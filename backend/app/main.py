import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.database import engine, Base
from app.core.logging_config import bind_request_context, configure_logging, reset_request_context
from app.core.redis_client import close_redis, get_redis

configure_logging(level=settings.log_level, json_logs=settings.json_logs)
logger = logging.getLogger(__name__)

from app.api.routes import market, options, model, trades, backtest
from app.api.routes import inference as inference_routes
from app.api.routes import signals as signals_routes
from app.api.routes import uncertainty as uncertainty_routes
from app.api.routes import regime as regime_routes
from app.api.routes import decision as decision_routes
from app.api.routes import governance as governance_routes
from app.api.metrics import router as metrics_router
from app.websocket.manager import websocket_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Database tables ───────────────────────────────────────────────────────
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("startup", extra={"event": "db_tables_created"})

    # ── Background monitoring scheduler ──────────────────────────────────────
    from app.governance.scheduler import create_scheduler
    symbols = [s.strip() for s in settings.default_symbol.split(",") if s.strip()]
    scheduler = create_scheduler(symbols=symbols or ["SPY"])
    await scheduler.start()
    logger.info("startup", extra={"event": "monitoring_scheduler_started"})

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    await scheduler.stop()
    logger.info("shutdown", extra={"event": "monitoring_scheduler_stopped"})
    await close_redis()
    await engine.dispose()
    logger.info("shutdown", extra={"event": "app_shutdown"})


app = FastAPI(
    title="Live Trading Predictor",
    description="Paper-trading options research with ML predictions, signals, and risk controls",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    ctx_token = bind_request_context(request_id=request_id)
    try:
        response = await call_next(request)
    finally:
        reset_request_context(ctx_token)
    duration_ms = (time.time() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round(duration_ms, 1),
        },
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Routers
app.include_router(market.router, prefix="/api/market", tags=["market"])
app.include_router(options.router, prefix="/api/options", tags=["options"])
app.include_router(model.router, prefix="/api/model", tags=["model"])
app.include_router(trades.router, prefix="/api/trades", tags=["trades"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["backtest"])
app.include_router(inference_routes.router, prefix="/api/inference", tags=["inference"])
app.include_router(signals_routes.router, prefix="/api/signals", tags=["signals"])
app.include_router(uncertainty_routes.router, prefix="/api/uncertainty", tags=["uncertainty"])
app.include_router(regime_routes.router, prefix="/api/regime", tags=["regime"])
app.include_router(decision_routes.router, prefix="/api/decision", tags=["decision"])
app.include_router(governance_routes.router, prefix="/api/governance", tags=["governance"])
app.include_router(metrics_router, tags=["ops"])
app.include_router(websocket_router)


@app.get("/health", tags=["ops"])
async def health():
    """Liveness probe — always returns 200 if the process is running."""
    return {"status": "ok", "version": app.version}


@app.get("/ready", tags=["ops"])
async def ready():
    """
    Readiness probe — returns 200 only when all critical services are reachable.
    Used by Docker Compose / Kubernetes to gate traffic.
    """
    import sqlalchemy

    services: dict[str, str] = {}
    ok = True

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    try:
        async with engine.connect() as conn:
            await conn.execute(sqlalchemy.text("SELECT 1"))
        services["postgres"] = "ok"
    except Exception as exc:
        services["postgres"] = f"error: {exc}"
        ok = False

    # ── Redis ─────────────────────────────────────────────────────────────────
    try:
        r = await get_redis()
        await r.ping()
        services["redis"] = "ok"
    except Exception as exc:
        services["redis"] = f"error: {exc}"
        ok = False

    body = {"status": "ready" if ok else "not_ready", "services": services}
    if not ok:
        return JSONResponse(status_code=503, content=body)
    return body
