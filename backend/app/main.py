from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.database import engine, Base
from app.api.routes import market, options, model, trades, backtest
from app.api.routes import inference as inference_routes
from app.api.routes import signals as signals_routes
from app.api.routes import uncertainty as uncertainty_routes
from app.api.routes import regime as regime_routes
from app.api.routes import decision as decision_routes
from app.api.routes import governance as governance_routes
from app.websocket.manager import websocket_router
import logging
import uuid
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created")
    yield
    await engine.dispose()


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
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "req_id=%s method=%s path=%s status=%d duration_ms=%.1f",
        request_id, request.method, request.url.path,
        response.status_code, duration_ms,
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
app.include_router(websocket_router)


@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/ready", tags=["ops"])
async def ready():
    """Readiness: verify DB is reachable."""
    try:
        import sqlalchemy
        async with engine.connect() as conn:
            await conn.execute(sqlalchemy.text("SELECT 1"))
        return {"status": "ready"}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "not_ready", "error": str(e)})
