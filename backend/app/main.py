from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.database import engine, Base
from app.api.routes import market, options, model, trades, backtest
from app.api.routes import inference as inference_router
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
    title="Options Research Platform",
    description="Paper-trading options research with ML predictions",
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


app.include_router(market.router, prefix="/api/market", tags=["market"])
app.include_router(options.router, prefix="/api/options", tags=["options"])
app.include_router(model.router, prefix="/api/model", tags=["model"])
app.include_router(trades.router, prefix="/api/trades", tags=["trades"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["backtest"])
app.include_router(inference_router.router, prefix="/api/inference", tags=["inference"])
app.include_router(websocket_router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/ready")
async def ready():
    """Readiness check: verify DB connection."""
    try:
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        return {"status": "ready"}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "not_ready", "error": str(e)})
