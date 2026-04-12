from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.database import engine, Base
from app.core.redis_client import get_redis
from app.api.routes import market, options, model, trades, backtest
from app.websocket.manager import websocket_router
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created")
    yield
    await engine.dispose()


app = FastAPI(
    title="Options Research Platform",
    description="Paper-trading options research with ML predictions",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(market.router, prefix="/api/market", tags=["market"])
app.include_router(options.router, prefix="/api/options", tags=["options"])
app.include_router(model.router, prefix="/api/model", tags=["model"])
app.include_router(trades.router, prefix="/api/trades", tags=["trades"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["backtest"])
app.include_router(websocket_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
