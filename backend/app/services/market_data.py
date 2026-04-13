import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import pandas as pd
import yfinance as yf

from app.core.redis_client import get_redis
from app.schemas.market import Candle, CandleResponse, MarketQuote

CACHE_TTL_CANDLES = 60  # 1 min for intraday
CACHE_TTL_QUOTE = 15    # 15s for quote


async def fetch_candles(
    symbol: str,
    interval: str = "5m",
    period: str = "5d",
) -> CandleResponse:
    cache_key = f"candles:{symbol}:{interval}:{period}"
    _redis = None
    try:
        _redis = await get_redis()
        cached = await _redis.get(cache_key)
        if cached:
            return CandleResponse(**json.loads(cached))
    except Exception:
        _redis = None  # Redis unavailable — skip cache

    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(None, _download_candles, symbol, interval, period)

    candles = []
    for ts, row in df.iterrows():
        if hasattr(ts, "timestamp"):
            t = int(ts.timestamp())
        else:
            t = int(pd.Timestamp(ts).timestamp())
        candles.append(
            Candle(
                time=t,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
            )
        )

    result = CandleResponse(symbol=symbol, interval=interval, candles=candles)
    if _redis is not None:
        try:
            await _redis.setex(cache_key, CACHE_TTL_CANDLES, result.model_dump_json())
        except Exception:
            pass
    return result


def _download_candles(symbol: str, interval: str, period: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    df.dropna(inplace=True)
    return df


async def fetch_quote(symbol: str) -> MarketQuote:
    cache_key = f"quote:{symbol}"
    _redis = None
    try:
        _redis = await get_redis()
        cached = await _redis.get(cache_key)
        if cached:
            return MarketQuote(**json.loads(cached))
    except Exception:
        _redis = None  # Redis unavailable — skip cache

    loop = asyncio.get_event_loop()
    info = await loop.run_in_executor(None, _get_fast_info, symbol)
    quote = MarketQuote(**info)
    if _redis is not None:
        try:
            await _redis.setex(cache_key, CACHE_TTL_QUOTE, quote.model_dump_json())
        except Exception:
            pass
    return quote


def _get_fast_info(symbol: str) -> dict:
    ticker = yf.Ticker(symbol)
    fi = ticker.fast_info
    hist = ticker.history(period="2d", interval="1d")
    prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else float(fi.last_price or 0)
    price = float(fi.last_price or prev_close)
    change = price - prev_close
    change_pct = (change / prev_close * 100) if prev_close else 0
    return {
        "symbol": symbol,
        "price": price,
        "bid": price - 0.01,
        "ask": price + 0.01,
        "volume": float(fi.three_month_average_volume or 0),
        "change": change,
        "change_pct": change_pct,
    }


async def invalidate_cache(symbol: str):
    try:
        redis = await get_redis()
        pattern = f"*{symbol}*"
        keys = await redis.keys(pattern)
        if keys:
            await redis.delete(*keys)
    except Exception:
        pass  # Redis unavailable — skip cache invalidation
