from fastapi import APIRouter, Query
from typing import Optional
from app.services.market_data import fetch_candles, fetch_quote
from app.schemas.market import CandleResponse, MarketQuote

router = APIRouter()


@router.get("/candles/{symbol}", response_model=CandleResponse)
async def get_candles(
    symbol: str,
    interval: str = Query("5m", description="Candle interval"),
    period: str = Query("5d", description="Lookback period"),
):
    return await fetch_candles(symbol.upper(), interval, period)


@router.get("/quote/{symbol}", response_model=MarketQuote)
async def get_quote(symbol: str):
    return await fetch_quote(symbol.upper())
