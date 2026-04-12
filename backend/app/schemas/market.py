from pydantic import BaseModel
from typing import List, Optional


class Candle(BaseModel):
    time: int  # unix timestamp seconds
    open: float
    high: float
    low: float
    close: float
    volume: float


class CandleResponse(BaseModel):
    symbol: str
    interval: str
    candles: List[Candle]


class MarketQuote(BaseModel):
    symbol: str
    price: float
    bid: float
    ask: float
    volume: float
    change: float
    change_pct: float
