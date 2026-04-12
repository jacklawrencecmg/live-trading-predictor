"""
YFinanceMarketDataProvider — wraps the existing yfinance usage.

This is the current default provider. It is free, requires no API key,
but is rate-limited and unsuitable for production HFT or intraday trading
at scale. Replace with PolygonMarketDataProvider for production.

Migration: wherever market_data.py or ingestion_service.py calls yf.Ticker()
directly, inject this provider instead.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import AsyncIterator, Optional

import pandas as pd

from .protocols import MarketDataProvider, ProviderError, Quote

logger = logging.getLogger(__name__)


class YFinanceMarketDataProvider:
    """Implements MarketDataProvider using yfinance."""

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _download_candles(symbol: str, interval: str, period: str) -> pd.DataFrame:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            raise ProviderError(f"yfinance returned empty DataFrame for {symbol}")
        df.dropna(inplace=True)
        df.columns = [c.lower() for c in df.columns]
        # Normalize index to bar_open_time column
        if "bar_open_time" not in df.columns:
            df["bar_open_time"] = df.index
            df = df.reset_index(drop=True)
        return df

    @staticmethod
    def _get_quote(symbol: str) -> Quote:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        fi = ticker.fast_info
        hist = ticker.history(period="2d", interval="1d")
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else float(fi.last_price or 0)
        price = float(fi.last_price or prev_close)
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0.0
        return Quote(
            symbol=symbol,
            price=price,
            bid=round(price - 0.01, 4),   # yfinance has no real-time bid/ask
            ask=round(price + 0.01, 4),
            volume=float(fi.three_month_average_volume or 0),
            change=round(change, 4),
            change_pct=round(change_pct, 4),
        )

    # ── MarketDataProvider interface ───────────────────────────────────────

    async def get_candles(self, symbol: str, interval: str, period: str) -> pd.DataFrame:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, self._download_candles, symbol, interval, period
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"yfinance candle fetch failed for {symbol}: {exc}") from exc

    async def get_quote(self, symbol: str) -> Quote:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, self._get_quote, symbol)
        except Exception as exc:
            raise ProviderError(f"yfinance quote fetch failed for {symbol}: {exc}") from exc

    async def stream_quotes(
        self,
        symbol: str,
        interval_seconds: float = 15.0,
    ) -> AsyncIterator[Quote]:
        """Poll-based streaming — yields one Quote per interval_seconds."""
        while True:
            try:
                yield await self.get_quote(symbol)
            except ProviderError as exc:
                logger.warning("stream_quotes: %s", exc)
            await asyncio.sleep(interval_seconds)
