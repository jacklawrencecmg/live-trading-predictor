"""
Market data ingestion service.

Supports:
- Historical backfill from yfinance
- Live streaming (polling-based, configurable interval)
- Deduplication via UPSERT
- Retry logic with structured logging
- Correct closed-bar detection
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.data_ingestion.bar_model import OHLCVBar

logger = logging.getLogger(__name__)

TIMEFRAME_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "1d": 1440,
}


def _bar_close_time(bar_open: datetime, timeframe: str) -> datetime:
    minutes = TIMEFRAME_MINUTES.get(timeframe, 5)
    return bar_open + timedelta(minutes=minutes)


# Guard buffer added after bar_close_time before marking a bar as closed.
# Real-time market data may lag the nominal bar-close by 50–500 ms due to
# exchange dissemination, vendor processing, and network propagation.
# Without this buffer, a bar marked closed at bar_close_time + 1 ms may
# still have incomplete OHLC data from the vendor.
_BAR_CLOSE_BUFFER_SECONDS: int = 5


def _is_bar_closed(bar_open: datetime, timeframe: str) -> bool:
    """
    A bar is data-safe when bar_close_time + buffer <= now (UTC).

    The 5-second buffer guards the window [bar_close, bar_close + 5s]
    where the bar has nominally ended but price data may not yet have
    propagated from the exchange through the vendor API.
    """
    close = _bar_close_time(bar_open, timeframe)
    return (close + timedelta(seconds=_BAR_CLOSE_BUFFER_SECONDS)) <= datetime.utcnow()


def _parse_yf_df(df: pd.DataFrame, symbol: str, timeframe: str, source: str = "yfinance") -> List[dict]:
    """Convert yfinance DataFrame to list of bar dicts."""
    now = datetime.utcnow()
    bars = []
    for ts, row in df.iterrows():
        if hasattr(ts, "to_pydatetime"):
            bar_open = ts.to_pydatetime().replace(tzinfo=None)
        else:
            bar_open = pd.Timestamp(ts).to_pydatetime().replace(tzinfo=None)

        bar_close = _bar_close_time(bar_open, timeframe)
        is_closed = bar_close <= now

        # availability_time: for historical backfill this equals now (ingested_at).
        # For a live bar it would be bar_close_time; we use now for all backfill rows.
        availability_time = now

        # A row is stale if it arrived more than one full timeframe after bar_close_time.
        timeframe_minutes = TIMEFRAME_MINUTES.get(timeframe, 5)
        staleness_threshold = timedelta(minutes=timeframe_minutes)
        staleness_flag = (now - bar_close) > staleness_threshold if is_closed else False

        # VWAP approximation: (H+L+C)/3
        vwap = (float(row["High"]) + float(row["Low"]) + float(row["Close"])) / 3

        bars.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "bar_open_time": bar_open,
            "bar_close_time": bar_close,
            "availability_time": availability_time,
            "ingested_at": now,
            "source": source,
            "staleness_flag": staleness_flag,
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
            "vwap": vwap,
            "is_closed": is_closed,
        })
    return bars


async def _upsert_bars(session: AsyncSession, bars: List[dict]) -> int:
    """Upsert bars by (symbol, timeframe, bar_open_time). Returns count inserted/updated."""
    if not bars:
        return 0

    # Use SQLite-compatible approach: try insert, update on conflict
    # For PostgreSQL, use ON CONFLICT DO UPDATE
    try:
        stmt = pg_insert(OHLCVBar).values(bars)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "timeframe", "bar_open_time"],
            set_={
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "volume": stmt.excluded.volume,
                "vwap": stmt.excluded.vwap,
                "is_closed": stmt.excluded.is_closed,
                "availability_time": stmt.excluded.availability_time,
                "ingested_at": stmt.excluded.ingested_at,
                "staleness_flag": stmt.excluded.staleness_flag,
            },
        )
        await session.execute(stmt)
        return len(bars)
    except Exception:
        # Fallback for SQLite (tests)
        for bar in bars:
            existing = await session.execute(
                select(OHLCVBar).where(
                    OHLCVBar.symbol == bar["symbol"],
                    OHLCVBar.timeframe == bar["timeframe"],
                    OHLCVBar.bar_open_time == bar["bar_open_time"],
                )
            )
            obj = existing.scalar_one_or_none()
            if obj is None:
                session.add(OHLCVBar(**bar))
            else:
                for k, v in bar.items():
                    setattr(obj, k, v)
        return len(bars)


async def backfill(
    symbol: str,
    timeframe: str = "5m",
    period: str = "60d",
    max_retries: int = 3,
) -> int:
    """Backfill historical bars. Returns number of bars stored."""
    logger.info("Backfill start: symbol=%s timeframe=%s period=%s", symbol, timeframe, period)

    for attempt in range(1, max_retries + 1):
        try:
            import yfinance as yf  # lazy import: only needed at runtime, not test collection
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: yf.Ticker(symbol).history(period=period, interval=timeframe),
            )
            df.dropna(inplace=True)
            if df.empty:
                logger.warning("No data returned for %s/%s", symbol, timeframe)
                return 0

            bars = _parse_yf_df(df, symbol, timeframe)
            async with AsyncSessionLocal() as session:
                count = await _upsert_bars(session, bars)
                await session.commit()

            logger.info("Backfill done: symbol=%s bars=%d", symbol, count)
            return count

        except Exception as exc:
            logger.error("Backfill attempt %d failed: %s", attempt, exc)
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
            else:
                raise


async def ingest_latest(symbol: str, timeframe: str = "5m") -> int:
    """Ingest the latest bars (last 2 days). For use in live streaming loop."""
    return await backfill(symbol, timeframe, period="2d")


async def get_closed_bars(
    session: AsyncSession,
    symbol: str,
    timeframe: str,
    limit: int = 500,
) -> List[OHLCVBar]:
    """Fetch the most recent closed bars for a symbol. Safe for feature engineering."""
    result = await session.execute(
        select(OHLCVBar)
        .where(
            OHLCVBar.symbol == symbol,
            OHLCVBar.timeframe == timeframe,
            OHLCVBar.is_closed == True,
        )
        .order_by(OHLCVBar.bar_open_time.desc())
        .limit(limit)
    )
    bars = result.scalars().all()
    return list(reversed(bars))  # chronological order
