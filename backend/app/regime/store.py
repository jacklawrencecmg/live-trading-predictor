"""
Async persistence layer for regime labels.

Saves per-bar regime context to the DB and provides queries for:
  - Loading recent history (regime timeline)
  - Loading regime distribution (for stats / monitoring)
  - Loading per-regime sample counts (for backtest segmentation)
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import select, func, delete
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.regime_label import RegimeLabel
from app.regime.detector import RegimeContext

logger = logging.getLogger(__name__)


async def save_regime_label(
    db: AsyncSession,
    symbol: str,
    timeframe: str,
    bar_open_time: str,
    context: RegimeContext,
) -> Optional[RegimeLabel]:
    """
    Upsert a regime label for a bar. Non-blocking — call from inference
    without awaiting if you don't need the result.

    Uses INSERT OR REPLACE semantics (upsert on unique constraint).
    Returns the saved row, or None on error.
    """
    try:
        row = RegimeLabel(
            symbol=symbol,
            timeframe=timeframe,
            bar_open_time=bar_open_time,
            regime=str(context.regime.value if hasattr(context.regime, "value") else context.regime),
            adx_proxy=context.adx_proxy,
            atr_ratio=context.atr_ratio,
            volume_ratio=context.volume_ratio,
            bar_range_ratio=context.bar_range_ratio,
            is_abnormal_move=context.is_abnormal_move,
            abnormal_sigma=context.abnormal_move_sigma,
            trend_direction=context.trend_direction,
            confidence_threshold=context.confidence_threshold,
            suppressed=context.suppressed,
            created_at=datetime.utcnow(),
        )

        # Upsert: delete existing then insert (portable across SQLite/PG)
        stmt_del = delete(RegimeLabel).where(
            RegimeLabel.symbol == symbol,
            RegimeLabel.timeframe == timeframe,
            RegimeLabel.bar_open_time == bar_open_time,
        )
        await db.execute(stmt_del)
        db.add(row)
        await db.commit()
        return row
    except Exception as e:
        logger.warning("Failed to save regime label: %s", e)
        await db.rollback()
        return None


async def load_recent_regime_labels(
    db: AsyncSession,
    symbol: str,
    timeframe: str,
    limit: int = 100,
) -> List[RegimeLabel]:
    """
    Load the most recent N regime labels for a symbol, newest first.
    """
    stmt = (
        select(RegimeLabel)
        .where(RegimeLabel.symbol == symbol, RegimeLabel.timeframe == timeframe)
        .order_by(RegimeLabel.bar_open_time.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()
    return list(rows)


async def load_regime_distribution(
    db: AsyncSession,
    symbol: str,
    timeframe: str,
    limit: int = 500,
) -> dict:
    """
    Count occurrences of each regime over the most recent `limit` stored bars.
    Returns a dict: {regime_name: fraction, ...}
    """
    # Get most recent `limit` rows
    stmt = (
        select(RegimeLabel.regime, func.count(RegimeLabel.id).label("cnt"))
        .where(RegimeLabel.symbol == symbol, RegimeLabel.timeframe == timeframe)
        .group_by(RegimeLabel.regime)
    )
    result = await db.execute(stmt)
    rows = result.all()

    total = sum(r.cnt for r in rows)
    if total == 0:
        return {}
    return {r.regime: round(r.cnt / total, 4) for r in rows}


async def load_regime_performance_data(
    db: AsyncSession,
    symbol: str,
    timeframe: str,
    limit: int = 1000,
) -> List[dict]:
    """
    Load recent regime labels as a list of dicts for downstream analysis
    (e.g., merging with inference history to compute per-regime accuracy).
    """
    stmt = (
        select(RegimeLabel)
        .where(RegimeLabel.symbol == symbol, RegimeLabel.timeframe == timeframe)
        .order_by(RegimeLabel.bar_open_time.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()
    return [
        {
            "bar_open_time": r.bar_open_time,
            "regime": r.regime,
            "adx_proxy": r.adx_proxy,
            "atr_ratio": r.atr_ratio,
            "volume_ratio": r.volume_ratio,
            "is_abnormal_move": r.is_abnormal_move,
            "suppressed": r.suppressed,
        }
        for r in reversed(rows)  # chronological order
    ]
