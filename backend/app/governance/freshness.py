"""
Data freshness monitoring service.

Checks whether each data source for a symbol has been updated within its
expected freshness window.  Staleness is defined per source:

    quote_feed      60 s   — live price/trade data
    candle_data    600 s   — 5-minute bars (max 2 missed bars = 10 min)
    options_chain 3600 s   — IV/chain snapshot (same threshold used in inference)
    model_artifact    —    — always 'fresh' unless file is missing

Writes DataFreshnessCheck rows on every check.
Callers should invoke check_and_record() after each data ingestion cycle.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.governance.models import DataFreshnessCheck

logger = logging.getLogger(__name__)

# Threshold in seconds for each source
STALENESS_THRESHOLDS: Dict[str, float] = {
    "quote_feed":     60.0,
    "candle_data":    600.0,
    "options_chain":  3600.0,
    "model_artifact": 86400.0 * 7,   # 7 days — warn if no new model in a week
}


class DataFreshnessService:

    @staticmethod
    async def record_check(
        db: AsyncSession,
        *,
        symbol: str,
        source: str,
        last_data_ts: Optional[datetime],
        override_threshold: Optional[float] = None,
    ) -> DataFreshnessCheck:
        """
        Record a freshness check for (symbol, source).

        last_data_ts : the timestamp of the most recent data point received.
                       Pass None if the source is completely unavailable (always stale).
        """
        threshold = override_threshold or STALENESS_THRESHOLDS.get(source, 600.0)
        now = datetime.utcnow()

        age_seconds: Optional[float] = None
        is_stale = True
        if last_data_ts is not None:
            age_seconds = (now - last_data_ts).total_seconds()
            is_stale = age_seconds > threshold

        row = DataFreshnessCheck(
            symbol=symbol,
            source=source,
            checked_at=now,
            last_data_ts=last_data_ts,
            age_seconds=age_seconds,
            is_stale=is_stale,
            staleness_threshold_seconds=threshold,
            alert_raised=False,
        )
        db.add(row)
        await db.flush()

        if is_stale:
            logger.warning(
                "Data freshness: %s source=%s is STALE (age=%.0fs threshold=%.0fs)",
                symbol, source, age_seconds or -1, threshold,
            )
        return row

    @staticmethod
    async def get_current_status(
        db: AsyncSession,
        symbol: str,
    ) -> Dict[str, Any]:
        """
        Return the most recent check result per source for symbol.

        Returns {source: {is_stale, age_seconds, last_data_ts, checked_at}}.
        """
        subq = (
            select(
                DataFreshnessCheck.source,
                func.max(DataFreshnessCheck.checked_at).label("max_ts"),
            )
            .where(DataFreshnessCheck.symbol == symbol)
            .group_by(DataFreshnessCheck.source)
            .subquery()
        )
        result = await db.execute(
            select(DataFreshnessCheck).join(
                subq,
                (DataFreshnessCheck.source == subq.c.source) &
                (DataFreshnessCheck.checked_at == subq.c.max_ts),
            ).where(DataFreshnessCheck.symbol == symbol)
        )
        rows = result.scalars().all()
        out: Dict[str, Any] = {}
        for row in rows:
            out[row.source] = {
                "is_stale":             row.is_stale,
                "age_seconds":          row.age_seconds,
                "last_data_ts":         row.last_data_ts.isoformat() if row.last_data_ts else None,
                "checked_at":           row.checked_at.isoformat(),
                "threshold_seconds":    row.staleness_threshold_seconds,
            }
        return out

    @staticmethod
    async def get_stale_feeds(
        db: AsyncSession,
        since_minutes: int = 15,
    ) -> List[DataFreshnessCheck]:
        """
        Return all freshness checks in the last `since_minutes` that are stale.
        Used by the governance summary dashboard.
        """
        cutoff = datetime.utcnow() - timedelta(minutes=since_minutes)
        result = await db.execute(
            select(DataFreshnessCheck)
            .where(
                DataFreshnessCheck.is_stale.is_(True),
                DataFreshnessCheck.checked_at >= cutoff,
            )
            .order_by(DataFreshnessCheck.checked_at.desc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_history(
        db: AsyncSession,
        symbol: str,
        source: str,
        limit: int = 50,
    ) -> List[DataFreshnessCheck]:
        result = await db.execute(
            select(DataFreshnessCheck)
            .where(
                DataFreshnessCheck.symbol == symbol,
                DataFreshnessCheck.source == source,
            )
            .order_by(DataFreshnessCheck.checked_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
