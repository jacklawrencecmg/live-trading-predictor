"""
Calibration monitoring service.

Writes periodic CalibrationSnapshot rows from ConfidenceTracker.get_stats().
Enables querying the historical trend of:
    - rolling Brier score vs baseline
    - ECE
    - degradation factor
    - needs_retrain flag

Call record_snapshot() after every N inference events (e.g. N=20)
or on a schedule (e.g. hourly cron).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.governance.models import CalibrationSnapshot

logger = logging.getLogger(__name__)


class CalibrationMonitor:

    @staticmethod
    async def record_snapshot(
        db: AsyncSession,
        *,
        symbol: str,
        tracker_stats: Any,            # TrackerStats from confidence_tracker.py (duck-typed)
        model_name: Optional[str] = None,
    ) -> CalibrationSnapshot:
        """
        Persist a calibration health snapshot from a TrackerStats object.
        Should be called periodically (every N inferences or on schedule).
        """
        rel_json: Optional[str] = None
        if (
            getattr(tracker_stats, "reliability_bins", None) is not None
            and tracker_stats.reliability_bins
        ):
            rel_json = json.dumps({
                "bins": tracker_stats.reliability_bins,
                "mean_predicted": tracker_stats.reliability_mean_pred,
                "fraction_positive": tracker_stats.reliability_frac_pos,
            })

        snap = CalibrationSnapshot(
            symbol=symbol,
            snapshot_at=datetime.utcnow(),
            model_name=model_name,
            window_size=getattr(tracker_stats, "window_size", None),
            rolling_brier=getattr(tracker_stats, "rolling_brier", None),
            baseline_brier=getattr(tracker_stats, "baseline_brier", None),
            degradation_factor=getattr(tracker_stats, "degradation_factor", None),
            ece_recent=getattr(tracker_stats, "ece_recent", None),
            calibration_health=getattr(tracker_stats, "calibration_health", None),
            needs_retrain=bool(getattr(tracker_stats, "needs_retrain", False)),
            retrain_reason=getattr(tracker_stats, "retrain_reason", None),
            reliability_json=rel_json,
        )
        db.add(snap)
        await db.flush()

        if snap.needs_retrain:
            logger.warning(
                "CalibrationSnapshot: %s needs_retrain=True reason=%s",
                symbol, snap.retrain_reason,
            )
        return snap

    @staticmethod
    async def get_history(
        db: AsyncSession,
        symbol: str,
        limit: int = 50,
    ) -> List[CalibrationSnapshot]:
        result = await db.execute(
            select(CalibrationSnapshot)
            .where(CalibrationSnapshot.symbol == symbol)
            .order_by(CalibrationSnapshot.snapshot_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_latest(
        db: AsyncSession,
        symbol: str,
    ) -> Optional[CalibrationSnapshot]:
        result = await db.execute(
            select(CalibrationSnapshot)
            .where(CalibrationSnapshot.symbol == symbol)
            .order_by(CalibrationSnapshot.snapshot_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def symbols_needing_retrain(
        db: AsyncSession,
    ) -> List[str]:
        """
        Return symbols whose most recent snapshot has needs_retrain=True.
        Uses a subquery to find latest snapshot per symbol.
        """
        from sqlalchemy import func
        subq = (
            select(
                CalibrationSnapshot.symbol,
                func.max(CalibrationSnapshot.snapshot_at).label("max_ts"),
            )
            .group_by(CalibrationSnapshot.symbol)
            .subquery()
        )
        result = await db.execute(
            select(CalibrationSnapshot)
            .join(
                subq,
                (CalibrationSnapshot.symbol == subq.c.symbol) &
                (CalibrationSnapshot.snapshot_at == subq.c.max_ts),
            )
            .where(CalibrationSnapshot.needs_retrain.is_(True))
        )
        return [r.symbol for r in result.scalars().all()]

    @staticmethod
    async def health_by_symbol(
        db: AsyncSession,
    ) -> Dict[str, str]:
        """Return {symbol: calibration_health} for latest snapshot per symbol."""
        from sqlalchemy import func
        subq = (
            select(
                CalibrationSnapshot.symbol,
                func.max(CalibrationSnapshot.snapshot_at).label("max_ts"),
            )
            .group_by(CalibrationSnapshot.symbol)
            .subquery()
        )
        result = await db.execute(
            select(CalibrationSnapshot)
            .join(
                subq,
                (CalibrationSnapshot.symbol == subq.c.symbol) &
                (CalibrationSnapshot.snapshot_at == subq.c.max_ts),
            )
        )
        return {r.symbol: (r.calibration_health or "unknown") for r in result.scalars().all()}

    @staticmethod
    def trend_direction(snapshots: List[CalibrationSnapshot]) -> str:
        """
        Given a time-ordered list (oldest first), classify Brier trend.
        Returns 'improving', 'stable', or 'degrading'.
        Requires at least 3 snapshots with rolling_brier set.
        """
        briers = [s.rolling_brier for s in snapshots if s.rolling_brier is not None]
        if len(briers) < 3:
            return "unknown"
        # Linear regression slope on last points
        import numpy as np
        x = list(range(len(briers)))
        slope = float(np.polyfit(x, briers, 1)[0])
        if slope > 0.002:   return "degrading"
        if slope < -0.002:  return "improving"
        return "stable"
