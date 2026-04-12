"""
Inference event log service.

InferenceLogService
    Records every prediction in inference_events.
    Supports querying the log and back-filling actual outcomes once bars resolve.

    The write path is designed to be non-blocking: callers can fire-and-forget
    with log_inference_result_bg() or await log_inference_result() directly.

    outcome_recorded_at is filled by record_outcome() which should be called
    after each bar closes and price direction is known.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.governance.models import InferenceEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# InferenceLogService
# ---------------------------------------------------------------------------

class InferenceLogService:

    @staticmethod
    async def log_inference_result(
        db: AsyncSession,
        *,
        symbol: str,
        result: Any,                       # InferenceResult from inference_service
        request_id: Optional[str] = None,
        model_version_id: Optional[int] = None,
        manifest_hash: Optional[str] = None,
    ) -> InferenceEvent:
        """
        Persist one inference result.  Extracts all 4-layer fields from the
        InferenceResult dataclass (duck-typed to avoid circular imports).
        """
        options_stale = False
        if hasattr(result, "abstain_reason") and result.abstain_reason:
            options_stale = "stale" in str(result.abstain_reason).lower()

        bar_time: Optional[datetime] = None
        if hasattr(result, "bar_open_time") and result.bar_open_time:
            try:
                bar_time = datetime.fromisoformat(str(result.bar_open_time).replace("Z", ""))
            except (ValueError, TypeError):
                pass

        event = InferenceEvent(
            request_id=request_id,
            symbol=symbol,
            bar_open_time=bar_time,
            inference_ts=getattr(result, "timestamp", 0),

            model_name=_extract(result, "model_version"),
            model_version_id=model_version_id,
            feature_snapshot_id=getattr(result, "feature_snapshot_id", None),
            manifest_hash=manifest_hash,

            prob_up=getattr(result, "prob_up", None),
            prob_down=getattr(result, "prob_down", None),
            calibrated_prob_up=getattr(result, "calibrated_prob_up", None),
            calibration_available=getattr(result, "calibration_available", None),

            tradeable_confidence=getattr(result, "tradeable_confidence", None),
            degradation_factor=getattr(result, "degradation_factor", None),

            action=getattr(result, "action", None),
            abstain_reason=_trunc(getattr(result, "abstain_reason", None), 128),

            calibration_health=getattr(result, "calibration_health", None),
            ece_recent=getattr(result, "ece_recent", None),
            rolling_brier=getattr(result, "rolling_brier", None),
            expected_move_pct=getattr(result, "expected_move_pct", None),
            regime=_trunc(getattr(result, "regime", None), 32),
            options_stale=options_stale,
        )
        db.add(event)
        await db.flush()
        return event

    @staticmethod
    async def record_outcome(
        db: AsyncSession,
        *,
        symbol: str,
        bar_open_time: datetime,
        actual_outcome: int,    # 1=up 0=down
    ) -> int:
        """
        Back-fill actual_outcome for all pending events matching (symbol, bar_open_time).
        Returns count of rows updated.
        """
        result = await db.execute(
            update(InferenceEvent)
            .where(
                InferenceEvent.symbol == symbol,
                InferenceEvent.bar_open_time == bar_open_time,
                InferenceEvent.actual_outcome.is_(None),
            )
            .values(
                actual_outcome=actual_outcome,
                outcome_recorded_at=datetime.utcnow(),
            )
        )
        n = result.rowcount
        if n:
            logger.debug("Recorded outcome=%d for %s @ %s (%d rows)", actual_outcome, symbol, bar_open_time, n)
        return n

    @staticmethod
    async def query(
        db: AsyncSession,
        *,
        symbol: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        from_ts: Optional[datetime] = None,
        to_ts: Optional[datetime] = None,
        pending_only: bool = False,
    ) -> List[InferenceEvent]:
        stmt = (
            select(InferenceEvent)
            .order_by(InferenceEvent.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if symbol:
            stmt = stmt.where(InferenceEvent.symbol == symbol)
        if action:
            stmt = stmt.where(InferenceEvent.action == action)
        if from_ts:
            stmt = stmt.where(InferenceEvent.created_at >= from_ts)
        if to_ts:
            stmt = stmt.where(InferenceEvent.created_at <= to_ts)
        if pending_only:
            stmt = stmt.where(InferenceEvent.actual_outcome.is_(None))
        result = await db.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def get_accuracy_stats(
        db: AsyncSession,
        symbol: str,
        window: int = 100,
    ) -> Dict[str, Any]:
        """
        Compute rolling accuracy over the most recent `window` resolved events.
        Returns: {n_resolved, n_correct, accuracy, n_pending, abstain_rate}
        """
        events = await InferenceLogService.query(
            db, symbol=symbol, limit=window
        )
        resolved   = [e for e in events if e.actual_outcome is not None]
        pending    = [e for e in events if e.actual_outcome is None]
        abstained  = [e for e in events if e.action == "abstain"]

        correct = 0
        for e in resolved:
            if e.actual_outcome is None:
                continue
            # buy → correct if price went up (outcome=1)
            # sell → correct if price went down (outcome=0)
            if e.action == "buy"  and e.actual_outcome == 1: correct += 1
            if e.action == "sell" and e.actual_outcome == 0: correct += 1

        n = len(resolved)
        return {
            "n_total":      len(events),
            "n_resolved":   n,
            "n_correct":    correct,
            "accuracy":     round(correct / n, 4) if n else None,
            "n_pending":    len(pending),
            "abstain_rate": round(len(abstained) / len(events), 4) if events else None,
        }

    @staticmethod
    async def count_24h(
        db: AsyncSession,
        symbol: Optional[str] = None,
    ) -> Dict[str, int]:
        """Return count of inference events in last 24 hours, broken down by action."""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(hours=24)
        stmt = (
            select(InferenceEvent.action, func.count(InferenceEvent.id).label("cnt"))
            .where(InferenceEvent.created_at >= cutoff)
            .group_by(InferenceEvent.action)
        )
        if symbol:
            stmt = stmt.where(InferenceEvent.symbol == symbol)
        result = await db.execute(stmt)
        rows = result.all()
        counts: Dict[str, int] = {}
        for action, cnt in rows:
            counts[action or "unknown"] = cnt
        return counts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract(obj: Any, attr: str) -> Optional[str]:
    v = getattr(obj, attr, None)
    return str(v) if v is not None else None


def _trunc(s: Optional[str], n: int) -> Optional[str]:
    if s is None:
        return None
    return s[:n] if len(s) > n else s
