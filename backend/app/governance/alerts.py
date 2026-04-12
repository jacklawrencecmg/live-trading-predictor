"""
Governance alert service.

GovernanceAlertService
    Manages the governance_alerts table lifecycle:
        raise  → acknowledge → (expire)

    Deduplication: if an active alert with the same dedup_key exists,
    the trigger timestamp is bumped rather than creating a duplicate row.
    This prevents alert storms during prolonged incidents.

    Severity routing:
        critical → logger.critical
        warning  → logger.warning
        info     → logger.info

    Relationship to alert_service.py:
        The existing alert_service.py writes to audit_logs and fires callbacks.
        GovernanceAlertService writes to governance_alerts (richer schema with
        acknowledge/expire lifecycle).  Both can be used concurrently; they are
        not exclusive.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.governance.models import GovernanceAlert

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert type constants
# ---------------------------------------------------------------------------

class GovernanceAlertType:
    FEED_STALE              = "feed_stale"
    DRIFT_MODERATE          = "drift_moderate"
    DRIFT_HIGH              = "drift_high"
    CALIBRATION_DEGRADED    = "calibration_degraded"
    RETRAIN_NEEDED          = "retrain_needed"
    KILL_SWITCH_ACTIVATED   = "kill_switch_activated"
    KILL_SWITCH_DEACTIVATED = "kill_switch_deactivated"
    MODEL_PROMOTED          = "model_promoted"
    MODEL_DEPRECATED        = "model_deprecated"
    RISK_BREACH             = "risk_breach"
    INFERENCE_ERROR_SPIKE   = "inference_error_spike"


_SEVERITY_MAP: Dict[str, str] = {
    GovernanceAlertType.FEED_STALE:              "warning",
    GovernanceAlertType.DRIFT_MODERATE:          "warning",
    GovernanceAlertType.DRIFT_HIGH:              "critical",
    GovernanceAlertType.CALIBRATION_DEGRADED:    "warning",
    GovernanceAlertType.RETRAIN_NEEDED:          "warning",
    GovernanceAlertType.KILL_SWITCH_ACTIVATED:   "critical",
    GovernanceAlertType.KILL_SWITCH_DEACTIVATED: "info",
    GovernanceAlertType.MODEL_PROMOTED:          "info",
    GovernanceAlertType.MODEL_DEPRECATED:        "info",
    GovernanceAlertType.RISK_BREACH:             "critical",
    GovernanceAlertType.INFERENCE_ERROR_SPIKE:   "critical",
}

# Default alert TTLs in hours
_DEFAULT_TTL_HOURS: Dict[str, float] = {
    GovernanceAlertType.FEED_STALE:           1.0,
    GovernanceAlertType.DRIFT_MODERATE:       24.0,
    GovernanceAlertType.DRIFT_HIGH:           48.0,
    GovernanceAlertType.CALIBRATION_DEGRADED: 24.0,
    GovernanceAlertType.RETRAIN_NEEDED:       72.0,
    GovernanceAlertType.RISK_BREACH:          4.0,
}


# ---------------------------------------------------------------------------
# GovernanceAlertService
# ---------------------------------------------------------------------------

class GovernanceAlertService:

    @staticmethod
    async def raise_alert(
        db: AsyncSession,
        *,
        alert_type: str,
        title: str,
        symbol: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: Optional[str] = None,
        expires_hours: Optional[float] = None,
        dedup_key: Optional[str] = None,
    ) -> GovernanceAlert:
        """
        Create or bump an alert.

        If an active alert with the same dedup_key exists:
            - triggered_at is updated to now
            - The existing row is returned (no duplicate created)
        """
        sev = severity or _SEVERITY_MAP.get(alert_type, "info")
        ttl = expires_hours or _DEFAULT_TTL_HOURS.get(alert_type)
        expires_at = datetime.utcnow() + timedelta(hours=ttl) if ttl else None
        dk = dedup_key or _make_dedup_key(alert_type, symbol)

        # Deduplication check
        if dk:
            existing_result = await db.execute(
                select(GovernanceAlert).where(
                    GovernanceAlert.dedup_key == dk,
                    GovernanceAlert.is_active.is_(True),
                )
            )
            existing: Optional[GovernanceAlert] = existing_result.scalar_one_or_none()
            if existing is not None:
                existing.triggered_at = datetime.utcnow()
                if details:
                    existing.details_json = json.dumps(details)
                await db.flush()
                return existing

        # New alert
        row = GovernanceAlert(
            alert_type=alert_type,
            severity=sev,
            symbol=symbol,
            title=title,
            details_json=json.dumps(details) if details else None,
            triggered_at=datetime.utcnow(),
            expires_at=expires_at,
            dedup_key=dk,
            is_active=True,
        )
        db.add(row)
        await db.flush()

        log_fn = (
            logger.critical if sev == "critical"
            else logger.warning if sev == "warning"
            else logger.info
        )
        log_fn("GovernanceAlert [%s] %s: %s", sev.upper(), alert_type, title)
        return row

    @staticmethod
    async def acknowledge(
        db: AsyncSession,
        alert_id: int,
        by: str,
    ) -> GovernanceAlert:
        result = await db.execute(
            select(GovernanceAlert).where(GovernanceAlert.id == alert_id)
        )
        row: Optional[GovernanceAlert] = result.scalar_one_or_none()
        if row is None:
            raise ValueError(f"Alert id={alert_id} not found")
        row.acknowledged_at = datetime.utcnow()
        row.acknowledged_by = by
        row.is_active = False
        await db.flush()
        return row

    @staticmethod
    async def get_active(
        db: AsyncSession,
        severity: Optional[str] = None,
        alert_type: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> List[GovernanceAlert]:
        stmt = (
            select(GovernanceAlert)
            .where(GovernanceAlert.is_active.is_(True))
            .order_by(GovernanceAlert.triggered_at.desc())
        )
        if severity:
            stmt = stmt.where(GovernanceAlert.severity == severity)
        if alert_type:
            stmt = stmt.where(GovernanceAlert.alert_type == alert_type)
        if symbol:
            stmt = stmt.where(GovernanceAlert.symbol == symbol)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def get_history(
        db: AsyncSession,
        limit: int = 50,
        alert_type: Optional[str] = None,
    ) -> List[GovernanceAlert]:
        stmt = (
            select(GovernanceAlert)
            .order_by(GovernanceAlert.triggered_at.desc())
            .limit(limit)
        )
        if alert_type:
            stmt = stmt.where(GovernanceAlert.alert_type == alert_type)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def clear_expired(db: AsyncSession) -> int:
        """Mark alerts past their expires_at as inactive.  Returns count cleared."""
        result = await db.execute(
            update(GovernanceAlert)
            .where(
                GovernanceAlert.is_active.is_(True),
                GovernanceAlert.expires_at <= datetime.utcnow(),
            )
            .values(is_active=False)
        )
        n = result.rowcount
        if n:
            logger.info("Cleared %d expired governance alerts", n)
        return n

    # ------------------------------------------------------------------
    # Typed convenience methods
    # ------------------------------------------------------------------

    @staticmethod
    async def alert_feed_stale(
        db: AsyncSession,
        symbol: str,
        source: str,
        age_seconds: float,
    ) -> GovernanceAlert:
        return await GovernanceAlertService.raise_alert(
            db,
            alert_type=GovernanceAlertType.FEED_STALE,
            title=f"{symbol} {source} is stale ({age_seconds:.0f}s)",
            symbol=symbol,
            details={"source": source, "age_seconds": age_seconds},
            dedup_key=f"feed_stale:{symbol}:{source}",
        )

    @staticmethod
    async def alert_drift(
        db: AsyncSession,
        symbol: str,
        drift_level: str,
        max_psi: float,
        high_features: List[str],
    ) -> GovernanceAlert:
        atype = (GovernanceAlertType.DRIFT_HIGH if drift_level == "high"
                 else GovernanceAlertType.DRIFT_MODERATE)
        return await GovernanceAlertService.raise_alert(
            db,
            alert_type=atype,
            title=f"{symbol} feature drift {drift_level.upper()} (max_psi={max_psi:.3f})",
            symbol=symbol,
            details={"drift_level": drift_level, "max_psi": max_psi, "high_features": high_features},
            dedup_key=f"drift:{drift_level}:{symbol}",
        )

    @staticmethod
    async def alert_calibration_degraded(
        db: AsyncSession,
        symbol: str,
        calibration_health: str,
        rolling_brier: Optional[float],
    ) -> GovernanceAlert:
        return await GovernanceAlertService.raise_alert(
            db,
            alert_type=GovernanceAlertType.CALIBRATION_DEGRADED,
            title=f"{symbol} calibration {calibration_health} (brier={rolling_brier})",
            symbol=symbol,
            details={"calibration_health": calibration_health, "rolling_brier": rolling_brier},
            dedup_key=f"cal_degraded:{symbol}",
        )

    @staticmethod
    async def alert_retrain_needed(
        db: AsyncSession,
        symbol: str,
        reason: str,
    ) -> GovernanceAlert:
        return await GovernanceAlertService.raise_alert(
            db,
            alert_type=GovernanceAlertType.RETRAIN_NEEDED,
            title=f"{symbol} retrain recommended: {reason[:80]}",
            symbol=symbol,
            details={"reason": reason},
            dedup_key=f"retrain:{symbol}",
        )

    @staticmethod
    async def alert_kill_switch(
        db: AsyncSession,
        active: bool,
        reason: Optional[str],
        by: Optional[str],
    ) -> GovernanceAlert:
        atype = (GovernanceAlertType.KILL_SWITCH_ACTIVATED if active
                 else GovernanceAlertType.KILL_SWITCH_DEACTIVATED)
        state = "ACTIVATED" if active else "DEACTIVATED"
        return await GovernanceAlertService.raise_alert(
            db,
            alert_type=atype,
            title=f"Kill switch {state}" + (f" by {by}" if by else ""),
            details={"reason": reason, "by": by},
            dedup_key=None,   # every toggle is a distinct alert
        )

    @staticmethod
    async def alert_risk_breach(
        db: AsyncSession,
        symbol: Optional[str],
        breach_type: str,
        details: Dict[str, Any],
    ) -> GovernanceAlert:
        return await GovernanceAlertService.raise_alert(
            db,
            alert_type=GovernanceAlertType.RISK_BREACH,
            title=f"Risk breach: {breach_type}" + (f" ({symbol})" if symbol else ""),
            symbol=symbol,
            details=details,
            dedup_key=f"risk_breach:{breach_type}:{symbol or 'global'}",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dedup_key(alert_type: str, symbol: Optional[str]) -> Optional[str]:
    """Default dedup key: {alert_type}:{symbol}  or  {alert_type}:global"""
    return f"{alert_type}:{symbol or 'global'}"
