"""
Kill switch service.

Two layers of kill-switch state:
    1. Settings.kill_switch   — environment variable override (immediate, no DB)
    2. KillSwitchState DB row — persisted state (survives restart, requires DB)

Active if EITHER layer is True.

The `is_active_cached()` function uses a module-level TTL cache (5-second TTL)
for use on the hot inference path — avoids a DB query on every request while
staying responsive to state changes.

Thread safety: the cache is updated atomically.  The DB query is authoritative;
the cache is an optimisation for the hot path only.

Kill switch activation:
    - Writes to KillSwitchState (id=1)
    - Creates a GovernanceAlert (severity=critical)
    - Writes to AuditLog
    - Updates module-level cache immediately

Kill switch deactivation:
    - Same pattern
    - Requires explicit reason ("who deactivated and why")
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.governance.models import KillSwitchState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level TTL cache
# ---------------------------------------------------------------------------

_cache_active: bool = False
_cache_ts:     float = 0.0
_CACHE_TTL:    float = 5.0   # seconds


def _cache_set(active: bool) -> None:
    global _cache_active, _cache_ts
    _cache_active = active
    _cache_ts = time.monotonic()


def _cache_get() -> Tuple[bool, bool]:
    """Returns (active, is_fresh)."""
    age = time.monotonic() - _cache_ts
    return _cache_active, age < _CACHE_TTL


# ---------------------------------------------------------------------------
# KillSwitchService
# ---------------------------------------------------------------------------

class KillSwitchService:

    @staticmethod
    async def _ensure_singleton(db: AsyncSession) -> KillSwitchState:
        """Get or create the singleton row (id=1)."""
        result = await db.execute(
            select(KillSwitchState).where(KillSwitchState.id == 1)
        )
        row = result.scalar_one_or_none()
        if row is None:
            row = KillSwitchState(id=1, active=False)
            db.add(row)
            await db.flush()
        return row

    @staticmethod
    async def get_state(db: AsyncSession) -> KillSwitchState:
        return await KillSwitchService._ensure_singleton(db)

    @staticmethod
    async def is_active_db(db: AsyncSession) -> bool:
        """
        Authoritative DB check.  Use for non-hot-path decisions.
        Also updates the module cache.
        """
        from app.core.config import get_settings
        settings = get_settings()
        if settings.kill_switch:          # env-var override always wins
            _cache_set(True)
            return True
        row = await KillSwitchService._ensure_singleton(db)
        active = bool(row.active)
        _cache_set(active)
        return active

    @staticmethod
    def is_active_cached() -> bool:
        """
        Fast hot-path check (no DB).
        Returns cached value; use is_active_db() to refresh.
        Always defers to settings.kill_switch env override.
        """
        from app.core.config import get_settings
        try:
            if get_settings().kill_switch:
                return True
        except Exception:
            pass
        active, fresh = _cache_get()
        if fresh:
            return active
        # Cache stale — optimistic: assume not active until refreshed
        return False

    @staticmethod
    async def activate(
        db: AsyncSession,
        *,
        reason: str,
        by: str,
    ) -> KillSwitchState:
        row = await KillSwitchService._ensure_singleton(db)
        if row.active:
            logger.info("Kill switch already active — updating reason")
        row.active = True
        row.reason = reason
        row.activated_at = datetime.utcnow()
        row.activated_by = by
        row.updated_at = datetime.utcnow()
        await db.flush()

        _cache_set(True)

        # Sync to Redis so the paper-trading risk_manager hot path is also halted.
        # The governance DB is the audit-authoritative record; Redis is the fast gate.
        try:
            from app.services.risk_manager import set_kill_switch as _redis_ks
            await _redis_ks(True)
        except Exception as _e:
            logger.warning("Kill switch Redis sync (activate) failed: %s", _e)

        # Audit
        from app.models.audit_log import AuditLog
        db.add(AuditLog(
            event_type="governance:kill_switch_activated",
            symbol=None,
            details={"reason": reason, "by": by},
            message=f"Kill switch ACTIVATED by {by}: {reason}",
        ))

        # Governance alert
        from app.governance.alerts import GovernanceAlertService
        await GovernanceAlertService.alert_kill_switch(db, active=True, reason=reason, by=by)

        logger.critical("KILL SWITCH ACTIVATED by %s: %s", by, reason)
        return row

    @staticmethod
    async def deactivate(
        db: AsyncSession,
        *,
        by: str,
        reason: Optional[str] = None,
    ) -> KillSwitchState:
        row = await KillSwitchService._ensure_singleton(db)
        if not row.active:
            logger.info("Kill switch already inactive")
        row.active = False
        row.updated_at = datetime.utcnow()
        await db.flush()

        _cache_set(False)

        # Sync to Redis so the paper-trading risk_manager hot path is also resumed.
        try:
            from app.services.risk_manager import set_kill_switch as _redis_ks
            await _redis_ks(False)
        except Exception as _e:
            logger.warning("Kill switch Redis sync (deactivate) failed: %s", _e)

        from app.models.audit_log import AuditLog
        db.add(AuditLog(
            event_type="governance:kill_switch_deactivated",
            symbol=None,
            details={"reason": reason, "by": by},
            message=f"Kill switch DEACTIVATED by {by}" + (f": {reason}" if reason else ""),
        ))

        from app.governance.alerts import GovernanceAlertService
        await GovernanceAlertService.alert_kill_switch(db, active=False, reason=reason, by=by)

        logger.warning("Kill switch DEACTIVATED by %s", by)
        return row

    @staticmethod
    async def toggle(
        db: AsyncSession,
        *,
        active: bool,
        reason: Optional[str] = None,
        by: str = "operator",
    ) -> KillSwitchState:
        """Activate or deactivate based on `active` flag."""
        if active:
            return await KillSwitchService.activate(
                db, reason=reason or "Manual activation", by=by
            )
        return await KillSwitchService.deactivate(db, by=by, reason=reason)
