"""
Background monitoring scheduler for governance checks.

Runs periodic asyncio tasks inside the FastAPI lifespan to automate every
monitoring obligation that would otherwise require manual API calls:

    Loop                     Default interval   Responsibility
    ────────────────────     ────────────────   ──────────────────────────────
    _freshness_loop          60 s               Record data freshness per source
    _calibration_loop         5 min             Persist CalibrationSnapshot
    _drift_loop              15 min             Run PSI drift check
    _alert_cleanup_loop      60 min             Expire stale alerts
    _outcome_backfill_loop    2 min             Back-fill actual_outcome on resolved bars

Each loop:
  • Creates its own DB session (independent of request sessions)
  • Catches all exceptions internally — a single failure never kills the loop
  • Logs at WARNING/ERROR on failure; DEBUG on normal operation

Usage (in FastAPI lifespan):
    scheduler = create_scheduler(symbols=["SPY", "QQQ"])
    await scheduler.start()
    yield
    await scheduler.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from app.core.database import AsyncSessionLocal  # noqa: E402 — imported at module level for patchability

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MonitoringConfig:
    """Intervals (seconds) and thresholds for each monitoring loop."""

    # How often each loop fires
    freshness_interval_s:        int = 60      # every minute
    calibration_interval_s:      int = 300     # every 5 min
    drift_interval_s:            int = 900     # every 15 min
    alert_cleanup_interval_s:    int = 3600    # every hour
    outcome_backfill_interval_s: int = 120     # every 2 min

    # Initial stagger delays so loops don't all hit the DB simultaneously at startup
    freshness_startup_delay_s:        int = 15
    calibration_startup_delay_s:      int = 30
    drift_startup_delay_s:            int = 60
    alert_cleanup_startup_delay_s:    int = 120
    outcome_backfill_startup_delay_s: int = 90

    # Symbols to monitor; overridden by create_scheduler()
    tracked_symbols: List[str] = field(default_factory=lambda: ["SPY"])

    # Minimum feature rows required to trigger a drift check
    min_rows_for_drift: int = 100

    # Rows fetched per drift window
    drift_window_bars: int = 300

    # How far past bar_open_time before we consider the bar resolved and ready
    # for outcome back-fill (needs to be > bar duration)
    outcome_lag_hours: float = 0.25   # 15 minutes after bar open

    # Maximum pending events to backfill in a single run
    outcome_batch_size: int = 100


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class MonitoringScheduler:
    """
    Manages five background asyncio tasks for governance monitoring.

    All tasks are created with asyncio.create_task() during start() and
    cancelled cleanly during stop().  The scheduler is designed for a single
    FastAPI process; do not start multiple instances.
    """

    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
        symbols: Optional[List[str]] = None,
    ) -> None:
        self._config = config or MonitoringConfig()
        if symbols:
            self._config.tracked_symbols = symbols
        self._tasks: List[asyncio.Task] = []
        self._running = False

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start all monitoring loops.  Idempotent."""
        if self._running:
            logger.warning("MonitoringScheduler.start() called while already running — no-op")
            return
        self._running = True
        self._tasks = [
            asyncio.create_task(self._freshness_loop(),        name="monitor:freshness"),
            asyncio.create_task(self._calibration_loop(),      name="monitor:calibration"),
            asyncio.create_task(self._drift_loop(),            name="monitor:drift"),
            asyncio.create_task(self._alert_cleanup_loop(),    name="monitor:alert_cleanup"),
            asyncio.create_task(self._outcome_backfill_loop(), name="monitor:outcome_backfill"),
        ]
        logger.info(
            "MonitoringScheduler started: %d symbols=%s",
            len(self._config.tracked_symbols),
            self._config.tracked_symbols,
        )

    async def stop(self) -> None:
        """Cancel all monitoring loops and wait for them to exit."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("MonitoringScheduler stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def task_names(self) -> List[str]:
        return [t.get_name() for t in self._tasks if not t.done()]

    # ------------------------------------------------------------------
    # Loop scaffolding
    # ------------------------------------------------------------------

    async def _freshness_loop(self) -> None:
        await asyncio.sleep(self._config.freshness_startup_delay_s)
        while self._running:
            try:
                await self._run_freshness_checks()
            except Exception as exc:
                logger.error("Freshness loop unhandled error: %s", exc, exc_info=True)
            await asyncio.sleep(self._config.freshness_interval_s)

    async def _calibration_loop(self) -> None:
        await asyncio.sleep(self._config.calibration_startup_delay_s)
        while self._running:
            try:
                await self._run_calibration_snapshots()
            except Exception as exc:
                logger.error("Calibration loop unhandled error: %s", exc, exc_info=True)
            await asyncio.sleep(self._config.calibration_interval_s)

    async def _drift_loop(self) -> None:
        await asyncio.sleep(self._config.drift_startup_delay_s)
        while self._running:
            try:
                await self._run_drift_checks()
            except Exception as exc:
                logger.error("Drift loop unhandled error: %s", exc, exc_info=True)
            await asyncio.sleep(self._config.drift_interval_s)

    async def _alert_cleanup_loop(self) -> None:
        await asyncio.sleep(self._config.alert_cleanup_startup_delay_s)
        while self._running:
            try:
                from app.governance.alerts import GovernanceAlertService
                async with AsyncSessionLocal() as db:
                    n = await GovernanceAlertService.clear_expired(db)
                    await db.commit()
                    if n:
                        logger.info("Alert cleanup: expired %d alerts", n)
            except Exception as exc:
                logger.error("Alert cleanup loop error: %s", exc, exc_info=True)
            await asyncio.sleep(self._config.alert_cleanup_interval_s)

    async def _outcome_backfill_loop(self) -> None:
        await asyncio.sleep(self._config.outcome_backfill_startup_delay_s)
        while self._running:
            try:
                await self._run_outcome_backfill()
            except Exception as exc:
                logger.error("Outcome backfill loop unhandled error: %s", exc, exc_info=True)
            await asyncio.sleep(self._config.outcome_backfill_interval_s)

    # ------------------------------------------------------------------
    # Task implementations
    # ------------------------------------------------------------------

    async def _run_freshness_checks(self) -> None:
        """Record data freshness for each (symbol, source) pair."""
        from app.governance.freshness import DataFreshnessService
        from app.governance.alerts import GovernanceAlertService
        try:
            async with AsyncSessionLocal() as db:
                for symbol in self._config.tracked_symbols:
                    last_bar_ts   = await self._get_last_bar_time(db, symbol)
                    last_quote_ts = await self._get_last_inference_time(db, symbol)

                    for source, last_ts in [
                        ("candle_data", last_bar_ts),
                        ("quote_feed",  last_quote_ts),
                    ]:
                        try:
                            row = await DataFreshnessService.record_check(
                                db,
                                symbol=symbol,
                                source=source,
                                last_data_ts=last_ts,
                            )
                            if row.is_stale and not row.alert_raised:
                                await GovernanceAlertService.alert_feed_stale(
                                    db, symbol, source, row.age_seconds or -1
                                )
                                row.alert_raised = True
                        except Exception as exc:
                            logger.warning(
                                "Freshness check failed for %s/%s: %s", symbol, source, exc
                            )
                await db.commit()
        except Exception as exc:
            logger.error("_run_freshness_checks error: %s", exc, exc_info=True)

    async def _run_calibration_snapshots(self) -> None:
        """Persist a CalibrationSnapshot for each tracked symbol."""
        from app.governance.calibration import CalibrationMonitor
        from app.governance.alerts import GovernanceAlertService

        try:
            from app.inference.confidence_tracker import get_tracker
            tracker = get_tracker()
        except Exception as exc:
            logger.debug("Calibration loop: tracker unavailable (%s) — skipping", exc)
            return

        try:
            async with AsyncSessionLocal() as db:
                for symbol in self._config.tracked_symbols:
                    try:
                        stats = tracker.get_stats(symbol)
                        await CalibrationMonitor.record_snapshot(
                            db, symbol=symbol, tracker_stats=stats
                        )
                        health = getattr(stats, "calibration_health", None)
                        if health == "degraded":
                            await GovernanceAlertService.alert_calibration_degraded(
                                db, symbol, health,
                                getattr(stats, "rolling_brier", None),
                            )
                        if getattr(stats, "needs_retrain", False):
                            reason = getattr(stats, "retrain_reason", None) or "threshold exceeded"
                            await GovernanceAlertService.alert_retrain_needed(db, symbol, reason)
                    except Exception as exc:
                        logger.warning("Calibration snapshot failed for %s: %s", symbol, exc)
                await db.commit()
        except Exception as exc:
            logger.error("_run_calibration_snapshots error: %s", exc, exc_info=True)

    async def _run_drift_checks(self) -> None:
        """Compute PSI drift for each tracked symbol and persist snapshot."""
        from app.governance.drift import DriftMonitor
        from app.governance.alerts import GovernanceAlertService

        try:
            async with AsyncSessionLocal() as db:
                for symbol in self._config.tracked_symbols:
                    try:
                        matrix, feature_names = await self._get_feature_matrix(db, symbol)
                        if matrix is None or len(matrix) < self._config.min_rows_for_drift:
                            logger.debug(
                                "Drift check: not enough rows for %s (have %d, need %d)",
                                symbol,
                                len(matrix) if matrix is not None else 0,
                                self._config.min_rows_for_drift,
                            )
                            continue

                        ref_stats, manifest_hash = await self._get_reference_stats(db)
                        psi_map = DriftMonitor.compute_psi_from_matrix(
                            matrix, feature_names, ref_stats
                        )
                        snap = await DriftMonitor.record_snapshot(
                            db,
                            symbol=symbol,
                            psi_by_feature=psi_map,
                            window_bars=len(matrix),
                            manifest_hash=manifest_hash,
                        )
                        if snap.drift_level != "none":
                            high_feats = json.loads(snap.high_drift_features_json or "[]")
                            await GovernanceAlertService.alert_drift(
                                db, symbol, snap.drift_level, snap.max_psi, high_feats
                            )
                            snap.alert_raised = True
                    except Exception as exc:
                        logger.warning("Drift check failed for %s: %s", symbol, exc)
                await db.commit()
        except Exception as exc:
            logger.error("_run_drift_checks error: %s", exc, exc_info=True)

    async def _run_outcome_backfill(self) -> None:
        """
        Back-fill actual_outcome for inference events whose bar has closed and settled.

        Strategy:
            1. Find InferenceEvent rows with actual_outcome IS NULL and
               bar_open_time < (now - outcome_lag_hours).
            2. For each, look up the OHLCV bar in the feature store.
            3. Determine direction: close > open → 1, else → 0.
            4. Write via InferenceLogService.record_outcome().
        """
        from app.governance.inference_log import InferenceLogService
        from app.governance.models import InferenceEvent
        from sqlalchemy import select

        cutoff = datetime.utcnow() - timedelta(hours=self._config.outcome_lag_hours)

        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(InferenceEvent)
                    .where(
                        InferenceEvent.actual_outcome.is_(None),
                        InferenceEvent.bar_open_time.is_not(None),
                        InferenceEvent.bar_open_time < cutoff,
                        InferenceEvent.action.in_(["buy", "sell"]),
                    )
                    .order_by(InferenceEvent.bar_open_time)
                    .limit(self._config.outcome_batch_size)
                )
                pending = result.scalars().all()
                if not pending:
                    return

                logger.debug("Outcome backfill: %d pending events", len(pending))

                filled = 0
                for event in pending:
                    outcome = await self._lookup_bar_outcome(
                        db, event.symbol, event.bar_open_time
                    )
                    if outcome is not None:
                        n = await InferenceLogService.record_outcome(
                            db,
                            symbol=event.symbol,
                            bar_open_time=event.bar_open_time,
                            actual_outcome=outcome,
                        )
                        filled += n

                if filled:
                    await db.commit()
                    logger.info("Outcome backfill: resolved %d events", filled)
        except Exception as exc:
            logger.error("_run_outcome_backfill error: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Data-access helpers (all return None on error — non-fatal)
    # ------------------------------------------------------------------

    async def _get_last_bar_time(self, db, symbol: str) -> Optional[datetime]:
        """Most recent bar_open_time in the feature store for symbol."""
        try:
            from sqlalchemy import select, func
            from app.models.feature_row import FeatureRow
            result = await db.execute(
                select(func.max(FeatureRow.bar_open_time))
                .where(FeatureRow.symbol == symbol)
            )
            return result.scalar_one_or_none()
        except Exception:
            return None

    async def _get_last_inference_time(self, db, symbol: str) -> Optional[datetime]:
        """
        Most recent inference event as a proxy for quote-feed freshness.
        Falls back gracefully if inference_events is empty.
        """
        try:
            from sqlalchemy import select, func
            from app.governance.models import InferenceEvent
            result = await db.execute(
                select(func.max(InferenceEvent.created_at))
                .where(InferenceEvent.symbol == symbol)
            )
            return result.scalar_one_or_none()
        except Exception:
            return None

    async def _get_feature_matrix(
        self, db, symbol: str
    ) -> Tuple[Optional["np.ndarray"], List[str]]:
        """
        Fetch recent feature rows and return (matrix, feature_names).
        Returns (None, []) if insufficient data or model not available.
        """
        try:
            import numpy as np
            from sqlalchemy import select
            from app.models.feature_row import FeatureRow

            result = await db.execute(
                select(FeatureRow)
                .where(FeatureRow.symbol == symbol)
                .order_by(FeatureRow.bar_open_time.desc())
                .limit(self._config.drift_window_bars)
            )
            rows = result.scalars().all()
            if len(rows) < 30:
                return None, []

            feature_names: List[str] = []
            matrices = []
            for r in rows:
                feats = json.loads(r.features_json) if r.features_json else {}
                if not feature_names and feats:
                    feature_names = list(feats.keys())
                if feature_names:
                    matrices.append([feats.get(k, float("nan")) for k in feature_names])

            if not matrices:
                return None, []
            return __import__("numpy").array(matrices, dtype=float), feature_names
        except Exception as exc:
            logger.debug("Feature matrix fetch failed for %s: %s", symbol, exc)
            return None, []

    async def _get_reference_stats(self, db):
        """Return (reference_stats_dict | None, manifest_hash | None)."""
        try:
            from app.feature_pipeline.registry import MANIFEST_HASH
            from app.governance.registry import FeatureRegistryService
            fv = await FeatureRegistryService.get(db, MANIFEST_HASH)
            ref = json.loads(fv.reference_stats_json) if fv and fv.reference_stats_json else None
            return ref, MANIFEST_HASH
        except Exception:
            return None, None

    async def _lookup_bar_outcome(
        self, db, symbol: str, bar_open_time: datetime
    ) -> Optional[int]:
        """
        Return 1 (up) or 0 (down) for the bar at bar_open_time, or None if unavailable.

        Looks for 'close' > 'open' in the feature row JSON.
        Feature names tried: close/open, bar_close/bar_open.
        """
        try:
            from sqlalchemy import select
            from app.models.feature_row import FeatureRow

            result = await db.execute(
                select(FeatureRow)
                .where(
                    FeatureRow.symbol == symbol,
                    FeatureRow.bar_open_time == bar_open_time,
                )
                .limit(1)
            )
            row = result.scalar_one_or_none()
            if row is None or not row.features_json:
                return None

            feats = json.loads(row.features_json)
            open_px  = feats.get("open")  or feats.get("bar_open")
            close_px = feats.get("close") or feats.get("bar_close")
            if open_px is not None and close_px is not None:
                return 1 if float(close_px) > float(open_px) else 0
        except Exception as exc:
            logger.debug(
                "Bar outcome lookup failed for %s @ %s: %s", symbol, bar_open_time, exc
            )
        return None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_scheduler: Optional[MonitoringScheduler] = None


def get_scheduler() -> MonitoringScheduler:
    """Return the process-level scheduler singleton (creates with defaults if needed)."""
    global _scheduler
    if _scheduler is None:
        _scheduler = MonitoringScheduler()
    return _scheduler


def create_scheduler(
    config: Optional[MonitoringConfig] = None,
    symbols: Optional[List[str]] = None,
) -> MonitoringScheduler:
    """Create and register the process-level singleton. Call once at startup."""
    global _scheduler
    _scheduler = MonitoringScheduler(config=config, symbols=symbols)
    return _scheduler
