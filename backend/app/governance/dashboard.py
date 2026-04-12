"""
Governance summary dashboard.

GovernanceDashboard.summary() aggregates all governance subsystems into a
single response that answers: "Is the system trustworthy right now?"

A skeptical reviewer should be able to answer all of these from the response:
    - Is the kill switch active?
    - Which model version is live?
    - What is the current calibration health per symbol?
    - Are any feeds stale?
    - Is feature drift detected?
    - Are there active critical alerts?
    - How many inferences ran in the last 24h and what was the abstain rate?
    - Which symbols need model retraining?
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.governance.alerts import GovernanceAlertService
from app.governance.calibration import CalibrationMonitor
from app.governance.drift import DriftMonitor
from app.governance.freshness import DataFreshnessService
from app.governance.inference_log import InferenceLogService
from app.governance.kill_switch import KillSwitchService
from app.governance.registry import ModelRegistryService

logger = logging.getLogger(__name__)

# Model names this system tracks
_TRACKED_MODELS = ("logistic", "gbt", "random_forest")


class GovernanceDashboard:

    @staticmethod
    async def summary(db: AsyncSession) -> Dict[str, Any]:
        """
        Aggregate all governance signals into one dict.
        Non-fatal errors in sub-queries are caught and reported as null fields
        so a partial failure does not block the dashboard.
        """
        now = datetime.utcnow()

        # ---- Kill switch -------------------------------------------------
        ks_active = False
        ks_reason: Optional[str] = None
        try:
            ks_active = await KillSwitchService.is_active_db(db)
            ks_row = await KillSwitchService.get_state(db)
            ks_reason = ks_row.reason if ks_active else None
        except Exception as e:
            logger.error("dashboard: kill switch error: %s", e)

        # ---- Active model -----------------------------------------------
        active_model: Optional[Dict[str, Any]] = None
        active_manifest: Optional[str] = None
        for mname in _TRACKED_MODELS:
            try:
                ver = await ModelRegistryService.get_active(db, mname)
                if ver:
                    active_model = {
                        "id":           ver.id,
                        "model_name":   ver.model_name,
                        "version_tag":  ver.version_tag,
                        "promoted_at":  ver.promoted_at.isoformat() if ver.promoted_at else None,
                        "artifact_sha256": ver.artifact_sha256,
                    }
                    active_manifest = ver.feature_manifest_hash
                    break
            except Exception as e:
                logger.error("dashboard: model registry error: %s", e)

        # ---- Calibration health per symbol -------------------------------
        cal_health: Dict[str, str] = {}
        retrain_symbols: List[str] = []
        try:
            cal_health = await CalibrationMonitor.health_by_symbol(db)
            retrain_symbols = await CalibrationMonitor.symbols_needing_retrain(db)
        except Exception as e:
            logger.error("dashboard: calibration error: %s", e)

        # ---- Drift summary -----------------------------------------------
        drift_summary: Dict[str, str] = {}
        try:
            drift_summary = await DriftMonitor.summary_all_symbols(db)
        except Exception as e:
            logger.error("dashboard: drift error: %s", e)

        # ---- Stale feeds -------------------------------------------------
        stale_feeds: List[Dict[str, Any]] = []
        try:
            checks = await DataFreshnessService.get_stale_feeds(db, since_minutes=15)
            stale_feeds = [
                {
                    "symbol":     c.symbol,
                    "source":     c.source,
                    "age_seconds": c.age_seconds,
                    "threshold":  c.staleness_threshold_seconds,
                }
                for c in checks
            ]
        except Exception as e:
            logger.error("dashboard: freshness error: %s", e)

        # ---- Alerts ------------------------------------------------------
        crit_count = 0
        warn_count = 0
        recent_titles: List[str] = []
        try:
            active_alerts = await GovernanceAlertService.get_active(db)
            crit_count = sum(1 for a in active_alerts if a.severity == "critical")
            warn_count = sum(1 for a in active_alerts if a.severity == "warning")
            recent_titles = [a.title for a in active_alerts[:5]]
        except Exception as e:
            logger.error("dashboard: alerts error: %s", e)

        # ---- Inference volume 24h ----------------------------------------
        inf_count_24h = 0
        abstain_rate: Optional[float] = None
        try:
            counts = await InferenceLogService.count_24h(db)
            inf_count_24h = sum(counts.values())
            total = inf_count_24h
            if total > 0:
                abstained = counts.get("abstain", 0)
                abstain_rate = round(abstained / total, 4)
        except Exception as e:
            logger.error("dashboard: inference count error: %s", e)

        return {
            "generated_at":           now.isoformat(),
            "kill_switch_active":     ks_active,
            "kill_switch_reason":     ks_reason,
            "active_model":           active_model,
            "active_feature_manifest": active_manifest,
            "calibration_health":     cal_health,
            "symbols_needing_retrain": retrain_symbols,
            "drift_summary":          drift_summary,
            "stale_feeds":            stale_feeds,
            "active_critical_alerts": crit_count,
            "active_warning_alerts":  warn_count,
            "recent_alert_titles":    recent_titles,
            "inference_count_24h":    inf_count_24h,
            "abstain_rate_24h":       abstain_rate,
        }

    @staticmethod
    async def rolling_performance(
        db: AsyncSession,
        symbol: str,
        window_days: int = 7,
    ) -> Dict[str, Any]:
        """
        Aggregate inference accuracy + calibration trend for one symbol.
        """
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=window_days)

        accuracy_stats: Dict[str, Any] = {}
        try:
            accuracy_stats = await InferenceLogService.get_accuracy_stats(
                db, symbol, window=500
            )
        except Exception as e:
            logger.error("rolling_performance: accuracy error: %s", e)

        cal_history: List[Any] = []
        trend = "unknown"
        latest_snap = None
        try:
            cal_history = await CalibrationMonitor.get_history(db, symbol, limit=30)
            trend = CalibrationMonitor.trend_direction(list(reversed(cal_history)))
            latest_snap = cal_history[0] if cal_history else None
        except Exception as e:
            logger.error("rolling_performance: calibration error: %s", e)

        drift_snap = None
        try:
            drift_snap = await DriftMonitor.get_latest(db, symbol)
        except Exception as e:
            logger.error("rolling_performance: drift error: %s", e)

        return {
            "symbol":          symbol,
            "window_days":     window_days,
            "generated_at":    datetime.utcnow().isoformat(),
            "accuracy":        accuracy_stats,
            "calibration_trend": trend,
            "latest_calibration": {
                "snapshot_at":     latest_snap.snapshot_at.isoformat() if latest_snap else None,
                "rolling_brier":   latest_snap.rolling_brier if latest_snap else None,
                "degradation_factor": latest_snap.degradation_factor if latest_snap else None,
                "calibration_health": latest_snap.calibration_health if latest_snap else None,
                "needs_retrain":   latest_snap.needs_retrain if latest_snap else None,
            } if latest_snap else None,
            "latest_drift": {
                "drift_level": drift_snap.drift_level if drift_snap else "unknown",
                "max_psi":     drift_snap.max_psi if drift_snap else None,
                "computed_at": drift_snap.computed_at.isoformat() if drift_snap else None,
            } if drift_snap else None,
        }
