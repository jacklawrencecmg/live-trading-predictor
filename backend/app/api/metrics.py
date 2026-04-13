"""
Prometheus-compatible metrics endpoint.

GET /metrics  →  text/plain; version=0.0.4 (Prometheus exposition format)

No prometheus_client dependency — writes the text format directly.
This keeps the dependency footprint minimal and the format easy to audit.

Metric naming follows Prometheus conventions:
    options_research_{metric_name}

All metrics are gauges (point-in-time values) scraped on demand.
The endpoint is non-blocking: each scrape runs independent DB queries.

Usage in Prometheus scrape config:
    - job_name: options_research
      static_configs:
        - targets: ['localhost:8000']
      metrics_path: /metrics
      scrape_interval: 30s
"""

from __future__ import annotations

import logging
import time
from typing import List, Tuple

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db

router = APIRouter()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text format helpers
# ---------------------------------------------------------------------------

MetricLines = List[str]


def _gauge(name: str, value, help_text: str, labels: dict = None) -> MetricLines:
    """Render one HELP/TYPE/metric line-set for a gauge."""
    label_str = ""
    if labels:
        kv = ",".join(f'{k}="{v}"' for k, v in labels.items())
        label_str = f"{{{kv}}}"
    v = 0 if value is None else (1 if value is True else (0 if value is False else value))
    return [
        f"# HELP {name} {help_text}",
        f"# TYPE {name} gauge",
        f"{name}{label_str} {v}",
    ]


def _counter(name: str, value, help_text: str) -> MetricLines:
    v = 0 if value is None else value
    return [
        f"# HELP {name} {help_text}",
        f"# TYPE {name} counter",
        f"{name}_total {v}",
    ]


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    include_in_schema=False,   # don't pollute the OpenAPI docs
    tags=["ops"],
)
async def prometheus_metrics(db: AsyncSession = Depends(get_db)) -> PlainTextResponse:
    """
    Prometheus-format metrics for the governance and operational plane.

    Designed for scraping by Prometheus, Grafana Agent, or any
    OpenMetrics-compatible collector.
    """
    lines: List[str] = []
    scrape_start = time.time()

    # ── Kill switch ───────────────────────────────────────────────────────────
    try:
        from app.governance.kill_switch import KillSwitchService
        ks_active = await KillSwitchService.is_active_db(db)
        lines += _gauge(
            "options_research_kill_switch_active",
            int(ks_active),
            "1 if the trading kill switch is active (all inference blocked), 0 otherwise",
        )
    except Exception as exc:
        logger.warning("metrics: kill_switch query failed: %s", exc)

    # ── Active alerts ─────────────────────────────────────────────────────────
    try:
        from app.governance.alerts import GovernanceAlertService
        active_alerts = await GovernanceAlertService.get_active(db)
        crit = sum(1 for a in active_alerts if a.severity == "critical")
        warn = sum(1 for a in active_alerts if a.severity == "warning")
        info = sum(1 for a in active_alerts if a.severity == "info")
        for sev, cnt in [("critical", crit), ("warning", warn), ("info", info)]:
            lines += _gauge(
                "options_research_active_alerts",
                cnt,
                "Number of active governance alerts by severity",
                labels={"severity": sev},
            )
    except Exception as exc:
        logger.warning("metrics: alerts query failed: %s", exc)

    # ── Inference volume (24 h) ───────────────────────────────────────────────
    try:
        from app.governance.inference_log import InferenceLogService
        counts_24h = await InferenceLogService.count_24h(db)
        total_24h = sum(counts_24h.values())
        abstain_24h = counts_24h.get("abstain", 0)
        lines += _gauge(
            "options_research_inference_events_24h",
            total_24h,
            "Total inference events in the last 24 hours",
        )
        lines += _gauge(
            "options_research_inference_abstain_rate_24h",
            round(abstain_24h / total_24h, 4) if total_24h else 0,
            "Fraction of 24-hour inference events that resulted in abstain",
        )
        for action, cnt in counts_24h.items():
            lines += _gauge(
                "options_research_inference_by_action_24h",
                cnt,
                "Inference event count in last 24 hours by action",
                labels={"action": action or "unknown"},
            )
    except Exception as exc:
        logger.warning("metrics: inference count query failed: %s", exc)

    # ── Calibration health ────────────────────────────────────────────────────
    try:
        from app.governance.calibration import CalibrationMonitor
        health_map = await CalibrationMonitor.health_by_symbol(db)
        _HEALTH_INT = {"good": 0, "fair": 1, "caution": 2, "degraded": 3, "unknown": -1}
        for symbol, health in health_map.items():
            lines += _gauge(
                "options_research_calibration_health",
                _HEALTH_INT.get(health, -1),
                "Calibration health encoded: 0=good 1=fair 2=caution 3=degraded -1=unknown",
                labels={"symbol": symbol},
            )

        # Degradation factor from most recent snapshot
        for symbol in health_map:
            snap = await CalibrationMonitor.get_latest(db, symbol)
            if snap and snap.degradation_factor is not None:
                lines += _gauge(
                    "options_research_calibration_degradation_factor",
                    round(snap.degradation_factor, 4),
                    "Rolling calibration degradation factor (1.0=no degradation; lower=worse)",
                    labels={"symbol": symbol},
                )
            if snap and snap.rolling_brier is not None:
                lines += _gauge(
                    "options_research_calibration_rolling_brier",
                    round(snap.rolling_brier, 4),
                    "Rolling Brier score (lower is better; 0.25=random)",
                    labels={"symbol": symbol},
                )
    except Exception as exc:
        logger.warning("metrics: calibration query failed: %s", exc)

    # ── Drift ─────────────────────────────────────────────────────────────────
    try:
        from app.governance.drift import DriftMonitor
        drift_map = await DriftMonitor.summary_all_symbols(db)
        _DRIFT_INT = {"none": 0, "moderate": 1, "high": 2}
        for symbol, drift_level in drift_map.items():
            lines += _gauge(
                "options_research_drift_level",
                _DRIFT_INT.get(drift_level, -1),
                "Feature drift level encoded: 0=none 1=moderate 2=high",
                labels={"symbol": symbol},
            )

        # Per-symbol max PSI
        for symbol in drift_map:
            snap = await DriftMonitor.get_latest(db, symbol)
            if snap:
                lines += _gauge(
                    "options_research_drift_max_psi",
                    round(snap.max_psi, 4),
                    "Maximum Population Stability Index across all features (0.10=moderate 0.25=high)",
                    labels={"symbol": symbol},
                )
    except Exception as exc:
        logger.warning("metrics: drift query failed: %s", exc)

    # ── Data freshness ────────────────────────────────────────────────────────
    try:
        from app.governance.freshness import DataFreshnessService
        stale = await DataFreshnessService.get_stale_feeds(db, since_minutes=5)
        lines += _gauge(
            "options_research_stale_feeds",
            len(stale),
            "Number of data feeds that are currently stale (checked in last 5 min)",
        )
    except Exception as exc:
        logger.warning("metrics: freshness query failed: %s", exc)

    # ── Model registry ────────────────────────────────────────────────────────
    try:
        from app.governance.registry import ModelRegistryService
        _TRACKED = ("logistic", "gbt", "random_forest")
        for mname in _TRACKED:
            ver = await ModelRegistryService.get_active(db, mname)
            lines += _gauge(
                "options_research_model_active",
                1 if ver else 0,
                "1 if a model has an active (promoted) version, 0 if none",
                labels={"model": mname},
            )
    except Exception as exc:
        logger.warning("metrics: model registry query failed: %s", exc)

    # ── Scrape latency (self-instrumentation) ─────────────────────────────────
    scrape_ms = round((time.time() - scrape_start) * 1000, 1)
    lines += _gauge(
        "options_research_metrics_scrape_duration_ms",
        scrape_ms,
        "Duration of the last /metrics scrape in milliseconds",
    )

    return PlainTextResponse(
        "\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
