"""
Alert system for paper-trading signals.

Alert types:
- HIGH_CONFIDENCE_BULLISH: strong buy signal
- HIGH_CONFIDENCE_BEARISH: strong sell signal
- NO_TRADE_RISK: blocked by risk controls
- MODEL_HEALTH_DEGRADED: Brier score / accuracy degraded
- DATA_FEED_STALE: no new bars ingested
- DAILY_LOSS_LIMIT: hit the max daily loss

Channels: database log (default) + extensible callback system.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Callable

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class AlertType(str, Enum):
    HIGH_CONFIDENCE_BULLISH = "high_confidence_bullish"
    HIGH_CONFIDENCE_BEARISH = "high_confidence_bearish"
    NO_TRADE_RISK = "no_trade_risk"
    MODEL_HEALTH_DEGRADED = "model_health_degraded"
    DATA_FEED_STALE = "data_feed_stale"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    KILL_SWITCH = "kill_switch"
    GENERAL = "general"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


SEVERITY_MAP = {
    AlertType.HIGH_CONFIDENCE_BULLISH: AlertSeverity.INFO,
    AlertType.HIGH_CONFIDENCE_BEARISH: AlertSeverity.INFO,
    AlertType.NO_TRADE_RISK: AlertSeverity.WARNING,
    AlertType.MODEL_HEALTH_DEGRADED: AlertSeverity.WARNING,
    AlertType.DATA_FEED_STALE: AlertSeverity.WARNING,
    AlertType.DAILY_LOSS_LIMIT: AlertSeverity.CRITICAL,
    AlertType.KILL_SWITCH: AlertSeverity.CRITICAL,
}

# Pluggable notification callbacks: fn(alert_type, message, details) -> None
_notification_channels: List[Callable] = []


def register_channel(fn: Callable):
    """Register a notification callback. fn(alert_type, message, details)."""
    _notification_channels.append(fn)


async def fire_alert(
    db: Optional[AsyncSession],
    alert_type: AlertType,
    symbol: Optional[str],
    message: str,
    details: Optional[Dict[str, Any]] = None,
):
    """Fire an alert: log it, store in DB, call registered channels."""
    severity = SEVERITY_MAP.get(alert_type, AlertSeverity.INFO)
    log_fn = logger.critical if severity == AlertSeverity.CRITICAL else (
        logger.warning if severity == AlertSeverity.WARNING else logger.info
    )
    log_fn("ALERT [%s] %s | %s", alert_type.value, symbol or "", message)

    # Store in audit_log
    if db is not None:
        from app.models.audit_log import AuditLog
        log = AuditLog(
            event_type=f"alert:{alert_type.value}",
            symbol=symbol,
            details=details or {},
            message=message,
        )
        db.add(log)

    # Notify channels
    for channel in _notification_channels:
        try:
            channel(alert_type, message, details or {})
        except Exception as e:
            logger.error("Notification channel error: %s", e)


async def check_and_fire_signal_alerts(
    db: Optional[AsyncSession],
    symbol: str,
    prob_up: float,
    prob_down: float,
    confidence: float,
    no_trade_reason: Optional[str],
    thresholds: dict = None,
):
    """Check signal conditions and fire appropriate alerts."""
    thresholds = thresholds or {}
    high_conf = thresholds.get("high_confidence", 0.70)

    if no_trade_reason and "risk" in no_trade_reason:
        await fire_alert(db, AlertType.NO_TRADE_RISK, symbol,
                         f"Trade blocked: {no_trade_reason}",
                         {"no_trade_reason": no_trade_reason})
    elif prob_up > high_conf:
        await fire_alert(db, AlertType.HIGH_CONFIDENCE_BULLISH, symbol,
                         f"Bullish signal: P(up)={prob_up:.2f} confidence={confidence:.2f}",
                         {"prob_up": prob_up, "confidence": confidence})
    elif prob_down > high_conf:
        await fire_alert(db, AlertType.HIGH_CONFIDENCE_BEARISH, symbol,
                         f"Bearish signal: P(down)={prob_down:.2f} confidence={confidence:.2f}",
                         {"prob_down": prob_down, "confidence": confidence})


async def check_model_health(
    db: Optional[AsyncSession],
    recent_brier: float,
    baseline_brier: float = 0.25,
    threshold_multiplier: float = 1.3,
):
    """Fire alert if Brier score has degraded significantly."""
    if recent_brier > baseline_brier * threshold_multiplier:
        await fire_alert(
            db, AlertType.MODEL_HEALTH_DEGRADED, None,
            f"Model degraded: Brier={recent_brier:.4f} vs baseline={baseline_brier:.4f}",
            {"recent_brier": recent_brier, "baseline_brier": baseline_brier},
        )
