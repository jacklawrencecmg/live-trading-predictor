"""
AlertChannel implementations.

Registered at startup via GovernanceAlertService.register_channel().
Multiple channels can be active simultaneously — all registered channels
receive every alert.

Channels must never raise. A broken notification channel must not affect
the trading path or kill-switch reliability.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from .protocols import AlertChannel, AlertPayload

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# NullAlertChannel — default in dev/test
# ─────────────────────────────────────────────────────────────────────────────

class NullAlertChannel:
    """No-op channel. Use in tests and dev environments."""

    async def dispatch(self, alert: AlertPayload) -> None:
        logger.debug("NullAlertChannel: %s [%s] %s", alert.severity, alert.alert_type, alert.title)


# ─────────────────────────────────────────────────────────────────────────────
# WebhookAlertChannel — generic HTTP POST
# ─────────────────────────────────────────────────────────────────────────────

class WebhookAlertChannel:
    """
    Posts alert payloads as JSON to a configurable URL.
    Compatible with Zapier, Make.com, n8n, custom ingestion endpoints.

    Environment variables:
        ALERT_WEBHOOK_URL      — required
        ALERT_WEBHOOK_SECRET   — optional; sent as X-Webhook-Secret header
        ALERT_MIN_SEVERITY     — "info" | "warning" | "critical" (default: "warning")
    """

    def __init__(
        self,
        url: str,
        secret: Optional[str] = None,
        min_severity: str = "warning",
    ):
        self.url = url
        self.secret = secret
        self._severity_rank = {"info": 0, "warning": 1, "critical": 2}
        self._min_rank = self._severity_rank.get(min_severity, 1)

    async def dispatch(self, alert: AlertPayload) -> None:
        rank = self._severity_rank.get(alert.severity, 0)
        if rank < self._min_rank:
            return  # below threshold — skip

        try:
            import httpx
            headers = {"Content-Type": "application/json"}
            if self.secret:
                headers["X-Webhook-Secret"] = self.secret

            payload = {
                "alert_type": alert.alert_type,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity,
                "symbol": alert.symbol,
                "details": alert.details,
                "timestamp": alert.timestamp.isoformat(),
            }

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(self.url, json=payload, headers=headers)
                if resp.status_code >= 400:
                    logger.warning(
                        "WebhookAlertChannel: HTTP %s for alert %s",
                        resp.status_code, alert.alert_type,
                    )
        except Exception as exc:
            # Must not propagate — log and swallow
            logger.error("WebhookAlertChannel dispatch failed: %s", exc)

    @classmethod
    def from_env(cls) -> Optional["WebhookAlertChannel"]:
        """Construct from environment variables. Returns None if URL not set."""
        import os
        url = os.getenv("ALERT_WEBHOOK_URL")
        if not url:
            return None
        return cls(
            url=url,
            secret=os.getenv("ALERT_WEBHOOK_SECRET"),
            min_severity=os.getenv("ALERT_MIN_SEVERITY", "warning"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# SlackAlertChannel — Slack incoming webhook
# ─────────────────────────────────────────────────────────────────────────────

class SlackAlertChannel:
    """
    Posts to a Slack channel via incoming webhook URL.

    Environment variables:
        ALERT_SLACK_WEBHOOK_URL    — required (Slack incoming webhook)
        ALERT_SLACK_MIN_SEVERITY   — "info" | "warning" | "critical" (default: "warning")

    Setup: Slack App → Incoming Webhooks → Add New Webhook to Workspace.
    """

    _SEVERITY_EMOJI = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
    _SEVERITY_COLOR = {"info": "#36a64f", "warning": "#ff9800", "critical": "#d32f2f"}

    def __init__(self, webhook_url: str, min_severity: str = "warning"):
        self.webhook_url = webhook_url
        self._severity_rank = {"info": 0, "warning": 1, "critical": 2}
        self._min_rank = self._severity_rank.get(min_severity, 1)

    async def dispatch(self, alert: AlertPayload) -> None:
        rank = self._severity_rank.get(alert.severity, 0)
        if rank < self._min_rank:
            return

        try:
            import httpx
            emoji = self._SEVERITY_EMOJI.get(alert.severity, "📢")
            color = self._SEVERITY_COLOR.get(alert.severity, "#888888")

            text = f"{emoji} *{alert.title}*"
            if alert.symbol:
                text += f" `{alert.symbol}`"

            body = {
                "attachments": [
                    {
                        "color": color,
                        "title": alert.title,
                        "text": alert.message,
                        "fields": [
                            {"title": "Severity", "value": alert.severity, "short": True},
                            {"title": "Type", "value": alert.alert_type, "short": True},
                        ],
                        "footer": "options-research",
                        "ts": int(alert.timestamp.timestamp()),
                    }
                ]
            }

            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(self.webhook_url, json=body)
        except Exception as exc:
            logger.error("SlackAlertChannel dispatch failed: %s", exc)

    @classmethod
    def from_env(cls) -> Optional["SlackAlertChannel"]:
        import os
        url = os.getenv("ALERT_SLACK_WEBHOOK_URL")
        if not url:
            return None
        return cls(url, min_severity=os.getenv("ALERT_SLACK_MIN_SEVERITY", "warning"))


# ─────────────────────────────────────────────────────────────────────────────
# Factory — build channels from environment at startup
# ─────────────────────────────────────────────────────────────────────────────

def build_alert_channels_from_env() -> list:
    """
    Returns a list of all AlertChannels configured via environment variables.
    If no channels are configured, returns [NullAlertChannel()].

    Call at application startup and register each channel:
        for ch in build_alert_channels_from_env():
            GovernanceAlertService.register_channel(ch)
    """
    channels = []

    webhook = WebhookAlertChannel.from_env()
    if webhook:
        channels.append(webhook)
        logger.info("Alert channel registered: WebhookAlertChannel -> %s", webhook.url)

    slack = SlackAlertChannel.from_env()
    if slack:
        channels.append(slack)
        logger.info("Alert channel registered: SlackAlertChannel")

    if not channels:
        channels.append(NullAlertChannel())
        logger.info("Alert channel registered: NullAlertChannel (no env channels configured)")

    return channels
