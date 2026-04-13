"""
Structured JSON logging defaults for the options-research platform.

Usage
-----
Call ``configure_logging()`` once at application startup (e.g., in ``main.py``
before the FastAPI app is created).  Every log record emitted after that point
will be serialised as a single-line JSON object, making it trivially parseable
by log-aggregation tools (Loki, Datadog, CloudWatch, etc.).

Format
------
{
  "ts":      "2026-04-12T14:22:01.123456Z",   # ISO-8601 UTC
  "level":   "INFO",
  "logger":  "app.inference.inference_service",
  "msg":     "inference complete",
  "request_id": "abc-123",                    # only present if bound
  ... (any extra kwargs passed to the log call)
}

Request-scoped context
----------------------
Use ``bind_request_context(request_id=..., symbol=...)`` inside a FastAPI
dependency or middleware to attach fields to every log record emitted during
that request.  The context is stored in a ``contextvars.ContextVar`` and is
never shared between concurrent requests.

    from app.core.logging_config import bind_request_context
    ctx_token = bind_request_context(request_id="abc-123", symbol="SPY")
    try:
        ...
    finally:
        reset_request_context(ctx_token)
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

# ── Context variable for per-request fields ──────────────────────────────────
_request_ctx: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "_request_ctx", default={}
)


def bind_request_context(**kwargs: Any) -> contextvars.Token:  # type: ignore[type-arg]
    """Attach extra fields to every log record in the current async context."""
    current = dict(_request_ctx.get())
    current.update(kwargs)
    return _request_ctx.set(current)


def reset_request_context(token: contextvars.Token) -> None:  # type: ignore[type-arg]
    """Remove the bound context fields (call in a ``finally`` block)."""
    _request_ctx.reset(token)


# ── JSON formatter ────────────────────────────────────────────────────────────
class _JsonFormatter(logging.Formatter):
    """Serialise every LogRecord to a single-line JSON string."""

    # Fields from LogRecord that we always include explicitly — everything else
    # lands in the record's __dict__ but we only forward known extra kwargs.
    _SKIP_ATTRS = frozenset(
        {
            "args",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "taskName",
            "thread",
            "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        # Base fields
        doc: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="microseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Exception info
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)

        # Extra kwargs passed to the log call (e.g., logger.info("x", extra={"k": v}))
        for key, val in record.__dict__.items():
            if key not in self._SKIP_ATTRS and not key.startswith("_"):
                doc[key] = val

        # Per-request context fields (override any same-named extras)
        doc.update(_request_ctx.get())

        return json.dumps(doc, default=str)


# ── Public entry point ────────────────────────────────────────────────────────
def configure_logging(
    level: str = "INFO",
    *,
    json_logs: bool = True,
    force: bool = False,
) -> None:
    """
    Configure root logger with structured JSON output.

    Parameters
    ----------
    level:
        Minimum log level (DEBUG / INFO / WARNING / ERROR / CRITICAL).
    json_logs:
        When True (default) use the JSON formatter.
        Set False for local development if you prefer human-readable output.
    force:
        Re-configure even if handlers are already attached (useful in tests).
    """
    root = logging.getLogger()

    if root.handlers and not force:
        return  # Already configured — don't double-register handlers.

    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if json_logs:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )

    root.addHandler(handler)
    root.setLevel(level)

    # Silence noisy third-party loggers
    for noisy in ("uvicorn.access", "sqlalchemy.engine", "httpx", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
