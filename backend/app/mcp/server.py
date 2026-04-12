"""
options-research MCP server.

Exposes live project state to Claude Code via the Model Context Protocol.
Runs as a stdio subprocess — Claude Code connects to it automatically
when configured in .claude/settings.json.

Tools exposed (developer workflow, not trading path):
    get_risk_summary        — capital, daily P&L, kill switch state
    get_governance_alerts   — active alerts by severity
    get_latest_inference    — most recent inference result for a symbol
    get_market_quote        — live quote via the configured market data provider
    get_options_chain       — live options chain for a symbol
    get_system_health       — aggregated health: feeds, model, calibration, kill switch
    get_active_positions    — current paper trading positions

This server connects to the running application's DB and Redis. It requires
the same environment variables as the main application (.env file at repo root).

Usage:
    python -m app.mcp.server              # run directly (stdio)

Claude Code configuration (.claude/settings.json):
    "mcpServers": {
        "options-research": {
            "command": "python",
            "args": ["-m", "app.mcp.server"],
            "cwd": "backend",
            "env": { "PYTHONPATH": "." }
        }
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import date, datetime
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Bootstrap: ensure backend/ is importable when run as __main__
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print(
        "ERROR: 'mcp' package not installed.\n"
        "Run: pip install mcp\n"
        "See: https://github.com/modelcontextprotocol/python-sdk",
        file=sys.stderr,
    )
    sys.exit(1)

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "options-research",
    instructions=(
        "Access live state of the options-research paper trading system. "
        "Use these tools to inspect risk state, governance alerts, inference results, "
        "and market data during development and debugging. "
        "These tools are read-only except where noted."
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json(obj: Any) -> str:
    """Serialize to JSON, handling datetime objects."""
    def default(o):
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        return str(o)
    return json.dumps(obj, indent=2, default=default)


async def _get_db_session():
    """Return an async DB session. Caller must close it."""
    from app.core.database import AsyncSessionLocal
    return AsyncSessionLocal()


# ---------------------------------------------------------------------------
# Tool: get_risk_summary
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_risk_summary() -> str:
    """
    Return the current risk manager state: capital, daily P&L,
    max loss limit, kill switch status, and cooldown configuration.

    Use this when debugging trade execution, checking whether the kill switch
    is active, or understanding the current capital allocation.
    """
    try:
        from app.services.risk_manager import get_risk_summary as _risk_summary
        summary = await _risk_summary()
        return _json(summary)
    except Exception as exc:
        return _json({"error": str(exc), "note": "Is the application running with Redis connected?"})


# ---------------------------------------------------------------------------
# Tool: get_governance_alerts
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_governance_alerts(
    severity: Optional[str] = None,
    limit: int = 20,
) -> str:
    """
    Return recent governance alerts from the database.

    Args:
        severity: Filter by "info", "warning", or "critical". None returns all.
        limit: Maximum number of alerts to return (default 20, max 100).

    Use this when investigating model degradation, drift alerts, calibration
    warnings, or kill switch events.
    """
    limit = min(limit, 100)
    try:
        from sqlalchemy import desc, select
        from app.governance.models import GovernanceAlert

        async with await _get_db_session() as db:
            q = select(GovernanceAlert).order_by(desc(GovernanceAlert.triggered_at)).limit(limit)
            if severity:
                q = q.where(GovernanceAlert.severity == severity)
            result = await db.execute(q)
            alerts = result.scalars().all()

        return _json([
            {
                "id": a.id,
                "alert_type": a.alert_type,
                "title": a.title,
                "severity": a.severity,
                "symbol": a.symbol,
                "triggered_at": a.triggered_at,
                "resolved_at": a.resolved_at,
                "details": a.details,
            }
            for a in alerts
        ])
    except Exception as exc:
        return _json({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: get_latest_inference
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_latest_inference(symbol: str = "SPY") -> str:
    """
    Return the most recent inference result stored for a symbol.

    Args:
        symbol: Ticker symbol (default "SPY").

    Returns the full 4-layer uncertainty bundle: raw probability,
    calibrated probability, tradeable confidence, and action.
    Use this to verify inference is running correctly or to inspect
    what signal was generated for a given symbol.
    """
    try:
        from sqlalchemy import desc, select
        from app.models.inference_log import InferenceLog

        async with await _get_db_session() as db:
            result = await db.execute(
                select(InferenceLog)
                .where(InferenceLog.symbol == symbol.upper())
                .order_by(desc(InferenceLog.timestamp))
                .limit(1)
            )
            row = result.scalar_one_or_none()

        if row is None:
            return _json({"symbol": symbol, "status": "no_inference_found"})

        return _json({
            "symbol": row.symbol,
            "timestamp": row.timestamp,
            "bar_open_time": row.bar_open_time,
            "raw_prob_up": row.raw_prob_up,
            "calibrated_prob_up": row.calibrated_prob_up,
            "tradeable_confidence": row.tradeable_confidence,
            "degradation_factor": row.degradation_factor,
            "action": row.action,
            "abstain_reason": row.abstain_reason,
            "calibration_health": row.calibration_health,
            "regime": row.regime,
            "model_version": row.model_version,
            "feature_snapshot_id": row.feature_snapshot_id,
        })
    except Exception as exc:
        return _json({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: get_market_quote
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_market_quote(symbol: str = "SPY") -> str:
    """
    Fetch a live market quote for a symbol using the configured provider.

    Args:
        symbol: Ticker symbol.

    Use this to verify that market data is flowing correctly, or to get
    the current price context when debugging inference or options decisions.
    """
    try:
        from app.services.market_data import fetch_quote
        quote = await fetch_quote(symbol.upper())
        return _json(quote)
    except Exception as exc:
        return _json({"error": str(exc), "symbol": symbol})


# ---------------------------------------------------------------------------
# Tool: get_options_chain
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_options_chain(symbol: str = "SPY", expiry: Optional[str] = None) -> str:
    """
    Fetch the options chain for a symbol.

    Args:
        symbol: Ticker symbol.
        expiry: ISO date string "YYYY-MM-DD". None selects the nearest expiry.

    Returns spot price, ATM IV, IV rank, and the first 5 calls and puts
    closest to ATM (not the full chain — use the /api/options/chain endpoint
    for the full chain).
    """
    try:
        from app.services.options_service import fetch_options_chain
        chain = await fetch_options_chain(symbol.upper(), expiry)

        # Return a summary (full chain is large)
        spot = chain.get("spot", 0)
        calls = sorted(chain.get("calls", []), key=lambda c: abs(c.get("strike", 0) - spot))[:5]
        puts  = sorted(chain.get("puts",  []), key=lambda p: abs(p.get("strike", 0) - spot))[:5]

        return _json({
            "symbol": chain.get("symbol"),
            "spot": spot,
            "expiry": chain.get("expiry"),
            "atm_iv": chain.get("atm_iv"),
            "iv_rank": chain.get("iv_rank"),
            "snapshot_time": chain.get("snapshot_time"),
            "nearest_calls": calls,
            "nearest_puts": puts,
            "total_calls": len(chain.get("calls", [])),
            "total_puts": len(chain.get("puts", [])),
        })
    except Exception as exc:
        return _json({"error": str(exc), "symbol": symbol})


# ---------------------------------------------------------------------------
# Tool: get_system_health
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_system_health() -> str:
    """
    Return an aggregated health snapshot of the trading system.

    Checks:
      - Kill switch state (Redis + governance DB)
      - Latest model version and calibration health
      - Data feed freshness (most recent bar timestamp)
      - Active critical alerts count
      - Daily P&L vs limit

    Use this as the first call when diagnosing production issues.
    """
    health: dict = {}

    # Kill switch
    try:
        from app.services.risk_manager import is_kill_switch_active
        health["kill_switch_redis"] = await is_kill_switch_active()
    except Exception as e:
        health["kill_switch_redis"] = f"error: {e}"

    try:
        from app.governance.kill_switch import KillSwitchService
        health["kill_switch_governance_cached"] = KillSwitchService.is_active_cached()
    except Exception as e:
        health["kill_switch_governance_cached"] = f"error: {e}"

    # Risk summary
    try:
        from app.services.risk_manager import get_capital, get_daily_pnl
        from app.core.config import settings
        capital = await get_capital()
        daily_pnl = await get_daily_pnl()
        health["capital"] = capital
        health["daily_pnl"] = daily_pnl
        health["daily_pnl_pct"] = round(daily_pnl / capital * 100, 2) if capital else 0
        health["max_daily_loss_pct"] = settings.max_daily_loss_pct * 100
    except Exception as e:
        health["risk"] = f"error: {e}"

    # Active critical alerts
    try:
        from sqlalchemy import func, select
        from app.governance.models import GovernanceAlert

        async with await _get_db_session() as db:
            result = await db.execute(
                select(func.count(GovernanceAlert.id))
                .where(GovernanceAlert.severity == "critical")
                .where(GovernanceAlert.resolved_at.is_(None))
            )
            health["unresolved_critical_alerts"] = result.scalar() or 0
    except Exception as e:
        health["critical_alerts"] = f"error: {e}"

    # Most recent bar
    try:
        from sqlalchemy import desc, select
        from app.models.ohlcv import OHLCVBar

        async with await _get_db_session() as db:
            result = await db.execute(
                select(OHLCVBar.bar_open_time, OHLCVBar.symbol)
                .order_by(desc(OHLCVBar.bar_open_time))
                .limit(1)
            )
            row = result.one_or_none()
            if row:
                health["latest_bar"] = {"symbol": row.symbol, "bar_open_time": row.bar_open_time}
                age_seconds = (datetime.utcnow() - row.bar_open_time).total_seconds()
                health["latest_bar_age_minutes"] = round(age_seconds / 60, 1)
                health["feed_status"] = "ok" if age_seconds < 600 else "stale"
            else:
                health["feed_status"] = "no_bars"
    except Exception as e:
        health["feed"] = f"error: {e}"

    health["checked_at"] = datetime.utcnow().isoformat()
    return _json(health)


# ---------------------------------------------------------------------------
# Tool: get_active_positions
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_active_positions() -> str:
    """
    Return current open paper trading positions from the database.

    Use this to verify what the system currently holds, check position
    sizes against risk limits, or understand PnL attribution.
    """
    try:
        from sqlalchemy import select
        from app.models.position import Position

        async with await _get_db_session() as db:
            result = await db.execute(
                select(Position).where(Position.status == "open")
            )
            positions = result.scalars().all()

        return _json([
            {
                "id": p.id,
                "symbol": p.symbol,
                "side": p.side,
                "quantity": float(p.quantity),
                "avg_cost": float(p.avg_cost) if p.avg_cost else None,
                "opened_at": p.opened_at,
            }
            for p in positions
        ])
    except Exception as exc:
        return _json({"error": str(exc)})


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load .env from the repo root (two levels above backend/app/mcp/)
    from pathlib import Path
    env_path = Path(__file__).parents[3] / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)

    mcp.run(transport="stdio")
