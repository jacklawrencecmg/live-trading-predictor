import logging
from datetime import datetime, date
from typing import Optional

from app.core.config import settings
from app.core.redis_client import get_redis  # module-level so tests can patch it

logger = logging.getLogger(__name__)

DAILY_PNL_KEY = "risk:daily_pnl:{date}"
LAST_TRADE_KEY = "risk:last_trade:{symbol}"
KILL_SWITCH_KEY = "risk:kill_switch"
CAPITAL_KEY = "risk:capital"


class RiskViolation(Exception):
    pass


# ---------------------------------------------------------------------------
# Redis-resilient helpers — all reads return safe defaults on connection error.
# Writes log a warning and no-op so the app stays functional without Redis.
# ---------------------------------------------------------------------------

async def get_capital() -> float:
    try:
        r = await get_redis()
        val = await r.get(CAPITAL_KEY)
        return float(val) if val else settings.starting_capital
    except Exception as exc:
        logger.warning("get_capital: Redis unavailable (%s) — using starting capital", exc)
        return settings.starting_capital


async def set_capital(capital: float):
    try:
        r = await get_redis()
        await r.set(CAPITAL_KEY, str(capital))
    except Exception as exc:
        logger.warning("set_capital: Redis unavailable (%s) — state not persisted", exc)


async def get_daily_pnl() -> float:
    try:
        r = await get_redis()
        key = DAILY_PNL_KEY.format(date=date.today().isoformat())
        val = await r.get(key)
        return float(val) if val else 0.0
    except Exception as exc:
        logger.warning("get_daily_pnl: Redis unavailable (%s) — returning 0", exc)
        return 0.0


async def add_pnl(pnl: float):
    try:
        r = await get_redis()
        key = DAILY_PNL_KEY.format(date=date.today().isoformat())
        val = await r.get(key)
        current = float(val) if val else 0.0
        await r.setex(key, 86400, str(current + pnl))
    except Exception as exc:
        logger.warning("add_pnl: Redis unavailable (%s) — PnL not recorded", exc)


async def is_kill_switch_active() -> bool:
    try:
        r = await get_redis()
        val = await r.get(KILL_SWITCH_KEY)
        return val == "1" or settings.kill_switch
    except Exception as exc:
        logger.warning("is_kill_switch_active: Redis unavailable (%s) — using settings default", exc)
        return settings.kill_switch


async def set_kill_switch(active: bool):
    try:
        r = await get_redis()
        if active:
            await r.set(KILL_SWITCH_KEY, "1")
        else:
            await r.delete(KILL_SWITCH_KEY)
    except Exception as exc:
        logger.warning("set_kill_switch: Redis unavailable (%s) — state not persisted", exc)


async def record_trade_time(symbol: str):
    try:
        r = await get_redis()
        key = LAST_TRADE_KEY.format(symbol=symbol)
        await r.setex(key, settings.cooldown_minutes * 60, datetime.utcnow().isoformat())
    except Exception as exc:
        logger.warning("record_trade_time: Redis unavailable (%s)", exc)


async def check_cooldown(symbol: str) -> bool:
    """Returns True if cooldown is active (trade blocked)."""
    try:
        r = await get_redis()
        key = LAST_TRADE_KEY.format(symbol=symbol)
        val = await r.get(key)
        return val is not None
    except Exception as exc:
        logger.warning("check_cooldown: Redis unavailable (%s) — cooldown not enforced", exc)
        return False


async def check_all_risks(symbol: str, trade_value: float) -> None:
    """Raises RiskViolation if any check fails."""
    if await is_kill_switch_active():
        raise RiskViolation("Kill switch is active — all trading halted")
    try:
        from app.governance.kill_switch import KillSwitchService as _GovKS
        if _GovKS.is_active_cached():
            raise RiskViolation("Governance kill switch is active — all trading halted")
    except RiskViolation:
        raise
    except Exception:
        pass  # governance module unavailable — degrade gracefully

    capital = await get_capital()
    daily_pnl = await get_daily_pnl()
    max_daily_loss = capital * settings.max_daily_loss_pct
    if daily_pnl < -max_daily_loss:
        await set_kill_switch(True)
        raise RiskViolation(
            f"Max daily loss breached: PnL={daily_pnl:.2f}, limit={-max_daily_loss:.2f}"
        )

    max_position = capital * settings.max_position_size_pct
    if trade_value > max_position:
        raise RiskViolation(
            f"Position size {trade_value:.2f} exceeds max {max_position:.2f}"
        )

    if await check_cooldown(symbol):
        raise RiskViolation(f"Cooldown active for {symbol} — wait {settings.cooldown_minutes} min")


async def get_risk_summary() -> dict:
    capital = await get_capital()
    daily_pnl = await get_daily_pnl()
    kill = await is_kill_switch_active()
    return {
        "capital": capital,
        "daily_pnl": daily_pnl,
        "daily_pnl_pct": daily_pnl / capital if capital else 0,
        "max_daily_loss": capital * settings.max_daily_loss_pct,
        "max_position_size": capital * settings.max_position_size_pct,
        "kill_switch": kill,
        "cooldown_minutes": settings.cooldown_minutes,
    }
