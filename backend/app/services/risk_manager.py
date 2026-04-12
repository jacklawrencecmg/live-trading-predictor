import asyncio
import json
from datetime import datetime, date
from typing import Optional

from app.core.config import settings
from app.core.redis_client import get_redis


DAILY_PNL_KEY = "risk:daily_pnl:{date}"
LAST_TRADE_KEY = "risk:last_trade:{symbol}"
KILL_SWITCH_KEY = "risk:kill_switch"
CAPITAL_KEY = "risk:capital"


class RiskViolation(Exception):
    pass


async def get_capital() -> float:
    redis = await get_redis()
    val = await redis.get(CAPITAL_KEY)
    return float(val) if val else settings.starting_capital


async def set_capital(capital: float):
    redis = await get_redis()
    await redis.set(CAPITAL_KEY, str(capital))


async def get_daily_pnl() -> float:
    redis = await get_redis()
    key = DAILY_PNL_KEY.format(date=date.today().isoformat())
    val = await redis.get(key)
    return float(val) if val else 0.0


async def add_pnl(pnl: float):
    redis = await get_redis()
    key = DAILY_PNL_KEY.format(date=date.today().isoformat())
    val = await redis.get(key)
    current = float(val) if val else 0.0
    await redis.setex(key, 86400, str(current + pnl))


async def is_kill_switch_active() -> bool:
    redis = await get_redis()
    val = await redis.get(KILL_SWITCH_KEY)
    return val == "1" or settings.kill_switch


async def set_kill_switch(active: bool):
    redis = await get_redis()
    if active:
        await redis.set(KILL_SWITCH_KEY, "1")
    else:
        await redis.delete(KILL_SWITCH_KEY)


async def record_trade_time(symbol: str):
    redis = await get_redis()
    key = LAST_TRADE_KEY.format(symbol=symbol)
    await redis.setex(key, settings.cooldown_minutes * 60, datetime.utcnow().isoformat())


async def check_cooldown(symbol: str) -> bool:
    """Returns True if cooldown is active (trade blocked)."""
    redis = await get_redis()
    key = LAST_TRADE_KEY.format(symbol=symbol)
    val = await redis.get(key)
    return val is not None


async def check_all_risks(
    symbol: str,
    trade_value: float,
) -> None:
    """Raises RiskViolation if any check fails."""
    # Check both the Redis kill switch (this module) and the governance DB-backed
    # kill switch (app.governance.kill_switch) so that activating either one halts
    # trading.  The governance cached check is TTL-based (5 s) and adds no I/O.
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
        # Auto-activate kill switch
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
