"""
Position arithmetic — pure DB layer, no Redis or external service deps.

Separated from paper_trader.py so it can be unit-tested without redis,
yfinance, sklearn, or any other heavy import.

The caller is responsible for persisting PnL to the risk ledger; this module
accepts an `on_pnl` coroutine so tests can inject a no-op or capture calls.
"""

from datetime import datetime
from typing import Awaitable, Callable, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.position import Position
from app.schemas.trade import TradeCreate


async def update_position(
    trade_req: TradeCreate,
    price: float,
    db: AsyncSession,
    on_pnl: Callable[[float], Awaitable[None]],
) -> None:
    """
    Update or create a position record after a trade.

    Sign convention:
    - BTO / STO: qty > 0 for long, qty < 0 for short (using signed quantity)
    - BTC / STC: reduces the position toward zero

    Realized PnL on full or partial close:
      pnl = (exit_price - avg_cost) * closed_qty          (long)
      pnl = (avg_cost - exit_price) * abs(closed_qty)     (short)

    Args:
        trade_req: The incoming trade request.
        price:     Execution price.
        db:        Async DB session; caller manages flush/commit.
        on_pnl:    Coroutine called with the realized PnL on any closing trade.
                   Injected so callers can wire to Redis (production) or a no-op
                   (tests).
    """
    result = await db.execute(
        select(Position).where(
            Position.symbol == trade_req.symbol,
            Position.option_symbol == trade_req.option_symbol,
            Position.is_open == True,
        )
    )
    pos = result.scalar_one_or_none()

    # Map action → signed quantity delta
    action = trade_req.action
    if action == "BTO":
        qty_delta = abs(trade_req.quantity)       # add long
    elif action == "STO":
        qty_delta = -abs(trade_req.quantity)      # add short
    elif action == "BTC":
        qty_delta = abs(trade_req.quantity)       # buy back (reduce short)
    elif action == "STC":
        qty_delta = -abs(trade_req.quantity)      # sell out (reduce long)
    else:
        qty_delta = trade_req.quantity

    if pos is None:
        pos = Position(
            symbol=trade_req.symbol,
            option_symbol=trade_req.option_symbol,
            quantity=qty_delta,
            avg_cost=price,
            current_price=price,
            strike=trade_req.strike,
            expiry=trade_req.expiry,
            option_type=trade_req.option_type,
        )
        db.add(pos)
    else:
        total_qty = pos.quantity + qty_delta
        is_reducing = (pos.quantity > 0 and qty_delta < 0) or (
            pos.quantity < 0 and qty_delta > 0
        )

        if is_reducing:
            closed_qty = min(abs(pos.quantity), abs(qty_delta))
            if pos.quantity > 0:
                pnl = (price - pos.avg_cost) * closed_qty
            else:
                pnl = (pos.avg_cost - price) * closed_qty
            await on_pnl(pnl)
            pos.realized_pnl += pnl

        if total_qty == 0:
            pos.is_open = False
            pos.closed_at = datetime.utcnow()
        else:
            if not is_reducing:
                # Adding to existing position: weighted average cost
                new_cost = (
                    pos.avg_cost * abs(pos.quantity) + price * abs(qty_delta)
                ) / abs(total_qty)
                pos.avg_cost = new_cost
            pos.quantity = total_qty
            pos.current_price = price
            if pos.quantity > 0:
                pos.unrealized_pnl = (price - pos.avg_cost) * pos.quantity
            else:
                pos.unrealized_pnl = (pos.avg_cost - price) * abs(pos.quantity)
