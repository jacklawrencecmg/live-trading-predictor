import json
from datetime import datetime
from typing import List, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.trade import Trade
from app.models.position import Position
from app.models.audit_log import AuditLog
from app.schemas.trade import TradeCreate, TradeOut, PositionOut, PortfolioSummary
from app.services.risk_manager import (
    check_all_risks, get_capital, set_capital, add_pnl, get_risk_summary,
    record_trade_time, RiskViolation, is_kill_switch_active,
)
from app.services.market_data import fetch_quote
from app.paper_trading.position_manager import update_position as _update_position_impl


async def execute_paper_trade(
    trade_req: TradeCreate,
    db: AsyncSession,
    model_prob_up: Optional[float] = None,
    model_prob_down: Optional[float] = None,
    model_confidence: Optional[float] = None,
) -> TradeOut:
    # Get price
    if trade_req.price:
        price = trade_req.price
    else:
        quote = await fetch_quote(trade_req.symbol)
        price = quote.price

    trade_value = price * abs(trade_req.quantity) * 100 if trade_req.option_symbol else price * abs(trade_req.quantity)

    # Risk check
    await check_all_risks(trade_req.symbol, trade_value)

    trade = Trade(
        symbol=trade_req.symbol,
        option_symbol=trade_req.option_symbol,
        action=trade_req.action,
        quantity=trade_req.quantity,
        price=price,
        underlying_price=None,
        strike=trade_req.strike,
        expiry=trade_req.expiry,
        option_type=trade_req.option_type,
        model_prob_up=model_prob_up,
        model_prob_down=model_prob_down,
        model_confidence=model_confidence,
        is_paper=True,
    )
    db.add(trade)

    # Update position
    await _update_position(trade_req, price, db)

    # Update capital (debit on open, credit on close)
    capital = await get_capital()
    is_opening = trade_req.action in ("BTO", "STO")
    if is_opening:
        await set_capital(capital - trade_value)
    else:
        # Credit back the close proceeds; realized PnL is tracked in _update_position
        await set_capital(capital + trade_value)

    # Record cooldown
    await record_trade_time(trade_req.symbol)

    # Audit log
    log = AuditLog(
        event_type="paper_trade",
        symbol=trade_req.symbol,
        details={
            "action": trade_req.action,
            "quantity": trade_req.quantity,
            "price": price,
            "trade_value": trade_value,
            "model_confidence": model_confidence,
        },
        message=f"Paper trade: {trade_req.action} {trade_req.quantity} {trade_req.symbol} @ {price:.2f}",
    )
    db.add(log)
    await db.flush()

    return TradeOut(
        id=trade.id,
        symbol=trade.symbol,
        option_symbol=trade.option_symbol,
        action=trade.action,
        quantity=trade.quantity,
        price=trade.price,
        executed_at=trade.executed_at,
        model_prob_up=model_prob_up,
        model_prob_down=model_prob_down,
        model_confidence=model_confidence,
    )


async def _update_position(trade_req: TradeCreate, price: float, db: AsyncSession):
    """Delegates to position_manager.update_position, wiring in add_pnl from risk_manager."""
    await _update_position_impl(trade_req, price, db, on_pnl=add_pnl)


async def get_positions(db: AsyncSession) -> List[PositionOut]:
    result = await db.execute(select(Position).where(Position.is_open == True))
    positions = result.scalars().all()
    return [PositionOut.model_validate(p) for p in positions]


async def get_trades(db: AsyncSession, limit: int = 100) -> List[TradeOut]:
    result = await db.execute(select(Trade).order_by(Trade.executed_at.desc()).limit(limit))
    trades = result.scalars().all()
    return [TradeOut.model_validate(t) for t in trades]


async def get_portfolio_summary(db: AsyncSession) -> PortfolioSummary:
    risk = await get_risk_summary()
    capital = risk["capital"]
    daily_pnl = risk["daily_pnl"]
    kill = risk["kill_switch"]

    result = await db.execute(select(Position).where(Position.is_open == True))
    positions = result.scalars().all()

    pos_value = sum(
        (p.current_price or p.avg_cost) * abs(p.quantity)
        for p in positions
    )
    total_value = capital + pos_value
    total_pnl = total_value - settings.starting_capital

    return PortfolioSummary(
        capital=round(capital, 2),
        cash=round(capital, 2),
        positions_value=round(pos_value, 2),
        total_value=round(total_value, 2),
        daily_pnl=round(daily_pnl, 2),
        daily_pnl_pct=round(risk["daily_pnl_pct"], 6),
        total_pnl=round(total_pnl, 2),
        open_positions=len(positions),
        kill_switch_active=kill,
    )
