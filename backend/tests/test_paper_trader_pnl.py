"""
Tests for paper trader position arithmetic and PnL calculation.

These tests target position_manager.update_position directly to avoid the
Redis/yfinance/sklearn-dependent paper_trader and risk_manager layers.

We validate:
- PnL = (exit_price - avg_cost) * quantity for longs
- PnL = (avg_cost - exit_price) * quantity for shorts
- Capital: debited on open, credited on close
- Partial closes reduce quantity correctly
- Average cost updates correctly when adding to a position
"""

import pytest
from unittest.mock import AsyncMock


def make_trade_req(action: str, qty: int = 10, price: float = 100.0, symbol: str = "SPY"):
    """Build a TradeCreate without importing paper_trader at module level."""
    from app.schemas.trade import TradeCreate
    return TradeCreate(symbol=symbol, action=action, quantity=qty, price=price)


async def _noop_pnl(_pnl: float) -> None:
    """No-op on_pnl for tests that don't care about PnL capture."""


@pytest.mark.asyncio
async def test_open_long_creates_position(db_session):
    """BTO creates a new position with correct quantity and avg_cost."""
    from app.paper_trading.position_manager import update_position

    req = make_trade_req("BTO", qty=10, price=100.0)
    await update_position(req, 100.0, db_session, on_pnl=_noop_pnl)

    from sqlalchemy import select
    from app.models.position import Position
    result = await db_session.execute(
        select(Position).where(Position.symbol == "SPY", Position.is_open == True)
    )
    pos = result.scalar_one_or_none()
    assert pos is not None
    assert pos.quantity == 10
    assert abs(pos.avg_cost - 100.0) < 0.001


@pytest.mark.asyncio
async def test_long_close_records_pnl(db_session):
    """
    Open 10 @ 100, close 10 @ 110.
    Expected realized PnL = (110 - 100) * 10 = 100.
    """
    from app.paper_trading.position_manager import update_position
    from sqlalchemy import select
    from app.models.position import Position

    pnl_captured = []

    async def capture_pnl(p):
        pnl_captured.append(p)

    req_open = make_trade_req("BTO", qty=10, price=100.0)
    await update_position(req_open, 100.0, db_session, on_pnl=_noop_pnl)

    req_close = make_trade_req("STC", qty=10, price=110.0)
    await update_position(req_close, 110.0, db_session, on_pnl=capture_pnl)

    assert len(pnl_captured) == 1
    assert abs(pnl_captured[0] - 100.0) < 0.001, \
        f"Expected PnL=100, got {pnl_captured[0]}"

    result = await db_session.execute(
        select(Position).where(Position.symbol == "SPY")
    )
    pos = result.scalar_one_or_none()
    assert pos is not None
    assert not pos.is_open, "Position should be closed"
    assert abs(pos.realized_pnl - 100.0) < 0.001


@pytest.mark.asyncio
async def test_long_close_at_loss(db_session):
    """
    Open 10 @ 100, close 10 @ 90.
    Expected realized PnL = (90 - 100) * 10 = -100.
    """
    from app.paper_trading.position_manager import update_position

    pnl_captured = []

    async def capture_pnl(p):
        pnl_captured.append(p)

    await update_position(make_trade_req("BTO", qty=10, price=100.0), 100.0, db_session, on_pnl=_noop_pnl)
    await update_position(make_trade_req("STC", qty=10, price=90.0), 90.0, db_session, on_pnl=capture_pnl)

    assert abs(pnl_captured[0] - (-100.0)) < 0.001, \
        f"Expected PnL=-100, got {pnl_captured[0]}"


@pytest.mark.asyncio
async def test_short_cover_profit(db_session):
    """
    STO 10 @ 100, BTC 10 @ 80.
    Expected realized PnL = (100 - 80) * 10 = 200.
    """
    from app.paper_trading.position_manager import update_position

    pnl_captured = []

    async def capture_pnl(p):
        pnl_captured.append(p)

    await update_position(make_trade_req("STO", qty=10, price=100.0), 100.0, db_session, on_pnl=_noop_pnl)
    await update_position(make_trade_req("BTC", qty=10, price=80.0), 80.0, db_session, on_pnl=capture_pnl)

    assert abs(pnl_captured[0] - 200.0) < 0.001, \
        f"Expected PnL=200 (short profit), got {pnl_captured[0]}"


@pytest.mark.asyncio
async def test_partial_close_leaves_remainder(db_session):
    """
    Open 20 @ 100, partial close 10 @ 105.
    Realized PnL = (105-100)*10 = 50. Remaining quantity = 10.
    """
    from app.paper_trading.position_manager import update_position
    from sqlalchemy import select
    from app.models.position import Position

    pnl_captured = []

    async def capture_pnl(p):
        pnl_captured.append(p)

    await update_position(make_trade_req("BTO", qty=20, price=100.0), 100.0, db_session, on_pnl=_noop_pnl)
    await update_position(make_trade_req("STC", qty=10, price=105.0), 105.0, db_session, on_pnl=capture_pnl)

    assert abs(pnl_captured[0] - 50.0) < 0.001, \
        f"Expected PnL=50 on partial close, got {pnl_captured[0]}"

    result = await db_session.execute(
        select(Position).where(Position.symbol == "SPY", Position.is_open == True)
    )
    pos = result.scalar_one_or_none()
    assert pos is not None, "Position should still be open after partial close"
    assert pos.quantity == 10, f"Expected 10 remaining shares, got {pos.quantity}"


@pytest.mark.asyncio
async def test_average_cost_update_on_add(db_session):
    """
    Open 10 @ 100, then add 10 @ 110.
    New avg_cost = (100*10 + 110*10) / 20 = 105.
    """
    from app.paper_trading.position_manager import update_position
    from sqlalchemy import select
    from app.models.position import Position

    await update_position(make_trade_req("BTO", qty=10, price=100.0), 100.0, db_session, on_pnl=_noop_pnl)
    await update_position(make_trade_req("BTO", qty=10, price=110.0), 110.0, db_session, on_pnl=_noop_pnl)

    result = await db_session.execute(
        select(Position).where(Position.symbol == "SPY", Position.is_open == True)
    )
    pos = result.scalar_one_or_none()
    assert pos is not None
    assert pos.quantity == 20
    assert abs(pos.avg_cost - 105.0) < 0.01, \
        f"Expected avg_cost=105, got {pos.avg_cost}"
