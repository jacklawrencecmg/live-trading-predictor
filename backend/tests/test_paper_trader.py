import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from app.schemas.trade import TradeCreate

pytestmark = pytest.mark.risk_critical


@pytest.mark.asyncio
async def test_execute_paper_trade(db_session):
    trade_req = TradeCreate(symbol="SPY", action="BTO", quantity=10, price=450.0)

    with patch("app.services.paper_trader.check_all_risks", new_callable=AsyncMock):
        with patch("app.services.paper_trader.get_capital", return_value=100_000.0):
            with patch("app.services.paper_trader.set_capital", new_callable=AsyncMock):
                with patch("app.services.paper_trader.add_pnl", new_callable=AsyncMock):
                    with patch("app.services.paper_trader.record_trade_time", new_callable=AsyncMock):
                        from app.services.paper_trader import execute_paper_trade
                        result = await execute_paper_trade(trade_req, db_session)

    assert result.symbol == "SPY"
    assert result.price == 450.0
    assert result.quantity == 10
