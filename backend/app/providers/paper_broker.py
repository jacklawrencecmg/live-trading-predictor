"""
PaperBrokerProvider — wraps the existing paper_trader.py logic.

This is always is_paper=True. It executes trades against the DB and Redis
state exactly as the current paper_trader.py does.

When connecting to a live or sandbox broker (Alpaca, IBKR), create a new
class implementing BrokerProvider and swap it in at startup. Do not modify
PaperBrokerProvider to handle live orders — keep the separation explicit.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from .protocols import AccountInfo, BrokerError, BrokerProvider, OrderRequest, OrderResult

logger = logging.getLogger(__name__)


class PaperBrokerProvider:
    """
    Implements BrokerProvider by delegating to paper_trader.py and risk_manager.py.
    Maintains the existing in-DB + Redis execution path.
    """

    @property
    def is_paper(self) -> bool:
        return True

    async def submit_order(self, order: OrderRequest) -> OrderResult:
        """
        Wraps execute_paper_trade(). The risk gate (check_all_risks) is
        called inside paper_trader.py — BrokerProvider does not need to
        call it separately.
        """
        from app.core.database import AsyncSessionLocal
        from app.schemas.trades import TradeCreate
        from app.services.paper_trader import execute_paper_trade

        trade_req = TradeCreate(
            symbol=order.symbol,
            action=order.side,
            quantity=order.quantity,
        )
        try:
            async with AsyncSessionLocal() as db:
                result = await execute_paper_trade(trade_req, db)
                await db.commit()
        except Exception as exc:
            raise BrokerError(
                str(exc),
                order_id=order.client_order_id,
                retryable=False,
            ) from exc

        return OrderResult(
            order_id=str(result.id),
            symbol=order.symbol,
            status="filled",
            filled_qty=float(order.quantity),
            avg_fill_price=result.price,
            submitted_at=datetime.utcnow(),
            filled_at=datetime.utcnow(),
        )

    async def cancel_order(self, order_id: str) -> bool:
        # Paper orders fill synchronously — nothing to cancel.
        logger.info("cancel_order: paper trades fill synchronously; nothing to cancel")
        return False

    async def get_positions(self) -> List[dict]:
        from app.core.database import AsyncSessionLocal
        from app.services.position_manager import get_open_positions
        async with AsyncSessionLocal() as db:
            positions = await get_open_positions(db)
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.quantity),
                "avg_cost": float(p.avg_cost) if p.avg_cost else None,
                "market_value": None,  # paper trader does not track MTM
                "side": p.side,
            }
            for p in positions
        ]

    async def get_account(self) -> AccountInfo:
        from app.services.risk_manager import get_capital, get_daily_pnl
        capital = await get_capital()
        daily_pnl = await get_daily_pnl()
        return AccountInfo(
            account_id="paper-account",
            cash=capital + daily_pnl,
            portfolio_value=capital + daily_pnl,
            buying_power=capital + daily_pnl,
            is_paper=True,
        )
