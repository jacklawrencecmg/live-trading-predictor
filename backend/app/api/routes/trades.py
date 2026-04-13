from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional

from app.core.database import get_db
from app.schemas.trade import TradeCreate, TradeOut, PositionOut, PortfolioSummary
from app.services.paper_trader import execute_paper_trade, get_positions, get_trades, get_portfolio_summary
from app.services.risk_manager import set_kill_switch, get_risk_summary, RiskViolation

router = APIRouter()


@router.post("/execute", response_model=TradeOut)
async def execute_trade(
    trade: TradeCreate,
    model_prob_up: Optional[float] = None,
    model_prob_down: Optional[float] = None,
    model_confidence: Optional[float] = None,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await execute_paper_trade(trade, db, model_prob_up, model_prob_down, model_confidence)
    except RiskViolation as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/positions", response_model=List[PositionOut])
async def list_positions(db: AsyncSession = Depends(get_db)):
    return await get_positions(db)


@router.get("/history", response_model=List[TradeOut])
async def trade_history(limit: int = 100, db: AsyncSession = Depends(get_db)):
    return await get_trades(db, limit)


@router.get("/portfolio", response_model=PortfolioSummary)
async def portfolio(db: AsyncSession = Depends(get_db)):
    return await get_portfolio_summary(db)


@router.post("/kill-switch")
async def toggle_kill_switch(active: bool):
    await set_kill_switch(active)
    return {"kill_switch": active}


@router.get("/risk")
async def risk_summary():
    return await get_risk_summary()


@router.get("/pnl-summary")
async def pnl_summary(db: AsyncSession = Depends(get_db)):
    """Rolling P&L summary computed from closed trades."""
    from app.models.trade import Trade
    cutoff_30d = date.today() - timedelta(days=30)
    cutoff_7d  = date.today() - timedelta(days=7)

    result = await db.execute(
        select(Trade).where(
            Trade.executed_at >= cutoff_30d,
            Trade.action.in_(["BTC", "STC"]),  # closing trades only
        ).order_by(Trade.executed_at)
    )
    trades = result.scalars().all()

    risk = await get_risk_summary()
    daily_pnl = risk["daily_pnl"]

    # Aggregate realized PnL from closing trades using fill price × quantity
    # (positions hold realized_pnl; this is a proxy from trade records)
    pnls_30d = [t.price * abs(t.quantity) * (-1 if t.action == "BTC" else 1) for t in trades]
    trades_7d = [t for t in trades if t.executed_at.date() >= cutoff_7d]
    pnls_7d   = [t.price * abs(t.quantity) * (-1 if t.action == "BTC" else 1) for t in trades_7d]

    wins  = [p for p in pnls_30d if p > 0]
    losses = [p for p in pnls_30d if p < 0]

    import math
    sharpe_7d = None
    if len(pnls_7d) >= 2:
        mean = sum(pnls_7d) / len(pnls_7d)
        std  = math.sqrt(sum((p - mean) ** 2 for p in pnls_7d) / len(pnls_7d))
        sharpe_7d = (mean / std * math.sqrt(252)) if std > 0 else None

    return {
        "daily_realized": daily_pnl,
        "daily_unrealized": 0.0,
        "rolling_7d":  round(sum(pnls_7d), 2),
        "rolling_30d": round(sum(pnls_30d), 2),
        "win_rate_30d": len(wins) / len(pnls_30d) if pnls_30d else None,
        "trades_30d": len(pnls_30d),
        "avg_win":  round(sum(wins) / len(wins), 2) if wins else None,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else None,
        "sharpe_7d": round(sharpe_7d, 3) if sharpe_7d is not None else None,
    }
