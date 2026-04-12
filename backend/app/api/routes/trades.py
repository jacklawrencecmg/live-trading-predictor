from fastapi import APIRouter, Depends, HTTPException
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
