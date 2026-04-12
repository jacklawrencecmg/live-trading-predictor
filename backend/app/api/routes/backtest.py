from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from app.core.database import get_db
from app.models.backtest import BacktestResult
from app.schemas.backtest import BacktestRequest, BacktestResultOut
from app.services.backtest_service import run_backtest

router = APIRouter()


@router.post("/run", response_model=BacktestResultOut)
async def run_backtest_route(req: BacktestRequest, db: AsyncSession = Depends(get_db)):
    result = await run_backtest(req, db)
    return BacktestResultOut.model_validate(result)


@router.get("/results", response_model=List[BacktestResultOut])
async def list_backtest_results(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(BacktestResult).order_by(BacktestResult.created_at.desc()).limit(20)
    )
    return [BacktestResultOut.model_validate(r) for r in result.scalars().all()]
