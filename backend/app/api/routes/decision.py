"""
Decision API.

GET /api/decision/{symbol}
    Run full options decision layer for the current bar.
    Returns a scored OptionsDecision with all four candidate structures,
    recommended structure (or abstain), IV analysis, and confidence score.

Query parameters:
    timeframe         : bar timeframe (default "5m")
    confidence_threshold : minimum calibrated probability edge (default 0.55)
    atm_iv            : ATM implied vol, annualized (default 0.0 → RV-based estimate)
    iv_rank           : IV rank [0, 1] over lookback period (default 0.50)
    dte               : days to expiry for candidate structures (default 7)
    liquidity_quality : "good" | "fair" | "poor" (default "fair")
    atm_bid_ask_pct   : ATM bid-ask as fraction of mid (default 0.05)
"""

import logging
import pandas as pd
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.core.database import get_db
from app.data_ingestion.ingestion_service import get_closed_bars
from app.inference.inference_service import run_inference
from app.decision.decision_engine import build_options_decision

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{symbol}")
async def get_decision(
    symbol: str,
    timeframe: str = Query("5m"),
    confidence_threshold: float = Query(0.55),
    atm_iv: float = Query(0.0, description="ATM IV annualized, e.g. 0.25 = 25%"),
    iv_rank: float = Query(0.50, description="IV rank [0, 1] over lookback"),
    dte: int = Query(7, description="Days to expiry"),
    liquidity_quality: str = Query("fair", description="good|fair|poor"),
    atm_bid_ask_pct: float = Query(0.05, description="ATM bid-ask / mid"),
    db: AsyncSession = Depends(get_db),
):
    """
    Full options decision for `symbol`.

    Runs inference, then evaluates all four candidate structures (long_call,
    long_put, debit_spread, credit_spread) given IV environment, breakeven
    feasibility, and liquidity. Returns a ranked list of candidates and a
    top-level recommendation or abstain signal.
    """
    symbol = symbol.upper()

    # -----------------------------------------------------------------------
    # Load OHLCV bars
    # -----------------------------------------------------------------------
    bars = await get_closed_bars(db, symbol, timeframe, limit=300)

    if len(bars) < 30:
        try:
            from app.services.market_data import fetch_candles
            candles_resp = await fetch_candles(symbol, timeframe, "5d")
            df = pd.DataFrame([
                {
                    "open": c.open, "high": c.high, "low": c.low,
                    "close": c.close, "volume": c.volume,
                    "vwap": (c.high + c.low + c.close) / 3,
                    "bar_open_time": pd.Timestamp(c.time, unit="s"),
                }
                for c in candles_resp.candles
            ])
        except Exception as e:
            return {"error": f"Insufficient data for {symbol}: {e}", "symbol": symbol}
    else:
        df = pd.DataFrame([
            {
                "open": b.open, "high": b.high, "low": b.low,
                "close": b.close, "volume": b.volume,
                "vwap": b.vwap or (b.high + b.low + b.close) / 3,
                "bar_open_time": b.bar_open_time,
            }
            for b in bars
        ])

    # Attach spot price from most recent close
    spot_price = float(df["close"].iloc[-1])

    # -----------------------------------------------------------------------
    # Run inference
    # -----------------------------------------------------------------------
    inference_result = run_inference(df, symbol, confidence_threshold)

    # Attach spot to the inference result (decision engine needs it)
    inference_result.spot_price = spot_price

    # -----------------------------------------------------------------------
    # Build options decision
    # -----------------------------------------------------------------------
    decision = build_options_decision(
        inference_result,
        atm_iv=atm_iv,
        iv_rank=iv_rank,
        dte=dte,
        liquidity_quality=liquidity_quality,
        atm_bid_ask_pct=atm_bid_ask_pct,
    )

    return decision.to_dict()
