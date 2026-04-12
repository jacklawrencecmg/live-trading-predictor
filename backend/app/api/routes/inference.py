"""
Inference API endpoints.

GET /api/inference/predict/{symbol}
  - Loads latest closed bars
  - Runs feature pipeline
  - Returns prediction + signal + explanation

GET /api/inference/regime/{symbol}
  - Returns current regime classification

GET /api/inference/history/{symbol}
  - Returns last N inference events
"""

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List

from app.core.database import get_db
from app.data_ingestion.bar_model import OHLCVBar
from app.data_ingestion.ingestion_service import get_closed_bars
from app.feature_pipeline.features import build_feature_matrix, FEATURE_COLS
from app.inference.inference_service import run_inference
from app.inference.signal_scorer import score_signal
from app.regime.detector import detect_regime

router = APIRouter()


@router.get("/predict/{symbol}")
async def predict(
    symbol: str,
    timeframe: str = Query("5m"),
    confidence_threshold: float = Query(0.55),
    db: AsyncSession = Depends(get_db),
):
    symbol = symbol.upper()
    bars = await get_closed_bars(db, symbol, timeframe, limit=300)

    if len(bars) < 30:
        # Try yfinance fallback
        from app.services.market_data import fetch_candles
        try:
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
            raise HTTPException(status_code=400, detail=f"Insufficient data: {e}")
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

    result = run_inference(df, symbol, confidence_threshold)

    # Score signal
    rv = float(df["close"].pct_change().tail(20).std() * 100 * (252 * 78) ** 0.5)
    signal = score_signal(
        prob_up=result.prob_up,
        prob_down=result.prob_down,
        expected_move_pct=result.expected_move_pct,
        realized_vol_pct=rv,
        regime=result.regime,
        no_trade_reason=result.no_trade_reason,
        explanation=result.explanation,
        top_features=result.top_features,
    )

    return {
        "prediction": result.to_dict(),
        "signal": {
            "direction": signal.direction,
            "probability": signal.probability,
            "confidence": signal.confidence,
            "confidence_band": signal.confidence_band,
            "expected_move_pct": signal.expected_move_pct,
            "realized_vol_pct": signal.realized_vol_pct,
            "volatility_context": signal.volatility_context,
            "regime": signal.regime,
            "signal_quality_score": signal.signal_quality_score,
            "no_trade_reason": signal.no_trade_reason,
            "explanation": signal.explanation,
            "top_features": signal.top_features,
        },
    }


@router.get("/regime/{symbol}")
async def regime(
    symbol: str,
    timeframe: str = Query("5m"),
    db: AsyncSession = Depends(get_db),
):
    symbol = symbol.upper()
    bars = await get_closed_bars(db, symbol, timeframe, limit=100)
    if len(bars) < 20:
        return {"regime": "unknown", "symbol": symbol}

    df = pd.DataFrame([
        {"open": b.open, "high": b.high, "low": b.low, "close": b.close,
         "volume": b.volume, "bar_open_time": b.bar_open_time}
        for b in bars
    ])
    regimes = detect_regime(df)
    return {"regime": str(regimes.iloc[-1]), "symbol": symbol, "timeframe": timeframe}
