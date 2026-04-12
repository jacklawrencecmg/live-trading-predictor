import asyncio
from fastapi import APIRouter, HTTPException, Query
from app.services.market_data import fetch_candles
from app.services.options_service import fetch_options_chain
from app.services.feature_pipeline import build_features
from app.services.model_service import predict, get_model_loaded
from app.schemas.model import ModelPrediction

router = APIRouter()


@router.get("/predict/{symbol}", response_model=ModelPrediction)
async def get_prediction(
    symbol: str,
    interval: str = Query("5m"),
    period: str = Query("5d"),
    confidence_threshold: float = Query(0.55),
):
    candles_resp = await fetch_candles(symbol.upper(), interval, period)
    if len(candles_resp.candles) < 30:
        raise HTTPException(status_code=400, detail="Insufficient candle data")

    import pandas as pd
    rows = [
        {"open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume}
        for c in candles_resp.candles
    ]
    df = pd.DataFrame(rows)

    try:
        chain = await fetch_options_chain(symbol.upper())
        iv_rank = chain.iv_rank
        pcr = chain.put_call_ratio
        atm_iv = chain.atm_iv
    except Exception:
        iv_rank, pcr, atm_iv = 0.5, 1.0, 0.2

    features = build_features(df, iv_rank, pcr, atm_iv)
    if features is None:
        raise HTTPException(status_code=400, detail="Could not build features")

    prediction = predict(features, confidence_threshold)
    prediction.symbol = symbol.upper()
    return prediction


@router.get("/status")
async def model_status():
    return {"model_loaded": get_model_loaded()}
