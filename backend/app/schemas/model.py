from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class FeatureSet(BaseModel):
    rsi_14: float
    rsi_5: float
    macd_line: float
    macd_signal: float
    macd_hist: float
    bb_position: float
    atr_norm: float
    volume_ratio: float
    momentum_5: float
    momentum_10: float
    momentum_20: float
    iv_rank: float
    put_call_ratio: float
    atm_iv: float


class ModelPrediction(BaseModel):
    symbol: str
    timestamp: int
    prob_up: float
    prob_down: float
    prob_flat: float
    expected_move_pct: float
    confidence: float
    trade_signal: str  # "buy", "sell", "no_trade"
    features: FeatureSet
    model_version: str


class CalibrationData(BaseModel):
    bin_centers: List[float]
    fraction_positive: List[float]
    brier_score: float
    log_loss: float
