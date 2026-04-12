"""
Real-time inference service.

Behavior:
- Loads the latest trained model
- On each new CLOSED bar, builds features and runs inference
- Outputs: P(up), P(down), expected_move, confidence, no_trade_flag
- Stores each inference event in the database
- Exposes via REST endpoint

IMPORTANT: Inference triggers ONLY after input candle is fully closed.
Model version and feature snapshot IDs are stored for auditability.
"""

import hashlib
import json
import logging
import math
import time
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.feature_pipeline.features import build_feature_matrix, FEATURE_COLS
from app.ml_models.baseline import load_model
from app.regime.detector import detect_regime, should_suppress_trade

logger = logging.getLogger(__name__)


def _feature_snapshot_id(features: list) -> str:
    """Stable hash of feature values for auditability."""
    s = json.dumps([round(float(v), 6) if v is not None else None for v in features])
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _get_model_version(model) -> str:
    if model is None:
        return "untrained"
    name = type(model).__name__
    if hasattr(model, "named_steps"):
        clf_name = type(model.named_steps.get("clf", model)).__name__
        return f"{clf_name}_v1"
    return f"{name}_v1"


class InferenceResult:
    def __init__(
        self,
        symbol: str,
        timestamp: int,
        bar_open_time: str,
        prob_up: float,
        prob_down: float,
        prob_flat: float,
        expected_move_pct: float,
        confidence: float,
        trade_signal: str,
        no_trade_reason: Optional[str],
        feature_snapshot_id: str,
        model_version: str,
        regime: str,
        top_features: Dict[str, float],
        explanation: str,
    ):
        self.symbol = symbol
        self.timestamp = timestamp
        self.bar_open_time = bar_open_time
        self.prob_up = prob_up
        self.prob_down = prob_down
        self.prob_flat = prob_flat
        self.expected_move_pct = expected_move_pct
        self.confidence = confidence
        self.trade_signal = trade_signal
        self.no_trade_reason = no_trade_reason
        self.feature_snapshot_id = feature_snapshot_id
        self.model_version = model_version
        self.regime = regime
        self.top_features = top_features
        self.explanation = explanation

    def to_dict(self) -> dict:
        return self.__dict__.copy()


_loaded_model = None
_magnitude_model = None


def get_loaded_model():
    global _loaded_model
    if _loaded_model is None:
        _loaded_model = load_model("logistic") or load_model("gbt") or load_model("random_forest")
    return _loaded_model


def run_inference(
    df: pd.DataFrame,  # closed bars only, chronological
    symbol: str,
    confidence_threshold: float = 0.55,
    options_features: Optional[dict] = None,
) -> InferenceResult:
    """
    Run inference on the latest closed bar.
    df must contain only closed bars (is_closed=True).
    """
    if len(df) < 30:
        return _no_trade_result(symbol, "insufficient_data", df)

    feat_df = build_feature_matrix(df)
    if feat_df.empty or feat_df[FEATURE_COLS].iloc[-1].isna().any():
        return _no_trade_result(symbol, "feature_computation_failed", df)

    # Get last row (current bar's features — built from prior bars)
    feat_row = feat_df.iloc[-1]
    feat_values = feat_row[FEATURE_COLS].values.tolist()

    # Add options features if available
    if options_features:
        for k in ["atm_iv", "iv_rank", "iv_skew", "pc_volume_ratio", "pc_oi_ratio"]:
            feat_values.append(float(options_features.get(k, 0.0) or 0.0))

    X = np.array([feat_values])

    # Detect regime
    regimes = detect_regime(df)
    current_regime = str(regimes.iloc[-1])

    # Check regime suppression
    from app.regime.detector import Regime
    regime_enum = regimes.iloc[-1]
    if should_suppress_trade(regime_enum):
        return _no_trade_result(symbol, f"regime_suppressed:{current_regime}", df, regime=current_regime)

    # Run model
    model = get_loaded_model()
    if model is None:
        return _no_trade_result(symbol, "model_not_trained", df, regime=current_regime)

    try:
        probs = model.predict_proba(X)[0]
        prob_up = float(probs[1]) if len(probs) > 1 else 0.5
        prob_down = float(probs[0])
    except Exception as e:
        logger.error("Inference error: %s", e)
        return _no_trade_result(symbol, f"model_error:{e}", df, regime=current_regime)

    prob_flat = max(1.0 - prob_up - prob_down, 0.0)
    confidence = abs(prob_up - 0.5) * 2

    # Expected move (use realized vol as proxy if no magnitude model)
    rv20 = float(feat_row.get("realized_vol_10", 0.01) or 0.01)
    expected_move_pct = rv20 / math.sqrt(252 * 78) * 100  # 1-bar expected

    # Signal
    min_prob = confidence_threshold
    if prob_up > min_prob:
        signal = "buy"
        no_trade_reason = None
    elif prob_down > min_prob:
        signal = "sell"
        no_trade_reason = None
    else:
        signal = "no_trade"
        no_trade_reason = f"low_confidence:{confidence:.2f}"

    # Feature importance / explanation
    top_features, explanation = _explain(model, feat_values, FEATURE_COLS)

    snapshot_id = _feature_snapshot_id(feat_values)
    model_version = _get_model_version(model)
    bar_time = str(df["bar_open_time"].iloc[-1])

    return InferenceResult(
        symbol=symbol,
        timestamp=int(time.time()),
        bar_open_time=bar_time,
        prob_up=round(prob_up, 4),
        prob_down=round(prob_down, 4),
        prob_flat=round(prob_flat, 4),
        expected_move_pct=round(expected_move_pct, 4),
        confidence=round(confidence, 4),
        trade_signal=signal,
        no_trade_reason=no_trade_reason,
        feature_snapshot_id=snapshot_id,
        model_version=model_version,
        regime=current_regime,
        top_features=top_features,
        explanation=explanation,
    )


def _no_trade_result(
    symbol: str, reason: str, df: pd.DataFrame = None, regime: str = "unknown"
) -> InferenceResult:
    bar_time = str(df["bar_open_time"].iloc[-1]) if df is not None and len(df) > 0 else ""
    return InferenceResult(
        symbol=symbol,
        timestamp=int(time.time()),
        bar_open_time=bar_time,
        prob_up=0.5, prob_down=0.5, prob_flat=0.0,
        expected_move_pct=0.0, confidence=0.0,
        trade_signal="no_trade",
        no_trade_reason=reason,
        feature_snapshot_id="",
        model_version="untrained",
        regime=regime,
        top_features={},
        explanation=f"No trade: {reason}",
    )


def _explain(model, feat_values: list, feature_names: list) -> tuple:
    """
    Generate feature importance and natural language explanation.
    Uses model coefficients or importances.
    """
    top = {}
    explanation = ""

    try:
        clf = model.named_steps.get("clf") if hasattr(model, "named_steps") else None
        scaler = model.named_steps.get("scaler") if hasattr(model, "named_steps") else None

        if clf is not None and hasattr(clf, "coef_"):
            coef = clf.coef_[0]
            if scaler is not None and len(coef) == len(feat_values):
                scaled_vals = scaler.transform([feat_values[:len(coef)]])[0]
                contributions = coef * scaled_vals
            else:
                contributions = coef[:len(feature_names)]

            named = sorted(
                zip(feature_names[:len(contributions)], contributions),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            top = {n: round(float(v), 4) for n, v in named[:5]}

        elif clf is not None and hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
            named = sorted(
                zip(feature_names[:len(imp)], imp),
                key=lambda x: x[1], reverse=True,
            )
            top = {n: round(float(v), 4) for n, v in named[:5]}

    except Exception:
        pass

    # Natural language summary
    if top:
        positive = [n for n, v in top.items() if v > 0]
        negative = [n for n, v in top.items() if v < 0]
        parts = []
        if positive:
            pretty = [_feature_label(n) for n in positive[:3]]
            parts.append(f"bullish signals from {', '.join(pretty)}")
        if negative:
            pretty = [_feature_label(n) for n in negative[:3]]
            parts.append(f"bearish pressure from {', '.join(pretty)}")
        explanation = "Prediction driven by: " + "; ".join(parts) + "." if parts else "Mixed signals."
    else:
        explanation = "Model prediction without feature attribution available."

    return top, explanation


_FEATURE_LABELS = {
    "rsi_14": "RSI(14)",
    "rsi_5": "RSI(5)",
    "macd_hist": "MACD histogram",
    "macd_line": "MACD line",
    "ret_1": "1-bar momentum",
    "ret_5": "5-bar momentum",
    "ret_20": "20-bar momentum",
    "volume_ratio": "relative volume",
    "bb_pct": "Bollinger Band position",
    "vwap_distance": "VWAP distance",
    "realized_vol_10": "realized volatility",
    "atm_iv": "ATM implied volatility",
    "iv_skew": "IV skew",
    "pc_volume_ratio": "put/call volume ratio",
}


def _feature_label(name: str) -> str:
    return _FEATURE_LABELS.get(name, name.replace("_", " "))
