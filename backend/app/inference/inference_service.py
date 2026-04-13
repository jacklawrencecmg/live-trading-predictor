"""
Real-time inference service.

Behavior:
- Loads the latest trained model
- On each new CLOSED bar, builds features and runs inference
- Outputs a 4-layer uncertainty bundle:
    raw probability → calibrated probability → tradeable confidence → action
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
from app.inference.uncertainty import (
    UncertaintyBundle,
    CalibrationMap,
    build_uncertainty_bundle,
)
from app.inference.confidence_tracker import get_tracker

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
        # --- Layer 1: raw probabilities ---
        prob_up: float,
        prob_down: float,
        prob_flat: float,
        # --- Layer 2: calibrated probabilities ---
        calibrated_prob_up: float,
        calibrated_prob_down: float,
        calibration_available: bool,
        # --- Layer 3: tradeable confidence ---
        tradeable_confidence: float,
        degradation_factor: float,
        # --- Layer 4: action ---
        action: str,                    # "buy" | "sell" | "abstain"
        abstain_reason: Optional[str],
        # --- Supporting uncertainty fields ---
        confidence_band: tuple,         # (low, high) ECE-driven
        calibration_health: str,
        rolling_brier: Optional[float],
        ece_recent: Optional[float],
        reliability_diagram: Optional[dict],
        # --- Legacy / existing fields ---
        expected_move_pct: float,
        # confidence kept for backward compat = tradeable_confidence
        confidence: float,
        trade_signal: str,
        no_trade_reason: Optional[str],
        feature_snapshot_id: str,
        model_version: str,
        regime: str,
        top_features: Dict[str, float],
        explanation: str,
    ):
        # Layer 1
        self.prob_up = prob_up
        self.prob_down = prob_down
        self.prob_flat = prob_flat
        # Layer 2
        self.calibrated_prob_up = calibrated_prob_up
        self.calibrated_prob_down = calibrated_prob_down
        self.calibration_available = calibration_available
        # Layer 3
        self.tradeable_confidence = tradeable_confidence
        self.degradation_factor = degradation_factor
        # Layer 4
        self.action = action
        self.abstain_reason = abstain_reason
        # Uncertainty supporting
        self.confidence_band = confidence_band
        self.calibration_health = calibration_health
        self.rolling_brier = rolling_brier
        self.ece_recent = ece_recent
        self.reliability_diagram = reliability_diagram
        # Other
        self.symbol = symbol
        self.timestamp = timestamp
        self.bar_open_time = bar_open_time
        self.expected_move_pct = expected_move_pct
        self.confidence = confidence          # = tradeable_confidence (backward compat)
        self.trade_signal = trade_signal      # = action (backward compat)
        self.no_trade_reason = no_trade_reason
        self.feature_snapshot_id = feature_snapshot_id
        self.model_version = model_version
        self.regime = regime
        self.top_features = top_features
        self.explanation = explanation

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        if isinstance(d.get("confidence_band"), tuple):
            d["confidence_band"] = list(d["confidence_band"])
        return d


# Maximum age (seconds) of an options snapshot before inference falls back to
# sentinel values.  5 minutes matches one 5-minute bar interval and is the
# tightest useful threshold for intraday options features.
# Previously this was 3600.0 (1 hour), which is far too permissive: a
# 60-minute-old options snapshot spans 12 completed bars and can reflect a
# completely different market state.
_MAX_CHAIN_STALENESS = 300.0

_loaded_model = None
_magnitude_model = None
_calibration_map: CalibrationMap = CalibrationMap.identity()


def get_loaded_model():
    global _loaded_model
    if _loaded_model is None:
        _loaded_model = load_model("logistic") or load_model("gbt") or load_model("random_forest")
    return _loaded_model


def get_calibration_map() -> CalibrationMap:
    """Return the active calibration map (identity when none has been loaded)."""
    return _calibration_map


def set_calibration_map(cal_map: CalibrationMap) -> None:
    """Register a new calibration map (called after model promotion)."""
    global _calibration_map
    _calibration_map = cal_map
    logger.info("Calibration map updated: kind=%s n=%d", cal_map.kind, cal_map.n_calibration_samples)


def run_inference(
    df: pd.DataFrame,  # closed bars only, chronological
    symbol: str,
    confidence_threshold: float = 0.55,
    options_features: Optional[dict] = None,
) -> InferenceResult:
    """
    Run inference on the latest closed bar.
    df must contain only closed bars (is_closed=True).

    Returns an InferenceResult with the full 4-layer uncertainty bundle.
    """
    if len(df) < 30:
        return _no_trade_result(symbol, "insufficient_data", df)

    # Guard: reject unclosed last bar
    if "is_closed" in df.columns and not bool(df["is_closed"].iloc[-1]):
        return _no_trade_result(symbol, "last_bar_not_closed", df)

    # Pass options data into the feature pipeline so the pipeline can fill
    # OPTIONS_FEATURE_COLS with real values and set is_null_options=0.
    # When options_data is None the pipeline sentinel-fills those columns (all
    # zeros, is_null_options=1), which is exactly what the model was trained on.
    #
    # IMPORTANT: the model input is always feat_row[FEATURE_COLS] — a vector of
    # exactly len(FEATURE_COLS) elements.  Options values are captured inside
    # FEATURE_COLS via the sentinel/indicator pattern (is_null_options) and via
    # OPTIONS_FEATURE_COLS which are part of ALL_FEATURE_COLS stored for the
    # decision engine, NOT appended manually here.  Manual appending caused a
    # dimension mismatch (28 vs 33) that sklearn caught as an exception, causing
    # a silent fallback to abstain on every inference call that supplied options data.
    _opts_for_features: Optional[dict] = None
    _options_stale = False
    if options_features:
        staleness = float(options_features.get("staleness_seconds", 0.0) or 0.0)
        if staleness > _MAX_CHAIN_STALENESS:
            logger.warning(
                "options_features for %s are stale (%.0fs > %.0fs) — using sentinel values",
                symbol, staleness, _MAX_CHAIN_STALENESS,
            )
            _options_stale = True
            # _opts_for_features stays None → pipeline sentinel-fills
        else:
            _opts_for_features = options_features

    feat_df = build_feature_matrix(df, options_data=_opts_for_features)
    if feat_df.empty or feat_df[FEATURE_COLS].iloc[-1].isna().any():
        return _no_trade_result(symbol, "feature_computation_failed", df)

    feat_row = feat_df.iloc[-1]
    feat_values = feat_row[FEATURE_COLS].values.tolist()

    X = np.array([feat_values])

    # Regime check — use detect_regime_row for full context
    from app.regime.detector import detect_regime_row, detect_regime, Regime
    try:
        regime_ctx = detect_regime_row(df)
        current_regime = str(regime_ctx.regime.value if hasattr(regime_ctx.regime, "value") else regime_ctx.regime)
        regime_abstain = regime_ctx.suppressed
        prior_abstain_reason = (
            f"regime_suppressed:{regime_ctx.suppress_reason}"
            if regime_abstain else None
        )
        # Use regime-specific threshold unless caller overrides to something higher
        effective_threshold = max(confidence_threshold, regime_ctx.confidence_threshold)
    except Exception as e:
        logger.warning("Regime detection failed: %s", e)
        current_regime = "unknown"
        prior_abstain_reason = "regime_suppressed:unknown_regime"
        effective_threshold = confidence_threshold
        regime_ctx = None

    # Model
    model = get_loaded_model()
    if model is None:
        return _no_trade_result(symbol, "model_not_trained", df, regime=current_regime)

    try:
        probs = model.predict_proba(X)[0]
        raw_prob_up = float(probs[1]) if len(probs) > 1 else 0.5
        raw_prob_down = float(probs[0])
    except Exception as e:
        logger.error("Inference error: %s", e)
        return _no_trade_result(symbol, f"model_error:{e}", df, regime=current_regime)

    # --- 4-layer uncertainty bundle ---
    cal_map = get_calibration_map()
    tracker = get_tracker()
    tracker_stats = tracker.get_stats(symbol)

    bundle = build_uncertainty_bundle(
        raw_prob_up=raw_prob_up,
        calibration_map=cal_map,
        tracker_stats=tracker_stats,
        confidence_threshold=effective_threshold,
        prior_abstain_reason=prior_abstain_reason,
    )

    # Reliability diagram as a serialisable dict
    rel_diag = None
    if bundle.reliability_bins is not None:
        rel_diag = {
            "bins": bundle.reliability_bins,
            "mean_predicted": bundle.reliability_mean_pred,
            "fraction_positive": bundle.reliability_frac_pos,
        }

    # Expected move (realized vol proxy)
    rv20 = float(feat_row.get("realized_vol_10", 0.01) or 0.01)
    expected_move_pct = rv20 / math.sqrt(252 * 78) * 100

    # Trade signal (backward compat: maps action → trade_signal vocabulary)
    trade_signal = {"buy": "buy", "sell": "sell", "abstain": "no_trade"}.get(bundle.action, "no_trade")
    no_trade_reason = bundle.abstain_reason if bundle.action == "abstain" else None

    # Feature explanation
    top_features, explanation = _explain(model, feat_values, FEATURE_COLS)

    snapshot_id = _feature_snapshot_id(feat_values)
    model_version = _get_model_version(model)
    bar_time = str(df["bar_open_time"].iloc[-1])

    # Confidence band as (low, high) tuple using the calibrated prob
    conf_band = (bundle.confidence_band_low, bundle.confidence_band_high)

    return InferenceResult(
        symbol=symbol,
        timestamp=int(time.time()),
        bar_open_time=bar_time,
        # Layer 1
        prob_up=round(raw_prob_up, 4),
        prob_down=round(raw_prob_down, 4),
        prob_flat=round(max(0.0, 1.0 - raw_prob_up - raw_prob_down), 4),
        # Layer 2
        calibrated_prob_up=bundle.calibrated_prob_up,
        calibrated_prob_down=bundle.calibrated_prob_down,
        calibration_available=bundle.calibration_available,
        # Layer 3
        tradeable_confidence=bundle.tradeable_confidence,
        degradation_factor=bundle.degradation_factor,
        # Layer 4
        action=bundle.action,
        abstain_reason=bundle.abstain_reason,
        # Uncertainty supporting
        confidence_band=conf_band,
        calibration_health=bundle.calibration_health,
        rolling_brier=bundle.rolling_brier,
        ece_recent=bundle.ece_recent,
        reliability_diagram=rel_diag,
        # Other
        expected_move_pct=round(expected_move_pct, 4),
        confidence=bundle.tradeable_confidence,   # backward compat
        trade_signal=trade_signal,
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
        calibrated_prob_up=0.5, calibrated_prob_down=0.5,
        calibration_available=False,
        tradeable_confidence=0.0,
        degradation_factor=1.0,
        action="abstain",
        abstain_reason=reason,
        confidence_band=(0.45, 0.55),
        calibration_health="unknown",
        rolling_brier=None,
        ece_recent=None,
        reliability_diagram=None,
        expected_move_pct=0.0,
        confidence=0.0,
        trade_signal="no_trade",
        no_trade_reason=reason,
        feature_snapshot_id="",
        model_version="untrained",
        regime=regime,
        top_features={},
        explanation=f"No trade: {reason}",
    )


def _explain(model, feat_values: list, feature_names: list) -> tuple:
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
                key=lambda x: abs(x[1]), reverse=True,
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

    if top:
        positive = [n for n, v in top.items() if v > 0]
        negative = [n for n, v in top.items() if v < 0]
        parts = []
        if positive:
            parts.append(f"bullish signals from {', '.join(_feature_label(n) for n in positive[:3])}")
        if negative:
            parts.append(f"bearish pressure from {', '.join(_feature_label(n) for n in negative[:3])}")
        explanation = "Prediction driven by: " + "; ".join(parts) + "." if parts else "Mixed signals."
    else:
        explanation = "Model prediction without feature attribution available."

    return top, explanation


_FEATURE_LABELS = {
    "rsi_14": "RSI(14)", "rsi_5": "RSI(5)", "macd_hist": "MACD histogram",
    "macd_line": "MACD line", "ret_1": "1-bar momentum", "ret_5": "5-bar momentum",
    "ret_20": "20-bar momentum", "volume_ratio": "relative volume",
    "bb_pct": "Bollinger Band position", "vwap_distance": "VWAP distance",
    "realized_vol_10": "realized volatility", "atm_iv": "ATM implied volatility",
    "iv_skew": "IV skew", "pc_volume_ratio": "put/call volume ratio",
}


def _feature_label(name: str) -> str:
    return _FEATURE_LABELS.get(name, name.replace("_", " "))
