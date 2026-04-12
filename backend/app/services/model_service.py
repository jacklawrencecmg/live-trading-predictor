import json
import math
import pickle
import base64
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline

from app.schemas.model import ModelPrediction, FeatureSet, CalibrationData
from app.services.feature_pipeline import features_to_array, build_features

MODEL_VERSION = "logistic_v1"

# In-memory model store (replace with Redis/DB persistence for prod)
_direction_model: Optional[Pipeline] = None
_magnitude_model: Optional[Pipeline] = None


def _make_direction_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")),
    ])


def _make_magnitude_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0)),
    ])


def train_models(
    X: np.ndarray,
    y_direction: np.ndarray,  # 1=up, 0=down/flat
    y_magnitude: np.ndarray,  # abs % move
) -> Tuple[Pipeline, Pipeline]:
    dir_model = _make_direction_model()
    dir_model.fit(X, y_direction)

    mag_model = _make_magnitude_model()
    mag_model.fit(X, y_magnitude)

    return dir_model, mag_model


def predict(features: FeatureSet, confidence_threshold: float = 0.55) -> ModelPrediction:
    global _direction_model, _magnitude_model

    import time
    ts = int(time.time())
    X = np.array([features_to_array(features)])

    if _direction_model is None:
        # Return uninformed prior with low confidence
        return ModelPrediction(
            symbol="UNKNOWN",
            timestamp=ts,
            prob_up=0.5,
            prob_down=0.5,
            prob_flat=0.0,
            expected_move_pct=0.0,
            confidence=0.0,
            trade_signal="no_trade",
            features=features,
            model_version="untrained",
        )

    probs = _direction_model.predict_proba(X)[0]
    prob_up = float(probs[1]) if len(probs) > 1 else 0.5
    prob_down = float(probs[0])

    expected_move = float(_magnitude_model.predict(X)[0])
    expected_move = max(expected_move, 0.0)

    confidence = abs(prob_up - 0.5) * 2  # 0 at 50/50, 1 at 100/0

    if confidence < (confidence_threshold - 0.5) * 2:
        signal = "no_trade"
    elif prob_up > 0.5 + (confidence_threshold - 0.5):
        signal = "buy"
    else:
        signal = "sell"

    return ModelPrediction(
        symbol="",
        timestamp=ts,
        prob_up=round(prob_up, 4),
        prob_down=round(prob_down, 4),
        prob_flat=round(max(1 - prob_up - prob_down, 0), 4),
        expected_move_pct=round(expected_move * 100, 4),
        confidence=round(confidence, 4),
        trade_signal=signal,
        features=features,
        model_version=MODEL_VERSION,
    )


def compute_calibration(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> CalibrationData:
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    bs = float(brier_score_loss(y_true, y_prob))
    ll = float(log_loss(y_true, y_prob))
    return CalibrationData(
        bin_centers=mean_pred.tolist(),
        fraction_positive=fraction_pos.tolist(),
        brier_score=round(bs, 6),
        log_loss=round(ll, 6),
    )


def set_models(dir_model: Pipeline, mag_model: Pipeline):
    global _direction_model, _magnitude_model
    _direction_model = dir_model
    _magnitude_model = mag_model


def get_model_loaded() -> bool:
    return _direction_model is not None
