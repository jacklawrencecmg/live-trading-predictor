import pytest
import numpy as np
import pandas as pd
from app.services.model_service import train_models, predict, set_models, compute_calibration
from app.services.feature_pipeline import build_features


def _train_and_set():
    np.random.seed(0)
    n = 200
    X = np.random.randn(n, 14)
    y_dir = (X[:, 0] > 0).astype(int)
    y_mag = np.abs(np.random.randn(n)) * 0.01
    dir_m, mag_m = train_models(X, y_dir, y_mag)
    set_models(dir_m, mag_m)
    return dir_m, mag_m


def test_train_returns_models():
    dir_m, mag_m = _train_and_set()
    assert dir_m is not None
    assert mag_m is not None


def test_predict_probs_sum_to_one():
    _train_and_set()
    from app.schemas.model import FeatureSet
    feat = FeatureSet(
        rsi_14=50, rsi_5=50, macd_line=0, macd_signal=0, macd_hist=0,
        bb_position=0.5, atr_norm=0.001, volume_ratio=1.0,
        momentum_5=0.001, momentum_10=0.001, momentum_20=0.001,
        iv_rank=0.5, put_call_ratio=1.0, atm_iv=0.2,
    )
    pred = predict(feat, confidence_threshold=0.55)
    assert abs(pred.prob_up + pred.prob_down - 1.0) < 0.01
    assert pred.trade_signal in ("buy", "sell", "no_trade")


def test_predict_confidence_range():
    _train_and_set()
    from app.schemas.model import FeatureSet
    feat = FeatureSet(
        rsi_14=80, rsi_5=85, macd_line=1.0, macd_signal=0.5, macd_hist=0.5,
        bb_position=0.9, atr_norm=0.002, volume_ratio=2.0,
        momentum_5=0.02, momentum_10=0.03, momentum_20=0.04,
        iv_rank=0.3, put_call_ratio=0.8, atm_iv=0.15,
    )
    pred = predict(feat)
    assert 0 <= pred.confidence <= 1


def test_calibration():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.8, 0.2, 0.7, 0.9, 0.3, 0.4, 0.6, 0.1])
    cal = compute_calibration(y_true, y_prob, n_bins=5)
    assert cal.brier_score >= 0
    assert cal.log_loss >= 0
