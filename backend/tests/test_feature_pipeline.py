import pytest
import pandas as pd
import numpy as np
from app.services.feature_pipeline import build_features, features_to_array


def make_df(n=100, seed=42):
    np.random.seed(seed)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    prices = np.maximum(prices, 1)
    return pd.DataFrame({
        "open": prices * 0.999,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
    })


def test_build_features_basic():
    df = make_df(100)
    features = build_features(df)
    assert features is not None
    arr = features_to_array(features)
    assert len(arr) == 14
    assert all(np.isfinite(v) for v in arr)


def test_insufficient_data():
    df = make_df(10)
    features = build_features(df)
    assert features is None


def test_rsi_range():
    df = make_df(100)
    features = build_features(df)
    assert 0 <= features.rsi_14 <= 100
    assert 0 <= features.rsi_5 <= 100


def test_volume_ratio_positive():
    df = make_df(100)
    features = build_features(df)
    assert features.volume_ratio >= 0


def test_feature_names_match():
    from app.services.feature_pipeline import FEATURE_NAMES
    df = make_df(100)
    features = build_features(df)
    arr = features_to_array(features)
    assert len(FEATURE_NAMES) == len(arr)
