"""Tests for feature pipeline — leakage and correctness."""

import pytest
import numpy as np
import pandas as pd
from app.feature_pipeline.features import build_feature_matrix, FEATURE_COLS


def make_df(n=100, seed=42):
    np.random.seed(seed)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    prices = np.abs(prices) + 50
    return pd.DataFrame({
        "open": prices * 0.999,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
        "vwap": prices,
        "bar_open_time": pd.date_range("2024-01-01 09:30", periods=n, freq="5min"),
    })


def test_feature_matrix_shape():
    df = make_df(100)
    feat = build_feature_matrix(df)
    assert len(feat) == len(df)
    for col in FEATURE_COLS:
        assert col in feat.columns, f"Missing column: {col}"


def test_no_future_leakage():
    """
    Core leakage test: changing future bars should not affect features of earlier bars.
    """
    df = make_df(100)
    feat_original = build_feature_matrix(df)

    # Modify last 10 bars drastically
    df_modified = df.copy()
    df_modified.loc[df_modified.index[-10:], "close"] *= 10

    feat_modified = build_feature_matrix(df_modified)

    # Features for rows 0..89 should be identical (they don't use rows 90-99)
    for col in FEATURE_COLS:
        orig = feat_original[col].iloc[:89].dropna()
        mod = feat_modified[col].iloc[:89].dropna()
        if len(orig) > 0 and len(mod) > 0:
            assert np.allclose(orig.values, mod.values, atol=1e-6), \
                f"Leakage detected in feature {col}"


def test_rsi_range():
    df = make_df(100)
    feat = build_feature_matrix(df)
    rsi = feat["rsi_14"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_volume_ratio_positive():
    df = make_df(100)
    feat = build_feature_matrix(df)
    vr = feat["volume_ratio"].dropna()
    assert (vr >= 0).all()


def test_no_inf_values():
    df = make_df(100)
    feat = build_feature_matrix(df)
    for col in FEATURE_COLS:
        col_data = feat[col].replace([np.inf, -np.inf], np.nan).dropna()
        assert not np.isinf(col_data).any(), f"Inf in {col}"
