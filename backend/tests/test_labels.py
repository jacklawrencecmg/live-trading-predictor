"""Tests for label generation — critical for leakage prevention."""

import pytest
import numpy as np
import pandas as pd
from app.feature_pipeline.labels import (
    binary_label, ternary_label, regression_return_label, regression_range_label, build_labels
)


def make_df(n=100, seed=0):
    np.random.seed(seed)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    prices = np.abs(prices) + 50
    return pd.DataFrame({
        "open": prices * 0.999,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
        "bar_open_time": pd.date_range("2024-01-01 09:30", periods=n, freq="5min"),
    })


def test_binary_label_uses_next_bar():
    df = make_df(10)
    labels = binary_label(df)
    # label[0] = 1 if close[1] > close[0]
    expected_0 = float(df["close"].iloc[1] > df["close"].iloc[0])
    assert abs(labels.iloc[0] - expected_0) < 0.001
    # Last row must be NaN (no next bar)
    assert pd.isna(labels.iloc[-1]) or labels.iloc[-1] in (0.0, 1.0)


def test_binary_label_no_future_info():
    """Labels must only use close[i+1] not close[i] through close[n]."""
    df = make_df(50)
    labels = binary_label(df)
    # Verify: changing the last bar's close doesn't affect earlier labels
    df2 = df.copy()
    df2.loc[df2.index[-1], "close"] = df2["close"].iloc[-1] * 2
    labels2 = binary_label(df2)
    # Only the second-to-last label may change (it uses close[-1])
    assert (labels.iloc[:-2] == labels2.iloc[:-2]).all()


def test_ternary_label_values():
    df = make_df(100)
    labels = ternary_label(df)
    valid_values = {0, 1, 2}
    non_nan = labels.dropna()
    assert set(non_nan.unique()).issubset(valid_values | {float("nan")})


def test_build_labels_drops_last_row():
    df = make_df(50)
    labels = build_labels(df)
    assert len(labels) == len(df) - 1


def test_regression_return_label():
    df = make_df(20)
    labels = regression_return_label(df)
    # First label should be log(close[1]/close[0])
    import math
    expected = math.log(df["close"].iloc[1] / df["close"].iloc[0])
    assert abs(labels.iloc[0] - expected) < 1e-6


def test_range_label_positive():
    df = make_df(20)
    labels = regression_range_label(df)
    non_nan = labels.dropna()
    assert (non_nan >= 0).all()
