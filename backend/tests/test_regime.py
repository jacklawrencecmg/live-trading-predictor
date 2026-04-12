import pytest
import numpy as np
import pandas as pd
from app.regime.detector import detect_regime, Regime, should_suppress_trade


def make_trending_df(n=100):
    """Consistently trending up prices."""
    prices = np.linspace(100, 150, n)
    return pd.DataFrame({
        "open": prices * 0.999,
        "high": prices * 1.002,
        "low": prices * 0.998,
        "close": prices,
        "volume": np.ones(n) * 1e6,
        "bar_open_time": pd.date_range("2024-01-01", periods=n, freq="5min"),
    })


def make_vol_df(n=100, vol_multiplier=3.0):
    """High volatility prices."""
    np.random.seed(1)
    prices = 100 + np.cumsum(np.random.randn(n) * vol_multiplier)
    prices = np.abs(prices) + 50
    return pd.DataFrame({
        "open": prices * 0.999,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "volume": np.ones(n) * 1e6,
        "bar_open_time": pd.date_range("2024-01-01", periods=n, freq="5min"),
    })


def test_regime_returns_series():
    df = make_trending_df()
    regimes = detect_regime(df)
    assert len(regimes) == len(df)


def test_suppress_high_vol():
    assert should_suppress_trade(Regime.HIGH_VOLATILITY) is True


def test_suppress_unknown():
    assert should_suppress_trade(Regime.UNKNOWN) is True


def test_no_suppress_trending():
    assert should_suppress_trade(Regime.TRENDING_UP) is False
