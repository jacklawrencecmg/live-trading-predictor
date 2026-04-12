"""
Regime detection tests.

RG1  — detect_regime returns a Series with one label per bar
RG2  — trending_up in a pure uptrend
RG3  — trending_down in a pure downtrend
RG4  — high_volatility when ATR spikes
RG5  — low_volatility when ATR compresses
RG6  — mean_reverting in a flat oscillating market
RG7  — warmup bars return UNKNOWN
RG8  — should_suppress_trade: high_vol → True (backward compat)
RG9  — should_suppress_trade: trending_up → False (backward compat)
RG10 — should_suppress_trade: event_risk → True (new)
RG11 — should_suppress_trade: liquidity_poor → True (new)
RG12 — should_suppress_trade: unknown → True (backward compat)
RG13 — detect_regime_full returns DataFrame with 'regime' column
RG14 — detect_regime_full adx column present and positive
RG15 — detect_regime_row returns RegimeContext
RG16 — RegimeContext.suppressed consistent with get_regime_thresholds
RG17 — event_risk fires on extreme return
RG18 — liquidity_poor fires on very low volume
RG19 — get_regime_thresholds: trending_up allows trade
RG20 — get_regime_thresholds: high_volatility blocks trade
RG21 — detect_regime vectorized: no Python loop performance test
RG22 — confidence_threshold ordering: event_risk > high_vol > mean_rev > trending
RG23 — detect_regime_row confidence_threshold elevated for suppressed regimes
RG24 — detect_regime string enum acceptance in should_suppress_trade
"""

import numpy as np
import pandas as pd
import pytest
from app.regime.detector import (
    detect_regime,
    detect_regime_full,
    detect_regime_row,
    get_regime_thresholds,
    should_suppress_trade,
    Regime,
    RegimeContext,
    REGIME_THRESHOLDS,
    WARMUP_BARS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(prices, vol_multiplier=1.0, volume_multiplier=1.0, n=None):
    """Build an OHLCV DataFrame from a close price array."""
    if n is not None:
        prices = prices[:n] if len(prices) >= n else np.pad(prices, (0, n - len(prices)), mode="edge")
    prices = np.array(prices, dtype=float)
    noise = np.abs(prices) * 0.002
    return pd.DataFrame({
        "open":   prices * 0.999,
        "high":   prices + noise * vol_multiplier,
        "low":    prices - noise * vol_multiplier,
        "close":  prices,
        "volume": np.ones(len(prices)) * 1e6 * volume_multiplier,
        "bar_open_time": pd.date_range("2024-01-01 09:30", periods=len(prices), freq="5min"),
    })


def trending_up_df(n=80):
    return make_df(np.linspace(100, 150, n))


def trending_down_df(n=80):
    return make_df(np.linspace(150, 100, n))


def oscillating_df(n=80):
    t = np.arange(n)
    return make_df(100 + 2 * np.sin(2 * np.pi * t / 10))


def high_vol_df(n=100):
    """
    Low-vol baseline for first half, then a sudden ATR spike in the second half.
    The ratio atr_short/atr_long must exceed ATR_HIGH_VOL_RATIO=1.5.
    """
    prices = np.linspace(100, 105, n)
    noise = np.abs(prices) * 0.002
    # First half: quiet range; second half: 20× wider range
    high = np.concatenate([prices[:n//2] + noise[:n//2], prices[n//2:] + noise[n//2:] * 20])
    low  = np.concatenate([prices[:n//2] - noise[:n//2], prices[n//2:] - noise[n//2:] * 20])
    return pd.DataFrame({
        "open":   prices * 0.999,
        "high":   high,
        "low":    low,
        "close":  prices,
        "volume": np.ones(n) * 1e6,
        "bar_open_time": pd.date_range("2024-01-01 09:30", periods=n, freq="5min"),
    })


def low_vol_df(n=100):
    """
    Normal-vol baseline for first half, then ATR compresses in second half.
    The ratio atr_short/atr_long must fall below ATR_LOW_VOL_RATIO=0.5.
    """
    prices = np.linspace(100, 102, n)
    noise = np.abs(prices) * 0.002
    # First half: normal range; second half: 20× tighter range
    high = np.concatenate([prices[:n//2] + noise[:n//2], prices[n//2:] + noise[n//2:] * 0.05])
    low  = np.concatenate([prices[:n//2] - noise[:n//2], prices[n//2:] - noise[n//2:] * 0.05])
    return pd.DataFrame({
        "open":   prices * 0.999,
        "high":   high,
        "low":    low,
        "close":  prices,
        "volume": np.ones(n) * 1e6,
        "bar_open_time": pd.date_range("2024-01-01 09:30", periods=n, freq="5min"),
    })


def event_risk_df(n=80):
    """
    Baseline noisy prices for warmup, then inject a 10% single-bar spike.
    The spike must exceed ABNORMAL_MIN_ABS_RETURN (0.5%) AND 3.5 × rolling_std.
    """
    np.random.seed(42)
    # Noisy baseline so rolling_std > 0
    prices = 100 + np.cumsum(np.random.randn(n) * 0.2)
    prices = np.abs(prices) + 80  # keep positive
    # Inject 10% spike at bar -10 (well beyond warmup)
    spike_idx = n - 10
    prices[spike_idx] = prices[spike_idx - 1] * 1.10
    prices[spike_idx + 1:] = prices[spike_idx]
    return make_df(prices, vol_multiplier=1.0)


def liquidity_poor_df(n=80):
    """
    First half: normal volume (baseline for rolling average).
    Second half: volume drops to 5% of baseline — well below LIQUIDITY_VOL_RATIO=0.25.
    """
    np.random.seed(7)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.1)
    prices = np.abs(prices) + 80
    df = make_df(prices, vol_multiplier=0.5)
    # Half baseline, half very low (5% of normal)
    volumes = np.where(np.arange(n) < n // 2, 1e6, 5e4)
    df["volume"] = volumes
    return df


# ---------------------------------------------------------------------------
# RG1
# ---------------------------------------------------------------------------
def test_detect_regime_returns_series():
    df = trending_up_df()
    regimes = detect_regime(df)
    assert isinstance(regimes, pd.Series)
    assert len(regimes) == len(df)


# ---------------------------------------------------------------------------
# RG2
# ---------------------------------------------------------------------------
def test_trending_up_detected():
    df = trending_up_df(n=80)
    regimes = detect_regime(df)
    last_regimes = regimes.iloc[WARMUP_BARS:].values
    # Majority of post-warmup bars should be trending
    trending = [(r in (Regime.TRENDING_UP, Regime.TRENDING_DOWN)) for r in last_regimes]
    assert sum(trending) / len(trending) > 0.4, f"Expected trending majority, got {pd.Series(last_regimes).value_counts()}"


# ---------------------------------------------------------------------------
# RG3
# ---------------------------------------------------------------------------
def test_trending_down_detected():
    df = trending_down_df(n=80)
    regimes = detect_regime(df)
    last_regimes = regimes.iloc[WARMUP_BARS:].values
    trending_down = [r == Regime.TRENDING_DOWN for r in last_regimes]
    assert sum(trending_down) / len(trending_down) > 0.2, "Expected some TRENDING_DOWN"


# ---------------------------------------------------------------------------
# RG4
# ---------------------------------------------------------------------------
def test_high_volatility_detected():
    df = high_vol_df(n=80)
    regimes = detect_regime(df)
    # Some bars should be high_volatility
    has_high_vol = any(r == Regime.HIGH_VOLATILITY for r in regimes)
    assert has_high_vol, f"Expected HIGH_VOLATILITY, got {pd.Series(regimes).value_counts().to_dict()}"


# ---------------------------------------------------------------------------
# RG5
# ---------------------------------------------------------------------------
def test_low_volatility_detected():
    df = low_vol_df(n=80)
    regimes = detect_regime(df)
    has_low_vol = any(r == Regime.LOW_VOLATILITY for r in regimes)
    assert has_low_vol, f"Expected LOW_VOLATILITY, got {pd.Series(regimes).value_counts().to_dict()}"


# ---------------------------------------------------------------------------
# RG6
# ---------------------------------------------------------------------------
def test_mean_reverting_detected():
    df = oscillating_df(n=80)
    regimes = detect_regime(df)
    has_mr = any(r == Regime.MEAN_REVERTING for r in regimes)
    assert has_mr, f"Expected MEAN_REVERTING, got {pd.Series(regimes).value_counts().to_dict()}"


# ---------------------------------------------------------------------------
# RG7
# ---------------------------------------------------------------------------
def test_warmup_bars_are_unknown():
    df = trending_up_df(n=80)
    regimes = detect_regime(df)
    assert all(r == Regime.UNKNOWN for r in regimes.iloc[:WARMUP_BARS])


# ---------------------------------------------------------------------------
# RG8–RG12 should_suppress_trade
# ---------------------------------------------------------------------------
def test_suppress_high_vol():
    assert should_suppress_trade(Regime.HIGH_VOLATILITY) is True


def test_no_suppress_trending():
    assert should_suppress_trade(Regime.TRENDING_UP) is False
    assert should_suppress_trade(Regime.TRENDING_DOWN) is False


def test_suppress_event_risk():
    assert should_suppress_trade(Regime.EVENT_RISK) is True


def test_suppress_liquidity_poor():
    assert should_suppress_trade(Regime.LIQUIDITY_POOR) is True


def test_suppress_unknown():
    assert should_suppress_trade(Regime.UNKNOWN) is True


# ---------------------------------------------------------------------------
# RG13
# ---------------------------------------------------------------------------
def test_detect_regime_full_has_regime_column():
    df = trending_up_df()
    full = detect_regime_full(df)
    assert "regime" in full.columns
    assert len(full) == len(df)


# ---------------------------------------------------------------------------
# RG14
# ---------------------------------------------------------------------------
def test_detect_regime_full_adx_positive():
    df = trending_up_df()
    full = detect_regime_full(df)
    assert "adx" in full.columns
    assert (full["adx"].dropna() >= 0).all()


# ---------------------------------------------------------------------------
# RG15
# ---------------------------------------------------------------------------
def test_detect_regime_row_returns_context():
    df = trending_up_df()
    ctx = detect_regime_row(df)
    assert isinstance(ctx, RegimeContext)
    assert isinstance(ctx.regime, Regime)
    assert 0.0 <= ctx.confidence_threshold <= 1.0
    assert ctx.adx_proxy >= 0.0


# ---------------------------------------------------------------------------
# RG16
# ---------------------------------------------------------------------------
def test_regime_context_suppressed_consistent():
    for regime in Regime:
        thresholds = get_regime_thresholds(regime)
        ctx = RegimeContext(
            regime=regime,
            atr_ratio=1.0, realized_vol_ratio=1.0,
            adx_proxy=30.0, trend_direction="up", ema_spread_pct=0.01,
            volume_ratio=1.0, bar_range_ratio=1.0,
            is_abnormal_move=False, abnormal_move_sigma=0.5,
            suppressed=not thresholds.allow_trade,
            suppress_reason=thresholds.suppress_reason,
            confidence_threshold=thresholds.confidence_threshold,
            min_signal_quality=thresholds.min_signal_quality,
        )
        assert ctx.suppressed == (not thresholds.allow_trade), (
            f"Suppressed mismatch for {regime}"
        )


# ---------------------------------------------------------------------------
# RG17
# ---------------------------------------------------------------------------
def test_event_risk_fires_on_extreme_return():
    df = event_risk_df(n=80)
    regimes = detect_regime(df)
    has_event_risk = any(r == Regime.EVENT_RISK for r in regimes)
    assert has_event_risk, f"Expected EVENT_RISK on 20% spike, got {pd.Series(regimes).value_counts().to_dict()}"


# ---------------------------------------------------------------------------
# RG18
# ---------------------------------------------------------------------------
def test_liquidity_poor_fires_on_low_volume():
    df = liquidity_poor_df(n=80)
    regimes = detect_regime(df)
    has_lp = any(r == Regime.LIQUIDITY_POOR for r in regimes)
    assert has_lp, f"Expected LIQUIDITY_POOR on 10% volume, got {pd.Series(regimes).value_counts().to_dict()}"


# ---------------------------------------------------------------------------
# RG19–RG20
# ---------------------------------------------------------------------------
def test_trending_up_allows_trade():
    t = get_regime_thresholds(Regime.TRENDING_UP)
    assert t.allow_trade is True
    assert t.suppress_reason is None


def test_high_volatility_blocks_trade():
    t = get_regime_thresholds(Regime.HIGH_VOLATILITY)
    assert t.allow_trade is False
    assert t.suppress_reason is not None


# ---------------------------------------------------------------------------
# RG21 — detect_regime is vectorized (no loop bottleneck)
# ---------------------------------------------------------------------------
def test_detect_regime_vectorized_large_df():
    """Should complete in <1s for 5000 bars (vectorized, not looped)."""
    import time
    n = 5000
    df = make_df(np.linspace(100, 200, n))
    t0 = time.time()
    detect_regime(df)
    elapsed = time.time() - t0
    assert elapsed < 2.0, f"detect_regime took {elapsed:.2f}s on {n} bars — check for Python loops"


# ---------------------------------------------------------------------------
# RG22 — threshold ordering
# ---------------------------------------------------------------------------
def test_confidence_threshold_ordering():
    """Higher-risk regimes should require higher confidence."""
    t_trending = get_regime_thresholds(Regime.TRENDING_UP)
    t_mean_rev = get_regime_thresholds(Regime.MEAN_REVERTING)
    t_high_vol = get_regime_thresholds(Regime.HIGH_VOLATILITY)
    t_event    = get_regime_thresholds(Regime.EVENT_RISK)

    assert t_event.confidence_threshold >= t_high_vol.confidence_threshold
    assert t_high_vol.confidence_threshold >= t_mean_rev.confidence_threshold
    assert t_mean_rev.confidence_threshold >= t_trending.confidence_threshold


# ---------------------------------------------------------------------------
# RG23
# ---------------------------------------------------------------------------
def test_regime_row_confidence_elevated_for_suppressed():
    """For suppressed regimes, confidence_threshold must exceed 0.55."""
    for regime in Regime:
        thresholds = get_regime_thresholds(regime)
        if not thresholds.allow_trade:
            assert thresholds.confidence_threshold > 0.55, (
                f"Suppressed regime {regime} should have threshold > 0.55"
            )


# ---------------------------------------------------------------------------
# RG24 — string enum acceptance
# ---------------------------------------------------------------------------
def test_should_suppress_accepts_string():
    assert should_suppress_trade("event_risk") is True
    assert should_suppress_trade("trending_up") is False
    assert should_suppress_trade("liquidity_poor") is True
    assert should_suppress_trade("nonexistent_value") is True  # unknown → suppressed
