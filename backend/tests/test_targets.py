"""
Tests for multi-horizon targets and regime labels.

Coverage
--------
T1  — Direction target: values, NaN positions, known-series correctness
T2  — Return target: known values, sign consistency with direction
T3  — Magnitude target: non-negative, equals abs(ret)
T4  — RVOL target: h=1 equals magnitude, non-negative, known values
T5  — Abstain flag: binary, matches FLAT class, rate grows with horizon
T6  — Leakage: appending future bars does not change past targets (CRITICAL)
T7  — API: column names, custom horizons, target_col_names()
T8  — Regime labels: dtype, column names, warmup mask
T9  — Regime correctness: each flag fires on the expected synthetic series
T10 — Regime shift-by-1: appending future bars does not change past labels
T11 — Embargo: NaN counts match horizon, overlap structure documented
"""

import math

import numpy as np
import pandas as pd
import pytest

from app.feature_pipeline.targets import (
    HORIZONS,
    FLAT_THRESHOLD_K,
    DirectionLabel,
    compute_targets,
    target_col_names,
    _atr_norm,
)
from app.feature_pipeline.regime_labels import (
    REGIME_LABEL_COLS,
    WARMUP_BARS,
    ADX_TREND_THRESHOLD,
    ATR_HIGH_VOL_RATIO,
    ATR_LOW_VOL_RATIO,
    LIQUIDITY_VOL_THRESHOLD,
    compute_regime_labels,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Random but valid OHLCV: high >= max(open,close), low <= min(open,close)."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
    close = np.clip(close, 1.0, None)
    opens = np.clip(close + rng.normal(0, 0.15, n), 0.01, None)
    oc_hi = np.maximum(close, opens)
    oc_lo = np.minimum(close, opens)
    highs = oc_hi + rng.uniform(0.05, 0.5, n)
    lows  = np.clip(oc_lo - rng.uniform(0.05, 0.5, n), 0.01, None)
    vols  = rng.integers(1_000, 10_000, n).astype(float)
    times = pd.date_range("2024-01-02 14:30", periods=n, freq="5min")
    return pd.DataFrame({
        "bar_open_time": times,
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  close,
        "volume": vols,
        "vwap":   (highs + lows + close) / 3,
    })


def _make_rising(n: int = 80) -> pd.DataFrame:
    """Strictly and strongly rising close prices (guaranteed UP direction at h=1)."""
    close = np.linspace(100.0, 200.0, n)          # +1 per bar — very large move
    opens = close - 0.3
    highs = close + 0.5
    lows  = close - 0.5
    vols  = np.full(n, 5000.0)
    times = pd.date_range("2024-01-02 14:30", periods=n, freq="5min")
    return pd.DataFrame({
        "bar_open_time": times, "open": opens, "high": highs,
        "low": lows, "close": close, "volume": vols,
        "vwap": (highs + lows + close) / 3,
    })


def _make_falling(n: int = 80) -> pd.DataFrame:
    """Strictly and strongly falling close prices."""
    close = np.linspace(200.0, 100.0, n)
    opens = close + 0.3
    highs = close + 0.5
    lows  = close - 0.5
    vols  = np.full(n, 5000.0)
    times = pd.date_range("2024-01-02 14:30", periods=n, freq="5min")
    return pd.DataFrame({
        "bar_open_time": times, "open": opens, "high": highs,
        "low": lows, "close": close, "volume": vols,
        "vwap": (highs + lows + close) / 3,
    })


def _make_flat(n: int = 80) -> pd.DataFrame:
    """Completely flat close prices — every label should be FLAT."""
    close = np.full(n, 100.0)
    opens = close.copy()
    highs = close + 0.01
    lows  = close - 0.01
    vols  = np.full(n, 5000.0)
    times = pd.date_range("2024-01-02 14:30", periods=n, freq="5min")
    return pd.DataFrame({
        "bar_open_time": times, "open": opens, "high": highs,
        "low": lows, "close": close, "volume": vols,
        "vwap": (highs + lows + close) / 3,
    })


# ---------------------------------------------------------------------------
# T1: Direction target
# ---------------------------------------------------------------------------

def test_T1_dir_values_in_valid_set():
    df = _make_ohlcv(100)
    t = compute_targets(df, horizons=(1,))
    valid = t["y_dir_h1"].dropna()
    assert set(valid.unique()).issubset({0.0, 1.0, 2.0})


def test_T1_dir_last_h_rows_nan_for_each_horizon():
    df = _make_ohlcv(100)
    t = compute_targets(df, horizons=(1, 3, 5))
    for h in (1, 3, 5):
        col = f"y_dir_h{h}"
        assert t[col].iloc[-h:].isna().all(), f"Expected last {h} rows NaN for {col}"
        if h < len(df):
            assert t[col].iloc[:-h].notna().any(), f"Expected some non-NaN rows for {col}"


def test_T1_strong_rising_series_gives_UP_at_h1():
    """A linearly rising series should produce UP for all non-NaN, non-warmup rows."""
    df = _make_rising(80)
    t = compute_targets(df, horizons=(1,))
    # Skip first 5 rows: ATR warmup may produce NaN threshold
    valid = t["y_dir_h1"].dropna().iloc[5:]
    assert (valid == float(DirectionLabel.UP)).all(), \
        f"Expected all UP for rising series; got unique values {valid.unique()}"


def test_T1_strong_falling_series_gives_DOWN_at_h1():
    """A linearly falling series should produce DOWN for all non-NaN, non-warmup rows."""
    df = _make_falling(80)
    t = compute_targets(df, horizons=(1,))
    valid = t["y_dir_h1"].dropna().iloc[5:]
    assert (valid == float(DirectionLabel.DOWN)).all(), \
        f"Expected all DOWN for falling series; got {valid.unique()}"


def test_T1_flat_series_gives_FLAT_at_h1():
    """Completely flat close prices → all FLAT (zero future return)."""
    df = _make_flat(80)
    t = compute_targets(df, horizons=(1,))
    valid = t["y_dir_h1"].dropna().iloc[5:]
    assert (valid == float(DirectionLabel.FLAT)).all()


def test_T1_dir_h5_nan_count_is_5():
    df = _make_ohlcv(50)
    t = compute_targets(df, horizons=(5,))
    assert t["y_dir_h5"].iloc[-5:].isna().all()
    assert t["y_dir_h5"].iloc[:-5].notna().any()


# ---------------------------------------------------------------------------
# T2: Return target
# ---------------------------------------------------------------------------

def test_T2_ret_known_value_at_h1():
    """y_ret_h1[0] = log(close[1] / close[0])."""
    df = _make_ohlcv(50)
    t  = compute_targets(df, horizons=(1,))
    expected = math.log(df["close"].iloc[1] / df["close"].iloc[0])
    assert abs(t["y_ret_h1"].iloc[0] - expected) < 1e-9


def test_T2_ret_known_value_at_h3():
    """y_ret_h3[0] = log(close[3] / close[0])."""
    df = _make_ohlcv(50)
    t  = compute_targets(df, horizons=(3,))
    expected = math.log(df["close"].iloc[3] / df["close"].iloc[0])
    assert abs(t["y_ret_h3"].iloc[0] - expected) < 1e-9


def test_T2_ret_last_h_rows_nan():
    df = _make_ohlcv(60)
    t  = compute_targets(df, horizons=(1, 3, 5))
    for h in (1, 3, 5):
        assert t[f"y_ret_h{h}"].iloc[-h:].isna().all()


def test_T2_ret_positive_when_UP():
    """When direction is UP, the return must be positive."""
    df = _make_ohlcv(100)
    t  = compute_targets(df, horizons=(1,))
    up_mask = t["y_dir_h1"] == float(DirectionLabel.UP)
    assert (t.loc[up_mask, "y_ret_h1"] > 0).all()


def test_T2_ret_negative_when_DOWN():
    """When direction is DOWN, the return must be negative."""
    df = _make_ohlcv(100)
    t  = compute_targets(df, horizons=(1,))
    dn_mask = t["y_dir_h1"] == float(DirectionLabel.DOWN)
    assert (t.loc[dn_mask, "y_ret_h1"] < 0).all()


# ---------------------------------------------------------------------------
# T3: Magnitude target
# ---------------------------------------------------------------------------

def test_T3_mag_nonnegative():
    df = _make_ohlcv(100)
    t  = compute_targets(df, horizons=(1, 3, 5))
    for h in (1, 3, 5):
        valid = t[f"y_mag_h{h}"].dropna()
        assert (valid >= 0).all()


def test_T3_mag_equals_abs_ret():
    """y_mag_h must be exactly |y_ret_h|."""
    df = _make_ohlcv(100)
    t  = compute_targets(df)
    for h in HORIZONS:
        diff = (t[f"y_mag_h{h}"] - t[f"y_ret_h{h}"].abs()).dropna()
        assert (diff.abs() < 1e-12).all()


def test_T3_mag_zero_for_flat_series():
    df = _make_flat(50)
    t  = compute_targets(df, horizons=(1,))
    valid = t["y_mag_h1"].dropna()
    assert (valid < 1e-9).all()


# ---------------------------------------------------------------------------
# T4: RVOL target
# ---------------------------------------------------------------------------

def test_T4_rvol_h1_equals_mag_h1():
    """For h=1, rvol = |one-bar log-return|, which equals magnitude."""
    df = _make_ohlcv(100)
    t  = compute_targets(df, horizons=(1,))
    diff = (t["y_rvol_h1"] - t["y_mag_h1"]).dropna()
    assert (diff.abs() < 1e-9).all()


def test_T4_rvol_nonnegative():
    df = _make_ohlcv(100)
    t  = compute_targets(df)
    for h in HORIZONS:
        valid = t[f"y_rvol_h{h}"].dropna()
        assert (valid >= 0).all()


def test_T4_rvol_last_h_rows_nan():
    df = _make_ohlcv(60)
    t  = compute_targets(df, horizons=(3, 5))
    for h in (3, 5):
        assert t[f"y_rvol_h{h}"].iloc[-h:].isna().all()


def test_T4_rvol_h3_known_value():
    """
    For a series with known returns, y_rvol_h3[0] = sqrt(mean(r1^2, r2^2, r3^2))
    where r_j = log(close[j] / close[j-1]).
    """
    close = np.array([100.0, 101.0, 103.0, 106.0, 110.0, 115.0])
    df = pd.DataFrame({
        "bar_open_time": pd.date_range("2024-01-02", periods=len(close), freq="5min"),
        "open": close - 0.1, "high": close + 0.5,
        "low": close - 0.5, "close": close,
        "volume": np.full(len(close), 5000.0),
        "vwap": close,
    })
    t = compute_targets(df, horizons=(3,))
    r1 = math.log(101.0 / 100.0)
    r2 = math.log(103.0 / 101.0)
    r3 = math.log(106.0 / 103.0)
    expected = math.sqrt((r1**2 + r2**2 + r3**2) / 3)
    assert abs(t["y_rvol_h3"].iloc[0] - expected) < 1e-9


def test_T4_rvol_zero_for_flat_series():
    df = _make_flat(50)
    t  = compute_targets(df, horizons=(3,))
    valid = t["y_rvol_h3"].dropna()
    assert (valid < 1e-9).all()


# ---------------------------------------------------------------------------
# T5: Abstain flag
# ---------------------------------------------------------------------------

def test_T5_abstain_values_binary_or_nan():
    df = _make_ohlcv(100)
    t  = compute_targets(df)
    for h in HORIZONS:
        valid = t[f"y_abstain_h{h}"].dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})


def test_T5_abstain_matches_flat_direction():
    """y_abstain_h == 1 exactly when y_dir_h == FLAT (1.0)."""
    df = _make_ohlcv(100)
    t  = compute_targets(df)
    for h in HORIZONS:
        valid = t[[f"y_dir_h{h}", f"y_abstain_h{h}"]].dropna()
        expected_abstain = (valid[f"y_dir_h{h}"] == float(DirectionLabel.FLAT)).astype(float)
        pd.testing.assert_series_equal(
            valid[f"y_abstain_h{h}"].reset_index(drop=True),
            expected_abstain.reset_index(drop=True),
            check_names=False,
        )


def test_T5_abstain_rate_increases_with_horizon():
    """
    Wider threshold at sqrt(h) means more bars classified as FLAT at longer horizons.
    For a random series, flat_rate_h5 >= flat_rate_h1 in expectation.
    """
    df = _make_ohlcv(500, seed=7)
    t  = compute_targets(df)
    flat_rate_1 = t["y_abstain_h1"].dropna().mean()
    flat_rate_5 = t["y_abstain_h5"].dropna().mean()
    assert flat_rate_5 >= flat_rate_1, (
        f"Expected flat_rate_h5 >= flat_rate_h1 but got {flat_rate_5:.3f} < {flat_rate_1:.3f}"
    )


def test_T5_flat_series_all_abstain():
    """A completely flat series → y_abstain == 1 for all non-NaN rows."""
    df = _make_flat(60)
    t  = compute_targets(df, horizons=(1,))
    valid = t["y_abstain_h1"].dropna()
    assert (valid == 1.0).all()


# ---------------------------------------------------------------------------
# T6: Leakage tests
# ---------------------------------------------------------------------------

def test_T6_appending_future_bars_does_not_change_past_targets():
    """
    CRITICAL leakage test. Appending 10 extra bars to the end of the series
    must not change any non-NaN target value computed on the original series.

    This verifies that:
      1. The ATR threshold uses only past data (shift-safe).
      2. y_ret and y_dir for row i depend only on close[i+h] — which is fixed
         as long as we compare rows where the label was already computable.
    """
    df = _make_ohlcv(100, seed=42)
    extra = _make_ohlcv(10, seed=99)
    extra["bar_open_time"] = df["bar_open_time"].iloc[-1] + pd.to_timedelta(
        (extra.index + 1) * 5, unit="min"
    )
    df_long = pd.concat([df, extra], ignore_index=True)

    t_short = compute_targets(df, horizons=(1, 3, 5))
    t_long  = compute_targets(df_long, horizons=(1, 3, 5))

    for h in (1, 3, 5):
        for col in [f"y_dir_h{h}", f"y_ret_h{h}", f"y_mag_h{h}", f"y_rvol_h{h}"]:
            # Compare only rows that were non-NaN in the shorter computation
            original_valid = t_short[col].notna()
            orig = t_short.loc[original_valid, col].values
            extended = t_long.iloc[:len(df)].loc[original_valid, col].values
            max_diff = np.nanmax(np.abs(orig - extended))
            assert max_diff < 1e-9, (
                f"{col}: max diff {max_diff:.2e} after appending future bars — possible leakage"
            )


def test_T6_target_window_exactly_i_to_i_plus_h():
    """
    y_dir_h3[i] depends on close[i+3] but NOT on close[i+4].
    Changing close[i+4] must not change y_dir_h3[i] or y_ret_h3[i].
    Changing close[i+3] MUST change y_ret_h3[i].
    """
    df = _make_ohlcv(50, seed=3)
    h  = 3
    i  = 10   # arbitrary test row

    t_original = compute_targets(df, horizons=(h,))

    # Change close[i+h+1] (beyond the label window)
    df2 = df.copy()
    df2.loc[df2.index[i + h + 1], "close"] *= 1.50
    t_beyond = compute_targets(df2, horizons=(h,))
    assert abs(t_original[f"y_ret_h{h}"].iloc[i] - t_beyond[f"y_ret_h{h}"].iloc[i]) < 1e-9

    # Change close[i+h] (inside the label window) — return MUST change
    df3 = df.copy()
    df3.loc[df3.index[i + h], "close"] *= 1.50
    t_inside = compute_targets(df3, horizons=(h,))
    assert abs(t_original[f"y_ret_h{h}"].iloc[i] - t_inside[f"y_ret_h{h}"].iloc[i]) > 0.01


def test_T6_atr_threshold_is_past_only():
    """
    The ATR-based threshold uses only past data (shift-safe).
    Computing _atr_norm on df[:50] vs df[:60] must give identical values at rows 0..49.
    """
    df = _make_ohlcv(60, seed=5)
    atr_short = _atr_norm(df.iloc[:50]).values
    atr_full  = _atr_norm(df).values[:50]
    assert np.nanmax(np.abs(atr_short - atr_full)) < 1e-9


def test_T6_nan_count_exactly_equals_horizon():
    """The last h rows must be NaN; earlier rows must be non-NaN (except ATR warmup)."""
    df = _make_ohlcv(100)
    t  = compute_targets(df, horizons=(1, 3, 5))
    for h in (1, 3, 5):
        nan_tail = t[f"y_ret_h{h}"].iloc[-h:].isna().sum()
        assert nan_tail == h, f"Expected exactly {h} trailing NaN for y_ret_h{h}"
        # The NaN should be at the very end, not scattered
        assert pd.notna(t[f"y_ret_h{h}"].iloc[-(h + 1)])


def test_T6_rvol_window_exactly_i_to_i_plus_h():
    """
    y_rvol_h3[i] depends on close[i+1], close[i+2], close[i+3].
    Changing close[i+4] must not change y_rvol_h3[i].
    """
    df = _make_ohlcv(50, seed=4)
    h  = 3
    i  = 10

    t_orig = compute_targets(df, horizons=(h,))

    df2 = df.copy()
    df2.loc[df2.index[i + h + 1], "close"] *= 2.0
    t_mod = compute_targets(df2, horizons=(h,))

    assert abs(t_orig[f"y_rvol_h{h}"].iloc[i] - t_mod[f"y_rvol_h{h}"].iloc[i]) < 1e-9


# ---------------------------------------------------------------------------
# T7: API
# ---------------------------------------------------------------------------

def test_T7_compute_targets_returns_all_columns():
    df = _make_ohlcv(60)
    t  = compute_targets(df)
    expected = ["bar_open_time"] + target_col_names()
    for col in expected:
        assert col in t.columns, f"Missing column: {col}"


def test_T7_target_col_names_matches_output_columns():
    names = target_col_names()
    df = _make_ohlcv(60)
    t  = compute_targets(df)
    for name in names:
        assert name in t.columns


def test_T7_custom_horizons():
    df = _make_ohlcv(60)
    t  = compute_targets(df, horizons=(2,))
    assert "y_dir_h2" in t.columns
    assert "y_ret_h2" in t.columns
    assert "y_dir_h1" not in t.columns


def test_T7_single_horizon_returns_correct_rows():
    n  = 60
    df = _make_ohlcv(n)
    t  = compute_targets(df, horizons=(1,))
    assert len(t) == n


def test_T7_target_col_names_custom_horizons():
    names = target_col_names((2, 4))
    assert names == ["y_dir_h2", "y_ret_h2", "y_mag_h2", "y_rvol_h2", "y_abstain_h2",
                     "y_dir_h4", "y_ret_h4", "y_mag_h4", "y_rvol_h4", "y_abstain_h4"]


# ---------------------------------------------------------------------------
# T8: Regime labels — basics
# ---------------------------------------------------------------------------

def test_T8_all_regime_cols_present():
    df = _make_ohlcv(80)
    r  = compute_regime_labels(df)
    for col in REGIME_LABEL_COLS:
        assert col in r.columns, f"Missing column: {col}"


def test_T8_all_bool_dtype():
    df = _make_ohlcv(80)
    r  = compute_regime_labels(df)
    for col in REGIME_LABEL_COLS:
        assert r[col].dtype == bool, f"{col} should be bool, got {r[col].dtype}"


def test_T8_warmup_rows_are_all_regime_warmup():
    df = _make_ohlcv(80)
    r  = compute_regime_labels(df)
    assert r["regime_warmup"].iloc[:WARMUP_BARS].all()
    assert not r["regime_warmup"].iloc[WARMUP_BARS:].any()


def test_T8_trending_and_mean_reverting_are_complements_post_warmup():
    """
    Post-warmup: exactly one of regime_trending / regime_mean_reverting is True
    at every row (they are strict complements, not independent).
    """
    df = _make_ohlcv(100)
    r  = compute_regime_labels(df)
    post = r.iloc[WARMUP_BARS:]
    xor  = post["regime_trending"] ^ post["regime_mean_reverting"]
    assert xor.all(), "regime_trending and regime_mean_reverting should be strict complements"


def test_T8_warmup_rows_have_no_other_flags():
    """During warmup, all regime flags except regime_warmup should be False."""
    df = _make_ohlcv(80)
    r  = compute_regime_labels(df)
    other_cols = [c for c in REGIME_LABEL_COLS if c != "regime_warmup"]
    warmup_rows = r.iloc[:WARMUP_BARS]
    for col in other_cols:
        assert not warmup_rows[col].any(), f"{col} should be False during warmup"


# ---------------------------------------------------------------------------
# T9: Regime label correctness
# ---------------------------------------------------------------------------

def test_T9_high_adx_series_triggers_trending():
    """
    A strongly directional series (long monotone trend) should generate
    regime_trending=True in the post-warmup window.
    """
    df = _make_rising(120)
    r  = compute_regime_labels(df)
    post = r.iloc[WARMUP_BARS + 20:]   # allow additional warmup for ADX convergence
    assert post["regime_trending"].any(), "Monotone rising series should trigger regime_trending"


def test_T9_low_atr_ratio_triggers_low_vol():
    """
    A series with compressed volatility (tiny range, constant returns) should
    eventually trigger regime_low_vol once the long-run ATR reference is established.
    """
    # Build series: start with normal vol to establish long-run ATR, then compress
    df_normal = _make_ohlcv(60, seed=0)
    close = df_normal["close"].values.copy()
    # Last 40 bars: near-flat with tiny range
    for i in range(60, 100):
        close_val = close[-1] * (1 + 0.0001)
        close = np.append(close, close_val)
    times = pd.date_range("2024-01-02 14:30", periods=100, freq="5min")
    opens = close - 0.01
    highs = close + 0.02
    lows  = close - 0.02
    vols  = np.full(100, 5000.0)
    df = pd.DataFrame({
        "bar_open_time": times, "open": opens, "high": highs,
        "low": lows, "close": close, "volume": vols,
        "vwap": (highs + lows + close) / 3,
    })
    r = compute_regime_labels(df)
    assert r["regime_low_vol"].any(), "Near-flat series should trigger regime_low_vol"


def test_T9_elevated_atr_triggers_high_vol():
    """
    A series with a sudden spike in bar ranges should trigger regime_high_vol.
    """
    df = _make_ohlcv(100, seed=1)
    df = df.copy()
    # Inject a volatility spike: expand ranges for bars 40-60
    df.loc[df.index[40:60], "high"] = df["close"].iloc[40:60] * 1.05
    df.loc[df.index[40:60], "low"]  = df["close"].iloc[40:60] * 0.95
    r = compute_regime_labels(df)
    assert r.iloc[50:]["regime_high_vol"].any(), "Spiked ranges should trigger regime_high_vol"


def test_T9_low_volume_triggers_liquidity_poor():
    """
    Sustained low volume should trigger regime_liquidity_poor.
    """
    df = _make_ohlcv(100, seed=2)
    df = df.copy()
    # Set last 30 bars to near-zero volume
    df.loc[df.index[70:], "volume"] = 1.0
    r = compute_regime_labels(df)
    assert r.iloc[75:]["regime_liquidity_poor"].any(), "Near-zero volume should trigger liquidity_poor"


def test_T9_large_return_triggers_abnormal():
    """
    A single very large return (>3σ) should trigger regime_abnormal.
    """
    df = _make_ohlcv(100, seed=3)
    df = df.copy()
    # Inject a 10% return spike at bar 50
    spike_close = df["close"].iloc[49] * 1.10
    df.loc[df.index[50], "close"] = spike_close
    df.loc[df.index[50], "high"]  = spike_close + 0.5
    r = compute_regime_labels(df)
    # Allow a few bars for the detection to propagate
    assert r.iloc[50:55]["regime_abnormal"].any(), "10% move should trigger regime_abnormal"


def test_T9_normal_series_has_no_abnormal():
    """A gentle random walk should rarely (or never) trigger regime_abnormal."""
    rng = np.random.default_rng(999)
    close = 100.0 + np.cumsum(rng.normal(0, 0.1, 200))    # very small steps
    close = np.clip(close, 1, None)
    times = pd.date_range("2024-01-02 14:30", periods=200, freq="5min")
    opens = close - 0.05
    highs = close + 0.1
    lows  = np.clip(close - 0.1, 0.01, None)
    vols  = np.full(200, 5000.0)
    df = pd.DataFrame({
        "bar_open_time": times, "open": opens, "high": highs,
        "low": lows, "close": close, "volume": vols,
        "vwap": (highs + lows + close) / 3,
    })
    r = compute_regime_labels(df)
    # Expect no abnormal flags for a very low-vol random walk
    assert not r["regime_abnormal"].any(), "Tiny-move series should not trigger regime_abnormal"


# ---------------------------------------------------------------------------
# T10: Regime shift-by-1 invariant
# ---------------------------------------------------------------------------

def test_T10_appending_future_bars_does_not_change_regime_labels():
    """
    Appending new bars to the end of the series must not change any regime label
    for prior bars. This confirms all signals use only past data (shift-safe).
    """
    df = _make_ohlcv(100, seed=10)
    extra = _make_ohlcv(10, seed=11)
    extra["bar_open_time"] = df["bar_open_time"].iloc[-1] + pd.to_timedelta(
        (extra.index + 1) * 5, unit="min"
    )
    df_long = pd.concat([df, extra], ignore_index=True)

    r_short = compute_regime_labels(df)
    r_long  = compute_regime_labels(df_long)

    for col in REGIME_LABEL_COLS:
        orig  = r_short[col].values
        ext   = r_long.iloc[:len(df)][col].values
        diffs = orig != ext
        assert not diffs.any(), (
            f"{col}: regime changed at rows {np.where(diffs)[0].tolist()} after appending bars — leakage"
        )


def test_T10_changing_future_bar_does_not_change_regime_at_i():
    """
    Modifying bar i+5 should not change the regime label at bar i.
    """
    df = _make_ohlcv(80, seed=12)
    i  = 30

    r_orig = compute_regime_labels(df)

    df2 = df.copy()
    df2.loc[df2.index[i + 5], "close"]  *= 5.0   # extreme change to future bar
    df2.loc[df2.index[i + 5], "high"]   *= 5.0
    df2.loc[df2.index[i + 5], "volume"] *= 0.001

    r_mod = compute_regime_labels(df2)

    for col in REGIME_LABEL_COLS:
        assert r_orig[col].iloc[i] == r_mod[col].iloc[i], (
            f"{col}: regime at bar {i} changed when bar {i+5} was modified — lookahead detected"
        )


# ---------------------------------------------------------------------------
# T11: Embargo / overlap structure
# ---------------------------------------------------------------------------

def test_T11_h5_last_5_rows_all_targets_nan():
    """For h=5, all target columns must be NaN in the last 5 rows."""
    df = _make_ohlcv(60)
    t  = compute_targets(df, horizons=(5,))
    for col in ["y_dir_h5", "y_ret_h5", "y_mag_h5", "y_rvol_h5", "y_abstain_h5"]:
        assert t[col].iloc[-5:].isna().all(), f"Expected NaN in last 5 rows for {col}"


def test_T11_rvol_h5_depends_on_exactly_5_future_bars():
    """
    y_rvol_h5[i] depends on log-returns at bars i+1..i+5.
    Changing close[i+6] must not change y_rvol_h5[i].
    Changing close[i+5] MUST change y_rvol_h5[i].
    """
    df = _make_ohlcv(60, seed=20)
    h  = 5
    i  = 10

    t_orig = compute_targets(df, horizons=(h,))

    # Perturb beyond window — must not change y_rvol_h5[i]
    df2 = df.copy()
    df2.loc[df2.index[i + h + 1], "close"] *= 3.0
    t_beyond = compute_targets(df2, horizons=(h,))
    assert abs(t_orig[f"y_rvol_h{h}"].iloc[i] - t_beyond[f"y_rvol_h{h}"].iloc[i]) < 1e-9

    # Perturb inside window — must change y_rvol_h5[i]
    df3 = df.copy()
    df3.loc[df3.index[i + h], "close"] *= 1.50
    t_inside = compute_targets(df3, horizons=(h,))
    assert abs(t_orig[f"y_rvol_h{h}"].iloc[i] - t_inside[f"y_rvol_h{h}"].iloc[i]) > 1e-6


def test_T11_embargo_requirement_is_horizon():
    """
    For consecutive rows i and i+1, y_rvol_h{h} shares h-1 bars.
    The minimum safe embargo for h-bar targets is h bars.
    This test verifies the overlap count explicitly.
    """
    h = 5
    df = _make_ohlcv(50, seed=21)
    # log-returns: r[j] = log(close[j] / close[j-1])
    log_ret = np.log(df["close"] / df["close"].shift(1))
    i = 10

    # Window for y_rvol_h5[i]:   r[i+1..i+5]
    window_i   = set(range(i + 1, i + h + 1))
    # Window for y_rvol_h5[i+1]: r[i+2..i+6]
    window_i1  = set(range(i + 2, i + h + 2))

    overlap = window_i & window_i1
    assert len(overlap) == h - 1, f"Expected {h-1} overlapping bars, got {len(overlap)}"


def test_T11_long_horizon_needs_more_nan():
    """Longer horizons produce more trailing NaN rows than shorter horizons."""
    df = _make_ohlcv(80)
    t  = compute_targets(df, horizons=(1, 3, 5))
    nan1 = t["y_ret_h1"].isna().sum()
    nan3 = t["y_ret_h3"].isna().sum()
    nan5 = t["y_ret_h5"].isna().sum()
    assert nan1 <= nan3 <= nan5
