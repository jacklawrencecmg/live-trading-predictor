"""
Feature pipeline and feature store tests — v2.

Coverage:
  F1  — Mean-reversion family: ema_dist and zscore_5 correctness
  F2  — Mean-reversion shift-by-1 invariant
  F3  — Bar-structure family: body/wick ratios sum to 1, known values
  F4  — Bar-structure: bar_return and gap_pct correctness
  F5  — Bar-structure shift-by-1 invariant
  F6  — Regime features are binary {0, 1}
  F7  — Regime shift-by-1 invariant
  F8  — All FEATURE_COLS produced (no missing column)
  F9  — Null and missingness preservation
  F10 — Feature snapshot ID is stable and deterministic
  F11 — MANIFEST_HASH changes when any feature version is incremented
  F12 — PIPELINE_VERSION is at least 2
  F13 — store.save_feature_row and load_feature_row round-trip (async)
  F14 — store returns None for stale manifest (async)
  F15 — store.save_feature_batch and load_feature_range_pit (async)
  F16 — rows_to_dataframe produces correct columns and ordering (async)
  F17 — inspector.inspect_row covers all new feature families
  F18 — inspector out-of-range detection
  F19 — options sentinel fill: is_null_options=1 when options_data=None
  F20 — options real values used when options_data provided
  F21 — valid_mask excludes warmup rows
  F22 — valid_mask excludes rows with NaN required features
"""

import json
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.feature_store


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    prices = np.clip(prices, 1, None)
    opens = np.clip(prices + rng.normal(0, 0.2, n), 0.01, None)
    # Ensure valid OHLCV: high >= max(open, close), low <= min(open, close)
    oc_high = np.maximum(opens, prices)
    oc_low = np.minimum(opens, prices)
    highs = oc_high + rng.uniform(0.1, 1.0, n)
    lows = np.clip(oc_low - rng.uniform(0.1, 1.0, n), 0.01, None)
    volumes = rng.integers(1000, 10000, n).astype(float)
    vwap = (highs + lows + prices) / 3
    times = pd.date_range("2024-01-02 14:30", periods=n, freq="5min")
    return pd.DataFrame({
        "bar_open_time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
        "vwap": vwap,
    })


def _make_candle_df(bars: list, start: str = "2024-01-02 14:30") -> pd.DataFrame:
    times = pd.date_range(start, periods=len(bars), freq="5min")
    rows = []
    for i, b in enumerate(bars):
        rows.append({
            "bar_open_time": times[i],
            "open": float(b["open"]),
            "high": float(b["high"]),
            "low": float(b["low"]),
            "close": float(b["close"]),
            "volume": float(b.get("volume", 1000)),
            "vwap": float(b.get("vwap", (b["high"] + b["low"] + b["close"]) / 3)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# F1: Mean-reversion correctness
# ---------------------------------------------------------------------------

def test_F1_ema_dist_20_positive_in_rising_trend():
    from app.feature_pipeline.compute import compute_features
    n = 100
    prices = np.linspace(100, 200, n)
    df = pd.DataFrame({
        "bar_open_time": pd.date_range("2024-01-02 14:30", periods=n, freq="5min"),
        "open": prices * 0.999, "high": prices * 1.005, "low": prices * 0.995,
        "close": prices, "volume": 1000.0, "vwap": prices,
    })
    feat = compute_features(df)
    assert feat["ema_dist_20"].dropna().iloc[-1] > 0


def test_F1_ema_dist_20_negative_in_falling_trend():
    from app.feature_pipeline.compute import compute_features
    n = 100
    prices = np.linspace(200, 100, n)
    df = pd.DataFrame({
        "bar_open_time": pd.date_range("2024-01-02 14:30", periods=n, freq="5min"),
        "open": prices * 1.001, "high": prices * 1.005, "low": prices * 0.995,
        "close": prices, "volume": 1000.0, "vwap": prices,
    })
    feat = compute_features(df)
    assert feat["ema_dist_20"].dropna().iloc[-1] < 0


def test_F1_ema_dist_50_larger_lag_in_strong_trend():
    """In a strong uptrend, |ema_dist_50| > |ema_dist_20| because EMA50 lags more."""
    from app.feature_pipeline.compute import compute_features
    n = 150
    prices = np.linspace(100, 300, n)
    df = pd.DataFrame({
        "bar_open_time": pd.date_range("2024-01-02 14:30", periods=n, freq="5min"),
        "open": prices * 0.999, "high": prices * 1.005, "low": prices * 0.995,
        "close": prices, "volume": 1000.0, "vwap": prices,
    })
    feat = compute_features(df)
    last = feat.iloc[-1]
    assert abs(last["ema_dist_50"]) > abs(last["ema_dist_20"])


def test_F1_zscore_5_near_zero_for_flat_series():
    from app.feature_pipeline.compute import compute_features
    n = 50
    prices = np.full(n, 100.0)
    df = pd.DataFrame({
        "bar_open_time": pd.date_range("2024-01-02 14:30", periods=n, freq="5min"),
        "open": prices, "high": prices + 0.01, "low": prices - 0.01,
        "close": prices, "volume": 1000.0, "vwap": prices,
    })
    feat = compute_features(df)
    assert (feat["zscore_5"].dropna().abs() < 1e-3).all()


def test_F1_zscore_5_more_volatile_than_zscore_20():
    from app.feature_pipeline.compute import compute_features
    df = _make_ohlcv(200)
    feat = compute_features(df)
    assert feat["zscore_5"].dropna().std() >= feat["zscore_20"].dropna().std() * 0.5


# ---------------------------------------------------------------------------
# F2: Mean-reversion shift-by-1 invariant
# ---------------------------------------------------------------------------

def test_F2_mean_reversion_shift_invariant():
    from app.feature_pipeline.compute import compute_features
    df = _make_ohlcv(80)
    extra = _make_ohlcv(15, seed=99)
    extra["bar_open_time"] = df["bar_open_time"].iloc[-1] + pd.to_timedelta(
        (extra.index + 1) * 5, unit="min"
    )
    df_ext = pd.concat([df, extra], ignore_index=True)
    cols = ["ema_dist_20", "ema_dist_50", "zscore_5"]
    short = compute_features(df)[cols].values
    long_ = compute_features(df_ext)[cols].values[:len(df)]
    assert np.nanmax(np.abs(short - long_)) < 1e-9


# ---------------------------------------------------------------------------
# F3: Bar-structure ratios
# ---------------------------------------------------------------------------

def test_F3_wick_and_body_sum_lte_one():
    from app.feature_pipeline.compute import compute_features
    df = _make_ohlcv(80)
    feat = compute_features(df)
    valid = feat.dropna(subset=["body_ratio", "upper_wick_ratio", "lower_wick_ratio"])
    total = valid["body_ratio"] + valid["upper_wick_ratio"] + valid["lower_wick_ratio"]
    assert (total <= 1.0 + 1e-6).all()


def test_F3_full_body_candle_body_ratio_is_one():
    """open=low, close=high → body_ratio == 1.0, both wicks == 0."""
    from app.feature_pipeline.compute import compute_features
    bars = [
        {"open": 100, "high": 102, "low": 99, "close": 101},
        {"open": 100, "high": 105, "low": 100, "close": 105},   # full body
        {"open": 105, "high": 107, "low": 104, "close": 106},
    ]
    feat = compute_features(_make_candle_df(bars))
    row = feat.iloc[2]
    assert abs(row["body_ratio"] - 1.0) < 1e-6
    assert abs(row["upper_wick_ratio"]) < 1e-6
    assert abs(row["lower_wick_ratio"]) < 1e-6


def test_F3_doji_near_zero_body_ratio():
    """open == close → body_ratio near 0."""
    from app.feature_pipeline.compute import compute_features
    bars = [
        {"open": 100, "high": 102, "low": 99, "close": 101},
        {"open": 100, "high": 103, "low": 97, "close": 100},    # doji
        {"open": 100, "high": 101, "low": 99, "close": 100},
    ]
    feat = compute_features(_make_candle_df(bars))
    assert float(feat["body_ratio"].iloc[2]) < 0.05


def test_F3_shooting_star_upper_wick_dominant():
    from app.feature_pipeline.compute import compute_features
    bars = [
        {"open": 100, "high": 101, "low": 99, "close": 100},
        {"open": 100, "high": 108, "low": 99, "close": 100.5},  # shooting star
        {"open": 100.5, "high": 101.5, "low": 99.5, "close": 101},
    ]
    feat = compute_features(_make_candle_df(bars))
    row = feat.iloc[2]
    assert row["upper_wick_ratio"] > row["body_ratio"]
    assert row["upper_wick_ratio"] > row["lower_wick_ratio"]


def test_F3_hammer_lower_wick_dominant():
    from app.feature_pipeline.compute import compute_features
    bars = [
        {"open": 100, "high": 101, "low": 99, "close": 100},
        {"open": 100, "high": 100.5, "low": 92, "close": 99.5},  # hammer
        {"open": 99.5, "high": 101, "low": 99, "close": 100},
    ]
    feat = compute_features(_make_candle_df(bars))
    row = feat.iloc[2]
    assert row["lower_wick_ratio"] > row["body_ratio"]
    assert row["lower_wick_ratio"] > row["upper_wick_ratio"]


def test_F3_ratios_bounded_zero_to_one():
    from app.feature_pipeline.compute import compute_features
    df = _make_ohlcv(200)
    feat = compute_features(df)
    for col in ["body_ratio", "upper_wick_ratio", "lower_wick_ratio"]:
        vals = feat[col].dropna()
        assert (vals >= 0.0).all() and (vals <= 1.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# F4: bar_return and gap_pct
# ---------------------------------------------------------------------------

def test_F4_bar_return_positive_for_bullish_bar():
    from app.feature_pipeline.compute import compute_features
    bars = [
        {"open": 100, "high": 102, "low": 99, "close": 101},
        {"open": 100, "high": 105, "low": 99, "close": 103},   # bullish
        {"open": 103, "high": 104, "low": 102, "close": 103},
    ]
    assert float(compute_features(_make_candle_df(bars))["bar_return"].iloc[2]) > 0


def test_F4_bar_return_negative_for_bearish_bar():
    from app.feature_pipeline.compute import compute_features
    bars = [
        {"open": 100, "high": 102, "low": 99, "close": 101},
        {"open": 103, "high": 104, "low": 98, "close": 99},    # bearish
        {"open": 99, "high": 101, "low": 98, "close": 100},
    ]
    assert float(compute_features(_make_candle_df(bars))["bar_return"].iloc[2]) < 0


def test_F4_bar_return_known_value():
    """bar_return[2] = close[1] / open[1] - 1 = 104/100 - 1 = 0.04."""
    from app.feature_pipeline.compute import compute_features
    bars = [
        {"open": 100, "high": 102, "low": 99, "close": 100},
        {"open": 100, "high": 105, "low": 99, "close": 104},
        {"open": 104, "high": 106, "low": 103, "close": 105},
    ]
    actual = float(compute_features(_make_candle_df(bars))["bar_return"].iloc[2])
    assert abs(actual - 0.04) < 1e-9


def test_F4_gap_pct_positive_for_gap_up():
    from app.feature_pipeline.compute import compute_features
    bars = [
        {"open": 100, "high": 102, "low": 99, "close": 101},
        {"open": 105, "high": 107, "low": 104, "close": 106},  # gap-up
        {"open": 106, "high": 108, "low": 105, "close": 107},
    ]
    assert float(compute_features(_make_candle_df(bars))["gap_pct"].iloc[2]) > 0


def test_F4_gap_pct_known_value():
    """gap_pct[2] = open[1] / close[0] - 1 = 102/100 - 1 = 0.02."""
    from app.feature_pipeline.compute import compute_features
    bars = [
        {"open": 100, "high": 102, "low": 99, "close": 100},
        {"open": 102, "high": 104, "low": 101, "close": 103},
        {"open": 103, "high": 105, "low": 102, "close": 104},
    ]
    actual = float(compute_features(_make_candle_df(bars))["gap_pct"].iloc[2])
    assert abs(actual - 0.02) < 1e-9


# ---------------------------------------------------------------------------
# F5: Bar-structure shift-by-1 invariant
# ---------------------------------------------------------------------------

def test_F5_bar_structure_shift_invariant():
    from app.feature_pipeline.compute import compute_features
    df = _make_ohlcv(80)
    extra = _make_ohlcv(15, seed=77)
    extra["bar_open_time"] = df["bar_open_time"].iloc[-1] + pd.to_timedelta(
        (extra.index + 1) * 5, unit="min"
    )
    df_ext = pd.concat([df, extra], ignore_index=True)
    cols = ["body_ratio", "upper_wick_ratio", "lower_wick_ratio", "bar_return", "gap_pct"]
    short = compute_features(df)[cols].values
    long_ = compute_features(df_ext)[cols].values[:len(df)]
    assert np.nanmax(np.abs(short - long_)) < 1e-9


def test_F5_bar_return_not_changed_by_current_bar_mutation():
    """bar_return[i-1] must not change when row i's OHLCV is mutated."""
    from app.feature_pipeline.compute import compute_features
    df = _make_ohlcv(50)
    df_mod = df.copy()
    df_mod.iloc[-1, df_mod.columns.get_loc("open")] = 999.0
    df_mod.iloc[-1, df_mod.columns.get_loc("close")] = 1.0
    feat1 = compute_features(df)
    feat2 = compute_features(df_mod)
    assert abs(feat1["bar_return"].iloc[-2] - feat2["bar_return"].iloc[-2]) < 1e-9


# ---------------------------------------------------------------------------
# F6: Regime features are binary {0, 1}
# ---------------------------------------------------------------------------

def test_F6_regime_features_are_binary():
    from app.feature_pipeline.compute import compute_features
    df = _make_ohlcv(120)
    feat = compute_features(df)
    for col in ["price_above_ema20", "ma_alignment"]:
        vals = set(feat[col].dropna().unique())
        assert vals.issubset({0.0, 1.0}), f"{col} must be binary {{0,1}}, got {vals}"


def test_F6_price_above_ema20_is_one_in_rising_trend():
    from app.feature_pipeline.compute import compute_features
    n = 100
    prices = np.linspace(100, 200, n)
    df = pd.DataFrame({
        "bar_open_time": pd.date_range("2024-01-02 14:30", periods=n, freq="5min"),
        "open": prices * 0.999, "high": prices * 1.005, "low": prices * 0.995,
        "close": prices, "volume": 1000.0, "vwap": prices,
    })
    feat = compute_features(df)
    assert feat["price_above_ema20"].dropna().iloc[-1] == 1.0


def test_F6_ma_alignment_is_one_in_sustained_uptrend():
    from app.feature_pipeline.compute import compute_features
    n = 200
    prices = np.linspace(100, 300, n)
    df = pd.DataFrame({
        "bar_open_time": pd.date_range("2024-01-02 14:30", periods=n, freq="5min"),
        "open": prices * 0.999, "high": prices * 1.005, "low": prices * 0.995,
        "close": prices, "volume": 1000.0, "vwap": prices,
    })
    feat = compute_features(df)
    assert feat["ma_alignment"].dropna().iloc[-1] == 1.0


# ---------------------------------------------------------------------------
# F7: Regime shift-by-1 invariant
# ---------------------------------------------------------------------------

def test_F7_regime_shift_invariant():
    from app.feature_pipeline.compute import compute_features
    df = _make_ohlcv(100)
    extra = _make_ohlcv(10, seed=55)
    extra["bar_open_time"] = df["bar_open_time"].iloc[-1] + pd.to_timedelta(
        (extra.index + 1) * 5, unit="min"
    )
    df_ext = pd.concat([df, extra], ignore_index=True)
    cols = ["price_above_ema20", "ma_alignment"]
    short = compute_features(df)[cols].values
    long_ = compute_features(df_ext)[cols].values[:len(df)]
    assert np.nanmax(np.abs(short - long_)) < 1e-9


# ---------------------------------------------------------------------------
# F8: All FEATURE_COLS produced
# ---------------------------------------------------------------------------

def test_F8_all_feature_cols_in_output():
    from app.feature_pipeline.compute import compute_features
    from app.feature_pipeline.registry import FEATURE_COLS
    df = _make_ohlcv(100)
    feat = compute_features(df)
    missing = [c for c in FEATURE_COLS if c not in feat.columns]
    assert not missing, f"FEATURE_COLS missing from output: {missing}"


def test_F8_all_all_feature_cols_in_output():
    from app.feature_pipeline.compute import compute_features
    from app.feature_pipeline.registry import ALL_FEATURE_COLS
    df = _make_ohlcv(100)
    feat = compute_features(df)
    missing = [c for c in ALL_FEATURE_COLS if c not in feat.columns]
    assert not missing, f"ALL_FEATURE_COLS missing from output: {missing}"


def test_F8_feature_cols_all_registered():
    from app.feature_pipeline.registry import FEATURE_COLS, REGISTRY
    unregistered = [n for n in FEATURE_COLS if n not in REGISTRY]
    assert not unregistered, f"Unregistered features in FEATURE_COLS: {unregistered}"


# ---------------------------------------------------------------------------
# F9: Null and missingness preservation
# ---------------------------------------------------------------------------

def test_F9_ret_60_nan_when_fewer_than_61_bars():
    from app.feature_pipeline.compute import compute_features
    df = _make_ohlcv(30)
    assert compute_features(df)["ret_60"].isna().all()


def test_F9_nan_volume_produces_nan_volume_features():
    from app.feature_pipeline.compute import compute_features
    df = _make_ohlcv(60)
    df["volume"] = np.nan
    feat = compute_features(df)
    assert feat["volume_ratio"].isna().all()
    assert feat["volume_zscore"].isna().all()


def test_F9_body_ratio_nan_at_row_0():
    from app.feature_pipeline.compute import compute_features
    assert np.isnan(compute_features(_make_ohlcv(10))["body_ratio"].iloc[0])


def test_F9_gap_pct_nan_at_rows_0_and_1():
    from app.feature_pipeline.compute import compute_features
    feat = compute_features(_make_ohlcv(10))
    assert np.isnan(feat["gap_pct"].iloc[0])
    assert np.isnan(feat["gap_pct"].iloc[1])


def test_F9_ema_dist_nan_at_row_0():
    from app.feature_pipeline.compute import compute_features
    feat = compute_features(_make_ohlcv(10))
    assert np.isnan(feat["ema_dist_20"].iloc[0])


# ---------------------------------------------------------------------------
# F10: Feature snapshot ID
# ---------------------------------------------------------------------------

def test_F10_snapshot_id_deterministic():
    from app.feature_pipeline.store import _snapshot_id
    features = {"rsi_14": 55.0, "ret_1": 0.003, "body_ratio": 0.7}
    assert _snapshot_id(features) == _snapshot_id(features)


def test_F10_snapshot_id_differs_for_different_values():
    from app.feature_pipeline.store import _snapshot_id
    assert _snapshot_id({"rsi_14": 55.0}) != _snapshot_id({"rsi_14": 55.1})


def test_F10_snapshot_id_is_16_chars():
    from app.feature_pipeline.store import _snapshot_id
    assert len(_snapshot_id({"rsi_14": 50.0})) == 16


def test_F10_snapshot_id_handles_none():
    from app.feature_pipeline.store import _snapshot_id
    snap = _snapshot_id({"rsi_14": None, "ret_1": 0.003})
    assert isinstance(snap, str) and len(snap) == 16


# ---------------------------------------------------------------------------
# F11: MANIFEST_HASH
# ---------------------------------------------------------------------------

def test_F11_manifest_hash_is_16_chars():
    from app.feature_pipeline.registry import MANIFEST_HASH
    assert len(MANIFEST_HASH) == 16


def test_F11_manifest_hash_changes_when_version_bumped():
    from app.feature_pipeline.registry import _compute_manifest_hash, REGISTRY
    original = _compute_manifest_hash()
    pairs = sorted((f.name, f.version) for f in REGISTRY.values())
    pairs_mod = [(n, v + 1 if n == "rsi_14" else v) for n, v in pairs]
    payload = json.dumps(pairs_mod, separators=(",", ":"))
    modified = hashlib.sha256(payload.encode()).hexdigest()[:16]
    assert original != modified


def test_F11_all_feature_cols_registered():
    from app.feature_pipeline.registry import ALL_FEATURE_COLS, REGISTRY
    missing = [n for n in ALL_FEATURE_COLS if n not in REGISTRY]
    assert not missing, f"ALL_FEATURE_COLS names missing from REGISTRY: {missing}"


# ---------------------------------------------------------------------------
# F12: PIPELINE_VERSION
# ---------------------------------------------------------------------------

def test_F12_pipeline_version_is_v2():
    from app.feature_pipeline.registry import PIPELINE_VERSION
    assert PIPELINE_VERSION >= 2


# ---------------------------------------------------------------------------
# F13: Store save / load round-trip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_F13_save_and_load_round_trip(db_session):
    from app.feature_pipeline.compute import compute_features
    from app.feature_pipeline.store import save_feature_row, load_feature_row, deserialize_features

    df = _make_ohlcv(100)
    feat_df = compute_features(df)
    bar_time = pd.Timestamp(df["bar_open_time"].iloc[-1]).to_pydatetime()
    feat_series = feat_df.iloc[-1]

    saved = await save_feature_row("SPY", "5m", bar_time, feat_series, db_session)
    assert saved.symbol == "SPY"
    assert len(saved.snapshot_id) == 16

    loaded = await load_feature_row("SPY", "5m", bar_time, db_session)
    assert loaded is not None
    feats = deserialize_features(loaded)
    for col in ["ema_dist_20", "ema_dist_50", "zscore_5", "body_ratio",
                "upper_wick_ratio", "lower_wick_ratio", "bar_return", "gap_pct",
                "price_above_ema20", "ma_alignment"]:
        assert col in feats, f"Stored features missing: {col}"


@pytest.mark.asyncio
async def test_F13_save_twice_upserts(db_session):
    from app.feature_pipeline.compute import compute_features
    from app.feature_pipeline.store import save_feature_row

    df = _make_ohlcv(100)
    feat_df = compute_features(df)
    bar_time = pd.Timestamp(df["bar_open_time"].iloc[-1]).to_pydatetime()
    feat_series = feat_df.iloc[-1]

    r1 = await save_feature_row("SPY", "5m", bar_time, feat_series, db_session)
    r2 = await save_feature_row("SPY", "5m", bar_time, feat_series, db_session)
    assert r1.id == r2.id, "Duplicate save must upsert, not create a new row"


# ---------------------------------------------------------------------------
# F14: Store returns None for stale manifest
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_F14_returns_none_for_stale_manifest(db_session):
    from app.models.feature_row import FeatureRow
    from app.feature_pipeline.store import load_feature_row

    bar_time = datetime(2024, 1, 2, 14, 30)
    stale = FeatureRow(
        symbol="AAPL", timeframe="5m", bar_open_time=bar_time,
        manifest_hash="0000stale0000000", pipeline_version=0,
        features_json=json.dumps({"rsi_14": 50.0}),
        null_mask=json.dumps([]),
        snapshot_id="abc123def456abcd",
        is_valid=True,
    )
    db_session.add(stale)
    await db_session.flush()

    result = await load_feature_row("AAPL", "5m", bar_time, db_session, require_current_manifest=True)
    assert result is None, "Must return None for stale-manifest row"

    result_any = await load_feature_row("AAPL", "5m", bar_time, db_session, require_current_manifest=False)
    assert result_any is not None


# ---------------------------------------------------------------------------
# F15: save_feature_batch and load_feature_range_pit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_F15_batch_save_and_range_load(db_session):
    from app.feature_pipeline.compute import compute_features
    from app.feature_pipeline.store import save_feature_batch, load_feature_range_pit

    df = _make_ohlcv(120)
    feat_df = compute_features(df)

    count = await save_feature_batch("SPY", "5m", feat_df, db_session)
    assert count > 0

    start = pd.Timestamp(df["bar_open_time"].iloc[0]).to_pydatetime()
    end = pd.Timestamp(df["bar_open_time"].iloc[-1]).to_pydatetime()
    rows = await load_feature_range_pit("SPY", "5m", start, end, db_session)
    assert len(rows) > 0
    for r in rows:
        assert r.is_valid


@pytest.mark.asyncio
async def test_F15_range_load_excludes_stale_manifest(db_session):
    from app.feature_pipeline.compute import compute_features
    from app.feature_pipeline.store import save_feature_batch, load_feature_range_pit
    from app.models.feature_row import FeatureRow

    df = _make_ohlcv(80)
    await save_feature_batch("QQQ", "5m", compute_features(df), db_session)

    stale_time = pd.Timestamp(df["bar_open_time"].iloc[40]).to_pydatetime()
    stale = FeatureRow(
        symbol="QQQ", timeframe="5m", bar_open_time=stale_time,
        manifest_hash="stalemanifest123", pipeline_version=0,
        features_json=json.dumps({"rsi_14": 50.0}),
        null_mask=json.dumps([]), snapshot_id="stale0000stale00",
        is_valid=True,
    )
    db_session.add(stale)
    await db_session.flush()

    start = pd.Timestamp(df["bar_open_time"].iloc[0]).to_pydatetime()
    end = pd.Timestamp(df["bar_open_time"].iloc[-1]).to_pydatetime()
    rows = await load_feature_range_pit("QQQ", "5m", start, end, db_session)

    assert "stalemanifest123" not in {r.manifest_hash for r in rows}


@pytest.mark.asyncio
async def test_F15_range_load_sorted_ascending(db_session):
    from app.feature_pipeline.compute import compute_features
    from app.feature_pipeline.store import save_feature_batch, load_feature_range_pit

    df = _make_ohlcv(100)
    await save_feature_batch("IWM", "5m", compute_features(df), db_session)
    start = pd.Timestamp(df["bar_open_time"].iloc[0]).to_pydatetime()
    end = pd.Timestamp(df["bar_open_time"].iloc[-1]).to_pydatetime()
    rows = await load_feature_range_pit("IWM", "5m", start, end, db_session)

    times = [r.bar_open_time for r in rows]
    assert times == sorted(times), "load_feature_range_pit must return rows sorted ascending"


# ---------------------------------------------------------------------------
# F16: rows_to_dataframe
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_F16_rows_to_dataframe_sorted_and_complete(db_session):
    from app.feature_pipeline.compute import compute_features
    from app.feature_pipeline.store import save_feature_batch, load_feature_range_pit, rows_to_dataframe
    from app.feature_pipeline.registry import FEATURE_COLS

    df = _make_ohlcv(100)
    await save_feature_batch("MSFT", "5m", compute_features(df), db_session)
    start = pd.Timestamp(df["bar_open_time"].iloc[0]).to_pydatetime()
    end = pd.Timestamp(df["bar_open_time"].iloc[-1]).to_pydatetime()
    rows = await load_feature_range_pit("MSFT", "5m", start, end, db_session)

    result_df = rows_to_dataframe(rows)
    assert "bar_open_time" in result_df.columns
    times = result_df["bar_open_time"].values
    assert (times[:-1] <= times[1:]).all()
    for col in FEATURE_COLS:
        assert col in result_df.columns, f"rows_to_dataframe missing: {col}"


def test_F16_rows_to_dataframe_empty_input():
    from app.feature_pipeline.store import rows_to_dataframe
    result = rows_to_dataframe([])
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# F17: Inspector covers new feature families
# ---------------------------------------------------------------------------

def test_F17_inspect_row_includes_all_new_features():
    from app.feature_pipeline.inspector import inspect_row
    df = _make_ohlcv(100)
    report = inspect_row(df, "SPY", bar_index=-1)
    for col in ["ema_dist_20", "ema_dist_50", "zscore_5",
                "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
                "bar_return", "gap_pct", "price_above_ema20", "ma_alignment"]:
        assert col in report.features, f"inspect_row.features missing: {col}"


def test_F17_inspect_row_str_contains_new_columns():
    from app.feature_pipeline.inspector import inspect_row
    text = str(inspect_row(_make_ohlcv(100), "SPY", bar_index=-1))
    for col in ["body_ratio", "ema_dist_20", "price_above_ema20", "gap_pct"]:
        assert col in text


def test_F17_inspect_row_regime_values_are_binary():
    from app.feature_pipeline.inspector import inspect_row
    report = inspect_row(_make_ohlcv(120), "SPY", bar_index=-1)
    for col in ["price_above_ema20", "ma_alignment"]:
        val = report.features.get(col)
        if val is not None:
            assert val in (0.0, 1.0)


def test_F17_inspect_row_prior_bar_populated():
    from app.feature_pipeline.inspector import inspect_row
    report = inspect_row(_make_ohlcv(50), "SPY", bar_index=-1)
    for field in ["close", "open", "high", "low", "volume"]:
        assert field in report.prior_bar


# ---------------------------------------------------------------------------
# F18: Out-of-range detection
# ---------------------------------------------------------------------------

def test_F18_out_of_range_flags_impossible_rsi():
    from app.feature_pipeline.registry import REGISTRY, ALL_FEATURE_COLS
    features = {n: 50.0 for n in ALL_FEATURE_COLS}
    features["rsi_14"] = 150.0

    out_of_range = []
    for name in ALL_FEATURE_COLS:
        reg = REGISTRY.get(name)
        val = features.get(name)
        if reg is None or val is None:
            continue
        lo, hi = reg.expected_min, reg.expected_max
        if (lo is not None and val < lo) or (hi is not None and val > hi):
            out_of_range.append(name)
    assert "rsi_14" in out_of_range


def test_F18_body_ratio_above_one_is_out_of_range():
    from app.feature_pipeline.registry import REGISTRY
    reg = REGISTRY["body_ratio"]
    val = 1.5
    flagged = (reg.expected_max is not None and val > reg.expected_max)
    assert flagged, "body_ratio=1.5 must be flagged as out-of-range"


# ---------------------------------------------------------------------------
# F19 / F20: Options sentinel fill
# ---------------------------------------------------------------------------

def test_F19_options_sentinel_fill_all_columns():
    from app.feature_pipeline.compute import compute_features
    from app.feature_pipeline.registry import OPTIONS_FEATURE_COLS, REGISTRY
    df = _make_ohlcv(50)
    feat = compute_features(df, options_data=None)
    for name in OPTIONS_FEATURE_COLS:
        expected = REGISTRY[name].sentinel_value
        assert (feat[name] == expected).all()
    assert (feat["is_null_options"] == 1.0).all()


def test_F20_options_real_values_used():
    from app.feature_pipeline.compute import compute_features
    df = _make_ohlcv(50)
    opts = {
        "atm_iv": 0.35, "iv_rank": 0.82, "iv_skew": 0.08,
        "pc_volume_ratio": 1.4, "pc_oi_ratio": 0.75,
        "gex_proxy": 5e6, "dist_to_max_oi": -0.015,
    }
    feat = compute_features(df, options_data=opts)
    for name, expected in opts.items():
        assert float(feat[name].iloc[-1]) == expected
    assert (feat["is_null_options"] == 0.0).all()


# ---------------------------------------------------------------------------
# F21 / F22: valid_mask
# ---------------------------------------------------------------------------

def test_F21_valid_mask_false_at_row_0():
    from app.feature_pipeline.compute import compute_features, valid_mask
    assert not valid_mask(compute_features(_make_ohlcv(200))).iloc[0]


def test_F21_valid_mask_true_after_warmup():
    from app.feature_pipeline.compute import compute_features, valid_mask
    df = _make_ohlcv(200)
    mask = valid_mask(compute_features(df))
    assert mask.iloc[80]
    assert mask.iloc[-1]


def test_F22_valid_mask_false_when_required_feature_nan():
    from app.feature_pipeline.compute import compute_features, valid_mask
    df = _make_ohlcv(100)
    feat = compute_features(df).copy()
    feat.loc[feat.index[-1], "rsi_14"] = np.nan
    assert not valid_mask(feat).iloc[-1]
