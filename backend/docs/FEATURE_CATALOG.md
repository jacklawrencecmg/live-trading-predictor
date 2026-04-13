# Feature Catalog

**Pipeline version:** 2
**Manifest hash:** computed at import time from sorted `(name, version)` pairs
**Scope:** All features registered in `app/feature_pipeline/registry.py`
**Total features:** 48 (40 in FEATURE_COLS model inputs + 7 options + 1 raw ATR)

---

## Design Invariants

| Invariant | Description |
|-----------|-------------|
| **Shift-by-1** | Every OHLCV-derived feature at row `i` uses only bars `0..i-1`. `df.shift(1)` is applied before all rolling/EWM operations. Appending future bars must not change any historical feature row (max diff < 1e-9). |
| **No Greek forward-fill** | None/NaN greek values are stored as NULL. The pipeline never propagates a value from a prior snapshot. |
| **FFILL_LIMIT = 78** | ffill propagation is capped at 78 bars (one full 5-min trading session). Values beyond the limit become NaN, preventing stale data from crossing session gaps. |
| **Null strategy** | `required` → row invalid if NaN, dropped from training. `optional_sentinel` → filled with sentinel value; `is_null_options` indicator set to 1. |
| **Manifest hash** | SHA-256 of sorted `(name, version)` pairs. Changes when any feature formula is updated, invalidating all cached `FeatureRow` records. |

---

## Feature Groups

### 1. Trend

Oscillators and band-based features measuring directional momentum via price relative to moving averages and volatility bands. All computed from `close.shift(1)`.

| Name | v | Formula | Units | Range | Null |
|------|---|---------|-------|-------|------|
| `rsi_14` | 1 | `100 - 100/(1 + EWM_gain(14)/EWM_loss(14))` applied to `close.shift(1)` | dimensionless | [0, 100] | required |
| `rsi_5` | 1 | Same as rsi_14 with window=5 — faster oscillator | dimensionless | [0, 100] | required |
| `macd_line` | 1 | `EMA(close,12) - EMA(close,26)`, normalised: divided by close | price-relative | unbounded | required |
| `macd_signal` | 1 | `EMA(macd_line, 9)` — signal smoothing | price-relative | unbounded | required |
| `macd_hist` | 1 | `macd_line - macd_signal` — momentum of momentum | price-relative | unbounded | required |
| `bb_pct` | 1 | `(close - lower_band) / (upper_band - lower_band)`, 20-bar/2σ bands | dimensionless | [-0.5, 1.5] | required |

**Notes:** `macd_line` and related features are price-normalised during training via StandardScaler. The absolute scale is not meaningful; the sign and cross-zero dynamics are.

---

### 2. Volatility

ATR-based and realized-vol features measuring recent price dispersion and volatility regime.

| Name | v | Formula | Units | Range | Null |
|------|---|---------|-------|-------|------|
| `atr_norm` | 1 | `ATR(14) / close` — ATR normalized to price level | dimensionless pct | [0, 0.1] | required |
| `atr_14` | 1 | Raw EWM true range (span=14). **Stored for inspection only; excluded from FEATURE_COLS.** | price units | [0, ∞) | required |
| `realized_vol_5` | 1 | `std(log_ret, 5) × √(252×78)` — annualized 5-bar vol | annualized vol | [0, 5] | required |
| `realized_vol_10` | 1 | `std(log_ret, 10) × √(252×78)` | annualized vol | [0, 5] | required |
| `realized_vol_20` | 1 | `std(log_ret, 20) × √(252×78)` | annualized vol | [0, 5] | required |
| `vol_regime` | 1 | `realized_vol_10 / (realized_vol_60 + ε)` — short/long vol ratio | dimensionless | [0, 5] | required |

**Notes:** `252 × 78` = bars per year for 5-minute intraday data (252 trading days × 78 bars/day). `realized_vol_60` is computed internally but not stored in FEATURE_COLS; it only contributes to `vol_regime`.

---

### 3. Momentum

Directional return features over multiple lookback windows, plus a slow mean-reversion z-score.

| Name | v | Formula | Units | Range | Null |
|------|---|---------|-------|-------|------|
| `ret_1` | 1 | `close[i-1] / close[i-2] - 1` | pct return | [-0.1, 0.1] | required |
| `ret_5` | 1 | `close[i-1] / close[i-6] - 1` | pct return | [-0.2, 0.2] | required |
| `ret_10` | 1 | `close[i-1] / close[i-11] - 1` | pct return | [-0.3, 0.3] | required |
| `ret_20` | 1 | `close[i-1] / close[i-21] - 1` | pct return | [-0.4, 0.4] | required |
| `ret_60` | 1 | `close[i-1] / close[i-61] - 1` — medium-term (≈5 h on 5-min bars) | pct return | [-0.6, 0.6] | required |
| `zscore_20` | 1 | `(close[i-1] - mean(close[i-21..i-1])) / std(close[i-21..i-1])` | std deviations | [-4, 4] | required |

**Notes:** All returns use `close.shift(1)` as the base series. `zscore_20` is a slow mean-reversion signal; for a faster version see `zscore_5` in the mean-reversion group.

---

### 4. Mean Reversion  *(added in v2)*

EMA-distance and fast z-score features measuring price overextension from trend anchors. Distinct from momentum (which measures raw direction/speed): these measure *deviation from the mean*.

| Name | v | Formula | Units | Range | Null |
|------|---|---------|-------|-------|------|
| `ema_dist_20` | 1 | `(close[i-1] - EMA(close,20)[i-1]) / (close[i-1] + ε)` | dimensionless pct | [-0.05, 0.05] | required |
| `ema_dist_50` | 1 | `(close[i-1] - EMA(close,50)[i-1]) / (close[i-1] + ε)` — slower anchor | dimensionless pct | [-0.1, 0.1] | required |
| `zscore_5` | 1 | `(close[i-1] - mean(close[i-6..i-1])) / (std(close[i-6..i-1]) + ε)` | std deviations | [-4, 4] | required |

**Notes:**
- `ema_dist_20` and `ema_dist_50` are positive above the EMA and negative below.
- In a sustained uptrend: `ema_dist_50 > ema_dist_20 > 0` because EMA50 lags further behind price.
- `zscore_5` reacts faster than `zscore_20` to short-term overextension; high-magnitude values signal potential near-term reversion.

---

### 5. VWAP-Relative

Features measuring price position and trend relative to the intraday volume-weighted average price.

| Name | v | Formula | Units | Range | Null |
|------|---|---------|-------|-------|------|
| `vwap_distance` | 1 | `(close[i-1] - vwap[i-1]) / (close[i-1] + ε)` | dimensionless pct | [-0.05, 0.05] | required |
| `vwap_slope` | 1 | `(vwap[i-1] - vwap[i-6]) / (\|vwap[i-6]\| + ε)` — 5-bar pct change of VWAP | dimensionless pct | [-0.02, 0.02] | required |

**Notes:** Falls back to `close` when `vwap` column is absent (e.g., some data providers omit it). `vwap_distance > 0` means price is above VWAP, a bullish intraday signal used by institutional desks.

---

### 6. Volume-Relative

Features measuring current volume activity relative to its recent distribution.

| Name | v | Formula | Units | Range | Null |
|------|---|---------|-------|-------|------|
| `volume_ratio` | 1 | `volume[i-1] / mean(volume[i-21..i-1])`, clipped to [0, 10] | dimensionless | [0, 10] | required |
| `volume_trend` | 1 | `mean(volume[i-6..i-1]) / mean(volume[i-21..i-1])`, clipped to [0, 5] — short/long volume trend | dimensionless | [0, 5] | required |
| `volume_zscore` | 1 | `(volume[i-1] - mean(volume[i-21..i-1])) / std(volume[i-21..i-1])` | std deviations | [-3, 10] | required |

**Notes:** `volume_ratio > 1` indicates above-average activity, often associated with larger moves. All features use `volume.shift(1)` as the base series.

---

### 7. Intraday Seasonality

Cyclic time encoding and session-position features derived from `bar_open_time`. These are **NOT shifted** — `bar_open_time` is a temporal property of the prediction target bar, not a value derived from past price data. No lookahead applies.

| Name | v | Formula | Units | Range | Null |
|------|---|---------|-------|-------|------|
| `hour_sin` | 1 | `sin(2π × hour_fraction / 24)` | dimensionless | [-1, 1] | required |
| `hour_cos` | 1 | `cos(2π × hour_fraction / 24)` | dimensionless | [-1, 1] | required |
| `minute_sin` | 1 | `sin(2π × minute / 60)` | dimensionless | [-1, 1] | required |
| `minute_cos` | 1 | `cos(2π × minute / 60)` | dimensionless | [-1, 1] | required |
| `session_progress` | 1 | `(time - 09:30 ET) / 390 min`, clipped to [0, 1] | dimensionless | [0, 1] | required |
| `is_first_30min` | 1 | `1` if `session_progress ∈ [0, 0.077]`, else `0` | binary {0,1} | [0, 1] | required |
| `is_last_30min` | 1 | `1` if `session_progress ∈ [0.923, 1.0]`, else `0` | binary {0,1} | [0, 1] | required |

**Notes:** Sin/cos encoding preserves the cyclic structure of time (23:59 is close to 00:00). The first and last 30 minutes of the NYSE session are high-activity windows with elevated vol and different liquidity dynamics.

---

### 8. Bar Structure  *(added in v2)*

Candlestick anatomy features derived from the OHLCV of bar `i-1`. These capture **intrabar microstructure** — how price moved within the prior bar — as a complement to close-to-close momentum returns.

All features reference bar `i-1` exclusively via `df.shift(1)`, satisfying the shift-by-1 invariant.

| Name | v | Formula | Units | Range | Null |
|------|---|---------|-------|-------|------|
| `body_ratio` | 1 | `\|close[i-1] - open[i-1]\| / (high[i-1] - low[i-1] + ε)`, clipped [0,1] | dimensionless | [0, 1] | required |
| `upper_wick_ratio` | 1 | `(high[i-1] - max(close[i-1], open[i-1])) / (high[i-1] - low[i-1] + ε)`, clipped [0,1] | dimensionless | [0, 1] | required |
| `lower_wick_ratio` | 1 | `(min(close[i-1], open[i-1]) - low[i-1]) / (high[i-1] - low[i-1] + ε)`, clipped [0,1] | dimensionless | [0, 1] | required |
| `bar_return` | 1 | `close[i-1] / (open[i-1] + ε) - 1` — intrabar open-to-close return | pct return | [-0.05, 0.05] | required |
| `gap_pct` | 1 | `open[i-1] / (close[i-2] + ε) - 1` — gap between prior open and pre-prior close | pct gap | [-0.03, 0.03] | required |

**Notes:**
- `body_ratio + upper_wick_ratio + lower_wick_ratio ≤ 1.0` always (clipping prevents floating-point excess).
- Interpretation: doji (`body_ratio ≈ 0`), shooting star (`upper_wick_ratio ≫ body_ratio`), hammer (`lower_wick_ratio ≫ body_ratio`), full-body engulfing (`body_ratio ≈ 1.0`).
- `bar_return` differs from `ret_1` (close-to-close): `bar_return` captures intrabar direction while `ret_1` captures bar-to-bar carry.
- `gap_pct` is near zero for continuous intraday 5-min bars but non-zero at session opens (overnight gap) or after trading halts.
- `gap_pct` is NaN at rows 0 and 1 (insufficient prior bars).

---

### 9. Regime  *(added in v2)*

Binary trend-direction indicators derived from EMA relationships. These complement `vol_regime` (volatility-based) by providing price-trend regime signals that the model can use to learn regime-conditional feature weights.

| Name | v | Formula | Units | Range | Null |
|------|---|---------|-------|-------|------|
| `price_above_ema20` | 1 | `1 if close[i-1] > EMA(close,20)[i-1] else 0` | binary {0,1} | [0, 1] | required |
| `ma_alignment` | 1 | `1 if EMA(close,20)[i-1] > EMA(close,50)[i-1] else 0` — bullish MA stack | binary {0,1} | [0, 1] | required |

**Notes:**
- `price_above_ema20 = 1` signals price is in short-term bullish territory relative to recent price memory.
- `ma_alignment = 1` signals the 20-bar EMA is above the 50-bar EMA — a classical "golden cross" condition at the intraday timescale.
- Both features are binary, so they act as interaction terms: the model can learn e.g. "RSI overbought while in uptrend is less bearish than RSI overbought while below EMA20".

---

### 10. Options Summary

Options chain summary features derived from the nearest available snapshot. These are **point-in-time correct**: only snapshots with `available_at ≤ bar.event_time` are used (see L7 in `LEAKAGE_AUDIT.md`).

All features in this group use `null_strategy = "optional_sentinel"`. When no options snapshot is available, they are filled with sentinel values and `is_null_options = 1`.

| Name | v | Formula | Sentinel | Units | Range | Null |
|------|---|---------|----------|-------|-------|------|
| `atm_iv` | 1 | Linear interpolation of IV at strikes bracketing spot price | 0.0 | annualized IV | [0, 3] | optional_sentinel |
| `iv_rank` | 1 | `(atm_iv - iv_52w_low) / (iv_52w_high - iv_52w_low + ε)` | 0.0 | dimensionless | [0, 1] | optional_sentinel |
| `iv_skew` | 1 | `IV(25Δ put) - IV(25Δ call)` — put demand premium | 0.0 | vol spread | [-0.2, 0.5] | optional_sentinel |
| `pc_volume_ratio` | 1 | `Σ put_vol / (Σ call_vol + ε)` across all strikes and expiries | 1.0 | dimensionless | [0, 5] | optional_sentinel |
| `pc_oi_ratio` | 1 | `Σ put_OI / (Σ call_OI + ε)` | 1.0 | dimensionless | [0, 5] | optional_sentinel |
| `gex_proxy` | 1 | `Σ(gamma_i × OI_i × 100 × spot)` for all contracts — net dealer gamma | 0.0 | dollar gamma | unbounded | optional_sentinel |
| `dist_to_max_oi` | 1 | `(spot - strike_max_OI) / spot` — distance to the max-OI strike | 0.0 | dimensionless pct | [-0.1, 0.1] | optional_sentinel |
| `is_null_options` | 1 | `1 if options_data is None else 0` — missingness indicator | n/a | binary {0,1} | [0, 1] | required |

**Notes:**
- Sentinels for ratios use 1.0 (neutral), not 0.0, to avoid signalling extreme put/call imbalance when data is absent.
- `gex_proxy` is an approximation: true GEX requires the full options chain with sign convention per dealer convention. This proxy assumes dealers are short the options (negative GEX = dealer buying pressure at lower strikes).
- `is_null_options = 1` is always present in FEATURE_COLS, allowing the model to distinguish genuine signal from sentinel fill.

---

## Column Lists

### FEATURE_COLS (40 features — model inputs)

```
# trend (6)
rsi_14, rsi_5, macd_line, macd_signal, macd_hist, bb_pct

# volatility (5)
atr_norm, realized_vol_5, realized_vol_10, realized_vol_20, vol_regime

# momentum (6)
ret_1, ret_5, ret_10, ret_20, ret_60, zscore_20

# mean reversion (3) — NEW v2
ema_dist_20, ema_dist_50, zscore_5

# vwap (2)
vwap_distance, vwap_slope

# volume (3)
volume_ratio, volume_trend, volume_zscore

# seasonality (7)
hour_sin, hour_cos, minute_sin, minute_cos,
session_progress, is_first_30min, is_last_30min

# bar structure (5) — NEW v2
body_ratio, upper_wick_ratio, lower_wick_ratio, bar_return, gap_pct

# regime (2) — NEW v2
price_above_ema20, ma_alignment

# options missingness indicator (1)
is_null_options
```

### OPTIONS_FEATURE_COLS (7 features — sentinel-filled when absent)

```
atm_iv, iv_rank, iv_skew, pc_volume_ratio, pc_oi_ratio, gex_proxy, dist_to_max_oi
```

### ALL_FEATURE_COLS (48 features — full storage set)

`FEATURE_COLS + OPTIONS_FEATURE_COLS + ["atr_14"]`

`atr_14` (raw ATR in price units) is stored for inspection but excluded from model input to avoid redundancy with `atr_norm`.

---

## Null Strategy Reference

| Strategy | Behaviour | When to use |
|----------|-----------|-------------|
| `required` | Row invalid when null; excluded from training via `valid_mask()` | Features where a null value indicates insufficient data for the model to make a valid prediction |
| `optional_sentinel` | Filled with `sentinel_value`; `is_null_options=1` signals the fill | Features that are optionally available; the model was trained to handle the sentinel pattern |

---

## Feature Versioning

Every `FeatureDef` has a `version` integer. The `MANIFEST_HASH` is SHA-256 of sorted `(name, version)` pairs. When any formula changes:

1. Increment the feature's `version` in `registry.py`
2. `MANIFEST_HASH` changes automatically
3. All cached `FeatureRow` records become stale (they will not be returned by `load_feature_row` / `load_feature_range_pit` with `require_current_manifest=True`)
4. If the change affects model inputs (`FEATURE_COLS`), also increment `PIPELINE_VERSION` and delete stale model artifacts from `model_artifacts/`

---

## Point-in-Time Correctness

Every feature at row `i` uses only data from bars `0..i-1`:

- **OHLCV features**: `df.shift(1)` applied before all rolling/EWM operations
- **VWAP features**: `vwap.shift(1)` used for distance; falls back to `close.shift(1)`
- **Volume features**: `volume.shift(1)` before all rolling operations
- **Options features**: only snapshots with `available_at ≤ bar.event_time` — enforced in `option_store.get_latest_chain_pit()`
- **Seasonality features**: `bar_open_time` of the prediction target bar — not shifted (it is a temporal property of the target, not a value derived from past prices)

The shift-by-1 invariant is regression-tested in `tests/test_leakage.py` (L2) and `tests/test_feature_store.py` (F2, F5, F7) for all feature families.

---

## Feature Store API

```python
from app.feature_pipeline.store import (
    save_feature_row,       # persist one bar's features
    save_feature_batch,     # persist a full compute_features() result
    load_feature_row,       # load single bar by (symbol, timeframe, bar_time)
    load_feature_range_pit, # load time range, manifest-checked, sorted
    rows_to_dataframe,      # convert List[FeatureRow] → pd.DataFrame
    deserialize_features,   # dict from FeatureRow.features_json
    to_feature_series,      # pd.Series from FeatureRow
)
```

All async functions require a SQLAlchemy `AsyncSession`. `load_feature_range_pit` enforces `require_current_manifest=True` and `valid_only=True` by default.

---

## Inspector API

```python
from app.feature_pipeline.inspector import inspect_row

report = inspect_row(df, symbol="SPY", bar_index=-1, options_data=None)
print(report)           # human-readable table with OK/WARN/NULL per feature
report.to_dict()        # machine-readable for API responses or logging
```

`FeatureInspection` fields: `features`, `null_features`, `out_of_range`, `is_valid`, `prior_bar`, `feature_meta`, `manifest_hash`, `pipeline_version`.
