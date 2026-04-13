# Feature Documentation

**Pipeline version:** 1
**Manifest hash:** computed at runtime from `app.feature_pipeline.registry.MANIFEST_HASH`
**Total core features (FEATURE_COLS):** 30
**Total options features (OPTIONS_FEATURE_COLS):** 7

All OHLCV-derived features apply `.shift(1)` before any rolling or EWM computation so that `feature[i]` uses only `close[0..i-1]`. Appending future bars to the DataFrame must not change historical feature values (max diff < 1e-9). See `docs/LEAKAGE_AUDIT.md §L2` for the formal regression test.

---

## How to Inspect One Feature Row End-to-End

```python
from app.feature_pipeline.inspector import inspect_row

# df: full closed-bar OHLCV history (needs ≥ 61 bars for full warmup)
report = inspect_row(df, symbol="SPY", bar_index=-1)

print(report)           # human-readable table with values, expected ranges, OK/WARN/NULL
report.to_dict()        # machine-readable dict for API responses
report.features         # {feature_name: float} for the target bar
report.null_features    # list of required features that were NaN
report.out_of_range     # list of features outside expected_min/max
report.prior_bar        # OHLCV summary of the bar whose data drove these features
```

---

## FEATURE_COLS — Core Model Inputs (30 features)

### Group: trend

| Name | Version | Formula | Units | Expected Range | Null Strategy |
|------|---------|---------|-------|---------------|---------------|
| `rsi_14` | 1 | `100 - 100/(1 + EWM_avg_gain(14) / EWM_avg_loss(14))` | dimensionless [0, 100] | [0, 100] | required |
| `rsi_5` | 1 | `100 - 100/(1 + EWM_avg_gain(5) / EWM_avg_loss(5))` | dimensionless [0, 100] | [0, 100] | required |
| `macd_line` | 1 | `EMA(close, 12) - EMA(close, 26)` | price-relative | unbounded | required |
| `macd_signal` | 1 | `EMA(macd_line, 9)` | price-relative | unbounded | required |
| `macd_hist` | 1 | `macd_line - macd_signal` | price-relative | unbounded | required |
| `bb_pct` | 1 | `(close - lower_band) / (upper_band - lower_band)` | dimensionless; 0=lower, 1=upper | [-0.5, 1.5] | required |

All trend features are computed on `close.shift(1).ffill(limit=78)` — the prior bar's close series.

### Group: volatility

| Name | Version | Formula | Units | Expected Range | Null Strategy |
|------|---------|---------|-------|---------------|---------------|
| `atr_norm` | 1 | `ATR(14) / close` | dimensionless pct | [0, 0.1] | required |
| `realized_vol_5` | 1 | `std(log_ret, 5) * sqrt(252 * 78)` | annualized vol | [0, 5] | required |
| `realized_vol_10` | 1 | `std(log_ret, 10) * sqrt(252 * 78)` | annualized vol | [0, 5] | required |
| `realized_vol_20` | 1 | `std(log_ret, 20) * sqrt(252 * 78)` | annualized vol | [0, 5] | required |
| `vol_regime` | 1 | `realized_vol_10 / (realized_vol_60 + eps)` | dimensionless ratio; >1 = elevated | [0, 5] | required |

`ATR(14)` uses `df.shift(1)` (high, low, and previous close are all one bar prior to the target bar). `vol_regime` uses a 60-bar realized vol as its denominator; this intermediate series is not stored as a model feature but is computed for the ratio.

### Group: momentum

| Name | Version | Formula | Units | Expected Range | Null Strategy |
|------|---------|---------|-------|---------------|---------------|
| `ret_1` | 1 | `close[i-1] / close[i-2] - 1` | pct return | [-0.1, 0.1] | required |
| `ret_5` | 1 | `close[i-1] / close[i-6] - 1` | pct return | [-0.2, 0.2] | required |
| `ret_10` | 1 | `close[i-1] / close[i-11] - 1` | pct return | [-0.3, 0.3] | required |
| `ret_20` | 1 | `close[i-1] / close[i-21] - 1` | pct return | [-0.4, 0.4] | required |
| `ret_60` | 1 | `close[i-1] / close[i-61] - 1` | pct return | [-0.6, 0.6] | required |
| `zscore_20` | 1 | `(close[i-1] - mean(close[i-21..i-1])) / std(close[i-21..i-1])` | standard deviations | [-4, 4] | required |

`ret_60` requires ≥ 62 bars of history after shift. `zscore_20` is a mean-reversion signal: large negative values indicate oversold conditions relative to the 20-bar window.

### Group: vwap

| Name | Version | Formula | Units | Expected Range | Null Strategy |
|------|---------|---------|-------|---------------|---------------|
| `vwap_distance` | 1 | `(close[i-1] - vwap[i-1]) / close[i-1]` | dimensionless pct; positive=above VWAP | [-0.05, 0.05] | required |
| `vwap_slope` | 1 | `(vwap[i-1] - vwap[i-6]) / \|vwap[i-6]\|` | dimensionless pct per 5 bars | [-0.02, 0.02] | required |

When `vwap` is not present in the input DataFrame, it is approximated with `close`. Both series use prior-bar values (`shift(1)`).

### Group: volume

| Name | Version | Formula | Units | Expected Range | Null Strategy |
|------|---------|---------|-------|---------------|---------------|
| `volume_ratio` | 1 | `volume[i-1] / mean(volume[i-21..i-1])` | dimensionless; >1=above-avg | [0, 10] | required |
| `volume_trend` | 1 | `mean(volume[i-6..i-1]) / mean(volume[i-21..i-1])` | dimensionless ratio | [0, 5] | required |
| `volume_zscore` | 1 | `(volume[i-1] - mean) / std` over 20 bars | standard deviations | [-3, 10] | required |

Volume features are clipped at their expected-max values to suppress outliers from data errors or halts.

### Group: seasonality

| Name | Version | Formula | Units | Expected Range | Null Strategy |
|------|---------|---------|-------|---------------|---------------|
| `hour_sin` | 1 | `sin(2π * (hour + minute/60) / 24)` | [-1, 1] | [-1, 1] | required |
| `hour_cos` | 1 | `cos(2π * (hour + minute/60) / 24)` | [-1, 1] | [-1, 1] | required |
| `minute_sin` | 1 | `sin(2π * minute / 60)` | [-1, 1] | [-1, 1] | required |
| `minute_cos` | 1 | `cos(2π * minute / 60)` | [-1, 1] | [-1, 1] | required |
| `session_progress` | 1 | `(bar_open_time - 09:30 ET) / 390 min`, clipped [0, 1] | [0, 1] | [0, 1] | required |
| `is_first_30min` | 1 | `1 if session_progress ∈ [0, 0.077]` | binary {0, 1} | [0, 1] | required |
| `is_last_30min` | 1 | `1 if session_progress ∈ [0.923, 1.0]` | binary {0, 1} | [0, 1] | required |

Seasonality features use `bar_open_time` of the target bar directly — no shift applied, because the time of day is a known input at prediction time, not a value derived from the price series.

### Group: missingness indicator

| Name | Version | Formula | Units | Expected Range | Null Strategy |
|------|---------|---------|-------|---------------|---------------|
| `is_null_options` | 1 | `1 if options_data is None, else 0` | binary {0, 1} | [0, 1] | required |

This is the last member of `FEATURE_COLS`. It preserves information about options data availability for the model — without it, rows with and without options data would have identical feature vectors in the optionless regime.

---

## OPTIONS_FEATURE_COLS — Options Chain Features (7 features)

These features use `null_strategy="optional_sentinel"`: when `options_data=None`, each is set to its sentinel value and `is_null_options=1`. When options data is provided, `is_null_options=0`.

| Name | Version | Formula | Units | Expected Range | Sentinel |
|------|---------|---------|-------|---------------|---------|
| `atm_iv` | 1 | Linear interpolation of IV at strikes bracketing spot | annualized implied vol | [0, 3] | 0.0 |
| `iv_rank` | 1 | `(atm_iv - iv_52w_low) / (iv_52w_high - iv_52w_low + eps)` | [0, 1]; 1=52w-high IV | [0, 1] | 0.0 |
| `iv_skew` | 1 | `IV(25-delta put) - IV(25-delta call)` | vol spread; positive=put premium | [-0.2, 0.5] | 0.0 |
| `pc_volume_ratio` | 1 | `sum(put_volume) / (sum(call_volume) + eps)` | ratio; >1=more puts | [0, 5] | 1.0 |
| `pc_oi_ratio` | 1 | `sum(put_OI) / (sum(call_OI) + eps)` | ratio | [0, 5] | 1.0 |
| `gex_proxy` | 1 | `sum(gamma_i × OI_i × 100 × spot)` | dollar gamma (proxy) | unbounded | 0.0 |
| `dist_to_max_oi` | 1 | `(spot - strike_max_OI) / spot` | pct; negative=max OI below spot | [-0.1, 0.1] | 0.0 |

**Important:** `pc_volume_ratio` and `pc_oi_ratio` use sentinel `1.0` (neutral ratio) rather than `0.0` to avoid the model misinterpreting absence of data as extreme put skew.

---

## Stored-Only Features (not in FEATURE_COLS)

| Name | Group | Notes |
|------|-------|-------|
| `atr_14` | volatility | Raw ATR in price units; stored in `ALL_FEATURE_COLS` for inspection but excluded from model input vector to avoid redundancy with `atr_norm` |

---

## Warmup Requirements

The minimum number of bars required before all `FEATURE_COLS` are non-NaN:

| Feature | Minimum bars (after shift) |
|---------|--------------------------|
| `rsi_14` | 15 |
| `bb_pct` / `volume_ratio` / `zscore_20` | 21 |
| `realized_vol_20` | 21 |
| `ret_20` | 22 |
| `ret_60` / `vol_regime` | 62 |

In practice, provide **≥ 80 bars** to ensure all features are populated for the last row. The `valid_mask()` function returns `True` for rows where all `FEATURE_COLS` are non-NaN.

---

## Forward-Fill Policy

`ffill(limit=78)` is applied to all OHLCV-derived shifted series before rolling computations. This allows up to one full 5-minute trading session (78 bars = 6.5 h × 12 bars/h) of gap-filling. Rows beyond the limit become NaN and are excluded from training by `valid_mask()`.

**For daily bars:** set `FFILL_LIMIT = 1` to prevent filling across non-trading days.

---

## Versioning

Each `FeatureDef` has an integer `version` field. When a formula changes:
1. Increment `version` in `registry.py`
2. `MANIFEST_HASH` recomputes automatically
3. All stored `FeatureRow` records with the old hash are treated as stale
4. Any pickled models trained on the old feature set are dimension-incompatible and must be retrained

Track model/feature-set co-versioning by storing the `manifest_hash` alongside serialized model files.
