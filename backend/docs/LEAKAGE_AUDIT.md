# Leakage Audit

**Audit date:** 2026-04-12 (updated; original: 2026-04-11)
**Auditor:** Automated analysis + manual review
**Scope:** Full ML pipeline — ingestion → features → labels → training → inference → execution

---

## Methodology

Each issue below was found by inspecting:
- Every rolling/EWM computation for shift-by-1 compliance
- Bar-close detection logic for race conditions
- Train/test split boundaries for future data bleed
- Feature-dimension consistency between training and inference
- Label generation for contemporaneous or future data in threshold computation
- ffill propagation for session-boundary violations
- Options join timestamps for snapshot_time > bar_open_time violations

Each issue is accompanied by a regression test in `tests/test_leakage.py`.

---

## Issues Found and Fixed

### L1 — Training/Inference Feature Dimension Mismatch (CRITICAL)

| Field | Detail |
|-------|--------|
| **Severity** | CRITICAL |
| **Category** | Feature pipeline consistency |
| **Impacted files** | `app/services/backtest_service.py`, `app/inference/inference_service.py` |
| **Regression tests** | `test_L1_backtest_prepare_dataset_produces_new_pipeline_feature_count`, `test_L1_prepare_dataset_and_inference_service_use_same_feature_names` |

**Issue**

`backtest_service._prepare_dataset()` built features using the OLD pipeline (`app/services/feature_pipeline.py`) which produced a **14-feature** `FeatureSet` vector. `inference_service.run_inference()` uses the NEW pipeline (`app/feature_pipeline/features.py`) which produces a **22-feature** `FEATURE_COLS` vector.

`run_backtest()` ended by calling `set_models(dir_m_final, mag_m_final)` — storing a model trained on 14 features in the global model registry. When inference then called `model.predict_proba(X)` with a 22-feature X, the result was either an `sklearn` dimension-mismatch exception or silent wrong-class predictions (if the model coincidentally accepted the input).

**Why it matters**

Any backtest result was completely disconnected from what the live model would actually produce. The production model was trained on one feature set and evaluated on another.

**Fix applied**

Rewrote `_prepare_dataset()` to use `build_feature_matrix()` from the new pipeline. The loop-based O(n²) approach was replaced with a single O(n) call. Both training and inference now use `FEATURE_COLS` (22 features).

**Residual risk**

If `FEATURE_COLS` is ever changed, any previously serialised models (pickle files) will become dimension-incompatible. Models should be versioned alongside their feature manifest.

---

### L2 — Feature Lookback Shift-by-1 Invariant (HIGH)

| Field | Detail |
|-------|--------|
| **Severity** | HIGH |
| **Category** | Feature lookahead |
| **Impacted files** | `app/feature_pipeline/features.py` |
| **Regression tests** | `test_L2_features_at_row_i_do_not_use_close_i`, `test_L2_feature_row_i_is_nan_for_row_0` |

**Issue**

The new pipeline correctly applies `c1 = c.shift(1)` before all rolling and EWM operations, ensuring `feat[i]` uses only `close[0..i-1]`. However, this invariant was not enforced by tests that verified *stability under future-bar appends*: appending N future bars to the DataFrame must not change any historical feature row.

**Why it matters**

If any feature at row i were to depend on close[i] (the target bar), the model would be trained on leaked data. Any regime in which a bar's own close is strongly predictive of the next bar's direction (e.g., momentum) would inflate accuracy metrics substantially.

**Fix applied**

Added `test_L2_features_at_row_i_do_not_use_close_i`: appends 10 new bars to a base DataFrame and asserts that the features of all original rows remain bit-for-bit identical (max diff < 1e-9). This test would fail immediately if any future code removed the shift.

**Residual risk**

The shift invariant is only checked for the 22 FEATURE_COLS features. Any new feature added to the pipeline must be manually verified for shift compliance before being added to FEATURE_COLS.

---

### L3 — Bar-Close Guard Buffer Missing (HIGH)

| Field | Detail |
|-------|--------|
| **Severity** | HIGH |
| **Category** | Bar-close timing |
| **Impacted files** | `app/data_ingestion/ingestion_service.py` |
| **Regression tests** | `test_L3_bar_not_closed_within_buffer_window`, `test_L3_bar_closed_after_buffer_elapsed`, `test_L3_buffer_constant_is_positive` |

**Issue**

`_is_bar_closed()` used `bar_close_time <= datetime.utcnow()` with no guard buffer. A 5m bar closing at 09:35:00.000 would be marked as closed at 09:35:00.001, before the vendor API has propagated the complete OHLC data. Real-world latency between exchange dissemination and API availability is typically 50–500ms; at peak load it can reach several seconds.

**Why it matters**

A prediction triggered immediately after `bar_close_time` may use an OHLC bar where `high` or `close` is still the last-seen tick rather than the bar's true closing value. Features computed from an incomplete bar (especially ATR and Bollinger) are systematically biased toward mid-bar values.

**Fix applied**

Added `_BAR_CLOSE_BUFFER_SECONDS = 5`. The check is now `(bar_close_time + 5s) <= utcnow()`. The 5-second buffer is conservative for yfinance (typically 15-minute delayed) and appropriate for direct exchange feeds.

**Residual risk**

For very low-latency systems (sub-second bars), 5 seconds may be too conservative. The constant is exported and configurable. For intraday 5m bars with yfinance data, the buffer has no practical effect since all historical data is already fully settled.

---

### L4 — Ternary Label ATR Threshold Uses Current-Bar OHLC (MEDIUM)

| Field | Detail |
|-------|--------|
| **Severity** | MEDIUM |
| **Category** | Label generation |
| **Impacted files** | `app/feature_pipeline/labels.py` |
| **Regression tests** | `test_L4_ternary_label_threshold_shift`, `test_L4_ternary_label_no_lookahead_in_threshold_non_atr` |

**Issue**

In `ternary_label()`, the volatility threshold for row i was computed using `h = df["high"]` and `l = df["low"]` (unshifted). This means `ATR[i]` incorporated `high[i]` and `low[i]` — the intrabar range of the bar being labelled — in its EWM computation. The threshold at row i thus adapted to current-bar volatility, which the model cannot observe at decision time.

While this is a label-metadata issue (not a feature lookahead), it creates a subtle training-time inconsistency: labels are harder to classify when computed with a stale threshold than the one the model would see at inference time.

**Why it matters**

The ternary label's NO_TRADE zone is determined by `threshold_multiplier * ATR`. If ATR[i] is abnormally high due to a large intrabar move at bar i, the same return that would be classified as UP in a low-volatility regime gets classified as NO_TRADE. This makes the label distribution non-stationary in a way the model cannot account for.

**Fix applied**

Shifted `high`, `low`, and previous-close by 1 so `ATR[i]` uses only `high[0..i-1]` and `low[0..i-1]`. Threshold for row i is now based strictly on historical volatility.

**Residual risk**

The non-ATR path (realized-vol rolling std) was already using `close.shift(1)` inside `_realized_vol`, so it was partially clean. The test `test_L4_ternary_label_no_lookahead_in_threshold_non_atr` confirms the non-ATR path is also stable.

---

### L5 — Inference Accepts Unclosed Bars Without Explicit Guard (MEDIUM)

| Field | Detail |
|-------|--------|
| **Severity** | MEDIUM |
| **Category** | Inference pipeline |
| **Impacted files** | `app/inference/inference_service.py` |
| **Regression tests** | `test_L5_inference_rejects_unclosed_last_bar`, `test_L5_inference_proceeds_when_all_bars_closed`, `test_L5_inference_proceeds_when_no_is_closed_column` |

**Issue**

`run_inference()` documented "df must contain only closed bars" but enforced this only by convention. Any caller that accidentally passed a DataFrame with an unclosed last bar would proceed to build features and run the model against a bar whose `high`, `low`, and `close` were still updating (live data). Although `.shift(1)` prevents the current bar's close from entering its OWN feature row, the current bar's high/low does enter the ATR and Bollinger for the NEXT row (the inference row).

**Why it matters**

A prediction made at 09:32:15 on a 5m bar that started at 09:30:00 and hasn't closed yet will use an ATR inflated by a partial bar's range. This can suppress or exaggerate signals depending on intrabar volatility.

**Fix applied**

Added an explicit check at the top of `run_inference()`:
```python
if "is_closed" in df.columns and not bool(df["is_closed"].iloc[-1]):
    return _no_trade_result(symbol, "last_bar_not_closed", df)
```
The check is conditional on column presence so legacy callers without `is_closed` are not broken.

**Residual risk**

Callers that do not include `is_closed` in their DataFrames bypass this guard. The recommended solution is to enforce `is_closed` column presence in the `get_closed_bars()` query in `ingestion_service.py` — the column is always available from that path.

---

### L6 — Session-Boundary ffill Without Limit (LOW-MEDIUM)

| Field | Detail |
|-------|--------|
| **Severity** | LOW-MEDIUM |
| **Category** | Feature stationarity |
| **Impacted files** | `app/feature_pipeline/features.py` |
| **Regression tests** | `test_L6_ffill_does_not_propagate_across_long_gap` |

**Issue**

`c1.ffill()` without a `limit` parameter would propagate the last known close value across arbitrarily long gaps in the data (e.g., a 3-day weekend, a trading halt, or a data vendor outage). During such a gap, features like RSI and MACD would remain at their pre-gap values rather than becoming NaN, making them appear stable when in reality no price data existed.

For daily bars this can mean RSI computed on a Tuesday uses Monday's close silently held flat for Saturday and Sunday.

**Why it matters**

Features that are artificially stable across session gaps look cleaner to the model than they actually are. If the model learns that stable RSI patterns are predictive, it may over-fit to stale-fill artifacts.

**Fix applied**

Added `FFILL_LIMIT = 78` (one full 5-minute trading session = 78 bars) as a module-level constant. All ffill calls now use `ffill(limit=FFILL_LIMIT)`. Rows beyond the limit become NaN and are dropped from the training set via the `~np.isnan(X).any(axis=1)` filter in `_prepare_dataset`.

**Residual risk**

The limit of 78 is calibrated for 5-minute bars. For daily bars, 78 would be 78 trading days (~15 weeks), which is too permissive. If the pipeline is ever run on daily frequency, `FFILL_LIMIT` should be set per timeframe (e.g., 1 for daily bars to never fill across non-trading days).

---

### L7 — Options Snapshot Join: No Enforcement of available_at ≤ bar_open_time (LOW → FIXED)

| Field | Detail |
|-------|--------|
| **Severity** | LOW → **FIXED** (now code-enforced in `option_store.py`) |
| **Category** | Options data alignment |
| **Impacted files** | `app/data_ingestion/option_store.py`, `docs/DATA_LINEAGE.md` |
| **Regression tests** | `test_L7_options_snapshot_before_bar_is_valid`, `test_L7_options_snapshot_after_bar_is_lookahead`, `test_L7_most_recent_valid_snapshot_selection`, `test_L11_L7_status_updated_to_code_enforced`, `test_L11_options_snapshot_cutoff_is_enforced_in_function_signature` |

**Issue**

The `OptionSnapshot` model stored `snapshot_time` but there was no database-level or application-level constraint preventing a snapshot with `snapshot_time > bar_open_time` from being joined to that bar's feature row. Such a join would let the model see options chain state (IV, Greeks, OI) that was not available when the bar opened.

**Why it matters**

Options chain changes substantially during the 5 minutes of a price bar. An IV snapshot taken at bar close that is then joined to features computed "at bar open" would let the model see the IV *outcome* of the bar, not the IV available as an input. This is a direct form of lookahead for any options-derived feature.

**Fix applied**

The new `app/data_ingestion/option_store.py` implements `get_latest_chain_pit(session, symbol, as_of_utc, ...)` which enforces `available_at <= as_of_utc` in SQL via a `WHERE option_quotes.available_at <= :as_of_utc` clause. The `as_of_utc` parameter is required (no default), preventing callers from inadvertently bypassing the guard. This upgrades L7 from advisory to **code-enforced**. The existing L7 prospective tests remain in place; two new L11 tests verify the enforcement at the function-signature level.

**Residual risk**

The SQL enforcement exists in the read path (`get_latest_chain_pit`) but not as a database constraint. Future code that queries `option_quotes` directly (bypassing `get_latest_chain_pit`) could reintroduce the violation. When options features are wired into `build_feature_matrix()`, that call site must use `get_latest_chain_pit`, not a raw query.

---

### L8 — Labels Drop Last Row and Use Correct Shift (VERIFIED CLEAN)

| Field | Detail |
|-------|--------|
| **Severity** | N/A — verified clean at audit |
| **Regression tests** | `test_L8_build_labels_drops_last_row`, `test_L8_binary_label_does_not_shift_minus_2`, `test_L8_regression_label_last_row_is_nan` |

`build_labels()` correctly drops the last row. `binary_label()` uses `shift(-1)` (not `shift(-2)`), correctly targeting `close[i+1]` as the label for row i. Regression tests are in place to catch any accidental shift-offset change.

---

### L9 — _prepare_dataset Feature Stability Under Dataset Size Change (VERIFIED CLEAN)

| Field | Detail |
|-------|--------|
| **Severity** | N/A — verified clean at audit |
| **Regression tests** | `test_L9_prepare_dataset_feature_stability`, `test_L9_prepare_dataset_labels_are_consistent` |

After the L1 fix, `_prepare_dataset` uses a single pass through `build_feature_matrix`. Because all rolling operations are causal, features for rows 0..k produced from a 150-row dataset are bit-for-bit identical to those from a 100-row dataset. Labels are also verified to match direct computation of `close[i+1] > close[i]`.

---

### L10 — Backtest Walk-Forward Loop Missing Embargo Gap (HIGH)

| Field | Detail |
|-------|--------|
| **Severity** | HIGH |
| **Category** | Train/test boundary contamination |
| **Impacted files** | `app/services/backtest_service.py` |
| **Regression tests** | `test_L10_embargo_bars_constant_exists_and_is_positive`, `test_L10_walk_forward_test_set_starts_after_embargo`, `test_L10_boundary_bar_excluded_from_test` |

**Issue**

`run_backtest()` implemented a custom walk-forward loop that set `X_test = X[train_end:test_end]` — starting the test set at the row immediately following the last training row. `PurgedWalkForwardSplit` (used by `trainer.py`) enforces `embargo_bars=1` by default, but `run_backtest` did not use that splitter and had no equivalent embargo.

The contamination vector: `label[train_end - 1] = sign(close[train_end] - close[train_end-1])`. This label uses `close[train_end]`. Features at row `train_end` (the first row of `X_test`) are computed from bars `0..train_end-1` via `shift(1)`, so `X_test[0]` itself is clean. However, the **label** that trained the model at the boundary used `close[train_end]`, and the **next-step price** that defines the test-set's first prediction target is `close[train_end+1]`. For multi-horizon labels or any feature that uses the close at the fold boundary, this creates an implicit dependency on one shared bar.

**Why it matters**

Reported backtest accuracy metrics are computed over a test window that starts one bar too early. With momentum-heavy features, the first bar of each test fold is the one most likely to continue the training-window trend. Removing the boundary bar reduces the optimistic bias in fold metrics, especially at short lookback windows.

**Fix applied**

Added module-level constant `_EMBARGO_BARS = 1` and updated the test slice:

```python
X_test = X[train_end + _EMBARGO_BARS:test_end]
y_dir_test = y_dir[train_end + _EMBARGO_BARS:test_end]
y_mag_test = y_mag[train_end + _EMBARGO_BARS:test_end]
```

This matches the default `embargo_bars=1` in `PurgedWalkForwardSplit`.

**Residual risk**

With `test_size=40` and `embargo_bars=1`, each fold evaluates 39 bars instead of 40. This is a minor reduction in test coverage but correct behaviour. For very small `test_size` values (<5), the embargo could reduce the test set below the `len(X_test) < 5` guard; this is acceptable (the fold is skipped).

---

### L12 — Options Staleness Threshold at Inference Too Permissive (MEDIUM)

| Field | Detail |
|-------|--------|
| **Severity** | MEDIUM |
| **Category** | Stale data at inference time |
| **Impacted files** | `app/inference/inference_service.py` |
| **Regression tests** | `test_L12_max_chain_staleness_constant_is_module_level`, `test_L12_max_chain_staleness_is_at_most_one_bar`, `test_L12_stale_options_triggers_sentinel_fill` |

**Issue**

`_MAX_CHAIN_STALENESS` was set to `3600.0` seconds (1 hour) as a **local variable** inside `run_inference()`. A local variable cannot be tested, overridden via configuration, or audited by static analysis tools. More critically, 3600 seconds allows 12 complete 5-minute bars to pass before options data is considered stale — the options chain state from an hour ago may reflect a completely different market regime (pre/post news, pre/post Fed, etc.).

This finding was noted in the original audit's "not unit-testable" table with the recommendation to enforce `staleness_seconds < 300`. It was not fixed at that time.

**Why it matters**

If options IV or greeks are stale by more than one bar interval, the model receives options features that correspond to market conditions that no longer apply. For example, if a large move caused IV to spike 30 minutes ago and the model is now predicting on a calmer bar, it will see inflated IV as if it is current — artificially boosting the "high volatility" signal.

**Fix applied**

1. Moved `_MAX_CHAIN_STALENESS = 300.0` to module level (from local variable inside `run_inference`).
2. Reduced from `3600.0` to `300.0` (one 5-minute bar interval).
3. The local re-assignment inside `run_inference` was removed; it now uses the module-level constant.

**Residual risk**

300 seconds is the right threshold for 5-minute bar inference. If the options snapshot service goes down for more than one bar (e.g., rate-limit outage), inference will fall back to sentinel values for that period. This is correct and safe behaviour. The sentinel fill (`is_null_options=1`) is what the model was trained on for the no-options-data case.

---

## Issues Identified but Not Unit-Testable (Require Integration Tests)

| Issue | Why not unit-testable | Recommended mitigaton |
|-------|----------------------|----------------------|
| yfinance adjusted-close revision bias | Requires fetching historical data, storing a snapshot, then refetching after a real split/dividend event and comparing | Document in `DATA_LINEAGE.md`; switch to `auto_adjust=False` for raw prices; store the `Close` vs `Adj Close` column explicitly |
| Walk-forward purge buffer | No purge gap (h-1 bars) exists between train-end and test-start for multi-horizon labels | Add `purge=h-1` parameter to walk-forward loop when multi-horizon labels are implemented |
| Normalizer fit on full dataset | `StandardScaler` in the sklearn Pipeline is fit per fold (clean), but verifying this requires integration with the actual training loop | Currently clean; add `check_is_fitted` assertion in the fold loop |
| Options chain staleness at inference time | Requires a live options snapshot store with real timestamps | Enforce `staleness_seconds < 300` check in inference service when options features are activated |

---

## Summary Table

| ID | Issue | Severity | File(s) | Fix | Test(s) |
|----|-------|----------|---------|-----|---------|
| L1 | Train/inference feature dimension mismatch | CRITICAL | `backtest_service.py` | Rewrote `_prepare_dataset` with new pipeline | `test_L1_*` (2 tests) |
| L2 | Shift-by-1 invariant not regression-tested | HIGH | `features.py` | Added stability test; invariant was already code-correct | `test_L2_*` (2 tests) |
| L3 | Bar-close buffer missing | HIGH | `ingestion_service.py` | Added 5s guard buffer | `test_L3_*` (3 tests) |
| L4 | Ternary ATR threshold uses current-bar range | MEDIUM | `labels.py` | Shifted high/low/pc in ATR computation | `test_L4_*` (2 tests) |
| L5 | Inference accepts unclosed bars silently | MEDIUM | `inference_service.py` | Added is_closed guard with explicit no_trade reason | `test_L5_*` (3 tests) |
| L6 | ffill propagates across session gaps indefinitely | LOW-MEDIUM | `features.py` | Added `ffill(limit=FFILL_LIMIT)` cap | `test_L6_*` (1 test) |
| L7 | Options snapshot join — available_at cutoff **now code-enforced** | LOW → FIXED | `option_store.py` | `get_latest_chain_pit` enforces `available_at <= as_of_utc` in SQL | `test_L7_*` (3) + `test_L11_*` (2) |
| L8 | Labels verified clean | — | `labels.py` | No change; regression tests added | `test_L8_*` (3 tests) |
| L9 | Prepare-dataset stability verified clean | — | `backtest_service.py` | No change; stability tests added | `test_L9_*` (2 tests) |
| L10 | Backtest walk-forward loop missing embargo gap | HIGH | `backtest_service.py` | Added `_EMBARGO_BARS = 1`; test slice starts at `train_end + _EMBARGO_BARS` | `test_L10_*` (3 tests) |
| L11 | L7 upgrade: options cutoff now SQL-enforced | — | `option_store.py` | `get_latest_chain_pit` requires `as_of_utc` arg | `test_L11_*` (2 tests) |
| L12 | Options staleness threshold 3600s → 300s | MEDIUM | `inference_service.py` | Moved to module-level constant; reduced to 300s (one bar) | `test_L12_*` (3 tests) |
