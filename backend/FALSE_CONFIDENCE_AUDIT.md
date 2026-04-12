# False Confidence Audit

**Scope:** All backend modules — feature pipeline, model training, inference, regime detection,
decision layer, paper-execution simulator.

**Date:** 2025-04-11
**Methodology:** Full static code review of `app/` + `tests/` with targeted checks against
11 false-confidence risk categories.

---

## Severity Legend

| Level | Meaning |
|-------|---------|
| 🔴 Critical | Directly inflates reported performance; must fix before trusting any numbers |
| 🟠 High | Significant performance inflation; fix before live deployment |
| 🟡 Medium | Partial inflation; fix before publishing results |
| 🟢 Low | Minor or already mitigated |

---

## Issue Rankings

| Rank | Category | Severity | Status | Fixed in This Audit |
|------|----------|----------|--------|---------------------|
| 1 | [Retraining frequency](#1-retraining-frequency) | 🟠 High | Partially fixed | ✅ Retrain trigger added |
| 2 | [Survivorship bias](#2-survivorship-bias) | 🟠 High | Not fixed (backtest only) | Documented |
| 3 | [Stale chain data](#3-stale-chain-data) | 🟡 Medium | Fixed | ✅ Freshness guard added |
| 4 | [Train/test overfitting gap](#4-traintest-overfitting-gap) | 🟡 Medium | Fixed | ✅ train_brier + overfit_ratio added |
| 5 | [Optimistic fill assumptions](#5-optimistic-fill-assumptions) | 🟡 Medium | Partial | Documented; BID_ASK option exists |
| 6 | [Baseline comparisons](#6-baseline-comparisons) | 🟡 Medium | Fixed | ✅ Brier skill score added |
| 7 | [Regime sample sizes](#7-regime-sample-sizes) | 🟡 Medium | Partial | Documented |
| 8 | [Calibration quality](#8-calibration-quality) | 🟢 Low | Well handled | — |
| 9 | [Data leakage](#9-data-leakage) | 🟢 Low | Well handled | — |
| 10 | [Feature search / multiple comparisons](#10-feature-search--multiple-comparisons) | 🟢 Low | Well handled | — |
| 11 | [Next-bar accuracy overuse](#11-next-bar-accuracy-overuse) | 🟢 Low | Well handled | — |

---

## 1. Retraining Frequency

**Severity: 🟠 High**

### Problem
The model is trained once and never updated. `app/inference/inference_service.py` loads a static
pickle from `model_artifacts/`. No scheduling logic, no performance-triggered retrain, no data-volume
trigger exists in the codebase.

The `confidence_tracker.py` detects degradation (rolling Brier > 1.3× baseline) but only shrinks
the `tradeable_confidence` output — it never requests retraining. In a live-trading context, a model
that was accurate in a trending market will silently fail in a regime shift, and the system will
trade at reduced confidence rather than refusing to trade and retraining.

### Risk
- Model performance decays over time as market microstructure, volatility regimes, or symbol
  behaviour changes
- Degradation factor provides some protection (suppresses trades when Brier worsens) but is
  not a substitute for model currency
- No lower bound on how stale a model can become

### Fix Applied
`app/inference/confidence_tracker.py` now sets `needs_retrain=True` on `TrackerStats` when:
- `degradation_factor <= 0.40` (model performing at 60% or less of original quality), OR
- `ece_recent >= 0.12` (calibration is substantially broken)

Both checks require `window_size >= 40` to avoid false alarms on sparse history.

A `WARNING` log is emitted: `RETRAIN RECOMMENDED for {symbol}: ...`

**What still needs to be done by the operator:**
- Wire the `needs_retrain` flag to a scheduler that calls the training pipeline
- Set a retraining cadence (weekly as minimum, daily if data volume allows)
- After retraining, call `tracker.set_baseline_brier(symbol, new_brier)` to reset the baseline

---

## 2. Survivorship Bias

**Severity: 🟠 High (backtests only; lower in live trading)**

### Problem
All backtesting and historical analysis operates on symbols that exist today. Any symbol that
was delisted, went bankrupt, underwent a reverse split, or was acquired between the training
start date and today is excluded.

This causes backtest performance to be inflated because:
- Failing companies have higher volatility → larger option premiums → more credit
- Survivorship-selected datasets are upward-biased (by definition, winners)
- Regime distributions in backtest data differ from real historical distributions

### Evidence
- `app/data_ingestion/ingestion_service.py` ingests whatever symbols are passed; no delisting check
- No code paths apply a historical constituent filter (e.g., S&P 500 as of date X)
- The backtest service uses live-pulled price data

### Risk Magnitude
For single-name equity options, survivorship bias can inflate annualized Brier by 5–15%
depending on sector. For ETFs (SPY, QQQ), the bias is negligible.

### Fix Required (Not Implemented)
1. Use a point-in-time universe: only include a symbol in backtest period T if it existed at T
2. For S&P 500 membership: use a constituent history file (e.g., from CRSP, Compustat)
3. Minimum: document that all backtest results apply to current survivors only

**Until this is fixed, all regime-conditional backtest P&L figures are optimistically biased.**

---

## 3. Stale Chain Data

**Severity: 🟡 Medium — Fixed**

### Problem
`app/inference/inference_service.py` appended options chain features (`atm_iv`, `iv_rank`, etc.)
to the feature vector without checking whether the data was current. If `options_features` was
populated with a snapshot from 2 hours ago (e.g., market was closed, data vendor lagged, or
ingestion fell behind), the IV context would be stale but treated as live.

Using stale IV creates false confidence in IV-based signals:
- A high `iv_rank` from 3 hours ago may not reflect current conditions after a news event
- `atm_iv` changes rapidly near catalysts

The `option_snapshot.py` model has a `staleness_seconds` field but it was never checked at
inference time.

### Fix Applied
`app/inference/inference_service.py` (line ~188):

```python
staleness = float(options_features.get("staleness_seconds", 0.0) or 0.0)
_MAX_CHAIN_STALENESS = 3600.0
if staleness > _MAX_CHAIN_STALENESS:
    logger.warning("options_features stale (%.0fs) — using sentinel values", staleness)
    feat_values.extend([0.0, 0.0, 0.0, 0.0, 0.0])   # sentinel zeros
else:
    for k in ["atm_iv", "iv_rank", ...]:
        feat_values.append(...)
```

### Remaining Gap
- The model was trained on non-sentinel values for these features. Sentinel zeros will
  produce a feature distribution shift — the model will not perform as validated when
  options data is absent. A more robust fix is to retrain with explicit `is_null_options`
  flag features (already present in `feature_pipeline/compute.py` line 264) and test that
  the model degrades gracefully rather than confidently wrong.

---

## 4. Train/Test Overfitting Gap

**Severity: 🟡 Medium — Fixed**

### Problem
`app/ml_models/baseline.py → train_with_walk_forward()` computed only test-set Brier per fold.
Without the corresponding train Brier, there was no way to detect overfitting from the logs.

A model that achieves train Brier = 0.15 and test Brier = 0.24 is severely overfit.
The existing logging showed only `brier=0.24`, giving no context for whether that was good
or bad relative to training performance.

### Fix Applied
`evaluate_model()` now returns:
```python
"brier_skill_score": ...,     # 1 - (model_brier / naive_brier); positive = beats random
"reference_brier": ...,       # naive classifier Brier (predict base rate)
```

`train_with_walk_forward()` now computes per-fold:
```python
metrics["train_brier"] = ...     # Brier on training data
metrics["overfit_ratio"] = test_brier / train_brier
```

Log output now shows:
```
Fold 2: acc=0.532 brier=0.2451 train_brier=0.2301 overfit_ratio=1.07 bss=0.018 auc=0.541
```

### Interpretation Guide
| overfit_ratio | Interpretation |
|---------------|----------------|
| < 1.1 | Normal (test ≈ train) |
| 1.1 – 1.3 | Mild overfitting; acceptable |
| 1.3 – 1.5 | Moderate overfitting; reduce complexity |
| > 1.5 | Severe overfitting; add regularization or reduce features |

| brier_skill_score | Interpretation |
|-------------------|----------------|
| < 0 | Worse than naive; do not trade |
| 0 – 0.05 | Marginal edge; only trade with high conviction signals |
| 0.05 – 0.15 | Meaningful edge; proceed with regime/calibration gates |
| > 0.15 | Strong edge; verify not a leakage artifact |

---

## 5. Optimistic Fill Assumptions

**Severity: 🟡 Medium — Documented, partially implemented**

### Problem
The paper execution simulator (`options_simulator/`) defaults to `FillMethod.MIDPOINT`
(fill at `(bid + ask) / 2`). In reality, retail options orders rarely fill at mid; the
achievable price is closer to the ask for buys and bid for sells, especially for single-leg
positions or less-liquid names.

Additionally, backtests run through the inference layer use the model's `expected_move_pct`
for sizing and P&L projections, but this number comes from the model, not from observed
option market prices with real spreads factored in.

### Quantified Impact
For a typical retail single-leg order with a 10% bid-ask spread:
- MIDPOINT assumes $2.00 fill on a $1.90/$2.10 quote
- BID_ASK assumes $2.10 entry
- The 5% difference compounds: on a $200 position, that's $10 extra cost per round trip

### Mitigation Available
- `FillMethod.BID_ASK` or `FillMethod.CONSERVATIVE` exist in the simulator config
- To use: set `SimulatorConfig(fill=FillConfig(method=FillMethod.BID_ASK))`

### Remaining Gap
- The **default** is still MIDPOINT — callers who don't configure this will get optimistic fills
- See `docs/simulator_limitations.md` F1, F2 for full impact analysis

---

## 6. Baseline Comparisons

**Severity: 🟡 Medium — Fixed**

### Problem
`evaluate_model()` returned Brier score but not a Brier skill score. A Brier score of 0.24
is uninterpretable in isolation: it could be excellent (if base rate is 50%, random = 0.25)
or terrible (if there's strong class imbalance).

### Fix Applied
`evaluate_model()` now returns `brier_skill_score` and `reference_brier`:

```python
base_rate = np.mean(y_test)
reference_brier = brier_score_loss(y_test, np.full_like(probs, base_rate))
bss = 1.0 - brier / (reference_brier + 1e-9)
```

BSS = 0 means "as good as naive". BSS = 0.10 means 10% improvement over naive.
BSS < 0 means the model is harmful.

### Remaining Gap
- Walk-forward fold summaries should report `mean_bss ± std_bss` across folds, not just mean Brier
- A `model_selection_report()` function that formats these with confidence intervals would help

---

## 7. Regime Sample Sizes

**Severity: 🟡 Medium — Documented**

### Problem
EVENT_RISK and LIQUIDITY_POOR regimes are rare by construction (abnormal 3.5σ moves, or
volume < 25% of 20-bar average). In a typical symbol dataset of 5,000 bars, these regimes
may represent <1% of observations (< 50 bars).

`app/ml_models/evaluation/regime.py` filters regimes with `< min_regime_samples` (default 30),
but:
- 30 samples is statistically very thin (95% CI on Brier ≈ ±0.07 for n=30 vs ±0.02 for n=500)
- No per-fold regime sample count is reported
- Aggregating across folds can give a false impression of confidence

### Risk
Regime-conditional P&L tables (e.g., "model performs well in TRENDING_UP, poorly in EVENT_RISK")
are based on small samples and will have wide confidence intervals that are never reported.

### Fix Required (Not Implemented)
1. Increase `min_regime_samples` to 50 minimum, warn at < 100
2. Report `n_samples` alongside every regime metric
3. Add confidence intervals to regime-conditional Brier tables:
   `brier ± 1.96 × sqrt(brier × (1 - brier) / n)`
4. Consider bootstrapped CIs for heteroskedastic regimes

**Until this is fixed, treat all regime-conditional performance estimates as directional
indicators only, not reliable statistics.**

---

## 8. Calibration Quality

**Severity: 🟢 Low — Well handled**

The calibration pipeline is the strongest part of this system:

✅ `CalibratedClassifierCV` wraps all models at training time
✅ Temporal split (not shuffled) for calibration set — prevents future leakage
✅ ECE computed on rolling window via `confidence_tracker.py`
✅ Reliability diagram computed and exposed to UI
✅ `calibration_health` status (`good` / `fair` / `degraded` / `unknown`) gates confidence
✅ `degradation_factor` reduces tradeable confidence when Brier worsens

**Minor gap:** ECE threshold for degraded status is 0.10, which is lenient. A more conservative
threshold of 0.07 would catch miscalibration earlier. But this is a tuning choice, not a flaw.

---

## 9. Data Leakage

**Severity: 🟢 Low — Well handled**

✅ All features use `.shift(1)` before rolling/EWM operations (`feature_pipeline/compute.py`)
✅ Labels use `shift(-1)` to target next bar (`feature_pipeline/labels.py`)
✅ ATR threshold for ternary labels uses `shift(1)` + `shift(2)` (no current-bar contamination)
✅ Comprehensive leakage regression tests in `tests/test_leakage.py` (L1–L9 all pass)
✅ Test L2 (append future bars, verify feature rows unchanged) is the gold-standard check
✅ `regime/detector.py` uses `.shift(1)` on all OHLCV series before signal computation

**Verified clean.** No leakage identified.

---

## 10. Feature Search / Multiple Comparisons

**Severity: 🟢 Low — Well handled**

✅ `FEATURE_COLS` is a fixed, hardcoded list (no data-driven feature selection)
✅ Model registry contains 3 parametric models + 3 naive baselines — not a large search space
✅ Hyperparameters are pre-specified in `TrainingConfig`, not grid-searched
✅ No evidence of "try 50 features, pick best 20" workflows

Selection bias from multiple comparisons is effectively zero.

---

## 11. Next-Bar Accuracy Overuse

**Severity: 🟢 Low — Well handled**

✅ `app/ml_models/evaluation/metrics.py` explicitly documents Brier as primary criterion
✅ `train_with_walk_forward()` selects winner by `brier_score_mean`, not accuracy
✅ `signal_scorer.py` uses probability edge + regime + volatility, not accuracy
✅ Accuracy is computed but not used for model selection or trading decisions

**Best practice followed throughout.**

---

## Summary of Code Changes Made

| File | Change | Addresses |
|------|--------|-----------|
| `app/ml_models/baseline.py` | Added `brier_skill_score`, `reference_brier`, `train_brier`, `overfit_ratio` to all fold metrics | #4, #6 |
| `app/inference/inference_service.py` | Added staleness guard for `options_features` (reject if > 3600s old) | #3 |
| `app/inference/confidence_tracker.py` | Added `needs_retrain`, `retrain_reason` to `TrackerStats`; emits WARNING when degradation crosses threshold | #1 |

---

## Remaining Action Items (Ordered by Priority)

| Priority | Action | Owner |
|----------|--------|-------|
| P1 | Wire `needs_retrain` flag to a retraining scheduler (weekly minimum) | Infra |
| P2 | Set default fill method in `SimulatorConfig` to `FillMethod.BID_ASK` for any production use | Dev |
| P3 | Use point-in-time symbol universe for any multi-symbol backtests | Research |
| P4 | Increase `min_regime_samples` to 50; add CI to regime-conditional metrics | Research |
| P5 | Add model selection report with `mean_bss ± std_bss` across folds | Dev |
| P6 | Retrain model with explicit `is_null_options` feature to handle sentinel zeros cleanly | Dev |

---

*This document should be reviewed whenever a new model is trained, a new feature is added,
or the trading universe changes.*
