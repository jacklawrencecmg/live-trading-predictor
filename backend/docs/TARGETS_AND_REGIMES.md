# Targets and Regimes

**Audit date**: 2026-04-12
**Pipeline version**: 2
**Horizons**: 1 bar · 3 bars · 5 bars
**Implementation**: `app/feature_pipeline/targets.py`, `app/feature_pipeline/regime_labels.py`
**Tests**: `tests/test_targets.py`

---

## Contents

1. [Design principles](#1-design-principles)
2. [Forecast horizons](#2-forecast-horizons)
3. [Target definitions](#3-target-definitions)
   - [Direction — P(up) / P(down) / P(flat)](#31-direction-pup--pdown--pflat)
   - [Expected return](#32-expected-return)
   - [Expected move magnitude](#33-expected-move-magnitude)
   - [Realized volatility forecast](#34-realized-volatility-forecast)
   - [Abstain / no-trade signal](#35-abstain--no-trade-signal)
4. [Regime labels](#4-regime-labels)
   - [Trending](#41-trending)
   - [Mean-reverting](#42-mean-reverting)
   - [Low volatility](#43-low-volatility)
   - [High volatility](#44-high-volatility)
   - [Liquidity-poor](#45-liquidity-poor)
   - [Abnormal movement](#46-abnormal-movement)
   - [Warmup](#47-warmup)
5. [Leakage analysis](#5-leakage-analysis)
   - [Feature–label overlap: none](#51-featurelabel-overlap-none)
   - [Threshold leakage: none](#52-threshold-leakage-none)
   - [Label–label overlap (training): requires larger embargo](#53-labellabel-overlap-training-requires-larger-embargo)
   - [Regime label leakage: none](#54-regime-label-leakage-none)
6. [When each target is useful](#6-when-each-target-is-useful)
7. [Embargo configuration](#7-embargo-configuration)
8. [Column reference](#8-column-reference)
9. [API reference](#9-api-reference)

---

## 1. Design Principles

**Point-in-time correctness**: Every target label for row i is a function of future bars
(i+1 through i+h). Every feature at row i is a function of past bars (0 through i-1).
There is zero overlap between the feature window and the label window at any row.

**ATR-adaptive thresholds**: The flat-zone boundary for direction classification is
`flat_threshold_k × ATR_norm(i) × sqrt(h)`, where ATR_norm(i) uses only bars 0..i-1.
The threshold adapts to market volatility without introducing lookahead.

**sqrt(h) scaling**: Threshold grows as sqrt(h) to account for the natural widening of
the price distribution over longer horizons. Under a random walk, variance grows linearly
with horizon, so the 1-sigma range grows as sqrt(h). Without this scaling, compound
returns would classify nearly all multi-bar targets as UP or DOWN, destroying the flat
zone's signal at longer horizons.

**Multi-label regimes**: Regime flags are independent booleans — a bar can be
simultaneously TRENDING and HIGH_VOL. This is intentional: it allows precise conditioning
during evaluation (e.g. "performance when trending and high-vol") and avoids the
information loss of a single mutually-exclusive enum.

---

## 2. Forecast Horizons

| Horizon | Bars | Wall time (5-min bars) | Typical use case |
|---------|------|------------------------|------------------|
| h=1     | 1    | 5 minutes              | Scalp / directional signal |
| h=3     | 3    | 15 minutes             | Intraday setup with breathing room |
| h=5     | 5    | 25 minutes             | Short-term swing, ATM option delta |

The default tuple is `HORIZONS = (1, 3, 5)`. Custom horizons are supported by passing
`horizons=(...)` to `compute_targets()`.

---

## 3. Target Definitions

### 3.1 Direction — P(up) / P(down) / P(flat)

**Column**: `y_dir_h{h}`
**Type**: float encoding of `DirectionLabel` enum
**Values**: 0.0 = DOWN, 1.0 = FLAT, 2.0 = UP
**NaN**: last h rows

**Definition**:

```
future_return[i] = log(close[i+h] / close[i])

threshold[i] = FLAT_THRESHOLD_K × atr_norm[i] × sqrt(h)
    where atr_norm[i] = ATR(bars 0..i-1) / close[i-1]

y_dir_h{h}[i] =
    2 (UP)   if future_return[i] >  threshold[i]
    0 (DOWN) if future_return[i] < -threshold[i]
    1 (FLAT) otherwise
```

**Default threshold_k**: 0.5.  A k=0.5 means the flat zone is ±0.5 ATR in log-return
terms. In typical equity intraday markets (ATR/close ≈ 0.3%), this corresponds to a
flat zone of ±0.15% per bar.

**Flat rate by horizon** (typical on SPY 5-min data):

| h | Flat zone half-width | Approx flat rate |
|---|----------------------|------------------|
| 1 | 0.5 × ATR × 1.0     | ~35–45%          |
| 3 | 0.5 × ATR × 1.73    | ~50–60%          |
| 5 | 0.5 × ATR × 2.24    | ~60–70%          |

**Why this is useful**: Options strategies are path-dependent. A delta trade on a 1-bar
flat move is neutral but an options position still decays. The flat class enables the
system to abstain from directional trades in noisy conditions. P(flat) is a direct
no-trade signal.

---

### 3.2 Expected Return

**Column**: `y_ret_h{h}`
**Type**: float, signed
**Range**: (-∞, +∞) in practice (−0.1 to +0.1 for most intraday bars)
**NaN**: last h rows

**Definition**:

```
y_ret_h{h}[i] = log(close[i+h] / close[i])
```

**Why log return**: Log returns are additive across time and symmetric around zero for a
random walk. `y_ret_h3 = y_ret_h1[i] + y_ret_h1[i+1] + y_ret_h1[i+2]` approximately
(exact additivity holds for log returns). This decomposability makes multi-horizon
consistency checking straightforward.

**Why this is useful**: For position sizing and strike selection. If the model's
expected return is 0.3% with ATM options implied at 0.5%, the option is likely
overpriced. Combined with the direction signal, this feeds the IV comparison logic.

**Leakage risk**: None from features. See [§5.1](#51-featurelabel-overlap-none).
There is label-label overlap in training; see [§5.3](#53-labellabel-overlap-training-requires-larger-embargo).

---

### 3.3 Expected Move Magnitude

**Column**: `y_mag_h{h}`
**Type**: float ≥ 0
**Range**: [0, ∞)
**NaN**: last h rows

**Definition**:

```
y_mag_h{h}[i] = |y_ret_h{h}[i]| = |log(close[i+h] / close[i])|
```

This is always non-negative. It is identical to `y_ret_h{h}` for UP bars and the
negation for DOWN bars.

**Why this is useful**: Expected magnitude drives strategy selection regardless of
direction. A model that forecasts a large move but cannot determine direction still has
edge: buy a straddle, or sell a straddle if the forecast is for a small move. Comparing
`y_mag_h1` against ATM implied volatility is the core signal for IV-relative strategies.

**Relationship to rvol**: `y_mag_h1 == y_rvol_h1` exactly (single-bar RMS = absolute
return). For h > 1 they diverge: `y_mag_h3` is the magnitude of the compound 3-bar
return; `y_rvol_h3` is the average per-bar volatility over those 3 bars.

---

### 3.4 Realized Volatility Forecast

**Column**: `y_rvol_h{h}`
**Type**: float ≥ 0
**Range**: [0, ∞), typically 0.001–0.010 per 5-min bar
**NaN**: last h rows

**Definition**:

```
r[j] = log(close[j] / close[j-1])      # per-bar log return

y_rvol_h{h}[i] = sqrt( mean( r[i+1]^2, r[i+2]^2, ..., r[i+h]^2 ) )
```

This is the root-mean-squared per-bar return over the next h bars, also known as
realized volatility in the RMS sense (mean-zero assumption, which is accurate intraday).

For h=1: `y_rvol_h1[i] = |r[i+1]| = y_mag_h1[i]`.

**Annualization**: To convert to annualized volatility (matching options convention):

```
rvol_annualized = y_rvol_h{h} × sqrt(252 × 78)   # 5-min bars, 252 trading days
```

**Why this is useful**:
- Direct comparison to implied volatility (IV): if model forecasts rvol > IV, options
  are cheap; if rvol < IV, options are expensive.
- Strike width selection: a wider realized vol forecast justifies wider spread strikes.
- Portfolio risk estimation: aggregate realized vol forecasts across open positions.

**Label overlap for rvol**: `y_rvol_h5[i]` and `y_rvol_h5[i+1]` share the same
per-bar returns at bars i+2 through i+5 (4 overlapping bars). This is the most
stringent overlap case. See [§5.3](#53-labellabel-overlap-training-requires-larger-embargo).

---

### 3.5 Abstain / No-Trade Signal

**Column**: `y_abstain_h{h}`
**Type**: float {0.0, 1.0}
**NaN**: last h rows

**Definition**:

```
y_abstain_h{h}[i] = 1.0  if y_dir_h{h}[i] == FLAT  (1.0)
                    0.0  otherwise
```

This is a derived binary target — it is deterministically derived from the direction
label. It is provided as a separate column to enable:

1. Training a dedicated "abstain classifier" that predicts whether the market will
   produce a meaningful move in the next h bars.
2. Evaluation of abstain coverage: what fraction of bars the system should skip.
3. Combining with the 4-layer confidence hierarchy: even a high-confidence directional
   prediction should be filtered if the abstain rate in the current regime is high.

**Why this is useful**: In options markets, trading cost per bar is high (bid-ask spread,
delta hedging friction). Skipping flat bars dramatically improves net edge. A well-trained
abstain classifier that achieves 60% flat detection at 30% false-negative rate is worth
more than a 3% improvement in directional accuracy.

---

## 4. Regime Labels

All regime labels are computed in `compute_regime_labels()`. They are multi-label
(independent booleans) and all use `shift(1)` on OHLCV data — point-in-time safe.

---

### 4.1 Trending

**Column**: `regime_trending`
**Condition**: `adx_proxy > 25`

```
dm_up[i]   = max(high[i-1] - high[i-2], 0)
dm_down[i] = max(low[i-2] - low[i-1], 0)
DX[i]      = |EWM(dm_up) - EWM(dm_down)| / (EWM(dm_up) + EWM(dm_down)) × 100
adx_proxy  = EWM(DX, com=13)    # smoothed directional index
```

**Leakage risk**: None. All OHLCV inputs shifted by 1.
**When useful**: Models often perform better in trending regimes (momentum features are
predictive). Separate evaluation reveals if the model's Brier skill score is regime-dependent.

---

### 4.2 Mean-Reverting

**Column**: `regime_mean_reverting`
**Condition**: `adx_proxy <= 25` (strict complement of regime_trending)

**Note**: This label is NOT the same as "mean-reverting in the statistical sense."
It simply marks bars where directional momentum is weak. The market may still trend
within this category — just without persistent directional strength.

**When useful**: In mean-reverting regimes, mean-reversion features (`zscore_5`,
`ema_dist_20`) tend to have more predictive power than momentum features (`ret_5`, `ret_20`).
A meta-model can switch feature importance weights by regime.

---

### 4.3 Low Volatility

**Column**: `regime_low_vol`
**Condition**: `atr_ratio < 0.50`

```
atr_ratio = atr_short / atr_long
    atr_short = EWM(True Range, com=13)   # ~14-bar
    atr_long  = EWM(True Range, com=49)   # ~50-bar
```

**Leakage risk**: None. ATR computed from shifted series.
**When useful**: Low vol compresses the flat zone — `y_dir_h1 == FLAT` rate is high.
Options are likely cheap (low IV, low rvol). Strategy: avoid directional trades; consider
selling premium. Confidence threshold should be raised (see `REGIME_THRESHOLDS` in
`app/regime/detector.py`).

---

### 4.4 High Volatility

**Column**: `regime_high_vol`
**Condition**: `atr_ratio > 1.50`

**Leakage risk**: None.
**When useful**: High vol expands bid-ask spreads, options execution costs, and
slippage. The inference-time detector suppresses trades in this regime (hard block).
For labeling purposes, high-vol periods should be quarantined from training or treated
as a separate distribution — models trained in normal vol often degrade in high vol.

---

### 4.5 Liquidity-Poor

**Column**: `regime_liquidity_poor`
**Condition**: `volume_ratio < 0.30`

```
volume_ratio = volume[i-1] / rolling_mean(volume[i-20..i-1], min_periods=5)
```

**Leakage risk**: None.
**When useful**: Thin markets widen spreads and make fills uncertain. Include this flag
as a training filter (exclude liquidity-poor bars from training data) and as an inference
gate. For options, thin markets mean large IV bid-ask spreads — any option price estimate
has high noise.

---

### 4.6 Abnormal Movement

**Column**: `regime_abnormal`
**Condition**: `|pct_change[i-1]| > ABNORMAL_SIGMA × rolling_std` AND `|pct_change[i-1]| > 0.5%`

```
pct_change[i-1] = (close[i-1] - close[i-2]) / close[i-2]
rolling_std     = std(pct_change, window=20, min_periods=10)
ABNORMAL_SIGMA  = 3.0
```

The dual guard (sigma ratio AND minimum absolute return) prevents false positives on
synthetic constant or near-constant series where rolling_std → 0 makes any tiny move
appear extreme.

**Leakage risk**: None. Both inputs shifted.
**When useful**: Abnormal moves proxy for news events, earnings, macro announcements.
The model was not trained for this distribution. Mark these bars as out-of-distribution
and abstain from inference. Distinct from high vol: high vol can be sustained structural
elevation; abnormal is a single-bar jump.

---

### 4.7 Warmup

**Column**: `regime_warmup`
**Condition**: first `WARMUP_BARS = 20` rows

All other regime flags are False during warmup. EWM-based signals (ATR, ADX) require
at least 20 bars to stabilize. During warmup, regime is unknown and the system should
abstain from all trading decisions.

---

## 5. Leakage Analysis

### 5.1 Feature–label overlap: none

```
Feature[i]  uses  bars 0..i-1  (enforced by shift(1) in compute.py)
Label[i]    uses  bars i+1..i+h
```

There is zero bar overlap between the feature window and the label window at any row.
This property holds for all target types (direction, return, magnitude, rvol, abstain)
and all horizons.

**Test**: `test_T6_appending_future_bars_does_not_change_past_targets` — appending 10
future bars does not change any previously-computable target.

---

### 5.2 Threshold leakage: none

The flat-zone threshold `flat_threshold_k × atr_norm[i] × sqrt(h)` depends on:
- `atr_norm[i]` = ATR(bars 0..i-1) / close[i-1] — past only (verified by `test_T6_atr_threshold_is_past_only`)
- `sqrt(h)` — a constant

The threshold is entirely computable from past data. It does not adapt to realized future
volatility or any future information.

---

### 5.3 Label–label overlap (training): requires larger embargo

For horizon h, the label windows for consecutive bars overlap:

```
y_rvol_h5[i]   uses  r[i+1], r[i+2], r[i+3], r[i+4], r[i+5]
y_rvol_h5[i+1] uses  r[i+2], r[i+3], r[i+4], r[i+5], r[i+6]
shared bars:         r[i+2], r[i+3], r[i+4], r[i+5]   (h-1 = 4 bars)
```

If row i is the last training sample and row i+1 is the first test sample, bars i+2..i+5
contribute to both a training label (y_rvol_h5[i]) and a test label (y_rvol_h5[i+1]).
This is a form of label leakage — the training set contains information about returns that
also appear in the test labels.

**Required embargo**:

| Horizon h | Label overlap | embargo_bars required |
|-----------|---------------|-----------------------|
| h = 1     | 0 bars        | 1 (current default)   |
| h = 3     | 2 bars        | **3**                 |
| h = 5     | 4 bars        | **5**                 |

When training on multi-horizon targets, create a `PurgedWalkForwardSplit` with
`embargo_bars=h`:

```python
from app.ml_models.training.splitter import PurgedWalkForwardSplit

# For h=5 targets
splitter = PurgedWalkForwardSplit(n_splits=5, embargo_bars=5)
```

The current `TrainingConfig.embargo_bars = 1` default is only correct for h=1.
Update `TrainingConfig` when using longer horizons.

**Test**: `test_T11_embargo_requirement_is_horizon` verifies the overlap count.

---

### 5.4 Regime label leakage: none

All regime signals use `df.shift(1)` on all OHLCV columns before any computation.
`regime_label[i]` reflects only bars 0..i-1.

**Test**: `test_T10_appending_future_bars_does_not_change_regime_labels` — appending
future bars does not change any prior regime label.

---

## 6. When Each Target Is Useful

| Target | Best for | Avoid when |
|--------|----------|------------|
| `y_dir_h1` | Fast directional trades, binary classifier baseline | Overnight gaps; binary ignores magnitude |
| `y_dir_h3/h5` | Swing setups, options delta alignment | High-vol regimes (large flat zone) |
| `y_ret_h{h}` | Position sizing, strike selection | Training alone (high noise, need magnitude to be useful) |
| `y_mag_h1` | IV comparison, straddle entry signal | Low-vol regimes (tiny moves hard to forecast) |
| `y_mag_h3/h5` | Spread width selection, expected P&L | Abnormal regimes (model out-of-distribution) |
| `y_rvol_h1` | Same as `y_mag_h1` — identical by definition | — |
| `y_rvol_h3/h5` | Realized-vol vs implied-vol comparison, IV richness | Liquidity-poor (realized vol estimate noisy) |
| `y_abstain_h1` | No-trade classifier for scalping | — |
| `y_abstain_h3/h5` | Pre-screening for multi-bar option trades | — |

**Combining targets for an options trade decision**:

```
Trade if all of:
  y_dir_h3    == UP or DOWN        (direction signal present)
  y_abstain_h3 == 0                (not predicted flat)
  y_rvol_h3   > option_implied_rvol  (realized vol forecast exceeds implied)
  regime_high_vol == False          (not in high-vol regime)
  regime_liquidity_poor == False     (liquid market)
```

---

## 7. Embargo Configuration

When using multi-horizon targets in the walk-forward backtest or trainer, update
`embargo_bars` to match the longest horizon in your label set:

```python
from app.ml_models.training.config import TrainingConfig

# Single h=5 horizon
cfg = TrainingConfig(embargo_bars=5)

# Mixed horizons: use the maximum
cfg = TrainingConfig(embargo_bars=max(HORIZONS))   # 5
```

The `PurgedWalkForwardSplit` docstring documents this requirement:
> "For k-bar lookahead use k."

The default `embargo_bars=1` in both `TrainingConfig` and `PurgedWalkForwardSplit`
is intentionally correct only for the current h=1 binary label.

---

## 8. Column Reference

### Target columns (produced by `compute_targets()`)

| Column | Type | Description | NaN rows |
|--------|------|-------------|----------|
| `y_dir_h1` | float | Direction class at h=1: 0=DOWN 1=FLAT 2=UP | last 1 |
| `y_ret_h1` | float | Signed log-return over 1 bar | last 1 |
| `y_mag_h1` | float | Unsigned magnitude over 1 bar | last 1 |
| `y_rvol_h1` | float | RMS per-bar rvol over 1 bar | last 1 |
| `y_abstain_h1` | float | 1 iff FLAT at h=1 | last 1 |
| `y_dir_h3` | float | Direction class at h=3 | last 3 |
| `y_ret_h3` | float | Signed log-return over 3 bars | last 3 |
| `y_mag_h3` | float | Unsigned magnitude over 3 bars | last 3 |
| `y_rvol_h3` | float | RMS per-bar rvol over 3 bars | last 3 |
| `y_abstain_h3` | float | 1 iff FLAT at h=3 | last 3 |
| `y_dir_h5` | float | Direction class at h=5 | last 5 |
| `y_ret_h5` | float | Signed log-return over 5 bars | last 5 |
| `y_mag_h5` | float | Unsigned magnitude over 5 bars | last 5 |
| `y_rvol_h5` | float | RMS per-bar rvol over 5 bars | last 5 |
| `y_abstain_h5` | float | 1 iff FLAT at h=5 | last 5 |

### Regime label columns (produced by `compute_regime_labels()`)

| Column | Type | Condition | Hard block? |
|--------|------|-----------|-------------|
| `regime_trending` | bool | adx_proxy > 25 | No |
| `regime_mean_reverting` | bool | adx_proxy ≤ 25 | No |
| `regime_low_vol` | bool | atr_ratio < 0.50 | No (raises threshold) |
| `regime_high_vol` | bool | atr_ratio > 1.50 | **Yes** (per detector.py) |
| `regime_liquidity_poor` | bool | volume_ratio < 0.30 | **Yes** (per detector.py) |
| `regime_abnormal` | bool | \|ret\| > 3σ AND > 0.5% | **Yes** (per detector.py) |
| `regime_warmup` | bool | first 20 rows | **Yes** (per detector.py) |

Note: hard-block semantics are enforced by `app/regime/detector.py` at inference time.
The regime label flags here are for labeling and evaluation only.

---

## 9. API Reference

### `compute_targets(df, horizons=(1,3,5), flat_threshold_k=0.5)`

```python
from app.feature_pipeline.targets import compute_targets, HORIZONS

df = ...    # OHLCV DataFrame with bar_open_time column
targets = compute_targets(df)              # default horizons (1, 3, 5)
targets = compute_targets(df, horizons=(1,))  # single horizon
targets = compute_targets(df, flat_threshold_k=0.3)  # tighter flat zone
```

Returns a DataFrame with the same index as `df` and columns listed in §8.

### `target_col_names(horizons=(1,3,5))`

```python
from app.feature_pipeline.targets import target_col_names

cols = target_col_names()          # all 15 columns for default horizons
cols = target_col_names((1,))      # 5 columns for h=1 only
```

### `compute_regime_labels(df)`

```python
from app.feature_pipeline.regime_labels import compute_regime_labels, REGIME_LABEL_COLS

regime = compute_regime_labels(df)    # DataFrame of bool flags
print(regime["regime_trending"].value_counts())
```

### Combined usage

```python
from app.feature_pipeline.targets import compute_targets
from app.feature_pipeline.regime_labels import compute_regime_labels
from app.feature_pipeline.compute import compute_features

features = compute_features(df)
targets  = compute_targets(df)
regimes  = compute_regime_labels(df)

# Join for training matrix (all share df.index)
train = features.join(targets).join(regimes)
```
