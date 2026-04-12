# Risk Register — Options Research Platform

**As of:** April 2026
**Scope:** All quantitative, statistical, engineering, and operational risks that could cause
the platform to produce misleading outputs, incorrect P&L accounting, or false confidence
in signals. This register does not cover live trading risk (the system is paper-only).

**Severity scale:** Critical → High → Medium → Low
**Likelihood scale:** Certain · Likely · Possible · Unlikely

---

## Summary Table

| # | Risk | Category | Severity | Likelihood | Residual |
|---|------|----------|----------|------------|---------|
| R01 | 15-minute delayed data makes 5m signals unactionable | Data | Critical | Certain | Certain until live feed added |
| R02 | Model trained once; no scheduled retraining | ML | Critical | Likely | High — degradation flag exists but unwired |
| R03 | Single global model across all symbols | ML | High | Certain | High — no per-symbol adaptation |
| R04 | Flat IV surface; no skew or term structure | Options | High | Certain | High — structural limitation |
| R05 | Survivorship bias in all backtests | Statistical | High | Certain | High — affects single-name performance claims |
| R06 | Fill model defaults to MIDPOINT (optimistic) | Simulation | High | Certain | Medium — BID_ASK mode available |
| R07 | No earnings / event calendar integration | Data | High | Certain | Medium — statistical EVENT_RISK exists |
| R08 | Actual outcome back-fill is manual | ML | High | Likely | High — calibration monitor is blind without it |
| R09 | Low regime sample sizes for rare regimes | Statistical | High | Likely | Medium — min 30 samples filter exists |
| R10 | No real options data for backtesting | Data | High | Certain | High — current chain is point-in-time only |
| R11 | No authentication on governance endpoints | Security | High | Possible | High — any network exposure is dangerous |
| R12 | model_artifacts lost on container restart (some envs) | Engineering | High | Possible | Low in Docker bind-mount; high otherwise |
| R13 | IV rank uses current-session extremes, not 52-week | Options | Medium | Certain | Medium — overstates IV rank in short sessions |
| R14 | WebSocket is polling; not real event stream | Engineering | Medium | Certain | Medium — latency/stale data in dashboard |
| R15 | PSI reference distribution missing for new models | ML/Governance | Medium | Likely | Medium — drift alerted but not detected |
| R16 | No purge buffer in walk-forward splits | Statistical | Medium | Certain | Medium — multi-bar label horizon not used yet |
| R17 | American-style early exercise not modeled | Simulation | Medium | Possible | Medium — affects short equity puts near divs |
| R18 | Volatility crush post-event not modeled | Simulation | Medium | Likely | High — long vol positions overstated |
| R19 | No margin/buying-power enforcement | Simulation | Medium | Certain | Medium — simulator opens any position |
| R20 | yfinance adjusted-close revision bias | Data | Medium | Possible | Low — affects longer backtests after splits |

---

## Detailed Risk Descriptions

---

### R01 — 15-Minute Delayed Data Makes 5m Signals Unactionable

**Category:** Data quality
**Severity:** Critical
**Likelihood:** Certain (yfinance is always delayed)

**Description:**
yfinance provides approximately 15-minute delayed data. A signal generated on the close
of a 5-minute bar cannot be acted upon until the next real-time bar — which has already
closed 15 minutes in the past from the data perspective. In practice, the signal lag is
at least 15 minutes, often 20+. For 5-minute bars with typical intraday moves of 0.05–0.15%,
the edge implied by a correctly-calibrated 0.56 calibrated_prob_up is consumed multiple
times over by a 15-minute execution delay.

**Current Control:**
- Dashboard labels data as "15-min delayed"
- PAPER ONLY badge on all UI surfaces
- System is framed as a research tool, not a real-time system

**Residual Risk:** The entire 5-minute signal framework is research-only until integrated
with a sub-second or at minimum sub-minute data feed.

**Mitigation Path:**
1. Integrate Polygon.io (real-time) or Alpaca (free, near-real-time) via the existing
   `app/providers/protocols.py` MarketDataProvider interface
2. Add a bar-latency field to `ohlcv_bars` tracking `ingested_at - bar_close_time`
3. Gate live trading on `bar_latency < 30s` for 5m bars

---

### R02 — Model Trained Once; No Scheduled Retraining

**Category:** ML operations
**Severity:** Critical
**Likelihood:** Likely

**Description:**
The model in `model_artifacts/logistic.pkl` is trained manually by the user running the
backtest. There is no cron job, no automated trigger, and no deployment pipeline that
keeps the model current. The governance module correctly detects degradation
(`needs_retrain=True` in `calibration_snapshots`) but that flag is never read by any
scheduler or job runner. A model trained on Q4 2024 data operating in Q2 2025 regime
conditions will silently degrade, with only the `tradeable_confidence` shrinkage providing
any protection — which is not a substitute for model currency.

**Current Control:**
- `confidence_tracker.py` sets `needs_retrain=True` when `degradation_factor <= 0.40` or `ece_recent >= 0.12`
- `tradeable_confidence` degradation factor shrinks signals when rolling Brier worsens
- WARNING log emitted: `"RETRAIN RECOMMENDED for {symbol}"`

**Residual Risk:** High. The degradation factor provides a soft signal reduction but the
model continues operating on stale patterns. In a regime shift (e.g., VIX spike, rate
shock), the model can be confidently wrong for weeks before the degradation factor fully
suppresses it.

**Mitigation Path:**
1. Add a daily retraining job (celery/APScheduler) that:
   - Reads `calibration_snapshots` for `needs_retrain=True` entries
   - Runs backtest with last 90 days of data
   - Registers new model version → promotes via governance API
2. Define a 7-day maximum model age; flag any model older than 7 days as `model_artifact_stale`
   (governance freshness threshold already exists for this at 604800 seconds)

---

### R03 — Single Global Model Across All Symbols

**Category:** ML design
**Severity:** High
**Likelihood:** Certain (by current design)

**Description:**
There is one `logistic.pkl` shared across all symbols. A model trained on SPY (large-cap
ETF, high liquidity, low spread) will be applied to NVDA (single-name, earnings risk,
higher vol) or TSLA (highly sentiment-driven) without any symbol-specific calibration.
The feature distributions, typical return magnitudes, and regime transitions differ
substantially between these symbols, making shared model weights inappropriate.

**Current Control:**
- Each backtest overwrites `logistic.pkl` — so the model is implicitly "trained on the
  last symbol backtested"
- Regime suppression provides some protection in high-vol symbols

**Residual Risk:** High. Feature distributions (RSI, vol_regime, volume_ratio) have
different statistical properties across symbols. A logistic regression trained on SPY
will have weight vectors that are miscalibrated for NVDA's return distribution.

**Mitigation Path:**
1. Store models keyed by symbol: `model_artifacts/logistic_{SYMBOL}.pkl`
2. Modify `get_loaded_model(symbol)` to load symbol-specific artifact
3. Update backtest to produce per-symbol artifacts
4. Update governance model registry to track `training_symbol` as part of the
   artifact identity (column already exists in `model_versions.training_symbol`)

---

### R04 — Flat IV Surface; No Skew or Term Structure

**Category:** Options pricing
**Severity:** High
**Likelihood:** Certain (by current design)

**Description:**
The options decision engine uses `atm_iv` as a single scalar representing implied
volatility. Real options markets have a volatility surface: IV varies by strike (skew)
and by expiry (term structure). Specifically:
- OTM puts carry higher IV than ATM calls for equity underlyings (put skew)
- Near-term IV spikes before events and collapses after
- The `iv_skew` feature (`IV(25-delta put) - IV(25-delta call)`) exists in
  `OPTIONS_FEATURE_COLS` but the Black-Scholes pricing in the simulator uses flat vol

This means: (a) the model underestimates the true cost of downside protection,
(b) credit spreads using OTM puts appear more favorable than they are, and (c) the
IV/RV ratio comparison uses ATM IV rather than the relevant strike's IV.

**Current Control:**
- `iv_skew` feature in OPTIONS_FEATURE_COLS provides some signal
- `structure_evaluator.py` uses bid/ask from the options chain for cost estimation
  (when chain data is available), partially mitigating this for live runs

**Mitigation Path:**
1. Store `iv_by_strike` in `option_snapshots` as a JSON column
2. Interpolate strike-specific IV during structure evaluation
3. Use strike IV (not ATM IV) for Black-Scholes pricing in the simulator

---

### R05 — Survivorship Bias in All Backtests

**Category:** Statistical validity
**Severity:** High
**Likelihood:** Certain

**Description:**
All historical backtesting operates on symbols that exist today. Any company that was
delisted, went bankrupt, merged, or was acquired during the test period is excluded by
construction. For ETFs (SPY, QQQ) the bias is negligible. For single-name equities the
bias inflates backtested returns by 5–15% annualized. The specific mechanism: failing
companies have higher options premium (put skew is elevated), meaning any short-vol
strategy backtested on survivors systematically excludes the cases where the strategy
would have lost most.

**Current Control:**
- Documented in `FALSE_CONFIDENCE_AUDIT.md §2`
- Documented in `simulator_limitations.md §B2`
- No code-level mitigation exists

**Mitigation Path:**
1. For SPY/QQQ: bias is negligible; acceptable
2. For single-name equities: use a point-in-time constituent list
3. Minimum: add a warning to backtest results when `symbol not in {SPY, QQQ, IWM, etc.}`
   stating that results are subject to survivorship bias

---

### R06 — Fill Model Defaults to MIDPOINT (Optimistic Fills)

**Category:** Simulation fidelity
**Severity:** High
**Likelihood:** Certain (current default)

**Description:**
`SimulatorConfig` defaults to `FillMethod.MIDPOINT`. For a quote of `$1.90 / $2.10`,
a MIDPOINT fill assumes entry at $2.00. In reality, retail market orders fill at the
natural price ($2.10 to buy, $1.90 to sell). For a typical 10% spread, this understates
round-trip cost by 10% of the option's premium. On a $200 position, this is $20 per
round trip — material relative to typical short-term options edge.

**Current Control:**
- `FillMethod.BID_ASK` and `FillMethod.CONSERVATIVE` modes exist and are documented
- `simulator_limitations.md §F1` documents this explicitly

**Mitigation Path:**
1. Change the default in `SimulatorConfig` to `FillMethod.BID_ASK` for all non-test
   use cases. MIDPOINT should require explicit opt-in.
2. Add a warning when MIDPOINT is used in backtest results output

---

### R07 — No Earnings or Event Calendar Integration

**Category:** Data
**Severity:** High
**Likelihood:** Certain

**Description:**
EVENT_RISK regime is detected reactively: it triggers when a bar's move exceeds 3.5σ of
recent realized volatility. This means the system detects an event **after** it has
already moved the market, not before. Earnings announcements, FOMC decisions, and macro
data releases cause IV to spike in advance (the market prices in uncertainty). A system
that cannot see these events in advance will:
1. Fail to suppress long-gamma trades ahead of events (where IV is expensive)
2. Fail to identify short-gamma opportunities after event IV crush
3. Potentially enter positions the bar before a catalyst explodes IV

**Current Control:**
- Statistical EVENT_RISK detection (3.5σ) provides post-hoc protection
- `is_abnormal_move` feature in `RegimeSignals` captures outlier bars

**Mitigation Path:**
1. Integrate an earnings calendar API (e.g., Alpha Vantage earnings endpoint, free)
2. Add `days_to_earnings` as a feature; suppress any directional trade within 2 bars
   of an earnings date
3. Add a separate `pre_event_iv_elevated` flag to suppress long-vol trades when
   near-term IV is > 1.5× 30-day IV (IV already pricing the event)

---

### R08 — Actual Outcome Back-Fill Is Manual

**Category:** ML operations
**Severity:** High
**Likelihood:** Likely

**Description:**
The `inference_events.actual_outcome` field determines whether a prediction was correct.
This field drives: rolling Brier score, ECE, calibration health, degradation factor, and
ultimately the `needs_retrain` flag. Without outcome back-fill, the calibration monitor
has a window of zero and reports `calibration_health = "unknown"`. All governance
calibration thresholds become meaningless. Currently, outcomes are populated only when
`POST /api/uncertainty/{symbol}/record` or `POST /api/governance/inference/outcomes/bulk`
is called manually.

**Current Control:**
- Endpoints for bulk outcome recording exist
- `window_size < 40` warning shown in frontend

**Mitigation Path:**
1. Add a periodic task (every 15m during market hours) that:
   - Queries `inference_events` with `actual_outcome IS NULL` and `bar_open_time < NOW() - 5m`
   - Fetches the closing price of the next bar from `ohlcv_bars`
   - Computes `actual_outcome = 1 if close[i+1] > close[i] else 0`
   - Back-fills via `InferenceLogService.record_outcome()`

---

### R09 — Low Sample Sizes for Rare Regimes

**Category:** Statistical validity
**Severity:** High
**Likelihood:** Likely

**Description:**
EVENT_RISK and LIQUIDITY_POOR regimes are rare by construction. For a typical symbol
with 5,000 bars, these regimes account for < 1% of observations (< 50 bars). With n=30
(the minimum), a 95% confidence interval on Brier score is approximately ±0.07 — which
is wider than the typical Brier improvement over baseline (0.02–0.05). Regime-conditional
performance tables are therefore statistically unreliable and should not drive hard
trading decisions.

**Current Control:**
- `min_regime_samples = 30` filter exists in `ml_models/evaluation/regime.py`
- Documented in `FALSE_CONFIDENCE_AUDIT.md §7`

**Mitigation Path:**
1. Increase `min_regime_samples` to 50; warn at < 100
2. Report `n_samples ± 1.96 * sqrt(brier * (1-brier) / n)` confidence interval
   alongside all regime-conditional metrics
3. In the UI, show "insufficient data" for any regime metric below 100 samples

---

### R10 — No Historical Options Data for Backtesting

**Category:** Data
**Severity:** High
**Likelihood:** Certain

**Description:**
Backtesting currently uses only OHLCV price data. Options chain data (`option_snapshots`)
is populated on demand from yfinance at the current point in time. There is no mechanism
to retrieve what the options chain looked like on a historical date. This means:
- The `OPTIONS_FEATURE_COLS` (IV rank, skew, GEX proxy) use point-in-time current data
  as a proxy for historical values during backtesting, which is lookahead bias
- Any backtest result that uses options features is inflated by this leakage
- The `is_null_options = 1` sentinel path is the only honest backtesting path

**Current Control:**
- `is_null_options` sentinel feature exists; options features zeroed when unavailable
- L7 leakage test documents the invariant for snapshot joins
- No code currently joins historical options to backtest bars (sentinel path is default)

**Mitigation Path:**
1. Subscribe to a historical options data provider (CBOE DataShop, OptionsDX, or
   Polygon.io options endpoints)
2. Store historical chains in `option_snapshots` with accurate `snapshot_time`
3. The join invariant (`snapshot_time <= bar_open_time`) is already documented and tested
4. Until then: treat all options-feature backtest results as inflated and use the
   price-only path for honest performance attribution

---

### R11 — No Authentication on Governance Endpoints

**Category:** Security
**Severity:** High
**Likelihood:** Possible (depends on deployment)

**Description:**
All `/api/governance/*` endpoints are unauthenticated. The kill switch, model promotion,
and alert acknowledgement endpoints can be called by any process that can reach the backend.
In local Docker or Codespace environments (where access is gated at the network level)
this is acceptable. In any networked or internet-exposed deployment, this is a serious risk:
an attacker or misconfigured client could activate the kill switch, promote a malicious
model, or pollute the governance alert table.

**Mitigation Path:**
1. Add HTTP Bearer token authentication as FastAPI middleware
2. Use `SECRET_KEY` from settings.py to sign tokens
3. Protect all write endpoints (`POST`, `DELETE`) with authentication
4. Consider separate `ADMIN_KEY` for kill switch and model promotion

---

### R12 — Model Artifacts Lost on Container Restart in Some Environments

**Category:** Engineering
**Severity:** High
**Likelihood:** Possible

**Description:**
In the standard Docker Compose setup, `./backend:/app` bind-mounts the backend directory,
so `model_artifacts/` persists on the host. However, in Codespaces, CI environments, or
container-as-a-service deployments without bind mounts, the artifacts directory is ephemeral.
After a container restart, `_loaded_model = None` and the first inference call returns
`model_not_trained`. The user must re-run the backtest manually each time.

**Current Control:**
- Bind mount in `docker-compose.yml` handles the common case
- `get_loaded_model()` retries from disk on every call when cache is empty

**Mitigation Path:**
1. Store model artifacts in PostgreSQL as binary blobs (in `model_versions.artifact_dir`
   with actual file path, or add `artifact_bytes` column for small models < 10MB)
2. On startup, check `model_versions` for the active model and load from DB if no pkl exists
3. Alternatively, mount a persistent volume specifically for `model_artifacts/`

---

### R13 — IV Rank Uses Current Session, Not 52-Week Extremes

**Category:** Options intelligence
**Severity:** Medium
**Likelihood:** Certain

**Description:**
The `iv_rank` feature is computed as `(atm_iv - iv_52w_low) / (iv_52w_high - iv_52w_low)`.
In practice, the 52-week high and low are estimated from the options chain data available
at call time (yfinance returns some historical IV data but not reliably). If the current
trading session is short or the symbol's IV history is limited, `iv_rank` will be computed
relative to a compressed range, making it appear higher or lower than it truly is.

**Mitigation Path:**
1. Store a rolling 252-bar percentile of ATM IV in a dedicated table
2. Compute `iv_rank` from stored historical IV data, not from the current chain call

---

### R14 — WebSocket Is Polling; Not a Real Event Stream

**Category:** Engineering
**Severity:** Medium
**Likelihood:** Certain

**Description:**
The frontend `useWebSocket` hook sends an HTTP GET every 15 seconds to simulate a live
feed. This means quote and candle data shown on the dashboard is up to 15 seconds stale
regardless of the data vendor's own latency. On the backend, the `ws/` route exists but
does not push server-initiated messages. Any rapid intrabar price movement appears
only on the next 15-second poll.

**Mitigation Path:**
1. Implement a server-side WebSocket that pushes bar closes and quote updates
2. Use Redis pub/sub as the internal broadcast mechanism (infrastructure already present)
3. The `websocket/manager.py` connection manager is already implemented; the data push
   side needs to be wired to the ingestion service

---

### R15 — PSI Reference Distribution Missing for New Models

**Category:** ML governance
**Severity:** Medium
**Likelihood:** Likely

**Description:**
PSI drift detection compares the live feature distribution to the training reference
distribution stored in `feature_versions.reference_stats_json`. If a new model is trained
and registered without populating this JSON (which requires extra work at training time),
PSI falls back to a Gaussian approximation from the current production window. In this
fallback mode, PSI will always be near zero and no drift alerts will fire, even if
features have genuinely shifted.

**Mitigation Path:**
1. Make `reference_stats_json` mandatory at model registration time
2. Automatically compute and attach it in `backtest_service.run_backtest()` using
   the training feature matrix

---

### R16 — No Purge Buffer in Walk-Forward Splits

**Category:** Statistical validity
**Severity:** Medium
**Likelihood:** Certain

**Description:**
The walk-forward splitter (`ml_models/training/splitter.py`) uses contiguous
train/test windows with no gap between them. When labels use multi-bar horizons
(e.g., "return over next N bars"), the last N bars of the training window overlap
with the first N bars of the test window's label computation. This constitutes a form
of indirect leakage for multi-bar horizon models. Currently, the label is strictly
next-bar (`shift(-1)`), so this is not an active issue — but adding any multi-bar
label without adding a purge buffer would reintroduce leakage.

**Current Control:**
- Label is strictly `shift(-1)` (next bar only) — no current leakage
- `FALSE_CONFIDENCE_AUDIT.md` documents this risk

**Mitigation Path:**
Preemptively add `purge_bars=0` parameter to the splitter. Set to `h-1` whenever a
label horizon `h > 1` is introduced.

---

### R17 — American-Style Early Exercise Not Modeled

**Category:** Simulation
**Severity:** Medium
**Likelihood:** Possible

**Description:**
US equity options are American-style and can be exercised at any time before expiry.
Short puts on dividend-paying stocks are routinely assigned early when the dividend
exceeds the remaining extrinsic value. The simulator only evaluates assignment at
expiry (DTE=0), so early-assignment losses on short equity puts near ex-dividend dates
are not captured. See `simulator_limitations.md §M1`.

**Mitigation Path:**
1. Add a dividend calendar check: flag positions within 5 days of ex-dividend
2. For deep ITM short puts near ex-date: compute early-exercise probability using
   `extrinsic_value < dividend_amount` heuristic

---

### R18 — Volatility Crush Post-Event Not Modeled

**Category:** Simulation
**Severity:** Medium
**Likelihood:** Likely

**Description:**
After earnings announcements, FOMC decisions, or other binary events, implied volatility
typically collapses 30–60% regardless of the magnitude of the price move. Any long
volatility position (long straddle, long call, long put) held through an event will
experience a vega loss from IV crush that can exceed the delta gain from a favorable
price move. The simulator marks positions at current IV, which does not anticipate or
model the post-event collapse. See `simulator_limitations.md §M4`.

**Mitigation Path:**
1. Add a `post_event_iv_crush_factor` (e.g., 0.5×) applied to IV after known event dates
2. Requires earnings calendar integration (see R07)

---

### R19 — No Margin or Buying Power Enforcement

**Category:** Simulation
**Severity:** Medium
**Likelihood:** Certain

**Description:**
The risk manager enforces `max_daily_loss` and `max_position_size` as dollar limits, but
does not check margin requirements or buying power. A naked short put requires substantial
margin (typically 20% of underlying value); a credit spread requires the width of the
spread as collateral. The simulator can open any position regardless of whether the
account has sufficient capital, causing simulated portfolio leverage to be unrealistic.
See `simulator_limitations.md §R1–R2`.

**Mitigation Path:**
1. Add `margin_required` computation to `risk_guard.py` per position type:
   - Long options: premium paid (already tracked)
   - Debit spread: net debit (already tracked)
   - Credit spread: spread width × 100 × contracts
   - Naked short: 20% of underlying + premium
2. Add buying power check before any order is opened

---

### R20 — yfinance Adjusted-Close Revision Bias

**Category:** Data
**Severity:** Medium
**Likelihood:** Possible

**Description:**
yfinance returns `Adj Close` by default, which retroactively adjusts historical prices
for splits and dividends. If a split occurs after a backtest is run, re-running the same
backtest will return different historical prices for the pre-split period. Features
(especially momentum returns and VWAP distance) computed from the pre-revision data
will be inconsistent with features computed from the post-revision data. This makes
it impossible to reliably reproduce a historical backtest result after a corporate action.

**Current Control:**
- Documented in `LEAKAGE_AUDIT.md` (integration test caveat)

**Mitigation Path:**
1. Store raw (unadjusted) OHLCV data in `ohlcv_bars` with a separate `adj_factor` column
2. Apply the adjustment factor at feature computation time, not at ingestion time
3. Track `adj_factor` changes (corporate action log) to maintain reproducibility

---

## Risk Appetite Statement

This platform is for **research only**. No real capital is at risk. The appropriate risk
posture is therefore:
- Accept operational risks (authentication, container restarts) as medium priority
- Treat statistical and quantitative risks (leakage, survivorship, calibration) as
  first-priority — they determine whether research conclusions are valid
- Treat simulation fidelity risks (fills, IV surface, assignment) as second-priority —
  they determine whether paper P&L is a reliable proxy for live P&L
- Do not attempt to "paper over" documented limitations — the audit documents exist
  precisely to prevent overconfidence in results
