# Implementation Plan — Options Research Platform

**As of:** April 2026
**Scope:** Turning the current research prototype into a serious, uncertainty-aware
options research platform with honest performance measurement.

This plan is organized into phases. Each phase builds on the previous and should be
completed before starting the next. Within a phase, items are ordered by priority.

---

## Where a Naive "Next Candle Predictor" Fails

Before the phases, this section explains why a simple ML model predicting "will the next
candle be up or down?" is insufficient for options research — and what this platform must
do instead.

### Failure Mode 1: Data Timing Makes 5m Signals Meaningless

A signal on bar N at 10:00:00 means: "the 09:55:00–10:00:00 bar closed bullishly."
With 15-minute delayed data, you cannot act until 10:15:00. The 10:00:00 bar is already
three bars in the past. The signal has expired before you can act on it.

**What this means:** A next-candle predictor is only valuable with sub-60-second data
latency. Everything else is research only.

### Failure Mode 2: The Option Already Prices In the Expected Move

If SPY has 15% ATM IV, the market is pricing an 0.94% daily move (15% / √252). A model
that predicts "up with 56% probability" has a tiny edge over the 50% prior — but the
option premium already incorporates far more uncertainty than your edge can overcome.
You need IV/RV edge (realized vol < implied vol) to be short options profitably, or a
directional edge large enough to overcome the premium you pay to be long.

**What this means:** Predicting direction is necessary but not sufficient. You must also
have a view on whether implied volatility is cheap or expensive relative to what will
realize. A next-candle predictor has no IV view.

### Failure Mode 3: Regime Dependency Destroys Out-of-Sample Performance

A logistic regression trained on 6 months of trending SPY data learns: "RSI > 60 +
positive MACD histogram = bullish." This works in a trending market. In a mean-reverting
or high-volatility regime, the same signal is reliably wrong. Out-of-sample performance
collapses if the regime shifts between training and inference.

**What this means:** A next-candle predictor without regime detection produces confident
wrong signals at the worst possible times — in regime transitions.

### Failure Mode 4: Options P&L Is Not Linear in Direction

A 5% directional prediction edge on a 5-minute bar produces roughly 0.05% return if you
are long stock. On a long option with 0.30 delta, the actual return depends on delta,
gamma (second-order move), theta (time decay per bar), and IV (if vol changes). You can
be right about direction and still lose money if:
- Theta decay exceeds delta gain (long options in low-vol grinding markets)
- IV decreases after entry (long vol position during IV crush)
- You are assigned early on a short put (American exercise)
- The spread you pay to enter exceeds your expected edge

**What this means:** Predicting "up 55%" is not a complete trading signal. You need to
know: by how much, over what horizon, and whether the option's premium reflects a fair
price for that uncertainty.

### Failure Mode 5: Accuracy Is the Wrong Metric

A model that predicts "up" every bar achieves ~52% accuracy for SPY (the base rate
of up-days since 2000). This looks like a performing model. Its Brier score is 0.2496,
marginally worse than random (0.2500). Its Brier skill score is -0.0002 (harmful).
You cannot build a trading strategy on this.

**What this means:** Every performance claim about a next-candle predictor must be
stated in terms of Brier skill score and calibration, not accuracy. A BSS below 0.05
should be considered marginal at best.

### Failure Mode 6: Transaction Costs at the Options Level Are Regime-Sensitive

Options bid-ask spreads widen sharply in high-volatility environments — exactly when
your directional model is most likely to be activated (because it sees large moves as
signal). In the regimes where the model fires most aggressively, real-world transaction
costs are also highest. A backtest using MIDPOINT fills systematically understates this
cost precisely when it matters most.

---

## Phase 1: Wire What Exists (2–4 weeks)

The governance, calibration, and feedback systems are built but not wired together.
Phase 1 closes these loops without adding new features.

### 1.1 Automated Outcome Back-Fill
**What:** A background task that records `actual_outcome` for resolved inference events.

**How:**
1. Add a `BackgroundTask` or APScheduler job running every 15 minutes during market hours
2. Query `inference_events WHERE actual_outcome IS NULL AND bar_open_time < NOW() - 10m`
3. For each: fetch `ohlcv_bars.close` for `bar_open_time` and the next bar
4. Compute `outcome = 1 if close_next > close_current else 0`
5. Call `InferenceLogService.record_outcome(event_id, outcome)`

**Why this matters:** Without this, `calibration_health` is permanently "unknown" and the
degradation factor never updates. The governance system is blind.

### 1.2 Per-Symbol Model Artifacts
**What:** Store and load models keyed by symbol, not globally.

**How:**
1. Rename artifact to `model_artifacts/logistic_{SYMBOL}.pkl`
2. Modify `backtest_service.run_backtest()` to use `save_model(model, f"logistic_{req.symbol}")`
3. Modify `inference_service.get_loaded_model(symbol)` to try `logistic_{symbol}` first,
   fall back to `logistic` (cross-symbol baseline)
4. Update governance model registry: `training_symbol` is now mandatory

### 1.3 Model Freshness Gate
**What:** Block inference when the model artifact is older than 7 days.

**How:**
1. Read `_meta.json` alongside the pkl at load time
2. Check `trained_at` against `datetime.utcnow()`
3. If age > 7 days: set `abstain_reason = "model_stale:N_days"` and log a governance alert
4. Show age in the Model Health panel

### 1.4 Change Default Fill to BID_ASK
**What:** Set `FillMethod.BID_ASK` as the default in `SimulatorConfig`.

**How:** One-line change in `paper_trading/options_simulator/config.py`.
**Risk:** Reduces paper P&L, which is correct — previously reported P&L was optimistic.

### 1.5 PSI Reference Distribution at Training
**What:** Populate `feature_versions.reference_stats_json` automatically during backtest.

**How:**
1. In `backtest_service.run_backtest()`, after building the full feature matrix,
   compute percentile stats for each feature in `FEATURE_COLS`
2. Call `FeatureRegistryService.ensure_manifest(db, reference_stats_json=...)`
3. This makes PSI drift detection produce accurate (not Gaussian-fallback) results

### 1.6 Basic Authentication on Governance Write Endpoints
**What:** Protect kill switch, model promotion, and alert acknowledgement with a token.

**How:**
1. Add `ADMIN_TOKEN` to `.env` and `settings.py`
2. Add a FastAPI `Depends` that checks `Authorization: Bearer {ADMIN_TOKEN}` on
   all `POST`/`DELETE` routes in `governance.py`
3. Frontend: pass the token from environment on admin actions

---

## Phase 2: Data Layer Upgrade (4–8 weeks)

The current data layer uses free delayed data. Phase 2 provides the data infrastructure
needed for research-quality signal evaluation.

### 2.1 Live/Near-Real-Time Market Data Feed
**Target:** Sub-60-second bar data for 5m bars during market hours.

**Options (in order of effort):**
1. **Alpaca Markets API** — free, real-time bars for US equities via WebSocket
2. **Polygon.io** — paid, comprehensive; supports options too
3. **Tradier** — free tier includes options chains

**How:**
1. Implement `AlpacaMarketDataProvider` in `app/providers/` implementing `MarketDataProvider` protocol
2. Wire to data ingestion service via provider interface
3. Add `source="alpaca"` to `ohlcv_bars` rows from the live feed
4. Maintain yfinance as the historical backfill source

### 2.2 True WebSocket Feed to Frontend
**What:** Server-pushed price updates; not polling.

**How:**
1. Backend: on each new closed bar from the live feed, publish to Redis channel
   `bars:{symbol}`
2. Backend WebSocket route (`/ws/{symbol}`) subscribes to Redis and pushes bar closes
   and quote ticks to the client
3. Frontend: `useWebSocket` hook connects to the real WS endpoint; remove the 15-second
   polling fallback (or keep as fallback only)

### 2.3 Earnings Calendar Integration
**What:** Know when earnings are scheduled; suppress directional positions within 2 bars.

**Options:**
1. Alpha Vantage earnings calendar (free tier, JSON)
2. Polygon.io ticker details (paid)
3. Open-source lists (manually maintained quarterly)

**How:**
1. New table: `earnings_calendar (symbol, report_date, fiscal_quarter, source)`
2. New provider method: `EventCalendarProvider.get_upcoming_events(symbol, days=10)`
3. In `inference_service.run_inference()`: if `days_to_next_earnings <= 2`, set
   `prior_abstain_reason = "pre_earnings_suppression"` unless directional conviction
   is very high (calibrated_prob > 0.70)
4. Add `days_to_earnings` to `FEATURE_COLS` (requires pipeline version bump + retrain)

### 2.4 Historical Options Data (Research Quality)
**What:** Historical options chains for honest backtesting of options features.

**Options:**
1. OptionsDX (free, daily CBOE data for SPY/QQQ going back to 2005)
2. CBOE DataShop (paid, comprehensive)
3. Polygon.io options endpoint (paid)

**How:**
1. Build a one-time historical options loader that populates `option_snapshots` with
   historical data, keyed by `snapshot_time = bar_open_time` for each trading date
2. The L7 leakage invariant (`snapshot_time <= bar_open_time`) is already documented
   and tested; no new safety work required
3. With historical options data: remove the `is_null_options = 1` sentinel path from
   backtests; use real IV/skew/GEX features historically

### 2.5 Outcome Recording Completeness Dashboard
**What:** Surface `n_pending / n_total` in the governance dashboard so operators can
see how complete the feedback loop is.

**How:** Extend `GET /api/governance/inference/stats/{symbol}` to include
`pct_outcomes_resolved` and show it in the Model Health panel.

---

## Phase 3: Model Layer Upgrade (4–8 weeks)

Phase 3 builds research-grade model infrastructure on top of the honest data from Phase 2.

### 3.1 Automated Retraining Pipeline
**What:** Weekly retraining triggered by `needs_retrain` flag or model age > 7 days.

**How:**
1. Add APScheduler job: `retrain_if_needed()` runs Sunday at midnight UTC
2. Reads `calibration_snapshots` for `needs_retrain=True` OR latest model age > 7d
3. Calls `run_backtest()` with last 90 calendar days of data (approximately 18 trading weeks)
4. On success: registers new model version → promotes via governance → sets baseline Brier
5. On failure: raises `CRITICAL` governance alert; keeps previous model active

### 3.2 Multi-Timeframe Features
**What:** Add daily bar context to the 5m prediction model.

**Rationale:** A 5m signal in the direction of the daily trend has a higher base rate
than a counter-trend 5m signal. Daily context (above/below 20-day SMA, daily RSI,
daily ATR regime) provides information the 5m feature set cannot capture.

**How:**
1. Add a daily bar table (`ohlcv_bars_daily`) or use a `timeframe='1d'` column
2. Compute daily features: `daily_rsi`, `daily_ret_5`, `daily_trend` (above/below SMA20)
3. Add 3–5 daily features to `FEATURE_COLS`; this requires `PIPELINE_VERSION` bump and retrain
4. Test: leakage tests must ensure daily features use the most recent *closed daily bar*
   (yesterday's data, not today's)

### 3.3 Symbol-Specific Baseline Calibration
**What:** Track per-symbol base rates to calibrate predictions against symbol history.

**Rationale:** SPY has a ~52% up-bar rate at 5m; high-beta names may differ. The
model's prior should reflect the symbol's actual distribution, not a generic 50%.

**How:**
1. Add `symbol_stats` table: `(symbol, timeframe, up_bar_rate, vol_percentile_10_90, n_bars)`
2. Compute from `ohlcv_bars` on each retraining run
3. Use symbol-specific base rate as the naive Brier reference in `brier_skill_score`
4. Adjust logistic regression's `class_weight` to match the symbol's observed base rate

### 3.4 Model Selection and Champion/Challenger Framework
**What:** Formal A/B evaluation between model types before promoting to active.

**How:**
1. After training, new models enter `staging` status in the governance registry
2. A "shadow run" period: the staging model's predictions are logged to `inference_events`
   with `model_version_id = staging_id` but do not affect paper trading
3. After 200 resolved bars: compare staging model Brier skill score vs active model
4. Promote staging to active only if BSS improvement > 0.005 (statistically meaningful
   given regime-specific sample sizes)

### 3.5 Hyperparameter Sensitivity Analysis
**What:** Understand how sensitive backtest results are to confidence threshold, train
size, and test size parameters.

**How:**
1. Add a `sensitivity_run` API endpoint: runs the same backtest across a grid of
   `confidence_threshold in [0.50, 0.55, 0.60, 0.65]` and reports Sharpe vs accuracy
2. Output: heatmap of performance vs threshold (visible in Backtest panel)
3. Key check: if performance peaks at threshold = 0.65+ on training data but not on OOS,
   this indicates threshold overfitting — the threshold itself has been overfitted to
   the training regime

---

## Phase 4: Options Intelligence (6–10 weeks)

Phase 4 builds the options-specific reasoning layer that makes this a real options
research platform rather than a price predictor with options output bolted on.

### 4.1 IV Skew and Term Structure
**What:** Model the IV surface, not just ATM IV.

**How:**
1. Extend `option_snapshots` with `iv_surface_json`: a JSON dict of `{strike: iv}` per expiry
2. Add IV surface features:
   - `skew_25d`: `IV(25Δ put) - IV(25Δ call)` — directional skew
   - `term_prem`: `IV(30d) - IV(7d)` — term structure premium
   - `skew_slope`: linear slope of IV vs moneyness
3. Use strike-specific IV in structure evaluation (replace ATM IV for OTM strikes)
4. Use strike-specific IV in Black-Scholes pricing in the simulator

### 4.2 GEX (Gamma Exposure) Market State
**What:** Estimate dealer gamma positioning and its effect on intraday price pinning.

**Rationale:** When dealers are short gamma (negative GEX), they amplify intraday moves
(buy more to hedge as price rises, sell more as price falls). When dealers are long gamma
(positive GEX), they suppress moves (sell as price rises, buy as price falls). GEX near
zero or transitioning to negative is associated with higher realized vol.

**How:**
1. `gex_proxy` feature already exists in `OPTIONS_FEATURE_COLS`; improve the computation:
   `GEX = sum(gamma_i × OI_i × 100 × spot)` per strike, sum across the chain
2. Add `gex_sign` (positive/negative) and `gex_magnitude` as features
3. Use sign flip as an additional component in EVENT_RISK regime detection

### 4.3 Real Expected Move Pricing
**What:** Use the options market's own expected move as the baseline, not realized vol.

**Rationale:** The model's `expected_move_pct` currently uses a realized vol proxy.
The market's consensus expected move is available directly from ATM straddle pricing:
`expected_move = (call_mid + put_mid) / spot`. This is a better benchmark for whether
a directional move thesis makes sense at current option prices.

**How:**
1. Compute `atm_straddle_expected_move = (ATM_call_mid + ATM_put_mid) / spot` from chain
2. Add to `IVAnalysis` in the decision engine
3. Gate debit spread recommendations on `model_expected_move > atm_straddle_expected_move`
   (only buy a directional spread if your model expects a larger move than the market does)

### 4.4 Liquidity Scoring at Strike Level
**What:** Score liquidity for the specific strikes being considered, not just ATM.

**How:**
1. Compute per-strike: `bid_ask_pct = (ask - bid) / mid` and `oi_rank` (OI percentile
   vs all strikes in the chain)
2. In `structure_evaluator.py`: add per-leg liquidity score to each candidate
3. Reject structures where any leg has `bid_ask_pct > 0.25` or `OI < 50`

### 4.5 Pin Risk and Expiry Risk Model
**What:** Flag positions approaching expiry near a strike with high open interest.

**Rationale:** When the underlying price converges on a strike with very high OI at
expiry, the probability of assignment and the direction of gamma effects are uncertain.
This is "pin risk."

**How:**
1. Add `dist_to_max_oi_strike` as an inference signal
2. At DTE = 0 or DTE = 1 with `|spot - strike_max_oi| / spot < 0.005`: flag as
   `pin_risk_elevated` in the governance alerts
3. Recommend closing any positions with elevated pin risk before the last trading hour

---

## Phase 5: Production Readiness (4–6 weeks)

Phase 5 handles the engineering work needed before this platform could support any
real-money research review or external user.

### 5.1 Authentication and Multi-User Support
- JWT-based authentication for all API endpoints
- Separate admin role (kill switch, model promotion) from read-only role
- Frontend: login screen + token storage

### 5.2 CI/CD Pipeline
- GitHub Actions workflow: on every push, run `pytest tests/ --tb=short`
- Leakage tests are a hard gate: PR is blocked if any `test_leakage.py` test fails
- Docker build check: `docker-compose build` must succeed

### 5.3 Observability
- Prometheus metrics endpoint (`/metrics`)
- Key counters: inference rate, abstain rate by reason, backtest runs, model promotions
- Grafana dashboard: model health, data freshness, inference volume, calibration trend
- Alert routing: critical governance alerts → Slack/webhook (`alert_channels.py` already supports this)

### 5.4 Scheduled Tasks Framework
Wire all automated jobs into a single scheduler:
- Every 15m: outcome back-fill (`BackgroundTask`)
- Every hour: data freshness checks for all active symbols
- Every 6h: PSI drift computation (`POST /api/governance/drift/run`)
- Every 6h: calibration snapshot (`POST /api/governance/calibration/{symbol}/snapshot`)
- Weekly (Sunday midnight): retraining check (`retrain_if_needed()`)

### 5.5 Data Retention and Archival
- `inference_events` will grow unboundedly; add a configurable retention window (default: 1 year)
- Archive old records to a cold storage table before deletion
- `ohlcv_bars`: retain 2 years for active symbols; prune older data

---

## Service, Data, and Testing Boundaries

### Service Boundaries

| Service | Owns | Does NOT own |
|---------|------|-------------|
| Ingestion service | Bar and options data freshness | Feature computation |
| Feature service | Feature manifest, feature rows | Model weights |
| Training service | Model artifacts, calibration maps | Live inference |
| Inference service | 4-layer uncertainty, inference events | Paper trading decisions |
| Decision service | Options structure scoring | Execution |
| Paper trading service | P&L accounting, positions | Signal generation |
| Governance service | All monitoring and alerting | Any trading decision |
| Scheduler service | Job orchestration | Business logic |

### Data Boundaries

| Data Store | What lives here | What does NOT |
|-----------|----------------|---------------|
| PostgreSQL | All durable records: bars, features, trades, governance | Model weights, caches |
| Redis | Kill switch state (TTL), quote cache (TTL 10s) | Durable records |
| Disk (`model_artifacts/`) | Model pkl + meta, confidence tracker JSON | Raw market data |
| Frontend state | UI display state | Any computed signals |

### Testing Boundaries

| Test scope | Must cover | Must NOT do |
|-----------|------------|-------------|
| Unit tests | One function/class in isolation; no I/O | Make network calls, require live DB |
| Integration tests | Multi-module interactions; uses in-memory DB | Require external services |
| Leakage tests | Mathematical invariants of feature pipeline | Mock the feature pipeline |
| Backtesting tests | Walk-forward split correctness | Use real market data (synthetic only) |
| Governance tests | Alert dedup, kill switch state | Require Redis (use null stub) |

---

## Non-Goals (Out of Scope)

These items are explicitly out of scope regardless of how simple they might seem:

- **Live trading.** All orders are paper-only. No broker connectivity for real orders.
- **Crypto.** The platform is US equity options only. The regime detection and IV surface
  models are calibrated for US equity market structure.
- **Options strategy management (rolling, adjustments).** The simulator supports
  open/close but not continuous position management. See `simulator_limitations.md §F5`.
- **Macro or fundamental signals.** The model uses price/volume/IV features only.
  No earnings estimates, no economic indicators, no sentiment data.
- **Portfolio optimization across positions.** The risk manager enforces per-position
  limits. Cross-position correlation and portfolio-level Greek neutrality are not modeled.

---

## Recommended First 30 Days

1. **Day 1–3:** Complete Phase 1.4 (BID_ASK fills) and 1.5 (PSI reference stats).
   These are one-line and two-line changes with high impact on measurement honesty.

2. **Day 4–7:** Build Phase 1.1 (automated outcome back-fill). Without this, the
   calibration monitor is dark and all governance health metrics are "unknown."

3. **Day 8–14:** Complete Phase 1.2 (per-symbol models) and 1.3 (model freshness gate).
   These make the platform honest about what each model actually knows.

4. **Day 15–21:** Start Phase 2.1 (live data feed). Even a free Alpaca integration
   transforms the platform from "delayed research tool" to "near-real-time research tool."
   This is the single highest-leverage infrastructure change available.

5. **Day 22–30:** Wire Phase 1.6 (basic authentication) and set up CI with leakage
   test gate. These are prerequisites for any shared or networked use.
