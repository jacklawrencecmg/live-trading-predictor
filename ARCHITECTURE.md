# Architecture — Options Research Platform

**As of:** April 2026
**Status:** Point-in-time accurate. This document describes what exists, not what is planned.
See `IMPLEMENTATION_PLAN.md` for roadmap.

---

## System Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│  Browser                                                               │
│  Next.js 14 + TypeScript + Tailwind (port 3000)                        │
│                                                                        │
│  Dashboard | Chart | Options Chain | Signals | Decision | Governance  │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │ REST (proxied via Next.js rewrites)
                               │ WebSocket (polling, 15s)
┌──────────────────────────────▼─────────────────────────────────────────┐
│  FastAPI (port 8000)                                                   │
│  12 router groups, async throughout                                    │
│                                                                        │
│  /api/market   /api/options  /api/model    /api/trades                 │
│  /api/backtest /api/signals  /api/inference /api/uncertainty           │
│  /api/regime   /api/decision /api/governance /ws/...                  │
└────┬──────────┬──────────────┬──────────────────────────────────────────┘
     │          │              │
     ▼          ▼              ▼
PostgreSQL    Redis         yfinance
(async/       (cache,        (15-min
asyncpg)      kill switch    delayed,
              state)         free tier)
```

---

## Layer Map

The backend is organized into 8 processing layers. Each layer has a strict
responsibility boundary — see Module Boundaries below.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 0: Data Ingestion                                             │
│  app/data_ingestion/  +  app/providers/                              │
│  yfinance → OHLCVBar → PostgreSQL ohlcv_bars                        │
│  yfinance → OptionSnapshot → PostgreSQL option_snapshots             │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ closed bars only (is_closed=True)
┌────────────────────────────────▼─────────────────────────────────────┐
│  Layer 1: Feature Pipeline                                           │
│  app/feature_pipeline/                                               │
│  30 FEATURE_COLS + 7 OPTIONS_FEATURE_COLS                            │
│  All OHLCV features use shift(1) — no lookahead                      │
│  Output: feature_row stored in DB; live row passed to inference      │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ feature vector (30 floats)
┌────────────────────────────────▼─────────────────────────────────────┐
│  Layer 2: ML Models                                                  │
│  app/ml_models/                                                      │
│  Training: walk-forward TimeSeriesSplit, no shuffle                  │
│  Models: LogisticRegression, RandomForest, GBT, NaiveBaseline        │
│  Calibration: Platt scaling or isotonic per fold                     │
│  Output: model_artifacts/logistic.pkl (+ _meta.json)                │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ predict_proba → raw (prob_up, prob_down)
┌────────────────────────────────▼─────────────────────────────────────┐
│  Layer 3: Inference (4-layer uncertainty)                            │
│  app/inference/inference_service.py                                  │
│                                                                      │
│  raw_prob → [CalibrationMap] → calibrated_prob                       │
│  calibrated_prob → [ConfidenceTracker] → tradeable_confidence        │
│  tradeable_confidence → [threshold + regime] → action/abstain       │
│                                                                      │
│  Output: InferenceResult persisted to inference_events               │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ InferenceResult
┌────────────────────────────────▼─────────────────────────────────────┐
│  Layer 4: Signal Scoring + Regime                                    │
│  app/inference/signal_scorer.py                                      │
│  app/regime/detector.py                                              │
│                                                                      │
│  8 regimes: TRENDING_UP/DOWN, MEAN_REVERTING, HIGH/LOW_VOLATILITY,   │
│             LIQUIDITY_POOR, EVENT_RISK, UNKNOWN                      │
│  Hard blocks: HIGH_VOLATILITY, LIQUIDITY_POOR, EVENT_RISK → ABSTAIN  │
│  Signal quality score 0–100 (edge 50% + regime 25% + vol 15% + move 10%)│
│                                                                      │
│  Output: ScoredSignal                                                │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ ScoredSignal + InferenceResult
┌────────────────────────────────▼─────────────────────────────────────┐
│  Layer 5: Options Decision Engine                                    │
│  app/decision/decision_engine.py                                     │
│  app/decision/iv_analysis.py                                         │
│  app/decision/structure_evaluator.py                                 │
│                                                                      │
│  IV analysis: ATM IV, realized vol, IV rank, IV/RV ratio             │
│  Structure candidates: long_call, long_put, debit_spread,            │
│                        credit_spread (scored + viability gated)      │
│  Hard disqualifiers: credit spread DTE > 1 day (horizon mismatch)    │
│                                                                      │
│  Output: OptionsDecision (direction + structure recommendation)      │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ OptionsDecision (advisory)
┌────────────────────────────────▼─────────────────────────────────────┐
│  Layer 6: Paper Trading + Risk                                       │
│  app/paper_trading/options_simulator/simulator.py                    │
│  app/services/risk_manager.py                                        │
│                                                                      │
│  Fill engine: MIDPOINT (default) or BID_ASK or CONSERVATIVE          │
│  Greeks: Black-Scholes (flat vol, no skew)                           │
│  Risk gates: max daily loss (2%), max position (5%), cooldown (15m)  │
│  Kill switch check on every order                                    │
│                                                                      │
│  Output: Trade/Position persisted to PostgreSQL                      │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ events
┌────────────────────────────────▼─────────────────────────────────────┐
│  Layer 7: Governance                                                 │
│  app/governance/                                                     │
│                                                                      │
│  ModelRegistry: version lifecycle (staging → active → deprecated)    │
│  FeatureRegistry: manifest hash tracking                             │
│  InferenceLog: append-only per-inference audit trail                 │
│  DriftMonitor: PSI-based feature distribution shift                  │
│  CalibrationMonitor: rolling Brier, ECE, degradation factor          │
│  DataFreshness: per-source staleness checks                          │
│  GovernanceAlerts: dedup-keyed alert table                           │
│  KillSwitch: two-layer (env-var hard override + DB-backed toggle)    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Inference Call (Request Path)

```
GET /api/signals/SPY
  │
  ├─ ingestion_service.get_closed_bars(SPY, limit=300)
  │    └─ SELECT * FROM ohlcv_bars WHERE symbol='SPY' AND is_closed=True
  │         ORDER BY bar_open_time DESC LIMIT 300
  │
  ├─ [optional] options_service.get_chain(SPY)
  │    └─ yfinance call → OptionSnapshot (staleness check: >3600s → sentinel)
  │
  ├─ inference_service.run_inference(df, symbol, options_features)
  │    ├─ feature_pipeline.build_feature_matrix(df, options_data)
  │    ├─ Guard: len(df) >= 30, last bar is_closed=True
  │    ├─ model.predict_proba(X)               ← Layer 2 pkl
  │    ├─ calibration_map.apply(raw_prob)       ← Layer 3a
  │    ├─ confidence_tracker.get_stats(symbol)  ← Layer 3b
  │    ├─ build_uncertainty_bundle(...)
  │    └─ return InferenceResult
  │
  ├─ regime.detect_regime_row(df)
  │
  ├─ signal_scorer.score(inference_result, regime_ctx)
  │
  └─ return SignalResponse{prediction, signal, trade_idea}
```

---

## Data Flow: Backtest + Training

```
POST /api/backtest/run {symbol, interval, period, n_folds, ...}
  │
  ├─ market_data.get_history(symbol, interval, period)  ← yfinance
  │
  ├─ backtest_service.run_backtest(df, req)
  │    ├─ _prepare_dataset(df)
  │    │    └─ feature_pipeline.build_feature_matrix(df)  ← same pipeline as inference
  │    │
  │    ├─ Walk-forward loop (n_folds):
  │    │    ├─ train on [start, start+train_size]
  │    │    ├─ test on [start+train_size, start+train_size+test_size]
  │    │    ├─ evaluate: accuracy, Brier, log-loss, MAE, Sharpe
  │    │    ├─ per-fold train_brier + overfit_ratio (catch overfitting)
  │    │    └─ per-fold Brier skill score vs naive baseline
  │    │
  │    ├─ Final model: train on ALL data
  │    │    ├─ model_service.train_models(X, y_dir, y_mag)
  │    │    ├─ model_service.set_models(dir_m, mag_m)    ← in-memory
  │    │    └─ baseline.save_model(dir_m, "logistic")    ← disk persistence
  │    │
  │    └─ Calibration: compute_calibration(all_y_true, all_y_prob)
  │
  └─ BacktestResult persisted to PostgreSQL
```

---

## Database Schema

```
ohlcv_bars            option_snapshots       feature_rows
─────────────         ────────────────       ────────────
id                    id                     id
symbol                underlying_symbol      symbol
timeframe             snapshot_time          bar_open_time
bar_open_time         expiry                 manifest_hash
bar_close_time        strike                 feature_values_json
open/high/low/close   option_type            [30 feature cols]
volume                bid/ask/mid
is_closed             iv / greeks
staleness_flag        open_interest
ingested_at           staleness_seconds
source                is_illiquid

trades                positions              backtest_results
──────                ─────────              ───────────────
id                    id                     id
symbol                symbol                 symbol / interval
option_symbol         option_symbol          n_folds
action (BTO/STC/...)  quantity               accuracy
quantity              avg_cost               brier_score
price                 current_price          brier_skill_score
executed_at           unrealized_pnl         log_loss
model_prob_up/down    realized_pnl           sharpe_ratio
model_confidence      is_open                fold_results_json

── Governance tables (app/governance/models.py) ──

model_versions        feature_versions       inference_events
─────────────         ────────────────       ────────────────
id                    manifest_hash          id / request_id
model_name            pipeline_version       symbol / bar_open_time
version_tag           feature_count          model_name / version_id
status                feature_list_json      prob_up / prob_down
trained_at            reference_stats_json   calibrated_prob_up
n_samples/features                           tradeable_confidence
feature_manifest_hash                        action / abstain_reason
artifact_sha256                              calibration_health
                                             regime / options_stale
                                             actual_outcome (back-fill)

drift_snapshots       calibration_snapshots  governance_alerts
───────────────       ─────────────────────  ─────────────────
symbol                symbol                 alert_type
computed_at           snapshot_at            severity
psi_by_feature_json   rolling_brier          dedup_key
max_psi / mean_psi    baseline_brier         title / details_json
drift_level           degradation_factor     triggered_at
                      ece_recent             acknowledged_at
                      calibration_health     is_active
                      needs_retrain

kill_switch_state     regime_labels          audit_log
─────────────────     ─────────────          ─────────
active (boolean)      symbol                 table_name
reason                bar_open_time          row_id
activated_at          regime                 action
activated_by          confidence             changed_at
```

---

## Module Boundaries

| Module | Responsibility | Must NOT |
|--------|---------------|----------|
| `data_ingestion` | Fetch raw OHLCV + options from vendors; write to DB | Compute features, train models |
| `feature_pipeline` | Build feature matrix from closed bars | Fetch data, call models |
| `ml_models` | Train, calibrate, serialize, load models | Fetch data, make execution decisions |
| `inference` | Load model, run 4-layer prediction | Train models |
| `regime` | Classify market state from bars | Make execution decisions |
| `decision` | Map inference output to options structure | Train models, execute trades |
| `paper_trading` | Simulate fills and track positions | Make prediction decisions |
| `risk_manager` | Enforce capital/position limits | Make prediction decisions |
| `governance` | Monitor, log, alert on system state | Modify inference output |
| `providers` | Abstract external data/broker APIs | Implement business logic |

---

## Current Gaps (Point-in-Time Accurate)

### Data Layer
- **No live data.** yfinance is 15-minute delayed. Signals computed on delayed data
  cannot be acted upon before the next bar closes. The system is a research tool, not
  a near-real-time trading system, until a live feed is integrated.
- **Options data is not persisted systematically.** `option_snapshots` table exists
  but is populated only on demand (per API call), not on a schedule. Historical
  options chain data for backtesting is unavailable.
- **No earnings/event calendar.** EVENT_RISK regime detection is based on statistical
  abnormal-move detection (3.5σ), not a calendar of known catalysts. Earnings, FOMC,
  and macro events are detected reactively, not anticipated.

### Model Layer
- **Single global model.** One `logistic.pkl` per process. All symbols share the same
  model weights. A model trained on SPY will be used to predict NVDA without
  symbol-specific adaptation.
- **No scheduled retraining.** The `needs_retrain` flag in `calibration_snapshots` is
  set correctly but never wired to a retraining job. Retraining is manual.
- **No multi-timeframe context.** The model sees only 5m bars. Daily or weekly regime
  context (e.g., market is in a weekly downtrend) is not available to the predictor.
- **Flat IV surface.** The `atm_iv` feature is a single scalar. No IV skew, no term
  structure. The Black-Scholes pricing in the simulator uses flat vol.
- **Actual outcome back-fill is manual.** `actual_outcome` in `inference_events` must
  be populated by calling `POST /api/uncertainty/{symbol}/record`. No automated
  scheduler closes the feedback loop.

### Engineering Layer
- **No authentication.** All governance endpoints are unauthenticated. This is acceptable
  for local/Codespace development but must be addressed before any networked deployment.
- **WebSocket is polling.** The `useWebSocket` hook polls every 15 seconds via REST.
  It is not a true WebSocket stream. The `ws/` route exists but is not used for real
  price streaming.
- **No CI/CD.** Tests are not run on push. The GitHub Actions workflow does not exist.
- **No observability.** No Prometheus metrics, no structured log export, no distributed
  tracing. Application health is measured only via the governance API.
- **`model_artifacts/` is inside the container.** In Docker, the backend volume mounts
  `./backend:/app`, so `model_artifacts/` is persisted to the host. In environments
  without a bind mount (e.g., pure container deployments), artifacts are ephemeral.

### Options Intelligence
- **No real IV skew.** The structure evaluator scores debit vs credit spreads based on
  a flat-vol proxy. Real options edge comes from skew, term structure, and GEX — none
  of which are modeled.
- **No liquidity scoring at the contract level.** `liquidity_quality` is estimated from
  ATM bid-ask spread. Individual OTM strike liquidity is not assessed before
  recommending a specific strike.
- **No pin risk or assignment risk.** See `simulator_limitations.md` M1–M2.

---

## Service Boundaries

### What lives in the database (shared, durable)
- OHLCV bars, option snapshots, feature rows
- Trades, positions, backtest results
- All governance tables (inference events, alerts, drift, calibration)
- Kill switch state

### What lives in Redis (fast, ephemeral)
- Kill switch active state (TTL cache, 5s)
- Quote cache (short TTL, avoids hammering yfinance)
- WebSocket session state

### What lives on disk (durable, process-local)
- `model_artifacts/logistic.pkl` and `_meta.json`
- `model_artifacts/trackers/tracker_{SYMBOL}.json` (rolling confidence window)

### What lives in memory (lost on restart)
- `inference_service._loaded_model` (reloads from pkl on next request)
- `inference_service._calibration_map` (must be reloaded after backtest)
- `model_service._direction_model` (legacy; use pkl-backed path instead)

---

## Testing Boundaries

| Test file | What it covers | Must pass before |
|-----------|----------------|------------------|
| `test_leakage.py` | L1–L9 leakage invariants | Any feature pipeline change |
| `test_feature_pipeline.py` | Feature computation correctness | Any feature change |
| `test_labels.py` | Label generation correctness | Any label change |
| `test_model_training.py` | Walk-forward training | Any model change |
| `test_model_service.py` | Inference consistency | Any inference change |
| `test_governance.py` | Alert, registry, kill switch | Any governance change |
| `test_regime.py` | Regime classification | Any regime change |
| `test_options_simulator.py` | Simulator mechanics | Any simulator change |
| `test_paper_trader.py` | P&L accounting | Any paper trader change |
| `test_risk_manager.py` | Risk gate logic | Any risk change |
| `test_decision_engine.py` | Decision logic | Any decision change |
| `test_data_lineage.py` | Timestamp invariants | Any ingestion change |

All tests use async fixtures and `aiosqlite` (in-memory) for DB. Redis is stubbed.
Tests must not make real network calls.
