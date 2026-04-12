# CLAUDE.md — Binding Guidance for This Repository

This file is read by Claude Code at the start of every session. Every instruction here is binding.
Treat it as the engineering constitution for this project. When in doubt, be more conservative.

---

## What This Project Is

A **paper-trading options research platform** for developing and validating quantitative
trading signals against US equity options. It is not a live trading system. No real money
is ever at stake. The goal is to build a rigorous research environment where model quality
can be honestly measured before any live deployment decision is made.

The system is **uncertainty-aware by design**. Every inference output carries a 4-layer
confidence hierarchy: raw probability → calibrated probability → tradeable confidence →
action/abstain. A signal that does not clear all four layers produces ABSTAIN. This is
correct behavior, not a bug.

---

## Non-Negotiable Invariants

### NI-1: No Lookahead Bias
`feature[i]` must only use data from bars `0..i-1`. This is enforced via `.shift(1)` on
all OHLCV series before any rolling or EWM computation. The leakage regression tests
(L1–L9 in `tests/test_leakage.py`) must always pass. Any change to `feature_pipeline/`
must be preceded by running these tests.

**If you modify any feature computation: run leakage tests before proceeding.**

### NI-2: Train/Inference Feature Consistency
Training and inference must use the identical feature manifest (`FEATURE_COLS` in
`app/feature_pipeline/registry.py`). The manifest hash is the contract. If you change
`FEATURE_COLS`, you must:
1. Increment the pipeline version
2. Delete all existing pkl files in `model_artifacts/`
3. Retrain before any inference runs

**Never modify FEATURE_COLS without deleting stale model artifacts.**

### NI-3: Paper Only
No code path may place a real order. `paper_trader.py` and the options simulator are the
only execution surfaces. If you are ever asked to add live broker connectivity, stop and
confirm the request explicitly before touching execution code.

### NI-4: Calibration Is Primary
Accuracy is not the success metric. Brier skill score and ECE are. A model with 54%
accuracy that is well-calibrated is better than a 60% accurate model that is overconfident.
Never optimize for accuracy at the expense of calibration.

### NI-5: Governance Log Is Append-Only
`inference_events` rows are never updated except for the `actual_outcome` back-fill column.
`governance_alerts` uses deduplication (same `dedup_key` → bump `triggered_at`), not
deletion. Do not add any code that deletes or mutates historical governance records.

### NI-6: Regime Blocks Are Hard Blocks
When regime detection suppresses a trade (HIGH_VOLATILITY, LIQUIDITY_POOR, EVENT_RISK),
the system must ABSTAIN. These are not soft suggestions. Do not add bypass logic or
override flags that allow trading through hard-suppressed regimes.

---

## Architecture Rules

### AR-1: Layer Isolation
Each layer must not reach into adjacent layers. The module boundary table in
`docs/architecture.md` is authoritative. Examples:
- `feature_pipeline` must not fetch data or call models
- `inference_service` must not train models
- `paper_trading` must not read feature state directly
- `governance` must not modify inference output

### AR-2: Async Throughout
All I/O is async. Do not introduce synchronous database calls, synchronous HTTP requests,
or blocking file I/O in hot paths. Use `asyncpg` for DB and `httpx` (async) for any HTTP.
`aiosqlite` is for tests only.

### AR-3: Provider Pattern for External Data
All external data comes through the `app/providers/` protocol interfaces. Do not call
`yfinance` directly from routers or services. When adding a new data source, implement
the appropriate `Protocol` class from `app/providers/protocols.py`.

### AR-4: Model Artifacts Are Versioned
Any time a model is saved to `model_artifacts/`, it must be accompanied by:
1. A `_meta.json` file with feature manifest hash, training symbol, trained_at timestamp
2. Registration in the governance model registry (`ModelRegistryService.register()`)

Do not write bare pkl files without metadata.

### AR-5: No Hardcoded URLs
The frontend uses Next.js rewrites to proxy `/api/*` and `/ws/*` to the backend.
`api.ts` uses `baseURL: ""`. Never hardcode `http://localhost:8000` or any Codespace URL
into frontend code. The `BACKEND_URL` env var in docker-compose.yml is the only place
this is configured.

---

## Testing Requirements

### Before any backend change:
```bash
cd backend && pytest tests/test_leakage.py -v   # must pass — hard gate
```

### Before any feature pipeline change:
```bash
cd backend && pytest tests/test_feature_pipeline.py tests/test_leakage.py -v
```

### Before any model or inference change:
```bash
cd backend && pytest tests/test_model_service.py tests/test_model_training.py -v
```

### Full suite:
```bash
cd backend && pytest tests/ -v --tb=short
```

Tests in `test_leakage.py` are load-bearing. A failing leakage test means a real
mathematical error exists in the feature pipeline, not a test configuration problem.
Do not modify leakage tests to make them pass — fix the code.

---

## What to Never Do

- **Never** add `--no-verify` to git commands
- **Never** shuffle time-series data for train/test splits
- **Never** use accuracy as a model selection criterion
- **Never** remove the `shift(1)` from any feature computation without running leakage tests
- **Never** set the default fill method to MIDPOINT in production simulation contexts
- **Never** call `set_models()` without also calling `save_model()` (in-memory models are lost on restart)
- **Never** join `option_snapshots` to `ohlcv_bars` using `snapshot_time > bar_open_time`
- **Never** commit secrets, API keys, or `.env` files
- **Never** disable the kill switch check in the inference hot path

---

## How to Think About Changes

### Adding a new feature
1. Add to `registry.py` with full metadata (formula, expected range, null strategy, version)
2. Add the computation in `compute.py` using `.shift(1)` on all OHLCV series
3. Add the feature name to `FEATURE_COLS` (or `OPTIONS_FEATURE_COLS` if options-derived)
4. Increment `PIPELINE_VERSION` in `registry.py`
5. Run leakage tests, then retrain

### Adding a new model type
1. Implement the model class in `ml_models/baseline.py` (fit, predict_proba, feature_importance)
2. Add to the training pipeline in `ml_models/training/trainer.py`
3. Register in the governance model registry after training
4. Do not replace the existing logistic baseline — add alongside it

### Adding a new data provider
1. Implement the Protocol from `app/providers/protocols.py`
2. Add a null stub in `null_providers.py`
3. Add an integration test that runs against the null stub
4. Wire into the router via the provider interface, not directly

### Adding a new endpoint
1. Add to the appropriate router in `app/api/routes/`
2. Add a Pydantic schema in `app/schemas/`
3. Wire the router in `main.py`
4. Add the corresponding TypeScript type and API function in `frontend/src/lib/api.ts`

---

## Key File Locations

| Purpose | Path |
|---------|------|
| Feature manifest | `app/feature_pipeline/registry.py` |
| Leakage tests | `tests/test_leakage.py` |
| Inference entry | `app/inference/inference_service.py` |
| Decision engine | `app/decision/decision_engine.py` |
| Regime detector | `app/regime/detector.py` |
| Governance models | `app/governance/models.py` |
| Kill switch | `app/governance/kill_switch.py` |
| Paper simulator | `app/paper_trading/options_simulator/simulator.py` |
| False confidence audit | `backend/FALSE_CONFIDENCE_AUDIT.md` |
| Leakage audit | `backend/docs/LEAKAGE_AUDIT.md` |
| Data lineage | `backend/docs/DATA_LINEAGE.md` |
| Simulator limits | `backend/docs/simulator_limitations.md` |
| Feature docs | `backend/docs/FEATURES.md` |
| Governance runbook | `backend/docs/governance.md` |

---

## Current State Summary (as of April 2026)

**Implemented and tested:** Feature pipeline (30 features), 4-layer inference, 8-regime detection,
governance suite, paper trading simulator, decision engine, IV analysis, structure evaluation,
leakage audit (L1–L9 clean), false confidence audit (partially addressed).

**Wired but fragile:** Model persistence (pkl saved after backtest), outcome back-fill (manual),
WebSocket (polling, not streaming), yfinance data (15-min delayed, no persistence across restarts
for in-flight data).

**Not yet implemented:** Scheduled retraining, live market data, real IV surface (skew/term
structure), earnings calendar integration, automated outcome recording, authentication, CI/CD,
observability metrics, multi-symbol model differentiation.

**Known limitations documented:** See `FALSE_CONFIDENCE_AUDIT.md` and `simulator_limitations.md`.
Do not paper over these — they exist to set honest expectations.
