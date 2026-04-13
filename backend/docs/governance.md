# Model Governance — Operator Guide

**System:** Live Trading Predictor v2.0.0
**Module:** `app.governance`
**Last updated:** 2025-04-11

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Database Tables](#2-database-tables)
3. [Configuration](#3-configuration)
4. [API Endpoints](#4-api-endpoints)
5. [Alert Types and Severity Matrix](#5-alert-types-and-severity-matrix)
6. [Kill Switch Procedure](#6-kill-switch-procedure)
7. [Drift Monitoring — PSI Thresholds](#7-drift-monitoring--psi-thresholds)
8. [Calibration Health Thresholds](#8-calibration-health-thresholds)
9. [Data Freshness Thresholds](#9-data-freshness-thresholds)
10. [Known Limitations](#10-known-limitations)
11. [Skeptical Reviewer Checklist](#11-skeptical-reviewer-checklist)
12. [Runbook — Common Incidents](#12-runbook--common-incidents)

---

## 1. Architecture Overview

The governance module is a self-contained observability layer that sits alongside (not inside) the inference pipeline. It records what the model decided, why it decided it, and whether those decisions held up over time.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Inference API  (/api/inference)                                    │
│    run_inference() → InferenceResult                                │
│    ↓  (API route logs immediately after calling run_inference)      │
│  InferenceLogService.log_inference_result()  ─────→ inference_events│
└─────────────────────────────────────────────────────────────────────┘
              ↓ (periodic / on-demand)
┌─────────────────────────────────────────────────────────────────────┐
│  Governance Services                                                │
│                                                                     │
│  ModelRegistryService    → model_versions                           │
│  FeatureRegistryService  → feature_versions                         │
│  DriftMonitor            → drift_snapshots                          │
│  CalibrationMonitor      → calibration_snapshots                    │
│  DataFreshnessService    → data_freshness_checks                    │
│  GovernanceAlertService  → governance_alerts                        │
│  KillSwitchService       → kill_switch_state                        │
│                                                                     │
│  GovernanceDashboard     (aggregates all of the above)              │
└─────────────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  REST API  /api/governance/*  (32 endpoints)                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Key design decisions

| Decision | Rationale |
|---|---|
| Governance is **append-only** for logs | Inference events and drift snapshots are never mutated after write. Outcome back-fill uses a separate column (`actual_outcome`), not an update to the original prediction row. |
| Inference log is **non-blocking** | `log_inference_result()` is called from the API route *after* the response is formed. A logging failure never fails a trade decision. |
| Kill switch has **two layers** | Env-var `KILL_SWITCH=true` bypasses the DB entirely (safe for config-management emergencies). The DB-backed state survives restarts and supports dynamic toggle without redeploy. |
| Alert **deduplication** | Repeated conditions bump `triggered_at` on the existing row; they don't spam new rows. This keeps alert tables readable. |
| Services are **stateless** | All services are static method collections. State lives in the DB. This makes unit testing straightforward. |

---

## 2. Database Tables

All tables are created by Alembic migration `002_governance`. Run `alembic upgrade 002_governance` to apply.

### 2.1 `model_versions`

Tracks every trained model artefact with its lifecycle status.

| Column | Type | Notes |
|---|---|---|
| `id` | Integer PK | Auto-increment |
| `model_name` | String(64) | e.g. `logistic`, `gbt`, `random_forest` |
| `version_tag` | String(64) | Auto-assigned `v1.0.0`, `v2.0.0`, … if omitted |
| `status` | String(32) | `staging` → `active` → `deprecated` |
| `trained_at` | DateTime | When training completed |
| `training_symbol` | String(16) | Symbol used for training (null = cross-symbol) |
| `n_samples` | Integer | Training set size |
| `n_features` | Integer | Feature count at training time |
| `feature_manifest_hash` | String(64) | FK-by-convention to `feature_versions.manifest_hash` |
| `train_metrics_json` | Text | Arbitrary JSON: accuracy, AUC, F1, etc. |
| `calibration_kind` | String(32) | e.g. `platt`, `isotonic`, `none` |
| `calibration_ece_at_fit` | Float | ECE measured immediately after calibration fit |
| `artifact_dir` | String(256) | Relative path under `model_artifacts/` |
| `artifact_sha256` | String(64) | SHA-256 of `model.pkl`; auto-computed on register |
| `promoted_at` | DateTime | When status changed to `active` |
| `deprecated_at` | DateTime | When status changed to `deprecated` |
| `notes` | Text | Free-form operator notes |

**Constraints:** `UNIQUE(model_name, version_tag)`. Only one row per `model_name` may have `status = active` at any time (enforced by `ModelRegistryService.promote()`).

**Indexes:** `model_name`, `status`, `feature_manifest_hash`

---

### 2.2 `feature_versions`

Records every distinct set of features seen in production. The `manifest_hash` is the primary identifier.

| Column | Type | Notes |
|---|---|---|
| `manifest_hash` | String(64) UNIQUE | SHA-256 of sorted feature names list |
| `pipeline_version` | Integer | Monotone pipeline version counter |
| `feature_count` | Integer | |
| `feature_list_json` | Text | JSON array of feature names |
| `reference_stats_json` | Text | JSON object: `{feature: {percentiles: [...], min, max, mean, std}}` |
| `recorded_at` | DateTime | |
| `description` | Text | Optional notes |

`ensure_manifest()` is idempotent — safe to call on every startup or training run.

---

### 2.3 `inference_events`

The core audit log. One row per inference call.

| Column | Type | Notes |
|---|---|---|
| `id` | BigInteger PK | |
| `request_id` | String(32) | X-Request-ID from HTTP middleware |
| `symbol` | String(16) | |
| `bar_open_time` | DateTime | Bar timestamp the model was evaluated on |
| `inference_ts` | BigInteger | Unix ms timestamp of the inference call |
| `model_name` | String(64) | |
| `model_version_id` | Integer | FK to `model_versions.id` (nullable) |
| `manifest_hash` | String(64) | Feature manifest used |
| `prob_up` | Float | Raw model output |
| `prob_down` | Float | |
| `calibrated_prob_up` | Float | Post-calibration probability |
| `calibration_available` | Boolean | |
| `tradeable_confidence` | Float | After degradation factor applied |
| `degradation_factor` | Float | 1.0 = healthy, lower = degraded |
| `action` | String(16) | `buy`, `sell`, `abstain` |
| `abstain_reason` | String(128) | Structured reason string when action=abstain |
| `calibration_health` | String(16) | `good`, `caution`, `degraded` |
| `ece_recent` | Float | ECE at inference time |
| `rolling_brier` | Float | |
| `expected_move_pct` | Float | |
| `regime` | String(32) | Market regime label |
| `options_stale` | Boolean | Whether options chain was stale |
| `actual_outcome` | SmallInteger | 1=up, 0=down, NULL=pending |
| `outcome_recorded_at` | DateTime | |
| `created_at` | DateTime | |

**Indexes:** `symbol`, `action`, `created_at`, compound `(symbol, created_at)`, compound `(symbol, bar_open_time)`

---

### 2.4 `drift_snapshots`

PSI-based feature drift snapshots. Typically computed every N bars or on a schedule.

| Column | Type | Notes |
|---|---|---|
| `symbol` | String(16) | |
| `computed_at` | DateTime | |
| `window_bars` | Integer | How many recent bars were used |
| `manifest_hash` | String(64) | Feature version used |
| `psi_by_feature_json` | Text | JSON: `{feature_name: psi_value, …}` |
| `max_psi` | Float | Highest single-feature PSI |
| `mean_psi` | Float | |
| `high_drift_features_json` | Text | JSON list of features with PSI > 0.10 |
| `drift_level` | String(16) | `none`, `moderate`, `high` |
| `alert_raised` | Boolean | Whether a governance alert was fired |

---

### 2.5 `calibration_snapshots`

Rolling calibration health measurements.

| Column | Type | Notes |
|---|---|---|
| `symbol` | String(16) | |
| `snapshot_at` | DateTime | |
| `model_name` | String(64) | |
| `window_size` | Integer | Number of resolved inferences in window |
| `rolling_brier` | Float | Lower is better (0.25 = random) |
| `baseline_brier` | Float | Brier score at fit time |
| `degradation_factor` | Float | `baseline / rolling`; < 1.0 means degraded |
| `ece_recent` | Float | Expected Calibration Error |
| `calibration_health` | String(16) | `good`, `caution`, `degraded` |
| `needs_retrain` | Boolean | |
| `retrain_reason` | Text | |
| `reliability_json` | Text | Reliability diagram bins as JSON |

---

### 2.6 `data_freshness_checks`

Point-in-time freshness checks for each data source.

| Column | Type | Notes |
|---|---|---|
| `symbol` | String(16) | |
| `source` | String(64) | `quote_feed`, `candle_data`, `options_chain`, `model_artifact` |
| `checked_at` | DateTime | |
| `last_data_ts` | DateTime | Most recent data point timestamp; NULL = no data |
| `age_seconds` | Float | `checked_at - last_data_ts` in seconds |
| `is_stale` | Boolean | `age_seconds > staleness_threshold_seconds` |
| `staleness_threshold_seconds` | Float | Threshold used at check time |
| `alert_raised` | Boolean | |

---

### 2.7 `governance_alerts`

Active and historical alert records.

| Column | Type | Notes |
|---|---|---|
| `alert_type` | String(64) | See §5 |
| `severity` | String(16) | `info`, `warning`, `critical` |
| `symbol` | String(16) | Nullable (system-wide alerts have no symbol) |
| `title` | String(256) | Short human-readable description |
| `details_json` | Text | Structured payload; content varies by alert type |
| `triggered_at` | DateTime | Most recent trigger time (bumped on dedup) |
| `expires_at` | DateTime | Nullable; when set, alert auto-clears after this time |
| `acknowledged_at` | DateTime | Nullable |
| `acknowledged_by` | String(64) | Nullable |
| `dedup_key` | String(128) | Unique active-alert key; format: `{alert_type}:{symbol}` |
| `is_active` | Boolean | False after acknowledge or expiry |

**Deduplication:** If an active row with the same `dedup_key` exists, `raise_alert()` bumps `triggered_at` and returns the existing row. No duplicate rows are created.

---

### 2.8 `kill_switch_state`

Singleton table. Always contains exactly one row (`id = 1`).

| Column | Type | Notes |
|---|---|---|
| `active` | Boolean | Whether trading is halted |
| `reason` | Text | Operator-supplied reason |
| `activated_at` | DateTime | When switch was last activated |
| `activated_by` | String(64) | User/process identifier |
| `updated_at` | DateTime | Last write time |

---

## 3. Configuration

Environment variables relevant to the governance module:

| Variable | Default | Description |
|---|---|---|
| `KILL_SWITCH` | `false` | Hard override — if `"true"`, `is_active_cached()` returns `True` without a DB query. Takes precedence over DB state. |
| `DATABASE_URL` | required | SQLAlchemy async DSN, e.g. `postgresql+asyncpg://user:pass@host/db` |

No other configuration is read from environment for governance services. Thresholds (PSI, calibration, freshness) are defined as constants in the respective service modules and are intentionally code-reviewed rather than runtime-configured, to prevent silent threshold drift.

---

## 4. API Endpoints

All endpoints are under `/api/governance`. Authentication is not enforced in this version — deploy behind a reverse proxy or internal network boundary.

### 4.1 Dashboard

| Method | Path | Description |
|---|---|---|
| `GET` | `/summary` | Full governance health summary (kill switch, active model, calibration, drift, stale feeds, alert counts, 24h inference volume) |
| `GET` | `/performance/{symbol}` | Rolling performance for a symbol: accuracy stats + calibration trend + latest drift |

`GET /summary` response shape:
```json
{
  "kill_switch_active": false,
  "active_model": { "model_name": "logistic", "version_tag": "v3.0.0", "status": "active" },
  "calibration_health": "good",
  "symbols_needing_retrain": [],
  "drift_summary": { "SPY": "none", "QQQ": "moderate" },
  "stale_feeds_count": 0,
  "active_alerts_count": 1,
  "critical_alerts_count": 0,
  "inference_volume_24h": { "SPY": { "buy": 12, "sell": 4, "abstain": 8 } }
}
```

Each subsystem is queried independently. A failure in one subsystem (e.g., no calibration snapshots yet) returns `null` for that field rather than failing the entire summary.

---

### 4.2 Model Registry

| Method | Path | Description |
|---|---|---|
| `POST` | `/models/register` | Register a new model version (status=staging) |
| `POST` | `/models/{version_id}/promote` | Promote to active; atomically deprecates current active |
| `POST` | `/models/{version_id}/deprecate` | Deprecate a specific version |
| `GET` | `/models/{model_name}/active` | Get the currently active version |
| `GET` | `/models/{model_name}/versions` | List versions (filterable by status, paginated) |

`POST /models/register` body:
```json
{
  "model_name": "logistic",
  "version_tag": "v3.0.0",        // optional; auto-assigned if omitted
  "trained_at": "2025-04-11T10:00:00",
  "training_symbol": "SPY",
  "n_samples": 4800,
  "n_features": 47,
  "feature_manifest_hash": "abc123...",
  "train_metrics_json": "{\"auc\": 0.61, \"accuracy\": 0.58}",
  "calibration_kind": "platt",
  "calibration_ece_at_fit": 0.028,
  "artifact_dir": "logistic_v3",
  "notes": "Retrained with 6 months extra data"
}
```

`POST /models/{version_id}/promote` body:
```json
{ "notes": "Approved after shadow run 2025-04-11" }
```

---

### 4.3 Feature Registry

| Method | Path | Description |
|---|---|---|
| `POST` | `/features/register` | Register a feature manifest (idempotent) |
| `GET` | `/features/{manifest_hash}` | Get a specific manifest |
| `GET` | `/features` | List all manifests (paginated) |
| `POST` | `/features/diff` | Diff two manifests — returns added/removed/version_bumped |

`POST /features/diff` body:
```json
{ "hash_a": "abc123...", "hash_b": "def456..." }
```

---

### 4.4 Inference Log

| Method | Path | Description |
|---|---|---|
| `GET` | `/inference` | Query inference events (symbol, action, time range, pending_only, pagination) |
| `POST` | `/inference/{event_id}/outcome` | Back-fill actual outcome for a specific event |
| `POST` | `/inference/outcomes/bulk` | Back-fill outcomes for all pending events at a given bar time |
| `GET` | `/inference/stats/{symbol}` | Accuracy stats for a symbol over a rolling window |
| `GET` | `/inference/volume/{symbol}` | Action counts for the last 24 hours |

`GET /inference` query parameters:
- `symbol` — filter by symbol (required)
- `action` — `buy`, `sell`, `abstain` (optional)
- `from_ts` — ISO datetime lower bound
- `to_ts` — ISO datetime upper bound
- `pending_only` — `true` to return only rows without actual_outcome
- `limit` — default 50, max 500
- `offset` — for pagination

`POST /inference/outcomes/bulk` body:
```json
{
  "symbol": "SPY",
  "bar_open_time": "2025-04-11T14:30:00",
  "actual_outcome": 1   // 1=up, 0=down
}
```

---

### 4.5 Drift Monitoring

| Method | Path | Description |
|---|---|---|
| `POST` | `/drift/run` | Compute PSI for a symbol over recent bars; store snapshot; raise alert if needed |
| `GET` | `/drift/{symbol}/latest` | Most recent drift snapshot |
| `GET` | `/drift/summary` | Latest drift level per symbol |

`POST /drift/run` body:
```json
{
  "symbol": "SPY",
  "window_bars": 100,
  "manifest_hash": "abc123..."  // optional; uses latest if omitted
}
```

The endpoint queries the most recent `window_bars` rows from `feature_rows` (the live feature store table), builds a feature matrix, computes PSI against the reference distribution stored in `feature_versions.reference_stats_json`, and saves a `drift_snapshots` row. If `drift_level` is `moderate` or `high`, a `GovernanceAlert` is raised.

---

### 4.6 Calibration Monitoring

| Method | Path | Description |
|---|---|---|
| `POST` | `/calibration/{symbol}/snapshot` | Take a calibration snapshot for a symbol |
| `GET` | `/calibration/{symbol}/latest` | Most recent snapshot |
| `GET` | `/calibration/retrain-needed` | List symbols flagged for retrain |

`POST /calibration/{symbol}/snapshot` reads the live `ConfidenceTracker` for the symbol, persists a `calibration_snapshots` row, and raises alerts if `calibration_health == "degraded"` or `needs_retrain == True`.

---

### 4.7 Data Freshness

| Method | Path | Description |
|---|---|---|
| `POST` | `/freshness/check` | Record a freshness check result |
| `GET` | `/freshness/{symbol}` | Current freshness status per source for a symbol |
| `GET` | `/freshness/stale` | All stale feeds checked in the last 15 minutes |

`POST /freshness/check` body:
```json
{
  "symbol": "SPY",
  "source": "quote_feed",
  "last_data_ts": "2025-04-11T15:29:55",
  "override_threshold_seconds": null  // use default if null
}
```

---

### 4.8 Alerts

| Method | Path | Description |
|---|---|---|
| `GET` | `/alerts` | List alerts (active_only, severity, symbol, limit) |
| `POST` | `/alerts/{alert_id}/acknowledge` | Acknowledge an alert |
| `POST` | `/alerts/clear-expired` | Sweep expired alerts (mark is_active=False) |

`POST /alerts/{alert_id}/acknowledge` body:
```json
{ "by": "ops-engineer-name" }
```

---

### 4.9 Kill Switch

| Method | Path | Description |
|---|---|---|
| `GET` | `/kill-switch` | Current kill switch state |
| `POST` | `/kill-switch/activate` | Halt trading |
| `POST` | `/kill-switch/deactivate` | Resume trading |

`POST /kill-switch/activate` body:
```json
{
  "reason": "Unexpected vol spike — manual halt",
  "by": "ops-engineer-name"
}
```

---

## 5. Alert Types and Severity Matrix

| Alert Type | Constant | Default Severity | Dedup Key | Auto-expires |
|---|---|---|---|---|
| Feed stale | `feed_stale` | `warning` | `feed_stale:{symbol}` | 1 hour |
| Drift moderate | `drift_moderate` | `warning` | `drift_moderate:{symbol}` | 24 hours |
| Drift high | `drift_high` | `critical` | `drift_high:{symbol}` | 24 hours |
| Calibration degraded | `calibration_degraded` | `warning` | `calibration_degraded:{symbol}` | 24 hours |
| Retrain needed | `retrain_needed` | `warning` | `retrain_needed:{symbol}` | 72 hours |
| Kill switch activated | `kill_switch_activated` | `critical` | none (always new row) | never |
| Kill switch deactivated | `kill_switch_deactivated` | `info` | none (always new row) | 24 hours |
| Model promoted | `model_promoted` | `info` | none (always new row) | 24 hours |
| Model deprecated | `model_deprecated` | `info` | none (always new row) | 24 hours |
| Risk breach | `risk_breach` | `critical` | `risk_breach:{symbol}` | 4 hours |
| Inference error spike | `inference_error_spike` | `critical` | `inference_error_spike:{symbol}` | 1 hour |

**Severity definitions:**

- `info` — Informational. No action required. Useful for audit trail.
- `warning` — Degraded state. Investigate within the trading session.
- `critical` — Trading integrity at risk. Requires immediate operator review. Consider activating kill switch.

---

## 6. Kill Switch Procedure

### Activating (halting trading)

```bash
curl -X POST http://localhost:8000/api/governance/kill-switch/activate \
  -H "Content-Type: application/json" \
  -d '{"reason": "Unexpected drawdown — manual halt pending review", "by": "ops"}'
```

Or set the environment variable for an immediate hot-path override without a DB round-trip:

```bash
# In your container environment / systemd unit:
KILL_SWITCH=true
# Then restart or send SIGHUP (depends on your deployment)
```

The env-var method is authoritative — it overrides DB state. Use it for emergency response when the DB might itself be degraded.

### Deactivating (resuming trading)

```bash
curl -X POST http://localhost:8000/api/governance/kill-switch/deactivate \
  -H "Content-Type: application/json" \
  -d '{"reason": "Manual review complete — no model issues found", "by": "ops"}'
```

Then, if using the env-var override, remove `KILL_SWITCH=true` from the environment and redeploy or restart.

### How the kill switch is checked during inference

`KillSwitchService.is_active_cached()` is the hot-path method. It:
1. Checks `settings.kill_switch` (env var, no I/O)
2. If the module-level TTL cache is fresh (< 5 seconds old), returns the cached value
3. Otherwise, falls back to False (optimistic) — the next DB-backed check at the inference route level will be authoritative

The intent is: the kill switch **stops trading reliably** when activated, but does not add DB latency to every inference when inactive.

### Audit trail

Every activate/deactivate call:
- Writes to `kill_switch_state` (updated_at, activated_by, reason)
- Raises a `kill_switch_activated` or `kill_switch_deactivated` governance alert (these are not deduplicated — each toggle creates its own alert row for a complete audit trail)

---

## 7. Drift Monitoring — PSI Thresholds

Population Stability Index (PSI) measures how much the distribution of a feature has shifted relative to the training reference distribution.

### Formula

```
PSI = Σ (actual_pct_i - expected_pct_i) × ln(actual_pct_i / expected_pct_i)
```

Where bins are computed over the feature's value range, `expected` is from the training reference, and `actual` is from the recent production window. An epsilon of `1e-4` is added to each bin proportion to avoid `log(0)`.

Returns `NaN` if fewer than 30 samples are present — this is treated as "insufficient data" and does not raise an alert.

### Threshold interpretation

| PSI | Drift Level | Interpretation | Recommended Action |
|---|---|---|---|
| < 0.10 | `none` | Distribution is stable | No action |
| 0.10 – 0.25 | `moderate` | Noticeable shift | Monitor more frequently; review high-drift features |
| > 0.25 | `high` | Major distribution shift | Investigate data pipeline; consider retraining |

### Reference distribution sources (priority order)

1. **Stored training percentiles** (`feature_versions.reference_stats_json`) — most accurate; set at training time
2. **Uniform bins using `expected_min`/`expected_max`** — less accurate but deterministic
3. **Gaussian approximation from current window** — fallback of last resort; PSI will always be low; document this limitation in your review

To populate accurate reference stats at training time:
```python
reference_stats = {
    feature_name: {
        "percentiles": np.percentile(X_train[feature_name], np.linspace(0, 100, 11)).tolist(),
        "min": float(X_train[feature_name].min()),
        "max": float(X_train[feature_name].max()),
        "mean": float(X_train[feature_name].mean()),
        "std": float(X_train[feature_name].std()),
    }
    for feature_name in feature_names
}
await FeatureRegistryService.ensure_manifest(db, reference_stats_json=json.dumps(reference_stats), ...)
```

### What PSI does not catch

- **Label drift** (the actual market outcome distribution changing) — monitored via calibration snapshots, not PSI
- **Correlation structure changes** — PSI is per-feature marginal only
- **Concept drift** — requires direct accuracy tracking in `inference_events`

---

## 8. Calibration Health Thresholds

Calibration health is reported per symbol based on the most recent `calibration_snapshots` row.

### Degradation factor

```
degradation_factor = baseline_brier / rolling_brier
```

Where `baseline_brier` is the Brier score measured at model fit time and `rolling_brier` is the rolling average over recent resolved inferences.

| Degradation Factor | Health | Meaning |
|---|---|---|
| ≥ 0.85 | `good` | Rolling performance near training baseline |
| 0.60 – 0.85 | `caution` | Moderate degradation — monitor; reduce position sizing |
| < 0.60 | `degraded` | Significant degradation — abstain from trading; retrain |

### Expected Calibration Error (ECE)

| ECE | Interpretation |
|---|---|
| < 0.05 | Well calibrated |
| 0.05 – 0.10 | Acceptable |
| > 0.10 | Recalibration needed |

### Trend direction

`CalibrationMonitor.trend_direction()` fits a linear regression to the `rolling_brier` series across the most recent snapshots:

- Slope > +0.002 per snapshot → `degrading`
- Slope < −0.002 per snapshot → `improving`
- Otherwise → `stable`

### Retrain triggers

`needs_retrain = True` is set in a calibration snapshot when any of:
- `calibration_health == "degraded"` (degradation factor < 0.60)
- `ece_recent > 0.15`
- `rolling_brier > baseline_brier * 1.5` (50% worse than baseline)

The `retrain_reason` column records which condition triggered it.

### Window size warning

Calibration metrics are unreliable with fewer than 40 resolved inferences in the window. The frontend ModelHealthPanel displays a warning when `window_size < 40`. Governance alerts are not raised for small windows — only the UI warns.

---

## 9. Data Freshness Thresholds

Defined in `app/governance/freshness.py`:

| Source Key | Threshold | Rationale |
|---|---|---|
| `quote_feed` | 60 seconds | Real-time equity quotes; >1 min = stale |
| `candle_data` | 600 seconds (10 min) | Bar data; one missed bar is an issue |
| `options_chain` | 3600 seconds (1 hour) | Options data refreshes less frequently |
| `model_artifact` | 604800 seconds (7 days) | Model should be retrained at least weekly |

**How checks are triggered:** Callers (inference route, scheduler, health checks) call `DataFreshnessService.record_check()` at appropriate points. The service does not autonomously poll data sources — it records the result of checks performed by the caller.

**Stale feed alerts** auto-expire after 1 hour. If the feed is still stale when the sweep runs, the alert is re-raised (dedup bumps the `triggered_at`).

---

## 10. Known Limitations

### PSI reference distribution quality

If `reference_stats_json` is null in the `feature_versions` row (e.g., the manifest was registered without training statistics), PSI computation falls back to a Gaussian approximation derived from the current production window. In this case:
- PSI will typically be near zero regardless of actual drift
- This will not raise drift alerts even if the distribution has shifted
- **Detection:** Check `feature_versions.reference_stats_json` is populated for active manifests

### Survivorship bias in accuracy stats

`get_accuracy_stats()` only counts inferences where `actual_outcome` is not null. If outcome back-fill is delayed or partial (e.g., market data is unavailable for some bars), the accuracy numbers will be optimistic. Operators should monitor `n_pending` in the accuracy stats response to assess completeness.

### Calibration monitor requires actual outcomes

The rolling Brier score requires resolved inferences. In a paper-trading context with delayed or manual outcome recording, the calibration window will lag reality. A `window_size` of 0 produces no health signal.

### Kill switch hot-path cache is optimistic

If the DB kill switch is activated while a service instance has a warm 5-second cache, that instance may process up to 5 more seconds of trades before picking up the activation. This is a deliberate latency vs. correctness trade-off. For hard emergencies, use `KILL_SWITCH=true` env-var + restart.

### No multi-symbol model differentiation

The model registry stores `training_symbol` per model, but the inference log's `model_name` is a free-form string. There is no enforced FK relationship between `inference_events.model_version_id` and `model_versions.id`. This means registry and inference log can drift if callers don't populate `model_version_id`.

### PSI is marginal-only

PSI measures each feature's distribution independently. Joint distribution shifts (e.g., two features moving in opposite directions while marginals stay stable) are not detected.

---

## 11. Skeptical Reviewer Checklist

This section maps each common governance failure mode to the control that addresses it.

| Failure Mode | Risk | Control | Where to Verify |
|---|---|---|---|
| Model is stale — never retrained after data regime change | Predictions based on outdated patterns | `calibration_snapshots.needs_retrain` flag; `model_versions.trained_at` audit trail | `GET /api/governance/calibration/retrain-needed` |
| Feature pipeline silently changes — model receives different features than it was trained on | Prediction accuracy degrades without any alert | `feature_versions.manifest_hash` compared between training and inference; `DriftMonitor` computes PSI vs training reference | `GET /api/governance/features/diff` on any two manifest hashes |
| Calibration probabilities are overconfident after regime change | Oversized positions relative to true edge | `CalibrationMonitor` computes ECE, degradation factor, and trend direction; `needs_retrain` flag; `caution`/`degraded` health states reduce `tradeable_confidence` in inference | `GET /api/governance/calibration/{symbol}/latest` |
| Stale market data feeds used for inference | Predictions based on hours-old prices | `DataFreshnessService` records check timestamps; stale sources → `options_stale` flag on inference event; frontend `inst-panel-stale` visual indicator | `GET /api/governance/freshness/{symbol}` |
| System continues trading despite known degradation | Real losses accrue while model is known bad | Kill switch: two-layer (env-var + DB); `KillSwitchService.is_active_cached()` called on every inference hot path | `GET /api/governance/kill-switch`; audit `governance_alerts` for `kill_switch_activated` rows |
| Operator activates/deactivates kill switch without leaving a record | No accountability; can't reconstruct timeline | Every toggle creates a new (non-deduplicated) `governance_alerts` row + updates `kill_switch_state.activated_by` | `GET /api/governance/alerts?active_only=false` |
| Gradual calibration degradation goes unnoticed | Model worsens over weeks without triggering hard threshold | `trend_direction()` uses linear regression over snapshot history; `degrading` trend raises alert before hard threshold is breached | `GET /api/governance/performance/{symbol}` |
| Old model version silently continues running after a new one is registered | Predictions from an outdated model | `ModelRegistryService.promote()` atomically deprecates the prior active version; inference events log `model_version_id` | `GET /api/governance/models/{model_name}/versions` |
| Duplicate alerts spam the alert table | Operators miss real signals in noise | `raise_alert()` deduplication: `dedup_key` prevents duplicate rows; bumps `triggered_at` instead | `GET /api/governance/alerts` — check row count vs trigger count |
| Inference logging fails silently, leaving gaps in audit trail | Can't reconstruct what the model decided | Log call is in the API route (not inside `run_inference()`); failures are logged at ERROR level; `request_id` correlates HTTP logs with inference events | Cross-reference Nginx/app logs `req_id` with `inference_events.request_id` |

### Audit queries

To verify governance is functioning, a skeptical reviewer should run:

```sql
-- 1. Is there exactly one active model per model_name?
SELECT model_name, COUNT(*) as active_count
FROM model_versions
WHERE status = 'active'
GROUP BY model_name
HAVING COUNT(*) != 1;
-- Expected: 0 rows

-- 2. Are recent inferences being logged?
SELECT DATE(created_at), COUNT(*)
FROM inference_events
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY 1 ORDER BY 1 DESC;

-- 3. Are outcomes being back-filled?
SELECT
    COUNT(*) FILTER (WHERE actual_outcome IS NULL) as pending,
    COUNT(*) FILTER (WHERE actual_outcome IS NOT NULL) as resolved,
    ROUND(100.0 * COUNT(*) FILTER (WHERE actual_outcome IS NOT NULL) / COUNT(*), 1) as pct_resolved
FROM inference_events
WHERE created_at > NOW() - INTERVAL '7 days';

-- 4. Are all active alerts acknowledged within SLA?
SELECT alert_type, severity, symbol, triggered_at, acknowledged_at,
    EXTRACT(EPOCH FROM (COALESCE(acknowledged_at, NOW()) - triggered_at))/3600 as hours_open
FROM governance_alerts
WHERE severity = 'critical'
ORDER BY triggered_at DESC
LIMIT 20;

-- 5. Is the feature manifest consistent between training and recent inference?
SELECT DISTINCT ie.manifest_hash, fv.id as fv_exists
FROM inference_events ie
LEFT JOIN feature_versions fv ON fv.manifest_hash = ie.manifest_hash
WHERE ie.created_at > NOW() - INTERVAL '24 hours'
AND fv.id IS NULL;
-- Expected: 0 rows (all manifest hashes are registered)
```

---

## 12. Runbook — Common Incidents

### INC-001: `drift_high` alert raised

1. `GET /api/governance/drift/{symbol}/latest` — identify which features have high PSI
2. `GET /api/governance/features/{manifest_hash}` — confirm reference stats are present
3. If reference stats are null, PSI is unreliable — investigate separately
4. If reference stats are present and PSI > 0.25: inspect the feature pipeline for upstream data changes
5. If drift is genuine: activate kill switch, retrain model on recent data, register new model version, promote, deactivate kill switch

### INC-002: `retrain_needed` alert raised

1. `GET /api/governance/calibration/{symbol}/latest` — review rolling_brier, ece_recent, retrain_reason
2. Assess whether degradation is symbol-specific or system-wide
3. Schedule retraining. Until retrained: the `tradeable_confidence` degradation factor automatically reduces trade signals
4. After retraining: `POST /api/governance/models/register` → `POST /api/governance/models/{id}/promote`
5. Acknowledge the alert: `POST /api/governance/alerts/{id}/acknowledge`

### INC-003: Feed stale alert (`feed_stale`)

1. `GET /api/governance/freshness/{symbol}` — identify which source is stale
2. Investigate upstream data provider connectivity
3. `quote_feed` stale > 5 minutes: seriously consider activating kill switch
4. `options_chain` stale: inference will still run but `options_stale = true` on inference events; options-based signals will be absent
5. After feed recovers, the next freshness check will auto-clear (dedup key expires after 1 hour if not re-triggered)

### INC-004: Kill switch activated unexpectedly

1. `GET /api/governance/kill-switch` — check `activated_by` and `reason`
2. Check `KILL_SWITCH` environment variable — if set to `"true"`, the env-var is overriding; the DB deactivate call alone is insufficient
3. Review recent governance alerts for the root cause
4. If the activation was automated (e.g., from a risk breach), resolve the underlying condition before deactivating
5. Deactivate: `POST /api/governance/kill-switch/deactivate` with your name in the `by` field
