# DATA_LINEAGE.md

Point-in-time data model for the options-research platform.

This document is authoritative.  Code that touches data ingestion, feature
engineering, training set construction, or backtesting must comply with the
invariants described here.  When code and documentation conflict, treat the
code as potentially wrong and file an issue before changing this document.

---

## Table of contents

1. [Temporal model](#1-temporal-model)
2. [Tables and their timestamp semantics](#2-tables-and-their-timestamp-semantics)
3. [Point-in-time query patterns](#3-point-in-time-query-patterns)
4. [Idempotency semantics](#4-idempotency-semantics)
5. [Revision workflow](#5-revision-workflow)
6. [Ingest batch rollback](#6-ingest-batch-rollback)
7. [Lookahead guard for options joins (L7)](#7-lookahead-guard-for-options-joins-l7)
8. [Building a correct training set](#8-building-a-correct-training-set)
9. [Research snapshots (reproducibility)](#9-research-snapshots-reproducibility)
10. [Data source registry](#10-data-source-registry)
11. [Governance integration](#11-governance-integration)
12. [Known limitations](#12-known-limitations)
13. [Schema diagram](#13-schema-diagram)
14. [Legacy table reference](#14-legacy-table-reference)
15. [Audit query examples](#15-audit-query-examples)

---

## 1. Temporal model

Every row in the point-in-time data layer carries **three distinct timestamps**.
Confusing them produces either lookahead bias or overconstrained training sets.

```
event_time    When the market event occurred.
              Bar: UTC bar open time.
              Options: exchange quote timestamp (when known).
              Ground truth — immutable even after corrections.

available_at  When data became available to downstream consumers.
              The formula is:
                available_at = event_time + bar_duration + source.typical_delay_s
              Example — yfinance 5m bar at 09:30 ET (14:30 UTC):
                event_time   = 2024-01-02T14:30:00Z   (bar opens)
                bar_duration = 300s  (5 min)
                delay        = 900s  (yfinance 15-min lag)
                available_at = 2024-01-02T14:50:00Z
              For real-time feeds: ≈ event_time + bar_duration.
              For historical backfills: ingested_at (conservative — see §12 L-LIM-1).

ingested_at   Wall-clock time of the DB INSERT.
              Auto-set by the server (server_default=now()).
              Immutable after write. Used for audit and DB-state replay.
```

### Why you must filter on `available_at`, not `ingested_at`

A model trained on data filtered by `ingested_at <= T` may still use data that
was **not available** to a trader at time T — if the data was ingested in a
bulk historical backfill that ran after T, while the bars themselves predate T.

The correct filter for all training/backtest queries is:

```sql
WHERE available_at <= :T AND is_current = TRUE
```

Never substitute `ingested_at` for `available_at` in this position.

---

## 2. Tables and their timestamp semantics

### 2.1 `market_bars` (PIT-correct, added in migration 003)

| Column        | Semantic role | Description |
|---------------|---------------|-------------|
| `event_time`  | Event time    | Bar open time (UTC). Never changed by corrections. |
| `available_at`| Availability  | When consumers may use this bar. See §1 for formula. NULL = unknown; treat as `ingested_at`. |
| `ingested_at` | Ingestion     | Wall-clock DB write time. Auto-set, immutable. |

**`bar_status` values**

| Value        | Meaning |
|--------------|---------|
| `PARTIAL`    | Bar still open — close is latest tick, not the final official close |
| `CLOSED`     | Bar finalized. Canonical state for all historical data. |
| `CORRECTED`  | Superseded by a newer revision. `is_current` = FALSE. |
| `BACKFILLED` | Gap-filled from a secondary source. Quality may be lower. |
| `INVALID`    | Vendor-flagged as bad data. Exclude from training and inference. |

**Revision columns**

| Column          | Meaning |
|-----------------|---------|
| `revision_seq`  | 1 = first write; 2 = first correction; etc. |
| `is_current`    | TRUE on the latest revision only. All read queries filter `WHERE is_current`. |
| `superseded_at` | Wall-clock time this row was replaced. NULL if current. |
| `superseded_by` | FK → replacement `market_bars.id`. NULL if current. |

### 2.2 `option_quotes` (PIT-correct, added in migration 003)

| Column        | Semantic role | Description |
|---------------|---------------|-------------|
| `event_time`  | Event time    | Exchange quote timestamp. NULL for yfinance (not provided). |
| `available_at`| Availability  | When our system received this quote from the source. Use in all join conditions. |
| `ingested_at` | Ingestion     | Wall-clock DB write time. |

Same revision columns as `market_bars`.

### 2.3 `inference_events` (governance layer)

| Column               | Semantic role | Description |
|----------------------|---------------|-------------|
| `bar_open_time`      | Event time    | event_time of the bar that triggered inference |
| `inference_ts`       | Wall clock    | Unix ms — when inference was executed |
| `created_at`         | Ingestion     | DB write time |
| `data_available_at`  | Availability  | max(bar.available_at, options.available_at) of all input data. Added in migration 003. |
| `outcome_quality`    | Data quality  | `final` \| `preliminary` \| `corrected` \| NULL |

### 2.4 `model_versions` (governance layer)

| Column            | Semantic role | Description |
|-------------------|---------------|-------------|
| `trained_at`      | Event time    | When training run completed |
| `promoted_at`     | Transition    | Service-level record of status → 'active' |
| `went_live_at`    | Transition    | DB-recorded timestamp of status → 'active'. Added in migration 003. |
| `went_offline_at` | Transition    | DB-recorded timestamp of status leaving 'active'. Added in migration 003. |

### 2.5 `ohlcv_bars` (legacy — kept for backward compatibility)

| Column              | Equivalent    |
|---------------------|---------------|
| `bar_open_time`     | `event_time`  |
| `availability_time` | `available_at` (nullable; may be NULL for legacy rows) |
| `ingested_at`       | `ingested_at` |
| `staleness_flag`    | True if ingested_at − bar_close_time > one bar duration |

No revision tracking. **Do not write new ingestion code targeting `ohlcv_bars`.**

### 2.6 `option_snapshots` (legacy)

| Column          | Equivalent    |
|-----------------|---------------|
| `snapshot_time` | `available_at`|
| `ingested_at`   | `ingested_at` |

No revision tracking. **Do not write new ingestion code targeting `option_snapshots`.**

---

## 3. Point-in-time query patterns

### 3.1 Current state (live inference and dashboards)

```sql
SELECT symbol, timeframe, event_time, open, high, low, close, volume
FROM   market_bars
WHERE  symbol     = 'SPY'
AND    timeframe  = '5m'
AND    event_time BETWEEN :start AND :end
AND    is_current = TRUE
AND    bar_status = 'CLOSED'
ORDER  BY event_time;
```

### 3.2 Training set boundary — data available as of market time T

The **only** correct way to construct a lookahead-free training set.
`T` = the `event_time` of the target bar being predicted.

```sql
SELECT symbol, timeframe, event_time, open, high, low, close, volume
FROM   market_bars
WHERE  symbol      = 'SPY'
AND    timeframe   = '5m'
AND    available_at <= :T          -- the critical filter
AND    is_current  = TRUE
AND    bar_status  = 'CLOSED'
ORDER  BY event_time;
```

### 3.3 DB-state replay — audit at wall-clock time T_wall

Reconstructs exactly what the database contained at wall-clock time T_wall.
Use for verifying what data a model had access to during a live inference.

```sql
SELECT *
FROM   market_bars
WHERE  symbol      = 'SPY'
AND    timeframe   = '5m'
AND    ingested_at  <= :T_wall
AND    (is_current OR superseded_at > :T_wall)
ORDER  BY event_time, revision_seq;
```

### 3.4 Options join — point-in-time correct (L7)

```sql
SELECT
    mb.event_time,
    mb.close,
    oq.implied_volatility,
    oq.delta,
    oq.iv_rank,
    oq.pc_volume_ratio
FROM  market_bars mb
LEFT JOIN option_quotes oq
    ON  oq.underlying_symbol = mb.symbol
    AND oq.available_at      <= mb.event_time   -- REQUIRED. Must be <=. Never >.
    AND oq.is_current        = TRUE
    AND oq.expiry            = :target_expiry
    AND oq.strike            = :target_strike
    AND oq.option_type       = 'call'
WHERE mb.symbol      = 'SPY'
AND   mb.timeframe   = '5m'
AND   mb.available_at <= :T
AND   mb.is_current   = TRUE
ORDER BY mb.event_time;
```

### 3.5 Nearest prior options snapshot per bar

Picks the most recent quote available before each bar opened:

```sql
SELECT DISTINCT ON (underlying_symbol, expiry, strike, option_type)
    *
FROM  option_quotes
WHERE underlying_symbol = 'SPY'
AND   available_at      <= :bar_event_time   -- PIT boundary
AND   is_current        = TRUE
ORDER BY
    underlying_symbol,
    expiry,
    strike,
    option_type,
    available_at DESC;   -- pick latest quote before the boundary
```

### 3.6 Which model was active at time T?

```sql
SELECT *
FROM   model_versions
WHERE  model_name      = 'logistic'
AND    went_live_at    <= :T
AND    (went_offline_at IS NULL OR went_offline_at > :T);
```

---

## 4. Idempotency semantics

Idempotency keys are encoded as unique constraints.
The ingestion layer handles collisions with `ON CONFLICT DO NOTHING` (skip, increment `rows_skipped`).

| Table           | Idempotency key |
|-----------------|-----------------|
| `market_bars`   | `(symbol, timeframe, event_time, source_id, revision_seq)` |
| `option_quotes` | `(underlying_symbol, expiry, strike, option_type, available_at, source_id, revision_seq)` |

**What "idempotent" means here**: writing the same bar twice is safe — the
second attempt hits the unique constraint and is skipped.

**What it does NOT mean**: writing the same bar with different OHLCV values
is a data error, not an idempotent write.  The ingestion layer must detect
value mismatches and route them through the correction workflow (§5).

---

## 5. Revision workflow

When a data provider issues a correction (split adjustment, dividend, data error):

```
Step 1 — Insert new revision
    INSERT INTO market_bars (
        symbol, timeframe, event_time, source_id,
        revision_seq  = <old_seq + 1>,
        is_current    = TRUE,
        bar_status    = 'CLOSED',   -- or appropriate status
        open, high, low, close, volume, ...
    );

Step 2 — Retire old revision (same transaction)
    UPDATE market_bars
    SET    is_current    = FALSE,
           bar_status    = 'CORRECTED',
           superseded_at = NOW(),
           superseded_by = <new_id>
    WHERE  id            = <old_id>;

Step 3 — Record in correction ledger (same transaction)
    INSERT INTO bar_corrections (
        original_bar_id, replacement_bar_id,
        correction_type, initiated_by, reason, changed_fields_json
    ) VALUES (
        <old_id>, <new_id>,
        'SPLIT_ADJUSTMENT', 'split_processor',
        '2:1 split on 2024-06-10',
        '{"close": {"from": 200.0, "to": 100.0},
          "split_factor": {"from": 1.0, "to": 2.0}}'
    );
```

All three steps must run inside a single database transaction.

**`correction_type` values**

| Type                  | Use when |
|-----------------------|----------|
| `SPLIT_ADJUSTMENT`    | Post-split backward price normalisation |
| `DIVIDEND_ADJUSTMENT` | Ex-dividend price adjustment |
| `DATA_ERROR`          | Vendor error — tick spike, wrong OHLC field |
| `LATE_DATA`           | Bar arrived after a gap; replaces a gap-fill row |
| `PARTIAL_TO_FINAL`    | Live partial bar promoted to official final close |

---

## 6. Ingest batch rollback

A `BarIngestBatch` groups all rows written in one ingestion run.
To roll back a corrupted or incorrect batch without deleting rows:

```sql
BEGIN;

UPDATE market_bars
SET    is_current    = FALSE,
       bar_status    = 'CORRECTED',
       superseded_at = NOW()
WHERE  ingest_batch_id = :batch_id
AND    is_current      = TRUE;

UPDATE bar_ingest_batches
SET    status = 'rolled_back'
WHERE  id     = :batch_id;

COMMIT;
```

Original rows are preserved for audit.  Re-running the batch will produce new
rows with `revision_seq = old_seq + 1` that become the current revision.

---

## 7. Lookahead guard for options joins (L7)

From `LEAKAGE_AUDIT.md §L7`:

> Options data must only be joined to a bar if it was available before
> that bar opened.  The invariant is:
>
>     option_quotes.available_at <= market_bars.event_time
>
> Using `>` or `>=` on the wrong side leaks future options state into
> bar features, introducing lookahead bias.

This is the most dangerous join in the feature pipeline because the violation
is subtle: using options data from 09:35 to price a feature for a 09:30 bar
is a 5-minute lookahead even though both timestamps fall in the same session.

**Checklist before any feature that uses options data:**

- [ ] Join condition is `oq.available_at <= mb.event_time`
- [ ] Not `oq.event_time` (may be NULL for yfinance; use `available_at`)
- [ ] Options data filtered to `is_current = TRUE`
- [ ] `available_at` accounts for source delay (yfinance options are 15-min delayed)
- [ ] Tested by adding a row with `available_at = bar.event_time + 1s` and verifying it is excluded

---

## 8. Building a correct training set

### The canonical recipe

```python
from datetime import datetime, timezone
from sqlalchemy import select
from app.models.market_data import MarketBar

T_start = datetime(2023, 1, 2, tzinfo=timezone.utc)
T_end   = datetime(2024, 1, 2, tzinfo=timezone.utc)   # exclusive

# Step 1: fetch bars with available_at <= bar's own event_time
stmt = (
    select(MarketBar)
    .where(
        MarketBar.symbol      == "SPY",
        MarketBar.timeframe   == "5m",
        MarketBar.event_time  >= T_start,
        MarketBar.event_time  <  T_end,
        MarketBar.available_at <= MarketBar.event_time,  # PIT boundary
        MarketBar.is_current  == True,
        MarketBar.bar_status  == "CLOSED",
    )
    .order_by(MarketBar.event_time)
)

# Step 2: for each bar, fetch nearest prior options snapshot (§3.5)

# Step 3: compute features — all rolling/EWM must use .shift(1) on OHLCV
# to ensure feature[i] uses only bars[0..i-1] (LEAKAGE_AUDIT.md §L2)
```

### Common mistakes

| Mistake | Effect | Fix |
|---------|--------|-----|
| Filter on `ingested_at` instead of `available_at` | May include bars not available at the modelled time | Use `available_at <= :T` |
| Include `PARTIAL` bars | Final close unknown; high/low may expand after close | Add `bar_status = 'CLOSED'` |
| Use `option_quotes.event_time` in join | NULL for yfinance; violates L7 | Use `option_quotes.available_at` |
| Omit `is_current = TRUE` | May include superseded (corrected) rows | Always filter `is_current = TRUE` |
| Use `>=` instead of `<=` on the PIT boundary | Future data leaks in | Must be `<=` |

---

## 9. Research snapshots (reproducibility)

A `ResearchSnapshot` pins a single `as_of_time` as the data boundary for
an entire research run.  All tables must be queried with:

```sql
WHERE available_at <= (
    SELECT as_of_time FROM research_snapshots WHERE name = :snap_name
)
AND is_current = TRUE
```

After verifying results, lock the snapshot:

```sql
UPDATE research_snapshots SET is_locked = TRUE WHERE name = :snap_name;
```

A locked snapshot is immutable.  Create a new snapshot with a new name rather
than modifying a locked one.  Locked snapshots are the unit of reproducibility
for published or committed research results.

---

## 10. Data source registry

`market_data_sources` is seeded in migration 003:

| source_id  | typical_delay_s | is_real_time | Notes |
|------------|-----------------|--------------|-------|
| `yfinance` | 900             | false        | 15-min delayed. Historical `available_at` is estimated (see §12 L-LIM-1). |
| `polygon`  | 0               | false        | REST history: no nominal delay. WebSocket: real-time. Requires subscription. |
| `alpaca`   | 0               | true         | WebSocket feed. Real-time. |
| `demo`     | 0               | false        | Synthetic seed data for local dev and CI. |

`typical_delay_s` is used to estimate `available_at` when the source API
does not provide it explicitly:

```python
available_at = event_time + timedelta(seconds=bar_duration_s + source.typical_delay_s)
```

When adding a new provider: add a row to `market_data_sources` in a new
migration **before** writing any bars that reference that `source_id`.

---

## 11. Governance integration

### `inference_events.data_available_at`

Set this field in `InferenceLogService` as:

```python
data_available_at = max(
    bar.available_at or bar.ingested_at,
    options_quote.available_at if options_quote else datetime.min.replace(tzinfo=timezone.utc),
)
```

Enables the query: was inference based on stale data?
```sql
SELECT COUNT(*)
FROM   inference_events
WHERE  data_available_at < (
           to_timestamp(inference_ts / 1000.0) AT TIME ZONE 'UTC'
           - INTERVAL '5 minutes'
       );
```

### `model_versions.went_live_at` / `went_offline_at`

Set in `ModelRegistryService`:

```python
# ModelRegistryService.promote():
version.went_live_at    = datetime.now(timezone.utc)

# ModelRegistryService.deprecate():
version.went_offline_at = datetime.now(timezone.utc)
```

---

## 12. Known limitations

### L-LIM-1: yfinance historical `available_at` is an estimate

yfinance does not expose the exact server-publication time of historical data.
For historical backfills, `available_at` should be computed as:

```python
available_at = event_time + timedelta(seconds=bar_duration_s + 900)
# Not ingested_at — that would be the fetch time, which can be years later.
```

If the original ingestion code used `ingested_at` as `available_at`, the
training set will be unnecessarily conservative (it will appear that less data
was available at time T than was actually the case).  This is safe (no
lookahead) but reduces effective training set size.

### L-LIM-2: `option_quotes.event_time` is NULL for yfinance

yfinance does not provide a per-quote exchange timestamp for options data.
`event_time` will be NULL.  Always use `available_at` (not `event_time`) in
join conditions for yfinance-sourced options.

### L-LIM-3: No cross-source revision coordination

If the same bar exists from two sources (yfinance + polygon), they are
independent rows with different `source_id` values.  There is no automatic
mechanism to mark one as authoritative.  Training pipelines must either:
- Choose one source exclusively per run, or
- Apply a priority rule (e.g., polygon > yfinance) and select one row per
  `(symbol, timeframe, event_time)` before computing features.

### L-LIM-4: Partial indexes are PostgreSQL-specific

The `WHERE is_current = TRUE` partial indexes are created with raw SQL in
the migration (`op.execute(CREATE INDEX ...)`).  They do not exist on
SQLite (used in unit tests) and will not be created on other engines.
Tests that run against SQLite will still work — queries are correct; they
just do full scans in the test environment.

### L-LIM-5: Legacy tables lack revision tracking

`ohlcv_bars` and `option_snapshots` have no revision chain.  They cannot
participate in the correction workflow.  New ingestion code must exclusively
target `market_bars` and `option_quotes`.

---

## 13. Schema diagram

```
market_data_sources (source registry)
  source_id PK            typical_delay_s    is_real_time
      │
      ├──────────────────────────┐
      │                          │
      ▼                          ▼
bar_ingest_batches ◄─────── bar_ingest_batches
  id PK                     (same table; FK from both market_bars and option_quotes)
  source_id FK
  symbol, timeframe, status
      │                              │
      ▼                              ▼
market_bars                    option_quotes
  id PK                          id PK
  symbol, timeframe               underlying_symbol, expiry, strike, option_type
  event_time  ◄────── L7: option_quotes.available_at <= market_bars.event_time
  available_at                    event_time
  ingested_at                     available_at ─────────────────────────────────────►
  source_id FK                    ingested_at
  ingest_batch_id FK              source_id FK
  bar_status                      ingest_batch_id FK
  open/high/low/close/volume      bid/ask/iv/greeks/chain_aggregates
  split_factor, div_factor        spread_pct, is_stale, is_illiquid
  revision_seq                    revision_seq
  is_current ◄── WHERE is_current = TRUE  (partial index on both tables)
  superseded_at                   superseded_at
  superseded_by FK (self)         superseded_by FK (self)
      │
      ▼
bar_corrections (append-only ledger)
  original_bar_id FK → market_bars
  replacement_bar_id FK → market_bars
  correction_type, reason, changed_fields_json

research_snapshots (reproducibility)
  as_of_time ── all PIT queries: WHERE available_at <= as_of_time
  is_locked  ── immutable once locked

inference_events (governance)
  bar_open_time       = event_time of triggering bar
  data_available_at   = max(bar.available_at, options.available_at)
  outcome_quality     = 'final' | 'preliminary' | 'corrected' | NULL

model_versions (governance)
  went_live_at    ─── PIT query: which model was active at T?
  went_offline_at ─── WHERE went_live_at <= T AND (went_offline_at IS NULL OR went_offline_at > T)
```

---

## 14. Legacy table reference

The original timestamp vocabulary (documented here for backward compatibility):

| Legacy field         | PIT-layer equivalent | Notes |
|----------------------|----------------------|-------|
| `bar_open_time`      | `event_time`         | In `ohlcv_bars` |
| `bar_close_time`     | `event_time + duration` | Derived; not stored in `market_bars` |
| `availability_time`  | `available_at`       | Nullable in legacy table |
| `snapshot_time`      | `available_at`       | In `option_snapshots` |
| `staleness_flag`     | `staleness_s > 0`    | In `ohlcv_bars` |
| `staleness_seconds`  | `staleness_s`        | In `option_snapshots` |
| `source`             | `source_id`          | Free-text string; FK in new tables |

Join rule in legacy code (kept for reference):
```python
# CORRECT (legacy tables)
WHERE option_snapshots.snapshot_time <= ohlcv_bars.bar_open_time

# CORRECT (new tables)
WHERE option_quotes.available_at <= market_bars.event_time
```

---

## 15. Audit query examples

```sql
-- Find market_bars ingested more than 30 min after estimated availability
SELECT symbol, timeframe, event_time, available_at, ingested_at, staleness_s
FROM   market_bars
WHERE  staleness_s > 1800
ORDER  BY staleness_s DESC
LIMIT  50;

-- Verify no PARTIAL bars slipped into the training window
SELECT COUNT(*) AS bad_bars
FROM   market_bars
WHERE  bar_status = 'PARTIAL'
AND    is_current = TRUE;

-- All corrections in the last 7 days with field-level diffs
SELECT bc.corrected_at, bc.correction_type, bc.reason,
       mb_old.symbol, mb_old.timeframe, mb_old.event_time,
       bc.changed_fields_json
FROM   bar_corrections bc
JOIN   market_bars mb_old ON mb_old.id = bc.original_bar_id
WHERE  bc.corrected_at >= NOW() - INTERVAL '7 days'
ORDER  BY bc.corrected_at DESC;

-- Options lookahead sanity check (should always return 0 rows)
SELECT COUNT(*) AS leakage_rows
FROM   option_quotes oq
JOIN   market_bars   mb
    ON mb.symbol    = oq.underlying_symbol
   AND mb.timeframe = '5m'
   AND mb.is_current = TRUE
WHERE  oq.available_at > mb.event_time    -- this condition must never match in feature queries
AND    oq.is_current    = TRUE;

-- Which model was live at a specific historical time?
SELECT model_name, version_tag, went_live_at, went_offline_at
FROM   model_versions
WHERE  went_live_at <= '2024-06-01 14:30:00+00'
AND    (went_offline_at IS NULL OR went_offline_at > '2024-06-01 14:30:00+00');

-- Inference events that used stale input data (lag > 5 min)
SELECT ie.symbol, ie.bar_open_time, ie.data_available_at,
       EXTRACT(EPOCH FROM (ie.bar_open_time - ie.data_available_at)) / 60 AS lag_minutes
FROM   inference_events ie
WHERE  ie.data_available_at IS NOT NULL
AND    ie.bar_open_time > ie.data_available_at + INTERVAL '5 minutes'
ORDER  BY lag_minutes DESC
LIMIT  50;

-- Research snapshot coverage — rows included per snapshot
SELECT rs.name, rs.as_of_time, rs.bar_count, rs.option_quote_count, rs.is_locked
FROM   research_snapshots rs
ORDER  BY rs.as_of_time DESC;
```
