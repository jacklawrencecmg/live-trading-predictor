# Architecture Overview

## System Layers

```
┌──────────────────────────────────────────────────────────┐
│  Next.js Frontend (port 3000)                            │
│  Dashboard | Chart | Options Chain | Signals | Backtest  │
└────────────────────┬─────────────────────────────────────┘
                     │ REST + WebSocket
┌────────────────────▼─────────────────────────────────────┐
│  FastAPI Backend (port 8000)                             │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Data         │  │ Feature      │  │ ML Models    │   │
│  │ Ingestion    │→ │ Pipeline     │→ │ (LR/RF/GBT)  │   │
│  └──────────────┘  └──────────────┘  └──────┬───────┘   │
│                                             │            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────▼───────┐   │
│  │ Backtesting  │  │ Signal       │  │ Inference    │   │
│  │ (WF splits)  │  │ Scoring      │← │ Service      │   │
│  └──────────────┘  └──────┬───────┘  └──────────────┘   │
│                           │                              │
│  ┌──────────────┐  ┌──────▼───────┐  ┌──────────────┐   │
│  │ Risk Manager │← │ Paper        │  │ Alerts       │   │
│  │              │  │ Trading      │  │              │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└────────────────────────┬─────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
   ┌──────▼──────┐ ┌─────▼──────┐ ┌────▼───────┐
   │ PostgreSQL  │ │   Redis    │ │  yfinance  │
   │ (bars,      │ │ (cache,    │ │  (market   │
   │  positions, │ │  risk      │ │   data)    │
   │  audit)     │ │  state)    │ └────────────┘
   └─────────────┘ └────────────┘
```

## Data Flow

```
yfinance → data_ingestion → OHLCVBar (PostgreSQL)
                ↓
         feature_pipeline → FeatureMatrix (in-memory)
                ↓
           ml_models → predict_proba
                ↓
         inference_service → InferenceResult
                ↓
          signal_scorer → ScoredSignal
                ↓
         paper_trading → Trade/Position (PostgreSQL)
```

## Leakage Prevention

See `docs/leakage_checklist.md` for full audit.

Key invariants:
1. `feature[i]` uses only `bar[0..i-1]` (shift-by-1 applied)
2. `label[i]` uses only `bar[i+1].close`
3. Train/test splits use `TimeSeriesSplit` (no shuffling)
4. Inference triggers only on `is_closed=True` bars
5. Options snapshots keyed by `snapshot_time < bar_open_time`

## Module Boundaries

| Module | Responsibility | Must NOT touch |
|--------|---------------|----------------|
| data_ingestion | Fetch + store raw bars | Features, models |
| feature_pipeline | Compute features from closed bars | Raw data fetch, models |
| ml_models | Train + serialize models | Data fetch, execution |
| inference | Load model, run prediction | Training |
| backtesting | Evaluate model over time | Live execution |
| paper_trading | Simulate fills | Model training |
| risk | Enforce limits | Predictions |
| alerts | Notify on events | Trading decisions |
