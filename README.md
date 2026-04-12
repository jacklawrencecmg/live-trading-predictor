# Options Research Platform

Paper-trading options research tool with ML predictions, Greeks, and walk-forward backtesting.

## Stack
- **Frontend**: Next.js 14, TypeScript, Tailwind CSS, lightweight-charts, Recharts
- **Backend**: FastAPI, SQLAlchemy async, asyncpg
- **Database**: PostgreSQL 16
- **Cache/Pub-Sub**: Redis 7
- **Market Data**: yfinance (free, 15-min delayed)
- **ML**: scikit-learn (Logistic Regression + Ridge, calibrated)

## Quick Start

```bash
cp .env.example .env
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

## Features

### Live Market Data
- 5-minute candlestick charts with volume
- Real-time quote updates via WebSocket (15-second polling)
- Supports any ticker available on yfinance

### Options Chain
- Near-the-money strikes (±10% from spot)
- Full Greeks: Δ, Γ, Θ, Vega, Rho (Black-Scholes)
- Implied volatility (Brent solver)
- IV Rank, put-call ratio, ATM IV
- Volume and open interest

### ML Model
- **Features**: RSI(5,14), MACD, Bollinger Band %, ATR, volume ratio, momentum(5,10,20), IV rank, P/C ratio, ATM IV
- **Direction**: Logistic Regression (prob_up, prob_down)
- **Magnitude**: Ridge Regression (expected move %)
- **Signal**: buy / sell / no_trade with confidence threshold
- Model trains during first backtest run

### Walk-Forward Backtest
- Rolling window: train on N candles, test on M, step forward
- Metrics: accuracy, Brier score, log loss, magnitude MAE, Sharpe, total return
- Calibration curve per run
- Per-fold breakdown table

### Paper Trader
- Submit BTO/STO/BTC/STC orders at market price
- Position tracking with unrealized P&L
- Trade history with model confidence annotation
- No live orders — paper only

### Risk Controls
- Max daily loss (default 2% of capital) → auto kill switch
- Max position size (default 5% of capital)
- Per-symbol cooldown (default 15 min after trade)
- Manual kill switch (halts all new orders)

## Architecture

```
┌─────────────┐    REST/WS    ┌──────────────┐
│  Next.js UI │ ◄──────────► │  FastAPI     │
└─────────────┘               │  (port 8000) │
                               └──────┬───────┘
                                      │
                     ┌────────────────┼────────────────┐
                     │                │                │
              ┌──────┴──────┐  ┌──────┴──────┐  ┌─────┴──────┐
              │  PostgreSQL  │  │    Redis    │  │  yfinance  │
              │  (trades,    │  │  (cache,    │  │  (market   │
              │   positions, │  │   risk      │  │   data)    │
              │   audit)     │  │   state)    │  └────────────┘
              └─────────────┘  └─────────────┘
```

## Development

### Backend only
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend only
```bash
cd frontend
npm install
npm run dev
```

### Tests
```bash
cd backend
pytest tests/ -v --cov=app
```

## Constraints
- **No live trading** — all orders are paper only (enforced in code)
- Market data is delayed ~15 minutes (yfinance)
- Model starts untrained; run a backtest first to train it on historical data
- Research and inference layers are separate from execution

## Extending

### Better data
Swap yfinance for Polygon.io or Alpaca by replacing `app/services/market_data.py`.

### Better models
Add XGBoost/LightGBM in `app/services/model_service.py` alongside the logistic baseline.

### Options execution
`paper_trader.py` supports `option_symbol`, `strike`, `expiry`, `option_type` — just pass them from the UI.
