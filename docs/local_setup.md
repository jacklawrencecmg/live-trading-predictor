# Local Setup Guide

## Prerequisites

- Docker Desktop (or Podman)
- Node.js 20+
- Python 3.11+
- Git

## Quick Start (Docker)

```bash
git clone https://github.com/jacklawrencecmg/live-trading-predictor
cd live-trading-predictor
cp .env.example .env
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

## Demo Mode (no external data)

```bash
# Seed synthetic data
docker compose run backend python -m app.demo.seed_data

# Use DEMO symbol in the UI
```

## Backend Only (local Python)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp ../.env.example .env
uvicorn app.main:app --reload --port 8000
```

## Frontend Only

```bash
cd frontend
npm install
npm run dev
```

## Running Tests

```bash
cd backend
pytest tests/ -v --cov=app --cov-report=term-missing
```

## Training a Model

1. Start the app
2. Navigate to Backtest tab
3. Click "Run Backtest" — this trains and saves the model
4. Return to Dashboard — predictions will now be live

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| DATABASE_URL | PostgreSQL async URL | postgres on localhost |
| REDIS_URL | Redis URL | localhost:6379 |
| SECRET_KEY | App secret | dev-only value |
| DEMO_MODE | Use synthetic data | false |

## Data Sources

The app uses **yfinance** by default — free but ~15-min delayed.
For real-time data, swap `app/services/market_data.py` for Polygon.io or Alpaca.
