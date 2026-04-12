"""
Demo mode seed data.

Generates realistic synthetic OHLCV bars and options snapshots
for a 60-day period so the app works without external market data.

Usage:
    python -m app.demo.seed_data
"""

import asyncio
import math
import random
from datetime import datetime, timedelta
from typing import List

import numpy as np

from app.core.database import AsyncSessionLocal, Base, engine
from app.data_ingestion.bar_model import OHLCVBar

DEMO_SYMBOL = "DEMO"
DEMO_TIMEFRAME = "5m"
BARS_PER_DAY = 78  # 6.5h trading day, 5m bars
DAYS = 60
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def _generate_bars(symbol: str, days: int = DAYS) -> List[dict]:
    """
    Generate synthetic OHLCV bars using geometric Brownian motion.
    Market hours: 09:30 - 16:00 ET.
    """
    bars = []
    price = 450.0
    vol = 0.015  # daily vol
    bar_vol = vol / math.sqrt(252 * BARS_PER_DAY / 78)

    start_date = datetime.utcnow() - timedelta(days=days)
    current = start_date.replace(hour=14, minute=30, second=0, microsecond=0)  # 09:30 ET = 14:30 UTC

    for day in range(days):
        day_start = current.replace(hour=14, minute=30)
        if day_start.weekday() >= 5:  # skip weekends
            current += timedelta(days=1)
            continue

        for bar_i in range(BARS_PER_DAY):
            bar_open = day_start + timedelta(minutes=bar_i * 5)
            bar_close = bar_open + timedelta(minutes=5)

            # GBM step
            ret = np.random.normal(0.00005, bar_vol)
            open_p = price
            close_p = price * math.exp(ret)
            high_p = max(open_p, close_p) * (1 + abs(np.random.normal(0, bar_vol * 0.5)))
            low_p = min(open_p, close_p) * (1 - abs(np.random.normal(0, bar_vol * 0.5)))
            volume = abs(np.random.normal(5_000_000, 1_500_000))
            vwap = (high_p + low_p + close_p) / 3

            bars.append({
                "symbol": symbol,
                "timeframe": DEMO_TIMEFRAME,
                "bar_open_time": bar_open,
                "bar_close_time": bar_close,
                "open": round(open_p, 4),
                "high": round(high_p, 4),
                "low": round(low_p, 4),
                "close": round(close_p, 4),
                "volume": round(volume, 2),
                "vwap": round(vwap, 4),
                "is_closed": bar_close <= datetime.utcnow(),
                "source": "demo",
            })
            price = close_p

        current += timedelta(days=1)

    return bars


async def seed_database():
    """Create tables and seed demo data."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    bars = _generate_bars(DEMO_SYMBOL)
    print(f"Generated {len(bars)} bars for {DEMO_SYMBOL}")

    async with AsyncSessionLocal() as session:
        # Batch insert
        batch_size = 500
        for i in range(0, len(bars), batch_size):
            batch = bars[i: i + batch_size]
            session.add_all([OHLCVBar(**b) for b in batch])
        await session.commit()

    print(f"Seeded {len(bars)} bars into database")


if __name__ == "__main__":
    asyncio.run(seed_database())
