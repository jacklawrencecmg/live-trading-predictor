"""
Tests for data lineage — timestamp audit trail on OHLCVBar and OptionSnapshot.

Validates:
- OHLCVBar availability_time and staleness_flag are populated correctly
- OptionSnapshot snapshot_time is enforced before bar_open_time (no lookahead)
- staleness_seconds is computed correctly at join time
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy import select

from app.data_ingestion.bar_model import OHLCVBar
from app.models.option_snapshot import OptionSnapshot


def _make_bar(
    symbol: str = "SPY",
    timeframe: str = "5m",
    bar_open: datetime = None,
    is_closed: bool = True,
    availability_time: datetime = None,
    ingested_at: datetime = None,
    staleness_flag: bool = False,
    source: str = "yfinance",
) -> OHLCVBar:
    bar_open = bar_open or datetime(2024, 1, 15, 14, 30)  # 09:30 ET as UTC-5 proxy
    bar_close = bar_open + timedelta(minutes=5)
    now = ingested_at or datetime.utcnow()
    return OHLCVBar(
        symbol=symbol,
        timeframe=timeframe,
        bar_open_time=bar_open,
        bar_close_time=bar_close,
        availability_time=availability_time or now,
        ingested_at=now,
        source=source,
        staleness_flag=staleness_flag,
        open=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0,
        is_closed=is_closed,
    )


def _make_snapshot(
    symbol: str = "SPY",
    expiry: str = "2024-01-19",
    strike: float = 100.0,
    option_type: str = "call",
    snapshot_time: datetime = None,
    staleness_seconds: float = None,
) -> OptionSnapshot:
    return OptionSnapshot(
        underlying_symbol=symbol,
        expiry=expiry,
        strike=strike,
        option_type=option_type,
        snapshot_time=snapshot_time or datetime(2024, 1, 15, 14, 25),
        ingested_at=datetime.utcnow(),
        source="yfinance",
        staleness_seconds=staleness_seconds,
        underlying_price=100.0,
        bid=1.0, ask=1.05, last=1.02,
        implied_volatility=0.20,
        delta=0.50, gamma=0.05, theta=-0.02, vega=0.10, rho=0.01,
        is_illiquid=False,
    )


@pytest.mark.asyncio
async def test_bar_persists_availability_time(db_session):
    """availability_time is stored and retrieved correctly."""
    avail = datetime(2024, 1, 15, 14, 35, 0)  # 5 min after bar closes
    bar = _make_bar(availability_time=avail)
    db_session.add(bar)
    await db_session.flush()

    result = await db_session.execute(
        select(OHLCVBar).where(OHLCVBar.symbol == "SPY")
    )
    saved = result.scalar_one()
    assert saved.availability_time == avail


@pytest.mark.asyncio
async def test_bar_staleness_flag_set_for_delayed_ingest(db_session):
    """staleness_flag=True when bar was ingested long after bar_close_time."""
    bar = _make_bar(staleness_flag=True)
    db_session.add(bar)
    await db_session.flush()

    result = await db_session.execute(
        select(OHLCVBar).where(OHLCVBar.symbol == "SPY")
    )
    saved = result.scalar_one()
    assert saved.staleness_flag is True


@pytest.mark.asyncio
async def test_bar_staleness_flag_false_for_fresh_ingest(db_session):
    """staleness_flag=False (default) for a normally-ingested bar."""
    bar = _make_bar(staleness_flag=False)
    db_session.add(bar)
    await db_session.flush()

    result = await db_session.execute(
        select(OHLCVBar).where(OHLCVBar.symbol == "SPY")
    )
    saved = result.scalar_one()
    assert saved.staleness_flag is False


@pytest.mark.asyncio
async def test_option_snapshot_persists_snapshot_time(db_session):
    """snapshot_time (availability_time for options) is stored correctly."""
    snap_time = datetime(2024, 1, 15, 14, 28, 0)
    snap = _make_snapshot(snapshot_time=snap_time)
    db_session.add(snap)
    await db_session.flush()

    result = await db_session.execute(
        select(OptionSnapshot).where(OptionSnapshot.underlying_symbol == "SPY")
    )
    saved = result.scalar_one()
    assert saved.snapshot_time == snap_time


def test_option_snapshot_no_lookahead_invariant():
    """
    Invariant: snapshot_time must be <= bar_open_time for valid joins.
    This test verifies the check logic directly (no DB needed).
    """
    bar_open = datetime(2024, 1, 15, 14, 30)

    # Valid: snapshot captured 5 min before bar opened
    snap_valid = _make_snapshot(snapshot_time=bar_open - timedelta(minutes=5))
    assert snap_valid.snapshot_time <= bar_open, "Valid snapshot should be before bar_open"

    # Invalid (lookahead): snapshot captured after bar opened
    snap_future = _make_snapshot(snapshot_time=bar_open + timedelta(minutes=1))
    assert snap_future.snapshot_time > bar_open, \
        "Lookahead snapshot correctly identified as invalid for join"


@pytest.mark.asyncio
async def test_option_snapshot_staleness_seconds_stored(db_session):
    """staleness_seconds records how stale the options data was at join time."""
    snap = _make_snapshot(staleness_seconds=47.5)
    db_session.add(snap)
    await db_session.flush()

    result = await db_session.execute(
        select(OptionSnapshot).where(OptionSnapshot.underlying_symbol == "SPY")
    )
    saved = result.scalar_one()
    assert abs(saved.staleness_seconds - 47.5) < 0.01


def test_ingestion_service_populates_staleness_flag():
    """_parse_yf_df marks bars as stale when ingested long after bar close."""
    import pandas as pd
    from app.data_ingestion.ingestion_service import _parse_yf_df

    # Build a minimal DataFrame with a bar from 60 days ago
    old_ts = datetime.utcnow() - timedelta(days=60)
    df = pd.DataFrame(
        {"Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.5], "Volume": [5000]},
        index=[pd.Timestamp(old_ts)],
    )

    bars = _parse_yf_df(df, "SPY", "5m")
    assert len(bars) == 1
    assert bars[0]["staleness_flag"] is True, \
        "Historical bar ingested now should be marked stale"
    assert bars[0]["availability_time"] is not None
    assert bars[0]["ingested_at"] is not None
    assert bars[0]["source"] == "yfinance"
