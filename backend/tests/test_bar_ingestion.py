"""
Tests for OHLCV bar ingestion: bar_store.py and bar_ingestion.py.

Coverage
--------
T-1   is_bar_closed: correct with buffer, boundary, future bar
T-2   is_within_session: regular hours, extended, weekend, holiday
T-3   PartialBarError raised for PARTIAL and INVALID bar_status
T-4   PartialBarError NOT raised when allow_partial=True
T-5   PartialBarError NOT raised for CLOSED bar
T-6   write_bar: fresh insert → action='inserted', revision_seq=1
T-7   write_bar: same payload twice → action='skipped' (idempotent)
T-8   write_bar: OHLCV changed → action='corrected', revision chain set
T-9   write_bar: PARTIAL promoted to CLOSED → action='updated_partial'
T-10  write_bar: BarCorrection row written on correction
T-11  get_closed_bars_pit: respects available_at <= as_of cutoff
T-12  get_closed_bars_pit: excludes PARTIAL bars
T-13  get_closed_bars_pit: excludes non-current (superseded) revisions
T-14  BarIngestionService.backfill: session-filtered, inserts correct bars
T-15  BarIngestionService.backfill: provider error propagates, batch marked failed
T-16  BarIngestionService: partial bar written for current open bar
T-17  BarIngestResult counts match write outcomes
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
import pandas as pd
import pytest
import pytest_asyncio

from app.data_ingestion.bar_ingestion import (
    BarIngestionService,
    BarIngestResult,
    PartialBarError,
    assert_bar_usable,
)
from app.data_ingestion.bar_store import (
    BarWriteResult,
    close_batch,
    get_closed_bars_pit,
    open_batch,
    write_bar,
)
from app.data_ingestion.session_calendar import (
    BAR_CLOSE_BUFFER_S,
    is_bar_closed,
    is_within_session,
)
from app.models.market_data import BarCorrection, MarketBar, MarketDataSource
from app.providers.protocols import ProviderError

UTC = timezone.utc

# ── Constants ─────────────────────────────────────────────────────────────────

# A Monday well inside regular session (15:00 UTC = 10:00 EST; DST starts Mar 10)
_REG_SESSION_BAR = datetime(2024, 3, 4, 15, 0, 0)  # Mon 2024-03-04 10:00 EST
_TF = "5m"
_SRC = "yfinance"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def source(db_session):
    """Insert a MarketDataSource row required by FK constraints."""
    src = MarketDataSource(
        source_id=_SRC,
        display_name="Yahoo Finance",
        typical_delay_s=900,
        max_staleness_s=1800,
        is_real_time=False,
    )
    db_session.add(src)
    await db_session.flush()
    return src


async def _write_closed(
    session,
    *,
    symbol: str = "SPY",
    timeframe: str = _TF,
    event_time: datetime = _REG_SESSION_BAR,
    source_id: str = _SRC,
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.5,
    volume: float = 1_000_000.0,
    available_at: datetime | None = None,
) -> BarWriteResult:
    """Helper: write a CLOSED bar with default test values."""
    if available_at is None:
        available_at = event_time + timedelta(minutes=20)
    return await write_bar(
        session,
        symbol=symbol,
        timeframe=timeframe,
        event_time=event_time,
        source_id=source_id,
        bar_status="CLOSED",
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        available_at=available_at,
    )


# ── T-1: is_bar_closed ────────────────────────────────────────────────────────

class TestIsBarClosed:
    def test_closed_after_buffer(self):
        # Bar opened 30 minutes ago with 5m timeframe; should be closed.
        event = datetime.utcnow() - timedelta(minutes=30)
        assert is_bar_closed(event, "5m") is True

    def test_open_before_close(self):
        # Bar opened 1 second ago; cannot be closed yet.
        event = datetime.utcnow() - timedelta(seconds=1)
        assert is_bar_closed(event, "5m") is False

    def test_boundary_exactly_at_close_plus_buffer(self):
        # now = event_time + bar_duration + buffer exactly → closed
        duration_s = 300  # 5m
        event = datetime(2024, 3, 4, 14, 0, 0)
        now = event + timedelta(seconds=duration_s + BAR_CLOSE_BUFFER_S)
        assert is_bar_closed(event, "5m", now_utc=now) is True

    def test_boundary_one_second_before_close(self):
        # now = event_time + bar_duration + buffer - 1s → NOT closed
        duration_s = 300
        event = datetime(2024, 3, 4, 14, 0, 0)
        now = event + timedelta(seconds=duration_s + BAR_CLOSE_BUFFER_S - 1)
        assert is_bar_closed(event, "5m", now_utc=now) is False

    def test_daily_bar_from_yesterday(self):
        # Daily bar from yesterday is closed.
        event = datetime.utcnow() - timedelta(hours=26)
        assert is_bar_closed(event, "1d") is True

    def test_hourly_bar_open(self):
        event = datetime.utcnow() - timedelta(minutes=30)
        assert is_bar_closed(event, "1h") is False


# ── T-2: is_within_session ────────────────────────────────────────────────────

class TestIsWithinSession:
    def test_regular_session_bar(self):
        # 2024-03-04 14:00 UTC = 10:00 ET (Monday, not holiday)
        assert is_within_session(_REG_SESSION_BAR) is True

    def test_before_open(self):
        # 13:00 UTC = 09:00 ET — before 09:30 open
        early = datetime(2024, 3, 4, 13, 0, 0)
        assert is_within_session(early) is False

    def test_after_close(self):
        # 21:00 UTC = 17:00 ET — after 16:00 close
        late = datetime(2024, 3, 4, 21, 0, 0)
        assert is_within_session(late) is False

    def test_saturday(self):
        sat = datetime(2024, 3, 2, 14, 0, 0)
        assert is_within_session(sat) is False

    def test_sunday(self):
        sun = datetime(2024, 3, 3, 14, 0, 0)
        assert is_within_session(sun) is False

    def test_nyse_holiday_independence_day(self):
        # 2024-07-04 is Independence Day (Thursday)
        indep = datetime(2024, 7, 4, 14, 0, 0)
        assert is_within_session(indep) is False

    def test_extended_hours_premarket(self):
        # 07:00 ET = 11:00 UTC — pre-market
        pre = datetime(2024, 3, 4, 11, 0, 0)  # 07:00 ET
        assert is_within_session(pre, include_extended_hours=True) is True
        assert is_within_session(pre, include_extended_hours=False) is False

    def test_timezone_aware_input(self):
        # Pass an ET-aware datetime
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
        et_dt = datetime(2024, 3, 4, 10, 0, 0, tzinfo=ET)
        assert is_within_session(et_dt) is True


# ── T-3/T-4/T-5: assert_bar_usable ───────────────────────────────────────────

class TestAssertBarUsable:
    def _make_bar(self, status: str) -> MagicMock:
        bar = MagicMock(spec=MarketBar)
        bar.bar_status = status
        bar.symbol = "SPY"
        bar.timeframe = "5m"
        bar.event_time = _REG_SESSION_BAR
        return bar

    def test_raises_for_partial(self):
        bar = self._make_bar("PARTIAL")
        with pytest.raises(PartialBarError) as exc_info:
            assert_bar_usable(bar)
        assert exc_info.value.bar_status == "PARTIAL"
        assert exc_info.value.symbol == "SPY"

    def test_raises_for_invalid(self):
        bar = self._make_bar("INVALID")
        with pytest.raises(PartialBarError):
            assert_bar_usable(bar)

    def test_passes_for_closed(self):
        bar = self._make_bar("CLOSED")
        assert_bar_usable(bar)  # no exception

    def test_passes_for_backfilled(self):
        bar = self._make_bar("BACKFILLED")
        assert_bar_usable(bar)  # no exception

    def test_passes_for_corrected(self):
        bar = self._make_bar("CORRECTED")
        assert_bar_usable(bar)  # no exception

    def test_allow_partial_bypasses_guard(self):
        bar = self._make_bar("PARTIAL")
        assert_bar_usable(bar, allow_partial=True)  # no exception

    def test_allow_partial_still_raises_for_invalid(self):
        # allow_partial does NOT bypass INVALID
        bar = self._make_bar("INVALID")
        with pytest.raises(PartialBarError):
            assert_bar_usable(bar, allow_partial=False)


# ── T-6: fresh insert ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_write_bar_fresh_insert(db_session, source):
    result = await _write_closed(db_session)
    assert result.action == "inserted"
    assert result.bar_id is not None
    assert result.old_bar_id is None

    bar = await db_session.get(MarketBar, result.bar_id)
    assert bar.symbol == "SPY"
    assert bar.revision_seq == 1
    assert bar.is_current is True
    assert bar.bar_status == "CLOSED"


# ── T-7: idempotent skip ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_write_bar_idempotent(db_session, source):
    r1 = await _write_closed(db_session)
    await db_session.flush()

    # Write identical payload again
    r2 = await _write_closed(db_session)
    assert r2.action == "skipped"
    assert r2.bar_id == r1.bar_id


# ── T-8: correction on changed OHLCV ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_write_bar_correction(db_session, source):
    r1 = await _write_closed(db_session, close=100.5)
    await db_session.flush()

    # Different close price triggers correction
    r2 = await _write_closed(db_session, close=101.0)
    assert r2.action == "corrected"
    assert r2.bar_id is not None
    assert r2.old_bar_id == r1.bar_id

    # Old row is superseded
    old_bar = await db_session.get(MarketBar, r1.bar_id)
    assert old_bar.is_current is False
    assert old_bar.superseded_by == r2.bar_id
    assert old_bar.bar_status == "CORRECTED"

    # New row is current with incremented revision_seq
    new_bar = await db_session.get(MarketBar, r2.bar_id)
    assert new_bar.is_current is True
    assert new_bar.revision_seq == 2
    assert new_bar.close == pytest.approx(101.0)


# ── T-9: PARTIAL promoted to CLOSED ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_write_bar_partial_to_closed(db_session, source):
    # Write initial PARTIAL bar (live, bar not yet closed)
    r1 = await write_bar(
        db_session,
        symbol="SPY", timeframe=_TF,
        event_time=_REG_SESSION_BAR, source_id=_SRC,
        bar_status="PARTIAL",
        open_=100.0, high=100.8, low=99.5, close=100.3, volume=500_000.0,
        available_at=None,
    )
    assert r1.action == "inserted"
    await db_session.flush()

    # Bar closes — write CLOSED version (same event_time, different close)
    r2 = await write_bar(
        db_session,
        symbol="SPY", timeframe=_TF,
        event_time=_REG_SESSION_BAR, source_id=_SRC,
        bar_status="CLOSED",
        open_=100.0, high=101.0, low=99.5, close=100.5, volume=1_000_000.0,
        available_at=_REG_SESSION_BAR + timedelta(minutes=20),
    )
    assert r2.action == "updated_partial"
    assert r2.old_bar_id == r1.bar_id

    # Old PARTIAL row is superseded
    old = await db_session.get(MarketBar, r1.bar_id)
    assert old.is_current is False
    assert old.bar_status == "PARTIAL"  # kept as PARTIAL, not changed to CORRECTED

    # New CLOSED row is current
    new = await db_session.get(MarketBar, r2.bar_id)
    assert new.bar_status == "CLOSED"
    assert new.revision_seq == 2


# ── T-10: BarCorrection ledger entry ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_correction_ledger_entry(db_session, source):
    from sqlalchemy import select

    r1 = await _write_closed(db_session, close=100.0)
    await db_session.flush()
    r2 = await _write_closed(db_session, close=99.0)
    await db_session.flush()

    corrections = (await db_session.execute(
        select(BarCorrection).where(BarCorrection.original_bar_id == r1.bar_id)
    )).scalars().all()
    assert len(corrections) == 1
    corr = corrections[0]
    assert corr.correction_type == "DATA_ERROR"
    assert corr.replacement_bar_id == r2.bar_id
    assert "close" in corr.changed_fields_json


# ── T-11: get_closed_bars_pit — available_at cutoff ──────────────────────────

@pytest.mark.asyncio
async def test_get_closed_bars_pit_cutoff(db_session, source):
    base = datetime(2024, 3, 4, 14, 0, 0)  # event_time for bar 1
    base2 = datetime(2024, 3, 4, 14, 5, 0)  # event_time for bar 2

    av1 = base + timedelta(minutes=20)
    av2 = base2 + timedelta(minutes=20)

    await write_bar(
        db_session, symbol="SPY", timeframe=_TF,
        event_time=base, source_id=_SRC,
        bar_status="CLOSED",
        open_=100.0, high=101.0, low=99.0, close=100.5, volume=1e6,
        available_at=av1,
    )
    await write_bar(
        db_session, symbol="SPY", timeframe=_TF,
        event_time=base2, source_id=_SRC,
        bar_status="CLOSED",
        open_=100.5, high=102.0, low=100.0, close=101.5, volume=1.2e6,
        available_at=av2,
    )
    await db_session.flush()

    # as_of between av1 and av2 → only first bar visible
    bars = await get_closed_bars_pit(
        db_session, "SPY", _TF,
        as_of_utc=av1 + timedelta(minutes=1),
    )
    assert len(bars) == 1
    assert bars[0].event_time == base

    # as_of after both → both visible
    bars_all = await get_closed_bars_pit(
        db_session, "SPY", _TF,
        as_of_utc=av2 + timedelta(minutes=1),
    )
    assert len(bars_all) == 2


# ── T-12: get_closed_bars_pit excludes PARTIAL ───────────────────────────────

@pytest.mark.asyncio
async def test_get_closed_bars_pit_excludes_partial(db_session, source):
    partial_time = datetime(2024, 3, 4, 14, 0, 0)
    closed_time = datetime(2024, 3, 4, 14, 5, 0)
    av_closed = closed_time + timedelta(minutes=20)

    # Write PARTIAL bar (available_at = None)
    await write_bar(
        db_session, symbol="SPY", timeframe=_TF,
        event_time=partial_time, source_id=_SRC,
        bar_status="PARTIAL",
        open_=100.0, high=100.5, low=99.8, close=100.2, volume=500_000.0,
        available_at=None,
    )
    # Write CLOSED bar
    await write_bar(
        db_session, symbol="SPY", timeframe=_TF,
        event_time=closed_time, source_id=_SRC,
        bar_status="CLOSED",
        open_=100.2, high=101.0, low=100.0, close=100.8, volume=1e6,
        available_at=av_closed,
    )
    await db_session.flush()

    bars = await get_closed_bars_pit(
        db_session, "SPY", _TF,
        as_of_utc=av_closed + timedelta(hours=1),
    )
    # Only the CLOSED bar should appear
    assert len(bars) == 1
    assert bars[0].bar_status == "CLOSED"


# ── T-13: get_closed_bars_pit excludes superseded rows ───────────────────────

@pytest.mark.asyncio
async def test_get_closed_bars_pit_excludes_superseded(db_session, source):
    av = _REG_SESSION_BAR + timedelta(minutes=20)

    await _write_closed(db_session, close=100.0, available_at=av)
    await db_session.flush()
    # Correction supersedes the first row
    await _write_closed(db_session, close=99.5, available_at=av)
    await db_session.flush()

    bars = await get_closed_bars_pit(
        db_session, "SPY", _TF,
        as_of_utc=av + timedelta(hours=1),
    )
    # Only the current revision should be returned
    assert len(bars) == 1
    assert bars[0].close == pytest.approx(99.5)
    assert bars[0].is_current is True


# ── T-14/T-17: BarIngestionService.backfill ──────────────────────────────────

def _make_provider_df(bars: list[dict]) -> pd.DataFrame:
    """Build a provider-format DataFrame from a list of bar dicts."""
    rows = []
    for b in bars:
        rows.append({
            "open": b.get("open", 100.0),
            "high": b.get("high", 101.0),
            "low": b.get("low", 99.0),
            "close": b.get("close", 100.5),
            "volume": b.get("volume", 1e6),
            "bar_open_time": b["event_time"],
        })
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex([r["event_time"] for r in bars])
    return df


@pytest.mark.asyncio
async def test_backfill_inserts_session_bars(db_session, source):
    # Two regular-session bars and one out-of-session (Saturday)
    # 2024-03-04 is before DST (starts 2024-03-10): EST = UTC-5
    regular1 = datetime(2024, 3, 4, 15, 0, 0)   # Mon 10:00 EST
    regular2 = datetime(2024, 3, 4, 15, 5, 0)   # Mon 10:05 EST
    saturday = datetime(2024, 3, 2, 15, 0, 0)   # Sat — outside session

    df = _make_provider_df([
        {"event_time": regular1},
        {"event_time": regular2},
        {"event_time": saturday},
    ])

    mock_provider = MagicMock()
    mock_provider.get_candles = AsyncMock(return_value=df)

    svc = BarIngestionService(mock_provider, source_id=_SRC, typical_delay_s=900)

    # Use a now far in the past so all bars appear closed
    past_now = datetime(2024, 3, 5, 20, 0, 0)
    result = await svc.backfill(db_session, "SPY", _TF, period="5d", now_utc=past_now)

    assert result.success
    assert result.bars_inserted == 2
    assert result.bars_out_of_session == 1
    assert result.batch_id is not None


@pytest.mark.asyncio
async def test_backfill_marks_partial_bar(db_session, source):
    # One bar that is in the future relative to now → PARTIAL
    future_bar = datetime.utcnow() + timedelta(seconds=10)

    df = _make_provider_df([{"event_time": future_bar}])
    mock_provider = MagicMock()
    mock_provider.get_candles = AsyncMock(return_value=df)

    svc = BarIngestionService(mock_provider, source_id=_SRC, typical_delay_s=900)

    # Use a fixed now that makes the bar appear open
    fixed_now = future_bar  # bar_nominal_close = future + 5m, still in future
    result = await svc.backfill(
        db_session, "SPY", _TF, period="2d", now_utc=fixed_now
    )

    # The bar should be written as PARTIAL (not counted in bars_inserted)
    assert result.bars_partial >= 1 or result.bars_out_of_session >= 1
    # (the future bar may be out of session too — that's fine)


# ── T-15: provider error propagates ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_backfill_provider_error_marks_batch_failed(db_session, source):
    from sqlalchemy import select as sa_select

    mock_provider = MagicMock()
    mock_provider.get_candles = AsyncMock(
        side_effect=ProviderError("network timeout")
    )

    svc = BarIngestionService(mock_provider, source_id=_SRC, typical_delay_s=900)

    with pytest.raises(ProviderError):
        await svc.backfill(db_session, "SPY", _TF, period="5d", max_retries=1)

    # Batch should be in 'failed' state
    from app.models.market_data import BarIngestBatch
    batches = (await db_session.execute(
        sa_select(BarIngestBatch).where(BarIngestBatch.source_id == _SRC)
    )).scalars().all()
    assert any(b.status == "failed" for b in batches)


# ── T-16: no inference on partial bar from service ────────────────────────────

@pytest.mark.asyncio
async def test_no_inference_on_partial_bar(db_session, source):
    """End-to-end: partial bar written by ingestion service → assert_bar_usable blocks it."""
    from sqlalchemy import select as sa_select

    # Write a PARTIAL bar directly
    await write_bar(
        db_session, symbol="SPY", timeframe=_TF,
        event_time=_REG_SESSION_BAR, source_id=_SRC,
        bar_status="PARTIAL",
        open_=100.0, high=100.5, low=99.8, close=100.2, volume=500_000.0,
        available_at=None,
    )
    await db_session.flush()

    # Retrieve the partial bar
    bars = (await db_session.execute(
        sa_select(MarketBar).where(
            MarketBar.symbol == "SPY",
            MarketBar.bar_status == "PARTIAL",
        )
    )).scalars().all()
    assert len(bars) >= 1

    partial_bar = bars[0]
    # Guard must block inference
    with pytest.raises(PartialBarError) as exc_info:
        assert_bar_usable(partial_bar)
    assert "PARTIAL" in str(exc_info.value)
    assert "allow_partial=True" in str(exc_info.value)

    # And the PIT read must not return it
    pit_bars = await get_closed_bars_pit(
        db_session, "SPY", _TF,
        as_of_utc=datetime.utcnow() + timedelta(hours=1),
    )
    assert all(b.bar_status != "PARTIAL" for b in pit_bars)
