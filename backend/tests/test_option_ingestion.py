"""
Tests for options chain ingestion: option_store.py and option_ingestion.py.

Coverage
--------
O-1   write_quote: fresh insert → action='inserted', fields stored correctly
O-2   write_quote: identical payload → action='skipped' (idempotent)
O-3   write_quote: changed bid/ask → action='corrected', revision chain set
O-4   write_quote: NaN greeks stored as NULL (no poisoning)
O-5   write_quote: None greeks stored as NULL (no forward-fill from prior)
O-6   compute_spread_pct: correct values, None on bad inputs
O-7   compute_is_illiquid: zero bid/ask, zero OI, normal quote
O-8   get_latest_chain_pit: respects available_at <= as_of_utc
O-9   get_latest_chain_pit: excludes superseded rows
O-10  get_latest_chain_pit: expiry filter works
O-11  OptionChainResponse: missingness counts correct
O-12  OptionIngestResult: counts match write outcomes
O-13  OptionChainIngestionService.snapshot: writes all contracts
O-14  OptionChainIngestionService.snapshot: provider error marks batch failed
O-15  No Greek forward-fill across snapshots (core invariant)
O-16  spread_pct = None when bid/ask missing (not 0)
O-17  is_illiquid True when OI=0, False when OI=1
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from app.data_ingestion.option_ingestion import (
    OptionChainIngestionService,
    OptionIngestResult,
)
from app.data_ingestion.option_store import (
    OptionChainResponse,
    OptionChainRow,
    OptionWriteResult,
    _compute_is_illiquid,
    _compute_spread_pct,
    get_latest_chain_pit,
    write_quote,
)
from app.models.market_data import MarketDataSource, OptionQuote
from app.providers.protocols import (
    OptionContract,
    OptionsChainSnapshot,
    ProviderError,
)

UTC = timezone.utc

# ── Constants ─────────────────────────────────────────────────────────────────

_SRC = "yfinance"
_SYMBOL = "SPY"
_EXPIRY = "2024-06-21"
_STRIKE = 500.0
_NOW = datetime(2024, 3, 4, 15, 0, 0)  # fixed "now" for determinism


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def source(db_session):
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


async def _write_call(
    session,
    *,
    available_at: datetime = _NOW,
    bid: float | None = 5.0,
    ask: float | None = 5.10,
    implied_volatility: float | None = 0.20,
    delta: float | None = 0.45,
    gamma: float | None = 0.02,
    theta: float | None = -0.05,
    vega: float | None = 0.30,
    volume: int | None = 1000,
    open_interest: int | None = 5000,
) -> OptionWriteResult:
    return await write_quote(
        session,
        underlying_symbol=_SYMBOL,
        expiry=_EXPIRY,
        strike=_STRIKE,
        option_type="call",
        source_id=_SRC,
        available_at=available_at,
        bid=bid,
        ask=ask,
        implied_volatility=implied_volatility,
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        volume=volume,
        open_interest=open_interest,
    )


# ── O-6: spread_pct ───────────────────────────────────────────────────────────

class TestComputeSpreadPct:
    def test_normal_spread(self):
        # (5.10 - 5.00) / 5.05 ≈ 0.0198
        pct = _compute_spread_pct(5.00, 5.10)
        assert pct is not None
        assert abs(pct - 0.10 / 5.05) < 1e-6

    def test_zero_spread(self):
        # bid == ask → spread = 0
        pct = _compute_spread_pct(5.00, 5.00)
        assert pct == pytest.approx(0.0)

    def test_missing_bid(self):
        assert _compute_spread_pct(None, 5.10) is None

    def test_missing_ask(self):
        assert _compute_spread_pct(5.00, None) is None

    def test_negative_spread_returns_none(self):
        # bid > ask → bad data
        assert _compute_spread_pct(5.10, 5.00) is None

    def test_zero_mid_returns_none(self):
        assert _compute_spread_pct(0.0, 0.0) is None

    def test_nan_inputs_return_none(self):
        import math
        assert _compute_spread_pct(float("nan"), 5.10) is None
        assert _compute_spread_pct(5.00, float("nan")) is None


# ── O-7: is_illiquid ──────────────────────────────────────────────────────────

class TestComputeIsIlliquid:
    def test_normal_quote_not_illiquid(self):
        assert _compute_is_illiquid(5.00, 5.10, 1000) is False

    def test_zero_bid_ask_is_illiquid(self):
        assert _compute_is_illiquid(0.0, 0.0, 100) is True

    def test_zero_oi_is_illiquid(self):
        assert _compute_is_illiquid(5.00, 5.10, 0) is True

    def test_none_oi_not_illiquid(self):
        # Unknown OI should not flag as illiquid
        assert _compute_is_illiquid(5.00, 5.10, None) is False

    def test_missing_bid_ask_not_illiquid(self):
        # Missing bid/ask alone does not make it illiquid
        assert _compute_is_illiquid(None, None, 100) is False

    def test_oi_one_not_illiquid(self):
        assert _compute_is_illiquid(0.05, 0.10, 1) is False


# ── O-1: fresh insert ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_write_quote_fresh_insert(db_session, source):
    result = await _write_call(db_session)
    assert result.action == "inserted"
    assert result.quote_id is not None
    assert result.old_quote_id is None

    quote = await db_session.get(OptionQuote, result.quote_id)
    assert quote.underlying_symbol == _SYMBOL
    assert quote.expiry == _EXPIRY
    assert quote.strike == _STRIKE
    assert quote.option_type == "call"
    assert quote.revision_seq == 1
    assert quote.is_current is True
    assert quote.spread_pct is not None
    assert quote.dte is not None and quote.dte > 0


# ── O-2: idempotent skip ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_write_quote_idempotent(db_session, source):
    r1 = await _write_call(db_session)
    await db_session.flush()
    r2 = await _write_call(db_session)
    assert r2.action == "skipped"
    assert r2.quote_id == r1.quote_id


# ── O-3: correction on changed bid/ask ───────────────────────────────────────

@pytest.mark.asyncio
async def test_write_quote_correction(db_session, source):
    r1 = await _write_call(db_session, bid=5.00, ask=5.10)
    await db_session.flush()

    # Re-stated price triggers correction
    r2 = await _write_call(db_session, bid=5.05, ask=5.15)
    assert r2.action == "corrected"
    assert r2.old_quote_id == r1.quote_id

    old = await db_session.get(OptionQuote, r1.quote_id)
    assert old.is_current is False
    assert old.superseded_by == r2.quote_id

    new = await db_session.get(OptionQuote, r2.quote_id)
    assert new.is_current is True
    assert new.revision_seq == 2
    assert new.bid == pytest.approx(5.05)


# ── O-4/O-5: NaN and None Greeks stored as NULL ───────────────────────────────

@pytest.mark.asyncio
async def test_nan_greek_stored_as_null(db_session, source):
    """NaN float values must be coerced to NULL, not stored as poisoned floats."""
    result = await _write_call(db_session, delta=float("nan"), gamma=float("inf"))
    await db_session.flush()
    quote = await db_session.get(OptionQuote, result.quote_id)
    assert quote.delta is None, "NaN delta must be stored as NULL"
    assert quote.gamma is None, "Inf gamma must be stored as NULL"


@pytest.mark.asyncio
async def test_none_greek_stored_as_null(db_session, source):
    """None greeks must remain NULL — never filled from defaults or prior rows."""
    result = await _write_call(db_session, delta=None, gamma=None, theta=None, vega=None)
    await db_session.flush()
    quote = await db_session.get(OptionQuote, result.quote_id)
    assert quote.delta is None
    assert quote.gamma is None
    assert quote.theta is None
    assert quote.vega is None


# ── O-15: No Greek forward-fill across snapshots ─────────────────────────────

@pytest.mark.asyncio
async def test_no_greek_forward_fill_across_snapshots(db_session, source):
    """
    If snapshot 1 has delta=0.45 and snapshot 2 has delta=None,
    the stored delta for snapshot 2 must be NULL — never 0.45.
    """
    t1 = _NOW
    t2 = _NOW + timedelta(minutes=5)

    r1 = await _write_call(db_session, available_at=t1, delta=0.45)
    await db_session.flush()

    r2 = await _write_call(db_session, available_at=t2, delta=None)
    await db_session.flush()

    q1 = await db_session.get(OptionQuote, r1.quote_id)
    q2 = await db_session.get(OptionQuote, r2.quote_id)

    assert q1.delta == pytest.approx(0.45)
    assert q2.delta is None, (
        "Delta from snapshot 1 must NOT be propagated to snapshot 2. "
        "This violates the no-forward-fill invariant."
    )


# ── O-16: spread_pct None when bid/ask missing ────────────────────────────────

@pytest.mark.asyncio
async def test_spread_pct_none_when_bid_missing(db_session, source):
    result = await _write_call(db_session, bid=None, ask=5.10)
    await db_session.flush()
    quote = await db_session.get(OptionQuote, result.quote_id)
    assert quote.spread_pct is None, "spread_pct must be NULL when bid is missing"


@pytest.mark.asyncio
async def test_spread_pct_computed_when_both_present(db_session, source):
    result = await _write_call(db_session, bid=5.00, ask=5.10)
    await db_session.flush()
    quote = await db_session.get(OptionQuote, result.quote_id)
    assert quote.spread_pct is not None
    assert quote.spread_pct > 0


# ── O-17: is_illiquid flag ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_is_illiquid_zero_oi(db_session, source):
    result = await _write_call(db_session, open_interest=0)
    await db_session.flush()
    quote = await db_session.get(OptionQuote, result.quote_id)
    assert quote.is_illiquid is True


@pytest.mark.asyncio
async def test_is_illiquid_false_normal(db_session, source):
    result = await _write_call(db_session, bid=5.00, ask=5.10, open_interest=1000)
    await db_session.flush()
    quote = await db_session.get(OptionQuote, result.quote_id)
    assert quote.is_illiquid is False


# ── O-8: get_latest_chain_pit cutoff ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_latest_chain_pit_cutoff(db_session, source):
    t1 = _NOW
    t2 = _NOW + timedelta(hours=1)

    await _write_call(db_session, available_at=t1)
    await write_quote(
        db_session,
        underlying_symbol=_SYMBOL, expiry=_EXPIRY, strike=505.0, option_type="call",
        source_id=_SRC, available_at=t2,
        bid=4.0, ask=4.10,
    )
    await db_session.flush()

    # As-of between t1 and t2 → only first quote visible
    response = await get_latest_chain_pit(
        db_session, _SYMBOL, as_of_utc=t1 + timedelta(minutes=30)
    )
    assert len(response.rows) == 1
    assert response.rows[0].strike == _STRIKE

    # As-of after t2 → both visible
    response2 = await get_latest_chain_pit(
        db_session, _SYMBOL, as_of_utc=t2 + timedelta(minutes=30)
    )
    assert len(response2.rows) == 2


# ── O-9: get_latest_chain_pit excludes superseded ────────────────────────────

@pytest.mark.asyncio
async def test_get_latest_chain_pit_excludes_superseded(db_session, source):
    r1 = await _write_call(db_session, available_at=_NOW, bid=5.00, ask=5.10)
    await db_session.flush()
    # Correction — same available_at, different bid
    r2 = await _write_call(db_session, available_at=_NOW, bid=5.05, ask=5.15)
    await db_session.flush()

    response = await get_latest_chain_pit(
        db_session, _SYMBOL, as_of_utc=_NOW + timedelta(hours=1)
    )
    assert len(response.rows) == 1
    assert response.rows[0].bid == pytest.approx(5.05)
    assert response.rows[0].revision_seq == 2


# ── O-10: expiry filter ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_latest_chain_pit_expiry_filter(db_session, source):
    other_expiry = "2024-09-20"

    await _write_call(db_session, available_at=_NOW)
    await write_quote(
        db_session,
        underlying_symbol=_SYMBOL, expiry=other_expiry, strike=_STRIKE,
        option_type="call", source_id=_SRC, available_at=_NOW,
        bid=6.0, ask=6.10,
    )
    await db_session.flush()

    response = await get_latest_chain_pit(
        db_session, _SYMBOL,
        as_of_utc=_NOW + timedelta(hours=1),
        expiry=_EXPIRY,
    )
    assert all(r.expiry == _EXPIRY for r in response.rows)
    assert len(response.rows) == 1


# ── O-11: OptionChainResponse missingness counts ──────────────────────────────

@pytest.mark.asyncio
async def test_chain_response_missingness_counts(db_session, source):
    # Write one quote with all greeks, one with no greeks, one illiquid
    await write_quote(
        db_session,
        underlying_symbol=_SYMBOL, expiry=_EXPIRY, strike=500.0, option_type="call",
        source_id=_SRC, available_at=_NOW,
        bid=5.0, ask=5.1,
        delta=0.45, gamma=0.02, theta=-0.05, vega=0.3,
    )
    await write_quote(
        db_session,
        underlying_symbol=_SYMBOL, expiry=_EXPIRY, strike=505.0, option_type="call",
        source_id=_SRC, available_at=_NOW,
        bid=3.0, ask=3.1,
        delta=None, gamma=None, theta=None, vega=None,  # all greeks missing
    )
    await write_quote(
        db_session,
        underlying_symbol=_SYMBOL, expiry=_EXPIRY, strike=510.0, option_type="call",
        source_id=_SRC, available_at=_NOW,
        bid=0.0, ask=0.0, open_interest=0,  # illiquid
    )
    await db_session.flush()

    response = await get_latest_chain_pit(
        db_session, _SYMBOL, as_of_utc=_NOW + timedelta(hours=1)
    )

    assert response.total == 3
    # Row 2 (no greeks) and Row 3 (illiquid, also no greeks passed) both lack complete greeks
    assert response.missing_greeks_count == 2
    assert response.illiquid_count == 1

    # Only first row has all greeks complete
    complete_rows = [r for r in response.rows if r.greeks_complete]
    assert len(complete_rows) == 1
    assert complete_rows[0].strike == 500.0


# ── O-12/O-13: OptionChainIngestionService.snapshot ──────────────────────────

def _make_option_contract(strike, option_type="call", delta=None):
    return OptionContract(
        strike=strike,
        expiry=_EXPIRY,
        option_type=option_type,
        bid=5.0,
        ask=5.10,
        mid=5.05,
        implied_volatility=0.20,
        delta=delta,
        gamma=0.02,
        theta=-0.05,
        vega=0.30,
        volume=100,
        open_interest=500,
    )


def _make_chain_snapshot(symbol=_SYMBOL, calls=None, puts=None):
    if calls is None:
        calls = [_make_option_contract(500.0, "call", delta=0.45)]
    if puts is None:
        puts = [_make_option_contract(500.0, "put", delta=-0.55)]
    return OptionsChainSnapshot(
        symbol=symbol,
        spot=500.0,
        expiry=_EXPIRY,
        calls=calls,
        puts=puts,
        atm_iv=0.20,
        iv_rank=0.45,
    )


@pytest.mark.asyncio
async def test_snapshot_writes_all_contracts(db_session, source):
    provider = MagicMock()
    provider.get_expirations = AsyncMock(return_value=[_EXPIRY])
    provider.get_chain = AsyncMock(return_value=_make_chain_snapshot())

    svc = OptionChainIngestionService(provider, source_id=_SRC)
    result = await svc.snapshot(db_session, _SYMBOL)

    assert result.success
    assert result.quotes_inserted == 2  # 1 call + 1 put
    assert result.quotes_skipped == 0
    assert result.batch_id is not None
    assert _EXPIRY in result.expiries_fetched


@pytest.mark.asyncio
async def test_snapshot_counts_missing_greeks(db_session, source):
    # Contracts with missing delta (common in yfinance for deep OTM)
    calls = [
        _make_option_contract(500.0, "call", delta=0.45),
        _make_option_contract(600.0, "call", delta=None),  # missing greek
    ]
    chain = _make_chain_snapshot(calls=calls, puts=[])

    provider = MagicMock()
    provider.get_expirations = AsyncMock(return_value=[_EXPIRY])
    provider.get_chain = AsyncMock(return_value=chain)

    svc = OptionChainIngestionService(provider, source_id=_SRC)
    result = await svc.snapshot(db_session, _SYMBOL)

    assert result.quotes_inserted == 2
    assert result.quotes_missing_greeks == 1


# ── O-14: provider error marks batch failed ───────────────────────────────────

@pytest.mark.asyncio
async def test_snapshot_provider_error_marks_batch_failed(db_session, source):
    from sqlalchemy import select as sa_select
    from app.models.market_data import BarIngestBatch

    provider = MagicMock()
    provider.get_expirations = AsyncMock(return_value=[_EXPIRY])
    provider.get_chain = AsyncMock(side_effect=ProviderError("timeout"))

    svc = OptionChainIngestionService(provider, source_id=_SRC, max_retries=1)

    with pytest.raises(ProviderError):
        await svc.snapshot(db_session, _SYMBOL)

    batches = (await db_session.execute(
        sa_select(BarIngestBatch).where(BarIngestBatch.source_id == _SRC)
    )).scalars().all()
    assert any(b.status == "failed" for b in batches)
