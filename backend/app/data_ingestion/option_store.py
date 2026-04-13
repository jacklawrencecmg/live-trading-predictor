"""
DB write layer for point-in-time-correct options chain storage.

Public API
----------
OptionWriteResult     Per-quote outcome from write_quote.
write_quote           Idempotent write of a single options quote row.
get_latest_chain_pit  PIT-safe read: latest quotes for a symbol as-of a time.
OptionChainRow        Typed read-side value for a single strike row.
OptionChainResponse   Typed read-side response wrapping all rows for a chain.

Idempotency key
---------------
(underlying_symbol, expiry, strike, option_type, available_at, source_id)
with is_current=TRUE identifies the live revision.

Duplicate payloads (same key, same bid/ask/IV) are silently skipped.
Re-stated payloads (same key, different values) open a revision chain.

No forward-fill of Greeks
--------------------------
If a Greek (delta, gamma, theta, vega, rho) is None/NaN on the incoming
payload, write_quote stores NULL — it never copies a prior row's value.
Consumers must treat NULL Greeks as "unknown" and must NOT substitute a
value from a different timestamp.

Stale detection
---------------
At snapshot time, is_stale is set based on available_at age vs a threshold.
The staleness_s column is reserved for join-time computation (option quote
latency relative to a market bar's event_time) and is not set here.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.market_data import OptionQuote

log = logging.getLogger(__name__)
UTC = timezone.utc

# Float tolerance for quote comparison — tighter than OHLCV since prices
# are already in dollars/cents; sub-cent differences should trigger updates.
_QUOTE_EPS = 1e-4

# Staleness threshold: a quote whose available_at is older than this is stale.
# For yfinance (15-min delayed), quotes are "fresh" right after ingestion.
# Set to 1 hour; a quote older than 1h at snapshot time is marked stale.
_STALE_THRESHOLD_S: int = 3_600


# ── Result and read types ─────────────────────────────────────────────────────

@dataclass
class OptionWriteResult:
    """Outcome of a single write_quote call."""
    action: Literal["inserted", "skipped", "corrected"]
    quote_id: int | None = None
    old_quote_id: int | None = None


@dataclass
class OptionChainRow:
    """
    Read-side typed row for a single option contract.

    Greeks may be None — callers must NOT substitute values from a prior
    snapshot.  Check greeks_complete before using any Greek in a formula.
    """
    quote_id: int
    underlying_symbol: str
    expiry: str          # 'YYYY-MM-DD'
    strike: float
    option_type: str     # 'call' | 'put'
    dte: int | None

    bid: float | None
    ask: float | None
    mid: float | None
    last: float | None
    volume: int | None
    open_interest: int | None

    implied_volatility: float | None
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None
    rho: float | None

    spread_pct: float | None
    is_illiquid: bool
    is_stale: bool

    # Derived missingness summary (True iff all of delta/gamma/theta/vega are set)
    greeks_complete: bool

    available_at: datetime
    ingested_at: datetime
    source_id: str
    revision_seq: int


@dataclass
class OptionChainResponse:
    """
    Typed response from get_latest_chain_pit.

    Aggregates missingness and quality statistics across all returned rows
    so callers can decide whether the chain is usable for research.
    """
    underlying_symbol: str
    as_of_time: datetime
    source_id: str | None
    rows: list[OptionChainRow] = field(default_factory=list)

    # Populated after rows are set
    total: int = 0
    stale_count: int = 0
    illiquid_count: int = 0
    missing_greeks_count: int = 0  # rows where greeks_complete=False

    def __post_init__(self) -> None:
        self.total = len(self.rows)
        self.stale_count = sum(1 for r in self.rows if r.is_stale)
        self.illiquid_count = sum(1 for r in self.rows if r.is_illiquid)
        self.missing_greeks_count = sum(1 for r in self.rows if not r.greeks_complete)


# ── Float helpers ─────────────────────────────────────────────────────────────

def _coerce(v: float | None) -> float | None:
    """Return None if v is NaN/inf, otherwise return v."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _quotes_differ(existing: OptionQuote, incoming: dict) -> bool:
    """Return True if any price/IV/Greek field differs materially."""
    for attr in ("bid", "ask", "implied_volatility", "delta", "gamma",
                 "theta", "vega", "volume", "open_interest"):
        a = getattr(existing, attr, None)
        b = incoming.get(attr)
        if a is None and b is None:
            continue
        if a is None or b is None:
            return True
        try:
            if abs(float(a) - float(b)) > _QUOTE_EPS:
                return True
        except (TypeError, ValueError):
            return True
    return False


def _compute_spread_pct(bid: float | None, ask: float | None) -> float | None:
    """
    Proportional spread: (ask - bid) / mid.

    Returns None if either leg is missing, zero, or negative.
    A zero spread on a non-zero mid is valid (but unusual); a negative
    spread indicates bad data and is returned as None.
    """
    b = _coerce(bid)
    a = _coerce(ask)
    if b is None or a is None:
        return None
    mid = (b + a) / 2.0
    if mid <= 0:
        return None
    spread = a - b
    if spread < 0:
        return None
    return spread / mid


def _compute_is_illiquid(
    bid: float | None,
    ask: float | None,
    open_interest: int | None,
) -> bool:
    """
    Mark a quote illiquid when:
    - both bid and ask are effectively zero, OR
    - open_interest is explicitly zero (no open contracts).

    Missing bid/ask is NOT automatically illiquid; the quote may still be
    tradeable at last or estimated mid.
    """
    b = _coerce(bid)
    a = _coerce(ask)
    if b is not None and a is not None and b <= 0 and a <= 0:
        return True
    if open_interest is not None and open_interest == 0:
        return True
    return False


def _compute_dte(expiry: str, as_of: datetime) -> int | None:
    """Days-to-expiry at *as_of* time."""
    try:
        exp_date = date.fromisoformat(expiry)
        ref_date = as_of.date() if isinstance(as_of, datetime) else as_of
        return max((exp_date - ref_date).days, 0)
    except (ValueError, AttributeError):
        return None


# ── Core write ────────────────────────────────────────────────────────────────

async def write_quote(
    session: AsyncSession,
    *,
    underlying_symbol: str,
    expiry: str,
    strike: float,
    option_type: str,
    source_id: str,
    available_at: datetime,
    # Quote data
    bid: float | None = None,
    ask: float | None = None,
    last: float | None = None,
    volume: int | None = None,
    open_interest: int | None = None,
    underlying_price: float | None = None,
    # Greeks and IV — explicit None = unknown, never forward-filled
    implied_volatility: float | None = None,
    delta: float | None = None,
    gamma: float | None = None,
    theta: float | None = None,
    vega: float | None = None,
    rho: float | None = None,
    # Chain-level aggregates
    iv_rank: float | None = None,
    iv_skew: float | None = None,
    pc_volume_ratio: float | None = None,
    pc_oi_ratio: float | None = None,
    gamma_exposure: float | None = None,
    # Metadata
    option_symbol: str | None = None,
    event_time: datetime | None = None,
    ingest_batch_id: int | None = None,
    stale_threshold_s: int = _STALE_THRESHOLD_S,
) -> OptionWriteResult:
    """
    Idempotent write of a single option quote row.

    Greeks (delta, gamma, theta, vega, rho) and IV are stored verbatim —
    if the incoming value is None/NaN, NULL is written to the database.
    Do NOT pass a value from a prior snapshot to fill gaps.

    Caller must commit after this call.  Do NOT commit inside.
    """
    now = datetime.utcnow()

    # Sanitize floats — coerce NaN/inf to None so we never store poison values
    bid = _coerce(bid)
    ask = _coerce(ask)
    last = _coerce(last)
    underlying_price = _coerce(underlying_price)
    implied_volatility = _coerce(implied_volatility)
    delta = _coerce(delta)
    gamma = _coerce(gamma)
    theta = _coerce(theta)
    vega = _coerce(vega)
    rho = _coerce(rho)
    iv_rank = _coerce(iv_rank)
    iv_skew = _coerce(iv_skew)
    pc_volume_ratio = _coerce(pc_volume_ratio)
    pc_oi_ratio = _coerce(pc_oi_ratio)
    gamma_exposure = _coerce(gamma_exposure)

    # Derived quality fields
    mid = ((bid + ask) / 2.0) if (bid is not None and ask is not None) else None
    spread_pct = _compute_spread_pct(bid, ask)
    is_illiquid = _compute_is_illiquid(bid, ask, open_interest)
    dte = _compute_dte(expiry, available_at)

    av_naive = (
        available_at.astimezone(UTC).replace(tzinfo=None)
        if available_at.tzinfo else available_at
    )
    is_stale = (now - av_naive).total_seconds() > stale_threshold_s

    incoming = dict(
        bid=bid, ask=ask, implied_volatility=implied_volatility,
        delta=delta, gamma=gamma, theta=theta, vega=vega,
        volume=volume, open_interest=open_interest,
    )

    # ── Look up current revision ───────────────────────────────────────────
    result = await session.execute(
        select(OptionQuote).where(
            OptionQuote.underlying_symbol == underlying_symbol,
            OptionQuote.expiry == expiry,
            OptionQuote.strike == strike,
            OptionQuote.option_type == option_type,
            OptionQuote.available_at == available_at,
            OptionQuote.source_id == source_id,
            OptionQuote.is_current == True,  # noqa: E712
        )
    )
    existing: OptionQuote | None = result.scalar_one_or_none()

    # ── Case 1: fresh insert ───────────────────────────────────────────────
    if existing is None:
        quote = OptionQuote(
            underlying_symbol=underlying_symbol,
            option_symbol=option_symbol,
            expiry=expiry,
            strike=strike,
            option_type=option_type,
            dte=dte,
            event_time=event_time,
            available_at=available_at,
            ingested_at=now,
            source_id=source_id,
            ingest_batch_id=ingest_batch_id,
            underlying_price=underlying_price,
            bid=bid,
            ask=ask,
            last=last,
            volume=volume,
            open_interest=open_interest,
            implied_volatility=implied_volatility,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            iv_rank=iv_rank,
            iv_skew=iv_skew,
            pc_volume_ratio=pc_volume_ratio,
            pc_oi_ratio=pc_oi_ratio,
            gamma_exposure=gamma_exposure,
            spread_pct=spread_pct,
            is_stale=is_stale,
            staleness_s=None,   # set at join time, not at write time
            is_illiquid=is_illiquid,
            revision_seq=1,
            is_current=True,
        )
        session.add(quote)
        await session.flush()
        return OptionWriteResult(action="inserted", quote_id=quote.id)

    # ── Case 2: identical payload ──────────────────────────────────────────
    if not _quotes_differ(existing, incoming):
        return OptionWriteResult(action="skipped", quote_id=existing.id)

    # ── Case 3: re-stated data — open revision chain ───────────────────────
    new_quote = OptionQuote(
        underlying_symbol=underlying_symbol,
        option_symbol=option_symbol,
        expiry=expiry,
        strike=strike,
        option_type=option_type,
        dte=dte,
        event_time=event_time,
        available_at=available_at,
        ingested_at=now,
        source_id=source_id,
        ingest_batch_id=ingest_batch_id,
        underlying_price=underlying_price,
        bid=bid,
        ask=ask,
        last=last,
        volume=volume,
        open_interest=open_interest,
        implied_volatility=implied_volatility,
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        rho=rho,
        iv_rank=iv_rank,
        iv_skew=iv_skew,
        pc_volume_ratio=pc_volume_ratio,
        pc_oi_ratio=pc_oi_ratio,
        gamma_exposure=gamma_exposure,
        spread_pct=spread_pct,
        is_stale=is_stale,
        staleness_s=None,
        is_illiquid=is_illiquid,
        revision_seq=existing.revision_seq + 1,
        is_current=True,
    )
    session.add(new_quote)
    await session.flush()

    existing.is_current = False
    existing.superseded_at = now
    existing.superseded_by = new_quote.id

    log.info(
        "option correction: %s %s %s %s @ %s rev=%d→%d",
        underlying_symbol, expiry, strike, option_type,
        available_at, existing.revision_seq, new_quote.revision_seq,
    )
    return OptionWriteResult(action="corrected", quote_id=new_quote.id,
                             old_quote_id=existing.id)


# ── Point-in-time read ────────────────────────────────────────────────────────

async def get_latest_chain_pit(
    session: AsyncSession,
    underlying_symbol: str,
    as_of_utc: datetime | None = None,
    expiry: str | None = None,
    source_id: str | None = None,
    min_dte: int = 0,
    max_dte: int | None = None,
) -> OptionChainResponse:
    """
    Return the latest option quotes for *underlying_symbol* as-of *as_of_utc*.

    Enforces the L7 lookahead guard:
        WHERE available_at <= as_of_utc AND is_current = TRUE

    Parameters
    ----------
    as_of_utc
        Training set availability cutoff.  None → current time.
    expiry
        Filter to a specific expiry date ('YYYY-MM-DD').  None → all expiries.
    source_id
        Filter to a specific data source.  None → all sources.
    min_dte / max_dte
        DTE filter applied server-side.  max_dte=None → no upper bound.

    Returns
    -------
    OptionChainResponse
        Typed response with rows, missingness stats, and quality flags.
        The `rows` list is sorted by (expiry, option_type, strike).
    """
    if as_of_utc is None:
        as_of_utc = datetime.utcnow()

    conditions = [
        OptionQuote.underlying_symbol == underlying_symbol,
        OptionQuote.is_current == True,  # noqa: E712
        OptionQuote.available_at <= as_of_utc,
    ]
    if expiry is not None:
        conditions.append(OptionQuote.expiry == expiry)
    if source_id is not None:
        conditions.append(OptionQuote.source_id == source_id)
    if min_dte > 0:
        conditions.append(OptionQuote.dte >= min_dte)
    if max_dte is not None:
        conditions.append(OptionQuote.dte <= max_dte)

    result = await session.execute(
        select(OptionQuote)
        .where(*conditions)
        .order_by(
            OptionQuote.expiry.asc(),
            OptionQuote.option_type.asc(),
            OptionQuote.strike.asc(),
        )
    )
    quotes: list[OptionQuote] = list(result.scalars().all())

    rows = [
        OptionChainRow(
            quote_id=q.id,
            underlying_symbol=q.underlying_symbol,
            expiry=q.expiry,
            strike=q.strike,
            option_type=q.option_type,
            dte=q.dte,
            bid=q.bid,
            ask=q.ask,
            mid=(
                (q.bid + q.ask) / 2.0
                if q.bid is not None and q.ask is not None else None
            ),
            last=q.last,
            volume=q.volume,
            open_interest=q.open_interest,
            implied_volatility=q.implied_volatility,
            delta=q.delta,
            gamma=q.gamma,
            theta=q.theta,
            vega=q.vega,
            rho=q.rho,
            spread_pct=q.spread_pct,
            is_illiquid=q.is_illiquid,
            is_stale=q.is_stale,
            greeks_complete=all(
                v is not None for v in (q.delta, q.gamma, q.theta, q.vega)
            ),
            available_at=q.available_at,
            ingested_at=q.ingested_at,
            source_id=q.source_id,
            revision_seq=q.revision_seq,
        )
        for q in quotes
    ]

    return OptionChainResponse(
        underlying_symbol=underlying_symbol,
        as_of_time=as_of_utc,
        source_id=source_id,
        rows=rows,
    )
