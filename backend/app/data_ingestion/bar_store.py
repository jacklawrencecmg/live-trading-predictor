"""
DB write layer for point-in-time-correct OHLCV bar storage.

Public API
----------
BarWriteResult   Per-bar outcome from write_bar.
open_batch       Create a BarIngestBatch session record; returns batch_id.
close_batch      Finalise a batch with row counts and completion status.
write_bar        Idempotent write: insert, skip, or open a correction chain.
get_closed_bars_pit  PIT-safe read filtered by available_at <= as_of_utc.

Idempotency key
---------------
(symbol, timeframe, event_time, source_id) with is_current=TRUE identifies
the live revision.  Duplicate payloads with identical OHLCV are silently
skipped.  Payloads with differing OHLCV open a new revision and mark the
old row superseded.

Correction flow (three-step transaction, caller must commit)
------------------------------------------------------------
1. INSERT new MarketBar(revision_seq = old.revision_seq + 1, is_current=TRUE)
2. UPDATE old MarketBar: is_current=FALSE, superseded_at=now, superseded_by=new.id
3. INSERT BarCorrection with field-level diff JSON
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.market_data import BarCorrection, BarIngestBatch, MarketBar

log = logging.getLogger(__name__)

UTC = timezone.utc

# Relative tolerance for floating-point OHLCV comparison.
# Avoids spurious corrections from harmless float representation drift.
_OHLCV_EPS = 1e-6


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class BarWriteResult:
    """
    Outcome of a single write_bar call.

    action values
    -------------
    inserted        Fresh row; revision_seq = 1.
    updated_partial PARTIAL bar promoted to CLOSED (normal bar close flow).
    corrected       Existing CLOSED/BACKFILLED bar replaced (data correction).
    skipped         Incoming payload identical to current revision; no-op.
    """
    action: Literal["inserted", "updated_partial", "skipped", "corrected"]
    bar_id: int | None = None       # id of the written (or existing) row
    old_bar_id: int | None = None   # id of the superseded row (correction only)


# ── Float comparison ──────────────────────────────────────────────────────────

def _ohlcv_near(a: float | None, b: float | None) -> bool:
    """Return True if two floats are within _OHLCV_EPS relative tolerance."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    denom = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= _OHLCV_EPS * denom


def _bars_differ(existing: MarketBar, incoming: dict) -> bool:
    """
    Return True if any OHLCV field or bar_status differs materially.

    A PARTIAL → CLOSED transition always counts as different (the bar must
    be promoted even if prices happen to match the partial snapshot).
    """
    for attr in ("open", "high", "low", "close", "volume"):
        if not _ohlcv_near(getattr(existing, attr), incoming.get(attr)):
            return True
    if existing.bar_status != incoming.get("bar_status", existing.bar_status):
        return True
    return False


def _changed_fields_json(existing: MarketBar, incoming: dict) -> str:
    """Build a JSON diff of OHLCV fields that changed."""
    diff: dict[str, dict] = {}
    for attr in ("open", "high", "low", "close", "volume"):
        old_val = getattr(existing, attr)
        new_val = incoming.get(attr)
        if not _ohlcv_near(old_val, new_val):
            diff[attr] = {"from": old_val, "to": new_val}
    if existing.bar_status != incoming.get("bar_status", existing.bar_status):
        diff["bar_status"] = {"from": existing.bar_status,
                              "to": incoming.get("bar_status")}
    return json.dumps(diff)


# ── Batch lifecycle ───────────────────────────────────────────────────────────

async def open_batch(
    session: AsyncSession,
    source_id: str,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> int:
    """
    Create a new BarIngestBatch row with status='running'.

    Returns the new batch_id.  The caller is responsible for calling
    close_batch when the run completes or fails.
    """
    batch = BarIngestBatch(
        source_id=source_id,
        symbol=symbol,
        timeframe=timeframe,
        started_at=datetime.utcnow(),
        status="running",
    )
    session.add(batch)
    await session.flush()
    log.debug("Opened batch %d source=%s symbol=%s tf=%s",
              batch.id, source_id, symbol, timeframe)
    return batch.id


async def close_batch(
    session: AsyncSession,
    batch_id: int,
    *,
    rows_written: int,
    rows_skipped: int = 0,
    rows_corrected: int = 0,
    status: Literal["completed", "failed", "rolled_back"] = "completed",
    error_detail: str | None = None,
) -> None:
    """
    Finalise a BarIngestBatch row.

    Call in a try/finally block so failed batches are always marked.
    """
    await session.execute(
        update(BarIngestBatch)
        .where(BarIngestBatch.id == batch_id)
        .values(
            completed_at=datetime.utcnow(),
            rows_written=rows_written,
            rows_skipped=rows_skipped,
            rows_corrected=rows_corrected,
            status=status,
            error_detail=error_detail,
        )
    )
    log.debug("Closed batch %d status=%s written=%d skipped=%d corrected=%d",
              batch_id, status, rows_written, rows_skipped, rows_corrected)


# ── Core write ────────────────────────────────────────────────────────────────

async def write_bar(
    session: AsyncSession,
    *,
    symbol: str,
    timeframe: str,
    event_time: datetime,
    source_id: str,
    bar_status: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    vwap: float | None = None,
    trade_count: int | None = None,
    available_at: datetime | None = None,
    ingest_batch_id: int | None = None,
    split_factor: float = 1.0,
    div_factor: float = 1.0,
    is_adjusted: bool = False,
) -> BarWriteResult:
    """
    Idempotent write of a single OHLCV bar into market_bars.

    Caller must commit (or let the session auto-commit) after calling this
    function.  Do NOT commit inside; the correction chain must be atomic.

    Parameters
    ----------
    event_time
        Bar open time.  Pass as UTC-aware or naive UTC.
    bar_status
        'PARTIAL' | 'CLOSED' | 'BACKFILLED' | 'INVALID'
    available_at
        When this bar became available to consumers.  None for PARTIAL bars
        (not yet finalized).
    """
    now = datetime.utcnow()
    incoming = dict(
        open=open_, high=high, low=low, close=close, volume=volume,
        bar_status=bar_status,
    )

    # ── Compute staleness ──────────────────────────────────────────────────
    staleness_s: float | None = None
    if available_at is not None:
        av_naive = (
            available_at.astimezone(UTC).replace(tzinfo=None)
            if available_at.tzinfo else available_at
        )
        staleness_s = (now - av_naive).total_seconds()

    # ── Look up current revision ───────────────────────────────────────────
    result = await session.execute(
        select(MarketBar).where(
            MarketBar.symbol == symbol,
            MarketBar.timeframe == timeframe,
            MarketBar.event_time == event_time,
            MarketBar.source_id == source_id,
            MarketBar.is_current == True,  # noqa: E712
        )
    )
    existing: MarketBar | None = result.scalar_one_or_none()

    # ── Case 1: fresh insert ───────────────────────────────────────────────
    if existing is None:
        bar = MarketBar(
            symbol=symbol,
            timeframe=timeframe,
            event_time=event_time,
            available_at=available_at,
            ingested_at=now,
            source_id=source_id,
            ingest_batch_id=ingest_batch_id,
            bar_status=bar_status,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            vwap=vwap,
            trade_count=trade_count,
            staleness_s=staleness_s,
            split_factor=split_factor,
            div_factor=div_factor,
            is_adjusted=is_adjusted,
            revision_seq=1,
            is_current=True,
        )
        session.add(bar)
        await session.flush()
        return BarWriteResult(action="inserted", bar_id=bar.id)

    # ── Case 2: identical payload — idempotent no-op ───────────────────────
    if not _bars_differ(existing, incoming):
        return BarWriteResult(action="skipped", bar_id=existing.id)

    # ── Case 3: data changed — open correction chain ───────────────────────
    action: Literal["updated_partial", "corrected"] = (
        "updated_partial" if existing.bar_status == "PARTIAL" else "corrected"
    )
    correction_type = (
        "PARTIAL_TO_FINAL" if action == "updated_partial" else "DATA_ERROR"
    )

    new_bar = MarketBar(
        symbol=symbol,
        timeframe=timeframe,
        event_time=event_time,
        available_at=available_at,
        ingested_at=now,
        source_id=source_id,
        ingest_batch_id=ingest_batch_id,
        bar_status=bar_status,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        vwap=vwap,
        trade_count=trade_count,
        staleness_s=staleness_s,
        split_factor=split_factor,
        div_factor=div_factor,
        is_adjusted=is_adjusted,
        revision_seq=existing.revision_seq + 1,
        is_current=True,
    )
    session.add(new_bar)
    await session.flush()  # materialise new_bar.id before FK reference

    # Supersede old row
    existing.is_current = False
    existing.superseded_at = now
    existing.superseded_by = new_bar.id
    if existing.bar_status not in ("PARTIAL", "INVALID"):
        existing.bar_status = "CORRECTED"

    # Immutable correction ledger entry
    correction = BarCorrection(
        original_bar_id=existing.id,
        replacement_bar_id=new_bar.id,
        corrected_at=now,
        correction_type=correction_type,
        initiated_by="auto_ingest",
        changed_fields_json=_changed_fields_json(existing, incoming),
    )
    session.add(correction)

    log.info(
        "Bar %s action=%s symbol=%s tf=%s event_time=%s rev=%d→%d",
        correction_type, action, symbol, timeframe, event_time,
        existing.revision_seq, new_bar.revision_seq,
    )
    return BarWriteResult(action=action, bar_id=new_bar.id, old_bar_id=existing.id)


# ── Point-in-time read ────────────────────────────────────────────────────────

async def get_closed_bars_pit(
    session: AsyncSession,
    symbol: str,
    timeframe: str,
    as_of_utc: datetime | None = None,
    limit: int = 500,
    source_id: str | None = None,
) -> list[MarketBar]:
    """
    Return closed bars available as of *as_of_utc*.

    This is the correct query for building training sets.  It enforces:

        available_at <= as_of_utc  AND  is_current = TRUE
        AND bar_status IN ('CLOSED', 'BACKFILLED')

    Do NOT use this for PARTIAL bars — they must never appear in feature
    engineering inputs.

    Parameters
    ----------
    as_of_utc
        Training set availability cutoff (UTC).  None → current time.
    source_id
        Restrict to a single data source.  None → all sources.

    Returns bars in ascending event_time order.
    """
    if as_of_utc is None:
        as_of_utc = datetime.utcnow()

    conditions = [
        MarketBar.symbol == symbol,
        MarketBar.timeframe == timeframe,
        MarketBar.is_current == True,  # noqa: E712
        MarketBar.bar_status.in_(["CLOSED", "BACKFILLED"]),
        MarketBar.available_at <= as_of_utc,
    ]
    if source_id is not None:
        conditions.append(MarketBar.source_id == source_id)

    result = await session.execute(
        select(MarketBar)
        .where(*conditions)
        .order_by(MarketBar.event_time.asc())
        .limit(limit)
    )
    return list(result.scalars().all())
