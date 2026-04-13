"""
OHLCV ingestion service.

Provides historical backfill, one-shot live update, and continuous live
polling against any MarketDataProvider implementation.

Public API
----------
PartialBarError       Exception — inference attempted on a non-final bar.
assert_bar_usable     Guard — raises PartialBarError unless bar is final.
BarIngestResult       Summary dataclass returned after each ingestion run.
BarIngestionService   Main service class.

Guard invariant
---------------
assert_bar_usable must be called by any downstream layer before using a
MarketBar row from this service for feature engineering or inference.  This
is the single enforcement point for NI-1 (no lookahead bias) at the
ingestion boundary.

Live polling
------------
start_live_polling creates an asyncio.Task that calls ingest_latest on the
configured interval.  Each iteration opens and closes its own DB session to
keep latency bounded and to isolate failures.  Structured log entries are
emitted for every iteration outcome so the polling health is observable.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.data_ingestion.bar_store import BarWriteResult, close_batch, open_batch, write_bar
from app.data_ingestion.session_calendar import (
    BAR_CLOSE_BUFFER_S,
    TIMEFRAME_SECONDS,
    compute_available_at,
    is_bar_closed,
    is_within_session,
)
from app.providers.protocols import MarketDataProvider, ProviderError

if TYPE_CHECKING:
    from app.models.market_data import MarketBar

log = logging.getLogger(__name__)
UTC = timezone.utc

# Seconds we consider a PARTIAL bar stale enough to warn (not error).
_STALE_PARTIAL_WARN_S: int = 3600


# ── Guard ─────────────────────────────────────────────────────────────────────

class PartialBarError(ValueError):
    """
    Raised when inference or feature engineering is attempted on a bar that
    is not yet finalised (bar_status = PARTIAL or INVALID).

    Set allow_partial=True on assert_bar_usable only in controlled contexts
    such as live quote display — never in training pipelines.
    """

    def __init__(
        self,
        bar_status: str,
        symbol: str,
        timeframe: str,
        event_time: datetime,
    ) -> None:
        self.bar_status = bar_status
        self.symbol = symbol
        self.timeframe = timeframe
        self.event_time = event_time
        super().__init__(
            f"Bar {symbol}/{timeframe} @ {event_time} has status "
            f"'{bar_status}' — use only CLOSED or BACKFILLED bars for inference. "
            f"Pass allow_partial=True to bypass (live display only)."
        )


def assert_bar_usable(
    bar: "MarketBar",
    *,
    allow_partial: bool = False,
) -> None:
    """
    Raise PartialBarError if bar is not safe for feature engineering.

    A bar is safe when bar_status in ('CLOSED', 'BACKFILLED', 'CORRECTED').
    PARTIAL and INVALID bars are unsafe; CORRECTED bars are superseded rows
    that should not appear in normal read paths but are safe if present.

    Parameters
    ----------
    bar
        MarketBar ORM row to validate.
    allow_partial
        If True, PARTIAL bars pass this guard.  Use only for live quote
        display contexts.  Never set True in training or inference paths.
    """
    unsafe = {"PARTIAL", "INVALID"}
    if bar.bar_status in unsafe and not allow_partial:
        raise PartialBarError(
            bar_status=bar.bar_status,
            symbol=bar.symbol,
            timeframe=bar.timeframe,
            event_time=bar.event_time,
        )


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class BarIngestResult:
    """Summary of a single ingestion run (backfill or live update)."""
    symbol: str
    timeframe: str
    source_id: str
    batch_id: int | None = None
    bars_inserted: int = 0
    bars_updated_partial: int = 0
    bars_corrected: int = 0
    bars_skipped: int = 0
    bars_partial: int = 0     # PARTIAL bars written (current live bar)
    bars_out_of_session: int = 0  # rows filtered by session calendar
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    error: str | None = None

    @property
    def total_fetched(self) -> int:
        return (self.bars_inserted + self.bars_updated_partial +
                self.bars_corrected + self.bars_skipped +
                self.bars_partial + self.bars_out_of_session)

    @property
    def success(self) -> bool:
        return self.error is None


# ── DataFrame → bar_data ──────────────────────────────────────────────────────

def _parse_df_row(
    row: pd.Series,
    bar_open: datetime,
    symbol: str,
    timeframe: str,
    source_id: str,
    typical_delay_s: int,
    now_utc: datetime,
    include_extended_hours: bool,
) -> dict | None:
    """
    Convert a single DataFrame row into a keyword-argument dict suitable for
    write_bar, or return None if the bar should be skipped (outside session,
    bad data).

    Returns None for bars outside the NYSE session window.
    """
    # Session filter
    if not is_within_session(bar_open, include_extended_hours=include_extended_hours):
        return None

    # Determine bar lifecycle status
    closed = is_bar_closed(bar_open, timeframe, now_utc=now_utc,
                           buffer_s=BAR_CLOSE_BUFFER_S)
    bar_status = "CLOSED" if closed else "PARTIAL"

    # available_at: only set for closed bars (PARTIAL bars are not yet usable)
    available_at: datetime | None = None
    if closed:
        available_at = compute_available_at(bar_open, timeframe, typical_delay_s)

    # VWAP approximation when not supplied by provider: (H+L+C)/3
    vwap: float | None = None
    if "vwap" in row.index and pd.notna(row["vwap"]):
        vwap = float(row["vwap"])
    elif all(k in row.index for k in ("high", "low", "close")):
        vwap = (float(row["high"]) + float(row["low"]) + float(row["close"])) / 3

    trade_count: int | None = None
    if "trade_count" in row.index and pd.notna(row.get("trade_count")):
        trade_count = int(row["trade_count"])

    return dict(
        symbol=symbol,
        timeframe=timeframe,
        event_time=bar_open,
        source_id=source_id,
        bar_status=bar_status,
        open_=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row["volume"]),
        vwap=vwap,
        trade_count=trade_count,
        available_at=available_at,
    )


# ── Service ───────────────────────────────────────────────────────────────────

class BarIngestionService:
    """
    OHLCV ingestion service.

    Usage
    -----
    provider = YFinanceMarketDataProvider()
    svc = BarIngestionService(provider, source_id="yfinance", typical_delay_s=900)

    # Historical backfill:
    async with AsyncSessionLocal() as session:
        result = await svc.backfill(session, "SPY", "5m", period="60d")
        await session.commit()

    # Start continuous live polling (creates a background asyncio.Task):
    await svc.start_live_polling("SPY", "5m", poll_interval_s=60)
    ...
    await svc.stop_live_polling()
    """

    def __init__(
        self,
        provider: MarketDataProvider,
        source_id: str = "yfinance",
        typical_delay_s: int = 900,
    ) -> None:
        self._provider = provider
        self._source_id = source_id
        self._typical_delay_s = typical_delay_s
        self._polling_tasks: dict[str, asyncio.Task] = {}

    # ── Internal helpers ───────────────────────────────────────────────────

    async def _fetch_with_retry(
        self,
        symbol: str,
        timeframe: str,
        period: str,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """
        Fetch candles from the provider with exponential back-off retry.

        Raises ProviderError after max_retries exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                df = await self._provider.get_candles(symbol, timeframe, period)
                if df.empty:
                    log.warning(
                        "fetch symbol=%s tf=%s period=%s returned empty DataFrame "
                        "(attempt %d/%d)",
                        symbol, timeframe, period, attempt, max_retries,
                    )
                    return df
                return df
            except ProviderError as exc:
                last_exc = exc
                if attempt < max_retries:
                    delay = 2 ** attempt
                    log.warning(
                        "fetch failed symbol=%s tf=%s attempt=%d/%d delay=%ds error=%s",
                        symbol, timeframe, attempt, max_retries, delay, exc,
                    )
                    await asyncio.sleep(delay)
                else:
                    log.error(
                        "fetch exhausted retries symbol=%s tf=%s error=%s",
                        symbol, timeframe, exc,
                    )
        raise ProviderError(f"All {max_retries} fetch attempts failed for {symbol}: {last_exc}")

    async def _ingest_df(
        self,
        session: AsyncSession,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        batch_id: int,
        include_extended_hours: bool,
        now_utc: datetime,
    ) -> BarIngestResult:
        """Convert a provider DataFrame into DB writes and return the result."""
        result = BarIngestResult(
            symbol=symbol,
            timeframe=timeframe,
            source_id=self._source_id,
            batch_id=batch_id,
        )

        for ts, row in df.iterrows():
            # Normalise index → naive UTC datetime
            if hasattr(ts, "to_pydatetime"):
                bar_open = ts.to_pydatetime()
            else:
                bar_open = pd.Timestamp(ts).to_pydatetime()
            if bar_open.tzinfo is not None:
                bar_open = bar_open.astimezone(UTC).replace(tzinfo=None)

            bar_data = _parse_df_row(
                row, bar_open, symbol, timeframe,
                self._source_id, self._typical_delay_s,
                now_utc, include_extended_hours,
            )
            if bar_data is None:
                result.bars_out_of_session += 1
                continue

            bar_data["ingest_batch_id"] = batch_id

            write_result: BarWriteResult = await write_bar(session, **bar_data)

            if write_result.action == "inserted":
                if bar_data["bar_status"] == "PARTIAL":
                    result.bars_partial += 1
                else:
                    result.bars_inserted += 1
            elif write_result.action == "updated_partial":
                result.bars_updated_partial += 1
            elif write_result.action == "corrected":
                result.bars_corrected += 1
            elif write_result.action == "skipped":
                result.bars_skipped += 1

        result.completed_at = datetime.utcnow()
        return result

    # ── Public interface ───────────────────────────────────────────────────

    async def backfill(
        self,
        session: AsyncSession,
        symbol: str,
        timeframe: str,
        period: str = "60d",
        max_retries: int = 3,
        include_extended_hours: bool = False,
        now_utc: datetime | None = None,
    ) -> BarIngestResult:
        """
        Backfill historical bars for *symbol* over *period*.

        Opens a BarIngestBatch before writing and closes it on completion or
        failure.  The session is NOT committed here — the caller must commit.

        Parameters
        ----------
        period
            yfinance-compatible period string: '5d', '60d', '1y', etc.
        include_extended_hours
            Include pre-market (04:00–09:30 ET) and after-hours (16:00–20:00)
            bars.  Default False (regular session only).
        now_utc
            Override "now" for deterministic testing.
        """
        if now_utc is None:
            now_utc = datetime.utcnow()

        log.info(
            "backfill start symbol=%s tf=%s period=%s source=%s",
            symbol, timeframe, period, self._source_id,
        )

        batch_id = await open_batch(
            session, self._source_id, symbol=symbol, timeframe=timeframe
        )
        result = BarIngestResult(
            symbol=symbol, timeframe=timeframe,
            source_id=self._source_id, batch_id=batch_id,
        )

        try:
            df = await self._fetch_with_retry(symbol, timeframe, period, max_retries)
            if df.empty:
                await close_batch(
                    session, batch_id,
                    rows_written=0, status="completed",
                    error_detail="provider returned empty DataFrame",
                )
                result.completed_at = datetime.utcnow()
                return result

            result = await self._ingest_df(
                session, df, symbol, timeframe, batch_id,
                include_extended_hours, now_utc,
            )
            await close_batch(
                session, batch_id,
                rows_written=result.bars_inserted + result.bars_updated_partial,
                rows_skipped=result.bars_skipped,
                rows_corrected=result.bars_corrected,
                status="completed",
            )
        except Exception as exc:
            result.error = str(exc)
            result.completed_at = datetime.utcnow()
            await close_batch(
                session, batch_id,
                rows_written=result.bars_inserted,
                rows_skipped=result.bars_skipped,
                rows_corrected=result.bars_corrected,
                status="failed",
                error_detail=str(exc),
            )
            log.error(
                "backfill failed symbol=%s tf=%s error=%s",
                symbol, timeframe, exc, exc_info=True,
            )
            raise

        log.info(
            "backfill done symbol=%s tf=%s inserted=%d updated_partial=%d "
            "corrected=%d skipped=%d out_of_session=%d",
            symbol, timeframe,
            result.bars_inserted, result.bars_updated_partial,
            result.bars_corrected, result.bars_skipped,
            result.bars_out_of_session,
        )
        return result

    async def ingest_latest(
        self,
        session: AsyncSession,
        symbol: str,
        timeframe: str = "5m",
        include_extended_hours: bool = False,
    ) -> BarIngestResult:
        """
        Ingest the last 2 days of bars.  For use in live polling loops.

        This is a convenience wrapper around backfill(period="2d").
        """
        return await self.backfill(
            session, symbol, timeframe,
            period="2d",
            include_extended_hours=include_extended_hours,
        )

    async def start_live_polling(
        self,
        symbol: str,
        timeframe: str,
        poll_interval_s: float = 60.0,
        include_extended_hours: bool = False,
    ) -> None:
        """
        Start a background asyncio.Task that calls ingest_latest repeatedly.

        The task runs until stop_live_polling is called or the event loop
        is shut down.  Each iteration opens its own DB session to bound
        connection hold time.

        Calling start_live_polling for an already-polling symbol is a no-op.
        """
        key = f"{symbol}:{timeframe}"
        if key in self._polling_tasks and not self._polling_tasks[key].done():
            log.warning("Live polling already running for %s", key)
            return

        async def _poll_loop() -> None:
            log.info(
                "Live polling started symbol=%s tf=%s interval=%.0fs",
                symbol, timeframe, poll_interval_s,
            )
            while True:
                try:
                    async with AsyncSessionLocal() as session:
                        result = await self.ingest_latest(
                            session, symbol, timeframe,
                            include_extended_hours=include_extended_hours,
                        )
                        await session.commit()
                    log.info(
                        "poll tick symbol=%s tf=%s inserted=%d skipped=%d partial=%d",
                        symbol, timeframe,
                        result.bars_inserted, result.bars_skipped, result.bars_partial,
                    )
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    log.error(
                        "poll tick error symbol=%s tf=%s error=%s",
                        symbol, timeframe, exc, exc_info=True,
                    )
                    # Back off on errors; keep running
                    await asyncio.sleep(min(poll_interval_s * 2, 300))
                    continue

                await asyncio.sleep(poll_interval_s)

            log.info("Live polling stopped symbol=%s tf=%s", symbol, timeframe)

        task = asyncio.create_task(_poll_loop(), name=f"poll:{key}")
        self._polling_tasks[key] = task

    async def stop_live_polling(self, symbol: str | None = None) -> None:
        """
        Cancel live polling.

        Parameters
        ----------
        symbol
            Stop only the task for this symbol (all timeframes if ambiguous).
            If None, stop all polling tasks.
        """
        if symbol is None:
            keys = list(self._polling_tasks.keys())
        else:
            keys = [k for k in self._polling_tasks if k.startswith(f"{symbol}:")]

        for key in keys:
            task = self._polling_tasks.pop(key, None)
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                log.info("Cancelled poll task %s", key)
