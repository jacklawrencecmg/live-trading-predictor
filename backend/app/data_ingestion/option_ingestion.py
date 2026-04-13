"""
Options chain ingestion service.

Provides snapshot-on-demand and periodic background snapshots of options
chain data against any OptionsChainProvider implementation.

Public API
----------
OptionIngestResult          Summary dataclass returned after each snapshot.
OptionChainIngestionService Main service class.

Snapshot semantics
------------------
Each call to snapshot() fetches the full options chain for a symbol (all
expirations, or a specific expiry) and writes one OptionQuote row per
contract.  The snapshot_time (available_at) is set to datetime.utcnow()
at the start of the call — all rows in a single snapshot share the same
available_at so that L7 joins work correctly.

No forward-fill of Greeks
--------------------------
If a Greek is None/NaN in the provider response, it is stored as NULL.
The ingestion service never propagates a value from a previous snapshot.
Downstream code must check for NULL before computing any Greek-derived
feature.  This is a hard invariant; do not relax it without a signed-off
FALSE_CONFIDENCE_AUDIT entry.

Periodic snapshots
------------------
start_periodic_snapshots creates one asyncio.Task per symbol.  The task
calls snapshot() on the configured interval and logs structured events.
Each iteration uses its own DB session to bound connection hold time and
to isolate row-level failures.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.data_ingestion.bar_store import close_batch, open_batch
from app.data_ingestion.option_store import OptionWriteResult, write_quote
from app.providers.protocols import OptionsChainProvider, OptionsChainSnapshot, ProviderError

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)
UTC = timezone.utc


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class OptionIngestResult:
    """Summary of a single chain snapshot."""
    underlying_symbol: str
    source_id: str
    batch_id: int | None = None
    expiries_fetched: list[str] = field(default_factory=list)
    quotes_inserted: int = 0
    quotes_corrected: int = 0
    quotes_skipped: int = 0
    quotes_missing_greeks: int = 0   # inserted rows where >=1 Greek is None
    quotes_illiquid: int = 0         # inserted rows flagged is_illiquid
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    error: str | None = None

    @property
    def total_written(self) -> int:
        return self.quotes_inserted + self.quotes_corrected

    @property
    def success(self) -> bool:
        return self.error is None


# ── Service ───────────────────────────────────────────────────────────────────

class OptionChainIngestionService:
    """
    Options chain ingestion service.

    Usage
    -----
    provider = YFinanceOptionsProvider()
    svc = OptionChainIngestionService(provider, source_id="yfinance")

    # Single snapshot (all expirations):
    async with AsyncSessionLocal() as session:
        result = await svc.snapshot(session, "SPY")
        await session.commit()

    # Periodic background snapshots every 5 minutes:
    await svc.start_periodic_snapshots("SPY", interval_s=300)
    ...
    await svc.stop_periodic_snapshots()
    """

    def __init__(
        self,
        provider: OptionsChainProvider,
        source_id: str = "yfinance",
        max_retries: int = 3,
    ) -> None:
        self._provider = provider
        self._source_id = source_id
        self._max_retries = max_retries
        self._polling_tasks: dict[str, asyncio.Task] = {}

    # ── Internal helpers ───────────────────────────────────────────────────

    async def _fetch_expirations_with_retry(self, symbol: str) -> list[str]:
        """Return available expiry dates, retrying on transient failures."""
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return await self._provider.get_expirations(symbol)
            except ProviderError as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    await asyncio.sleep(2 ** attempt)
        raise ProviderError(
            f"get_expirations failed after {self._max_retries} attempts "
            f"for {symbol}: {last_exc}"
        )

    async def _fetch_chain_with_retry(
        self, symbol: str, expiry: str | None
    ) -> OptionsChainSnapshot:
        """Fetch a single expiry chain with exponential back-off retry."""
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return await self._provider.get_chain(symbol, expiry)
            except ProviderError as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    delay = 2 ** attempt
                    log.warning(
                        "chain fetch failed symbol=%s expiry=%s attempt=%d/%d "
                        "delay=%ds error=%s",
                        symbol, expiry, attempt, self._max_retries, delay, exc,
                    )
                    await asyncio.sleep(delay)
        raise ProviderError(
            f"get_chain failed after {self._max_retries} attempts "
            f"for {symbol}/{expiry}: {last_exc}"
        )

    async def _ingest_chain_snapshot(
        self,
        session: AsyncSession,
        chain: OptionsChainSnapshot,
        snapshot_time: datetime,
        batch_id: int,
        result: OptionIngestResult,
    ) -> None:
        """Write all contracts from one chain snapshot into the database."""
        contracts = (chain.calls or []) + (chain.puts or [])
        for contract in contracts:
            write_result: OptionWriteResult = await write_quote(
                session,
                underlying_symbol=chain.symbol,
                expiry=contract.expiry,
                strike=contract.strike,
                option_type=contract.option_type,
                source_id=self._source_id,
                available_at=snapshot_time,
                bid=contract.bid,
                ask=contract.ask,
                volume=contract.volume,
                open_interest=contract.open_interest,
                underlying_price=chain.spot,
                implied_volatility=contract.implied_volatility,
                delta=contract.delta,
                gamma=contract.gamma,
                theta=contract.theta,
                vega=contract.vega,
                # rho is not in OptionContract; store None (no forward-fill)
                rho=None,
                iv_rank=chain.iv_rank,
                ingest_batch_id=batch_id,
            )

            if write_result.action == "inserted":
                result.quotes_inserted += 1
                # Count quality issues for inserted rows
                if any(v is None for v in (
                    contract.delta, contract.gamma,
                    contract.theta, contract.vega,
                )):
                    result.quotes_missing_greeks += 1
                if (
                    (contract.bid is None or contract.bid <= 0)
                    and (contract.ask is None or contract.ask <= 0)
                ):
                    result.quotes_illiquid += 1
            elif write_result.action == "corrected":
                result.quotes_corrected += 1
            elif write_result.action == "skipped":
                result.quotes_skipped += 1

    # ── Public interface ───────────────────────────────────────────────────

    async def snapshot(
        self,
        session: AsyncSession,
        symbol: str,
        expiry: str | None = None,
        snapshot_time: datetime | None = None,
    ) -> OptionIngestResult:
        """
        Capture a full options chain snapshot for *symbol*.

        All rows in the snapshot share the same available_at (snapshot_time)
        so that L7 joins between option_quotes and market_bars are correct.

        Parameters
        ----------
        expiry
            Specific expiry to fetch ('YYYY-MM-DD').  None → all expiries.
        snapshot_time
            Override snapshot timestamp (useful for testing).
            Defaults to datetime.utcnow() at call time.

        The session is NOT committed inside; the caller must commit.
        """
        if snapshot_time is None:
            snapshot_time = datetime.utcnow()

        log.info(
            "option snapshot start symbol=%s expiry=%s source=%s",
            symbol, expiry or "all", self._source_id,
        )

        batch_id = await open_batch(session, self._source_id, symbol=symbol)
        result = OptionIngestResult(
            underlying_symbol=symbol,
            source_id=self._source_id,
            batch_id=batch_id,
        )

        try:
            if expiry is not None:
                expirations = [expiry]
            else:
                expirations = await self._fetch_expirations_with_retry(symbol)

            result.expiries_fetched = expirations

            for exp in expirations:
                chain = await self._fetch_chain_with_retry(symbol, exp)
                await self._ingest_chain_snapshot(
                    session, chain, snapshot_time, batch_id, result
                )

            await close_batch(
                session, batch_id,
                rows_written=result.total_written,
                rows_skipped=result.quotes_skipped,
                rows_corrected=result.quotes_corrected,
                status="completed",
            )
        except Exception as exc:
            result.error = str(exc)
            result.completed_at = datetime.utcnow()
            await close_batch(
                session, batch_id,
                rows_written=result.quotes_inserted,
                rows_skipped=result.quotes_skipped,
                rows_corrected=result.quotes_corrected,
                status="failed",
                error_detail=str(exc),
            )
            log.error(
                "option snapshot failed symbol=%s error=%s",
                symbol, exc, exc_info=True,
            )
            raise

        result.completed_at = datetime.utcnow()
        log.info(
            "option snapshot done symbol=%s expiries=%d inserted=%d "
            "corrected=%d skipped=%d missing_greeks=%d illiquid=%d",
            symbol, len(expirations),
            result.quotes_inserted, result.quotes_corrected, result.quotes_skipped,
            result.quotes_missing_greeks, result.quotes_illiquid,
        )
        return result

    async def start_periodic_snapshots(
        self,
        symbol: str,
        interval_s: float = 300.0,
        expiry: str | None = None,
    ) -> None:
        """
        Start a background asyncio.Task that calls snapshot() on *interval_s*.

        A separate DB session is opened for each iteration to bound
        connection hold time and isolate row-level failures.

        Calling this for an already-running symbol is a no-op.
        """
        key = f"{symbol}:{expiry or 'all'}"
        if key in self._polling_tasks and not self._polling_tasks[key].done():
            log.warning("Periodic options snapshot already running for %s", key)
            return

        async def _loop() -> None:
            log.info(
                "Periodic options snapshot started symbol=%s interval=%.0fs",
                symbol, interval_s,
            )
            while True:
                try:
                    async with AsyncSessionLocal() as session:
                        result = await self.snapshot(session, symbol, expiry=expiry)
                        await session.commit()
                    log.info(
                        "options tick symbol=%s inserted=%d missing_greeks=%d",
                        symbol, result.quotes_inserted, result.quotes_missing_greeks,
                    )
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    log.error(
                        "options tick error symbol=%s error=%s",
                        symbol, exc, exc_info=True,
                    )
                    # Back off on persistent errors; keep running
                    await asyncio.sleep(min(interval_s * 2, 600))
                    continue

                await asyncio.sleep(interval_s)

            log.info("Periodic options snapshot stopped symbol=%s", symbol)

        task = asyncio.create_task(_loop(), name=f"opts:{key}")
        self._polling_tasks[key] = task

    async def stop_periodic_snapshots(self, symbol: str | None = None) -> None:
        """
        Cancel periodic snapshot tasks.

        Parameters
        ----------
        symbol
            Stop only the task for this symbol.  None → stop all tasks.
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
                log.info("Cancelled options snapshot task %s", key)
