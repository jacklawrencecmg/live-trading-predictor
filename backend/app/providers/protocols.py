"""
Provider protocols — runtime abstraction layer.

Every external system the application depends on at runtime is represented
here as a typing.Protocol. New vendors are added by implementing the protocol,
not by forking existing code.

Protocols are structural (duck-typed): any class implementing the required
methods satisfies the protocol without inheriting from it.

Four provider categories:
  MarketDataProvider    — OHLCV bars, live quotes, streaming ticks
  OptionsChainProvider  — options chains, expirations, IV surface
  BrokerProvider        — order submission, position/account queries
  AlertChannel          — outbound notification dispatch

See docs/mcp-integration-plan.md for the full integration architecture.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator, List, Optional, Protocol, runtime_checkable

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Shared value types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Quote:
    symbol: str
    price: float
    bid: float
    ask: float
    volume: float
    change: float
    change_pct: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OptionContract:
    strike: float
    expiry: str                  # ISO date string "YYYY-MM-DD"
    option_type: str             # "call" | "put"
    bid: float
    ask: float
    mid: float
    implied_volatility: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    open_interest: Optional[int] = None
    volume: Optional[int] = None


@dataclass
class OptionsChainSnapshot:
    symbol: str
    spot: float
    expiry: str
    calls: List[OptionContract]
    puts: List[OptionContract]
    atm_iv: float
    iv_rank: Optional[float] = None
    snapshot_time: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OrderRequest:
    symbol: str
    side: str                    # "buy" | "sell"
    quantity: float
    order_type: str = "market"   # "market" | "limit"
    limit_price: Optional[float] = None
    time_in_force: str = "day"   # "day" | "gtc" | "ioc"
    client_order_id: Optional[str] = None


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    status: str                  # "filled" | "pending" | "rejected" | "cancelled"
    filled_qty: float
    avg_fill_price: Optional[float]
    submitted_at: datetime
    filled_at: Optional[datetime] = None
    reject_reason: Optional[str] = None


@dataclass
class AccountInfo:
    account_id: str
    cash: float
    portfolio_value: float
    buying_power: float
    currency: str = "USD"
    is_paper: bool = True


@dataclass
class AlertPayload:
    alert_type: str
    title: str
    message: str
    severity: str                # "info" | "warning" | "critical"
    symbol: Optional[str] = None
    details: Optional[dict] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# MarketDataProvider
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class MarketDataProvider(Protocol):
    """
    Source of OHLCV bars and live quotes.

    Implementations:
        YFinanceMarketDataProvider  — current default (free, rate-limited)
        PolygonMarketDataProvider   — production grade, requires API key
        AlpacaMarketDataProvider    — via Alpaca Data API v2, requires key
    """

    async def get_candles(
        self,
        symbol: str,
        interval: str,          # "1m" | "5m" | "15m" | "1h" | "1d"
        period: str,            # e.g. "5d", "60d", "1y"
    ) -> pd.DataFrame:
        """
        Return a DataFrame with columns: open, high, low, close, volume,
        bar_open_time (datetime). Sorted ascending by bar_open_time.
        Raises ProviderError on failure.
        """
        ...

    async def get_quote(self, symbol: str) -> Quote:
        """Return the latest quote for symbol. Raises ProviderError on failure."""
        ...

    async def stream_quotes(
        self,
        symbol: str,
        interval_seconds: float = 15.0,
    ) -> AsyncIterator[Quote]:
        """
        Yield quotes at approximately interval_seconds frequency.
        Implementors that have a real push feed should ignore interval_seconds
        and yield on each incoming tick. Poll-based implementations use it as
        the sleep interval.
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# OptionsChainProvider
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class OptionsChainProvider(Protocol):
    """
    Source of live options chain data including strikes, greeks, and IV.

    Implementations:
        YFinanceOptionsProvider   — current default (free, delayed, limited greeks)
        TDAmeritradeOptionsProvider
        IBKROptionsProvider
        TastytradeOptionsProvider
    """

    async def get_expirations(self, symbol: str) -> List[str]:
        """Return available expiry dates as ISO strings, sorted ascending."""
        ...

    async def get_chain(
        self,
        symbol: str,
        expiry: Optional[str] = None,   # None → nearest expiry
    ) -> OptionsChainSnapshot:
        """
        Return the full chain for a single expiry.
        Raises ProviderError if symbol or expiry is not found.
        """
        ...

    async def get_atm_iv(self, symbol: str) -> float:
        """
        Return the at-the-money implied volatility as a decimal (0.20 = 20%).
        Convenience method; implementors may call get_chain internally.
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# BrokerProvider
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class BrokerProvider(Protocol):
    """
    Execution venue: submit orders, query positions and account state.

    Implementations:
        PaperBrokerProvider    — current (DB + Redis, no real fills)
        AlpacaBrokerProvider   — Alpaca paper or live API
        IBKRBrokerProvider     — IBKR via ib_insync or IBKR REST

    The application's risk layer (risk_manager.py) gates every call to
    submit_order. Broker implementations must NOT apply their own risk
    logic — they are pure execution adapters.
    """

    @property
    def is_paper(self) -> bool:
        """True when connected to a paper/sandbox environment."""
        ...

    async def submit_order(self, order: OrderRequest) -> OrderResult:
        """
        Submit an order. Returns immediately with status "pending" if the
        broker is async, or "filled" if synchronously confirmed.
        Raises BrokerError on rejection or connectivity failure.
        """
        ...

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True if cancelled, False if already filled."""
        ...

    async def get_positions(self) -> List[dict]:
        """Return open positions as a list of dicts (symbol, qty, avg_cost, market_value)."""
        ...

    async def get_account(self) -> AccountInfo:
        """Return current account state (cash, portfolio value, buying power)."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# AlertChannel
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class AlertChannel(Protocol):
    """
    Outbound notification channel. Zero or more channels are registered with
    the GovernanceAlertService at startup.

    Implementations:
        NullAlertChannel      — no-op (dev/test default)
        WebhookAlertChannel   — HTTP POST to a URL (generic, works with Zapier etc.)
        SlackAlertChannel     — Slack incoming webhook
        PagerDutyAlertChannel — PagerDuty Events API v2 (critical only)
    """

    async def dispatch(self, alert: AlertPayload) -> None:
        """
        Send the alert. Must not raise — swallow errors and log them.
        Failures in a notification channel must never affect the trading path.
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────

class ProviderError(Exception):
    """Raised when a MarketDataProvider or OptionsChainProvider call fails."""


class BrokerError(Exception):
    """Raised when a BrokerProvider call fails."""
    def __init__(self, message: str, order_id: Optional[str] = None, retryable: bool = False):
        super().__init__(message)
        self.order_id = order_id
        self.retryable = retryable
