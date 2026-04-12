"""
Null / stub provider implementations for tests.

Import these in test fixtures instead of the live providers. They return
deterministic data and never make network calls.

Usage:
    from app.providers.null_providers import (
        NullMarketDataProvider, NullOptionsProvider, NullBrokerProvider
    )
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import AsyncIterator, List, Optional

import numpy as np
import pandas as pd

from .protocols import (
    AccountInfo, OptionContract, OptionsChainSnapshot,
    OrderRequest, OrderResult, ProviderError, Quote,
)


class NullMarketDataProvider:
    """
    Returns deterministic synthetic bars and quotes.
    No network calls. Suitable for unit tests and CI.
    """

    def __init__(self, base_price: float = 100.0, seed: int = 42):
        self.base_price = base_price
        self._rng = np.random.default_rng(seed)

    async def get_candles(self, symbol: str, interval: str, period: str) -> pd.DataFrame:
        n = 200
        prices = self.base_price + self._rng.normal(0, 1, n).cumsum()
        opens  = prices + self._rng.normal(0, 0.1, n)
        highs  = np.maximum(opens, prices) + self._rng.uniform(0, 0.5, n)
        lows   = np.minimum(opens, prices) - self._rng.uniform(0, 0.5, n)
        vols   = self._rng.integers(10_000, 1_000_000, n).astype(float)
        times  = [datetime(2024, 1, 2) + timedelta(minutes=5 * i) for i in range(n)]
        return pd.DataFrame({
            "bar_open_time": times,
            "open": opens, "high": highs, "low": lows,
            "close": prices, "volume": vols,
        })

    async def get_quote(self, symbol: str) -> Quote:
        return Quote(
            symbol=symbol,
            price=self.base_price,
            bid=self.base_price - 0.01,
            ask=self.base_price + 0.01,
            volume=500_000.0,
            change=0.5,
            change_pct=0.5,
        )

    async def stream_quotes(
        self, symbol: str, interval_seconds: float = 1.0
    ) -> AsyncIterator[Quote]:
        while True:
            yield await self.get_quote(symbol)
            await asyncio.sleep(interval_seconds)


class NullOptionsProvider:
    """Returns a synthetic options chain. No network calls."""

    def __init__(self, base_price: float = 100.0, atm_iv: float = 0.20):
        self.base_price = base_price
        self.atm_iv = atm_iv

    async def get_expirations(self, symbol: str) -> List[str]:
        base = datetime(2024, 1, 5)
        return [(base + timedelta(days=7 * i)).strftime("%Y-%m-%d") for i in range(8)]

    async def get_chain(
        self, symbol: str, expiry: Optional[str] = None
    ) -> OptionsChainSnapshot:
        exps = await self.get_expirations(symbol)
        chosen = expiry if (expiry and expiry in exps) else exps[0]
        strikes = [self.base_price * m for m in [0.90, 0.95, 0.975, 1.0, 1.025, 1.05, 1.10]]
        calls = [
            OptionContract(
                strike=s, expiry=chosen, option_type="call",
                bid=max(0.01, self.base_price - s + 1.5),
                ask=max(0.02, self.base_price - s + 1.7),
                mid=max(0.015, self.base_price - s + 1.6),
                implied_volatility=self.atm_iv + abs(s - self.base_price) / self.base_price * 0.1,
                delta=max(0.01, min(0.99, 0.5 - (s - self.base_price) / self.base_price * 5)),
            )
            for s in strikes
        ]
        puts = [
            OptionContract(
                strike=s, expiry=chosen, option_type="put",
                bid=max(0.01, s - self.base_price + 1.5),
                ask=max(0.02, s - self.base_price + 1.7),
                mid=max(0.015, s - self.base_price + 1.6),
                implied_volatility=self.atm_iv + abs(s - self.base_price) / self.base_price * 0.1,
                delta=min(-0.01, max(-0.99, -0.5 + (s - self.base_price) / self.base_price * 5)),
            )
            for s in strikes
        ]
        return OptionsChainSnapshot(
            symbol=symbol, spot=self.base_price, expiry=chosen,
            calls=calls, puts=puts, atm_iv=self.atm_iv, iv_rank=0.45,
        )

    async def get_atm_iv(self, symbol: str) -> float:
        return self.atm_iv


class NullBrokerProvider:
    """
    Records submitted orders in memory. Never touches DB or Redis.
    Useful for testing the order path in isolation.
    """

    def __init__(self, reject_symbols: Optional[List[str]] = None):
        self._reject = set(reject_symbols or [])
        self.submitted: List[OrderResult] = []
        self._id = 0

    @property
    def is_paper(self) -> bool:
        return True

    async def submit_order(self, order: OrderRequest) -> OrderResult:
        if order.symbol in self._reject:
            from .protocols import BrokerError
            raise BrokerError(f"NullBrokerProvider: {order.symbol} is in reject list")
        self._id += 1
        result = OrderResult(
            order_id=f"null-{self._id}",
            symbol=order.symbol,
            status="filled",
            filled_qty=order.quantity,
            avg_fill_price=100.0,
            submitted_at=datetime.utcnow(),
            filled_at=datetime.utcnow(),
        )
        self.submitted.append(result)
        return result

    async def cancel_order(self, order_id: str) -> bool:
        return False

    async def get_positions(self) -> List[dict]:
        return []

    async def get_account(self) -> AccountInfo:
        return AccountInfo(
            account_id="null-account",
            cash=100_000.0,
            portfolio_value=100_000.0,
            buying_power=100_000.0,
        )
