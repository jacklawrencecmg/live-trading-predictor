"""
YFinanceOptionsProvider — wraps options_service.py's chain fetching.

Limitations (document explicitly so future engineers know what to replace):
  - IV is self-computed via Brent's method, not sourced from exchange
  - Greeks use Black-Scholes approximation, not market greeks
  - Data is delayed (typically 15 min for free feed)
  - No IV surface / term structure — single expiry only per call
  - Bid/ask is wide for illiquid strikes; yfinance does not flag these

Replace with TDAmeritradeOptionsProvider or IBKROptionsProvider for
production-quality IV surface data.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import List, Optional

from .protocols import OptionContract, OptionsChainProvider, OptionsChainSnapshot, ProviderError

logger = logging.getLogger(__name__)


class YFinanceOptionsProvider:
    """Implements OptionsChainProvider using yfinance. Thin wrapper over options_service.py."""

    async def get_expirations(self, symbol: str) -> List[str]:
        loop = asyncio.get_event_loop()
        try:
            import yfinance as yf
            ticker = await loop.run_in_executor(None, yf.Ticker, symbol)
            exps = await loop.run_in_executor(None, lambda: ticker.options)
            return list(exps) if exps else []
        except Exception as exc:
            raise ProviderError(f"yfinance expirations failed for {symbol}: {exc}") from exc

    async def get_chain(
        self,
        symbol: str,
        expiry: Optional[str] = None,
    ) -> OptionsChainSnapshot:
        """
        Delegates to the existing options_service._build_chain() logic.
        Wraps the result into the OptionsChainSnapshot protocol type.
        """
        loop = asyncio.get_event_loop()
        try:
            from app.services.options_service import _build_chain
            raw = await loop.run_in_executor(None, _build_chain, symbol, expiry)
        except Exception as exc:
            raise ProviderError(f"yfinance options chain failed for {symbol}: {exc}") from exc

        # _build_chain returns app.schemas.options.OptionsChain — adapt to protocol type
        calls = [
            OptionContract(
                strike=c.strike,
                expiry=raw.expiry,
                option_type="call",
                bid=c.bid,
                ask=c.ask,
                mid=c.mid,
                implied_volatility=c.implied_volatility,
                delta=c.delta,
                gamma=c.gamma,
                theta=c.theta,
                vega=c.vega,
                open_interest=c.open_interest,
                volume=c.volume,
            )
            for c in (raw.calls or [])
        ]
        puts = [
            OptionContract(
                strike=p.strike,
                expiry=raw.expiry,
                option_type="put",
                bid=p.bid,
                ask=p.ask,
                mid=p.mid,
                implied_volatility=p.implied_volatility,
                delta=p.delta,
                gamma=p.gamma,
                theta=p.theta,
                vega=p.vega,
                open_interest=p.open_interest,
                volume=p.volume,
            )
            for p in (raw.puts or [])
        ]

        return OptionsChainSnapshot(
            symbol=symbol,
            spot=raw.spot,
            expiry=raw.expiry,
            calls=calls,
            puts=puts,
            atm_iv=raw.atm_iv,
            iv_rank=raw.iv_rank,
        )

    async def get_atm_iv(self, symbol: str) -> float:
        try:
            chain = await self.get_chain(symbol)
            return chain.atm_iv
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"get_atm_iv failed for {symbol}: {exc}") from exc
