"""
app.providers — vendor-agnostic integration layer.

Runtime providers (replace yfinance calls in production):
    MarketDataProvider    protocols.py
    OptionsChainProvider  protocols.py
    BrokerProvider        protocols.py
    AlertChannel          protocols.py

Current implementations:
    YFinanceMarketDataProvider   yfinance_market.py
    YFinanceOptionsProvider      yfinance_options.py
    PaperBrokerProvider          paper_broker.py
    WebhookAlertChannel          alert_channels.py
    SlackAlertChannel            alert_channels.py
    NullAlertChannel             alert_channels.py

Test stubs (no network calls):
    NullMarketDataProvider       null_providers.py
    NullOptionsProvider          null_providers.py
    NullBrokerProvider           null_providers.py
"""

from .protocols import (
    AccountInfo,
    AlertChannel,
    AlertPayload,
    BrokerError,
    BrokerProvider,
    MarketDataProvider,
    OptionContract,
    OptionsChainProvider,
    OptionsChainSnapshot,
    OrderRequest,
    OrderResult,
    ProviderError,
    Quote,
)

__all__ = [
    "MarketDataProvider", "OptionsChainProvider", "BrokerProvider", "AlertChannel",
    "Quote", "OptionsChainSnapshot", "OptionContract",
    "OrderRequest", "OrderResult", "AccountInfo", "AlertPayload",
    "ProviderError", "BrokerError",
]
