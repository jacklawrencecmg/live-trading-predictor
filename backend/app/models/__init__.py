# Import all models so Base.metadata is fully populated before create_all.
# ── Legacy tables (kept for backward compatibility) ──────────────────────────
from app.models.trade import Trade                          # noqa: F401
from app.models.position import Position                    # noqa: F401
from app.models.audit_log import AuditLog                   # noqa: F401
from app.models.backtest import BacktestResult              # noqa: F401
from app.models.option_snapshot import OptionSnapshot       # noqa: F401
from app.data_ingestion.bar_model import OHLCVBar           # noqa: F401
from app.models.feature_row import FeatureRow               # noqa: F401
from app.models.regime_label import RegimeLabel             # noqa: F401

# ── Point-in-time market data layer (migration 003) ───────────────────────────
from app.models.market_data import (                        # noqa: F401
    MarketDataSource,
    BarIngestBatch,
    MarketBar,
    BarCorrection,
    OptionQuote,
    ResearchSnapshot,
)
