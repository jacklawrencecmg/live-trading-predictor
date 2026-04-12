"""
DB model for per-bar regime labels.

One row per (symbol, timeframe, bar_open_time). The regime label and key
context signals are stored so that:
  - Backtests can segment performance by regime historically
  - The UI can show regime history without re-running detection
  - Regime drift can be monitored over time
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Index, UniqueConstraint
from app.core.database import Base


class RegimeLabel(Base):
    __tablename__ = "regime_labels"

    id            = Column(Integer, primary_key=True, index=True)
    symbol        = Column(String(20), nullable=False)
    timeframe     = Column(String(10), nullable=False)
    bar_open_time = Column(String(32), nullable=False)  # ISO string, matches bar model

    regime = Column(String(30), nullable=False)

    # Context signals (stored for trend analysis and debugging)
    adx_proxy         = Column(Float, nullable=True)
    atr_ratio         = Column(Float, nullable=True)
    volume_ratio      = Column(Float, nullable=True)
    bar_range_ratio   = Column(Float, nullable=True)
    is_abnormal_move  = Column(Boolean, nullable=True)
    abnormal_sigma    = Column(Float, nullable=True)
    trend_direction   = Column(String(10), nullable=True)

    # Thresholds in effect
    confidence_threshold = Column(Float, nullable=True)
    suppressed           = Column(Boolean, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "bar_open_time", name="uq_regime_bar"),
        Index("ix_regime_symbol_time", "symbol", "timeframe", "bar_open_time"),
    )
