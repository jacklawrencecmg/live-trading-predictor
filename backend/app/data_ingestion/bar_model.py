"""
OHLCV bar data models and deduplication logic.

Closed-bar logic:
- A bar is "closed" when its end timestamp has passed relative to server time.
- For a 5m bar starting at 09:30:00, it closes at 09:35:00.
- We only ingest a bar as closed if current_time >= bar_open + interval.
- Partial bars (the current live bar) are stored with is_closed=False and updated.
- Predictions MUST only use bars with is_closed=True to avoid lookahead bias.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, UniqueConstraint, Index
from app.core.database import Base


class OHLCVBar(Base):
    __tablename__ = "ohlcv_bars"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)  # "1m", "5m", "15m", "1h", "1d"
    bar_open_time = Column(DateTime, nullable=False)  # UTC, start of bar
    bar_close_time = Column(DateTime, nullable=False)  # UTC, end of bar (exclusive)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    vwap = Column(Float, nullable=True)
    is_closed = Column(Boolean, default=False, nullable=False)  # True = bar is final
    source = Column(String(50), nullable=True)  # "yfinance", "polygon", "demo"
    ingested_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "bar_open_time", name="uq_bar"),
        Index("ix_bar_symbol_tf_time", "symbol", "timeframe", "bar_open_time"),
    )
