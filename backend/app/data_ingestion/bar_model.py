"""
OHLCV bar data models and deduplication logic.

Closed-bar logic:
- A bar is "closed" when its end timestamp has passed relative to server time.
- For a 5m bar starting at 09:30:00, it closes at 09:35:00.
- We only ingest a bar as closed if current_time >= bar_open + interval.
- Partial bars (the current live bar) are stored with is_closed=False and updated.
- Predictions MUST only use bars with is_closed=True to avoid lookahead bias.

Timestamp semantics (per Project Charter):
  bar_open_time      — event_time:        when the price event actually occurred (UTC)
  bar_close_time     — nominal close:     bar_open_time + timeframe duration
  availability_time  — availability_time: when this bar became readable by our pipeline.
                       For historical backfill this equals ingested_at; for live streaming
                       it equals bar_close_time (the moment the bar was complete).
  ingested_at        — ingestion_time:    wall-clock time we wrote the row to the DB.
  source             — source_id:         data vendor ("yfinance", "polygon", "demo", …)
  staleness_flag     — True if the bar was ingested more than one full timeframe period
                       after bar_close_time, indicating a delayed or backfilled row that
                       may not have been available in real-time.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, UniqueConstraint, Index
from app.core.database import Base


class OHLCVBar(Base):
    __tablename__ = "ohlcv_bars"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)  # "1m", "5m", "15m", "1h", "1d"

    # --- Timestamp audit trail ---
    bar_open_time = Column(DateTime, nullable=False)   # event_time: UTC start of bar
    bar_close_time = Column(DateTime, nullable=False)  # nominal close: bar_open + interval
    availability_time = Column(DateTime, nullable=True)  # when data was readable by pipeline
    ingested_at = Column(DateTime, default=datetime.utcnow, nullable=False)  # DB write time
    source = Column(String(50), nullable=True)         # source_id: "yfinance", "polygon", …
    staleness_flag = Column(Boolean, default=False, nullable=False)  # True = delayed ingest

    # --- OHLCV ---
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    vwap = Column(Float, nullable=True)

    # --- Bar state ---
    is_closed = Column(Boolean, default=False, nullable=False)  # True = bar is final

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "bar_open_time", name="uq_bar"),
        Index("ix_bar_symbol_tf_time", "symbol", "timeframe", "bar_open_time"),
    )
