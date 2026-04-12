"""
OptionSnapshot — persisted options chain row with full timestamp audit trail.

Timestamp semantics:
  snapshot_time     — availability_time: when the options data was captured from the
                      source API. This is the timestamp used for all join operations.
                      INVARIANT: snapshot_time must be <= bar_open_time of any bar
                      this snapshot is joined to, to prevent lookahead.
  ingested_at       — ingestion_time: wall-clock time the row was written to the DB.
  source            — source_id: "yfinance", "polygon", "demo", etc.
  staleness_seconds — seconds elapsed between the bar's close_time and snapshot_time.
                      Nonzero means the options data lagged behind the price bar.
                      Used to flag joins where stale Greeks may affect feature quality.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean,
    UniqueConstraint, Index,
)
from app.core.database import Base


class OptionSnapshot(Base):
    __tablename__ = "option_snapshots"

    id = Column(Integer, primary_key=True, index=True)

    # --- Instrument identity ---
    underlying_symbol = Column(String(20), nullable=False)
    option_symbol = Column(String(50), nullable=True)   # OCC symbol if available
    expiry = Column(String(12), nullable=False)          # "YYYY-MM-DD"
    strike = Column(Float, nullable=False)
    option_type = Column(String(4), nullable=False)      # "call" or "put"

    # --- Timestamp audit trail ---
    snapshot_time = Column(DateTime, nullable=False)     # availability_time (UTC)
    ingested_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    source = Column(String(50), nullable=True)           # source_id
    # Seconds between the reference bar's close_time and snapshot_time.
    # Set at join time; None means no bar has been joined yet.
    staleness_seconds = Column(Float, nullable=True)

    # --- Market data ---
    underlying_price = Column(Float, nullable=True)
    bid = Column(Float, nullable=True)
    ask = Column(Float, nullable=True)
    last = Column(Float, nullable=True)
    volume = Column(Integer, nullable=True)
    open_interest = Column(Integer, nullable=True)
    implied_volatility = Column(Float, nullable=True)

    # --- Greeks ---
    delta = Column(Float, nullable=True)
    gamma = Column(Float, nullable=True)
    theta = Column(Float, nullable=True)
    vega = Column(Float, nullable=True)
    rho = Column(Float, nullable=True)

    # --- Derived chain aggregates (optional, computed at ingest time) ---
    iv_rank = Column(Float, nullable=True)              # current IV vs 52-week range
    iv_skew = Column(Float, nullable=True)              # OTM put IV - OTM call IV
    pc_volume_ratio = Column(Float, nullable=True)      # put vol / call vol
    pc_oi_ratio = Column(Float, nullable=True)          # put OI / call OI
    gamma_exposure = Column(Float, nullable=True)       # GEX: sum(gamma * OI * 100 * spot)

    # --- Staleness / quality flags ---
    # True if bid==ask==0 or OI==0 (illiquid/no-print contract)
    is_illiquid = Column(Boolean, default=False, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "underlying_symbol", "expiry", "strike", "option_type", "snapshot_time",
            name="uq_option_snapshot",
        ),
        Index("ix_opt_snap_symbol_time", "underlying_symbol", "snapshot_time"),
        Index("ix_opt_snap_expiry", "underlying_symbol", "expiry", "snapshot_time"),
    )
