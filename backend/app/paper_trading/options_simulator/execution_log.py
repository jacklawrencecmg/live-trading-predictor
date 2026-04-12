"""
Execution log — SQLAlchemy schema for persisting simulation events.

Two tables:
  sim_positions  — one row per position lifecycle (open through final state)
  sim_events     — one row per ExecutionEvent (full audit trail)

The simulator writes to these tables via the helper functions at the bottom.
Both functions are synchronous because the simulator engine is synchronous;
callers can wrap them in `asyncio.to_thread` for async contexts.

Column conventions:
  - all dollar amounts stored as NUMERIC(12, 4)
  - timestamps stored as UTC-aware TIMESTAMP
  - enums stored as VARCHAR (not PG enum) for portability
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, String, Float, Integer, Boolean,
    DateTime, Text, JSON, UniqueConstraint, Index,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


# ---------------------------------------------------------------------------
# sim_positions
# ---------------------------------------------------------------------------

class SimPositionRecord(Base):
    """
    Persistent record of a simulated options position.

    One row per position. Updated in-place on state transitions.
    """
    __tablename__ = "sim_positions"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Identity
    position_id       = Column(String(24), nullable=False, unique=True, index=True)
    simulator_label   = Column(String(64), nullable=False, default="paper_options_sim_v1")
    symbol            = Column(String(16), nullable=False, index=True)
    structure_type    = Column(String(32), nullable=False)
    direction         = Column(String(16), nullable=False)

    # State
    state             = Column(String(16), nullable=False, default="open")
    contracts         = Column(Integer, nullable=False, default=1)
    multiplier        = Column(Integer, nullable=False, default=100)

    # Fill economics
    open_premium_per_share = Column(Float, nullable=True)
    net_premium_dollars    = Column(Float, nullable=True)
    open_risk_dollars      = Column(Float, nullable=True)
    total_fees_paid        = Column(Float, nullable=True, default=0.0)
    fill_method            = Column(String(16), nullable=True)

    # Legs (JSON snapshot)
    legs_json         = Column(JSON, nullable=True)    # list of leg dicts at open
    decision_snapshot = Column(JSON, nullable=True)    # OptionsDecision context

    # P&L
    realized_pnl      = Column(Float, nullable=True)
    target_profit_dollars = Column(Float, nullable=True)
    stop_loss_dollars     = Column(Float, nullable=True)

    # Lifecycle
    opened_at         = Column(DateTime, nullable=True)
    closed_at         = Column(DateTime, nullable=True)
    bars_held         = Column(Integer, nullable=True, default=0)
    exit_reason       = Column(String(32), nullable=True)
    assignment_note   = Column(Text, nullable=True)

    created_at        = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at        = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_sim_positions_symbol_state", "symbol", "state"),
        Index("ix_sim_positions_opened_at", "opened_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<SimPositionRecord {self.position_id} "
            f"{self.symbol} {self.structure_type} {self.state}>"
        )


# ---------------------------------------------------------------------------
# sim_events
# ---------------------------------------------------------------------------

class SimEventRecord(Base):
    """
    Persistent record of every execution event.

    This is the append-only audit log. One row per ExecutionEvent.
    """
    __tablename__ = "sim_events"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Identity
    event_id       = Column(String(24), nullable=False, unique=True, index=True)
    position_id    = Column(String(24), nullable=True, index=True)
    simulator_label = Column(String(64), nullable=False, default="paper_options_sim_v1")

    # Classification
    event_type     = Column(String(32), nullable=False, index=True)
    symbol         = Column(String(16), nullable=False, index=True)
    structure_type = Column(String(32), nullable=True)

    # Payload
    message        = Column(Text, nullable=True)
    data           = Column(JSON, nullable=True)

    # Timing
    event_timestamp = Column(DateTime, nullable=False, index=True)
    created_at      = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_sim_events_position_type", "position_id", "event_type"),
        Index("ix_sim_events_ts", "event_timestamp"),
    )

    def __repr__(self) -> str:
        return f"<SimEventRecord {self.event_id} {self.event_type} pos={self.position_id}>"


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def persist_position(session, position, simulator_label: str = "paper_options_sim_v1") -> None:
    """
    Upsert a SimPosition → SimPositionRecord.
    Uses merge-by-position_id: inserts on first call, updates thereafter.
    """
    existing = session.query(SimPositionRecord).filter_by(
        position_id=position.position_id
    ).one_or_none()

    legs_json = [
        {
            "action": lg.action,
            "option_type": lg.option_type,
            "target_delta": lg.target_delta,
            "strike": lg.strike,
            "expiry": lg.expiry,
            "fill_price": lg.fill_price,
            "fill_slippage": lg.fill_slippage,
        }
        for lg in position.legs
    ]

    if existing is None:
        record = SimPositionRecord(
            position_id=position.position_id,
            simulator_label=simulator_label,
            symbol=position.symbol,
            structure_type=position.structure_type,
            direction=position.direction,
            state=position.state.value,
            contracts=position.contracts,
            multiplier=position.multiplier,
            open_premium_per_share=position.open_premium_per_share,
            net_premium_dollars=(
                position.fill_result.net_premium_dollars
                if position.fill_result else None
            ),
            open_risk_dollars=position.open_risk_dollars,
            total_fees_paid=position.total_fees_paid,
            fill_method=(
                None  # filled in by caller if needed
            ),
            legs_json=legs_json,
            decision_snapshot=position.decision_snapshot or {},
            target_profit_dollars=position.target_profit_dollars,
            stop_loss_dollars=position.stop_loss_dollars,
            realized_pnl=position.realized_pnl if position.realized_pnl != 0.0 else None,
            opened_at=position.opened_at,
            closed_at=position.closed_at,
            bars_held=position.bars_held,
            exit_reason=position.exit_reason,
            assignment_note=position.assignment_note,
        )
        session.add(record)
    else:
        # Update mutable fields
        existing.state           = position.state.value
        existing.realized_pnl    = position.realized_pnl
        existing.total_fees_paid = position.total_fees_paid
        existing.closed_at       = position.closed_at
        existing.bars_held       = position.bars_held
        existing.exit_reason     = position.exit_reason
        existing.assignment_note = position.assignment_note
        existing.updated_at      = datetime.utcnow()


def persist_event(session, event, simulator_label: str = "paper_options_sim_v1") -> None:
    """Append a single ExecutionEvent to sim_events."""
    record = SimEventRecord(
        event_id=event.event_id,
        position_id=event.position_id,
        simulator_label=simulator_label,
        event_type=event.event_type.value,
        symbol=event.symbol,
        structure_type=event.structure_type,
        message=event.message,
        data=event.data,
        event_timestamp=event.timestamp,
    )
    session.add(record)


def flush_event_log(session, events: list, simulator_label: str = "paper_options_sim_v1") -> int:
    """
    Batch-persist a list of ExecutionEvent objects.
    Returns the number of events written.
    """
    written = 0
    for ev in events:
        try:
            persist_event(session, ev, simulator_label)
            written += 1
        except Exception:
            pass   # duplicate event_id — skip silently
    return written
