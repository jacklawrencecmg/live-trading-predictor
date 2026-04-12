"""
In-memory state models for the options simulator.

These are pure dataclasses with no DB dependency. The execution_log module
provides the SQLAlchemy schema for persistence.

State machine for SimPosition:
    PENDING  ──open_fill──► OPEN
    OPEN     ──exit──────► CLOSED | EXPIRED | ASSIGNED
    OPEN     ──kill──────► CLOSED (force_closed=True)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PositionState(str, Enum):
    PENDING  = "pending"    # order accepted, not yet filled
    OPEN     = "open"       # filled, active
    CLOSED   = "closed"     # exited by stop/target/time rule
    EXPIRED  = "expired"    # expired OTM (zero intrinsic value)
    ASSIGNED = "assigned"   # ITM at expiry; assignment event noted


class EventType(str, Enum):
    POSITION_OPENED  = "position_opened"   # fill completed, position active
    POSITION_CLOSED  = "position_closed"   # explicit exit (rule-driven)
    POSITION_EXPIRED = "position_expired"  # OTM expiry — full loss on long
    POSITION_ASSIGNED = "position_assigned" # ITM expiry note
    FILL_SIMULATED   = "fill_simulated"    # individual leg fill detail
    MARK_TO_MARKET   = "mark_to_market"    # P&L snapshot on update
    RISK_BLOCKED     = "risk_blocked"      # new order rejected by risk guard
    RISK_EVENT       = "risk_event"        # risk state change (kill switch, daily loss)
    SESSION_BLOCKED  = "session_blocked"   # new order rejected by session rules
    FILL_QUALITY     = "fill_quality"      # fill quality warning (wide spread etc)


class ExitReason(str, Enum):
    TARGET_PROFIT    = "target_profit"
    STOP_LOSS        = "stop_loss"
    DTE_THRESHOLD    = "dte_threshold"
    MAX_HOLDING_BARS = "max_holding_bars"
    SESSION_CLOSE    = "session_close"     # forced close near end of session
    EXPIRED_OTM      = "expired_otm"
    ASSIGNED_ITM     = "assigned_itm"
    FORCE_CLOSE      = "force_close"       # kill switch or manual
    SPREAD_TOO_WIDE  = "spread_too_wide"   # fill rejected, not an exit


# ---------------------------------------------------------------------------
# Leg-level models
# ---------------------------------------------------------------------------

@dataclass
class SimLeg:
    """One option contract within a simulated position."""
    action: str           # "buy" | "sell"
    option_type: str      # "call" | "put"
    target_delta: float
    strike: Optional[float] = None
    expiry: Optional[str] = None

    # Fill state (populated after execution)
    fill_price: float = 0.0          # price per share (premium / contract × multiplier)
    fill_bid: Optional[float] = None
    fill_ask: Optional[float] = None
    fill_mid: Optional[float] = None
    fill_slippage: float = 0.0       # dollars per share slippage vs mid


@dataclass
class LegQuote:
    """Current market quote for one option leg (used for mark-to-market)."""
    bid: float
    ask: float
    mid: float                        # (bid + ask) / 2; supply explicitly if available
    underlying_price: float = 0.0
    dte: int = 0                      # days to expiry remaining


@dataclass
class FillResult:
    """
    Output of the fill engine for a complete structure.

    `net_premium_per_share` is the signed premium per-share:
      positive = debit paid (long structures)
      negative = credit received (short structures / credit spreads)

    `net_premium_dollars` accounts for multiplier × contracts.
    """
    legs: List[SimLeg]
    net_premium_per_share: float     # signed
    net_premium_dollars: float       # net_premium_per_share × multiplier × contracts
    total_fees: float
    fill_quality_warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Position model
# ---------------------------------------------------------------------------

@dataclass
class SimPosition:
    """
    Full lifecycle record for one simulated options position.

    Scoring glossary:
      open_risk_dollars  : maximum possible loss at open (cost for debits, width−credit for spreads)
      current_pnl        : live unrealized P&L in dollars (updated on each mark-to-market)
      realized_pnl       : final P&L recorded when position closes (includes fees)
    """
    # Identity
    position_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    symbol: str = ""
    structure_type: str = ""          # "long_call" | "long_put" | "debit_spread" | "credit_spread"
    direction: str = ""               # "bullish" | "bearish"

    # State machine
    state: PositionState = PositionState.PENDING

    # Legs
    legs: List[SimLeg] = field(default_factory=list)
    contracts: int = 1
    multiplier: int = 100

    # Fill economics
    fill_result: Optional[FillResult] = None
    open_premium_per_share: float = 0.0   # = fill_result.net_premium_per_share at open
    open_risk_dollars: float = 0.0        # max possible loss in dollars

    # P&L targets (set at open in dollars)
    target_profit_dollars: Optional[float] = None
    stop_loss_dollars: Optional[float] = None

    # Live P&L
    current_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_fees_paid: float = 0.0

    # Timing
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    bars_held: int = 0

    # Close state
    exit_reason: Optional[str] = None
    assignment_note: Optional[str] = None   # populated if ITM at expiry

    # Full event trail
    events: List["ExecutionEvent"] = field(default_factory=list)

    # Context snapshot (from decision engine at open)
    decision_snapshot: Dict[str, Any] = field(default_factory=dict)

    def is_debit(self) -> bool:
        return self.open_premium_per_share > 0

    def is_credit(self) -> bool:
        return self.open_premium_per_share < 0


# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------

@dataclass
class ExecutionEvent:
    """
    Immutable audit record for every state change or notable condition.

    Every event is appended to position.events AND to the simulator's
    global event_log list, which can be flushed to the DB.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: EventType = EventType.MARK_TO_MARKET

    # References
    position_id: Optional[str] = None
    symbol: str = ""
    structure_type: str = ""

    # Payload
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_id":       self.event_id,
            "timestamp":      self.timestamp.isoformat(),
            "event_type":     str(self.event_type.value),
            "position_id":    self.position_id,
            "symbol":         self.symbol,
            "structure_type": self.structure_type,
            "message":        self.message,
            "data":           self.data,
        }


# ---------------------------------------------------------------------------
# Result types returned by simulator public methods
# ---------------------------------------------------------------------------

@dataclass
class OpenResult:
    """
    Result of a simulator.open_position() call.

    If `position` is None, the order was blocked. Check `blocked_reason`.
    """
    position: Optional[SimPosition]
    blocked_reason: Optional[str] = None
    events: List[ExecutionEvent] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.position is not None


@dataclass
class ExitEvent:
    """
    Emitted by simulator.update_positions() when a position exits.
    """
    position: SimPosition
    exit_reason: ExitReason
    realized_pnl: float
    timestamp: datetime
    events: List[ExecutionEvent] = field(default_factory=list)


@dataclass
class RiskCheck:
    """Result of a risk guard evaluation."""
    approved: bool
    reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
