"""
Risk guard — stateful risk control layer.

RiskGuard is instantiated once per simulator session. It tracks:
  - Session-level realized + unrealized P&L
  - Total open risk across all live positions
  - Concurrent open position count
  - Cooldown timer after loss events
  - Kill switch state

All checks return a RiskCheck(approved, reason). Callers act on the
`approved` flag; the `reason` string goes into the event log.

Design rule: risk controls only RAISE the bar — they never lower it.
A position that passes all guards when opened remains open even if a
guard that would block it later is tripped (guards only block NEW opens).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from app.paper_trading.options_simulator.config import RiskConfig
from app.paper_trading.options_simulator.models import RiskCheck


class RiskGuard:
    """
    Stateful session-level risk controller.

    Thread safety: not thread-safe. The simulator is single-threaded.
    """

    def __init__(self, config: RiskConfig) -> None:
        self.config = config

        # Mutable session state
        self._daily_realized_pnl: float = 0.0
        self._daily_unrealized_pnl: float = 0.0
        self._open_risk_dollars: float = 0.0   # sum of max_loss per open position
        self._open_position_count: int = 0
        self._kill_switch: bool = config.kill_switch

        # Cooldown state
        self._last_loss_amount: float = 0.0
        self._cooldown_until: Optional[datetime] = None

        # Diagnostic counters
        self.blocks_today: int = 0
        self.risk_events: list = []   # plain list of (timestamp, reason) tuples

    # -----------------------------------------------------------------------
    # Primary check: called before every new-position open
    # -----------------------------------------------------------------------

    def check_new_position(
        self,
        open_risk_dollars: float,
        timestamp: datetime,
    ) -> RiskCheck:
        """
        Return RiskCheck(approved=True) iff all guards pass for a new position.

        Parameters
        ----------
        open_risk_dollars : float
            Max possible loss of the proposed position (positive).
        timestamp : datetime
            Current bar timestamp (used for cooldown evaluation).
        """
        # 1. Kill switch — hard block
        if self._kill_switch:
            return self._block("kill_switch", {})

        # 2. Max concurrent positions
        if self._open_position_count >= self.config.max_concurrent_positions:
            return self._block("max_concurrent_positions", {
                "current": self._open_position_count,
                "max": self.config.max_concurrent_positions,
            })

        # 3. Daily loss limit
        total_daily_pnl = self._daily_realized_pnl + self._daily_unrealized_pnl
        if total_daily_pnl <= -self.config.max_daily_loss:
            return self._block("max_daily_loss_exceeded", {
                "daily_pnl": round(total_daily_pnl, 2),
                "limit": -self.config.max_daily_loss,
            })

        # 4. Max open risk
        projected_open_risk = self._open_risk_dollars + open_risk_dollars
        if projected_open_risk > self.config.max_open_risk:
            return self._block("max_open_risk_exceeded", {
                "current_open_risk": round(self._open_risk_dollars, 2),
                "proposed_addition": round(open_risk_dollars, 2),
                "limit": self.config.max_open_risk,
            })

        # 5. Cooldown
        if self._cooldown_until is not None and timestamp < self._cooldown_until:
            remaining = (self._cooldown_until - timestamp).total_seconds() / 60
            return self._block("cooldown_active", {
                "cooldown_until": self._cooldown_until.isoformat(),
                "minutes_remaining": round(remaining, 1),
                "triggered_by_loss": round(self._last_loss_amount, 2),
            })

        return RiskCheck(approved=True)

    # -----------------------------------------------------------------------
    # State updates (called by simulator after position events)
    # -----------------------------------------------------------------------

    def on_position_opened(self, open_risk_dollars: float) -> None:
        """Register a newly opened position."""
        self._open_risk_dollars += open_risk_dollars
        self._open_position_count += 1

    def on_position_closed(
        self,
        realized_pnl: float,
        open_risk_dollars: float,
        timestamp: datetime,
    ) -> None:
        """
        Register a closed position.

        Deducts its open_risk from the running total, adds realized P&L,
        and triggers cooldown if the loss exceeds the threshold.
        """
        self._daily_realized_pnl += realized_pnl
        self._open_risk_dollars = max(0.0, self._open_risk_dollars - open_risk_dollars)
        self._open_position_count = max(0, self._open_position_count - 1)

        # Cooldown check
        if (
            realized_pnl < 0
            and abs(realized_pnl) >= self.config.cooldown_after_loss
            and self.config.cooldown_minutes > 0
        ):
            self._last_loss_amount = abs(realized_pnl)
            self._cooldown_until = timestamp + timedelta(minutes=self.config.cooldown_minutes)
            self.risk_events.append((timestamp, f"cooldown_set:{self._cooldown_until.isoformat()}"))

    def update_unrealized_pnl(self, total_unrealized: float) -> None:
        """Update the aggregate unrealized P&L for all open positions."""
        self._daily_unrealized_pnl = total_unrealized

    def trigger_kill_switch(self, reason: str = "manual", timestamp: Optional[datetime] = None) -> None:
        """Activate the kill switch. No new positions will be allowed."""
        self._kill_switch = True
        ts = timestamp or datetime.utcnow()
        self.risk_events.append((ts, f"kill_switch_activated:{reason}"))

    def reset_daily(self) -> None:
        """Reset session P&L counters (call at start of each trading day)."""
        self._daily_realized_pnl = 0.0
        self._daily_unrealized_pnl = 0.0
        self._cooldown_until = None
        self.blocks_today = 0
        # Note: kill switch is NOT auto-reset — must be explicitly cleared.

    def clear_kill_switch(self) -> None:
        self._kill_switch = False

    # -----------------------------------------------------------------------
    # Read-only properties
    # -----------------------------------------------------------------------

    @property
    def daily_pnl(self) -> float:
        return self._daily_realized_pnl + self._daily_unrealized_pnl

    @property
    def daily_realized_pnl(self) -> float:
        return self._daily_realized_pnl

    @property
    def open_risk_dollars(self) -> float:
        return self._open_risk_dollars

    @property
    def open_position_count(self) -> int:
        return self._open_position_count

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch

    @property
    def in_cooldown(self) -> bool:
        if self._cooldown_until is None:
            return False
        return datetime.utcnow() < self._cooldown_until

    def snapshot(self) -> Dict[str, Any]:
        """Return a plain dict snapshot for logging."""
        return {
            "daily_realized_pnl": round(self._daily_realized_pnl, 2),
            "daily_unrealized_pnl": round(self._daily_unrealized_pnl, 2),
            "open_risk_dollars": round(self._open_risk_dollars, 2),
            "open_position_count": self._open_position_count,
            "kill_switch": self._kill_switch,
            "cooldown_until": self._cooldown_until.isoformat() if self._cooldown_until else None,
        }

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _block(self, reason: str, details: Dict[str, Any]) -> RiskCheck:
        self.blocks_today += 1
        self.risk_events.append((datetime.utcnow(), reason))
        return RiskCheck(approved=False, reason=reason, details=details)
