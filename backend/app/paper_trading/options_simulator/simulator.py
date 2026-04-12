"""
PaperOptionsSimulator — main orchestrator.

Usage pattern:
    config  = SimulatorConfig()
    sim     = PaperOptionsSimulator(config)

    # Open a position from a OptionsDecision (decision layer output)
    result  = sim.open_position(decision, contracts=1, timestamp=now)
    if result.accepted:
        position_id = result.position.position_id

    # On each new bar, supply updated quotes for every open position
    exits = sim.update_positions(
        {position_id: [LegQuote(bid, ask, mid, ...), ...]},
        timestamp=bar_time,
    )

    # Inspect the global event log
    for ev in sim.event_log:
        print(ev.to_dict())

Session / session-close handling:
    The simulator does NOT track wall-clock time internally. All time
    awareness comes from the `timestamp` argument the caller provides.
    The caller is responsible for:
      - Passing the bar's open-time as `timestamp`
      - Calling update_positions at the end of session with the final
        quotes to trigger time-based exit checks
      - Calling sim.risk_guard.reset_daily() at the start of each day

Assignment / exercise handling:
    When DTE reaches 0 during update_positions, the simulator evaluates
    each leg's intrinsic value relative to the underlying price (from
    LegQuote.underlying_price). If a LONG leg is ITM, an ASSIGNED event
    is noted with the assignment note. For paper-trading purposes, all
    expiry exits are settled at intrinsic value — no physical delivery
    is modeled.

Pin risk:
    When the underlying is within one ATM-tick (0.5% of strike) of a
    short leg at expiry, an assignment-risk note is added to the event
    but the position is closed at intrinsic value with a warning.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo   # type: ignore

from app.paper_trading.options_simulator.config import SimulatorConfig, ExitConfig
from app.paper_trading.options_simulator.models import (
    SimLeg,
    SimPosition,
    PositionState,
    LegQuote,
    FillResult,
    OpenResult,
    ExitEvent,
    ExitReason,
    ExecutionEvent,
    EventType,
    RiskCheck,
)
from app.paper_trading.options_simulator.fill_engine import (
    simulate_fill,
    estimate_close_fill,
    compute_open_risk,
)
from app.paper_trading.options_simulator.risk_guard import RiskGuard

_ET = ZoneInfo("America/New_York")


class PaperOptionsSimulator:
    """
    Stateful paper-execution simulator for options structures.

    Not thread-safe. One instance per paper-trading session.
    """

    def __init__(self, config: Optional[SimulatorConfig] = None) -> None:
        self.config = config or SimulatorConfig()
        self.risk_guard = RiskGuard(self.config.risk)

        # Mutable state
        self._positions: Dict[str, SimPosition] = {}  # id → position
        self._event_log: List[ExecutionEvent] = []

    # -----------------------------------------------------------------------
    # Public: open a position
    # -----------------------------------------------------------------------

    def open_position(
        self,
        structure_type: str,
        direction: str,
        symbol: str,
        legs: List[SimLeg],
        quotes: List[LegQuote],
        *,
        contracts: int = 1,
        timestamp: Optional[datetime] = None,
        spread_width_per_share: float = 0.0,
        decision_snapshot: Optional[Dict[str, Any]] = None,
    ) -> OpenResult:
        """
        Attempt to open a new options position.

        Returns OpenResult(position=None, blocked_reason=...) if any guard fires.
        Returns OpenResult(position=<SimPosition>) on successful simulated fill.

        Parameters
        ----------
        structure_type : str
            "long_call" | "long_put" | "debit_spread" | "credit_spread"
        direction : str
            "bullish" | "bearish"
        symbol : str
            Underlying ticker.
        legs : List[SimLeg]
            Ordered option legs (action, option_type, strike etc.).
        quotes : List[LegQuote]
            Current market quotes, parallel to `legs`.
        contracts : int
            Number of contracts (standard lot size = 1).
        timestamp : datetime
            Bar time for session/risk checks. Defaults to utcnow.
        spread_width_per_share : float
            For credit spreads: distance between strikes per share.
            Used to compute max loss. Pass 0 for outright long options.
        decision_snapshot : dict
            Arbitrary context from the decision layer (for logging).
        """
        ts = timestamp or datetime.utcnow()
        events: List[ExecutionEvent] = []

        # ------------------------------------------------------------------
        # 1. Session check
        # ------------------------------------------------------------------
        session_block = self._check_session(ts)
        if session_block:
            ev = self._log(
                EventType.SESSION_BLOCKED,
                symbol=symbol,
                structure_type=structure_type,
                message=session_block,
                data={"timestamp": ts.isoformat()},
            )
            events.append(ev)
            return OpenResult(position=None, blocked_reason=session_block, events=events)

        # ------------------------------------------------------------------
        # 2. Fill quality pre-check (warn but don't block unless configured)
        # ------------------------------------------------------------------
        fill_result = simulate_fill(
            legs, quotes,
            self.config.fill,
            self.config.fees,
            contracts,
            self.config.contracts.multiplier,
        )

        if fill_result.fill_quality_warnings and not self.config.risk.allow_wide_spreads:
            for w in fill_result.fill_quality_warnings:
                ev = self._log(
                    EventType.FILL_QUALITY,
                    symbol=symbol,
                    structure_type=structure_type,
                    message=f"fill_quality_warning: {w}",
                    data={"warning": w},
                )
                events.append(ev)

        # ------------------------------------------------------------------
        # 3. Compute open risk
        # ------------------------------------------------------------------
        open_risk = compute_open_risk(
            fill_result.net_premium_per_share,
            structure_type,
            spread_width_per_share,
            contracts,
            self.config.contracts.multiplier,
        )

        # ------------------------------------------------------------------
        # 4. Risk guard check
        # ------------------------------------------------------------------
        risk_check: RiskCheck = self.risk_guard.check_new_position(open_risk, ts)
        if not risk_check.approved:
            ev = self._log(
                EventType.RISK_BLOCKED,
                symbol=symbol,
                structure_type=structure_type,
                message=f"risk_blocked:{risk_check.reason}",
                data={"reason": risk_check.reason, "details": risk_check.details},
            )
            events.append(ev)
            return OpenResult(
                position=None, blocked_reason=risk_check.reason, events=events
            )

        # ------------------------------------------------------------------
        # 5. Build position
        # ------------------------------------------------------------------
        multiplier = self.config.contracts.multiplier
        net_pps = fill_result.net_premium_per_share

        # Dollar targets
        target_dollars: Optional[float] = None
        stop_dollars: Optional[float] = None

        if self.config.exit.target_profit_pct is not None:
            # For debit: target = cost × target_pct (50% of premium paid)
            # For credit: target = credit × target_pct (decay 50% of received)
            target_dollars = abs(net_pps) * multiplier * contracts * self.config.exit.target_profit_pct

        if self.config.exit.stop_loss_pct is not None:
            # Max loss × stop_pct
            stop_dollars = open_risk * self.config.exit.stop_loss_pct

        position = SimPosition(
            symbol=symbol,
            structure_type=structure_type,
            direction=direction,
            state=PositionState.OPEN,
            legs=fill_result.legs,
            contracts=contracts,
            multiplier=multiplier,
            fill_result=fill_result,
            open_premium_per_share=net_pps,
            open_risk_dollars=open_risk,
            target_profit_dollars=target_dollars,
            stop_loss_dollars=stop_dollars,
            total_fees_paid=fill_result.total_fees,
            opened_at=ts,
            decision_snapshot=decision_snapshot or {},
        )

        self._positions[position.position_id] = position
        self.risk_guard.on_position_opened(open_risk)

        # ------------------------------------------------------------------
        # 6. Log open event
        # ------------------------------------------------------------------
        ev = self._log(
            EventType.POSITION_OPENED,
            position_id=position.position_id,
            symbol=symbol,
            structure_type=structure_type,
            message=f"opened {structure_type} {direction} | "
                    f"net_pps={net_pps:.4f} | "
                    f"open_risk=${open_risk:.2f} | "
                    f"fees=${fill_result.total_fees:.2f}",
            data={
                "net_premium_per_share": net_pps,
                "net_premium_dollars": fill_result.net_premium_dollars,
                "open_risk_dollars": open_risk,
                "total_fees": fill_result.total_fees,
                "fill_method": self.config.fill.method.value,
                "legs": [
                    {
                        "action": lg.action,
                        "option_type": lg.option_type,
                        "strike": lg.strike,
                        "fill_price": lg.fill_price,
                        "slippage": lg.fill_slippage,
                    }
                    for lg in position.legs
                ],
            },
        )
        events.append(ev)
        position.events.append(ev)

        return OpenResult(position=position, events=events)

    # -----------------------------------------------------------------------
    # Public: update all open positions (mark-to-market + exit checks)
    # -----------------------------------------------------------------------

    def update_positions(
        self,
        position_quotes: Dict[str, List[LegQuote]],
        timestamp: Optional[datetime] = None,
    ) -> List[ExitEvent]:
        """
        Mark all open positions to market and check exit conditions.

        Parameters
        ----------
        position_quotes : Dict[str, List[LegQuote]]
            Mapping of position_id → current leg quotes (same order as position.legs).
        timestamp : datetime
            Current bar time.

        Returns
        -------
        List[ExitEvent] — one per position that exited this update.
        """
        ts = timestamp or datetime.utcnow()
        exit_events: List[ExitEvent] = []

        # Update unrealized P&L for risk guard
        total_unrealized = sum(
            self._mark_position(pid, quotes, ts)
            for pid, quotes in position_quotes.items()
            if pid in self._positions and self._positions[pid].state == PositionState.OPEN
        )
        self.risk_guard.update_unrealized_pnl(total_unrealized)

        # Check exits for each updated position
        for pid, quotes in position_quotes.items():
            pos = self._positions.get(pid)
            if pos is None or pos.state != PositionState.OPEN:
                continue

            exit_reason = self._check_exit_conditions(pos, quotes, ts)
            if exit_reason is not None:
                pos.bars_held += 1
                exit_ev = self._close_position(pos, quotes, exit_reason, ts)
                exit_events.append(exit_ev)
            else:
                pos.bars_held += 1

        return exit_events

    # -----------------------------------------------------------------------
    # Public: force-close a position
    # -----------------------------------------------------------------------

    def force_close_position(
        self,
        position_id: str,
        quotes: List[LegQuote],
        timestamp: Optional[datetime] = None,
        reason: str = "force_close",
    ) -> Optional[ExitEvent]:
        """Immediately close a position regardless of exit rules."""
        pos = self._positions.get(position_id)
        if pos is None or pos.state != PositionState.OPEN:
            return None
        ts = timestamp or datetime.utcnow()
        return self._close_position(pos, quotes, ExitReason.FORCE_CLOSE, ts)

    def close_all_positions(
        self,
        position_quotes: Dict[str, List[LegQuote]],
        timestamp: Optional[datetime] = None,
    ) -> List[ExitEvent]:
        """Force-close every open position (e.g. session-end or kill switch)."""
        ts = timestamp or datetime.utcnow()
        exits = []
        for pid in list(self._positions):
            pos = self._positions[pid]
            if pos.state == PositionState.OPEN:
                quotes = position_quotes.get(pid, [])
                if quotes:
                    ev = self._close_position(pos, quotes, ExitReason.FORCE_CLOSE, ts)
                    exits.append(ev)
        return exits

    # -----------------------------------------------------------------------
    # Public: read-only properties
    # -----------------------------------------------------------------------

    @property
    def open_positions(self) -> List[SimPosition]:
        return [p for p in self._positions.values() if p.state == PositionState.OPEN]

    @property
    def all_positions(self) -> List[SimPosition]:
        return list(self._positions.values())

    @property
    def event_log(self) -> List[ExecutionEvent]:
        return list(self._event_log)

    @property
    def daily_pnl(self) -> float:
        return self.risk_guard.daily_pnl

    # -----------------------------------------------------------------------
    # Session rules
    # -----------------------------------------------------------------------

    def _check_session(self, ts: datetime) -> Optional[str]:
        """Return a block reason string if session rules prevent opening, else None."""
        cfg = self.config.session
        try:
            ts_et = ts.astimezone(_ET)
        except Exception:
            ts_et = ts  # fallback: no timezone conversion

        t = ts_et.time()

        if t < cfg.market_open:
            return f"pre_market:{t.strftime('%H:%M')}"
        if t > cfg.market_close:
            return f"after_hours:{t.strftime('%H:%M')}"

        # No-trade after open
        open_dt = ts_et.replace(
            hour=cfg.market_open.hour,
            minute=cfg.market_open.minute,
            second=0, microsecond=0,
        )
        mins_since_open = (ts_et - open_dt).total_seconds() / 60
        if cfg.no_trade_after_open_mins > 0 and mins_since_open < cfg.no_trade_after_open_mins:
            return f"too_close_to_open:{mins_since_open:.1f}min"

        # No-trade before close
        if cfg.no_trade_before_close_mins > 0:
            close_dt = ts_et.replace(
                hour=cfg.market_close.hour,
                minute=cfg.market_close.minute,
                second=0, microsecond=0,
            )
            mins_to_close = (close_dt - ts_et).total_seconds() / 60
            if mins_to_close <= cfg.no_trade_before_close_mins:
                return f"too_close_to_close:{mins_to_close:.1f}min"

        return None

    # -----------------------------------------------------------------------
    # Mark-to-market
    # -----------------------------------------------------------------------

    def _mark_position(
        self, position_id: str, quotes: List[LegQuote], ts: datetime
    ) -> float:
        """
        Update position.current_pnl from fresh quotes.
        Returns current unrealized P&L (dollars).
        """
        pos = self._positions.get(position_id)
        if pos is None or pos.state != PositionState.OPEN:
            return 0.0
        if len(quotes) != len(pos.legs):
            return pos.current_pnl   # stale — don't overwrite

        pnl = _compute_unrealized_pnl(pos, quotes)
        pos.current_pnl = pnl

        ev = self._log(
            EventType.MARK_TO_MARKET,
            position_id=pos.position_id,
            symbol=pos.symbol,
            structure_type=pos.structure_type,
            message=f"mtm pnl=${pnl:.2f}",
            data={"current_pnl": round(pnl, 2), "bars_held": pos.bars_held},
        )
        pos.events.append(ev)
        return pnl

    # -----------------------------------------------------------------------
    # Exit condition checks
    # -----------------------------------------------------------------------

    def _check_exit_conditions(
        self,
        pos: SimPosition,
        quotes: List[LegQuote],
        ts: datetime,
    ) -> Optional[ExitReason]:
        """Return the first firing ExitReason, or None if position stays open."""
        cfg = self.config.exit
        pnl = pos.current_pnl

        # 1. Target profit
        if cfg.target_profit_pct is not None and pos.target_profit_dollars is not None:
            if pnl >= pos.target_profit_dollars:
                return ExitReason.TARGET_PROFIT

        # 2. Stop loss
        if cfg.stop_loss_pct is not None and pos.stop_loss_dollars is not None:
            if pnl <= -pos.stop_loss_dollars:
                return ExitReason.STOP_LOSS

        # 3. DTE threshold / expiry
        if quotes:
            min_dte = min(q.dte for q in quotes)
            if min_dte == 0:
                # Always evaluate expiry mechanics regardless of close_at_dte config
                return self._check_expiry(pos, quotes)
            elif cfg.close_at_dte > 0 and min_dte <= cfg.close_at_dte:
                return ExitReason.DTE_THRESHOLD

        # 4. Max holding bars
        if cfg.max_holding_bars > 0 and pos.bars_held >= cfg.max_holding_bars:
            return ExitReason.MAX_HOLDING_BARS

        # 5. Session close (near-close check)
        session_block = self._check_session_close(ts)
        if session_block:
            return ExitReason.SESSION_CLOSE

        return None

    def _check_expiry(
        self, pos: SimPosition, quotes: List[LegQuote]
    ) -> ExitReason:
        """
        Evaluate expiry mechanics when DTE == 0.

        Uses intrinsic value if ExitConfig.use_intrinsic_at_expiry is True.
        Returns EXPIRED (OTM) or ASSIGNED (ITM) exit reason.
        """
        for leg, quote in zip(pos.legs, quotes):
            if quote.underlying_price <= 0 or leg.strike is None:
                continue
            intrinsic = _intrinsic_value(leg, quote.underlying_price)
            if intrinsic > 0.01 and leg.action == "sell":
                pos.assignment_note = (
                    f"Short {leg.option_type} strike={leg.strike} "
                    f"underlying={quote.underlying_price:.2f} "
                    f"intrinsic={intrinsic:.4f} — assignment likely."
                )
                # Pin risk note
                if abs(quote.underlying_price - leg.strike) / leg.strike < 0.005:
                    pos.assignment_note += " WARNING: pin risk — outcome uncertain."
                return ExitReason.ASSIGNED_ITM

        # Check if any long leg has value (ITM long option)
        for leg, quote in zip(pos.legs, quotes):
            if leg.action == "buy" and leg.strike is not None:
                intrinsic = _intrinsic_value(leg, quote.underlying_price)
                if intrinsic > 0.01:
                    return ExitReason.ASSIGNED_ITM  # exercise in value

        return ExitReason.EXPIRED_OTM

    def _check_session_close(self, ts: datetime) -> bool:
        """Return True if we're so close to close that open positions should be exited."""
        # This is separate from the open-position block: here we're asking whether
        # an already-open position should be forced out near close.
        # By default we don't force-exit open positions near close (only block new opens).
        # Callers can override by calling close_all_positions() explicitly.
        return False

    # -----------------------------------------------------------------------
    # Position close
    # -----------------------------------------------------------------------

    def _close_position(
        self,
        pos: SimPosition,
        quotes: List[LegQuote],
        exit_reason: ExitReason,
        ts: datetime,
    ) -> ExitEvent:
        """
        Close a position: compute closing fill, finalize P&L, update state.
        """
        # Compute closing fill (reversing the legs)
        if exit_reason in (ExitReason.EXPIRED_OTM, ExitReason.ASSIGNED_ITM) and \
                self.config.exit.use_intrinsic_at_expiry:
            realized_pnl = _compute_expiry_pnl(pos, quotes)
        else:
            close_fill = estimate_close_fill(
                pos.legs, quotes,
                self.config.fill,
                self.config.fees,
                pos.contracts,
                pos.multiplier,
            )
            # P&L = close proceeds - open cost
            open_net = pos.open_premium_per_share * pos.multiplier * pos.contracts
            close_net = close_fill.net_premium_per_share * pos.multiplier * pos.contracts
            # For debit: open_net > 0 (paid); close proceeds positive → pnl = close - open
            # For credit: open_net < 0 (received); close is cost to buy back
            realized_pnl = -close_net - open_net
            pos.total_fees_paid += close_fill.total_fees

        # Subtract total fees from P&L
        realized_pnl -= pos.total_fees_paid

        pos.realized_pnl = round(realized_pnl, 2)
        pos.current_pnl = realized_pnl
        pos.exit_reason = exit_reason.value
        pos.closed_at = ts
        pos.state = _exit_reason_to_state(exit_reason)

        # Risk guard update
        self.risk_guard.on_position_closed(realized_pnl, pos.open_risk_dollars, ts)

        # Determine event type for log
        if exit_reason == ExitReason.EXPIRED_OTM:
            ev_type = EventType.POSITION_EXPIRED
        elif exit_reason == ExitReason.ASSIGNED_ITM:
            ev_type = EventType.POSITION_ASSIGNED
        else:
            ev_type = EventType.POSITION_CLOSED

        data = {
            "exit_reason": exit_reason.value,
            "realized_pnl": pos.realized_pnl,
            "total_fees_paid": pos.total_fees_paid,
            "bars_held": pos.bars_held,
            "open_premium_per_share": pos.open_premium_per_share,
        }
        if pos.assignment_note:
            data["assignment_note"] = pos.assignment_note

        ev = self._log(
            ev_type,
            position_id=pos.position_id,
            symbol=pos.symbol,
            structure_type=pos.structure_type,
            message=f"closed {pos.structure_type} | reason={exit_reason.value} | pnl=${realized_pnl:.2f}",
            data=data,
        )
        pos.events.append(ev)

        return ExitEvent(
            position=pos,
            exit_reason=exit_reason,
            realized_pnl=realized_pnl,
            timestamp=ts,
            events=[ev],
        )

    # -----------------------------------------------------------------------
    # Internal event logging
    # -----------------------------------------------------------------------

    def _log(
        self,
        event_type: EventType,
        *,
        position_id: Optional[str] = None,
        symbol: str = "",
        structure_type: str = "",
        message: str = "",
        data: Optional[dict] = None,
    ) -> ExecutionEvent:
        ev = ExecutionEvent(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            position_id=position_id,
            symbol=symbol,
            structure_type=structure_type,
            message=message,
            data=data or {},
        )
        self._event_log.append(ev)
        return ev


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def _compute_unrealized_pnl(pos: SimPosition, quotes: List[LegQuote]) -> float:
    """
    Unrealized P&L = sum over legs of (current_price - fill_price) × sign × multiplier × contracts.

    sign = +1 for buy legs, -1 for sell legs.
    """
    pnl = 0.0
    for leg, quote in zip(pos.legs, quotes):
        sign = 1.0 if leg.action == "buy" else -1.0
        pnl += (quote.mid - leg.fill_price) * sign * pos.multiplier * pos.contracts
    return round(pnl, 2)


def _compute_expiry_pnl(pos: SimPosition, quotes: List[LegQuote]) -> float:
    """
    P&L at expiry using intrinsic value.

    For each leg: intrinsic = max(S - K, 0) for call, max(K - S, 0) for put.
    The pnl contribution = (intrinsic - fill_price) × sign × multiplier × contracts.
    """
    pnl = 0.0
    for leg, quote in zip(pos.legs, quotes):
        sign = 1.0 if leg.action == "buy" else -1.0
        if leg.strike is not None and quote.underlying_price > 0:
            intrinsic = _intrinsic_value(leg, quote.underlying_price)
        else:
            intrinsic = max(quote.mid, 0.0)
        pnl += (intrinsic - leg.fill_price) * sign * pos.multiplier * pos.contracts
    return round(pnl, 2)


def _intrinsic_value(leg: SimLeg, underlying: float) -> float:
    """Option intrinsic value per share."""
    if leg.strike is None:
        return 0.0
    if leg.option_type == "call":
        return max(underlying - leg.strike, 0.0)
    return max(leg.strike - underlying, 0.0)


def _exit_reason_to_state(reason: ExitReason) -> PositionState:
    if reason == ExitReason.EXPIRED_OTM:
        return PositionState.EXPIRED
    if reason == ExitReason.ASSIGNED_ITM:
        return PositionState.ASSIGNED
    return PositionState.CLOSED
