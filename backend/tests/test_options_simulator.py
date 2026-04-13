"""
Options paper-execution simulator tests.

OS1  — FillMethod.MIDPOINT: buy leg fills at mid
OS2  — FillMethod.MIDPOINT: sell leg fills at mid
OS3  — FillMethod.BID_ASK: buy leg fills at ask
OS4  — FillMethod.BID_ASK: sell leg fills at bid
OS5  — FillMethod.CONSERVATIVE: buy leg > ask (extra slippage)
OS6  — FillMethod.CONSERVATIVE: sell leg < bid (extra slippage)
OS5b — FillMethod.MIDPOINT_PLUS_SLIPPAGE: buy leg fills between mid and ask
OS6b — FillMethod.MIDPOINT_PLUS_SLIPPAGE: sell leg fills between bid and mid
OS7  — Net premium: long call debit is positive
OS8  — Net premium: credit spread is negative
OS9  — Fee calculation: per_contract × contracts + regulatory
OS10 — Fee calculation: minimum fee applied when per_contract < min_per_leg
OS11 — open_risk for debit = premium × multiplier × contracts
OS12 — open_risk for credit_spread = (width - credit) × multiplier × contracts
OS13 — RiskGuard: kill switch blocks new positions
OS14 — RiskGuard: max_concurrent_positions blocks when at limit
OS15 — RiskGuard: daily loss limit blocks when exceeded
OS16 — RiskGuard: max_open_risk blocks when adding would exceed limit
OS17 — RiskGuard: cooldown blocks within window after qualified loss
OS18 — RiskGuard: cooldown expires after cooldown_minutes
OS19 — RiskGuard: approved when all guards green
OS20 — RiskGuard: on_position_opened increments counters
OS21 — RiskGuard: on_position_closed decrements counters + triggers cooldown
OS22 — Session: pre-market timestamp blocked
OS23 — Session: after-hours timestamp blocked
OS24 — Session: within no_trade_before_close_mins blocked
OS25 — Session: normal trading hours accepted
OS26 — Simulator.open_position: long_call creates position with correct net debit
OS27 — Simulator.open_position: credit_spread creates position with net credit
OS28 — Simulator.open_position: risk block returns None position + RISK_BLOCKED event
OS29 — Simulator.open_position: session block returns None position + SESSION_BLOCKED event
OS30 — Simulator.update_positions: unrealized P&L calculated correctly (long call)
OS31 — Simulator.update_positions: unrealized P&L calculated correctly (credit spread)
OS32 — Simulator.update_positions: target_profit triggers exit
OS33 — Simulator.update_positions: stop_loss triggers exit
OS34 — Simulator.update_positions: DTE threshold triggers exit
OS35 — Simulator.update_positions: DTE=0 OTM → EXPIRED state + full loss
OS36 — Simulator.update_positions: DTE=0 ITM short leg → ASSIGNED state + note
OS37 — Simulator.force_close_position: exits open position
OS38 — Simulator.close_all_positions: closes all open positions
OS39 — RiskGuard.trigger_kill_switch: blocks all subsequent opens
OS40 — EventLog: POSITION_OPENED event present after successful open
OS41 — EventLog: RISK_BLOCKED event present after risk-blocked open
OS42 — EventLog: POSITION_CLOSED event present after target-profit close
OS43 — EventLog: POSITION_EXPIRED event present after OTM expiry
OS44 — EventLog: POSITION_ASSIGNED event present after ITM expiry
OS45 — EventLog: all events have timestamp, event_id, event_type fields
OS46 — ExecutionEvent.to_dict(): serializes without error
OS47 — RiskGuard.reset_daily(): clears daily P&L and cooldown
OS48 — debit_spread: both legs present, correct direction
OS49 — long_put: bearish direction, put legs
OS50 — fill_quality_warning logged for wide spread
"""

import pytest
from datetime import datetime, timezone, timedelta, time
from unittest.mock import patch

from app.paper_trading.options_simulator.config import (
    SimulatorConfig, FillConfig, FeeConfig, ContractSelectionConfig,
    SessionConfig, ExitConfig, RiskConfig, FillMethod,
)
from app.paper_trading.options_simulator.models import (
    SimLeg, LegQuote, SimPosition, PositionState, EventType, ExitReason,
)
from app.paper_trading.options_simulator.fill_engine import (
    simulate_fill, compute_open_risk,
)
from app.paper_trading.options_simulator.risk_guard import RiskGuard
from app.paper_trading.options_simulator.simulator import (
    PaperOptionsSimulator, _compute_unrealized_pnl, _intrinsic_value,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ET = timezone(timedelta(hours=-4))  # EDT approximation for testing

def _et(h, m=0):
    """Return a timezone-aware datetime in Eastern time on a fixed date."""
    return datetime(2025, 6, 10, h, m, 0, tzinfo=ET)


def _make_legs(structure_type="long_call", bullish=True):
    if structure_type == "long_call":
        return [SimLeg(action="buy", option_type="call", target_delta=0.40, strike=105.0)]
    if structure_type == "long_put":
        return [SimLeg(action="buy", option_type="put", target_delta=0.40, strike=95.0)]
    if structure_type == "debit_spread":
        return [
            SimLeg(action="buy", option_type="call", target_delta=0.40, strike=103.0),
            SimLeg(action="sell", option_type="call", target_delta=0.20, strike=108.0),
        ]
    if structure_type == "credit_spread":
        return [
            SimLeg(action="sell", option_type="put", target_delta=0.30, strike=96.0),
            SimLeg(action="buy", option_type="put", target_delta=0.15, strike=91.0),
        ]
    raise ValueError(structure_type)


def _make_quotes(bid=1.80, ask=2.20, underlying=100.0, dte=7):
    """Single-leg quote."""
    return [LegQuote(bid=bid, ask=ask, mid=(bid+ask)/2, underlying_price=underlying, dte=dte)]


def _two_quotes(q1, q2):
    return [q1, q2]


def _default_sim(
    *,
    fill_method=FillMethod.MIDPOINT,
    target_profit_pct=0.50,
    stop_loss_pct=1.00,
    close_at_dte=1,
    no_trade_before_close=15,
    no_trade_after_open=0,
    max_daily_loss=500.0,
    max_open_risk=2000.0,
    max_concurrent=5,
    cooldown_after_loss=100.0,
    cooldown_minutes=30,
    kill_switch=False,
    slippage_factor=0.25,
):
    return PaperOptionsSimulator(SimulatorConfig(
        fill=FillConfig(method=fill_method, slippage_factor=slippage_factor),
        fees=FeeConfig(per_contract=0.65, regulatory_fee_per_contract=0.0),
        session=SessionConfig(
            no_trade_before_close_mins=no_trade_before_close,
            no_trade_after_open_mins=no_trade_after_open,
        ),
        exit=ExitConfig(
            target_profit_pct=target_profit_pct,
            stop_loss_pct=stop_loss_pct,
            close_at_dte=close_at_dte,
        ),
        risk=RiskConfig(
            max_daily_loss=max_daily_loss,
            max_open_risk=max_open_risk,
            max_concurrent_positions=max_concurrent,
            cooldown_after_loss=cooldown_after_loss,
            cooldown_minutes=cooldown_minutes,
            kill_switch=kill_switch,
        ),
    ))


def _open_long_call(sim, ts=None, bid=1.80, ask=2.20, underlying=100.0):
    legs = _make_legs("long_call")
    quotes = _make_quotes(bid=bid, ask=ask, underlying=underlying)
    ts = ts or _et(10, 30)
    return sim.open_position(
        "long_call", "bullish", "TEST", legs, quotes,
        contracts=1, timestamp=ts,
    )


# ---------------------------------------------------------------------------
# OS1 – OS6: Fill engine
# ---------------------------------------------------------------------------

def test_midpoint_buy_fills_at_mid():
    legs = [SimLeg(action="buy", option_type="call", target_delta=0.4)]
    quotes = _make_quotes(bid=1.80, ask=2.20)
    cfg = FillConfig(method=FillMethod.MIDPOINT)
    result = simulate_fill(legs, quotes, cfg, FeeConfig(), contracts=1)
    assert result.legs[0].fill_price == pytest.approx(2.00, abs=0.01)


def test_midpoint_sell_fills_at_mid():
    legs = [SimLeg(action="sell", option_type="put", target_delta=0.3)]
    quotes = _make_quotes(bid=1.80, ask=2.20)
    cfg = FillConfig(method=FillMethod.MIDPOINT)
    result = simulate_fill(legs, quotes, cfg, FeeConfig(), contracts=1)
    assert result.legs[0].fill_price == pytest.approx(2.00, abs=0.01)


def test_bid_ask_buy_fills_at_ask():
    legs = [SimLeg(action="buy", option_type="call", target_delta=0.4)]
    quotes = _make_quotes(bid=1.80, ask=2.20)
    cfg = FillConfig(method=FillMethod.BID_ASK)
    result = simulate_fill(legs, quotes, cfg, FeeConfig(), contracts=1)
    assert result.legs[0].fill_price == pytest.approx(2.20, abs=0.01)


def test_bid_ask_sell_fills_at_bid():
    legs = [SimLeg(action="sell", option_type="put", target_delta=0.3)]
    quotes = _make_quotes(bid=1.80, ask=2.20)
    cfg = FillConfig(method=FillMethod.BID_ASK)
    result = simulate_fill(legs, quotes, cfg, FeeConfig(), contracts=1)
    assert result.legs[0].fill_price == pytest.approx(1.80, abs=0.01)


def test_conservative_buy_greater_than_ask():
    legs = [SimLeg(action="buy", option_type="call", target_delta=0.4)]
    quotes = _make_quotes(bid=1.80, ask=2.20)
    cfg = FillConfig(method=FillMethod.CONSERVATIVE, slippage_factor=0.25)
    result = simulate_fill(legs, quotes, cfg, FeeConfig(), contracts=1)
    # fill = ask + 0.25 × (ask - bid) = 2.20 + 0.25 × 0.40 = 2.30
    assert result.legs[0].fill_price > 2.20
    assert result.legs[0].fill_price == pytest.approx(2.30, abs=0.01)


def test_conservative_sell_less_than_bid():
    legs = [SimLeg(action="sell", option_type="put", target_delta=0.3)]
    quotes = _make_quotes(bid=1.80, ask=2.20)
    cfg = FillConfig(method=FillMethod.CONSERVATIVE, slippage_factor=0.25)
    result = simulate_fill(legs, quotes, cfg, FeeConfig(), contracts=1)
    # fill = bid - 0.25 × (ask - bid) = 1.80 - 0.10 = 1.70
    assert result.legs[0].fill_price < 1.80
    assert result.legs[0].fill_price == pytest.approx(1.70, abs=0.01)


def test_midpoint_plus_slippage_buy_between_mid_and_ask():
    # OS5b: buy fills at mid + slippage_factor × spread (between mid and ask)
    legs = [SimLeg(action="buy", option_type="call", target_delta=0.4)]
    quotes = _make_quotes(bid=1.80, ask=2.20)  # mid=2.00, spread=0.40
    cfg = FillConfig(method=FillMethod.MIDPOINT_PLUS_SLIPPAGE, slippage_factor=0.25)
    result = simulate_fill(legs, quotes, cfg, FeeConfig(), contracts=1)
    # fill = 2.00 + 0.25 × 0.40 = 2.10
    assert 2.00 < result.legs[0].fill_price < 2.20
    assert result.legs[0].fill_price == pytest.approx(2.10, abs=0.01)


def test_midpoint_plus_slippage_sell_between_bid_and_mid():
    # OS6b: sell fills at mid - slippage_factor × spread (between bid and mid)
    legs = [SimLeg(action="sell", option_type="put", target_delta=0.3)]
    quotes = _make_quotes(bid=1.80, ask=2.20)  # mid=2.00, spread=0.40
    cfg = FillConfig(method=FillMethod.MIDPOINT_PLUS_SLIPPAGE, slippage_factor=0.25)
    result = simulate_fill(legs, quotes, cfg, FeeConfig(), contracts=1)
    # fill = 2.00 - 0.25 × 0.40 = 1.90
    assert 1.80 < result.legs[0].fill_price < 2.00
    assert result.legs[0].fill_price == pytest.approx(1.90, abs=0.01)


# ---------------------------------------------------------------------------
# OS7 – OS8: Net premium sign
# ---------------------------------------------------------------------------

def test_long_call_net_premium_is_positive():
    legs = _make_legs("long_call")
    quotes = _make_quotes(bid=1.80, ask=2.20)
    result = simulate_fill(legs, quotes, FillConfig(), FeeConfig(), contracts=1)
    assert result.net_premium_per_share > 0


def test_credit_spread_net_premium_is_negative():
    # Sell put at 1.80/2.20 mid=2.00, buy put at 0.90/1.10 mid=1.00
    legs = _make_legs("credit_spread")
    q1 = LegQuote(bid=1.80, ask=2.20, mid=2.00, underlying_price=100.0, dte=7)
    q2 = LegQuote(bid=0.90, ask=1.10, mid=1.00, underlying_price=100.0, dte=7)
    result = simulate_fill(legs, [q1, q2], FillConfig(), FeeConfig(), contracts=1)
    # sell mid 2.00, buy mid 1.00 → net = −2.00 + 1.00 = −1.00 (credit)
    assert result.net_premium_per_share < 0
    assert result.net_premium_per_share == pytest.approx(-1.00, abs=0.01)


# ---------------------------------------------------------------------------
# OS9 – OS10: Fees
# ---------------------------------------------------------------------------

def test_fee_per_contract_times_contracts():
    legs = [SimLeg(action="buy", option_type="call", target_delta=0.4)]
    quotes = _make_quotes()
    fee_cfg = FeeConfig(per_contract=0.65, regulatory_fee_per_contract=0.03)
    result = simulate_fill(legs, quotes, FillConfig(), fee_cfg, contracts=2)
    expected = (0.65 + 0.03) * 2
    assert result.total_fees == pytest.approx(expected, abs=0.01)


def test_fee_minimum_per_leg_applied():
    legs = [SimLeg(action="buy", option_type="call", target_delta=0.4)]
    quotes = _make_quotes()
    fee_cfg = FeeConfig(per_contract=0.10, min_per_leg=1.00, regulatory_fee_per_contract=0.0)
    result = simulate_fill(legs, quotes, FillConfig(), fee_cfg, contracts=1)
    assert result.total_fees == pytest.approx(1.00, abs=0.01)


# ---------------------------------------------------------------------------
# OS11 – OS12: open_risk calculation
# ---------------------------------------------------------------------------

def test_open_risk_debit_equals_premium():
    # long call, 2.00 per share, 1 contract, 100 multiplier
    risk = compute_open_risk(
        net_premium_per_share=2.00,
        structure_type="long_call",
        spread_width_per_share=0.0,
        contracts=1,
        multiplier=100,
    )
    assert risk == pytest.approx(200.0, abs=0.01)


def test_open_risk_credit_spread_is_width_minus_credit():
    # credit_spread: sell at 2.00, buy at 1.00 → net = -1.00 (credit)
    # spread width = 5.00 per share → max loss = 4.00 × 100 = 400
    risk = compute_open_risk(
        net_premium_per_share=-1.00,
        structure_type="credit_spread",
        spread_width_per_share=5.00,
        contracts=1,
        multiplier=100,
    )
    assert risk == pytest.approx(400.0, abs=0.01)


# ---------------------------------------------------------------------------
# OS13 – OS21: RiskGuard
# ---------------------------------------------------------------------------

def test_kill_switch_blocks():
    guard = RiskGuard(RiskConfig(kill_switch=True))
    check = guard.check_new_position(100.0, datetime.utcnow())
    assert not check.approved
    assert check.reason == "kill_switch"


def test_max_concurrent_blocks_at_limit():
    guard = RiskGuard(RiskConfig(max_concurrent_positions=2))
    guard.on_position_opened(100.0)
    guard.on_position_opened(100.0)
    check = guard.check_new_position(100.0, datetime.utcnow())
    assert not check.approved
    assert check.reason == "max_concurrent_positions"


def test_daily_loss_limit_blocks():
    guard = RiskGuard(RiskConfig(max_daily_loss=200.0))
    guard.on_position_closed(-250.0, 200.0, datetime.utcnow())
    check = guard.check_new_position(50.0, datetime.utcnow())
    assert not check.approved
    assert check.reason == "max_daily_loss_exceeded"


def test_max_open_risk_blocks():
    guard = RiskGuard(RiskConfig(max_open_risk=500.0))
    guard.on_position_opened(400.0)
    check = guard.check_new_position(200.0, datetime.utcnow())
    assert not check.approved
    assert check.reason == "max_open_risk_exceeded"


def test_cooldown_blocks_within_window():
    guard = RiskGuard(RiskConfig(cooldown_after_loss=100.0, cooldown_minutes=30))
    now = datetime.utcnow()
    guard.on_position_closed(-150.0, 200.0, now)
    check = guard.check_new_position(50.0, now + timedelta(minutes=5))
    assert not check.approved
    assert check.reason == "cooldown_active"


def test_cooldown_expires_after_window():
    guard = RiskGuard(RiskConfig(cooldown_after_loss=100.0, cooldown_minutes=30))
    now = datetime.utcnow()
    guard.on_position_closed(-150.0, 200.0, now)
    # After 31 minutes, cooldown should have expired
    check = guard.check_new_position(50.0, now + timedelta(minutes=31))
    assert check.approved


def test_risk_guard_approves_when_all_clear():
    guard = RiskGuard(RiskConfig())
    check = guard.check_new_position(100.0, datetime.utcnow())
    assert check.approved


def test_on_position_opened_increments_counters():
    guard = RiskGuard(RiskConfig())
    assert guard.open_position_count == 0
    assert guard.open_risk_dollars == 0.0
    guard.on_position_opened(300.0)
    assert guard.open_position_count == 1
    assert guard.open_risk_dollars == pytest.approx(300.0, abs=0.01)


def test_on_position_closed_decrements_counters():
    guard = RiskGuard(RiskConfig(cooldown_after_loss=1000.0))  # high threshold: no cooldown
    guard.on_position_opened(300.0)
    guard.on_position_closed(50.0, 300.0, datetime.utcnow())
    assert guard.open_position_count == 0
    assert guard.open_risk_dollars == pytest.approx(0.0, abs=0.01)
    assert guard.daily_realized_pnl == pytest.approx(50.0, abs=0.01)


# ---------------------------------------------------------------------------
# OS22 – OS25: Session rules
# ---------------------------------------------------------------------------

def test_pre_market_blocked():
    sim = _default_sim()
    result = _open_long_call(sim, ts=_et(8, 0))
    assert not result.accepted
    assert "pre_market" in result.blocked_reason


def test_after_hours_blocked():
    sim = _default_sim()
    result = _open_long_call(sim, ts=_et(17, 0))
    assert not result.accepted
    assert "after_hours" in result.blocked_reason


def test_near_close_blocked():
    sim = _default_sim(no_trade_before_close=15)
    # 15:50 ET = 10 minutes to close
    result = _open_long_call(sim, ts=_et(15, 50))
    assert not result.accepted
    assert "too_close_to_close" in result.blocked_reason


def test_normal_hours_accepted():
    sim = _default_sim(no_trade_before_close=15, no_trade_after_open=0)
    result = _open_long_call(sim, ts=_et(11, 0))
    assert result.accepted


# ---------------------------------------------------------------------------
# OS26 – OS29: Simulator open_position
# ---------------------------------------------------------------------------

def test_open_long_call_creates_position():
    sim = _default_sim()
    result = _open_long_call(sim, bid=1.80, ask=2.20)
    assert result.accepted
    pos = result.position
    assert pos.structure_type == "long_call"
    assert pos.state == PositionState.OPEN
    assert pos.open_premium_per_share > 0       # debit
    assert len(pos.legs) == 1


def test_open_credit_spread_net_credit():
    sim = _default_sim()
    legs = _make_legs("credit_spread")
    q1 = LegQuote(bid=1.80, ask=2.20, mid=2.00, underlying_price=100.0, dte=7)
    q2 = LegQuote(bid=0.90, ask=1.10, mid=1.00, underlying_price=100.0, dte=7)
    result = sim.open_position(
        "credit_spread", "bullish", "TEST", legs, [q1, q2],
        contracts=1, timestamp=_et(10, 30),
        spread_width_per_share=5.0,
    )
    assert result.accepted
    pos = result.position
    assert pos.open_premium_per_share < 0    # credit received
    assert len(pos.legs) == 2


def test_risk_block_returns_none_position():
    sim = _default_sim(kill_switch=True)
    result = _open_long_call(sim)
    assert not result.accepted
    assert result.position is None
    assert result.blocked_reason == "kill_switch"


def test_session_block_returns_none_position():
    sim = _default_sim()
    result = _open_long_call(sim, ts=_et(8, 0))
    assert not result.accepted
    assert result.position is None
    assert result.blocked_reason is not None


# ---------------------------------------------------------------------------
# OS30 – OS31: Mark-to-market P&L
# ---------------------------------------------------------------------------

def test_unrealized_pnl_long_call():
    # Long call: bought at 2.00, now worth 3.00
    pos = SimPosition(
        legs=[SimLeg(action="buy", option_type="call", target_delta=0.4, fill_price=2.00)],
        contracts=1, multiplier=100,
    )
    quotes = [LegQuote(bid=2.80, ask=3.20, mid=3.00, underlying_price=105.0, dte=5)]
    pnl = _compute_unrealized_pnl(pos, quotes)
    assert pnl == pytest.approx(100.0, abs=0.01)  # (3.00 - 2.00) * 1 * 100


def test_unrealized_pnl_credit_spread():
    # Credit spread: sell put at 2.00, buy put at 1.00
    # Now: sell worth 1.20, buy worth 0.60
    # pnl per share = (-1.20 + 0.60) - (-2.00 + 1.00) = -0.60 + 1.00 = +0.40
    pos = SimPosition(
        legs=[
            SimLeg(action="sell", option_type="put", target_delta=0.3, fill_price=2.00),
            SimLeg(action="buy",  option_type="put", target_delta=0.15, fill_price=1.00),
        ],
        contracts=1, multiplier=100,
    )
    quotes = [
        LegQuote(bid=1.10, ask=1.30, mid=1.20, underlying_price=101.0, dte=5),
        LegQuote(bid=0.50, ask=0.70, mid=0.60, underlying_price=101.0, dte=5),
    ]
    pnl = _compute_unrealized_pnl(pos, quotes)
    # Sell leg: (1.20 - 2.00) * -1 * 100 = +80
    # Buy  leg: (0.60 - 1.00) * +1 * 100 = -40
    # Total = +40
    assert pnl == pytest.approx(40.0, abs=0.01)


# ---------------------------------------------------------------------------
# OS32 – OS34: Exit triggers
# ---------------------------------------------------------------------------

def test_target_profit_triggers_exit():
    sim = _default_sim(target_profit_pct=0.50, fill_method=FillMethod.MIDPOINT)
    result = _open_long_call(sim, bid=1.80, ask=2.20)
    pos = result.position
    # target_profit = 2.00 × 100 × 0.50 = $100
    # Update with quote that gives pnl ≥ $100
    # Current value = 3.05 → pnl = (3.05 - 2.00) * 100 = $105
    exits = sim.update_positions(
        {pos.position_id: [LegQuote(bid=2.90, ask=3.20, mid=3.05, underlying_price=106.0, dte=5)]},
        timestamp=_et(13, 0),
    )
    assert len(exits) == 1
    assert exits[0].exit_reason == ExitReason.TARGET_PROFIT


def test_stop_loss_triggers_exit():
    sim = _default_sim(stop_loss_pct=0.50, target_profit_pct=None, fill_method=FillMethod.MIDPOINT)
    # open_risk = 2.00 * 100 = 200; stop = 200 * 0.50 = $100
    result = _open_long_call(sim, bid=1.80, ask=2.20)
    pos = result.position
    # pnl of -$101 should trigger stop: current_mid = 2.00 - 1.01 = 0.99
    exits = sim.update_positions(
        {pos.position_id: [LegQuote(bid=0.88, ask=1.10, mid=0.99, underlying_price=97.0, dte=5)]},
        timestamp=_et(13, 0),
    )
    assert len(exits) == 1
    assert exits[0].exit_reason == ExitReason.STOP_LOSS


def test_dte_threshold_triggers_exit():
    sim = _default_sim(close_at_dte=1)
    result = _open_long_call(sim, bid=1.80, ask=2.20)
    pos = result.position
    exits = sim.update_positions(
        {pos.position_id: [LegQuote(bid=1.80, ask=2.20, mid=2.00, underlying_price=100.0, dte=1)]},
        timestamp=_et(13, 0),
    )
    assert len(exits) == 1
    assert exits[0].exit_reason == ExitReason.DTE_THRESHOLD


# ---------------------------------------------------------------------------
# OS35 – OS36: Expiry simulation
# ---------------------------------------------------------------------------

def test_dte_zero_otm_expires_worthless():
    sim = _default_sim(close_at_dte=0, target_profit_pct=None, stop_loss_pct=None)
    legs = [SimLeg(action="buy", option_type="call", target_delta=0.4, strike=105.0)]
    quotes = [LegQuote(bid=1.80, ask=2.20, mid=2.00, underlying_price=100.0, dte=7)]
    result = sim.open_position("long_call", "bullish", "TEST", legs, quotes,
                               timestamp=_et(10, 30))
    pos = result.position
    # Underlying stays below strike at expiry
    exits = sim.update_positions(
        {pos.position_id: [LegQuote(bid=0.0, ask=0.0, mid=0.0, underlying_price=103.0, dte=0)]},
        timestamp=_et(15, 45),
    )
    assert len(exits) == 1
    assert exits[0].exit_reason == ExitReason.EXPIRED_OTM
    assert exits[0].position.state == PositionState.EXPIRED


def test_dte_zero_itm_short_leg_assigned():
    sim = _default_sim(close_at_dte=0, target_profit_pct=None, stop_loss_pct=None)
    # credit spread: short put at 96, underlying falls to 93
    legs = [
        SimLeg(action="sell", option_type="put", target_delta=0.3, strike=96.0, fill_price=2.00),
        SimLeg(action="buy",  option_type="put", target_delta=0.15, strike=91.0, fill_price=1.00),
    ]
    quotes = [
        LegQuote(bid=1.80, ask=2.20, mid=2.00, underlying_price=100.0, dte=7),
        LegQuote(bid=0.90, ask=1.10, mid=1.00, underlying_price=100.0, dte=7),
    ]
    result = sim.open_position("credit_spread", "bullish", "TEST", legs, quotes,
                               contracts=1, timestamp=_et(10, 30),
                               spread_width_per_share=5.0)
    pos = result.position
    # At expiry: underlying = 93 → short put (strike 96) is ITM by 3 points
    exits = sim.update_positions(
        {pos.position_id: [
            LegQuote(bid=0.0, ask=0.0, mid=3.00, underlying_price=93.0, dte=0),
            LegQuote(bid=0.0, ask=0.0, mid=0.0, underlying_price=93.0, dte=0),
        ]},
        timestamp=_et(16, 0),
    )
    assert len(exits) == 1
    assert exits[0].exit_reason == ExitReason.ASSIGNED_ITM
    assert exits[0].position.state == PositionState.ASSIGNED
    assert exits[0].position.assignment_note is not None


# ---------------------------------------------------------------------------
# OS37 – OS38: Force close
# ---------------------------------------------------------------------------

def test_force_close_position():
    sim = _default_sim(target_profit_pct=None, stop_loss_pct=None)
    result = _open_long_call(sim)
    pos = result.position
    quotes = _make_quotes(bid=1.80, ask=2.20, underlying=100.0, dte=5)
    exit_ev = sim.force_close_position(pos.position_id, quotes, timestamp=_et(14, 0))
    assert exit_ev is not None
    assert exit_ev.exit_reason == ExitReason.FORCE_CLOSE
    assert pos.state == PositionState.CLOSED


def test_close_all_positions():
    sim = _default_sim(target_profit_pct=None, stop_loss_pct=None)
    r1 = _open_long_call(sim, ts=_et(10, 30))
    r2 = _open_long_call(sim, ts=_et(11, 0))
    assert len(sim.open_positions) == 2
    quotes = _make_quotes(bid=1.80, ask=2.20, underlying=100.0, dte=5)
    exits = sim.close_all_positions(
        {r1.position.position_id: quotes, r2.position.position_id: quotes},
        timestamp=_et(15, 55),
    )
    assert len(exits) == 2
    assert len(sim.open_positions) == 0


# ---------------------------------------------------------------------------
# OS39: Kill switch via trigger_kill_switch
# ---------------------------------------------------------------------------

def test_trigger_kill_switch_blocks_new_positions():
    sim = _default_sim()
    sim.risk_guard.trigger_kill_switch(reason="circuit_breaker")
    result = _open_long_call(sim, ts=_et(10, 30))
    assert not result.accepted
    assert result.blocked_reason == "kill_switch"


# ---------------------------------------------------------------------------
# OS40 – OS45: Event log
# ---------------------------------------------------------------------------

def test_position_opened_event_logged():
    sim = _default_sim()
    result = _open_long_call(sim)
    types = [e.event_type for e in sim.event_log]
    assert EventType.POSITION_OPENED in types


def test_risk_blocked_event_logged():
    sim = _default_sim(kill_switch=True)
    _open_long_call(sim)
    types = [e.event_type for e in sim.event_log]
    assert EventType.RISK_BLOCKED in types


def test_position_closed_event_on_target_profit():
    sim = _default_sim(target_profit_pct=0.50, fill_method=FillMethod.MIDPOINT)
    result = _open_long_call(sim, bid=1.80, ask=2.20)
    pos = result.position
    sim.update_positions(
        {pos.position_id: [LegQuote(bid=2.90, ask=3.20, mid=3.05, underlying_price=106.0, dte=5)]},
        timestamp=_et(13, 0),
    )
    all_types = [e.event_type for e in sim.event_log]
    assert EventType.POSITION_CLOSED in all_types


def test_position_expired_event_logged():
    sim = _default_sim(close_at_dte=0, target_profit_pct=None, stop_loss_pct=None)
    legs = [SimLeg(action="buy", option_type="call", target_delta=0.4, strike=105.0)]
    quotes = [LegQuote(bid=1.80, ask=2.20, mid=2.00, underlying_price=100.0, dte=7)]
    result = sim.open_position("long_call", "bullish", "TEST", legs, quotes, timestamp=_et(10, 30))
    sim.update_positions(
        {result.position.position_id: [LegQuote(bid=0, ask=0, mid=0, underlying_price=103.0, dte=0)]},
        timestamp=_et(15, 55),
    )
    types = [e.event_type for e in sim.event_log]
    assert EventType.POSITION_EXPIRED in types


def test_position_assigned_event_logged():
    sim = _default_sim(close_at_dte=0, target_profit_pct=None, stop_loss_pct=None)
    legs = [SimLeg(action="sell", option_type="put", target_delta=0.3, strike=96.0, fill_price=2.00),
            SimLeg(action="buy",  option_type="put", target_delta=0.15, strike=91.0, fill_price=1.00)]
    quotes = [LegQuote(bid=1.80, ask=2.20, mid=2.00, underlying_price=100.0, dte=7),
              LegQuote(bid=0.90, ask=1.10, mid=1.00, underlying_price=100.0, dte=7)]
    result = sim.open_position("credit_spread", "bullish", "TEST", legs, quotes,
                               contracts=1, timestamp=_et(10, 30), spread_width_per_share=5.0)
    sim.update_positions(
        {result.position.position_id: [
            LegQuote(bid=0, ask=0, mid=3.0, underlying_price=93.0, dte=0),
            LegQuote(bid=0, ask=0, mid=0.0, underlying_price=93.0, dte=0),
        ]},
        timestamp=_et(16, 0),
    )
    types = [e.event_type for e in sim.event_log]
    assert EventType.POSITION_ASSIGNED in types


def test_all_events_have_required_fields():
    sim = _default_sim()
    _open_long_call(sim)
    for ev in sim.event_log:
        assert ev.event_id is not None and len(ev.event_id) > 0
        assert ev.timestamp is not None
        assert ev.event_type is not None


# ---------------------------------------------------------------------------
# OS46: to_dict serialization
# ---------------------------------------------------------------------------

def test_execution_event_to_dict():
    sim = _default_sim()
    result = _open_long_call(sim)
    ev = sim.event_log[0]
    d = ev.to_dict()
    assert "event_id" in d
    assert "timestamp" in d
    assert "event_type" in d
    assert "position_id" in d
    assert "data" in d


# ---------------------------------------------------------------------------
# OS47: RiskGuard reset_daily
# ---------------------------------------------------------------------------

def test_reset_daily_clears_pnl_and_cooldown():
    guard = RiskGuard(RiskConfig(cooldown_after_loss=50.0, cooldown_minutes=60))
    now = datetime.utcnow()
    guard.on_position_closed(-100.0, 200.0, now)
    assert guard.in_cooldown
    guard.reset_daily()
    assert not guard.in_cooldown
    assert guard.daily_realized_pnl == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# OS48 – OS49: Structure-specific sanity checks
# ---------------------------------------------------------------------------

def test_debit_spread_two_legs():
    sim = _default_sim()
    legs = _make_legs("debit_spread")
    q1 = LegQuote(bid=2.80, ask=3.20, mid=3.00, underlying_price=100.0, dte=7)
    q2 = LegQuote(bid=0.90, ask=1.10, mid=1.00, underlying_price=100.0, dte=7)
    result = sim.open_position(
        "debit_spread", "bullish", "TEST", legs, [q1, q2],
        contracts=1, timestamp=_et(10, 30),
    )
    assert result.accepted
    pos = result.position
    assert len(pos.legs) == 2
    assert pos.legs[0].action == "buy"
    assert pos.legs[1].action == "sell"
    assert pos.open_premium_per_share > 0    # net debit (buy higher delta, sell lower)


def test_long_put_bearish():
    sim = _default_sim()
    legs = _make_legs("long_put")
    quotes = [LegQuote(bid=1.80, ask=2.20, mid=2.00, underlying_price=100.0, dte=7)]
    result = sim.open_position(
        "long_put", "bearish", "TEST", legs, quotes,
        contracts=1, timestamp=_et(10, 30),
    )
    assert result.accepted
    pos = result.position
    assert pos.structure_type == "long_put"
    assert pos.direction == "bearish"
    assert pos.legs[0].option_type == "put"


# ---------------------------------------------------------------------------
# OS50: Fill quality warning logged for wide spread
# ---------------------------------------------------------------------------

def test_fill_quality_warning_logged_for_wide_spread():
    sim = _default_sim()
    legs = _make_legs("long_call")
    # bid=0.50, ask=4.00 → spread/mid = 3.50/2.25 = 155% >> max 20%
    quotes = [LegQuote(bid=0.50, ask=4.00, mid=2.25, underlying_price=100.0, dte=7)]
    sim.open_position("long_call", "bullish", "TEST", legs, quotes, timestamp=_et(10, 30))
    fill_quality_events = [
        e for e in sim.event_log if e.event_type == EventType.FILL_QUALITY
    ]
    assert len(fill_quality_events) > 0
