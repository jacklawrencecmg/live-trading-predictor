"""
Simulator configuration models.

All config objects are plain dataclasses — no external dependencies.
The SimulatorConfig combines every sub-config; callers only need to
instantiate and override the fields they care about.
"""

from dataclasses import dataclass, field
from datetime import time
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FillMethod(str, Enum):
    MIDPOINT    = "midpoint"     # fill at (bid + ask) / 2
    BID_ASK     = "bid_ask"      # buy at ask, sell at bid
    CONSERVATIVE = "conservative" # buy at ask + slippage, sell at bid - slippage


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class FillConfig:
    """Controls how simulated fills are priced."""
    method: FillMethod = FillMethod.MIDPOINT

    # For CONSERVATIVE fills: extra slippage as a fraction of the bid-ask spread.
    # 0.25 = add 25% of spread width on top of the natural side.
    # Applied symmetrically (buyers pay more, sellers receive less).
    slippage_factor: float = 0.25

    # Hard minimum option premium per contract (e.g. $0.05 tick).
    # Fills below this are floored here.
    min_tick: float = 0.01

    # Maximum acceptable bid-ask spread as a fraction of mid-price.
    # Orders rejected (not blocked, just warned) above this level.
    max_spread_pct: float = 0.20


@dataclass
class FeeConfig:
    """Commission and fee model."""

    # Per-contract fee in dollars (e.g. $0.65 = typical retail broker).
    per_contract: float = 0.65

    # Minimum fee per order leg (many brokers: $0 or $1.00).
    min_per_leg: float = 0.0

    # Maximum fee per order leg (cap for large orders; 0 = no cap).
    max_per_leg: float = 0.0

    # Assignment or exercise fee per contract.
    exercise_fee: float = 0.0

    # Exchange regulatory fee per contract (SEC/FINRA pass-through).
    regulatory_fee_per_contract: float = 0.03


@dataclass
class ContractSelectionConfig:
    """Rules for which contracts the simulator selects when chain data is absent."""

    # Target delta for long option legs (e.g. 0.40 = slightly OTM).
    target_delta_long: float = 0.40

    # Target delta for short legs in spreads (e.g. 0.20).
    target_delta_short: float = 0.20

    # Spread width in percent of spot (for debit/credit spreads).
    spread_width_pct: float = 0.03   # 3% of spot = typical single-name spread

    # Contract multiplier (standard US equity options = 100 shares per contract).
    multiplier: int = 100


@dataclass
class SessionConfig:
    """Trading session time constraints (all times Eastern)."""

    market_open: time = field(default_factory=lambda: time(9, 30))
    market_close: time = field(default_factory=lambda: time(16, 0))

    # Do not open new positions within this many minutes of close.
    # 0 disables the rule.
    no_trade_before_close_mins: int = 15

    # Do not open new positions within this many minutes of open
    # (fills are wide right at the bell).
    no_trade_after_open_mins: int = 2


@dataclass
class ExitConfig:
    """Rules for closing existing positions."""

    # Take-profit: close when unrealized P&L reaches this fraction of
    # the maximum theoretical profit of the structure.
    # None = disabled.
    target_profit_pct: Optional[float] = 0.50   # close at 50% of max profit

    # Stop-loss: close when unrealized loss reaches this fraction of
    # the maximum theoretical loss (cost paid, or spread width - credit).
    # None = disabled.
    stop_loss_pct: Optional[float] = 1.00       # let it ride to full loss by default

    # Close any position whose DTE at update time falls at or below this value.
    # Avoids expiry/pin-risk complications.
    close_at_dte: int = 1

    # Force-close positions that have been open longer than this many bars,
    # regardless of P&L. 0 = disabled.
    max_holding_bars: int = 0

    # When closing a position at expiry simulation (DTE=0), use intrinsic
    # value calculation instead of a market quote (approximates exercise).
    use_intrinsic_at_expiry: bool = True


@dataclass
class RiskConfig:
    """Hard risk controls that gate new-position opens."""

    # Maximum realized + unrealized loss for the session (positive dollars).
    # New positions are blocked once this is exceeded.
    max_daily_loss: float = 500.0

    # Maximum aggregate open risk across all positions (positive dollars).
    # Open risk = sum of max_loss per position (cost for debits, width-credit for spreads).
    max_open_risk: float = 2000.0

    # Maximum number of simultaneously open positions.
    max_concurrent_positions: int = 5

    # After any position closes at a loss greater than this (in dollars),
    # block new positions for `cooldown_minutes`. 0 = no cooldown.
    cooldown_after_loss: float = 100.0
    cooldown_minutes: int = 30

    # If True, no new positions may be opened (existing ones can still close).
    # Set programmatically to halt the simulator after a critical event.
    kill_switch: bool = False

    # If True, log a warning but do not block when max_spread_pct is exceeded.
    # If False (default), reject the fill (not a risk block — just a fill quality check).
    allow_wide_spreads: bool = False


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class SimulatorConfig:
    """Complete simulator configuration."""

    fill: FillConfig = field(default_factory=FillConfig)
    fees: FeeConfig = field(default_factory=FeeConfig)
    contracts: ContractSelectionConfig = field(default_factory=ContractSelectionConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    # Simulator name / label (used in log entries).
    label: str = "paper_options_sim_v1"
