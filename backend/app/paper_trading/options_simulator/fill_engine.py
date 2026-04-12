"""
Fill simulation engine.

Translates a list of option legs + market quotes into simulated fill prices,
net premium, and fee totals. Three fill methods are supported:

  MIDPOINT    — fill at (bid + ask) / 2. Optimistic; reasonable for liquid
                names where mid is often achievable in practice.

  BID_ASK     — buy at ask, sell at bid. Assumes no price improvement.
                Conservative for single-leg; realistic for illiquid chains.

  CONSERVATIVE — BID_ASK + additional slippage proportional to the spread.
                 buy_fill  = ask + slippage_factor × (ask − bid)
                 sell_fill = bid − slippage_factor × (ask − bid)
                 Models partial adverse impact when size is meaningful
                 or the market is thin.

Returned FillResult.net_premium_per_share is SIGNED:
  positive = net debit (you paid; applies to long options and debit spreads)
  negative = net credit (you received; applies to credit spreads)

All prices are per-share (not per-contract). Multiply by multiplier × contracts
to get dollar exposure.
"""

from typing import List
from app.paper_trading.options_simulator.config import FillConfig, FeeConfig, FillMethod
from app.paper_trading.options_simulator.models import SimLeg, FillResult


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_fill(
    legs: List[SimLeg],
    quotes: list,            # List[LegQuote], same length and order as legs
    fill_config: FillConfig,
    fee_config: FeeConfig,
    contracts: int,
    multiplier: int = 100,
) -> FillResult:
    """
    Compute simulated fill for a multi-leg options structure.

    Parameters
    ----------
    legs : List[SimLeg]
        Ordered list of option legs (action, option_type, strike etc.).
        This list is mutated in-place: fill_price and slippage are written.
    quotes : List[LegQuote]
        Current market quotes, parallel to `legs`.
    fill_config : FillConfig
    fee_config : FeeConfig
    contracts : int
    multiplier : int

    Returns
    -------
    FillResult with net_premium_per_share, net_premium_dollars, total_fees.
    """
    if len(legs) != len(quotes):
        raise ValueError(
            f"legs ({len(legs)}) and quotes ({len(quotes)}) must have equal length"
        )

    warnings: list = []
    net_pps = 0.0   # net premium per share (signed)

    for leg, quote in zip(legs, quotes):
        fill_price, slippage = _fill_one_leg(leg, quote, fill_config)

        # Record on leg
        leg.fill_price = fill_price
        leg.fill_slippage = slippage
        leg.fill_bid = quote.bid
        leg.fill_ask = quote.ask
        leg.fill_mid = quote.mid

        # Wide-spread quality warning
        if quote.mid > 0:
            spread_pct = (quote.ask - quote.bid) / quote.mid
            if spread_pct > fill_config.max_spread_pct:
                warnings.append(
                    f"{leg.action} {leg.option_type} "
                    f"spread {spread_pct:.1%} > max {fill_config.max_spread_pct:.1%}"
                )

        # Net premium: buying adds cost (positive), selling subtracts (negative)
        sign = 1.0 if leg.action == "buy" else -1.0
        net_pps += sign * fill_price

    # Fees
    total_fees = _compute_fees(legs, contracts, fee_config)
    net_dollars = net_pps * multiplier * contracts

    return FillResult(
        legs=legs,
        net_premium_per_share=round(net_pps, 4),
        net_premium_dollars=round(net_dollars, 2),
        total_fees=round(total_fees, 2),
        fill_quality_warnings=warnings,
    )


def estimate_close_fill(
    legs: List[SimLeg],
    quotes: list,           # List[LegQuote]
    fill_config: FillConfig,
    fee_config: FeeConfig,
    contracts: int,
    multiplier: int = 100,
) -> FillResult:
    """
    Estimate the fill to CLOSE (unwind) an existing position.

    Reverses the action of each leg (buy→sell, sell→buy) and runs the
    same fill logic. The returned net_premium_per_share is the proceeds
    from unwinding (positive = you received money, negative = you paid).
    """
    closing_legs = [
        SimLeg(
            action="sell" if leg.action == "buy" else "buy",
            option_type=leg.option_type,
            target_delta=leg.target_delta,
            strike=leg.strike,
            expiry=leg.expiry,
        )
        for leg in legs
    ]
    return simulate_fill(
        closing_legs, quotes, fill_config, fee_config, contracts, multiplier
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _fill_one_leg(leg: SimLeg, quote, fill_config: FillConfig):
    """
    Return (fill_price, slippage_per_share) for a single leg.

    For buy legs   : fill_price >= ask (CONSERVATIVE) or == ask (BID_ASK) or == mid (MIDPOINT)
    For sell legs  : fill_price <= bid (CONSERVATIVE) or == bid (BID_ASK) or == mid (MIDPOINT)
    """
    bid = max(quote.bid, fill_config.min_tick)
    ask = max(quote.ask, fill_config.min_tick)
    mid = max(quote.mid, (bid + ask) / 2)

    spread = max(ask - bid, 0.0)

    if fill_config.method == FillMethod.MIDPOINT:
        price = mid
        slippage = 0.0

    elif fill_config.method == FillMethod.BID_ASK:
        if leg.action == "buy":
            price = ask
            slippage = ask - mid
        else:
            price = bid
            slippage = mid - bid

    else:   # CONSERVATIVE
        sf = fill_config.slippage_factor
        if leg.action == "buy":
            price = ask + sf * spread
            slippage = price - mid
        else:
            price = max(bid - sf * spread, fill_config.min_tick)
            slippage = mid - price

    price = max(round(price, 4), fill_config.min_tick)
    return price, round(slippage, 4)


def _compute_fees(legs: List[SimLeg], contracts: int, fee_config: FeeConfig) -> float:
    """
    Total round-trip commission for one order (all legs, one direction).

    Per-leg fee = max(min_per_leg, per_contract × contracts) + regulatory
    Optionally capped at max_per_leg if > 0.
    """
    total = 0.0
    for _ in legs:
        leg_fee = fee_config.per_contract * contracts
        leg_fee += fee_config.regulatory_fee_per_contract * contracts
        if fee_config.min_per_leg > 0:
            leg_fee = max(leg_fee, fee_config.min_per_leg)
        if fee_config.max_per_leg > 0:
            leg_fee = min(leg_fee, fee_config.max_per_leg)
        total += leg_fee
    return total


def compute_open_risk(
    net_premium_per_share: float,
    structure_type: str,
    spread_width_per_share: float = 0.0,
    contracts: int = 1,
    multiplier: int = 100,
) -> float:
    """
    Compute max possible loss (positive dollars) for a structure.

    Long options (debit structures):
        max_loss = cost paid = net_premium_per_share × multiplier × contracts

    Credit spreads:
        max_loss = (spread_width - credit_received) × multiplier × contracts
        NOTE: credit_received = abs(net_premium_per_share)

    For naked short options this would be theoretically unlimited; this
    simulator does not support uncapped risk structures.
    """
    if net_premium_per_share >= 0:
        # Debit paid — max loss is the entire premium
        return abs(net_premium_per_share) * multiplier * contracts

    # Credit received
    credit = abs(net_premium_per_share)
    if spread_width_per_share > 0 and structure_type == "credit_spread":
        max_loss_pps = max(spread_width_per_share - credit, 0.0)
    else:
        # Unknown spread width — treat credit as zero protection (conservative)
        max_loss_pps = credit  # at minimum, the received credit is at risk
    return max_loss_pps * multiplier * contracts
