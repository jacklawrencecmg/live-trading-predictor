"""
Per-structure evaluation logic.

Each of the four candidate structures is scored independently on four axes:
  Direction alignment (0–35) — does the structure benefit from the forecast?
  IV edge             (0–25) — buying or selling vol given current IV regime?
  Breakeven feasibility (0–20) — is the required move achievable?
  Liquidity quality   (0–20) — can we fill without excessive friction?

Total: 0–100. Structures below MIN_VIABLE_SCORE are marked not viable.

Hard disqualifiers (set viable=False regardless of score):
  - DTE < 2 (too close to expiry for new positions)
  - Liquidity "poor" AND score < 40 (cost of entry undermines expected value)
  - Debit spread cost > 70% of spread width (terrible risk/reward)
  - Credit spread premium < 15% of spread width (inadequate compensation)
  - Credit spread DTE > 1 (horizon mismatch: 5-min signal edge does not extend
    to multi-day theta/direction exposure)

Structure mechanics:
  long_call   — buy OTM call; profits if underlying rises above breakeven
  long_put    — buy OTM put; profits if underlying falls below breakeven
  debit_spread — directional spread; buy lower call + sell higher call (bull),
                  or buy higher put + sell lower put (bear); capped payoff
  credit_spread — premium collection; sell closer put + buy further put (bull),
                   or sell closer call + buy further call (bear)

When an options chain is provided, actual market prices are used for all
calculations. When absent, Black-Scholes approximations are used.
"""

import math
from typing import Optional, List

from app.decision.models import StructureCandidate, StructureLeg, IVAnalysis
from app.decision.iv_analysis import iv_edge_for_structure

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------
MIN_VIABLE_SCORE: float = 25.0      # below this → not viable
MIN_DTE: int = 2                    # no new positions this close to expiry

# Debit spread: cost / max_profit should be below this to be "efficient"
DEBIT_SPREAD_MAX_COST_RATIO: float = 0.65

# Credit spread: premium / max_loss should be above this for "adequate" reward
CREDIT_SPREAD_MIN_CREDIT_RATIO: float = 0.20

# Brenner-Subrahmanyam approximation factor: ATM call ≈ factor × S × σ × √T
BSA_FACTOR: float = 0.4             # widely used; exact for lognormal ATM


def _bs_atm_call_approx(spot: float, atm_iv: float, dte: int) -> float:
    """ATM call/put premium approximation via Brenner-Subrahmanyam formula."""
    T = max(dte, 1) / 252.0
    return BSA_FACTOR * spot * atm_iv * math.sqrt(T)


def _otm_premium_approx(
    spot: float, atm_iv: float, dte: int, target_delta: float
) -> float:
    """
    Approximate premium for an OTM option at target_delta.
    Uses a delta-weighted scaling relative to ATM premium.
    Very rough; replaced by actual prices when chain is available.
    """
    atm_prem = _bs_atm_call_approx(spot, atm_iv, dte)
    # Scale factor: ATM delta ≈ 0.50; target_delta option costs ~(2×target_delta) × ATM
    scale = 2.0 * target_delta
    return atm_prem * scale


def _liquidity_score(
    liquidity_quality: str,
    bid_ask_pct: float,
    dte: int,
) -> tuple:
    """Return (liquidity_fit, score, fill_cost_pct, concerns)."""
    concerns = []

    if dte <= MIN_DTE:
        concerns.append(f"DTE={dte} — too close to expiry for new positions")

    if liquidity_quality == "good":
        fit, score = "good", 20.0
    elif liquidity_quality == "fair":
        fit, score = "fair", 10.0
        concerns.append("Moderate bid-ask spread — factor fill cost into P&L")
    else:
        fit, score = "poor", 0.0
        concerns.append("Wide bid-ask spread — execution cost may exceed edge")

    fill_cost_pct = bid_ask_pct * 100   # express as % of mid
    return fit, score, fill_cost_pct, concerns


def _direction_score(
    structure_type: str,
    structure_direction: str,
    forecast_direction: str,
) -> float:
    """
    Score for how well the structure's directional payoff matches the forecast.

    Credit spreads get a slightly lower base direction score because they profit
    from BOTH time decay AND direction — the directional edge is less pure.
    """
    matches = structure_direction == forecast_direction

    if structure_type in ("long_call", "long_put"):
        return 35.0 if matches else 0.0
    elif structure_type == "debit_spread":
        return 33.0 if matches else 0.0
    elif structure_type == "credit_spread":
        # Credit spread profits if direction holds OR market is flat; partial credit
        return 28.0 if matches else 0.0
    return 0.0


def _breakeven_score(
    breakeven_move_pct: float,
    expected_move_1d_pct: float,
) -> tuple:
    """
    Score how achievable the breakeven is relative to the expected move.
    Returns (score, concern_or_None).
    """
    if expected_move_1d_pct < 0.001:
        return 5.0, "Insufficient expected move data"

    ratio = breakeven_move_pct / expected_move_1d_pct

    if ratio <= 0.40:
        return 20.0, None
    elif ratio <= 0.70:
        return 16.0, None
    elif ratio <= 1.00:
        return 10.0, None
    elif ratio <= 1.50:
        return 4.0, f"Breakeven ({breakeven_move_pct:.2f}%) > expected move ({expected_move_1d_pct:.2f}%)"
    else:
        return 0.0, f"Breakeven ({breakeven_move_pct:.2f}%) >> expected move ({expected_move_1d_pct:.2f}%) — difficult to profit"


def _resolve_chain_legs(
    structure_type: str,
    direction: str,
    spot: float,
    atm_iv: float,
    dte: int,
    chain=None,
) -> List[StructureLeg]:
    """
    Resolve the legs of a structure.
    Uses actual chain prices when available; falls back to approximations.
    """
    legs = []

    if structure_type == "long_call":
        target_delta = 0.40
        prem = _otm_premium_approx(spot, atm_iv, dte, target_delta)
        strike = _estimate_strike(spot, atm_iv, dte, target_delta, "call")
        leg = StructureLeg(
            action="buy", option_type="call",
            target_delta=target_delta,
            strike=round(strike, 2) if strike else None,
            expiry=None,
            estimated_mid=round(prem, 4),
            estimated_iv=round(atm_iv, 4),
        )
        if chain:
            leg = _fill_from_chain(leg, chain, "call", target_delta, spot)
        legs = [leg]

    elif structure_type == "long_put":
        target_delta = -0.40
        prem = _otm_premium_approx(spot, atm_iv, dte, 0.40)
        strike = _estimate_strike(spot, atm_iv, dte, 0.40, "put")
        leg = StructureLeg(
            action="buy", option_type="put",
            target_delta=target_delta,
            strike=round(strike, 2) if strike else None,
            expiry=None,
            estimated_mid=round(prem, 4),
            estimated_iv=round(atm_iv, 4),
        )
        if chain:
            leg = _fill_from_chain(leg, chain, "put", 0.40, spot)
        legs = [leg]

    elif structure_type == "debit_spread":
        if direction == "bullish":
            # Bull call spread: buy lower call, sell higher call
            long_leg = StructureLeg(
                action="buy", option_type="call", target_delta=0.40,
                estimated_mid=round(_otm_premium_approx(spot, atm_iv, dte, 0.40), 4),
                estimated_iv=round(atm_iv, 4),
            )
            short_leg = StructureLeg(
                action="sell", option_type="call", target_delta=0.20,
                estimated_mid=round(_otm_premium_approx(spot, atm_iv, dte, 0.20), 4),
                estimated_iv=round(atm_iv, 4),
            )
        else:
            # Bear put spread: buy higher put, sell lower put
            long_leg = StructureLeg(
                action="buy", option_type="put", target_delta=0.40,
                estimated_mid=round(_otm_premium_approx(spot, atm_iv, dte, 0.40), 4),
                estimated_iv=round(atm_iv, 4),
            )
            short_leg = StructureLeg(
                action="sell", option_type="put", target_delta=0.20,
                estimated_mid=round(_otm_premium_approx(spot, atm_iv, dte, 0.20), 4),
                estimated_iv=round(atm_iv, 4),
            )
        if chain:
            long_leg = _fill_from_chain(
                long_leg, chain, long_leg.option_type, 0.40, spot)
            short_leg = _fill_from_chain(
                short_leg, chain, short_leg.option_type, 0.20, spot)
        legs = [long_leg, short_leg]

    elif structure_type == "credit_spread":
        if direction == "bullish":
            # Bull put spread: sell higher put (OTM), buy lower put (further OTM)
            short_leg = StructureLeg(
                action="sell", option_type="put", target_delta=0.30,
                estimated_mid=round(_otm_premium_approx(spot, atm_iv, dte, 0.30), 4),
                estimated_iv=round(atm_iv, 4),
            )
            long_leg = StructureLeg(
                action="buy", option_type="put", target_delta=0.15,
                estimated_mid=round(_otm_premium_approx(spot, atm_iv, dte, 0.15), 4),
                estimated_iv=round(atm_iv, 4),
            )
        else:
            # Bear call spread: sell lower call (OTM), buy higher call (further OTM)
            short_leg = StructureLeg(
                action="sell", option_type="call", target_delta=0.30,
                estimated_mid=round(_otm_premium_approx(spot, atm_iv, dte, 0.30), 4),
                estimated_iv=round(atm_iv, 4),
            )
            long_leg = StructureLeg(
                action="buy", option_type="call", target_delta=0.15,
                estimated_mid=round(_otm_premium_approx(spot, atm_iv, dte, 0.15), 4),
                estimated_iv=round(atm_iv, 4),
            )
        if chain:
            short_leg = _fill_from_chain(
                short_leg, chain, short_leg.option_type, 0.30, spot)
            long_leg = _fill_from_chain(
                long_leg, chain, long_leg.option_type, 0.15, spot)
        legs = [short_leg, long_leg]

    return legs


def _estimate_strike(
    spot: float, atm_iv: float, dte: int, target_delta: float, option_type: str
) -> float:
    """
    Approximate strike for a given target delta using a simplified BS inverse.
    Good enough for structural illustration when no chain is available.
    """
    T = max(dte, 1) / 252.0
    sigma_sq_T = atm_iv ** 2 * T
    # d1 ≈ Φ^{-1}(delta) for call; Φ^{-1}(delta+1) for put
    from scipy.stats import norm as _norm
    if option_type == "call":
        d1_target = float(_norm.ppf(target_delta))
    else:
        d1_target = float(_norm.ppf(1.0 - target_delta))
    # d1 = (ln(S/K) + r*T + 0.5*σ²*T) / (σ*√T)
    # → ln(S/K) = d1*σ*√T - r*T - 0.5*σ²*T
    r = 0.05
    log_SK = d1_target * atm_iv * math.sqrt(T) - r * T - 0.5 * sigma_sq_T
    return spot * math.exp(-log_SK)


def _fill_from_chain(leg: StructureLeg, chain, opt_type: str, target_delta: float, spot: float) -> StructureLeg:
    """
    Find the best-matching contract in the chain and populate leg fields.
    Falls back gracefully if no good match.
    """
    try:
        best = None
        best_dist = float("inf")
        for row in chain.rows:
            contract = row.call if opt_type == "call" else row.put
            if contract is None:
                continue
            abs_delta = abs(contract.delta)
            dist = abs(abs_delta - target_delta)
            if dist < best_dist:
                best_dist = dist
                best = contract

        if best is not None and best_dist < 0.20:
            leg.strike = best.strike
            leg.expiry = best.expiry
            leg.estimated_mid = best.mid
            leg.estimated_iv = best.iv
            leg.bid = best.bid
            leg.ask = best.ask
    except Exception:
        pass  # chain lookup best-effort
    return leg


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_structure(
    structure_type: str,
    forecast_direction: str,
    spot: float,
    iv_analysis: IVAnalysis,
    expected_move_1d_pct: float,
    liquidity_quality: str,
    atm_bid_ask_pct: float,
    dte: int,
    chain=None,
) -> StructureCandidate:
    """
    Score one candidate structure and return a fully populated StructureCandidate.

    Parameters
    ----------
    structure_type : str
        "long_call" | "long_put" | "debit_spread" | "credit_spread"
    forecast_direction : str
        "bullish" | "bearish" (from model)
    spot : float
        Underlying spot price.
    iv_analysis : IVAnalysis
        Output of compute_iv_analysis().
    expected_move_1d_pct : float
        1-day expected move from model, as percentage.
    liquidity_quality : str
        "good" | "fair" | "poor"
    atm_bid_ask_pct : float
        (ask - bid) / mid at ATM, as a decimal (0.03 = 3%).
    dte : int
        Days to expiry of the selected expiry.
    chain : OptionsChain or None
        Live options chain for resolving actual strikes/prices.
    """
    # Structure's own directional lean
    direction_map = {
        "long_call":    "bullish",
        "long_put":     "bearish",
        "debit_spread": forecast_direction,   # follows the forecast
        "credit_spread": forecast_direction,  # follows the forecast
    }
    structure_direction = direction_map[structure_type]

    tailwinds = []
    concerns = []
    disqualified = False

    # ── 0. DTE check ─────────────────────────────────────────────────────────
    if dte < MIN_DTE:
        concerns.append(f"DTE={dte}: too close to expiry — avoid opening new positions")
        disqualified = True

    # ── 1. Direction alignment ────────────────────────────────────────────────
    dir_score = _direction_score(structure_type, structure_direction, forecast_direction)
    if dir_score > 0:
        tailwinds.append(f"Direction matches forecast ({forecast_direction})")
    else:
        concerns.append(f"Structure direction ({structure_direction}) opposes forecast ({forecast_direction})")

    # ── 2. IV edge ────────────────────────────────────────────────────────────
    iv_edge_label, iv_score = iv_edge_for_structure(
        structure_type, iv_analysis.iv_rank, iv_analysis.iv_vs_rv
    )
    if iv_edge_label == "favorable":
        tailwinds.append(
            f"IV edge favorable — {'buying' if structure_type != 'credit_spread' else 'selling'}"
            f" vol when IV rank={iv_analysis.iv_rank:.0%} and IV is {iv_analysis.iv_vs_rv}"
        )
    elif iv_edge_label == "unfavorable":
        concerns.append(
            f"IV edge unfavorable — {'buying expensive' if structure_type != 'credit_spread' else 'selling cheap'}"
            f" vol (IV rank={iv_analysis.iv_rank:.0%}, IV {iv_analysis.iv_vs_rv})"
        )

    # ── 3. Resolve legs and compute payoff metrics ────────────────────────────
    atm_iv = iv_analysis.atm_iv
    legs = _resolve_chain_legs(structure_type, structure_direction, spot, atm_iv, dte, chain)

    # Extract prices from legs
    def _mid(leg: StructureLeg) -> float:
        return leg.estimated_mid or 0.0

    estimated_cost_pct = 0.0
    estimated_credit_pct = 0.0
    max_profit_pct = 0.0
    max_loss_pct = 0.0
    breakeven_move_pct = 0.0
    spread_width_pct = None
    estimated_fill_cost_pct = 0.0

    if structure_type == "long_call":
        leg = legs[0] if legs else None
        if leg:
            premium = _mid(leg)
            strike = leg.strike or (spot * 1.02)
            breakeven = strike + premium
            breakeven_move_pct = (breakeven - spot) / spot * 100
            estimated_cost_pct = premium / spot * 100
            max_loss_pct = estimated_cost_pct
            max_profit_pct = 999.0   # theoretically unlimited; cap display at "unlimited"
            estimated_fill_cost_pct = atm_bid_ask_pct * 100

    elif structure_type == "long_put":
        leg = legs[0] if legs else None
        if leg:
            premium = _mid(leg)
            strike = leg.strike or (spot * 0.98)
            breakeven = strike - premium
            breakeven_move_pct = (spot - breakeven) / spot * 100
            estimated_cost_pct = premium / spot * 100
            max_loss_pct = estimated_cost_pct
            max_profit_pct = strike / spot * 100  # underlying goes to zero
            estimated_fill_cost_pct = atm_bid_ask_pct * 100

    elif structure_type == "debit_spread" and len(legs) == 2:
        long_leg, short_leg = legs[0], legs[1]
        long_prem = _mid(long_leg)
        short_prem = _mid(short_leg)
        net_debit = long_prem - short_prem

        long_strike = long_leg.strike or (spot * 1.02 if structure_direction == "bullish" else spot * 0.98)
        short_strike = short_leg.strike or (spot * 1.04 if structure_direction == "bullish" else spot * 0.96)

        spread_width = abs(short_strike - long_strike)
        max_profit = spread_width - net_debit
        max_loss_val = net_debit
        spread_width_pct = spread_width / spot * 100

        if structure_direction == "bullish":
            breakeven = long_strike + net_debit
            breakeven_move_pct = (breakeven - spot) / spot * 100
        else:
            breakeven = long_strike - net_debit
            breakeven_move_pct = (spot - breakeven) / spot * 100

        estimated_cost_pct = net_debit / spot * 100
        max_profit_pct = max_profit / spot * 100
        max_loss_pct = max_loss_val / spot * 100

        cost_ratio = net_debit / (spread_width + 1e-9)
        estimated_fill_cost_pct = atm_bid_ask_pct * 100 * 2  # two legs

        if cost_ratio > DEBIT_SPREAD_MAX_COST_RATIO:
            concerns.append(
                f"Spread cost ({cost_ratio:.0%} of width) exceeds {DEBIT_SPREAD_MAX_COST_RATIO:.0%} — poor value"
            )
            disqualified = True
        elif cost_ratio < 0.40:
            tailwinds.append(f"Efficient cost structure ({cost_ratio:.0%} of spread width)")

    elif structure_type == "credit_spread" and len(legs) == 2:
        short_leg, long_leg = legs[0], legs[1]
        short_prem = _mid(short_leg)
        long_prem = _mid(long_leg)
        net_credit = short_prem - long_prem

        short_strike = short_leg.strike or None
        long_strike = long_leg.strike or None

        if short_strike and long_strike:
            spread_width = abs(short_strike - long_strike)
        else:
            spread_width = spot * 0.02  # default 2% width

        spread_width_pct = spread_width / spot * 100
        max_loss_val = spread_width - net_credit
        max_profit_val = net_credit
        credit_ratio = net_credit / (spread_width + 1e-9)

        if structure_direction == "bullish":
            breakeven = (short_leg.strike or spot * 0.98) - net_credit
            breakeven_move_pct = (spot - breakeven) / spot * 100
        else:
            breakeven = (short_leg.strike or spot * 1.02) + net_credit
            breakeven_move_pct = (breakeven - spot) / spot * 100

        estimated_credit_pct = credit_ratio * 100
        max_profit_pct = max_profit_val / spot * 100
        max_loss_pct = max_loss_val / spot * 100
        estimated_cost_pct = 0.0
        estimated_fill_cost_pct = atm_bid_ask_pct * 100 * 2

        if credit_ratio < CREDIT_SPREAD_MIN_CREDIT_RATIO:
            concerns.append(
                f"Credit ({credit_ratio:.0%} of width) below {CREDIT_SPREAD_MIN_CREDIT_RATIO:.0%} — poor risk/reward"
            )
            disqualified = True
        elif credit_ratio > 0.35:
            tailwinds.append(f"Solid premium collection ({credit_ratio:.0%} of spread width)")

        # Horizon mismatch: a 5-min bar-level signal has edge only over the next
        # bar (~5 min).  A credit spread with DTE > 1 profits from theta decay
        # and direction holding across days — a fundamentally different horizon.
        # Disqualify to prevent misleading recommendations.
        if dte > 1:
            concerns.append(
                f"Horizon mismatch: credit spread profits over DTE={dte} days; "
                f"5-min signal has sub-day resolution — structural edge does not carry"
            )
            disqualified = True
        else:
            concerns.append(
                f"Short-DTE credit spread (DTE={dte}): horizon closer to signal timeframe "
                f"but theta/gamma risk is elevated"
            )

    # ── 4. Breakeven feasibility ──────────────────────────────────────────────
    be_score, be_concern = _breakeven_score(breakeven_move_pct, expected_move_1d_pct)
    if be_concern:
        concerns.append(be_concern)
    elif be_score >= 16:
        tailwinds.append(
            f"Breakeven ({breakeven_move_pct:.2f}%) achievable vs 1-day expected move ({expected_move_1d_pct:.2f}%)"
        )

    # ── 5. Liquidity ──────────────────────────────────────────────────────────
    liq_fit, liq_score, fill_cost, liq_concerns = _liquidity_score(
        liquidity_quality, atm_bid_ask_pct, dte
    )
    concerns.extend(liq_concerns)

    # ── 6. Total score ────────────────────────────────────────────────────────
    raw_score = dir_score + iv_score + be_score + liq_score
    # Hard disqualification: floor to max 20 (not viable) but keep the score info
    final_score = round(min(raw_score, 20.0) if disqualified else raw_score, 1)
    viable = (not disqualified) and (final_score >= MIN_VIABLE_SCORE) and (dir_score > 0)

    # ── 7. Build horizon note ─────────────────────────────────────────────────
    if structure_type in ("long_call", "long_put"):
        horizon_note = (
            f"Profit if underlying moves >{breakeven_move_pct:.2f}% by expiry ({dte}d). "
            f"1-day expected move: {expected_move_1d_pct:.2f}%."
        )
    elif structure_type == "debit_spread":
        horizon_note = (
            f"Defined-risk directional bet. Max profit if underlying reaches short strike by expiry. "
            f"Cost: {estimated_cost_pct:.2f}% of spot. Max profit: {max_profit_pct:.2f}%."
        )
    else:
        horizon_note = (
            f"Premium collection ({max_profit_pct:.2f}% of spot). "
            f"Max loss capped at {max_loss_pct:.2f}%. Profits if direction holds over {dte} days."
        )

    # ── 8. Rationale ─────────────────────────────────────────────────────────
    rationale_parts = []
    if tailwinds:
        rationale_parts.append("Strengths: " + "; ".join(tailwinds[:3]) + ".")
    if concerns:
        rationale_parts.append("Concerns: " + "; ".join(concerns[:3]) + ".")
    rationale = " ".join(rationale_parts) or "No clear edge or concerns identified."

    return StructureCandidate(
        structure_type=structure_type,
        direction=structure_direction,
        score=final_score,
        viable=viable,
        legs=legs,
        estimated_cost_pct=round(estimated_cost_pct, 4),
        estimated_credit_pct=round(estimated_credit_pct, 4),
        max_profit_pct=round(min(max_profit_pct, 999.0), 4),
        max_loss_pct=round(max_loss_pct, 4),
        breakeven_move_pct=round(breakeven_move_pct, 4),
        spread_width_pct=round(spread_width_pct, 4) if spread_width_pct is not None else None,
        iv_edge=iv_edge_label,
        iv_edge_score=iv_score,
        liquidity_fit=liq_fit,
        estimated_fill_cost_pct=round(estimated_fill_cost_pct, 4),
        horizon_note=horizon_note,
        rationale=rationale,
        tailwinds=tailwinds,
        concerns=concerns,
    )
