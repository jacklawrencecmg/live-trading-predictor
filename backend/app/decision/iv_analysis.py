"""
IV vs Realized Volatility analysis.

This module answers three questions:
  1. Is implied volatility cheap, fair, or rich relative to recent realized vol?
  2. What 1-day move is the options market pricing in?
  3. What does the model's own realized-vol forecast imply about that same move?

The IV/RV comparison is the primary determinant of structure selection:
  - Rich IV  → prefer selling premium (credit spreads)
  - Cheap IV → prefer buying premium (outright longs, debit spreads)
  - Fair IV  → direction quality dominates structure choice

IV Rank complements the IV/RV ratio by normalizing current IV against its
own historical range — useful when IV is absolutely high but at a historical
low (IV Rank = 0.20 means cheap despite being "high" in absolute terms).

All inputs are annualized vols (0.20 = 20%).
"""

import math
from app.decision.models import IVAnalysis


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# IV/RV ratio classification
_RICH_RATIO: float = 1.30     # IV more than 30% above RV → rich
_CHEAP_RATIO: float = 0.85    # IV more than 15% below RV → cheap

# IV rank thresholds (0-1 scale)
_HIGH_IV_RANK: float = 0.65   # above this → elevated IV (bias toward selling)
_LOW_IV_RANK: float = 0.35    # below this → suppressed IV (bias toward buying)


def compute_iv_analysis(
    atm_iv: float,
    realized_vol_ann: float,
    iv_rank: float,
) -> IVAnalysis:
    """
    Compute the full IV context from three numbers.

    Parameters
    ----------
    atm_iv : float
        At-the-money implied volatility, annualized (e.g. 0.25 = 25%).
        If unavailable, pass 0.0 — the engine will fall back to a RV-based estimate.
    realized_vol_ann : float
        Annualized realized volatility from recent price history (e.g. 0.18 = 18%).
    iv_rank : float
        IV rank in [0, 1] over some lookback (often 1-year).

    Returns
    -------
    IVAnalysis
    """
    # Fall back: if ATM IV is missing, estimate from RV using a standard VRP ratio
    _VRP_PREMIUM = 1.10   # options typically priced at ~10% premium to RV on average
    if atm_iv <= 0.001:
        atm_iv = realized_vol_ann * _VRP_PREMIUM if realized_vol_ann > 0.001 else 0.20

    if realized_vol_ann <= 0.001:
        realized_vol_ann = 0.01   # floor

    iv_rv_ratio = atm_iv / realized_vol_ann

    # Primary classification: use BOTH ratio and rank for robustness
    # A stock can have "low" absolute IV that is still historically high (rank 0.70)
    if iv_rv_ratio >= _RICH_RATIO and iv_rank >= _LOW_IV_RANK:
        iv_vs_rv = "rich"
    elif iv_rv_ratio <= _CHEAP_RATIO and iv_rank <= _HIGH_IV_RANK:
        iv_vs_rv = "cheap"
    else:
        iv_vs_rv = "fair"

    # 1-day implied move ≈ ATM_IV / sqrt(252)
    iv_1d = atm_iv / math.sqrt(252) * 100      # as percentage
    rv_1d = realized_vol_ann / math.sqrt(252) * 100

    return IVAnalysis(
        atm_iv=round(atm_iv, 4),
        realized_vol_ann=round(realized_vol_ann, 4),
        iv_rank=round(iv_rank, 4),
        iv_rv_ratio=round(iv_rv_ratio, 3),
        iv_vs_rv=iv_vs_rv,
        iv_implied_1d_move_pct=round(iv_1d, 3),
        rv_implied_1d_move_pct=round(rv_1d, 3),
    )


def iv_edge_for_structure(
    structure_type: str,
    iv_rank: float,
    iv_vs_rv: str,
) -> tuple:
    """
    Return (iv_edge_label, iv_edge_score) for a structure given the IV context.

    Parameters
    ----------
    structure_type : str
        One of: "long_call", "long_put", "debit_spread", "credit_spread"
    iv_rank : float
        IV rank in [0, 1].
    iv_vs_rv : str
        "cheap" | "fair" | "rich"

    Returns
    -------
    (iv_edge_label, iv_edge_score)
        iv_edge_label : "favorable" | "neutral" | "unfavorable"
        iv_edge_score : float 0–25 (raw component in the structure score)
    """
    is_buying_vol = structure_type in ("long_call", "long_put")
    is_debit_spread = structure_type == "debit_spread"    # net long vol, but hedged
    is_credit_spread = structure_type == "credit_spread"  # net short vol

    if is_buying_vol:
        # Buying vol: want cheap IV
        if iv_rank < 0.25:
            label, score = "favorable", 25.0
        elif iv_rank < 0.40:
            label, score = "favorable", 20.0
        elif iv_rank < 0.60:
            label, score = "neutral", 12.0
        elif iv_rank < 0.75:
            label, score = "unfavorable", 5.0
        else:
            label, score = "unfavorable", 0.0
        # Reinforce with iv_vs_rv
        if iv_vs_rv == "cheap":
            score = min(25.0, score + 3.0)
        elif iv_vs_rv == "rich":
            score = max(0.0, score - 3.0)

    elif is_debit_spread:
        # Net long vol but the short leg offsets some IV cost; less sensitive to IV
        if iv_rank < 0.35:
            label, score = "favorable", 22.0
        elif iv_rank < 0.55:
            label, score = "neutral", 15.0
        elif iv_rank < 0.70:
            label, score = "neutral", 10.0
        else:
            label, score = "unfavorable", 4.0
        if iv_vs_rv == "cheap":
            score = min(22.0, score + 2.0)

    elif is_credit_spread:
        # Selling vol: want rich IV
        if iv_rank >= 0.75:
            label, score = "favorable", 25.0
        elif iv_rank >= 0.60:
            label, score = "favorable", 20.0
        elif iv_rank >= 0.40:
            label, score = "neutral", 12.0
        elif iv_rank >= 0.25:
            label, score = "unfavorable", 5.0
        else:
            label, score = "unfavorable", 0.0
        if iv_vs_rv == "rich":
            score = min(25.0, score + 3.0)
        elif iv_vs_rv == "cheap":
            score = max(0.0, score - 3.0)
    else:
        label, score = "neutral", 12.0

    return label, round(score, 1)
