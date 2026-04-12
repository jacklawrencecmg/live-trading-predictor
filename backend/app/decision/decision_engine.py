"""
Decision engine: converts an InferenceResult into a full OptionsDecision.

Orchestration order:
  1. Derive realized vol from the inference expected-move forecast
  2. Compute IV analysis (IV/RV ratio, IV rank classification, 1-day moves)
  3. Determine direction thesis and direction-specific calibrated probability
  4. Scale expected move to 1-day (options pricing context)
  5. Evaluate all four candidate structures (long_call, long_put, debit_spread, credit_spread)
  6. Sort candidates by score; select recommended structure
  7. Compute composite confidence score
  8. Set abstain flag and reason
  9. Return OptionsDecision

Key design constraints:
  - Directional view alone does NOT imply a good structure; IV regime, breakeven
    feasibility, and liquidity all gate the recommendation independently.
  - Regime suppression is a hard block: if the inference layer suppressed the signal,
    all structures are marked non-viable and the decision abstains.
  - Calibration health degrades the confidence score continuously, not via a hard
    threshold, so marginal setups are naturally filtered.
"""

import math
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from app.decision.models import OptionsDecision, StructureCandidate, IVAnalysis
from app.decision.iv_analysis import compute_iv_analysis
from app.decision.structure_evaluator import evaluate_structure

# ---------------------------------------------------------------------------
# Defaults used when live options market data is unavailable
# ---------------------------------------------------------------------------
DEFAULT_DTE: int = 7                  # 1-week horizon assumption
DEFAULT_ATM_BID_ASK_PCT: float = 0.05  # 5% round-trip friction assumption
DEFAULT_LIQUIDITY: str = "fair"

# ---------------------------------------------------------------------------
# 5-minute bar scaling constants
# ---------------------------------------------------------------------------
# 78 five-minute bars per trading day; 252 trading days per year
BARS_PER_DAY: int = 78
TRADING_DAYS: int = 252
ANNUAL_BARS: int = BARS_PER_DAY * TRADING_DAYS   # 19 656

# ---------------------------------------------------------------------------
# Confidence score weights
# ---------------------------------------------------------------------------
_PROB_EDGE_WEIGHT: float = 0.45       # how strongly calibrated prob drives score
_IV_EDGE_WEIGHT: float = 0.30         # IV environment fit
_CALIBRATION_WEIGHT: float = 0.25     # calibration health modifier

# Calibration health multipliers (applied to raw composite)
_HEALTH_MULTIPLIER: Dict[str, float] = {
    "good": 1.00,
    "fair": 0.85,
    "degraded": 0.65,
    "unknown": 0.55,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_options_decision(
    inference_result,
    *,
    atm_iv: float = 0.0,
    iv_rank: float = 0.50,
    expiry: Optional[str] = None,
    dte: Optional[int] = None,
    liquidity_quality: Optional[str] = None,
    atm_bid_ask_pct: Optional[float] = None,
    chain: Optional[Any] = None,
) -> OptionsDecision:
    """
    Convert an InferenceResult into a scored OptionsDecision.

    Parameters
    ----------
    inference_result : InferenceResult
        Output from inference_service; must carry calibrated probs, regime, etc.
    atm_iv : float
        At-the-money implied volatility, annualized (0.25 = 25%).
        If 0.0 or missing, falls back to RV-based estimate.
    iv_rank : float
        IV rank [0, 1] over some lookback period (typically 1 year).
    expiry : str
        Target expiry date string (e.g. "2025-01-17"). Optional display field.
    dte : int
        Days to expiry. Defaults to DEFAULT_DTE if not provided.
    liquidity_quality : str
        "good" | "fair" | "poor". Defaults to DEFAULT_LIQUIDITY.
    atm_bid_ask_pct : float
        Bid-ask spread at ATM as a fraction of mid. Defaults to DEFAULT_ATM_BID_ASK_PCT.
    chain : optional
        Live options chain object. If provided, passed to structure_evaluator for
        actual market price calculations instead of BS approximations.

    Returns
    -------
    OptionsDecision
    """
    r = inference_result

    # Resolve defaults
    dte_val = dte if dte is not None else DEFAULT_DTE
    liq = liquidity_quality or DEFAULT_LIQUIDITY
    ba_pct = atm_bid_ask_pct if atm_bid_ask_pct is not None else DEFAULT_ATM_BID_ASK_PCT
    expiry_str = expiry or _default_expiry(dte_val)

    # -----------------------------------------------------------------------
    # Step 1: Derive realized vol from inference expected_move_pct
    # expected_move_pct is the 1-bar (5-min) move, annualized via:
    #   rv_ann = expected_move_pct/100 * sqrt(ANNUAL_BARS)
    # -----------------------------------------------------------------------
    rv_ann = _annualized_rv(r.expected_move_pct)

    # -----------------------------------------------------------------------
    # Step 2: IV analysis
    # -----------------------------------------------------------------------
    iv_analysis = compute_iv_analysis(atm_iv, rv_ann, iv_rank)

    # -----------------------------------------------------------------------
    # Step 3: Direction thesis and direction-specific calibrated probability
    # -----------------------------------------------------------------------
    regime_suppressed = getattr(r, "regime_suppressed", False)
    # Backward compat: some inference builds store suppression in action
    if not regime_suppressed and r.action == "abstain" and r.abstain_reason:
        regime_suppressed = str(r.abstain_reason).startswith("regime_suppressed")

    cal_up   = r.calibrated_prob_up
    cal_down = r.calibrated_prob_down

    if cal_up > cal_down and cal_up > 0.50:
        direction_thesis = "bullish"
        calibrated_prob = cal_up
    elif cal_down > cal_up and cal_down > 0.50:
        direction_thesis = "bearish"
        calibrated_prob = cal_down
    elif r.action == "abstain":
        direction_thesis = "abstain"
        calibrated_prob = max(cal_up, cal_down)
    else:
        direction_thesis = "neutral"
        calibrated_prob = max(cal_up, cal_down)

    # -----------------------------------------------------------------------
    # Step 4: Expected move scaling
    # -----------------------------------------------------------------------
    # 1-bar expected move is in percent; scale to 1-day for options context
    expected_move_1bar_pct = r.expected_move_pct
    expected_move_1d_pct = _scale_to_1d(expected_move_1bar_pct)

    spot = getattr(r, "spot_price", None) or _extract_spot(r)
    expected_range_low  = spot * (1.0 - expected_move_1d_pct / 100.0)
    expected_range_high = spot * (1.0 + expected_move_1d_pct / 100.0)

    # -----------------------------------------------------------------------
    # Step 5: Evaluate all four candidate structures
    # -----------------------------------------------------------------------
    structure_types = ["long_call", "long_put", "debit_spread", "credit_spread"]
    candidates: List[StructureCandidate] = []

    for stype in structure_types:
        candidate = evaluate_structure(
            structure_type=stype,
            forecast_direction=direction_thesis,
            spot=spot,
            iv_analysis=iv_analysis,
            expected_move_1d_pct=expected_move_1d_pct,
            liquidity_quality=liq,
            atm_bid_ask_pct=ba_pct,
            dte=dte_val,
            chain=chain,
        )
        # Regime suppression overrides all structure viability
        if regime_suppressed:
            candidate.viable = False
            if "regime_suppressed" not in candidate.concerns:
                candidate.concerns.insert(0, "regime suppressed — no new positions")
        candidates.append(candidate)

    # Sort by score descending
    candidates.sort(key=lambda c: c.score, reverse=True)

    # -----------------------------------------------------------------------
    # Step 6: Recommended structure
    # -----------------------------------------------------------------------
    viable_candidates = [c for c in candidates if c.viable]
    recommended_structure: Optional[str] = None
    recommendation_rationale: str = ""

    if viable_candidates:
        best = viable_candidates[0]
        recommended_structure = best.structure_type
        recommendation_rationale = _build_recommendation_rationale(
            best, direction_thesis, iv_analysis, calibrated_prob
        )
    else:
        recommendation_rationale = _build_abstain_rationale(
            candidates, direction_thesis, regime_suppressed,
            r.action, r.abstain_reason
        )

    # -----------------------------------------------------------------------
    # Step 7: Composite confidence score (0–100)
    # -----------------------------------------------------------------------
    confidence_score = _compute_confidence_score(
        calibrated_prob=calibrated_prob,
        direction_thesis=direction_thesis,
        iv_analysis=iv_analysis,
        calibration_health=r.calibration_health,
        signal_quality_score=getattr(r, "signal_quality_score", r.tradeable_confidence * 100),
    )

    # -----------------------------------------------------------------------
    # Step 8: Abstain flag
    # -----------------------------------------------------------------------
    inference_abstained = r.action == "abstain"
    no_viable_structures = len(viable_candidates) == 0
    abstain = regime_suppressed or no_viable_structures or inference_abstained

    abstain_reason: Optional[str] = None
    if regime_suppressed:
        abstain_reason = f"regime_suppressed:{getattr(r, 'suppress_reason', r.abstain_reason)}"
    elif inference_abstained and r.abstain_reason:
        abstain_reason = r.abstain_reason
    elif no_viable_structures:
        abstain_reason = "no_viable_structure"

    # -----------------------------------------------------------------------
    # Step 9: Build and return OptionsDecision
    # -----------------------------------------------------------------------
    return OptionsDecision(
        symbol=r.symbol,
        generated_at=datetime.now(timezone.utc).isoformat(),
        spot_price=round(spot, 4),

        direction_thesis=direction_thesis,
        horizon=f"1-bar (5 min), ~{dte_val}d expiry",

        calibrated_prob=round(calibrated_prob, 4),
        prob_up=round(cal_up, 4),
        prob_down=round(cal_down, 4),
        confidence_band=r.confidence_band,

        expected_move_1bar_pct=round(expected_move_1bar_pct, 4),
        expected_move_1d_pct=round(expected_move_1d_pct, 4),
        expected_range_low=round(expected_range_low, 4),
        expected_range_high=round(expected_range_high, 4),

        iv_analysis=iv_analysis,

        expiry=expiry_str,
        dte=dte_val,
        liquidity_quality=liq,
        atm_bid_ask_pct=round(ba_pct, 4),

        regime=r.regime,
        regime_suppressed=regime_suppressed,
        calibration_health=r.calibration_health,
        signal_quality_score=round(
            getattr(r, "signal_quality_score", r.tradeable_confidence * 100), 2
        ),

        confidence_score=round(confidence_score, 2),
        abstain=abstain,
        abstain_reason=abstain_reason,

        candidates=candidates,
        recommended_structure=recommended_structure,
        recommendation_rationale=recommendation_rationale,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _annualized_rv(expected_move_pct: float) -> float:
    """
    Convert the inference 1-bar expected-move percentage to annualized realized vol.

    expected_move_pct is expressed as a percentage (e.g. 0.65 means 0.65%).
    For a 5-minute bar:
        rv_1bar = expected_move_pct / 100
        rv_ann  = rv_1bar * sqrt(BARS_PER_DAY * TRADING_DAYS)
    """
    if expected_move_pct <= 0:
        return 0.20  # fallback: 20% annualized
    rv_1bar = expected_move_pct / 100.0
    return rv_1bar * math.sqrt(ANNUAL_BARS)


def _scale_to_1d(expected_move_1bar_pct: float) -> float:
    """
    Scale 1-bar (5-min) expected move to a 1-day expected move.
    Assumes independent bar returns: 1d_move ≈ 1bar_move × sqrt(BARS_PER_DAY).
    """
    return expected_move_1bar_pct * math.sqrt(BARS_PER_DAY)


def _extract_spot(r) -> float:
    """Attempt to pull spot price from various inference result shapes."""
    for attr in ("spot_price", "close_price", "last_price", "price"):
        v = getattr(r, attr, None)
        if v and v > 0:
            return float(v)
    return 100.0   # absolute fallback; caller should always supply spot


def _default_expiry(dte: int) -> str:
    """Return a default expiry string roughly `dte` trading days from now."""
    from datetime import timedelta
    # Approximate: dte trading days ≈ dte * 7/5 calendar days
    calendar_days = max(dte, 1) * 7 // 5 + 1
    expiry_date = datetime.now(timezone.utc).date() + timedelta(days=calendar_days)
    return expiry_date.strftime("%Y-%m-%d")


def _compute_confidence_score(
    calibrated_prob: float,
    direction_thesis: str,
    iv_analysis: IVAnalysis,
    calibration_health: str,
    signal_quality_score: float,
) -> float:
    """
    Composite 0–100 confidence score.

    Components:
      - Probability edge (0–50): distance of calibrated_prob from 0.5, scaled
      - IV fit (0–25): how well the IV environment supports the best structure
      - Signal quality (0–25): from inference layer signal quality score (0–100)

    Then multiplied by calibration health degradation multiplier.
    """
    # Probability edge: edge = calibrated_prob - 0.5; map [0, 0.5] → [0, 50]
    if direction_thesis in ("abstain", "neutral"):
        prob_component = 0.0
    else:
        edge = max(calibrated_prob - 0.50, 0.0)   # 0 to 0.5
        prob_component = edge * 100.0              # 0 to 50

    # IV fit: use the IV regime quality as a proxy
    # "fair" IV environment → moderate; rich or cheap (whichever matches structure) → full
    iv_component = _iv_fit_score(iv_analysis, direction_thesis)

    # Signal quality passthrough (scaled from 0–100 to 0–25)
    sq_component = min(signal_quality_score / 100.0, 1.0) * 25.0

    raw = prob_component + iv_component + sq_component   # 0 to 100

    # Apply calibration health multiplier
    multiplier = _HEALTH_MULTIPLIER.get(calibration_health, 0.55)
    return max(0.0, min(100.0, raw * multiplier))


def _iv_fit_score(iv_analysis: IVAnalysis, direction_thesis: str) -> float:
    """
    Return 0–25 score representing how good the IV environment is for a
    directional options trade, regardless of structure.

    Logic: neutral IV (fair, mid rank) → 12.5 baseline.
    Cheap IV favors buying strategies (bullish/bearish); rich IV is a mixed
    signal for directional plays (good for credit but bad for debit).
    """
    if direction_thesis in ("abstain", "neutral"):
        return 12.5  # no directional edge to assess

    iv_rank = iv_analysis.iv_rank
    iv_vs_rv = iv_analysis.iv_vs_rv

    # Base on IV rank absolute level (lower rank → cheaper options → better for buying)
    if iv_rank < 0.30:
        base = 22.0
    elif iv_rank < 0.50:
        base = 16.0
    elif iv_rank < 0.70:
        base = 12.0
    else:
        base = 8.0

    # Reinforce: cheap IV is universally better for directional trades
    if iv_vs_rv == "cheap":
        base = min(25.0, base + 3.0)

    return base


def _build_recommendation_rationale(
    best: StructureCandidate,
    direction_thesis: str,
    iv_analysis: IVAnalysis,
    calibrated_prob: float,
) -> str:
    iv_summary = (
        f"IV rank {iv_analysis.iv_rank:.0%} ({iv_analysis.iv_vs_rv}), "
        f"IV/RV ratio {iv_analysis.iv_rv_ratio:.2f}"
    )
    prob_summary = f"calibrated {direction_thesis} probability {calibrated_prob:.1%}"
    return (
        f"Recommend {best.structure_type.replace('_', ' ')} "
        f"(score {best.score:.0f}/100). "
        f"{prob_summary.capitalize()}. {iv_summary}. "
        f"IV edge: {best.iv_edge}. Liquidity: {best.liquidity_fit}."
    )


def _build_abstain_rationale(
    candidates: List[StructureCandidate],
    direction_thesis: str,
    regime_suppressed: bool,
    action: str,
    abstain_reason: Optional[str],
) -> str:
    if regime_suppressed:
        reason = abstain_reason or "regime suppressed"
        return f"Abstaining: regime conditions prevent new positions ({reason})."
    if action == "abstain":
        return f"Abstaining: inference layer abstained ({abstain_reason or 'low confidence'})."
    top = candidates[0] if candidates else None
    if top:
        return (
            f"Abstaining: highest-scoring structure ({top.structure_type}, "
            f"score {top.score:.0f}) did not clear viability threshold. "
            f"Direction thesis: {direction_thesis}."
        )
    return "Abstaining: no viable structure found."
