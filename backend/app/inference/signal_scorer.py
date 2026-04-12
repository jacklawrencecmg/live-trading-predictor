"""
Signal scoring layer on top of raw model probabilities.

Signal quality score (0-100) combines:
- Probability edge (how far from 50/50)
- Calibration confidence
- Regime suitability
- Volatility context
- Risk limit headroom
"""

from dataclasses import dataclass
from typing import Optional
from app.regime.detector import Regime


@dataclass
class ScoredSignal:
    direction: str           # "up", "down", "no_trade"
    probability: float       # raw model probability
    confidence: float        # abs(p - 0.5) * 2
    confidence_band: tuple   # (low, high) ±1 std estimate
    expected_move_pct: float
    realized_vol_pct: float
    volatility_context: str  # "expanding", "contracting", "normal"
    regime: str
    signal_quality_score: float  # 0-100
    no_trade_reason: Optional[str]
    prob_up: float
    prob_down: float
    explanation: str
    top_features: dict


def score_signal(
    prob_up: float,
    prob_down: float,
    expected_move_pct: float,
    realized_vol_pct: float,
    regime: str,
    no_trade_reason: Optional[str],
    explanation: str,
    top_features: dict,
    min_edge: float = 0.05,   # minimum P(up) - 0.5 for a trade
    min_expected_move: float = 0.10,  # percent
) -> ScoredSignal:

    direction = "no_trade"
    prob = 0.5
    if prob_up > 0.5 + min_edge:
        direction = "up"
        prob = prob_up
    elif prob_down > 0.5 + min_edge:
        direction = "down"
        prob = prob_down

    confidence = abs(prob_up - 0.5) * 2

    # Confidence band: simple heuristic ±0.05
    conf_low = max(0.0, prob - 0.05)
    conf_high = min(1.0, prob + 0.05)

    # Volatility context
    if realized_vol_pct > 30:
        vol_context = "expanding"
    elif realized_vol_pct < 10:
        vol_context = "contracting"
    else:
        vol_context = "normal"

    # Score components (each 0-100)
    edge_score = min(100, max(0, (abs(prob_up - 0.5) - min_edge) / (0.5 - min_edge) * 100))

    # Regime score
    regime_score_map = {
        "trending_up": 80,
        "trending_down": 80,
        "mean_reverting": 60,
        "low_volatility": 40,
        "high_volatility": 20,
        "unknown": 0,
    }
    regime_score = regime_score_map.get(regime, 50)

    # Volatility context score
    vol_score = 70 if vol_context == "normal" else (40 if vol_context == "contracting" else 30)

    # Expected move adequacy
    move_score = min(100, (expected_move_pct / max(min_expected_move, 0.01)) * 50)

    # Combined score
    if no_trade_reason:
        quality_score = 0.0
    else:
        quality_score = (
            edge_score * 0.50
            + regime_score * 0.25
            + vol_score * 0.15
            + move_score * 0.10
        )

    if no_trade_reason and direction != "no_trade":
        direction = "no_trade"

    return ScoredSignal(
        direction=direction,
        probability=round(prob, 4),
        confidence=round(confidence, 4),
        confidence_band=(round(conf_low, 4), round(conf_high, 4)),
        expected_move_pct=round(expected_move_pct, 4),
        realized_vol_pct=round(realized_vol_pct, 2),
        volatility_context=vol_context,
        regime=regime,
        signal_quality_score=round(quality_score, 1),
        no_trade_reason=no_trade_reason,
        prob_up=round(prob_up, 4),
        prob_down=round(prob_down, 4),
        explanation=explanation,
        top_features=top_features,
    )
