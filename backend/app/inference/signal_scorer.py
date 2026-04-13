"""
Signal scoring layer.

Combines the 4-layer uncertainty bundle with market context to produce a
single actionable ScoredSignal. Uses calibrated_prob (not raw_prob) for the
quality score, and ECE-driven confidence band (not hardcoded ±0.05).

Signal quality score (0-100) combines:
- Probability edge (how far calibrated prob is from 50/50)
- Regime suitability
- Volatility context
- Expected move adequacy
- Calibration health penalty (degrades score when model is miscalibrated)

The distinction between the four probability layers is preserved in the output:
  probability       = calibrated_prob (best estimate)
  raw_probability   = raw model output (for transparency)
  tradeable_confidence = signal layer 3 (threshold-adjusted)
  confidence        = tradeable_confidence (backward compat alias)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ScoredSignal:
    direction: str                  # "up" | "down" | "no_trade"

    # Probability layers (explicit distinction)
    raw_probability: float          # Layer 1: raw model output
    probability: float              # Layer 2: calibrated probability
    tradeable_confidence: float     # Layer 3: calibrated × degradation
    confidence: float               # = tradeable_confidence (backward compat)

    # Confidence interval
    confidence_band: tuple          # (low, high) — ECE-driven width

    # Uncertainty context
    degradation_factor: float
    calibration_health: str         # "good" | "fair" | "degraded" | "unknown"
    calibration_available: bool
    ece_recent: Optional[float]
    rolling_brier: Optional[float]
    abstain_reason: Optional[str]   # None when action != "abstain"

    # Market context
    expected_move_pct: float
    realized_vol_pct: float
    volatility_context: str
    regime: str

    # Composite score
    signal_quality_score: float     # 0-100

    # Audit
    no_trade_reason: Optional[str]
    prob_up: float                  # calibrated (consistent with 'probability')
    prob_down: float
    explanation: str
    top_features: dict


def score_signal(
    # Layer 1
    raw_prob_up: float,
    raw_prob_down: float,
    # Layer 2
    calibrated_prob_up: float,
    calibrated_prob_down: float,
    calibration_available: bool,
    # Layer 3
    tradeable_confidence: float,
    degradation_factor: float,
    # Layer 4 context
    abstain_reason: Optional[str],
    calibration_health: str,
    ece_recent: Optional[float],
    rolling_brier: Optional[float],
    confidence_band: tuple,
    # Market context
    expected_move_pct: float,
    realized_vol_pct: float,
    regime: str,
    no_trade_reason: Optional[str],
    explanation: str,
    top_features: dict,
    min_edge: float = 0.05,
    min_expected_move: float = 0.10,
) -> ScoredSignal:

    # Direction from calibrated probability (not raw, not tradeable)
    direction = "no_trade"
    if abstain_reason is None:
        if calibrated_prob_up > 0.5 + min_edge:
            direction = "up"
        elif calibrated_prob_down > 0.5 + min_edge:
            direction = "down"

    # Volatility context
    if realized_vol_pct > 30:
        vol_context = "expanding"
    elif realized_vol_pct < 10:
        vol_context = "contracting"
    else:
        vol_context = "normal"

    # Score components (each 0–100)
    edge = abs(calibrated_prob_up - 0.5)
    edge_score = min(100, max(0, (edge - min_edge) / (0.5 - min_edge) * 100))

    regime_score_map = {
        "trending_up": 80, "trending_down": 80, "mean_reverting": 60,
        "low_volatility": 40, "high_volatility": 20, "unknown": 0,
    }
    regime_score = regime_score_map.get(regime, 50)

    vol_score = 70 if vol_context == "normal" else (40 if vol_context == "contracting" else 30)
    move_score = min(100, (expected_move_pct / max(min_expected_move, 0.01)) * 50)

    # Calibration health penalty (applied to raw quality score)
    cal_penalty_map = {
        "good":     1.00,
        "fair":     0.85,
        "caution":  0.70,
        "degraded": 0.60,
        "unknown":  0.75,
    }
    cal_penalty = cal_penalty_map.get(calibration_health, 0.75)

    # Combined quality score
    if abstain_reason or direction == "no_trade":
        quality_score = 0.0
    else:
        raw_score = (
            edge_score * 0.50
            + regime_score * 0.25
            + vol_score * 0.15
            + move_score * 0.10
        )
        quality_score = round(raw_score * cal_penalty, 1)

    return ScoredSignal(
        direction=direction,
        raw_probability=round(raw_prob_up, 4),
        probability=round(calibrated_prob_up, 4),
        tradeable_confidence=round(tradeable_confidence, 4),
        confidence=round(tradeable_confidence, 4),
        confidence_band=confidence_band,
        degradation_factor=round(degradation_factor, 4),
        calibration_health=calibration_health,
        calibration_available=calibration_available,
        ece_recent=ece_recent,
        rolling_brier=rolling_brier,
        abstain_reason=abstain_reason,
        expected_move_pct=round(expected_move_pct, 4),
        realized_vol_pct=round(realized_vol_pct, 2),
        volatility_context=vol_context,
        regime=regime,
        signal_quality_score=quality_score,
        no_trade_reason=no_trade_reason,
        prob_up=round(calibrated_prob_up, 4),
        prob_down=round(calibrated_prob_down, 4),
        explanation=explanation,
        top_features=top_features,
    )
