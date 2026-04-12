"""
Uncertainty quantification — 4-layer decision model.

The 4 layers are distinct and must be treated separately:

  Layer 1 — raw_prob_up
    Direct output of model.predict_proba()[:, 1]. Uncorrected, often
    overconfident for tree models. Cannot be trusted as a probability.

  Layer 2 — calibrated_prob_up
    Raw probability after applying the fitted calibration map (isotonic or
    Platt scaling). Closer to the true P(event=up) as measured on held-out
    data. If no calibration map is available, equals raw_prob_up with a flag.

  Layer 3 — tradeable_confidence
    Calibrated probability shrunk toward 0.5 by a degradation factor derived
    from recent rolling Brier score. When the model has been performing worse
    than its training baseline, tradeable_confidence < calibrated_prob. This
    layer answers: "given recent performance, how much should I trust this?"

    tradeable_confidence = 0.5 + (calibrated_prob - 0.5) * degradation_factor

  Layer 4 — action
    The final recommendation: "buy", "sell", or "abstain".
    Abstain when:
      - tradeable_confidence < confidence_threshold
      - degradation_factor < hard_abstain_degradation (severe deterioration)
      - regime is suppressed (existing logic, propagated here)

  Supporting fields:
    confidence_band: (low, high) calibration uncertainty range.
      Computed as calibrated_prob ± ece_recent.

      IMPORTANT — this is NOT a statistical confidence interval.
      ECE (Expected Calibration Error) is a scalar average miscalibration
      measure across ALL predictions; it has no per-prediction probabilistic
      interpretation. The band gives a rough sense of typical model mis-
      calibration magnitude. A narrow band (low ECE) means the model is
      generally well-calibrated on average; it does NOT mean this specific
      prediction is accurate to within ±ECE.

    calibration_health: "good" | "fair" | "degraded" | "unknown"
      good:      rolling_brier ≤ 1.1 * baseline_brier AND ece ≤ 0.05
      fair:      rolling_brier ≤ 1.3 * baseline_brier OR ece ≤ 0.10
      degraded:  rolling_brier > 1.3 * baseline_brier
      unknown:   insufficient history
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Calibration map — stored alongside a model artifact, applied at inference
# ---------------------------------------------------------------------------

@dataclass
class CalibrationMap:
    """
    Piecewise-linear mapping from raw predicted probability to calibrated
    probability. Serialised as two parallel lists (can be stored as JSON).

    For isotonic regression: x = mean_predicted, y = fraction_positive per bin.
    For Platt scaling: use a_coef / b_coef for the logistic transform.
    """
    kind: str               # "isotonic" | "platt" | "identity"
    x_raw: List[float]      # sorted raw probability breakpoints
    y_cal: List[float]      # corresponding calibrated probabilities
    # Platt parameters (used when kind="platt")
    platt_a: Optional[float] = None
    platt_b: Optional[float] = None
    n_calibration_samples: int = 0
    ece_at_fit: Optional[float] = None

    def apply(self, raw_prob: float) -> float:
        """Apply calibration map to a single raw probability."""
        if self.kind == "identity":
            return raw_prob
        if self.kind == "platt" and self.platt_a is not None and self.platt_b is not None:
            import math
            return 1.0 / (1.0 + math.exp(-(self.platt_a * raw_prob + self.platt_b)))
        if self.kind == "isotonic" and len(self.x_raw) >= 2:
            # Linear interpolation between stored breakpoints
            raw_prob = max(self.x_raw[0], min(self.x_raw[-1], raw_prob))
            for i in range(len(self.x_raw) - 1):
                if self.x_raw[i] <= raw_prob <= self.x_raw[i + 1]:
                    t = (raw_prob - self.x_raw[i]) / (self.x_raw[i + 1] - self.x_raw[i] + 1e-10)
                    return self.y_cal[i] + t * (self.y_cal[i + 1] - self.y_cal[i])
        return raw_prob  # fallback: identity

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def identity(cls) -> "CalibrationMap":
        return cls(kind="identity", x_raw=[], y_cal=[], n_calibration_samples=0)

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationMap":
        return cls(**d)


# ---------------------------------------------------------------------------
# Uncertainty bundle — full output of the 4-layer model
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyBundle:
    # --- Layer 1: raw model output ---
    raw_prob_up: float
    raw_prob_down: float

    # --- Layer 2: calibrated probability ---
    calibrated_prob_up: float
    calibrated_prob_down: float
    calibration_available: bool     # False → calibrated = raw, no map loaded

    # --- Layer 3: tradeable confidence ---
    tradeable_confidence: float     # in [0, 1]; threshold determines action
    degradation_factor: float       # 1.0 = nominal; <1.0 = recently underperforming
    rolling_brier: Optional[float]  # None when window too small
    baseline_brier: Optional[float]
    ece_recent: Optional[float]

    # --- Layer 4: action ---
    action: str                     # "buy" | "sell" | "abstain"
    abstain_reason: Optional[str]

    # --- Calibration uncertainty range ---
    # calibrated_prob ± ece_recent.  NOT a statistical confidence interval;
    # ECE is an average miscalibration scalar, not a per-prediction CI.
    # Renamed from "confidence_band" to make the semantics explicit.
    confidence_band_low: float      # calibrated_prob - ece_recent (floor 0.0)
    confidence_band_high: float     # calibrated_prob + ece_recent (cap 1.0)

    # --- Calibration health ---
    calibration_health: str         # "good" | "fair" | "degraded" | "unknown"

    # --- Reliability diagram (sparse, serialisable) ---
    reliability_bins: Optional[List[float]]
    reliability_mean_pred: Optional[List[float]]
    reliability_frac_pos: Optional[List[float]]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _calibration_health(
    rolling_brier: Optional[float],
    baseline_brier: Optional[float],
    ece_recent: Optional[float],
) -> str:
    if rolling_brier is None or baseline_brier is None:
        return "unknown"
    ratio = rolling_brier / (baseline_brier + 1e-9)
    ece = ece_recent or 0.0
    if ratio <= 1.1 and ece <= 0.05:
        return "good"
    if ratio <= 1.3 or ece <= 0.10:
        return "fair"
    return "degraded"


def build_uncertainty_bundle(
    raw_prob_up: float,
    calibration_map: CalibrationMap,
    tracker_stats,          # TrackerStats from confidence_tracker, or None
    confidence_threshold: float = 0.55,
    hard_abstain_degradation: float = 0.30,
    prior_abstain_reason: Optional[str] = None,
) -> UncertaintyBundle:
    """
    Build the full 4-layer UncertaintyBundle.

    Parameters
    ----------
    raw_prob_up : float
        Direct model.predict_proba()[:, 1] output.
    calibration_map : CalibrationMap
        Loaded from the model artifact. Use CalibrationMap.identity() when none available.
    tracker_stats : TrackerStats or None
        Rolling performance statistics from ConfidenceTracker. None → no degradation.
    confidence_threshold : float
        Minimum tradeable_confidence to generate a buy/sell action.
    hard_abstain_degradation : float
        If degradation_factor falls below this level, always abstain regardless of
        tradeable_confidence. Prevents trading when the model is badly off.
    prior_abstain_reason : str or None
        If set (regime suppressed, bar not closed, etc.), the action is forced to abstain.
    """
    raw_prob_down = 1.0 - raw_prob_up
    raw_prob_up = float(max(0.0, min(1.0, raw_prob_up)))
    raw_prob_down = float(max(0.0, min(1.0, raw_prob_down)))

    # --- Layer 2: calibration ---
    calibration_available = calibration_map.kind != "identity"
    cal_up = float(calibration_map.apply(raw_prob_up))
    cal_up = max(0.0, min(1.0, cal_up))
    cal_down = 1.0 - cal_up

    # --- Layer 3: degradation ---
    if tracker_stats is not None:
        deg = float(getattr(tracker_stats, "degradation_factor", 1.0))
        rolling_brier = getattr(tracker_stats, "rolling_brier", None)
        baseline_brier = getattr(tracker_stats, "baseline_brier", None)
        ece_recent = getattr(tracker_stats, "ece_recent", None)
    else:
        deg = 1.0
        rolling_brier = None
        baseline_brier = None
        ece_recent = None

    deg = max(0.0, min(1.0, deg))
    # Tradeable confidence: shrink calibrated signal toward 0.5
    tradeable = 0.5 + (cal_up - 0.5) * deg
    tradeable = max(0.0, min(1.0, tradeable))
    tradeable_conf = abs(tradeable - 0.5) * 2   # [0, 1]

    # --- Confidence band (ECE-driven) ---
    ece_width = ece_recent if ece_recent is not None else 0.05
    band_low = max(0.0, cal_up - ece_width)
    band_high = min(1.0, cal_up + ece_width)

    # --- Calibration health ---
    health = _calibration_health(rolling_brier, baseline_brier, ece_recent)

    # --- Layer 4: action ---
    abstain_reason = prior_abstain_reason
    action = "abstain"

    if abstain_reason is not None:
        pass  # already abstaining, keep reason
    elif deg < hard_abstain_degradation:
        abstain_reason = f"degraded_performance:factor={deg:.2f}"
    elif tradeable_conf < confidence_threshold:
        abstain_reason = f"low_tradeable_confidence:{tradeable_conf:.2f}"
    else:
        # Determine direction from calibrated probability
        if tradeable > 0.5:
            action = "buy"
        else:
            action = "sell"
        abstain_reason = None

    return UncertaintyBundle(
        raw_prob_up=round(raw_prob_up, 4),
        raw_prob_down=round(raw_prob_down, 4),
        calibrated_prob_up=round(cal_up, 4),
        calibrated_prob_down=round(cal_down, 4),
        calibration_available=calibration_available,
        tradeable_confidence=round(tradeable_conf, 4),
        degradation_factor=round(deg, 4),
        rolling_brier=round(rolling_brier, 6) if rolling_brier is not None else None,
        baseline_brier=round(baseline_brier, 6) if baseline_brier is not None else None,
        ece_recent=round(ece_recent, 6) if ece_recent is not None else None,
        action=action,
        abstain_reason=abstain_reason,
        confidence_band_low=round(band_low, 4),
        confidence_band_high=round(band_high, 4),
        calibration_health=health,
        reliability_bins=getattr(tracker_stats, "reliability_bins", None),
        reliability_mean_pred=getattr(tracker_stats, "reliability_mean_pred", None),
        reliability_frac_pos=getattr(tracker_stats, "reliability_frac_pos", None),
    )
