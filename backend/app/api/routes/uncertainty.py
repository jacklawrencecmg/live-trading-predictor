"""
Uncertainty API — dedicated endpoints for calibration health and tracker state.

GET /api/uncertainty/{symbol}
    Full uncertainty context: reliability diagram, confidence buckets,
    calibration health, degradation state.

GET /api/uncertainty/{symbol}/record
    POST endpoint to record an actual outcome for a past prediction.
    Called after bar close to update the rolling tracker.
"""

from fastapi import APIRouter, Body, Query
from typing import Optional
from app.inference.confidence_tracker import get_tracker

router = APIRouter()


@router.get("/{symbol}")
async def get_uncertainty(symbol: str):
    """
    Return the full uncertainty context for a symbol.

    Includes reliability diagram data (for rendering in the UI),
    rolling Brier score, ECE, degradation factor, and calibration health.
    """
    symbol = symbol.upper()
    tracker = get_tracker()
    stats = tracker.get_stats(symbol)

    return {
        "symbol": symbol,
        "window_size": stats.window_size,
        "calibration_health": stats.calibration_health,
        "degradation_factor": stats.degradation_factor,
        "rolling_brier": stats.rolling_brier,
        "baseline_brier": stats.baseline_brier,
        "ece_recent": stats.ece_recent,
        "reliability_diagram": (
            {
                "bins": stats.reliability_bins,
                "mean_predicted": stats.reliability_mean_pred,
                "fraction_positive": stats.reliability_frac_pos,
            }
            if stats.reliability_bins is not None else None
        ),
    }


@router.post("/{symbol}/record")
async def record_outcome(
    symbol: str,
    calibrated_prob: float = Body(...),
    actual_outcome: int = Body(...),       # 1 = price went up, 0 = went down
    baseline_brier: Optional[float] = Body(None),
):
    """
    Record a prediction outcome to update the rolling calibration tracker.

    Call this endpoint once per bar after the close is confirmed
    (bar_open_time + bar_duration has elapsed and the actual direction is known).
    """
    symbol = symbol.upper()
    if actual_outcome not in (0, 1):
        return {"error": "actual_outcome must be 0 or 1"}
    if not 0.0 <= calibrated_prob <= 1.0:
        return {"error": "calibrated_prob must be in [0, 1]"}

    tracker = get_tracker()
    tracker.record(
        symbol=symbol,
        calibrated_prob=calibrated_prob,
        actual_outcome=actual_outcome,
        baseline_brier=baseline_brier,
    )

    stats = tracker.get_stats(symbol)
    return {
        "symbol": symbol,
        "recorded": True,
        "window_size": stats.window_size,
        "calibration_health": stats.calibration_health,
        "degradation_factor": stats.degradation_factor,
        "rolling_brier": stats.rolling_brier,
    }
