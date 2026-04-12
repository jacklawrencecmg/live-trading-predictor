"""
Probability calibration.

Raw classifiers often output poorly-calibrated probabilities:
  - Logistic regression can be well-calibrated when features are independent,
    but tends to be overconfident with correlated features.
  - GBT and RF are typically overconfident (probabilities cluster near 0 and 1).

Calibration wraps an already-fitted model with CalibratedClassifierCV to
adjust raw probabilities toward observed frequencies.

Calibration itself is evaluated via ECE (Expected Calibration Error) and
reliability diagrams. A model with low ECE produces probabilities you can
trust: "60% confidence" means ~60% of such predictions should be correct.

IMPORTANT: calibration must be fit on held-out data, not the training set.
We use a portion of each fold's training data for calibration fitting,
or rely on sklearn's CalibratedClassifierCV with cv="prefit".
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Only import sklearn when needed to keep test collection fast
def _calibrate(model, X_calib: np.ndarray, y_calib: np.ndarray, method: str = "isotonic"):
    """
    Wrap a fitted model with isotonic or Platt calibration.

    Fits calibration on (X_calib, y_calib) which must NOT overlap with the
    training data used to fit `model`.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_calib, y_calib : held-out calibration split
    method : "isotonic" (non-parametric, needs ≥ ~1000 samples) or
             "sigmoid" (Platt scaling, works on smaller sets)
    """
    from sklearn.calibration import CalibratedClassifierCV
    calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrated.fit(X_calib, y_calib)
    return calibrated


def calibrate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "isotonic",
    calib_fraction: float = 0.2,
    random_seed: int = 42,
):
    """
    Split training data, fit model on train portion, calibrate on held-out portion.

    Uses a temporal split (last calib_fraction of training data) to preserve
    the time-series property — no shuffling.

    Returns (calibrated_model, calib_split_size).
    """
    n = len(X_train)
    split = int(n * (1 - calib_fraction))
    if split < 50 or (n - split) < 30:
        # Not enough data for split calibration — return uncalibrated
        logger.warning(
            "Insufficient data for calibration split (n=%d). Skipping calibration.", n
        )
        return model, 0

    X_tr, y_tr = X_train[:split], y_train[:split]
    X_cal, y_cal = X_train[split:], y_train[split:]

    from sklearn.base import clone
    try:
        base = clone(model)
        base.fit(X_tr, y_tr)
        calibrated = _calibrate(base, X_cal, y_cal, method=method)
        return calibrated, len(X_cal)
    except Exception as e:
        logger.warning("Calibration failed (%s). Falling back to uncalibrated model.", e)
        # Fall back: fit on full training data without calibration
        try:
            fallback = clone(model)
            fallback.fit(X_train, y_train)
            return fallback, 0
        except Exception:
            return model, 0


def calibration_summary(
    y_true: np.ndarray,
    prob_up: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration curve and ECE for reporting.

    Returns dict with:
      mean_predicted_prob: list of n_bins bin centres
      fraction_positive:   list of n_bins observed positive fractions
      ece:                 Expected Calibration Error
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_pred = []
    frac_pos = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (prob_up >= lo) & (prob_up < hi) if hi < 1.0 else (prob_up >= lo) & (prob_up <= hi)
        if not mask.any():
            continue
        mean_pred.append(float(np.mean(prob_up[mask])))
        frac_pos.append(float(np.mean(y_true[mask])))

    n = len(y_true)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (prob_up >= lo) & (prob_up < hi) if hi < 1.0 else (prob_up >= lo) & (prob_up <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(float(np.mean(y_true[mask])) - float(np.mean(prob_up[mask])))

    return {
        "mean_predicted_prob": mean_pred,
        "fraction_positive": frac_pos,
        "ece": round(ece, 6),
        "n_bins": n_bins,
    }
