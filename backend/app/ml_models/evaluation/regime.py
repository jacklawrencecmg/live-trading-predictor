"""
Regime-segmented evaluation.

Evaluates model performance broken down by the market regime active at
prediction time. Regime is determined by app.regime.detector.detect_regime(),
which classifies each bar as one of:
  trending_up | trending_down | mean_reverting | high_volatility | low_volatility | unknown

This answers: "Is this model regime-dependent? Does it only work in
trending markets? Does it blow up in high-vol?"
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

from app.ml_models.evaluation.metrics import (
    _brier_score,
    _log_loss,
    _roc_auc,
    _balanced_accuracy,
)


@dataclass
class RegimeMetrics:
    regime: str
    n_samples: int
    sample_fraction: float

    brier_score: float
    log_loss: float
    roc_auc: Optional[float]
    accuracy: float
    balanced_accuracy: float

    # Trading utility at threshold
    directional_accuracy_confident: Optional[float]   # accuracy when confidence > threshold
    n_confident_samples: Optional[int]

    def to_dict(self) -> dict:
        return asdict(self)


def regime_segmented_evaluation(
    y_true: np.ndarray,
    prob_up: np.ndarray,
    regimes: np.ndarray,            # string label per sample
    confidence_threshold: float = 0.55,
    min_samples: int = 30,
) -> List[RegimeMetrics]:
    """
    Compute evaluation metrics separately for each market regime.

    Parameters
    ----------
    y_true : ndarray (n,)        — binary labels
    prob_up : ndarray (n,)       — predicted P(up)
    regimes : ndarray (n,)       — regime string per sample
    confidence_threshold : float — min confidence for the "confident" subset
    min_samples : int            — regimes with fewer samples are excluded

    Returns
    -------
    List of RegimeMetrics sorted by n_samples descending.
    """
    unique_regimes = np.unique(regimes)
    n_total = len(y_true)
    results = []

    for regime in unique_regimes:
        mask = regimes == regime
        if mask.sum() < min_samples:
            continue

        yt = y_true[mask]
        pp = prob_up[mask]
        y_pred = (pp >= 0.5).astype(int)
        n = int(mask.sum())

        accuracy = float(np.mean(y_pred == yt))
        bal_acc = _balanced_accuracy(yt, y_pred)

        # Confident-sample directional accuracy
        confidence = np.abs(pp - 0.5) * 2
        conf_mask = confidence >= (confidence_threshold - 0.5) * 2
        if conf_mask.sum() >= 10:
            dca = float(np.mean(y_pred[conf_mask] == yt[conf_mask]))
            n_conf = int(conf_mask.sum())
        else:
            dca = None
            n_conf = None

        results.append(RegimeMetrics(
            regime=str(regime),
            n_samples=n,
            sample_fraction=round(n / n_total, 4),
            brier_score=round(_brier_score(yt, pp), 6),
            log_loss=round(_log_loss(yt, pp), 6),
            roc_auc=_roc_auc(yt, pp),
            accuracy=round(accuracy, 4),
            balanced_accuracy=round(bal_acc, 4),
            directional_accuracy_confident=round(dca, 4) if dca is not None else None,
            n_confident_samples=n_conf,
        ))

    results.sort(key=lambda r: -r.n_samples)
    return results
