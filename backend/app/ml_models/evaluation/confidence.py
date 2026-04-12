"""
Confidence-bucket analysis.

Separates predictive quality from trading utility by answering:
  "When the model is most confident, does it actually make money?"

Confidence is defined as |P(up) - 0.5| * 2 ∈ [0, 1].
Each sample is assigned to one of n_bins equal-width confidence buckets.

For each bucket we report:
  - Sample count and fraction of total
  - Accuracy (directional accuracy within bucket)
  - Mean confidence
  - Expected return (mean return of target bar when we act)
  - Return std (volatility of the signal)
  - Sharpe proxy (expected_return / return_std * sqrt(252 * 78))
  - Signal rate (how often does the model fall in this bucket)

Trading utility analysis distinguishes a model that is:
  (a) Accurate but untradeable: high accuracy only at low confidence → no edge
  (b) Tradeable: high accuracy specifically at high confidence → actionable
"""

from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np


@dataclass
class ConfidenceBucket:
    bucket_index: int
    confidence_min: float
    confidence_max: float

    n_samples: int
    signal_rate: float          # fraction of total test samples in this bucket
    mean_confidence: float

    # Predictive quality
    accuracy: float             # fraction correct (0.5 = coin flip)
    directional_accuracy: float # same as accuracy for binary labels

    # Trading utility (requires return_at_bar to be passed)
    expected_return: Optional[float]   # mean return when signal fires
    return_std: Optional[float]        # std of returns when signal fires
    sharpe_proxy: Optional[float]      # annualized Sharpe proxy

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConfidenceAnalysis:
    buckets: List[ConfidenceBucket]
    n_bins: int
    total_samples: int
    high_confidence_accuracy: Optional[float]   # accuracy in top confidence bin
    skill_monotone: bool   # True if accuracy generally increases with confidence

    def to_dict(self) -> dict:
        return {
            "buckets": [b.to_dict() for b in self.buckets],
            "n_bins": self.n_bins,
            "total_samples": self.total_samples,
            "high_confidence_accuracy": self.high_confidence_accuracy,
            "skill_monotone": self.skill_monotone,
        }


def confidence_bucket_analysis(
    y_true: np.ndarray,
    prob_up: np.ndarray,
    n_bins: int = 5,
    returns: Optional[np.ndarray] = None,
    annualize_factor: float = 252 * 78,
) -> ConfidenceAnalysis:
    """
    Analyse model performance broken down by prediction confidence.

    Parameters
    ----------
    y_true : ndarray (n,)   — binary labels {0, 1}
    prob_up : ndarray (n,)  — predicted P(up) ∈ [0, 1]
    n_bins : int            — number of confidence buckets
    returns : ndarray (n,) or None
        Actual one-bar returns for each prediction. When provided, trading
        utility metrics (expected_return, sharpe_proxy) are computed. The sign
        convention is: positive return = price went up.
    annualize_factor : float
        Bars per year used for Sharpe annualization. Default = 252 × 78 (5m bars).
    """
    confidence = np.abs(prob_up - 0.5) * 2  # [0, 1]
    y_pred = (prob_up >= 0.5).astype(int)
    n_total = len(y_true)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    buckets: List[ConfidenceBucket] = []

    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)

        n = int(mask.sum())
        if n == 0:
            buckets.append(ConfidenceBucket(
                bucket_index=i,
                confidence_min=round(lo, 4),
                confidence_max=round(hi, 4),
                n_samples=0,
                signal_rate=0.0,
                mean_confidence=float((lo + hi) / 2),
                accuracy=float("nan"),
                directional_accuracy=float("nan"),
                expected_return=None,
                return_std=None,
                sharpe_proxy=None,
            ))
            continue

        acc = float(np.mean(y_pred[mask] == y_true[mask]))
        mean_conf = float(np.mean(confidence[mask]))

        # Trading utility: signed return when we follow the signal
        expected_ret = None
        ret_std = None
        sharpe = None
        if returns is not None:
            # Signal: buy when P(up) > 0.5, sell when P(up) < 0.5
            signal_direction = np.where(prob_up[mask] >= 0.5, 1.0, -1.0)
            signed_returns = signal_direction * returns[mask]
            expected_ret = float(np.mean(signed_returns))
            ret_std = float(np.std(signed_returns))
            if ret_std > 0:
                sharpe = float(expected_ret / ret_std * np.sqrt(annualize_factor))
            else:
                sharpe = 0.0

        buckets.append(ConfidenceBucket(
            bucket_index=i,
            confidence_min=round(lo, 4),
            confidence_max=round(hi, 4),
            n_samples=n,
            signal_rate=round(n / n_total, 4),
            mean_confidence=round(mean_conf, 4),
            accuracy=round(acc, 4),
            directional_accuracy=round(acc, 4),
            expected_return=round(expected_ret, 6) if expected_ret is not None else None,
            return_std=round(ret_std, 6) if ret_std is not None else None,
            sharpe_proxy=round(sharpe, 4) if sharpe is not None else None,
        ))

    # High-confidence accuracy (top bin)
    top_bucket = buckets[-1] if buckets else None
    hc_acc = top_bucket.accuracy if top_bucket and top_bucket.n_samples > 0 else None

    # Skill monotonicity: do non-empty bins show increasing accuracy with confidence?
    valid_accs = [b.accuracy for b in buckets if b.n_samples > 0 and not np.isnan(b.accuracy)]
    if len(valid_accs) >= 2:
        # Kendall-tau-like: count concordant vs discordant adjacent pairs
        concordant = sum(1 for a, b in zip(valid_accs, valid_accs[1:]) if b >= a)
        monotone = concordant >= len(valid_accs) - 1 - concordant  # majority non-decreasing
    else:
        monotone = False

    return ConfidenceAnalysis(
        buckets=buckets,
        n_bins=n_bins,
        total_samples=n_total,
        high_confidence_accuracy=hc_acc,
        skill_monotone=monotone,
    )
