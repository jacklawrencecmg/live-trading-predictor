"""
Model evaluation metrics.

Primary selection criterion: Brier score (proper scoring rule for probabilities).
Secondary: log-loss.
Tertiary: trading utility metrics (directional accuracy at high confidence).

Do NOT select models by accuracy alone — class imbalance makes it unreliable.
"""

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FoldMetrics:
    fold: int
    train_size: int
    test_size: int

    # Probabilistic quality (primary selection criteria)
    brier_score: float       # proper scoring rule; lower is better; 0.25 = coin-flip
    log_loss: float          # lower is better
    ece: float               # Expected Calibration Error; lower is better

    # Discriminative quality
    roc_auc: Optional[float]
    accuracy: float
    balanced_accuracy: float

    # Per-class (class 1 = up)
    precision_up: float
    recall_up: float
    f1_up: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AggregatedMetrics:
    """Mean ± std across all folds."""
    brier_score_mean: float
    brier_score_std: float
    log_loss_mean: float
    log_loss_std: float
    ece_mean: float
    ece_std: float
    roc_auc_mean: Optional[float]
    roc_auc_std: Optional[float]
    accuracy_mean: float
    accuracy_std: float
    balanced_accuracy_mean: float
    balanced_accuracy_std: float
    n_folds: int
    total_test_samples: int

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_str(self) -> str:
        auc_str = f"{self.roc_auc_mean:.3f}±{self.roc_auc_std:.3f}" if self.roc_auc_mean else "N/A"
        return (
            f"Brier={self.brier_score_mean:.4f}±{self.brier_score_std:.4f}  "
            f"LogLoss={self.log_loss_mean:.4f}±{self.log_loss_std:.4f}  "
            f"ECE={self.ece_mean:.4f}  AUC={auc_str}  "
            f"BalAcc={self.balanced_accuracy_mean:.3f}±{self.balanced_accuracy_std:.3f}"
        )


# ---------------------------------------------------------------------------
# Per-metric computation helpers
# ---------------------------------------------------------------------------

def _brier_score(y_true: np.ndarray, prob_up: np.ndarray) -> float:
    return float(np.mean((y_true - prob_up) ** 2))


def _log_loss(y_true: np.ndarray, prob_up: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(prob_up, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def _roc_auc(y_true: np.ndarray, prob_up: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    # Mann-Whitney U implementation (no sklearn dependency at metric level)
    pos = prob_up[y_true == 1]
    neg = prob_up[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # Count concordant pairs
    concordant = float(np.sum(pos[:, None] > neg[None, :]))
    tied = float(np.sum(pos[:, None] == neg[None, :]))
    total = float(len(pos) * len(neg))
    return (concordant + 0.5 * tied) / total


def _ece(
    y_true: np.ndarray,
    prob_up: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error.

    Bins predictions by predicted probability; computes weighted mean absolute
    difference between predicted probability and observed frequency.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob_up >= lo) & (prob_up < hi)
        if not mask.any():
            continue
        frac = float(np.mean(y_true[mask]))
        conf = float(np.mean(prob_up[mask]))
        ece += (mask.sum() / n) * abs(frac - conf)
    return float(ece)


def _precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float, float]:
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return float(precision), float(recall), float(f1)


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            continue
        recalls.append(float(np.mean(y_pred[mask] == c)))
    return float(np.mean(recalls)) if recalls else 0.5


# ---------------------------------------------------------------------------
# Main computation function
# ---------------------------------------------------------------------------

def compute_fold_metrics(
    fold: int,
    train_size: int,
    y_true: np.ndarray,
    prob_up: np.ndarray,
) -> FoldMetrics:
    """
    Compute all metrics for one fold's test predictions.

    Parameters
    ----------
    fold : int
    train_size : int
    y_true : ndarray of shape (n,)  — binary labels {0, 1}
    prob_up : ndarray of shape (n,) — predicted P(up)
    """
    y_pred = (prob_up >= 0.5).astype(int)
    precision, recall, f1 = _precision_recall_f1(y_true, y_pred)
    accuracy = float(np.mean(y_pred == y_true))
    bal_acc = _balanced_accuracy(y_true, y_pred)

    return FoldMetrics(
        fold=fold,
        train_size=train_size,
        test_size=len(y_true),
        brier_score=_brier_score(y_true, prob_up),
        log_loss=_log_loss(y_true, prob_up),
        ece=_ece(y_true, prob_up),
        roc_auc=_roc_auc(y_true, prob_up),
        accuracy=accuracy,
        balanced_accuracy=bal_acc,
        precision_up=precision,
        recall_up=recall,
        f1_up=f1,
    )


def aggregate_fold_metrics(folds: List[FoldMetrics]) -> AggregatedMetrics:
    """Aggregate a list of per-fold metrics into mean ± std."""
    if not folds:
        raise ValueError("No folds to aggregate")

    def _m(attr: str) -> Tuple[float, float]:
        vals = [getattr(f, attr) for f in folds]
        vals = [v for v in vals if v is not None]
        if not vals:
            return (None, None)
        return float(np.mean(vals)), float(np.std(vals))

    bs_m, bs_s = _m("brier_score")
    ll_m, ll_s = _m("log_loss")
    ec_m, ec_s = _m("ece")
    au_m, au_s = _m("roc_auc")
    ac_m, ac_s = _m("accuracy")
    ba_m, ba_s = _m("balanced_accuracy")

    return AggregatedMetrics(
        brier_score_mean=bs_m, brier_score_std=bs_s,
        log_loss_mean=ll_m, log_loss_std=ll_s,
        ece_mean=ec_m, ece_std=ec_s,
        roc_auc_mean=au_m, roc_auc_std=au_s,
        accuracy_mean=ac_m, accuracy_std=ac_s,
        balanced_accuracy_mean=ba_m, balanced_accuracy_std=ba_s,
        n_folds=len(folds),
        total_test_samples=sum(f.test_size for f in folds),
    )
