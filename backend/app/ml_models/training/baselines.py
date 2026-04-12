"""
Naive baselines — sklearn-compatible estimators.

All baselines implement the sklearn estimator interface:
  fit(X, y) → self
  predict_proba(X) → ndarray of shape (n, 2)  [P(down), P(up)]
  predict(X) → ndarray of shape (n,)

Baselines serve two purposes:
1. Sanity floor: any trained model that cannot beat them is worthless.
2. Calibration reference: baselines that achieve low Brier score by exploiting
   class imbalance reveal how much of the model's score is free information.

Baselines included:
  PriorBaseline       — always predicts the training-set class proportion
  MomentumBaseline    — follow the sign of the last bar's return (ret_1)
  AntiMomentumBaseline — oppose the sign (contrarian)
"""

import numpy as np
from typing import Optional


class PriorBaseline:
    """
    Predict the training-set class proportion for every sample.

    This is the "always predict base rate" baseline. A model with
    Brier score worse than this adds no value over raw class frequencies.
    """

    def __init__(self):
        self.prior_up: float = 0.5
        self.classes_: np.ndarray = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PriorBaseline":
        n = len(y)
        if n == 0:
            self.prior_up = 0.5
        else:
            self.prior_up = float(np.sum(y == 1) / n)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        p = self.prior_up
        return np.column_stack([np.full(n, 1 - p), np.full(n, p)])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def __repr__(self) -> str:
        return f"PriorBaseline(prior_up={self.prior_up:.3f})"


class MomentumBaseline:
    """
    Follow the sign of the prior bar's return.

    Confidence is proportional to the magnitude of ret_1, capped at [0.51, 0.80].
    Uses the ret_1 feature column, identified by name or by a fixed index.

    Parameters
    ----------
    ret1_feature_idx : int
        Column index of ret_1 in the feature matrix. Defaults to 8, which is
        the position in FEATURE_COLS (from app.feature_pipeline.registry).
    confidence_cap : float
        Maximum P(up) assigned (symmetric around 0.5).
    """

    def __init__(
        self,
        ret1_feature_idx: int = 8,
        confidence_cap: float = 0.70,
    ):
        self.ret1_feature_idx = ret1_feature_idx
        self.confidence_cap = confidence_cap
        self._scale: float = 1.0
        self.classes_: np.ndarray = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MomentumBaseline":
        # Scale by the 90th percentile of |ret_1| so confidence spans [0.5, cap]
        if X.shape[1] > self.ret1_feature_idx:
            ret1 = X[:, self.ret1_feature_idx]
            p90 = float(np.nanpercentile(np.abs(ret1), 90))
            self._scale = p90 if p90 > 0 else 1.0
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        if X.shape[1] <= self.ret1_feature_idx:
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])

        ret1 = X[:, self.ret1_feature_idx]
        half_range = self.confidence_cap - 0.5
        # Map |ret1| / scale into [0, half_range], then add/subtract from 0.5
        magnitude = np.clip(np.abs(ret1) / (self._scale + 1e-9), 0, 1) * half_range
        prob_up = np.where(ret1 >= 0, 0.5 + magnitude, 0.5 - magnitude)
        prob_up = np.clip(prob_up, 0.5 - half_range, 0.5 + half_range)
        return np.column_stack([1 - prob_up, prob_up])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def __repr__(self) -> str:
        return f"MomentumBaseline(ret1_idx={self.ret1_feature_idx}, cap={self.confidence_cap})"


class AntiMomentumBaseline:
    """
    Oppose the sign of the prior bar's return (contrarian / mean-reversion baseline).

    This is the mirror of MomentumBaseline. If the market is trend-following,
    Momentum beats Anti-Momentum. If it mean-reverts, the opposite holds.
    Comparing both reveals the prevailing regime in-sample.
    """

    def __init__(
        self,
        ret1_feature_idx: int = 8,
        confidence_cap: float = 0.70,
    ):
        self._momentum = MomentumBaseline(ret1_feature_idx, confidence_cap)
        self.classes_: np.ndarray = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AntiMomentumBaseline":
        self._momentum.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = self._momentum.predict_proba(X)
        # Flip P(up) and P(down)
        return probs[:, ::-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def __repr__(self) -> str:
        return f"AntiMomentumBaseline(ret1_idx={self._momentum.ret1_feature_idx})"


# ---------------------------------------------------------------------------
# Registry of all baseline constructors by name
# ---------------------------------------------------------------------------

BASELINE_REGISTRY = {
    "prior": PriorBaseline,
    "momentum": MomentumBaseline,
    "anti_momentum": AntiMomentumBaseline,
}
