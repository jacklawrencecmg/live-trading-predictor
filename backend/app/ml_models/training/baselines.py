"""
Naive baselines — sklearn-compatible estimators.

All baselines implement the sklearn estimator interface:
  fit(X, y) → self
  predict_proba(X) → ndarray of shape (n, n_classes)
  predict(X) → ndarray of shape (n,)

Baselines serve two purposes:
1. Sanity floor: any trained model that cannot beat them is worthless.
2. Calibration reference: baselines that achieve low Brier score by exploiting
   class imbalance reveal how much of the model's score is free information.

Baselines included:
  PriorBaseline            — always predicts the training-set class proportion
  MomentumBaseline         — follow the sign of the last bar's return (ret_1)
  AntiMomentumBaseline     — oppose the sign (contrarian)
  PersistenceBaseline      — predict maximum uncertainty / no-trade outcome
  VolatilityNoTradeBaseline — abstain on high-vol bars, use priors on low-vol

Binary vs ternary label support
---------------------------------
PersistenceBaseline and VolatilityNoTradeBaseline automatically detect whether
labels are binary {0, 1} or ternary {0=DOWN, 1=FLAT, 2=UP} from the y array
passed to fit(). They output (n, 2) or (n, 3) proba accordingly.

The older baselines (Prior, Momentum, AntiMomentum) remain binary-only as they
have existing tests that verify their (n, 2) output contract.
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


class PersistenceBaseline:
    """
    Predict the most uncertain / "no-trade" outcome for every sample.

    Binary (y ∈ {0,1}): always predict 50/50 — maximum entropy, no directional view.
    Ternary (y ∈ {0,1,2}): always predict P(FLAT)=1.0 — the market does nothing.

    This baseline sets the floor for ternary direction models: any model that cannot
    beat PersistenceBaseline on Brier score has no usable directional signal.

    n_classes_ is inferred from the training labels in fit().
    """

    def __init__(self):
        self.n_classes_: int = 2
        self.classes_: np.ndarray = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PersistenceBaseline":
        unique = np.unique(y[~np.isnan(y.astype(float))]).astype(int)
        self.n_classes_ = len(unique)
        self.classes_ = unique
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        if self.n_classes_ == 3:
            # All mass on FLAT (class index 1 in {DOWN=0, FLAT=1, UP=2})
            proba = np.zeros((n, 3))
            proba[:, 1] = 1.0
            return proba
        # Binary: uniform uncertainty
        return np.full((n, 2), 0.5)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def __repr__(self) -> str:
        return f"PersistenceBaseline(n_classes={self.n_classes_})"


class VolatilityNoTradeBaseline:
    """
    Abstain (predict flat/neutral) on high-volatility bars; use class priors otherwise.

    Intuition: when recent realized volatility is compressed the market may be in a
    predictable regime; when it expands we should stay out.  This tests whether
    volatility-conditioned abstention adds value over a plain prior.

    The volatility feature is realized_vol_5 (feature index 7 in FEATURE_COLS).
    The "high-vol" threshold is the vol_threshold_pct-ile of the training distribution.

    Works for both binary and ternary label spaces:
    - Binary:  high-vol → [0.5, 0.5];     low-vol → [1−p, p] class priors
    - Ternary: high-vol → [0, 1, 0] FLAT; low-vol → class priors [p_down, p_flat, p_up]

    n_classes_ is inferred from the training labels in fit().
    """

    # Matches FEATURE_COLS index of realized_vol_5 in app/feature_pipeline/registry.py
    VOL_FEATURE_IDX: int = 7

    def __init__(
        self,
        vol_feature_idx: int = 7,
        vol_threshold_pct: float = 80.0,
    ):
        self.vol_feature_idx = vol_feature_idx
        self.vol_threshold_pct = vol_threshold_pct
        self._threshold: float = 0.0
        self._priors: np.ndarray = np.array([0.5, 0.5])
        self.n_classes_: int = 2
        self.classes_: np.ndarray = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VolatilityNoTradeBaseline":
        unique = np.unique(y[~np.isnan(y.astype(float))]).astype(int)
        self.n_classes_ = len(unique)
        self.classes_ = unique

        # Volatility threshold learned from training data
        if X.shape[1] > self.vol_feature_idx:
            vol = X[:, self.vol_feature_idx]
            self._threshold = float(np.nanpercentile(vol, self.vol_threshold_pct))

        # Class priors from training labels
        counts = np.bincount(y.astype(int), minlength=self.n_classes_)
        total = counts.sum()
        self._priors = counts.astype(float) / (total + 1e-9)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        # Start with class priors for every row
        proba = np.tile(self._priors, (n, 1)).astype(float)

        if X.shape[1] > self.vol_feature_idx:
            vol = X[:, self.vol_feature_idx]
            high_vol = vol > self._threshold

            if self.n_classes_ == 3:
                flat_row = np.array([0.0, 1.0, 0.0])
                proba[high_vol] = flat_row
            else:
                proba[high_vol] = np.array([0.5, 0.5])

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def __repr__(self) -> str:
        return (
            f"VolatilityNoTradeBaseline("
            f"vol_idx={self.vol_feature_idx}, "
            f"threshold_pct={self.vol_threshold_pct})"
        )


# ---------------------------------------------------------------------------
# Registry of all baseline constructors by name
# ---------------------------------------------------------------------------

BASELINE_REGISTRY = {
    "prior": PriorBaseline,
    "momentum": MomentumBaseline,
    "anti_momentum": AntiMomentumBaseline,
    "persistence": PersistenceBaseline,
    "vol_no_trade": VolatilityNoTradeBaseline,
}
