"""
Feature importance and ablation analysis.

Three complementary importance measures:

1. Model-intrinsic importance
   - Logistic: |coef * scale| (contribution of each feature to the decision boundary)
   - GBT / RF: mean decrease in impurity (feature_importances_)
   Both are biased toward continuous or high-cardinality features, so treat
   as directional, not definitive.

2. Permutation importance (model-agnostic)
   Randomly shuffles each feature and measures the drop in Brier score.
   A large drop → the model depends on that feature for calibration.
   A small drop → the feature is redundant or not used.
   Does NOT require re-training; fast and model-agnostic.

3. Group ablation (re-training based)
   Drops one feature group at a time (per app.feature_pipeline.registry.FEATURE_GROUPS),
   retrains the model on the first fold, and compares Brier scores.
   The Brier score delta tells you how much predictive power each group contributes.
   This is the most reliable measure but requires re-training n_groups times.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FeatureImportance:
    feature: str
    intrinsic_importance: Optional[float]   # from model coefficients/impurity
    permutation_importance: Optional[float] # brier score increase when shuffled
    permutation_std: Optional[float]        # std across n_repeats shuffles
    rank_intrinsic: Optional[int]
    rank_permutation: Optional[int]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AblationResult:
    group: str
    features_dropped: List[str]
    brier_score_baseline: float
    brier_score_ablated: float
    brier_delta: float              # positive = group helped; negative = group hurt
    pct_change: float               # (ablated - baseline) / baseline * 100

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Intrinsic importance
# ---------------------------------------------------------------------------

def extract_intrinsic_importance(
    model,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    Extract model-intrinsic feature importance.

    Supports sklearn Pipeline with 'scaler' and 'clf' named steps.
    For logistic: uses scaled coefficient magnitudes.
    For tree models: uses feature_importances_.
    Returns empty dict if the model does not expose importance.
    """
    clf = model.named_steps.get("clf") if hasattr(model, "named_steps") else model
    scaler = model.named_steps.get("scaler") if hasattr(model, "named_steps") else None

    if clf is None:
        return {}

    n = len(feature_names)

    if hasattr(clf, "coef_"):
        coef = clf.coef_[0][:n]
        if scaler is not None and hasattr(scaler, "scale_"):
            # Multiply by feature std to get "per-std-unit" contribution
            scale = scaler.scale_[:n]
            importance = np.abs(coef * scale)
        else:
            importance = np.abs(coef)
        return dict(zip(feature_names, importance.tolist()))

    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_[:n]
        return dict(zip(feature_names, imp.tolist()))

    return {}


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

def _brier_score(y: np.ndarray, prob_up: np.ndarray) -> float:
    return float(np.mean((y - prob_up) ** 2))


def permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> List[FeatureImportance]:
    """
    Compute permutation importance for each feature.

    For each feature:
      1. Compute baseline Brier score on (X, y).
      2. Shuffle that feature column n_repeats times.
      3. Mean Brier score increase = importance (positive = feature is useful).

    Returns list of FeatureImportance sorted by permutation_importance descending.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    baseline_probs = model.predict_proba(X)[:, 1]
    baseline_brier = _brier_score(y, baseline_probs)

    results = []
    n_features = min(X.shape[1], len(feature_names))

    for i, name in enumerate(feature_names[:n_features]):
        deltas = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            perm_probs = model.predict_proba(X_perm)[:, 1]
            perm_brier = _brier_score(y, perm_probs)
            deltas.append(perm_brier - baseline_brier)

        results.append(FeatureImportance(
            feature=name,
            intrinsic_importance=None,   # filled later
            permutation_importance=round(float(np.mean(deltas)), 6),
            permutation_std=round(float(np.std(deltas)), 6),
            rank_intrinsic=None,
            rank_permutation=None,
        ))

    # Assign permutation ranks
    results.sort(key=lambda x: -(x.permutation_importance or 0))
    for rank, r in enumerate(results, 1):
        r.rank_permutation = rank

    return results


def merge_importance(
    perm: List[FeatureImportance],
    intrinsic: Dict[str, float],
) -> List[FeatureImportance]:
    """Merge permutation and intrinsic importance into a single sorted list."""
    by_name = {r.feature: r for r in perm}

    # Fill intrinsic importance
    for name, val in intrinsic.items():
        if name in by_name:
            by_name[name].intrinsic_importance = round(float(val), 6)

    # Assign intrinsic ranks
    with_intrinsic = [(n, v) for n, v in intrinsic.items() if n in by_name]
    with_intrinsic.sort(key=lambda x: -x[1])
    for rank, (name, _) in enumerate(with_intrinsic, 1):
        by_name[name].rank_intrinsic = rank

    return sorted(by_name.values(), key=lambda x: -(x.permutation_importance or 0))


# ---------------------------------------------------------------------------
# Group ablation
# ---------------------------------------------------------------------------

def group_ablation(
    build_model_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    feature_groups: Dict[str, List[str]],
) -> List[AblationResult]:
    """
    Measure the Brier score impact of removing each feature group.

    Parameters
    ----------
    build_model_fn : callable
        Returns an unfitted sklearn-compatible model.
    X_train, y_train : training data
    X_test, y_test : test data
    feature_names : list of column names corresponding to X columns
    feature_groups : {group_name: [feature_name, ...]}
        From app.feature_pipeline.registry.FEATURE_GROUPS.

    For each group, columns belonging to that group are zeroed out in both
    X_train and X_test before fitting. Zeroing (not dropping) preserves the
    feature matrix shape, which is required for models that expect a fixed
    input dimension. The model is retrained from scratch on each ablation.
    """
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    # Baseline
    baseline_model = build_model_fn()
    baseline_model.fit(X_train, y_train)
    baseline_prob = baseline_model.predict_proba(X_test)[:, 1]
    baseline_brier = _brier_score(y_test, baseline_prob)

    results = []
    for group, group_features in feature_groups.items():
        indices = [name_to_idx[f] for f in group_features if f in name_to_idx]
        if not indices:
            continue

        X_tr_abl = X_train.copy()
        X_te_abl = X_test.copy()
        X_tr_abl[:, indices] = 0.0
        X_te_abl[:, indices] = 0.0

        try:
            model = build_model_fn()
            model.fit(X_tr_abl, y_train)
            abl_prob = model.predict_proba(X_te_abl)[:, 1]
            abl_brier = _brier_score(y_test, abl_prob)
        except Exception:
            abl_brier = float("nan")

        delta = abl_brier - baseline_brier
        pct = (delta / (abs(baseline_brier) + 1e-9)) * 100

        results.append(AblationResult(
            group=group,
            features_dropped=group_features,
            brier_score_baseline=round(baseline_brier, 6),
            brier_score_ablated=round(abl_brier, 6),
            brier_delta=round(delta, 6),
            pct_change=round(pct, 2),
        ))

    # Sort by delta descending (largest positive = most valuable group)
    results.sort(key=lambda r: -(r.brier_delta if not np.isnan(r.brier_delta) else -999))
    return results
