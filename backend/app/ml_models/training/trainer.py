"""
Model training pipeline.

Responsibilities:
  1. Build all model candidates (baselines + parametric)
  2. Run walk-forward evaluation with purge/embargo
  3. Collect fold metrics, confidence analysis, importance
  4. Select the best model by Brier score (not headline accuracy)
  5. Re-train the winner on the full dataset
  6. Optionally run ablation study on fold 0
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.ml_models.training.config import TrainingConfig
from app.ml_models.training.splitter import PurgedWalkForwardSplit, FoldIndices
from app.ml_models.training.baselines import BASELINE_REGISTRY
from app.ml_models.training.calibration import calibrate_model
from app.ml_models.evaluation.metrics import (
    FoldMetrics,
    AggregatedMetrics,
    compute_fold_metrics,
    aggregate_fold_metrics,
)
from app.ml_models.evaluation.confidence import ConfidenceAnalysis, confidence_bucket_analysis
from app.ml_models.evaluation.importance import (
    FeatureImportance,
    AblationResult,
    extract_intrinsic_importance,
    permutation_importance,
    merge_importance,
    group_ablation,
)
from app.ml_models.evaluation.regime import RegimeMetrics, regime_segmented_evaluation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _build_sklearn_model(name: str, cfg: TrainingConfig):
    """
    Build an unfitted sklearn model (Pipeline) by name.

    Returns None for baseline names (handled separately).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "logistic_l2":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=cfg.lr_C, penalty="l2", max_iter=cfg.lr_max_iter,
                solver=cfg.lr_solver, class_weight="balanced",
                random_state=cfg.random_seed,
            )),
        ])
    if name == "logistic_l1":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=cfg.lr_C, penalty="l1", max_iter=cfg.lr_max_iter,
                solver="saga", class_weight="balanced",
                random_state=cfg.random_seed,
            )),
        ])
    if name == "logistic_elasticnet":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=cfg.lr_C, penalty="elasticnet", l1_ratio=0.5,
                max_iter=cfg.lr_max_iter, solver="saga",
                class_weight="balanced", random_state=cfg.random_seed,
            )),
        ])
    if name == "gbt":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(
                max_iter=cfg.gbt_max_iter,
                max_depth=cfg.gbt_max_depth,
                learning_rate=cfg.gbt_learning_rate,
                min_samples_leaf=cfg.gbt_min_samples_leaf,
                l2_regularization=cfg.gbt_l2_regularization,
                random_state=cfg.random_seed,
            )),
        ])
    if name == "random_forest":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=cfg.rf_n_estimators,
                max_depth=cfg.rf_max_depth,
                min_samples_leaf=cfg.rf_min_samples_leaf,
                class_weight="balanced",
                random_state=cfg.random_seed,
                n_jobs=-1,
            )),
        ])
    return None


# ---------------------------------------------------------------------------
# Helpers for binary / ternary evaluation
# ---------------------------------------------------------------------------

def _get_prob_up(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Extract P(UP) from a model's predict_proba output.

    Binary model  (proba shape (n, 2)): return proba[:, 1]
    Ternary model (proba shape (n, 3)): return proba[:, 2]  (UP = class index 2)
    """
    proba = model.predict_proba(X)
    if proba.shape[1] == 3:
        return proba[:, 2]
    return proba[:, 1]


def _binarize_for_eval(y: np.ndarray, label_type: str) -> np.ndarray:
    """
    Binarize labels for metric computation (always returns {0, 1}).

    binary  → identity
    ternary → UP-vs-rest: (y == 2).astype(int)
    """
    if label_type == "ternary":
        return (y == 2).astype(int)
    return y.astype(int)


# ---------------------------------------------------------------------------
# Data structures for results
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    model_name: str
    is_baseline: bool
    fold_metrics: List[FoldMetrics]
    aggregated: AggregatedMetrics
    confidence_analysis: Optional[ConfidenceAnalysis]      # pooled across folds
    regime_metrics: Optional[List[RegimeMetrics]]          # pooled across folds
    importance: Optional[List[FeatureImportance]]          # last fold
    ablation: Optional[List[AblationResult]]               # fold 0 only
    calibration_summary: Optional[dict]                    # ECE + curve
    # The final model trained on all data (None until finalize() is called)
    final_model: Optional[Any] = None
    # Pooled OOS predictions for artifact saving / offline analysis
    pooled_oos_y: Optional[np.ndarray] = None         # binarized y_true
    pooled_oos_prob_up: Optional[np.ndarray] = None   # P(UP) from model

    def primary_score(self, metric: str = "brier_score") -> float:
        """Return the mean value of the primary selection metric."""
        val = getattr(self.aggregated, f"{metric}_mean", None)
        return float(val) if val is not None else float("inf")


@dataclass
class TrainingReport:
    config: TrainingConfig
    model_results: List[ModelResult]
    winner: ModelResult
    splitter_description: List[dict]
    feature_names: List[str]

    # ------------------------------------------------------------------
    # Convenience accessors (tests and downstream consumers use these)
    # ------------------------------------------------------------------

    @property
    def brier_score(self) -> Optional[float]:
        """Winner's mean Brier score across walk-forward folds."""
        return self.winner.aggregated.brier_score_mean

    @property
    def metrics(self) -> dict:
        """Winner's aggregated metrics as a flat dict (JSON-serialisable)."""
        ag = self.winner.aggregated
        return {
            "brier_score": ag.brier_score_mean,
            "brier_score_std": ag.brier_score_std,
            "log_loss": ag.log_loss_mean,
            "log_loss_std": ag.log_loss_std,
            "roc_auc": ag.roc_auc_mean,
            "balanced_accuracy": ag.balanced_accuracy_mean,
            "ece": ag.ece_mean,
            "n_folds": ag.n_folds,
            "total_test_samples": ag.total_test_samples,
        }

    def to_summary_dict(self) -> dict:
        return {
            "config_hash": self.config.config_hash(),
            "n_models": len(self.model_results),
            "winner": self.winner.model_name,
            "winner_brier": self.winner.primary_score("brier_score"),
            "models": [
                {
                    "name": r.model_name,
                    "is_baseline": r.is_baseline,
                    "brier_mean": r.aggregated.brier_score_mean,
                    "brier_std": r.aggregated.brier_score_std,
                    "log_loss_mean": r.aggregated.log_loss_mean,
                    "roc_auc_mean": r.aggregated.roc_auc_mean,
                    "balanced_accuracy_mean": r.aggregated.balanced_accuracy_mean,
                    "ece_mean": r.aggregated.ece_mean,
                    "n_folds": r.aggregated.n_folds,
                    "total_test_samples": r.aggregated.total_test_samples,
                }
                for r in self.model_results
            ],
        }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_all_models(
    X: np.ndarray,
    y: np.ndarray,
    cfg: TrainingConfig,
    feature_names: List[str],
    returns: Optional[np.ndarray] = None,
    regimes: Optional[np.ndarray] = None,
) -> TrainingReport:
    """
    Run walk-forward training and evaluation for all models in cfg.model_names.

    Parameters
    ----------
    X : ndarray (n, p) — feature matrix (valid rows only, time-ordered)
    y : ndarray (n,)   — binary labels {0, 1}
    cfg : TrainingConfig
    feature_names : list of column names corresponding to X[:, i]
    returns : ndarray (n,) or None
        One-bar returns aligned with X/y. Used for trading utility metrics.
    regimes : ndarray (n,) or None
        Regime string per sample. Used for regime-segmented evaluation.
        If None, regime evaluation is skipped.
    """
    rng = np.random.default_rng(cfg.random_seed)
    splitter = PurgedWalkForwardSplit(
        n_splits=cfg.n_splits,
        test_window_bars=cfg.test_window_bars,
        embargo_bars=cfg.embargo_bars,
        min_train_bars=cfg.min_train_bars,
    )

    model_names = list(cfg.model_names)
    if cfg.include_random_forest and "random_forest" not in model_names:
        model_names.append("random_forest")

    all_folds = list(splitter.split(X))
    if not all_folds:
        raise ValueError(
            f"No valid folds generated. n={len(X)} samples, "
            f"n_splits={cfg.n_splits}, test_window={cfg.test_window_bars}, "
            f"embargo={cfg.embargo_bars}, min_train={cfg.min_train_bars}"
        )

    splitter_desc = splitter.describe(len(X))
    logger.info(
        "Walk-forward: %d folds, test_window=%s, embargo=%d",
        len(all_folds), cfg.test_window_bars, cfg.embargo_bars,
    )

    # Containers for pooled predictions (for confidence + regime analysis)
    pooled: Dict[str, Dict] = {
        name: {"y": [], "prob_up": [], "returns": [], "regimes": []}
        for name in model_names
    }

    model_fold_metrics: Dict[str, List[FoldMetrics]] = {n: [] for n in model_names}

    # -----------------------------------------------------------------------
    # Walk-forward loop
    # -----------------------------------------------------------------------
    for fi in all_folds:
        X_tr, y_tr = X[fi.train_idx], y[fi.train_idx]
        X_te, y_te = X[fi.test_idx], y[fi.test_idx]
        ret_te = returns[fi.test_idx] if returns is not None else None
        reg_te = regimes[fi.test_idx] if regimes is not None else None

        if len(np.unique(y_tr)) < 2:
            logger.warning("Fold %d: single class in training set, skipping.", fi.fold)
            continue

        for name in model_names:
            try:
                model = _fit_model(name, X_tr, y_tr, cfg)
            except Exception as e:
                logger.warning("Fold %d: failed to fit %s: %s", fi.fold, name, e)
                continue

            prob_up = _get_prob_up(model, X_te)
            y_eval = _binarize_for_eval(y_te, cfg.label_type)
            fm = compute_fold_metrics(fi.fold, len(fi.train_idx), y_eval, prob_up)
            model_fold_metrics[name].append(fm)

            # Accumulate pooled predictions (store binarized y for consistent evaluation)
            pooled[name]["y"].append(y_eval)
            pooled[name]["prob_up"].append(prob_up)
            if ret_te is not None:
                pooled[name]["returns"].append(ret_te)
            if reg_te is not None:
                pooled[name]["regimes"].append(reg_te)

        logger.info(
            "Fold %d: train=%d test=%d",
            fi.fold, len(fi.train_idx), len(fi.test_idx),
        )

    # -----------------------------------------------------------------------
    # Assemble ModelResult for each model
    # -----------------------------------------------------------------------
    model_results: List[ModelResult] = []

    for name in model_names:
        fold_metrics = model_fold_metrics[name]
        if not fold_metrics:
            logger.warning("No folds completed for model %s — skipping.", name)
            continue

        aggregated = aggregate_fold_metrics(fold_metrics)

        # Pooled predictions across all folds
        y_pool = np.concatenate(pooled[name]["y"])
        p_pool = np.concatenate(pooled[name]["prob_up"])
        ret_pool = (
            np.concatenate(pooled[name]["returns"])
            if pooled[name]["returns"] else None
        )
        reg_pool = (
            np.concatenate(pooled[name]["regimes"])
            if pooled[name]["regimes"] else None
        )

        conf_analysis = confidence_bucket_analysis(
            y_pool, p_pool, n_bins=cfg.n_confidence_bins, returns=ret_pool
        )

        regime_eval = None
        if reg_pool is not None:
            regime_eval = regime_segmented_evaluation(
                y_pool, p_pool, reg_pool,
                confidence_threshold=cfg.confidence_threshold,
                min_samples=cfg.min_regime_samples,
            )

        from app.ml_models.training.calibration import calibration_summary as _cal_summary
        cal_summary = _cal_summary(y_pool, p_pool)

        # Importance: use last fold's test data with the last fold's model
        importance_list = None
        if name not in BASELINE_REGISTRY:
            try:
                last_fi = all_folds[-1]
                X_tr_last = X[last_fi.train_idx]
                y_tr_last = y[last_fi.train_idx]
                X_te_last = X[last_fi.test_idx]
                y_te_last = y[last_fi.test_idx]
                last_model = _fit_model(name, X_tr_last, y_tr_last, cfg)
                perm = permutation_importance(
                    last_model, X_te_last, y_te_last,
                    feature_names, n_repeats=3, rng=rng,
                )
                intrinsic = extract_intrinsic_importance(last_model, feature_names)
                importance_list = merge_importance(perm, intrinsic)
            except Exception as e:
                logger.warning("Importance computation failed for %s: %s", name, e)

        # Ablation: fold 0, parametric models only, if requested
        ablation_results = None
        if cfg.run_ablation and name not in BASELINE_REGISTRY and len(all_folds) > 0:
            try:
                from app.feature_pipeline.registry import FEATURE_GROUPS
                fi0 = all_folds[0]
                ablation_results = group_ablation(
                    build_model_fn=lambda: _build_sklearn_model(name, cfg),
                    X_train=X[fi0.train_idx],
                    y_train=y[fi0.train_idx],
                    X_test=X[fi0.test_idx],
                    y_test=y[fi0.test_idx],
                    feature_names=feature_names,
                    feature_groups={
                        g: [f for f in feats if f in feature_names]
                        for g, feats in FEATURE_GROUPS.items()
                    },
                )
            except Exception as e:
                logger.warning("Ablation failed for %s: %s", name, e)

        model_results.append(ModelResult(
            model_name=name,
            is_baseline=name in BASELINE_REGISTRY,
            fold_metrics=fold_metrics,
            aggregated=aggregated,
            confidence_analysis=conf_analysis,
            regime_metrics=regime_eval,
            importance=importance_list,
            ablation=ablation_results,
            calibration_summary=cal_summary,
            final_model=None,
            pooled_oos_y=y_pool,
            pooled_oos_prob_up=p_pool,
        ))

    # -----------------------------------------------------------------------
    # Select winner (non-baseline with best Brier score)
    # -----------------------------------------------------------------------
    non_baselines = [r for r in model_results if not r.is_baseline]
    if not non_baselines:
        raise ValueError("No parametric models were successfully trained.")

    direction = cfg.selection_metric_direction
    key = cfg.selection_metric + "_mean"

    def _score(r: ModelResult) -> float:
        val = getattr(r.aggregated, key, float("inf"))
        return float(val) if val is not None else float("inf")

    if direction == "lower":
        winner = min(non_baselines, key=_score)
    else:
        winner = max(non_baselines, key=lambda r: -(_score(r)))

    logger.info(
        "Winner: %s  %s=%.4f",
        winner.model_name, key, _score(winner),
    )

    # -----------------------------------------------------------------------
    # Finalize: re-train winner on full dataset
    # -----------------------------------------------------------------------
    try:
        final = _fit_model(winner.model_name, X, y, cfg)
        winner.final_model = final
    except Exception as e:
        logger.error("Failed to train final model for %s: %s", winner.model_name, e)

    return TrainingReport(
        config=cfg,
        model_results=model_results,
        winner=winner,
        splitter_description=splitter_desc,
        feature_names=feature_names,
    )


def _fit_model(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: TrainingConfig,
):
    """Fit one model by name. Returns the fitted model."""
    if name in BASELINE_REGISTRY:
        model = BASELINE_REGISTRY[name]()
        model.fit(X_train, y_train)
        return model

    sklearn_model = _build_sklearn_model(name, cfg)
    if sklearn_model is None:
        raise ValueError(f"Unknown model name: {name}")

    sklearn_model.fit(X_train, y_train)
    return sklearn_model
