"""
Baseline models for next-candle direction prediction.

Models:
1. Naive: predict same direction as last closed candle
2. Logistic Regression
3. Random Forest
4. Gradient Boosted Trees (scikit-learn HistGradientBoosting)

All models output:
- predict_proba(X) → [prob_down, prob_up]
- feature_importance() → dict {name: importance}
- serialize() / load()

Training uses TimeSeriesSplit to avoid leakage.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
    precision_score, recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "model_artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


class NaiveBaseline:
    """Predict the same direction as the last bar."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBaseline":
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Use last feature (ret_1) to determine direction."""
        # X[:,8] is ret_1 (index in FEATURE_COLS)
        ret1_idx = 8  # ret_1 position in feature array
        if X.shape[1] > ret1_idx:
            ret1 = X[:, ret1_idx]
            prob_up = np.where(ret1 >= 0, 0.6, 0.4)
        else:
            prob_up = np.full(len(X), 0.5)
        return np.column_stack([1 - prob_up, prob_up])

    def feature_importance(self) -> Dict[str, float]:
        return {}

    def __repr__(self):
        return "NaiveBaseline"


def _make_lr_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=500, class_weight="balanced", solver="lbfgs")),
    ])


def _make_rf_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ])


def _make_gbt_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=100, max_depth=4, min_samples_leaf=20,
            learning_rate=0.05, random_state=42,
        )),
    ])


MODEL_REGISTRY = {
    "naive": NaiveBaseline,
    "logistic": _make_lr_pipeline,
    "random_forest": _make_rf_pipeline,
    "gbt": _make_gbt_pipeline,
}


def evaluate_model(
    model, X_test: np.ndarray, y_test: np.ndarray, feature_names: list = None
) -> Dict[str, Any]:
    """Compute all evaluation metrics for a model."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    brier = float(brier_score_loss(y_test, probs))
    # Brier skill score: 1 - (model_brier / reference_brier)
    # Reference = always predict base rate (naive classifier).
    # A random binary classifier has Brier ≈ 0.25.
    # bss > 0 means model is better than naive; bss < 0 means worse.
    base_rate = float(np.mean(y_test))
    reference_brier = float(brier_score_loss(y_test, np.full_like(probs, base_rate)))
    bss = 1.0 - brier / (reference_brier + 1e-9)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "brier_score": brier,
        "brier_skill_score": round(bss, 4),   # positive = beats naive
        "reference_brier": round(reference_brier, 6),
        "log_loss": float(log_loss(y_test, probs)),
        "n_samples": int(len(y_test)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, probs))
    except Exception:
        metrics["roc_auc"] = None

    # Calibration
    try:
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="uniform")
        metrics["calibration"] = {
            "bin_centers": mean_pred.tolist(),
            "fraction_positive": frac_pos.tolist(),
        }
    except Exception:
        metrics["calibration"] = None

    # Feature importance
    clf = model.named_steps.get("clf") if hasattr(model, "named_steps") else None
    if clf is not None and hasattr(clf, "feature_importances_") and feature_names:
        imp = clf.feature_importances_
        metrics["feature_importance"] = dict(zip(feature_names, imp.tolist()))
    elif clf is not None and hasattr(clf, "coef_") and feature_names:
        scaler = model.named_steps.get("scaler")
        coef = clf.coef_[0]
        if scaler is not None:
            coef = coef * scaler.scale_  # un-standardize
        metrics["feature_importance"] = dict(zip(feature_names, np.abs(coef).tolist()))
    else:
        metrics["feature_importance"] = {}

    return metrics


def train_with_walk_forward(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "logistic",
    n_splits: int = 5,
    feature_names: list = None,
) -> Tuple[Any, list]:
    """
    Walk-forward evaluation using TimeSeriesSplit.
    Returns (final_model_trained_on_all, fold_metrics).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) < 2:
            logger.warning("Fold %d: only one class in training set, skipping", fold)
            continue

        model = MODEL_REGISTRY[model_name]() if model_name != "naive" else NaiveBaseline()
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test, feature_names)
        metrics["fold"] = fold
        metrics["train_size"] = len(train_idx)
        metrics["test_size"] = len(test_idx)

        # Train-set Brier — gap between train and test reveals overfitting.
        # Ratio > 1.5 (test Brier 50% worse than train) is a red flag.
        train_probs = model.predict_proba(X_train)[:, 1]
        train_brier = float(brier_score_loss(y_train, train_probs))
        metrics["train_brier"] = round(train_brier, 6)
        metrics["overfit_ratio"] = round(
            metrics["brier_score"] / (train_brier + 1e-9), 3
        )

        fold_metrics.append(metrics)

        logger.info(
            "Fold %d: acc=%.3f brier=%.4f train_brier=%.4f overfit_ratio=%.2f bss=%.3f auc=%s",
            fold, metrics["accuracy"], metrics["brier_score"], train_brier,
            metrics["overfit_ratio"], metrics["brier_skill_score"],
            f"{metrics['roc_auc']:.3f}" if metrics["roc_auc"] else "N/A"
        )

    # Train final model on all data
    final = MODEL_REGISTRY[model_name]() if model_name != "naive" else NaiveBaseline()
    if len(np.unique(y)) >= 2:
        final.fit(X, y)
    return final, fold_metrics


def save_model(model, model_name: str, metadata: dict = None):
    path = ARTIFACTS_DIR / f"{model_name}.pkl"
    meta_path = ARTIFACTS_DIR / f"{model_name}_meta.json"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    if metadata:
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
    logger.info("Saved model: %s", path)


def load_model(model_name: str):
    path = ARTIFACTS_DIR / f"{model_name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
