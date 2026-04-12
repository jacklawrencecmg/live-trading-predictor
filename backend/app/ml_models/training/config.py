"""
Training configuration — single source of reproducible hyperparameters.

Every pipeline run is stamped with the config hash so that artifacts can
be traced back to the exact hyperparameters that produced them.
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Tuple

from app.feature_pipeline.registry import (
    FEATURE_COLS,
    MANIFEST_HASH,
    PIPELINE_VERSION,
)


@dataclass(frozen=True)
class TrainingConfig:
    # ---------------------------------------------------------------------------
    # Walk-forward split
    # ---------------------------------------------------------------------------
    n_splits: int = 5
    # Fixed-width test window (bars). None = expanding from each split point.
    test_window_bars: int = 500
    # Bars dropped between train-end and test-start to prevent label overlap.
    # For 1-bar lookahead labels, embargo=1 is sufficient.
    # For k-bar lookahead, set embargo = k.
    embargo_bars: int = 1
    # Minimum train bars required; folds with fewer are skipped.
    min_train_bars: int = 250

    # ---------------------------------------------------------------------------
    # Models to include (order controls report ordering)
    # ---------------------------------------------------------------------------
    # Baselines always included: "prior", "momentum", "anti_momentum"
    # Parametric models: "logistic_l2", "logistic_l1", "logistic_elasticnet"
    # Trees: "gbt" (always), "random_forest" (only if justified)
    model_names: Tuple[str, ...] = (
        "prior",
        "momentum",
        "anti_momentum",
        "logistic_l2",
        "logistic_l1",
        "gbt",
    )
    # RF is excluded by default — include explicitly when justified
    include_random_forest: bool = False

    # ---------------------------------------------------------------------------
    # Logistic regression
    # ---------------------------------------------------------------------------
    lr_C: float = 0.1                 # regularization (inverse)
    lr_max_iter: int = 500
    lr_solver: str = "lbfgs"

    # ---------------------------------------------------------------------------
    # Gradient boosted trees (HistGradientBoosting)
    # ---------------------------------------------------------------------------
    gbt_max_iter: int = 200
    gbt_max_depth: int = 4
    gbt_learning_rate: float = 0.05
    gbt_min_samples_leaf: int = 20
    gbt_l2_regularization: float = 0.1

    # ---------------------------------------------------------------------------
    # Random forest
    # ---------------------------------------------------------------------------
    rf_n_estimators: int = 200
    rf_max_depth: int = 6
    rf_min_samples_leaf: int = 20

    # ---------------------------------------------------------------------------
    # Probability calibration
    # ---------------------------------------------------------------------------
    # "isotonic" is better for large datasets; "sigmoid" (Platt scaling) for small.
    calibration_method: str = "isotonic"
    calibration_cv: int = 3

    # ---------------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------------
    # Number of confidence bins for bucket analysis
    n_confidence_bins: int = 5
    # Minimum confidence to count as a trade signal
    confidence_threshold: float = 0.55
    # Run feature-group ablation (adds ~n_groups × fold_0 training cost)
    run_ablation: bool = True
    # Minimum samples per regime to report regime metrics
    min_regime_samples: int = 30

    # ---------------------------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------------------------
    random_seed: int = 42

    # ---------------------------------------------------------------------------
    # Feature metadata (baked into config for artifact traceability)
    # ---------------------------------------------------------------------------
    feature_cols: Tuple[str, ...] = tuple(FEATURE_COLS)
    manifest_hash: str = MANIFEST_HASH
    pipeline_version: int = PIPELINE_VERSION

    # ---------------------------------------------------------------------------
    # Model selection criterion
    # ---------------------------------------------------------------------------
    # Primary: "brier_score" (lower is better) | "log_loss" | "roc_auc" | "balanced_accuracy"
    # DO NOT use "accuracy" alone — class imbalance makes it misleading.
    selection_metric: str = "brier_score"
    selection_metric_direction: str = "lower"   # "lower" | "higher"

    def config_hash(self) -> str:
        """SHA-256 of the config dict — changes if any hyperparameter changes."""
        d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}
        payload = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        d = asdict(self)
        # Tuples aren't JSON-serializable; convert to lists
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


# Default config singleton — override by constructing TrainingConfig(...)
DEFAULT_CONFIG = TrainingConfig()
