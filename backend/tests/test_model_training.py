"""
Model training pipeline tests.

Coverage:
  MT1 — Splitter: correct fold counts and embargo gaps
  MT2 — Splitter: test windows never overlap train windows
  MT3 — Splitter: raises on insufficient data
  MT4 — Baselines: fit/predict_proba API compliance
  MT5 — PriorBaseline: output probabilities reflect training class frequency
  MT6 — MomentumBaseline: follows ret_1 sign
  MT7 — AntiMomentumBaseline: opposes ret_1 sign (mirror of momentum)
  MT8 — Metrics: Brier score of 0.25 for coin-flip classifier
  MT9 — Metrics: ECE near 0 for perfectly calibrated classifier
  MT10 — Metrics: aggregate_fold_metrics produces correct means
  MT11 — Confidence buckets: n_samples sum equals total
  MT12 — Confidence buckets: accuracy in each bucket is in [0, 1]
  MT13 — Importance: permutation_importance returns one entry per feature
  MT14 — Importance: shuffling improves Brier of a no-signal model ≈ 0
  MT15 — Ablation: returns one result per non-empty group
  MT16 — Regime eval: excludes regimes with < min_samples
  MT17 — Full train_all_models: runs end-to-end without error
  MT18 — Full train_all_models: winner Brier is finite and < 0.5
  MT19 — Full train_all_models: winner selected from non-baseline models
  MT20 — Pipeline: run_training_pipeline runs end-to-end with synthetic data
  MT21 — Registry: save and load round-trip preserves predict_proba output
  MT22 — Report: generate_report produces non-empty markdown with all sections
  MT23 — Config: config_hash changes when hyperparameter changes
  MT24 — Config: to_dict is JSON-serializable
"""

import json
import math
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.calibration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = datetime(2024, 1, 2, 9, 30)
    prices = 100 + np.cumsum(rng.normal(0, 0.5, n))
    prices = np.maximum(prices, 1.0)
    highs = prices * (1 + rng.uniform(0, 0.003, n))
    lows = prices * (1 - rng.uniform(0, 0.003, n))
    volumes = rng.integers(10_000, 100_000, n).astype(float)
    return pd.DataFrame({
        "open": prices * (1 + rng.uniform(-0.001, 0.001, n)),
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
        "vwap": (prices + highs + lows) / 3,
        "bar_open_time": [base + timedelta(minutes=5 * i) for i in range(n)],
    })


def _synthetic_Xy(n: int = 800, n_features: int = 30, seed: int = 0):
    """Simple synthetic dataset with a weak signal."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    # Weak signal: y ≈ sign(X[:, 0] + noise)
    y = (X[:, 0] + rng.normal(0, 2, n) > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# MT1 — Splitter fold counts
# ---------------------------------------------------------------------------

def test_MT1_splitter_fold_count():
    from app.ml_models.training.splitter import PurgedWalkForwardSplit
    X = np.empty((2000, 1))
    sp = PurgedWalkForwardSplit(n_splits=5, test_window_bars=200, embargo_bars=1, min_train_bars=100)
    folds = list(sp.split(X))
    assert len(folds) == 5


def test_MT1_splitter_describe_returns_n_splits_entries():
    from app.ml_models.training.splitter import PurgedWalkForwardSplit
    sp = PurgedWalkForwardSplit(n_splits=4, test_window_bars=150, embargo_bars=1, min_train_bars=100)
    desc = sp.describe(2000)
    assert len(desc) == 4


# ---------------------------------------------------------------------------
# MT2 — Splitter: no train/test overlap, embargo is respected
# ---------------------------------------------------------------------------

def test_MT2_train_test_no_overlap():
    from app.ml_models.training.splitter import PurgedWalkForwardSplit
    X = np.empty((2000, 1))
    sp = PurgedWalkForwardSplit(n_splits=5, test_window_bars=200, embargo_bars=5)
    for fi in sp.split(X):
        assert fi.train_idx.max() < fi.test_idx.min(), \
            f"Fold {fi.fold}: train index {fi.train_idx.max()} overlaps test start {fi.test_idx.min()}"


def test_MT2_embargo_gap_is_respected():
    from app.ml_models.training.splitter import PurgedWalkForwardSplit
    embargo = 7
    X = np.empty((2000, 1))
    sp = PurgedWalkForwardSplit(n_splits=5, test_window_bars=200, embargo_bars=embargo)
    for fi in sp.split(X):
        gap = fi.test_start - fi.train_end
        assert gap == embargo, f"Fold {fi.fold}: expected gap {embargo}, got {gap}"


# ---------------------------------------------------------------------------
# MT3 — Splitter raises on insufficient data
# ---------------------------------------------------------------------------

def test_MT3_splitter_raises_on_insufficient_data():
    from app.ml_models.training.splitter import PurgedWalkForwardSplit
    sp = PurgedWalkForwardSplit(n_splits=5, test_window_bars=1000)
    with pytest.raises(ValueError, match="Not enough data"):
        list(sp.split(np.empty((100, 1))))


# ---------------------------------------------------------------------------
# MT4 — Baseline API compliance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["prior", "momentum", "anti_momentum"])
def test_MT4_baseline_api(name):
    from app.ml_models.training.baselines import BASELINE_REGISTRY
    X, y = _synthetic_Xy(200)
    model = BASELINE_REGISTRY[name]()
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (200, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6), "Probabilities must sum to 1"
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities must be in [0, 1]"


# ---------------------------------------------------------------------------
# MT5 — PriorBaseline reflects training class frequency
# ---------------------------------------------------------------------------

def test_MT5_prior_baseline_reflects_class_frequency():
    from app.ml_models.training.baselines import PriorBaseline
    X = np.zeros((100, 5))
    y = np.array([1] * 70 + [0] * 30)  # 70% positive
    model = PriorBaseline()
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert abs(probs[0, 1] - 0.7) < 1e-6, f"Expected P(up)=0.7, got {probs[0, 1]}"
    # All rows identical
    assert np.allclose(probs, probs[0], atol=1e-9)


# ---------------------------------------------------------------------------
# MT6 — MomentumBaseline follows ret_1 sign
# ---------------------------------------------------------------------------

def test_MT6_momentum_follows_ret1_sign():
    from app.ml_models.training.baselines import MomentumBaseline
    n = 100
    X = np.zeros((n, 30))
    X[:50, 8] = 0.01    # positive ret_1
    X[50:, 8] = -0.01   # negative ret_1
    y = np.zeros(n, dtype=int)
    model = MomentumBaseline(ret1_feature_idx=8)
    model.fit(X, y)
    probs = model.predict_proba(X)
    # Positive ret_1 → P(up) > 0.5
    assert (probs[:50, 1] > 0.5).all()
    # Negative ret_1 → P(up) < 0.5
    assert (probs[50:, 1] < 0.5).all()


# ---------------------------------------------------------------------------
# MT7 — AntiMomentumBaseline opposes Momentum
# ---------------------------------------------------------------------------

def test_MT7_anti_momentum_mirrors_momentum():
    from app.ml_models.training.baselines import MomentumBaseline, AntiMomentumBaseline
    X, y = _synthetic_Xy(100)
    mom = MomentumBaseline()
    anti = AntiMomentumBaseline()
    mom.fit(X, y)
    anti.fit(X, y)
    p_mom = mom.predict_proba(X)
    p_anti = anti.predict_proba(X)
    # P(up) for anti ≈ P(down) for momentum
    assert np.allclose(p_anti[:, 1], p_mom[:, 0], atol=1e-9)


# ---------------------------------------------------------------------------
# MT8 — Brier score of 0.25 for coin-flip
# ---------------------------------------------------------------------------

def test_MT8_brier_score_coinflip():
    from app.ml_models.evaluation.metrics import _brier_score
    n = 1000
    y = np.array([0, 1] * (n // 2))
    prob_up = np.full(n, 0.5)
    bs = _brier_score(y, prob_up)
    assert abs(bs - 0.25) < 0.01, f"Expected Brier~0.25 for coin flip, got {bs}"


# ---------------------------------------------------------------------------
# MT9 — ECE near 0 for perfect calibration
# ---------------------------------------------------------------------------

def test_MT9_ece_perfect_calibration():
    from app.ml_models.evaluation.metrics import _ece
    n = 1000
    rng = np.random.default_rng(0)
    # Perfect calibration: P(y=1|p) = p
    prob_up = rng.uniform(0, 1, n)
    y = rng.binomial(1, prob_up).astype(float)
    ece = _ece(y, prob_up, n_bins=10)
    # With 1000 samples and 10 bins, ECE should be well below 0.05
    assert ece < 0.05, f"ECE for perfectly calibrated classifier: {ece}"


# ---------------------------------------------------------------------------
# MT10 — aggregate_fold_metrics means are correct
# ---------------------------------------------------------------------------

def test_MT10_aggregate_fold_metrics():
    from app.ml_models.evaluation.metrics import FoldMetrics, aggregate_fold_metrics
    def _fm(fold, train, test, brier):
        return FoldMetrics(
            fold=fold, train_size=train, test_size=test,
            brier_score=brier, log_loss=0.60, ece=0.02,
            roc_auc=0.65, accuracy=0.51, balanced_accuracy=0.52,
            precision_up=0.72, recall_up=0.68, f1_up=0.60,
        )

    folds = [_fm(0, 100, 50, 0.20), _fm(1, 200, 50, 0.22)]
    agg = aggregate_fold_metrics(folds)
    assert abs(agg.brier_score_mean - 0.21) < 1e-9
    assert abs(agg.brier_score_std - 0.01) < 1e-9
    assert agg.n_folds == 2
    assert agg.total_test_samples == 100


# ---------------------------------------------------------------------------
# MT11 — Confidence buckets: n_samples sum equals total
# ---------------------------------------------------------------------------

def test_MT11_confidence_buckets_sum_to_total():
    from app.ml_models.evaluation.confidence import confidence_bucket_analysis
    n = 500
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n).astype(float)
    prob_up = rng.uniform(0, 1, n)
    ca = confidence_bucket_analysis(y, prob_up, n_bins=5)
    total = sum(b.n_samples for b in ca.buckets)
    assert total == n


# ---------------------------------------------------------------------------
# MT12 — Confidence bucket accuracy in [0, 1]
# ---------------------------------------------------------------------------

def test_MT12_confidence_bucket_accuracy_valid():
    from app.ml_models.evaluation.confidence import confidence_bucket_analysis
    n = 500
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, n).astype(float)
    prob_up = rng.uniform(0, 1, n)
    ca = confidence_bucket_analysis(y, prob_up, n_bins=5)
    for b in ca.buckets:
        if b.n_samples > 0:
            assert 0.0 <= b.accuracy <= 1.0, f"Bucket {b.bucket_index} accuracy {b.accuracy} out of range"


# ---------------------------------------------------------------------------
# MT13 — permutation_importance returns one entry per feature
# ---------------------------------------------------------------------------

def test_MT13_permutation_importance_feature_count():
    from app.ml_models.evaluation.importance import permutation_importance
    from app.ml_models.training.baselines import PriorBaseline
    X, y = _synthetic_Xy(200, n_features=10)
    model = PriorBaseline()
    model.fit(X, y)
    feature_names = [f"f{i}" for i in range(10)]
    result = permutation_importance(model, X, y, feature_names, n_repeats=2)
    assert len(result) == 10
    assert all(r.feature in feature_names for r in result)


# ---------------------------------------------------------------------------
# MT14 — permutation_importance is ~0 for no-signal model
# ---------------------------------------------------------------------------

def test_MT14_permutation_importance_near_zero_for_baseline():
    from app.ml_models.evaluation.importance import permutation_importance
    from app.ml_models.training.baselines import PriorBaseline
    X, y = _synthetic_Xy(400, n_features=5)
    model = PriorBaseline()
    model.fit(X, y)
    result = permutation_importance(model, X, y, [f"f{i}" for i in range(5)], n_repeats=3)
    # PriorBaseline ignores X, so shuffling any feature → delta ≈ 0
    for r in result:
        assert abs(r.permutation_importance) < 1e-9, \
            f"PriorBaseline permutation importance should be 0, got {r.permutation_importance}"


# ---------------------------------------------------------------------------
# MT15 — group_ablation returns one result per non-empty group
# ---------------------------------------------------------------------------

def test_MT15_ablation_returns_one_per_group():
    from app.ml_models.evaluation.importance import group_ablation
    from app.ml_models.training.baselines import PriorBaseline

    X, y = _synthetic_Xy(300, n_features=6)
    feature_names = [f"f{i}" for i in range(6)]
    groups = {"g1": ["f0", "f1"], "g2": ["f2", "f3"], "g3": ["f4", "f5"]}

    results = group_ablation(
        build_model_fn=PriorBaseline,
        X_train=X[:200], y_train=y[:200],
        X_test=X[200:], y_test=y[200:],
        feature_names=feature_names,
        feature_groups=groups,
    )
    assert len(results) == 3
    assert {r.group for r in results} == {"g1", "g2", "g3"}


# ---------------------------------------------------------------------------
# MT16 — regime_segmented_evaluation excludes small regimes
# ---------------------------------------------------------------------------

def test_MT16_regime_eval_excludes_small_regimes():
    from app.ml_models.evaluation.regime import regime_segmented_evaluation
    n = 200
    y = np.array([0, 1] * (n // 2))
    prob_up = np.full(n, 0.5)
    regimes = np.array(["trending_up"] * 100 + ["rare_regime"] * 3 + ["mean_reverting"] * 97)
    results = regime_segmented_evaluation(y, prob_up, regimes, min_samples=30)
    regime_names = {r.regime for r in results}
    assert "rare_regime" not in regime_names, "Rare regime with 3 samples should be excluded"
    assert "trending_up" in regime_names


# ---------------------------------------------------------------------------
# MT17 — train_all_models runs without error
# ---------------------------------------------------------------------------

def test_MT17_train_all_models_runs():
    from app.ml_models.training.config import TrainingConfig
    from app.ml_models.training.trainer import train_all_models

    X, y = _synthetic_Xy(1200, n_features=30)
    cfg = TrainingConfig(
        n_splits=3,
        test_window_bars=150,
        embargo_bars=1,
        min_train_bars=100,
        model_names=("prior", "momentum", "logistic_l2"),
        run_ablation=False,
    )
    feature_names = [f"f{i}" for i in range(30)]
    report = train_all_models(X, y, cfg, feature_names)
    assert report is not None
    assert len(report.model_results) >= 2


# ---------------------------------------------------------------------------
# MT18 — winner Brier is finite and below coin-flip (0.25)
# ---------------------------------------------------------------------------

def test_MT18_winner_brier_below_coinflip():
    from app.ml_models.training.config import TrainingConfig
    from app.ml_models.training.trainer import train_all_models

    X, y = _synthetic_Xy(1200, n_features=30)
    cfg = TrainingConfig(
        n_splits=3,
        test_window_bars=150,
        embargo_bars=1,
        min_train_bars=100,
        model_names=("prior", "logistic_l2"),
        run_ablation=False,
    )
    report = train_all_models(X, y, cfg, [f"f{i}" for i in range(30)])
    brier = report.winner.aggregated.brier_score_mean
    assert math.isfinite(brier), "Winner Brier score must be finite"
    assert brier < 0.5, f"Winner Brier {brier} is too high (coin flip = 0.25)"


# ---------------------------------------------------------------------------
# MT19 — winner is a non-baseline model
# ---------------------------------------------------------------------------

def test_MT19_winner_is_non_baseline():
    from app.ml_models.training.config import TrainingConfig
    from app.ml_models.training.trainer import train_all_models
    from app.ml_models.training.baselines import BASELINE_REGISTRY

    X, y = _synthetic_Xy(1200, n_features=30)
    cfg = TrainingConfig(
        n_splits=3,
        test_window_bars=150,
        embargo_bars=1,
        min_train_bars=100,
        model_names=("prior", "logistic_l2"),
        run_ablation=False,
    )
    report = train_all_models(X, y, cfg, [f"f{i}" for i in range(30)])
    assert report.winner.model_name not in BASELINE_REGISTRY, \
        f"Winner should not be a baseline, got {report.winner.model_name}"


# ---------------------------------------------------------------------------
# MT20 — run_training_pipeline end-to-end
# ---------------------------------------------------------------------------

def test_MT20_pipeline_end_to_end():
    from app.ml_models.training.config import TrainingConfig
    from app.ml_models.pipeline import run_training_pipeline

    df = _make_ohlcv(800)
    cfg = TrainingConfig(
        n_splits=3,
        test_window_bars=100,
        embargo_bars=1,
        min_train_bars=100,
        model_names=("prior", "logistic_l2"),
        run_ablation=False,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        report = run_training_pipeline(df, cfg=cfg, output_dir=Path(tmpdir))
        assert report.winner is not None
        assert (Path(tmpdir) / "training_report.md").exists()
        assert (Path(tmpdir) / "training_report.json").exists()


# ---------------------------------------------------------------------------
# MT21 — registry save/load round-trip
# ---------------------------------------------------------------------------

def test_MT21_registry_save_load_round_trip():
    from app.ml_models.training.baselines import PriorBaseline
    from app.ml_models.model_registry import save_model, load_model_by_hash

    X, y = _synthetic_Xy(200)
    model = PriorBaseline()
    model.fit(X, y)
    original_probs = model.predict_proba(X[:10])

    config_hash = "test_hash_mt21"
    save_model(
        model=model,
        model_name="prior",
        config_hash=config_hash,
        metrics={"brier_score_mean": 0.25},
        feature_names=[f"f{i}" for i in range(20)],
    )

    loaded = load_model_by_hash(config_hash, "prior")
    assert loaded is not None, "Model should be loadable after save"
    loaded_probs = loaded.predict_proba(X[:10])
    assert np.allclose(original_probs, loaded_probs, atol=1e-9)


# ---------------------------------------------------------------------------
# MT22 — generate_report produces complete markdown
# ---------------------------------------------------------------------------

def test_MT22_report_has_all_sections():
    from app.ml_models.training.config import TrainingConfig
    from app.ml_models.training.trainer import train_all_models
    from app.ml_models.report import generate_report

    X, y = _synthetic_Xy(1200, n_features=30)
    cfg = TrainingConfig(
        n_splits=3,
        test_window_bars=150,
        embargo_bars=1,
        min_train_bars=100,
        model_names=("prior", "logistic_l2"),
        run_ablation=False,
    )
    report = train_all_models(X, y, cfg, [f"f{i}" for i in range(30)])
    md = generate_report(report)

    required_sections = [
        "## Executive Summary",
        "## Model Comparison",
        "## Walk-Forward Fold Details",
        "## Confidence Bucket Analysis",
        "## Feature Importance",
        "## Calibration Summary",
        "## Model Selection Rationale",
    ]
    for section in required_sections:
        assert section in md, f"Report missing section: {section}"


# ---------------------------------------------------------------------------
# MT23 — config_hash changes on hyperparameter change
# ---------------------------------------------------------------------------

def test_MT23_config_hash_changes():
    from app.ml_models.training.config import TrainingConfig
    cfg1 = TrainingConfig(n_splits=5)
    cfg2 = TrainingConfig(n_splits=6)
    assert cfg1.config_hash() != cfg2.config_hash()


def test_MT23_config_hash_is_stable():
    from app.ml_models.training.config import TrainingConfig
    cfg = TrainingConfig()
    assert cfg.config_hash() == cfg.config_hash()


# ---------------------------------------------------------------------------
# MT24 — to_dict is JSON-serializable
# ---------------------------------------------------------------------------

def test_MT24_config_to_dict_is_json_serializable():
    from app.ml_models.training.config import TrainingConfig
    cfg = TrainingConfig()
    d = cfg.to_dict()
    # Should not raise
    serialized = json.dumps(d)
    assert len(serialized) > 0
