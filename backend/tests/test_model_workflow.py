"""
Model development workflow tests.

Coverage:
  MW1  — PersistenceBaseline binary: always outputs (n, 2), sums to 1, always 0.5
  MW2  — PersistenceBaseline ternary: always outputs (n, 3), all mass on FLAT
  MW3  — PersistenceBaseline detects n_classes from y in fit()
  MW4  — VolatilityNoTradeBaseline binary: high-vol → 0.5/0.5
  MW5  — VolatilityNoTradeBaseline ternary: high-vol → all mass on FLAT
  MW6  — VolatilityNoTradeBaseline threshold learned from training data
  MW7  — VolatilityNoTradeBaseline low-vol rows use class priors
  MW8  — _get_prob_up binary (returns proba[:, 1])
  MW9  — _get_prob_up ternary (returns proba[:, 2])
  MW10 — _binarize_for_eval binary (identity)
  MW11 — _binarize_for_eval ternary (UP-vs-rest)
  MW12 — train_all_models ternary labels runs end-to-end
  MW13 — ternary winner Brier is finite
  MW14 — pooled_oos_y and pooled_oos_prob_up populated after training
  MW15 — pooled_oos_y is binary {0, 1} for ternary training
  MW16 — embargo_bars set to horizon in TrainingConfig
  MW17 — label_type field defaults to "binary"
  MW18 — config_hash differs for binary vs ternary label_type
  MW19 — new baselines in BASELINE_REGISTRY
  MW20 — new baselines API compliance (fit/predict_proba/predict)
  MW21 — _save_prediction_artifacts writes CSVs with correct columns
  MW22 — run_multi_horizon_pipeline returns dict keyed by horizon
  MW23 — per-horizon embargo matches horizon value
  MW24 — generate_multi_horizon_report contains required sections
  MW25 — prediction artifacts CSV round-trip (write + read)
"""

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

def _make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = datetime(2024, 1, 2, 9, 30)
    prices = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    prices = np.clip(prices, 1.0, None)
    opens = np.clip(prices + rng.normal(0, 0.2, n), 0.01, None)
    oc_high = np.maximum(opens, prices)
    oc_low = np.minimum(opens, prices)
    highs = oc_high + rng.uniform(0.1, 0.5, n)
    lows = np.clip(oc_low - rng.uniform(0.1, 0.5, n), 0.01, None)
    volumes = rng.integers(10_000, 100_000, n).astype(float)
    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
        "vwap": (prices + highs + lows) / 3,
        "bar_open_time": [base + timedelta(minutes=5 * i) for i in range(n)],
    })


def _ternary_y(n: int = 800, seed: int = 0) -> tuple:
    """Synthetic ternary labels and feature matrix."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 30))
    # Balanced ternary labels with weak signal
    raw = X[:, 0] + rng.normal(0, 1.5, n)
    y = np.where(raw > 0.5, 2, np.where(raw < -0.5, 0, 1)).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# MW1 — PersistenceBaseline binary
# ---------------------------------------------------------------------------

def test_MW1_persistence_binary_shape_and_values():
    from app.ml_models.training.baselines import PersistenceBaseline
    X = np.zeros((100, 5))
    y = np.array([0, 1] * 50)
    m = PersistenceBaseline()
    m.fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (100, 2)
    assert np.allclose(p, 0.5), "Binary persistence must always predict 0.5/0.5"
    assert np.allclose(p.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# MW2 — PersistenceBaseline ternary
# ---------------------------------------------------------------------------

def test_MW2_persistence_ternary_all_flat():
    from app.ml_models.training.baselines import PersistenceBaseline
    X = np.zeros((100, 5))
    y = np.array([0, 1, 2] * 33 + [0])
    m = PersistenceBaseline()
    m.fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (100, 3)
    assert np.allclose(p[:, 1], 1.0), "Ternary persistence must predict P(FLAT)=1"
    assert np.allclose(p[:, 0], 0.0) and np.allclose(p[:, 2], 0.0)
    assert np.allclose(p.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# MW3 — PersistenceBaseline detects n_classes
# ---------------------------------------------------------------------------

def test_MW3_persistence_detects_n_classes():
    from app.ml_models.training.baselines import PersistenceBaseline
    m2 = PersistenceBaseline()
    m2.fit(np.zeros((10, 2)), np.array([0, 1] * 5))
    assert m2.n_classes_ == 2

    m3 = PersistenceBaseline()
    m3.fit(np.zeros((9, 2)), np.array([0, 1, 2] * 3))
    assert m3.n_classes_ == 3


# ---------------------------------------------------------------------------
# MW4 — VolatilityNoTradeBaseline binary: high-vol → 0.5/0.5
# ---------------------------------------------------------------------------

def test_MW4_vol_no_trade_binary_high_vol_neutral():
    from app.ml_models.training.baselines import VolatilityNoTradeBaseline
    n = 100
    X_tr = np.zeros((n, 10))
    X_tr[:, 7] = np.linspace(0.001, 0.05, n)   # increasing volatility
    y_tr = np.array([0, 1] * (n // 2))

    m = VolatilityNoTradeBaseline(vol_feature_idx=7, vol_threshold_pct=80.0)
    m.fit(X_tr, y_tr)

    # High-vol test rows (above 80th pct of training)
    X_high = np.zeros((10, 10))
    X_high[:, 7] = 0.10   # well above threshold
    p = m.predict_proba(X_high)
    assert p.shape == (10, 2)
    assert np.allclose(p, 0.5), "High-vol binary should predict 0.5/0.5"


# ---------------------------------------------------------------------------
# MW5 — VolatilityNoTradeBaseline ternary: high-vol → P(FLAT)=1
# ---------------------------------------------------------------------------

def test_MW5_vol_no_trade_ternary_high_vol_flat():
    from app.ml_models.training.baselines import VolatilityNoTradeBaseline
    n = 120
    X_tr = np.zeros((n, 10))
    X_tr[:, 7] = np.linspace(0.001, 0.05, n)
    y_tr = np.array([0, 1, 2] * (n // 3))

    m = VolatilityNoTradeBaseline(vol_feature_idx=7, vol_threshold_pct=80.0)
    m.fit(X_tr, y_tr)

    X_high = np.zeros((10, 10))
    X_high[:, 7] = 0.10
    p = m.predict_proba(X_high)
    assert p.shape == (10, 3)
    assert np.allclose(p[:, 1], 1.0), "High-vol ternary should predict P(FLAT)=1"
    assert np.allclose(p[:, 0], 0.0) and np.allclose(p[:, 2], 0.0)


# ---------------------------------------------------------------------------
# MW6 — VolatilityNoTradeBaseline threshold learned from training data
# ---------------------------------------------------------------------------

def test_MW6_vol_no_trade_threshold_from_training():
    from app.ml_models.training.baselines import VolatilityNoTradeBaseline
    rng = np.random.default_rng(7)
    X_tr = np.zeros((200, 10))
    X_tr[:, 7] = rng.uniform(0, 1, 200)
    y_tr = rng.integers(0, 2, 200)

    m = VolatilityNoTradeBaseline(vol_feature_idx=7, vol_threshold_pct=80.0)
    m.fit(X_tr, y_tr)

    expected = float(np.percentile(X_tr[:, 7], 80.0))
    assert abs(m._threshold - expected) < 1e-9


# ---------------------------------------------------------------------------
# MW7 — VolatilityNoTradeBaseline low-vol rows use class priors
# ---------------------------------------------------------------------------

def test_MW7_vol_no_trade_low_vol_uses_priors():
    from app.ml_models.training.baselines import VolatilityNoTradeBaseline
    n = 100
    X_tr = np.zeros((n, 10))
    X_tr[:, 7] = 0.02   # fixed low vol in training
    y_tr = np.array([1] * 70 + [0] * 30)  # 70% class-1

    m = VolatilityNoTradeBaseline(vol_feature_idx=7, vol_threshold_pct=80.0)
    m.fit(X_tr, y_tr)

    # Low-vol test rows (below threshold)
    X_low = np.zeros((10, 10))
    X_low[:, 7] = 0.001
    p = m.predict_proba(X_low)
    assert abs(p[0, 1] - 0.70) < 0.01, f"Low-vol should use prior P(1)≈0.70, got {p[0,1]}"


# ---------------------------------------------------------------------------
# MW8 — _get_prob_up binary
# ---------------------------------------------------------------------------

def test_MW8_get_prob_up_binary():
    from app.ml_models.training.trainer import _get_prob_up
    from app.ml_models.training.baselines import PriorBaseline
    X = np.zeros((50, 5))
    y = np.array([0, 1] * 25)
    m = PriorBaseline()
    m.fit(X, y)
    prob = _get_prob_up(m, X)
    assert prob.shape == (50,)
    # PriorBaseline with 50/50 → P(up) = 0.5
    assert np.allclose(prob, 0.5)


# ---------------------------------------------------------------------------
# MW9 — _get_prob_up ternary
# ---------------------------------------------------------------------------

def test_MW9_get_prob_up_ternary():
    from app.ml_models.training.trainer import _get_prob_up
    from app.ml_models.training.baselines import PersistenceBaseline
    X = np.zeros((50, 5))
    y = np.array([0, 1, 2] * 16 + [0, 1])
    m = PersistenceBaseline()
    m.fit(X, y)
    prob = _get_prob_up(m, X)
    assert prob.shape == (50,)
    # PersistenceBaseline ternary always predicts P(FLAT)=1, so P(UP)=0
    assert np.allclose(prob, 0.0)


# ---------------------------------------------------------------------------
# MW10 — _binarize_for_eval binary
# ---------------------------------------------------------------------------

def test_MW10_binarize_binary_is_identity():
    from app.ml_models.training.trainer import _binarize_for_eval
    y = np.array([0, 1, 0, 1, 1])
    result = _binarize_for_eval(y, "binary")
    assert np.array_equal(result, y.astype(int))


# ---------------------------------------------------------------------------
# MW11 — _binarize_for_eval ternary
# ---------------------------------------------------------------------------

def test_MW11_binarize_ternary_up_vs_rest():
    from app.ml_models.training.trainer import _binarize_for_eval
    y = np.array([0, 1, 2, 0, 2, 1])
    result = _binarize_for_eval(y, "ternary")
    expected = np.array([0, 0, 1, 0, 1, 0])
    assert np.array_equal(result, expected)


# ---------------------------------------------------------------------------
# MW12 — train_all_models ternary runs end-to-end
# ---------------------------------------------------------------------------

def test_MW12_train_all_models_ternary_runs():
    from app.ml_models.training.config import TrainingConfig
    from app.ml_models.training.trainer import train_all_models

    X, y = _ternary_y(900)
    cfg = TrainingConfig(
        n_splits=3,
        test_window_bars=150,
        embargo_bars=3,
        min_train_bars=100,
        label_type="ternary",
        model_names=("persistence", "logistic_l2"),
        run_ablation=False,
    )
    report = train_all_models(X, y, cfg, [f"f{i}" for i in range(30)])
    assert report is not None
    assert len(report.model_results) >= 1


# ---------------------------------------------------------------------------
# MW13 — ternary winner Brier is finite
# ---------------------------------------------------------------------------

def test_MW13_ternary_winner_brier_finite():
    from app.ml_models.training.config import TrainingConfig
    from app.ml_models.training.trainer import train_all_models

    X, y = _ternary_y(900)
    cfg = TrainingConfig(
        n_splits=3,
        test_window_bars=150,
        embargo_bars=3,
        min_train_bars=100,
        label_type="ternary",
        model_names=("persistence", "logistic_l2"),
        run_ablation=False,
    )
    report = train_all_models(X, y, cfg, [f"f{i}" for i in range(30)])
    brier = report.winner.aggregated.brier_score_mean
    assert math.isfinite(brier), f"Winner Brier must be finite, got {brier}"
    assert brier <= 0.5, f"Winner Brier {brier} is unexpectedly high"


# ---------------------------------------------------------------------------
# MW14 — pooled_oos arrays populated after training
# ---------------------------------------------------------------------------

def test_MW14_pooled_oos_arrays_populated():
    from app.ml_models.training.config import TrainingConfig
    from app.ml_models.training.trainer import train_all_models

    X, y = _ternary_y(900)
    cfg = TrainingConfig(
        n_splits=2,
        test_window_bars=200,
        embargo_bars=1,
        min_train_bars=100,
        label_type="ternary",
        model_names=("persistence", "logistic_l2"),
        run_ablation=False,
    )
    report = train_all_models(X, y, cfg, [f"f{i}" for i in range(30)])
    for result in report.model_results:
        assert result.pooled_oos_y is not None, f"{result.model_name}: pooled_oos_y is None"
        assert result.pooled_oos_prob_up is not None, f"{result.model_name}: pooled_oos_prob_up is None"
        assert len(result.pooled_oos_y) == len(result.pooled_oos_prob_up)
        assert len(result.pooled_oos_y) > 0


# ---------------------------------------------------------------------------
# MW15 — pooled_oos_y is binary {0, 1} for ternary training
# ---------------------------------------------------------------------------

def test_MW15_pooled_oos_y_is_binary_for_ternary():
    from app.ml_models.training.config import TrainingConfig
    from app.ml_models.training.trainer import train_all_models

    X, y = _ternary_y(900)
    cfg = TrainingConfig(
        n_splits=2,
        test_window_bars=200,
        embargo_bars=1,
        min_train_bars=100,
        label_type="ternary",
        model_names=("persistence", "logistic_l2"),
        run_ablation=False,
    )
    report = train_all_models(X, y, cfg, [f"f{i}" for i in range(30)])
    for result in report.model_results:
        unique_vals = set(result.pooled_oos_y.tolist())
        assert unique_vals <= {0, 1}, (
            f"{result.model_name}: pooled_oos_y contains non-binary values {unique_vals}"
        )


# ---------------------------------------------------------------------------
# MW16 — embargo_bars set to horizon in TrainingConfig
# ---------------------------------------------------------------------------

def test_MW16_embargo_bars_matches_horizon():
    from app.ml_models.training.config import TrainingConfig
    for h in (1, 3, 5):
        cfg = TrainingConfig(embargo_bars=h, label_type="ternary")
        assert cfg.embargo_bars == h


# ---------------------------------------------------------------------------
# MW17 — label_type defaults to "binary"
# ---------------------------------------------------------------------------

def test_MW17_label_type_defaults_binary():
    from app.ml_models.training.config import TrainingConfig
    cfg = TrainingConfig()
    assert cfg.label_type == "binary"


# ---------------------------------------------------------------------------
# MW18 — config_hash differs for binary vs ternary
# ---------------------------------------------------------------------------

def test_MW18_config_hash_differs_binary_vs_ternary():
    from app.ml_models.training.config import TrainingConfig
    binary_cfg = TrainingConfig(label_type="binary")
    ternary_cfg = TrainingConfig(label_type="ternary")
    assert binary_cfg.config_hash() != ternary_cfg.config_hash()


# ---------------------------------------------------------------------------
# MW19 — new baselines in BASELINE_REGISTRY
# ---------------------------------------------------------------------------

def test_MW19_new_baselines_in_registry():
    from app.ml_models.training.baselines import BASELINE_REGISTRY
    assert "persistence" in BASELINE_REGISTRY
    assert "vol_no_trade" in BASELINE_REGISTRY


# ---------------------------------------------------------------------------
# MW20 — new baselines API compliance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["persistence", "vol_no_trade"])
def test_MW20_new_baseline_api_binary(name):
    from app.ml_models.training.baselines import BASELINE_REGISTRY
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 30))
    y = rng.integers(0, 2, 100)
    m = BASELINE_REGISTRY[name]()
    m.fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (100, 2), f"{name}: expected (100, 2) got {p.shape}"
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert (p >= 0).all() and (p <= 1).all()
    pred = m.predict(X)
    assert pred.shape == (100,)


@pytest.mark.parametrize("name", ["persistence", "vol_no_trade"])
def test_MW20_new_baseline_api_ternary(name):
    from app.ml_models.training.baselines import BASELINE_REGISTRY
    rng = np.random.default_rng(1)
    X = rng.standard_normal((120, 30))
    y = np.array([0, 1, 2] * 40)
    m = BASELINE_REGISTRY[name]()
    m.fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (120, 3), f"{name}: expected (120, 3) got {p.shape}"
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert (p >= 0).all() and (p <= 1).all()


# ---------------------------------------------------------------------------
# MW21 — _save_prediction_artifacts writes CSVs
# ---------------------------------------------------------------------------

def test_MW21_save_prediction_artifacts_writes_csvs():
    from app.ml_models.training.config import TrainingConfig
    from app.ml_models.training.trainer import train_all_models
    from app.ml_models.pipeline import _save_prediction_artifacts

    X, y = _ternary_y(900)
    cfg = TrainingConfig(
        n_splits=2,
        test_window_bars=200,
        embargo_bars=1,
        min_train_bars=100,
        label_type="ternary",
        model_names=("persistence", "logistic_l2"),
        run_ablation=False,
    )
    report = train_all_models(X, y, cfg, [f"f{i}" for i in range(30)])

    with tempfile.TemporaryDirectory() as tmpdir:
        _save_prediction_artifacts(report, horizon=3, output_dir=Path(tmpdir), symbol="TEST")
        csvs = list(Path(tmpdir).glob("*.csv"))
        assert len(csvs) == len(report.model_results), "Expected one CSV per model"
        for csv_path in csvs:
            df = pd.read_csv(csv_path)
            assert "y_true" in df.columns
            assert "prob_up" in df.columns
            assert len(df) > 0


# ---------------------------------------------------------------------------
# MW22 — run_multi_horizon_pipeline returns dict keyed by horizon
# ---------------------------------------------------------------------------

def test_MW22_multi_horizon_pipeline_returns_dict():
    from app.ml_models.training.config import TrainingConfig
    from app.ml_models.pipeline import run_multi_horizon_pipeline

    df = _make_ohlcv(800)
    small_cfg = TrainingConfig(
        n_splits=2,
        test_window_bars=100,
        min_train_bars=100,
        model_names=("persistence", "logistic_l2"),
        run_ablation=False,
    )
    reports = run_multi_horizon_pipeline(df, horizons=(1, 3), symbol="TEST", base_cfg=small_cfg)
    assert isinstance(reports, dict)
    assert 1 in reports
    assert 3 in reports
    for h, report in reports.items():
        assert report.winner is not None


# ---------------------------------------------------------------------------
# MW23 — per-horizon embargo matches horizon value
# ---------------------------------------------------------------------------

def test_MW23_per_horizon_embargo_equals_horizon():
    from app.ml_models.training.config import TrainingConfig
    from app.ml_models.pipeline import run_multi_horizon_pipeline

    df = _make_ohlcv(800)
    small_cfg = TrainingConfig(
        n_splits=2,
        test_window_bars=100,
        min_train_bars=100,
        model_names=("persistence", "logistic_l2"),
        run_ablation=False,
    )
    reports = run_multi_horizon_pipeline(df, horizons=(1, 3), symbol="TEST", base_cfg=small_cfg)
    for h, report in reports.items():
        assert report.config.embargo_bars == h, (
            f"h={h}: expected embargo_bars={h}, got {report.config.embargo_bars}"
        )


# ---------------------------------------------------------------------------
# MW24 — generate_multi_horizon_report contains required sections
# ---------------------------------------------------------------------------

def test_MW24_multi_horizon_report_sections():
    from app.ml_models.training.config import TrainingConfig
    from app.ml_models.pipeline import run_multi_horizon_pipeline
    from app.ml_models.report import generate_multi_horizon_report

    df = _make_ohlcv(800)
    small_cfg = TrainingConfig(
        n_splits=2,
        test_window_bars=100,
        min_train_bars=100,
        model_names=("persistence", "logistic_l2"),
        run_ablation=False,
    )
    reports = run_multi_horizon_pipeline(df, horizons=(1, 3), symbol="TEST", base_cfg=small_cfg)
    md = generate_multi_horizon_report(reports)

    for section in [
        "## Winner Summary",
        "## Per-Horizon Model Comparison",
        "## Baseline Beat Summary",
    ]:
        assert section in md, f"Missing section: {section}"

    # Should mention each trained horizon
    for h in reports:
        assert f"h={h}" in md, f"Horizon h={h} not mentioned in report"


# ---------------------------------------------------------------------------
# MW25 — prediction artifact CSV round-trip
# ---------------------------------------------------------------------------

def test_MW25_prediction_artifact_csv_round_trip():
    from app.ml_models.training.config import TrainingConfig
    from app.ml_models.training.trainer import train_all_models
    from app.ml_models.pipeline import _save_prediction_artifacts

    X, y = _ternary_y(900)
    cfg = TrainingConfig(
        n_splits=2,
        test_window_bars=200,
        embargo_bars=1,
        min_train_bars=100,
        label_type="ternary",
        model_names=("persistence", "logistic_l2"),
        run_ablation=False,
    )
    report = train_all_models(X, y, cfg, [f"f{i}" for i in range(30)])
    # Use first model result that has OOS data
    result = next(r for r in report.model_results if r.pooled_oos_y is not None)

    with tempfile.TemporaryDirectory() as tmpdir:
        _save_prediction_artifacts(report, horizon=1, output_dir=Path(tmpdir), symbol="SYM")
        csvs = list(Path(tmpdir).glob("*.csv"))
        assert csvs, "No CSVs written"
        # Find the CSV for this specific model
        model_csv = next(
            (p for p in csvs if result.model_name in p.name), csvs[0]
        )
        df_loaded = pd.read_csv(model_csv)

        # Values must match the stored arrays exactly
        assert np.array_equal(df_loaded["y_true"].values, result.pooled_oos_y)
        assert np.allclose(df_loaded["prob_up"].values, result.pooled_oos_prob_up)
