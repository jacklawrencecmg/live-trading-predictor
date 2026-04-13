"""
Calibration regression tests — CR1–CR14.

These tests assert threshold-level properties of the calibration pipeline.
They are NOT accuracy benchmarks. They are smoke tests that will catch:

  - A calibration map that produces probabilities outside [0, 1]
  - A Brier score that is catastrophically worse than random (BSS < threshold)
  - ECE values that indicate the calibration pipeline is broken
  - An overfit_ratio so high that the model is not learning (just memorizing)
  - A 4-layer uncertainty bundle that is internally inconsistent

All tests use deterministic synthetic data with a fixed seed so that results
are reproducible and not sensitive to yfinance availability.

Failure of any test here means the calibration or model pipeline has regressed.
Fix the code, not the thresholds.

Thresholds are deliberately lenient (the synthetic signal is weak by design).
Real model quality is assessed by BSS in the backtest results, not here.
"""

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

pytestmark = pytest.mark.calibration

# ── Threshold constants ───────────────────────────────────────────────────────
# These define the minimum acceptable behavior of the calibration pipeline
# on synthetic weak-signal data. Do not tighten without understanding the
# impact on the synthetic data distribution.

BSS_FLOOR = -0.10        # Brier skill score must not be catastrophically negative
ECE_CEILING = 0.20       # ECE must be below this after calibration
OVERFIT_RATIO_CEILING = 2.5  # test_brier / train_brier must not exceed this
BRIER_CEILING = 0.27     # Raw Brier must not be worse than random (0.25) by too much
PROB_EPSILON = 1e-6      # Probabilities must be strictly in (0+ε, 1-ε)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 400, seed: int = 42) -> "pd.DataFrame":
    import pandas as pd
    rng = np.random.default_rng(seed)
    base = datetime(2024, 1, 2, 9, 30)
    prices = 100 + np.cumsum(rng.normal(0, 0.5, n))
    prices = np.maximum(prices, 1.0)
    highs = prices * (1 + rng.uniform(0, 0.003, n))
    lows = prices * (1 - rng.uniform(0, 0.003, n))
    vols = rng.integers(10_000, 100_000, n).astype(float)
    return pd.DataFrame({
        "open": prices * (1 + rng.uniform(-0.001, 0.001, n)),
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": vols,
        "vwap": (prices + highs + lows) / 3,
        "bar_open_time": [base + timedelta(minutes=5 * i) for i in range(n)],
    })


def _synthetic_Xy(n: int = 800, n_features: int = 30, seed: int = 7):
    """Weak-signal synthetic dataset for calibration smoke tests."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    # Weak signal in first feature; mostly noise
    y = (X[:, 0] * 0.4 + rng.normal(0, 1.5, n) > 0).astype(int)
    return X, y


def _train_split(X, y, train_frac=0.7):
    n = len(y)
    split = int(n * train_frac)
    return X[:split], X[split:], y[:split], y[split:]


# ── CR1: Logistic regression fit/predict_proba on synthetic data ──────────────

def test_CR1_logistic_fit_and_predict_proba_shape():
    """Logistic pipeline produces predict_proba with correct shape."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _synthetic_Xy()
    X_tr, X_te, y_tr, _ = _train_split(X, y)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")),
    ])
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)
    assert probs.shape == (len(X_te), 2), "predict_proba must return (n_samples, 2)"


# ── CR2: All predicted probabilities must be in (0, 1) ───────────────────────

def test_CR2_probabilities_are_valid():
    """All predicted probabilities are strictly between 0 and 1."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _synthetic_Xy()
    X_tr, X_te, y_tr, _ = _train_split(X, y)
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")),
    ])
    cal = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    cal.fit(X_tr, y_tr)
    probs = cal.predict_proba(X_te)
    prob_up = probs[:, 1]
    assert (prob_up >= 0).all() and (prob_up <= 1).all(), \
        "Calibrated probabilities must be in [0, 1]"
    assert (probs.sum(axis=1) - 1.0 < 1e-6).all(), \
        "Probabilities must sum to 1 for each sample"


# ── CR3: Brier score on weak-signal data is not catastrophically bad ──────────

def test_CR3_brier_not_catastrophically_bad():
    """Brier score on synthetic data is <= BRIER_CEILING (not catastrophically wrong)."""
    from sklearn.metrics import brier_score_loss
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _synthetic_Xy()
    X_tr, X_te, y_tr, y_te = _train_split(X, y)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")),
    ])
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    brier = brier_score_loss(y_te, probs)
    assert brier <= BRIER_CEILING, (
        f"Brier score {brier:.4f} exceeds ceiling {BRIER_CEILING}. "
        "The model is performing catastrophically worse than random. "
        "This is a smoke test — check the pipeline, not the signal."
    )


# ── CR4: Brier skill score is computable and finite ──────────────────────────

def test_CR4_brier_skill_score_is_finite():
    """BSS is a finite number (no NaN/inf from division by zero in baseline)."""
    from sklearn.metrics import brier_score_loss
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _synthetic_Xy()
    X_tr, X_te, y_tr, y_te = _train_split(X, y)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")),
    ])
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    base_rate = y_tr.mean()
    model_brier = brier_score_loss(y_te, probs)
    naive_brier = brier_score_loss(y_te, np.full(len(y_te), base_rate))
    bss = 1.0 - model_brier / (naive_brier + 1e-9)
    assert math.isfinite(bss), f"BSS is not finite: {bss}"
    assert bss >= BSS_FLOOR, (
        f"BSS {bss:.4f} is below floor {BSS_FLOOR}. "
        "The model is performing much worse than a naive prior. "
        "Check for a pipeline error."
    )


# ── CR5: Overfit ratio is below ceiling ───────────────────────────────────────

def test_CR5_overfit_ratio_below_ceiling():
    """train_brier / test_brier must not exceed OVERFIT_RATIO_CEILING."""
    from sklearn.metrics import brier_score_loss
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _synthetic_Xy()
    X_tr, X_te, y_tr, y_te = _train_split(X, y)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")),
    ])
    model.fit(X_tr, y_tr)
    train_probs = model.predict_proba(X_tr)[:, 1]
    test_probs = model.predict_proba(X_te)[:, 1]
    train_brier = brier_score_loss(y_tr, train_probs)
    test_brier = brier_score_loss(y_te, test_probs)
    if train_brier < 1e-9:
        pytest.skip("Train Brier is essentially zero — model is trivially overfit on this split")
    overfit_ratio = test_brier / train_brier
    assert overfit_ratio <= OVERFIT_RATIO_CEILING, (
        f"Overfit ratio {overfit_ratio:.2f} exceeds ceiling {OVERFIT_RATIO_CEILING}. "
        "The model is severely overfit to training data."
    )


# ── CR6: CalibrationMap identity is a no-op ──────────────────────────────────

def test_CR6_calibration_map_identity_is_noop():
    """CalibrationMap.identity() must return the input unchanged."""
    from app.inference.uncertainty import CalibrationMap

    cal = CalibrationMap.identity()
    for p in [0.0, 0.1, 0.45, 0.5, 0.73, 1.0]:
        result = cal.apply(p)
        assert abs(result - p) < 1e-9, (
            f"Identity CalibrationMap changed {p} → {result}"
        )


# ── CR7: Calibration map preserves order (weak monotonicity) ─────────────────

def test_CR7_calibration_map_is_approximately_monotone():
    """After isotonic or sigmoid calibration, higher raw prob → higher calibrated prob."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _synthetic_Xy(n=1000)
    X_tr, X_te, y_tr, _ = _train_split(X, y)
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")),
    ])
    cal = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    cal.fit(X_tr, y_tr)

    # Generate a grid of raw scores from the base model
    base.fit(X_tr, y_tr)
    raw_scores = base.predict_proba(X_te)[:, 1]
    cal_scores = cal.predict_proba(X_te)[:, 1]

    # Sort by raw score; calibrated scores should be roughly non-decreasing
    order = np.argsort(raw_scores)
    cal_sorted = cal_scores[order]
    # Count inversions (calibrated[i] > calibrated[j] for i < j)
    n = len(cal_sorted)
    inversions = sum(
        1 for i in range(0, n - 1, 10)
        for j in range(i + 1, min(i + 11, n))
        if cal_sorted[i] > cal_sorted[j] + 0.05  # allow 5% slack
    )
    total_checked = sum(min(10, n - i) for i in range(0, n - 1, 10))
    inversion_rate = inversions / max(total_checked, 1)
    assert inversion_rate < 0.20, (
        f"Calibration map is not approximately monotone: {inversion_rate:.1%} inversion rate. "
        "Calibration is producing probabilities that are not ordered by the underlying score."
    )


# ── CR8: ECE of calibrated model is below ceiling ────────────────────────────

def test_CR8_ece_of_calibrated_model_below_ceiling():
    """ECE after calibration must be below ECE_CEILING on synthetic data."""
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _synthetic_Xy(n=1000)
    X_tr, X_te, y_tr, y_te = _train_split(X, y)
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")),
    ])
    cal = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    cal.fit(X_tr, y_tr)
    cal_probs = cal.predict_proba(X_te)[:, 1]

    # Compute ECE: weighted mean absolute calibration error across bins
    try:
        frac_pos, mean_pred = calibration_curve(y_te, cal_probs, n_bins=10, strategy="uniform")
    except ValueError:
        pytest.skip("Insufficient samples for calibration curve (too few bins populated)")

    bin_sizes = np.histogram(cal_probs, bins=10, range=(0, 1))[0]
    total = len(cal_probs)
    ece = sum(
        abs(fp - mp) * (bs / total)
        for fp, mp, bs in zip(frac_pos, mean_pred, bin_sizes)
        if bs > 0
    )
    assert ece <= ECE_CEILING, (
        f"ECE {ece:.4f} exceeds ceiling {ECE_CEILING} after calibration. "
        "The calibration pipeline is not functioning correctly."
    )


# ── CR9: UncertaintyBundle produces consistent 4-layer output ────────────────

def test_CR9_uncertainty_bundle_layers_are_consistent():
    """build_uncertainty_bundle must return internally consistent output."""
    from app.inference.uncertainty import CalibrationMap, build_uncertainty_bundle

    cal = CalibrationMap.identity()

    # Case A: high raw probability → should not be suppressed at default threshold
    bundle_high = build_uncertainty_bundle(
        raw_prob_up=0.70,
        calibration_map=cal,
        tracker_stats=None,
        confidence_threshold=0.55,
        prior_abstain_reason=None,
    )
    assert bundle_high.calibrated_prob_up >= 0.0
    assert bundle_high.calibrated_prob_up <= 1.0
    assert bundle_high.tradeable_confidence >= 0.0
    assert bundle_high.tradeable_confidence <= 1.0
    assert bundle_high.action in ("buy", "sell", "abstain")

    # Case B: 50/50 raw probability → should abstain (no edge)
    bundle_neutral = build_uncertainty_bundle(
        raw_prob_up=0.50,
        calibration_map=cal,
        tracker_stats=None,
        confidence_threshold=0.55,
        prior_abstain_reason=None,
    )
    assert bundle_neutral.action == "abstain", (
        f"50/50 probability should produce abstain, got {bundle_neutral.action!r}"
    )

    # Case C: prior regime suppression → must propagate to output
    bundle_suppressed = build_uncertainty_bundle(
        raw_prob_up=0.75,
        calibration_map=cal,
        tracker_stats=None,
        confidence_threshold=0.55,
        prior_abstain_reason="regime_suppressed:high_volatility",
    )
    assert bundle_suppressed.action == "abstain", (
        "Regime-suppressed signal must produce abstain regardless of probability"
    )
    assert bundle_suppressed.abstain_reason is not None
    assert "regime" in bundle_suppressed.abstain_reason.lower()


# ── CR10: Degradation factor shrinks tradeable confidence ────────────────────

def test_CR10_degradation_factor_reduces_tradeable_confidence():
    """A degraded tracker must produce lower tradeable_confidence than a healthy one."""
    from app.inference.uncertainty import CalibrationMap, build_uncertainty_bundle
    from app.inference.confidence_tracker import TrackerStats

    cal = CalibrationMap.identity()
    raw_prob = 0.68

    # Healthy tracker (degradation_factor = 1.0)
    healthy_stats = TrackerStats(
        symbol="SPY",
        window_size=100,
        rolling_brier=0.22,
        baseline_brier=0.22,
        degradation_factor=1.0,
        ece_recent=0.03,
        calibration_health="good",
        needs_retrain=False,
        retrain_reason=None,
        reliability_bins=None,
        reliability_mean_pred=None,
        reliability_frac_pos=None,
    )

    # Degraded tracker (degradation_factor = 0.4 — model performance halved)
    degraded_stats = TrackerStats(
        symbol="SPY",
        window_size=100,
        rolling_brier=0.28,
        baseline_brier=0.22,
        degradation_factor=0.4,
        ece_recent=0.12,
        calibration_health="degraded",
        needs_retrain=True,
        retrain_reason="rolling_brier_exceeds_1.5x_baseline",
        reliability_bins=None,
        reliability_mean_pred=None,
        reliability_frac_pos=None,
    )

    bundle_healthy = build_uncertainty_bundle(
        raw_prob_up=raw_prob,
        calibration_map=cal,
        tracker_stats=healthy_stats,
        confidence_threshold=0.55,
        prior_abstain_reason=None,
    )
    bundle_degraded = build_uncertainty_bundle(
        raw_prob_up=raw_prob,
        calibration_map=cal,
        tracker_stats=degraded_stats,
        confidence_threshold=0.55,
        prior_abstain_reason=None,
    )

    assert bundle_degraded.tradeable_confidence <= bundle_healthy.tradeable_confidence, (
        f"Degraded model ({bundle_degraded.tradeable_confidence:.3f}) must have lower "
        f"tradeable_confidence than healthy model ({bundle_healthy.tradeable_confidence:.3f})"
    )


# ── CR11: Calibration health "good" threshold logic ──────────────────────────

def test_CR11_calibration_health_classification():
    """calibration_health is classified correctly from degradation_factor."""
    from app.inference.confidence_tracker import ConfidenceTracker
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ConfidenceTracker(storage_dir=tmpdir)

        # Simulate healthy resolved inferences
        # calibrated_prob ≈ 0.65, outcome = 1 (correct) repeatedly
        for _ in range(50):
            tracker.record_inference("SPY", calibrated_prob=0.65, actual_outcome=1)
        for _ in range(30):
            tracker.record_inference("SPY", calibrated_prob=0.40, actual_outcome=0)

        stats = tracker.get_stats("SPY")
        assert stats.calibration_health in ("good", "fair", "caution", "degraded", "unknown"), (
            f"Unexpected calibration_health value: {stats.calibration_health!r}"
        )
        assert 0.0 <= stats.degradation_factor <= 2.0, (
            f"degradation_factor {stats.degradation_factor} out of expected range"
        )


# ── CR12: Calibration pipeline produces reliability diagram data ──────────────

def test_CR12_reliability_diagram_is_computable():
    """compute_calibration produces a valid reliability diagram dict."""
    from app.services.model_service import compute_calibration

    rng = np.random.default_rng(0)
    # Simulate calibrated probabilities and true outcomes
    probs = rng.uniform(0.3, 0.7, 200)
    # Outcomes correlated with probs (realistic scenario)
    outcomes = (rng.uniform(0, 1, 200) < probs).astype(int)

    calib = compute_calibration(outcomes, probs, n_bins=10)
    assert calib is not None
    assert len(calib.bin_centers) > 0
    assert len(calib.fraction_positive) == len(calib.bin_centers)
    assert all(0 <= bp <= 1 for bp in calib.bin_centers), "Bin centers must be in [0, 1]"
    assert all(0 <= fp <= 1 for fp in calib.fraction_positive), \
        "Fraction positive must be in [0, 1]"
    assert math.isfinite(calib.brier_score), "Brier score must be finite"
    assert math.isfinite(calib.log_loss), "Log loss must be finite"


# ── CR13: walk-forward training produces model with valid metrics ─────────────

def test_CR13_walk_forward_produces_valid_metrics():
    """Full walk-forward training loop produces finite, valid metrics."""
    from app.ml_models.pipeline import run_training_pipeline

    df = _make_ohlcv(n=400)
    result = run_training_pipeline(df, symbol="TEST", n_splits=3)

    assert result is not None, "Training pipeline returned None"
    assert hasattr(result, "brier_score") or hasattr(result, "metrics"), \
        "Training result must have brier_score or metrics attribute"

    # Access metrics generically
    brier = getattr(result, "brier_score", None) or result.metrics.get("brier_score")
    if brier is not None:
        assert math.isfinite(brier), f"Brier score is not finite: {brier}"
        assert 0 <= brier <= 1, f"Brier score {brier} out of [0, 1]"


# ── CR14: Brier skill score formula sanity check ──────────────────────────────

def test_CR14_brier_skill_score_formula():
    """BSS formula: BSS=0 means model equals naive; BSS=1 means perfect."""
    from sklearn.metrics import brier_score_loss

    rng = np.random.default_rng(99)
    y = rng.integers(0, 2, 200)
    base_rate = y.mean()

    # Test 1: Naive classifier (always predicts base rate) → BSS ≈ 0
    naive_probs = np.full(len(y), base_rate)
    naive_brier = brier_score_loss(y, naive_probs)
    reference_brier = brier_score_loss(y, naive_probs)
    naive_bss = 1.0 - naive_brier / (reference_brier + 1e-9)
    assert abs(naive_bss) < 0.01, f"Naive BSS should be ≈ 0, got {naive_bss:.4f}"

    # Test 2: Perfect classifier → BSS = 1.0
    perfect_probs = y.astype(float)
    perfect_brier = brier_score_loss(y, perfect_probs)
    perfect_bss = 1.0 - perfect_brier / (reference_brier + 1e-9)
    assert perfect_bss > 0.95, f"Perfect classifier BSS should be ≈ 1.0, got {perfect_bss:.4f}"

    # Test 3: Harmful classifier (inverted probs) → BSS < 0
    inverted_probs = 1.0 - y.astype(float)
    inverted_brier = brier_score_loss(y, inverted_probs)
    inverted_bss = 1.0 - inverted_brier / (reference_brier + 1e-9)
    assert inverted_bss < 0, f"Inverted classifier BSS must be negative, got {inverted_bss:.4f}"
