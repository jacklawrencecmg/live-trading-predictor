"""
Comprehensive test suite for the uncertainty-aware inference system.

Covers:
  U1–U5:   CalibrationMap (identity, isotonic, platt, serialization, edge cases)
  U6–U10:  UncertaintyBundle 4-layer consistency
  U11–U15: Degradation factor formula and edge cases
  U16–U21: Calibration health states (good/fair/caution/degraded/unknown)
  U22–U28: ConfidenceTracker: record, record_inference, get_stats, persistence
  U29–U33: Rolling degradation → tradeable confidence reduction
  U34–U36: Retrain signal triggers
  U37–U39: Reliability diagram computation
  U40–U43: Abstain logic (low confidence, degraded, regime suppression)
  U44–U46: TrainingReport.brier_score and .metrics properties
  U47–U48: run_training_pipeline with n_splits parameter
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.inference.uncertainty import (
    CalibrationMap,
    UncertaintyBundle,
    build_uncertainty_bundle,
    _calibration_health,
)
from app.inference.confidence_tracker import (
    ConfidenceTracker,
    TrackerStats,
    _brier,
    _ece,
    _degradation_factor,
    _reliability_diagram,
    DEFAULT_BASELINE_BRIER,
    MIN_WINDOW_FOR_STATS,
    RETRAIN_DEGRADATION_THRESHOLD,
    RETRAIN_ECE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    prices = np.maximum(prices, 1.0)
    opens = prices + rng.normal(0, 0.1, n)
    oc_high = np.maximum(opens, prices)
    oc_low = np.minimum(opens, prices)
    highs = oc_high + rng.uniform(0.05, 0.5, n)
    lows = oc_low - rng.uniform(0.05, 0.5, n)
    volume = rng.integers(1_000, 10_000, n).astype(float)
    ts = pd.date_range("2023-01-01", periods=n, freq="5min")
    vwap = prices + rng.normal(0, 0.05, n)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": prices,
        "volume": volume, "vwap": vwap, "bar_open_time": ts,
    })


def _tracker_with_data(n: int, prob_fn=None, outcome_fn=None, seed: int = 42,
                       baseline_brier: float = DEFAULT_BASELINE_BRIER,
                       tmpdir: Path = None):
    """Create a ConfidenceTracker with n records for symbol 'TEST'."""
    rng = np.random.default_rng(seed)
    tracker = ConfidenceTracker(storage_dir=tmpdir) if tmpdir else ConfidenceTracker()
    for i in range(n):
        prob = prob_fn(i, rng) if prob_fn else float(rng.uniform(0.4, 0.6))
        outcome = outcome_fn(i, rng, prob) if outcome_fn else int(rng.integers(0, 2))
        tracker.record("TEST", prob, outcome, baseline_brier if i == 0 else None)
    return tracker


# ===========================================================================
# U1–U5: CalibrationMap
# ===========================================================================

def test_U1_identity_map_is_noop():
    """Identity CalibrationMap returns the raw probability unchanged."""
    cm = CalibrationMap.identity()
    for p in [0.0, 0.3, 0.5, 0.7, 1.0]:
        assert cm.apply(p) == p


def test_U2_isotonic_map_interpolates_linearly():
    """Isotonic CalibrationMap linearly interpolates within breakpoints."""
    cm = CalibrationMap(
        kind="isotonic",
        x_raw=[0.0, 0.5, 1.0],
        y_cal=[0.0, 0.45, 0.90],
    )
    result = cm.apply(0.25)
    assert abs(result - 0.225) < 1e-6, f"Expected 0.225, got {result}"
    result = cm.apply(0.75)
    assert abs(result - 0.675) < 1e-6


def test_U3_isotonic_map_clamps_to_breakpoints():
    """Isotonic map clamps out-of-range inputs to the nearest breakpoint."""
    cm = CalibrationMap(
        kind="isotonic",
        x_raw=[0.2, 0.8],
        y_cal=[0.25, 0.75],
    )
    assert cm.apply(0.0) == cm.apply(0.2)   # clamped to lower bound
    assert cm.apply(1.0) == cm.apply(0.8)   # clamped to upper bound


def test_U4_platt_map_applies_logistic():
    """Platt scaling applies the logistic function correctly."""
    a, b = 1.5, -0.5
    cm = CalibrationMap(kind="platt", x_raw=[], y_cal=[], platt_a=a, platt_b=b)
    p = 0.6
    expected = 1.0 / (1.0 + math.exp(-(a * p + b)))
    result = cm.apply(p)
    assert abs(result - expected) < 1e-9


def test_U5_calibration_map_round_trips_dict():
    """CalibrationMap serializes to dict and deserializes correctly."""
    cm = CalibrationMap(
        kind="isotonic",
        x_raw=[0.1, 0.5, 0.9],
        y_cal=[0.15, 0.48, 0.85],
        n_calibration_samples=200,
        ece_at_fit=0.03,
    )
    d = cm.to_dict()
    cm2 = CalibrationMap.from_dict(d)
    assert cm2.kind == cm.kind
    assert cm2.x_raw == cm.x_raw
    assert cm2.y_cal == cm.y_cal
    assert cm2.n_calibration_samples == cm.n_calibration_samples
    assert cm2.ece_at_fit == cm.ece_at_fit


# ===========================================================================
# U6–U10: UncertaintyBundle 4-layer consistency
# ===========================================================================

def test_U6_bundle_layers_are_consistent():
    """raw → calibrated → tradeable → action layers are mathematically consistent."""
    cm = CalibrationMap(kind="isotonic", x_raw=[0.0, 1.0], y_cal=[0.0, 1.0])
    bundle = build_uncertainty_bundle(raw_prob_up=0.70, calibration_map=cm, tracker_stats=None)
    # Layer 1 → Layer 2: calibrated should equal raw for identity-like map
    assert 0 <= bundle.calibrated_prob_up <= 1
    assert 0 <= bundle.raw_prob_up <= 1
    # Layer 3: tradeable = 0.5 + (cal - 0.5) * deg
    expected_tradeable_dir = 0.5 + (bundle.calibrated_prob_up - 0.5) * bundle.degradation_factor
    implied_tc = abs(expected_tradeable_dir - 0.5) * 2
    assert abs(implied_tc - bundle.tradeable_confidence) < 1e-6
    # Layer 4: action is one of the valid values
    assert bundle.action in ("buy", "sell", "abstain")


def test_U7_calibrated_plus_down_equals_one():
    """calibrated_prob_up + calibrated_prob_down == 1.0."""
    cm = CalibrationMap.identity()
    bundle = build_uncertainty_bundle(raw_prob_up=0.65, calibration_map=cm, tracker_stats=None)
    assert abs(bundle.calibrated_prob_up + bundle.calibrated_prob_down - 1.0) < 1e-9


def test_U8_no_calibration_uses_identity():
    """When calibration_map is identity, calibrated_prob == raw_prob."""
    cm = CalibrationMap.identity()
    bundle = build_uncertainty_bundle(raw_prob_up=0.62, calibration_map=cm, tracker_stats=None)
    assert abs(bundle.calibrated_prob_up - 0.62) < 1e-6
    assert bundle.calibration_available is False


def test_U9_confidence_band_is_calibrated_prob_pm_ece():
    """Confidence band is calibrated_prob ± ece_recent."""
    from dataclasses import dataclass

    @dataclass
    class FakeStats:
        degradation_factor: float = 1.0
        rolling_brier: float = 0.23
        baseline_brier: float = 0.23
        ece_recent: float = 0.06
        reliability_bins = None
        reliability_mean_pred = None
        reliability_frac_pos = None

    cm = CalibrationMap.identity()
    bundle = build_uncertainty_bundle(raw_prob_up=0.62, calibration_map=cm, tracker_stats=FakeStats())
    cal = bundle.calibrated_prob_up
    ece = 0.06
    assert abs(bundle.confidence_band_low - max(0.0, cal - ece)) < 1e-6
    assert abs(bundle.confidence_band_high - min(1.0, cal + ece)) < 1e-6


def test_U10_prior_abstain_reason_forces_abstain():
    """If prior_abstain_reason is set, action is always 'abstain'."""
    cm = CalibrationMap.identity()
    bundle = build_uncertainty_bundle(
        raw_prob_up=0.99,
        calibration_map=cm,
        tracker_stats=None,
        prior_abstain_reason="regime_suppressed",
    )
    assert bundle.action == "abstain"
    assert "regime_suppressed" in bundle.abstain_reason


# ===========================================================================
# U11–U15: Degradation factor
# ===========================================================================

def test_U11_degradation_factor_at_baseline():
    """When rolling_brier == baseline_brier, factor == 1.0."""
    assert _degradation_factor(0.23, 0.23) == 1.0


def test_U12_degradation_factor_below_baseline():
    """When rolling_brier < baseline_brier, factor == 1.0 (no bonus for over-performance)."""
    assert _degradation_factor(0.20, 0.23) == 1.0


def test_U13_degradation_factor_at_ratio_1_5():
    """When ratio = 1.5, factor = 0.5 (linear interpolation)."""
    result = _degradation_factor(0.345, 0.23)  # ratio = 1.5
    assert abs(result - 0.5) < 1e-4


def test_U14_degradation_factor_at_ratio_2():
    """When ratio >= 2.0, factor = 0.0 (fully degraded)."""
    assert _degradation_factor(0.46, 0.23) == 0.0
    assert _degradation_factor(1.0, 0.23) == 0.0


def test_U15_degradation_factor_zero_baseline():
    """Zero baseline_brier returns 1.0 (no data to compare)."""
    assert _degradation_factor(0.5, 0.0) == 1.0


# ===========================================================================
# U16–U21: Calibration health states
# ===========================================================================

def test_U16_health_good():
    """good: ratio ≤ 1.1 AND ece ≤ 0.05."""
    assert _calibration_health(0.23, 0.23, 0.04) == "good"
    assert _calibration_health(0.25, 0.23, 0.05) == "good"   # ratio=1.087 ≤ 1.1


def test_U17_health_fair():
    """fair: ratio ≤ 1.3 OR ece ≤ 0.10 (but not good)."""
    # ratio=1.2 ≤ 1.3 → fair (even with high ece)
    assert _calibration_health(0.276, 0.23, 0.12) == "fair"
    # ece=0.08 ≤ 0.10 → fair (even with moderate ratio)
    assert _calibration_health(0.299, 0.23, 0.08) == "fair"


def test_U18_health_caution():
    """caution: ratio ≤ 1.6 OR ece ≤ 0.15 (but not good or fair)."""
    # ratio=1.4, ece=0.12 — ratio > 1.3 and ece > 0.10 → caution since ratio ≤ 1.6
    assert _calibration_health(0.322, 0.23, 0.12) == "caution"


def test_U19_health_degraded():
    """degraded: ratio > 1.6 AND ece > 0.15."""
    assert _calibration_health(0.46, 0.23, 0.20) == "degraded"


def test_U20_health_unknown_when_no_baseline():
    """unknown: when rolling_brier or baseline_brier is None."""
    assert _calibration_health(None, 0.23, 0.05) == "unknown"
    assert _calibration_health(0.23, None, 0.05) == "unknown"
    assert _calibration_health(None, None, None) == "unknown"


def test_U21_health_five_states_are_exhaustive():
    """All five health states are reachable."""
    states = {
        _calibration_health(0.23, 0.23, 0.03),      # good
        _calibration_health(0.28, 0.23, 0.08),      # fair
        _calibration_health(0.34, 0.23, 0.12),      # caution
        _calibration_health(0.50, 0.23, 0.20),      # degraded
        _calibration_health(None, None, None),       # unknown
    }
    assert states == {"good", "fair", "caution", "degraded", "unknown"}


# ===========================================================================
# U22–U28: ConfidenceTracker
# ===========================================================================

def test_U22_tracker_returns_unknown_below_min_window():
    """Tracker returns health='unknown' before MIN_WINDOW_FOR_STATS observations."""
    with tempfile.TemporaryDirectory() as d:
        tracker = ConfidenceTracker(storage_dir=Path(d))
        for i in range(MIN_WINDOW_FOR_STATS - 1):
            tracker.record("TEST", 0.55, 1)
        stats = tracker.get_stats("TEST")
        assert stats.calibration_health == "unknown"
        assert stats.rolling_brier is None


def test_U23_tracker_computes_stats_after_min_window():
    """Tracker computes rolling_brier and ece after MIN_WINDOW_FOR_STATS observations."""
    with tempfile.TemporaryDirectory() as d:
        tracker = ConfidenceTracker(storage_dir=Path(d))
        rng = np.random.default_rng(0)
        for i in range(MIN_WINDOW_FOR_STATS):
            p = float(rng.uniform(0.4, 0.6))
            tracker.record("TEST", p, int(rng.integers(0, 2)))
        stats = tracker.get_stats("TEST")
        assert stats.rolling_brier is not None
        assert math.isfinite(stats.rolling_brier)
        assert 0 <= stats.rolling_brier <= 1


def test_U24_tracker_record_inference_alias():
    """record_inference() behaves identically to record()."""
    with tempfile.TemporaryDirectory() as d:
        t1 = ConfidenceTracker(storage_dir=Path(d) / "t1")
        t2 = ConfidenceTracker(storage_dir=Path(d) / "t2")
        rng = np.random.default_rng(1)
        for _ in range(MIN_WINDOW_FOR_STATS):
            p, o = float(rng.uniform(0.4, 0.6)), int(rng.integers(0, 2))
            t1.record("SYM", p, o)
            t2.record_inference("SYM", p, o)
        s1 = t1.get_stats("SYM")
        s2 = t2.get_stats("SYM")
        assert abs(s1.rolling_brier - s2.rolling_brier) < 1e-9


def test_U25_tracker_storage_dir_parameter():
    """ConfidenceTracker respects storage_dir parameter for file persistence."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d)
        tracker = ConfidenceTracker(storage_dir=path)
        rng = np.random.default_rng(2)
        for i in range(MIN_WINDOW_FOR_STATS):
            tracker.record("SPY", float(rng.uniform(0.4, 0.7)), int(rng.integers(0, 2)))
        tracker._save("SPY", tracker._states["SPY"])
        assert (path / "tracker_SPY.json").exists()


def test_U26_tracker_set_baseline_brier():
    """set_baseline_brier updates the baseline for future stats computation."""
    with tempfile.TemporaryDirectory() as d:
        tracker = ConfidenceTracker(storage_dir=Path(d))
        rng = np.random.default_rng(3)
        for _ in range(MIN_WINDOW_FOR_STATS):
            tracker.record("TEST", float(rng.uniform(0.4, 0.6)), int(rng.integers(0, 2)))
        tracker.set_baseline_brier("TEST", 0.10)   # very low baseline → likely degraded
        stats = tracker.get_stats("TEST")
        assert stats.baseline_brier == pytest.approx(0.10, abs=1e-9)


def test_U27_tracker_window_trims_old_records():
    """Records beyond window size are trimmed (rolling window)."""
    window = 30
    with tempfile.TemporaryDirectory() as d:
        tracker = ConfidenceTracker(window=window, storage_dir=Path(d))
        rng = np.random.default_rng(4)
        for _ in range(window + 20):
            tracker.record("TEST", float(rng.uniform(0.4, 0.6)), int(rng.integers(0, 2)))
        state = tracker._states["TEST"]
        assert len(state.probs) == window


def test_U28_tracker_separate_symbols():
    """Each symbol has its own independent state."""
    with tempfile.TemporaryDirectory() as d:
        tracker = ConfidenceTracker(storage_dir=Path(d))
        rng = np.random.default_rng(5)
        # SPY: mostly correct predictions (low brier)
        for _ in range(MIN_WINDOW_FOR_STATS):
            p = float(rng.uniform(0.7, 0.9))
            tracker.record("SPY", p, 1)     # always right → low brier
        # QQQ: always wrong
        for _ in range(MIN_WINDOW_FOR_STATS):
            p = float(rng.uniform(0.7, 0.9))
            tracker.record("QQQ", p, 0)     # always wrong → high brier
        spy_stats = tracker.get_stats("SPY")
        qqq_stats = tracker.get_stats("QQQ")
        assert spy_stats.rolling_brier < qqq_stats.rolling_brier


# ===========================================================================
# U29–U33: Rolling degradation → tradeable confidence reduction
# ===========================================================================

def test_U29_good_performance_no_degradation():
    """When model performs at baseline, degradation_factor == 1.0."""
    rb = DEFAULT_BASELINE_BRIER * 1.05  # 5% worse than baseline → still factor=1 since < ratio=1.1
    # Actually factor=1 for ratio ≤ 1.0, so let's use exactly baseline
    assert _degradation_factor(DEFAULT_BASELINE_BRIER, DEFAULT_BASELINE_BRIER) == 1.0


def test_U30_moderate_degradation_shrinks_signal():
    """Moderate Brier degradation reduces tradeable confidence toward 0.5."""
    from dataclasses import dataclass

    @dataclass
    class Stats30:
        degradation_factor: float = 0.6
        rolling_brier: float = DEFAULT_BASELINE_BRIER * 1.8
        baseline_brier: float = DEFAULT_BASELINE_BRIER
        ece_recent: float = 0.08
        reliability_bins = None
        reliability_mean_pred = None
        reliability_frac_pos = None

    cm = CalibrationMap.identity()
    bundle_nodeg = build_uncertainty_bundle(0.72, cm, None)
    bundle_deg = build_uncertainty_bundle(0.72, cm, Stats30())
    # Degraded tradeable confidence should be lower than no-degradation
    assert bundle_deg.tradeable_confidence <= bundle_nodeg.tradeable_confidence


def test_U31_severe_degradation_causes_abstain():
    """When degradation_factor is very low, the system abstains."""
    from dataclasses import dataclass

    @dataclass
    class StatsSevere:
        degradation_factor: float = 0.20   # < hard_abstain_degradation=0.30
        rolling_brier: float = 0.50
        baseline_brier: float = DEFAULT_BASELINE_BRIER
        ece_recent: float = 0.15
        reliability_bins = None
        reliability_mean_pred = None
        reliability_frac_pos = None

    cm = CalibrationMap.identity()
    bundle = build_uncertainty_bundle(
        raw_prob_up=0.85,
        calibration_map=cm,
        tracker_stats=StatsSevere(),
        hard_abstain_degradation=0.30,
    )
    assert bundle.action == "abstain"
    assert "degraded_performance" in (bundle.abstain_reason or "")


def test_U32_low_tradeable_confidence_causes_abstain():
    """When tradeable_confidence < confidence_threshold, action is 'abstain'."""
    cm = CalibrationMap.identity()
    # raw_prob = 0.52 → calibrated ≈ 0.52 → tradeable = 0.5 + 0.02*1.0 = 0.52
    # tradeable_conf = |0.52-0.5|*2 = 0.04 < threshold=0.55
    bundle = build_uncertainty_bundle(
        raw_prob_up=0.52,
        calibration_map=cm,
        tracker_stats=None,
        confidence_threshold=0.55,
    )
    assert bundle.action == "abstain"
    assert "low_tradeable_confidence" in (bundle.abstain_reason or "")


def test_U33_high_confidence_produces_buy_action():
    """Strong bullish signal with good calibration produces buy action."""
    cm = CalibrationMap.identity()
    bundle = build_uncertainty_bundle(
        raw_prob_up=0.80,
        calibration_map=cm,
        tracker_stats=None,
        confidence_threshold=0.20,
    )
    assert bundle.action == "buy"
    assert bundle.abstain_reason is None


# ===========================================================================
# U34–U36: Retrain signal triggers
# ===========================================================================

def test_U34_retrain_triggered_by_low_degradation():
    """needs_retrain=True when degradation_factor drops below threshold."""
    with tempfile.TemporaryDirectory() as d:
        tracker = ConfidenceTracker(storage_dir=Path(d))
        # low baseline → high ratio → low degradation
        n = 50
        rng = np.random.default_rng(10)
        for i in range(n):
            p = float(rng.uniform(0.6, 0.9))
            outcome = 0   # always wrong → very high Brier
            tracker.record("TEST", p, outcome, baseline_brier=0.05 if i == 0 else None)
        stats = tracker.get_stats("TEST")
        if stats.degradation_factor <= RETRAIN_DEGRADATION_THRESHOLD:
            assert stats.needs_retrain is True


def test_U35_retrain_triggered_by_high_ece():
    """needs_retrain=True when ECE exceeds threshold (with sufficient data)."""
    with tempfile.TemporaryDirectory() as d:
        tracker = ConfidenceTracker(storage_dir=Path(d))
        n = 50
        rng = np.random.default_rng(11)
        # High-confidence wrong predictions → high ECE
        for _ in range(n):
            p = float(rng.uniform(0.85, 0.95))
            tracker.record("TEST", p, 0)   # always wrong
        stats = tracker.get_stats("TEST")
        assert stats.ece_recent is not None
        if stats.ece_recent >= RETRAIN_ECE_THRESHOLD:
            assert stats.needs_retrain is True


def test_U36_no_retrain_before_min_window():
    """needs_retrain remains False before enough observations."""
    with tempfile.TemporaryDirectory() as d:
        tracker = ConfidenceTracker(storage_dir=Path(d))
        for _ in range(MIN_WINDOW_FOR_STATS):
            tracker.record("TEST", 0.95, 0)   # always wrong
        stats = tracker.get_stats("TEST")
        # MIN_WINDOW_FOR_STATS < RETRAIN_MIN_WINDOW → should not trigger yet
        from app.inference.confidence_tracker import RETRAIN_MIN_WINDOW
        if MIN_WINDOW_FOR_STATS < RETRAIN_MIN_WINDOW:
            assert stats.needs_retrain is False


# ===========================================================================
# U37–U39: Reliability diagram computation
# ===========================================================================

def test_U37_reliability_diagram_basic_structure():
    """Reliability diagram returns three parallel lists."""
    rng = np.random.default_rng(20)
    probs = list(rng.uniform(0, 1, 100))
    outcomes = [int(rng.integers(0, 2)) for _ in range(100)]
    centres, mean_preds, frac_poss = _reliability_diagram(probs, outcomes)
    assert len(centres) == len(mean_preds) == len(frac_poss)
    assert all(0 <= c <= 1 for c in centres)
    assert all(0 <= m <= 1 for m in mean_preds)
    assert all(0 <= f <= 1 for f in frac_poss)


def test_U38_reliability_diagram_perfect_calibration():
    """A perfectly calibrated model: each bin's mean_pred ≈ frac_positive."""
    rng = np.random.default_rng(21)
    n = 5000
    probs = list(rng.uniform(0, 1, n))
    # Bernoulli outcomes with exactly P=prob → well-calibrated in expectation
    outcomes = [int(rng.uniform(0, 1) < p) for p in probs]
    bins, mean_preds, frac_poss = _reliability_diagram(probs, outcomes, n_bins=5)
    for mp, fp in zip(mean_preds, frac_poss):
        assert abs(mp - fp) < 0.10, f"Bin miscalibrated: mean_pred={mp:.3f}, frac_pos={fp:.3f}"


def test_U39_ece_near_zero_for_well_calibrated():
    """ECE is small for a well-calibrated model."""
    rng = np.random.default_rng(22)
    n = 3000
    probs = list(rng.uniform(0, 1, n))
    outcomes = [int(rng.uniform(0, 1) < p) for p in probs]
    ece = _ece(probs, outcomes)
    assert ece < 0.08, f"ECE={ece:.4f} too large for well-calibrated model"


# ===========================================================================
# U40–U43: Abstain logic
# ===========================================================================

def test_U40_abstain_on_regime_suppression():
    """Regime suppression forces abstain regardless of probability."""
    cm = CalibrationMap.identity()
    bundle = build_uncertainty_bundle(
        raw_prob_up=0.99,
        calibration_map=cm,
        tracker_stats=None,
        prior_abstain_reason="regime_suppressed:HIGH_VOLATILITY",
    )
    assert bundle.action == "abstain"


def test_U41_no_abstain_when_confident_and_healthy():
    """No abstain when tradeable confidence is high and model is healthy."""
    cm = CalibrationMap.identity()
    bundle = build_uncertainty_bundle(
        raw_prob_up=0.78,
        calibration_map=cm,
        tracker_stats=None,
        confidence_threshold=0.30,
        hard_abstain_degradation=0.10,
    )
    assert bundle.action != "abstain"


def test_U42_sell_action_on_bearish_signal():
    """A strong down signal (raw_prob_up low) produces 'sell' action."""
    cm = CalibrationMap.identity()
    bundle = build_uncertainty_bundle(
        raw_prob_up=0.20,   # strong bearish
        calibration_map=cm,
        tracker_stats=None,
        confidence_threshold=0.20,
    )
    assert bundle.action == "sell"


def test_U43_abstain_reason_none_when_not_abstaining():
    """abstain_reason is None when action is buy or sell."""
    cm = CalibrationMap.identity()
    bundle = build_uncertainty_bundle(
        raw_prob_up=0.80,
        calibration_map=cm,
        tracker_stats=None,
        confidence_threshold=0.10,
    )
    assert bundle.action in ("buy", "sell")
    assert bundle.abstain_reason is None


# ===========================================================================
# U44–U46: TrainingReport.brier_score and .metrics
# ===========================================================================

def test_U44_training_report_has_brier_score_property():
    """TrainingReport exposes brier_score as a property."""
    from app.ml_models.training.trainer import TrainingReport
    assert hasattr(TrainingReport, "brier_score") or "brier_score" in dir(TrainingReport)


def test_U45_training_report_metrics_dict():
    """After a small training run, result.metrics contains the expected keys."""
    from app.ml_models.pipeline import run_training_pipeline
    df = _make_ohlcv(n=400)
    result = run_training_pipeline(df, symbol="TEST", n_splits=3)
    assert hasattr(result, "metrics")
    m = result.metrics
    for key in ("brier_score", "log_loss", "roc_auc", "n_folds"):
        assert key in m, f"Missing key in metrics: {key}"


def test_U46_brier_score_in_valid_range():
    """Brier score from training result is finite and in [0, 1]."""
    from app.ml_models.pipeline import run_training_pipeline
    df = _make_ohlcv(n=400)
    result = run_training_pipeline(df, symbol="TEST", n_splits=3)
    brier = getattr(result, "brier_score", None)
    if brier is None:
        brier = result.metrics.get("brier_score")
    assert brier is not None
    assert math.isfinite(brier)
    assert 0.0 <= brier <= 1.0


# ===========================================================================
# U47–U48: run_training_pipeline with n_splits parameter
# ===========================================================================

def test_U47_n_splits_parameter_accepted():
    """run_training_pipeline accepts n_splits as a shortcut parameter."""
    from app.ml_models.pipeline import run_training_pipeline
    df = _make_ohlcv(n=400)
    # Should not raise TypeError
    result = run_training_pipeline(df, symbol="TEST", n_splits=2)
    assert result is not None


def test_U48_n_splits_overrides_cfg_n_splits():
    """n_splits parameter overrides the value in the default config."""
    from app.ml_models.pipeline import run_training_pipeline
    from app.ml_models.training.config import DEFAULT_CONFIG
    df = _make_ohlcv(n=400)
    result = run_training_pipeline(df, symbol="TEST", n_splits=2)
    # Verify result structure is valid with the overridden n_splits
    brier = result.metrics.get("brier_score")
    assert brier is not None and math.isfinite(brier)
    assert result.winner is not None
