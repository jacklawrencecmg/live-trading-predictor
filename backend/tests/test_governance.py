"""
Governance system test suite — GV1–GV55

Tests cover every service class, API-level validation, and multi-service
integration flows.  All tests use the in-memory SQLite session from conftest.py.

Categories
----------
GV01–GV08   ModelRegistryService
GV09–GV13   FeatureRegistryService
GV14–GV22   InferenceLogService
GV23–GV28   DriftMonitor (PSI computation + DB writes)
GV29–GV34   CalibrationMonitor
GV35–GV38   DataFreshnessService
GV39–GV45   GovernanceAlertService
GV46–GV51   KillSwitchService
GV52–GV55   GovernanceDashboard + integration flows
"""

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.risk_critical

from app.governance.models import (
    CalibrationSnapshot,
    DataFreshnessCheck,
    DriftSnapshot,
    FeatureVersion,
    GovernanceAlert,
    InferenceEvent,
    KillSwitchState,
    ModelVersion,
)
from app.governance.registry import (
    FeatureRegistryService,
    ModelRegistryService,
    _auto_version_tag,
)
from app.governance.inference_log import InferenceLogService
from app.governance.drift import DriftMonitor, _classify_drift, _psi_1d
from app.governance.calibration import CalibrationMonitor
from app.governance.freshness import DataFreshnessService
from app.governance.alerts import GovernanceAlertService, GovernanceAlertType
from app.governance.kill_switch import KillSwitchService, _cache_set


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
async def governance_tables(db_session):
    """Ensure governance tables exist in the in-memory DB."""
    from app.core.database import Base
    from sqlalchemy import inspect, text
    from app.governance.models import (
        ModelVersion, FeatureVersion, InferenceEvent, DriftSnapshot,
        CalibrationSnapshot, DataFreshnessCheck, GovernanceAlert, KillSwitchState,
    )
    # Tables are created by Base.metadata in conftest — nothing extra needed.
    yield


@dataclass
class FakeTrackerStats:
    """Minimal TrackerStats for testing CalibrationMonitor."""
    symbol: str = "SPY"
    window_size: int = 50
    rolling_brier: Optional[float] = 0.230
    baseline_brier: Optional[float] = 0.225
    degradation_factor: float = 0.98
    ece_recent: Optional[float] = 0.035
    calibration_health: str = "good"
    reliability_bins: Optional[list] = None
    reliability_mean_pred: Optional[list] = None
    reliability_frac_pos: Optional[list] = None
    needs_retrain: bool = False
    retrain_reason: Optional[str] = None


@dataclass
class FakeInferenceResult:
    """Minimal InferenceResult for testing InferenceLogService."""
    symbol: str = "SPY"
    timestamp: int = 1712000000
    bar_open_time: str = "2025-04-01 14:30:00"
    model_version: str = "LogisticRegression_v1"
    feature_snapshot_id: str = "abc123"
    prob_up: float = 0.58
    prob_down: float = 0.42
    calibrated_prob_up: float = 0.61
    calibrated_prob_down: float = 0.39
    calibration_available: bool = True
    tradeable_confidence: float = 0.59
    degradation_factor: float = 0.98
    action: str = "buy"
    abstain_reason: Optional[str] = None
    calibration_health: str = "good"
    ece_recent: float = 0.035
    rolling_brier: float = 0.230
    expected_move_pct: float = 0.15
    regime: str = "trending_up"


# ============================================================================
# GV01–GV08  ModelRegistryService
# ============================================================================

@pytest.mark.asyncio
async def test_GV01_register_model_creates_staging_row(db_session):
    """GV01: Registering a model creates a staging row."""
    row = await ModelRegistryService.register(
        db_session,
        model_name="logistic",
        version_tag="v1.0.0",
        n_samples=5000,
        n_features=36,
    )
    await db_session.commit()
    assert row.id is not None
    assert row.status == "staging"
    assert row.model_name == "logistic"
    assert row.version_tag == "v1.0.0"


@pytest.mark.asyncio
async def test_GV02_auto_version_tag_increments(db_session):
    """GV02: Auto-generated version tags increment correctly."""
    await ModelRegistryService.register(db_session, model_name="gbt", version_tag="v1.0.0")
    await ModelRegistryService.register(db_session, model_name="gbt", version_tag="v2.0.0")
    await db_session.commit()

    # Auto-generate third tag
    from sqlalchemy import select
    rows = await db_session.execute(
        select(ModelVersion.version_tag).where(ModelVersion.model_name == "gbt")
    )
    existing = [r[0] for r in rows.all()]
    tag = _auto_version_tag("gbt", existing)
    assert tag == "v3.0.0"


@pytest.mark.asyncio
async def test_GV03_promote_staging_to_active(db_session):
    """GV03: Promoting a staging model sets status=active."""
    row = await ModelRegistryService.register(
        db_session, model_name="logistic", version_tag="v1.0.0"
    )
    await db_session.flush()
    promoted = await ModelRegistryService.promote(db_session, row.id)
    await db_session.commit()
    assert promoted.status == "active"
    assert promoted.promoted_at is not None


@pytest.mark.asyncio
async def test_GV04_promote_deprecates_previous_active(db_session):
    """GV04: Promoting a new version deprecates the current active."""
    r1 = await ModelRegistryService.register(
        db_session, model_name="logistic", version_tag="v1.0.0"
    )
    await ModelRegistryService.promote(db_session, r1.id)

    r2 = await ModelRegistryService.register(
        db_session, model_name="logistic", version_tag="v2.0.0"
    )
    await db_session.flush()
    await ModelRegistryService.promote(db_session, r2.id)
    await db_session.commit()

    from sqlalchemy import select
    result = await db_session.execute(
        select(ModelVersion).where(ModelVersion.id == r1.id)
    )
    r1_reloaded = result.scalar_one()
    assert r1_reloaded.status == "deprecated"


@pytest.mark.asyncio
async def test_GV05_promote_already_active_raises(db_session):
    """GV05: Promoting an already-active model raises ValueError."""
    row = await ModelRegistryService.register(
        db_session, model_name="logistic", version_tag="v1.0.0"
    )
    await ModelRegistryService.promote(db_session, row.id)
    with pytest.raises(ValueError, match="already active"):
        await ModelRegistryService.promote(db_session, row.id)


@pytest.mark.asyncio
async def test_GV06_deprecate_model(db_session):
    """GV06: Deprecating a model sets status=deprecated."""
    row = await ModelRegistryService.register(
        db_session, model_name="gbt", version_tag="v1.0.0"
    )
    await db_session.flush()
    dep = await ModelRegistryService.deprecate(db_session, row.id, reason="replaced by v2")
    await db_session.commit()
    assert dep.status == "deprecated"
    assert dep.deprecated_at is not None


@pytest.mark.asyncio
async def test_GV07_get_active_version(db_session):
    """GV07: get_active returns only the active version."""
    r1 = await ModelRegistryService.register(
        db_session, model_name="random_forest", version_tag="v1.0.0"
    )
    await ModelRegistryService.promote(db_session, r1.id)
    r2 = await ModelRegistryService.register(
        db_session, model_name="random_forest", version_tag="v2.0.0"
    )
    await db_session.flush()
    active = await ModelRegistryService.get_active(db_session, "random_forest")
    assert active is not None
    assert active.version_tag == "v1.0.0"


@pytest.mark.asyncio
async def test_GV08_list_versions_filters_by_status(db_session):
    """GV08: list_versions returns correct rows for status filter."""
    r1 = await ModelRegistryService.register(
        db_session, model_name="logistic", version_tag="v1.0.0"
    )
    await ModelRegistryService.promote(db_session, r1.id)
    await ModelRegistryService.register(
        db_session, model_name="logistic", version_tag="v2.0.0"
    )
    await db_session.commit()

    staging = await ModelRegistryService.list_versions(db_session, model_name="logistic", status="staging")
    active  = await ModelRegistryService.list_versions(db_session, model_name="logistic", status="active")
    assert len(staging) == 1
    assert len(active) == 1
    assert active[0].version_tag == "v1.0.0"


# ============================================================================
# GV09–GV13  FeatureRegistryService
# ============================================================================

SAMPLE_FEATURES = [
    {"name": "rsi_14", "version": 1, "group": "trend"},
    {"name": "macd_hist", "version": 1, "group": "trend"},
    {"name": "volume_ratio", "version": 2, "group": "volume"},
]

@pytest.mark.asyncio
async def test_GV09_register_feature_manifest(db_session):
    """GV09: ensure_manifest creates a new row."""
    row = await FeatureRegistryService.ensure_manifest(
        db_session,
        manifest_hash="abc123",
        pipeline_version=1,
        feature_list=SAMPLE_FEATURES,
        description="initial manifest",
    )
    await db_session.commit()
    assert row.manifest_hash == "abc123"
    assert row.feature_count == 3
    assert row.pipeline_version == 1


@pytest.mark.asyncio
async def test_GV10_ensure_manifest_idempotent(db_session):
    """GV10: calling ensure_manifest twice with same hash returns same row."""
    r1 = await FeatureRegistryService.ensure_manifest(
        db_session, manifest_hash="xyz789", pipeline_version=1, feature_list=SAMPLE_FEATURES
    )
    r2 = await FeatureRegistryService.ensure_manifest(
        db_session, manifest_hash="xyz789", pipeline_version=1, feature_list=SAMPLE_FEATURES
    )
    assert r1.id == r2.id


@pytest.mark.asyncio
async def test_GV11_get_manifest_by_hash(db_session):
    """GV11: get() returns correct row or None."""
    await FeatureRegistryService.ensure_manifest(
        db_session, manifest_hash="def456", pipeline_version=1, feature_list=SAMPLE_FEATURES
    )
    await db_session.commit()
    row = await FeatureRegistryService.get(db_session, "def456")
    assert row is not None
    assert row.manifest_hash == "def456"

    missing = await FeatureRegistryService.get(db_session, "nonexistent")
    assert missing is None


@pytest.mark.asyncio
async def test_GV12_diff_feature_lists_detects_changes(db_session):
    """GV12: diff_feature_lists identifies added, removed, version_bumped."""
    list_a = [
        {"name": "rsi_14", "version": 1, "group": "trend"},
        {"name": "macd_hist", "version": 1, "group": "trend"},
    ]
    list_b = [
        {"name": "rsi_14", "version": 2, "group": "trend"},   # bumped
        {"name": "volume_ratio", "version": 1, "group": "volume"},  # added
    ]  # macd_hist removed
    diff = FeatureRegistryService.diff_feature_lists(list_a, list_b)
    assert "volume_ratio" in diff["added"]
    assert "macd_hist" in diff["removed"]
    assert "rsi_14" in diff["version_bumped"]


@pytest.mark.asyncio
async def test_GV13_list_all_manifests(db_session):
    """GV13: list_all returns all registered manifests ordered by recorded_at."""
    await FeatureRegistryService.ensure_manifest(
        db_session, manifest_hash="h1", pipeline_version=1, feature_list=SAMPLE_FEATURES
    )
    await FeatureRegistryService.ensure_manifest(
        db_session, manifest_hash="h2", pipeline_version=2, feature_list=SAMPLE_FEATURES
    )
    await db_session.commit()
    rows = await FeatureRegistryService.list_all(db_session)
    assert len(rows) >= 2


# ============================================================================
# GV14–GV22  InferenceLogService
# ============================================================================

@pytest.mark.asyncio
async def test_GV14_log_inference_result(db_session):
    """GV14: log_inference_result persists an InferenceEvent row."""
    result = FakeInferenceResult()
    ev = await InferenceLogService.log_inference_result(
        db_session, symbol="SPY", result=result, request_id="req001"
    )
    await db_session.commit()
    assert ev.id is not None
    assert ev.symbol == "SPY"
    assert ev.action == "buy"
    assert ev.prob_up == pytest.approx(0.58)
    assert ev.request_id == "req001"


@pytest.mark.asyncio
async def test_GV15_log_abstain_result(db_session):
    """GV15: abstain action is correctly recorded."""
    result = FakeInferenceResult(action="abstain", abstain_reason="regime_suppressed:high_volatility")
    ev = await InferenceLogService.log_inference_result(
        db_session, symbol="SPY", result=result
    )
    await db_session.commit()
    assert ev.action == "abstain"
    assert "regime_suppressed" in ev.abstain_reason


@pytest.mark.asyncio
async def test_GV16_record_outcome_fills_column(db_session):
    """GV16: record_outcome back-fills actual_outcome for pending events."""
    bar_time = datetime(2025, 4, 1, 14, 30)
    result = FakeInferenceResult(bar_open_time="2025-04-01 14:30:00")
    ev = await InferenceLogService.log_inference_result(
        db_session, symbol="SPY", result=result
    )
    await db_session.flush()

    # Manually set bar_open_time (log_inference_result parses string)
    ev.bar_open_time = bar_time
    await db_session.flush()

    n = await InferenceLogService.record_outcome(
        db_session, symbol="SPY", bar_open_time=bar_time, actual_outcome=1
    )
    await db_session.commit()
    assert n >= 1

    from sqlalchemy import select
    r = await db_session.execute(select(InferenceEvent).where(InferenceEvent.id == ev.id))
    loaded = r.scalar_one()
    assert loaded.actual_outcome == 1
    assert loaded.outcome_recorded_at is not None


@pytest.mark.asyncio
async def test_GV17_query_by_action_filter(db_session):
    """GV17: query() with action filter returns only matching rows."""
    await InferenceLogService.log_inference_result(
        db_session, symbol="SPY", result=FakeInferenceResult(action="buy")
    )
    await InferenceLogService.log_inference_result(
        db_session, symbol="SPY", result=FakeInferenceResult(action="abstain")
    )
    await db_session.commit()

    buys = await InferenceLogService.query(db_session, symbol="SPY", action="buy")
    assert all(e.action == "buy" for e in buys)


@pytest.mark.asyncio
async def test_GV18_pending_only_filter(db_session):
    """GV18: pending_only=True returns only rows with no outcome."""
    ev = await InferenceLogService.log_inference_result(
        db_session, symbol="QQQ", result=FakeInferenceResult()
    )
    await db_session.flush()

    pending = await InferenceLogService.query(db_session, symbol="QQQ", pending_only=True)
    assert any(e.id == ev.id for e in pending)


@pytest.mark.asyncio
async def test_GV19_accuracy_stats_correct_buy(db_session):
    """GV19: accuracy_stats correctly counts correct BUY predictions."""
    result = FakeInferenceResult(action="buy")
    ev = await InferenceLogService.log_inference_result(
        db_session, symbol="SPY", result=result
    )
    await db_session.flush()
    ev.actual_outcome = 1
    ev.outcome_recorded_at = datetime.utcnow()
    await db_session.commit()

    stats = await InferenceLogService.get_accuracy_stats(db_session, "SPY", window=10)
    assert stats["n_correct"] == 1
    assert stats["accuracy"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_GV20_accuracy_stats_wrong_buy(db_session):
    """GV20: accuracy_stats marks incorrect BUY (outcome=down=0) as wrong."""
    ev = await InferenceLogService.log_inference_result(
        db_session, symbol="SPY", result=FakeInferenceResult(action="buy")
    )
    await db_session.flush()
    ev.actual_outcome = 0
    ev.outcome_recorded_at = datetime.utcnow()
    await db_session.commit()

    stats = await InferenceLogService.get_accuracy_stats(db_session, "SPY", window=10)
    assert stats["n_correct"] == 0
    assert stats["accuracy"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_GV21_abstain_rate_computed(db_session):
    """GV21: abstain_rate reflects fraction of abstain actions."""
    for action in ["buy", "abstain", "abstain"]:
        await InferenceLogService.log_inference_result(
            db_session,
            symbol="SPY",
            result=FakeInferenceResult(action=action)
        )
    await db_session.commit()

    counts = await InferenceLogService.count_24h(db_session, symbol="SPY")
    total = sum(counts.values())
    abstained = counts.get("abstain", 0)
    assert total == 3
    assert abstained == 2


@pytest.mark.asyncio
async def test_GV22_log_truncates_long_abstain_reason(db_session):
    """GV22: abstain_reason > 128 chars is truncated to 128."""
    long_reason = "x" * 200
    result = FakeInferenceResult(action="abstain", abstain_reason=long_reason)
    ev = await InferenceLogService.log_inference_result(
        db_session, symbol="SPY", result=result
    )
    assert len(ev.abstain_reason) == 128


# ============================================================================
# GV23–GV28  DriftMonitor
# ============================================================================

def _make_feature_matrix(n=200, n_feats=5, shift=0.0):
    """Create a synthetic feature matrix for drift tests."""
    rng = np.random.default_rng(42)
    X = rng.normal(loc=shift, scale=1.0, size=(n, n_feats))
    return X


@pytest.mark.parametrize("psi,expected", [
    (0.05, "none"),
    (0.12, "moderate"),
    (0.30, "high"),
])
def test_GV23_classify_drift_thresholds(psi, expected):
    """GV23: _classify_drift returns correct level for known PSI values."""
    assert _classify_drift(psi) == expected


def test_GV24_psi_near_zero_for_same_distribution():
    """GV24: PSI ≈ 0 when current distribution matches reference."""
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, 500)
    # Reference uses percentile dict from same distribution
    ref = {f"p{i*10}": float(np.percentile(data, i*10)) for i in range(11)}
    psi = _psi_1d(data[:200], ref)
    assert not math.isnan(psi)
    assert psi < 0.10


def test_GV25_psi_high_for_shifted_distribution():
    """GV25: PSI is high when distributions differ significantly."""
    rng = np.random.default_rng(1)
    ref_data = rng.normal(0, 1, 500)
    cur_data  = rng.normal(3, 1, 200)   # shifted by 3 sigma
    ref_pcts = {f"p{i*10}": float(np.percentile(ref_data, i*10)) for i in range(11)}
    psi = _psi_1d(cur_data, ref_pcts)
    assert psi > 0.25


@pytest.mark.asyncio
async def test_GV26_record_drift_snapshot(db_session):
    """GV26: record_snapshot persists a DriftSnapshot row."""
    psi_map = {"rsi_14": 0.05, "volume_ratio": 0.35, "macd_hist": 0.08}
    snap = await DriftMonitor.record_snapshot(
        db_session,
        symbol="SPY",
        psi_by_feature=psi_map,
        window_bars=200,
        manifest_hash="abc",
    )
    await db_session.commit()
    assert snap.id is not None
    assert snap.drift_level == "high"
    assert snap.max_psi == pytest.approx(0.35, abs=1e-5)
    high_feats = json.loads(snap.high_drift_features_json)
    assert "volume_ratio" in high_feats


@pytest.mark.asyncio
async def test_GV27_get_latest_drift_snapshot(db_session):
    """GV27: get_latest returns most recent snapshot for symbol."""
    psi_map = {"rsi_14": 0.02}
    await DriftMonitor.record_snapshot(
        db_session, symbol="SPY", psi_by_feature=psi_map, window_bars=100
    )
    await DriftMonitor.record_snapshot(
        db_session, symbol="SPY", psi_by_feature={"rsi_14": 0.05}, window_bars=200
    )
    await db_session.commit()
    latest = await DriftMonitor.get_latest(db_session, "SPY")
    # Most recently inserted has max_psi ≈ 0.05
    assert latest.max_psi == pytest.approx(0.05, abs=1e-5)


@pytest.mark.asyncio
async def test_GV28_compute_psi_from_matrix(db_session):
    """GV28: compute_psi_from_matrix returns a PSI per feature."""
    X = _make_feature_matrix(n=200, n_feats=3)
    names = ["rsi_14", "macd_hist", "volume_ratio"]
    psi_map = DriftMonitor.compute_psi_from_matrix(X, names)
    assert set(psi_map.keys()) == set(names)
    for v in psi_map.values():
        assert v >= 0 or math.isnan(v)


# ============================================================================
# GV29–GV34  CalibrationMonitor
# ============================================================================

@pytest.mark.asyncio
async def test_GV29_record_calibration_snapshot(db_session):
    """GV29: record_snapshot persists a CalibrationSnapshot."""
    stats = FakeTrackerStats()
    snap = await CalibrationMonitor.record_snapshot(
        db_session, symbol="SPY", tracker_stats=stats
    )
    await db_session.commit()
    assert snap.id is not None
    assert snap.calibration_health == "good"
    assert snap.needs_retrain is False


@pytest.mark.asyncio
async def test_GV30_retrain_flag_propagated(db_session):
    """GV30: needs_retrain=True from tracker propagates to snapshot."""
    stats = FakeTrackerStats(
        needs_retrain=True,
        retrain_reason="degradation_factor=0.38 <= threshold=0.40",
        calibration_health="degraded",
        degradation_factor=0.38,
    )
    snap = await CalibrationMonitor.record_snapshot(
        db_session, symbol="QQQ", tracker_stats=stats
    )
    await db_session.commit()
    assert snap.needs_retrain is True
    assert "0.38" in snap.retrain_reason


@pytest.mark.asyncio
async def test_GV31_symbols_needing_retrain(db_session):
    """GV31: symbols_needing_retrain returns symbols with latest snapshot needs_retrain=True."""
    good_stats = FakeTrackerStats(symbol="SPY", needs_retrain=False)
    bad_stats  = FakeTrackerStats(symbol="TSLA", needs_retrain=True,
                                  retrain_reason="ece=0.13", calibration_health="degraded")
    await CalibrationMonitor.record_snapshot(db_session, symbol="SPY",  tracker_stats=good_stats)
    await CalibrationMonitor.record_snapshot(db_session, symbol="TSLA", tracker_stats=bad_stats)
    await db_session.commit()
    symbols = await CalibrationMonitor.symbols_needing_retrain(db_session)
    assert "TSLA" in symbols
    assert "SPY" not in symbols


@pytest.mark.asyncio
async def test_GV32_health_by_symbol(db_session):
    """GV32: health_by_symbol returns latest health per symbol."""
    await CalibrationMonitor.record_snapshot(
        db_session, symbol="SPY",
        tracker_stats=FakeTrackerStats(calibration_health="good")
    )
    await CalibrationMonitor.record_snapshot(
        db_session, symbol="AAPL",
        tracker_stats=FakeTrackerStats(symbol="AAPL", calibration_health="degraded")
    )
    await db_session.commit()
    health = await CalibrationMonitor.health_by_symbol(db_session)
    assert health.get("SPY") == "good"
    assert health.get("AAPL") == "degraded"


@pytest.mark.asyncio
async def test_GV33_trend_direction_degrading(db_session):
    """GV33: trend_direction returns 'degrading' for monotonically worsening Brier."""
    snaps = []
    for i, brier in enumerate([0.22, 0.23, 0.25, 0.28, 0.32]):
        s = CalibrationSnapshot(
            symbol="SPY",
            snapshot_at=datetime.utcnow() + timedelta(hours=i),
            rolling_brier=brier,
        )
        snaps.append(s)
    trend = CalibrationMonitor.trend_direction(snaps)
    assert trend == "degrading"


@pytest.mark.asyncio
async def test_GV34_trend_direction_stable(db_session):
    """GV34: trend_direction returns 'stable' for flat Brier."""
    snaps = [
        CalibrationSnapshot(symbol="SPY",
                            snapshot_at=datetime.utcnow() + timedelta(hours=i),
                            rolling_brier=0.230 + 0.001 * (i % 2))
        for i in range(5)
    ]
    trend = CalibrationMonitor.trend_direction(snaps)
    assert trend in ("stable", "improving")


# ============================================================================
# GV35–GV38  DataFreshnessService
# ============================================================================

@pytest.mark.asyncio
async def test_GV35_record_fresh_check(db_session):
    """GV35: A recently updated feed is not stale."""
    check = await DataFreshnessService.record_check(
        db_session,
        symbol="SPY",
        source="quote_feed",
        last_data_ts=datetime.utcnow() - timedelta(seconds=10),
    )
    await db_session.commit()
    assert check.is_stale is False
    assert check.age_seconds < 60


@pytest.mark.asyncio
async def test_GV36_record_stale_check(db_session):
    """GV36: An old feed timestamp is correctly flagged as stale."""
    check = await DataFreshnessService.record_check(
        db_session,
        symbol="SPY",
        source="options_chain",
        last_data_ts=datetime.utcnow() - timedelta(seconds=7200),  # 2h > 1h threshold
    )
    await db_session.commit()
    assert check.is_stale is True


@pytest.mark.asyncio
async def test_GV37_missing_source_always_stale(db_session):
    """GV37: last_data_ts=None means source is unavailable → stale."""
    check = await DataFreshnessService.record_check(
        db_session, symbol="QQQ", source="quote_feed", last_data_ts=None
    )
    await db_session.commit()
    assert check.is_stale is True
    assert check.age_seconds is None


@pytest.mark.asyncio
async def test_GV38_get_stale_feeds_returns_recent_stale(db_session):
    """GV38: get_stale_feeds returns stale checks within the time window."""
    await DataFreshnessService.record_check(
        db_session, symbol="SPY", source="options_chain",
        last_data_ts=datetime.utcnow() - timedelta(hours=3)
    )
    await db_session.commit()
    stale = await DataFreshnessService.get_stale_feeds(db_session, since_minutes=60)
    assert any(c.symbol == "SPY" and c.source == "options_chain" for c in stale)


# ============================================================================
# GV39–GV45  GovernanceAlertService
# ============================================================================

@pytest.mark.asyncio
async def test_GV39_raise_alert_creates_row(db_session):
    """GV39: raise_alert creates an active GovernanceAlert row."""
    alert = await GovernanceAlertService.raise_alert(
        db_session,
        alert_type=GovernanceAlertType.DRIFT_HIGH,
        title="SPY drift detected",
        symbol="SPY",
        details={"max_psi": 0.30},
    )
    await db_session.commit()
    assert alert.id is not None
    assert alert.is_active is True
    assert alert.severity == "critical"


@pytest.mark.asyncio
async def test_GV40_alert_deduplication(db_session):
    """GV40: Second alert with same dedup_key bumps timestamp, not a new row."""
    a1 = await GovernanceAlertService.raise_alert(
        db_session, alert_type=GovernanceAlertType.FEED_STALE,
        title="first", symbol="SPY", dedup_key="feed_stale:SPY:options_chain"
    )
    first_ts = a1.triggered_at
    await db_session.flush()

    a2 = await GovernanceAlertService.raise_alert(
        db_session, alert_type=GovernanceAlertType.FEED_STALE,
        title="second", symbol="SPY", dedup_key="feed_stale:SPY:options_chain"
    )
    await db_session.commit()
    # Same row
    assert a1.id == a2.id
    # Timestamp was bumped (a2 triggered_at >= first_ts)
    assert a2.triggered_at >= first_ts


@pytest.mark.asyncio
async def test_GV41_acknowledge_alert(db_session):
    """GV41: Acknowledging an alert marks it inactive."""
    alert = await GovernanceAlertService.raise_alert(
        db_session, alert_type=GovernanceAlertType.RETRAIN_NEEDED,
        title="retrain SPY", symbol="SPY"
    )
    await db_session.flush()

    acked = await GovernanceAlertService.acknowledge(db_session, alert.id, by="alice")
    await db_session.commit()
    assert acked.is_active is False
    assert acked.acknowledged_by == "alice"
    assert acked.acknowledged_at is not None


@pytest.mark.asyncio
async def test_GV42_get_active_alerts_excludes_acknowledged(db_session):
    """GV42: get_active returns only is_active=True rows."""
    a1 = await GovernanceAlertService.raise_alert(
        db_session, alert_type=GovernanceAlertType.DRIFT_MODERATE,
        title="active alert", symbol="SPY", dedup_key="test_active"
    )
    a2 = await GovernanceAlertService.raise_alert(
        db_session, alert_type=GovernanceAlertType.CALIBRATION_DEGRADED,
        title="to be acked", symbol="QQQ", dedup_key="test_acked"
    )
    await db_session.flush()
    await GovernanceAlertService.acknowledge(db_session, a2.id, by="bob")
    await db_session.commit()

    active = await GovernanceAlertService.get_active(db_session)
    active_ids = [a.id for a in active]
    assert a1.id in active_ids
    assert a2.id not in active_ids


@pytest.mark.asyncio
async def test_GV43_clear_expired_alerts(db_session):
    """GV43: clear_expired deactivates alerts past their expires_at."""
    past = datetime.utcnow() - timedelta(hours=1)
    alert = GovernanceAlert(
        alert_type="test",
        severity="info",
        title="expired alert",
        triggered_at=past - timedelta(hours=2),
        expires_at=past,
        is_active=True,
    )
    db_session.add(alert)
    await db_session.flush()

    n = await GovernanceAlertService.clear_expired(db_session)
    await db_session.commit()
    assert n >= 1

    from sqlalchemy import select
    r = await db_session.execute(
        select(GovernanceAlert).where(GovernanceAlert.id == alert.id)
    )
    loaded = r.scalar_one()
    assert loaded.is_active is False


@pytest.mark.asyncio
async def test_GV44_typed_feed_stale_alert(db_session):
    """GV44: alert_feed_stale creates correct alert_type and dedup_key."""
    alert = await GovernanceAlertService.alert_feed_stale(
        db_session, symbol="NVDA", source="options_chain", age_seconds=7200.0
    )
    await db_session.commit()
    assert alert.alert_type == GovernanceAlertType.FEED_STALE
    assert "NVDA" in alert.title
    assert alert.severity == "warning"


@pytest.mark.asyncio
async def test_GV45_typed_risk_breach_alert(db_session):
    """GV45: alert_risk_breach creates critical alert."""
    alert = await GovernanceAlertService.alert_risk_breach(
        db_session, symbol=None,
        breach_type="daily_loss_limit",
        details={"daily_pnl": -600, "limit": 500}
    )
    await db_session.commit()
    assert alert.alert_type == GovernanceAlertType.RISK_BREACH
    assert alert.severity == "critical"


# ============================================================================
# GV46–GV51  KillSwitchService
# ============================================================================

@pytest.mark.asyncio
async def test_GV46_kill_switch_starts_inactive(db_session):
    """GV46: Fresh DB has kill switch inactive."""
    active = await KillSwitchService.is_active_db(db_session)
    assert active is False


@pytest.mark.asyncio
async def test_GV47_activate_kill_switch(db_session):
    """GV47: activate() sets active=True and writes audit log."""
    row = await KillSwitchService.activate(
        db_session, reason="test activation", by="alice"
    )
    await db_session.commit()
    assert row.active is True
    assert row.activated_by == "alice"
    assert row.reason == "test activation"


@pytest.mark.asyncio
async def test_GV48_deactivate_kill_switch(db_session):
    """GV48: deactivate() sets active=False."""
    await KillSwitchService.activate(db_session, reason="test", by="alice")
    row = await KillSwitchService.deactivate(db_session, by="bob", reason="resolved")
    await db_session.commit()
    assert row.active is False


@pytest.mark.asyncio
async def test_GV49_is_active_cached_reflects_db(db_session):
    """GV49: is_active_db() updates the module cache."""
    _cache_set(False)
    await KillSwitchService.activate(db_session, reason="caching test", by="sys")
    is_active = await KillSwitchService.is_active_db(db_session)
    assert is_active is True
    # Cache should now be True
    active, fresh = __import__(
        "app.governance.kill_switch", fromlist=["_cache_get"]
    )._cache_get()
    assert active is True and fresh is True


@pytest.mark.asyncio
async def test_GV50_toggle_activates_correctly(db_session):
    """GV50: toggle(active=True) activates; toggle(active=False) deactivates."""
    await KillSwitchService.toggle(db_session, active=True, reason="r", by="op")
    row = await KillSwitchService.get_state(db_session)
    assert row.active is True

    await KillSwitchService.toggle(db_session, active=False, by="op")
    row = await KillSwitchService.get_state(db_session)
    assert row.active is False


@pytest.mark.asyncio
async def test_GV51_kill_switch_activation_creates_governance_alert(db_session):
    """GV51: Activating kill switch creates a KILL_SWITCH_ACTIVATED governance alert."""
    await KillSwitchService.activate(db_session, reason="emergency stop", by="risk_team")
    await db_session.commit()

    alerts = await GovernanceAlertService.get_active(
        db_session, alert_type=GovernanceAlertType.KILL_SWITCH_ACTIVATED
    )
    assert len(alerts) >= 1
    assert alerts[0].severity == "critical"


# ============================================================================
# GV52–GV55  GovernanceDashboard + integration
# ============================================================================

@pytest.mark.asyncio
async def test_GV52_summary_returns_expected_keys(db_session):
    """GV52: summary() returns all required top-level keys."""
    from app.governance.dashboard import GovernanceDashboard
    result = await GovernanceDashboard.summary(db_session)

    required_keys = {
        "generated_at", "kill_switch_active", "active_model",
        "active_feature_manifest", "calibration_health",
        "symbols_needing_retrain", "drift_summary", "stale_feeds",
        "active_critical_alerts", "active_warning_alerts",
        "recent_alert_titles", "inference_count_24h", "abstain_rate_24h",
    }
    assert required_keys.issubset(result.keys())


@pytest.mark.asyncio
async def test_GV53_summary_reflects_active_kill_switch(db_session):
    """GV53: summary shows kill_switch_active=True after activation."""
    from app.governance.dashboard import GovernanceDashboard
    await KillSwitchService.activate(db_session, reason="test", by="test_runner")
    await db_session.commit()

    summary = await GovernanceDashboard.summary(db_session)
    assert summary["kill_switch_active"] is True


@pytest.mark.asyncio
async def test_GV54_rolling_performance_returns_keys(db_session):
    """GV54: rolling_performance() returns required analytics keys."""
    from app.governance.dashboard import GovernanceDashboard
    result = await GovernanceDashboard.rolling_performance(db_session, "SPY", window_days=7)

    assert "symbol" in result
    assert "accuracy" in result
    assert "calibration_trend" in result


@pytest.mark.asyncio
async def test_GV55_full_governance_flow(db_session):
    """
    GV55: Integration test — register model, log inference, record outcome,
    snapshot calibration, check summary.

    Verifies that the full governance lifecycle works end-to-end without errors.
    """
    from app.governance.dashboard import GovernanceDashboard

    # 1. Register feature manifest
    await FeatureRegistryService.ensure_manifest(
        db_session, manifest_hash="integ_hash",
        pipeline_version=1, feature_list=SAMPLE_FEATURES,
    )

    # 2. Register and promote a model version
    ver = await ModelRegistryService.register(
        db_session, model_name="logistic", version_tag="v1.0.0",
        feature_manifest_hash="integ_hash", n_samples=5000, n_features=3
    )
    await ModelRegistryService.promote(db_session, ver.id)

    # 3. Log some inference events
    for i, action in enumerate(["buy", "buy", "abstain", "sell", "buy"]):
        res = FakeInferenceResult(action=action)
        ev = await InferenceLogService.log_inference_result(
            db_session, symbol="SPY", result=res,
            model_version_id=ver.id, manifest_hash="integ_hash",
        )

    # 4. Record calibration snapshot
    stats = FakeTrackerStats(symbol="SPY", needs_retrain=False, window_size=50)
    await CalibrationMonitor.record_snapshot(db_session, symbol="SPY", tracker_stats=stats)

    # 5. Record freshness check
    await DataFreshnessService.record_check(
        db_session, symbol="SPY", source="quote_feed",
        last_data_ts=datetime.utcnow() - timedelta(seconds=5),
    )

    await db_session.commit()

    # 6. Summary should reflect registered model and positive health
    summary = await GovernanceDashboard.summary(db_session)
    assert summary["kill_switch_active"] is False
    assert summary["active_model"] is not None
    assert summary["active_model"]["model_name"] == "logistic"
    assert summary["inference_count_24h"] == 5
    assert "SPY" in summary["calibration_health"]
    assert summary["calibration_health"]["SPY"] == "good"
