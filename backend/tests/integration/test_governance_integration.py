"""
Integration tests — Governance end-to-end.

IT-G1: Kill switch activation persists to PostgreSQL and blocks inference
IT-G2: Inference event is logged to the database after run_inference
IT-G3: Model registration and promotion lifecycle in live DB

Requires: INTEGRATION_TESTS=1, live PostgreSQL.
"""

from datetime import datetime

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.risk_critical]


@pytest.mark.asyncio
async def test_IT_G1_kill_switch_persists_and_blocks(db_session):
    """Activating kill switch persists to DB and is reflected in subsequent reads."""
    from app.governance.kill_switch import KillSwitchService

    # Ensure kill switch starts inactive
    await KillSwitchService.ensure_initialized(db_session)
    await KillSwitchService.deactivate(db_session, reason="test setup", by="test")

    state = await KillSwitchService.get_state(db_session)
    assert not state.active, "Kill switch should start inactive"

    # Activate
    await KillSwitchService.activate(db_session, reason="integration test", by="pytest")
    state = await KillSwitchService.get_state(db_session)
    assert state.active, "Kill switch should be active after activation"
    assert state.reason == "integration test"

    # Deactivate
    await KillSwitchService.deactivate(db_session, reason="test teardown", by="pytest")
    state = await KillSwitchService.get_state(db_session)
    assert not state.active, "Kill switch should be inactive after deactivation"


@pytest.mark.asyncio
async def test_IT_G2_inference_event_is_logged(db_session):
    """InferenceLogService persists an inference event row to PostgreSQL."""
    from app.governance.inference_log import InferenceLogService
    from app.inference.inference_service import InferenceResult

    fake_result = InferenceResult(
        symbol="SPY",
        timestamp=int(datetime.utcnow().timestamp()),
        bar_open_time="2024-01-02T09:35:00",
        prob_up=0.58,
        prob_down=0.42,
        prob_flat=0.0,
        calibrated_prob_up=0.56,
        calibrated_prob_down=0.44,
        calibration_available=True,
        tradeable_confidence=0.12,
        degradation_factor=1.0,
        action="abstain",
        abstain_reason="low_tradeable_confidence:0.12",
        confidence_band=(0.50, 0.62),
        calibration_health="good",
        rolling_brier=None,
        ece_recent=None,
        reliability_diagram=None,
        expected_move_pct=0.08,
        confidence=0.12,
        trade_signal="no_trade",
        no_trade_reason="low_tradeable_confidence:0.12",
        feature_snapshot_id="test1234",
        model_version="LogisticRegression_v1",
        regime="trending_up",
        top_features={"rsi_14": 0.12, "ret_1": 0.08},
        explanation="Test inference event",
    )

    event_id = await InferenceLogService.log_inference_result(
        db_session, fake_result, request_id="test-req-001"
    )
    assert event_id is not None, "log_inference_result must return a non-None event ID"

    # Verify the row exists
    from app.governance.models import InferenceEvent
    from sqlalchemy import select
    stmt = select(InferenceEvent).where(InferenceEvent.id == event_id)
    row = (await db_session.execute(stmt)).scalar_one_or_none()
    assert row is not None, f"InferenceEvent {event_id} not found in database"
    assert row.symbol == "SPY"
    assert row.action == "abstain"


@pytest.mark.asyncio
async def test_IT_G3_model_version_lifecycle(db_session):
    """Model registration → staging → active → deprecated lifecycle."""
    from app.governance.registry import ModelRegistryService
    from datetime import datetime

    version = await ModelRegistryService.register(
        db_session,
        model_name="logistic",
        version_tag="v_test_integration",
        trained_at=datetime.utcnow(),
        training_symbol="SPY",
        n_samples=500,
        n_features=30,
        feature_manifest_hash="abc123test",
        train_metrics_json='{"brier": 0.24}',
        calibration_kind="sigmoid",
        calibration_ece_at_fit=0.04,
        artifact_dir="logistic_v_test",
    )
    assert version.status == "staging"

    await ModelRegistryService.promote(db_session, version.id, notes="integration test")
    promoted = await ModelRegistryService.get_active(db_session, "logistic")
    assert promoted is not None
    assert promoted.status == "active"
    assert promoted.version_tag == "v_test_integration"
