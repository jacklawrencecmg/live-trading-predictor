"""
Governance API route tests — GR01–GR30

Tests exercise the FastAPI HTTP layer for all governance endpoints.
Uses httpx.AsyncClient against a minimal test app that mounts the governance
router with the in-memory SQLite db_session injected via dependency override.

Categories
----------
GR01–GR03   GET  /summary and GET /performance/{symbol}
GR04–GR05   GET/POST /kill-switch
GR06–GR10   Model registry (list, register, get, promote, deprecate)
GR11–GR13   Feature registry (list, register, get)
GR14–GR17   Inference log (query, record single outcome, bulk outcomes)
GR18–GR19   Drift (get history, run on-demand — 422 when no feature rows)
GR20–GR21   Calibration history and force snapshot
GR22–GR23   Data freshness (all stale feeds, per-symbol status)
GR24–GR27   Alerts (list active, list history, acknowledge, clear expired)
GR28–GR30   /metrics Prometheus endpoint
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, AsyncGenerator

import pytest
import pytest_asyncio

pytestmark = pytest.mark.risk_critical

# ---------------------------------------------------------------------------
# Dependency guard — skip entire module if web framework not installed.
# FastAPI and httpx are listed in requirements.txt but may not be installed
# in stripped-down CI environments.  When they are present, all 30 tests run.
# ---------------------------------------------------------------------------
fastapi    = pytest.importorskip("fastapi",    reason="fastapi not installed — skipping route tests")
httpx      = pytest.importorskip("httpx",      reason="httpx not installed — skipping route tests")

# ---------------------------------------------------------------------------
# Test app and client setup
# ---------------------------------------------------------------------------

from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport

from app.core.database import get_db
from app.api.routes.governance import router as gov_router
from app.api.metrics import router as metrics_router

_test_app = FastAPI()
_test_app.include_router(gov_router, prefix="/api/governance")
_test_app.include_router(metrics_router)


@pytest_asyncio.fixture
async def client(db_session) -> AsyncGenerator[AsyncClient, None]:
    """
    AsyncClient wired to the governance + metrics routers,
    with get_db overridden to yield the test SQLite session.
    """
    async def _override_get_db():
        yield db_session

    _test_app.dependency_overrides[get_db] = _override_get_db
    async with AsyncClient(
        transport=ASGITransport(app=_test_app),
        base_url="http://test",
    ) as ac:
        yield ac
    _test_app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _register_model(client: AsyncClient, name: str = "logistic", tag: str = "v1.0.0"):
    resp = await client.post("/api/governance/models", json={
        "model_name": name,
        "version_tag": tag,
        "n_samples": 1000,
        "n_features": 30,
    })
    assert resp.status_code == 201
    return resp.json()


# ---------------------------------------------------------------------------
# GR01–GR03: Summary and performance
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_summary_returns_200(client):
    """GR01: GET /summary returns 200 with expected top-level keys."""
    resp = await client.get("/api/governance/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert "kill_switch_active" in body
    assert "active_model" in body
    assert "calibration_health" in body
    assert "drift_summary" in body
    assert "active_critical_alerts" in body
    assert "inference_count_24h" in body


@pytest.mark.asyncio
async def test_get_summary_kill_switch_default_false(client):
    """GR02: kill switch is inactive by default."""
    resp = await client.get("/api/governance/summary")
    assert resp.status_code == 200
    assert resp.json()["kill_switch_active"] is False


@pytest.mark.asyncio
async def test_get_performance_returns_200(client):
    """GR03: GET /performance/SPY returns 200 with accuracy and calibration fields."""
    resp = await client.get("/api/governance/performance/SPY")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "SPY"
    assert "accuracy" in body
    assert "calibration_trend" in body


# ---------------------------------------------------------------------------
# GR04–GR05: Kill switch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_kill_switch_state(client):
    """GR04: GET /kill-switch returns current state."""
    resp = await client.get("/api/governance/kill-switch")
    assert resp.status_code == 200
    body = resp.json()
    assert "active" in body
    assert body["active"] is False


@pytest.mark.asyncio
async def test_post_kill_switch_activates(client):
    """GR05: POST /kill-switch with active=true activates the switch."""
    resp = await client.post("/api/governance/kill-switch", json={
        "active": True,
        "reason": "Emergency halt — test",
        "by": "test_runner",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["active"] is True
    assert "Emergency halt" in (body.get("reason") or "")

    # Deactivate for cleanliness
    await client.post("/api/governance/kill-switch", json={
        "active": False, "by": "test_runner", "reason": "cleanup"
    })


# ---------------------------------------------------------------------------
# GR06–GR10: Model registry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_models_empty(client):
    """GR06: GET /models returns empty list when no models registered."""
    resp = await client.get("/api/governance/models")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_register_model(client):
    """GR07: POST /models creates a staging model version."""
    body = await _register_model(client)
    assert body["model_name"] == "logistic"
    assert body["version_tag"] == "v1.0.0"
    assert body["status"] == "staging"


@pytest.mark.asyncio
async def test_get_model_by_id(client):
    """GR08: GET /models/{id} returns the registered model."""
    created = await _register_model(client, tag="v1.0.1")
    resp = await client.get(f"/api/governance/models/{created['id']}")
    assert resp.status_code == 200
    assert resp.json()["version_tag"] == "v1.0.1"


@pytest.mark.asyncio
async def test_get_model_not_found(client):
    """GR08b: GET /models/9999 returns 404."""
    resp = await client.get("/api/governance/models/9999")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_promote_model(client):
    """GR09: POST /models/{id}/promote moves staging → active."""
    created = await _register_model(client, tag="v2.0.0")
    resp = await client.post(f"/api/governance/models/{created['id']}/promote", json={})
    assert resp.status_code == 200
    assert resp.json()["status"] == "active"


@pytest.mark.asyncio
async def test_deprecate_model(client):
    """GR10: POST /models/{id}/deprecate moves model → deprecated."""
    created = await _register_model(client, tag="v3.0.0")
    resp = await client.post(
        f"/api/governance/models/{created['id']}/deprecate",
        json={"reason": "superseded by v4"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "deprecated"


# ---------------------------------------------------------------------------
# GR11–GR13: Feature registry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_features_empty(client):
    """GR11: GET /features returns empty list initially."""
    resp = await client.get("/api/governance/features")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_register_feature_manifest(client):
    """GR12: POST /features registers a manifest (idempotent)."""
    manifest = {
        "manifest_hash": "abc123def456",
        "pipeline_version": 3,
        "feature_list": [{"name": "rsi_14", "version": 1, "group": "momentum"}],
        "description": "baseline features",
    }
    resp = await client.post("/api/governance/features", json=manifest)
    assert resp.status_code == 201
    body = resp.json()
    assert body["manifest_hash"] == "abc123def456"
    assert body["feature_count"] == 1

    # Idempotent — second call returns same record
    resp2 = await client.post("/api/governance/features", json=manifest)
    assert resp2.status_code == 201
    assert resp2.json()["manifest_hash"] == "abc123def456"


@pytest.mark.asyncio
async def test_get_feature_manifest(client):
    """GR13: GET /features/{hash} returns the manifest."""
    await client.post("/api/governance/features", json={
        "manifest_hash": "deadbeef0001",
        "pipeline_version": 1,
        "feature_list": [{"name": "macd", "version": 1, "group": "trend"}],
    })
    resp = await client.get("/api/governance/features/deadbeef0001")
    assert resp.status_code == 200
    assert resp.json()["manifest_hash"] == "deadbeef0001"


# ---------------------------------------------------------------------------
# GR14–GR17: Inference log
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_inference_log_empty(client):
    """GR14: GET /inference-log returns empty list when no events."""
    resp = await client.get("/api/governance/inference-log")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_record_single_outcome(client, db_session):
    """GR15: POST /inference-log/{event_id}/outcome back-fills actual_outcome."""
    from app.governance.models import InferenceEvent
    event = InferenceEvent(
        symbol="SPY",
        bar_open_time=datetime(2024, 1, 2, 10, 0),
        inference_ts=1704182400,
        action="buy",
    )
    db_session.add(event)
    await db_session.flush()
    await db_session.commit()

    resp = await client.post(
        f"/api/governance/inference-log/{event.id}/outcome",
        json={"actual_outcome": 1},
    )
    assert resp.status_code == 200
    assert resp.json()["actual_outcome"] == 1


@pytest.mark.asyncio
async def test_record_single_outcome_not_found(client):
    """GR16: POST /inference-log/9999/outcome returns 404."""
    resp = await client.post(
        "/api/governance/inference-log/9999/outcome",
        json={"actual_outcome": 1},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_bulk_outcome_recording(client, db_session):
    """GR17: POST /inference-log/bulk-outcomes updates multiple rows."""
    from app.governance.models import InferenceEvent
    bar1 = datetime(2024, 1, 3, 10, 0)
    bar2 = datetime(2024, 1, 3, 10, 5)

    for bar_t, action in [(bar1, "buy"), (bar2, "sell")]:
        event = InferenceEvent(
            symbol="SPY", bar_open_time=bar_t,
            inference_ts=1704182400, action=action,
        )
        db_session.add(event)
    await db_session.flush()
    await db_session.commit()

    resp = await client.post("/api/governance/inference-log/bulk-outcomes", json={
        "outcomes": [
            {"symbol": "SPY", "bar_open_time": bar1.isoformat(), "actual_outcome": 1},
            {"symbol": "SPY", "bar_open_time": bar2.isoformat(), "actual_outcome": 0},
        ]
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_submitted"] == 2
    assert body["total_updated"] == 2
    assert body["by_symbol"]["SPY"] == 2


# ---------------------------------------------------------------------------
# GR18–GR19: Drift
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_drift_empty(client):
    """GR18: GET /drift/SPY returns empty snapshots list when none exist."""
    resp = await client.get("/api/governance/drift/SPY")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "SPY"
    assert body["count"] == 0
    assert body["snapshots"] == []


@pytest.mark.asyncio
async def test_run_drift_returns_422_when_no_feature_rows(client):
    """GR19: POST /drift/run returns 422 when the feature store is empty."""
    resp = await client.post("/api/governance/drift/run", json={
        "symbol": "SPY",
        "window_bars": 200,
    })
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GR20–GR21: Calibration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_calibration_history_empty(client):
    """GR20: GET /calibration/SPY returns empty list initially."""
    resp = await client.get("/api/governance/calibration/SPY")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_force_calibration_snapshot(client):
    """GR21: POST /calibration/SPY/snapshot writes a row (tracker via mock)."""
    from dataclasses import dataclass

    @dataclass
    class _Stats:
        rolling_brier: float = 0.23
        baseline_brier: float = 0.25
        degradation_factor: float = 1.0
        ece_recent: float = 0.04
        calibration_health: str = "good"
        needs_retrain: bool = False
        retrain_reason: str = None
        window_size: int = 50
        reliability_bins: list = None
        reliability_mean_pred: list = None
        reliability_frac_pos: list = None

    mock_tracker = type("T", (), {"get_stats": lambda self, sym: _Stats()})()

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(
            "app.inference.confidence_tracker.get_tracker",
            lambda: mock_tracker,
        )
        resp = await client.post("/api/governance/calibration/SPY/snapshot")

    assert resp.status_code == 201
    body = resp.json()
    assert body["symbol"] == "SPY"
    assert body["calibration_health"] == "good"


# ---------------------------------------------------------------------------
# GR22–GR23: Data freshness
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_all_stale_feeds_empty(client):
    """GR22: GET /freshness returns empty list when no stale checks in DB."""
    resp = await client.get("/api/governance/freshness")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_get_symbol_freshness(client):
    """GR23: GET /freshness/SPY returns sources dict (may be empty)."""
    resp = await client.get("/api/governance/freshness/SPY")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "SPY"
    assert "sources" in body
    assert "any_stale" in body


# ---------------------------------------------------------------------------
# GR24–GR27: Alerts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_active_alerts_empty(client):
    """GR24: GET /alerts returns empty list when no active alerts."""
    resp = await client.get("/api/governance/alerts")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_alert_history(client, db_session):
    """GR25: GET /alerts?active_only=false includes inactive alerts."""
    from app.governance.alerts import GovernanceAlertService, GovernanceAlertType
    await GovernanceAlertService.raise_alert(
        db_session,
        alert_type=GovernanceAlertType.FEED_STALE,
        title="test alert",
        symbol="SPY",
        dedup_key="test:SPY",
    )
    await db_session.commit()

    resp = await client.get("/api/governance/alerts?active_only=false")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


@pytest.mark.asyncio
async def test_acknowledge_alert(client, db_session):
    """GR26: POST /alerts/{id}/acknowledge marks alert inactive."""
    from app.governance.alerts import GovernanceAlertService, GovernanceAlertType
    alert = await GovernanceAlertService.raise_alert(
        db_session,
        alert_type=GovernanceAlertType.DRIFT_MODERATE,
        title="drift alert",
        symbol="QQQ",
        dedup_key="drift_moderate:QQQ:test",
    )
    await db_session.commit()

    resp = await client.post(
        f"/api/governance/alerts/{alert.id}/acknowledge",
        json={"acknowledged_by": "test_operator"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_active"] is False
    assert body["acknowledged_by"] == "test_operator"


@pytest.mark.asyncio
async def test_clear_expired_alerts(client, db_session):
    """GR27: POST /alerts/clear-expired removes past-expiry alerts."""
    from app.governance.models import GovernanceAlert
    from datetime import timedelta

    expired = GovernanceAlert(
        alert_type="feed_stale",
        severity="warning",
        title="stale",
        triggered_at=datetime.utcnow() - timedelta(hours=5),
        expires_at=datetime.utcnow() - timedelta(hours=1),
        is_active=True,
        dedup_key="stale:test:clear",
    )
    db_session.add(expired)
    await db_session.flush()
    await db_session.commit()

    resp = await client.post("/api/governance/alerts/clear-expired")
    assert resp.status_code == 200
    assert resp.json()["cleared"] >= 1


# ---------------------------------------------------------------------------
# GR28–GR30: /metrics Prometheus endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_metrics_endpoint_returns_200(client):
    """GR28: GET /metrics returns 200 with Prometheus content-type."""
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_metrics_contains_kill_switch_metric(client):
    """GR29: /metrics output contains options_research_kill_switch_active."""
    resp = await client.get("/metrics")
    assert "options_research_kill_switch_active" in resp.text


@pytest.mark.asyncio
async def test_metrics_contains_scrape_duration(client):
    """GR30: /metrics output contains the self-instrumentation scrape duration gauge."""
    resp = await client.get("/metrics")
    assert "options_research_metrics_scrape_duration_ms" in resp.text
    # Verify Prometheus exposition format structure
    assert "# HELP" in resp.text
    assert "# TYPE" in resp.text
