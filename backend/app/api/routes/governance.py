"""
Governance REST endpoints.

All endpoints are under /api/governance/.

Security note:
    Kill switch endpoints should be protected by authentication middleware
    before production deployment.  Currently open to match the rest of the
    dev API (no auth on any route).  See docs/governance.md §Security.

Endpoint map:
    GET  /summary                       — one-page governance health view
    GET  /performance/{symbol}          — rolling performance dashboard

    GET  /kill-switch                   — current state
    POST /kill-switch                   — activate or deactivate

    GET  /models                        — list all model versions
    POST /models                        — register new model version
    GET  /models/{version_id}           — get one version
    POST /models/{version_id}/promote   — staging → active
    POST /models/{version_id}/deprecate — active/staging → deprecated

    GET  /features                      — list feature manifests
    POST /features                      — register manifest
    GET  /features/{manifest_hash}      — get one manifest

    GET  /inference-log                 — query inference events
    POST /inference-log/{event_id}/outcome  — record actual outcome

    GET  /drift/{symbol}                — latest drift snapshot
    POST /drift/run                     — run drift check on demand

    GET  /calibration/{symbol}          — calibration history
    POST /calibration/{symbol}/snapshot — force a snapshot now

    GET  /freshness                     — all stale feeds
    GET  /freshness/{symbol}            — per-symbol freshness

    GET  /alerts                        — active and recent alerts
    POST /alerts/{alert_id}/acknowledge — acknowledge an alert
    POST /alerts/clear-expired          — clear all expired alerts
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.governance.alerts import GovernanceAlertService
from app.governance.calibration import CalibrationMonitor
from app.governance.dashboard import GovernanceDashboard
from app.governance.drift import DriftMonitor
from app.governance.freshness import DataFreshnessService
from app.governance.inference_log import InferenceLogService
from app.governance.kill_switch import KillSwitchService
from app.governance.registry import FeatureRegistryService, ModelRegistryService
from app.governance.schemas import (
    AcknowledgeRequest,
    CalibrationSnapshotOut,
    DeprecateRequest,
    DriftSnapshotOut,
    FeatureVersionOut,
    FreshnessCheckOut,
    GovernanceAlertOut,
    GovernanceSummaryOut,
    InferenceEventOut,
    KillSwitchOut,
    KillSwitchRequest,
    ModelVersionOut,
    PromoteRequest,
    RecordOutcomeRequest,
    RegisterFeatureManifestRequest,
    RegisterModelRequest,
    RunDriftCheckRequest,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_out(v) -> ModelVersionOut:
    metrics = None
    if v.train_metrics_json:
        try:
            metrics = json.loads(v.train_metrics_json)
        except Exception:
            pass
    return ModelVersionOut(
        id=v.id, model_name=v.model_name, version_tag=v.version_tag,
        status=v.status, trained_at=v.trained_at, training_symbol=v.training_symbol,
        n_samples=v.n_samples, n_features=v.n_features,
        feature_manifest_hash=v.feature_manifest_hash,
        calibration_kind=v.calibration_kind,
        calibration_ece_at_fit=v.calibration_ece_at_fit,
        artifact_dir=v.artifact_dir, artifact_sha256=v.artifact_sha256,
        promoted_at=v.promoted_at, deprecated_at=v.deprecated_at,
        notes=v.notes, train_metrics=metrics,
    )


def _feature_out(v) -> FeatureVersionOut:
    feat_list = None
    if v.feature_list_json:
        try:
            feat_list = json.loads(v.feature_list_json)
        except Exception:
            pass
    return FeatureVersionOut(
        id=v.id, manifest_hash=v.manifest_hash, pipeline_version=v.pipeline_version,
        feature_count=v.feature_count, recorded_at=v.recorded_at,
        description=v.description, feature_list=feat_list,
    )


def _drift_out(d) -> DriftSnapshotOut:
    psi_map: Dict[str, float] = {}
    if d.psi_by_feature_json:
        try:
            raw = json.loads(d.psi_by_feature_json)
            psi_map = {k: (v or 0.0) for k, v in raw.items()}
        except Exception:
            pass
    high_feats: List[str] = []
    if d.high_drift_features_json:
        try:
            high_feats = json.loads(d.high_drift_features_json)
        except Exception:
            pass
    return DriftSnapshotOut(
        id=d.id, symbol=d.symbol, computed_at=d.computed_at,
        window_bars=d.window_bars, manifest_hash=d.manifest_hash,
        psi_by_feature=psi_map, max_psi=d.max_psi, mean_psi=d.mean_psi,
        high_drift_features=high_feats, drift_level=d.drift_level,
        alert_raised=d.alert_raised,
    )


def _alert_out(a) -> GovernanceAlertOut:
    details = None
    if a.details_json:
        try:
            details = json.loads(a.details_json)
        except Exception:
            pass
    return GovernanceAlertOut(
        id=a.id, alert_type=a.alert_type, severity=a.severity,
        symbol=a.symbol, title=a.title, details=details,
        triggered_at=a.triggered_at, expires_at=a.expires_at,
        acknowledged_at=a.acknowledged_at, acknowledged_by=a.acknowledged_by,
        dedup_key=a.dedup_key, is_active=a.is_active,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@router.get("/summary", response_model=Dict[str, Any])
async def get_summary(db: AsyncSession = Depends(get_db)):
    """
    One-page governance health summary.
    Aggregates kill switch, model registry, calibration, drift, freshness,
    and alert counts into a single response.
    """
    return await GovernanceDashboard.summary(db)


@router.get("/performance/{symbol}", response_model=Dict[str, Any])
async def get_performance(
    symbol: str,
    window_days: int = Query(7, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
):
    """Rolling performance dashboard for one symbol."""
    return await GovernanceDashboard.rolling_performance(db, symbol.upper(), window_days)


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------

@router.get("/kill-switch", response_model=KillSwitchOut)
async def get_kill_switch(db: AsyncSession = Depends(get_db)):
    """Return current kill switch state."""
    row = await KillSwitchService.get_state(db)
    return KillSwitchOut(
        active=row.active, reason=row.reason, activated_at=row.activated_at,
        activated_by=row.activated_by, updated_at=row.updated_at,
    )


@router.post("/kill-switch", response_model=KillSwitchOut)
async def set_kill_switch(
    req: KillSwitchRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Activate or deactivate the kill switch.

    - active=true  : halts all trading signals
    - active=false : resumes trading (requires explicit deactivation reason)
    """
    row = await KillSwitchService.toggle(
        db,
        active=req.active,
        reason=req.reason,
        by=req.by,
    )
    await db.commit()
    return KillSwitchOut(
        active=row.active, reason=row.reason, activated_at=row.activated_at,
        activated_by=row.activated_by, updated_at=row.updated_at,
    )


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@router.get("/models", response_model=List[ModelVersionOut])
async def list_models(
    model_name: Optional[str] = None,
    status: Optional[str] = Query(None, pattern="^(staging|active|deprecated)$"),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    rows = await ModelRegistryService.list_versions(
        db, model_name=model_name, status=status, limit=limit
    )
    return [_model_out(r) for r in rows]


@router.post("/models", response_model=ModelVersionOut, status_code=201)
async def register_model(
    req: RegisterModelRequest,
    db: AsyncSession = Depends(get_db),
):
    """Register a new model version in staging status."""
    try:
        row = await ModelRegistryService.register(
            db,
            model_name=req.model_name,
            version_tag=req.version_tag,
            training_symbol=req.training_symbol,
            n_samples=req.n_samples,
            n_features=req.n_features,
            feature_manifest_hash=req.feature_manifest_hash,
            train_metrics=req.train_metrics,
            calibration_kind=req.calibration_kind,
            calibration_ece_at_fit=req.calibration_ece_at_fit,
            artifact_dir=req.artifact_dir,
            notes=req.notes,
        )
        await db.commit()
        return _model_out(row)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/models/{version_id}", response_model=ModelVersionOut)
async def get_model(
    version_id: int,
    db: AsyncSession = Depends(get_db),
):
    row = await ModelRegistryService.get_by_id(db, version_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Model version not found")
    return _model_out(row)


@router.post("/models/{version_id}/promote", response_model=ModelVersionOut)
async def promote_model(
    version_id: int,
    req: PromoteRequest,
    db: AsyncSession = Depends(get_db),
):
    """Promote a staging model to active.  Deprecates the current active version."""
    try:
        row = await ModelRegistryService.promote(db, version_id, notes=req.notes)
        await db.commit()
        return _model_out(row)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/models/{version_id}/deprecate", response_model=ModelVersionOut)
async def deprecate_model(
    version_id: int,
    req: DeprecateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Deprecate a model version."""
    try:
        row = await ModelRegistryService.deprecate(db, version_id, reason=req.reason)
        await db.commit()
        return _model_out(row)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------

@router.get("/features", response_model=List[FeatureVersionOut])
async def list_features(
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    rows = await FeatureRegistryService.list_all(db, limit=limit)
    return [_feature_out(r) for r in rows]


@router.post("/features", response_model=FeatureVersionOut, status_code=201)
async def register_feature_manifest(
    req: RegisterFeatureManifestRequest,
    db: AsyncSession = Depends(get_db),
):
    """Register a feature manifest (idempotent by manifest_hash)."""
    row = await FeatureRegistryService.ensure_manifest(
        db,
        manifest_hash=req.manifest_hash,
        pipeline_version=req.pipeline_version,
        feature_list=req.feature_list,
        reference_stats=req.reference_stats,
        description=req.description,
    )
    await db.commit()
    return _feature_out(row)


@router.get("/features/{manifest_hash}", response_model=FeatureVersionOut)
async def get_feature_manifest(
    manifest_hash: str,
    db: AsyncSession = Depends(get_db),
):
    row = await FeatureRegistryService.get(db, manifest_hash)
    if row is None:
        raise HTTPException(status_code=404, detail="Feature manifest not found")
    return _feature_out(row)


# ---------------------------------------------------------------------------
# Inference log
# ---------------------------------------------------------------------------

@router.get("/inference-log", response_model=List[InferenceEventOut])
async def get_inference_log(
    symbol: Optional[str] = None,
    action: Optional[str] = Query(None, pattern="^(buy|sell|abstain)$"),
    pending_only: bool = False,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """
    Query the inference log.

    Use pending_only=true to find predictions that still need outcome recording.
    """
    events = await InferenceLogService.query(
        db,
        symbol=symbol.upper() if symbol else None,
        action=action,
        limit=limit,
        offset=offset,
        pending_only=pending_only,
    )
    return [InferenceEventOut.from_orm(e) for e in events]


@router.post("/inference-log/{event_id}/outcome", status_code=200)
async def record_inference_outcome(
    event_id: int,
    req: RecordOutcomeRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Record the actual price direction for a specific inference event.
    actual_outcome: 1=up, 0=down.
    """
    from sqlalchemy import select as sa_select, update as sa_update
    from app.governance.models import InferenceEvent
    result = await db.execute(
        sa_select(InferenceEvent).where(InferenceEvent.id == event_id)
    )
    ev = result.scalar_one_or_none()
    if ev is None:
        raise HTTPException(status_code=404, detail="Inference event not found")
    ev.actual_outcome = req.actual_outcome
    ev.outcome_recorded_at = datetime.utcnow()
    await db.commit()
    return {"id": event_id, "actual_outcome": req.actual_outcome}


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------

@router.get("/drift/{symbol}", response_model=Dict[str, Any])
async def get_drift(
    symbol: str,
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    """Return the most recent drift snapshot(s) for a symbol."""
    snapshots = await DriftMonitor.get_history(db, symbol.upper(), limit=limit)
    return {
        "symbol":   symbol.upper(),
        "count":    len(snapshots),
        "snapshots": [_drift_out(s).dict() for s in snapshots],
    }


@router.post("/drift/run", response_model=Dict[str, Any], status_code=201)
async def run_drift_check(
    req: RunDriftCheckRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Run an on-demand drift check using recent feature rows from the DB.

    Note: this endpoint queries FeatureRow from the feature store.
    If the feature store is empty (e.g. test environment), returns a 422.
    """
    symbol = req.symbol.upper()
    from app.models.feature_row import FeatureRow
    from sqlalchemy import select as sa_select
    import numpy as np

    result = await db.execute(
        sa_select(FeatureRow)
        .where(FeatureRow.symbol == symbol)
        .order_by(FeatureRow.bar_open_time.desc())
        .limit(req.window_bars)
    )
    rows = result.scalars().all()
    if len(rows) < 30:
        raise HTTPException(
            status_code=422,
            detail=f"Not enough feature rows for {symbol} (found {len(rows)}, need ≥30)",
        )

    # Build feature matrix
    feature_names: List[str] = []
    matrices = []
    for r in rows:
        try:
            feats = json.loads(r.features_json) if r.features_json else {}
            if not feature_names:
                feature_names = list(feats.keys())
            matrices.append([feats.get(k, float("nan")) for k in feature_names])
        except Exception:
            pass

    if not matrices:
        raise HTTPException(status_code=422, detail="Could not parse feature rows")

    matrix = np.array(matrices, dtype=float)

    # Get reference stats if available
    from app.feature_pipeline.registry import MANIFEST_HASH
    fv = await FeatureRegistryService.get(db, MANIFEST_HASH)
    ref_stats = None
    if fv and fv.reference_stats_json:
        try:
            ref_stats = json.loads(fv.reference_stats_json)
        except Exception:
            pass

    psi_map = DriftMonitor.compute_psi_from_matrix(matrix, feature_names, ref_stats)
    snap = await DriftMonitor.record_snapshot(
        db,
        symbol=symbol,
        psi_by_feature=psi_map,
        window_bars=len(rows),
        manifest_hash=MANIFEST_HASH,
    )

    # Raise alert if drift detected
    if snap.drift_level != "none":
        high_feats = json.loads(snap.high_drift_features_json or "[]")
        await GovernanceAlertService.alert_drift(
            db, symbol, snap.drift_level, snap.max_psi, high_feats
        )
        snap.alert_raised = True

    await db.commit()
    return _drift_out(snap).dict()


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@router.get("/calibration/{symbol}", response_model=List[CalibrationSnapshotOut])
async def get_calibration_history(
    symbol: str,
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    snaps = await CalibrationMonitor.get_history(db, symbol.upper(), limit=limit)
    return [CalibrationSnapshotOut.from_orm(s) for s in snaps]


@router.post("/calibration/{symbol}/snapshot", response_model=CalibrationSnapshotOut, status_code=201)
async def force_calibration_snapshot(
    symbol: str,
    db: AsyncSession = Depends(get_db),
):
    """Force a calibration snapshot now by reading the live ConfidenceTracker."""
    from app.inference.confidence_tracker import get_tracker
    tracker = get_tracker()
    stats = tracker.get_stats(symbol.upper())

    snap = await CalibrationMonitor.record_snapshot(
        db, symbol=symbol.upper(), tracker_stats=stats
    )

    # Alert if degraded
    if stats.calibration_health == "degraded":
        await GovernanceAlertService.alert_calibration_degraded(
            db, symbol.upper(), stats.calibration_health, stats.rolling_brier
        )
    if stats.needs_retrain:
        await GovernanceAlertService.alert_retrain_needed(
            db, symbol.upper(), stats.retrain_reason or "threshold exceeded"
        )

    await db.commit()
    return CalibrationSnapshotOut.from_orm(snap)


# ---------------------------------------------------------------------------
# Data freshness
# ---------------------------------------------------------------------------

@router.get("/freshness", response_model=List[Dict[str, Any]])
async def get_all_stale_feeds(
    since_minutes: int = Query(15, ge=1, le=60),
    db: AsyncSession = Depends(get_db),
):
    """Return all stale feed checks in the last `since_minutes`."""
    checks = await DataFreshnessService.get_stale_feeds(db, since_minutes=since_minutes)
    return [
        {
            "symbol":      c.symbol,
            "source":      c.source,
            "age_seconds": c.age_seconds,
            "threshold":   c.staleness_threshold_seconds,
            "checked_at":  c.checked_at.isoformat(),
        }
        for c in checks
    ]


@router.get("/freshness/{symbol}", response_model=Dict[str, Any])
async def get_symbol_freshness(
    symbol: str,
    db: AsyncSession = Depends(get_db),
):
    """Per-source freshness status for a symbol."""
    sources = await DataFreshnessService.get_current_status(db, symbol.upper())
    any_stale = any(v["is_stale"] for v in sources.values())
    return {
        "symbol":    symbol.upper(),
        "sources":   sources,
        "any_stale": any_stale,
        "checked_at": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

@router.get("/alerts", response_model=List[GovernanceAlertOut])
async def get_alerts(
    active_only: bool = True,
    severity: Optional[str] = Query(None, pattern="^(info|warning|critical)$"),
    alert_type: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    if active_only:
        alerts = await GovernanceAlertService.get_active(
            db, severity=severity, alert_type=alert_type,
            symbol=symbol.upper() if symbol else None,
        )
        return [_alert_out(a) for a in alerts[:limit]]
    else:
        alerts = await GovernanceAlertService.get_history(db, limit=limit, alert_type=alert_type)
        return [_alert_out(a) for a in alerts]


@router.post("/alerts/{alert_id}/acknowledge", response_model=GovernanceAlertOut)
async def acknowledge_alert(
    alert_id: int,
    req: AcknowledgeRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        row = await GovernanceAlertService.acknowledge(db, alert_id, by=req.acknowledged_by)
        await db.commit()
        return _alert_out(row)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/alerts/clear-expired", response_model=Dict[str, int])
async def clear_expired_alerts(db: AsyncSession = Depends(get_db)):
    n = await GovernanceAlertService.clear_expired(db)
    await db.commit()
    return {"cleared": n}
