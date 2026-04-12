"""
Pydantic schemas for governance API request/response bodies.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

class ModelVersionOut(BaseModel):
    id: int
    model_name: str
    version_tag: str
    status: str
    trained_at: datetime
    training_symbol: Optional[str]
    n_samples: Optional[int]
    n_features: Optional[int]
    feature_manifest_hash: Optional[str]
    calibration_kind: Optional[str]
    calibration_ece_at_fit: Optional[float]
    artifact_dir: Optional[str]
    artifact_sha256: Optional[str]
    promoted_at: Optional[datetime]
    deprecated_at: Optional[datetime]
    notes: Optional[str]
    # Metrics parsed from JSON
    train_metrics: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class RegisterModelRequest(BaseModel):
    model_name: str = Field(..., description="logistic | gbt | random_forest | naive")
    version_tag: str = Field(..., description="e.g. v1.0.0 — must be unique per model_name")
    training_symbol: Optional[str] = None
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    feature_manifest_hash: Optional[str] = None
    train_metrics: Optional[Dict[str, Any]] = None
    calibration_kind: Optional[str] = None
    calibration_ece_at_fit: Optional[float] = None
    artifact_dir: Optional[str] = None
    artifact_sha256: Optional[str] = None
    notes: Optional[str] = None


class PromoteRequest(BaseModel):
    notes: Optional[str] = None


class DeprecateRequest(BaseModel):
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------

class FeatureVersionOut(BaseModel):
    id: int
    manifest_hash: str
    pipeline_version: int
    feature_count: int
    recorded_at: datetime
    description: Optional[str]
    # feature_list parsed from JSON
    feature_list: Optional[List[Dict[str, Any]]] = None

    class Config:
        from_attributes = True


class RegisterFeatureManifestRequest(BaseModel):
    manifest_hash: str
    pipeline_version: int
    feature_list: List[Dict[str, Any]] = Field(
        ...,
        description="List of {name, version, group} dicts"
    )
    reference_stats: Optional[Dict[str, Any]] = Field(
        None,
        description="Training-time per-feature stats: {feat: {mean, std, p5, p50, p95}}"
    )
    description: Optional[str] = None


# ---------------------------------------------------------------------------
# Inference log
# ---------------------------------------------------------------------------

class InferenceEventOut(BaseModel):
    id: int
    request_id: Optional[str]
    symbol: str
    bar_open_time: Optional[datetime]
    inference_ts: int
    model_name: Optional[str]
    model_version_id: Optional[int]
    feature_snapshot_id: Optional[str]
    manifest_hash: Optional[str]
    prob_up: Optional[float]
    prob_down: Optional[float]
    calibrated_prob_up: Optional[float]
    calibration_available: Optional[bool]
    tradeable_confidence: Optional[float]
    degradation_factor: Optional[float]
    action: Optional[str]
    abstain_reason: Optional[str]
    calibration_health: Optional[str]
    ece_recent: Optional[float]
    rolling_brier: Optional[float]
    expected_move_pct: Optional[float]
    regime: Optional[str]
    options_stale: bool
    actual_outcome: Optional[int]
    outcome_recorded_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


class RecordOutcomeRequest(BaseModel):
    actual_outcome: int = Field(..., ge=0, le=1, description="1=up, 0=down")


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------

class DriftSnapshotOut(BaseModel):
    id: int
    symbol: str
    computed_at: datetime
    window_bars: int
    manifest_hash: Optional[str]
    psi_by_feature: Dict[str, float]
    max_psi: float
    mean_psi: float
    high_drift_features: List[str]
    drift_level: str
    alert_raised: bool

    class Config:
        from_attributes = True


class RunDriftCheckRequest(BaseModel):
    symbol: str
    window_bars: int = Field(200, ge=50, le=2000)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class CalibrationSnapshotOut(BaseModel):
    id: int
    symbol: str
    snapshot_at: datetime
    model_name: Optional[str]
    window_size: Optional[int]
    rolling_brier: Optional[float]
    baseline_brier: Optional[float]
    degradation_factor: Optional[float]
    ece_recent: Optional[float]
    calibration_health: Optional[str]
    needs_retrain: bool
    retrain_reason: Optional[str]

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Data freshness
# ---------------------------------------------------------------------------

class FreshnessCheckOut(BaseModel):
    id: int
    symbol: str
    source: str
    checked_at: datetime
    last_data_ts: Optional[datetime]
    age_seconds: Optional[float]
    is_stale: bool
    staleness_threshold_seconds: float
    alert_raised: bool

    class Config:
        from_attributes = True


class FreshnessStatusOut(BaseModel):
    symbol: str
    sources: Dict[str, Dict[str, Any]]
    any_stale: bool
    checked_at: datetime


# ---------------------------------------------------------------------------
# Governance alerts
# ---------------------------------------------------------------------------

class GovernanceAlertOut(BaseModel):
    id: int
    alert_type: str
    severity: str
    symbol: Optional[str]
    title: str
    details: Optional[Dict[str, Any]] = None
    triggered_at: datetime
    expires_at: Optional[datetime]
    acknowledged_at: Optional[datetime]
    acknowledged_by: Optional[str]
    dedup_key: Optional[str]
    is_active: bool

    class Config:
        from_attributes = True


class AcknowledgeRequest(BaseModel):
    acknowledged_by: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------

class KillSwitchOut(BaseModel):
    active: bool
    reason: Optional[str]
    activated_at: Optional[datetime]
    activated_by: Optional[str]
    updated_at: datetime

    class Config:
        from_attributes = True


class KillSwitchRequest(BaseModel):
    active: bool
    reason: Optional[str] = None
    by: str = Field("operator", description="Who is toggling the switch")


# ---------------------------------------------------------------------------
# Governance summary dashboard
# ---------------------------------------------------------------------------

class GovernanceSummaryOut(BaseModel):
    generated_at: datetime
    kill_switch_active: bool

    active_model: Optional[Dict[str, Any]]
    active_feature_manifest: Optional[str]

    # Per-symbol calibration health (latest snapshot)
    calibration_health: Dict[str, str]    # symbol → "good"|"fair"|"degraded"|"unknown"
    symbols_needing_retrain: List[str]

    # Drift
    drift_summary: Dict[str, str]         # symbol → "none"|"moderate"|"high"|"unknown"

    # Freshness
    stale_feeds: List[Dict[str, Any]]

    # Alerts
    active_critical_alerts: int
    active_warning_alerts: int
    recent_alert_titles: List[str]        # last 5

    # Inference volume
    inference_count_24h: int
    abstain_rate_24h: Optional[float]
