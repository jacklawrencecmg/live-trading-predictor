"""
SQLAlchemy ORM models for the governance system.

Tables
------
model_versions          Versioned model artifact registry
feature_versions        Feature manifest history (hash → definition list)
inference_events        Persistent per-prediction log with outcome column
drift_snapshots         Population Stability Index snapshots per symbol
calibration_snapshots   Periodic calibration health checkpoints
data_freshness_checks   Feed staleness audit records
governance_alerts       Persistent alerts with acknowledge / expire lifecycle
kill_switch_state       Singleton row (id=1) for persisted kill switch state
"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)

from app.core.database import Base


# ---------------------------------------------------------------------------
# Model version registry
# ---------------------------------------------------------------------------

class ModelVersion(Base):
    """
    One row per trained model artifact.

    Lifecycle:  staging  →  active  →  deprecated
    Only one row per model_name may have status='active' at a time;
    that invariant is enforced by ModelRegistryService, not a DB constraint,
    to allow atomic swap during promotion.
    """
    __tablename__ = "model_versions"

    id                     = Column(Integer, primary_key=True, autoincrement=True)
    model_name             = Column(String(64), nullable=False, index=True)
    version_tag            = Column(String(64), nullable=False)
    status                 = Column(String(32), nullable=False, default="staging", index=True)
    # "staging" | "active" | "deprecated"

    trained_at             = Column(DateTime, nullable=False, default=datetime.utcnow)
    training_symbol        = Column(String(16), nullable=True)   # "SPY", "QQQ", "multi", …

    n_samples              = Column(Integer, nullable=True)
    n_features             = Column(Integer, nullable=True)
    feature_manifest_hash  = Column(String(64), nullable=True, index=True)

    # Walk-forward fold metrics (JSON-serialised dict)
    train_metrics_json     = Column(Text, nullable=True)

    # Calibration metadata
    calibration_kind       = Column(String(32), nullable=True)   # "isotonic"|"platt"|"identity"
    calibration_ece_at_fit = Column(Float, nullable=True)

    # Artifact provenance
    artifact_dir           = Column(String(256), nullable=True)  # relative to model_artifacts/
    artifact_sha256        = Column(String(64), nullable=True)   # SHA-256 of model.pkl

    # Governance lifecycle
    promoted_at            = Column(DateTime, nullable=True)
    deprecated_at          = Column(DateTime, nullable=True)
    notes                  = Column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("model_name", "version_tag", name="uq_model_version_tag"),
    )


# ---------------------------------------------------------------------------
# Feature version registry
# ---------------------------------------------------------------------------

class FeatureVersion(Base):
    """
    One row per unique feature manifest (identified by manifest_hash).

    The manifest_hash is derived from (feature_name, feature_version) pairs
    sorted deterministically — see feature_pipeline/registry.py.
    Changing any feature formula bumps its version, which changes the hash,
    which triggers a new row here and invalidates all cached FeatureRow entries.
    """
    __tablename__ = "feature_versions"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    manifest_hash     = Column(String(64), nullable=False, unique=True, index=True)
    pipeline_version  = Column(Integer, nullable=False)
    feature_count     = Column(Integer, nullable=False)
    # JSON: [{name, version, group, description}]
    feature_list_json = Column(Text, nullable=False)
    # JSON: {feature_name: {mean, std, p5, p25, p50, p75, p95}} — from training data
    reference_stats_json = Column(Text, nullable=True)
    recorded_at       = Column(DateTime, nullable=False, default=datetime.utcnow)
    description       = Column(Text, nullable=True)   # human note on what changed


# ---------------------------------------------------------------------------
# Inference event log
# ---------------------------------------------------------------------------

class InferenceEvent(Base):
    """
    One row per call to run_inference() that returns a result.
    Stores the full 4-layer uncertainty bundle for retrospective analysis.

    actual_outcome is populated asynchronously once the target bar closes:
        1 = price moved up, 0 = price moved down, NULL = pending.
    """
    __tablename__ = "inference_events"

    id                    = Column(BigInteger, primary_key=True, autoincrement=True)
    request_id            = Column(String(32), nullable=True)    # X-Request-ID header
    symbol                = Column(String(16), nullable=False, index=True)
    bar_open_time         = Column(DateTime, nullable=True)      # bar that triggered inference
    inference_ts          = Column(BigInteger, nullable=False)   # unix timestamp

    # Model provenance
    model_name            = Column(String(64), nullable=True)
    model_version_id      = Column(Integer, nullable=True)       # → model_versions.id
    feature_snapshot_id   = Column(String(16), nullable=True)    # SHA-256 prefix of feature vector
    manifest_hash         = Column(String(64), nullable=True)

    # Layer 1 — raw
    prob_up               = Column(Float, nullable=True)
    prob_down             = Column(Float, nullable=True)

    # Layer 2 — calibrated
    calibrated_prob_up    = Column(Float, nullable=True)
    calibration_available = Column(Boolean, nullable=True)

    # Layer 3 — tradeable
    tradeable_confidence  = Column(Float, nullable=True)
    degradation_factor    = Column(Float, nullable=True)

    # Layer 4 — action
    action                = Column(String(16), nullable=True, index=True)   # buy|sell|abstain
    abstain_reason        = Column(String(128), nullable=True)

    # Calibration health context
    calibration_health    = Column(String(16), nullable=True)
    ece_recent            = Column(Float, nullable=True)
    rolling_brier         = Column(Float, nullable=True)
    expected_move_pct     = Column(Float, nullable=True)
    regime                = Column(String(32), nullable=True)
    options_stale         = Column(Boolean, nullable=False, default=False)

    # Outcome (back-filled once bar resolves)
    actual_outcome        = Column(SmallInteger, nullable=True)   # 1=up 0=down NULL=pending
    outcome_recorded_at   = Column(DateTime, nullable=True)

    created_at            = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_inf_symbol_created", "symbol", "created_at"),
        Index("ix_inf_bar_time",       "symbol", "bar_open_time"),
    )


# ---------------------------------------------------------------------------
# Drift snapshots
# ---------------------------------------------------------------------------

class DriftSnapshot(Base):
    """
    Population Stability Index (PSI) snapshot for feature distributions.

    PSI < 0.10   → no significant drift   (drift_level = 'none')
    PSI 0.10-0.25 → moderate drift        (drift_level = 'moderate')
    PSI > 0.25   → significant drift      (drift_level = 'high')

    Reference distribution comes from FeatureVersion.reference_stats_json
    when available; otherwise bins are derived from [expected_min, expected_max]
    from the feature registry.
    """
    __tablename__ = "drift_snapshots"

    id                       = Column(Integer, primary_key=True, autoincrement=True)
    symbol                   = Column(String(16), nullable=False, index=True)
    computed_at              = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    window_bars              = Column(Integer, nullable=False)    # number of recent bars used
    manifest_hash            = Column(String(64), nullable=True)

    # Per-feature PSI values — JSON: {"rsi_14": 0.03, "volume_ratio": 0.28, ...}
    psi_by_feature_json      = Column(Text, nullable=False)

    # Summary
    max_psi                  = Column(Float, nullable=False)
    mean_psi                 = Column(Float, nullable=False)
    # JSON: ["volume_ratio", "atr_norm"] — features with PSI > 0.25
    high_drift_features_json = Column(Text, nullable=True)

    drift_level              = Column(String(16), nullable=False, index=True)
    # "none" | "moderate" | "high"

    alert_raised             = Column(Boolean, nullable=False, default=False)


# ---------------------------------------------------------------------------
# Calibration snapshots
# ---------------------------------------------------------------------------

class CalibrationSnapshot(Base):
    """
    Point-in-time snapshot of the rolling calibration health for a symbol.
    Written by CalibrationMonitor after every N inference events or on schedule.
    Provides a queryable time series of model health — detects slow degradation.
    """
    __tablename__ = "calibration_snapshots"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    symbol             = Column(String(16), nullable=False, index=True)
    snapshot_at        = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    model_name         = Column(String(64), nullable=True)
    window_size        = Column(Integer, nullable=True)

    rolling_brier      = Column(Float, nullable=True)
    baseline_brier     = Column(Float, nullable=True)
    degradation_factor = Column(Float, nullable=True)
    ece_recent         = Column(Float, nullable=True)
    calibration_health = Column(String(16), nullable=True)

    needs_retrain      = Column(Boolean, nullable=False, default=False)
    retrain_reason     = Column(Text, nullable=True)

    # JSON: {bins: [...], mean_predicted: [...], fraction_positive: [...]}
    reliability_json   = Column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Data freshness checks
# ---------------------------------------------------------------------------

class DataFreshnessCheck(Base):
    """
    Records each staleness check for each data source.
    Sources: 'quote_feed', 'options_chain', 'candle_data', 'model_artifact'.
    """
    __tablename__ = "data_freshness_checks"

    id                           = Column(Integer, primary_key=True, autoincrement=True)
    symbol                       = Column(String(16), nullable=False, index=True)
    source                       = Column(String(64), nullable=False, index=True)

    checked_at                   = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    last_data_ts                 = Column(DateTime, nullable=True)
    age_seconds                  = Column(Float, nullable=True)

    is_stale                     = Column(Boolean, nullable=False)
    staleness_threshold_seconds  = Column(Float, nullable=False)

    alert_raised                 = Column(Boolean, nullable=False, default=False)


# ---------------------------------------------------------------------------
# Governance alerts
# ---------------------------------------------------------------------------

class GovernanceAlert(Base):
    """
    Persisted alert with full lifecycle: active → acknowledged | expired.

    dedup_key prevents alert spam: if a dedup_key already has an active alert,
    a new one is not created — only the triggered_at is bumped.
    """
    __tablename__ = "governance_alerts"

    id              = Column(BigInteger, primary_key=True, autoincrement=True)
    alert_type      = Column(String(64), nullable=False, index=True)
    severity        = Column(String(16), nullable=False, index=True)
    # "info" | "warning" | "critical"
    symbol          = Column(String(16), nullable=True, index=True)

    title           = Column(String(256), nullable=False)
    details_json    = Column(Text, nullable=True)

    triggered_at    = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    expires_at      = Column(DateTime, nullable=True)

    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by = Column(String(64), nullable=True)

    dedup_key       = Column(String(128), nullable=True, index=True)
    is_active       = Column(Boolean, nullable=False, default=True, index=True)


# ---------------------------------------------------------------------------
# Kill switch state
# ---------------------------------------------------------------------------

class KillSwitchState(Base):
    """
    Singleton row (id always = 1).
    Persisted kill switch — survives process restarts.
    Toggling is idempotent: activating an already-active switch is a no-op
    (unless reason changes); deactivating an already-inactive switch likewise.
    """
    __tablename__ = "kill_switch_state"

    id           = Column(Integer, primary_key=True, default=1)
    active       = Column(Boolean, nullable=False, default=False)
    reason       = Column(Text, nullable=True)
    activated_at = Column(DateTime, nullable=True)
    activated_by = Column(String(64), nullable=True)
    updated_at   = Column(DateTime, nullable=False, default=datetime.utcnow)
