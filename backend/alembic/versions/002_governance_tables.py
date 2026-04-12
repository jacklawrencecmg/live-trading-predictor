"""Add governance tables

Revision ID: 002_governance
Revises: 001_initial
Create Date: 2025-04-11

New tables
----------
model_versions
feature_versions
inference_events
drift_snapshots
calibration_snapshots
data_freshness_checks
governance_alerts
kill_switch_state
"""

from alembic import op
import sqlalchemy as sa

revision = "002_governance"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── model_versions ───────────────────────────────────────────────────────
    op.create_table(
        "model_versions",
        sa.Column("id",                     sa.Integer,     primary_key=True, autoincrement=True),
        sa.Column("model_name",             sa.String(64),  nullable=False),
        sa.Column("version_tag",            sa.String(64),  nullable=False),
        sa.Column("status",                 sa.String(32),  nullable=False, server_default="staging"),
        sa.Column("trained_at",             sa.DateTime,    nullable=False),
        sa.Column("training_symbol",        sa.String(16),  nullable=True),
        sa.Column("n_samples",              sa.Integer,     nullable=True),
        sa.Column("n_features",             sa.Integer,     nullable=True),
        sa.Column("feature_manifest_hash",  sa.String(64),  nullable=True),
        sa.Column("train_metrics_json",     sa.Text,        nullable=True),
        sa.Column("calibration_kind",       sa.String(32),  nullable=True),
        sa.Column("calibration_ece_at_fit", sa.Float,       nullable=True),
        sa.Column("artifact_dir",           sa.String(256), nullable=True),
        sa.Column("artifact_sha256",        sa.String(64),  nullable=True),
        sa.Column("promoted_at",            sa.DateTime,    nullable=True),
        sa.Column("deprecated_at",          sa.DateTime,    nullable=True),
        sa.Column("notes",                  sa.Text,        nullable=True),
        sa.UniqueConstraint("model_name", "version_tag", name="uq_model_version_tag"),
    )
    op.create_index("ix_mv_model_name", "model_versions", ["model_name"])
    op.create_index("ix_mv_status",     "model_versions", ["status"])
    op.create_index("ix_mv_manifest",   "model_versions", ["feature_manifest_hash"])

    # ── feature_versions ─────────────────────────────────────────────────────
    op.create_table(
        "feature_versions",
        sa.Column("id",                   sa.Integer,     primary_key=True, autoincrement=True),
        sa.Column("manifest_hash",        sa.String(64),  nullable=False, unique=True),
        sa.Column("pipeline_version",     sa.Integer,     nullable=False),
        sa.Column("feature_count",        sa.Integer,     nullable=False),
        sa.Column("feature_list_json",    sa.Text,        nullable=False),
        sa.Column("reference_stats_json", sa.Text,        nullable=True),
        sa.Column("recorded_at",          sa.DateTime,    nullable=False),
        sa.Column("description",          sa.Text,        nullable=True),
    )
    op.create_index("ix_fv_hash", "feature_versions", ["manifest_hash"])

    # ── inference_events ─────────────────────────────────────────────────────
    op.create_table(
        "inference_events",
        sa.Column("id",                   sa.BigInteger,  primary_key=True, autoincrement=True),
        sa.Column("request_id",           sa.String(32),  nullable=True),
        sa.Column("symbol",               sa.String(16),  nullable=False),
        sa.Column("bar_open_time",        sa.DateTime,    nullable=True),
        sa.Column("inference_ts",         sa.BigInteger,  nullable=False),
        sa.Column("model_name",           sa.String(64),  nullable=True),
        sa.Column("model_version_id",     sa.Integer,     nullable=True),
        sa.Column("feature_snapshot_id",  sa.String(16),  nullable=True),
        sa.Column("manifest_hash",        sa.String(64),  nullable=True),
        sa.Column("prob_up",              sa.Float,       nullable=True),
        sa.Column("prob_down",            sa.Float,       nullable=True),
        sa.Column("calibrated_prob_up",   sa.Float,       nullable=True),
        sa.Column("calibration_available",sa.Boolean,     nullable=True),
        sa.Column("tradeable_confidence", sa.Float,       nullable=True),
        sa.Column("degradation_factor",   sa.Float,       nullable=True),
        sa.Column("action",               sa.String(16),  nullable=True),
        sa.Column("abstain_reason",       sa.String(128), nullable=True),
        sa.Column("calibration_health",   sa.String(16),  nullable=True),
        sa.Column("ece_recent",           sa.Float,       nullable=True),
        sa.Column("rolling_brier",        sa.Float,       nullable=True),
        sa.Column("expected_move_pct",    sa.Float,       nullable=True),
        sa.Column("regime",               sa.String(32),  nullable=True),
        sa.Column("options_stale",        sa.Boolean,     nullable=False, server_default="false"),
        sa.Column("actual_outcome",       sa.SmallInteger,nullable=True),
        sa.Column("outcome_recorded_at",  sa.DateTime,    nullable=True),
        sa.Column("created_at",           sa.DateTime,    nullable=False),
    )
    op.create_index("ix_ie_symbol",     "inference_events", ["symbol"])
    op.create_index("ix_ie_action",     "inference_events", ["action"])
    op.create_index("ix_ie_created_at", "inference_events", ["created_at"])
    op.create_index("ix_inf_symbol_created", "inference_events", ["symbol", "created_at"])
    op.create_index("ix_inf_bar_time",       "inference_events", ["symbol", "bar_open_time"])

    # ── drift_snapshots ──────────────────────────────────────────────────────
    op.create_table(
        "drift_snapshots",
        sa.Column("id",                       sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol",                   sa.String(16), nullable=False),
        sa.Column("computed_at",              sa.DateTime,   nullable=False),
        sa.Column("window_bars",              sa.Integer,    nullable=False),
        sa.Column("manifest_hash",            sa.String(64), nullable=True),
        sa.Column("psi_by_feature_json",      sa.Text,       nullable=False),
        sa.Column("max_psi",                  sa.Float,      nullable=False),
        sa.Column("mean_psi",                 sa.Float,      nullable=False),
        sa.Column("high_drift_features_json", sa.Text,       nullable=True),
        sa.Column("drift_level",              sa.String(16), nullable=False),
        sa.Column("alert_raised",             sa.Boolean,    nullable=False, server_default="false"),
    )
    op.create_index("ix_ds_symbol",      "drift_snapshots", ["symbol"])
    op.create_index("ix_ds_computed_at", "drift_snapshots", ["computed_at"])
    op.create_index("ix_ds_drift_level", "drift_snapshots", ["drift_level"])

    # ── calibration_snapshots ────────────────────────────────────────────────
    op.create_table(
        "calibration_snapshots",
        sa.Column("id",                 sa.Integer,     primary_key=True, autoincrement=True),
        sa.Column("symbol",             sa.String(16),  nullable=False),
        sa.Column("snapshot_at",        sa.DateTime,    nullable=False),
        sa.Column("model_name",         sa.String(64),  nullable=True),
        sa.Column("window_size",        sa.Integer,     nullable=True),
        sa.Column("rolling_brier",      sa.Float,       nullable=True),
        sa.Column("baseline_brier",     sa.Float,       nullable=True),
        sa.Column("degradation_factor", sa.Float,       nullable=True),
        sa.Column("ece_recent",         sa.Float,       nullable=True),
        sa.Column("calibration_health", sa.String(16),  nullable=True),
        sa.Column("needs_retrain",      sa.Boolean,     nullable=False, server_default="false"),
        sa.Column("retrain_reason",     sa.Text,        nullable=True),
        sa.Column("reliability_json",   sa.Text,        nullable=True),
    )
    op.create_index("ix_cs_symbol",      "calibration_snapshots", ["symbol"])
    op.create_index("ix_cs_snapshot_at", "calibration_snapshots", ["snapshot_at"])

    # ── data_freshness_checks ────────────────────────────────────────────────
    op.create_table(
        "data_freshness_checks",
        sa.Column("id",                          sa.Integer,     primary_key=True, autoincrement=True),
        sa.Column("symbol",                      sa.String(16),  nullable=False),
        sa.Column("source",                      sa.String(64),  nullable=False),
        sa.Column("checked_at",                  sa.DateTime,    nullable=False),
        sa.Column("last_data_ts",                sa.DateTime,    nullable=True),
        sa.Column("age_seconds",                 sa.Float,       nullable=True),
        sa.Column("is_stale",                    sa.Boolean,     nullable=False),
        sa.Column("staleness_threshold_seconds", sa.Float,       nullable=False),
        sa.Column("alert_raised",                sa.Boolean,     nullable=False, server_default="false"),
    )
    op.create_index("ix_dfc_symbol",     "data_freshness_checks", ["symbol"])
    op.create_index("ix_dfc_source",     "data_freshness_checks", ["source"])
    op.create_index("ix_dfc_checked_at", "data_freshness_checks", ["checked_at"])

    # ── governance_alerts ────────────────────────────────────────────────────
    op.create_table(
        "governance_alerts",
        sa.Column("id",              sa.BigInteger,  primary_key=True, autoincrement=True),
        sa.Column("alert_type",      sa.String(64),  nullable=False),
        sa.Column("severity",        sa.String(16),  nullable=False),
        sa.Column("symbol",          sa.String(16),  nullable=True),
        sa.Column("title",           sa.String(256), nullable=False),
        sa.Column("details_json",    sa.Text,        nullable=True),
        sa.Column("triggered_at",    sa.DateTime,    nullable=False),
        sa.Column("expires_at",      sa.DateTime,    nullable=True),
        sa.Column("acknowledged_at", sa.DateTime,    nullable=True),
        sa.Column("acknowledged_by", sa.String(64),  nullable=True),
        sa.Column("dedup_key",       sa.String(128), nullable=True),
        sa.Column("is_active",       sa.Boolean,     nullable=False, server_default="true"),
    )
    op.create_index("ix_ga_alert_type",   "governance_alerts", ["alert_type"])
    op.create_index("ix_ga_severity",     "governance_alerts", ["severity"])
    op.create_index("ix_ga_symbol",       "governance_alerts", ["symbol"])
    op.create_index("ix_ga_triggered_at", "governance_alerts", ["triggered_at"])
    op.create_index("ix_ga_dedup_key",    "governance_alerts", ["dedup_key"])
    op.create_index("ix_ga_is_active",    "governance_alerts", ["is_active"])

    # ── kill_switch_state ────────────────────────────────────────────────────
    op.create_table(
        "kill_switch_state",
        sa.Column("id",           sa.Integer, primary_key=True, default=1),
        sa.Column("active",       sa.Boolean, nullable=False, server_default="false"),
        sa.Column("reason",       sa.Text,    nullable=True),
        sa.Column("activated_at", sa.DateTime,nullable=True),
        sa.Column("activated_by", sa.String(64), nullable=True),
        sa.Column("updated_at",   sa.DateTime,nullable=False),
    )


def downgrade() -> None:
    op.drop_table("kill_switch_state")
    op.drop_table("governance_alerts")
    op.drop_table("data_freshness_checks")
    op.drop_table("calibration_snapshots")
    op.drop_table("drift_snapshots")
    op.drop_table("inference_events")
    op.drop_table("feature_versions")
    op.drop_table("model_versions")
