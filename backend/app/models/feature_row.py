"""
FeatureRow — persisted feature snapshot for one bar.

Each row stores:
  - Identity: symbol, timeframe, bar_open_time
  - Versioning: manifest_hash (invalidates on formula change), pipeline_version
  - Feature data: features_json (all feature values), null_mask (which are NaN)
  - Snapshot ID: SHA-256 of feature values for auditability
  - Validity flag: False when any required feature is NaN

The manifest_hash is sourced from app.feature_pipeline.registry.MANIFEST_HASH
and changes whenever any (name, version) pair in the registry changes. Stale
rows (manifest_hash != current) should be treated as invalid and recomputed.
"""

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Index, Integer, String, Text, UniqueConstraint
)
from app.core.database import Base


class FeatureRow(Base):
    __tablename__ = "feature_rows"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Identity
    symbol = Column(String(16), nullable=False, index=True)
    timeframe = Column(String(8), nullable=False)           # e.g. "5m", "1d"
    bar_open_time = Column(DateTime, nullable=False, index=True)

    # Versioning
    manifest_hash = Column(String(16), nullable=False, index=True)
    pipeline_version = Column(Integer, nullable=False)

    # Feature payload
    features_json = Column(Text, nullable=False)            # JSON dict: {name: value}
    null_mask = Column(Text, nullable=True)                 # JSON list of null feature names

    # Auditability
    snapshot_id = Column(String(16), nullable=False, index=True)

    # Validity: False when any "required" feature is NaN
    is_valid = Column(Boolean, nullable=False, default=True)

    __table_args__ = (
        UniqueConstraint(
            "symbol", "timeframe", "bar_open_time", "manifest_hash",
            name="uq_feature_row_identity",
        ),
        Index("ix_feature_row_lookup", "symbol", "timeframe", "bar_open_time"),
    )
