"""
Feature store — save and load FeatureRow records.

Provides a thin async CRUD layer over the feature_rows table.
The snapshot_id is a SHA-256 of rounded feature values (same algorithm as
inference_service._feature_snapshot_id) for cross-service auditability.
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Optional, List

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.feature_pipeline.registry import (
    FEATURE_COLS,
    ALL_FEATURE_COLS,
    MANIFEST_HASH,
    PIPELINE_VERSION,
    REGISTRY,
)
from app.models.feature_row import FeatureRow

logger = logging.getLogger(__name__)


def _snapshot_id(features: dict) -> str:
    """Stable SHA-256 hash of feature values for cross-service auditability."""
    vals = [
        round(float(v), 6) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None
        for v in [features.get(n) for n in sorted(features)]
    ]
    s = json.dumps(vals)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _null_mask(features: dict) -> List[str]:
    """Return list of feature names whose value is NaN or None."""
    return [
        k for k, v in features.items()
        if v is None or (isinstance(v, float) and np.isnan(v))
    ]


def _is_valid(features: dict) -> bool:
    """
    True when all "required" features have non-null values.
    Optional_sentinel features are always non-null (replaced by sentinel).
    """
    for name in FEATURE_COLS:
        if REGISTRY.get(name) and REGISTRY[name].null_strategy == "required":
            v = features.get(name)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return False
    return True


async def save_feature_row(
    symbol: str,
    timeframe: str,
    bar_open_time: datetime,
    feat_row: pd.Series,
    db: AsyncSession,
) -> FeatureRow:
    """
    Persist one feature row to the database.

    feat_row: a pandas Series with index = ALL_FEATURE_COLS (or a superset).
    Upserts by (symbol, timeframe, bar_open_time, manifest_hash) — safe to
    call repeatedly; duplicate writes update the existing record.
    """
    features = {}
    for col in ALL_FEATURE_COLS:
        if col not in feat_row.index:
            continue
        v = feat_row[col]
        features[col] = None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

    null_mask_list = _null_mask(features)
    is_valid = _is_valid(features)
    snap_id = _snapshot_id(features)

    # Check for existing row
    stmt = select(FeatureRow).where(
        FeatureRow.symbol == symbol,
        FeatureRow.timeframe == timeframe,
        FeatureRow.bar_open_time == bar_open_time,
        FeatureRow.manifest_hash == MANIFEST_HASH,
    )
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing is not None:
        existing.features_json = json.dumps(features)
        existing.null_mask = json.dumps(null_mask_list)
        existing.snapshot_id = snap_id
        existing.is_valid = is_valid
        existing.pipeline_version = PIPELINE_VERSION
        return existing

    row = FeatureRow(
        symbol=symbol,
        timeframe=timeframe,
        bar_open_time=bar_open_time,
        manifest_hash=MANIFEST_HASH,
        pipeline_version=PIPELINE_VERSION,
        features_json=json.dumps(features),
        null_mask=json.dumps(null_mask_list),
        snapshot_id=snap_id,
        is_valid=is_valid,
    )
    db.add(row)
    await db.flush()
    return row


async def load_feature_row(
    symbol: str,
    timeframe: str,
    bar_open_time: datetime,
    db: AsyncSession,
    require_current_manifest: bool = True,
) -> Optional[FeatureRow]:
    """
    Load the most recent valid FeatureRow for (symbol, timeframe, bar_open_time).

    When require_current_manifest=True (default), only rows whose manifest_hash
    matches the current MANIFEST_HASH are returned. Stale rows (from an older
    formula version) return None, prompting the caller to recompute.
    """
    stmt = select(FeatureRow).where(
        FeatureRow.symbol == symbol,
        FeatureRow.timeframe == timeframe,
        FeatureRow.bar_open_time == bar_open_time,
        FeatureRow.is_valid == True,  # noqa: E712
    )
    if require_current_manifest:
        stmt = stmt.where(FeatureRow.manifest_hash == MANIFEST_HASH)

    result = await db.execute(stmt)
    return result.scalar_one_or_none()


def deserialize_features(row: FeatureRow) -> dict:
    """Return feature dict from a FeatureRow, decoding stored JSON."""
    return json.loads(row.features_json)


def to_feature_series(row: FeatureRow) -> pd.Series:
    """Return feature values as a pandas Series indexed by feature name."""
    return pd.Series(deserialize_features(row))
