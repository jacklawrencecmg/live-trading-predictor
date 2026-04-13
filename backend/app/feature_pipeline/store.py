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

# re-export for callers that only import from store
__all__ = [
    "save_feature_row",
    "save_feature_batch",
    "load_feature_row",
    "load_feature_range_pit",
    "deserialize_features",
    "to_feature_series",
    "rows_to_dataframe",
]
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


async def save_feature_batch(
    symbol: str,
    timeframe: str,
    feat_df: pd.DataFrame,
    db: AsyncSession,
) -> int:
    """
    Persist a full feature matrix to the database.

    feat_df: result of compute_features() — one row per bar, indexed from 0.
    Must contain a 'bar_open_time' column.

    Rows with no valid bar_open_time or where is_valid==False are still saved
    (callers can filter with is_valid later); this preserves the null-mask
    information for diagnostics.

    Returns the number of rows written/updated.
    """
    if "bar_open_time" not in feat_df.columns:
        raise ValueError("feat_df must contain a 'bar_open_time' column")

    count = 0
    for _, row in feat_df.iterrows():
        bot = row["bar_open_time"]
        if bot is None or (isinstance(bot, float) and np.isnan(bot)):
            continue
        bar_time = pd.Timestamp(bot).to_pydatetime()
        await save_feature_row(symbol, timeframe, bar_time, row, db)
        count += 1
    return count


async def load_feature_range_pit(
    symbol: str,
    timeframe: str,
    start_utc: datetime,
    end_utc: datetime,
    db: AsyncSession,
    require_current_manifest: bool = True,
    valid_only: bool = True,
) -> List[FeatureRow]:
    """
    Load feature rows for a symbol between start_utc and end_utc (inclusive).

    Point-in-time correct: only rows whose bar_open_time ∈ [start_utc, end_utc]
    are returned.  No future-bar contamination is possible because features are
    computed from bars strictly before bar_open_time (shift-by-1 invariant).

    Parameters
    ----------
    require_current_manifest
        When True (default), only rows with the current MANIFEST_HASH are
        returned.  Stale rows (computed under an old formula version) are
        excluded.  Set False only for diagnostic/replay purposes.
    valid_only
        When True (default), only rows where is_valid=True are returned.
        Invalid rows (warmup NaN, missing Greeks) are excluded from training.

    Returns
    -------
    List[FeatureRow] sorted by bar_open_time ascending.
    """
    stmt = select(FeatureRow).where(
        FeatureRow.symbol == symbol,
        FeatureRow.timeframe == timeframe,
        FeatureRow.bar_open_time >= start_utc,
        FeatureRow.bar_open_time <= end_utc,
    )
    if require_current_manifest:
        stmt = stmt.where(FeatureRow.manifest_hash == MANIFEST_HASH)
    if valid_only:
        stmt = stmt.where(FeatureRow.is_valid == True)  # noqa: E712
    stmt = stmt.order_by(FeatureRow.bar_open_time.asc())

    result = await db.execute(stmt)
    return list(result.scalars().all())


def rows_to_dataframe(rows: List[FeatureRow]) -> pd.DataFrame:
    """
    Convert a list of FeatureRow objects into a pandas DataFrame.

    Returns a DataFrame indexed by bar_open_time with columns = ALL_FEATURE_COLS.
    Useful for assembling a training matrix from persisted feature rows.
    """
    if not rows:
        return pd.DataFrame(columns=["bar_open_time"] + ALL_FEATURE_COLS)

    records = []
    for row in rows:
        feats = deserialize_features(row)
        feats["bar_open_time"] = row.bar_open_time
        records.append(feats)

    df = pd.DataFrame(records)
    df["bar_open_time"] = pd.to_datetime(df["bar_open_time"])
    df = df.sort_values("bar_open_time").reset_index(drop=True)
    return df
