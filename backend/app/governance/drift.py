"""
Feature distribution drift monitor.

Uses Population Stability Index (PSI) to detect covariate shift between
the reference (training-time) feature distribution and the current window.

PSI formula (per bin):
    PSI_bin = (actual_pct - expected_pct) × ln(actual_pct / expected_pct)
    PSI = Σ PSI_bin

Interpretation:
    PSI < 0.10   → no significant drift      (drift_level = 'none')
    PSI 0.10-0.25 → moderate drift           (drift_level = 'moderate')
    PSI > 0.25   → significant shift         (drift_level = 'high')

Reference distribution source priority:
    1. FeatureVersion.reference_stats_json (stored training percentiles)
    2. FeatureRegistry expected_min/expected_max (uniform assumption — noted as less accurate)
    3. Gaussian approximation from current window mean/std (poorest quality — flags this)

All PSI values are stored verbatim in drift_snapshots.psi_by_feature_json so
a reviewer can inspect per-feature drift without re-running computation.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.governance.models import DriftSnapshot

logger = logging.getLogger(__name__)

# PSI thresholds
_PSI_MODERATE = 0.10
_PSI_HIGH     = 0.25
_N_BINS       = 10


# ---------------------------------------------------------------------------
# PSI computation
# ---------------------------------------------------------------------------

def _psi_1d(
    current: np.ndarray,
    reference_percentiles: Optional[Dict[str, float]],
    expected_min: Optional[float] = None,
    expected_max: Optional[float] = None,
    n_bins: int = _N_BINS,
) -> float:
    """
    Compute PSI for a single feature vector.

    Parameters
    ----------
    current : recent feature values
    reference_percentiles : dict with keys p0, p10, p20, ..., p100 from training
    expected_min / expected_max : fallback for uniform bin edges
    """
    if len(current) < 30:
        return float("nan")   # insufficient data

    # Determine bin edges
    if reference_percentiles:
        pct_keys = sorted([k for k in reference_percentiles if k.startswith("p")],
                          key=lambda k: int(k[1:]))
        edges = np.array([reference_percentiles[k] for k in pct_keys], dtype=float)
        # deduplicate edges (constant features)
        edges = np.unique(edges)
        if len(edges) < 3:
            return 0.0   # constant reference → no drift possible
    elif expected_min is not None and expected_max is not None and expected_min < expected_max:
        edges = np.linspace(expected_min, expected_max, n_bins + 1)
    else:
        # Gaussian fallback using current window stats
        mu, sigma = float(np.nanmean(current)), float(np.nanstd(current))
        if sigma < 1e-9:
            return 0.0
        edges = np.array([mu + sigma * z for z in np.linspace(-3, 3, n_bins + 1)])

    # Compute actual distribution over the current window
    actual_counts, _ = np.histogram(current, bins=edges)
    expected_counts  = np.full(len(actual_counts), len(current) / len(actual_counts))

    total_actual   = actual_counts.sum()
    total_expected = expected_counts.sum()
    if total_actual == 0 or total_expected == 0:
        return float("nan")

    psi = 0.0
    eps = 1e-4  # prevents log(0)
    for a, e in zip(actual_counts, expected_counts):
        a_pct = (a / total_actual)   + eps
        e_pct = (e / total_expected) + eps
        psi += (a_pct - e_pct) * math.log(a_pct / e_pct)
    return max(0.0, psi)


def _classify_drift(max_psi: float) -> str:
    if max_psi >= _PSI_HIGH:     return "high"
    if max_psi >= _PSI_MODERATE: return "moderate"
    return "none"


# ---------------------------------------------------------------------------
# DriftMonitor
# ---------------------------------------------------------------------------

class DriftMonitor:

    @staticmethod
    def compute_psi_from_matrix(
        feature_matrix: np.ndarray,
        feature_names: List[str],
        reference_stats: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        """
        Compute PSI for each column of feature_matrix.

        Parameters
        ----------
        feature_matrix  : shape (n_bars, n_features), recent observations
        feature_names   : column labels for feature_matrix
        reference_stats : {feat_name: {p0, p10, ..., p100}} from training

        Returns {feature_name: psi_value}
        """
        psi_map: Dict[str, float] = {}
        for i, name in enumerate(feature_names):
            if i >= feature_matrix.shape[1]:
                break
            col = feature_matrix[:, i]
            col = col[~np.isnan(col)]
            ref_pcts = (reference_stats or {}).get(name)
            psi_map[name] = _psi_1d(col, ref_pcts)
        return psi_map

    @staticmethod
    async def record_snapshot(
        db: AsyncSession,
        *,
        symbol: str,
        psi_by_feature: Dict[str, float],
        window_bars: int,
        manifest_hash: Optional[str] = None,
    ) -> DriftSnapshot:
        """Persist a drift snapshot and return the ORM row."""
        valid_psis = [v for v in psi_by_feature.values() if not math.isnan(v)]
        max_psi    = max(valid_psis, default=0.0)
        mean_psi   = float(np.mean(valid_psis)) if valid_psis else 0.0

        high_feats = [k for k, v in psi_by_feature.items() if v >= _PSI_HIGH]
        drift_level = _classify_drift(max_psi)

        row = DriftSnapshot(
            symbol=symbol,
            window_bars=window_bars,
            manifest_hash=manifest_hash,
            psi_by_feature_json=json.dumps({k: round(v, 6) if not math.isnan(v) else None
                                            for k, v in psi_by_feature.items()}),
            max_psi=round(max_psi, 6),
            mean_psi=round(mean_psi, 6),
            high_drift_features_json=json.dumps(high_feats),
            drift_level=drift_level,
            alert_raised=False,
        )
        db.add(row)
        await db.flush()

        if drift_level != "none":
            logger.warning(
                "Drift detected for %s: level=%s max_psi=%.4f high_features=%s",
                symbol, drift_level, max_psi, high_feats,
            )
        return row

    @staticmethod
    async def get_latest(
        db: AsyncSession,
        symbol: str,
    ) -> Optional[DriftSnapshot]:
        result = await db.execute(
            select(DriftSnapshot)
            .where(DriftSnapshot.symbol == symbol)
            .order_by(DriftSnapshot.computed_at.desc(), DriftSnapshot.id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_history(
        db: AsyncSession,
        symbol: str,
        limit: int = 20,
    ) -> List[DriftSnapshot]:
        result = await db.execute(
            select(DriftSnapshot)
            .where(DriftSnapshot.symbol == symbol)
            .order_by(DriftSnapshot.computed_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def summary_all_symbols(
        db: AsyncSession,
    ) -> Dict[str, str]:
        """Return {symbol: drift_level} for the most recent snapshot per symbol."""
        # Get max computed_at per symbol, then fetch those rows
        from sqlalchemy import func
        subq = (
            select(DriftSnapshot.symbol, func.max(DriftSnapshot.computed_at).label("max_ts"))
            .group_by(DriftSnapshot.symbol)
            .subquery()
        )
        result = await db.execute(
            select(DriftSnapshot)
            .join(subq, (DriftSnapshot.symbol == subq.c.symbol) &
                  (DriftSnapshot.computed_at == subq.c.max_ts))
        )
        rows = result.scalars().all()
        return {r.symbol: r.drift_level for r in rows}
