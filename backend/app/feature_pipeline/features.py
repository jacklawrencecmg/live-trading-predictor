"""
Feature engineering pipeline.

LEAKAGE PREVENTION:
- All features are computed from bars with bar_open_time < target_bar_open_time
- The target bar is NEVER included in the feature window
- Rolling stats use .shift(1) before .rolling() to exclude the current bar
- Options features use snapshot_time < bar_open_time of the target bar

This module is the public API for the feature pipeline. Computation is
delegated to app.feature_pipeline.compute. Feature definitions and column
lists are sourced from app.feature_pipeline.registry.
"""

from typing import Optional, Dict, Any

import pandas as pd

from app.feature_pipeline.compute import compute_features, valid_mask
from app.feature_pipeline.registry import (
    FEATURE_COLS,
    OPTIONS_FEATURE_COLS,
    ALL_FEATURE_COLS,
    MANIFEST_HASH,
    PIPELINE_VERSION,
    FFILL_LIMIT,
)

# Re-export for callers that import from this module
__all__ = [
    "build_feature_matrix",
    "compute_features",
    "valid_mask",
    "FEATURE_COLS",
    "OPTIONS_FEATURE_COLS",
    "ALL_FEATURE_COLS",
    "MANIFEST_HASH",
    "PIPELINE_VERSION",
    "FFILL_LIMIT",
]


def build_feature_matrix(
    df: pd.DataFrame,
    options_data: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Build feature matrix from a closed-bar OHLCV DataFrame.

    Thin wrapper around compute_features() — preserved for backward
    compatibility with callers that import build_feature_matrix from here.

    df columns: open, high, low, close, volume, vwap (optional), bar_open_time
    Returns a DataFrame with ALL_FEATURE_COLS + bar_open_time columns.
    """
    return compute_features(df, options_data=options_data)
