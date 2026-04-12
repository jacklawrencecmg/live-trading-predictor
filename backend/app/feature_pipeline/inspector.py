"""
Feature inspector — end-to-end debug tool for a single feature row.

Usage:
    from app.feature_pipeline.inspector import inspect_row
    report = inspect_row(df, symbol="SPY", bar_index=-1)
    print(report)           # human-readable text table
    report.to_dict()        # machine-readable dict for API responses
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

from app.feature_pipeline.compute import compute_features, valid_mask
from app.feature_pipeline.registry import (
    FEATURE_COLS,
    OPTIONS_FEATURE_COLS,
    ALL_FEATURE_COLS,
    REGISTRY,
    MANIFEST_HASH,
    PIPELINE_VERSION,
)


@dataclass
class FeatureInspection:
    """Full inspection result for one feature row."""

    # Target bar identity
    symbol: str
    bar_open_time: str
    bar_index: int

    # Pipeline metadata
    manifest_hash: str
    pipeline_version: int

    # Feature values
    features: Dict[str, float]

    # Diagnostics
    null_features: List[str]                    # FEATURE_COLS with NaN
    out_of_range: List[Dict[str, Any]]          # features outside expected_min/max
    is_valid: bool                              # True iff no required feature is NaN

    # Input bar summary (prior bar, used for features)
    prior_bar: Dict[str, Any]

    # Optional: per-feature metadata from registry
    feature_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        lines = [
            f"Feature Inspection — {self.symbol} @ {self.bar_open_time}",
            f"  Manifest: {self.manifest_hash}  Pipeline: v{self.pipeline_version}",
            f"  Valid: {self.is_valid}",
            "",
            f"{'Feature':<22} {'Value':>12}  {'Expected range':<20}  {'OK'}",
            "-" * 64,
        ]
        for name in ALL_FEATURE_COLS:
            val = self.features.get(name)
            reg = REGISTRY.get(name)
            if reg is None:
                continue
            lo = reg.expected_min
            hi = reg.expected_max
            range_str = f"[{lo}, {hi}]" if lo is not None or hi is not None else "—"
            if val is None or (isinstance(val, float) and np.isnan(val)):
                ok_str = "NULL"
                val_str = "NaN"
            else:
                in_range = (lo is None or val >= lo) and (hi is None or val <= hi)
                ok_str = "OK" if in_range else "WARN"
                val_str = f"{val:12.6f}"
            lines.append(f"  {name:<20} {val_str}  {range_str:<20}  {ok_str}")

        if self.null_features:
            lines.append("")
            lines.append(f"  NULL required features: {', '.join(self.null_features)}")
        if self.out_of_range:
            lines.append("")
            lines.append("  Out-of-range features:")
            for item in self.out_of_range:
                lines.append(
                    f"    {item['name']}: {item['value']:.6f}  expected [{item['expected_min']}, {item['expected_max']}]"
                )
        return "\n".join(lines)


def inspect_row(
    df: pd.DataFrame,
    symbol: str,
    bar_index: int = -1,
    options_data: Optional[Dict[str, Any]] = None,
) -> FeatureInspection:
    """
    Compute and inspect features for a single bar.

    Parameters
    ----------
    df : pd.DataFrame
        Closed-bar OHLCV DataFrame (full history needed for warmup).
    symbol : str
        Ticker symbol (for display only).
    bar_index : int
        Which row to inspect. Default -1 = last bar (most recent closed bar).
        Can be any integer index into df.
    options_data : dict or None
        Options chain summary for this bar.

    Returns
    -------
    FeatureInspection
        Full inspection dataclass. Call str() for a human-readable table,
        or .to_dict() for a machine-readable payload.
    """
    # Compute the full feature matrix
    feat_df = compute_features(df, options_data=options_data)

    # Target row
    n = len(feat_df)
    if bar_index < 0:
        bar_index = n + bar_index
    bar_index = max(0, min(bar_index, n - 1))

    feat_row = feat_df.iloc[bar_index]
    bar_time = str(feat_row.get("bar_open_time", "unknown"))

    # Feature values dict
    features = {}
    for col in ALL_FEATURE_COLS:
        if col in feat_row.index:
            v = feat_row[col]
            features[col] = None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
        else:
            features[col] = None

    # Null required features
    null_features = [
        name for name in FEATURE_COLS
        if features.get(name) is None
        and REGISTRY.get(name) and REGISTRY[name].null_strategy == "required"
    ]

    # Out-of-range features
    out_of_range = []
    for name in ALL_FEATURE_COLS:
        reg = REGISTRY.get(name)
        val = features.get(name)
        if reg is None or val is None:
            continue
        lo, hi = reg.expected_min, reg.expected_max
        if (lo is not None and val < lo) or (hi is not None and val > hi):
            out_of_range.append({
                "name": name,
                "value": val,
                "expected_min": lo,
                "expected_max": hi,
            })

    is_valid = len(null_features) == 0

    # Prior bar summary (the bar whose data drove these features)
    prior_idx = bar_index - 1
    prior_bar: Dict[str, Any] = {}
    if prior_idx >= 0:
        prior = df.iloc[prior_idx]
        for col in ["open", "high", "low", "close", "volume", "vwap", "bar_open_time"]:
            if col in prior.index:
                v = prior[col]
                prior_bar[col] = str(v) if col == "bar_open_time" else (
                    None if (isinstance(v, float) and np.isnan(v)) else float(v)
                )

    # Per-feature metadata
    feature_meta = {
        name: {
            "version": reg.version,
            "group": reg.group,
            "description": reg.description,
            "formula": reg.formula,
            "units": reg.units,
            "null_strategy": reg.null_strategy,
        }
        for name, reg in REGISTRY.items()
        if name in ALL_FEATURE_COLS
    }

    return FeatureInspection(
        symbol=symbol,
        bar_open_time=bar_time,
        bar_index=bar_index,
        manifest_hash=MANIFEST_HASH,
        pipeline_version=PIPELINE_VERSION,
        features=features,
        null_features=null_features,
        out_of_range=out_of_range,
        is_valid=is_valid,
        prior_bar=prior_bar,
        feature_meta=feature_meta,
    )
