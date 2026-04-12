"""
Model registry — versioned artifact storage.

Layout:
  model_artifacts/
    registry.json                      — index of all saved models
    {model_name}_{config_hash}_{ts}/
      model.pkl                        — serialized estimator
      metadata.json                    — config, metrics, feature manifest
      training_report.json             — full fold + evaluation results

Models saved here can be loaded by:
  1. The inference service (get_production_model)
  2. Backtesting (load_model_by_hash)
  3. Comparison/analysis scripts

The inference service in inference_service.py uses the simple load_model()
from the old baseline.py for backward compatibility. To switch it to the
versioned registry, call promote_to_production() after saving.
"""

import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.feature_pipeline.registry import MANIFEST_HASH, PIPELINE_VERSION

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "model_artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

REGISTRY_PATH = ARTIFACTS_DIR / "registry.json"
PRODUCTION_SYMLINK = ARTIFACTS_DIR / "production"  # symlink → latest promoted model dir


# ---------------------------------------------------------------------------
# Registry index
# ---------------------------------------------------------------------------

def _load_index() -> List[dict]:
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text())
        except Exception:
            return []
    return []


def _save_index(index: List[dict]) -> None:
    REGISTRY_PATH.write_text(json.dumps(index, indent=2))


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_model(
    model: Any,
    model_name: str,
    config_hash: str,
    metrics: Dict[str, float],
    feature_names: List[str],
    training_report_dict: Optional[dict] = None,
    notes: str = "",
) -> Path:
    """
    Serialize a fitted model and its metadata to a versioned artifact directory.

    Returns the artifact directory path.
    """
    ts = int(time.time())
    dir_name = f"{model_name}_{config_hash}_{ts}"
    artifact_dir = ARTIFACTS_DIR / dir_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Model pickle
    model_path = artifact_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Metadata
    metadata = {
        "model_name": model_name,
        "config_hash": config_hash,
        "timestamp": ts,
        "timestamp_iso": datetime.utcfromtimestamp(ts).isoformat() + "Z",
        "manifest_hash": MANIFEST_HASH,
        "pipeline_version": PIPELINE_VERSION,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "metrics": metrics,
        "notes": notes,
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if training_report_dict is not None:
        (artifact_dir / "training_report.json").write_text(
            json.dumps(training_report_dict, indent=2, default=str)
        )

    # Update registry index
    index = _load_index()
    index.append({
        "dir": dir_name,
        "model_name": model_name,
        "config_hash": config_hash,
        "timestamp": ts,
        "manifest_hash": MANIFEST_HASH,
        "primary_metric": metrics.get("brier_score_mean"),
        "is_production": False,
    })
    _save_index(index)

    logger.info("Saved model artifact: %s", artifact_dir)
    return artifact_dir


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_model_from_dir(artifact_dir: Path) -> Any:
    """Load a pickled model from an artifact directory."""
    model_path = artifact_dir / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No model.pkl in {artifact_dir}")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_model_by_hash(config_hash: str, model_name: Optional[str] = None) -> Optional[Any]:
    """
    Load the most recent model matching config_hash (and optionally model_name).
    Returns None if not found.
    """
    index = _load_index()
    candidates = [
        e for e in index
        if e["config_hash"] == config_hash
        and (model_name is None or e["model_name"] == model_name)
    ]
    if not candidates:
        return None
    # Most recent
    best = max(candidates, key=lambda e: e["timestamp"])
    return load_model_from_dir(ARTIFACTS_DIR / best["dir"])


def get_production_model() -> Optional[Any]:
    """
    Load the current production model (if any has been promoted).
    Returns None if no production model is set.
    """
    # Check registry index for promoted models
    index = _load_index()
    prod = [e for e in index if e.get("is_production")]
    if not prod:
        return None
    latest = max(prod, key=lambda e: e["timestamp"])
    try:
        return load_model_from_dir(ARTIFACTS_DIR / latest["dir"])
    except Exception as e:
        logger.error("Failed to load production model: %s", e)
        return None


def promote_to_production(config_hash: str, model_name: Optional[str] = None) -> bool:
    """
    Mark the most recent matching model as the production model.
    Clears the production flag from all other entries.
    Returns True if a model was found and promoted.
    """
    index = _load_index()
    candidates = [
        e for e in index
        if e["config_hash"] == config_hash
        and (model_name is None or e["model_name"] == model_name)
    ]
    if not candidates:
        logger.warning("No model found for config_hash=%s", config_hash)
        return False

    latest = max(candidates, key=lambda e: e["timestamp"])

    for e in index:
        e["is_production"] = (e is latest)
    _save_index(index)
    logger.info("Promoted to production: %s", latest["dir"])
    return True


# ---------------------------------------------------------------------------
# List and inspect
# ---------------------------------------------------------------------------

def list_models(model_name: Optional[str] = None) -> List[dict]:
    """Return registry index, optionally filtered by model name."""
    index = _load_index()
    if model_name:
        index = [e for e in index if e["model_name"] == model_name]
    return sorted(index, key=lambda e: -e["timestamp"])


def get_metadata(dir_name: str) -> Optional[dict]:
    """Return the metadata.json for a given artifact directory name."""
    meta_path = ARTIFACTS_DIR / dir_name / "metadata.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())
