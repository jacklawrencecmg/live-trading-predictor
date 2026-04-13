"""
Confidence tracker — rolling window of prediction outcomes per symbol.

Records (calibrated_prob, actual_outcome) pairs and derives:
  - rolling_brier: mean squared error over recent window
  - ece_recent:    expected calibration error over recent window
  - degradation_factor: how much to shrink tradeable confidence

The tracker is file-backed so state survives process restarts.
Writes are debounced to avoid excessive I/O on every prediction.

Usage:
    tracker = ConfidenceTracker()
    tracker.record("SPY", calibrated_prob=0.58, actual_outcome=1)
    stats = tracker.get_stats("SPY")
    # stats.degradation_factor, stats.rolling_brier, etc.
"""

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TRACKER_DIR = Path(__file__).parent.parent.parent / "model_artifacts" / "trackers"
TRACKER_DIR.mkdir(parents=True, exist_ok=True)

# How many recent (prob, outcome) pairs to retain per symbol
DEFAULT_WINDOW = 100
MIN_WINDOW_FOR_STATS = 20    # minimum observations before computing degradation
DEFAULT_BASELINE_BRIER = 0.23  # fallback when no training baseline is available


RETRAIN_DEGRADATION_THRESHOLD: float = 0.40   # degradation_factor below this → retrain recommended
RETRAIN_ECE_THRESHOLD: float = 0.12            # ECE above this → retrain recommended
RETRAIN_MIN_WINDOW: int = 40                   # minimum observations before recommending retrain


@dataclass
class TrackerStats:
    symbol: str
    window_size: int          # actual observations in the window
    rolling_brier: Optional[float]
    baseline_brier: Optional[float]
    degradation_factor: float  # 1.0 = no degradation; 0.0 = fully degraded
    ece_recent: Optional[float]
    calibration_health: str   # "good" | "fair" | "degraded" | "unknown"

    # Reliability diagram data (10-bin)
    reliability_bins: Optional[List[float]]
    reliability_mean_pred: Optional[List[float]]
    reliability_frac_pos: Optional[List[float]]

    # Retraining signal
    needs_retrain: bool = False
    retrain_reason: Optional[str] = None


def _brier(probs: List[float], outcomes: List[int]) -> float:
    n = len(probs)
    if n == 0:
        return float("nan")
    return sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / n


def _ece(probs: List[float], outcomes: List[int], n_bins: int = 10) -> float:
    n = len(probs)
    if n == 0:
        return float("nan")
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        indices = [i for i, p in enumerate(probs) if lo <= p < hi] if hi < 1.0 else \
                  [i for i, p in enumerate(probs) if lo <= p <= hi]
        if not indices:
            continue
        mean_pred = sum(probs[i] for i in indices) / len(indices)
        frac_pos = sum(outcomes[i] for i in indices) / len(indices)
        ece += (len(indices) / n) * abs(frac_pos - mean_pred)
    return ece


def _reliability_diagram(
    probs: List[float],
    outcomes: List[int],
    n_bins: int = 10,
) -> Tuple[List[float], List[float], List[float]]:
    """Returns (bin_centres, mean_predicted, fraction_positive)."""
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    centres, mean_preds, frac_poss = [], [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        indices = [i for i, p in enumerate(probs) if lo <= p < hi] if hi < 1.0 else \
                  [i for i, p in enumerate(probs) if lo <= p <= hi]
        if not indices:
            continue
        centres.append(round((lo + hi) / 2, 3))
        mean_preds.append(round(sum(probs[i] for i in indices) / len(indices), 4))
        frac_poss.append(round(sum(outcomes[i] for i in indices) / len(indices), 4))

    return centres, mean_preds, frac_poss


def _degradation_factor(
    rolling_brier: float,
    baseline_brier: float,
) -> float:
    """
    Compute degradation factor in [0, 1].

    Logic:
      ratio = rolling_brier / baseline_brier
      ratio ≤ 1.0   → factor = 1.0  (performing at or better than baseline)
      ratio = 1.5   → factor = 0.5  (50% signal reduction)
      ratio ≥ 2.0   → factor = 0.0  (full abstain)

    This is linear in the range [1.0, 2.0] and clipped outside.
    """
    if baseline_brier <= 0:
        return 1.0
    ratio = rolling_brier / baseline_brier
    if ratio <= 1.0:
        return 1.0
    if ratio >= 2.0:
        return 0.0
    return round(1.0 - (ratio - 1.0), 4)   # linear: 1.5 → 0.5


# ---------------------------------------------------------------------------
# Tracker state
# ---------------------------------------------------------------------------

@dataclass
class _SymbolState:
    probs: List[float] = field(default_factory=list)
    outcomes: List[int] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    baseline_brier: float = DEFAULT_BASELINE_BRIER
    window: int = DEFAULT_WINDOW
    last_save_time: float = 0.0


class ConfidenceTracker:
    """
    Per-symbol rolling tracker of prediction calibration quality.

    Thread-safe. File-backed (writes are debounced to 30s).
    """

    _SAVE_DEBOUNCE_SECONDS = 30

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        tracker_dir: Path = TRACKER_DIR,
        storage_dir: Optional[Path] = None,   # alias for tracker_dir
    ):
        self._window = window
        # storage_dir is an alias for tracker_dir; storage_dir takes precedence when both provided
        self._dir = Path(storage_dir) if storage_dir is not None else tracker_dir
        self._states: Dict[str, _SymbolState] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def record(
        self,
        symbol: str,
        calibrated_prob: float,
        actual_outcome: int,
        baseline_brier: Optional[float] = None,
    ) -> None:
        """
        Record a prediction outcome.

        Parameters
        ----------
        symbol : str
        calibrated_prob : float  — calibrated P(up) for this bar's prediction
        actual_outcome : int     — 1 if price went up, 0 if down
        baseline_brier : float or None
            Training-time Brier score. If provided, updates the baseline for
            this symbol (useful when a new model is promoted).
        """
        with self._lock:
            state = self._get_or_load(symbol)
            state.probs.append(float(calibrated_prob))
            state.outcomes.append(int(actual_outcome))
            state.timestamps.append(time.time())

            # Trim to window
            if len(state.probs) > state.window:
                state.probs = state.probs[-state.window:]
                state.outcomes = state.outcomes[-state.window:]
                state.timestamps = state.timestamps[-state.window:]

            if baseline_brier is not None:
                state.baseline_brier = float(baseline_brier)

            self._maybe_save(symbol, state)

    def get_stats(self, symbol: str) -> TrackerStats:
        """Return rolling calibration statistics for symbol."""
        with self._lock:
            state = self._get_or_load(symbol)
            n = len(state.probs)

        if n < MIN_WINDOW_FOR_STATS:
            return TrackerStats(
                symbol=symbol,
                window_size=n,
                rolling_brier=None,
                baseline_brier=state.baseline_brier,
                degradation_factor=1.0,
                ece_recent=None,
                calibration_health="unknown",
                reliability_bins=None,
                reliability_mean_pred=None,
                reliability_frac_pos=None,
            )

        rb = _brier(state.probs, state.outcomes)
        ece = _ece(state.probs, state.outcomes)
        deg = _degradation_factor(rb, state.baseline_brier)

        bins, mean_preds, frac_poss = _reliability_diagram(state.probs, state.outcomes)

        # Calibration health — four states after "unknown" cleared by window check above:
        #   good:    rolling performance at or near training baseline, low ECE
        #   fair:    modest Brier increase or slightly elevated ECE — tradeable with penalty
        #   caution: notable drift; signal still usable but degradation factor applied
        #   degraded: severe drift; hard-abstain threshold may trigger
        ratio = rb / (state.baseline_brier + 1e-9)
        if ratio <= 1.1 and ece <= 0.05:
            health = "good"
        elif ratio <= 1.3 or ece <= 0.10:
            health = "fair"
        elif ratio <= 1.6 or ece <= 0.15:
            health = "caution"
        else:
            health = "degraded"

        # Retraining trigger: fire when degradation is severe AND we have enough data
        needs_retrain = False
        retrain_reason: Optional[str] = None
        if n >= RETRAIN_MIN_WINDOW:
            if deg <= RETRAIN_DEGRADATION_THRESHOLD:
                needs_retrain = True
                retrain_reason = f"degradation_factor={deg:.2f} <= threshold={RETRAIN_DEGRADATION_THRESHOLD}"
            elif ece is not None and not math.isnan(ece) and ece >= RETRAIN_ECE_THRESHOLD:
                needs_retrain = True
                retrain_reason = f"ece_recent={ece:.4f} >= threshold={RETRAIN_ECE_THRESHOLD}"
        if needs_retrain:
            logger.warning(
                "RETRAIN RECOMMENDED for %s: %s (window=%d)", symbol, retrain_reason, n
            )

        return TrackerStats(
            symbol=symbol,
            window_size=n,
            rolling_brier=round(rb, 6),
            baseline_brier=state.baseline_brier,
            degradation_factor=deg,
            ece_recent=round(ece, 6),
            calibration_health=health,
            reliability_bins=bins,
            reliability_mean_pred=mean_preds,
            reliability_frac_pos=frac_poss,
            needs_retrain=needs_retrain,
            retrain_reason=retrain_reason,
        )

    def record_inference(
        self,
        symbol: str,
        calibrated_prob: float,
        actual_outcome: int,
        baseline_brier: Optional[float] = None,
    ) -> None:
        """
        Alias for record(). Preferred name at inference callsites.

        Records a resolved prediction outcome for the rolling calibration window.
        Call this when an outcome (price went up/down) becomes known for a bar
        on which inference was previously run.
        """
        self.record(symbol, calibrated_prob, actual_outcome, baseline_brier)

    def set_baseline_brier(self, symbol: str, baseline_brier: float) -> None:
        """Update the training baseline for a symbol (called after model promotion)."""
        with self._lock:
            state = self._get_or_load(symbol)
            state.baseline_brier = float(baseline_brier)
            self._save(symbol, state)

    # ---------------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------------

    def _path(self, symbol: str) -> Path:
        return self._dir / f"tracker_{symbol.upper()}.json"

    def _get_or_load(self, symbol: str) -> _SymbolState:
        if symbol in self._states:
            return self._states[symbol]
        state = self._load(symbol)
        self._states[symbol] = state
        return state

    def _load(self, symbol: str) -> _SymbolState:
        path = self._path(symbol)
        if not path.exists():
            return _SymbolState(window=self._window)
        try:
            d = json.loads(path.read_text())
            s = _SymbolState(window=self._window)
            s.probs = d.get("probs", [])
            s.outcomes = d.get("outcomes", [])
            s.timestamps = d.get("timestamps", [])
            s.baseline_brier = float(d.get("baseline_brier", DEFAULT_BASELINE_BRIER))
            return s
        except Exception as e:
            logger.warning("Failed to load tracker for %s: %s", symbol, e)
            return _SymbolState(window=self._window)

    def _maybe_save(self, symbol: str, state: _SymbolState) -> None:
        now = time.time()
        if now - state.last_save_time >= self._SAVE_DEBOUNCE_SECONDS:
            self._save(symbol, state)
            state.last_save_time = now

    def _save(self, symbol: str, state: _SymbolState) -> None:
        try:
            self._path(symbol).write_text(json.dumps({
                "probs": state.probs,
                "outcomes": state.outcomes,
                "timestamps": state.timestamps,
                "baseline_brier": state.baseline_brier,
            }))
        except Exception as e:
            logger.warning("Failed to save tracker for %s: %s", symbol, e)


# ---------------------------------------------------------------------------
# Singleton (module-level, shared across requests in one process)
# ---------------------------------------------------------------------------
_tracker: Optional[ConfidenceTracker] = None
_tracker_lock = threading.Lock()


def get_tracker() -> ConfidenceTracker:
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = ConfidenceTracker()
    return _tracker
