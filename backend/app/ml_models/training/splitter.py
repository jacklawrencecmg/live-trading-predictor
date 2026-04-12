"""
PurgedWalkForwardSplit — time-series-safe cross-validation splitter.

For each fold k of n_splits total:
  train_end   = offset + k * step
  embargo_end = train_end + embargo_bars   ← dropped, not in train or test
  test_start  = embargo_end
  test_end    = test_start + test_window_bars  (or end of data if window is None)

The purge gap (embargo) prevents label leakage when the target variable for
bar i is close[i+h] and consecutive target bars overlap. For 1-bar lookahead
labels (standard binary up/down) set embargo=1. For k-bar lookahead use k.

This implements the spirit of Lopez de Prado's "Advances in Financial ML"
purged k-fold CV, simplified for a single-horizon sequential setting.
"""

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np


@dataclass
class FoldIndices:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_end: int       # last train bar index (exclusive)
    test_start: int      # first test bar index (inclusive)
    test_end: int        # last test bar index (exclusive)


class PurgedWalkForwardSplit:
    """
    Walk-forward splitter with embargo gap.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    test_window_bars : int or None
        Number of bars in each test window. None = expanding (all remaining bars).
    embargo_bars : int
        Bars dropped between train_end and test_start to prevent label overlap.
    min_train_bars : int
        Folds with fewer training bars are skipped.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_window_bars: Optional[int] = 500,
        embargo_bars: int = 1,
        min_train_bars: int = 250,
    ):
        self.n_splits = n_splits
        self.test_window_bars = test_window_bars
        self.embargo_bars = embargo_bars
        self.min_train_bars = min_train_bars

    def split(self, X: np.ndarray) -> Iterator[FoldIndices]:
        """
        Yield FoldIndices for each fold in chronological order.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Only shape[0] is used; values are ignored.
        """
        n = len(X)
        tw = self.test_window_bars
        em = self.embargo_bars

        if tw is None:
            # Expanding test: split data into n_splits+1 equal chunks
            # Train on first k chunks, test on chunk k+1
            chunk = n // (self.n_splits + 1)
            splits = [chunk * (i + 1) for i in range(self.n_splits)]
        else:
            # Fixed test window: determine split points so each test window
            # fits within the data
            total_test = self.n_splits * tw + (self.n_splits - 1) * em
            if total_test >= n:
                raise ValueError(
                    f"Not enough data: need {total_test} bars for test+embargo "
                    f"across {self.n_splits} folds, only {n} available."
                )
            # Space split points evenly across the back half of the data
            first_split = n - total_test
            step = (tw + em)
            splits = [first_split + i * step for i in range(self.n_splits)]

        for fold, train_end in enumerate(splits):
            test_start = train_end + em
            if tw is None:
                test_end = n
            else:
                test_end = min(test_start + tw, n)

            if test_start >= n or test_start >= test_end:
                continue

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) < self.min_train_bars:
                continue

            yield FoldIndices(
                fold=fold,
                train_idx=train_idx,
                test_idx=test_idx,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )

    def n_valid_folds(self, n_samples: int) -> int:
        """Return the number of folds that will actually be generated."""
        return sum(
            1 for _ in self.split(np.empty((n_samples, 1)))
        )

    def describe(self, n_samples: int) -> List[dict]:
        """Return a list of dicts describing each fold for logging."""
        result = []
        for fi in self.split(np.empty((n_samples, 1))):
            result.append({
                "fold": fi.fold,
                "train_bars": len(fi.train_idx),
                "embargo_bars": self.embargo_bars,
                "test_bars": len(fi.test_idx),
                "train_end": fi.train_end,
                "test_start": fi.test_start,
                "test_end": fi.test_end,
            })
        return result
