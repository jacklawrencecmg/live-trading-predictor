"""
Multi-horizon target labels for the options-research forecasting system.

Supported horizons (default): 1 bar, 3 bars, 5 bars.

Target types per horizon h
--------------------------
  y_dir_h{h}     3-class direction: 0=DOWN  1=FLAT  2=UP
  y_ret_h{h}     Signed log return: log(close[i+h] / close[i])
  y_mag_h{h}     Unsigned magnitude: |y_ret_h|
  y_rvol_h{h}    Realized-vol forecast: RMS of per-bar log-returns over next h bars
  y_abstain_h{h} 1 when direction is FLAT, else 0

Leakage safety
--------------
Feature[i] uses bars 0..i-1 (enforced by compute.py shift(1)).
Label y_*_h[i] uses close[i+1]..close[i+h] — strictly future data.
There is NO overlap between the feature window and the label window at any row.

The ATR-based flat threshold is computed from shifted high/low/close (bars 0..i-1),
so the classification boundary for row i is knowable at bar-i open time.

Label overlap in training (critical for PurgedWalkForwardSplit configuration)
------------------------------------------------------------------------------
For horizon h, consecutive training labels share h-1 bars:

  y_rvol_h5[i]   uses r[i+1], r[i+2], r[i+3], r[i+4], r[i+5]
  y_rvol_h5[i+1] uses r[i+2], r[i+3], r[i+4], r[i+5], r[i+6]

If i is the last training bar and i+1 is the first test bar, bars i+2..i+5 appear
in both a training label and a test label — information leakage.

  PurgedWalkForwardSplit must use embargo_bars >= h for h-bar targets:
    h=1 → embargo_bars = 1   (default is already correct)
    h=3 → embargo_bars = 3
    h=5 → embargo_bars = 5
"""

import math
from enum import IntEnum
from typing import Tuple

import numpy as np
import pandas as pd

HORIZONS: Tuple[int, ...] = (1, 3, 5)
FLAT_THRESHOLD_K: float = 0.5    # multiplied by ATR_norm × sqrt(h)


class DirectionLabel(IntEnum):
    DOWN = 0
    FLAT = 1
    UP   = 2


def _atr_norm(df: pd.DataFrame) -> pd.Series:
    """
    Per-bar ATR / close, using only bars 0..i-1 (shift-safe).

    Identical formula to labels.py:ternary_label so the flat-zone boundary is
    consistent with the single-horizon baseline when h=1.
    """
    h   = df["high"].shift(1)
    lw  = df["low"].shift(1)
    pc  = df["close"].shift(2)    # prior close for gap component of True Range
    tr  = pd.concat([h - lw, (h - pc).abs(), (lw - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(com=13, adjust=False).mean()
    return atr / (df["close"].shift(1) + 1e-9)


def _dir_target(
    df: pd.DataFrame,
    h: int,
    atr_norm: pd.Series,
    flat_threshold_k: float,
) -> pd.Series:
    """
    Ternary direction label for horizon h. Last h rows are NaN.

    Classification:
        UP   if log(close[i+h]/close[i]) >  flat_threshold_k * atr_norm[i] * sqrt(h)
        DOWN if log(close[i+h]/close[i]) < -flat_threshold_k * atr_norm[i] * sqrt(h)
        FLAT otherwise

    The sqrt(h) scaling is grounded in random-walk theory: variance grows linearly
    with horizon, so the natural "noise" threshold grows as sqrt(h).
    Using a fixed absolute threshold would classify all h>1 bars as UP or DOWN
    simply because compound returns accumulate over time.
    """
    future_ret = np.log(df["close"].shift(-h) / (df["close"] + 1e-9))
    threshold  = flat_threshold_k * atr_norm * math.sqrt(h)

    labels = pd.Series(float(DirectionLabel.FLAT), index=df.index, dtype=float)
    labels[future_ret > threshold]  = float(DirectionLabel.UP)
    labels[future_ret < -threshold] = float(DirectionLabel.DOWN)
    labels.iloc[-h:] = np.nan
    return labels


def _ret_target(df: pd.DataFrame, h: int) -> pd.Series:
    """
    Signed log return over h bars: log(close[i+h] / close[i]).
    NaN for the last h rows (no complete future window).
    """
    ret = np.log(df["close"].shift(-h) / (df["close"] + 1e-9))
    ret = ret.copy()
    ret.iloc[-h:] = np.nan
    return ret


def _rvol_target(df: pd.DataFrame, h: int) -> pd.Series:
    """
    Realized-volatility forecast for horizon h.

    Definition: RMS of per-bar log-returns over the next h bars.

        rvol_h[i] = sqrt( mean( r[i+1]^2, r[i+2]^2, ..., r[i+h]^2 ) )

    where r[j] = log(close[j] / close[j-1]).

    For h=1:  rvol_1[i] = |r[i+1]|  (identical to y_mag_h1).
    NaN for the last h rows.

    Implementation note
    ~~~~~~~~~~~~~~~~~~~
    sq_ret.rolling(h).mean() at position j covers sq_ret[j-h+1..j].
    After .shift(-h), position i maps to j=i+h, giving sq_ret[i+1..i+h].
    This is exactly the next-h per-bar squared returns — no future leakage
    beyond bar i+h.
    """
    log_ret = np.log(df["close"] / (df["close"].shift(1) + 1e-9))
    sq_ret  = log_ret ** 2
    rvol    = np.sqrt(sq_ret.rolling(h).mean().shift(-h))
    return rvol


def compute_targets(
    df: pd.DataFrame,
    horizons: Tuple[int, ...] = HORIZONS,
    flat_threshold_k: float = FLAT_THRESHOLD_K,
) -> pd.DataFrame:
    """
    Compute all multi-horizon target labels for a bar DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV bars. Required columns: high, low, close, bar_open_time.
    horizons : tuple of int
        Forecast horizons in bars. Default: (1, 3, 5).
    flat_threshold_k : float
        Scaling constant for the ATR-adaptive flat zone. Default: 0.5.
        Larger values → wider flat zone → more FLAT labels → more abstain signals.

    Returns
    -------
    pd.DataFrame
        Same row count as df. Indexed identically. Columns per horizon h:
            y_dir_h{h}      float  0=DOWN, 1=FLAT, 2=UP   (NaN last h rows)
            y_ret_h{h}      float  signed log-return       (NaN last h rows)
            y_mag_h{h}      float  |y_ret_h|               (NaN last h rows)
            y_rvol_h{h}     float  RMS per-bar rvol        (NaN last h rows)
            y_abstain_h{h}  float  1 iff FLAT, 0 otherwise (NaN last h rows)
    """
    atr_n = _atr_norm(df)
    parts: dict = {"bar_open_time": df["bar_open_time"].values}

    for h in sorted(horizons):
        ret  = _ret_target(df, h)
        mag  = ret.abs()
        dir_ = _dir_target(df, h, atr_n, flat_threshold_k)
        rvol = _rvol_target(df, h)

        # Abstain flag: 1 when the expected move is below the flat threshold
        abstain = pd.Series(np.nan, index=df.index, dtype=float)
        valid   = dir_.notna()
        abstain[valid] = (dir_[valid] == float(DirectionLabel.FLAT)).astype(float)

        parts[f"y_dir_h{h}"]     = dir_
        parts[f"y_ret_h{h}"]     = ret
        parts[f"y_mag_h{h}"]     = mag
        parts[f"y_rvol_h{h}"]    = rvol
        parts[f"y_abstain_h{h}"] = abstain

    return pd.DataFrame(parts, index=df.index)


def target_col_names(horizons: Tuple[int, ...] = HORIZONS) -> list:
    """Return ordered list of all target column names for the given horizons."""
    cols = []
    for h in sorted(horizons):
        cols += [
            f"y_dir_h{h}",
            f"y_ret_h{h}",
            f"y_mag_h{h}",
            f"y_rvol_h{h}",
            f"y_abstain_h{h}",
        ]
    return cols
