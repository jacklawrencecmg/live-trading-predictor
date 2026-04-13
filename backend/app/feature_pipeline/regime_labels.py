"""
Multi-label regime classification for training data segmentation.

Unlike the inference-time regime detector (app/regime/detector.py), which produces
a single mutually-exclusive regime per bar via a priority cascade, this module
produces independent boolean flags. A bar can simultaneously be HIGH_VOL and
TRENDING, or ABNORMAL and LIQUIDITY_POOR.

Purpose
-------
1. Training stratification: understand model performance by regime:

       results.groupby("regime_trending")[["brier", "accuracy"]].mean()

2. Regime target: train a companion meta-classifier that predicts which regime
   the market will be in at time i+h, feeding regime awareness into the inference
   confidence calculation.

3. Conditional model evaluation: compute Brier scores and calibration curves
   separately inside/outside each regime flag.

Shift-by-1 guarantee
--------------------
All OHLCV signals are shifted by 1 before any computation. regime_label[i]
depends only on bars 0..i-1. The label is knowable at bar-i open time —
point-in-time correct with no lookahead.

Columns produced by compute_regime_labels()
-------------------------------------------
regime_trending        adx_proxy > ADX_TREND_THRESHOLD (strong directional momentum)
regime_mean_reverting  adx_proxy <= ADX_TREND_THRESHOLD (rangebound / low momentum)
regime_low_vol         atr_ratio < ATR_LOW_VOL_RATIO (compressed vs long-run ATR)
regime_high_vol        atr_ratio > ATR_HIGH_VOL_RATIO (expanded vs long-run ATR)
regime_liquidity_poor  volume_ratio < LIQUIDITY_VOL_THRESHOLD (thin market)
regime_abnormal        |pct_change| > sigma threshold AND > min abs return
regime_warmup          first WARMUP_BARS rows (insufficient history for signals)

Note: regime_trending and regime_mean_reverting are complements — exactly one
is True post-warmup. The other five flags are independent orthogonal indicators.
"""

import numpy as np
import pandas as pd

# Tuning constants — intentionally aligned with app/regime/detector.py where
# semantically equivalent, with minor differences noted below.
WARMUP_BARS: int = 20
ADX_TREND_THRESHOLD: float = 25.0
ATR_HIGH_VOL_RATIO: float = 1.50
ATR_LOW_VOL_RATIO: float = 0.50
LIQUIDITY_VOL_THRESHOLD: float = 0.30    # slightly wider than detector's 0.25:
                                          # labels are symmetric; detector is asymmetric
                                          # (combines volume + bar-range conditions)
ABNORMAL_SIGMA: float = 3.0              # slightly tighter than detector's 3.5:
                                          # labeling benefits from higher sensitivity
ABNORMAL_MIN_STD: float = 0.0005
ABNORMAL_MIN_ABS_RETURN: float = 0.005

REGIME_LABEL_COLS = [
    "regime_trending",
    "regime_mean_reverting",
    "regime_low_vol",
    "regime_high_vol",
    "regime_liquidity_poor",
    "regime_abnormal",
    "regime_warmup",
]


def compute_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute independent boolean regime flags for each bar.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV bars. Required columns: high, low, close, volume.

    Returns
    -------
    pd.DataFrame
        Same index as df. Columns: REGIME_LABEL_COLS (all dtype=bool).
        Multiple flags can be True simultaneously for the same row.
    """
    # Shift all OHLCV series by 1: regime[i] uses only data from bars 0..i-1
    c  = df["close"].shift(1).ffill()
    h  = df["high"].shift(1).ffill()
    lw = df["low"].shift(1).ffill()
    v  = df["volume"].shift(1).ffill()

    # -----------------------------------------------------------------------
    # ATR (short and long) for volatility regime classification
    # -----------------------------------------------------------------------
    pc       = c.shift(1)
    tr       = pd.concat([h - lw, (h - pc).abs(), (lw - pc).abs()], axis=1).max(axis=1)
    atr_short = tr.ewm(com=13, adjust=False).mean()    # ~14-bar half-life
    atr_long  = tr.ewm(com=49, adjust=False).mean()    # ~50-bar half-life
    atr_ratio = atr_short / (atr_long + 1e-9)

    # -----------------------------------------------------------------------
    # ADX proxy — directional movement strength (same formula as detector.py)
    # -----------------------------------------------------------------------
    dm_up   = (h - h.shift(1)).clip(lower=0)
    dm_down = (lw.shift(1) - lw).clip(lower=0)
    dmu_s   = dm_up.ewm(com=13, adjust=False).mean()
    dmd_s   = dm_down.ewm(com=13, adjust=False).mean()
    dx      = (dmu_s - dmd_s).abs() / (dmu_s + dmd_s + 1e-9) * 100
    adx     = dx.ewm(com=13, adjust=False).mean()

    # -----------------------------------------------------------------------
    # Volume ratio (current bar vs 20-bar rolling average)
    # -----------------------------------------------------------------------
    vol_avg      = v.rolling(20, min_periods=5).mean()
    volume_ratio = v / (vol_avg + 1e-9)

    # -----------------------------------------------------------------------
    # Abnormal move detection
    # -----------------------------------------------------------------------
    ret          = c.pct_change()
    roll_std     = ret.rolling(20, min_periods=10).std()
    roll_std_s   = roll_std.clip(lower=ABNORMAL_MIN_STD)
    abnormal_sig = ret.abs() / roll_std_s
    # Dual guard: high sigma ratio AND minimum absolute return.
    # The absolute return guard prevents false positives on synthetic constant
    # series where rolling_std → 0 making any tiny move appear extreme.
    is_abnormal  = (abnormal_sig > ABNORMAL_SIGMA) & (ret.abs() > ABNORMAL_MIN_ABS_RETURN)

    # -----------------------------------------------------------------------
    # Warmup mask
    # -----------------------------------------------------------------------
    warmup_mask = pd.Series(False, index=df.index)
    warmup_mask.iloc[:WARMUP_BARS] = True

    out = pd.DataFrame(index=df.index)
    out["regime_trending"]       = (adx > ADX_TREND_THRESHOLD)          & ~warmup_mask
    out["regime_mean_reverting"] = (adx <= ADX_TREND_THRESHOLD)         & ~warmup_mask
    out["regime_low_vol"]        = (atr_ratio < ATR_LOW_VOL_RATIO)      & ~warmup_mask
    out["regime_high_vol"]       = (atr_ratio > ATR_HIGH_VOL_RATIO)     & ~warmup_mask
    out["regime_liquidity_poor"] = (volume_ratio < LIQUIDITY_VOL_THRESHOLD) & ~warmup_mask
    out["regime_abnormal"]       = is_abnormal                           & ~warmup_mask
    out["regime_warmup"]         = warmup_mask

    return out.astype(bool)
