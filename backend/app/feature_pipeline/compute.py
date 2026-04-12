"""
Feature computation — group-by-group implementation.

Every OHLCV-derived series uses .shift(1) BEFORE any rolling or EWM call so
that feature[i] uses only bars 0..i-1 (point-in-time correct). This is the
shift-by-1 invariant: appending future bars to df must never change historical
feature rows.

Options features use optional_sentinel fill: when options_data is None all
options columns are set to their sentinel values and is_null_options = 1.

Entry point:  compute_features(df, options_data=None) -> pd.DataFrame
"""

import math
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from app.feature_pipeline.registry import (
    FEATURE_COLS,
    OPTIONS_FEATURE_COLS,
    ALL_FEATURE_COLS,
    REGISTRY,
    FFILL_LIMIT,
)

# ---------------------------------------------------------------------------
# Low-level helpers (all operate on already-shifted series)
# ---------------------------------------------------------------------------

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s: pd.Series, n: int) -> pd.Series:
    """Wilder RSI using EWM (com = n-1 ≡ span = 2n-1 in classic form)."""
    d = s.diff()
    gain = d.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    loss = (-d).clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-9))


def _macd(s: pd.Series):
    """Returns (macd_line, signal_line, histogram)."""
    fast = _ema(s, 12)
    slow = _ema(s, 26)
    line = fast - slow
    sig = _ema(line, 9)
    return line, sig, line - sig


def _atr(df_sh: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Average True Range from a *shifted* OHLCV DataFrame.

    df_sh must already be shifted by 1 so that row i contains bar i-1 values.
    prev_close (pc) is then shift(1) of df_sh['close'], i.e., close[i-2].
    This ensures ATR[i] references only bars 0..i-1.
    """
    h = df_sh["high"]
    l = df_sh["low"]
    pc = df_sh["close"].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(com=n - 1, adjust=False).mean()


def _bollinger_pct(s: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    ma = s.rolling(n).mean()
    sd = s.rolling(n).std()
    upper = ma + k * sd
    lower = ma - k * sd
    return (s - lower) / (upper - lower + 1e-9)


def _realized_vol(s: pd.Series, n: int) -> pd.Series:
    """Annualised realized vol from log returns of an already-shifted close series."""
    log_ret = np.log(s / s.shift(1))
    return log_ret.rolling(n).std() * math.sqrt(252 * 78)


# ---------------------------------------------------------------------------
# Group computation functions
# ---------------------------------------------------------------------------

def _compute_trend(c1: pd.Series) -> Dict[str, pd.Series]:
    """
    Trend features: RSI(14), RSI(5), MACD line/signal/hist, Bollinger %B.

    c1: close.shift(1) with ffill(limit=FFILL_LIMIT) already applied.
    """
    rsi14 = _rsi(c1, 14)
    rsi5 = _rsi(c1, 5)
    macd_l, macd_s, macd_h = _macd(c1)
    bb = _bollinger_pct(c1, 20)
    return {
        "rsi_14": rsi14,
        "rsi_5": rsi5,
        "macd_line": macd_l,
        "macd_signal": macd_s,
        "macd_hist": macd_h,
        "bb_pct": bb,
    }


def _compute_volatility(
    c1: pd.Series,
    df_shifted: pd.DataFrame,
) -> Dict[str, pd.Series]:
    """
    Volatility features: ATR(14), ATR norm, realized vols (5/10/20/60), vol_regime.

    df_shifted: df.shift(1).ffill(limit=FFILL_LIMIT) — OHLCV shifted by 1 bar.
    c1:         close.shift(1).ffill(limit=FFILL_LIMIT)
    """
    atr14 = _atr(df_shifted, 14)
    atr_norm = atr14 / (c1 + 1e-9)

    rv5 = _realized_vol(c1, 5)
    rv10 = _realized_vol(c1, 10)
    rv20 = _realized_vol(c1, 20)
    rv60 = _realized_vol(c1, 60)  # intermediate for vol_regime

    vol_regime = rv10 / (rv60 + 1e-9)

    return {
        "atr_14": atr14,
        "atr_norm": atr_norm,
        "realized_vol_5": rv5,
        "realized_vol_10": rv10,
        "realized_vol_20": rv20,
        "vol_regime": vol_regime,
    }


def _compute_momentum(c1: pd.Series) -> Dict[str, pd.Series]:
    """
    Momentum and mean-reversion features: pct returns (1/5/10/20/60) and z-score.

    c1: close.shift(1) — no additional ffill needed; pct_change handles NaN.
    """
    ret1 = c1.pct_change(1, fill_method=None)
    ret5 = c1.pct_change(5, fill_method=None)
    ret10 = c1.pct_change(10, fill_method=None)
    ret20 = c1.pct_change(20, fill_method=None)
    ret60 = c1.pct_change(60, fill_method=None)

    # 20-bar z-score of close
    roll20_mean = c1.rolling(20).mean()
    roll20_std = c1.rolling(20).std()
    zscore20 = (c1 - roll20_mean) / (roll20_std + 1e-9)

    return {
        "ret_1": ret1,
        "ret_5": ret5,
        "ret_10": ret10,
        "ret_20": ret20,
        "ret_60": ret60,
        "zscore_20": zscore20,
    }


def _compute_vwap(df: pd.DataFrame, c1: pd.Series) -> Dict[str, pd.Series]:
    """
    VWAP distance and VWAP slope.

    df:  original (unshifted) DataFrame — we shift inside this function.
    c1:  close.shift(1) (already shifted, no additional ffill needed here).

    vwap_distance: (close[i-1] - vwap[i-1]) / close[i-1]
    vwap_slope:    5-bar pct change of vwap[i-1]
    """
    # Fall back to close when vwap is missing
    vwap = df["vwap"].fillna(df["close"])
    vwap1 = vwap.shift(1)  # prior bar's VWAP

    vwap_dist = (c1 - vwap1) / (c1 + 1e-9)

    # 5-bar slope: (vwap[i-1] - vwap[i-6]) / |vwap[i-6]|
    vwap_slope = vwap1.pct_change(5, fill_method=None)

    return {
        "vwap_distance": vwap_dist,
        "vwap_slope": vwap_slope,
    }


def _compute_volume(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Volume features: ratio, trend, z-score.
    All use volume.shift(1) for point-in-time correctness.
    """
    vol = df["volume"].shift(1)
    vol_ma20 = vol.rolling(20).mean()
    vol_ma5 = vol.rolling(5).mean()
    vol_std20 = vol.rolling(20).std()

    vol_ratio = (vol / (vol_ma20 + 1e-9)).clip(0, 10)
    vol_trend = (vol_ma5 / (vol_ma20 + 1e-9)).clip(0, 5)
    vol_zscore = (vol - vol_ma20) / (vol_std20 + 1e-9)

    return {
        "volume_ratio": vol_ratio,
        "volume_trend": vol_trend,
        "volume_zscore": vol_zscore,
    }


def _compute_seasonality(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Intraday seasonality features from bar_open_time.
    These are NOT shifted — bar_open_time is a property of the prediction target bar,
    not a value derived from past price data, so no lookahead applies.
    """
    ts = pd.to_datetime(df["bar_open_time"])
    hour_frac = ts.dt.hour + ts.dt.minute / 60

    hour_sin = np.sin(2 * np.pi * hour_frac / 24)
    hour_cos = np.cos(2 * np.pi * hour_frac / 24)
    min_sin = np.sin(2 * np.pi * ts.dt.minute / 60)
    min_cos = np.cos(2 * np.pi * ts.dt.minute / 60)

    # Session progress: minutes since 09:30 ET / 390 total minutes, clipped [0, 1]
    session_minutes = (hour_frac - 9.5) * 60  # minutes after 09:30
    session_progress = (session_minutes / 390.0).clip(0.0, 1.0)

    is_first30 = ((session_minutes >= 0) & (session_minutes <= 30)).astype(float)
    is_last30 = ((session_minutes >= 360) & (session_minutes <= 390)).astype(float)

    return {
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "minute_sin": min_sin,
        "minute_cos": min_cos,
        "session_progress": session_progress,
        "is_first_30min": is_first30,
        "is_last_30min": is_last30,
    }


def _compute_options(
    options_data: Optional[Dict[str, Any]],
    n_rows: int,
) -> Dict[str, pd.Series]:
    """
    Options chain summary features.

    When options_data is None (no snapshot available), all features are filled
    with their sentinel values and is_null_options is set to 1.

    options_data keys (all optional within the dict):
        atm_iv, iv_rank, iv_skew, pc_volume_ratio, pc_oi_ratio,
        gex_proxy, dist_to_max_oi
    """
    idx = range(n_rows)

    if options_data is None:
        result = {}
        for name in OPTIONS_FEATURE_COLS:
            result[name] = pd.Series(
                REGISTRY[name].sentinel_value, index=idx, dtype=float
            )
        result["is_null_options"] = pd.Series(1.0, index=idx)
        return result

    result = {}
    for name in OPTIONS_FEATURE_COLS:
        val = options_data.get(name)
        sentinel = REGISTRY[name].sentinel_value
        result[name] = pd.Series(
            float(val) if val is not None else sentinel,
            index=idx,
            dtype=float,
        )
    result["is_null_options"] = pd.Series(0.0, index=idx)
    return result


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def compute_features(
    df: pd.DataFrame,
    options_data: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix from a closed-bar OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Columns: open, high, low, close, volume, vwap (optional), bar_open_time
        Rows must be in chronological order. All bars should be closed.

    options_data : dict or None
        Point-in-time options chain summary for the last bar. Keys:
        atm_iv, iv_rank, iv_skew, pc_volume_ratio, pc_oi_ratio,
        gex_proxy, dist_to_max_oi.
        When None, options features are sentinel-filled and is_null_options=1.

    Returns
    -------
    pd.DataFrame
        One row per input row. Column set = ALL_FEATURE_COLS + ['bar_open_time'].
        Features for row i use only bars 0..i-1 (shift-by-1 invariant).
        Rows with insufficient warmup data (< lookback) will have NaN in the
        corresponding features — callers should filter with valid_mask().

    Notes
    -----
    CRITICAL invariant: all OHLCV-derived series call .shift(1) BEFORE any
    rolling/EWM operation. Appending N future bars to df must not change any
    historical feature row (max diff < 1e-9). See test_L2_* in test_leakage.py.
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)

    # ------------------------------------------------------------------
    # Build shifted series — the foundation of all OHLCV features
    # ------------------------------------------------------------------
    c = df["close"]
    c1 = c.shift(1).ffill(limit=FFILL_LIMIT)
    df_shifted = df.shift(1)
    df_shifted_filled = df_shifted.ffill(limit=FFILL_LIMIT)

    # ------------------------------------------------------------------
    # Compute each feature group
    # ------------------------------------------------------------------
    trend = _compute_trend(c1)
    vol = _compute_volatility(c1, df_shifted_filled)
    mom = _compute_momentum(c1)
    vwap = _compute_vwap(df, c1)
    volume = _compute_volume(df)
    season = _compute_seasonality(df)
    options = _compute_options(options_data, n)

    # ------------------------------------------------------------------
    # Assemble into a single DataFrame
    # ------------------------------------------------------------------
    parts = {
        "bar_open_time": df["bar_open_time"],
        **trend,
        **vol,
        **mom,
        **vwap,
        **volume,
        **season,
        **options,
    }
    feat_df = pd.DataFrame(parts)

    return feat_df


def valid_mask(feat_df: pd.DataFrame) -> pd.Series:
    """
    Boolean mask: True for rows where all FEATURE_COLS are non-NaN.

    Use this to filter the training set:
        mask = valid_mask(feat_df)
        X = feat_df.loc[mask, FEATURE_COLS].values
    """
    return ~feat_df[FEATURE_COLS].isna().any(axis=1)
