"""
Feature engineering pipeline.

LEAKAGE PREVENTION:
- All features are computed from bars with bar_open_time < target_bar_open_time
- The target bar is NEVER included in the feature window
- Rolling stats use .shift(1) before .rolling() to exclude the current bar
- Options features use snapshot_time < bar_open_time of the target bar
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
import pandas as pd


@dataclass
class FeatureVector:
    """Single feature row for one prediction target."""
    # Identity
    symbol: str
    bar_open_time: str  # ISO string, the target bar open time
    timeframe: str

    # Price features
    rsi_14: float
    rsi_5: float
    macd_line: float
    macd_signal: float
    macd_hist: float
    bb_pct: float          # Bollinger Band %B: 0=lower, 1=upper
    atr_norm: float        # ATR(14) / close
    vwap_distance: float   # (close - vwap) / close

    # Momentum
    ret_1: float           # 1-bar return
    ret_5: float
    ret_10: float
    ret_20: float

    # Volume
    volume_ratio: float    # volume / 20-bar mean volume
    volume_trend: float    # 5-bar mean / 20-bar mean

    # Intraday time features
    hour_sin: float
    hour_cos: float
    minute_sin: float
    minute_cos: float
    is_first_30min: float  # 1 if within first 30min of session
    is_last_30min: float   # 1 if within last 30min of session

    # Volatility
    realized_vol_10: float  # 10-bar realized vol annualized
    realized_vol_20: float
    atr_14: float

    # Options chain features (None if unavailable)
    atm_iv: Optional[float] = None
    iv_rank: Optional[float] = None
    iv_skew: Optional[float] = None          # OTM put IV - OTM call IV (25-delta proxy)
    pc_volume_ratio: Optional[float] = None  # put volume / call volume
    pc_oi_ratio: Optional[float] = None      # put OI / call OI
    gamma_exposure: Optional[float] = None   # sum(gamma * OI * 100 * spot)
    dist_to_max_oi: Optional[float] = None   # (spot - strike_max_oi) / spot

    # Regime
    regime: Optional[str] = None  # "trending", "mean_reverting", "high_vol", "low_vol"


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s: pd.Series, n: int) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    l_ = (-d).clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    return 100 - 100 / (1 + g / (l_ + 1e-9))


def _macd(s: pd.Series):
    fast = _ema(s, 12)
    slow = _ema(s, 26)
    line = fast - slow
    sig = _ema(line, 9)
    return line, sig, line - sig


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(com=n - 1, adjust=False).mean()


def _bollinger_pct(s: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    ma = s.rolling(n).mean()
    sd = s.rolling(n).std()
    upper = ma + k * sd
    lower = ma - k * sd
    return (s - lower) / (upper - lower + 1e-9)


def _realized_vol(s: pd.Series, n: int) -> pd.Series:
    log_ret = np.log(s / s.shift(1))
    return log_ret.rolling(n).std() * math.sqrt(252 * 78)  # annualize for 5m bars


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from OHLCV DataFrame.

    CRITICAL: All features use .shift(1) BEFORE any rolling, so feature[i]
    uses data from bar[i-1] and earlier. This means features for bar[i]
    are available at the OPEN of bar[i], never using bar[i]'s own data.

    df columns: open, high, low, close, volume, vwap, bar_open_time (datetime)
    """
    df = df.copy().reset_index(drop=True)
    c = df["close"]

    # Shift by 1 to prevent lookahead: feature at row i uses data from rows 0..i-1
    c1 = c.shift(1)

    rsi14 = _rsi(c1.fillna(method="ffill"), 14)
    rsi5 = _rsi(c1.fillna(method="ffill"), 5)
    macd_l, macd_s, macd_h = _macd(c1.fillna(method="ffill"))

    # ATR and BB based on shifted close
    df_shifted = df.shift(1)
    atr14 = _atr(df_shifted.fillna(method="ffill"), 14)
    bb = _bollinger_pct(c1.fillna(method="ffill"), 20)

    vwap = df["vwap"].fillna(c)
    vwap_dist = (c1 - vwap.shift(1)) / (c1 + 1e-9)

    # Returns (using shifted close)
    ret1 = c1.pct_change(1)
    ret5 = c1.pct_change(5)
    ret10 = c1.pct_change(10)
    ret20 = c1.pct_change(20)

    # Volume
    vol = df["volume"].shift(1)
    vol_ma20 = vol.rolling(20).mean()
    vol_ma5 = vol.rolling(5).mean()
    vol_ratio = vol / (vol_ma20 + 1e-9)
    vol_trend = vol_ma5 / (vol_ma20 + 1e-9)

    # Time features
    if hasattr(df["bar_open_time"].iloc[0], "hour"):
        hours = df["bar_open_time"].dt.hour + df["bar_open_time"].dt.minute / 60
    else:
        hours = pd.to_datetime(df["bar_open_time"]).dt.hour + pd.to_datetime(df["bar_open_time"]).dt.minute / 60

    ts = pd.to_datetime(df["bar_open_time"])
    hour_sin = np.sin(2 * np.pi * (ts.dt.hour + ts.dt.minute / 60) / 24)
    hour_cos = np.cos(2 * np.pi * (ts.dt.hour + ts.dt.minute / 60) / 24)
    min_sin = np.sin(2 * np.pi * ts.dt.minute / 60)
    min_cos = np.cos(2 * np.pi * ts.dt.minute / 60)
    session_hour = ts.dt.hour + ts.dt.minute / 60 - 9.5
    is_first30 = ((session_hour >= 0) & (session_hour <= 0.5)).astype(float)
    is_last30 = ((session_hour >= 6.0) & (session_hour <= 6.5)).astype(float)

    # Realized vol
    rv10 = _realized_vol(c1.fillna(method="ffill"), 10)
    rv20 = _realized_vol(c1.fillna(method="ffill"), 20)

    feat_df = pd.DataFrame({
        "bar_open_time": df["bar_open_time"],
        "rsi_14": rsi14,
        "rsi_5": rsi5,
        "macd_line": macd_l,
        "macd_signal": macd_s,
        "macd_hist": macd_h,
        "bb_pct": bb,
        "atr_norm": atr14 / (c1 + 1e-9),
        "atr_14": atr14,
        "vwap_distance": vwap_dist,
        "ret_1": ret1,
        "ret_5": ret5,
        "ret_10": ret10,
        "ret_20": ret20,
        "volume_ratio": vol_ratio.clip(0, 10),
        "volume_trend": vol_trend.clip(0, 5),
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "minute_sin": min_sin,
        "minute_cos": min_cos,
        "is_first_30min": is_first30,
        "is_last_30min": is_last30,
        "realized_vol_10": rv10,
        "realized_vol_20": rv20,
    })

    return feat_df


FEATURE_COLS = [
    "rsi_14", "rsi_5", "macd_line", "macd_signal", "macd_hist",
    "bb_pct", "atr_norm", "vwap_distance",
    "ret_1", "ret_5", "ret_10", "ret_20",
    "volume_ratio", "volume_trend",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "is_first_30min", "is_last_30min",
    "realized_vol_10", "realized_vol_20",
]

OPTIONS_FEATURE_COLS = [
    "atm_iv", "iv_rank", "iv_skew", "pc_volume_ratio", "pc_oi_ratio",
    "gamma_exposure", "dist_to_max_oi",
]

ALL_FEATURE_COLS = FEATURE_COLS + OPTIONS_FEATURE_COLS
