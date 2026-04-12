import math
from typing import Optional
import numpy as np
import pandas as pd
from app.schemas.model import FeatureSet


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - 100 / (1 + rs)


def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _bollinger_position(series: pd.Series, period=20, std=2.0) -> pd.Series:
    ma = series.rolling(period).mean()
    sd = series.rolling(period).std()
    upper = ma + std * sd
    lower = ma - std * sd
    return (series - lower) / (upper - lower + 1e-9)


def build_features(
    df: pd.DataFrame,  # columns: open, high, low, close, volume (lowercase)
    iv_rank: float = 0.5,
    put_call_ratio: float = 1.0,
    atm_iv: float = 0.2,
) -> Optional[FeatureSet]:
    """
    Build feature vector from the most recent complete candle.
    Requires at least 30 rows.
    """
    if len(df) < 30:
        return None

    df = df.copy().reset_index(drop=True)
    close = df["close"]
    volume = df["volume"]

    rsi14 = _rsi(close, 14)
    rsi5 = _rsi(close, 5)
    macd_line, macd_signal, macd_hist = _macd(close)
    atr = _atr(df, 14)
    bb_pos = _bollinger_position(close)

    vol_ma = volume.rolling(20).mean()
    vol_ratio = volume / (vol_ma + 1e-9)

    # Momentum: return over N periods
    mom5 = close.pct_change(5)
    mom10 = close.pct_change(10)
    mom20 = close.pct_change(20)

    idx = -2  # Use second-to-last as last complete candle
    current_price = float(close.iloc[idx])

    def safe(s, i=-2):
        v = s.iloc[i]
        return 0.0 if (np.isnan(v) or np.isinf(v)) else float(v)

    return FeatureSet(
        rsi_14=safe(rsi14),
        rsi_5=safe(rsi5),
        macd_line=safe(macd_line),
        macd_signal=safe(macd_signal),
        macd_hist=safe(macd_hist),
        bb_position=safe(bb_pos),
        atr_norm=safe(atr) / (current_price + 1e-9),
        volume_ratio=min(safe(vol_ratio), 10.0),
        momentum_5=safe(mom5),
        momentum_10=safe(mom10),
        momentum_20=safe(mom20),
        iv_rank=iv_rank,
        put_call_ratio=put_call_ratio,
        atm_iv=atm_iv,
    )


def features_to_array(f: FeatureSet) -> list:
    return [
        f.rsi_14, f.rsi_5, f.macd_line, f.macd_signal, f.macd_hist,
        f.bb_position, f.atr_norm, f.volume_ratio,
        f.momentum_5, f.momentum_10, f.momentum_20,
        f.iv_rank, f.put_call_ratio, f.atm_iv,
    ]


FEATURE_NAMES = [
    "rsi_14", "rsi_5", "macd_line", "macd_signal", "macd_hist",
    "bb_position", "atr_norm", "volume_ratio",
    "momentum_5", "momentum_10", "momentum_20",
    "iv_rank", "put_call_ratio", "atm_iv",
]
