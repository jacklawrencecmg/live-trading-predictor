"""
Market regime detection from OHLCV features.

Regimes:
- trending_up: sustained upward momentum
- trending_down: sustained downward momentum
- mean_reverting: oscillating price action
- high_volatility: ATR well above recent mean
- low_volatility: ATR well below recent mean

Uses only market-derived features. No external news.
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Optional


class Regime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


def detect_regime(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Detect regime for each bar. Uses only data prior to each bar (shift applied).

    Logic:
    1. Volatility regime: ATR vs long-run ATR
       - high_vol if current_atr > 1.5 * long_atr
       - low_vol if current_atr < 0.5 * long_atr
    2. Trend regime: ADX proxy using directional movement
       - trending if directional strength > threshold
       - mean_reverting otherwise
    """
    c = df["close"].shift(1)
    h = df["high"].shift(1)
    l = df["low"].shift(1)

    # ATR
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(com=13, adjust=False).mean()
    atr_long = tr.ewm(com=49, adjust=False).mean()

    # Directional movement (simplified ADX proxy)
    dm_up = (h - h.shift(1)).clip(lower=0)
    dm_down = (l.shift(1) - l).clip(lower=0)
    dm_up_smooth = dm_up.ewm(com=13, adjust=False).mean()
    dm_down_smooth = dm_down.ewm(com=13, adjust=False).mean()
    dx = (dm_up_smooth - dm_down_smooth).abs() / (dm_up_smooth + dm_down_smooth + 1e-9) * 100
    adx = dx.ewm(com=13, adjust=False).mean()

    # Trend direction (simple: 10-bar vs 30-bar EMA)
    ema_fast = c.ewm(span=10, adjust=False).mean()
    ema_slow = c.ewm(span=30, adjust=False).mean()

    regimes = []
    for i in range(len(df)):
        if i < lookback:
            regimes.append(Regime.UNKNOWN)
            continue

        atr_v = atr.iloc[i]
        atr_l = atr_long.iloc[i]
        adx_v = adx.iloc[i]
        fast = ema_fast.iloc[i]
        slow = ema_slow.iloc[i]

        if np.isnan(atr_v) or np.isnan(adx_v):
            regimes.append(Regime.UNKNOWN)
            continue

        # Volatility first
        if atr_v > 1.5 * (atr_l + 1e-9):
            regimes.append(Regime.HIGH_VOLATILITY)
        elif atr_v < 0.5 * (atr_l + 1e-9):
            regimes.append(Regime.LOW_VOLATILITY)
        elif adx_v > 25:
            regimes.append(Regime.TRENDING_UP if fast > slow else Regime.TRENDING_DOWN)
        else:
            regimes.append(Regime.MEAN_REVERTING)

    return pd.Series(regimes, index=df.index)


def should_suppress_trade(regime: Regime) -> bool:
    """Suppress paper trades in poor regimes."""
    return regime in (Regime.HIGH_VOLATILITY, Regime.UNKNOWN)
