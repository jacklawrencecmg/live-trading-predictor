"""
Target label generation for next-candle prediction.

Label types:
1. Binary: next close > current close → 1, else 0
2. Ternary: up/down/no-trade based on volatility-scaled threshold
3. Regression (return): next bar log return
4. Regression (range): next bar (high - low) / close

LEAKAGE PREVENTION:
- label[i] = f(close[i+1]) — uses only the NEXT bar's close
- label[i] must never use information from bar[i] itself for feature alignment
- All labels are shifted so they align with feature rows
"""

import math
import numpy as np
import pandas as pd
from enum import IntEnum
from typing import Tuple


class TernaryLabel(IntEnum):
    DOWN = 0
    NO_TRADE = 1
    UP = 2


def binary_label(df: pd.DataFrame) -> pd.Series:
    """
    1 if next bar close > current close, else 0.
    label[i] uses df['close'][i+1] vs df['close'][i].
    Last row has NaN.
    """
    return (df["close"].shift(-1) > df["close"]).astype(float)


def ternary_label(
    df: pd.DataFrame,
    threshold_multiplier: float = 0.5,
    use_atr: bool = True,
) -> pd.Series:
    """
    3-class label based on move magnitude vs local volatility.

    - UP   if next_return > +threshold
    - DOWN if next_return < -threshold
    - NO_TRADE otherwise

    threshold = threshold_multiplier * realized_vol (10-bar)

    When use_atr=True, threshold = threshold_multiplier * ATR / close
    Configurable by volatility so no-trade zone adapts to market conditions.
    """
    next_ret = df["close"].shift(-1) / df["close"] - 1

    if use_atr:
        h, l, pc = df["high"], df["low"], df["close"].shift(1)
        tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        atr = tr.ewm(com=13, adjust=False).mean()
        threshold = threshold_multiplier * atr / (df["close"] + 1e-9)
    else:
        log_ret = np.log(df["close"] / df["close"].shift(1))
        vol = log_ret.rolling(10).std()
        threshold = threshold_multiplier * vol

    labels = pd.Series(TernaryLabel.NO_TRADE, index=df.index, dtype=int)
    labels[next_ret > threshold] = int(TernaryLabel.UP)
    labels[next_ret < -threshold] = int(TernaryLabel.DOWN)
    labels.iloc[-1] = np.nan  # last bar has no label

    return labels


def regression_return_label(df: pd.DataFrame) -> pd.Series:
    """
    Log return of next bar: log(close[i+1] / close[i]).
    Preferably used for magnitude estimation.
    """
    return np.log(df["close"].shift(-1) / (df["close"] + 1e-9))


def regression_range_label(df: pd.DataFrame) -> pd.Series:
    """
    Next bar range normalized by close: (high[i+1] - low[i+1]) / close[i].
    Estimates expected volatility magnitude.
    """
    next_range = df["high"].shift(-1) - df["low"].shift(-1)
    return next_range / (df["close"] + 1e-9)


def build_labels(df: pd.DataFrame, threshold_multiplier: float = 0.5) -> pd.DataFrame:
    """Build all labels for a bar DataFrame. Drop last row (no label available)."""
    labels = pd.DataFrame({
        "bar_open_time": df["bar_open_time"],
        "binary": binary_label(df),
        "ternary": ternary_label(df, threshold_multiplier),
        "ret_next": regression_return_label(df),
        "range_next": regression_range_label(df),
    })
    return labels.iloc[:-1].copy()  # drop last row — no next bar


"""
When to use each label:

1. Binary (up/down):
   - Simplest, highest signal count
   - Good for logistic regression baseline
   - Ignores magnitude — win rate alone not sufficient for edge

2. Ternary (up/down/no-trade):
   - Better for options because you want to skip choppy conditions
   - no-trade zone filters out low-edge setups
   - Use when expected move size matters for strategy selection

3. Regression return:
   - Best for sizing and strike selection
   - Higher noise than classification
   - Combine with direction signal for full picture

4. Regression range:
   - Use for IV comparison: if model_range > implied_range, IV may be cheap
   - Useful for selecting spread widths in options strategies
"""
