"""
Market regime detection from OHLCV features.

Regime priority (first match wins at each bar):
  1. EVENT_RISK      — abnormal price move (> ABNORMAL_SIGMA × recent σ)
  2. LIQUIDITY_POOR  — volume and/or bar range anomalously low
  3. HIGH_VOLATILITY — short-run ATR >> long-run ATR
  4. LOW_VOLATILITY  — short-run ATR << long-run ATR
  5. TRENDING_UP     — ADX-proxy strong, fast EMA above slow EMA
  6. TRENDING_DOWN   — ADX-proxy strong, fast EMA below slow EMA
  7. MEAN_REVERTING  — ADX-proxy weak, normal volatility
  8. UNKNOWN         — insufficient warmup data

Design notes:
- All signals use .shift(1) so feature[i] depends only on data up to bar i-1.
  Point-in-time safe: the regime label at bar i is knowable at bar i open time.
- Fully vectorized — no Python row loop.
- detect_regime()      : returns pd.Series[Regime] (backward compat)
- detect_regime_full() : returns pd.DataFrame with all intermediate signals
- detect_regime_row()  : extracts RegimeContext from the last row (for inference)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
WARMUP_BARS: int = 20           # rows with insufficient history → UNKNOWN
ADX_TREND_THRESHOLD: float = 25  # ADX proxy above this → trending
ATR_HIGH_VOL_RATIO: float = 1.5  # current_atr / long_atr > this → HIGH_VOL
ATR_LOW_VOL_RATIO: float = 0.50  # current_atr / long_atr < this → LOW_VOL
ABNORMAL_SIGMA: float = 3.5          # |ret_1| > N × rolling_std → EVENT_RISK
ABNORMAL_MIN_STD: float = 0.0005     # minimum std to avoid division by near-zero
ABNORMAL_MIN_ABS_RETURN: float = 0.005  # absolute return must also exceed 0.5% per bar
LIQUIDITY_VOL_RATIO: float = 0.25    # volume / avg_volume < this → LIQUIDITY_POOR
LIQUIDITY_RANGE_RATIO: float = 0.20  # bar_range / atr < this (combined with low vol)


class Regime(str, Enum):
    TRENDING_UP    = "trending_up"
    TRENDING_DOWN  = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY  = "low_volatility"
    LIQUIDITY_POOR  = "liquidity_poor"
    EVENT_RISK      = "event_risk"
    UNKNOWN         = "unknown"


@dataclass(frozen=True)
class RegimeThresholds:
    """Per-regime trading parameters applied at inference time."""
    confidence_threshold: float   # minimum calibrated_prob distance from 0.5
    min_signal_quality: float     # minimum signal quality score (0-100)
    allow_trade: bool             # hard block regardless of confidence
    suppress_reason: Optional[str]  # block reason when allow_trade=False


# ---------------------------------------------------------------------------
# Per-regime thresholds
# Rational:
#   TRENDING_*    — model works best here; default thresholds
#   MEAN_REVERTING — momentum features less predictive; raise bar slightly
#   LOW_VOLATILITY — signal noise ratio is lower; tighter quality threshold
#   HIGH_VOLATILITY — execution risk; hard block
#   LIQUIDITY_POOR  — execution slippage risk; hard block
#   EVENT_RISK      — discontinuous move; model not trained for this; hard block
#   UNKNOWN         — insufficient data; hard block
# ---------------------------------------------------------------------------
REGIME_THRESHOLDS: dict = {
    Regime.TRENDING_UP:    RegimeThresholds(0.55, 40.0, True,  None),
    Regime.TRENDING_DOWN:  RegimeThresholds(0.55, 40.0, True,  None),
    Regime.MEAN_REVERTING: RegimeThresholds(0.58, 48.0, True,  None),
    Regime.LOW_VOLATILITY: RegimeThresholds(0.60, 52.0, True,  None),
    Regime.HIGH_VOLATILITY: RegimeThresholds(0.65, 60.0, False, "high_volatility"),
    Regime.LIQUIDITY_POOR:  RegimeThresholds(0.70, 65.0, False, "liquidity_poor"),
    Regime.EVENT_RISK:      RegimeThresholds(0.70, 70.0, False, "event_risk"),
    Regime.UNKNOWN:         RegimeThresholds(0.60, 50.0, False, "unknown_regime"),
}


@dataclass
class RegimeContext:
    """Full signal decomposition for the most recent bar."""
    regime: Regime

    # Volatility signals
    atr_ratio: float          # current_atr / long_atr; 1.0 = at baseline
    realized_vol_ratio: float # recent realized vol / baseline

    # Trend signals
    adx_proxy: float          # directional movement strength (0–100)
    trend_direction: str      # "up" | "down" | "flat"
    ema_spread_pct: float     # (fast_ema - slow_ema) / slow_ema

    # Volume / liquidity signals
    volume_ratio: float       # bar volume / 20-bar average
    bar_range_ratio: float    # bar range / ATR

    # Abnormal move
    is_abnormal_move: bool
    abnormal_move_sigma: float  # how many sigmas the last return was

    # Trading gate
    suppressed: bool
    suppress_reason: Optional[str]
    confidence_threshold: float
    min_signal_quality: float


def _build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all intermediate regime signals. All series are shift(1)-safe.
    Returns a DataFrame aligned with df.index.
    """
    c = df["close"].shift(1).ffill()
    h = df["high"].shift(1).ffill()
    l = df["low"].shift(1).ffill()
    v = df["volume"].shift(1).ffill()

    # -----------------------------------------------------------------------
    # ATR (short and long)
    # -----------------------------------------------------------------------
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr_short = tr.ewm(com=13, adjust=False).mean()   # ~14-bar half-life
    atr_long  = tr.ewm(com=49, adjust=False).mean()   # ~50-bar half-life
    atr_ratio = atr_short / (atr_long + 1e-9)

    # -----------------------------------------------------------------------
    # ADX proxy (directional movement)
    # -----------------------------------------------------------------------
    dm_up   = (h - h.shift(1)).clip(lower=0)
    dm_down = (l.shift(1) - l).clip(lower=0)
    dmu_s = dm_up.ewm(com=13, adjust=False).mean()
    dmd_s = dm_down.ewm(com=13, adjust=False).mean()
    dx    = (dmu_s - dmd_s).abs() / (dmu_s + dmd_s + 1e-9) * 100
    adx   = dx.ewm(com=13, adjust=False).mean()

    # -----------------------------------------------------------------------
    # EMA cross (trend direction)
    # -----------------------------------------------------------------------
    ema_fast = c.ewm(span=10, adjust=False).mean()
    ema_slow = c.ewm(span=30, adjust=False).mean()
    ema_spread_pct = (ema_fast - ema_slow) / (ema_slow.abs() + 1e-9)

    # -----------------------------------------------------------------------
    # Volume ratio
    # -----------------------------------------------------------------------
    vol_avg    = v.rolling(20, min_periods=5).mean()
    volume_ratio = v / (vol_avg + 1e-9)

    # -----------------------------------------------------------------------
    # Bar range ratio (bar range vs ATR; very low = thin/illiquid market)
    # -----------------------------------------------------------------------
    bar_range = h - l
    bar_range_ratio = bar_range / (atr_short + 1e-9)

    # -----------------------------------------------------------------------
    # Abnormal move detection
    # -----------------------------------------------------------------------
    ret = c.pct_change()
    roll_std = ret.rolling(20, min_periods=10).std()
    roll_std_safe = roll_std.clip(lower=ABNORMAL_MIN_STD)
    abnormal_sigma = ret.abs() / roll_std_safe
    # Require both a high sigma ratio AND a minimum absolute move to avoid
    # false positives on constant-return data (e.g. synthetic linear trends
    # where rolling_std → 0 but returns are predictable and small).
    is_abnormal = (abnormal_sigma > ABNORMAL_SIGMA) & (ret.abs() > ABNORMAL_MIN_ABS_RETURN)

    return pd.DataFrame({
        "atr_ratio": atr_ratio,
        "atr_short": atr_short,
        "atr_long": atr_long,
        "adx": adx,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "ema_spread_pct": ema_spread_pct,
        "volume_ratio": volume_ratio,
        "bar_range_ratio": bar_range_ratio,
        "is_abnormal": is_abnormal.astype(bool),
        "abnormal_sigma": abnormal_sigma,
    }, index=df.index)


def detect_regime_full(df: pd.DataFrame, lookback: int = WARMUP_BARS) -> pd.DataFrame:
    """
    Returns a DataFrame with all intermediate signals AND the final regime label.
    Columns: all signal columns + 'regime' (Regime enum).
    """
    sig = _build_signals(df)
    n = len(df)

    warmup_mask = pd.Series(False, index=df.index)
    warmup_mask.iloc[:lookback] = True

    # Regime conditions (evaluated in priority order)
    event_risk_cond   = sig["is_abnormal"] & ~warmup_mask
    liquidity_cond    = (
        (sig["volume_ratio"] < LIQUIDITY_VOL_RATIO) |
        ((sig["volume_ratio"] < 0.40) & (sig["bar_range_ratio"] < LIQUIDITY_RANGE_RATIO))
    ) & ~warmup_mask & ~event_risk_cond
    high_vol_cond     = (sig["atr_ratio"] > ATR_HIGH_VOL_RATIO) & ~warmup_mask & ~event_risk_cond & ~liquidity_cond
    low_vol_cond      = (sig["atr_ratio"] < ATR_LOW_VOL_RATIO) & ~warmup_mask & ~event_risk_cond & ~liquidity_cond & ~high_vol_cond
    trending_up_cond  = (sig["adx"] > ADX_TREND_THRESHOLD) & (sig["ema_fast"] >= sig["ema_slow"]) & ~warmup_mask & ~event_risk_cond & ~liquidity_cond & ~high_vol_cond & ~low_vol_cond
    trending_dn_cond  = (sig["adx"] > ADX_TREND_THRESHOLD) & (sig["ema_fast"] < sig["ema_slow"]) & ~warmup_mask & ~event_risk_cond & ~liquidity_cond & ~high_vol_cond & ~low_vol_cond
    mean_rev_cond     = ~warmup_mask & ~event_risk_cond & ~liquidity_cond & ~high_vol_cond & ~low_vol_cond & ~trending_up_cond & ~trending_dn_cond

    conditions = [
        warmup_mask,
        event_risk_cond,
        liquidity_cond,
        high_vol_cond,
        low_vol_cond,
        trending_up_cond,
        trending_dn_cond,
        mean_rev_cond,
    ]
    # Use string values in np.select to avoid numpy enum truncation issues,
    # then convert back to Regime enum instances.
    choices_str = [
        Regime.UNKNOWN.value,
        Regime.EVENT_RISK.value,
        Regime.LIQUIDITY_POOR.value,
        Regime.HIGH_VOLATILITY.value,
        Regime.LOW_VOLATILITY.value,
        Regime.TRENDING_UP.value,
        Regime.TRENDING_DOWN.value,
        Regime.MEAN_REVERTING.value,
    ]

    regime_str_arr = np.select(conditions, choices_str, default=Regime.UNKNOWN.value)
    sig["regime"] = [Regime(v) for v in regime_str_arr]
    return sig


def detect_regime(df: pd.DataFrame, lookback: int = WARMUP_BARS) -> pd.Series:
    """
    Backward-compatible: returns pd.Series[Regime] aligned with df.index.
    """
    full = detect_regime_full(df, lookback)
    return pd.Series(full["regime"].values, index=df.index)


def detect_regime_row(df: pd.DataFrame, lookback: int = WARMUP_BARS) -> RegimeContext:
    """
    Compute regime for the LAST bar of df. Returns a RegimeContext dataclass
    for use at inference time.
    """
    full = detect_regime_full(df, lookback)
    last = full.iloc[-1]

    regime = last["regime"]
    thresholds = get_regime_thresholds(regime)

    ema_spread = float(last["ema_spread_pct"])
    trend_dir = "up" if ema_spread > 0 else ("down" if ema_spread < 0 else "flat")

    return RegimeContext(
        regime=regime,
        atr_ratio=round(float(last["atr_ratio"]), 4),
        realized_vol_ratio=round(float(last["atr_ratio"]), 4),  # ATR ratio is a vol ratio proxy
        adx_proxy=round(float(last["adx"]), 2),
        trend_direction=trend_dir,
        ema_spread_pct=round(ema_spread, 6),
        volume_ratio=round(float(last["volume_ratio"]), 4),
        bar_range_ratio=round(float(last["bar_range_ratio"]), 4),
        is_abnormal_move=bool(last["is_abnormal"]),
        abnormal_move_sigma=round(float(last["abnormal_sigma"]), 2),
        suppressed=not thresholds.allow_trade,
        suppress_reason=thresholds.suppress_reason,
        confidence_threshold=thresholds.confidence_threshold,
        min_signal_quality=thresholds.min_signal_quality,
    )


def get_regime_thresholds(regime) -> RegimeThresholds:
    """Look up trading thresholds for a regime. Accepts Regime enum or string."""
    if isinstance(regime, str):
        try:
            regime = Regime(regime)
        except ValueError:
            regime = Regime.UNKNOWN
    return REGIME_THRESHOLDS.get(regime, REGIME_THRESHOLDS[Regime.UNKNOWN])


def should_suppress_trade(regime) -> bool:
    """Backward-compatible: returns True if trading should be suppressed."""
    return not get_regime_thresholds(regime).allow_trade
