"""
Regime API.

GET /api/regime/{symbol}
    Current regime context: label, signals, thresholds, suppression state.

GET /api/regime/{symbol}/history
    Recent per-bar regime history (for timeline visualization).

GET /api/regime/{symbol}/distribution
    Fraction of time spent in each regime over stored history.

POST /api/regime/{symbol}/label
    Manually trigger regime detection for the current bar and persist it.
    Normally called automatically from inference; useful for testing/backfill.
"""

import logging
import pandas as pd
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.data_ingestion.ingestion_service import get_closed_bars
from app.regime.detector import (
    detect_regime_row,
    detect_regime_full,
    Regime,
    REGIME_THRESHOLDS,
)
from app.regime.store import (
    save_regime_label,
    load_recent_regime_labels,
    load_regime_distribution,
    load_regime_performance_data,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Human-readable regime descriptions for UI
REGIME_DESCRIPTIONS = {
    "trending_up":    "Sustained upward directional move — momentum strategies favored.",
    "trending_down":  "Sustained downward directional move — momentum strategies favored.",
    "mean_reverting": "Oscillating price action without clear direction — mean-reversion signals more reliable.",
    "high_volatility": "ATR significantly above baseline — execution risk elevated; trading suppressed.",
    "low_volatility":  "ATR significantly below baseline — limited opportunity; higher bar for entry.",
    "liquidity_poor":  "Volume and/or bar range anomalously low — execution slippage risk; trading suppressed.",
    "event_risk":      "Abnormal price movement detected (> 3.5σ) — discontinuous move; model unreliable; trading suppressed.",
    "unknown":         "Insufficient historical data for regime classification.",
}

# Which regimes suppress trading
SUPPRESSED_REGIMES = {r for r, t in REGIME_THRESHOLDS.items() if not t.allow_trade}


def _bars_to_df(bars) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "open": b.open, "high": b.high, "low": b.low, "close": b.close,
            "volume": b.volume,
            "vwap": b.vwap or (b.high + b.low + b.close) / 3,
            "bar_open_time": b.bar_open_time,
        }
        for b in bars
    ])


@router.get("/{symbol}")
async def get_regime(
    symbol: str,
    timeframe: str = Query("5m"),
    db: AsyncSession = Depends(get_db),
):
    """
    Current regime for a symbol: label, context signals, trading thresholds.
    """
    symbol = symbol.upper()
    bars = await get_closed_bars(db, symbol, timeframe, limit=100)

    if len(bars) < 30:
        try:
            from app.services.market_data import fetch_candles
            candles_resp = await fetch_candles(symbol, timeframe, "5d")
            df = pd.DataFrame([
                {
                    "open": c.open, "high": c.high, "low": c.low, "close": c.close,
                    "volume": c.volume,
                    "vwap": (c.high + c.low + c.close) / 3,
                    "bar_open_time": pd.Timestamp(c.time, unit="s"),
                }
                for c in candles_resp.candles
            ])
        except Exception as e:
            return {"error": f"Insufficient data: {e}", "symbol": symbol}
    else:
        df = _bars_to_df(bars)

    try:
        ctx = detect_regime_row(df)
    except Exception as e:
        logger.error("Regime detection error for %s: %s", symbol, e)
        return {"error": str(e), "symbol": symbol}

    regime_str = str(ctx.regime.value if hasattr(ctx.regime, "value") else ctx.regime)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "regime": regime_str,
        "description": REGIME_DESCRIPTIONS.get(regime_str, ""),
        "suppressed": ctx.suppressed,
        "suppress_reason": ctx.suppress_reason,
        "confidence_threshold": ctx.confidence_threshold,
        "min_signal_quality": ctx.min_signal_quality,
        "signals": {
            "adx_proxy": ctx.adx_proxy,
            "atr_ratio": ctx.atr_ratio,
            "volume_ratio": ctx.volume_ratio,
            "bar_range_ratio": ctx.bar_range_ratio,
            "trend_direction": ctx.trend_direction,
            "ema_spread_pct": ctx.ema_spread_pct,
            "is_abnormal_move": ctx.is_abnormal_move,
            "abnormal_move_sigma": ctx.abnormal_move_sigma,
        },
        "thresholds": {
            r.value if hasattr(r, "value") else r: {
                "confidence_threshold": t.confidence_threshold,
                "min_signal_quality": t.min_signal_quality,
                "allow_trade": t.allow_trade,
            }
            for r, t in REGIME_THRESHOLDS.items()
        },
    }


@router.get("/{symbol}/history")
async def get_regime_history(
    symbol: str,
    timeframe: str = Query("5m"),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """
    Recent stored regime labels (chronological, newest last).
    Used for the regime timeline visualization in the UI.
    """
    symbol = symbol.upper()
    rows = await load_recent_regime_labels(db, symbol, timeframe, limit=limit)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "count": len(rows),
        "history": [
            {
                "bar_open_time": r.bar_open_time,
                "regime": r.regime,
                "adx_proxy": r.adx_proxy,
                "atr_ratio": r.atr_ratio,
                "volume_ratio": r.volume_ratio,
                "is_abnormal_move": r.is_abnormal_move,
                "suppressed": r.suppressed,
            }
            for r in reversed(rows)  # chronological
        ],
    }


@router.get("/{symbol}/distribution")
async def get_regime_distribution(
    symbol: str,
    timeframe: str = Query("5m"),
    db: AsyncSession = Depends(get_db),
):
    """
    Fraction of time spent in each regime over stored history.
    Also returns which regimes suppress trading.
    """
    symbol = symbol.upper()
    distribution = await load_regime_distribution(db, symbol, timeframe)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "distribution": distribution,
        "suppressed_regimes": list(
            r.value if hasattr(r, "value") else r
            for r in SUPPRESSED_REGIMES
        ),
        "descriptions": REGIME_DESCRIPTIONS,
    }


@router.post("/{symbol}/label")
async def label_regime(
    symbol: str,
    timeframe: str = Query("5m"),
    db: AsyncSession = Depends(get_db),
):
    """
    Detect and persist the current-bar regime. Idempotent (upserts).
    Normally called automatically from inference; useful for backfill.
    """
    symbol = symbol.upper()
    bars = await get_closed_bars(db, symbol, timeframe, limit=100)
    if len(bars) < 30:
        return {"error": "Insufficient bars for regime detection", "symbol": symbol}

    df = _bars_to_df(bars)
    try:
        ctx = detect_regime_row(df)
    except Exception as e:
        return {"error": str(e), "symbol": symbol}

    bar_time = str(df["bar_open_time"].iloc[-1])
    await save_regime_label(db, symbol, timeframe, bar_time, ctx)

    regime_str = str(ctx.regime.value if hasattr(ctx.regime, "value") else ctx.regime)
    return {
        "symbol": symbol,
        "bar_open_time": bar_time,
        "regime": regime_str,
        "suppressed": ctx.suppressed,
        "stored": True,
    }
