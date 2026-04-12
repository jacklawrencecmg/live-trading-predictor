"""
Signals API — combines inference + rules engine into a single actionable response.

GET /api/signals/{symbol}
  Returns: prediction (4-layer) + scored signal + trade idea + uncertainty context
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd

from app.core.database import get_db
from app.data_ingestion.ingestion_service import get_closed_bars
from app.inference.inference_service import run_inference
from app.inference.signal_scorer import score_signal
from app.paper_trading.rules_engine import evaluate_rules, RulesConfig

router = APIRouter()


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
async def get_signal(
    symbol: str,
    timeframe: str = Query("5m"),
    confidence_threshold: float = Query(0.55),
    min_signal_quality: float = Query(40.0),
    db: AsyncSession = Depends(get_db),
):
    symbol = symbol.upper()
    bars = await get_closed_bars(db, symbol, timeframe, limit=300)

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

    result = run_inference(df, symbol, confidence_threshold)

    rv = float(df["close"].pct_change().tail(20).std() * 100 * (252 * 78) ** 0.5)
    signal = score_signal(
        raw_prob_up=result.prob_up,
        raw_prob_down=result.prob_down,
        calibrated_prob_up=result.calibrated_prob_up,
        calibrated_prob_down=result.calibrated_prob_down,
        calibration_available=result.calibration_available,
        tradeable_confidence=result.tradeable_confidence,
        degradation_factor=result.degradation_factor,
        abstain_reason=result.abstain_reason,
        calibration_health=result.calibration_health,
        ece_recent=result.ece_recent,
        rolling_brier=result.rolling_brier,
        confidence_band=result.confidence_band,
        expected_move_pct=result.expected_move_pct,
        realized_vol_pct=rv,
        regime=result.regime,
        no_trade_reason=result.no_trade_reason,
        explanation=result.explanation,
        top_features=result.top_features,
    )

    config = RulesConfig(min_signal_quality_score=min_signal_quality)
    trade_idea = evaluate_rules(
        symbol=symbol,
        prob_up=result.calibrated_prob_up,
        prob_down=result.calibrated_prob_down,
        confidence=result.tradeable_confidence,
        expected_move_pct=result.expected_move_pct,
        signal_quality_score=signal.signal_quality_score,
        regime=result.regime,
        config=config,
    )

    return {
        "symbol": symbol,
        "prediction": {
            # Layer 1
            "prob_up": result.prob_up,
            "prob_down": result.prob_down,
            # Layer 2
            "calibrated_prob_up": result.calibrated_prob_up,
            "calibrated_prob_down": result.calibrated_prob_down,
            "calibration_available": result.calibration_available,
            # Layer 3
            "tradeable_confidence": result.tradeable_confidence,
            "degradation_factor": result.degradation_factor,
            # Layer 4
            "action": result.action,
            "abstain_reason": result.abstain_reason,
            # Supporting
            "confidence_band": list(result.confidence_band),
            "calibration_health": result.calibration_health,
            "rolling_brier": result.rolling_brier,
            "ece_recent": result.ece_recent,
            "reliability_diagram": result.reliability_diagram,
            # Meta
            "expected_move_pct": result.expected_move_pct,
            "model_version": result.model_version,
            "bar_open_time": result.bar_open_time,
            "feature_snapshot_id": result.feature_snapshot_id,
            # Backward compat
            "confidence": result.confidence,
        },
        "signal": {
            "direction": signal.direction,
            "raw_probability": signal.raw_probability,
            "probability": signal.probability,
            "tradeable_confidence": signal.tradeable_confidence,
            "confidence": signal.confidence,
            "confidence_band": list(signal.confidence_band),
            "degradation_factor": signal.degradation_factor,
            "calibration_health": signal.calibration_health,
            "calibration_available": signal.calibration_available,
            "ece_recent": signal.ece_recent,
            "rolling_brier": signal.rolling_brier,
            "abstain_reason": signal.abstain_reason,
            "signal_quality_score": signal.signal_quality_score,
            "volatility_context": signal.volatility_context,
            "regime": signal.regime,
            "explanation": signal.explanation,
            "top_features": signal.top_features,
        },
        "trade_idea": {
            "direction": trade_idea.direction,
            "strategy": trade_idea.strategy,
            "target_delta": trade_idea.target_delta,
            "blocked": trade_idea.blocked,
            "block_reason": trade_idea.block_reason,
            "rationale": trade_idea.rationale,
        },
    }
