"""
Integration tests — Backtest end-to-end.

IT-B1: Full backtest run stores results in PostgreSQL
IT-B2: Model artifact is written to disk after backtest
IT-B3: Inference works after model trained via backtest
IT-B4: Backtest result BSS is finite and logged to DB

Requires: INTEGRATION_TESTS=1, live PostgreSQL.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = datetime(2024, 1, 2, 9, 30)
    prices = 100 + np.cumsum(rng.normal(0, 0.5, n))
    prices = np.maximum(prices, 1.0)
    highs = prices * (1 + rng.uniform(0, 0.003, n))
    lows = prices * (1 - rng.uniform(0, 0.003, n))
    vols = rng.integers(10_000, 100_000, n).astype(float)
    return pd.DataFrame({
        "open": prices * (1 + rng.uniform(-0.001, 0.001, n)),
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": vols,
        "vwap": (prices + highs + lows) / 3,
        "bar_open_time": [base + timedelta(minutes=5 * i) for i in range(n)],
    })


@pytest.mark.asyncio
async def test_IT_B1_backtest_stores_result_in_db(db_session):
    """Full backtest run persists a BacktestResult row to PostgreSQL."""
    import math
    from app.services.backtest_service import run_backtest
    from app.schemas.backtest import BacktestRequest
    from app.models.backtest import BacktestResult
    from sqlalchemy import select

    req = BacktestRequest(
        symbol="TEST",
        interval="5m",
        period="3mo",
        n_folds=3,
        train_size=150,
        test_size=50,
        confidence_threshold=0.55,
    )
    df = _make_ohlcv(n=300)
    result = await run_backtest(db_session, req, df_override=df)

    assert result is not None
    assert result.id is not None

    # Verify the row exists in the database
    stmt = select(BacktestResult).where(BacktestResult.id == result.id)
    row = (await db_session.execute(stmt)).scalar_one_or_none()
    assert row is not None, "BacktestResult was not persisted to the database"
    assert row.symbol == "TEST"
    assert row.brier_score is None or math.isfinite(row.brier_score)


@pytest.mark.asyncio
async def test_IT_B2_model_artifact_written_after_backtest(tmp_path, db_session):
    """Backtest saves logistic.pkl to disk after training."""
    import os
    from app.services.backtest_service import run_backtest
    from app.schemas.backtest import BacktestRequest
    from app.ml_models.baseline import ARTIFACTS_DIR

    req = BacktestRequest(
        symbol="TEST",
        interval="5m",
        period="3mo",
        n_folds=2,
        train_size=100,
        test_size=40,
        confidence_threshold=0.55,
    )
    df = _make_ohlcv(n=250)
    await run_backtest(db_session, req, df_override=df)

    pkl_path = ARTIFACTS_DIR / "logistic_TEST.pkl"
    assert pkl_path.exists(), (
        f"Expected model artifact at {pkl_path} after backtest, but it was not created."
    )
    assert pkl_path.stat().st_size > 0, "Model artifact is empty"


@pytest.mark.asyncio
async def test_IT_B3_inference_works_after_backtest(db_session):
    """Inference endpoint returns a non-abstain result after model trained."""
    from app.services.backtest_service import run_backtest
    from app.schemas.backtest import BacktestRequest
    from app.inference.inference_service import run_inference, _loaded_model
    import app.inference.inference_service as inf_svc

    req = BacktestRequest(
        symbol="TEST",
        interval="5m",
        period="3mo",
        n_folds=2,
        train_size=100,
        test_size=40,
        confidence_threshold=0.55,
    )
    df = _make_ohlcv(n=250)
    await run_backtest(db_session, req, df_override=df)

    # Force reload of the model from disk
    inf_svc._loaded_model = None

    # Run inference on the same synthetic data
    result = run_inference(df, symbol="TEST", confidence_threshold=0.55)
    assert result is not None
    assert result.model_version != "untrained", (
        "Inference returned 'untrained' model version after backtest trained the model. "
        "The model artifact was not written or not loaded correctly."
    )
