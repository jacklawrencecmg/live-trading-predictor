import asyncio
import math
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.backtest import BacktestResult
from app.schemas.backtest import BacktestRequest, BacktestResultOut
from app.services.feature_pipeline import build_features, features_to_array, FEATURE_NAMES
from app.services.model_service import train_models, compute_calibration, set_models
from app.core.config import settings


def _prepare_dataset(df: pd.DataFrame, iv_rank: float = 0.5, put_call_ratio: float = 1.0, atm_iv: float = 0.2):
    """Build features and labels from OHLCV dataframe."""
    df = df.copy().reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]

    rows = []
    labels_dir = []
    labels_mag = []

    for i in range(30, len(df) - 1):
        window = df.iloc[: i + 1]
        feat = build_features(window, iv_rank, put_call_ratio, atm_iv)
        if feat is None:
            continue
        close_now = float(df["close"].iloc[i])
        close_next = float(df["close"].iloc[i + 1])
        direction = 1 if close_next > close_now else 0
        magnitude = abs(close_next / close_now - 1)
        rows.append(features_to_array(feat))
        labels_dir.append(direction)
        labels_mag.append(magnitude)

    X = np.array(rows)
    y_dir = np.array(labels_dir)
    y_mag = np.array(labels_mag)
    return X, y_dir, y_mag


def _simulate_trades(
    X_test: np.ndarray,
    y_dir_test: np.ndarray,
    dir_model,
    mag_model,
    confidence_threshold: float = 0.60,
    capital: float = 100_000,
    risk_per_trade: float = 0.01,
) -> Dict:
    probs = dir_model.predict_proba(X_test)[:, 1]
    signals = []
    for p in probs:
        confidence = abs(p - 0.5) * 2
        if p > confidence_threshold:
            signals.append(1)
        elif p < (1 - confidence_threshold):
            signals.append(-1)
        else:
            signals.append(0)

    pnl_series = []
    n_trades = 0
    cash = capital

    for sig, actual in zip(signals, y_dir_test):
        if sig == 0:
            pnl_series.append(0.0)
            continue
        position_pnl = risk_per_trade * capital * (1 if sig == actual else -1)
        cash += position_pnl
        pnl_series.append(position_pnl)
        n_trades += 1

    total_return = (cash - capital) / capital
    pnl_arr = np.array(pnl_series)
    active = pnl_arr[pnl_arr != 0]
    if len(active) > 1:
        sharpe = float(active.mean() / (active.std() + 1e-9) * math.sqrt(252))
    else:
        sharpe = 0.0

    return {
        "total_return": round(total_return, 6),
        "sharpe_ratio": round(sharpe, 4),
        "n_trades": n_trades,
        "pnl_series": pnl_series,
    }


async def run_backtest(req: BacktestRequest, db: AsyncSession) -> BacktestResult:
    loop = asyncio.get_event_loop()
    df_raw = await loop.run_in_executor(
        None,
        lambda: yf.Ticker(req.symbol).history(period=req.period, interval=req.interval),
    )
    df_raw.dropna(inplace=True)
    df_raw.columns = [c.lower() for c in df_raw.columns]

    start_date = str(df_raw.index[0].date()) if hasattr(df_raw.index[0], 'date') else str(df_raw.index[0])
    end_date = str(df_raw.index[-1].date()) if hasattr(df_raw.index[-1], 'date') else str(df_raw.index[-1])

    X, y_dir, y_mag = _prepare_dataset(df_raw)

    n = len(X)
    fold_results = []
    all_y_true, all_y_prob = [], []
    all_metrics = {"accuracy": [], "brier": [], "ll": [], "mae": [], "sharpe": [], "return": []}

    step = req.test_size
    n_folds_actual = 0

    start_idx = req.train_size
    while start_idx + req.test_size <= n:
        train_end = start_idx
        test_end = min(start_idx + req.test_size, n)

        X_train = X[max(0, train_end - req.train_size): train_end]
        y_dir_train = y_dir[max(0, train_end - req.train_size): train_end]
        y_mag_train = y_mag[max(0, train_end - req.train_size): train_end]

        X_test = X[train_end:test_end]
        y_dir_test = y_dir[train_end:test_end]
        y_mag_test = y_mag[train_end:test_end]

        if len(X_train) < 20 or len(X_test) < 5:
            start_idx += step
            continue
        if len(np.unique(y_dir_train)) < 2:
            start_idx += step
            continue

        try:
            dir_m, mag_m = train_models(X_train, y_dir_train, y_mag_train)
        except Exception:
            start_idx += step
            continue

        probs = dir_m.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        mag_preds = mag_m.predict(X_test)

        acc = float((preds == y_dir_test).mean())
        bs = float(brier_score_loss(y_dir_test, probs))
        ll = float(log_loss(y_dir_test, probs))
        mae = float(mean_absolute_error(y_mag_test, np.abs(mag_preds)))

        sim = _simulate_trades(X_test, y_dir_test, dir_m, mag_m)

        fold_results.append({
            "fold": n_folds_actual + 1,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy": round(acc, 4),
            "brier_score": round(bs, 6),
            "log_loss": round(ll, 6),
            "magnitude_mae": round(mae, 6),
            "sharpe_ratio": sim["sharpe_ratio"],
            "total_return": sim["total_return"],
            "n_trades": sim["n_trades"],
        })

        all_y_true.extend(y_dir_test.tolist())
        all_y_prob.extend(probs.tolist())
        all_metrics["accuracy"].append(acc)
        all_metrics["brier"].append(bs)
        all_metrics["ll"].append(ll)
        all_metrics["mae"].append(mae)
        all_metrics["sharpe"].append(sim["sharpe_ratio"])
        all_metrics["return"].append(sim["total_return"])

        n_folds_actual += 1
        start_idx += step
        if n_folds_actual >= req.n_folds:
            break

    # Train final model on all data
    if len(X) > 20 and len(np.unique(y_dir)) >= 2:
        dir_m_final, mag_m_final = train_models(X, y_dir, y_mag)
        set_models(dir_m_final, mag_m_final)

    def avg(lst):
        return round(sum(lst) / len(lst), 6) if lst else None

    calib_data = None
    if all_y_true:
        from app.services.model_service import compute_calibration
        calib = compute_calibration(np.array(all_y_true), np.array(all_y_prob))
        calib_data = calib.model_dump()

    result = BacktestResult(
        symbol=req.symbol,
        interval=req.interval,
        start_date=start_date,
        end_date=end_date,
        n_folds=n_folds_actual,
        train_size=req.train_size,
        test_size=req.test_size,
        accuracy=avg(all_metrics["accuracy"]),
        brier_score=avg(all_metrics["brier"]),
        log_loss=avg(all_metrics["ll"]),
        magnitude_mae=avg(all_metrics["mae"]),
        sharpe_ratio=avg(all_metrics["sharpe"]),
        total_return=avg(all_metrics["return"]),
        n_trades=sum(f["n_trades"] for f in fold_results),
        fold_results=fold_results,
        calibration_data=calib_data,
    )
    db.add(result)
    await db.flush()
    return result
