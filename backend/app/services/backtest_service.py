import asyncio
import math
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.backtest import BacktestResult
from app.schemas.backtest import BacktestRequest, BacktestResultOut
from app.feature_pipeline.features import build_feature_matrix, FEATURE_COLS
from app.services.model_service import train_models, compute_calibration, set_models
from app.ml_models.baseline import save_model
from app.core.config import settings


def _prepare_dataset(df: pd.DataFrame, iv_rank: float = 0.5, put_call_ratio: float = 1.0, atm_iv: float = 0.2):
    """
    Build features and labels from OHLCV dataframe using the leakage-safe pipeline.

    Feature alignment (IMPORTANT):
    - build_feature_matrix() applies .shift(1) so feat[i] uses only bars 0..i-1.
      feat[i] represents information available at the OPEN of bar i.
    - label[i] = sign(close[i+1] - close[i]) — direction of the bar that follows i.
    - Row i in X predicts row i in y: no future data appears in any feature row.

    This replaces the previous O(n²) loop that called build_features(window[:i+1])
    per bar. That loop also produced a 14-feature vector (the old pipeline) while
    inference_service uses a 22-feature vector (the new pipeline), causing a
    training/inference feature-dimension mismatch that would crash or silently
    corrupt predictions at run time.

    Options features (iv_rank, put_call_ratio, atm_iv) are broadcast constants
    here because the backtest only uses OHLCV data. In production they should be
    time-stamped snapshots filtered to snapshot_time <= bar_open_time[i].
    """
    df = df.copy().reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]

    # build_feature_matrix requires a bar_open_time column.
    if "bar_open_time" not in df.columns:
        if hasattr(df.index, 'to_pydatetime') or str(df.index.dtype).startswith('datetime'):
            df["bar_open_time"] = df.index.values
        else:
            # Placeholder timestamps preserve row ordering; time features will be
            # inaccurate but do not affect correctness of the leakage test.
            df["bar_open_time"] = pd.date_range(
                start="2020-01-02 14:30", periods=len(df), freq="5min"
            )

    # VWAP approximation if not already present
    if "vwap" not in df.columns:
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3

    # Build feature matrix in O(n) — leakage-safe via .shift(1)
    feat_df = build_feature_matrix(df)

    # Labels aligned to the same row index as features
    close = df["close"]
    y_dir_series = (close.shift(-1) > close).astype(int)   # 1 = up, 0 = down/flat
    y_mag_series = (close.shift(-1) / close - 1).abs()

    # Drop last row: label[n-1] requires close[n] which does not exist
    X_raw = feat_df[FEATURE_COLS].values[:-1]
    y_dir = y_dir_series.values[:-1]
    y_mag = y_mag_series.values[:-1]

    # Remove warmup rows where features are NaN (initial lookback windows not yet full)
    valid = ~np.isnan(X_raw).any(axis=1)
    return X_raw[valid], y_dir[valid], y_mag[valid]


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
    # Sharpe includes all periods (active and idle) so idle time penalises the ratio.
    # Using only active trades would inflate Sharpe by ignoring the cost of waiting.
    #
    # Annualization: pnl_arr contains one value per 5-minute bar.
    # Periods per year for 5-min bars = 252 trading days × 78 bars/day = 19,656.
    # Using sqrt(252) instead would inflate Sharpe by sqrt(78) ≈ 8.83×.
    _BARS_PER_YEAR = 252 * 78
    if len(pnl_arr) > 1 and pnl_arr.std() > 0:
        sharpe = float(pnl_arr.mean() / pnl_arr.std() * math.sqrt(_BARS_PER_YEAR))
    else:
        sharpe = 0.0

    return {
        "total_return": round(total_return, 6),
        "sharpe_ratio": round(sharpe, 4),
        "n_trades": n_trades,
        "pnl_series": pnl_series,
    }


# Embargo gap between the end of the training window and the start of the test
# window.  label[i] = sign(close[i+1] - close[i]), so the last training label
# (at index train_end-1) uses close[train_end].  Without an embargo, the test
# set starts at train_end, and its first label uses close[train_end+1] — which
# is clean — but the bar at train_end itself straddles the boundary.  One bar
# of embargo ensures no single bar contributes to both the training label and
# the test feature set.  This matches the default in PurgedWalkForwardSplit.
_EMBARGO_BARS = 1


async def run_backtest(req: BacktestRequest, db: AsyncSession) -> BacktestResult:
    import yfinance as yf  # lazy import: only needed at runtime, not test collection
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

        # Apply embargo: skip _EMBARGO_BARS rows after train_end to prevent
        # the boundary bar's label from contaminating the test feature set.
        X_test = X[train_end + _EMBARGO_BARS:test_end]
        y_dir_test = y_dir[train_end + _EMBARGO_BARS:test_end]
        y_mag_test = y_mag[train_end + _EMBARGO_BARS:test_end]

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

    # Train and promote the production model on all available data.
    # NOTE: This is intentional. Walk-forward folds evaluate on held-out windows
    # using per-fold models. The final model is trained on all data so inference
    # can use the most recent information. It is NOT evaluated on the same data
    # it was trained on — that evaluation was done fold-by-fold above.
    # Callers should treat fold metrics as the honest out-of-sample estimate.
    if len(X) > 20 and len(np.unique(y_dir)) >= 2:
        dir_m_final, mag_m_final = train_models(X, y_dir, y_mag)
        set_models(dir_m_final, mag_m_final)
        save_model(dir_m_final, "logistic")

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
