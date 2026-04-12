"""
Training pipeline — top-level orchestrator.

Usage:
    from app.ml_models.pipeline import run_training_pipeline
    from app.ml_models.training.config import TrainingConfig

    report = run_training_pipeline(df, cfg=TrainingConfig())
    # report.winner.final_model  → use for inference
    # report to markdown:
    from app.ml_models.report import generate_report
    md = generate_report(report)

The pipeline:
  1. Builds feature matrix from df via compute_features()
  2. Applies valid_mask() to drop warmup rows
  3. Builds labels from close prices
  4. Extracts returns and regime labels for trading utility analysis
  5. Calls train_all_models() with the configured models
  6. Saves the winner to the model registry
  7. Optionally writes the markdown report to output_dir
  8. Returns TrainingReport (full results)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.ml_models.training.config import TrainingConfig, DEFAULT_CONFIG
from app.ml_models.training.trainer import TrainingReport, train_all_models
from app.ml_models.model_registry import save_model
from app.ml_models.report import generate_report
from app.feature_pipeline.compute import compute_features, valid_mask
from app.feature_pipeline.registry import FEATURE_COLS

logger = logging.getLogger(__name__)


def run_training_pipeline(
    df: pd.DataFrame,
    cfg: TrainingConfig = DEFAULT_CONFIG,
    options_data: Optional[dict] = None,
    output_dir: Optional[Path] = None,
    symbol: str = "unknown",
) -> TrainingReport:
    """
    End-to-end training pipeline from raw OHLCV DataFrame to a trained model.

    Parameters
    ----------
    df : pd.DataFrame
        Closed-bar OHLCV history. Columns: open, high, low, close, volume,
        vwap (optional), bar_open_time.
        Minimum recommended length: 1500 bars (more = better walk-forward folds).
    cfg : TrainingConfig
        Reproducible hyperparameter configuration. Defaults to DEFAULT_CONFIG.
    options_data : dict or None
        Point-in-time options summary for the LAST bar only. Passed to
        compute_features(). In training, options features are typically not
        available per-bar, so most training runs will use None.
    output_dir : Path or None
        If provided, writes training_report.md and training_report.json here.
    symbol : str
        Ticker symbol for logging and artifact metadata.

    Returns
    -------
    TrainingReport with full evaluation results and the winner's final model.
    """
    logger.info("=== Training pipeline start: symbol=%s, config=%s ===", symbol, cfg.config_hash())

    # ------------------------------------------------------------------
    # 1. Feature matrix
    # ------------------------------------------------------------------
    feat_df = compute_features(df, options_data=options_data)
    mask = valid_mask(feat_df)

    logger.info(
        "Feature matrix: %d total rows, %d valid after warmup (%.1f%% dropped)",
        len(feat_df), int(mask.sum()), (1 - mask.mean()) * 100,
    )

    feat_valid = feat_df[mask].reset_index(drop=True)
    df_valid = df[mask.values].reset_index(drop=True)

    X = feat_valid[list(cfg.feature_cols)].values.astype(float)
    feature_names = list(cfg.feature_cols)

    # ------------------------------------------------------------------
    # 2. Labels: binary direction (1 = close[i+1] > close[i])
    # ------------------------------------------------------------------
    close = df_valid["close"].values
    # Label for row i is whether close[i+1] > close[i]
    # Last row has no label → drop it
    labels = (close[1:] > close[:-1]).astype(int)
    X = X[:-1]          # align: drop last row (no label available)
    feat_valid = feat_valid.iloc[:-1].reset_index(drop=True)
    df_valid = df_valid.iloc[:-1].reset_index(drop=True)

    y = labels
    n = len(y)
    logger.info(
        "Labels: %d samples | up=%.1f%% down=%.1f%%",
        n, y.mean() * 100, (1 - y.mean()) * 100,
    )

    # ------------------------------------------------------------------
    # 3. Returns (for trading utility metrics)
    # ------------------------------------------------------------------
    returns = (close[1:] - close[:-1]) / (close[:-1] + 1e-9)  # 1-bar pct return

    # ------------------------------------------------------------------
    # 4. Regime labels (optional — skip gracefully if detector fails)
    # ------------------------------------------------------------------
    regimes = None
    try:
        from app.regime.detector import detect_regime
        regime_series = detect_regime(df_valid)
        regimes = regime_series.values.astype(str)
    except Exception as e:
        logger.warning("Regime detection failed, skipping regime evaluation: %s", e)

    # ------------------------------------------------------------------
    # 5. Sanity checks
    # ------------------------------------------------------------------
    if n < cfg.min_train_bars + 100:
        raise ValueError(
            f"Insufficient data after warmup: {n} samples. "
            f"Need at least {cfg.min_train_bars + 100}. "
            f"Provide more history (recommend ≥ 1500 bars)."
        )

    if len(np.unique(y)) < 2:
        raise ValueError("All labels are the same class — check the price data.")

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    report = train_all_models(
        X=X, y=y, cfg=cfg,
        feature_names=feature_names,
        returns=returns,
        regimes=regimes,
    )

    # ------------------------------------------------------------------
    # 7. Save winner to registry
    # ------------------------------------------------------------------
    winner = report.winner
    if winner.final_model is not None:
        metrics = {
            "brier_score_mean": winner.aggregated.brier_score_mean,
            "brier_score_std": winner.aggregated.brier_score_std,
            "log_loss_mean": winner.aggregated.log_loss_mean,
            "roc_auc_mean": winner.aggregated.roc_auc_mean,
            "balanced_accuracy_mean": winner.aggregated.balanced_accuracy_mean,
            "ece_mean": winner.aggregated.ece_mean,
            "n_folds": winner.aggregated.n_folds,
            "total_test_samples": winner.aggregated.total_test_samples,
        }
        artifact_dir = save_model(
            model=winner.final_model,
            model_name=winner.model_name,
            config_hash=cfg.config_hash(),
            metrics=metrics,
            feature_names=feature_names,
            training_report_dict=report.to_summary_dict(),
            notes=f"symbol={symbol}",
        )
        logger.info("Artifact saved: %s", artifact_dir)

    # ------------------------------------------------------------------
    # 8. Write report files
    # ------------------------------------------------------------------
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        md = generate_report(report)
        (output_dir / "training_report.md").write_text(md, encoding="utf-8")
        logger.info("Report written: %s", output_dir / "training_report.md")

        import json
        (output_dir / "training_report.json").write_text(
            json.dumps(report.to_summary_dict(), indent=2, default=str),
            encoding="utf-8",
        )

    logger.info(
        "=== Pipeline complete: winner=%s brier=%.4f ===",
        winner.model_name, winner.aggregated.brier_score_mean,
    )
    return report
