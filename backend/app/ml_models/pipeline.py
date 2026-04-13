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

import dataclasses
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from app.ml_models.training.config import TrainingConfig, DEFAULT_CONFIG
from app.ml_models.training.trainer import TrainingReport, train_all_models
from app.ml_models.model_registry import save_model
from app.ml_models.report import generate_report, generate_multi_horizon_report
from app.feature_pipeline.compute import compute_features, valid_mask
from app.feature_pipeline.registry import FEATURE_COLS

logger = logging.getLogger(__name__)


def run_training_pipeline(
    df: pd.DataFrame,
    cfg: TrainingConfig = DEFAULT_CONFIG,
    options_data: Optional[dict] = None,
    output_dir: Optional[Path] = None,
    symbol: str = "unknown",
    n_splits: Optional[int] = None,
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
    n_splits : int or None
        Shortcut to override cfg.n_splits without constructing a new TrainingConfig.

    Returns
    -------
    TrainingReport with full evaluation results and the winner's final model.
    """
    _n_splits_explicit = n_splits is not None
    if n_splits is not None:
        cfg = dataclasses.replace(cfg, n_splits=n_splits)
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
    # 4b. Auto-scale test_window_bars when n_splits was explicitly set and
    #     the default test window is too large for the available data.
    #     This allows small-data tests (e.g. n≈400) to run with n_splits=3.
    # ------------------------------------------------------------------
    if _n_splits_explicit and cfg.test_window_bars is not None:
        required = cfg.n_splits * cfg.test_window_bars + (cfg.n_splits - 1) * cfg.embargo_bars
        if required >= n:
            new_tw = max(20, (n - cfg.n_splits * cfg.embargo_bars) // (cfg.n_splits + 1))
            new_min_train = max(20, new_tw // 2)
            cfg = dataclasses.replace(cfg, test_window_bars=new_tw, min_train_bars=new_min_train)
            logger.info(
                "Auto-scaled test_window_bars=%d, min_train_bars=%d for n=%d, n_splits=%d",
                new_tw, new_min_train, n, cfg.n_splits,
            )

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


# ---------------------------------------------------------------------------
# Multi-horizon pipeline
# ---------------------------------------------------------------------------

# Default model set for ternary multi-horizon training.
# Persistence and vol_no_trade replace the binary-only momentum baselines.
_MULTI_HORIZON_MODELS = (
    "persistence",
    "vol_no_trade",
    "logistic_l2",
    "logistic_l1",
    "gbt",
)


def run_multi_horizon_pipeline(
    df: pd.DataFrame,
    horizons: Tuple[int, ...] = (1, 3, 5),
    output_dir: Optional[Path] = None,
    symbol: str = "unknown",
    base_cfg: Optional[TrainingConfig] = None,
) -> Dict[int, TrainingReport]:
    """
    Multi-horizon ternary training pipeline.

    For each horizon h in ``horizons``:
      - Uses ternary direction labels (y ∈ {0=DOWN, 1=FLAT, 2=UP}) from
        ``compute_targets()``.
      - Sets ``embargo_bars = h`` to prevent h-bar label-overlap leakage.
      - Evaluates using UP-vs-rest binarization (P(UP) as the signal score).
      - Saves per-model prediction artifact CSVs if ``output_dir`` is given.
      - Saves the winner model artifact via the model registry.

    Parameters
    ----------
    df : pd.DataFrame
        Closed-bar OHLCV history.  Columns: open, high, low, close, volume,
        bar_open_time.  Recommended minimum: 1500 bars.
    horizons : tuple of int
        Forecast horizons in bars.  Default: (1, 3, 5).
    output_dir : Path or None
        If provided, writes per-horizon report MD/JSON and prediction CSVs.
    symbol : str
        Ticker symbol for logging and artifact metadata.
    base_cfg : TrainingConfig or None
        If provided, ``embargo_bars`` and ``label_type`` are overridden
        per-horizon; all other hyperparameters are inherited.  Useful for
        tests and for controlling n_splits, test_window_bars, etc.
        If None, uses sensible production defaults.

    Returns
    -------
    dict[int, TrainingReport]
        One ``TrainingReport`` per horizon.
    """
    from app.feature_pipeline.targets import compute_targets

    logger.info(
        "=== Multi-horizon pipeline start: symbol=%s, horizons=%s ===",
        symbol, horizons,
    )

    # ------------------------------------------------------------------
    # 1. Feature matrix (computed once, shared across all horizons)
    # ------------------------------------------------------------------
    feat_df = compute_features(df, options_data=None)
    mask = valid_mask(feat_df)

    logger.info(
        "Feature matrix: %d total rows, %d valid after warmup",
        len(feat_df), int(mask.sum()),
    )

    feat_valid = feat_df[mask].reset_index(drop=True)
    df_valid = df[mask.values].reset_index(drop=True)
    X_all = feat_valid[list(FEATURE_COLS)].values.astype(float)
    feature_names = list(FEATURE_COLS)
    n_total = len(X_all)

    # ------------------------------------------------------------------
    # 2. Multi-horizon targets
    # ------------------------------------------------------------------
    targets_df = compute_targets(df_valid)

    # ------------------------------------------------------------------
    # 3. 1-bar log returns (for trading-utility analysis, aligned to X_all)
    # ------------------------------------------------------------------
    close_vals = df_valid["close"].values
    log_ret_full = np.log(close_vals / np.roll(close_vals, 1))
    log_ret_full[0] = np.nan

    # ------------------------------------------------------------------
    # 4. Regime labels
    # ------------------------------------------------------------------
    regimes_full = None
    try:
        from app.regime.detector import detect_regime
        regimes_full = detect_regime(df_valid).values.astype(str)
    except Exception as e:
        logger.warning("Regime detection failed: %s", e)

    # ------------------------------------------------------------------
    # 5. Per-horizon training loop
    # ------------------------------------------------------------------
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    reports: Dict[int, TrainingReport] = {}

    for h in sorted(horizons):
        logger.info("--- Horizon h=%d ---", h)

        y_raw = targets_df[f"y_dir_h{h}"].values
        valid_rows = ~np.isnan(y_raw)

        X = X_all[valid_rows]
        y = y_raw[valid_rows].astype(int)
        rets = log_ret_full[valid_rows]
        regs = regimes_full[valid_rows] if regimes_full is not None else None

        if len(np.unique(y)) < 2:
            logger.warning("Horizon h=%d: fewer than 2 unique label values, skipping.", h)
            continue

        if base_cfg is not None:
            # Inherit all hyperparameters from base_cfg; override per-horizon fields.
            cfg = dataclasses.replace(
                base_cfg,
                embargo_bars=h,
                label_type="ternary",
            )
        else:
            cfg = TrainingConfig(
                embargo_bars=h,
                label_type="ternary",
                model_names=_MULTI_HORIZON_MODELS,
                run_ablation=False,
            )

        n = len(y)
        if n < cfg.min_train_bars + 100:
            logger.warning(
                "Horizon h=%d: only %d valid samples (need %d), skipping.",
                h, n, cfg.min_train_bars + 100,
            )
            continue

        report = train_all_models(
            X=X, y=y, cfg=cfg,
            feature_names=feature_names,
            returns=rets,
            regimes=regs,
        )

        # ---------------------------------------------------------------
        # Save winner to model registry
        # ---------------------------------------------------------------
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
                model_name=f"{winner.model_name}_h{h}",
                config_hash=cfg.config_hash(),
                metrics=metrics,
                feature_names=feature_names,
                training_report_dict=report.to_summary_dict(),
                notes=f"symbol={symbol}, horizon={h}, label_type=ternary",
            )
            logger.info("h=%d artifact saved: %s", h, artifact_dir)

        # ---------------------------------------------------------------
        # Write per-horizon report and prediction artifacts
        # ---------------------------------------------------------------
        if output_dir is not None:
            h_dir = output_dir / f"h{h}"
            h_dir.mkdir(exist_ok=True)

            md = generate_report(report)
            (h_dir / "training_report.md").write_text(md, encoding="utf-8")
            (h_dir / "training_report.json").write_text(
                json.dumps(report.to_summary_dict(), indent=2, default=str),
                encoding="utf-8",
            )

            _save_prediction_artifacts(report, h, h_dir, symbol)

        reports[h] = report
        logger.info(
            "h=%d complete: winner=%s brier=%.4f",
            h, winner.model_name, winner.aggregated.brier_score_mean,
        )

    # ------------------------------------------------------------------
    # 6. Multi-horizon summary report
    # ------------------------------------------------------------------
    if reports and output_dir is not None:
        summary_md = generate_multi_horizon_report(reports)
        (output_dir / "multi_horizon_report.md").write_text(summary_md, encoding="utf-8")

    logger.info("=== Multi-horizon pipeline complete: %d horizons ===", len(reports))
    return reports


def _save_prediction_artifacts(
    report: TrainingReport,
    horizon: int,
    output_dir: Path,
    symbol: str,
) -> None:
    """Save OOS prediction CSVs for every model in the report."""
    for result in report.model_results:
        if result.pooled_oos_y is None or result.pooled_oos_prob_up is None:
            continue
        df_pred = pd.DataFrame({
            "y_true": result.pooled_oos_y,
            "prob_up": result.pooled_oos_prob_up,
        })
        fname = output_dir / f"predictions_{symbol}_h{horizon}_{result.model_name}.csv"
        df_pred.to_csv(fname, index=False)
        logger.debug("Saved predictions: %s (%d rows)", fname.name, len(df_pred))
