"""
Markdown report generator.

Produces a complete human-readable report from a TrainingReport.

Report structure:
  1. Executive summary — config, winner, key metrics table
  2. Model comparison table — all models vs baselines
  3. Walk-forward fold details — per-fold table for the winner
  4. Confidence bucket analysis — accuracy and trading utility per bucket
  5. Feature importance — top 15 by permutation importance
  6. Feature group ablation — Brier delta per group
  7. Regime-segmented evaluation — per-regime accuracy and Brier score
  8. Calibration summary — ECE per model
  9. Model selection rationale — why the winner was chosen
"""

from datetime import datetime
from typing import Dict, Optional

from app.ml_models.training.trainer import TrainingReport, ModelResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(v, decimals: int = 4) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _table(headers: list, rows: list) -> str:
    """Render a markdown table."""
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    header_row = "| " + " | ".join(headers) + " |"
    body = "\n".join(
        "| " + " | ".join(str(c) for c in row) + " |"
        for row in rows
    )
    return "\n".join([header_row, sep, body])


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _section_summary(report: TrainingReport) -> str:
    cfg = report.config
    w = report.winner
    lines = [
        "## Executive Summary",
        "",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Config hash:** `{cfg.config_hash()}`  ",
        f"**Feature manifest:** `{cfg.manifest_hash}`  (pipeline v{cfg.pipeline_version})  ",
        f"**Features:** {len(cfg.feature_cols)}  ",
        f"**Walk-forward folds:** {cfg.n_splits}  embargo={cfg.embargo_bars} bar(s)  ",
        f"**Selection criterion:** `{cfg.selection_metric}` ({cfg.selection_metric_direction})  ",
        "",
        f"**Winner:** `{w.model_name}`",
        "",
        f"| Metric | Value |",
        f"| --- | --- |",
        f"| Brier score (mean ± std) | {_fmt(w.aggregated.brier_score_mean)} ± {_fmt(w.aggregated.brier_score_std)} |",
        f"| Log-loss (mean ± std) | {_fmt(w.aggregated.log_loss_mean)} ± {_fmt(w.aggregated.log_loss_std)} |",
        f"| ROC-AUC (mean ± std) | {_fmt(w.aggregated.roc_auc_mean)} ± {_fmt(w.aggregated.roc_auc_std)} |",
        f"| Balanced accuracy (mean ± std) | {_fmt(w.aggregated.balanced_accuracy_mean)} ± {_fmt(w.aggregated.balanced_accuracy_std)} |",
        f"| ECE (mean) | {_fmt(w.aggregated.ece_mean)} |",
        f"| Total test samples | {w.aggregated.total_test_samples:,} |",
        "",
    ]
    return "\n".join(lines)


def _section_model_comparison(report: TrainingReport) -> str:
    lines = ["## Model Comparison", ""]
    headers = [
        "Model", "Type", "Brier ↓ (mean ± std)", "LogLoss ↓",
        "AUC ↑", "BalAcc ↑", "ECE ↓", "Folds", "Test N"
    ]
    rows = []
    # Baselines first, then parametric, sorted by Brier within each group
    baselines = sorted(
        [r for r in report.model_results if r.is_baseline],
        key=lambda r: r.aggregated.brier_score_mean or 999,
    )
    parametric = sorted(
        [r for r in report.model_results if not r.is_baseline],
        key=lambda r: r.aggregated.brier_score_mean or 999,
    )
    winner_name = report.winner.model_name

    for r in baselines + parametric:
        tag = " ← **winner**" if r.model_name == winner_name else ""
        kind = "baseline" if r.is_baseline else "parametric"
        ag = r.aggregated
        brier_str = f"{_fmt(ag.brier_score_mean)} ± {_fmt(ag.brier_score_std)}"
        rows.append([
            f"`{r.model_name}`{tag}", kind, brier_str,
            _fmt(ag.log_loss_mean), _fmt(ag.roc_auc_mean),
            _fmt(ag.balanced_accuracy_mean), _fmt(ag.ece_mean),
            ag.n_folds, f"{ag.total_test_samples:,}",
        ])

    lines.append(_table(headers, rows))
    lines.append("")
    lines.append(
        "> **Note:** Model selection criterion is Brier score (lower is better), "
        "not headline accuracy. A model must beat the naive baselines on Brier score "
        "to be considered useful. Brier score = 0.25 for a coin-flip classifier."
    )
    lines.append("")
    return "\n".join(lines)


def _section_fold_details(report: TrainingReport) -> str:
    w = report.winner
    lines = ["## Walk-Forward Fold Details (Winner)", ""]
    lines.append(f"Model: `{w.model_name}`")
    lines.append("")

    headers = [
        "Fold", "Train N", "Test N", "Brier ↓", "LogLoss ↓", "AUC ↑", "BalAcc ↑", "ECE ↓"
    ]
    rows = []
    for fm in w.fold_metrics:
        rows.append([
            fm.fold, f"{fm.train_size:,}", f"{fm.test_size:,}",
            _fmt(fm.brier_score), _fmt(fm.log_loss),
            _fmt(fm.roc_auc), _fmt(fm.balanced_accuracy), _fmt(fm.ece),
        ])
    lines.append(_table(headers, rows))
    lines.append("")

    # Splitter description
    lines.append("**Splitter configuration:**")
    lines.append("")
    sd_headers = ["Fold", "Train bars", "Embargo bars", "Test bars", "Test start", "Test end"]
    sd_rows = [[
        s["fold"], s["train_bars"], s["embargo_bars"], s["test_bars"],
        s["test_start"], s["test_end"],
    ] for s in report.splitter_description]
    lines.append(_table(sd_headers, sd_rows))
    lines.append("")
    return "\n".join(lines)


def _section_confidence_buckets(report: TrainingReport) -> str:
    w = report.winner
    lines = ["## Confidence Bucket Analysis (Winner)", ""]
    lines.append(
        "Each sample is assigned to a confidence bucket based on |P(up) − 0.5| × 2. "
        "High accuracy only in low-confidence buckets → no tradeable edge."
    )
    lines.append("")

    ca = w.confidence_analysis
    if ca is None:
        lines.append("_Not available._")
        lines.append("")
        return "\n".join(lines)

    has_returns = any(b.expected_return is not None for b in ca.buckets)
    if has_returns:
        headers = [
            "Confidence", "N samples", "Signal rate", "Accuracy ↑",
            "Exp. return", "Ret. std", "Sharpe proxy ↑"
        ]
    else:
        headers = ["Confidence", "N samples", "Signal rate", "Accuracy ↑", "Mean confidence"]

    rows = []
    for b in ca.buckets:
        conf_range = f"[{b.confidence_min:.2f}, {b.confidence_max:.2f})"
        if b.n_samples == 0:
            if has_returns:
                rows.append([conf_range, 0, "—", "—", "—", "—", "—"])
            else:
                rows.append([conf_range, 0, "—", "—", "—"])
            continue

        if has_returns:
            rows.append([
                conf_range, f"{b.n_samples:,}", _fmt(b.signal_rate, 3),
                _fmt(b.accuracy, 4),
                _fmt(b.expected_return, 6) if b.expected_return is not None else "—",
                _fmt(b.return_std, 6) if b.return_std is not None else "—",
                _fmt(b.sharpe_proxy, 3) if b.sharpe_proxy is not None else "—",
            ])
        else:
            rows.append([
                conf_range, f"{b.n_samples:,}", _fmt(b.signal_rate, 3),
                _fmt(b.accuracy, 4), _fmt(b.mean_confidence, 4),
            ])

    lines.append(_table(headers, rows))
    lines.append("")
    lines.append(f"**Skill monotone:** {'Yes ✓' if ca.skill_monotone else 'No ✗'}  ")
    if ca.high_confidence_accuracy is not None:
        lines.append(f"**Top-bucket accuracy:** {_fmt(ca.high_confidence_accuracy, 4)}  ")
    lines.append("")

    # Also show all other models' high-confidence accuracy for comparison
    lines.append("**High-confidence accuracy by model (top bucket only):**")
    lines.append("")
    mc_headers = ["Model", "Type", "Top-bucket accuracy", "Top-bucket N"]
    mc_rows = []
    for r in report.model_results:
        ca_r = r.confidence_analysis
        if ca_r is None:
            continue
        top = ca_r.buckets[-1] if ca_r.buckets else None
        mc_rows.append([
            f"`{r.model_name}`",
            "baseline" if r.is_baseline else "parametric",
            _fmt(top.accuracy, 4) if top and top.n_samples > 0 else "—",
            f"{top.n_samples:,}" if top else "—",
        ])
    lines.append(_table(mc_headers, mc_rows))
    lines.append("")
    return "\n".join(lines)


def _section_importance(report: TrainingReport) -> str:
    w = report.winner
    lines = ["## Feature Importance (Winner)", ""]

    if w.importance is None:
        lines.append("_Not available for baseline models._")
        lines.append("")
        return "\n".join(lines)

    top15 = w.importance[:15]
    headers = [
        "Rank (perm.)", "Feature", "Perm. importance ↑",
        "Perm. std", "Intrinsic importance", "Rank (intrinsic)"
    ]
    rows = []
    for feat in top15:
        rows.append([
            feat.rank_permutation or "—",
            f"`{feat.feature}`",
            _fmt(feat.permutation_importance, 6),
            _fmt(feat.permutation_std, 6),
            _fmt(feat.intrinsic_importance, 6),
            feat.rank_intrinsic or "—",
        ])
    lines.append(_table(headers, rows))
    lines.append("")
    lines.append(
        "> **Permutation importance** = mean Brier score increase when the feature is shuffled. "
        "Positive = feature is used by the model; negative/zero = feature adds noise or is redundant. "
        "**Intrinsic importance** is model-specific (logistic: scaled coefficient magnitude; "
        "tree models: mean impurity decrease)."
    )
    lines.append("")
    return "\n".join(lines)


def _section_ablation(report: TrainingReport) -> str:
    w = report.winner
    lines = ["## Feature Group Ablation (Winner, Fold 0)", ""]

    if not w.ablation:
        lines.append("_Ablation not run or not available._")
        lines.append("")
        return "\n".join(lines)

    lines.append(
        "Each row shows the Brier score impact of zeroing out all features in a group. "
        "**Positive Δ = group hurt the model if removed (group is valuable)**. "
        "Negative Δ = group is not helping (or hurting). "
        "Groups are sorted by Brier Δ descending."
    )
    lines.append("")
    headers = [
        "Group", "Features dropped", "Baseline Brier", "Ablated Brier",
        "Brier Δ ↑", "% Change"
    ]
    rows = []
    for abl in w.ablation:
        rows.append([
            abl.group,
            ", ".join(f"`{f}`" for f in abl.features_dropped[:5])
            + ("…" if len(abl.features_dropped) > 5 else ""),
            _fmt(abl.brier_score_baseline),
            _fmt(abl.brier_score_ablated),
            _fmt(abl.brier_delta, 6),
            f"{abl.pct_change:+.2f}%",
        ])
    lines.append(_table(headers, rows))
    lines.append("")
    return "\n".join(lines)


def _section_regime(report: TrainingReport) -> str:
    w = report.winner
    lines = ["## Regime-Segmented Evaluation (Winner)", ""]

    if not w.regime_metrics:
        lines.append("_Regime evaluation not available (no regime data passed)._")
        lines.append("")
        return "\n".join(lines)

    headers = [
        "Regime", "N samples", "Fraction", "Brier ↓",
        "LogLoss ↓", "AUC ↑", "BalAcc ↑", "High-conf acc ↑"
    ]
    rows = []
    for rm in w.regime_metrics:
        rows.append([
            f"`{rm.regime}`",
            f"{rm.n_samples:,}",
            _fmt(rm.sample_fraction, 3),
            _fmt(rm.brier_score),
            _fmt(rm.log_loss),
            _fmt(rm.roc_auc),
            _fmt(rm.balanced_accuracy),
            _fmt(rm.directional_accuracy_confident) if rm.directional_accuracy_confident is not None else "—",
        ])
    lines.append(_table(headers, rows))
    lines.append("")
    return "\n".join(lines)


def _section_calibration(report: TrainingReport) -> str:
    lines = ["## Calibration Summary", ""]
    lines.append(
        "ECE (Expected Calibration Error) measures how far predicted probabilities "
        "are from observed frequencies. ECE = 0 is perfect calibration; "
        "a coin-flip classifier has ECE ≈ 0."
    )
    lines.append("")

    headers = ["Model", "Type", "ECE ↓ (mean ± std)", "Brier ↓ (mean)"]
    rows = []
    all_sorted = sorted(report.model_results, key=lambda r: r.aggregated.ece_mean or 999)
    for r in all_sorted:
        ag = r.aggregated
        rows.append([
            f"`{r.model_name}`",
            "baseline" if r.is_baseline else "parametric",
            f"{_fmt(ag.ece_mean)} ± {_fmt(ag.ece_std)}",
            _fmt(ag.brier_score_mean),
        ])
    lines.append(_table(headers, rows))
    lines.append("")
    return "\n".join(lines)


def _section_selection_rationale(report: TrainingReport) -> str:
    w = report.winner
    cfg = report.config
    lines = ["## Model Selection Rationale", ""]

    lines.append(
        f"The winner **`{w.model_name}`** was selected using "
        f"`{cfg.selection_metric}` (direction: {cfg.selection_metric_direction})."
    )
    lines.append("")

    # Rank all parametric models
    parametric = sorted(
        [r for r in report.model_results if not r.is_baseline],
        key=lambda r: r.aggregated.brier_score_mean or 999,
    )
    lines.append("**Parametric model ranking by Brier score:**")
    lines.append("")
    rank_headers = ["Rank", "Model", "Brier (mean)", "Brier (std)", "AUC (mean)"]
    rank_rows = [
        [i + 1, f"`{r.model_name}`",
         _fmt(r.aggregated.brier_score_mean),
         _fmt(r.aggregated.brier_score_std),
         _fmt(r.aggregated.roc_auc_mean)]
        for i, r in enumerate(parametric)
    ]
    lines.append(_table(rank_headers, rank_rows))
    lines.append("")

    # Baseline comparison
    best_baseline_brier = min(
        (r.aggregated.brier_score_mean for r in report.model_results if r.is_baseline),
        default=None,
    )
    if best_baseline_brier is not None:
        winner_brier = w.aggregated.brier_score_mean
        if winner_brier < best_baseline_brier:
            margin = best_baseline_brier - winner_brier
            lines.append(
                f"✓ Winner beats best baseline by **{margin:.4f} Brier points** "
                f"(best baseline: {best_baseline_brier:.4f})."
            )
        else:
            lines.append(
                f"⚠ Winner does **not** beat best baseline on Brier score "
                f"({winner_brier:.4f} vs baseline {best_baseline_brier:.4f}). "
                f"Model may not be adding value — consider collecting more data or "
                f"re-evaluating the feature set."
            )
    lines.append("")
    lines.append(
        "> Walk-forward evaluation uses purged splits with "
        f"{cfg.embargo_bars} bar embargo. "
        "All results are out-of-sample. "
        "The final model is re-trained on all available data after evaluation."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level report generation
# ---------------------------------------------------------------------------

def generate_report(report: TrainingReport) -> str:
    """
    Generate a complete markdown training report.

    Returns the report as a string. Write it to a file with:
        Path("reports/training_report.md").write_text(generate_report(report))
    """
    sections = [
        f"# Model Training Report",
        "",
        _section_summary(report),
        _section_model_comparison(report),
        _section_fold_details(report),
        _section_confidence_buckets(report),
        _section_importance(report),
        _section_ablation(report),
        _section_regime(report),
        _section_calibration(report),
        _section_selection_rationale(report),
    ]
    return "\n".join(sections)


def generate_multi_horizon_report(reports: Dict[int, "TrainingReport"]) -> str:
    """
    Generate a consolidated multi-horizon markdown report.

    Produces a single document comparing winners and model performance across
    all trained horizons.  Per-horizon detailed reports are in sub-directories.

    Parameters
    ----------
    reports : dict[int, TrainingReport]
        Mapping of horizon → TrainingReport, as returned by
        ``run_multi_horizon_pipeline()``.
    """
    horizons = sorted(reports.keys())
    lines = [
        "# Multi-Horizon Training Report",
        "",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Horizons evaluated:** {', '.join(f'h={h}' for h in horizons)}  ",
        f"**Label type:** ternary (DOWN=0 / FLAT=1 / UP=2), evaluated as UP-vs-rest  ",
        "",
        "---",
        "",
        "## Winner Summary",
        "",
    ]

    # Winner table
    w_headers = [
        "Horizon", "Winner", "Type", "Brier ↓ (mean ± std)",
        "AUC ↑", "BalAcc ↑", "ECE ↓", "Embargo bars", "Test N"
    ]
    w_rows = []
    for h in horizons:
        r = reports[h]
        w = r.winner
        ag = w.aggregated
        kind = "baseline" if w.is_baseline else "parametric"
        w_rows.append([
            f"h={h}",
            f"`{w.model_name}`",
            kind,
            f"{_fmt(ag.brier_score_mean)} ± {_fmt(ag.brier_score_std)}",
            _fmt(ag.roc_auc_mean),
            _fmt(ag.balanced_accuracy_mean),
            _fmt(ag.ece_mean),
            r.config.embargo_bars,
            f"{ag.total_test_samples:,}",
        ])
    lines.append(_table(w_headers, w_rows))
    lines.append("")

    # Per-horizon model comparison
    lines.append("---")
    lines.append("")
    lines.append("## Per-Horizon Model Comparison")
    lines.append("")
    lines.append(
        "> Each sub-table shows all models for that horizon, sorted by Brier score. "
        "Baselines appear first. Brier = 0.25 = coin-flip. "
        "See `h{N}/training_report.md` for full per-horizon details."
    )
    lines.append("")

    for h in horizons:
        r = reports[h]
        lines.append(f"### Horizon h={h}  (embargo={r.config.embargo_bars} bar(s))")
        lines.append("")
        lines.append(_section_model_comparison(r).split("\n", 3)[-1])  # strip heading

    # Best-model-beats-baseline summary per horizon
    lines.append("---")
    lines.append("")
    lines.append("## Baseline Beat Summary")
    lines.append("")
    bb_headers = ["Horizon", "Winner Brier", "Best Baseline Brier", "Margin", "Beats baseline?"]
    bb_rows = []
    for h in horizons:
        r = reports[h]
        w = r.winner
        baselines = [res for res in r.model_results if res.is_baseline]
        best_bl = (
            min(b.aggregated.brier_score_mean for b in baselines)
            if baselines else None
        )
        if best_bl is not None:
            margin = best_bl - w.aggregated.brier_score_mean
            beats = "Yes ✓" if margin > 0 else "No ✗"
            bb_rows.append([
                f"h={h}",
                _fmt(w.aggregated.brier_score_mean),
                _fmt(best_bl),
                _fmt(margin),
                beats,
            ])
    lines.append(_table(bb_headers, bb_rows))
    lines.append("")
    lines.append(
        "> A positive margin means the winner beat the best naive baseline. "
        "A model that cannot beat its baselines on Brier score has no predictive value."
    )
    lines.append("")

    return "\n".join(lines)
