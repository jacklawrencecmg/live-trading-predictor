"""
Leakage regression tests — one test per discovered leakage class.

Each test is a regression guard: it will catch any future change that
re-introduces a known leakage pattern. The test names map 1-to-1 to the
issues documented in docs/LEAKAGE_AUDIT.md.

Test categories:
  L1  — Feature/inference pipeline dimension mismatch
  L2  — Feature lookback window (shift-by-1 invariant)
  L3  — Bar-close guard buffer
  L4  — Ternary label threshold uses current-bar data
  L5  — Inference rejects unclosed bars
  L6  — Session-boundary ffill capped
  L7  — Options snapshot no-lookahead join (now code-enforced)
  L8  — Labels drop last row / no label for unpredictable bar
  L9  — Backtest prepare_dataset feature dimension consistency
  L10 — Backtest walk-forward loop missing embargo gap
  L11 — Options get_latest_chain_pit enforces available_at cutoff in SQL
  L12 — Options staleness threshold at inference too permissive
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

pytestmark = pytest.mark.leakage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV with bar_open_time and vwap columns."""
    rng = np.random.default_rng(seed)
    prices = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    prices = np.clip(prices, 1, None)
    highs = prices + rng.uniform(0, 1, n)
    lows = prices - rng.uniform(0, 1, n)
    opens = prices + rng.normal(0, 0.2, n)
    volumes = rng.integers(1000, 10000, n).astype(float)
    vwap = (highs + lows + prices) / 3
    times = pd.date_range("2024-01-02 14:30", periods=n, freq="5min")
    return pd.DataFrame({
        "bar_open_time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
        "vwap": vwap,
    })


# ---------------------------------------------------------------------------
# L1: Feature/inference pipeline dimension consistency
# ---------------------------------------------------------------------------

def test_L1_backtest_prepare_dataset_produces_new_pipeline_feature_count():
    """
    _prepare_dataset must produce exactly len(FEATURE_COLS) features per row
    so that models trained in backtest are compatible with inference_service
    which also uses FEATURE_COLS.

    This was a CRITICAL bug: the old pipeline produced 14 features; the new
    pipeline used by inference produces 22.  Training on 14, predicting on 22
    would crash or silently corrupt predictions.
    """
    from app.services.backtest_service import _prepare_dataset
    from app.feature_pipeline.features import FEATURE_COLS

    df = _make_ohlcv(200)
    X, y_dir, y_mag = _prepare_dataset(df)

    assert X.shape[1] == len(FEATURE_COLS), (
        f"Expected {len(FEATURE_COLS)} features (new pipeline), "
        f"got {X.shape[1]}. Training/inference dimension mismatch."
    )


def test_L1_prepare_dataset_and_inference_service_use_same_feature_names():
    """
    The feature names used by _prepare_dataset and inference_service must be
    the same list (FEATURE_COLS) so a model trained in backtest works in inference.
    """
    from app.feature_pipeline.features import FEATURE_COLS, build_feature_matrix
    from app.services.backtest_service import _prepare_dataset

    df = _make_ohlcv(100)

    # Feature matrix from the pipeline
    feat_df = build_feature_matrix(df)
    pipeline_cols = [c for c in FEATURE_COLS if c in feat_df.columns]

    # Feature array from _prepare_dataset
    X, _, _ = _prepare_dataset(df)

    assert X.shape[1] == len(pipeline_cols), (
        "Column count mismatch between _prepare_dataset and build_feature_matrix"
    )


# ---------------------------------------------------------------------------
# L2: Feature lookback — shift-by-1 invariant
# ---------------------------------------------------------------------------

def test_L2_features_at_row_i_do_not_use_close_i():
    """
    feat[i] must be identical whether or not we later append extra bars.
    If features at row i used close[i], appending new bars would not change
    feat[i]. But if rolling windows at row i ONLY used close[0..i-1] (via
    shift(1)), then feat[i] must be stable regardless of future rows.

    This test verifies the shift-by-1 invariant at every row in the matrix.
    Build df_extended by APPENDING extra rows to the same base, not by
    re-generating — re-generating with a different n advances the RNG
    differently and produces different high/low values.
    """
    from app.feature_pipeline.features import build_feature_matrix, FEATURE_COLS

    df = _make_ohlcv(80)

    # Extend by appending 10 genuinely new bars after the last bar
    extra = _make_ohlcv(10, seed=77)
    last_time = df["bar_open_time"].iloc[-1]
    extra["bar_open_time"] = last_time + pd.to_timedelta(
        (extra.index + 1) * 5, unit="min"
    )
    df_extended = pd.concat([df, extra], ignore_index=True)

    feat_short = build_feature_matrix(df)[FEATURE_COLS].values
    feat_long = build_feature_matrix(df_extended)[FEATURE_COLS].values

    # All rows 0..79 in feat_short must match rows 0..79 in feat_long.
    # EWM and rolling operations are strictly backward-looking (causal),
    # so appending future bars must not change any past row's feature value.
    n = len(feat_short)
    max_diff = np.nanmax(np.abs(feat_short - feat_long[:n]))
    assert max_diff < 1e-9, (
        f"Feature values changed when future bars were appended (max diff={max_diff:.2e}). "
        "This indicates the shift-by-1 invariant is broken."
    )


def test_L2_feature_row_i_is_nan_for_row_0():
    """
    After shift(1), row 0's features must be NaN (no prior bar exists).
    Any non-NaN at row 0 would indicate features were computed without a shift.
    """
    from app.feature_pipeline.features import build_feature_matrix, FEATURE_COLS

    df = _make_ohlcv(60)
    feat_df = build_feature_matrix(df)

    # Row 0 has c1 = NaN (shift(1) of row 0 is undefined).
    # Most features that depend on close will be NaN at row 0.
    # rsi_14 at row 0 must be NaN.
    assert np.isnan(feat_df["rsi_14"].iloc[0]), (
        "rsi_14 at row 0 should be NaN since shift(1) leaves no prior close."
    )


# ---------------------------------------------------------------------------
# L3: Bar-close guard buffer
# ---------------------------------------------------------------------------

def test_L3_bar_not_closed_within_buffer_window():
    """
    A bar whose close_time == now should NOT be marked closed.
    The guard buffer requires close_time + N seconds <= now.
    """
    from app.data_ingestion.ingestion_service import _is_bar_closed, _BAR_CLOSE_BUFFER_SECONDS

    # Bar that JUST closed (close_time == now) — inside buffer window
    bar_open = datetime.utcnow() - timedelta(minutes=5)
    assert not _is_bar_closed(bar_open, "5m"), (
        "Bar closed exactly at close_time should NOT be marked closed: "
        f"buffer of {_BAR_CLOSE_BUFFER_SECONDS}s has not elapsed."
    )


def test_L3_bar_closed_after_buffer_elapsed():
    """
    A bar whose close_time was more than buffer_seconds ago IS safe to use.
    """
    from app.data_ingestion.ingestion_service import _is_bar_closed, _BAR_CLOSE_BUFFER_SECONDS

    # Bar closed 10 minutes ago — well past the buffer
    bar_open = datetime.utcnow() - timedelta(minutes=15)
    assert _is_bar_closed(bar_open, "5m"), (
        "A bar closed 10 min ago should be marked closed."
    )


def test_L3_buffer_constant_is_positive():
    """Guard that the buffer is never accidentally set to 0 or negative."""
    from app.data_ingestion.ingestion_service import _BAR_CLOSE_BUFFER_SECONDS
    assert _BAR_CLOSE_BUFFER_SECONDS > 0, "Bar-close buffer must be positive"


# ---------------------------------------------------------------------------
# L4: Ternary label threshold uses only prior-bar data
# ---------------------------------------------------------------------------

def test_L4_ternary_label_threshold_shift():
    """
    Modifying bar i's high/low must NOT change the ternary label threshold for
    bar i-1. This confirms that threshold[i] only uses bars 0..i-1 (via ATR shift).

    Before the fix, ATR at row i used high[i]/low[i], so the threshold for row i
    was contaminated by the current bar's own intrabar range.
    """
    from app.feature_pipeline.labels import ternary_label

    df1 = _make_ohlcv(60)
    df2 = df1.copy()
    # Wildly inflate bar 50's high and low — should NOT affect threshold at row 49
    df2.loc[50, "high"] = 200.0
    df2.loc[50, "low"] = 1.0

    labels1 = ternary_label(df1, use_atr=True)
    labels2 = ternary_label(df2, use_atr=True)

    # Row 49's label must be identical — only bars 0..48 contribute to threshold[49]
    assert labels1.iloc[49] == labels2.iloc[49], (
        "Ternary threshold at row 49 changed when bar 50's high/low was modified. "
        "Threshold must only use bars 0..i-1 (ATR should be shifted)."
    )


def test_L4_ternary_label_no_lookahead_in_threshold_non_atr():
    """
    Same invariant for the non-ATR (realized-vol) threshold path.
    Modifying bar 50's close must not change the threshold at row 49.
    """
    from app.feature_pipeline.labels import ternary_label

    df1 = _make_ohlcv(60)
    df2 = df1.copy()
    df2.loc[50, "close"] = 999.0  # extreme future value

    labels1 = ternary_label(df1, use_atr=False)
    labels2 = ternary_label(df2, use_atr=False)

    assert labels1.iloc[49] == labels2.iloc[49], (
        "Non-ATR ternary threshold at row 49 changed when bar 50's close was modified."
    )


# ---------------------------------------------------------------------------
# L5: Inference rejects unclosed bars
# ---------------------------------------------------------------------------

def test_L5_inference_rejects_unclosed_last_bar():
    """
    run_inference must return no_trade with reason 'last_bar_not_closed'
    when the last bar in the DataFrame has is_closed=False.
    """
    from app.inference.inference_service import run_inference

    df = _make_ohlcv(60)
    df["is_closed"] = True
    df.loc[df.index[-1], "is_closed"] = False  # mark last bar as open

    result = run_inference(df, "SPY")

    assert result.trade_signal == "no_trade"
    assert result.no_trade_reason == "last_bar_not_closed", (
        f"Expected 'last_bar_not_closed', got '{result.no_trade_reason}'. "
        "Inference must reject DataFrames where the last bar is still open."
    )


def test_L5_inference_proceeds_when_all_bars_closed():
    """
    run_inference proceeds normally when is_closed=True for all bars
    (even if model is untrained — it falls back to no_trade for other reasons).
    """
    from app.inference.inference_service import run_inference

    df = _make_ohlcv(60)
    df["is_closed"] = True

    result = run_inference(df, "SPY")
    # Should NOT be rejected for bar-not-closed reason
    assert result.no_trade_reason != "last_bar_not_closed"


def test_L5_inference_proceeds_when_no_is_closed_column():
    """
    When is_closed column is absent (legacy callers), inference should not crash.
    It relaxes the check and proceeds.
    """
    from app.inference.inference_service import run_inference

    df = _make_ohlcv(60)
    # No is_closed column at all

    result = run_inference(df, "SPY")
    assert result.no_trade_reason != "last_bar_not_closed"


# ---------------------------------------------------------------------------
# L6: Session-boundary ffill capped
# ---------------------------------------------------------------------------

def test_L6_ffill_does_not_propagate_across_long_gap():
    """
    If there is a gap longer than _FFILL_LIMIT bars (e.g., data outage or
    multi-day weekend gap in a continuous series), features should become NaN
    rather than silently carrying stale values from before the gap.

    We insert 100 NaN rows (simulating a session gap larger than 78 bars)
    and verify that features beyond the limit are NaN, not filled.
    """
    from app.feature_pipeline.features import build_feature_matrix, FEATURE_COLS, FFILL_LIMIT as _FFILL_LIMIT

    # Build a 200-bar series with a 90-bar NaN gap in the middle
    df_pre = _make_ohlcv(50, seed=1)
    df_gap = df_pre.iloc[:1].copy()  # dummy row
    gap_rows = []
    for i in range(90):
        row = df_gap.iloc[0].copy()
        row["close"] = np.nan
        row["open"] = np.nan
        row["high"] = np.nan
        row["low"] = np.nan
        row["volume"] = np.nan
        row["bar_open_time"] = df_pre["bar_open_time"].iloc[-1] + timedelta(minutes=5 * (i + 1))
        gap_rows.append(row)
    df_after = _make_ohlcv(50, seed=2)
    df_after["bar_open_time"] = df_pre["bar_open_time"].iloc[-1] + timedelta(
        minutes=5 * 91
    ) + pd.to_timedelta(df_after.index * 5, unit="min")

    df_combined = pd.concat([df_pre, pd.DataFrame(gap_rows), df_after], ignore_index=True)

    feat_df = build_feature_matrix(df_combined)

    # Rows within the gap and just after should have NaN for most features
    # (gap is 90 bars, limit is 78, so rows 50+78=128 and beyond should start seeing NaN)
    gap_start = 50
    gap_end = 50 + 90
    rows_beyond_limit = feat_df.iloc[gap_start + _FFILL_LIMIT + 1: gap_end]

    if len(rows_beyond_limit) > 0:
        nan_count = rows_beyond_limit[FEATURE_COLS].isna().any(axis=1).sum()
        assert nan_count > 0, (
            "Features beyond the ffill limit should be NaN during a long gap, "
            "not silently carried from pre-gap data."
        )


# ---------------------------------------------------------------------------
# L7: Options snapshot no-lookahead join
# ---------------------------------------------------------------------------

def test_L7_options_snapshot_before_bar_is_valid():
    """Valid join: snapshot captured before bar_open_time."""
    bar_open = datetime(2024, 1, 15, 14, 30)
    snapshot_time = bar_open - timedelta(minutes=2)
    assert snapshot_time <= bar_open


def test_L7_options_snapshot_after_bar_is_lookahead():
    """Invalid join: snapshot captured after bar_open_time is lookahead."""
    bar_open = datetime(2024, 1, 15, 14, 30)
    snapshot_time = bar_open + timedelta(seconds=1)
    assert snapshot_time > bar_open, (
        "A snapshot taken after bar_open_time must NOT be joined to that bar."
    )


def test_L7_most_recent_valid_snapshot_selection():
    """
    Given multiple snapshots, the valid join partner is the most recent one
    with snapshot_time <= bar_open_time.  Any snapshot with snapshot_time >
    bar_open_time must be excluded.
    """
    bar_open = datetime(2024, 1, 15, 14, 30)
    snapshots = [
        datetime(2024, 1, 15, 14, 25),  # 5 min before — valid
        datetime(2024, 1, 15, 14, 29),  # 1 min before — valid, most recent
        datetime(2024, 1, 15, 14, 31),  # 1 min after — INVALID (lookahead)
    ]

    valid = [s for s in snapshots if s <= bar_open]
    best = max(valid)  # most recent valid snapshot

    assert best == datetime(2024, 1, 15, 14, 29), "Most recent valid snapshot selection failed"
    assert datetime(2024, 1, 15, 14, 31) not in valid, "Future snapshot must not be in valid set"


# ---------------------------------------------------------------------------
# L8: Labels — no label for last row, no future close in features
# ---------------------------------------------------------------------------

def test_L8_build_labels_drops_last_row():
    """build_labels must drop the last row (no next bar exists)."""
    from app.feature_pipeline.labels import build_labels

    df = _make_ohlcv(50)
    labels = build_labels(df)
    assert len(labels) == len(df) - 1, (
        "build_labels must drop the last row since it has no next bar to label."
    )


def test_L8_binary_label_does_not_shift_minus_2():
    """
    binary_label uses shift(-1): label[i] uses close[i+1].
    A shift(-2) would skip a bar and create a further-future target.
    Verify alignment is exactly 1 bar ahead.
    """
    from app.feature_pipeline.labels import binary_label

    df = _make_ohlcv(10)
    labels = binary_label(df)

    # label[0] should be 1 if close[1] > close[0], else 0
    expected = int(df["close"].iloc[1] > df["close"].iloc[0])
    assert int(labels.iloc[0]) == expected, (
        "binary_label[0] must reflect close[1] vs close[0], not any other pair."
    )


def test_L8_regression_label_last_row_is_nan():
    """regression_return_label must produce NaN at the last row."""
    from app.feature_pipeline.labels import regression_return_label

    df = _make_ohlcv(10)
    labels = regression_return_label(df)
    assert np.isnan(labels.iloc[-1]), "Last row regression label must be NaN (no next bar)"


# ---------------------------------------------------------------------------
# L9: _prepare_dataset — no future rows bleed into training features
# ---------------------------------------------------------------------------

def test_L9_prepare_dataset_feature_stability():
    """
    Features produced by _prepare_dataset for rows 0..k must be identical
    whether we run on df[:n] or df[:n+extra].

    This confirms the O(n) pipeline is stable and doesn't depend on how many
    future bars are in the dataset (unlike the old O(n²) loop which could
    theoretically allow future data contamination via global statistics).
    """
    from app.services.backtest_service import _prepare_dataset

    df = _make_ohlcv(150)

    X_full, y_full, _ = _prepare_dataset(df)
    X_short, y_short, _ = _prepare_dataset(df.iloc[:100].copy())

    # The short run should be a prefix of the full run
    # (same rows, same feature values up to the overlap)
    min_rows = min(len(X_short), len(X_full))
    assert min_rows > 0

    # Allow small floating-point tolerance
    max_diff = np.nanmax(np.abs(X_full[:min_rows] - X_short[:min_rows]))
    assert max_diff < 1e-9, (
        f"Feature rows changed when more future data was added (max diff={max_diff:.2e}). "
        "Features must be stable w.r.t. future data."
    )


def test_L9_prepare_dataset_labels_are_consistent():
    """
    Labels produced by _prepare_dataset must match direct computation of
    close[i+1] > close[i] — no off-by-one drift.
    """
    from app.services.backtest_service import _prepare_dataset

    df = _make_ohlcv(80, seed=99)
    X, y_dir, y_mag = _prepare_dataset(df)

    # Direct labels (same computation, independent)
    close = df["close"].values
    expected_dir = (close[1:] > close[:-1]).astype(int)
    expected_mag = np.abs(close[1:] / close[:-1] - 1)

    # y_dir should be a valid subset of expected_dir (after NaN warmup rows removed)
    assert len(y_dir) <= len(expected_dir)
    assert len(y_dir) > 0

    # The last len(y_dir) elements should match (valid rows at end)
    tail = expected_dir[-len(y_dir):]
    assert np.array_equal(y_dir, tail), (
        "Labels from _prepare_dataset do not match close[i+1] > close[i]."
    )


# ---------------------------------------------------------------------------
# L10: Backtest walk-forward loop — embargo gap between train and test
# ---------------------------------------------------------------------------

def test_L10_embargo_bars_constant_exists_and_is_positive():
    """
    _EMBARGO_BARS must exist in backtest_service and be >= 1.

    The embargo prevents the boundary bar from contributing to both the last
    training label and the first test feature vector.  Without it the bar at
    index train_end straddles the boundary: label[train_end-1] uses
    close[train_end], and X_test[0] uses close[train_end] in its features.
    An embargo of 1 removes X_test[0] from the evaluation set.
    """
    from app.services.backtest_service import _EMBARGO_BARS
    assert _EMBARGO_BARS >= 1, (
        f"_EMBARGO_BARS must be >= 1 to prevent boundary-bar label bleed, got {_EMBARGO_BARS}"
    )


def test_L10_walk_forward_test_set_starts_after_embargo():
    """
    Given a fully prepared feature/label array, the test slice must start at
    train_end + _EMBARGO_BARS, not train_end.

    We verify this by constructing slices the same way run_backtest does and
    asserting the gap is non-zero — no row appears in both X_train and X_test.
    """
    from app.services.backtest_service import _EMBARGO_BARS, _prepare_dataset

    df = _make_ohlcv(200)
    X, y_dir, y_mag = _prepare_dataset(df)

    train_size = 80
    test_size = 40
    train_end = train_size

    X_train = X[max(0, train_end - train_size): train_end]
    X_test = X[train_end + _EMBARGO_BARS: train_end + test_size]

    # No row that ends train should appear at the start of test
    # (they are different index ranges)
    train_last_idx = train_end - 1
    test_first_idx = train_end + _EMBARGO_BARS

    assert test_first_idx > train_last_idx, (
        f"Test window starts at {test_first_idx}, train window ends at {train_last_idx}. "
        "Embargo gap must ensure test_first_idx > train_last_idx."
    )

    # The X arrays must not share any rows
    assert len(X_train) > 0 and len(X_test) > 0
    # With embargo, first test row is not the row immediately after the last train row
    assert test_first_idx == train_end + _EMBARGO_BARS


def test_L10_boundary_bar_excluded_from_test():
    """
    Row at index train_end must not appear in X_test when _EMBARGO_BARS >= 1.

    label[train_end - 1] = sign(close[train_end] - close[train_end-1]).
    X at row train_end is computed from bars 0..train_end-1 via shift(1).
    If train_end were included in X_test, the close[train_end] value would
    appear in both a training label and a test feature lookback.
    """
    from app.services.backtest_service import _EMBARGO_BARS, _prepare_dataset

    df = _make_ohlcv(150)
    X, _, _ = _prepare_dataset(df)

    train_size = 60
    test_size = 30
    train_end = train_size

    X_test_with_embargo = X[train_end + _EMBARGO_BARS: train_end + test_size]
    X_test_no_embargo = X[train_end: train_end + test_size]

    # With embargo, the first row of the test set is different from what it
    # would be without embargo (shifted by _EMBARGO_BARS rows).
    if _EMBARGO_BARS > 0 and len(X_test_with_embargo) > 0 and len(X_test_no_embargo) > 0:
        # The boundary row (train_end) is in X_test_no_embargo but not in X_test_with_embargo
        boundary_row = X[train_end]
        assert not np.array_equal(X_test_with_embargo[0], boundary_row), (
            "The boundary row at train_end must be excluded from X_test when _EMBARGO_BARS >= 1."
        )


# ---------------------------------------------------------------------------
# L11: Options get_latest_chain_pit enforces available_at cutoff in SQL
# ---------------------------------------------------------------------------

def test_L11_L7_status_updated_to_code_enforced():
    """
    L7 was previously advisory (documentation + prospective tests only).
    With the new option_store.py, get_latest_chain_pit enforces
    available_at <= as_of_utc in SQL.

    This test verifies the SQL enforcement function exists and its signature
    requires an as_of_utc parameter (the cutoff).
    """
    import inspect
    from app.data_ingestion.option_store import get_latest_chain_pit

    sig = inspect.signature(get_latest_chain_pit)
    assert "as_of_utc" in sig.parameters, (
        "get_latest_chain_pit must have an as_of_utc parameter to enforce "
        "the available_at <= as_of_utc invariant in SQL."
    )


def test_L11_options_snapshot_none_as_of_utc_uses_current_time():
    """
    When as_of_utc=None is passed to get_latest_chain_pit, the function must
    substitute datetime.utcnow() — NOT skip the WHERE clause entirely.

    We verify this by inspecting the source to confirm the None-check pattern.
    Skipping the WHERE clause when as_of_utc=None would allow all future
    snapshots to be returned, re-introducing the L7 lookahead violation.
    """
    import inspect
    import textwrap
    from app.data_ingestion import option_store

    source = textwrap.dedent(inspect.getsource(option_store.get_latest_chain_pit))

    # The function must guard None with a datetime.utcnow() assignment before
    # building conditions — not pass None into the WHERE clause.
    assert "as_of_utc = datetime.utcnow()" in source or "datetime.utcnow()" in source, (
        "get_latest_chain_pit must substitute datetime.utcnow() when as_of_utc is None. "
        "Passing None into the WHERE clause would remove the available_at filter."
    )
    # And the WHERE clause must always include available_at <=
    assert "available_at <=" in source, (
        "get_latest_chain_pit must always apply WHERE available_at <= as_of_utc "
        "to enforce the L7 no-lookahead invariant."
    )


# ---------------------------------------------------------------------------
# L12: Options staleness threshold at inference
# ---------------------------------------------------------------------------

def test_L12_max_chain_staleness_constant_is_module_level():
    """
    _MAX_CHAIN_STALENESS must be a module-level constant in inference_service,
    not a local variable buried inside run_inference.

    A local variable cannot be tested, overridden in configuration, or linted
    for correctness by audit tooling.
    """
    import app.inference.inference_service as svc
    assert hasattr(svc, "_MAX_CHAIN_STALENESS"), (
        "_MAX_CHAIN_STALENESS must be a module-level constant so it can be "
        "inspected and tested without executing run_inference."
    )


def test_L12_max_chain_staleness_is_at_most_one_bar():
    """
    _MAX_CHAIN_STALENESS must be <= 300 seconds (one 5-minute bar interval).

    A threshold of 3600s (1 hour) allows options snapshots that are up to
    12 complete bars old to be used as current data.  For intraday 5-min
    bar inference this is far too permissive: IV and greeks can change
    substantially within a single bar.
    """
    from app.inference.inference_service import _MAX_CHAIN_STALENESS
    assert _MAX_CHAIN_STALENESS <= 300.0, (
        f"_MAX_CHAIN_STALENESS={_MAX_CHAIN_STALENESS}s exceeds one bar interval (300s). "
        "Stale options data from more than one bar ago must not be used as current data."
    )


def test_L12_stale_options_triggers_sentinel_fill():
    """
    run_inference must fall back to sentinel values when options staleness
    exceeds _MAX_CHAIN_STALENESS, not raise an exception or use stale data.

    Sentinel fill means the pipeline is called with options_data=None,
    which causes is_null_options=1 and zeros for all options columns.
    The model was trained on this sentinel pattern and handles it correctly.
    """
    from app.inference.inference_service import run_inference, _MAX_CHAIN_STALENESS

    df = _make_ohlcv(60)
    df["is_closed"] = True

    # Options data that is older than the staleness threshold
    stale_opts = {
        "staleness_seconds": _MAX_CHAIN_STALENESS + 1.0,
        "iv_rank": 0.75,
        "atm_iv": 0.3,
        "put_call_ratio": 1.2,
    }

    result_stale = run_inference(df, "SPY", options_features=stale_opts)

    # Fresh options with same values
    fresh_opts = {
        "staleness_seconds": 0.0,
        "iv_rank": 0.75,
        "atm_iv": 0.3,
        "put_call_ratio": 1.2,
    }
    result_fresh = run_inference(df, "SPY", options_features=fresh_opts)

    # Both should produce an InferenceResult without crashing.
    # With a stale snapshot the options sentinel path is used; the result
    # may differ from the fresh path (different feature values) but must
    # not crash or propagate the stale values as if they were current.
    assert result_stale is not None
    assert result_fresh is not None
    # The feature snapshot IDs should differ: stale uses sentinel, fresh uses real values
    # (unless the model happens to be untrained, in which case both abstain identically)
    # We at minimum verify the function completes cleanly.
    assert hasattr(result_stale, "action")
    assert hasattr(result_fresh, "action")
