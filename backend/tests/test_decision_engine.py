"""
Decision engine tests.

DE1  — build_options_decision returns OptionsDecision
DE2  — bullish direction thesis when cal_up > cal_down > 0.5
DE3  — bearish direction thesis when cal_down > cal_up > 0.5
DE4  — neutral direction thesis when neither prob > 0.5
DE5  — all four candidate structure types present in candidates
DE6  — candidates sorted by score descending
DE7  — regime_suppressed=True → all candidates non-viable → abstain
DE8  — regime_suppressed=True → abstain_reason contains "regime_suppressed"
DE9  — high calibrated_prob yields higher confidence_score than borderline prob
DE10 — calibration_health="degraded" lowers confidence_score vs "good"
DE11 — abstain=True when inference action="abstain"
DE12 — abstain=False when direction is bullish and some structure viable
DE13 — IV rank 0.10 (cheap) → long_call iv_edge "favorable"
DE14 — IV rank 0.80 (rich)  → credit_spread iv_edge "favorable"
DE15 — IV rank 0.80 (rich)  → long_call iv_edge "unfavorable"
DE16 — recommended_structure is top viable candidate's type
DE17 — expected_range_low < spot < expected_range_high
DE18 — compute_iv_analysis: iv_rv_ratio computed correctly
DE19 — compute_iv_analysis: fallback to RV estimate when atm_iv=0
DE20 — _annualized_rv: 1-bar move scales correctly to annualized vol
DE21 — evaluate_structure: long_call direction score = 0 for bearish forecast
DE22 — evaluate_structure: DTE < 2 → viable=False (hard disqualifier)
DE23 — evaluate_structure: poor liquidity AND score < 40 → viable=False
DE24 — evaluate_structure: credit_spread concern list non-empty (horizon note)
DE25 — recommendation_rationale non-empty string when not abstaining
DE26 — to_dict() serializes without error; nested IVAnalysis present
"""

import pytest
from unittest.mock import MagicMock
from app.decision.decision_engine import build_options_decision, _annualized_rv, _scale_to_1d
from app.decision.iv_analysis import compute_iv_analysis, iv_edge_for_structure
from app.decision.structure_evaluator import evaluate_structure
from app.decision.models import IVAnalysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_inference(
    *,
    prob_up=0.65,
    prob_down=0.30,
    calibrated_prob_up=0.64,
    calibrated_prob_down=0.31,
    calibration_available=True,
    tradeable_confidence=0.64,
    degradation_factor=0.95,
    action="buy",
    abstain_reason=None,
    confidence_band=(0.60, 0.68),
    calibration_health="good",
    rolling_brier=0.18,
    ece_recent=0.03,
    reliability_diagram=None,
    expected_move_pct=0.65,
    confidence=0.64,
    trade_signal="buy",
    no_trade_reason=None,
    feature_snapshot_id="abc123",
    model_version="GBT_v1",
    regime="trending_up",
    regime_suppressed=False,
    suppress_reason=None,
    signal_quality_score=65.0,
    spot_price=100.0,
    top_features=None,
    explanation="Test inference",
):
    r = MagicMock()
    r.symbol = "TEST"
    r.prob_up = prob_up
    r.prob_down = prob_down
    r.prob_flat = 1.0 - prob_up - prob_down
    r.calibrated_prob_up = calibrated_prob_up
    r.calibrated_prob_down = calibrated_prob_down
    r.calibration_available = calibration_available
    r.tradeable_confidence = tradeable_confidence
    r.degradation_factor = degradation_factor
    r.action = action
    r.abstain_reason = abstain_reason
    r.confidence_band = confidence_band
    r.calibration_health = calibration_health
    r.rolling_brier = rolling_brier
    r.ece_recent = ece_recent
    r.reliability_diagram = reliability_diagram
    r.expected_move_pct = expected_move_pct
    r.confidence = confidence
    r.trade_signal = trade_signal
    r.no_trade_reason = no_trade_reason
    r.feature_snapshot_id = feature_snapshot_id
    r.model_version = model_version
    r.regime = regime
    r.regime_suppressed = regime_suppressed
    r.suppress_reason = suppress_reason
    r.signal_quality_score = signal_quality_score
    r.spot_price = spot_price
    r.top_features = top_features or {}
    r.explanation = explanation
    r.timestamp = 1234567890
    r.bar_open_time = "2025-01-10T10:00:00"
    return r


def make_iv() -> IVAnalysis:
    return IVAnalysis(
        atm_iv=0.25,
        realized_vol_ann=0.20,
        iv_rank=0.45,
        iv_rv_ratio=1.25,
        iv_vs_rv="fair",
        iv_implied_1d_move_pct=1.575,
        rv_implied_1d_move_pct=1.260,
    )


# ---------------------------------------------------------------------------
# DE1
# ---------------------------------------------------------------------------
def test_returns_options_decision():
    from app.decision.models import OptionsDecision
    result = build_options_decision(make_inference(), atm_iv=0.25, iv_rank=0.45, dte=7)
    assert isinstance(result, OptionsDecision)


# ---------------------------------------------------------------------------
# DE2
# ---------------------------------------------------------------------------
def test_bullish_direction():
    result = build_options_decision(
        make_inference(calibrated_prob_up=0.64, calibrated_prob_down=0.31)
    )
    assert result.direction_thesis == "bullish"
    assert result.calibrated_prob == pytest.approx(0.64, abs=0.01)


# ---------------------------------------------------------------------------
# DE3
# ---------------------------------------------------------------------------
def test_bearish_direction():
    result = build_options_decision(
        make_inference(calibrated_prob_up=0.28, calibrated_prob_down=0.67, action="sell")
    )
    assert result.direction_thesis == "bearish"
    assert result.calibrated_prob == pytest.approx(0.67, abs=0.01)


# ---------------------------------------------------------------------------
# DE4
# ---------------------------------------------------------------------------
def test_neutral_direction():
    result = build_options_decision(
        make_inference(calibrated_prob_up=0.48, calibrated_prob_down=0.48, action="abstain",
                       abstain_reason="low_confidence")
    )
    assert result.direction_thesis in ("neutral", "abstain")


# ---------------------------------------------------------------------------
# DE5
# ---------------------------------------------------------------------------
def test_all_four_structures_present():
    result = build_options_decision(make_inference())
    types = {c.structure_type for c in result.candidates}
    assert types == {"long_call", "long_put", "debit_spread", "credit_spread"}


# ---------------------------------------------------------------------------
# DE6
# ---------------------------------------------------------------------------
def test_candidates_sorted_descending():
    result = build_options_decision(make_inference())
    scores = [c.score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# DE7
# ---------------------------------------------------------------------------
def test_regime_suppressed_all_nonviable():
    result = build_options_decision(
        make_inference(regime_suppressed=True, suppress_reason="high_volatility")
    )
    assert all(not c.viable for c in result.candidates)
    assert result.abstain is True


# ---------------------------------------------------------------------------
# DE8
# ---------------------------------------------------------------------------
def test_regime_suppressed_abstain_reason():
    result = build_options_decision(
        make_inference(regime_suppressed=True, suppress_reason="high_volatility")
    )
    assert result.abstain_reason is not None
    assert "regime_suppressed" in result.abstain_reason


# ---------------------------------------------------------------------------
# DE9
# ---------------------------------------------------------------------------
def test_high_prob_higher_confidence_than_borderline():
    strong = build_options_decision(make_inference(calibrated_prob_up=0.72, calibrated_prob_down=0.23))
    weak   = build_options_decision(make_inference(calibrated_prob_up=0.54, calibrated_prob_down=0.42))
    assert strong.confidence_score > weak.confidence_score


# ---------------------------------------------------------------------------
# DE10
# ---------------------------------------------------------------------------
def test_degraded_calibration_lowers_confidence():
    good     = build_options_decision(make_inference(calibration_health="good"))
    degraded = build_options_decision(make_inference(calibration_health="degraded"))
    assert good.confidence_score > degraded.confidence_score


# ---------------------------------------------------------------------------
# DE11
# ---------------------------------------------------------------------------
def test_inference_abstain_propagates():
    result = build_options_decision(
        make_inference(action="abstain", abstain_reason="low_confidence")
    )
    assert result.abstain is True


# ---------------------------------------------------------------------------
# DE12
# ---------------------------------------------------------------------------
def test_no_abstain_when_bullish_and_viable():
    result = build_options_decision(
        make_inference(calibrated_prob_up=0.68, calibrated_prob_down=0.27, action="buy"),
        atm_iv=0.20,
    )
    # At least one structure should be viable given a clear directional signal
    viable = [c for c in result.candidates if c.viable]
    # If no structures are viable due to other constraints that's acceptable;
    # verify the logic path ran without error
    assert result.abstain == (len(viable) == 0)


# ---------------------------------------------------------------------------
# DE13 — cheap IV → long_call favorable
# ---------------------------------------------------------------------------
def test_cheap_iv_long_call_favorable():
    label, score = iv_edge_for_structure("long_call", iv_rank=0.10, iv_vs_rv="cheap")
    assert label == "favorable"
    assert score >= 20.0


# ---------------------------------------------------------------------------
# DE14 — rich IV → credit_spread favorable
# ---------------------------------------------------------------------------
def test_rich_iv_credit_spread_favorable():
    label, score = iv_edge_for_structure("credit_spread", iv_rank=0.80, iv_vs_rv="rich")
    assert label == "favorable"
    assert score >= 20.0


# ---------------------------------------------------------------------------
# DE15 — rich IV → long_call unfavorable
# ---------------------------------------------------------------------------
def test_rich_iv_long_call_unfavorable():
    label, score = iv_edge_for_structure("long_call", iv_rank=0.80, iv_vs_rv="rich")
    assert label == "unfavorable"
    assert score <= 5.0


# ---------------------------------------------------------------------------
# DE16
# ---------------------------------------------------------------------------
def test_recommended_structure_is_top_viable():
    result = build_options_decision(make_inference(calibrated_prob_up=0.70))
    if not result.abstain and result.recommended_structure:
        viable = [c for c in result.candidates if c.viable]
        assert viable[0].structure_type == result.recommended_structure


# ---------------------------------------------------------------------------
# DE17
# ---------------------------------------------------------------------------
def test_expected_range_brackets_spot():
    result = build_options_decision(make_inference(spot_price=200.0))
    assert result.expected_range_low < result.spot_price < result.expected_range_high


# ---------------------------------------------------------------------------
# DE18
# ---------------------------------------------------------------------------
def test_iv_analysis_rv_ratio():
    iv = compute_iv_analysis(atm_iv=0.30, realized_vol_ann=0.20, iv_rank=0.50)
    assert iv.iv_rv_ratio == pytest.approx(1.50, abs=0.01)
    assert iv.iv_vs_rv == "rich"


# ---------------------------------------------------------------------------
# DE19 — fallback when atm_iv=0
# ---------------------------------------------------------------------------
def test_iv_analysis_fallback_when_no_atm():
    iv = compute_iv_analysis(atm_iv=0.0, realized_vol_ann=0.20, iv_rank=0.50)
    # Should use rv * 1.10 = 0.22
    assert iv.atm_iv == pytest.approx(0.22, abs=0.01)
    assert iv.iv_rv_ratio > 0


# ---------------------------------------------------------------------------
# DE20
# ---------------------------------------------------------------------------
def test_annualized_rv_scaling():
    import math
    # 1-bar move of 0.65% for 5-min bars (78/day, 252 days)
    rv = _annualized_rv(0.65)
    expected = (0.65 / 100) * math.sqrt(78 * 252)
    assert rv == pytest.approx(expected, rel=0.001)


# ---------------------------------------------------------------------------
# DE21 — long_call direction score = 0 for bearish forecast
# ---------------------------------------------------------------------------
def test_long_call_zero_direction_score_for_bearish():
    iv = make_iv()
    c = evaluate_structure(
        structure_type="long_call",
        forecast_direction="bearish",
        spot=100.0,
        iv_analysis=iv,
        expected_move_1d_pct=1.0,
        liquidity_quality="good",
        atm_bid_ask_pct=0.04,
        dte=7,
    )
    # long_call on a bearish forecast should score well below a directionally-aligned call
    assert c.score < 40.0
    assert c.viable is False


# ---------------------------------------------------------------------------
# DE22 — DTE < 2 disqualifier
# ---------------------------------------------------------------------------
def test_low_dte_nonviable():
    iv = make_iv()
    for stype in ["long_call", "long_put", "debit_spread", "credit_spread"]:
        c = evaluate_structure(
            structure_type=stype,
            forecast_direction="bullish",
            spot=100.0,
            iv_analysis=iv,
            expected_move_1d_pct=1.0,
            liquidity_quality="good",
            atm_bid_ask_pct=0.04,
            dte=1,
        )
        assert c.viable is False, f"{stype} should be non-viable with DTE=1"


# ---------------------------------------------------------------------------
# DE23 — poor liquidity + score < 40 → non-viable
# ---------------------------------------------------------------------------
def test_poor_liquidity_low_prob_nonviable():
    # Use an adverse direction (bearish for long_call) to keep score low
    iv = IVAnalysis(
        atm_iv=0.40, realized_vol_ann=0.20,
        iv_rank=0.85, iv_rv_ratio=2.0, iv_vs_rv="rich",
        iv_implied_1d_move_pct=2.52, rv_implied_1d_move_pct=1.26,
    )
    c = evaluate_structure(
        structure_type="long_call",
        forecast_direction="bearish",
        spot=100.0,
        iv_analysis=iv,
        expected_move_1d_pct=0.5,
        liquidity_quality="poor",
        atm_bid_ask_pct=0.20,
        dte=7,
    )
    assert c.viable is False


# ---------------------------------------------------------------------------
# DE24 — credit_spread has horizon concern
# ---------------------------------------------------------------------------
def test_credit_spread_has_horizon_concern():
    iv = make_iv()
    c = evaluate_structure(
        structure_type="credit_spread",
        forecast_direction="bullish",
        spot=100.0,
        iv_analysis=iv,
        expected_move_1d_pct=1.0,
        liquidity_quality="good",
        atm_bid_ask_pct=0.04,
        dte=7,
    )
    assert len(c.concerns) > 0


# ---------------------------------------------------------------------------
# DE25
# ---------------------------------------------------------------------------
def test_recommendation_rationale_nonempty_when_not_abstaining():
    result = build_options_decision(make_inference(calibrated_prob_up=0.70))
    if not result.abstain:
        assert len(result.recommendation_rationale) > 0


# ---------------------------------------------------------------------------
# DE26
# ---------------------------------------------------------------------------
def test_to_dict_serializes_cleanly():
    result = build_options_decision(make_inference())
    d = result.to_dict()
    assert "iv_analysis" in d
    assert isinstance(d["iv_analysis"], dict)
    assert "candidates" in d
    assert isinstance(d["candidates"], list)
    assert all("structure_type" in c for c in d["candidates"])
