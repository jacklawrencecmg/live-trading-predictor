import pytest
from app.paper_trading.rules_engine import evaluate_rules, RulesConfig


def _config(**kwargs) -> RulesConfig:
    base = dict(
        min_prob_bullish=0.60, min_prob_bearish=0.60, min_confidence=0.20,
        min_signal_quality_score=40.0, min_expected_move_pct=0.15,
        suppress_last_minutes=0,  # disable session close check for tests
    )
    base.update(kwargs)
    return RulesConfig(**base)


def test_bullish_trade_idea():
    idea = evaluate_rules(
        "SPY", prob_up=0.70, prob_down=0.30, confidence=0.40,
        expected_move_pct=0.30, signal_quality_score=70.0,
        regime="trending_up", config=_config(),
    )
    assert idea.direction == "bullish"
    assert not idea.blocked


def test_bearish_trade_idea():
    idea = evaluate_rules(
        "SPY", prob_up=0.30, prob_down=0.70, confidence=0.40,
        expected_move_pct=0.30, signal_quality_score=70.0,
        regime="trending_down", config=_config(),
    )
    assert idea.direction == "bearish"
    assert not idea.blocked


def test_blocked_low_confidence():
    idea = evaluate_rules(
        "SPY", prob_up=0.55, prob_down=0.45, confidence=0.10,
        expected_move_pct=0.30, signal_quality_score=70.0,
        regime="trending_up", config=_config(),
    )
    assert idea.blocked
    assert "confidence" in idea.block_reason


def test_blocked_bad_regime():
    idea = evaluate_rules(
        "SPY", prob_up=0.70, prob_down=0.30, confidence=0.40,
        expected_move_pct=0.30, signal_quality_score=70.0,
        regime="high_volatility", config=_config(),
    )
    assert idea.blocked
    assert "regime" in idea.block_reason


def test_blocked_low_quality():
    idea = evaluate_rules(
        "SPY", prob_up=0.70, prob_down=0.30, confidence=0.40,
        expected_move_pct=0.30, signal_quality_score=20.0,
        regime="trending_up", config=_config(),
    )
    assert idea.blocked
    assert "quality" in idea.block_reason


def test_strategy_selection_large_move():
    idea = evaluate_rules(
        "SPY", prob_up=0.70, prob_down=0.30, confidence=0.40,
        expected_move_pct=1.0, signal_quality_score=70.0,
        regime="trending_up", config=_config(),
    )
    assert idea.strategy == "long_call"


def test_strategy_selection_small_move():
    idea = evaluate_rules(
        "SPY", prob_up=0.70, prob_down=0.30, confidence=0.40,
        expected_move_pct=0.20, signal_quality_score=70.0,
        regime="trending_up", config=_config(),
    )
    assert idea.strategy == "call_spread"
