"""
Strategy rules engine — converts model predictions into paper-trade ideas.

Rules are configurable and kept SEPARATE from the model.
The engine is a pure function: given a signal + market state → TradeIdea or None.

Default rule set:
- Only consider bullish setups when P(up) > min_prob_threshold
- Only consider bearish setups when P(down) > min_prob_threshold
- Require minimum expected move (avoids micro-moves)
- Require acceptable spread width (liquidity filter)
- Prefer strikes near configurable delta target
- Suppress in low-confidence or high-volatility regimes
- No new trades in last 30 min of session (configurable)
"""

from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime, time, timezone
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore


@dataclass
class RulesConfig:
    # Probability thresholds
    min_prob_bullish: float = 0.60
    min_prob_bearish: float = 0.60
    min_confidence: float = 0.20   # abs(p - 0.5) * 2

    # Signal quality
    min_signal_quality_score: float = 40.0
    min_expected_move_pct: float = 0.15   # minimum expected move %

    # Options liquidity
    max_spread_pct: float = 0.05       # max (ask-bid)/mid allowed
    min_open_interest: int = 100
    min_volume: int = 10

    # Strike selection
    target_delta_long: float = 0.40    # target delta for long calls/puts
    target_delta_short: float = 0.20   # target delta for short legs
    delta_tolerance: float = 0.10      # acceptable range around target

    # Session rules
    suppress_last_minutes: int = 30    # suppress trades in last N minutes
    session_end_hour: int = 16         # 4pm ET
    session_end_minute: int = 0

    # Regime rules
    allowed_regimes: List[str] = None  # None = all except suppressed

    def __post_init__(self):
        if self.allowed_regimes is None:
            self.allowed_regimes = [
                "trending_up", "trending_down", "mean_reverting", "low_volatility"
            ]


@dataclass
class TradeIdea:
    symbol: str
    direction: str          # "bullish" or "bearish"
    strategy: str           # "long_call", "long_put", "call_spread", "put_spread"
    target_strike: Optional[float]
    target_delta: float
    max_spread_width: Optional[float]
    prob_signal: float
    expected_move_pct: float
    signal_quality: float
    regime: str
    rationale: str
    blocked: bool = False
    block_reason: Optional[str] = None


def evaluate_rules(
    symbol: str,
    prob_up: float,
    prob_down: float,
    confidence: float,
    expected_move_pct: float,
    signal_quality_score: float,
    regime: str,
    current_time: Optional[datetime] = None,
    config: Optional[RulesConfig] = None,
) -> TradeIdea:
    """
    Apply strategy rules and return a TradeIdea.
    If rules block the trade, TradeIdea.blocked=True with reason.
    """
    config = config or RulesConfig()
    now = current_time or datetime.utcnow()

    # --- Pre-flight checks ---

    # 1. Confidence check
    if confidence < config.min_confidence:
        return _blocked(symbol, "low_confidence", prob_up, prob_down,
                        expected_move_pct, signal_quality_score, regime, config,
                        f"confidence={confidence:.2f} < min={config.min_confidence:.2f}")

    # 2. Signal quality
    if signal_quality_score < config.min_signal_quality_score:
        return _blocked(symbol, "low_signal_quality", prob_up, prob_down,
                        expected_move_pct, signal_quality_score, regime, config,
                        f"quality={signal_quality_score:.0f} < min={config.min_signal_quality_score:.0f}")

    # 3. Expected move
    if expected_move_pct < config.min_expected_move_pct:
        return _blocked(symbol, "insufficient_expected_move", prob_up, prob_down,
                        expected_move_pct, signal_quality_score, regime, config,
                        f"expected_move={expected_move_pct:.2f}% < min={config.min_expected_move_pct:.2f}%")

    # 4. Regime check
    if regime not in config.allowed_regimes:
        return _blocked(symbol, f"regime_not_allowed:{regime}", prob_up, prob_down,
                        expected_move_pct, signal_quality_score, regime, config,
                        f"regime={regime} not in allowed={config.allowed_regimes}")

    # 5. Session close check — DST-safe UTC→ET conversion
    #    US/Eastern is UTC-5 (EST) Nov–Mar and UTC-4 (EDT) Mar–Nov.
    #    We use zoneinfo to get the correct offset at the moment of evaluation.
    ET = ZoneInfo("America/New_York")
    if now.tzinfo is None:
        # Assume UTC if naive
        now_aware = now.replace(tzinfo=timezone.utc)
    else:
        now_aware = now
    now_et_dt = now_aware.astimezone(ET)
    now_et = now_et_dt.time()

    session_end = time(config.session_end_hour, config.session_end_minute)
    total_cutoff_minutes = (
        config.session_end_hour * 60 + config.session_end_minute
        - config.suppress_last_minutes
    )
    session_cutoff = time(total_cutoff_minutes // 60, total_cutoff_minutes % 60)
    if session_cutoff <= now_et <= session_end:
        return _blocked(symbol, "near_session_close", prob_up, prob_down,
                        expected_move_pct, signal_quality_score, regime, config,
                        f"within {config.suppress_last_minutes} min of session close")

    # --- Determine direction and strategy ---

    if prob_up >= config.min_prob_bullish and prob_up > prob_down:
        direction = "bullish"
        strategy = _select_strategy(direction, expected_move_pct)
        target_delta = config.target_delta_long
        rationale = (
            f"Bullish: P(up)={prob_up:.2f}, expected_move={expected_move_pct:.2f}%, "
            f"quality={signal_quality_score:.0f}, regime={regime}"
        )
    elif prob_down >= config.min_prob_bearish and prob_down > prob_up:
        direction = "bearish"
        strategy = _select_strategy(direction, expected_move_pct)
        target_delta = config.target_delta_long
        rationale = (
            f"Bearish: P(down)={prob_down:.2f}, expected_move={expected_move_pct:.2f}%, "
            f"quality={signal_quality_score:.0f}, regime={regime}"
        )
    else:
        return _blocked(symbol, "no_directional_edge", prob_up, prob_down,
                        expected_move_pct, signal_quality_score, regime, config,
                        f"P(up)={prob_up:.2f} P(down)={prob_down:.2f} — no clear edge")

    prob_signal = prob_up if direction == "bullish" else prob_down

    return TradeIdea(
        symbol=symbol,
        direction=direction,
        strategy=strategy,
        target_strike=None,  # resolved at execution using live chain
        target_delta=target_delta,
        max_spread_width=None,
        prob_signal=round(prob_signal, 4),
        expected_move_pct=round(expected_move_pct, 4),
        signal_quality=round(signal_quality_score, 1),
        regime=regime,
        rationale=rationale,
        blocked=False,
        block_reason=None,
    )


def _select_strategy(direction: str, expected_move_pct: float) -> str:
    """
    Choose strategy based on expected move size.
    - Large expected move → outright long (more gamma)
    - Small expected move → debit spread (defined risk, lower cost)
    """
    if expected_move_pct > 0.5:
        return "long_call" if direction == "bullish" else "long_put"
    else:
        return "call_spread" if direction == "bullish" else "put_spread"


def _blocked(
    symbol: str, reason: str,
    prob_up: float, prob_down: float,
    expected_move_pct: float, signal_quality: float,
    regime: str, config: RulesConfig, message: str,
) -> TradeIdea:
    return TradeIdea(
        symbol=symbol,
        direction="no_trade",
        strategy="none",
        target_strike=None,
        target_delta=0.0,
        max_spread_width=None,
        prob_signal=max(prob_up, prob_down),
        expected_move_pct=expected_move_pct,
        signal_quality=signal_quality,
        regime=regime,
        rationale=f"Blocked: {message}",
        blocked=True,
        block_reason=reason,
    )
