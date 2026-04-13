"""
Options decision data model.

OptionsDecision is the top-level output. It contains:
  - The directional thesis and probability context
  - IV analysis and liquidity quality
  - Expected move and range (scaled to options horizon)
  - Four scored candidate structures
  - One recommended structure (or abstain)

StructureCandidate holds the per-structure evaluation. All four structures
are always scored; the consumer chooses how many to display.

StructureLeg is a single option contract within a multi-leg structure.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple


@dataclass
class StructureLeg:
    """One option contract in a structure (buy or sell)."""
    action: str              # "buy" | "sell"
    option_type: str         # "call" | "put"
    target_delta: float
    strike: Optional[float] = None
    expiry: Optional[str] = None
    estimated_mid: Optional[float] = None
    estimated_iv: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None

    def bid_ask_spread(self) -> Optional[float]:
        if self.bid is not None and self.ask is not None and self.estimated_mid:
            return (self.ask - self.bid) / (self.estimated_mid + 1e-9)
        return None


@dataclass
class StructureCandidate:
    """
    Full evaluation of one candidate options structure.

    Score interpretation (0–100):
      < 25  — not viable; abstain from this structure
      25–50 — marginal; conditional use only
      50–70 — acceptable; proceed with caution
      > 70  — strong setup; favorable conditions
    """
    structure_type: str      # "long_call" | "long_put" | "debit_spread" | "credit_spread"
    direction: str           # "bullish" | "bearish" (the structure's directional lean)

    # Scoring
    score: float             # 0–100 composite
    viable: bool             # score >= MIN_VIABLE_SCORE and no hard disqualifiers

    # Legs (populated when options chain is available)
    legs: List[StructureLeg] = field(default_factory=list)

    # Cost / payoff profile
    estimated_cost_pct: float = 0.0      # net debit as % of spot (for debit structs)
    estimated_credit_pct: float = 0.0    # net credit as % of max loss (for credit structs)
    max_profit_pct: float = 0.0          # max profit as % of spot
    max_loss_pct: float = 0.0            # max loss as % of spot
    breakeven_move_pct: float = 0.0      # % move required to break even

    # Spread geometry (None for outright structures)
    spread_width_pct: Optional[float] = None   # (short_strike - long_strike) / spot

    # IV assessment
    iv_edge: str = "neutral"             # "favorable" | "neutral" | "unfavorable"
    iv_edge_score: float = 0.0           # raw contribution to total score

    # Liquidity
    liquidity_fit: str = "unknown"       # "good" | "fair" | "poor" | "unknown"
    estimated_fill_cost_pct: float = 0.0 # round-trip bid-ask friction as % of premium

    # Horizon note
    horizon_note: str = ""

    # Human explanation
    rationale: str = ""
    tailwinds: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class IVAnalysis:
    """Implied volatility context relative to realized volatility."""
    atm_iv: float                  # ATM implied vol (annualized)
    realized_vol_ann: float        # annualized realized vol from model features
    iv_rank: float                 # 0–1 percentile rank over lookback
    iv_rv_ratio: float             # atm_iv / realized_vol_ann
    iv_vs_rv: str                  # "cheap" | "fair" | "rich"
    iv_implied_1d_move_pct: float  # ATM IV scaled to 1-day move
    rv_implied_1d_move_pct: float  # RV scaled to 1-day move

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OptionsDecision:
    """
    Full options research output for one symbol at one point in time.

    All four candidate structures are always included and scored.
    `recommended_structure` names the top viable candidate, or None if abstaining.
    `abstain` is True when no structure clears the minimum viable threshold
    or when regime/calibration conditions prevent trading.
    """
    symbol: str
    generated_at: str
    spot_price: float

    # ─── Directional thesis ──────────────────────────────────────────────────
    direction_thesis: str          # "bullish" | "bearish" | "neutral" | "abstain"
    horizon: str                   # human-readable, e.g. "1-bar (5 min)"

    # ─── Probability context ─────────────────────────────────────────────────
    calibrated_prob: float         # direction-specific prob (prob_up if bullish, else prob_down)
    prob_up: float
    prob_down: float
    confidence_band: Tuple[float, float]

    # ─── Move expectations ───────────────────────────────────────────────────
    expected_move_1bar_pct: float  # 1-bar model expected move (raw)
    expected_move_1d_pct: float    # scaled to 1 day for options context
    expected_range_low: float      # spot × (1 − expected_move_1d_pct/100)
    expected_range_high: float     # spot × (1 + expected_move_1d_pct/100)

    # ─── IV context ──────────────────────────────────────────────────────────
    iv_analysis: IVAnalysis

    # ─── Options market context ──────────────────────────────────────────────
    expiry: str
    dte: int                        # days to expiry
    liquidity_quality: str          # "good" | "fair" | "poor"
    atm_bid_ask_pct: float          # (ask−bid)/mid at ATM strike

    # ─── Model context ───────────────────────────────────────────────────────
    regime: str
    regime_suppressed: bool
    calibration_health: str
    signal_quality_score: float

    # ─── Decision output ─────────────────────────────────────────────────────
    confidence_score: float         # 0–100 composite decision confidence
    abstain: bool
    abstain_reason: Optional[str]

    # ─── Candidate structures ────────────────────────────────────────────────
    candidates: List[StructureCandidate]    # sorted by score descending
    recommended_structure: Optional[str]   # None if abstain
    recommendation_rationale: str

    # ─── OI context (optional) ───────────────────────────────────────────────
    # Strikes with heavy open interest provided to the evaluator.
    # Heavy OI creates gamma-pinning pressure; included here for audit/display.
    oi_concentrations: Optional[List[float]] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d
