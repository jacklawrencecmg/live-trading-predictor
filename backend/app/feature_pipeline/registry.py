"""
Feature Registry — single source of truth for feature definitions.

Every feature used by the ML pipeline is declared here with:
  - name: canonical identifier (valid Python identifier)
  - version: integer version; increment when formula changes
  - group: functional grouping for documentation and selective computation
  - description: what it measures
  - formula: human-readable formula string
  - units: units of measurement or mathematical domain
  - expected_min / expected_max: typical range (not hard bounds)
  - null_strategy:
      "required"          — row is invalid when null; model training drops it
      "optional_sentinel" — null → sentinel_value; a per-group missingness
                             indicator feature is added to the feature vector
  - sentinel_value: replacement for null when null_strategy == "optional_sentinel"

The MANIFEST_HASH is SHA-256 of sorted (name, version) pairs — it changes
whenever any feature formula is updated, invalidating cached FeatureRow rows.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

PIPELINE_VERSION: int = 2  # global bump triggers full re-compute of all stored rows

# Maximum consecutive bars over which ffill() propagates NaN values.
# One full 5-minute trading session = 78 bars (6.5 h × 12 bars/h).
# Prevents stale values from crossing overnight/weekend session gaps.
FFILL_LIMIT: int = 78


@dataclass(frozen=True)
class FeatureDef:
    name: str
    version: int
    group: str
    description: str
    formula: str
    units: str
    expected_min: Optional[float]
    expected_max: Optional[float]
    null_strategy: str          # "required" | "optional_sentinel"
    sentinel_value: float = 0.0


# ---------------------------------------------------------------------------
# Registry — defines all features in the pipeline
# ---------------------------------------------------------------------------

_RAW: List[FeatureDef] = [

    # -----------------------------------------------------------------------
    # TREND: oscillators and band-based features
    # -----------------------------------------------------------------------
    FeatureDef(
        name="rsi_14",  version=1,  group="trend",
        description="Relative Strength Index over 14 bars",
        formula="100 - 100/(1 + EWM_avg_gain(14) / EWM_avg_loss(14))",
        units="dimensionless [0, 100]",
        expected_min=0.0, expected_max=100.0,
        null_strategy="required",
    ),
    FeatureDef(
        name="rsi_5",   version=1,  group="trend",
        description="Relative Strength Index over 5 bars — faster oscillator",
        formula="100 - 100/(1 + EWM_avg_gain(5) / EWM_avg_loss(5))",
        units="dimensionless [0, 100]",
        expected_min=0.0, expected_max=100.0,
        null_strategy="required",
    ),
    FeatureDef(
        name="macd_line",   version=1,  group="trend",
        description="MACD line: fast EMA(12) minus slow EMA(26)",
        formula="EMA(close, 12) - EMA(close, 26)",
        units="price-relative (divided by close in pipeline)",
        expected_min=None, expected_max=None,
        null_strategy="required",
    ),
    FeatureDef(
        name="macd_signal", version=1,  group="trend",
        description="9-bar EMA of the MACD line",
        formula="EMA(macd_line, 9)",
        units="price-relative",
        expected_min=None, expected_max=None,
        null_strategy="required",
    ),
    FeatureDef(
        name="macd_hist",   version=1,  group="trend",
        description="MACD histogram: macd_line minus macd_signal",
        formula="macd_line - macd_signal",
        units="price-relative",
        expected_min=None, expected_max=None,
        null_strategy="required",
    ),
    FeatureDef(
        name="bb_pct",  version=1,  group="trend",
        description="Bollinger Band %%B: position within 2-sigma band",
        formula="(close - lower_band) / (upper_band - lower_band)",
        units="dimensionless; 0 = lower band, 1 = upper band",
        expected_min=-0.5, expected_max=1.5,
        null_strategy="required",
    ),

    # -----------------------------------------------------------------------
    # VOLATILITY: ATR, realized vol, vol-regime
    # -----------------------------------------------------------------------
    FeatureDef(
        name="atr_norm", version=1, group="volatility",
        description="ATR(14) normalized by close price",
        formula="ATR(14) / close",
        units="dimensionless pct",
        expected_min=0.0, expected_max=0.1,
        null_strategy="required",
    ),
    FeatureDef(
        name="atr_14", version=1, group="volatility",
        description="Raw ATR(14) in price units",
        formula="EWM(True-Range, span=14)",
        units="price",
        expected_min=0.0, expected_max=None,
        null_strategy="required",
    ),
    FeatureDef(
        name="realized_vol_5", version=1, group="volatility",
        description="5-bar realized volatility, annualized",
        formula="std(log_ret, window=5) * sqrt(252 * 78)",
        units="annualized vol (5-min bars)",
        expected_min=0.0, expected_max=5.0,
        null_strategy="required",
    ),
    FeatureDef(
        name="realized_vol_10", version=1, group="volatility",
        description="10-bar realized volatility, annualized",
        formula="std(log_ret, window=10) * sqrt(252 * 78)",
        units="annualized vol (5-min bars)",
        expected_min=0.0, expected_max=5.0,
        null_strategy="required",
    ),
    FeatureDef(
        name="realized_vol_20", version=1, group="volatility",
        description="20-bar realized volatility, annualized",
        formula="std(log_ret, window=20) * sqrt(252 * 78)",
        units="annualized vol (5-min bars)",
        expected_min=0.0, expected_max=5.0,
        null_strategy="required",
    ),
    FeatureDef(
        name="vol_regime", version=1, group="volatility",
        description="Volatility regime: ratio of short-term to long-term realized vol",
        formula="realized_vol_10 / (realized_vol_60 + eps)",
        units="dimensionless ratio; >1 = elevated short-term vol",
        expected_min=0.0, expected_max=5.0,
        null_strategy="required",
    ),

    # -----------------------------------------------------------------------
    # MOMENTUM / REVERSION: returns and z-score
    # -----------------------------------------------------------------------
    FeatureDef(
        name="ret_1",  version=1, group="momentum",
        description="1-bar percentage return (prior bar)",
        formula="close[i-1] / close[i-2] - 1",
        units="pct return",
        expected_min=-0.1, expected_max=0.1,
        null_strategy="required",
    ),
    FeatureDef(
        name="ret_5",  version=1, group="momentum",
        description="5-bar percentage return",
        formula="close[i-1] / close[i-6] - 1",
        units="pct return",
        expected_min=-0.2, expected_max=0.2,
        null_strategy="required",
    ),
    FeatureDef(
        name="ret_10", version=1, group="momentum",
        description="10-bar percentage return",
        formula="close[i-1] / close[i-11] - 1",
        units="pct return",
        expected_min=-0.3, expected_max=0.3,
        null_strategy="required",
    ),
    FeatureDef(
        name="ret_20", version=1, group="momentum",
        description="20-bar percentage return",
        formula="close[i-1] / close[i-21] - 1",
        units="pct return",
        expected_min=-0.4, expected_max=0.4,
        null_strategy="required",
    ),
    FeatureDef(
        name="ret_60", version=1, group="momentum",
        description="60-bar percentage return — medium-term momentum (≈ 5 h on 5-min bars)",
        formula="close[i-1] / close[i-61] - 1",
        units="pct return",
        expected_min=-0.6, expected_max=0.6,
        null_strategy="required",
    ),
    FeatureDef(
        name="zscore_20", version=1, group="momentum",
        description="20-bar z-score of close — mean-reversion signal",
        formula="(close[i-1] - mean(close[i-21..i-1])) / std(close[i-21..i-1])",
        units="standard deviations",
        expected_min=-4.0, expected_max=4.0,
        null_strategy="required",
    ),

    # -----------------------------------------------------------------------
    # VWAP: position and slope relative to intraday VWAP
    # -----------------------------------------------------------------------
    FeatureDef(
        name="vwap_distance", version=1, group="vwap",
        description="Distance of prior close from prior-bar session VWAP",
        formula="(close[i-1] - vwap[i-1]) / close[i-1]",
        units="dimensionless pct; positive = above VWAP",
        expected_min=-0.05, expected_max=0.05,
        null_strategy="required",
    ),
    FeatureDef(
        name="vwap_slope", version=1, group="vwap",
        description="5-bar slope of session VWAP — VWAP trend direction",
        formula="(vwap[i-1] - vwap[i-6]) / (|vwap[i-6]| + eps)",
        units="dimensionless pct per 5 bars",
        expected_min=-0.02, expected_max=0.02,
        null_strategy="required",
    ),

    # -----------------------------------------------------------------------
    # VOLUME: relative volume and z-score
    # -----------------------------------------------------------------------
    FeatureDef(
        name="volume_ratio", version=1, group="volume",
        description="Volume relative to 20-bar moving average",
        formula="volume[i-1] / mean(volume[i-21..i-1])",
        units="dimensionless ratio; >1 = above-average volume",
        expected_min=0.0, expected_max=10.0,
        null_strategy="required",
    ),
    FeatureDef(
        name="volume_trend", version=1, group="volume",
        description="Short-term vs long-term volume trend",
        formula="mean(volume[i-6..i-1]) / mean(volume[i-21..i-1])",
        units="dimensionless ratio",
        expected_min=0.0, expected_max=5.0,
        null_strategy="required",
    ),
    FeatureDef(
        name="volume_zscore", version=1, group="volume",
        description="Z-score of volume vs 20-bar distribution",
        formula="(volume[i-1] - mean(volume[i-21..i-1])) / std(volume[i-21..i-1])",
        units="standard deviations",
        expected_min=-3.0, expected_max=10.0,
        null_strategy="required",
    ),

    # -----------------------------------------------------------------------
    # INTRADAY SEASONALITY: cyclic time encoding + session markers
    # -----------------------------------------------------------------------
    FeatureDef(
        name="hour_sin",  version=1, group="seasonality",
        description="Sine component of hour-of-day (24-h cycle)",
        formula="sin(2π * hour_fraction / 24)",
        units="dimensionless [-1, 1]",
        expected_min=-1.0, expected_max=1.0,
        null_strategy="required",
    ),
    FeatureDef(
        name="hour_cos",  version=1, group="seasonality",
        description="Cosine component of hour-of-day (24-h cycle)",
        formula="cos(2π * hour_fraction / 24)",
        units="dimensionless [-1, 1]",
        expected_min=-1.0, expected_max=1.0,
        null_strategy="required",
    ),
    FeatureDef(
        name="minute_sin", version=1, group="seasonality",
        description="Sine component of minute-within-hour (60-min cycle)",
        formula="sin(2π * minute / 60)",
        units="dimensionless [-1, 1]",
        expected_min=-1.0, expected_max=1.0,
        null_strategy="required",
    ),
    FeatureDef(
        name="minute_cos", version=1, group="seasonality",
        description="Cosine component of minute-within-hour (60-min cycle)",
        formula="cos(2π * minute / 60)",
        units="dimensionless [-1, 1]",
        expected_min=-1.0, expected_max=1.0,
        null_strategy="required",
    ),
    FeatureDef(
        name="session_progress", version=1, group="seasonality",
        description="Fraction of the regular trading session elapsed",
        formula="(current_time - 09:30 ET) / 390 min, clipped to [0, 1]",
        units="dimensionless [0, 1]",
        expected_min=0.0, expected_max=1.0,
        null_strategy="required",
    ),
    FeatureDef(
        name="is_first_30min", version=1, group="seasonality",
        description="1 if within the first 30 minutes of the trading session",
        formula="1 if session_progress ∈ [0, 0.077], else 0",
        units="binary {0, 1}",
        expected_min=0.0, expected_max=1.0,
        null_strategy="required",
    ),
    FeatureDef(
        name="is_last_30min", version=1, group="seasonality",
        description="1 if within the last 30 minutes of the trading session",
        formula="1 if session_progress ∈ [0.923, 1.0], else 0",
        units="binary {0, 1}",
        expected_min=0.0, expected_max=1.0,
        null_strategy="required",
    ),

    # -----------------------------------------------------------------------
    # OPTIONS (optional_sentinel): options chain summary features
    # Null when no options snapshot is available for this bar.
    # Filled with sentinel_value=0.0; is_null_options indicator set to 1.
    # -----------------------------------------------------------------------
    FeatureDef(
        name="atm_iv",  version=1, group="options",
        description="At-the-money implied volatility from the nearest-expiry chain",
        formula="linear interpolation of IV at strikes bracketing the spot price",
        units="annualized implied vol (e.g., 0.20 = 20%)",
        expected_min=0.0, expected_max=3.0,
        null_strategy="optional_sentinel", sentinel_value=0.0,
    ),
    FeatureDef(
        name="iv_rank", version=1, group="options",
        description="IV rank: where current ATM IV sits in the 52-week IV range",
        formula="(atm_iv - iv_52w_low) / (iv_52w_high - iv_52w_low + eps)",
        units="dimensionless [0, 1]; 1 = at 52-week high IV",
        expected_min=0.0, expected_max=1.0,
        null_strategy="optional_sentinel", sentinel_value=0.0,
    ),
    FeatureDef(
        name="iv_skew", version=1, group="options",
        description="25-delta put IV minus 25-delta call IV — demand for downside protection",
        formula="IV(25-delta put) - IV(25-delta call)",
        units="annualized vol spread; positive = put premium",
        expected_min=-0.2, expected_max=0.5,
        null_strategy="optional_sentinel", sentinel_value=0.0,
    ),
    FeatureDef(
        name="pc_volume_ratio", version=1, group="options",
        description="Put-to-call volume ratio across all strikes and expiries",
        formula="sum(put_volume) / (sum(call_volume) + eps)",
        units="dimensionless ratio; >1 = more put activity",
        expected_min=0.0, expected_max=5.0,
        null_strategy="optional_sentinel", sentinel_value=1.0,
    ),
    FeatureDef(
        name="pc_oi_ratio", version=1, group="options",
        description="Put-to-call open interest ratio",
        formula="sum(put_OI) / (sum(call_OI) + eps)",
        units="dimensionless ratio",
        expected_min=0.0, expected_max=5.0,
        null_strategy="optional_sentinel", sentinel_value=1.0,
    ),
    FeatureDef(
        name="gex_proxy", version=1, group="options",
        description="Gamma exposure proxy: net dealer gamma times open interest",
        formula="sum(gamma_i * OI_i * 100 * spot) for all contracts",
        units="dollar gamma (proxy, unnormalized)",
        expected_min=None, expected_max=None,
        null_strategy="optional_sentinel", sentinel_value=0.0,
    ),
    FeatureDef(
        name="dist_to_max_oi", version=1, group="options",
        description="Distance from spot to the strike with highest open interest",
        formula="(spot - strike_max_OI) / spot",
        units="dimensionless pct; negative = max OI strike below spot",
        expected_min=-0.1, expected_max=0.1,
        null_strategy="optional_sentinel", sentinel_value=0.0,
    ),

    # Missingness indicator for the options group (always computable)
    FeatureDef(
        name="is_null_options", version=1, group="options",
        description="1 when no options snapshot was available for this bar",
        formula="1 if options_data is None, else 0",
        units="binary {0, 1}",
        expected_min=0.0, expected_max=1.0,
        null_strategy="required",
    ),

    # -----------------------------------------------------------------------
    # MEAN REVERSION: EMA-distance and fast z-score
    # Distinct from the momentum group: these capture deviation from slow
    # trend anchors rather than raw directional returns.
    # -----------------------------------------------------------------------
    FeatureDef(
        name="ema_dist_20", version=1, group="mean_reversion",
        description="Signed distance of prior close from its 20-bar EMA, pct-normalised",
        formula="(close[i-1] - EMA(close, 20)[i-1]) / (close[i-1] + eps)",
        units="dimensionless pct; positive = above EMA20",
        expected_min=-0.05, expected_max=0.05,
        null_strategy="required",
    ),
    FeatureDef(
        name="ema_dist_50", version=1, group="mean_reversion",
        description="Signed distance of prior close from its 50-bar EMA, pct-normalised",
        formula="(close[i-1] - EMA(close, 50)[i-1]) / (close[i-1] + eps)",
        units="dimensionless pct; positive = above EMA50",
        expected_min=-0.1, expected_max=0.1,
        null_strategy="required",
    ),
    FeatureDef(
        name="zscore_5", version=1, group="mean_reversion",
        description="5-bar z-score of close — fast mean-reversion oscillator",
        formula="(close[i-1] - mean(close[i-6..i-1])) / (std(close[i-6..i-1]) + eps)",
        units="standard deviations",
        expected_min=-4.0, expected_max=4.0,
        null_strategy="required",
    ),

    # -----------------------------------------------------------------------
    # BAR STRUCTURE: candlestick anatomy of the prior bar
    # These use OHLC of bar i-1 to capture intrabar price structure:
    # body size, wick rejection strength, intrabar direction, and gap size.
    # All values reference bar i-1 exclusively (shift-by-1 invariant holds).
    # -----------------------------------------------------------------------
    FeatureDef(
        name="body_ratio", version=1, group="bar_structure",
        description="Body of prior bar as fraction of total range; 0 = doji, 1 = full body",
        formula="|close[i-1] - open[i-1]| / (high[i-1] - low[i-1] + eps)",
        units="dimensionless [0, 1]",
        expected_min=0.0, expected_max=1.0,
        null_strategy="required",
    ),
    FeatureDef(
        name="upper_wick_ratio", version=1, group="bar_structure",
        description="Upper wick of prior bar as fraction of total range — seller rejection above",
        formula="(high[i-1] - max(close[i-1], open[i-1])) / (high[i-1] - low[i-1] + eps)",
        units="dimensionless [0, 1]",
        expected_min=0.0, expected_max=1.0,
        null_strategy="required",
    ),
    FeatureDef(
        name="lower_wick_ratio", version=1, group="bar_structure",
        description="Lower wick of prior bar as fraction of total range — buyer rejection below",
        formula="(min(close[i-1], open[i-1]) - low[i-1]) / (high[i-1] - low[i-1] + eps)",
        units="dimensionless [0, 1]",
        expected_min=0.0, expected_max=1.0,
        null_strategy="required",
    ),
    FeatureDef(
        name="bar_return", version=1, group="bar_structure",
        description="Intrabar return of prior bar: open-to-close pct change",
        formula="close[i-1] / (open[i-1] + eps) - 1",
        units="pct return; positive = bullish bar",
        expected_min=-0.05, expected_max=0.05,
        null_strategy="required",
    ),
    FeatureDef(
        name="gap_pct", version=1, group="bar_structure",
        description="Gap at open of prior bar relative to the bar before it",
        formula="open[i-1] / (close[i-2] + eps) - 1",
        units="pct gap; positive = gap-up open",
        expected_min=-0.03, expected_max=0.03,
        null_strategy="required",
    ),

    # -----------------------------------------------------------------------
    # REGIME: trend-direction binary signals derived from EMA relationships
    # These complement vol_regime (ratio of short/long realized vol) with
    # price-trend regime indicators.
    # -----------------------------------------------------------------------
    FeatureDef(
        name="price_above_ema20", version=1, group="regime",
        description="1 when prior close is above its 20-bar EMA; 0 otherwise",
        formula="1 if close[i-1] > EMA(close, 20)[i-1] else 0",
        units="binary {0, 1}",
        expected_min=0.0, expected_max=1.0,
        null_strategy="required",
    ),
    FeatureDef(
        name="ma_alignment", version=1, group="regime",
        description="1 when EMA20 > EMA50 (bullish MA alignment); 0 otherwise",
        formula="1 if EMA(close, 20)[i-1] > EMA(close, 50)[i-1] else 0",
        units="binary {0, 1}",
        expected_min=0.0, expected_max=1.0,
        null_strategy="required",
    ),
]

# Build the registry dict keyed by feature name
REGISTRY: Dict[str, FeatureDef] = {f.name: f for f in _RAW}

# Ordered feature group lists for documentation and selective computation
FEATURE_GROUPS: Dict[str, List[str]] = {}
for _f in _RAW:
    FEATURE_GROUPS.setdefault(_f.group, []).append(_f.name)

# ---------------------------------------------------------------------------
# Canonical feature column lists
# ---------------------------------------------------------------------------

# Core features: always computable from OHLCV data — used as model inputs
FEATURE_COLS: List[str] = [
    # trend
    "rsi_14", "rsi_5",
    "macd_line", "macd_signal", "macd_hist",
    "bb_pct",
    # volatility
    "atr_norm",
    "realized_vol_5", "realized_vol_10", "realized_vol_20",
    "vol_regime",
    # momentum (directional returns)
    "ret_1", "ret_5", "ret_10", "ret_20", "ret_60",
    "zscore_20",
    # mean reversion (EMA-distance and fast z-score)
    "ema_dist_20", "ema_dist_50",
    "zscore_5",
    # vwap
    "vwap_distance",
    "vwap_slope",
    # volume
    "volume_ratio", "volume_trend", "volume_zscore",
    # intraday seasonality
    "hour_sin", "hour_cos",
    "minute_sin", "minute_cos",
    "session_progress",
    "is_first_30min", "is_last_30min",
    # bar structure (candlestick anatomy of the prior bar)
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
    "bar_return", "gap_pct",
    # regime (trend-direction binary signals)
    "price_above_ema20", "ma_alignment",
    # options missingness indicator (always present, informs model when options absent)
    "is_null_options",
]

# Options-derived features: sentinel-filled when absent
OPTIONS_FEATURE_COLS: List[str] = [
    "atm_iv", "iv_rank", "iv_skew",
    "pc_volume_ratio", "pc_oi_ratio",
    "gex_proxy", "dist_to_max_oi",
]

# All features including options and raw ATR (for storage / inspection)
ALL_FEATURE_COLS: List[str] = FEATURE_COLS + OPTIONS_FEATURE_COLS + ["atr_14"]

# Verify every FEATURE_COLS name is in REGISTRY
_missing = [n for n in ALL_FEATURE_COLS if n not in REGISTRY]
assert not _missing, f"Unregistered features: {_missing}"

# ---------------------------------------------------------------------------
# Manifest hash — changes whenever any (name, version) pair changes
# ---------------------------------------------------------------------------

def _compute_manifest_hash() -> str:
    pairs = sorted((f.name, f.version) for f in REGISTRY.values())
    payload = json.dumps(pairs, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


MANIFEST_HASH: str = _compute_manifest_hash()
