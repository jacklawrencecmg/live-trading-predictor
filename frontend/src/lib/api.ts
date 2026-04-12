import axios from "axios";

// Use relative URLs — Next.js rewrites /api/* and /ws/* to the backend.
// This works in local docker-compose, Codespaces, and any deployment
// without needing to know the backend's URL at build time.
const api = axios.create({ baseURL: "" });

export interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface MarketQuote {
  symbol: string;
  price: number;
  bid: number;
  ask: number;
  volume: number;
  change: number;
  change_pct: number;
}

export interface OptionContract {
  symbol: string;
  strike: number;
  expiry: string;
  option_type: string;
  bid: number;
  ask: number;
  mid: number;
  last: number;
  volume: number;
  open_interest: number;
  iv: number;
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  rho: number;
  in_the_money: boolean;
  intrinsic_value: number;
  time_value: number;
}

export interface OptionsChainRow {
  strike: number;
  call: OptionContract | null;
  put: OptionContract | null;
}

export interface OptionsChain {
  symbol: string;
  underlying_price: number;
  expiry: string;
  rows: OptionsChainRow[];
  iv_rank: number;
  put_call_ratio: number;
  atm_iv: number;
}

export interface ModelPrediction {
  symbol: string;
  timestamp: number;
  prob_up: number;
  prob_down: number;
  prob_flat: number;
  expected_move_pct: number;
  confidence: number;
  trade_signal: string;
  model_version: string;
}

export interface ReliabilityDiagram {
  bins: number[];
  mean_predicted: number[];
  fraction_positive: number[];
}

export interface PredictionBundle {
  // Layer 1: raw
  prob_up: number;
  prob_down: number;
  // Layer 2: calibrated
  calibrated_prob_up: number;
  calibrated_prob_down: number;
  calibration_available: boolean;
  // Layer 3: tradeable
  tradeable_confidence: number;
  degradation_factor: number;
  // Layer 4: action
  action: string;
  abstain_reason: string | null;
  // Uncertainty context
  confidence_band: [number, number];
  calibration_health: "good" | "fair" | "degraded" | "unknown";
  rolling_brier: number | null;
  ece_recent: number | null;
  reliability_diagram: ReliabilityDiagram | null;
  // Other
  expected_move_pct: number;
  model_version: string;
  bar_open_time: string;
  feature_snapshot_id: string;
  // Backward compat
  confidence: number;
}

export interface ScoredSignal {
  direction: string;
  raw_probability: number;
  probability: number;
  tradeable_confidence: number;
  confidence: number;
  confidence_band: [number, number];
  degradation_factor: number;
  calibration_health: "good" | "fair" | "degraded" | "unknown";
  calibration_available: boolean;
  ece_recent: number | null;
  rolling_brier: number | null;
  abstain_reason: string | null;
  signal_quality_score: number;
  volatility_context: string;
  regime: string;
  explanation: string;
  top_features: Record<string, number>;
}

export interface TradeIdea {
  direction: string;
  strategy: string;
  target_delta: number;
  blocked: boolean;
  block_reason: string | null;
  rationale: string;
}

export interface SignalResponse {
  symbol: string;
  prediction: PredictionBundle;
  signal: ScoredSignal;
  trade_idea: TradeIdea;
}

export interface UncertaintyStats {
  symbol: string;
  window_size: number;
  calibration_health: "good" | "fair" | "degraded" | "unknown";
  degradation_factor: number;
  rolling_brier: number | null;
  baseline_brier: number | null;
  ece_recent: number | null;
  reliability_diagram: ReliabilityDiagram | null;
}

export interface PortfolioSummary {
  capital: number;
  cash: number;
  positions_value: number;
  total_value: number;
  daily_pnl: number;
  daily_pnl_pct: number;
  total_pnl: number;
  open_positions: number;
  kill_switch_active: boolean;
}

export interface Position {
  id: number;
  symbol: string;
  option_symbol: string | null;
  quantity: number;
  avg_cost: number;
  current_price: number | null;
  unrealized_pnl: number;
  realized_pnl: number;
  is_open: boolean;
}

export interface Trade {
  id: number;
  symbol: string;
  option_symbol: string | null;
  action: string;
  quantity: number;
  price: number;
  executed_at: string;
  model_prob_up: number | null;
  model_prob_down: number | null;
  model_confidence: number | null;
}

export interface BacktestRequest {
  symbol: string;
  interval: string;
  period: string;
  n_folds: number;
  train_size: number;
  test_size: number;
  confidence_threshold: number;
}

export interface BacktestResult {
  id: number;
  symbol: string;
  interval: string;
  start_date: string;
  end_date: string;
  n_folds: number;
  accuracy: number | null;
  brier_score: number | null;
  log_loss: number | null;
  magnitude_mae: number | null;
  sharpe_ratio: number | null;
  total_return: number | null;
  n_trades: number | null;
  fold_results: any[] | null;
  calibration_data: any | null;
}

export interface RiskSummary {
  capital: number;
  daily_pnl: number;
  daily_pnl_pct: number;
  max_daily_loss: number;
  max_position_size: number;
  kill_switch: boolean;
  cooldown_minutes: number;
}

// Market
export const getCandles = (symbol: string, interval = "5m", period = "5d") =>
  api.get<{ symbol: string; interval: string; candles: Candle[] }>(`/api/market/candles/${symbol}`, {
    params: { interval, period },
  });

export const getQuote = (symbol: string) =>
  api.get<MarketQuote>(`/api/market/quote/${symbol}`);

// Options
export const getOptionsChain = (symbol: string, expiry?: string) =>
  api.get<OptionsChain>(`/api/options/chain/${symbol}`, {
    params: expiry ? { expiry } : {},
  });

export const getExpirations = (symbol: string) =>
  api.get<string[]>(`/api/options/expirations/${symbol}`);

// Model
export const getPrediction = (symbol: string, interval = "5m", confidence_threshold = 0.55) =>
  api.get<ModelPrediction>(`/api/model/predict/${symbol}`, {
    params: { interval, confidence_threshold },
  });

// Trades
export const executeTrade = (trade: {
  symbol: string;
  action: string;
  quantity: number;
  price?: number;
  option_symbol?: string;
  strike?: number;
  expiry?: string;
  option_type?: string;
}) => api.post<Trade>("/api/trades/execute", trade);

export const getPositions = () => api.get<Position[]>("/api/trades/positions");
export const getTradeHistory = (limit = 50) =>
  api.get<Trade[]>(`/api/trades/history?limit=${limit}`);
export const getPortfolio = () => api.get<PortfolioSummary>("/api/trades/portfolio");
export const getRiskSummary = () => api.get<RiskSummary>("/api/trades/risk");
export const toggleKillSwitch = (active: boolean) =>
  api.post("/api/trades/kill-switch", null, { params: { active } });

// Backtest
export const runBacktest = (req: BacktestRequest) =>
  api.post<BacktestResult>("/api/backtest/run", req);
export const getBacktestResults = () => api.get<BacktestResult[]>("/api/backtest/results");

// Signals
export const getSignal = (
  symbol: string,
  timeframe = "5m",
  confidence_threshold = 0.55,
  min_signal_quality = 40.0,
) =>
  api.get<SignalResponse>(`/api/signals/${symbol}`, {
    params: { timeframe, confidence_threshold, min_signal_quality },
  });

// Uncertainty
export const getUncertainty = (symbol: string) =>
  api.get<UncertaintyStats>(`/api/uncertainty/${symbol}`);

export const recordOutcome = (
  symbol: string,
  calibrated_prob: number,
  actual_outcome: 0 | 1,
  baseline_brier?: number,
) =>
  api.post(`/api/uncertainty/${symbol}/record`, {
    calibrated_prob,
    actual_outcome,
    baseline_brier,
  });

// Regime
export type RegimeName =
  | "trending_up"
  | "trending_down"
  | "mean_reverting"
  | "high_volatility"
  | "low_volatility"
  | "liquidity_poor"
  | "event_risk"
  | "unknown";

export interface RegimeSignals {
  adx_proxy: number;
  atr_ratio: number;
  volume_ratio: number;
  bar_range_ratio: number;
  trend_direction: "up" | "down" | "flat";
  ema_spread_pct: number;
  is_abnormal_move: boolean;
  abnormal_move_sigma: number;
}

export interface RegimeContext {
  symbol: string;
  timeframe: string;
  regime: RegimeName;
  description: string;
  suppressed: boolean;
  suppress_reason: string | null;
  confidence_threshold: number;
  min_signal_quality: number;
  signals: RegimeSignals;
  thresholds: Record<RegimeName, { confidence_threshold: number; min_signal_quality: number; allow_trade: boolean }>;
}

export interface RegimeHistoryEntry {
  bar_open_time: string;
  regime: RegimeName;
  adx_proxy: number | null;
  atr_ratio: number | null;
  volume_ratio: number | null;
  is_abnormal_move: boolean | null;
  suppressed: boolean | null;
}

export interface RegimeHistory {
  symbol: string;
  timeframe: string;
  count: number;
  history: RegimeHistoryEntry[];
}

export interface RegimeDistribution {
  symbol: string;
  timeframe: string;
  distribution: Partial<Record<RegimeName, number>>;
  suppressed_regimes: RegimeName[];
  descriptions: Record<RegimeName, string>;
}

export const getRegime = (symbol: string, timeframe = "5m") =>
  api.get<RegimeContext>(`/api/regime/${symbol}`, { params: { timeframe } });

export const getRegimeHistory = (symbol: string, timeframe = "5m", limit = 100) =>
  api.get<RegimeHistory>(`/api/regime/${symbol}/history`, { params: { timeframe, limit } });

export const getRegimeDistribution = (symbol: string, timeframe = "5m") =>
  api.get<RegimeDistribution>(`/api/regime/${symbol}/distribution`, { params: { timeframe } });

// ---------------------------------------------------------------------------
// Decision layer types
// ---------------------------------------------------------------------------

export interface IVAnalysis {
  atm_iv: number;
  realized_vol_ann: number;
  iv_rank: number;
  iv_rv_ratio: number;
  iv_vs_rv: "cheap" | "fair" | "rich";
  iv_implied_1d_move_pct: number;
  rv_implied_1d_move_pct: number;
}

export interface StructureLeg {
  action: "buy" | "sell";
  option_type: "call" | "put";
  target_delta: number;
  strike: number | null;
  expiry: string | null;
  estimated_mid: number | null;
  estimated_iv: number | null;
  bid: number | null;
  ask: number | null;
}

export interface StructureCandidate {
  structure_type: "long_call" | "long_put" | "debit_spread" | "credit_spread";
  direction: "bullish" | "bearish";
  score: number;
  viable: boolean;
  legs: StructureLeg[];
  estimated_cost_pct: number;
  estimated_credit_pct: number;
  max_profit_pct: number;
  max_loss_pct: number;
  breakeven_move_pct: number;
  spread_width_pct: number | null;
  iv_edge: "favorable" | "neutral" | "unfavorable";
  iv_edge_score: number;
  liquidity_fit: "good" | "fair" | "poor" | "unknown";
  estimated_fill_cost_pct: number;
  horizon_note: string;
  rationale: string;
  tailwinds: string[];
  concerns: string[];
}

export interface OptionsDecision {
  symbol: string;
  generated_at: string;
  spot_price: number;
  direction_thesis: "bullish" | "bearish" | "neutral" | "abstain";
  horizon: string;
  calibrated_prob: number;
  prob_up: number;
  prob_down: number;
  confidence_band: [number, number];
  expected_move_1bar_pct: number;
  expected_move_1d_pct: number;
  expected_range_low: number;
  expected_range_high: number;
  iv_analysis: IVAnalysis;
  expiry: string;
  dte: number;
  liquidity_quality: "good" | "fair" | "poor";
  atm_bid_ask_pct: number;
  regime: string;
  regime_suppressed: boolean;
  calibration_health: "good" | "fair" | "degraded" | "unknown";
  signal_quality_score: number;
  confidence_score: number;
  abstain: boolean;
  abstain_reason: string | null;
  candidates: StructureCandidate[];
  recommended_structure: string | null;
  recommendation_rationale: string;
}

export const getDecision = (
  symbol: string,
  params?: {
    timeframe?: string;
    confidence_threshold?: number;
    atm_iv?: number;
    iv_rank?: number;
    dte?: number;
    liquidity_quality?: string;
    atm_bid_ask_pct?: number;
  }
) =>
  api.get<OptionsDecision>(`/api/decision/${symbol}`, { params });

// ---------------------------------------------------------------------------
// Signal history (forecast vs realized comparison)
// ---------------------------------------------------------------------------

export interface SignalHistoryEntry {
  id: string | number;
  symbol: string;
  bar_open_time: string;
  direction: "bullish" | "bearish" | "neutral" | "abstain";
  calibrated_prob: number;
  confidence_score: number | null;
  regime: string;
  action: string;
  abstain_reason: string | null;
  // Outcome fields — null if outcome not yet recorded
  actual_outcome: "up" | "down" | "flat" | null;
  outcome_pct: number | null;
  correct: boolean | null;
}

export const getSignalHistory = (symbol: string, limit = 20) =>
  api.get<SignalHistoryEntry[]>(`/api/signals/${symbol}/history`, { params: { limit } });

// ---------------------------------------------------------------------------
// P&L summary
// ---------------------------------------------------------------------------

export interface PnLSummary {
  daily_realized: number;
  daily_unrealized: number;
  rolling_7d: number;
  rolling_30d: number;
  win_rate_30d: number | null;
  trades_30d: number;
  avg_win: number | null;
  avg_loss: number | null;
  sharpe_7d: number | null;
}

export const getPnLSummary = () =>
  api.get<PnLSummary>("/api/trades/pnl-summary");
