import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const api = axios.create({ baseURL: API_URL });

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
