"use client";
import { useState, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";
import { useWebSocket, WSMessage } from "@/hooks/useWebSocket";
import { useMarketData } from "@/hooks/useMarketData";
import { Candle } from "@/lib/api";
import OptionsChainTable from "@/components/OptionsChain/OptionsChainTable";
import RegimePanel from "@/components/RegimePanel/RegimePanel";
import PaperTrader from "@/components/PaperTrader/PaperTrader";
import RiskPanel from "@/components/RiskPanel/RiskPanel";

// New institutional panels
import PredictionPanel from "@/components/Dashboard/PredictionPanel";
import ConfidencePanel from "@/components/Dashboard/ConfidencePanel";
import ModelHealthPanel from "@/components/Dashboard/ModelHealthPanel";
import DataFreshnessPanel from "@/components/Dashboard/DataFreshnessPanel";
import LiquidityPanel from "@/components/Dashboard/LiquidityPanel";
import RecentSignalsTable from "@/components/Dashboard/RecentSignalsTable";
import PnLPanel from "@/components/Dashboard/PnLPanel";
import StructuresPanel from "@/components/Dashboard/StructuresPanel";

import clsx from "clsx";

const CandlestickChart = dynamic(() => import("@/components/Chart/CandlestickChart"), {
  ssr: false,
});

const SYMBOLS = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "AMZN"];

// Collapsible section header for panels that may want to be hidden
function SectionHeader({ title, stale }: { title: string; stale?: boolean }) {
  return (
    <div className="px-3 py-1.5 border-b border-border flex items-center justify-between">
      <span className="text-[10px] font-semibold tracking-[0.12em] uppercase text-zinc-500">
        {title}
      </span>
      {stale && (
        <span className="text-[10px] text-amber-400 border border-amber-400/30 px-1 rounded-[2px]">
          STALE
        </span>
      )}
    </div>
  );
}


export default function Dashboard() {
  const [symbol, setSymbol] = useState("SPY");
  const [symbolInput, setSymbolInput] = useState("SPY");
  const [livePrice, setLivePrice] = useState<number | null>(null);
  const [showBacktest, setShowBacktest] = useState(false);

  const { quote, candles, loading: candlesLoading } = useMarketData(symbol);
  const [liveCandles, setLiveCandles] = useState<Candle[]>([]);

  useEffect(() => { setLiveCandles(candles); }, [candles]);

  const handleWsMessage = useCallback((msg: WSMessage) => {
    if (msg.type === "quote") setLivePrice(msg.price);
    if (msg.type === "candle" && msg.candle) {
      setLiveCandles((prev) => {
        const idx = prev.findIndex((c) => c.time === msg.candle.time);
        if (idx >= 0) { const next = [...prev]; next[idx] = msg.candle; return next; }
        return [...prev, msg.candle];
      });
    }
  }, []);

  const { connected } = useWebSocket(symbol, handleWsMessage);
  const currentPrice = livePrice ?? quote?.price;

  const handleSymbolSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const s = symbolInput.toUpperCase().trim();
    if (s) setSymbol(s);
  };

  return (
    <div className="min-h-screen bg-surface flex flex-col text-zinc-200">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header className="bg-panel border-b border-border px-4 py-2 flex items-center gap-3 flex-wrap shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-zinc-200 font-semibold text-sm tracking-tight">
            Options Research
          </span>
          <span className="text-[10px] text-zinc-600 border border-zinc-800 rounded-[2px] px-1.5 py-0.5 font-mono">
            PAPER ONLY
          </span>
        </div>

        {/* Symbol quick-picks */}
        <div className="flex gap-1 flex-wrap">
          {SYMBOLS.map((s) => (
            <button
              key={s}
              onClick={() => { setSymbol(s); setSymbolInput(s); }}
              className={clsx(
                "text-xs px-2 py-0.5 rounded-[2px] font-mono",
                symbol === s
                  ? "bg-accent/15 text-accent border border-accent/30 font-semibold"
                  : "text-zinc-500 hover:text-zinc-200 border border-transparent"
              )}
            >
              {s}
            </button>
          ))}
        </div>

        {/* Custom symbol input */}
        <form onSubmit={handleSymbolSubmit} className="flex gap-1">
          <input
            value={symbolInput}
            onChange={(e) => setSymbolInput(e.target.value.toUpperCase())}
            placeholder="TICKER"
            className="bg-surface border border-border text-zinc-200 text-xs px-2 py-0.5 rounded-[2px] w-20 uppercase font-mono placeholder-zinc-700 focus:border-border-2 focus:outline-none"
          />
          <button type="submit" className="text-xs text-accent hover:text-blue-300 font-mono">
            Go
          </button>
        </form>

        {/* Price and connection status */}
        <div className="ml-auto flex items-center gap-4 text-xs">
          {currentPrice != null && (
            <span className="text-zinc-100 font-mono font-semibold text-sm tabular-nums">
              ${currentPrice.toFixed(2)}
            </span>
          )}
          {quote && (
            <span className={clsx(
              "font-mono tabular-nums",
              quote.change >= 0 ? "text-emerald-400" : "text-red-400"
            )}>
              {quote.change >= 0 ? "+" : ""}{quote.change.toFixed(2)}
              {" "}({quote.change >= 0 ? "+" : ""}{quote.change_pct.toFixed(2)}%)
            </span>
          )}
          <div className={clsx(
            "flex items-center gap-1.5 text-[11px]",
            connected ? "text-emerald-400" : "text-zinc-600"
          )}>
            <span className={clsx("w-1.5 h-1.5 rounded-full", connected ? "bg-emerald-400" : "bg-zinc-700")} />
            <span>{connected ? "Live" : "Offline"}</span>
          </div>
        </div>
      </header>

      {/* ── 3-Column layout ──────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden gap-1.5 p-1.5">

        {/* ──── LEFT: Chart + Options chain + Recent signals ───────────── */}
        <div className="flex flex-col gap-1.5 overflow-hidden" style={{ width: "42%" }}>

          {/* Chart */}
          <div className="inst-panel flex-none">
            <div className="px-3 py-1.5 border-b border-border flex items-center justify-between">
              <span className="inst-label">{symbol} — 5m</span>
              <div className="flex items-center gap-2">
                {candlesLoading && <span className="text-zinc-700 text-[10px]">loading…</span>}
                <button
                  onClick={() => setShowBacktest((v) => !v)}
                  className={clsx(
                    "text-[10px] px-2 py-0.5 rounded-[2px] border",
                    showBacktest
                      ? "text-accent border-accent/30 bg-accent/5"
                      : "text-zinc-600 border-zinc-800 hover:text-zinc-400"
                  )}
                >
                  Backtest
                </button>
              </div>
            </div>
            <div className="p-2">
              {showBacktest ? (
                // Lazy-load backtest panel to avoid cluttering the default view
                <div className="text-zinc-500 text-[11px] p-2">
                  Backtest panel hidden. Import BacktestPanel to enable.
                </div>
              ) : (
                <CandlestickChart candles={liveCandles} symbol={symbol} height={320} />
              )}
            </div>
          </div>

          {/* Options chain */}
          <div className="inst-panel flex-none">
            <SectionHeader title="Options Chain" />
            <div className="overflow-hidden">
              <OptionsChainTable symbol={symbol} />
            </div>
          </div>

          {/* Recent signals table */}
          <div className="flex-1 min-h-0 overflow-auto">
            <RecentSignalsTable symbol={symbol} />
          </div>
        </div>

        {/* ──── MIDDLE: Prediction + Confidence + Structures + Liquidity ─ */}
        <div
          className="flex flex-col gap-1.5 overflow-y-auto overflow-x-hidden"
          style={{ width: "29%" }}
        >
          <PredictionPanel symbol={symbol} />
          <ConfidencePanel symbol={symbol} />
          <StructuresPanel symbol={symbol} />
          <LiquidityPanel symbol={symbol} />
        </div>

        {/* ──── RIGHT: Regime + Model health + Freshness + PnL + Positions */}
        <div
          className="flex flex-col gap-1.5 overflow-y-auto overflow-x-hidden"
          style={{ width: "29%" }}
        >
          <RegimePanel symbol={symbol} />
          <ModelHealthPanel symbol={symbol} />
          <DataFreshnessPanel symbol={symbol} connected={connected} />
          <PnLPanel />
          <PaperTrader symbol={symbol} currentPrice={currentPrice} />
          <RiskPanel />
        </div>
      </div>

      {/* ── Footer ──────────────────────────────────────────────────────── */}
      <footer className="bg-panel border-t border-border px-4 py-1 flex items-center gap-3 shrink-0">
        <span className="text-[10px] font-semibold text-red-400/60 font-mono">PAPER TRADING ONLY</span>
        <span className="text-zinc-800">·</span>
        <span className="text-[10px] text-zinc-700">Data: yfinance (delayed)</span>
        <span className="text-zinc-800">·</span>
        <span className="text-[10px] text-zinc-700">
          Probabilities are model outputs — not investment advice
        </span>
        <span className="ml-auto text-[10px] text-zinc-700">
          <a
            href="https://github.com/jacklawrencecmg/live-trading-predictor"
            target="_blank"
            rel="noreferrer"
            className="hover:text-zinc-400"
          >
            GitHub
          </a>
        </span>
      </footer>
    </div>
  );
}
