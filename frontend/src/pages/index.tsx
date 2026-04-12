"use client";
import { useState, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";
import { useWebSocket, WSMessage } from "@/hooks/useWebSocket";
import { useMarketData } from "@/hooks/useMarketData";
import { Candle } from "@/lib/api";
import OptionsChainTable from "@/components/OptionsChain/OptionsChainTable";
import ModelPanel from "@/components/ModelPanel/ModelPanel";
import PaperTrader from "@/components/PaperTrader/PaperTrader";
import RiskPanel from "@/components/RiskPanel/RiskPanel";
import BacktestPanel from "@/components/Backtest/BacktestPanel";
import clsx from "clsx";

const CandlestickChart = dynamic(() => import("@/components/Chart/CandlestickChart"), { ssr: false });

const SYMBOLS = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "AMZN", "MSFT", "META"];

function Panel({ title, children, className }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={clsx("bg-panel border border-border rounded-lg p-3", className)}>
      <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">{title}</h2>
      {children}
    </div>
  );
}

export default function Dashboard() {
  const [symbol, setSymbol] = useState("SPY");
  const [symbolInput, setSymbolInput] = useState("SPY");
  const [livePrice, setLivePrice] = useState<number | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [activeTab, setActiveTab] = useState<"chart" | "backtest">("chart");

  const { quote, candles, loading, refresh } = useMarketData(symbol);
  const [liveCandles, setLiveCandles] = useState<Candle[]>([]);

  useEffect(() => {
    setLiveCandles(candles);
  }, [candles]);

  const handleWsMessage = useCallback((msg: WSMessage) => {
    if (msg.type === "quote") {
      setLivePrice(msg.price);
    }
    if (msg.type === "candle" && msg.candle) {
      setLiveCandles((prev) => {
        const idx = prev.findIndex((c) => c.time === msg.candle.time);
        if (idx >= 0) {
          const next = [...prev];
          next[idx] = msg.candle;
          return next;
        }
        return [...prev, msg.candle];
      });
    }
  }, []);

  const { connected } = useWebSocket(symbol, handleWsMessage);

  const currentPrice = livePrice ?? quote?.price;

  const handleSymbolSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSymbol(symbolInput.toUpperCase().trim());
  };

  return (
    <div className="min-h-screen bg-surface flex flex-col">
      {/* Top bar */}
      <header className="bg-panel border-b border-border px-4 py-2 flex items-center gap-4">
        <span className="text-white font-bold text-base tracking-tight">Options Research</span>

        {/* Symbol selector */}
        <div className="flex gap-1">
          {SYMBOLS.map((s) => (
            <button
              key={s}
              onClick={() => { setSymbol(s); setSymbolInput(s); }}
              className={clsx(
                "text-xs px-2 py-0.5 rounded",
                symbol === s ? "bg-accent text-surface font-semibold" : "text-gray-400 hover:text-white"
              )}
            >
              {s}
            </button>
          ))}
        </div>

        {/* Custom symbol */}
        <form onSubmit={handleSymbolSubmit} className="flex gap-1">
          <input
            value={symbolInput}
            onChange={(e) => setSymbolInput(e.target.value.toUpperCase())}
            placeholder="Symbol"
            className="bg-surface border border-border text-white text-xs px-2 py-0.5 rounded w-20 uppercase"
          />
          <button type="submit" className="text-xs text-accent hover:underline">Go</button>
        </form>

        <div className="ml-auto flex items-center gap-4 text-xs">
          {currentPrice && (
            <span className="text-white font-semibold text-sm">${currentPrice.toFixed(2)}</span>
          )}
          {quote && (
            <span className={clsx(quote.change >= 0 ? "text-green-trade" : "text-red-trade")}>
              {quote.change >= 0 ? "+" : ""}{quote.change.toFixed(2)} ({quote.change_pct.toFixed(2)}%)
            </span>
          )}
          <span className={clsx("flex items-center gap-1", connected ? "text-green-trade" : "text-gray-500")}>
            <span className={clsx("w-1.5 h-1.5 rounded-full", connected ? "bg-green-trade" : "bg-gray-600")} />
            {connected ? "Live" : "Disconnected"}
          </span>
          <span className="text-gray-600">{symbol}</span>
        </div>
      </header>

      {/* Main layout */}
      <div className="flex flex-1 overflow-hidden gap-2 p-2">
        {/* Left column: chart + options chain */}
        <div className="flex flex-col gap-2 flex-1 min-w-0">
          {/* Tabs */}
          <div className="flex gap-2">
            {(["chart", "backtest"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setActiveTab(t)}
                className={clsx(
                  "text-xs px-3 py-1 rounded capitalize",
                  activeTab === t ? "bg-accent text-surface font-semibold" : "text-gray-400 hover:text-white"
                )}
              >
                {t}
              </button>
            ))}
          </div>

          {activeTab === "chart" ? (
            <>
              <Panel title={`${symbol} — 5m Candlestick`} className="flex-none">
                {loading && <div className="text-gray-500 text-xs">Loading...</div>}
                <CandlestickChart candles={liveCandles} symbol={symbol} height={380} />
              </Panel>

              <Panel title="Options Chain" className="flex-1 min-h-0 overflow-hidden">
                <OptionsChainTable symbol={symbol} />
              </Panel>
            </>
          ) : (
            <Panel title="Walk-Forward Backtest" className="flex-1">
              <BacktestPanel symbol={symbol} />
            </Panel>
          )}
        </div>

        {/* Right column: model + trader + risk */}
        <div className="flex flex-col gap-2 w-80 flex-none">
          <Panel title="Model Predictions">
            <ModelPanel symbol={symbol} />
          </Panel>

          <Panel title="Paper Trader" className="flex-1">
            <PaperTrader symbol={symbol} currentPrice={currentPrice} />
          </Panel>

          <Panel title="Risk Controls">
            <RiskPanel />
          </Panel>
        </div>
      </div>

      {/* Status bar */}
      <footer className="bg-panel border-t border-border px-4 py-1 flex items-center gap-4 text-xs text-gray-600">
        <span>Paper trading only — no live orders</span>
        <span>·</span>
        <span>Data: yfinance (15-min delayed)</span>
        <span>·</span>
        <span>Model: LogisticRegression baseline</span>
        <span className="ml-auto">Options Research Platform v1.0</span>
      </footer>
    </div>
  );
}
