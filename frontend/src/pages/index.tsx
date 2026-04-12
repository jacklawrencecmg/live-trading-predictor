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
import SignalCard from "@/components/SignalCard/SignalCard";
import clsx from "clsx";

const CandlestickChart = dynamic(() => import("@/components/Chart/CandlestickChart"), {
  ssr: false,
});

const SYMBOLS = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "AMZN"];

function Panel({
  title,
  children,
  className,
}: {
  title: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={clsx("bg-panel border border-border rounded-lg p-3", className)}>
      <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
        {title}
      </h2>
      {children}
    </div>
  );
}

type TabId = "chart" | "backtest";

export default function Dashboard() {
  const [symbol, setSymbol] = useState("SPY");
  const [symbolInput, setSymbolInput] = useState("SPY");
  const [livePrice, setLivePrice] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>("chart");
  const [rightTab, setRightTab] = useState<"signal" | "model" | "trader" | "risk">("signal");

  const { quote, candles, loading } = useMarketData(symbol);
  const [liveCandles, setLiveCandles] = useState<Candle[]>([]);

  useEffect(() => {
    setLiveCandles(candles);
  }, [candles]);

  const handleWsMessage = useCallback((msg: WSMessage) => {
    if (msg.type === "quote") setLivePrice(msg.price);
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
    const s = symbolInput.toUpperCase().trim();
    if (s) setSymbol(s);
  };

  return (
    <div className="min-h-screen bg-surface flex flex-col">
      {/* Header */}
      <header className="bg-panel border-b border-border px-4 py-2 flex items-center gap-3 flex-wrap">
        <span className="text-white font-bold text-sm tracking-tight whitespace-nowrap">
          Live Trading Predictor
        </span>
        <span className="text-xs text-gray-600 border border-border rounded px-1.5 py-0.5">
          PAPER ONLY
        </span>

        <div className="flex gap-1 flex-wrap">
          {SYMBOLS.map((s) => (
            <button
              key={s}
              onClick={() => {
                setSymbol(s);
                setSymbolInput(s);
              }}
              className={clsx(
                "text-xs px-2 py-0.5 rounded",
                symbol === s
                  ? "bg-accent text-surface font-semibold"
                  : "text-gray-400 hover:text-white"
              )}
            >
              {s}
            </button>
          ))}
        </div>

        <form onSubmit={handleSymbolSubmit} className="flex gap-1">
          <input
            value={symbolInput}
            onChange={(e) => setSymbolInput(e.target.value.toUpperCase())}
            placeholder="Symbol"
            className="bg-surface border border-border text-white text-xs px-2 py-0.5 rounded w-20 uppercase"
          />
          <button type="submit" className="text-xs text-accent hover:underline">
            Go
          </button>
        </form>

        <div className="ml-auto flex items-center gap-3 text-xs">
          {currentPrice && (
            <span className="text-white font-semibold">${currentPrice.toFixed(2)}</span>
          )}
          {quote && (
            <span
              className={clsx(
                quote.change >= 0 ? "text-green-trade" : "text-red-trade"
              )}
            >
              {quote.change >= 0 ? "+" : ""}
              {quote.change.toFixed(2)} ({quote.change_pct.toFixed(2)}%)
            </span>
          )}
          <span
            className={clsx(
              "flex items-center gap-1",
              connected ? "text-green-trade" : "text-gray-500"
            )}
          >
            <span
              className={clsx(
                "w-1.5 h-1.5 rounded-full",
                connected ? "bg-green-trade" : "bg-gray-600"
              )}
            />
            {connected ? "Live" : "Offline"}
          </span>
        </div>
      </header>

      {/* Main layout: left (chart/options) + right (signal/model/trader/risk) */}
      <div className="flex flex-1 overflow-hidden gap-2 p-2">
        {/* Left column */}
        <div className="flex flex-col gap-2 flex-1 min-w-0">
          <div className="flex gap-2">
            {(["chart", "backtest"] as TabId[]).map((t) => (
              <button
                key={t}
                onClick={() => setActiveTab(t)}
                className={clsx(
                  "text-xs px-3 py-1 rounded capitalize",
                  activeTab === t
                    ? "bg-accent text-surface font-semibold"
                    : "text-gray-400 hover:text-white"
                )}
              >
                {t}
              </button>
            ))}
          </div>

          {activeTab === "chart" ? (
            <>
              <Panel title={`${symbol} — 5m`} className="flex-none">
                {loading && (
                  <div className="text-gray-500 text-xs">Loading...</div>
                )}
                <CandlestickChart
                  candles={liveCandles}
                  symbol={symbol}
                  height={360}
                />
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

        {/* Right column */}
        <div className="flex flex-col gap-2 w-80 flex-none">
          {/* Right panel tabs */}
          <div className="flex gap-1 bg-panel border border-border rounded-lg p-1">
            {(
              [
                { id: "signal", label: "Signal" },
                { id: "model", label: "Model" },
                { id: "trader", label: "Trade" },
                { id: "risk", label: "Risk" },
              ] as const
            ).map(({ id, label }) => (
              <button
                key={id}
                onClick={() => setRightTab(id)}
                className={clsx(
                  "flex-1 text-xs py-1 rounded",
                  rightTab === id
                    ? "bg-accent/20 text-accent font-semibold"
                    : "text-gray-500 hover:text-white"
                )}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="flex-1 overflow-auto">
            {rightTab === "signal" && (
              <Panel title="Signal & Trade Idea">
                <SignalCard symbol={symbol} />
              </Panel>
            )}
            {rightTab === "model" && (
              <Panel title="Model Predictions">
                <ModelPanel symbol={symbol} />
              </Panel>
            )}
            {rightTab === "trader" && (
              <Panel title="Paper Trader" className="flex-1">
                <PaperTrader symbol={symbol} currentPrice={currentPrice} />
              </Panel>
            )}
            {rightTab === "risk" && (
              <Panel title="Risk Controls">
                <RiskPanel />
              </Panel>
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-panel border-t border-border px-4 py-1 flex items-center gap-3 text-xs text-gray-600">
        <span className="text-red-trade/70 font-semibold">PAPER TRADING ONLY</span>
        <span>·</span>
        <span>Data: yfinance (delayed)</span>
        <span>·</span>
        <span>Model: LogisticRegression baseline</span>
        <span className="ml-auto">
          <a
            href="https://github.com/jacklawrencecmg/live-trading-predictor"
            target="_blank"
            rel="noreferrer"
            className="hover:text-white"
          >
            GitHub
          </a>
        </span>
      </footer>
    </div>
  );
}
