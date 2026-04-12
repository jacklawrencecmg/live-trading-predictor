"use client";
import { useState, useEffect } from "react";
import { executeTrade, getPositions, getTradeHistory, getPortfolio, Position, Trade, PortfolioSummary } from "@/lib/api";
import clsx from "clsx";

interface Props {
  symbol: string;
  currentPrice?: number;
}

export default function PaperTrader({ symbol, currentPrice }: Props) {
  const [quantity, setQuantity] = useState(1);
  const [action, setAction] = useState<"BTO" | "STO">("BTO");
  const [positions, setPositions] = useState<Position[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [portfolio, setPortfolio] = useState<PortfolioSummary | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"positions" | "history">("positions");

  const load = async () => {
    const [pos, hist, port] = await Promise.all([
      getPositions(),
      getTradeHistory(20),
      getPortfolio(),
    ]);
    setPositions(pos.data);
    setTrades(hist.data);
    setPortfolio(port.data);
  };

  useEffect(() => {
    load();
  }, []);

  const handleTrade = async () => {
    setSubmitting(true);
    setError(null);
    try {
      await executeTrade({
        symbol,
        action,
        quantity,
        price: currentPrice,
      });
      await load();
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Portfolio summary */}
      {portfolio && (
        <div className="grid grid-cols-4 gap-2">
          {[
            { label: "Cash", value: `$${portfolio.cash.toLocaleString("en-US", { maximumFractionDigits: 0 })}` },
            { label: "Positions", value: `$${portfolio.positions_value.toLocaleString("en-US", { maximumFractionDigits: 0 })}` },
            {
              label: "Daily P&L",
              value: `${portfolio.daily_pnl >= 0 ? "+" : ""}$${portfolio.daily_pnl.toFixed(0)}`,
              color: portfolio.daily_pnl >= 0 ? "text-green-trade" : "text-red-trade",
            },
            {
              label: "Total P&L",
              value: `${portfolio.total_pnl >= 0 ? "+" : ""}$${portfolio.total_pnl.toFixed(0)}`,
              color: portfolio.total_pnl >= 0 ? "text-green-trade" : "text-red-trade",
            },
          ].map((s) => (
            <div key={s.label} className="bg-surface rounded p-2">
              <div className="text-xs text-gray-500">{s.label}</div>
              <div className={clsx("text-sm font-semibold", s.color ?? "text-white")}>{s.value}</div>
            </div>
          ))}
        </div>
      )}

      {/* Kill switch warning */}
      {portfolio?.kill_switch_active && (
        <div className="bg-red-trade/10 border border-red-trade/50 rounded p-2 text-red-trade text-xs font-semibold">
          KILL SWITCH ACTIVE — Trading halted
        </div>
      )}

      {/* Order entry */}
      <div className="flex gap-2 items-end">
        <div>
          <label className="text-xs text-gray-400 block mb-1">Action</label>
          <select
            value={action}
            onChange={(e) => setAction(e.target.value as any)}
            className="bg-panel border border-border text-white text-sm px-2 py-1 rounded"
          >
            <option value="BTO">Buy to Open</option>
            <option value="STO">Sell to Open</option>
            <option value="BTC">Buy to Close</option>
            <option value="STC">Sell to Close</option>
          </select>
        </div>
        <div>
          <label className="text-xs text-gray-400 block mb-1">Qty</label>
          <input
            type="number"
            value={quantity}
            min={1}
            onChange={(e) => setQuantity(Number(e.target.value))}
            className="bg-panel border border-border text-white text-sm px-2 py-1 rounded w-20"
          />
        </div>
        <div>
          <label className="text-xs text-gray-400 block mb-1">Price</label>
          <div className="text-sm text-white bg-panel border border-border px-2 py-1 rounded w-24">
            {currentPrice ? `$${currentPrice.toFixed(2)}` : "Market"}
          </div>
        </div>
        <button
          onClick={handleTrade}
          disabled={submitting || portfolio?.kill_switch_active}
          className="bg-accent text-surface font-semibold text-sm px-4 py-1.5 rounded hover:bg-accent/80 disabled:opacity-40 transition-colors"
        >
          {submitting ? "..." : "Submit"}
        </button>
      </div>

      {error && <div className="text-red-trade text-xs">{error}</div>}

      {/* Tabs */}
      <div className="flex gap-2 border-b border-border">
        {(["positions", "history"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={clsx(
              "text-xs pb-1 px-2 capitalize",
              tab === t ? "border-b border-accent text-white" : "text-gray-500"
            )}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === "positions" && (
        <div className="overflow-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-gray-500">
                <th className="text-left pb-1">Symbol</th>
                <th className="text-right pb-1">Qty</th>
                <th className="text-right pb-1">Cost</th>
                <th className="text-right pb-1">Current</th>
                <th className="text-right pb-1">Unreal P&L</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((p) => (
                <tr key={p.id} className="border-t border-border/50">
                  <td className="py-1">{p.option_symbol || p.symbol}</td>
                  <td className="text-right">{p.quantity}</td>
                  <td className="text-right">${p.avg_cost.toFixed(2)}</td>
                  <td className="text-right">{p.current_price ? `$${p.current_price.toFixed(2)}` : "—"}</td>
                  <td className={clsx("text-right", p.unrealized_pnl >= 0 ? "text-green-trade" : "text-red-trade")}>
                    {p.unrealized_pnl >= 0 ? "+" : ""}${p.unrealized_pnl.toFixed(2)}
                  </td>
                </tr>
              ))}
              {positions.length === 0 && (
                <tr><td colSpan={5} className="text-center text-gray-600 py-3">No open positions</td></tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {tab === "history" && (
        <div className="overflow-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-gray-500">
                <th className="text-left pb-1">Time</th>
                <th className="text-left pb-1">Symbol</th>
                <th className="text-left pb-1">Action</th>
                <th className="text-right pb-1">Qty</th>
                <th className="text-right pb-1">Price</th>
                <th className="text-right pb-1">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t) => (
                <tr key={t.id} className="border-t border-border/50">
                  <td className="py-1">{new Date(t.executed_at).toLocaleTimeString()}</td>
                  <td>{t.option_symbol || t.symbol}</td>
                  <td className={clsx(
                    t.action.startsWith("B") ? "text-green-trade" : "text-red-trade"
                  )}>{t.action}</td>
                  <td className="text-right">{t.quantity}</td>
                  <td className="text-right">${t.price.toFixed(2)}</td>
                  <td className="text-right text-gray-400">
                    {t.model_confidence != null ? `${(t.model_confidence * 100).toFixed(0)}%` : "—"}
                  </td>
                </tr>
              ))}
              {trades.length === 0 && (
                <tr><td colSpan={6} className="text-center text-gray-600 py-3">No trades yet</td></tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
