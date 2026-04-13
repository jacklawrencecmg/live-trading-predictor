"use client";
import { useState, useEffect } from "react";
import { getPortfolio, getPnLSummary, PortfolioSummary, PnLSummary } from "@/lib/api";

function PnLRow({ label, value, pct }: { label: string; value: number; pct?: number }) {
  const isPos = value >= 0;
  const cls = isPos ? "text-emerald-400" : "text-red-400";
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-zinc-500 text-[11px]">{label}</span>
      <div className="text-right">
        <span className={`text-[11px] font-mono tabular-nums ${cls}`}>
          {isPos ? "+" : "−"}${Math.abs(value).toFixed(2)}
        </span>
        {pct != null && (
          <span className={`text-[10px] font-mono ml-1 ${cls}`}>
            ({isPos ? "+" : "−"}{Math.abs(pct).toFixed(2)}%)
          </span>
        )}
      </div>
    </div>
  );
}

function MetricRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-zinc-500 text-[11px]">{label}</span>
      <span className="text-zinc-200 text-[11px] font-mono tabular-nums">{value}</span>
    </div>
  );
}

export default function PnLPanel() {
  const [portfolio, setPortfolio] = useState<PortfolioSummary | null>(null);
  const [summary, setSummary] = useState<PnLSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    const load = () =>
      Promise.all([getPortfolio(), getPnLSummary()])
        .then(([portRes, sumRes]) => {
          if (!active) return;
          setPortfolio(portRes.data);
          setSummary(sumRes.data);
          setLoading(false);
        })
        .catch(() => {
          if (active) setLoading(false);
        });
    load();
    const t = setInterval(load, 15_000);
    return () => { active = false; clearInterval(t); };
  }, []);

  return (
    <div className="inst-panel">
      <div className="inst-header">
        <span className="inst-label">Paper P&amp;L</span>
        {portfolio?.kill_switch_active && (
          <span className="text-[10px] font-semibold text-red-400 border border-red-400/40 px-1.5 py-0.5 rounded-[2px]">
            KILL SW ON
          </span>
        )}
      </div>

      {loading && <div className="inst-body text-zinc-600 text-[11px]">Loading…</div>}
      {!loading && !portfolio && <div className="inst-body text-zinc-600 text-[11px]">No portfolio data</div>}

      {!loading && portfolio && (
        <div className="inst-body space-y-1.5">
          {/* Daily P&L */}
          <PnLRow label="Daily P&L" value={portfolio.daily_pnl} pct={portfolio.daily_pnl_pct} />
          {portfolio.total_pnl != null && (
            <PnLRow label="Total P&L" value={portfolio.total_pnl} />
          )}

          <div className="border-t border-border my-0.5" />

          <MetricRow
            label="Portfolio value"
            value={`$${portfolio.total_value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          />
          <MetricRow
            label="Cash"
            value={`$${portfolio.cash.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          />
          <MetricRow label="Open positions" value={portfolio.open_positions} />

          {/* Rolling P&L metrics */}
          {summary && (
            <>
              <div className="border-t border-border my-0.5" />
              {summary.rolling_7d != null && (
                <PnLRow label="Rolling 7d" value={summary.rolling_7d} />
              )}
              {summary.rolling_30d != null && (
                <PnLRow label="Rolling 30d" value={summary.rolling_30d} />
              )}
              {summary.win_rate_30d != null && (
                <MetricRow
                  label="Win rate (30d)"
                  value={
                    <span className={summary.win_rate_30d >= 0.5 ? "text-emerald-400" : "text-amber-400"}>
                      {(summary.win_rate_30d * 100).toFixed(0)}%
                    </span>
                  }
                />
              )}
              {summary.sharpe_7d != null && (
                <MetricRow
                  label="Sharpe (7d)"
                  value={
                    <span className={
                      summary.sharpe_7d >= 1.0 ? "text-emerald-400" :
                      summary.sharpe_7d >= 0 ? "text-zinc-200" : "text-red-400"
                    }>
                      {summary.sharpe_7d.toFixed(2)}
                    </span>
                  }
                />
              )}
              {summary.trades_30d > 0 && (
                <MetricRow label="Trades (30d)" value={summary.trades_30d} />
              )}
            </>
          )}

          {portfolio.kill_switch_active && (
            <div className="bg-red-400/5 border border-red-400/30 rounded-[2px] p-1.5 mt-1">
              <span className="text-[10px] text-red-400">Kill switch active — all trading halted</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
