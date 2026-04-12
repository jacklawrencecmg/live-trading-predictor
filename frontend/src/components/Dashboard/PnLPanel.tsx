"use client";
import { useState, useEffect } from "react";
import { getPortfolio, PortfolioSummary } from "@/lib/api";

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

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-zinc-500 text-[11px]">{label}</span>
      <span className="text-zinc-200 text-[11px] font-mono tabular-nums">{value}</span>
    </div>
  );
}

export default function PnLPanel() {
  const [data, setData] = useState<PortfolioSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    const load = () =>
      getPortfolio()
        .then((r) => { if (active) { setData(r.data); setLoading(false); } })
        .catch(() => { if (active) setLoading(false); });
    load();
    const t = setInterval(load, 15_000);
    return () => { active = false; clearInterval(t); };
  }, []);

  const netPnL = data ? data.daily_pnl : null;
  const totalPnL = data ? data.total_pnl : null;

  return (
    <div className="inst-panel">
      <div className="inst-header">
        <span className="inst-label">Paper P&amp;L</span>
        {data?.kill_switch_active && (
          <span className="text-[10px] font-semibold text-red-400 border border-red-400/40 px-1.5 py-0.5 rounded-[2px]">
            KILL SW ON
          </span>
        )}
      </div>

      {loading && <div className="inst-body text-zinc-600 text-[11px]">Loading…</div>}
      {!loading && !data && <div className="inst-body text-zinc-600 text-[11px]">No portfolio data</div>}

      {!loading && data && (
        <div className="inst-body space-y-1.5">
          {/* Daily P&L */}
          <PnLRow
            label="Daily P&L"
            value={data.daily_pnl}
            pct={data.daily_pnl_pct}
          />
          {/* Total P&L */}
          {totalPnL != null && (
            <PnLRow label="Total P&L" value={totalPnL} />
          )}

          <div className="border-t border-border my-0.5" />

          <Row label="Portfolio value" value={`$${data.total_value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`} />
          <Row label="Cash" value={`$${data.cash.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`} />
          <Row label="Open positions" value={data.open_positions} />

          {data.kill_switch_active && (
            <div className="bg-red-400/5 border border-red-400/30 rounded-[2px] p-1.5 mt-1">
              <span className="text-[10px] text-red-400">Kill switch active — all trading halted</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
