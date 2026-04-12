"use client";
import { useState, useEffect } from "react";
import { getSignalHistory, SignalHistoryEntry } from "@/lib/api";

function dirLabel(d: string) {
  if (d === "bullish")  return { text: "BULL ↑", cls: "text-emerald-400" };
  if (d === "bearish")  return { text: "BEAR ↓", cls: "text-red-400" };
  if (d === "abstain")  return { text: "ABST",   cls: "text-amber-400" };
  return                       { text: "NEUT",   cls: "text-zinc-500" };
}

function outcomeLabel(entry: SignalHistoryEntry) {
  if (entry.actual_outcome == null) return { text: "pending", cls: "text-zinc-600" };
  const sign = entry.outcome_pct != null && entry.outcome_pct >= 0 ? "+" : "";
  const pct  = entry.outcome_pct != null ? `${sign}${entry.outcome_pct.toFixed(2)}%` : entry.actual_outcome;
  if (entry.correct === true)  return { text: pct, cls: "text-emerald-400" };
  if (entry.correct === false) return { text: pct, cls: "text-red-400" };
  return { text: pct, cls: "text-zinc-400" };
}

function correctMark(entry: SignalHistoryEntry) {
  if (entry.correct === true)  return <span className="text-emerald-400">✓</span>;
  if (entry.correct === false) return <span className="text-red-400">✗</span>;
  return <span className="text-zinc-700">·</span>;
}

function formatTime(iso: string) {
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return iso.slice(11, 16);
  }
}

function EmptyState() {
  return (
    <div className="p-4 text-center">
      <p className="text-zinc-600 text-[11px]">No signal history available</p>
      <p className="text-zinc-700 text-[10px] mt-1">
        History requires the <code className="text-zinc-500">/api/signals/&#123;symbol&#125;/history</code> endpoint.
      </p>
    </div>
  );
}

export default function RecentSignalsTable({ symbol }: { symbol: string }) {
  const [rows, setRows] = useState<SignalHistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [unavailable, setUnavailable] = useState(false);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setUnavailable(false);
    getSignalHistory(symbol, 15)
      .then((r) => {
        if (!active) return;
        setRows(r.data ?? []);
        setLoading(false);
      })
      .catch(() => {
        if (active) {
          setUnavailable(true);
          setLoading(false);
        }
      });
    return () => { active = false; };
  }, [symbol]);

  const correctCount = rows.filter((r) => r.correct === true).length;
  const resolvedCount = rows.filter((r) => r.correct !== null).length;

  return (
    <div className="inst-panel">
      <div className="inst-header">
        <span className="inst-label">Recent Signals — Forecast vs Realized</span>
        {resolvedCount > 0 && (
          <span className="text-[10px] font-mono text-zinc-500">
            {correctCount}/{resolvedCount} correct
          </span>
        )}
      </div>

      {loading && <div className="inst-body text-zinc-600 text-[11px]">Loading…</div>}
      {!loading && unavailable && <EmptyState />}

      {!loading && !unavailable && rows.length === 0 && (
        <div className="inst-body text-zinc-600 text-[11px]">No signals recorded yet</div>
      )}

      {!loading && !unavailable && rows.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-[10px] font-mono">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left px-3 py-1.5 text-zinc-600 font-normal">Time</th>
                <th className="text-left px-2 py-1.5 text-zinc-600 font-normal">Dir.</th>
                <th className="text-right px-2 py-1.5 text-zinc-600 font-normal">Prob</th>
                <th className="text-left px-2 py-1.5 text-zinc-600 font-normal">Regime</th>
                <th className="text-right px-2 py-1.5 text-zinc-600 font-normal">Outcome</th>
                <th className="px-3 py-1.5 text-zinc-600 font-normal text-center">✓</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((entry) => {
                const dir = dirLabel(entry.direction);
                const out = outcomeLabel(entry);
                return (
                  <tr key={entry.id} className="border-b border-border/50 hover:bg-panel-2">
                    <td className="px-3 py-1 text-zinc-500">{formatTime(entry.bar_open_time)}</td>
                    <td className={`px-2 py-1 ${dir.cls}`}>{dir.text}</td>
                    <td className="px-2 py-1 text-right text-zinc-300">
                      {entry.action === "abstain"
                        ? <span className="text-zinc-700">—</span>
                        : `${(entry.calibrated_prob * 100).toFixed(0)}%`}
                    </td>
                    <td className="px-2 py-1 text-zinc-600 truncate max-w-[80px]">
                      {entry.regime.replace(/_/g, " ")}
                    </td>
                    <td className={`px-2 py-1 text-right ${out.cls}`}>{out.text}</td>
                    <td className="px-3 py-1 text-center">{correctMark(entry)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
