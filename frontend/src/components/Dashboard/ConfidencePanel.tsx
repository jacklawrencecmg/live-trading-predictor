"use client";
import { useState, useEffect } from "react";
import { getSignal, SignalResponse } from "@/lib/api";

function Layer({
  n,
  label,
  value,
  sub,
  dimmed,
}: {
  n: number;
  label: string;
  value: React.ReactNode;
  sub?: string;
  dimmed?: boolean;
}) {
  return (
    <div className={`flex items-start gap-2 ${dimmed ? "opacity-40" : ""}`}>
      <span className="text-[10px] text-zinc-700 font-mono w-3 shrink-0 pt-0.5">{n}</span>
      <div className="flex-1 flex items-baseline justify-between gap-2">
        <span className="text-zinc-500 text-[11px]">{label}</span>
        <div className="text-right">
          <span className="text-zinc-200 text-[11px] font-mono tabular-nums">{value}</span>
          {sub && <div className="text-[10px] text-zinc-600">{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function HealthBadge({ health }: { health: string }) {
  const map: Record<string, string> = {
    good:     "text-emerald-400 border-emerald-400/30",
    fair:     "text-amber-400 border-amber-400/30",
    degraded: "text-red-400 border-red-400/30",
    unknown:  "text-zinc-500 border-zinc-700",
  };
  return (
    <span className={`text-[10px] font-semibold tracking-wider uppercase px-1.5 py-0.5 border rounded-[2px] ${map[health] ?? map.unknown}`}>
      {health}
    </span>
  );
}

function ActionBadge({ action }: { action: string }) {
  if (action === "buy")     return <span className="text-emerald-400 text-[11px] font-semibold">BUY</span>;
  if (action === "sell")    return <span className="text-red-400 text-[11px] font-semibold">SELL</span>;
  if (action === "abstain") return <span className="text-amber-400 text-[11px] font-semibold">ABSTAIN</span>;
  return <span className="text-zinc-400 text-[11px] font-semibold">NO TRADE</span>;
}

function AbstainReasonTree({ reason }: { reason: string }) {
  // Parse "reason:detail" pairs
  const parts = reason.split(":");
  const primary = parts[0].replace(/_/g, " ");
  const detail = parts.slice(1).join(":").replace(/_/g, " ");
  return (
    <div className="bg-amber-400/5 border border-amber-400/20 rounded-[2px] p-2 space-y-1">
      <div className="text-[10px] text-amber-400/80 font-semibold uppercase tracking-wide">Abstain reason</div>
      <div className="text-[11px] text-amber-300">{primary}</div>
      {detail && <div className="text-[10px] text-zinc-500 pl-2">→ {detail}</div>}
    </div>
  );
}

export default function ConfidencePanel({ symbol }: { symbol: string }) {
  const [data, setData] = useState<SignalResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    setLoading(true);
    const load = () =>
      getSignal(symbol)
        .then((r) => { if (active) { setData(r.data); setLoading(false); } })
        .catch(() => { if (active) setLoading(false); });
    load();
    const t = setInterval(load, 30_000);
    return () => { active = false; clearInterval(t); };
  }, [symbol]);

  const p = data?.prediction;
  const s = data?.signal;
  const isAbstain = s?.action === "abstain" || p?.action === "abstain";
  const abstainReason = s?.abstain_reason ?? p?.abstain_reason;
  const health = s?.calibration_health ?? p?.calibration_health ?? "unknown";
  const deg = s?.degradation_factor ?? p?.degradation_factor ?? 1.0;

  const calAdj =
    p?.prob_up != null && p?.calibrated_prob_up != null
      ? p.calibrated_prob_up - p.prob_up
      : null;

  return (
    <div className="inst-panel">
      <div className="inst-header">
        <span className="inst-label">Confidence Layers</span>
        {p && <HealthBadge health={health} />}
      </div>

      {loading && <div className="inst-body text-zinc-600 text-[11px]">Loading…</div>}
      {!loading && !data && <div className="inst-body text-zinc-600 text-[11px]">No data</div>}

      {!loading && p && s && (
        <div className="inst-body space-y-2">
          {/* 4-layer stack */}
          <Layer
            n={1}
            label="Raw P(up)"
            value={p.prob_up != null ? `${(p.prob_up * 100).toFixed(2)}%` : "—"}
          />
          <Layer
            n={2}
            label="Calibrated P(up)"
            value={p.calibrated_prob_up != null ? `${(p.calibrated_prob_up * 100).toFixed(2)}%` : "—"}
            sub={calAdj != null && Math.abs(calAdj) > 0.001
              ? `${calAdj >= 0 ? "+" : ""}${(calAdj * 100).toFixed(2)}% cal. adj.`
              : undefined}
          />
          <Layer
            n={3}
            label="Degradation factor"
            value={
              <span className={deg < 0.7 ? "text-red-400" : deg < 0.9 ? "text-amber-400" : "text-zinc-200"}>
                ×{deg.toFixed(3)}
              </span>
            }
            sub={deg < 0.7 ? "model degraded" : deg < 0.9 ? "slight degradation" : undefined}
          />
          <Layer
            n={4}
            label="Tradeable confidence"
            value={
              <span className={
                (p.tradeable_confidence ?? 0) >= 0.6 ? "text-emerald-400" :
                (p.tradeable_confidence ?? 0) >= 0.5 ? "text-zinc-200" : "text-zinc-500"
              }>
                {p.tradeable_confidence != null ? (p.tradeable_confidence * 100).toFixed(1) + "%" : "—"}
              </span>
            }
          />

          <div className="border-t border-border my-0.5" />

          {/* Action */}
          <div className="flex items-center justify-between">
            <span className="text-zinc-500 text-[11px]">Action</span>
            <ActionBadge action={p.action ?? "abstain"} />
          </div>

          {/* Abstain reason tree */}
          {isAbstain && abstainReason && (
            <AbstainReasonTree reason={abstainReason} />
          )}

          <div className="border-t border-border my-0.5" />

          {/* Calibration stats */}
          <div className="flex items-center justify-between">
            <span className="text-zinc-500 text-[11px]">Calibration health</span>
            <HealthBadge health={health} />
          </div>
          {p.ece_recent != null && (
            <div className="flex items-baseline justify-between">
              <span className="text-zinc-500 text-[11px]">ECE (recent)</span>
              <span className={`text-[11px] font-mono tabular-nums ${
                p.ece_recent < 0.05 ? "text-emerald-400" :
                p.ece_recent < 0.10 ? "text-amber-400" : "text-red-400"
              }`}>
                {p.ece_recent.toFixed(4)}
              </span>
            </div>
          )}
          {p.rolling_brier != null && (
            <div className="flex items-baseline justify-between">
              <span className="text-zinc-500 text-[11px]">Rolling Brier</span>
              <span className="text-zinc-200 text-[11px] font-mono tabular-nums">
                {p.rolling_brier.toFixed(4)}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
