"use client";
import { useState, useEffect } from "react";
import { getUncertainty, UncertaintyStats } from "@/lib/api";

function Row({ label, value, cls }: { label: string; value: React.ReactNode; cls?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-zinc-500 text-[11px] shrink-0">{label}</span>
      <span className={`text-[11px] font-mono tabular-nums text-right ${cls ?? "text-zinc-200"}`}>
        {value}
      </span>
    </div>
  );
}

function HealthBadge({ health }: { health: string }) {
  const map: Record<string, string> = {
    good:     "text-emerald-400 border-emerald-400/30 bg-emerald-400/5",
    fair:     "text-amber-400 border-amber-400/30 bg-amber-400/5",
    degraded: "text-red-400 border-red-400/30 bg-red-400/5",
    unknown:  "text-zinc-500 border-zinc-700 bg-zinc-800/30",
  };
  return (
    <span className={`text-[10px] font-semibold tracking-wider uppercase px-1.5 py-0.5 border rounded-[2px] ${map[health] ?? map.unknown}`}>
      {health}
    </span>
  );
}

function DegradationBar({ factor }: { factor: number }) {
  const pct = Math.round(factor * 100);
  const color = factor >= 0.85 ? "#34d399" : factor >= 0.60 ? "#fbbf24" : "#f87171";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1 bg-zinc-800 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-[10px] font-mono tabular-nums text-zinc-400 w-8 text-right">
        {factor.toFixed(2)}
      </span>
    </div>
  );
}

// Compact reliability diagram: bins as vertical bars
function ReliabilityMini({
  bins,
  meanPred,
  fracPos,
}: {
  bins: number[];
  meanPred: number[];
  fracPos: number[];
}) {
  if (!bins.length) return null;
  return (
    <div className="space-y-1">
      <div className="text-[10px] text-zinc-600 uppercase tracking-wide">Reliability diagram</div>
      <div className="flex items-end gap-0.5 h-10">
        {bins.map((bin, i) => {
          const ideal = meanPred[i] ?? bin;
          const actual = fracPos[i] ?? 0;
          const diff = actual - ideal;
          const barH = Math.round(actual * 40); // max 40px
          const color = Math.abs(diff) < 0.05 ? "#34d399" : Math.abs(diff) < 0.12 ? "#fbbf24" : "#f87171";
          return (
            <div key={i} className="flex-1 flex flex-col items-center justify-end" title={`bin ${bin.toFixed(1)}: pred=${ideal.toFixed(2)} actual=${actual.toFixed(2)}`}>
              <div style={{ height: `${barH}px`, backgroundColor: color, minHeight: "1px" }} className="w-full rounded-[1px]" />
            </div>
          );
        })}
      </div>
      <div className="flex justify-between text-[9px] text-zinc-700">
        <span>0.0</span>
        <span>ideal = diagonal</span>
        <span>1.0</span>
      </div>
    </div>
  );
}

export default function ModelHealthPanel({ symbol }: { symbol: string }) {
  const [data, setData] = useState<UncertaintyStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    setLoading(true);
    const load = () =>
      getUncertainty(symbol)
        .then((r) => { if (active) { setData(r.data); setLoading(false); } })
        .catch(() => { if (active) setLoading(false); });
    load();
    const t = setInterval(load, 60_000);
    return () => { active = false; clearInterval(t); };
  }, [symbol]);

  const health = data?.calibration_health ?? "unknown";

  const needsRetrain: boolean = data?.needs_retrain ?? false;
  const retrainReason: string | null = data?.retrain_reason ?? null;

  return (
    <div className="inst-panel">
      <div className="inst-header">
        <span className="inst-label">Model Health</span>
        {data && <HealthBadge health={health} />}
      </div>

      {loading && <div className="inst-body text-zinc-600 text-[11px]">Loading…</div>}
      {!loading && !data && <div className="inst-body text-zinc-600 text-[11px]">No data</div>}

      {!loading && data && (
        <div className="inst-body space-y-1.5">
          {/* Retrain alert */}
          {needsRetrain && (
            <div className="bg-red-400/5 border border-red-400/30 rounded-[2px] p-2 mb-2">
              <div className="text-[10px] text-red-400 font-semibold uppercase tracking-wide">
                ⚠ Retrain recommended
              </div>
              {retrainReason && (
                <div className="text-[10px] text-zinc-500 mt-0.5">{retrainReason}</div>
              )}
            </div>
          )}

          {/* Metrics */}
          <Row
            label="Rolling Brier"
            value={data.rolling_brier != null ? data.rolling_brier.toFixed(4) : "—"}
            cls={
              data.rolling_brier != null && data.baseline_brier != null
                ? data.rolling_brier <= data.baseline_brier * 1.1 ? "text-emerald-400"
                  : data.rolling_brier <= data.baseline_brier * 1.3 ? "text-amber-400"
                  : "text-red-400"
                : "text-zinc-200"
            }
          />
          <Row
            label="Baseline Brier"
            value={data.baseline_brier != null ? data.baseline_brier.toFixed(4) : "—"}
          />
          <Row
            label="ECE (recent)"
            value={data.ece_recent != null ? data.ece_recent.toFixed(4) : "—"}
            cls={
              data.ece_recent != null
                ? data.ece_recent < 0.05 ? "text-emerald-400"
                  : data.ece_recent < 0.10 ? "text-amber-400"
                  : "text-red-400"
                : "text-zinc-200"
            }
          />

          {/* Degradation bar */}
          <div className="space-y-0.5">
            <div className="flex items-baseline justify-between">
              <span className="text-zinc-500 text-[11px]">Degradation factor</span>
            </div>
            <DegradationBar factor={data.degradation_factor} />
          </div>

          <Row
            label="Window size"
            value={`${data.window_size} bars`}
            cls={data.window_size < 40 ? "text-amber-400" : "text-zinc-200"}
          />

          {/* Reliability diagram */}
          {data.reliability_diagram &&
            data.reliability_diagram.bins.length > 0 && (
              <>
                <div className="border-t border-border my-1" />
                <ReliabilityMini
                  bins={data.reliability_diagram.bins}
                  meanPred={data.reliability_diagram.mean_predicted}
                  fracPos={data.reliability_diagram.fraction_positive}
                />
              </>
            )}
        </div>
      )}
    </div>
  );
}
