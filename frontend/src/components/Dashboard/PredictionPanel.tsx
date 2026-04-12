"use client";
import { useState, useEffect } from "react";
import { getSignal, SignalResponse } from "@/lib/api";

function Row({ label, value, sub }: { label: string; value: React.ReactNode; sub?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-zinc-500 text-[11px] shrink-0">{label}</span>
      <span className="text-zinc-200 text-[11px] font-mono tabular-nums text-right">
        {value}
        {sub && <span className="text-zinc-600 ml-1">{sub}</span>}
      </span>
    </div>
  );
}

function DirectionBadge({ thesis }: { thesis: string }) {
  const map: Record<string, { label: string; cls: string }> = {
    bullish:  { label: "BULLISH",  cls: "text-emerald-400 border-emerald-400/30 bg-emerald-400/5" },
    bearish:  { label: "BEARISH",  cls: "text-red-400 border-red-400/30 bg-red-400/5" },
    neutral:  { label: "NEUTRAL",  cls: "text-zinc-400 border-zinc-600 bg-zinc-800/30" },
    abstain:  { label: "ABSTAIN",  cls: "text-amber-400 border-amber-400/30 bg-amber-400/5" },
  };
  const { label, cls } = map[thesis] ?? map.neutral;
  return (
    <span className={`text-[10px] font-semibold tracking-wider px-1.5 py-0.5 border rounded-[2px] ${cls}`}>
      {label}
    </span>
  );
}

export default function PredictionPanel({ symbol }: { symbol: string }) {
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

  const s = data?.signal;
  const p = data?.prediction;
  const thesis = s?.direction ?? "neutral";
  const calProb = s?.probability ?? p?.calibrated_prob_up ?? null;
  const [bandLo, bandHi] = s?.confidence_band ?? p?.confidence_band ?? [null, null];

  return (
    <div className="inst-panel">
      <div className="inst-header">
        <span className="inst-label">Prediction</span>
        {s && <DirectionBadge thesis={thesis} />}
      </div>

      {loading && <div className="inst-body text-zinc-600 text-[11px]">Loading…</div>}

      {!loading && !data && (
        <div className="inst-body text-zinc-600 text-[11px]">No data</div>
      )}

      {!loading && s && p && (
        <div className="inst-body space-y-1.5">
          {/* Primary probability */}
          <div className="flex items-baseline justify-between pb-1.5 border-b border-border">
            <span className="text-zinc-500 text-[11px]">Calibrated P(up)</span>
            <div className="text-right">
              <span className={`text-xl font-mono font-semibold tabular-nums ${
                thesis === "bullish" ? "text-emerald-400" :
                thesis === "bearish" ? "text-red-400" : "text-zinc-300"
              }`}>
                {calProb != null ? (calProb * 100).toFixed(1) : "—"}%
              </span>
              {bandLo != null && bandHi != null && (
                <div className="text-[10px] text-zinc-600 font-mono" title="±ECE (avg. miscalibration) — not a statistical confidence interval">
                  ±ECE [{(bandLo * 100).toFixed(1)}% – {(bandHi * 100).toFixed(1)}%]
                </div>
              )}
            </div>
          </div>

          {/* Probability breakdown */}
          <Row label="Raw P(up)" value={p.prob_up != null ? `${(p.prob_up * 100).toFixed(1)}%` : "—"} />
          <Row label="P(down)" value={p.prob_down != null ? `${(p.prob_down * 100).toFixed(1)}%` : "—"} />

          {/* Divider */}
          <div className="border-t border-border my-1" />

          {/* Expected move */}
          <Row
            label="Expected move (1-bar)"
            value={p.expected_move_pct != null ? `±${p.expected_move_pct.toFixed(3)}%` : "—"}
          />
          {s.volatility_context && (
            <Row label="Volatility context" value={s.volatility_context} />
          )}

          {/* Divider */}
          <div className="border-t border-border my-1" />

          {/* Signal quality */}
          <Row
            label="Signal quality"
            value={
              <span className={
                s.signal_quality_score >= 60 ? "text-emerald-400" :
                s.signal_quality_score >= 40 ? "text-amber-400" : "text-red-400"
              }>
                {s.signal_quality_score.toFixed(0)} / 100
              </span>
            }
          />

          {/* Top features */}
          {s.top_features && Object.keys(s.top_features).length > 0 && (
            <>
              <div className="border-t border-border my-1" />
              <div className="space-y-0.5">
                {Object.entries(s.top_features).slice(0, 4).map(([feat, val]) => (
                  <div key={feat} className="flex items-center justify-between">
                    <span className="text-zinc-600 text-[10px]">{feat.replace(/_/g, " ")}</span>
                    <span className={`text-[10px] font-mono tabular-nums ${val >= 0 ? "text-emerald-400/70" : "text-red-400/70"}`}>
                      {val >= 0 ? "+" : ""}{val.toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            </>
          )}

          {/* Explanation */}
          {s.explanation && (
            <p className="text-[10px] text-zinc-600 pt-1 border-t border-border leading-relaxed">
              {s.explanation}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
