"use client";
import { useState, useEffect } from "react";
import { getDecision, OptionsDecision, StructureCandidate } from "@/lib/api";

const STRUCTURE_LABELS: Record<string, string> = {
  long_call:     "Long Call",
  long_put:      "Long Put",
  debit_spread:  "Debit Spread",
  credit_spread: "Credit Spread",
};

function ScoreBar({ score, max = 100 }: { score: number; max?: number }) {
  const pct = Math.round((score / max) * 100);
  const color = score >= 70 ? "#34d399" : score >= 45 ? "#60a5fa" : "#52525b";
  return (
    <div className="flex items-center gap-1.5">
      <div className="flex-1 h-1 bg-zinc-800 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="text-[10px] font-mono tabular-nums text-zinc-400 w-5 text-right">
        {score.toFixed(0)}
      </span>
    </div>
  );
}

function IVEdgeBadge({ edge }: { edge: string }) {
  const map: Record<string, string> = {
    favorable:   "text-emerald-400/80",
    neutral:     "text-zinc-500",
    unfavorable: "text-red-400/80",
  };
  return (
    <span className={`text-[10px] ${map[edge] ?? "text-zinc-600"}`}>
      IV {edge}
    </span>
  );
}

function StructureRow({
  c,
  isRecommended,
}: {
  c: StructureCandidate;
  isRecommended: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const label = STRUCTURE_LABELS[c.structure_type] ?? c.structure_type;
  const dirColor = c.direction === "bullish" ? "text-emerald-400/70" : "text-red-400/70";

  return (
    <div className={`border rounded-[2px] ${
      !c.viable ? "border-zinc-800 opacity-50" :
      isRecommended ? "border-accent/30 bg-accent/3" : "border-border"
    }`}>
      <button
        className="w-full text-left p-2 space-y-1"
        onClick={() => c.viable && setExpanded((v) => !v)}
        disabled={!c.viable}
      >
        <div className="flex items-center gap-1.5">
          {isRecommended && (
            <span className="text-[9px] text-accent font-semibold">★</span>
          )}
          <span className="text-[11px] text-zinc-200">{label}</span>
          <span className={`text-[10px] ${dirColor}`}>{c.direction}</span>
          {!c.viable && (
            <span className="ml-auto text-[10px] text-zinc-600">not viable</span>
          )}
        </div>
        {c.viable && <ScoreBar score={c.score} />}
        {c.viable && (
          <div className="flex items-center justify-between">
            <IVEdgeBadge edge={c.iv_edge} />
            <span className="text-[10px] text-zinc-600">
              {c.estimated_cost_pct > 0
                ? `cost ${c.estimated_cost_pct.toFixed(1)}%`
                : `credit ${c.estimated_credit_pct.toFixed(1)}%`}
              {" · "}break {c.breakeven_move_pct.toFixed(1)}%
            </span>
          </div>
        )}
      </button>

      {expanded && c.viable && (
        <div className="px-2 pb-2 space-y-1 border-t border-border">
          <div className="pt-1.5 text-[10px] text-zinc-500 leading-relaxed">{c.rationale}</div>
          {c.tailwinds.length > 0 && (
            <div>
              <div className="text-[9px] text-zinc-700 uppercase tracking-wide mb-0.5">Tailwinds</div>
              {c.tailwinds.map((t, i) => (
                <div key={i} className="text-[10px] text-emerald-400/70">+ {t}</div>
              ))}
            </div>
          )}
          {c.concerns.length > 0 && (
            <div>
              <div className="text-[9px] text-zinc-700 uppercase tracking-wide mb-0.5">Concerns</div>
              {c.concerns.map((c2, i) => (
                <div key={i} className="text-[10px] text-amber-400/70">− {c2}</div>
              ))}
            </div>
          )}
          {c.horizon_note && (
            <div className="text-[10px] text-zinc-600 italic">{c.horizon_note}</div>
          )}
        </div>
      )}
    </div>
  );
}

export default function StructuresPanel({ symbol }: { symbol: string }) {
  const [data, setData] = useState<OptionsDecision | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    setLoading(true);
    const load = () =>
      getDecision(symbol)
        .then((r) => { if (active) { setData(r.data); setLoading(false); } })
        .catch(() => { if (active) setLoading(false); });
    load();
    const t = setInterval(load, 60_000);
    return () => { active = false; clearInterval(t); };
  }, [symbol]);

  return (
    <div className="inst-panel">
      <div className="inst-header">
        <span className="inst-label">Candidate Structures</span>
        {data?.recommended_structure && (
          <span className="text-[10px] text-accent font-mono">
            {STRUCTURE_LABELS[data.recommended_structure] ?? data.recommended_structure}
          </span>
        )}
      </div>

      {loading && <div className="inst-body text-zinc-600 text-[11px]">Loading…</div>}
      {!loading && !data && <div className="inst-body text-zinc-600 text-[11px]">No decision data</div>}

      {!loading && data && (
        <div className="inst-body space-y-1.5">
          {/* Abstain banner */}
          {data.abstain && data.abstain_reason && (
            <div className="bg-amber-400/5 border border-amber-400/20 rounded-[2px] p-2 mb-1">
              <div className="text-[10px] text-amber-400/80 font-semibold">SYSTEM ABSTAINING</div>
              <div className="text-[10px] text-zinc-500 mt-0.5">
                {data.abstain_reason.replace(/_/g, " ")}
              </div>
            </div>
          )}

          {/* Confidence score */}
          {!data.abstain && (
            <div className="flex items-center justify-between pb-1.5 border-b border-border">
              <span className="text-zinc-500 text-[11px]">Decision confidence</span>
              <span className={`text-[13px] font-mono font-semibold tabular-nums ${
                data.confidence_score >= 55 ? "text-emerald-400" :
                data.confidence_score >= 35 ? "text-amber-400" : "text-zinc-500"
              }`}>
                {data.confidence_score.toFixed(0)} / 100
              </span>
            </div>
          )}

          {/* Structure cards */}
          {data.candidates.map((c) => (
            <StructureRow
              key={c.structure_type}
              c={c}
              isRecommended={data.recommended_structure === c.structure_type}
            />
          ))}

          {/* Recommendation rationale */}
          {data.recommendation_rationale && !data.abstain && (
            <p className="text-[10px] text-zinc-600 pt-1 border-t border-border leading-relaxed">
              {data.recommendation_rationale}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
