import { useEffect, useState } from "react";
import { getDecision, OptionsDecision, StructureCandidate, IVAnalysis } from "../../lib/api";

interface Props {
  symbol: string;
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function ScoreBadge({ score, viable }: { score: number; viable: boolean }) {
  const color = !viable
    ? "bg-zinc-700 text-zinc-400"
    : score >= 70
    ? "bg-emerald-900 text-emerald-300"
    : score >= 50
    ? "bg-blue-900 text-blue-300"
    : score >= 25
    ? "bg-amber-900 text-amber-300"
    : "bg-zinc-700 text-zinc-400";
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-mono font-bold ${color}`}>
      {score.toFixed(0)}
    </span>
  );
}

function IVEdgeBadge({ edge }: { edge: string }) {
  const styles: Record<string, string> = {
    favorable: "bg-emerald-900 text-emerald-300",
    neutral: "bg-zinc-700 text-zinc-300",
    unfavorable: "bg-red-900 text-red-300",
  };
  return (
    <span className={`px-1.5 py-0.5 rounded text-xs ${styles[edge] ?? styles.neutral}`}>
      {edge}
    </span>
  );
}

function LiqBadge({ fit }: { fit: string }) {
  const styles: Record<string, string> = {
    good: "text-emerald-400",
    fair: "text-amber-400",
    poor: "text-red-400",
    unknown: "text-zinc-400",
  };
  return <span className={`text-xs ${styles[fit] ?? "text-zinc-400"}`}>{fit}</span>;
}

function DirectionBadge({ thesis }: { thesis: string }) {
  const styles: Record<string, string> = {
    bullish: "bg-emerald-900 text-emerald-300",
    bearish: "bg-red-900 text-red-300",
    neutral: "bg-zinc-700 text-zinc-300",
    abstain: "bg-amber-900 text-amber-300",
  };
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-semibold ${styles[thesis] ?? styles.neutral}`}>
      {thesis.toUpperCase()}
    </span>
  );
}

function IVSummary({ iv }: { iv: IVAnalysis }) {
  const ratioColor =
    iv.iv_vs_rv === "rich"
      ? "text-amber-400"
      : iv.iv_vs_rv === "cheap"
      ? "text-emerald-400"
      : "text-zinc-300";

  const rankPct = (iv.iv_rank * 100).toFixed(0);

  return (
    <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs text-zinc-400 mt-1">
      <div>
        ATM IV <span className="text-zinc-200 font-mono">{(iv.atm_iv * 100).toFixed(1)}%</span>
      </div>
      <div>
        RV <span className="text-zinc-200 font-mono">{(iv.realized_vol_ann * 100).toFixed(1)}%</span>
      </div>
      <div>
        IV/RV{" "}
        <span className={`font-mono ${ratioColor}`}>
          {iv.iv_rv_ratio.toFixed(2)} ({iv.iv_vs_rv})
        </span>
      </div>
      <div>
        IV Rank <span className="text-zinc-200 font-mono">{rankPct}th</span>
      </div>
      <div>
        Implied 1d{" "}
        <span className="text-zinc-200 font-mono">±{iv.iv_implied_1d_move_pct.toFixed(2)}%</span>
      </div>
      <div>
        Model 1d{" "}
        <span className="text-zinc-200 font-mono">±{iv.rv_implied_1d_move_pct.toFixed(2)}%</span>
      </div>
    </div>
  );
}

function ScoreBar({ score }: { score: number }) {
  const pct = Math.max(0, Math.min(100, score));
  const color =
    pct >= 70 ? "bg-emerald-500" : pct >= 50 ? "bg-blue-500" : pct >= 25 ? "bg-amber-500" : "bg-zinc-600";
  return (
    <div className="w-full bg-zinc-800 rounded h-1.5 mt-1">
      <div className={`${color} h-1.5 rounded`} style={{ width: `${pct}%` }} />
    </div>
  );
}

function CandidateCard({
  candidate,
  isRecommended,
}: {
  candidate: StructureCandidate;
  isRecommended: boolean;
}) {
  const [open, setOpen] = useState(false);
  const label = candidate.structure_type.replace(/_/g, " ");
  const borderColor = isRecommended
    ? "border-emerald-600"
    : candidate.viable
    ? "border-zinc-600"
    : "border-zinc-800";

  return (
    <div className={`border rounded-lg p-3 ${borderColor} bg-zinc-900`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-zinc-200 capitalize">{label}</span>
          {isRecommended && (
            <span className="text-xs bg-emerald-700 text-emerald-200 px-1.5 py-0.5 rounded font-semibold">
              RECOMMENDED
            </span>
          )}
          {!candidate.viable && (
            <span className="text-xs bg-zinc-800 text-zinc-500 px-1.5 py-0.5 rounded">
              NOT VIABLE
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <IVEdgeBadge edge={candidate.iv_edge} />
          <LiqBadge fit={candidate.liquidity_fit} />
          <ScoreBadge score={candidate.score} viable={candidate.viable} />
        </div>
      </div>

      <ScoreBar score={candidate.score} />

      <div className="grid grid-cols-3 gap-x-4 mt-2 text-xs text-zinc-400">
        {candidate.breakeven_move_pct > 0 && (
          <div>
            BE{" "}
            <span className="text-zinc-200 font-mono">
              {candidate.breakeven_move_pct.toFixed(2)}%
            </span>
          </div>
        )}
        {candidate.estimated_cost_pct > 0 && (
          <div>
            Cost{" "}
            <span className="text-zinc-200 font-mono">
              {candidate.estimated_cost_pct.toFixed(2)}%
            </span>
          </div>
        )}
        {candidate.estimated_credit_pct > 0 && (
          <div>
            Credit{" "}
            <span className="text-zinc-200 font-mono">
              {candidate.estimated_credit_pct.toFixed(2)}%
            </span>
          </div>
        )}
        {candidate.max_loss_pct > 0 && candidate.max_loss_pct < 999 && (
          <div>
            Max loss{" "}
            <span className="text-zinc-200 font-mono">{candidate.max_loss_pct.toFixed(2)}%</span>
          </div>
        )}
      </div>

      {candidate.legs.length > 0 && (
        <div className="mt-2 flex gap-2 flex-wrap">
          {candidate.legs.map((leg, i) => (
            <div key={i} className="text-xs bg-zinc-800 rounded px-2 py-1 text-zinc-300 font-mono">
              {leg.action.toUpperCase()} {leg.option_type.toUpperCase()}
              {leg.strike != null ? ` @${leg.strike.toFixed(1)}` : ""}
              {leg.estimated_mid != null ? ` ~$${leg.estimated_mid.toFixed(2)}` : ""}
              {` Δ${leg.target_delta.toFixed(2)}`}
            </div>
          ))}
        </div>
      )}

      <button
        onClick={() => setOpen((v) => !v)}
        className="text-xs text-zinc-500 hover:text-zinc-300 mt-2 underline"
      >
        {open ? "hide details" : "show details"}
      </button>

      {open && (
        <div className="mt-2 space-y-1">
          {candidate.horizon_note && (
            <p className="text-xs text-zinc-400 italic">{candidate.horizon_note}</p>
          )}
          {candidate.tailwinds.length > 0 && (
            <ul className="text-xs text-emerald-400 space-y-0.5">
              {candidate.tailwinds.map((t, i) => (
                <li key={i}>✓ {t}</li>
              ))}
            </ul>
          )}
          {candidate.concerns.length > 0 && (
            <ul className="text-xs text-amber-400 space-y-0.5">
              {candidate.concerns.map((c, i) => (
                <li key={i}>⚠ {c}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

export default function DecisionPanel({ symbol }: Props) {
  const [decision, setDecision] = useState<OptionsDecision | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchDecision = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getDecision(symbol);
      setDecision(res.data);
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || "Failed to load decision");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (symbol) fetchDecision();
  }, [symbol]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-40 text-zinc-500 text-sm">
        Computing decision…
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 rounded bg-red-950 border border-red-800 text-red-300 text-sm">
        {error}
      </div>
    );
  }

  if (!decision) return null;

  const regimeColor = decision.regime_suppressed ? "text-red-400" : "text-zinc-300";

  return (
    <div className="space-y-4 text-zinc-300">
      {/* Header row */}
      <div className="flex items-start justify-between flex-wrap gap-2">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-lg font-bold text-zinc-100">{symbol}</span>
            <DirectionBadge thesis={decision.direction_thesis} />
            {decision.abstain && (
              <span className="text-xs bg-amber-900 text-amber-300 px-2 py-0.5 rounded">
                ABSTAIN
              </span>
            )}
          </div>
          <div className="text-xs text-zinc-500 mt-0.5">{decision.horizon}</div>
        </div>
        <div className="text-right">
          <div className="text-sm font-mono text-zinc-200">
            Confidence{" "}
            <span
              className={
                decision.confidence_score >= 60
                  ? "text-emerald-400"
                  : decision.confidence_score >= 40
                  ? "text-amber-400"
                  : "text-zinc-400"
              }
            >
              {decision.confidence_score.toFixed(0)}/100
            </span>
          </div>
          <div className={`text-xs ${regimeColor}`}>
            {decision.regime.replace(/_/g, " ")}
            {decision.regime_suppressed && " — suppressed"}
          </div>
        </div>
      </div>

      {/* Abstain callout */}
      {decision.abstain && (
        <div className="rounded bg-amber-950 border border-amber-800 px-3 py-2 text-sm text-amber-300">
          {decision.recommendation_rationale}
        </div>
      )}

      {/* Probability + range row */}
      <div className="grid grid-cols-2 gap-4 text-xs">
        <div className="bg-zinc-900 rounded-lg p-3 border border-zinc-800">
          <div className="text-zinc-500 mb-1 font-semibold uppercase tracking-wide text-[10px]">
            Probabilities
          </div>
          <div className="flex gap-4">
            <div>
              ↑ Up{" "}
              <span className="text-emerald-400 font-mono">
                {(decision.prob_up * 100).toFixed(1)}%
              </span>
            </div>
            <div>
              ↓ Down{" "}
              <span className="text-red-400 font-mono">
                {(decision.prob_down * 100).toFixed(1)}%
              </span>
            </div>
          </div>
          <div className="text-zinc-500 mt-1">
            Band {(decision.confidence_band[0] * 100).toFixed(1)}% –{" "}
            {(decision.confidence_band[1] * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-zinc-900 rounded-lg p-3 border border-zinc-800">
          <div className="text-zinc-500 mb-1 font-semibold uppercase tracking-wide text-[10px]">
            Expected Range (1d)
          </div>
          <div className="font-mono text-zinc-200">
            {decision.expected_range_low.toFixed(2)} – {decision.expected_range_high.toFixed(2)}
          </div>
          <div className="text-zinc-500 mt-0.5">
            ±{decision.expected_move_1d_pct.toFixed(2)}% from spot{" "}
            <span className="text-zinc-300">{decision.spot_price.toFixed(2)}</span>
          </div>
        </div>
      </div>

      {/* IV analysis */}
      <div className="bg-zinc-900 rounded-lg p-3 border border-zinc-800">
        <div className="text-zinc-500 mb-1 font-semibold uppercase tracking-wide text-[10px]">
          IV Environment
        </div>
        <IVSummary iv={decision.iv_analysis} />
      </div>

      {/* Candidate structures */}
      <div>
        <div className="text-zinc-500 font-semibold uppercase tracking-wide text-[10px] mb-2">
          Candidate Structures
        </div>
        <div className="space-y-2">
          {decision.candidates.map((c) => (
            <CandidateCard
              key={c.structure_type}
              candidate={c}
              isRecommended={c.structure_type === decision.recommended_structure}
            />
          ))}
        </div>
      </div>

      {/* Recommendation rationale */}
      {!decision.abstain && decision.recommendation_rationale && (
        <div className="text-xs text-zinc-400 italic border-l-2 border-zinc-700 pl-3">
          {decision.recommendation_rationale}
        </div>
      )}

      {/* Refresh */}
      <button
        onClick={fetchDecision}
        className="text-xs text-zinc-500 hover:text-zinc-300 underline"
      >
        refresh
      </button>
    </div>
  );
}
