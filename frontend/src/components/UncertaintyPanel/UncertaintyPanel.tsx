"use client";
import { useState, useEffect } from "react";
import clsx from "clsx";
import { getUncertainty, UncertaintyStats, ReliabilityDiagram } from "@/lib/api";

interface Props {
  symbol: string;
}

const HEALTH_LABEL: Record<string, string> = {
  good: "Well-calibrated",
  fair: "Slightly off",
  degraded: "Miscalibrated",
  unknown: "No data yet",
};

const HEALTH_COLOR: Record<string, string> = {
  good: "text-green-trade",
  fair: "text-yellow-400",
  degraded: "text-red-trade",
  unknown: "text-gray-500",
};

const HEALTH_BG: Record<string, string> = {
  good: "bg-green-trade",
  fair: "bg-yellow-500",
  degraded: "bg-red-trade",
  unknown: "bg-gray-600",
};

/** Mini SVG reliability diagram: predicted vs actual */
function ReliabilityChart({ diagram }: { diagram: ReliabilityDiagram }) {
  const { bins, mean_predicted, fraction_positive } = diagram;
  if (!bins || bins.length === 0) return null;

  const W = 200;
  const H = 120;
  const PAD = 20;
  const innerW = W - PAD * 2;
  const innerH = H - PAD * 2;

  // Perfect calibration diagonal
  const diag = `M ${PAD} ${H - PAD} L ${W - PAD} ${PAD}`;

  // Actual calibration line: points are (mean_predicted[i], fraction_positive[i])
  const pts = mean_predicted
    .map((x, i) => {
      const cx = PAD + x * innerW;
      const cy = H - PAD - fraction_positive[i] * innerH;
      return `${cx},${cy}`;
    })
    .join(" L ");
  const actualPath = mean_predicted.length > 0 ? `M ${pts}` : "";

  // Bar chart backgrounds: bins are bucket boundaries
  const barElems = mean_predicted.map((x, i) => {
    const barW = innerW / mean_predicted.length;
    const bx = PAD + i * barW;
    const bh = fraction_positive[i] * innerH;
    return (
      <rect
        key={i}
        x={bx}
        y={H - PAD - bh}
        width={barW - 1}
        height={bh}
        fill="#58a6ff"
        opacity={0.15}
      />
    );
  });

  return (
    <svg width={W} height={H} className="overflow-visible">
      {/* Axes */}
      <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#21262d" strokeWidth={1} />
      <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#21262d" strokeWidth={1} />
      {/* Bar fill */}
      {barElems}
      {/* Perfect calibration diagonal */}
      <path d={diag} stroke="#3fb950" strokeWidth={1} strokeDasharray="3 2" fill="none" opacity={0.5} />
      {/* Actual calibration */}
      {actualPath && (
        <path d={actualPath} stroke="#58a6ff" strokeWidth={1.5} fill="none" />
      )}
      {/* Dots on actual */}
      {mean_predicted.map((x, i) => (
        <circle
          key={i}
          cx={PAD + x * innerW}
          cy={H - PAD - fraction_positive[i] * innerH}
          r={2.5}
          fill="#58a6ff"
        />
      ))}
      {/* Axis labels */}
      <text x={PAD + innerW / 2} y={H} fontSize={8} fill="#6e7681" textAnchor="middle">
        Predicted prob.
      </text>
      <text
        x={8}
        y={H - PAD - innerH / 2}
        fontSize={8}
        fill="#6e7681"
        textAnchor="middle"
        transform={`rotate(-90, 8, ${H - PAD - innerH / 2})`}
      >
        Actual freq.
      </text>
      {/* 0/1 labels */}
      <text x={PAD} y={H - PAD + 10} fontSize={7} fill="#6e7681" textAnchor="middle">0</text>
      <text x={W - PAD} y={H - PAD + 10} fontSize={7} fill="#6e7681" textAnchor="middle">1</text>
      <text x={PAD - 4} y={H - PAD} fontSize={7} fill="#6e7681" textAnchor="end">0</text>
      <text x={PAD - 4} y={PAD + 4} fontSize={7} fill="#6e7681" textAnchor="end">1</text>
    </svg>
  );
}

function StatRow({
  label,
  value,
  colorClass,
}: {
  label: string;
  value: string;
  colorClass?: string;
}) {
  return (
    <div className="flex items-center justify-between text-xs">
      <span className="text-gray-500">{label}</span>
      <span className={colorClass ?? "text-white"}>{value}</span>
    </div>
  );
}

function DegradationGauge({ factor }: { factor: number }) {
  const segments = [
    { threshold: 0.8, color: "bg-green-trade", label: "Strong" },
    { threshold: 0.5, color: "bg-yellow-500", label: "Moderate" },
    { threshold: 0, color: "bg-red-trade", label: "Weak" },
  ];
  const { color, label } = segments.find((s) => factor >= s.threshold) ?? segments[2];
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-gray-500">Degradation factor</span>
        <span className={clsx("font-medium", color.replace("bg-", "text-").replace("500", "400"))}>
          {(factor * 100).toFixed(0)}% — {label}
        </span>
      </div>
      <div className="relative h-2 bg-border rounded-full overflow-hidden">
        <div className={clsx("h-2 rounded-full", color)} style={{ width: `${factor * 100}%` }} />
        {/* threshold markers */}
        <div className="absolute top-0 left-[80%] w-px h-2 bg-gray-700" />
        <div className="absolute top-0 left-[50%] w-px h-2 bg-gray-700" />
      </div>
      <div className="flex justify-between text-[9px] text-gray-700">
        <span>Weak (0)</span>
        <span>50%</span>
        <span>80%</span>
        <span>Strong (100%)</span>
      </div>
    </div>
  );
}

export default function UncertaintyPanel({ symbol }: Props) {
  const [stats, setStats] = useState<UncertaintyStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await getUncertainty(symbol);
      setStats(r.data);
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
    const id = setInterval(fetchStats, 60_000);
    return () => clearInterval(id);
  }, [symbol]);

  if (loading && !stats) {
    return (
      <div className="flex items-center justify-center h-32 text-gray-500 text-xs">
        Loading uncertainty data...
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-red-trade text-xs p-2">
        {error}
        <button onClick={fetchStats} className="ml-2 text-accent hover:underline">retry</button>
      </div>
    );
  }

  if (!stats) return null;

  const health = stats.calibration_health;
  const ratio =
    stats.rolling_brier !== null && stats.baseline_brier
      ? stats.rolling_brier / stats.baseline_brier
      : null;

  return (
    <div className="flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div
            className={clsx(
              "w-2.5 h-2.5 rounded-full",
              HEALTH_BG[health] ?? "bg-gray-600"
            )}
          />
          <span className={clsx("text-sm font-semibold", HEALTH_COLOR[health] ?? "text-gray-400")}>
            {HEALTH_LABEL[health] ?? health}
          </span>
        </div>
        <button
          onClick={fetchStats}
          disabled={loading}
          className="text-xs text-accent hover:underline disabled:opacity-40"
        >
          {loading ? "..." : "↻"}
        </button>
      </div>

      {/* 4-Layer reference */}
      <div className="bg-surface rounded p-2 flex flex-col gap-1 text-xs">
        <div className="text-[10px] text-gray-600 uppercase tracking-wider mb-1">
          Decision Pipeline
        </div>
        {[
          { n: 1, label: "Raw model probability", note: "direct predict_proba()" },
          { n: 2, label: "Calibrated probability", note: "isotonic / Platt map" },
          { n: 3, label: "Tradeable confidence", note: "× degradation factor" },
          { n: 4, label: "Action", note: "buy / sell / abstain" },
        ].map(({ n, label, note }) => (
          <div key={n} className="flex items-start gap-2">
            <span className="text-[10px] text-gray-600 w-4 text-right flex-none pt-0.5">
              L{n}
            </span>
            <div>
              <span className="text-gray-300">{label}</span>
              <span className="text-gray-600 ml-1">— {note}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Key metrics */}
      <div className="flex flex-col gap-1.5">
        <StatRow
          label="Observations (window)"
          value={stats.window_size.toString()}
          colorClass={stats.window_size >= 30 ? "text-white" : "text-yellow-400"}
        />
        {stats.rolling_brier !== null && (
          <StatRow
            label="Rolling Brier score"
            value={stats.rolling_brier.toFixed(4)}
            colorClass={
              stats.rolling_brier < 0.2
                ? "text-green-trade"
                : stats.rolling_brier < 0.25
                ? "text-yellow-400"
                : "text-red-trade"
            }
          />
        )}
        {stats.baseline_brier !== null && (
          <StatRow
            label="Baseline Brier (training)"
            value={stats.baseline_brier.toFixed(4)}
          />
        )}
        {ratio !== null && (
          <StatRow
            label="Brier ratio (rolling/base)"
            value={ratio.toFixed(2)}
            colorClass={
              ratio <= 1.1 ? "text-green-trade" : ratio <= 1.5 ? "text-yellow-400" : "text-red-trade"
            }
          />
        )}
        {stats.ece_recent !== null && (
          <StatRow
            label="ECE (recent)"
            value={`${(stats.ece_recent * 100).toFixed(1)}%`}
            colorClass={
              stats.ece_recent < 0.05
                ? "text-green-trade"
                : stats.ece_recent < 0.10
                ? "text-yellow-400"
                : "text-red-trade"
            }
          />
        )}
      </div>

      {/* Degradation gauge */}
      <DegradationGauge factor={stats.degradation_factor} />

      {/* Reliability diagram */}
      {stats.reliability_diagram ? (
        <div>
          <div className="text-xs text-gray-500 mb-2">
            Reliability diagram
            <span className="text-gray-700 ml-1">— dashed = perfect calibration</span>
          </div>
          <div className="flex justify-center">
            <ReliabilityChart diagram={stats.reliability_diagram} />
          </div>
          <div className="flex gap-3 justify-center mt-1 text-[10px] text-gray-600">
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 border-t border-dashed border-green-trade/60" /> Perfect
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 border-t border-accent" /> Actual
            </span>
          </div>
        </div>
      ) : (
        <div className="text-xs text-gray-600 italic text-center py-2">
          Reliability diagram requires ≥ 20 observations
        </div>
      )}

      {/* ECE context */}
      {stats.ece_recent !== null && (
        <div className="text-xs text-gray-600 bg-surface rounded p-2">
          <span className="text-gray-500">ECE interpretation: </span>
          {stats.ece_recent < 0.05
            ? "Predictions are well-aligned with actual outcomes."
            : stats.ece_recent < 0.10
            ? "Minor miscalibration — predictions slightly over/underconfident."
            : "Significant miscalibration — confidence bands are widened accordingly."}
        </div>
      )}
    </div>
  );
}
