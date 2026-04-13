"use client";
import { useState, useEffect } from "react";
import clsx from "clsx";
import {
  getRegime,
  getRegimeHistory,
  getRegimeDistribution,
  RegimeContext,
  RegimeHistory,
  RegimeDistribution,
  RegimeName,
} from "@/lib/api";

interface Props {
  symbol: string;
}

// ─── Visual mappings ────────────────────────────────────────────────────────

const REGIME_COLOR: Record<string, string> = {
  trending_up:    "text-green-trade",
  trending_down:  "text-red-trade",
  mean_reverting: "text-yellow-400",
  high_volatility:"text-red-trade",
  low_volatility: "text-gray-400",
  liquidity_poor: "text-orange-400",
  event_risk:     "text-red-500",
  unknown:        "text-gray-600",
};

const REGIME_BG: Record<string, string> = {
  trending_up:    "bg-green-trade",
  trending_down:  "bg-red-trade",
  mean_reverting: "bg-yellow-500",
  high_volatility:"bg-red-trade",
  low_volatility: "bg-gray-500",
  liquidity_poor: "bg-orange-400",
  event_risk:     "bg-red-600",
  unknown:        "bg-gray-700",
};

const REGIME_LABEL: Record<string, string> = {
  trending_up:    "Trending ↑",
  trending_down:  "Trending ↓",
  mean_reverting: "Mean Rev.",
  high_volatility:"High Vol",
  low_volatility: "Low Vol",
  liquidity_poor: "Illiquid",
  event_risk:     "Event Risk",
  unknown:        "Unknown",
};

// ─── Sub-components ─────────────────────────────────────────────────────────

function RegimeBadge({ regime, suppressed }: { regime: string; suppressed: boolean }) {
  return (
    <div className="flex items-center gap-2">
      <div
        className={clsx("w-2 h-2 rounded-full flex-none", REGIME_BG[regime] ?? "bg-zinc-600")}
      />
      <span className={clsx("text-[11px] font-semibold tracking-wide", REGIME_COLOR[regime] ?? "text-zinc-400")}>
        {REGIME_LABEL[regime] ?? regime.replace(/_/g, " ")}
      </span>
      {suppressed && (
        <span className="text-[10px] px-1.5 py-0.5 rounded-[2px] border border-red-400/40 bg-red-400/10 text-red-400 font-semibold">
          NO TRADE
        </span>
      )}
    </div>
  );
}

function SignalBar({
  label,
  value,
  min = 0,
  max = 2,
  thresholdLow,
  thresholdHigh,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  thresholdLow?: number;
  thresholdHigh?: number;
}) {
  const clamped = Math.max(min, Math.min(max, value));
  const pct = ((clamped - min) / (max - min)) * 100;
  const color =
    thresholdHigh !== undefined && value > thresholdHigh
      ? "bg-red-trade"
      : thresholdLow !== undefined && value < thresholdLow
      ? "bg-yellow-500"
      : "bg-accent";

  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] text-zinc-500 w-24 flex-none truncate">{label}</span>
      <div className="flex-1 bg-border rounded-full h-1.5 relative">
        <div className={clsx("h-1.5 rounded-full", color)} style={{ width: `${pct}%` }} />
        {thresholdHigh !== undefined && (
          <div
            className="absolute top-0 w-px h-1.5 bg-red-400/60"
            style={{ left: `${((thresholdHigh - min) / (max - min)) * 100}%` }}
          />
        )}
        {thresholdLow !== undefined && (
          <div
            className="absolute top-0 w-px h-1.5 bg-amber-400/60"
            style={{ left: `${((thresholdLow - min) / (max - min)) * 100}%` }}
          />
        )}
      </div>
      <span className="text-[10px] text-zinc-200 font-mono w-10 text-right tabular-nums">{value.toFixed(2)}</span>
    </div>
  );
}

/** Timeline strip: one cell per bar, colored by regime */
function RegimeTimeline({ history }: { history: RegimeHistory["history"] }) {
  if (history.length === 0) return null;
  const CELL_W = 4;
  const H = 20;
  const W = history.length * CELL_W;

  // Group runs for tooltip performance
  return (
    <div>
      <div className="text-[10px] text-zinc-500 mb-1">
        Regime history — last {history.length} bars
      </div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="rounded overflow-hidden" style={{ height: H }}>
        {history.map((entry, i) => {
          const bg =
            entry.regime === "trending_up"    ? "#3fb950" :
            entry.regime === "trending_down"  ? "#f85149" :
            entry.regime === "mean_reverting" ? "#e3b341" :
            entry.regime === "high_volatility"? "#f85149" :
            entry.regime === "low_volatility" ? "#8b949e" :
            entry.regime === "liquidity_poor" ? "#fb8500" :
            entry.regime === "event_risk"     ? "#ff0000" :
            "#30363d";
          return (
            <rect
              key={i}
              x={i * CELL_W}
              y={0}
              width={CELL_W - 0.5}
              height={H}
              fill={bg}
              opacity={entry.suppressed ? 0.9 : 0.6}
            >
              <title>{entry.bar_open_time}: {entry.regime}</title>
            </rect>
          );
        })}
      </svg>
      {/* Legend */}
      <div className="flex flex-wrap gap-2 mt-1">
        {(
          [
            ["trending_up", "#3fb950", "Trend ↑"],
            ["trending_down", "#f85149", "Trend ↓"],
            ["mean_reverting", "#e3b341", "Mean Rev"],
            ["high_volatility", "#f85149", "High Vol"],
            ["low_volatility", "#8b949e", "Low Vol"],
            ["liquidity_poor", "#fb8500", "Illiquid"],
            ["event_risk", "#ff0000", "Event"],
          ] as [string, string, string][]
        ).map(([, color, label]) => (
          <span key={label} className="flex items-center gap-1 text-[10px] text-zinc-600">
            <span className="inline-block w-2 h-2 rounded-sm" style={{ background: color, opacity: 0.7 }} />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}

// Hex colors for regime distribution bars (avoids Tailwind class name extraction issues)
const REGIME_HEX: Record<string, string> = {
  trending_up:    "#34d399",
  trending_down:  "#f87171",
  mean_reverting: "#fbbf24",
  high_volatility:"#f87171",
  low_volatility: "#71717a",
  liquidity_poor: "#fb923c",
  event_risk:     "#ef4444",
  unknown:        "#3f3f46",
};

/** Bar chart of regime distribution */
function RegimeDistributionChart({ distribution }: { distribution: Partial<Record<RegimeName, number>> }) {
  const entries = Object.entries(distribution).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) return <div className="text-[10px] text-zinc-600 italic">No stored data yet</div>;

  return (
    <div className="flex flex-col gap-1">
      {entries.map(([regime, frac]) => (
        <div key={regime} className="flex items-center gap-2">
          <span className="text-[10px] text-zinc-500 w-20 truncate flex-none">
            {REGIME_LABEL[regime] ?? regime}
          </span>
          <div className="flex-1 bg-border rounded-full h-1.5">
            <div
              className="h-1.5 rounded-full"
              style={{
                width: `${(frac * 100).toFixed(1)}%`,
                backgroundColor: REGIME_HEX[regime] ?? "#58a6ff",
              }}
            />
          </div>
          <span className="text-[10px] text-zinc-500 w-8 text-right">
            {(frac * 100).toFixed(0)}%
          </span>
        </div>
      ))}
    </div>
  );
}

// ─── Per-regime threshold table ──────────────────────────────────────────────
function ThresholdTable({
  thresholds,
}: {
  thresholds: RegimeContext["thresholds"];
}) {
  const rows = Object.entries(thresholds).sort(([a], [b]) => {
    const order = [
      "trending_up", "trending_down", "mean_reverting", "low_volatility",
      "high_volatility", "liquidity_poor", "event_risk", "unknown",
    ];
    return order.indexOf(a) - order.indexOf(b);
  });

  return (
    <table className="w-full text-[10px] border-collapse">
      <thead>
        <tr className="text-zinc-600 border-b border-border">
          <th className="text-left py-1 font-normal">Regime</th>
          <th className="text-right py-1 font-normal">Conf.</th>
          <th className="text-right py-1 font-normal">Quality</th>
          <th className="text-right py-1 font-normal">Trade?</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([regime, t]) => (
          <tr key={regime} className="border-b border-border/40">
            <td className={clsx("py-0.5", REGIME_COLOR[regime] ?? "text-zinc-400")}>
              {REGIME_LABEL[regime] ?? regime}
            </td>
            <td className="text-right text-zinc-500 font-mono">{(t.confidence_threshold * 100).toFixed(0)}%</td>
            <td className="text-right text-zinc-500 font-mono">{t.min_signal_quality.toFixed(0)}</td>
            <td className={clsx("text-right font-medium font-mono", t.allow_trade ? "text-emerald-400" : "text-red-400")}>
              {t.allow_trade ? "✓" : "✗"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ─── Main component ──────────────────────────────────────────────────────────

export default function RegimePanel({ symbol }: Props) {
  const [ctx, setCtx] = useState<RegimeContext | null>(null);
  const [history, setHistory] = useState<RegimeHistory | null>(null);
  const [distribution, setDistribution] = useState<RegimeDistribution | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"current" | "history" | "thresholds">("current");

  const fetchAll = async () => {
    setLoading(true);
    setError(null);
    try {
      const [ctxRes, histRes, distRes] = await Promise.all([
        getRegime(symbol),
        getRegimeHistory(symbol, "5m", 100),
        getRegimeDistribution(symbol),
      ]);
      setCtx(ctxRes.data);
      setHistory(histRes.data);
      setDistribution(distRes.data);
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAll();
    const id = setInterval(fetchAll, 60_000);
    return () => clearInterval(id);
  }, [symbol]);

  if (loading && !ctx) {
    return (
      <div className="inst-panel">
        <div className="inst-header"><span className="inst-label">Regime</span></div>
        <div className="inst-body text-zinc-600 text-[11px]">Detecting regime…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="inst-panel">
        <div className="inst-header"><span className="inst-label">Regime</span></div>
        <div className="inst-body">
          <span className="text-red-400 text-[11px]">{error}</span>
          <button onClick={fetchAll} className="ml-2 text-[11px] text-accent hover:underline">retry</button>
        </div>
      </div>
    );
  }

  if (!ctx) return null;

  return (
    <div className="inst-panel">
      {/* Header */}
      <div className="inst-header">
        <RegimeBadge regime={ctx.regime} suppressed={ctx.suppressed} />
        <button
          onClick={fetchAll}
          disabled={loading}
          className="text-[11px] text-accent hover:underline disabled:opacity-40"
        >
          {loading ? "…" : "↻"}
        </button>
      </div>

      <div className="inst-body flex flex-col gap-3">
        {/* Description */}
        <div className="text-[11px] text-zinc-500 leading-relaxed">{ctx.description}</div>

        {/* Suppression callout */}
        {ctx.suppressed && ctx.suppress_reason && (
          <div className="text-[11px] bg-red-400/5 border border-red-400/20 rounded-[2px] px-2 py-1.5 text-red-400/80">
            Trading suppressed: {ctx.suppress_reason.replace(/_/g, " ")}
            {" · "}confidence threshold raised to {(ctx.confidence_threshold * 100).toFixed(0)}%
          </div>
        )}

        {/* Tabs */}
        <div className="flex gap-1 border-b border-border pb-1">
          {(["current", "history", "thresholds"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={clsx(
                "text-[10px] px-2 py-0.5 rounded-[2px] capitalize",
                tab === t ? "bg-accent/15 text-accent font-semibold" : "text-zinc-500 hover:text-zinc-200"
              )}
            >
              {t}
            </button>
          ))}
        </div>

        {/* Tab: current signals */}
        {tab === "current" && (
          <div className="flex flex-col gap-2">
            <div className="text-[10px] text-zinc-600 uppercase tracking-[0.10em]">Signals</div>
            <SignalBar
              label="ADX (strength)"
              value={ctx.signals.adx_proxy}
              min={0} max={60}
              thresholdHigh={25}
            />
            <SignalBar
              label="ATR ratio"
              value={ctx.signals.atr_ratio}
              min={0} max={3}
              thresholdHigh={1.5}
              thresholdLow={0.5}
            />
            <SignalBar
              label="Volume ratio"
              value={ctx.signals.volume_ratio}
              min={0} max={3}
              thresholdLow={0.25}
            />
            <SignalBar
              label="Range / ATR"
              value={ctx.signals.bar_range_ratio}
              min={0} max={3}
              thresholdLow={0.2}
            />

            <div className="grid grid-cols-2 gap-1 mt-1">
              <div className="bg-surface rounded-[2px] p-1.5">
                <div className="text-[10px] text-zinc-600">Trend dir</div>
                <div className={clsx(
                  "text-[11px] font-mono",
                  ctx.signals.trend_direction === "up" ? "text-emerald-400" :
                  ctx.signals.trend_direction === "down" ? "text-red-400" : "text-zinc-400"
                )}>
                  {ctx.signals.trend_direction === "up" ? "↑ Up" :
                   ctx.signals.trend_direction === "down" ? "↓ Down" : "→ Flat"}
                </div>
              </div>
              <div className="bg-surface rounded-[2px] p-1.5">
                <div className="text-[10px] text-zinc-600">Abnormal move</div>
                <div className={clsx(
                  "text-[11px] font-mono",
                  ctx.signals.is_abnormal_move ? "text-red-400 font-semibold" : "text-zinc-400"
                )}>
                  {ctx.signals.is_abnormal_move
                    ? `Yes (${ctx.signals.abnormal_move_sigma.toFixed(1)}σ)`
                    : `No (${ctx.signals.abnormal_move_sigma.toFixed(1)}σ)`}
                </div>
              </div>
              <div className="bg-surface rounded-[2px] p-1.5">
                <div className="text-[10px] text-zinc-600">Conf. threshold</div>
                <div className={clsx("text-[11px] font-mono", ctx.suppressed ? "text-red-400" : "text-zinc-200")}>
                  {(ctx.confidence_threshold * 100).toFixed(0)}%
                </div>
              </div>
              <div className="bg-surface rounded-[2px] p-1.5">
                <div className="text-[10px] text-zinc-600">Quality min</div>
                <div className="text-[11px] font-mono text-zinc-200">{ctx.min_signal_quality.toFixed(0)}</div>
              </div>
            </div>

            {/* Distribution (if available) */}
            {distribution && Object.keys(distribution.distribution).length > 0 && (
              <div className="mt-1">
                <div className="text-[10px] text-zinc-600 uppercase tracking-[0.10em] mb-1">
                  Distribution (stored history)
                </div>
                <RegimeDistributionChart distribution={distribution.distribution} />
              </div>
            )}
          </div>
        )}

        {/* Tab: history timeline */}
        {tab === "history" && (
          <div>
            {history && history.count > 0 ? (
              <RegimeTimeline history={history.history} />
            ) : (
              <div className="text-[11px] text-zinc-600 italic text-center py-4">
                No stored regime history yet.
                <br />
                Regime labels are saved at each inference call.
              </div>
            )}
          </div>
        )}

        {/* Tab: threshold table */}
        {tab === "thresholds" && (
          <div>
            <div className="text-[10px] text-zinc-600 mb-2">
              Per-regime trading thresholds. Suppressed regimes (✗) never generate trade ideas
              regardless of confidence.
            </div>
            <ThresholdTable thresholds={ctx.thresholds} />
          </div>
        )}
      </div>
    </div>
  );
}
