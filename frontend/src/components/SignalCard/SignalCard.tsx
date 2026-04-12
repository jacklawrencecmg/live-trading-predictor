"use client";
import { useState, useEffect } from "react";
import clsx from "clsx";
import { getSignal, SignalResponse } from "@/lib/api";

interface Props {
  symbol: string;
}

const HEALTH_COLOR: Record<string, string> = {
  good: "text-green-trade border-green-trade/40 bg-green-trade/10",
  fair: "text-yellow-400 border-yellow-400/40 bg-yellow-400/10",
  degraded: "text-red-trade border-red-trade/40 bg-red-trade/10",
  unknown: "text-gray-500 border-gray-600 bg-gray-700/20",
};

const REGIME_COLOR: Record<string, string> = {
  trending_up: "text-green-trade",
  trending_down: "text-red-trade",
  mean_reverting: "text-yellow-400",
  high_volatility: "text-red-trade",
  low_volatility: "text-gray-400",
  unknown: "text-gray-600",
};

function HealthBadge({ health }: { health: string }) {
  return (
    <span
      className={clsx(
        "text-xs px-1.5 py-0.5 rounded border font-medium",
        HEALTH_COLOR[health] ?? HEALTH_COLOR.unknown
      )}
    >
      {health}
    </span>
  );
}

function ProbLayer({
  label,
  value,
  sublabel,
  color = "bg-accent",
  band,
}: {
  label: string;
  value: number;
  sublabel?: string;
  color?: string;
  band?: [number, number];
}) {
  const pct = (value * 100).toFixed(1);
  const bandWidth = band ? ((band[1] - band[0]) * 100).toFixed(1) : null;
  return (
    <div className="flex items-center gap-2">
      <div className="w-28 flex-none">
        <div className="text-xs text-gray-400 leading-tight">{label}</div>
        {sublabel && <div className="text-[10px] text-gray-600">{sublabel}</div>}
      </div>
      <div className="flex-1 relative">
        <div className="bg-border rounded-full h-2">
          <div
            className={clsx("h-2 rounded-full transition-all duration-500", color)}
            style={{ width: `${Math.min(100, value * 100).toFixed(1)}%` }}
          />
        </div>
        {band && (
          <div
            className="absolute top-0 h-2 bg-accent/20 rounded"
            style={{
              left: `${(band[0] * 100).toFixed(1)}%`,
              width: `${bandWidth}%`,
            }}
          />
        )}
      </div>
      <span className="text-xs text-white w-12 text-right tabular-nums">{pct}%</span>
    </div>
  );
}

function DegradationBar({ factor }: { factor: number }) {
  const color =
    factor >= 0.8 ? "bg-green-trade" : factor >= 0.5 ? "bg-yellow-500" : "bg-red-trade";
  const label =
    factor >= 0.8 ? "Strong" : factor >= 0.5 ? "Degraded" : "Weak";
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-500 w-28 flex-none">Degradation</span>
      <div className="flex-1 bg-border rounded-full h-1.5">
        <div className={clsx("h-1.5 rounded-full", color)} style={{ width: `${factor * 100}%` }} />
      </div>
      <span className={clsx("text-xs w-12 text-right", color.replace("bg-", "text-").replace("500","400"))}>
        {(factor * 100).toFixed(0)}% · {label}
      </span>
    </div>
  );
}

function QualityMeter({ score }: { score: number }) {
  const color =
    score >= 70 ? "bg-green-trade" : score >= 40 ? "bg-yellow-500" : "bg-red-trade";
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-400 w-28 flex-none">Signal quality</span>
      <div className="flex-1 bg-border rounded-full h-2">
        <div className={clsx("h-2 rounded-full", color)} style={{ width: `${score}%` }} />
      </div>
      <span className="text-xs text-white w-12 text-right">{score.toFixed(0)}/100</span>
    </div>
  );
}

export default function SignalCard({ symbol }: Props) {
  const [data, setData] = useState<SignalResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchSignal = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await getSignal(symbol);
      setData(r.data);
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSignal();
    const id = setInterval(fetchSignal, 60_000);
    return () => clearInterval(id);
  }, [symbol]);

  if (loading && !data) {
    return (
      <div className="flex items-center justify-center h-32 text-gray-500 text-xs">
        Loading signal...
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-red-trade text-xs p-2">
        {error}
        <button onClick={fetchSignal} className="ml-2 text-accent hover:underline">retry</button>
      </div>
    );
  }

  if (!data) return null;

  const { signal, trade_idea, prediction } = data;
  const isUp = signal.direction === "up";
  const isDown = signal.direction === "down";
  const isNoTrade = signal.direction === "no_trade";
  const dirColor = isUp ? "text-green-trade" : isDown ? "text-red-trade" : "text-gray-400";

  const band = prediction.confidence_band;
  const ece = prediction.ece_recent;

  return (
    <div className="flex flex-col gap-3">
      {/* Header: direction + action badge + refresh */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={clsx("text-lg font-bold uppercase tracking-wide", dirColor)}>
            {isNoTrade ? "No Trade" : isUp ? "↑ Bullish" : "↓ Bearish"}
          </span>
          <span
            className={clsx(
              "text-xs px-1.5 py-0.5 rounded border font-medium uppercase",
              prediction.action === "buy"
                ? "text-green-trade border-green-trade/40 bg-green-trade/10"
                : prediction.action === "sell"
                ? "text-red-trade border-red-trade/40 bg-red-trade/10"
                : "text-gray-500 border-gray-600 bg-gray-700/20"
            )}
          >
            {prediction.action}
          </span>
          <HealthBadge health={prediction.calibration_health} />
        </div>
        <button
          onClick={fetchSignal}
          disabled={loading}
          className="text-xs text-accent hover:underline disabled:opacity-40"
        >
          {loading ? "..." : "↻"}
        </button>
      </div>

      {/* Abstain reason */}
      {prediction.abstain_reason && (
        <div className="text-xs text-yellow-400/80 bg-yellow-400/5 border border-yellow-400/20 rounded px-2 py-1">
          Abstain: {prediction.abstain_reason.replace(/_/g, " ")}
        </div>
      )}

      {/* 4-layer probability stack */}
      <div className="flex flex-col gap-1.5 bg-surface rounded p-2">
        <div className="text-[10px] text-gray-600 uppercase tracking-wider mb-0.5">
          4-Layer Uncertainty
        </div>
        <ProbLayer
          label="Raw model"
          sublabel="Layer 1"
          value={prediction.prob_up}
          color="bg-gray-500"
        />
        <ProbLayer
          label="Calibrated"
          sublabel="Layer 2"
          value={prediction.calibrated_prob_up}
          color="bg-accent"
          band={band}
        />
        <ProbLayer
          label="Tradeable conf."
          sublabel="Layer 3"
          value={prediction.tradeable_confidence}
          color={
            prediction.tradeable_confidence >= 0.55
              ? "bg-green-trade"
              : prediction.tradeable_confidence <= 0.45
              ? "bg-red-trade"
              : "bg-gray-500"
          }
        />
        {ece !== null && ece !== undefined && (
          <div className="text-[10px] text-gray-600 pl-[7.5rem]">
            ECE band ±{(ece * 100).toFixed(1)}% · [{(band[0] * 100).toFixed(1)}%,{" "}
            {(band[1] * 100).toFixed(1)}%]
          </div>
        )}
      </div>

      {/* Degradation + quality */}
      <div className="flex flex-col gap-1.5">
        <DegradationBar factor={prediction.degradation_factor} />
        <QualityMeter score={signal.signal_quality_score} />
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-1 text-xs">
        <div className="bg-surface rounded p-1.5">
          <div className="text-gray-500">Expected Δ</div>
          <div className="text-white">±{prediction.expected_move_pct.toFixed(2)}%</div>
        </div>
        <div className="bg-surface rounded p-1.5">
          <div className="text-gray-500">Regime</div>
          <div className={clsx("truncate", REGIME_COLOR[signal.regime] ?? "text-gray-400")}>
            {signal.regime.replace(/_/g, " ")}
          </div>
        </div>
        <div className="bg-surface rounded p-1.5">
          <div className="text-gray-500">Volatility</div>
          <div
            className={
              signal.volatility_context === "expanding"
                ? "text-red-trade"
                : signal.volatility_context === "contracting"
                ? "text-gray-400"
                : "text-white"
            }
          >
            {signal.volatility_context}
          </div>
        </div>
      </div>

      {/* Brier / ECE info */}
      {(prediction.rolling_brier !== null || prediction.ece_recent !== null) && (
        <div className="flex gap-2 text-xs text-gray-500">
          {prediction.rolling_brier !== null && (
            <span>
              Brier:{" "}
              <span
                className={
                  prediction.rolling_brier < 0.2
                    ? "text-green-trade"
                    : prediction.rolling_brier < 0.25
                    ? "text-yellow-400"
                    : "text-red-trade"
                }
              >
                {prediction.rolling_brier.toFixed(3)}
              </span>
            </span>
          )}
          {prediction.ece_recent !== null && (
            <span>
              ECE:{" "}
              <span
                className={
                  prediction.ece_recent < 0.05
                    ? "text-green-trade"
                    : prediction.ece_recent < 0.10
                    ? "text-yellow-400"
                    : "text-red-trade"
                }
              >
                {(prediction.ece_recent * 100).toFixed(1)}%
              </span>
            </span>
          )}
          {!prediction.calibration_available && (
            <span className="text-gray-600">calibration: unavailable</span>
          )}
        </div>
      )}

      {/* Trade idea */}
      {!trade_idea.blocked ? (
        <div className="border border-green-trade/30 bg-green-trade/5 rounded p-2 text-xs">
          <div className="text-green-trade font-semibold mb-0.5">
            Idea: {trade_idea.strategy.replace(/_/g, " ")} @ Δ{trade_idea.target_delta.toFixed(2)}
          </div>
          <div className="text-gray-400">{trade_idea.rationale}</div>
        </div>
      ) : (
        <div className="border border-border bg-surface rounded p-2 text-xs">
          <div className="text-gray-500 font-semibold mb-0.5">No trade</div>
          <div className="text-gray-600">{trade_idea.rationale}</div>
        </div>
      )}

      {/* Explanation */}
      <div className="text-xs text-gray-500 leading-relaxed">{signal.explanation}</div>

      {/* Top features */}
      {Object.keys(signal.top_features).length > 0 && (
        <div>
          <div className="text-xs text-gray-600 mb-1">Key features</div>
          <div className="flex flex-col gap-0.5">
            {Object.entries(signal.top_features)
              .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
              .slice(0, 4)
              .map(([name, val]) => (
                <div key={name} className="flex items-center gap-1 text-xs">
                  <span
                    className={clsx(
                      "w-1.5 h-1.5 rounded-full flex-none",
                      val > 0 ? "bg-green-trade" : "bg-red-trade"
                    )}
                  />
                  <span className="text-gray-400 flex-1">{name.replace(/_/g, " ")}</span>
                  <span className={val > 0 ? "text-green-trade" : "text-red-trade"}>
                    {val > 0 ? "+" : ""}
                    {val.toFixed(3)}
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Audit footer */}
      <div className="text-[10px] text-gray-700 font-mono">
        snap:{prediction.feature_snapshot_id} · {prediction.model_version}
      </div>
    </div>
  );
}
