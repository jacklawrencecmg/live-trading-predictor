"use client";
import { useState, useEffect } from "react";
import { getSignal, SignalResponse } from "@/lib/api";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  Legend,
} from "recharts";
import clsx from "clsx";

interface Props {
  symbol: string;
}

interface HistoryPoint {
  time: number;
  prob_up: number;
  calibrated: number;
  tradeable: number;
}

function SignalBadge({ action }: { action: string }) {
  const cls: Record<string, string> = {
    buy: "bg-green-trade/20 text-green-trade border-green-trade/50",
    sell: "bg-red-trade/20 text-red-trade border-red-trade/50",
    abstain: "bg-gray-700/30 text-gray-400 border-gray-600",
  };
  return (
    <span
      className={clsx(
        "px-3 py-1 rounded border text-sm font-semibold uppercase tracking-wide",
        cls[action] ?? cls.abstain
      )}
    >
      {action}
    </span>
  );
}

function ProbBar({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-400 w-24">{label}</span>
      <div className="flex-1 bg-border rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${(value * 100).toFixed(1)}%` }}
        />
      </div>
      <span className="text-xs text-white w-12 text-right tabular-nums">
        {(value * 100).toFixed(1)}%
      </span>
    </div>
  );
}

function EceBadge({ ece }: { ece: number }) {
  const color =
    ece < 0.05 ? "text-green-trade border-green-trade/40 bg-green-trade/10"
    : ece < 0.10 ? "text-yellow-400 border-yellow-400/40 bg-yellow-400/10"
    : "text-red-trade border-red-trade/40 bg-red-trade/10";
  return (
    <span className={clsx("text-xs px-1.5 py-0.5 rounded border", color)}>
      ECE {(ece * 100).toFixed(1)}%
    </span>
  );
}

export default function ModelPanel({ symbol }: Props) {
  const [data, setData] = useState<SignalResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await getSignal(symbol);
      setData(r.data);
      const p = r.data.prediction;
      setHistory((prev) => [
        ...prev.slice(-29),
        {
          time: Date.now(),
          prob_up: p.prob_up,
          calibrated: p.calibrated_prob_up,
          tradeable: p.tradeable_confidence,
        },
      ]);
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 60_000);
    return () => clearInterval(id);
  }, [symbol]);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-white">Model Prediction</h3>
        <button
          onClick={fetchData}
          disabled={loading}
          className="text-xs text-accent hover:underline disabled:opacity-50"
        >
          {loading ? "Loading..." : "Refresh"}
        </button>
      </div>

      {error && <div className="text-red-trade text-xs">{error}</div>}

      {data && (() => {
        const p = data.prediction;
        return (
          <>
            <div className="flex items-center gap-2 flex-wrap">
              <SignalBadge action={p.action} />
              <span className="text-xs text-gray-400">
                Expected move:{" "}
                <span className="text-white">±{p.expected_move_pct.toFixed(2)}%</span>
              </span>
              {p.ece_recent !== null && <EceBadge ece={p.ece_recent} />}
            </div>

            {/* 4-layer bars */}
            <div className="flex flex-col gap-1.5">
              <div className="text-[10px] text-gray-600 uppercase tracking-wider">
                Probability layers
              </div>
              <ProbBar label="L1 Raw ↑" value={p.prob_up} color="bg-gray-500" />
              <ProbBar label="L1 Raw ↓" value={p.prob_down} color="bg-gray-600" />
              <ProbBar label="L2 Calibrated ↑" value={p.calibrated_prob_up} color="bg-accent" />
              <ProbBar
                label="L3 Tradeable"
                value={p.tradeable_confidence}
                color={
                  p.tradeable_confidence >= 0.55
                    ? "bg-green-trade"
                    : p.tradeable_confidence <= 0.45
                    ? "bg-red-trade"
                    : "bg-gray-500"
                }
              />
            </div>

            {/* Calibration health + degradation */}
            <div className="grid grid-cols-2 gap-1 text-xs">
              <div className="bg-surface rounded p-1.5">
                <div className="text-gray-500">Cal. health</div>
                <div
                  className={clsx(
                    "font-medium",
                    p.calibration_health === "good"
                      ? "text-green-trade"
                      : p.calibration_health === "fair"
                      ? "text-yellow-400"
                      : p.calibration_health === "degraded"
                      ? "text-red-trade"
                      : "text-gray-500"
                  )}
                >
                  {p.calibration_health}
                </div>
              </div>
              <div className="bg-surface rounded p-1.5">
                <div className="text-gray-500">Degradation</div>
                <div
                  className={
                    p.degradation_factor >= 0.8
                      ? "text-green-trade"
                      : p.degradation_factor >= 0.5
                      ? "text-yellow-400"
                      : "text-red-trade"
                  }
                >
                  {(p.degradation_factor * 100).toFixed(0)}%
                </div>
              </div>
              {p.rolling_brier !== null && (
                <div className="bg-surface rounded p-1.5">
                  <div className="text-gray-500">Brier score</div>
                  <div
                    className={
                      p.rolling_brier < 0.2
                        ? "text-green-trade"
                        : p.rolling_brier < 0.25
                        ? "text-yellow-400"
                        : "text-red-trade"
                    }
                  >
                    {p.rolling_brier.toFixed(4)}
                  </div>
                </div>
              )}
              <div className="bg-surface rounded p-1.5" title="±ECE (avg. miscalibration) — not a statistical confidence interval">
                <div className="text-gray-500">Cal. range (±ECE)</div>
                <div className="text-white tabular-nums">
                  [{(p.confidence_band[0] * 100).toFixed(0)},{" "}
                  {(p.confidence_band[1] * 100).toFixed(0)}]%
                </div>
              </div>
            </div>

            <div className="text-xs text-gray-600">
              Model: {p.model_version} · Bar: {p.bar_open_time}
            </div>
          </>
        );
      })()}

      {/* History chart: all 3 probability layers */}
      {history.length > 1 && (
        <div>
          <div className="text-xs text-gray-500 mb-1">
            Probability history — last {history.length} refreshes
          </div>
          <ResponsiveContainer width="100%" height={100}>
            <LineChart data={history} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
              <XAxis dataKey="time" hide />
              <YAxis domain={[0, 1]} hide />
              <Tooltip
                formatter={(v: number, name: string) => [
                  `${(v * 100).toFixed(1)}%`,
                  name === "prob_up" ? "Raw ↑" : name === "calibrated" ? "Calibrated" : "Tradeable",
                ]}
                contentStyle={{
                  background: "#161b22",
                  border: "1px solid #21262d",
                  fontSize: 10,
                }}
              />
              <ReferenceLine y={0.5} stroke="#21262d" strokeDasharray="3 3" />
              <Line
                type="monotone"
                dataKey="prob_up"
                stroke="#6e7681"
                dot={false}
                strokeWidth={1}
                strokeDasharray="3 2"
                name="prob_up"
              />
              <Line
                type="monotone"
                dataKey="calibrated"
                stroke="#58a6ff"
                dot={false}
                strokeWidth={1.5}
                name="calibrated"
              />
              <Line
                type="monotone"
                dataKey="tradeable"
                stroke="#3fb950"
                dot={false}
                strokeWidth={1}
                strokeDasharray="4 2"
                name="tradeable"
              />
            </LineChart>
          </ResponsiveContainer>
          <div className="flex gap-3 justify-center text-[10px] text-gray-600 mt-0.5">
            <span className="flex items-center gap-1">
              <span className="inline-block w-4 border-t border-dashed border-gray-500" /> Raw
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block w-4 border-t border-accent" /> Calibrated
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block w-4 border-t border-dashed border-green-trade" /> Tradeable
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
