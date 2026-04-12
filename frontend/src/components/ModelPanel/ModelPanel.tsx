"use client";
import { useState, useEffect } from "react";
import { getPrediction, ModelPrediction } from "@/lib/api";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine } from "recharts";
import clsx from "clsx";

interface Props {
  symbol: string;
}

function SignalBadge({ signal }: { signal: string }) {
  const cls = {
    buy: "bg-green-trade/20 text-green-trade border-green-trade/50",
    sell: "bg-red-trade/20 text-red-trade border-red-trade/50",
    no_trade: "bg-gray-700/30 text-gray-400 border-gray-600",
  }[signal] ?? "bg-gray-700/30 text-gray-400 border-gray-600";

  return (
    <span className={clsx("px-3 py-1 rounded border text-sm font-semibold uppercase tracking-wide", cls)}>
      {signal.replace("_", " ")}
    </span>
  );
}

function ProbBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-400 w-16">{label}</span>
      <div className="flex-1 bg-border rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${(value * 100).toFixed(1)}%` }}
        />
      </div>
      <span className="text-xs text-white w-12 text-right">{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

export default function ModelPanel({ symbol }: Props) {
  const [prediction, setPrediction] = useState<ModelPrediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<{ time: number; prob_up: number; confidence: number }[]>([]);

  const fetchPrediction = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await getPrediction(symbol);
      setPrediction(r.data);
      setHistory((prev) => [
        ...prev.slice(-29),
        { time: r.data.timestamp, prob_up: r.data.prob_up, confidence: r.data.confidence },
      ]);
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPrediction();
    const id = setInterval(fetchPrediction, 60_000);
    return () => clearInterval(id);
  }, [symbol]);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-white">Model Prediction</h3>
        <button
          onClick={fetchPrediction}
          disabled={loading}
          className="text-xs text-accent hover:underline disabled:opacity-50"
        >
          {loading ? "Loading..." : "Refresh"}
        </button>
      </div>

      {error && <div className="text-red-trade text-xs">{error}</div>}

      {prediction && (
        <>
          <div className="flex items-center gap-3">
            <SignalBadge signal={prediction.trade_signal} />
            <span className="text-xs text-gray-400">
              Confidence: <span className="text-white">{(prediction.confidence * 100).toFixed(1)}%</span>
            </span>
            <span className="text-xs text-gray-400">
              Expected move: <span className="text-white">±{prediction.expected_move_pct.toFixed(2)}%</span>
            </span>
          </div>

          <div className="flex flex-col gap-1.5">
            <ProbBar label="Prob Up" value={prediction.prob_up} color="bg-green-trade" />
            <ProbBar label="Prob Down" value={prediction.prob_down} color="bg-red-trade" />
            <ProbBar label="Confidence" value={prediction.confidence} color="bg-accent" />
          </div>

          <div className="text-xs text-gray-600 mt-1">
            Model: {prediction.model_version} · Updated {new Date(prediction.timestamp * 1000).toLocaleTimeString()}
          </div>
        </>
      )}

      {history.length > 1 && (
        <div>
          <div className="text-xs text-gray-500 mb-1">Prob Up — last {history.length} refreshes</div>
          <ResponsiveContainer width="100%" height={80}>
            <LineChart data={history}>
              <XAxis dataKey="time" hide />
              <YAxis domain={[0, 1]} hide />
              <Tooltip
                formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
                contentStyle={{ background: "#161b22", border: "1px solid #21262d", fontSize: 11 }}
              />
              <ReferenceLine y={0.5} stroke="#21262d" strokeDasharray="3 3" />
              <Line type="monotone" dataKey="prob_up" stroke="#3fb950" dot={false} strokeWidth={1.5} />
              <Line type="monotone" dataKey="confidence" stroke="#58a6ff" dot={false} strokeWidth={1} strokeDasharray="4 2" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
