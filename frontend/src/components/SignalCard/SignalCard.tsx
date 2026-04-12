"use client";
import { useState, useEffect } from "react";
import axios from "axios";
import clsx from "clsx";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Signal {
  direction: string;
  probability: number;
  confidence: number;
  confidence_band: [number, number];
  signal_quality_score: number;
  volatility_context: string;
  regime: string;
  explanation: string;
  top_features: Record<string, number>;
}

interface TradeIdea {
  direction: string;
  strategy: string;
  target_delta: number;
  blocked: boolean;
  block_reason: string | null;
  rationale: string;
}

interface SignalResponse {
  symbol: string;
  prediction: {
    prob_up: number;
    prob_down: number;
    confidence: number;
    expected_move_pct: number;
    model_version: string;
    bar_open_time: string;
    feature_snapshot_id: string;
  };
  signal: Signal;
  trade_idea: TradeIdea;
}

interface Props {
  symbol: string;
}

function QualityMeter({ score }: { score: number }) {
  const color =
    score >= 70 ? "bg-green-trade" : score >= 40 ? "bg-yellow-500" : "bg-red-trade";
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-400 w-12">Quality</span>
      <div className="flex-1 bg-border rounded-full h-2">
        <div className={`h-2 rounded-full ${color}`} style={{ width: `${score}%` }} />
      </div>
      <span className="text-xs text-white w-8 text-right">{score.toFixed(0)}</span>
    </div>
  );
}

export default function SignalCard({ symbol }: Props) {
  const [data, setData] = useState<SignalResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetch = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await axios.get<SignalResponse>(`${API_URL}/api/signals/${symbol}`);
      setData(r.data);
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetch();
    const id = setInterval(fetch, 60_000);
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
        <button onClick={fetch} className="ml-2 text-accent hover:underline">retry</button>
      </div>
    );
  }

  if (!data) return null;

  const { signal, trade_idea, prediction } = data;
  const isUp = signal.direction === "up";
  const isDown = signal.direction === "down";
  const isNoTrade = signal.direction === "no_trade";

  const directionColor = isUp
    ? "text-green-trade"
    : isDown
    ? "text-red-trade"
    : "text-gray-400";

  const regimeColor: Record<string, string> = {
    trending_up: "text-green-trade",
    trending_down: "text-red-trade",
    mean_reverting: "text-yellow-400",
    high_volatility: "text-red-trade",
    low_volatility: "text-gray-400",
    unknown: "text-gray-600",
  };

  return (
    <div className="flex flex-col gap-3">
      {/* Direction + quality */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className={clsx(
              "text-lg font-bold uppercase tracking-wide",
              directionColor
            )}
          >
            {isNoTrade ? "No Trade" : isUp ? "↑ Bullish" : "↓ Bearish"}
          </span>
          {!isNoTrade && (
            <span className="text-xs text-gray-400">
              {(signal.probability * 100).toFixed(1)}%
            </span>
          )}
        </div>
        <button
          onClick={fetch}
          disabled={loading}
          className="text-xs text-accent hover:underline disabled:opacity-40"
        >
          {loading ? "..." : "↻"}
        </button>
      </div>

      {/* Quality meter */}
      <QualityMeter score={signal.signal_quality_score} />

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-1 text-xs">
        <div className="bg-surface rounded p-1.5">
          <div className="text-gray-500">Confidence</div>
          <div className="text-white">{(signal.confidence * 100).toFixed(0)}%</div>
        </div>
        <div className="bg-surface rounded p-1.5">
          <div className="text-gray-500">Expected Δ</div>
          <div className="text-white">±{prediction.expected_move_pct.toFixed(2)}%</div>
        </div>
        <div className="bg-surface rounded p-1.5">
          <div className="text-gray-500">Regime</div>
          <div className={clsx("truncate", regimeColor[signal.regime] || "text-gray-400")}>
            {signal.regime.replace("_", " ")}
          </div>
        </div>
      </div>

      {/* Trade idea */}
      {!trade_idea.blocked ? (
        <div className="border border-green-trade/30 bg-green-trade/5 rounded p-2 text-xs">
          <div className="text-green-trade font-semibold mb-0.5">
            Idea: {trade_idea.strategy.replace("_", " ")} @ Δ{trade_idea.target_delta.toFixed(2)}
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

      {/* Auditability footer */}
      <div className="text-xs text-gray-700 font-mono">
        snap:{prediction.feature_snapshot_id} · {prediction.model_version}
      </div>
    </div>
  );
}
