"use client";
import { useState, useEffect } from "react";
import { runBacktest, getBacktestResults, BacktestRequest, BacktestResult } from "@/lib/api";
import {
  ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, Tooltip, Line, LineChart, CartesianGrid
} from "recharts";
import clsx from "clsx";

interface Props {
  symbol: string;
}

function MetricCard({ label, value, unit = "" }: { label: string; value: number | null; unit?: string }) {
  return (
    <div className="bg-surface rounded p-2">
      <div className="text-xs text-gray-500">{label}</div>
      <div className="text-sm font-semibold text-white">
        {value != null ? `${(value * (unit === "%" ? 100 : 1)).toFixed(unit === "%" ? 1 : 4)}${unit}` : "—"}
      </div>
    </div>
  );
}

export default function BacktestPanel({ symbol }: Props) {
  const [results, setResults] = useState<BacktestResult[]>([]);
  const [running, setRunning] = useState(false);
  const [selected, setSelected] = useState<BacktestResult | null>(null);
  const [form, setForm] = useState<BacktestRequest>({
    symbol,
    interval: "5m",
    period: "60d",
    n_folds: 5,
    train_size: 200,
    test_size: 50,
    confidence_threshold: 0.60,
  });

  useEffect(() => {
    setForm((f) => ({ ...f, symbol }));
  }, [symbol]);

  useEffect(() => {
    getBacktestResults().then((r) => setResults(r.data));
  }, []);

  const handleRun = async () => {
    setRunning(true);
    try {
      const r = await runBacktest(form);
      setResults((prev) => [r.data, ...prev]);
      setSelected(r.data);
    } finally {
      setRunning(false);
    }
  };

  const calibData = selected?.calibration_data
    ? (selected.calibration_data.bin_centers as number[]).map((x: number, i: number) => ({
        predicted: x,
        actual: (selected.calibration_data.fraction_positive as number[])[i],
      }))
    : [];

  return (
    <div className="flex flex-col gap-4">
      {/* Config */}
      <div className="grid grid-cols-3 gap-2 text-xs">
        {[
          { key: "interval", label: "Interval", type: "select", options: ["1m", "5m", "15m", "1h", "1d"] },
          { key: "period", label: "Period", type: "select", options: ["30d", "60d", "1y", "2y"] },
          { key: "n_folds", label: "Folds", type: "number" },
          { key: "train_size", label: "Train size", type: "number" },
          { key: "test_size", label: "Test size", type: "number" },
          { key: "confidence_threshold", label: "Min conf.", type: "number", step: 0.01 },
        ].map(({ key, label, type, options, step }: any) => (
          <div key={key}>
            <label className="text-gray-500 block mb-0.5">{label}</label>
            {type === "select" ? (
              <select
                value={(form as any)[key]}
                onChange={(e) => setForm((f) => ({ ...f, [key]: e.target.value }))}
                className="bg-panel border border-border text-white text-xs px-1 py-0.5 rounded w-full"
              >
                {options!.map((o: string) => <option key={o} value={o}>{o}</option>)}
              </select>
            ) : (
              <input
                type="number"
                value={(form as any)[key]}
                step={step}
                onChange={(e) => setForm((f) => ({ ...f, [key]: Number(e.target.value) }))}
                className="bg-panel border border-border text-white text-xs px-1 py-0.5 rounded w-full"
              />
            )}
          </div>
        ))}
      </div>

      <button
        onClick={handleRun}
        disabled={running}
        className="bg-accent text-surface font-semibold text-sm px-4 py-1.5 rounded hover:bg-accent/80 disabled:opacity-40 w-full"
      >
        {running ? "Running walk-forward backtest..." : "Run Backtest"}
      </button>

      {selected && (
        <>
          <div className="grid grid-cols-4 gap-2">
            <MetricCard label="Accuracy" value={selected.accuracy} unit="%" />
            <MetricCard label="Brier Score" value={selected.brier_score} />
            <MetricCard label="Log Loss" value={selected.log_loss} />
            <MetricCard label="Mag MAE" value={selected.magnitude_mae} />
            <MetricCard label="Sharpe" value={selected.sharpe_ratio} />
            <MetricCard label="Total Return" value={selected.total_return} unit="%" />
            <MetricCard label="# Trades" value={selected.n_trades} />
            <MetricCard label="# Folds" value={selected.n_folds} />
          </div>

          {calibData.length > 1 && (
            <div>
              <div className="text-xs text-gray-500 mb-1">
                Calibration Curve (Brier={selected.calibration_data?.brier_score?.toFixed(4)})
              </div>
              <ResponsiveContainer width="100%" height={160}>
                <ScatterChart>
                  <CartesianGrid stroke="#21262d" strokeDasharray="3 3" />
                  <XAxis dataKey="predicted" type="number" domain={[0, 1]} tick={{ fontSize: 10 }} name="Predicted" />
                  <YAxis dataKey="actual" type="number" domain={[0, 1]} tick={{ fontSize: 10 }} name="Actual" />
                  <Tooltip
                    cursor={false}
                    contentStyle={{ background: "#161b22", border: "1px solid #21262d", fontSize: 11 }}
                    formatter={(v: number) => v.toFixed(3)}
                  />
                  {/* Perfect calibration line */}
                  <Line
                    data={[{ predicted: 0, actual: 0 }, { predicted: 1, actual: 1 }]}
                    type="linear"
                    dataKey="actual"
                    stroke="#21262d"
                    dot={false}
                    strokeDasharray="4 2"
                  />
                  <Scatter data={calibData} fill="#58a6ff" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )}

          {selected.fold_results && (
            <div>
              <div className="text-xs text-gray-500 mb-1">Per-fold results</div>
              <div className="overflow-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-gray-500">
                      <th className="text-left pb-1">Fold</th>
                      <th className="text-right pb-1">Acc</th>
                      <th className="text-right pb-1">Brier</th>
                      <th className="text-right pb-1">Sharpe</th>
                      <th className="text-right pb-1">Return</th>
                      <th className="text-right pb-1">Trades</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selected.fold_results.map((f: any) => (
                      <tr key={f.fold} className="border-t border-border/50">
                        <td className="py-0.5">{f.fold}</td>
                        <td className="text-right">{(f.accuracy * 100).toFixed(1)}%</td>
                        <td className="text-right">{f.brier_score.toFixed(4)}</td>
                        <td className="text-right">{f.sharpe_ratio.toFixed(2)}</td>
                        <td className={clsx("text-right", f.total_return >= 0 ? "text-green-trade" : "text-red-trade")}>
                          {(f.total_return * 100).toFixed(2)}%
                        </td>
                        <td className="text-right">{f.n_trades}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {results.length > 0 && !selected && (
        <div className="text-xs text-gray-500">
          {results.length} previous backtest{results.length > 1 ? "s" : ""} — run a new one to see results.
        </div>
      )}
    </div>
  );
}
