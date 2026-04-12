"use client";
import { useState, useEffect } from "react";
import { getOptionsChain, getExpirations, OptionsChain, OptionsChainRow } from "@/lib/api";
import clsx from "clsx";

interface Props {
  symbol: string;
}

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

function fmt(v: number, d = 2) {
  return v.toFixed(d);
}

export default function OptionsChainTable({ symbol }: Props) {
  const [chain, setChain] = useState<OptionsChain | null>(null);
  const [expirations, setExpirations] = useState<string[]>([]);
  const [selectedExp, setSelectedExp] = useState<string>("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getExpirations(symbol).then((r) => {
      setExpirations(r.data);
      if (r.data.length > 0) setSelectedExp(r.data[0]);
    });
  }, [symbol]);

  useEffect(() => {
    if (!selectedExp) return;
    setLoading(true);
    getOptionsChain(symbol, selectedExp)
      .then((r) => setChain(r.data))
      .finally(() => setLoading(false));
  }, [symbol, selectedExp]);

  return (
    <div className="flex flex-col gap-2">
      {/* Header */}
      <div className="flex items-center gap-3">
        <select
          value={selectedExp}
          onChange={(e) => setSelectedExp(e.target.value)}
          className="bg-panel border border-border text-sm text-white px-2 py-1 rounded"
        >
          {expirations.map((e) => (
            <option key={e} value={e}>{e}</option>
          ))}
        </select>
        {chain && (
          <div className="flex gap-4 text-xs text-gray-400">
            <span>Spot: <span className="text-white">${fmt(chain.underlying_price)}</span></span>
            <span>ATM IV: <span className="text-white">{pct(chain.atm_iv)}</span></span>
            <span>IV Rank: <span className={clsx(chain.iv_rank > 0.5 ? "text-red-trade" : "text-green-trade")}>
              {pct(chain.iv_rank)}
            </span></span>
            <span>P/C: <span className="text-white">{fmt(chain.put_call_ratio)}</span></span>
          </div>
        )}
      </div>

      {loading && <div className="text-gray-500 text-xs">Loading chain...</div>}

      {chain && (
        <div className="overflow-auto max-h-80">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="text-gray-500 border-b border-border">
                <th className="text-right py-1 px-2">Bid</th>
                <th className="text-right px-2">Ask</th>
                <th className="text-right px-2">IV</th>
                <th className="text-right px-2">Δ</th>
                <th className="text-right px-2">Γ</th>
                <th className="text-right px-2">Θ</th>
                <th className="text-right px-2">V</th>
                <th className="text-right px-2">OI</th>
                <th className="text-center px-4 font-bold text-white">STRIKE</th>
                <th className="text-left px-2">Bid</th>
                <th className="text-left px-2">Ask</th>
                <th className="text-left px-2">IV</th>
                <th className="text-left px-2">Δ</th>
                <th className="text-left px-2">Γ</th>
                <th className="text-left px-2">Θ</th>
                <th className="text-left px-2">V</th>
                <th className="text-left px-2">OI</th>
              </tr>
            </thead>
            <tbody>
              {chain.rows.map((row) => {
                const atm = Math.abs(row.strike / chain.underlying_price - 1) < 0.005;
                return (
                  <tr
                    key={row.strike}
                    className={clsx(
                      "border-b border-border/50 hover:bg-border/30",
                      atm && "bg-accent/10"
                    )}
                  >
                    {/* Call side */}
                    {["bid", "ask", "iv", "delta", "gamma", "theta", "volume", "open_interest"].map((field) => (
                      <td key={field} className={clsx(
                        "text-right px-2 py-0.5",
                        row.call?.in_the_money ? "text-green-trade" : "text-gray-300"
                      )}>
                        {row.call
                          ? field === "iv" ? pct(row.call.iv)
                          : field === "volume" || field === "open_interest"
                          ? (row.call as any)[field].toLocaleString()
                          : fmt((row.call as any)[field], 4)
                          : "—"}
                      </td>
                    ))}

                    {/* Strike */}
                    <td className={clsx(
                      "text-center px-4 py-0.5 font-semibold",
                      atm ? "text-accent" : "text-white"
                    )}>
                      {fmt(row.strike)}
                    </td>

                    {/* Put side */}
                    {["bid", "ask", "iv", "delta", "gamma", "theta", "volume", "open_interest"].map((field) => (
                      <td key={field} className={clsx(
                        "text-left px-2 py-0.5",
                        row.put?.in_the_money ? "text-red-trade" : "text-gray-300"
                      )}>
                        {row.put
                          ? field === "iv" ? pct(row.put.iv)
                          : field === "volume" || field === "open_interest"
                          ? (row.put as any)[field].toLocaleString()
                          : fmt((row.put as any)[field], 4)
                          : "—"}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
