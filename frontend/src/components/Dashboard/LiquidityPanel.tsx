"use client";
import { useState, useEffect } from "react";
import { getOptionsChain, OptionsChain } from "@/lib/api";

function Row({ label, value, cls }: { label: string; value: React.ReactNode; cls?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-zinc-500 text-[11px] shrink-0">{label}</span>
      <span className={`text-[11px] font-mono tabular-nums text-right ${cls ?? "text-zinc-200"}`}>
        {value}
      </span>
    </div>
  );
}

function LiqBadge({ quality }: { quality: "good" | "fair" | "poor" | "unknown" }) {
  const map = {
    good:    "text-emerald-400 border-emerald-400/30 bg-emerald-400/5",
    fair:    "text-amber-400 border-amber-400/30 bg-amber-400/5",
    poor:    "text-red-400 border-red-400/30 bg-red-400/5",
    unknown: "text-zinc-500 border-zinc-700 bg-zinc-800/30",
  };
  return (
    <span className={`text-[10px] font-semibold tracking-wider uppercase px-1.5 py-0.5 border rounded-[2px] ${map[quality]}`}>
      {quality}
    </span>
  );
}

function spreadQuality(spreadPct: number): "good" | "fair" | "poor" {
  if (spreadPct < 0.04) return "good";
  if (spreadPct < 0.10) return "fair";
  return "poor";
}

export default function LiquidityPanel({ symbol }: { symbol: string }) {
  const [chain, setChain] = useState<OptionsChain | null>(null);
  const [loading, setLoading] = useState(true);
  const [isStale, setIsStale] = useState(false);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setIsStale(false);
    const load = () =>
      getOptionsChain(symbol)
        .then((r) => {
          if (!active) return;
          setChain(r.data);
          setLoading(false);
        })
        .catch(() => {
          if (active) { setLoading(false); setIsStale(true); }
        });
    load();
    const t = setInterval(load, 60_000);
    return () => { active = false; clearInterval(t); };
  }, [symbol]);

  // Compute ATM spread from chain rows
  const atm = (() => {
    if (!chain) return null;
    const underlying = chain.underlying_price;
    const sorted = [...chain.rows].sort(
      (a, b) => Math.abs(a.strike - underlying) - Math.abs(b.strike - underlying)
    );
    return sorted[0] ?? null;
  })();

  const atmCallSpread = atm?.call && atm.call.ask > 0 && atm.call.bid > 0
    ? atm.call.ask - atm.call.bid : null;
  const atmCallMid = atm?.call && atm.call.ask > 0
    ? (atm.call.bid + atm.call.ask) / 2 : null;
  const spreadPct = atmCallSpread != null && atmCallMid != null && atmCallMid > 0
    ? atmCallSpread / atmCallMid : null;

  const quality: "good" | "fair" | "poor" | "unknown" =
    spreadPct != null ? spreadQuality(spreadPct) : "unknown";

  // Total volume
  const totalCallVol = chain?.rows.reduce((s, r) => s + (r.call?.volume ?? 0), 0) ?? 0;
  const totalPutVol  = chain?.rows.reduce((s, r) => s + (r.put?.volume ?? 0), 0) ?? 0;
  const totalCallOI  = chain?.rows.reduce((s, r) => s + (r.call?.open_interest ?? 0), 0) ?? 0;
  const totalPutOI   = chain?.rows.reduce((s, r) => s + (r.put?.open_interest ?? 0), 0) ?? 0;

  return (
    <div className={isStale ? "inst-panel-stale" : "inst-panel"}>
      <div className="inst-header">
        <span className="inst-label">
          Liquidity
          {isStale && <span className="ml-2 text-amber-400 text-[10px]">STALE</span>}
        </span>
        <LiqBadge quality={quality} />
      </div>

      {loading && <div className="inst-body text-zinc-600 text-[11px]">Loading…</div>}
      {!loading && !chain && <div className="inst-body text-zinc-600 text-[11px]">No chain data</div>}

      {!loading && chain && (
        <div className="inst-body space-y-1.5">
          <Row
            label="Underlying price"
            value={`$${chain.underlying_price.toFixed(2)}`}
          />
          <Row
            label="ATM IV"
            value={chain.atm_iv != null ? `${(chain.atm_iv * 100).toFixed(1)}%` : "—"}
          />
          <Row
            label="IV rank"
            value={chain.iv_rank != null ? `${(chain.iv_rank * 100).toFixed(0)}th pct.` : "—"}
            cls={
              chain.iv_rank != null
                ? chain.iv_rank > 0.7 ? "text-red-400"
                  : chain.iv_rank > 0.4 ? "text-amber-400"
                  : "text-emerald-400"
                : "text-zinc-200"
            }
          />
          <div className="border-t border-border my-0.5" />
          {atm?.call && (
            <>
              <Row
                label="ATM call spread"
                value={atmCallSpread != null ? `$${atmCallSpread.toFixed(2)}` : "—"}
                cls={quality === "good" ? "text-emerald-400" : quality === "fair" ? "text-amber-400" : "text-red-400"}
              />
              <Row
                label="ATM spread pct."
                value={spreadPct != null ? `${(spreadPct * 100).toFixed(1)}%` : "—"}
                cls={quality === "good" ? "text-emerald-400" : quality === "fair" ? "text-amber-400" : "text-red-400"}
              />
            </>
          )}
          <Row
            label="P/C vol. ratio"
            value={chain.put_call_ratio != null ? chain.put_call_ratio.toFixed(2) : "—"}
            cls={
              chain.put_call_ratio != null
                ? chain.put_call_ratio > 1.2 ? "text-red-400"  // bearish skew
                  : chain.put_call_ratio < 0.7 ? "text-emerald-400"  // bullish skew
                  : "text-zinc-200"
                : "text-zinc-200"
            }
          />
          <div className="border-t border-border my-0.5" />
          <Row
            label="Total call vol."
            value={totalCallVol.toLocaleString()}
          />
          <Row
            label="Total put vol."
            value={totalPutVol.toLocaleString()}
          />
          <Row
            label="Total call OI"
            value={totalCallOI.toLocaleString()}
          />
          <Row
            label="Total put OI"
            value={totalPutOI.toLocaleString()}
          />
        </div>
      )}
    </div>
  );
}
