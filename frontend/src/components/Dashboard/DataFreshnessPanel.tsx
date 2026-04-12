"use client";
import { useState, useEffect } from "react";
import { getSignal } from "@/lib/api";

interface Source {
  name: string;
  ageSeconds: number | null;
  status: "live" | "fresh" | "stale" | "unknown";
  detail?: string;
}

function FreshnessRow({ src }: { src: Source }) {
  const dotColor = {
    live:    "bg-emerald-400",
    fresh:   "bg-emerald-400/60",
    stale:   "bg-amber-400",
    unknown: "bg-zinc-600",
  }[src.status];

  const label = {
    live:    "LIVE",
    fresh:   src.ageSeconds != null ? formatAge(src.ageSeconds) : "FRESH",
    stale:   src.ageSeconds != null ? `STALE (${formatAge(src.ageSeconds)})` : "STALE",
    unknown: "UNKNOWN",
  }[src.status];

  const textColor = {
    live:    "text-emerald-400",
    fresh:   "text-zinc-400",
    stale:   "text-amber-400",
    unknown: "text-zinc-600",
  }[src.status];

  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-zinc-500 text-[11px]">{src.name}</span>
      <div className="flex items-center gap-1.5">
        <span className={`status-dot ${dotColor}`} />
        <span className={`text-[10px] font-mono ${textColor}`}>{label}</span>
        {src.detail && (
          <span className="text-[10px] text-zinc-700">{src.detail}</span>
        )}
      </div>
    </div>
  );
}

function formatAge(secs: number): string {
  if (secs < 60)   return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m`;
  return `${(secs / 3600).toFixed(1)}h`;
}

function ageStatus(secs: number): "fresh" | "stale" {
  return secs < 300 ? "fresh" : "stale";
}

export default function DataFreshnessPanel({
  symbol,
  connected,
}: {
  symbol: string;
  connected: boolean;
}) {
  const [modelVersion, setModelVersion] = useState<string | null>(null);
  const [barTime, setBarTime] = useState<string | null>(null);
  const [modelAge, setModelAge] = useState<number | null>(null);
  const [tick, setTick] = useState(0);
  const [lastLoad, setLastLoad] = useState<number>(Date.now());

  useEffect(() => {
    let active = true;
    getSignal(symbol)
      .then((r) => {
        if (!active) return;
        setModelVersion(r.data.prediction.model_version ?? null);
        setBarTime(r.data.prediction.bar_open_time ?? null);
        setLastLoad(Date.now());
      })
      .catch(() => {});
    return () => { active = false; };
  }, [symbol]);

  // Age counter — update every 10s
  useEffect(() => {
    const t = setInterval(() => setTick((n) => n + 1), 10_000);
    return () => clearInterval(t);
  }, []);

  const modelAgeSecs = Math.floor((Date.now() - lastLoad) / 1000);
  const quoteStatus: Source["status"] = connected ? "live" : "stale";

  const sources: Source[] = [
    {
      name: "Quote / price",
      ageSeconds: connected ? 0 : null,
      status: quoteStatus,
      detail: connected ? "WebSocket" : "WebSocket disconnected",
    },
    {
      name: "Signal / model",
      ageSeconds: modelAgeSecs,
      status: ageStatus(modelAgeSecs),
      detail: modelVersion ?? undefined,
    },
    {
      name: "Options chain",
      ageSeconds: modelAgeSecs, // proxy — updated when signal was last loaded
      status: ageStatus(modelAgeSecs),
    },
  ];

  return (
    <div className="inst-panel">
      <div className="inst-header">
        <span className="inst-label">Data Freshness</span>
        <span className={`text-[10px] font-mono ${connected ? "text-emerald-400" : "text-amber-400"}`}>
          {connected ? "● WS LIVE" : "○ WS OFF"}
        </span>
      </div>
      <div className="inst-body space-y-1.5">
        {sources.map((src) => (
          <FreshnessRow key={src.name} src={src} />
        ))}

        {barTime && (
          <>
            <div className="border-t border-border my-1" />
            <div className="flex items-baseline justify-between">
              <span className="text-zinc-500 text-[11px]">Last bar</span>
              <span className="text-zinc-400 text-[10px] font-mono">{barTime}</span>
            </div>
          </>
        )}

        <div className="border-t border-border mt-1 pt-1.5">
          <p className="text-[10px] text-zinc-700 leading-relaxed">
            Signals auto-refresh every 30s. Options chain may be delayed up to 15 min during off-hours.
          </p>
        </div>
      </div>
    </div>
  );
}
