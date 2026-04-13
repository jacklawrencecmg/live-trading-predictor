"use client";
import { useState, useEffect } from "react";
import {
  getGovernanceFreshness,
  GovernanceFreshnessStatus,
  FreshnessSourceStatus,
} from "@/lib/api";

function formatAge(secs: number): string {
  if (secs < 60)   return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m`;
  return `${(secs / 3600).toFixed(1)}h`;
}

function FreshnessRow({
  name,
  status,
}: {
  name: string;
  status: FreshnessSourceStatus;
}) {
  const isStale = status.is_stale;
  const ageLabel =
    status.age_seconds != null
      ? isStale
        ? `STALE (${formatAge(status.age_seconds)})`
        : formatAge(status.age_seconds)
      : "UNKNOWN";

  const dotCls = isStale ? "bg-amber-400" : "bg-emerald-400/60";
  const textCls = isStale ? "text-amber-400" : "text-zinc-400";

  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-zinc-500 text-[11px]">{name}</span>
      <div className="flex items-center gap-1.5">
        <span className={`status-dot ${dotCls}`} />
        <span className={`text-[10px] font-mono ${textCls}`}>{ageLabel}</span>
      </div>
    </div>
  );
}

function humanizeName(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function DataFreshnessPanel({
  symbol,
  connected,
}: {
  symbol: string;
  connected: boolean;
}) {
  const [data, setData] = useState<GovernanceFreshnessStatus | null>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let active = true;
    const load = () =>
      getGovernanceFreshness(symbol)
        .then((r) => { if (active) setData(r.data); })
        .catch(() => {});
    load();
    const t = setInterval(load, 30_000);
    return () => { active = false; clearInterval(t); };
  }, [symbol]);

  // Re-render every 15s to keep "age" labels current
  useEffect(() => {
    const t = setInterval(() => setTick((n) => n + 1), 15_000);
    return () => clearInterval(t);
  }, []);

  const anyStale = data?.any_stale ?? false;

  return (
    <div className={anyStale ? "inst-panel-stale" : "inst-panel"}>
      <div className="inst-header">
        <span className="inst-label">
          Data Freshness
          {anyStale && <span className="ml-2 text-amber-400 text-[10px]">STALE</span>}
        </span>
        <span className={`text-[10px] font-mono ${connected ? "text-emerald-400" : "text-amber-400"}`}>
          {connected ? "● WS" : "○ WS"}
        </span>
      </div>
      <div className="inst-body space-y-1.5">
        {/* WebSocket feed — derived from props */}
        <div className="flex items-center justify-between gap-2">
          <span className="text-zinc-500 text-[11px]">Quote / WebSocket</span>
          <div className="flex items-center gap-1.5">
            <span className={`status-dot ${connected ? "bg-emerald-400" : "bg-amber-400"}`} />
            <span className={`text-[10px] font-mono ${connected ? "text-emerald-400" : "text-amber-400"}`}>
              {connected ? "LIVE" : "DISCONNECTED"}
            </span>
          </div>
        </div>

        {/* Governance-layer source feeds */}
        {data && Object.entries(data.sources).map(([key, src]) => (
          <FreshnessRow key={key} name={humanizeName(key)} status={src} />
        ))}

        {!data && (
          <p className="text-zinc-600 text-[10px]">Freshness data unavailable</p>
        )}

        <div className="border-t border-border mt-1 pt-1.5">
          <p className="text-[10px] text-zinc-700 leading-relaxed">
            Stale feeds suppress inference; options chain may be delayed up to 15 min off-hours.
          </p>
        </div>
      </div>
    </div>
  );
}
