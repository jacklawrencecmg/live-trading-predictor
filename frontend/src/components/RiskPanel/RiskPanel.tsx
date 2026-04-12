"use client";
import { useState, useEffect } from "react";
import { getRiskSummary, toggleKillSwitch, RiskSummary } from "@/lib/api";
import clsx from "clsx";

export default function RiskPanel() {
  const [risk, setRisk] = useState<RiskSummary | null>(null);
  const [toggling, setToggling] = useState(false);

  const load = async () => {
    const r = await getRiskSummary();
    setRisk(r.data);
  };

  useEffect(() => {
    load();
    const id = setInterval(load, 30_000);
    return () => clearInterval(id);
  }, []);

  const handleKillSwitch = async () => {
    if (!risk) return;
    setToggling(true);
    await toggleKillSwitch(!risk.kill_switch);
    await load();
    setToggling(false);
  };

  if (!risk) return null;

  const dailyPnlPct = risk.daily_pnl_pct * 100;
  const lossUsedPct = Math.min(Math.abs(dailyPnlPct) / (risk.max_daily_loss / risk.capital * 100) * 100, 100);

  return (
    <div className="flex flex-col gap-3">
      <h3 className="text-sm font-semibold text-white">Risk Controls</h3>

      {/* Kill switch */}
      <div className={clsx(
        "flex items-center justify-between p-2 rounded border",
        risk.kill_switch
          ? "border-red-trade/50 bg-red-trade/10"
          : "border-border bg-panel"
      )}>
        <div>
          <div className="text-xs font-semibold text-white">Kill Switch</div>
          <div className={clsx("text-xs", risk.kill_switch ? "text-red-trade" : "text-gray-500")}>
            {risk.kill_switch ? "ACTIVE — All trading halted" : "Inactive"}
          </div>
        </div>
        <button
          onClick={handleKillSwitch}
          disabled={toggling}
          className={clsx(
            "text-xs px-3 py-1 rounded font-semibold",
            risk.kill_switch
              ? "bg-green-trade text-surface"
              : "bg-red-trade text-white"
          )}
        >
          {risk.kill_switch ? "Resume" : "Halt"}
        </button>
      </div>

      {/* Daily loss meter */}
      <div>
        <div className="flex justify-between text-xs mb-1">
          <span className="text-gray-400">Daily P&L</span>
          <span className={clsx(risk.daily_pnl >= 0 ? "text-green-trade" : "text-red-trade")}>
            {risk.daily_pnl >= 0 ? "+" : ""}${risk.daily_pnl.toFixed(0)}
            {" "}({dailyPnlPct.toFixed(2)}%)
          </span>
        </div>
        <div className="bg-border rounded-full h-2">
          <div
            className={clsx("h-2 rounded-full transition-all", risk.daily_pnl < 0 ? "bg-red-trade" : "bg-green-trade")}
            style={{ width: `${lossUsedPct}%` }}
          />
        </div>
        <div className="text-xs text-gray-600 mt-0.5">
          Max loss: ${risk.max_daily_loss.toFixed(0)}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="bg-surface rounded p-2">
          <div className="text-gray-500">Capital</div>
          <div className="text-white">${risk.capital.toLocaleString("en-US", { maximumFractionDigits: 0 })}</div>
        </div>
        <div className="bg-surface rounded p-2">
          <div className="text-gray-500">Max Position</div>
          <div className="text-white">${risk.max_position_size.toLocaleString("en-US", { maximumFractionDigits: 0 })}</div>
        </div>
        <div className="bg-surface rounded p-2">
          <div className="text-gray-500">Cooldown</div>
          <div className="text-white">{risk.cooldown_minutes} min</div>
        </div>
        <div className="bg-surface rounded p-2">
          <div className="text-gray-500">Max Daily Loss</div>
          <div className="text-white">${risk.max_daily_loss.toFixed(0)}</div>
        </div>
      </div>
    </div>
  );
}
