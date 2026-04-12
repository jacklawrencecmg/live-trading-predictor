#!/usr/bin/env bash
# lib/risk-files.sh — canonical list of risk-critical and execution-critical files.
# Any TODO/FIXME found in these files triggers a warning.
# Sourced by post-edit.sh.

RISK_CRITICAL_FILES=(
  "backend/app/services/risk_manager.py"
  "backend/app/governance/kill_switch.py"
  "backend/app/services/paper_trader.py"
  "backend/app/inference/inference_service.py"
  "backend/app/inference/uncertainty.py"
  "backend/app/decision/structure_evaluator.py"
  "backend/app/decision/decision_engine.py"
  "backend/app/core/config.py"
)

# Returns 0 (true) if the given repo-relative path is risk-critical.
is_risk_critical() {
  local rel="$1"
  for f in "${RISK_CRITICAL_FILES[@]}"; do
    if [[ "$rel" == "$f" ]]; then
      return 0
    fi
  done
  return 1
}
