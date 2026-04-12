#!/usr/bin/env bash
# lib/paths.sh — shared path utilities sourced by hook scripts.
# Sourced, not executed. Always: source "$(dirname "${BASH_SOURCE[0]}")/lib/paths.sh"

# ── Repo root ────────────────────────────────────────────────────────────────
# Resolve from this file's location: hooks/lib/paths.sh → .claude/hooks/lib/
# Two levels up from lib/ → .claude/ → repo root
HOOKS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$HOOKS_DIR/.." && pwd)"

BACKEND_DIR="$REPO_ROOT/backend"
FRONTEND_DIR="$REPO_ROOT/frontend"
LOG_FILE="$HOOKS_DIR/validation.log"

# ── Normalize a path from JSON (Windows backslashes → forward slashes) ───────
normalize_path() {
  local p="${1//\\//}"   # backslash → forward slash
  echo "$p"
}

# ── Strip repo root prefix to produce a repo-relative path ───────────────────
rel_path() {
  local p
  p=$(normalize_path "$1")
  # Remove drive letter prefix if present (e.g. /c/Users/... or C:/Users/...)
  # Then strip REPO_ROOT
  echo "${p#$REPO_ROOT/}"
}

# ── Extract tool_input.file_path from the JSON payload on stdin ──────────────
extract_file_path() {
  python3 - <<'PYEOF'
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get("tool_input", {}).get("file_path", ""))
except Exception:
    print("")
PYEOF
}

# ── Append a structured line to the validation log ───────────────────────────
log_entry() {
  local level="$1"   # EDIT | FORMAT | LINT | TEST | TODO | STOP | ERROR
  local target="$2"  # file or test name
  local status="$3"  # PASS | WARN | FAIL | OK | SKIP
  local detail="${4:-}"
  printf '%s %-6s %-7s %s%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$level" \
    "$status" \
    "$target" \
    "${detail:+  ($detail)}" \
    >> "$LOG_FILE"
}

# ── Colour helpers (no-op when not in a terminal) ────────────────────────────
RED=''  YELLOW=''  GREEN=''  BOLD=''  RESET=''
if [ -t 1 ]; then
  RED='\033[0;31m' YELLOW='\033[0;33m' GREEN='\033[0;32m'
  BOLD='\033[1m'   RESET='\033[0m'
fi
