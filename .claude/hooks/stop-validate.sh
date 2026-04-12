#!/usr/bin/env bash
# stop-validate.sh — Stop hook: runs when Claude finishes a task.
#
# Gates:
#   1. Leakage tests     — ALWAYS run. Exit 2 (block) on any failure.
#   2. Calibration tests — Run when backend Python files changed. Exit 2 on failure.
#
# Exit codes:
#   0  All gates passed.
#   2  One or more gates failed → Claude cannot mark the task complete until fixed.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/paths.sh"

OVERALL_STATUS=0   # 0 = pass, 2 = block

# ─────────────────────────────────────────────────────────────────────────────
_header() { echo ""; echo -e "${BOLD}$1${RESET}"; }
_pass()   { echo -e "  ${GREEN}PASS${RESET}  $1"; }
_fail()   { echo -e "  ${RED}FAIL${RESET}  $1"; }
_info()   { echo -e "        $1"; }

# ─────────────────────────────────────────────────────────────────────────────
# Detect whether any backend Python files have been modified since last commit.
# Uses git status (includes staged, unstaged, and untracked .py files in backend/).
# ─────────────────────────────────────────────────────────────────────────────
backend_py_changed() {
  git -C "$REPO_ROOT" status --porcelain -- backend/ 2>/dev/null \
    | grep '\.py' \
    | grep -v '^?' \
    | wc -l | tr -d '[:space:]'
}

CHANGED_COUNT=$(backend_py_changed)

echo ""
echo -e "${BOLD}═══ Options Research — Validation Gate ═══${RESET}"
echo "  Backend .py files modified: ${CHANGED_COUNT}"
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# GATE 1: LEAKAGE TESTS (always)
# ═════════════════════════════════════════════════════════════════════════════
_header "Gate 1 / Leakage regression tests"

LEAKAGE_START=$(date +%s%N 2>/dev/null || date +%s)

LEAKAGE_OUT=$(cd "$BACKEND_DIR" && python -m pytest tests/test_leakage.py \
  -v --tb=short --no-header 2>&1)
LEAKAGE_EXIT=$?

LEAKAGE_END=$(date +%s%N 2>/dev/null || date +%s)

# Parse pass/fail counts
LEAKAGE_PASSED=$(echo "$LEAKAGE_OUT" | grep -cE '^\s*tests/test_leakage\.py::.*PASSED' || echo 0)
LEAKAGE_FAILED=$(echo "$LEAKAGE_OUT" | grep -cE '^\s*tests/test_leakage\.py::.*FAILED' || echo 0)
LEAKAGE_SUMMARY=$(echo "$LEAKAGE_OUT" | grep -E '^(FAILED|ERROR|[0-9]+ (passed|failed))' | tail -3)

if [ $LEAKAGE_EXIT -eq 0 ]; then
  _pass "test_leakage.py — ${LEAKAGE_PASSED} passed, ${LEAKAGE_FAILED} failed"
  log_entry "STOP" "test_leakage.py" "PASS" "${LEAKAGE_PASSED}/${LEAKAGE_PASSED} passed"
else
  _fail "test_leakage.py — ${LEAKAGE_FAILED} FAILED"
  echo ""
  echo "$LEAKAGE_OUT" | grep -E '(FAILED|AssertionError|assert )' | head -10 | sed 's/^/    /'
  echo ""
  _info "Full output:"
  echo "$LEAKAGE_OUT" | tail -30 | sed 's/^/    /'
  log_entry "STOP" "test_leakage.py" "FAIL" "${LEAKAGE_FAILED} failure(s)"
  OVERALL_STATUS=2
fi

# ═════════════════════════════════════════════════════════════════════════════
# GATE 2: CALIBRATION / MODEL TRAINING TESTS (when backend files changed)
# ═════════════════════════════════════════════════════════════════════════════
_header "Gate 2 / Calibration & model training tests"

if [ "$CHANGED_COUNT" -eq 0 ]; then
  echo "  SKIP  No backend Python files changed — skipping calibration gate"
  log_entry "STOP" "test_model_training.py" "SKIP" "no backend changes"
else
  # Run calibration-specific tests. The keyword filter selects tests whose
  # name contains "calibr", "brier", "ece", or "isoton" (calibration map).
  # Failures in these tests indicate a Brier score or calibration regression.
  CALIB_OUT=$(cd "$BACKEND_DIR" && python -m pytest tests/test_model_training.py \
    -v --tb=short --no-header \
    -k "calibr or brier or ece or isoton" \
    2>&1)
  CALIB_EXIT=$?

  CALIB_PASSED=$(echo "$CALIB_OUT" | grep -cE 'PASSED' || echo 0)
  CALIB_FAILED=$(echo "$CALIB_OUT" | grep -cE 'FAILED' || echo 0)
  CALIB_DESELECTED=$(echo "$CALIB_OUT" | grep -oE '[0-9]+ deselected' | head -1)

  if [ $CALIB_EXIT -eq 0 ]; then
    _pass "test_model_training.py (calibration) — ${CALIB_PASSED} passed, ${CALIB_FAILED} failed  ${CALIB_DESELECTED}"
    log_entry "STOP" "test_model_training.py" "PASS" "${CALIB_PASSED} calibration tests passed"
  else
    _fail "test_model_training.py (calibration) — ${CALIB_FAILED} FAILED"
    echo ""
    _info "Failing tests:"
    echo "$CALIB_OUT" | grep 'FAILED' | sed 's/^/    /'
    echo ""
    _info "Failure details:"
    echo "$CALIB_OUT" | tail -40 | sed 's/^/    /'
    echo ""
    echo -e "  ${RED}${BOLD}Calibration regression detected.${RESET}"
    echo "  A calibration test failure means a Brier score or ECE threshold was breached."
    echo "  Fix the regression before marking this task complete."
    log_entry "STOP" "test_model_training.py" "FAIL" "${CALIB_FAILED} calibration regression(s)"
    OVERALL_STATUS=2
  fi

  # ── Full model training suite (non-calibration) — warn only ─────────────
  # Run the remaining tests but do not block on them.
  OTHER_OUT=$(cd "$BACKEND_DIR" && python -m pytest tests/test_model_training.py \
    -q --tb=line --no-header \
    -k "not (calibr or brier or ece or isoton)" \
    2>&1) || true
  OTHER_EXIT=$?

  OTHER_FAILED=$(echo "$OTHER_OUT" | grep -cE '^FAILED' || echo 0)
  if [ $OTHER_EXIT -ne 0 ] && [ "$OTHER_FAILED" -gt 0 ]; then
    echo ""
    echo -e "  ${YELLOW}WARN${RESET}  ${OTHER_FAILED} non-calibration model test(s) failed (not blocking):"
    echo "$OTHER_OUT" | grep '^FAILED' | sed 's/^/    /'
    log_entry "STOP" "test_model_training.py" "WARN" "${OTHER_FAILED} non-calibration failure(s)"
  fi
fi

# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}═══════════════════════════════════════════${RESET}"

if [ $OVERALL_STATUS -eq 0 ]; then
  echo -e "${GREEN}${BOLD}All validation gates passed.${RESET}"
  log_entry "STOP" "all-gates" "PASS"
else
  echo -e "${RED}${BOLD}Validation blocked. Fix the failures above before completing.${RESET}"
  echo ""
  echo "Quick fix commands:"
  echo "  Leakage:     cd backend && python -m pytest tests/test_leakage.py -v"
  echo "  Calibration: cd backend && python -m pytest tests/test_model_training.py -v -k 'calibr or brier'"
  log_entry "STOP" "all-gates" "FAIL" "blocked"
fi

echo ""
exit $OVERALL_STATUS
