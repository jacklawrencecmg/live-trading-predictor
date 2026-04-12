#!/usr/bin/env bash
# post-edit.sh — PostToolUse hook for Edit and Write tool calls.
#
# Runs after every file edit:
#   • Logs the changed file
#   • Backend .py  → black + isort (format), flake8 (lint), related unit test
#   • Frontend .ts/.tsx → tsc --noEmit (type check), next lint
#   • Risk-critical files → TODO/FIXME warning
#
# Exit behaviour:
#   0  always — this hook warns but never blocks individual edits.
#      (Blocking happens at Stop time via stop-validate.sh.)

set -uo pipefail

# ── Bootstrap ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/paths.sh"
source "$SCRIPT_DIR/lib/risk-files.sh"
source "$SCRIPT_DIR/lib/find-test.sh"

# ── Parse payload ─────────────────────────────────────────────────────────────
PAYLOAD=$(cat)
RAW_PATH=$(printf '%s' "$PAYLOAD" | extract_file_path)

if [ -z "$RAW_PATH" ]; then
  exit 0
fi

FILE_PATH=$(normalize_path "$RAW_PATH")
REL=$(rel_path "$FILE_PATH")

# Skip if file is outside the repo (e.g. tmp files) or is a log/config
if [[ "$REL" == "$FILE_PATH" ]]; then
  # rel_path returned the same string → not under REPO_ROOT
  exit 0
fi

# ── Log the edit ──────────────────────────────────────────────────────────────
log_entry "EDIT" "$REL" "—"
echo "hook: edited $REL"

# ── Route by file type ────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════════════════
# BACKEND PYTHON
# ════════════════════════════════════════════════════════════════════════════
if [[ "$REL" == backend/*.py || "$REL" == backend/**/*.py ]]; then

  # ── Format: black ──────────────────────────────────────────────────────────
  echo "  → black..."
  if (cd "$BACKEND_DIR" && python -m black --quiet "$FILE_PATH" 2>&1); then
    log_entry "FORMAT" "$REL" "OK" "black"
  else
    log_entry "FORMAT" "$REL" "WARN" "black failed"
    echo -e "  ${YELLOW}WARN${RESET} black could not format $REL"
  fi

  # ── Format: isort ─────────────────────────────────────────────────────────
  if (cd "$BACKEND_DIR" && python -m isort --profile black --quiet "$FILE_PATH" 2>&1); then
    log_entry "FORMAT" "$REL" "OK" "isort"
  else
    log_entry "FORMAT" "$REL" "WARN" "isort failed"
    echo -e "  ${YELLOW}WARN${RESET} isort could not sort imports in $REL"
  fi

  # ── Lint: flake8 ───────────────────────────────────────────────────────────
  echo "  → flake8..."
  FLAKE_OUT=$(cd "$BACKEND_DIR" && python -m flake8 \
    --max-line-length=120 --ignore=E203,W503 \
    "$FILE_PATH" 2>&1) || true

  if [ -z "$FLAKE_OUT" ]; then
    log_entry "LINT" "$REL" "OK" "flake8 clean"
  else
    ISSUE_COUNT=$(echo "$FLAKE_OUT" | wc -l | tr -d ' ')
    log_entry "LINT" "$REL" "WARN" "flake8 $ISSUE_COUNT issue(s)"
    echo -e "  ${YELLOW}WARN${RESET} flake8 found $ISSUE_COUNT issue(s) in $REL:"
    echo "$FLAKE_OUT" | sed 's/^/    /'
  fi

  # ── TODO/FIXME check in risk-critical files ────────────────────────────────
  if is_risk_critical "$REL"; then
    TODO_HITS=$(grep -n "TODO\|FIXME\|HACK\|XXX" "$FILE_PATH" 2>/dev/null || true)
    if [ -n "$TODO_HITS" ]; then
      TODO_COUNT=$(echo "$TODO_HITS" | wc -l | tr -d ' ')
      log_entry "TODO" "$REL" "WARN" "$TODO_COUNT marker(s)"
      echo -e "  ${YELLOW}WARN${RESET} $TODO_COUNT TODO/FIXME marker(s) in risk-critical file $REL:"
      echo "$TODO_HITS" | sed 's/^/    /'
    fi
  fi

  # ── Run related unit test ──────────────────────────────────────────────────
  TEST_FILE=$(find_related_test "$FILE_PATH" "$BACKEND_DIR/tests")

  if [ -n "$TEST_FILE" ]; then
    TEST_REL=$(rel_path "$TEST_FILE")
    echo "  → running $TEST_REL..."
    TEST_OUT=$(cd "$BACKEND_DIR" && python -m pytest "$TEST_FILE" \
      -q --tb=short --no-header \
      --ignore=tests/test_leakage.py \
      2>&1) || true
    TEST_EXIT=$?

    if [ $TEST_EXIT -eq 0 ]; then
      PASS_LINE=$(echo "$TEST_OUT" | grep -E '^[0-9]+ passed' | head -1)
      log_entry "TEST" "$TEST_REL" "PASS" "${PASS_LINE:-tests passed}"
      echo -e "  ${GREEN}PASS${RESET} $TEST_REL  ${PASS_LINE}"
    else
      FAIL_LINE=$(echo "$TEST_OUT" | grep -E '^(FAILED|ERROR|[0-9]+ failed)' | head -3 | tr '\n' ' ')
      log_entry "TEST" "$TEST_REL" "FAIL" "$FAIL_LINE"
      echo -e "  ${RED}FAIL${RESET} $TEST_REL"
      echo "$TEST_OUT" | tail -20 | sed 's/^/    /'
      echo ""
      echo -e "  ${YELLOW}NOTE${RESET} Test failure logged. Leakage gate runs at task completion."
    fi
  else
    log_entry "TEST" "$REL" "SKIP" "no matching test file"
    echo "  → no test file found for $REL"
  fi

# ════════════════════════════════════════════════════════════════════════════
# FRONTEND TYPESCRIPT / TSX
# ════════════════════════════════════════════════════════════════════════════
elif [[ "$REL" == frontend/src/*.ts   || "$REL" == frontend/src/*.tsx   || \
        "$REL" == frontend/src/**/*.ts || "$REL" == frontend/src/**/*.tsx ]]; then

  # ── Type check: tsc --noEmit ───────────────────────────────────────────────
  echo "  → tsc --noEmit..."
  TSC_OUT=$(cd "$FRONTEND_DIR" && npx --yes tsc --noEmit 2>&1) || true
  TSC_EXIT=$?

  if [ $TSC_EXIT -eq 0 ]; then
    log_entry "LINT" "$REL" "OK" "tsc clean"
    echo -e "  ${GREEN}OK${RESET}   tsc: no type errors"
  else
    ERROR_COUNT=$(echo "$TSC_OUT" | grep -c "error TS" 2>/dev/null || echo "?")
    log_entry "LINT" "$REL" "WARN" "tsc $ERROR_COUNT error(s)"
    echo -e "  ${YELLOW}WARN${RESET} tsc: $ERROR_COUNT type error(s)"
    echo "$TSC_OUT" | grep "error TS" | head -10 | sed 's/^/    /'
    [ $(echo "$TSC_OUT" | grep -c "error TS") -gt 10 ] && echo "    ... (truncated)"
  fi

  # ── ESLint via next lint ───────────────────────────────────────────────────
  echo "  → next lint..."
  LINT_OUT=$(cd "$FRONTEND_DIR" && npx next lint --fix 2>&1) || true
  LINT_EXIT=$?

  if [ $LINT_EXIT -eq 0 ]; then
    log_entry "LINT" "$REL" "OK" "eslint clean"
    echo -e "  ${GREEN}OK${RESET}   eslint: clean"
  else
    log_entry "LINT" "$REL" "WARN" "eslint issues"
    echo -e "  ${YELLOW}WARN${RESET} eslint: issues found"
    echo "$LINT_OUT" | grep -v "^$" | head -15 | sed 's/^/    /'
  fi

# ════════════════════════════════════════════════════════════════════════════
# OTHER FILES (docs, config, etc.) — log only
# ════════════════════════════════════════════════════════════════════════════
else
  log_entry "EDIT" "$REL" "SKIP" "no validation for this file type"
fi

exit 0
