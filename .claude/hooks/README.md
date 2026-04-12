# Claude Code Hooks — options-research

## Overview

Two hook types are active, configured in `.claude/settings.json`:

| Hook | Trigger | Blocks? |
|------|---------|---------|
| `post-edit.sh` | After every `Edit` or `Write` tool call | No — warns only |
| `stop-validate.sh` | When Claude marks a task complete | Yes — exits 2 on failure |

---

## post-edit.sh — Per-file validation

Runs immediately after each file edit. Never blocks (exit 0 always). Outputs
inline warnings that Claude reads and can act on.

### Backend `.py` files

| Step | Tool | Action |
|------|------|--------|
| Format | `python -m black` | Formats in place |
| Sort imports | `python -m isort --profile black` | Formats in place |
| Lint | `python -m flake8 --max-line-length=120 --ignore=E203,W503` | Warns on violations |
| Unit test | `pytest <related_test_file>` | Warns on failures |
| TODO scan | `grep TODO\|FIXME` | Warns if in a risk-critical file |

**Test discovery** (`lib/find-test.sh`) — in order of preference:
1. `tests/test_<module_name>.py` — exact match on filename
2. `tests/test_<parent_package>.py` — match on containing directory
3. `tests/test_*<module_name>*.py` — glob fallback

Example: editing `app/services/risk_manager.py` → runs `tests/test_risk_manager.py`.

**Risk-critical files** (`lib/risk-files.sh`) — any TODO/FIXME in these files
produces a warning:
- `app/services/risk_manager.py`
- `app/governance/kill_switch.py`
- `app/services/paper_trader.py`
- `app/inference/inference_service.py`
- `app/inference/uncertainty.py`
- `app/decision/structure_evaluator.py`
- `app/decision/decision_engine.py`
- `app/core/config.py`

### Frontend `.ts` / `.tsx` files

| Step | Tool | Action |
|------|------|--------|
| Type check | `npx tsc --noEmit` | Warns on type errors (whole project) |
| Lint | `npx next lint --fix` | Applies auto-fixes, warns on remainder |

---

## stop-validate.sh — Completion gate

Runs when Claude finishes a task. Exit code 2 prevents task completion and
injects the failure output back into the conversation.

### Gate 1 — Leakage regression (always runs)

```
pytest tests/test_leakage.py -v --tb=short
```

Runs all 9 leakage regression tests (L1–L9). Any failure **blocks completion**.
These tests verify the shift-by-1 invariant, label alignment, warmup masking,
and feature/inference dimension consistency. They are fast (<5s) and must always
pass regardless of what was changed.

### Gate 2 — Calibration regression (runs when backend .py files changed)

```
pytest tests/test_model_training.py -v --tb=short -k "calibr or brier or ece or isoton"
```

Keyword-filtered to tests covering:
- Brier score thresholds
- ECE (Expected Calibration Error) bounds
- Calibration map fitting (isotonic regression)
- Degradation factor computation

Any failure **blocks completion**. The remaining model training tests
(non-calibration) are run but produce warnings only.

**Threshold source**: thresholds are encoded in the test assertions in
`tests/test_model_training.py`. To tighten a threshold, update the test.

---

## Validation log

All hook events are appended to `.claude/hooks/validation.log`:

```
2024-01-15T10:30:00Z EDIT   —      backend/app/services/risk_manager.py
2024-01-15T10:30:01Z FORMAT OK     backend/app/services/risk_manager.py  (black)
2024-01-15T10:30:01Z FORMAT OK     backend/app/services/risk_manager.py  (isort)
2024-01-15T10:30:02Z LINT   WARN   backend/app/services/risk_manager.py  (flake8 2 issue(s))
2024-01-15T10:30:03Z TEST   PASS   backend/tests/test_risk_manager.py    (12 passed)
2024-01-15T10:31:00Z STOP   PASS   test_leakage.py                       (9/9 passed)
2024-01-15T10:31:02Z STOP   PASS   test_model_training.py                (5 calibration tests passed)
2024-01-15T10:31:02Z STOP   PASS   all-gates
```

The log is append-only. Add `validation.log` to `.gitignore` if not already present.

---

## Prerequisites

The hooks assume:
- `python` (3.11+) is in PATH with `black`, `isort`, `flake8`, and `pytest` installed.
  The simplest setup: `pip install black isort flake8 pytest` in the backend venv.
- `node` and `npm` are in PATH for frontend hooks.
- `git` is in PATH (used by the Stop hook to detect changed files).

If running from within a virtual environment, activate it before starting
`claude` or ensure the venv's `python` is on PATH.

---

## Adding a new risk-critical file

Edit `.claude/hooks/lib/risk-files.sh` and add the repo-relative path to the
`RISK_CRITICAL_FILES` array.

## Adding a new calibration threshold

Add or modify a test in `backend/tests/test_model_training.py` with a name
containing `calibr`, `brier`, `ece`, or `isoton`. The Stop hook's keyword
filter will automatically pick it up.

## Disabling a gate temporarily

To skip the Stop gate during exploratory work, set `SKIP_STOP_GATE=1` in
your shell before running claude. (Add a check to `stop-validate.sh` if you
add this escape hatch — it is not currently implemented by design.)
