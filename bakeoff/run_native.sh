#!/usr/bin/env bash
# Run models with the ACTIVE python (no Docker) — for models whose deps live in
# the current env. Reuses your torch, so no image build / multi-GB re-download.
#
# Default: the two transformers VLMs that share a stack (internvl3, qwen25vl).
# Pass model names to run others — each needs ITS deps in the active env, so use
# a per-model micromamba env for the conflicting ones (see README "Running
# without Docker").
#
# Usage:
#   micromamba activate <env>
#   bakeoff/run_native.sh                 # internvl3 + qwen25vl
#   bakeoff/run_native.sh internvl3       # just one
#
# Run from anywhere; it cds to the repo root itself.
set -uo pipefail

cd "$(dirname "$0")/.."                       # repo root
export BAKEOFF_CODE_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

PY="$(command -v python3 || command -v python || true)"
[ -n "$PY" ] || { echo "FATAL: no python found — activate your env first"; exit 1; }

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(internvl3 qwen25vl)

# --- preflight ---
[ -f sample.pdf ]        || { echo "FATAL: sample.pdf missing — scp it to the repo root"; exit 1; }
[ -d fixtures/gt/cells ] || { echo "FATAL: fixtures/gt missing"; exit 1; }

echo "python: $PY"
echo "models: ${MODELS[*]}  (native, no Docker)"

DONE=(); FAILED=()      # explicit init — `declare -a` can trip `set -u` on empty arrays
for m in "${MODELS[@]}"; do
  echo
  echo "==================== $m (native) ===================="
  if ! "$PY" bakeoff/run.py   --model "$m"; then echo "[$m] RUN failed";   FAILED+=("$m:run");   continue; fi
  if ! "$PY" bakeoff/score.py --model "$m"; then echo "[$m] SCORE failed"; FAILED+=("$m:score"); continue; fi
  DONE+=("$m")
done

echo
echo "==================== scorecard ===================="
"$PY" bakeoff/report.py || echo "(report skipped — no scores yet)"

ok_list="none";  [ ${#DONE[@]}   -gt 0 ] && ok_list="${DONE[*]}"
bad_list="none"; [ ${#FAILED[@]} -gt 0 ] && bad_list="${FAILED[*]}"
echo
echo "finished. ok: ${ok_list}  |  failed: ${bad_list}"
[ ${#FAILED[@]} -eq 0 ] || echo "for a failure, inspect runs/<model>/page1.meta.json -> .error"
