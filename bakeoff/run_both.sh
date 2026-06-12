#!/usr/bin/env bash
# Run a prompt-using image VLM through BOTH tracks, then the scorecard:
#   Track A  (run.py -> score.py)  — whole-document parse
#   Track B  (kie.py)              — field extraction (KIE)
# Native (uses the active python/env). Pin a GPU with CUDA_VISIBLE_DEVICES.
#
# Usage:
#   micromamba activate <env>
#   CUDA_VISIBLE_DEVICES=1 bakeoff/run_both.sh             # default: qwen3vl
#   CUDA_VISIBLE_DEVICES=1 bakeoff/run_both.sh internvl3   # any image VLM
#   CUDA_VISIBLE_DEVICES=1 bakeoff/run_both.sh qwen3vl page2
#
# Run from anywhere; it cds to the repo root itself.
set -uo pipefail

cd "$(dirname "$0")/.."                       # repo root
export BAKEOFF_CODE_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
PY="$(command -v python3 || command -v python || true)"
[ -n "$PY" ] || { echo "FATAL: no python found — activate your env first"; exit 1; }

MODEL="${1:-qwen3vl}"
PAGES=("${@:2}")
PAGE_ARGS=()
[ ${#PAGES[@]} -gt 0 ] && PAGE_ARGS=(--pages "${PAGES[@]}")

[ -f sample.pdf ]        || { echo "FATAL: sample.pdf missing — scp it to the repo root"; exit 1; }
[ -d fixtures/gt/cells ] || { echo "FATAL: fixtures/gt missing"; exit 1; }

echo "python: $PY"
echo "model:  $MODEL  (Track A parse + Track B kie)"
if [ ${#PAGES[@]} -gt 0 ]; then
  echo "pages:  ${PAGES[*]}"
else
  echo "pages:  config default"
fi
echo "gpu:    ${CUDA_VISIBLE_DEVICES:-all visible}"

FAILED=()

echo
echo "==================== $MODEL — Track A (parse) ===================="
if "$PY" bakeoff/run.py --model "$MODEL" "${PAGE_ARGS[@]}"; then
  "$PY" bakeoff/score.py --model "$MODEL" "${PAGE_ARGS[@]}" || FAILED+=("scoreA")
else
  FAILED+=("runA")
fi

echo
echo "==================== $MODEL — Track B (kie) ===================="
"$PY" bakeoff/kie.py --model "$MODEL" "${PAGE_ARGS[@]}" || FAILED+=("kieB")

echo
echo "==================== scorecard ===================="
"$PY" bakeoff/report.py || echo "(report skipped)"

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "done: $MODEL — both tracks ok"
else
  echo "done: $MODEL — failures: ${FAILED[*]}"
  echo "inspect runs/$MODEL/page1.meta.json -> .error (Track A) or rerun kie.py (Track B)"
fi
