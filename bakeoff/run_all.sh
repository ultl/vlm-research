#!/usr/bin/env bash
# Build, run, and score each VLM in turn, then emit the scorecard.
#
# Sequential by design: one container at a time, so each model gets the whole
# GPU (safe on a constrained card). Default order is smallest-VRAM-first, so the
# four sub-3B models finish before the 7B (which may need a max_pixels cap on a
# <20 GB GPU). A model that fails is logged and skipped — the rest still run.
#
# Usage:
#   bakeoff/run_all.sh                       # all five, smallest first
#   bakeoff/run_all.sh paddleocr_vl mineru   # only these, in this order
#
# Run from anywhere; it cds to the repo root itself.
set -uo pipefail

cd "$(dirname "$0")/.."                       # repo root
COMPOSE="docker compose -f bakeoff/docker-compose.yml"
export BAKEOFF_CODE_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(paddleocr_vl mineru internvl3 deepseek_ocr qwen25vl)
fi

# --- preflight: fail fast on missing inputs ---
[ -f sample.pdf ]        || { echo "FATAL: sample.pdf missing — scp it to the repo root"; exit 1; }
[ -d fixtures/gt/cells ] || { echo "FATAL: fixtures/gt missing — scp fixtures/ to the repo root"; exit 1; }
command -v docker >/dev/null || { echo "FATAL: docker not found"; exit 1; }
# auto-detect the interpreter (micromamba/conda env may expose `python`, not `python3`)
PY="$(command -v python3 || command -v python || true)"
[ -n "$PY" ] || { echo "FATAL: no python found (python3 or python) — activate your env first"; exit 1; }

# forward GPU selection into the container (set CUDA_VISIBLE_DEVICES to pick a card)
DEV_ARGS=""
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && DEV_ARGS="-e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

echo "models: ${MODELS[*]}"
echo "python: $PY"
echo "gpu: ${CUDA_VISIBLE_DEVICES:-all visible}"
echo "code sha: $BAKEOFF_CODE_SHA"

DONE=(); FAILED=()      # explicit init — `declare -a` can trip `set -u` on empty arrays
for m in "${MODELS[@]}"; do
  echo
  echo "==================== $m ===================="
  if ! $COMPOSE build "$m";              then echo "[$m] BUILD failed"; FAILED+=("$m:build"); continue; fi
  if ! $COMPOSE run --rm $DEV_ARGS "$m"; then echo "[$m] RUN failed";   FAILED+=("$m:run");   continue; fi
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
