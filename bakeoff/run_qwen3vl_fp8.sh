#!/usr/bin/env bash
# Run Qwen3-VL-8B-Instruct-FP8 on selected bakeoff pages with vLLM.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bakeoff/run_qwen3vl_fp8.sh        # config pages
#   CUDA_VISIBLE_DEVICES=1 bakeoff/run_qwen3vl_fp8.sh page2  # one page
#   CUDA_VISIBLE_DEVICES=1 bakeoff/run_qwen3vl_fp8.sh page1 page3
set -uo pipefail

cd "$(dirname "$0")/.."

PY="$(command -v python3 || command -v python || true)"
[ -n "$PY" ] || { echo "FATAL: no python found - activate your qwen3vl-fp8 env"; exit 1; }

PAGE_ARGS=()
[ "$#" -gt 0 ] && PAGE_ARGS=(--pages "$@")

echo "python: $PY"
echo "model:  qwen3vl_fp8"
if [ "$#" -gt 0 ]; then
  echo "pages:  $*"
else
  echo "pages:  config default"
fi
echo "gpu:    ${CUDA_VISIBLE_DEVICES:-all visible}"

"$PY" bakeoff/run.py --model qwen3vl_fp8 "${PAGE_ARGS[@]}" || exit $?
"$PY" bakeoff/score.py --model qwen3vl_fp8 "${PAGE_ARGS[@]}" || exit $?
"$PY" bakeoff/report.py || echo "(report skipped)"
