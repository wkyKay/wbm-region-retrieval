#!/usr/bin/env bash
set -euo pipefail
ABL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ABL_ROOT"
ABL_PYTHON="${B1_ABLATION_PYTHON:-$(command -v python)}"
for arg in "$@"; do
  case "$arg" in --output-root*|--resume-root*|--device*|--dry-run|--check-inputs|--audit-only)
    echo "Use run.py directly for output/resume/audit overrides" >&2; exit 2;; esac
done
"$ABL_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "Activate the original CUDA environment"'
ABL_STAMP="$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S%z)"
ABL_SESSION="b1_ablation_${ABL_STAMP//+/_}"
ABL_OUTPUT="$ABL_ROOT/artificial/b1_ablation_${ABL_STAMP}"
[[ ! -e "$ABL_OUTPUT" ]] || { echo "Existing output" >&2; exit 2; }
"$ABL_PYTHON" b1_ablation/run.py --output-root "$ABL_OUTPUT" --device cuda --dry-run "$@" >/dev/null
mkdir "$ABL_OUTPUT"
tmux new-session -d -s "$ABL_SESSION" -c "$ABL_ROOT" \
  env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash -c 'set -o pipefail; log=$1; shift; "$@" 2>&1 | tee "$log"' bash "$ABL_OUTPUT/launch.log" \
  "$ABL_PYTHON" -u b1_ablation/run.py --output-root "$ABL_OUTPUT" --device cuda "$@"
printf 'session=%s\noutput=%s\n' "$ABL_SESSION" "$ABL_OUTPUT"
