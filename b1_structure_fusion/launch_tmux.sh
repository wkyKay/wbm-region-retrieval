#!/usr/bin/env bash
# Run after activating the original CUDA Python environment.
set -euo pipefail

B1_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$B1_ROOT"
B1_STAGE="${1:-valid}"
if [[ $# -gt 0 ]]; then shift; fi
case "$B1_STAGE" in valid|test|all) ;; *) echo "Usage: bash b1_structure_fusion/launch_tmux.sh {valid|test|all} [run.py options]" >&2; exit 2 ;; esac
for B1_ARG in "$@"; do
  case "$B1_ARG" in
    --stage|--stage=*|--output-root|--output-root=*|--resume-root|--resume-root=*|--device|--device=*|--dry-run|--check-inputs)
      echo "The launcher owns stage/output/device; use run.py directly for resume or checks." >&2
      exit 2 ;;
  esac
done
command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 2; }
B1_PYTHON="$(command -v python)"
"$B1_PYTHON" -c 'import numpy, scipy, torch; assert torch.cuda.is_available(), "Activate the original PyTorch CUDA environment first"; print("CUDA:", torch.cuda.get_device_name(0))'
B1_STAMP="$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S%z)"
B1_SESSION="b1_${B1_STAGE}_${B1_STAMP//+/_}"
B1_OUTPUT="$B1_ROOT/artificial/b1_structure_${B1_STAGE}_${B1_STAMP}"
# Refuse existing identities/output instead of overwriting earlier experiments.
if [[ -e "$B1_OUTPUT" ]] || tmux has-session -t "$B1_SESSION" 2>/dev/null; then
  echo "Output/session already exists; wait one second or resume the prior run explicitly." >&2
  exit 2
fi
"$B1_PYTHON" b1_structure_fusion/run.py --stage "$B1_STAGE" --output-root "$B1_OUTPUT" --device cuda --prepare-workers 4 "$@" --dry-run >/dev/null
mkdir -p "$B1_OUTPUT"
tmux new-session -d -s "$B1_SESSION" -c "$B1_ROOT" \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash -c 'set -o pipefail; log=$1; shift; "$@" 2>&1 | tee "$log"' bash "$B1_OUTPUT/launch.log" \
  "$B1_PYTHON" -u "$B1_ROOT/b1_structure_fusion/run.py" \
  --stage "$B1_STAGE" --output-root "$B1_OUTPUT" --device cuda --prepare-workers 4 "$@"
printf 'tmux session: %s\noutput: %s\n' "$B1_SESSION" "$B1_OUTPUT"
printf 'Watch: tail -f %q\n' "$B1_OUTPUT/launch.log"
printf 'Attach: tmux attach -t %q\n' "$B1_SESSION"
