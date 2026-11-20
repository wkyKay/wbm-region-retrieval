#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v tmux >/dev/null
command -v python >/dev/null
python_bin=$(command -v python)
stamp=$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S%z)
session="region_j_${stamp//+/}"
out="artificial/f_region_structure_${stamp}"
mkdir -p f_region_structure/launch_logs
log="f_region_structure/launch_logs/${stamp}.log"
exit_file="f_region_structure/launch_logs/${stamp}.exitcode"
printf -v command '%q ' env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$python_bin" -u -m f_region_structure.run --stage all --device cuda --prepare-workers 4 --output-root "$out" "$@"
printf -v log_q '%q' "$log"
printf -v exit_q '%q' "$exit_file"
tmux new-session -d -s "$session" "${command} >${log_q} 2>&1; code=\$?; printf '%s\\n' \"\$code\" >${exit_q}; exit \"\$code\""
printf 'session=%s\noutput=%s\nlog=%s\n' "$session" "$out" "$log"
