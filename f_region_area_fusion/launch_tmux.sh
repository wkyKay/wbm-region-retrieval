#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v tmux >/dev/null
python_bin=$(command -v python)
stamp=$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S%z)
session="region_area_${stamp//+/}"
valid_out="artificial/f_region_area_fusion_valid_${stamp}"
mkdir -p f_region_area_fusion/launch_logs
log="f_region_area_fusion/launch_logs/${stamp}.log"
exit_file="f_region_area_fusion/launch_logs/${stamp}.exitcode"
printf -v command '%q ' env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$python_bin" -u -m f_region_area_fusion.run --stage valid --workers 8 --output-root "$valid_out"
printf -v test_command '%q ' env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$python_bin" -u -m f_region_area_fusion.run --stage test --workers 8 --source-run "$valid_out"
printf -v log_q '%q' "$log"
printf -v exit_q '%q' "$exit_file"
tmux new-session -d -s "$session" "{ ${command} && ${test_command}; } >${log_q} 2>&1; code=\$?; printf '%s\\n' \"\$code\" >${exit_q}; exit \"\$code\""
printf 'session=%s\nvalid_output=%s\nlog=%s\n' "$session" "$valid_out" "$log"
