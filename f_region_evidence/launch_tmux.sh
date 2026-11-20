#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
python="${EVIDENCE_PYTHON:-/home/wangkaiyan2024/miniconda3/envs/wbm/bin/python}"
stamp="$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S%z)"
output="$root/artificial/f_region_evidence_$stamp"
logs="$root/f_region_evidence/launch_logs"
session="f_region_evidence_${stamp//[^0-9A-Za-z]/_}"
mkdir -p "$logs"
script="$logs/$stamp.sh"
log="$logs/$stamp.log"
status="$logs/$stamp.exit"
{
  printf '#!/usr/bin/env bash\nset -uo pipefail\n'
  printf 'cd %q\n' "$root"
  printf 'export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0\n'
  printf '%q -u -m f_region_evidence.run --stage all --device cuda --prepare-workers 4 --workers 8 --output-root %q > %q 2>&1\n' "$python" "$output" "$log"
  printf 'status=$?\nprintf "%%s\\n" "$status" > %q\nexit "$status"\n' "$status"
} > "$script"
tmux new-session -d -s "$session" "bash $(printf '%q' "$script")"
printf 'session=%s\noutput=%s\nlog=%s\nstatus=%s\n' "$session" "$output" "$log" "$status"
