#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

usage() {
  cat <<'USAGE'
Usage: ./run_demo_animals_parallel.sh [--gpus GPU_IDS]

Examples:
  ./run_demo_animals_parallel.sh --gpus 0,2
  GPU_IDS=1,3,5 ./run_demo_animals_parallel.sh

If no GPU list is provided, the script defaults to GPUs 0 and 1.
USAGE
}

log_dir="logs"
timestamp="$(date +"%Y%m%d_%H%M%S")"
run_log="${log_dir}/run_demo_animals_parallel_${timestamp}.log"
job_log="${log_dir}/run_demo_animals_parallel_${timestamp}.joblog"

animals=(
  bear boar cat cougar cow deer dog elephant fox goat hippo horse
  leopard moose panther pig rabbit racoon rhino sheep tiger wolf zebra
)

mkdir -p "$log_dir"

gpu_csv="${GPU_IDS:-0,1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      gpu_csv="${2:-}"
      shift 2
      ;;
    --gpus=*)
      gpu_csv="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

IFS=',' read -r -a gpu_ids <<< "$gpu_csv"

for i in "${!gpu_ids[@]}"; do
  gpu_ids[$i]="${gpu_ids[$i]//[[:space:]]/}"
  if [[ -z "${gpu_ids[$i]}" ]]; then
    echo "Invalid GPU list: '$gpu_csv'" >&2
    exit 1
  fi
done

num_jobs="${#gpu_ids[@]}"
if [[ "$num_jobs" -eq 0 ]]; then
  echo "No GPUs provided." >&2
  exit 1
fi

parallel_cmd='slot={%}; animal="{}"; animal_log_dir='"$(printf '%q' "$log_dir")"'/"$animal"; animal_log="$animal_log_dir"/run_demo_animals_parallel_'"$(printf '%q' "$timestamp")"'.log; mkdir -p "$animal_log_dir"; case "$slot" in'
for i in "${!gpu_ids[@]}"; do
  slot=$((i + 1))
  quoted_gpu="$(printf '%q' "${gpu_ids[$i]}")"
  parallel_cmd+=" ${slot}) gpu=${quoted_gpu} ;;"
done
parallel_cmd+=' *) echo "Unexpected parallel slot: $slot" >&2; exit 1 ;; esac; CUDA_VISIBLE_DEVICES="$gpu" python demo_uncropped.py --skip-existing --animal "$animal" 2>&1 | tee "$animal_log"'

echo "Writing run log to $run_log"
echo "Writing job log to $job_log"
echo "Writing per-animal logs to ${log_dir}/<animal>/run_demo_animals_parallel_${timestamp}.log"
echo "Using GPUs: ${gpu_ids[*]}"

parallel -j "$num_jobs" --line-buffer --joblog "$job_log" "$parallel_cmd" ::: "${animals[@]}" 2>&1 | tee "$run_log"
