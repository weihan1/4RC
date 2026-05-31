#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

usage() {
  cat <<'USAGE'
Usage: ./run_process_masks_animals_parallel.sh [--gpus 0,1,...]

Examples:
  ./run_process_masks_animals_parallel.sh --gpus 0,1
  GPU_IDS=2,3 ./run_process_masks_animals_parallel.sh
USAGE
}

gpu_csv="${GPU_IDS:-0,1}"
animals=(
  boar cat cougar cow deer dog elephant fox goat hippo horse leopard
  moose panther pig rabbit racoon rhino sheep tiger wolf zebra
)

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == --gpus=* ]]; then
  gpu_csv="${1#*=}"
  shift
elif [[ "${1:-}" == "--gpus" ]]; then
  gpu_csv="${2:-}"
  shift 2
fi

if ! command -v parallel >/dev/null 2>&1; then
  echo "GNU parallel is required but was not found in PATH." >&2
  exit 1
fi

IFS=',' read -r -a gpu_ids <<< "$gpu_csv"
num_jobs="${#gpu_ids[@]}"
python_bin="${PYTHON_BIN:-python}"
gpu_assignments=()

for ((i = 0; i < ${#animals[@]}; i++)); do
  gpu_assignments+=("${gpu_ids[$((i % num_jobs))]}")
done

echo "Using GPUs: ${gpu_ids[*]}"
echo "Animals: ${animals[*]}"

parallel -j "$num_jobs" --line-buffer --link \
  'echo "Running {2} on GPU {1}"; CUDA_VISIBLE_DEVICES="{1}" '"$python_bin"' process_masks.py --animal "{2}"' \
  ::: "${gpu_assignments[@]}" ::: "${animals[@]}"
