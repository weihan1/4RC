#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

usage() {
  cat <<'USAGE'
Usage: ./run_process_masks_animals.sh animal [animal ...] [-- extra process_masks.py args]

Examples:
  ./run_process_masks_animals.sh wolf sheep tiger
  ./run_process_masks_animals.sh wolf sheep -- --max-rgb-frames 50 --skip-existing

The script runs:
  python process_masks.py --animal <animal>

for each provided animal in sequence.
USAGE
}

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 1
fi

animals=()
extra_args=()
parsing_animals=true

while [[ $# -gt 0 ]]; do
  if [[ "$parsing_animals" == true ]]; then
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      --)
        parsing_animals=false
        shift
        continue
        ;;
      *)
        animals+=("$1")
        shift
        continue
        ;;
    esac
  fi

  extra_args+=("$1")
  shift
done

if [[ "${#animals[@]}" -eq 0 ]]; then
  echo "No animals provided." >&2
  usage >&2
  exit 1
fi

python_bin="${PYTHON_BIN:-python}"

for animal in "${animals[@]}"; do
  echo "Running process_masks.py for animal: ${animal}"
  "$python_bin" process_masks.py --animal "$animal" "${extra_args[@]}"
done
