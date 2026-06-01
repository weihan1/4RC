#!/usr/bin/env bash
set -euo pipefail

animals=(
  boar
  cat
  cougar
  cow
  deer
  dog
  elephant
  fox
  goat
  hippo
  horse
  leopard
  moose
  panther
  pig
  rabbit
  racoon
  rhino
  sheep
  tiger
  wolf
  zebra
)

for animal in "${animals[@]}"; do
  echo "doing animal $animal"
  python demo_only_depth.py --animal "${animal}" --skip-existing
done
