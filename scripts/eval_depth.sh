#!/usr/bin/env bash
set -euo pipefail

# -------- Defaults (relative to repo root: Sparfels) --------
alignment="median"
iter=1000
set=0

# Relative paths from ./ (Sparfels)
dataset_dir="../../Datasets_ICCV/mvs_training/dtu/Depths_raw"
test_dir="../../Datasets_ICCV/DTU_TEST"

export PYTHONPATH="${PYTHONPATH:-}:./utils"
export CUDA_VISIBLE_DEVICES=0

# Parse flags: -a alignment, -i iter, -s set
while getopts a:i:s: flag; do
  case "${flag}" in
    a) alignment=${OPTARG} ;;
    i) iter=${OPTARG} ;;
    s) set=${OPTARG} ;;
  esac
done

# Resolve to absolute for clearer errors (if readlink -f exists)
if command -v readlink >/dev/null 2>&1; then
  dataset_dir_abs="$(readlink -f "$dataset_dir" || echo "$dataset_dir")"
  test_dir_abs="$(readlink -f "$test_dir" || echo "$test_dir")"
else
  dataset_dir_abs="$dataset_dir"
  test_dir_abs="$test_dir"
fi

# Basic validation (fail fast if paths are wrong)
[[ -d "$dataset_dir_abs" ]] || { echo "ERROR: dataset_dir not found: $dataset_dir_abs"; exit 2; }
[[ -d "$test_dir_abs" ]] || { echo "ERROR: test_dir not found: $test_dir_abs"; exit 2; }

data_dir="./dtu_${set}"
scenes=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)
meshdir="./output_set_${set}/"

# Optional: quick check a mask folder exists (warn only)
if [[ ! -d "${test_dir_abs}/scan24/mask" ]]; then
  echo "WARN: ${test_dir_abs}/scan24/mask not found; verify test_dir layout (should be .../DTU_TEST/scanXX/mask/)."
fi

# Run sequentially for clear errors
for scene in "${scenes[@]}"; do
  python evaluation/depth_normal_metrics.py \
    --dataset_dir "$dataset_dir_abs" \
    --test_dir "$test_dir_abs" \
    --mesh_dir "$meshdir" \
    --ckpt "$iter" \
    --scan "$scene" \
    --alignment "$alignment" \
    --set "$set" &
done

echo "Done. set=$set, ckpt=$iter, alignment=$alignment, meshdir=$meshdir"
