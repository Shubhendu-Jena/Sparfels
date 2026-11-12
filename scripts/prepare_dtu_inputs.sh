#!/usr/bin/env bash
# Build the 3-view DTU inputs (./dtu_0 and ./dtu_1) used by dtu_train.sh and eval_depth.sh
# from the DTU_TEST folder of the 3-view splits download.
#
#   set 0: views 23, 24, 33
#   set 1: views 42, 43, 44 (scan 37: views 1, 8, 9)
#
# Usage: bash scripts/prepare_dtu_inputs.sh [path/to/DTU_TEST]
set -euo pipefail

test_dir="${1:-../../Datasets_ICCV/DTU_TEST}"
[[ -d "$test_dir" ]] || { echo "ERROR: DTU_TEST not found: $test_dir"; exit 2; }

scenes=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)

for set_id in 0 1; do
  for scene in "${scenes[@]}"; do
    if [[ $set_id -eq 0 ]]; then
      views=(23 24 33)
    elif [[ $scene -eq 37 ]]; then
      views=(1 8 9)
    else
      views=(42 43 44)
    fi
    out="./dtu_${set_id}/scan${scene}"
    mkdir -p "$out/images" "$out/mask"
    for v in "${views[@]}"; do
      cp "$test_dir/scan${scene}/image/$(printf '%06d' "$v").png" "$out/images/"
      cp "$test_dir/scan${scene}/mask/$(printf '%03d' "$v").png" "$out/mask/"
    done
  done
done

echo "Done: ./dtu_0 and ./dtu_1 created from $test_dir"
