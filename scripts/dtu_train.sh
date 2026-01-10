#!/usr/bin/env bash
set -euo pipefail

# defaults
set_id=0
iter=1000
n_view=3

# flags: -s for set, -i for iterations (optional: -n for n_view)
while getopts "s:i:n:" flag; do
  case "${flag}" in
    s) set_id=${OPTARG} ;;
    i) iter=${OPTARG} ;;
    n) n_view=${OPTARG} ;;
  esac
done

export PYTHONPATH="${PYTHONPATH:-}:./utils"
export CUDA_VISIBLE_DEVICES=0

data_dir="./dtu_${set_id}"
out_root="./output_set_${set_id}"
scenes=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)

dataset_dir="../../Datasets_ICCV/SampleSet"
test_dir="../../Datasets_ICCV/DTU_TEST"

for scene in "${scenes[@]}"; do
  scene_name="scan${scene}"
  img_base="${data_dir}/${scene_name}"
  out_dir="${out_root}/${scene_name}"

  python coarse_init_infer_mast3r_matches.py --img_base_path "$img_base" --n_views "$n_view" --focal_avg
  python train.py --iterations "$iter" -s "$img_base" -m "$out_dir" --optim_pose --object_centric
  python render.py -m "$out_dir" -s "$img_base"
  python evaluation/clean_and_register_dtu_mesh_eval_depth_normal.py \
    --set "$set_id" --dataset_dir "$dataset_dir" --test_dir "$test_dir" \
    --dtu_input_dir "$img_base" --mesh_dir "$out_dir/train/ours_${iter}" --scan "$scene"
  python evaluation/metrics_pose.py \
    --iterations "$iter" --set "$set_id" --dataset_dir "$dataset_dir" --test_dir "$test_dir" \
    --dtu_input_dir "$img_base" --mesh_dir "$out_dir" --scan "$scene"
done
