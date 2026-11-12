export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Dataset root:
#   - MVImgNet
#   - Tanks and Temples
#   - MipNeRF360
data_dir="${DATA_DIR:-../../Datasets_ICCV/collated_instantsplat_data/eval/Tanks}"

# Scene list (edit depending on the dataset)
scenes=(Ballroom Barn Church Family Francis Horse Ignatius Museum)

# Number of input views: 3, 6, or 12
n_view=${N_VIEW:-3}

# Training / finetuning iterations
iter=2000

for scene in "${scenes[@]}"; do
  python coarse_init_infer_mast3r_matches_nvs.py \
    --img_base_path "$data_dir/$scene/24_views" \
    --n_views "$n_view" \
    --focal_avg

  python train.py \
    --iterations "$iter" \
    -s "$data_dir/$scene/24_views/dust3r_${n_view}_views" \
    -m "./output_tnt/${n_view}_views/$scene" \
    -r 1 \
    --optim_pose \
    --nvs

  python init_test_pose.py \
    --img_base_path "$data_dir/$scene/24_views" \
    --n_views "$n_view" \
    --focal_avg

  python finetune_cams.py \
    --iterations "$iter" \
    -s "$data_dir/$scene/24_views/dust3r_${n_view}_views" \
    -m "./output_tnt/${n_view}_views/$scene" \
    --checkpoint_iterations "$iter" \
    --start_checkpoint "./output_tnt/${n_view}_views/$scene/chkpnt${iter}.pth" \
    -r 1 \
    --eval \
    --optim_pose \
    --nvs

  python render.py \
    -m "./output_tnt/${n_view}_views/$scene" \
    -s "$data_dir/$scene/24_views/dust3r_${n_view}_views" \
    --iteration "$iter" \
    --eval \
    --nvs

  python metrics.py \
    -m "./output_tnt/${n_view}_views/$scene" \
    --gt_pose_path "$data_dir/$scene/24_views" \
    --iter "$iter" \
    --n_views "$n_view"
done
