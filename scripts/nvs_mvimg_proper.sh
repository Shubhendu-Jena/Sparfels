export CUDA_VISIBLE_DEVICES=0

# Dataset root:
#   - MVImgNet
#   - Tanks and Temples
#   - MipNeRF360
data_dir="../../Datasets_ICCV/collated_instantsplat_data/eval/MVimgNet"

# Scene list (edit depending on the dataset)
scenes=(bench bicycle car chair ladder suv table)

# Number of input views: 3, 6, or 12
n_view=3

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
    -m "./output_mvimg/$scene" \
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
    -m "./output_mvimg/$scene" \
    --checkpoint_iterations "$iter" \
    --start_checkpoint "./output_mvimg/$scene/chkpnt${iter}.pth" \
    -r 1 \
    --eval \
    --optim_pose \
    --nvs

  python render.py \
    -m "./output_mvimg/$scene" \
    -s "$data_dir/$scene/24_views/dust3r_${n_view}_views" \
    --iteration "$iter" \
    --eval \
    --nvs

  python metrics.py \
    -m "./output_mvimg/$scene" \
    --gt_pose_path "$data_dir/$scene/24_views" \
    --iter "$iter" \
    --n_views "$n_view"
done
