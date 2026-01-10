set -euo pipefail

export CUDA_VISIBLE_DEVICES=1        # GPU index to use (0,1,...)

ROOT="$(pwd)"                        # repo/project root (run script from here)
export PYTHONPATH="${ROOT}:${ROOT}/utils:${PYTHONPATH:-}"   # add project + utils/ to imports

data_dir="${ROOT}/mvimg/bench"       # <-- SET THIS: path to your dataset folder
n_view=3                             # number of input views (for coarse init)
iter=1000                            # training iterations

echo "Using data_dir: $data_dir"
test -d "$data_dir" || { echo "ERROR: data_dir does not exist: $data_dir"; exit 1; }

python coarse_init_infer_mast3r_matches.py --img_base_path "$data_dir" --n_views "$n_view" --focal_avg
python train.py --iterations "$iter" -s "$data_dir" -m "${ROOT}/output_custom" -r 1 --optim_pose
python render.py -m "${ROOT}/output_custom" -s "$data_dir"
