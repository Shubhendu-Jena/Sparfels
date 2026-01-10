import os
import sys
import glob
import argparse
import numpy as np
import torch
from tqdm import tqdm

from utils.graphics_utils import getWorld2View
from utils.align_utils import (
    align_pose,
    align_ate_c2b_use_a2b,
    compute_ATE,
    compute_rpe
)
from scene.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    read_extrinsics_binary,
    read_intrinsics_binary,
    qvec2rotmat
)

np.bool = np.bool_  # Patch deprecated alias

# === GLOBAL VALID EXTENSIONS ===
VALID_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")


# === UTILITY ===
def get_first_extension(folder_path, valid_exts):
    all_files = glob.glob(os.path.join(folder_path, "*"))
    valid_files = [f for f in all_files if f.lower().endswith(valid_exts)]
    if not valid_files:
        raise FileNotFoundError(f"No valid files with extensions {valid_exts} found in {folder_path}")
    _, ext = os.path.splitext(valid_files[0])
    return ext


def read_cam_file_new(filename):
    with open(filename) as f:
        lines = [line.strip() for line in f.readlines()]
    extrinsics = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ").reshape((4, 4))
    intrinsics = np.fromstring(" ".join(lines[7:10]), dtype=np.float32, sep=" ").reshape((3, 3))
    intrinsics_ = np.eye(4, dtype=np.float32)
    intrinsics_[:3, :3] = intrinsics
    return intrinsics_, extrinsics


def read_cam_file(filename):
    intrinsics_, extrinsics = read_cam_file_new(filename)
    return intrinsics_ @ extrinsics


def camera_pose_metrics(root_dir, instance_dir, new_pose_dir, iteration):
    pose_path = os.path.join(new_pose_dir, f"pose/pose_{iteration}.npy")
    pose_opt = np.load(pose_path)

    cameras_pc_extrinsics, cameras_pc_poses, cameras_pc_intrinsics = [], [], []
    for vid in imgs_idx:
        intr, extr = read_cam_file_new(os.path.join(root_dir, f'cameras/{vid:08d}_cam.txt'))
        cameras_pc_extrinsics.append(extr)
        cameras_pc_poses.append(np.linalg.inv(extr))
        cameras_pc_intrinsics.append(intr)

    image_dir = os.path.join(instance_dir, "images")
    image_ext = get_first_extension(image_dir, VALID_IMAGE_EXTS)
    image_paths = sorted(glob.glob(os.path.join(image_dir, f"*{image_ext}")))

    try:
        cam_ext = read_extrinsics_text(os.path.join(instance_dir, "sparse/0/images.txt"))
        cam_int = read_intrinsics_text(os.path.join(instance_dir, "sparse/0/cameras.txt"))
    except:
        cam_ext = read_extrinsics_binary(os.path.join(instance_dir, "sparse/0/images.bin"))
        cam_int = read_intrinsics_binary(os.path.join(instance_dir, "sparse/0/cameras.bin"))

    cameras_mesh_extrinsics, cameras_mesh_intrinsics = [], []
    for idx, key in enumerate(cam_ext):
        extr = cam_ext[key]
        intr = cam_int[extr.camera_id]
        assert extr.name == os.path.basename(image_paths[idx])

        R = qvec2rotmat(extr.qvec).T
        T = np.array(extr.tvec)
        intrinsics_ = np.eye(4, dtype=np.float32)
        intrinsics_[0, 0], intrinsics_[1, 1] = intr.params[0], intr.params[1]
        intrinsics_[0, 2], intrinsics_[1, 2] = intr.params[2], intr.params[3]

        world_view_transform = getWorld2View(R, T)
        cameras_mesh_intrinsics.append(intrinsics_)
        cameras_mesh_extrinsics.append(world_view_transform)

    pose_ours = np.linalg.inv(pose_opt).copy()
    pose_ours = torch.from_numpy(pose_ours)

    poses_gt_np = np.stack([
        np.vstack([pose[:3, :], [0, 0, 0, 1]]) for pose in cameras_pc_poses
    ])
    poses_gt = torch.from_numpy(poses_gt_np)

    trans_gt, trans_est, _ = align_pose(
        poses_gt[:, :3, 3].numpy(),
        pose_ours[:, :3, 3].numpy()
    )
    poses_gt[:, :3, 3] = torch.from_numpy(trans_gt)
    pose_ours[:, :3, 3] = torch.from_numpy(trans_est)

    c2ws_est_aligned = align_ate_c2b_use_a2b(pose_ours, poses_gt)

    ate = compute_ATE(poses_gt.numpy(), c2ws_est_aligned.numpy())
    rpe_trans, rpe_rot = compute_rpe(poses_gt.numpy(), c2ws_est_aligned.numpy())

    print(f"\nRPE_trans: {rpe_trans * 100:.3f}, RPE_rot: {rpe_rot * 180 / np.pi:.3f}, ATE: {ate:.3f}\n")

    with open(os.path.join(new_pose_dir, "pose_eval.txt"), "w") as f:
        f.write(f"RPE_trans: {rpe_trans * 100:.4f}, RPE_rot: {rpe_rot * 180 / np.pi:.4f}, ATE: {ate:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=int, default=1)
    parser.add_argument("--dtu_input_dir", type=str, default="./sparsecraft_data/reconstruction/mvs_data/set0/scan55")
    parser.add_argument("--test_dir", type=str, default="./sparsecraft_data/reconstruction/mvs_data/set0/scan55")
    parser.add_argument("--mesh_dir", type=str, default="./exp_dir")
    parser.add_argument("--dataset_dir", type=str, default="/dataset/dtu_official/SampleSet/MVS_Data")
    parser.add_argument("--n_view", type=int, default=3)
    parser.add_argument("--set", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    scan = args.scan
    DTU_DIR = args.dtu_input_dir
    TEST_DIR = args.test_dir
    mesh_dir = args.mesh_dir

    if args.set == 0:
        view_list = [23, 24, 33, 22, 15, 34, 14, 32, 16, 35, 25]
    else:
        view_list = [42, 43, 44, 33, 34, 32, 45, 23, 41, 24, 31] if scan != 37 else [1, 8, 9, 33, 34, 32, 45, 23, 41, 24, 31]

    imgs_idx = view_list[:args.n_view]
    camera_pose_metrics(TEST_DIR, DTU_DIR, mesh_dir, args.iterations)
