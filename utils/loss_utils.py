#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
from utils.pose_utils import get_camera_from_tensor
import kornia.geometry as KG
from torch import Tensor, nn
from typing import Literal, Optional
rng = np.random.RandomState(234)
_EPS = np.finfo(float).eps * 4.0
TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision
VALID_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")

def to_homogeneous(points):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1]+(1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1]+(1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError
    
def from_homogeneous(points):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / (points[..., -1:] + 1e-6)

def batch_project_to_other_img(kpi, di, Ki, Kj, T_itoj, return_depth=False):
    """
    Project pixels of one image to the other. 
    Args:
        kpi: BxNx2 coordinates in pixels of image i
        di: BxN, corresponding depths of image i
        Ki: intrinsics of image i, Bx3x3
        Kj: intrinsics of image j, Bx3x3
        T_itoj: Transform matrix from coordinate system of i to j, Bx4x4
        return_depth: Bool

    Returns:
        kpi_j: Pixels projection in image j, BxNx2
        di_j: Depth of the projections in image j, BxN
    """
    if len(di.shape) == len(kpi.shape):
        # di must be BxNx1
        di = di.squeeze(-1)
    kpi_3d_i = to_homogeneous(kpi) @ torch.inverse(Ki).transpose(-1, -2)
    kpi_3d_i = kpi_3d_i * di[..., None]  # non-homogeneous coordinates
    kpi_3d_j = from_homogeneous(
            to_homogeneous(kpi_3d_i) @ T_itoj.transpose(-1, -2))
    kpi_j = from_homogeneous(kpi_3d_j @ Kj.transpose(-1, -2))
    if return_depth:
        di_j = kpi_3d_j[..., -1]
        return kpi_j, di_j
    return kpi_j

def reprojection_loss(depth, intrinsic, intrinsic_src_list, pose, p_src_list, match_list, conf_list):
    num_pairs = len(match_list)

    # inlier_rates = []

    loss = torch.Tensor([0]).cuda()
    for idx, m in enumerate(match_list):
        idxs = m[:,:2]
        T_itoj = p_src_list[idx].inverse() @ pose
        
        kpi_j = batch_project_to_other_img(m[:, 0:2], depth[:, idxs[:,1].long(), idxs[:,0].long()].squeeze(0), intrinsic[:3, :3], intrinsic_src_list[idx][:3, :3], T_itoj, return_depth=False)

        weight=conf_list[idx]
        diff = kpi_j - m[:,2:]
        if m[:,2:].nelement() != 0:
            loss += ((weight[:,None]) * F.huber_loss(diff, torch.zeros_like(diff), reduction='none')).mean()
        else:
            num_pairs = num_pairs - 1

    if loss.sum() != 0:
        loss = loss / num_pairs
    return loss

def angular_dist_between_2_vectors(vec1, vec2):
    vec1_unit = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + TINY_NUMBER)
    vec2_unit = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + TINY_NUMBER)
    angular_dists = np.arccos(np.clip(np.sum(vec1_unit*vec2_unit, axis=-1), -1.0, 1.0))
    return angular_dists

def batched_angular_dist_rot_matrix(R1, R2):
    '''
    calculate the angular distance between two rotation matrices (batched)
    Args:
        R1: the first rotation matrix [N, 3, 3]
        R2: the second rotation matrix [N, 3, 3]
    Retugns: angular distance in radiance [N, ]
    '''
    assert R1.shape[-1] == 3 and R2.shape[-1] == 3 and R1.shape[-2] == 3 and R2.shape[-2] == 3
    return np.arccos(np.clip((np.trace(np.matmul(R2.transpose(0, 2, 1), R1), axis1=1, axis2=2) - 1) / 2.,
                             a_min=-1 + TINY_NUMBER, a_max=1 - TINY_NUMBER))

def get_nearest_pose_ids(tar_pose_c2w, ref_poses_c2w, num_select, 
                         tar_id=-1, angular_dist_method='vector',
                         scene_center=(0, 0, 0)):
    '''
    Args:
        tar_pose_c2w: target pose [3, 3]
        ref_poses_c2w: reference poses [N, 3, 3]
        num_select: the number of nearest views to select
        angular_dist_method: matrix, vector, dist, random
    Returns: the selected indices
    '''
    num_cams = len(ref_poses_c2w)
    if tar_id > 0:
        # the image to render is one of the training images
        num_select = min(num_select, num_cams-1)
    else:
        num_select = min(num_select, num_cams)

    batched_tar_pose = tar_pose_c2w[None, ...].repeat(num_cams, 0)

    if angular_dist_method == 'matrix':
        dists = batched_angular_dist_rot_matrix(batched_tar_pose[:, :3, :3], ref_poses_c2w[:, :3, :3])
    elif angular_dist_method == 'vector':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses_c2w[:, :3, 3]
        scene_center = np.array(scene_center)[None, ...]
        tar_vectors = tar_cam_locs - scene_center
        ref_vectors = ref_cam_locs - scene_center
        dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
    elif angular_dist_method == 'dist':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses_c2w[:, :3, 3]
        dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)
    elif angular_dist_method == 'random':
        dists = np.random.rand(num_cams)
    else:
        raise Exception('unknown angular distance calculation method!')

    if tar_id >= 0:
        assert tar_id < num_cams
        dists[tar_id] = 1e3  # make sure not to select the target id itself
    sorted_ids = np.argsort(dists)
    selected_ids = sorted_ids[:num_select]

    # print(angular_dists[selected_ids] * 180 / np.pi)
    return selected_ids

def sample_pose(poses_c2w_quat_t, id_self, pose_c2w_self):
    """Sample a new pose, as an interpolation between two training poses. 

    Args:
        poses_c2w (torch.Tensor): c2w poses, shape (B, 4, 4)
        id_self (int): index corresponding to pose_w2c_self
        pose_w2c_self (torch.Tensor): sampled pose within the training poses. 
    """
    # sample a pose between the two others
    poses_c2w = []
    for idx in range(poses_c2w_quat_t.shape[0]):
        poses_c2w.append(get_camera_from_tensor(poses_c2w_quat_t[idx]).inverse())

    poses_c2w = torch.stack(poses_c2w)
    poses_c2w_numpy = poses_c2w.detach().cpu().numpy() 
    id_other = get_nearest_pose_ids(tar_pose_c2w=pose_c2w_self.detach().cpu().numpy(), ref_poses_c2w=poses_c2w_numpy, 
                                    num_select=1, tar_id=id_self, angular_dist_method='vector')[0]
    w = np.random.rand()
    pose_c2w_self = pose_c2w_self.detach()
    pose_c2w_other = poses_c2w[id_other].detach()
    
    pose_c2w_at_unseen =  w * pose_c2w_self + (1 - w) * pose_c2w_other
    return pose_c2w_at_unseen

def detect_common_image_extension(scene_all, image_name, prefix=""):
    """
    Finds an extension that exists for this image in scene_all.matches.
    Example keys in matches: "name.png" or "1_name.jpg" depending on prefix.
    """
    for ext in VALID_IMAGE_EXTS:
        key = f"{prefix}{image_name}{ext}"
        if key in scene_all.matches:
            return ext
    raise RuntimeError(f"Could not find a consistent image extension for {prefix}{image_name}")

def sample_matches(
    pose_ref,
    uid_tgt,
    viewpoint_cam_tgt,
    viewpoint_cams,
    scene_all,
    gaussians_all,
    args=None,
    max_matches=1500,
):
    use_nvs = bool(args is not None and getattr(args, "nvs", False))

    # V2 behavior: eval uses "1_" prefix, train uses no prefix
    prefix = ""
    if use_nvs and getattr(args, "eval", False):
        prefix = "1_"

    # Detect extension once for the target (then assume consistent across all in matches dict)
    common_ext = detect_common_image_extension(scene_all, viewpoint_cam_tgt.image_name, prefix=prefix)

    def make_key(cam):
        return f"{prefix}{cam.image_name}{common_ext}"

    # ref pose
    pose = get_camera_from_tensor(pose_ref).inverse()

    # source selection policy
    if use_nvs:
        src_indices = range(min(3, len(viewpoint_cams)))  # V2: first 3
        src_cams = [viewpoint_cams[i] for i in src_indices]
    else:
        src_cams = list(viewpoint_cams)                  # V1: all

    ref_key = make_key(viewpoint_cam_tgt)

    match_list, conf_list, intrinsic_src_list, pose_src_list = [], [], [], []
    for cam_src in src_cams:
        if cam_src.uid == uid_tgt:
            continue

        src_key = make_key(cam_src)

        # V1 safe behavior: if missing, skip
        try:
            match = scene_all.matches[ref_key][src_key]
            confs = scene_all.confs[ref_key][src_key]
        except KeyError:
            continue

        # intrinsics
        W, H = cam_src.image_width, cam_src.image_height
        ndc2pix = torch.tensor(
            [[W / 2, 0, 0, (W - 1) / 2],
             [0, H / 2, 0, (H - 1) / 2],
             [0, 0, 0, 1]],
            dtype=torch.float32, device="cuda"
        ).T
        intrinsic_src_mat = (cam_src.projection_matrix @ ndc2pix)[:3, :3].T
        intrinsic_src = torch.eye(4, dtype=torch.float32, device="cuda")
        intrinsic_src[:3, :3] = intrinsic_src_mat

        # downsample
        if match.shape[0] > max_matches:
            idx = np.random.choice(match.shape[0], max_matches, replace=False)  # V1
            match = match[idx]
            confs = confs[idx]

        match = torch.from_numpy(match).float().cuda()
        confs = torch.from_numpy(confs).float().cuda()

        pose_src = get_camera_from_tensor(gaussians_all.get_RT(cam_src.uid)).inverse()

        match_list.append(match)
        conf_list.append(confs)
        pose_src_list.append(pose_src)
        intrinsic_src_list.append(intrinsic_src)

    # target intrinsic
    W, H = viewpoint_cam_tgt.image_width, viewpoint_cam_tgt.image_height
    ndc2pix = torch.tensor(
        [[W / 2, 0, 0, (W - 1) / 2],
         [0, H / 2, 0, (H - 1) / 2],
         [0, 0, 0, 1]],
        dtype=torch.float32, device="cuda"
    ).T
    intrinsic_mat = (viewpoint_cam_tgt.projection_matrix @ ndc2pix)[:3, :3].T
    intrinsic = torch.eye(4, dtype=torch.float32, device="cuda")
    intrinsic[:3, :3] = intrinsic_mat

    return intrinsic, pose, intrinsic_src_list, pose_src_list, match_list, conf_list

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

