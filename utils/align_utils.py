import numpy as np
import torch
from scipy.spatial.transform import Rotation as RotLib
import utils.third_party.ATE.transformations as tfs
import utils.third_party.ATE.align_trajectory as align
import scipy

def rotation_error(pose_error):
    """Compute rotation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        rot_error (float): rotation error
    """
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5*(a+b+c-1.0)
    rot_error = np.arccos(max(min(d, 1.0), -1.0))
    return rot_error

def translation_error(pose_error):
    """Compute translation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        trans_error (float): translation error
    """
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    trans_error = np.sqrt(dx**2+dy**2+dz**2)
    return trans_error

def compute_rpe(gt, pred):
    trans_errors = []
    rot_errors = []
    for i in range(len(gt)-1):
        gt1 = gt[i]
        gt2 = gt[i+1]
        gt_rel = np.linalg.inv(gt1) @ gt2

        pred1 = pred[i]
        pred2 = pred[i+1]
        pred_rel = np.linalg.inv(pred1) @ pred2
        rel_err = np.linalg.inv(gt_rel) @ pred_rel
        
        trans_errors.append(translation_error(rel_err))
        rot_errors.append(rotation_error(rel_err))
    rpe_trans = np.mean(np.asarray(trans_errors))
    rpe_rot = np.mean(np.asarray(rot_errors))
    return rpe_trans, rpe_rot

def compute_ATE(gt, pred):
    """Compute RMSE of ATE
    Args:
        gt: ground-truth poses
        pred: predicted poses
    """
    errors = []

    for i in range(len(pred)):
        # cur_gt = np.linalg.inv(gt_0) @ gt[i]
        cur_gt = gt[i]
        gt_xyz = cur_gt[:3, 3] 

        # cur_pred = np.linalg.inv(pred_0) @ pred[i]
        cur_pred = pred[i]
        pred_xyz = cur_pred[:3, 3]

        align_err = gt_xyz - pred_xyz

        errors.append(np.sqrt(np.sum(align_err ** 2)))
    ate = np.sqrt(np.mean(np.asarray(errors) ** 2)) 
    return ate

def align_ate_c2b_use_a2b(traj_a, traj_b, traj_c=None, method='sim3'):
    """Align c to b using the sim3 from a to b.
    :param traj_a:  (N0, 3/4, 4) torch tensor
    :param traj_b:  (N0, 3/4, 4) torch tensor
    :param traj_c:  None or (N1, 3/4, 4) torch tensor
    :return:        (N1, 4,   4) torch tensor
    """
    device = traj_a.device
    if traj_c is None:
        traj_c = traj_a.clone()

    traj_a = traj_a.float().cpu().numpy()
    traj_b = traj_b.float().cpu().numpy()
    traj_c = traj_c.float().cpu().numpy()

    R_a = traj_a[:, :3, :3]  # (N0, 3, 3)
    t_a = traj_a[:, :3, 3]  # (N0, 3)
    quat_a = SO3_to_quat(R_a)  # (N0, 4)

    R_b = traj_b[:, :3, :3]  # (N0, 3, 3)
    t_b = traj_b[:, :3, 3]  # (N0, 3)
    quat_b = SO3_to_quat(R_b)  # (N0, 4)

    # This function works in quaternion.
    # scalar, (3, 3), (3, ) gt = R * s * est + t.
    s, R, t = alignTrajectory(t_a, t_b, quat_a, quat_b, method=method)

    # reshape tensors
    R = R[None, :, :].astype(np.float32)  # (1, 3, 3)
    t = t[None, :, None].astype(np.float32)  # (1, 3, 1)
    s = float(s)

    R_c = traj_c[:, :3, :3]  # (N1, 3, 3)
    t_c = traj_c[:, :3, 3:4]  # (N1, 3, 1)

    R_c_aligned = R @ R_c  # (N1, 3, 3)
    t_c_aligned = s * (R @ t_c) + t  # (N1, 3, 1)
    traj_c_aligned = np.concatenate([R_c_aligned, t_c_aligned], axis=2)  # (N1, 3, 4)

    # append the last row
    traj_c_aligned = convert3x4_4x4(traj_c_aligned)  # (N1, 4, 4)

    traj_c_aligned = torch.from_numpy(traj_c_aligned).to(device)
    return traj_c_aligned  # (N1, 4, 4)

def align_pose(pose1, pose2):
    mtx1 = np.array(pose1, dtype=np.double, copy=True)
    mtx2 = np.array(pose2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = scipy.linalg.orthogonal_procrustes(mtx1, mtx2)
    mtx2 = mtx2 * s

    return mtx1, mtx2, R

def convert3x4_4x4(input):
    """
    Args:
        input:  (N, 3, 4) or (3, 4) torch or np
    Returns: 
        (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor([[0,0,0,1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate([input, np.array([[0,0,0,1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output

def pad_poses(p):
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.], p[Ellipsis, :1, :4].shape)
  return np.concatenate([p[Ellipsis, :3, :4], bottom], axis=-2)

def pose_inverse_4x4(mat, use_inverse: bool=False):
    """
    Transforms world2cam into cam2world or vice-versa, without computing the inverse.
    Args:
        mat (torch.Tensor): pose matrix (B, 4, 4) or (4, 4)
    """
    # invert a camera pose
    out_mat = torch.zeros_like(mat)

    if len(out_mat.shape) == 3:
        # must be (B, 4, 4)
        out_mat[:, 3, 3] = 1
        R,t = mat[:, :3, :3],mat[:,:3, 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
        t_inv = (-R_inv@t)[..., 0]

        pose_inv = torch.cat([R_inv,t_inv[...,None]],dim=-1) # [...,3,4]

        out_mat[:, :3] = pose_inv
    else:
        out_mat[3, 3] = 1
        R,t = mat[:3, :3], mat[:3, 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
        t_inv = (-R_inv@t)[..., 0]
        pose_inv = torch.cat([R_inv,t_inv[...,None]],dim=-1) # [3,4]
        out_mat[:3] = pose_inv
    # assert torch.equal(out_mat, torch.inverse(mat))
    return out_mat

def rotation_distance(R1, R2, eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1@R2.transpose(-2,-1)
    trace = R_diff[...,0,0]+R_diff[...,1,1]+R_diff[...,2,2]
    angle = ((trace-1)/2).clamp(-1+eps,1-eps).acos_() # numerical stability near -1/+1
    return angle

def SO3_to_quat(R):
    """
    Args:
        R:  (N, 3, 3) or (3, 3) np
    Returns:   
        (N, 4, ) or (4, ) np
    """
    x = RotLib.from_matrix(R)
    quat = x.as_quat()
    return quat

def _getIndices(n_aligned, total_n):
    if n_aligned == -1:
        idxs = np.arange(0, total_n)
    else:
        assert n_aligned <= total_n and n_aligned >= 1
        idxs = np.arange(0, n_aligned)
    return idxs

def alignPositionYawSingle(p_es, p_gt, q_es, q_gt):
    '''
    calcualte the 4DOF transformation: yaw R and translation t so that:
        gt = R * est + t
    '''

    p_es_0, q_es_0 = p_es[0, :], q_es[0, :]
    p_gt_0, q_gt_0 = p_gt[0, :], q_gt[0, :]
    g_rot = tfs.quaternion_matrix(q_gt_0)
    g_rot = g_rot[0:3, 0:3]
    est_rot = tfs.quaternion_matrix(q_es_0)
    est_rot = est_rot[0:3, 0:3]

    C_R = np.dot(est_rot, g_rot.transpose())
    theta = align.get_best_yaw(C_R)
    R = align.rot_z(theta)
    t = p_gt_0 - np.dot(R, p_es_0)

    return R, t


def alignPositionYaw(p_es, p_gt, q_es, q_gt, n_aligned=1):
    if n_aligned == 1:
        R, t = alignPositionYawSingle(p_es, p_gt, q_es, q_gt)
        return R, t
    else:
        idxs = _getIndices(n_aligned, p_es.shape[0])
        est_pos = p_es[idxs, 0:3]
        gt_pos = p_gt[idxs, 0:3]
        _, R, t = align.align_umeyama(gt_pos, est_pos, known_scale=True,
                                      yaw_only=True)  # note the order
        t = np.array(t)
        t = t.reshape((3, ))
        R = np.array(R)
        return R, t


# align by a SE3 transformation
def alignSE3Single(p_es, p_gt, q_es, q_gt, pose_id_to_align=0):
    '''
    Calculate SE3 transformation R and t so that:
        gt = R * est + t
    Using only the first poses of est and gt
    '''

    p_es_0, q_es_0 = p_es[pose_id_to_align, :], q_es[pose_id_to_align, :]
    p_gt_0, q_gt_0 = p_gt[pose_id_to_align, :], q_gt[pose_id_to_align, :]

    g_rot = tfs.quaternion_matrix(q_gt_0)
    g_rot = g_rot[0:3, 0:3]
    est_rot = tfs.quaternion_matrix(q_es_0)
    est_rot = est_rot[0:3, 0:3]

    R = np.dot(g_rot, np.transpose(est_rot))
    t = p_gt_0 - np.dot(R, p_es_0)

    return R, t


def alignSE3(p_es, p_gt, q_es, q_gt, pose_id_to_align=0, n_aligned=-1):
    '''
    Calculate SE3 transformation R and t so that:
        gt = R * est + t
    '''
    if n_aligned == 1:
        R, t = alignSE3Single(p_es, p_gt, q_es, q_gt, pose_id_to_align=pose_id_to_align)
        return R, t
    else:
        idxs = _getIndices(n_aligned, p_es.shape[0])
        est_pos = p_es[idxs, 0:3]
        gt_pos = p_gt[idxs, 0:3]
        s, R, t = align.align_umeyama(gt_pos, est_pos,
                                      known_scale=True)  # note the order
        t = np.array(t)
        t = t.reshape((3, ))
        R = np.array(R)
        return R, t


# align by similarity transformation
def alignSIM3(p_es, p_gt, q_es, q_gt, n_aligned=-1):
    '''
    calculate s, R, t so that:
        gt = R * s * est + t
    '''
    idxs = _getIndices(n_aligned, p_es.shape[0])
    est_pos = p_es[idxs, 0:3]
    gt_pos = p_gt[idxs, 0:3]
    s, R, t = align.align_umeyama(gt_pos, est_pos)  # note the order
    return s, R, t


def alignTrajectory(p_es, p_gt, q_es, q_gt, method, pose_id_to_align=0, n_aligned=-1):
    '''
    calculate s, R, t so that:
        gt = R * s * est + t
    method can be: sim3, se3, posyaw, none;
    n_aligned: -1 means using all the frames
    '''
    assert p_es.shape[1] == 3
    assert p_gt.shape[1] == 3
    assert q_es.shape[1] == 4
    assert q_gt.shape[1] == 4

    s = 1
    R = None
    t = None
    if method == 'sim3':
        assert n_aligned >= 2 or n_aligned == -1, "sim3 uses at least 2 frames"
        s, R, t = alignSIM3(p_es, p_gt, q_es, q_gt, n_aligned)
    elif method == 'se3':
        R, t = alignSE3(p_es, p_gt, q_es, q_gt, pose_id_to_align=pose_id_to_align, n_aligned=1)
    elif method == 'posyaw':
        R, t = alignPositionYaw(p_es, p_gt, q_es, q_gt, n_aligned)
    elif method == 'none':
        R = np.identity(3)
        t = np.zeros((3, ))
    else:
        assert False, 'unknown alignment method'

    return s, R, t