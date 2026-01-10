import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2 as cv

import skimage.transform
import re
import argparse
import os
import json

def to_native(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_native(v) for v in o]
    return o


def read_dtu_camera(file_path):
    """
    Reads a DTU camera parameter file and returns intrinsic and extrinsic parameters.

    Args:
        file_path (str): Path to the DTU camera parameter file.

    Returns:
        dict: A dictionary containing:
            - "intrinsic": The intrinsic matrix (3x3).
            - "extrinsic": The extrinsic matrix (4x4).
    """
    with open(file_path) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsics = intrinsics.reshape((3, 3))
    intrinsics_ = np.float32(np.diag([1, 1, 1, 1]))
    intrinsics_[:3, :3] = intrinsics
    
    return {
        "extrinsic": extrinsics,
        "intrinsic": intrinsics_
    }

def depth_to_normals_np(depth_map, K):
    """
    Convert depth map to normal map in camera coordinate system using NumPy.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        K (np.ndarray): Intrinsic camera matrix of shape (3, 3).

    Returns:
        np.ndarray: Normal map of shape (H, W, 3).
    """
    H, W = depth_map.shape

    # Create pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")

    # Camera intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Unproject depth to 3D
    X = (u - cx) * depth_map / fx
    Y = (v - cy) * depth_map / fy
    Z = depth_map

    # Stack 3D points
    P = np.stack((X, Y, Z), axis=-1)  # Shape: (H, W, 3)

    # Compute spatial gradients
    dPdu = np.zeros_like(P)
    dPdv = np.zeros_like(P)
    
    dPdu[1:, :, :] = P[1:, :, :] - P[:-1, :, :]  # Gradient along u (vertical)
    dPdv[:, 1:, :] = P[:, 1:, :] - P[:, :-1, :]  # Gradient along v (horizontal)

    # Compute cross product for normals
    normals = np.cross(dPdu, dPdv, axis=-1)  # Shape: (H, W, 3)

    # Normalize normals to unit vectors
    norm = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8  # Avoid division by zero
    normals = normals / norm

    return normals

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def valid_mean(arr, mask, axis=None, keepdims=np._NoValue):
    """Compute mean of elements across given dimensions of an array, considering only valid elements.

    Args:
        arr: The array to compute the mean.
        mask: Array with numerical or boolean values for element weights or validity. For bool, False means invalid.
        axis: Dimensions to reduce.
        keepdims: If true, retains reduced dimensions with length 1.

    Returns:
        Mean array/scalar and a valid array/scalar that indicates where the mean could be computed successfully.
    """

    mask = mask.astype(arr.dtype) if mask.dtype == bool else mask
    num_valid = np.sum(mask, axis=axis, keepdims=keepdims)
    masked_arr = arr * mask
    masked_arr_sum = np.sum(masked_arr, axis=axis, keepdims=keepdims)

    with np.errstate(divide='ignore', invalid='ignore'):
        valid_mean = masked_arr_sum / num_valid
        is_valid = np.isfinite(valid_mean)
        valid_mean = np.nan_to_num(np.asarray(valid_mean), nan=0.0, posinf=0.0, neginf=0.0)

    return valid_mean, is_valid

def m_rel_ae(gt, pred, mask=None, output_scaling_factor=1.0):
    """Computes the mean-relative-absolute-error for a predicted and ground truth depth map.

    Args:
        gt: Ground truth depth map as numpy array of shape HxW. Negative or 0 values are invalid and ignored.
        pred: Predicted depth map as numpy array of shape HxW.
        mask: Array of shape HxW with numerical or boolean values for element weights or validity.
            For bool, False means invalid.
        output_scaling_factor: Scaling factor that is applied after computing the metrics (e.g. to get [%]).


    Returns:
        Scalar that indicates the mean-relative-absolute-error. Scalar is np.nan if the result is invalid.
    """
    mask = (gt > 0).astype(np.float32) * mask if mask is not None else (gt > 0).astype(np.float32)

    e = pred - gt
    ae = np.abs(e)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_ae = np.nan_to_num(ae / gt, nan=0, posinf=0, neginf=0)

    m_rel_ae, valid = valid_mean(rel_ae, mask)

    m_rel_ae = m_rel_ae * output_scaling_factor
    m_rel_ae = m_rel_ae if valid else np.nan

    return m_rel_ae




def thresh_inliers(gt, pred, thresh, mask=None, output_scaling_factor=1.0):
    """Computes the inlier (=error within a threshold) ratio for a predicted and ground truth depth map.

    Args:
        gt: Ground truth depth map as numpy array of shape HxW. Negative or 0 values are invalid and ignored.
        pred: Predicted depth map as numpy array of shape HxW.
        thresh: Threshold for the relative difference between the prediction and ground truth.
        mask: Array of shape HxW with numerical or boolean values for element weights or validity.
            For bool, False means invalid.
        output_scaling_factor: Scaling factor that is applied after computing the metrics (e.g. to get [%]).

    Returns:
        Scalar that indicates the inlier ratio. Scalar is np.nan if the result is invalid.
    """
    mask = (gt > 0).astype(np.float32) * mask if mask is not None else (gt > 0).astype(np.float32)

    with np.errstate(divide='ignore', invalid='ignore'):
        rel_1 = np.nan_to_num(gt / pred, nan=thresh+1, posinf=thresh+1, neginf=thresh+1)  # pred=0 should be an outlier
        rel_2 = np.nan_to_num(pred / gt, nan=0, posinf=0, neginf=0)  # gt=0 is masked out anyways

    max_rel = np.maximum(rel_1, rel_2)
    inliers = ((0 < max_rel) & (max_rel < thresh)).astype(np.float32)  # 1 for inliers, 0 for outliers

    inlier_ratio, valid = valid_mean(inliers, mask)

    inlier_ratio = inlier_ratio * output_scaling_factor
    inlier_ratio = inlier_ratio if valid else np.nan

    return inlier_ratio


def align_depths(gt_depth, pred_depth,alignment = "median"):


    #pred_depth = skimage.transform.resize(pred_depth, gt_depth.shape, order=0, anti_aliasing=False)

    pred_mask = pred_depth != 0 
    gt_mask = gt_depth > 0

    if alignment == "median":
        mask = gt_mask & pred_mask
        ratio = np.median(gt_depth[mask]) / np.median(pred_depth[mask])

        if mask.any() and np.isfinite(ratio):
            pred_depth = pred_depth * ratio
        else:
            ratio = np.nan

    elif alignment == 'lsq_scale_shift':
        mask = gt_mask & pred_mask
        with np.errstate(divide='ignore', invalid='ignore'):
            pred_invdepth = np.nan_to_num(1 / pred_depth, nan=0, posinf=0, neginf=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            gt_invdepth = np.nan_to_num(1 / gt_depth, nan=0, posinf=0, neginf=0)

        if mask.any():
            masked_gt_invdepth = (gt_invdepth[mask]).astype(np.float64)
            masked_pred_invdepth = (pred_invdepth[mask]).astype(np.float64)

            # system matrix: A = [[a_00, a_01], [a_10, a_11]]
            a_00 = np.sum(masked_pred_invdepth * masked_pred_invdepth)
            a_01 = np.sum(masked_pred_invdepth)
            a_11 = np.sum(mask.astype(np.float64))

            # right hand side: b = [b_0, b_1]
            b_0 = np.sum(masked_gt_invdepth * masked_pred_invdepth)
            b_1 = np.sum(masked_gt_invdepth)

            det = a_00 * a_11 - a_01 * a_01
            valid = det > 0

            if valid:
                scale = ((a_11 * b_0 - a_01 * b_1) / det).astype(np.float32)
                shift = ((-a_01 * b_0 + a_00 * b_1) / det).astype(np.float32)
            else:
                scale = np.nan
                shift = np.nan

        else:
            scale = np.nan
            shift = np.nan

        pred_invdepth = scale * pred_invdepth + shift
        with np.errstate(divide='ignore', invalid='ignore'):
            pred_depth = np.nan_to_num(1 / pred_invdepth, nan=0, posinf=0, neginf=0)
    return pred_depth

def compute_normal_consistency_images(normal_img1, normal_img2, mask):
    """
    Compute normal consistency between two normal maps using NumPy.

    Args:
        normal_img1: (H, W, 3) NumPy array for the first normal map.
        normal_img2: (H, W, 3) NumPy array for the second normal map.

    Returns:
        Normal consistency (scalar value between 0 and 1).
    """
    # Ensure both normal images have the same shape
    assert normal_img1.shape == normal_img2.shape, "Normal images must have the same shape"

    # Compute the norms of the normal maps
    norm1 = np.linalg.norm(normal_img1, axis=-1, keepdims=True)
    norm2 = np.linalg.norm(normal_img2, axis=-1, keepdims=True)

    # Replace zeros with a small epsilon to avoid division by zero
    epsilon = 1e-8
    norm1[norm1 < epsilon] = epsilon
    norm2[norm2 < epsilon] = epsilon

    # Normalize the normal maps
    normal_img1 = normal_img1 / norm1
    normal_img2 = normal_img2 / norm2

    # Compute dot products between corresponding normals
    dot_products = np.sum(normal_img1 * normal_img2, axis=-1)

    # Take the absolute value of dot products
    abs_dot_products = np.abs(dot_products)

    # Compute mean normal consistency
    normal_consistency = np.mean(abs_dot_products[mask>0])
    return normal_consistency.astype(np.float64)

def compute_depth_metrics(root_dir ,dataset_dir, meshdir,scan,   imgs_idx, alignment ='lsq_scale_shift' ):
        #imgs_idx = sorted(view_list[:args.n_view])
        metrics_dict = {"relative_error": [], "thresh_inlier": [], "normal_consistency": []}

        for depth_idx, image_idx in enumerate(imgs_idx):
            mask_filename = os.path.join(root_dir, 'scan{}/mask/{:0>3}.png'.format(scan, image_idx))
            mask_image = cv.imread(mask_filename)
            kernel_size = 11  # default 101
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask_image = cv.dilate(mask_image, kernel, iterations=1)
            mask_image = (mask_image[:, :, 0] > 128)
            depth_path = os.path.join(meshdir, f"vis_new/depth_0000{depth_idx}.tiff")
            gt_depth, scale = read_pfm(os.path.join(dataset_dir, f"scan{scan}/depth_map_{image_idx:04d}.pfm"))
            mask_depth_raster = ~(gt_depth <= 0)
            mask_full = mask_image*mask_depth_raster

            dtu_cams = read_dtu_camera(os.path.join('/', os.path.join(*dataset_dir.split('/')[:-1]), f"Cameras/{image_idx:08d}_cam.txt"))
            gt_normal = depth_to_normals_np(gt_depth, dtu_cams["intrinsic"][:3,:3])

            normal_path = os.path.join(meshdir, f"vis_new/normal_0000{depth_idx}.npy")
            pred_normal = np.load(normal_path)
            pred_normal = skimage.transform.resize(pred_normal, gt_normal.shape, order=0, anti_aliasing=False)
            gt_normal = gt_normal * np.repeat(mask_full[:, :, np.newaxis], 3, axis=2) #* np.repeat(mask_depth_raster[:, :, np.newaxis], 3, axis=2)
            pred_normal = pred_normal * np.repeat(mask_full[:, :, np.newaxis], 3, axis=2) #* np.repeat(mask_depth_raster[:, :, np.newaxis], 3, axis=2)
            normal_consistency = compute_normal_consistency_images(pred_normal, gt_normal, mask_full)
            # normal_consistency = normal_error_img[mask_full>0].mean()
            metrics_dict["normal_consistency"].append(normal_consistency)
            # import ipdb; ipdb.set_trace()
            # cv.imwrite(os.path.join(meshdir, "error_normal.png" + str(image_idx)+ "_ours.png"), (normal_error_img*255).astype(np.uint8))

            pred_normal = (pred_normal + 1.0) / 2.0
            gt_normal = (gt_normal + 1.0) / 2.0
            pred_normal[mask_full==0,:] = 1.0
            gt_normal[mask_full==0,:] = 1.0
            cv.imwrite(os.path.join(meshdir, "pred_normal" + str(image_idx)+ "_ours.png"), (pred_normal[:,:,::-1]*255).astype(np.uint8))
            cv.imwrite(os.path.join(meshdir, "gt_normal.png" + str(image_idx)+ "_ours.png"), (gt_normal[:,:,::-1]*255).astype(np.uint8))

            pred_depth = cv.imread(depth_path, cv.IMREAD_UNCHANGED)
            pred_depth = skimage.transform.resize(pred_depth, gt_depth.shape, order=0, anti_aliasing=False)
            gt_depth = gt_depth * mask_full #mask_image * mask_depth_raster
            pred_depth = pred_depth * mask_full #mask_image * mask_depth_raster
            pred_depth = align_depths(gt_depth, pred_depth, alignment=alignment)
            pred_depth = pred_depth * mask_full #mask_image * mask_depth_raster

            # pred_depth[mask_full==0] = 255.0
            # gt_depth[mask_full==0] = 255.0
            cv.imwrite(os.path.join(meshdir, "pred_depth" + str(image_idx)+ "_ours.png"), pred_depth.astype(np.uint8))
            cv.imwrite(os.path.join(meshdir, "gt_depth.png" + str(image_idx)+ "_ours.png"), gt_depth.astype(np.uint8))

            rel_absolute_error = m_rel_ae(gt_depth, pred_depth, mask=mask_full, output_scaling_factor=100.0)
            thresh_inlier = thresh_inliers(gt_depth, pred_depth, thresh=1.03, mask=mask_full, output_scaling_factor=100.0)
            metrics_dict["relative_error"].append(rel_absolute_error)
            metrics_dict["thresh_inlier"].append(thresh_inlier)
            print(f'Image {image_idx}:')
            print('relative error:', rel_absolute_error)
            print('thresh_inlier:', thresh_inlier)
            print('normal_consistency:', normal_consistency)

        # Calculate average metrics
        metrics_dict["average_relative_error"] = np.mean(metrics_dict["relative_error"])
        metrics_dict["average_thresh_inlier"] = np.mean(metrics_dict["thresh_inlier"])
        metrics_dict["average_normal_consistency"] = np.mean(metrics_dict["normal_consistency"])

        print("\nAverage metrics:")
        print('average relative error:', metrics_dict["average_relative_error"])
        print('average thresh_inlier:', metrics_dict["average_thresh_inlier"])
        print('average normal_consistency:', metrics_dict["average_normal_consistency"])

        return metrics_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=int, default=1)
    parser.add_argument(
        "--dataset_dir", type=str, default="/dataset/dtu_official/SampleSet/MVS_Data"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="./sparsecraft_data/reconstruction/mvs_data/set0/scan55",
        help="Path to the DTU scan directory used for training and that contains the images and masks",
    )
    parser.add_argument(
        "--mesh_dir",
        type=str,
        default="./sparsecraft_data/reconstruction/mvs_data/set0/scan55",
        help="Path to the DTU scan directory used for training and that contains the images and masks",
    )
    parser.add_argument("--ckpt", type=int, default=1)
    #parser.add_argument("--ckpt", type=int, default=1)
    parser.add_argument('--n_view', dest='n_view', type=int, default=3)
    parser.add_argument('--set', dest='set', type=int, default=0)
    parser.add_argument(
        "--alignment", type=str, default="lsq_scale_shift"
    )
    args = parser.parse_args()
    #args.meshdir = './variance_ablations/grid_search//scan83/2024-11-12_03-48-03/train/ours_2999'
    #args.meshdir = f'/media/shubhendujena/DATA/2d-gaussian-splatting-variance/output/scan{args.scan}/train/ours_1000/'
    #args.meshdir = '/media/shubhendujena/DATA/2d-gaussian-splatting-duster-depth/output/' + f'scan{args.scan}/train/ours_{args.ckpt}/'
    if args.set==0:
        view_list = [23, 24, 33, 22, 15, 34, 14, 32, 16, 35, 25]
    else:
        if args.scan != 37:
            view_list = [42, 43, 44, 33, 34, 32, 45, 23, 41, 24, 3]
        else:
            view_list = [1, 8, 9, 33, 34, 32, 45, 23, 41, 24, 31]

    imgs_idx = sorted(view_list[:args.n_view] )

    root_dir = args.test_dir
    dataset_dir = args.dataset_dir
    scan = args.scan
    meshdir = args.mesh_dir +f'scan{args.scan}/train/ours_{args.ckpt}/'
    print(meshdir)
    
    metrics_dict = compute_depth_metrics(root_dir ,dataset_dir, meshdir, scan, imgs_idx, args.alignment)

    with open(meshdir + f'/depth_metrics_{args.alignment}.json', "w") as f:
        json.dump(to_native(metrics_dict) ,  f, indent=4)
    #print(pred_depth.max(),gt_depth[0].max(), gt_depth[1] )


