import os
import shutil
import torch
import numpy as np
import argparse
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "submodules", "mast3r")))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from mast3r.fast_nn import fast_reciprocal_NNs
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from mast3r.model import AsymmetricMASt3R
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from utils.dust3r_utils import  compute_global_alignment, load_images, storePly, save_colmap_cameras, save_colmap_images, save_depth_images
import pickle 

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    # parser.add_argument("--model_path", type=str, default="./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", help="path to the model weights")
    parser.add_argument("--model_path", type=str, default="submodules/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth", help="path to the model weights")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--schedule", type=str, default='linear')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--niter", type=int, default=300)
    parser.add_argument("--focal_avg", action="store_true")
    # parser.add_argument("--focal_avg", type=bool, default=True)

    parser.add_argument("--llffhold", type=int, default=2)
    parser.add_argument("--n_views", type=int, default=12)
    parser.add_argument("--img_base_path", type=str, default="./dtu_0/scan24")

    return parser

if __name__ == '__main__':

    parser = get_args_parser()
    args = parser.parse_args()

    model_path = args.model_path
    device = args.device
    batch_size = args.batch_size
    schedule = args.schedule
    lr = args.lr
    niter = args.niter
    n_views = args.n_views
    img_base_path = args.img_base_path
    img_folder_path = os.path.join(img_base_path, "images")
    os.makedirs(img_folder_path, exist_ok=True)
    model = AsymmetricMASt3R.from_pretrained(model_path).to(device)
    ##########################################################################################################################################################################################

    train_img_list = sorted(os.listdir(img_folder_path))
    assert len(train_img_list)==n_views, f"Number of images ({len(train_img_list)}) in the folder ({img_folder_path}) is not equal to {n_views}"

    images, paths, ori_size = load_images(img_folder_path, size=512)
    print("ori_size", ori_size)

    start_time = time.time()
    
    ##########################################################################################################################################################################################
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    pairs_path = make_pairs(paths, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, args.device, batch_size=batch_size)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
    conf1, conf2 = pred1['desc_conf'].squeeze(0).detach(), pred2['desc_conf'].squeeze(0).detach()

    dict_correspondences = {}
    dict_confidences = {}
    for idx in range(desc1.shape[0]):
        # find 2D-2D matches between the two images
        matches_im0, matches_im1 = fast_reciprocal_NNs(desc1[idx], desc2[idx], subsample_or_initxy1=8,
                                                    device=device, dist='dot', block_size=2**13)

        # ignore small border around the edge
        H0, W0 = view1['true_shape'][idx]
        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
            matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

        H1, W1 = view2['true_shape'][idx]
        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
            matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

        viz_matches_im0, viz_matches_im1 = matches_im0, matches_im1
        conf_im0 = conf1[idx][viz_matches_im0[:,1], viz_matches_im0[:,0]]
        conf_im1 = conf2[idx][viz_matches_im1[:,1], viz_matches_im1[:,0]]
        matches_confs = torch.min(conf_im0, conf_im1)

        viz_matches_im0 = viz_matches_im0[matches_confs >= 10.0]
        viz_matches_im1 = viz_matches_im1[matches_confs >= 10.0]
        matches_confs = matches_confs[matches_confs >= 10.0]

        rows, cols = view1['original_img'][idx].shape[1:]
        resized_rows, resized_cols = view1['img'][idx].shape[1:]

        x_original_center = (cols-1) / 2
        y_original_center = (rows-1) / 2

        x_scaled_center = (resized_cols-1) / 2
        y_scaled_center = (resized_rows-1) / 2

        # Subtract the center, scale, and add the "scaled center".
        
        viz_matches_im0_original = np.copy(viz_matches_im0)
        viz_matches_im1_original = np.copy(viz_matches_im1)

        scale_x = cols/resized_cols
        scale_y = rows/resized_rows

        viz_matches_im0_original[:,0] = (viz_matches_im0_original[:,0] - x_scaled_center)*scale_x + x_original_center
        viz_matches_im0_original[:,1] = (viz_matches_im0_original[:,1] - y_scaled_center)*scale_y + y_original_center

        viz_matches_im1_original[:,0] = (viz_matches_im1_original[:,0] - x_scaled_center)*scale_x + x_original_center
        viz_matches_im1_original[:,1] = (viz_matches_im1_original[:,1] - y_scaled_center)*scale_y + y_original_center

        if pairs_path[idx][0]['img_path'] in dict_correspondences:
            dict_correspondences[pairs_path[idx][0]['img_path']].update({pairs_path[idx][1]['img_path']: np.concatenate((viz_matches_im0_original, viz_matches_im1_original), axis=-1)})
            dict_confidences[pairs_path[idx][0]['img_path']].update({pairs_path[idx][1]['img_path']: matches_confs.cpu().numpy()})
        else:
            dict_correspondences[pairs_path[idx][0]['img_path']] = {pairs_path[idx][1]['img_path']: np.concatenate((viz_matches_im0_original, viz_matches_im1_original), axis=-1)}
            dict_confidences[pairs_path[idx][0]['img_path']] = {pairs_path[idx][1]['img_path']: matches_confs.cpu().numpy()}

    output_matches_path=img_folder_path.replace("images", "matches")

    if os.path.exists(os.path.join(output_matches_path, 'matches.pkl')):
        os.remove(os.path.join(output_matches_path, 'matches.pkl'))

    os.makedirs(output_matches_path, exist_ok=True)

    with open(os.path.join(output_matches_path, 'matches.pkl'), 'wb') as f:
        pickle.dump(dict_correspondences, f)

    output_conf_path=img_folder_path.replace("images", "confidence")

    if os.path.exists(os.path.join(output_conf_path, 'conf.pkl')):
        os.remove(os.path.join(output_conf_path, 'conf.pkl'))

    os.makedirs(output_conf_path, exist_ok=True)

    with open(os.path.join(output_conf_path, 'conf.pkl'), 'wb') as f:
        pickle.dump(dict_confidences, f)

    output_colmap_path=img_folder_path.replace("images", "sparse/0")
    os.makedirs(output_colmap_path, exist_ok=True)

    depth_folder_path=img_folder_path.replace("images", "depths")
    os.makedirs(depth_folder_path, exist_ok=True)

    scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = compute_global_alignment(scene=scene, init="mst", niter=niter, schedule=schedule, lr=lr, focal_avg=args.focal_avg)
    scene = scene.clean_pointcloud()

    imgs = to_numpy(scene.imgs)
    focals = scene.get_focals()
    poses = to_numpy(scene.get_im_poses())
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(10.0))) #float(scene.conf_trf(torch.tensor(1.0)))
    confidence_masks = to_numpy(scene.get_masks())
    intrinsics = to_numpy(scene.get_intrinsics())
    depths = to_numpy(scene.get_depthmaps())
    ##########################################################################################################################################################################################
    end_time = time.time()
    print(f"Time taken for {n_views} views: {end_time-start_time} seconds")

    # save
    save_colmap_cameras(ori_size, intrinsics, os.path.join(output_colmap_path, 'cameras.txt'))
    save_colmap_images(poses, os.path.join(output_colmap_path, 'images.txt'), train_img_list)
    save_depth_images(depths, depth_folder_path, train_img_list)

    pts_4_3dgs = np.concatenate([p[m] for p, m in zip(pts3d, confidence_masks)])
    color_4_3dgs = np.concatenate([p[m] for p, m in zip(imgs, confidence_masks)])
    color_4_3dgs = (color_4_3dgs * 255.0).astype(np.uint8)
    storePly(os.path.join(output_colmap_path, "points3D.ply"), pts_4_3dgs, color_4_3dgs)
    pts_4_3dgs_all = np.array(pts3d).reshape(-1, 3)
    np.save(output_colmap_path + "/pts_4_3dgs_all.npy", pts_4_3dgs_all)
    np.save(output_colmap_path + "/focal.npy", np.array(focals.cpu()))
