import os
import shutil
import torch
import numpy as np
import argparse
import time
import roma

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "submodules", "mast3r")))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# from dust3r.inference import inference
# from dust3r.model import AsymmetricCroCo3DStereo
# from dust3r.utils.device import to_numpy
# from dust3r.image_pairs import make_pairs
# from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from mast3r.fast_nn import fast_reciprocal_NNs
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from mast3r.model import AsymmetricMASt3R
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from utils.align_utils import pad_poses, pose_inverse_4x4, rotation_distance, SO3_to_quat, alignTrajectory, convert3x4_4x4
from os import makedirs, path
import cv2
import kornia
import pickle 
from utils.dust3r_utils import  (compute_global_alignment, load_images, storePly, save_colmap_cameras, save_colmap_images, 
                                 round_python3, rigid_points_registration)

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
    parser.add_argument("--img_base_path", type=str, default="/home/workspace/datasets/instantsplat/Tanks_dust3r/Barn/24_views")

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
    # train_img_folder = os.path.join(img_base_path, f"dust3r_{n_views}_views/init_test_pose/train/images")
    # test_img_folder = os.path.join(img_base_path, f"dust3r_{n_views}_views/init_test_pose/test/images")
    all_img_folder = os.path.join(img_base_path, f"dust3r_{n_views}_views/init_test_pose/images")
    if os.path.exists(all_img_folder):
        shutil.rmtree(all_img_folder)
    os.makedirs(all_img_folder, exist_ok=True)
    model = AsymmetricMASt3R.from_pretrained(model_path).to(device)
    #################################################################################################################################################



    # ---------------- (1) Prepare Train & Test images list ---------------- 
    all_img_list = sorted(os.listdir(os.path.join(img_base_path, "images")))
    if args.llffhold > 0:
        train_img_list = [c for idx, c in enumerate(all_img_list) if (idx+1) % args.llffhold != 0]
        test_img_list = [c for idx, c in enumerate(all_img_list) if (idx+1) % args.llffhold == 0]
    # sample sparse view
    indices = np.linspace(0, len(train_img_list) - 1, n_views, dtype=int)
    print(indices)
    tmp_img_list = [train_img_list[i] for i in indices]
    train_img_list = tmp_img_list    
    assert len(train_img_list)==n_views, f"Number of images in the folder is not equal to {n_views}"

    #---------------- (2) Load train pointcloud and intrinsic (define as m1) ---------------- 
    train_pts_all_path = os.path.join(img_base_path, f"dust3r_{n_views}_views", "sparse/0", "pts_4_3dgs_all.npy")
    train_pts_all = np.load(train_pts_all_path)
    train_pts3d_m1 = train_pts_all
    
    if args.focal_avg:
        focal_path = os.path.join(img_base_path, f"dust3r_{n_views}_views", "sparse/0", "focal.npy")
        preset_focal = np.load(focal_path)     # load focal calculated by dust3r_coarse_geometry_initialization



    #---------------- (3) Get N_views pointcloud and 12 test pose (define as n1) ---------------- 
    # import pdb; pdb.set_trace()
    # if len(os.listdir(all_img_folder)) != len(test_img_list):
    for img_name in test_img_list:
        src_path = os.path.join(img_base_path, "images", img_name)
        tgt_path = os.path.join(all_img_folder, "1_"+img_name)
        print(src_path, tgt_path)
        shutil.copy(src_path, tgt_path)
    # if len(os.listdir(all_img_folder)) != len(train_img_list):
    for img_name in train_img_list[:3]:
        src_path = os.path.join(img_base_path, "images", img_name)
        tgt_path = os.path.join(all_img_folder, "0_"+img_name)
        print(src_path, tgt_path)
        shutil.copy(src_path, tgt_path)
    
    # # read all images
    # all_img_folder = os.path.join(img_base_path, "images")
    images, paths, ori_size = load_images(all_img_folder, size=512)
    print("ori_size", ori_size)
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

    output_matches_path=all_img_folder.replace("images", "matches")

    if os.path.exists(os.path.join(output_matches_path, 'matches.pkl')):
        os.remove(os.path.join(output_matches_path, 'matches.pkl'))

    os.makedirs(output_matches_path, exist_ok=True)

    with open(os.path.join(output_matches_path, 'matches.pkl'), 'wb') as f:
        pickle.dump(dict_correspondences, f)

    output_conf_path=all_img_folder.replace("images", "confidence")

    if os.path.exists(os.path.join(output_conf_path, 'conf.pkl')):
        os.remove(os.path.join(output_conf_path, 'conf.pkl'))

    os.makedirs(output_conf_path, exist_ok=True)

    with open(os.path.join(output_conf_path, 'conf.pkl'), 'wb') as f:
        pickle.dump(dict_confidences, f)


    test_output_colmap_path=all_img_folder.replace("images", "sparse/0")
    os.makedirs(test_output_colmap_path, exist_ok=True)

    scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = compute_global_alignment(scene, init="mst", niter=niter, schedule=schedule, lr=lr, focal_avg=args.focal_avg, known_focal=torch.tensor(preset_focal[0][0]))   
    all_poses = to_numpy(scene.get_im_poses())
    all_pts3d = to_numpy(scene.get_pts3d())

    # train_poses_n1 = [c for idx, c in enumerate(all_poses) if (idx+1) % args.llffhold != 0]    
    # train_pts3d_n1 = [c for idx, c in enumerate(all_pts3d) if (idx+1) % args.llffhold != 0]        
    # train_pts3d_n1 = [c for idx, c in enumerate(all_pts3d) if (idx+1) % args.llffhold != 0]     
    train_pts3d_n1 = all_pts3d[:3] 


    # test_poses_n1 = [c for idx, c in enumerate(all_poses) if (idx+1) % args.llffhold == 0]
    test_poses_n1 = all_poses[3:] 

    # all_poses_n1 =  np.array(to_numpy(all_poses))
    # train_poses_n1 =  np.array(to_numpy(train_poses_n1))
    train_pts3d_n1 = np.array(to_numpy(train_pts3d_n1)).reshape(-1,3)
    test_poses_n1  = np.array(to_numpy(test_poses_n1))              # test_pose_n1: c2w



    #---------------- (4) Applying pointcloud registration & Calculate transform_matrix & Save initial_test_pose---------------- ##########
    # compute transform that goes from cam to world
    train_pts3d_n1 = torch.from_numpy(train_pts3d_n1)
    train_pts3d_m1 = torch.from_numpy(train_pts3d_m1)
    train_pts3d_m1 = train_pts3d_m1[:(336*512*3), :]
    s, R, T = rigid_points_registration(train_pts3d_n1, train_pts3d_m1)

    transform_matrix = torch.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = T
    transform_matrix[:3, 3] *= s
    transform_matrix = transform_matrix.numpy()

    test_poses_m1 = transform_matrix @ test_poses_n1
    save_colmap_images(test_poses_m1, os.path.join(test_output_colmap_path, 'images.txt'), test_img_list)


    


  