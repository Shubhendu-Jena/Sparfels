import os
import re
import glob
import argparse
import numpy as np
import cv2 as cv
import torch
import trimesh
import open3d as o3d
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.renderer import (
    PerspectiveCameras, 
    PointsRasterizationSettings,
    RasterizationSettings,
    PointsRasterizer,
    MeshRasterizer
)
from pytorch3d.ops import interpolate_face_attributes
from utils.render_utils import save_img_f32
from scipy.spatial.transform import Rotation as R

from scene.colmap_loader import (
    read_extrinsics_text, read_intrinsics_text,
    read_extrinsics_binary, read_intrinsics_binary,
    qvec2rotmat
)

np.bool = np.bool_  # Patch deprecation
# Accepted image file extensions
VALID_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
VALID_MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")

def get_first_extension(folder_path, valid_exts):
    """
    Detect the extension used in a directory by checking the first valid file.
    Returns the extension (e.g., '.png').
    """
    all_files = glob.glob(os.path.join(folder_path, "*"))
    valid_files = [f for f in all_files if f.lower().endswith(valid_exts)]
    if not valid_files:
        raise FileNotFoundError(f"No valid files with extensions {valid_exts} found in {folder_path}")
    _, ext = os.path.splitext(valid_files[0])
    return ext


# ===========================
# Camera & Projection Utils
# ===========================
def read_cam_file_new(filename):
    """
    Load camera file e.g., 00000000_cam.txt
    """
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsics = intrinsics.reshape((3, 3))
    intrinsics_ = np.float32(np.diag([1, 1, 1, 1]))
    intrinsics_[:3, :3] = intrinsics
    #P = intrinsics_ @ extrinsics

    return intrinsics_, extrinsics

def read_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    intrinsics_ = np.eye(4, dtype=np.float32)
    intrinsics_[:3, :3] = intrinsics
    P = intrinsics_ @ extrinsics
    return P

def get_ndc_f_c_batched(fx, fy, px, py, image_width, image_height):
    s = min(image_width, image_height)
    fx_ndc = fx * 2.0 / (s - 1)
    fy_ndc = fy * 2.0 / (s - 1)

    px_ndc = - (px - (image_width - 1) / 2.0) * 2.0 / (s - 1)
    py_ndc = - (py - (image_height - 1) / 2.0) * 2.0 / (s - 1)

    return fx_ndc, fy_ndc, px_ndc, py_ndc

def gen_rays_from_single_image(H, W, image, intrinsic, c2w, depth=None, mask=None):
    device = image.device
    ys, xs = torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W), indexing='ij')
    p = torch.stack([xs, ys, torch.ones_like(ys)], dim=-1)  # H, W, 3

    ndc_u = 2 * xs / (W - 1) - 1
    ndc_v = 2 * ys / (H - 1) - 1
    rays_ndc_uv = torch.stack([ndc_u, ndc_v], dim=-1).view(-1, 2).float().to(device)

    intrinsic_inv = torch.inverse(intrinsic).float()
    p = torch.matmul(intrinsic_inv[None, :3, :3], p.view(-1, 3).float().to(device)[:, :, None]).squeeze()
    rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
    rays_v = torch.matmul(c2w[None, :3, :3], rays_v[:, :, None]).squeeze()
    rays_o = c2w[None, :3, 3].expand(rays_v.shape)

    color = image.permute(1, 2, 0).view(-1, 3)
    depth = depth.view(-1, 1) if depth is not None else None
    mask = mask.view(-1, 1) if mask is not None else torch.ones([H * W, 1], device=device)

    sample = {
        "rays_o": rays_o,
        "rays_v": rays_v,
        "rays_ndc_uv": rays_ndc_uv,
        "rays_color": color,
        "rays_mask": mask,
        "rays_norm_XYZ_cam": p
    }
    if depth is not None:
        sample["rays_depth"] = depth
    return sample

def load_K_Rt_from_P(filename=None, P=None):
    """Decompose projection matrix into intrinsics and pose"""
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = np.array([[float(v) for v in l.split()] for l in lines])
        P = lines.astype(np.float32).squeeze()

    K, R, t, *_ = cv.decomposeProjectionMatrix(P)
    K /= K[2, 2]

    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.T
    pose[:3, 3] = (t[:3] / t[3])[:, 0]
    return intrinsics, pose

def export_depth_and_normals_with_masking(root_dir, instance_dir, pose_path, source_mesh, target_pts, imgs_idx):
    # Load point cloud cameras
    cameras_pc_extrinsics = []
    cameras_pc_intrinsics = []
    for vid in imgs_idx:
        cam_file = os.path.join(root_dir, 'cameras/{:08d}_cam.txt'.format(vid))
        intrinsics, extrinsics = read_cam_file_new(cam_file)
        cameras_pc_intrinsics.append(intrinsics)
        cameras_pc_extrinsics.append(extrinsics)

    # Load COLMAP intrinsics/extrinsics
    try:
        cam_ext = read_extrinsics_text(os.path.join(instance_dir, "sparse/0", "images.txt"))
        cam_int = read_intrinsics_text(os.path.join(instance_dir, "sparse/0", "cameras.txt"))
    except:
        cam_ext = read_extrinsics_binary(os.path.join(instance_dir, "sparse/0", "images.bin"))
        cam_int = read_intrinsics_binary(os.path.join(instance_dir, "sparse/0", "cameras.bin"))

    pose_opt = np.load(pose_path)

    image_dir = os.path.join(instance_dir, "images")
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))

    cameras_mesh_extrinsics = []
    cameras_mesh_intrinsics = []
    for idx, key in enumerate(cam_ext):
        extr = cam_ext[key]
        intr = cam_int[extr.camera_id]
        assert extr.name == os.path.basename(image_paths[idx])
        intrinsics = np.eye(4, dtype=np.float32)
        intrinsics[0, 0], intrinsics[1, 1] = intr.params[0], intr.params[1]
        intrinsics[0, 2], intrinsics[1, 2] = intr.params[2], intr.params[3]
        cameras_mesh_intrinsics.append(intrinsics)
        cameras_mesh_extrinsics.append(pose_opt[idx])

    height, width = cam_int[1].height, cam_int[1].width
    mesh_raster_settings = RasterizationSettings(image_size=(height, width), faces_per_pixel=1)
    pc_raster_settings = PointsRasterizationSettings(image_size=(height, width), radius=0.001, points_per_pixel=1)

    mesh_src = Meshes(
        verts=[torch.tensor(np.asarray(source_mesh.vertices)).float().cuda()],
        faces=[torch.tensor(np.asarray(source_mesh.faces)).long().cuda()]
    )
    pc_tgt = Pointclouds(points=[torch.tensor(target_pts).float().cuda()])

    for i in range(min(3, len(imgs_idx))):
        # PC camera
        intr_pc = torch.tensor(cameras_pc_intrinsics[i]).cuda()
        fx_pc, fy_pc, px_pc, py_pc = get_ndc_f_c_batched(
            intr_pc[0, 0], intr_pc[1, 1],
            width - intr_pc[0, 2], height - intr_pc[1, 2],
            width, height
        )
        R_pc = torch.tensor(cameras_pc_extrinsics[i][:3, :3]).float().T.unsqueeze(0).cuda()
        T_pc = torch.tensor(cameras_pc_extrinsics[i][:3, 3]).float().unsqueeze(0).cuda()
        cam_pc = PerspectiveCameras(device="cuda", R=R_pc, T=T_pc,
                                    focal_length=((fx_pc, fy_pc),),
                                    principal_point=((px_pc, py_pc),))

        # Mesh camera
        intr_m = torch.tensor(cameras_mesh_intrinsics[i]).cuda()
        fx_m, fy_m, px_m, py_m = get_ndc_f_c_batched(
            intr_m[0, 0], intr_m[1, 1], intr_m[0, 2], intr_m[1, 2], width, height
        )
        R_m = torch.tensor(cameras_mesh_extrinsics[i][:3, :3]).float().T.unsqueeze(0).cuda()
        T_m = torch.tensor(cameras_mesh_extrinsics[i][:3, 3]).float().unsqueeze(0).cuda()
        cam_m = PerspectiveCameras(device="cuda", R=R_m, T=T_m,
                                   focal_length=((fx_m, fy_m),),
                                   principal_point=((px_m, py_m),))

        # Rasterize both
        raster_pc = PointsRasterizer(cameras=cam_pc, raster_settings=pc_raster_settings)(pc_tgt)
        raster_m = MeshRasterizer(cameras=cam_m, raster_settings=mesh_raster_settings)(mesh_src)

        mask_pc = ~(raster_pc.idx[0, ..., 0] == -1)
        mask_m = ~(raster_m.pix_to_face[0, ..., 0] == -1)
        mask = mask_pc & mask_m

        # Depth map
        depth = raster_m.zbuf[0, ..., 0]
        depth = torch.flip(depth, dims=[0, 1])
        depth[depth < 0] = 0

        # Normals
        faces = mesh_src.faces_packed()
        verts = mesh_src.verts_packed()
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        e1, e2 = v1 - v0, v2 - v0
        face_normals = torch.cross(e1, e2, dim=1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        face_normals = torch.einsum("ij,fj->fi", R_m[0], face_normals)
        face_attr = face_normals[:, None, :].expand(-1, 3, -1)
        normals = interpolate_face_attributes(raster_m.pix_to_face, raster_m.bary_coords, face_attr)[..., 0, :]
        normals = torch.flip(normals, dims=[1, 2])

        # Save
        iter_id = os.path.splitext(os.path.basename(pose_path))[0].split("_")[1]
        base_dir = os.path.dirname(os.path.dirname(pose_path))
        vis_path = os.path.join(base_dir, "train", f"ours_{iter_id}", "vis_new")
        os.makedirs(vis_path, exist_ok=True)
        save_img_f32(depth.cpu().numpy(), os.path.join(vis_path, f"depth_{i:05d}.tiff"))
        save_img_f32(normals[0].cpu().numpy(), os.path.join(vis_path, f"normal_{i:05d}.npy"))

# ===========================
# Main Cleaning Functions
# ===========================

def clean_points_by_mask_pc(root_dir, points, scan, imgs_idx, minimal_vis=0, mask_dilated_size=11):
    """Filter points using image masks (camera projection only)"""
    cameras, mask_lis = [], []
    # Determine consistent mask extension in this scan folder
    mask_dir = os.path.join(root_dir, f'scan{scan}/mask')
    mask_ext = get_first_extension(mask_dir, VALID_MASK_EXTS)

    # Read camera matrices and mask paths
    cameras, mask_lis = [], []
    for vid in imgs_idx:
        cam_path = os.path.join(root_dir, f'cameras/{vid:08d}_cam.txt')
        mask_path = os.path.join(mask_dir, f"{vid:03d}{mask_ext}")

        P = read_cam_file(cam_path)
        cameras.append(P)
        mask_lis.append(mask_path)

    inside_mask = np.zeros(len(points))
    for i, P in enumerate(cameras):
        pts_img = (P[:3, :3] @ points.T + P[:3, 3:4]).T
        pts_img /= pts_img[:, 2:3]
        pts_img = np.round(pts_img).astype(int) + 1

        mask = cv.imread(mask_lis[i], cv.IMREAD_COLOR)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (mask_dilated_size, mask_dilated_size))
        mask = cv.dilate(mask, kernel)[:, :, 0] > 128

        mask = np.pad(mask, ((1, 1), (1, 1)), constant_values=1)
        h, w = mask.shape
        valid = (pts_img[:, 0] >= 0) & (pts_img[:, 0] < w) & (pts_img[:, 1] >= 0) & (pts_img[:, 1] < h)
        curr_mask = np.zeros_like(valid, dtype=float)
        curr_mask[valid] = mask[pts_img[valid, 1], pts_img[valid, 0]]
        inside_mask += curr_mask
    return inside_mask > minimal_vis

def clean_points_by_mask_mesh(instance_dir, pose_path, points, scan, imgs_idx, minimal_vis=0, mask_dilated_size=11):
    """Filter mesh vertices using masks and COLMAP projection"""
    # Resolve consistent image extension
    image_dir = os.path.join(instance_dir, "images")
    image_ext = get_first_extension(image_dir, VALID_IMAGE_EXTS)
    image_paths = sorted(glob.glob(os.path.join(image_dir, f"*{image_ext}")))

    # Resolve consistent mask extension
    mask_dir = os.path.join(instance_dir, "mask")
    mask_ext = get_first_extension(mask_dir, VALID_MASK_EXTS)

    try:
        cam_ext = read_extrinsics_text(os.path.join(instance_dir, "sparse/0", "images.txt"))
        cam_int = read_intrinsics_text(os.path.join(instance_dir, "sparse/0", "cameras.txt"))
    except:
        cam_ext = read_extrinsics_binary(os.path.join(instance_dir, "sparse/0", "images.bin"))
        cam_int = read_intrinsics_binary(os.path.join(instance_dir, "sparse/0", "cameras.bin"))

    pose_opt = np.load(pose_path)
    cameras, mask_lis = [], []

    for idx, key in enumerate(cam_ext):
        extr = cam_ext[key]
        intr = cam_int[extr.camera_id]

        # Check filename matches the sorted image list
        assert extr.name == os.path.basename(image_paths[idx])

        # Build mask path using extracted index and detected extension
        img_id = int(extr.name.split(".")[0])
        mask_path = os.path.join(mask_dir, f"{img_id:03d}{mask_ext}")
        mask_lis.append(mask_path)

        # Compose projection matrix
        R = qvec2rotmat(extr.qvec).T
        T = np.array(extr.tvec)
        intrinsics = np.eye(4, dtype=np.float32)
        intrinsics[0, 0], intrinsics[1, 1] = intr.params[0], intr.params[1]
        intrinsics[0, 2], intrinsics[1, 2] = intr.params[2], intr.params[3]
        P = intrinsics @ pose_opt[idx]
        cameras.append(P)

    inside_mask = np.zeros(len(points))
    for i, P in enumerate(cameras):
        pts_img = (P[:3, :3] @ points.T + P[:3, 3:4]).T
        pts_img /= pts_img[:, 2:3]
        pts_img = np.round(pts_img).astype(int) + 1

        mask = cv.imread(mask_lis[i], cv.IMREAD_COLOR)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (mask_dilated_size, mask_dilated_size))
        mask = cv.dilate(mask, kernel)[:, :, 0] > 128
        mask = np.pad(mask, ((1, 1), (1, 1)), constant_values=1)

        h, w = mask.shape
        valid = (pts_img[:, 0] >= 0) & (pts_img[:, 0] < w) & (pts_img[:, 1] >= 0) & (pts_img[:, 1] < h)
        curr_mask = np.zeros_like(valid, dtype=float)
        curr_mask[valid] = mask[pts_img[valid, 1], pts_img[valid, 0]]
        inside_mask += curr_mask
    return inside_mask > minimal_vis

def clean_mesh_faces_by_mask(DTU_DIR, TEST_DIR, pose_path, scan, pc_file, mesh_file, new_mesh_file, imgs_idx, minimal_vis=0, mask_dilated_size=11):
    """Remove mesh faces with vertices outside masks"""
    mesh = trimesh.load(mesh_file)
    vertices, faces = mesh.vertices, mesh.faces
    pc_points = np.asarray(o3d.io.read_point_cloud(pc_file).points)

    mask_pc = clean_points_by_mask_pc(TEST_DIR, pc_points, scan, imgs_idx, minimal_vis, mask_dilated_size)
    mask_mesh = clean_points_by_mask_mesh(DTU_DIR, pose_path, vertices, scan, imgs_idx, minimal_vis, mask_dilated_size)

    indices = np.full(len(vertices), -1)
    indices[np.where(mask_mesh)] = np.arange(np.count_nonzero(mask_mesh))

    face_mask = mask_mesh[faces[:, 0]] & mask_mesh[faces[:, 1]] & mask_mesh[faces[:, 2]]
    new_faces = faces[face_mask]
    new_faces = indices[new_faces]
    new_vertices = vertices[mask_mesh]

    masked_gt_pc = pc_points[np.where(mask_pc)]

    new_mesh = trimesh.Trimesh(new_vertices, new_faces)
    new_mesh.export(new_mesh_file)
    return masked_gt_pc

def clean_mesh_faces_outside_frustum(DTU_DIR, TEST_DIR, masked_gt_pc, pose_path, old_mesh_file, new_mesh_file, imgs_idx, H=1200, W=1600, mask_dilated_size=11):
    """Remove mesh faces not visible in frustum of any camera"""
    # Load mesh and build ray intersector
    mesh = trimesh.load(old_mesh_file)
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    # Load COLMAP intrinsics and extrinsics
    try:
        cam_ext = read_extrinsics_text(os.path.join(DTU_DIR, "sparse/0", "images.txt"))
        cam_int = read_intrinsics_text(os.path.join(DTU_DIR, "sparse/0", "cameras.txt"))
    except:
        cam_ext = read_extrinsics_binary(os.path.join(DTU_DIR, "sparse/0", "images.bin"))
        cam_int = read_intrinsics_binary(os.path.join(DTU_DIR, "sparse/0", "cameras.bin"))

    # Load optimized poses
    pose_opt = np.load(pose_path)

    # Resolve consistent mask extension
    mask_dir = os.path.join(DTU_DIR, "mask")
    mask_ext = get_first_extension(mask_dir, VALID_MASK_EXTS)

    # Generate list of mask file paths
    mask_lis = [
        os.path.join(mask_dir, f"{int(extr.name.split('.')[0]):03d}{mask_ext}")
        for extr in cam_ext.values()
    ]

    cameras = []

    for idx, key in enumerate(cam_ext):
        extr = cam_ext[key]
        intr = cam_int[extr.camera_id]
        intrinsics = np.eye(4, dtype=np.float32)
        intrinsics[0, 0], intrinsics[1, 1] = intr.params[0], intr.params[1]
        intrinsics[0, 2], intrinsics[1, 2] = intr.params[2], intr.params[3]
        P = intrinsics @ pose_opt[idx]
        cameras.append(P)

    all_indices = []
    for i in range(len(imgs_idx)):
        mask_img = cv.imread(mask_lis[i])
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (mask_dilated_size, mask_dilated_size))
        mask_img = cv.dilate(mask_img, kernel)

        P = cameras[i]
        K, pose = load_K_Rt_from_P(P=P[:3, :])
        rays = gen_rays_from_single_image(H, W, torch.from_numpy(mask_img).permute(2, 0, 1).float(), torch.from_numpy(K)[:3, :3], torch.from_numpy(pose))
        rays_o, rays_d = rays['rays_o'], rays['rays_v']
        rays_mask = rays['rays_color'][:, 0] > 128

        for o, d in zip(rays_o[rays_mask].split(4096), rays_d[rays_mask].split(4096)):
            hits = intersector.intersects_first(o.numpy(), d.numpy())
            all_indices.append(hits)

    unique_faces = np.unique(np.concatenate(all_indices))
    face_mask = np.ones(len(mesh.faces), dtype=bool)
    face_mask[unique_faces[1:]] = False

    mesh_o3d = o3d.io.read_triangle_mesh(old_mesh_file)
    mesh_o3d.remove_triangles_by_mask(face_mask)
    o3d.io.write_triangle_mesh(new_mesh_file, mesh_o3d)

    mesh_clean = trimesh.load(new_mesh_file)
    cc = trimesh.graph.connected_components(mesh_clean.face_adjacency, min_len=500)
    mask = np.zeros(len(mesh_clean.faces), dtype=bool)
    mask[np.concatenate(cc)] = True
    mesh_clean.update_faces(mask)
    mesh_clean.remove_unreferenced_vertices()
    mesh_clean.export(new_mesh_file)

    export_depth_and_normals_with_masking(TEST_DIR, DTU_DIR, pose_path, mesh_clean, masked_gt_pc, imgs_idx)
    o3d.io.write_triangle_mesh(new_mesh_file.replace(".ply", "_raw.ply"), mesh_o3d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=int, default=1)
    parser.add_argument("--dtu_input_dir", type=str, default="./sparsecraft_data/reconstruction/mvs_data/set0/scan55")
    parser.add_argument("--test_dir", type=str, default="./sparsecraft_data/reconstruction/mvs_data/set0/scan55")
    parser.add_argument("--mesh_dir", type=str, default="./exp_dir")
    parser.add_argument("--dataset_dir", type=str, default="/dataset/dtu_official/SampleSet/MVS_Data")
    parser.add_argument("--n_view", type=int, default=3)
    parser.add_argument("--set", type=int, default=0)
    args = parser.parse_args()

    scan = args.scan
    GT_DIR = os.path.join(args.dataset_dir, "MVS_Data/Points/stl")
    args.gt = os.path.join(GT_DIR, f"stl{scan:03d}_total.ply")

    DTU_DIR = args.dtu_input_dir
    TEST_DIR = args.test_dir
    mesh_dir = args.mesh_dir
    mask_kernel_size = 11

    view_list = ([23, 24, 33, 22, 15, 34, 14, 32, 16, 35, 25] if args.set == 0
                 else ([42, 43, 44, 33, 34, 32, 45, 23, 41, 24, 31] if scan != 37
                       else [1, 8, 9, 33, 34, 32, 45, 23, 41, 24, 31]))

    imgs_idx = view_list[:args.n_view]
    old_mesh_file = os.path.join(mesh_dir, "fuse_post.ply")
    clean_mesh_file = os.path.join(mesh_dir, "clean_mesh.ply")
    final_mesh_file = os.path.join(mesh_dir, "final_mesh.ply")

    match = re.search(r'(\d+)$', mesh_dir)
    if not match:
        raise ValueError(f"Could not extract iteration number from mesh_dir: {mesh_dir}")
    iter_str = match.group(1)
    pose_path = os.path.join(*mesh_dir.split('/')[:-2], 'pose', f'pose_{iter_str}.npy')

    masked_gt_pc = clean_mesh_faces_by_mask(
        DTU_DIR, TEST_DIR, pose_path, scan,
        args.gt, old_mesh_file, clean_mesh_file,
        imgs_idx=imgs_idx, minimal_vis=1,
        mask_dilated_size=mask_kernel_size
    )

    clean_mesh_faces_outside_frustum(
        DTU_DIR, TEST_DIR, masked_gt_pc, pose_path, clean_mesh_file, final_mesh_file,
        imgs_idx=imgs_idx, mask_dilated_size=mask_kernel_size
    )
