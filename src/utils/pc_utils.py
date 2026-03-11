"""Point cloud utility functions shared across the project."""
import os
import math
import numpy as np
import torch
import h5py
from einops import rearrange, repeat


def pc_normalize(pc):
    """Normalize a point cloud to zero mean and unit sphere."""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def normalize_point_cloud(input, centroid=None, furthest_distance=None):
    """Normalize a batch of point clouds. Input: (B, 3, N) tensor."""
    if centroid is None:
        centroid = torch.mean(input, dim=-1, keepdim=True)
    input = input - centroid
    if furthest_distance is None:
        furthest_distance = torch.max(
            torch.norm(input, p=2, dim=1, keepdim=True), dim=-1, keepdim=True
        )[0]
    input = input / furthest_distance
    return input, centroid, furthest_distance


def pc_normalization(input):
    """Normalize a batch to unit sphere. Accepts numpy or tensor."""
    is_tensor = isinstance(input, torch.Tensor)
    if is_tensor:
        device = input.device
        input = input.detach().cpu().numpy()
    input_centroid = np.mean(input, axis=1, keepdims=True)
    input = input - input_centroid
    input_furthest_distance = np.amax(
        np.sqrt(np.sum(input ** 2, axis=-1)), axis=1, keepdims=True
    )
    input = input / np.expand_dims(input_furthest_distance, axis=-1)
    if is_tensor:
        input = torch.from_numpy(input).to(device)
    return input


def index_points(pts, idx):
    """
    Index into point clouds.
    pts: (B, C, N), idx: (B, S, [K]) -> (B, C, S, [K])
    """
    batch_size = idx.shape[0]
    sample_num = idx.shape[1]
    fdim = pts.shape[1]
    reshape = False
    if len(idx.shape) == 3:
        reshape = True
        idx = idx.reshape(batch_size, -1)
    res = torch.gather(pts, 2, idx[:, None].repeat(1, fdim, 1))
    if reshape:
        res = rearrange(res, 'b c (s k) -> b c s k', s=sample_num)
    return res


def FPS(pts, fps_pts_num):
    """Furthest Point Sampling using pointops CUDA kernel.
    Input: (B, 3, N). Output: (B, 3, fps_pts_num).
    """
    from src.ops.pointops.functions import pointops
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    sample_idx = pointops.furthestsampling(pts_trans, fps_pts_num).long()
    sample_pts = index_points(pts, sample_idx)
    return sample_pts


def get_knn_pts(k, pts, center_pts, return_idx=False):
    """KNN query using pointops CUDA kernel.
    Input: (B, 3, N). Output: (B, 3, M, K).
    """
    from src.ops.pointops.functions import pointops
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    center_pts_trans = rearrange(center_pts, 'b c m -> b m c').contiguous()
    knn_idx = pointops.knnquery_heap(k, pts_trans, center_pts_trans).long()
    knn_pts = index_points(pts, knn_idx)
    if not return_idx:
        return knn_pts
    return knn_pts, knn_idx


def midpoint_interpolate(sparse_pts, up_rate=4, normal=False):
    """Midpoint interpolation for upsampling.
    sparse_pts: (B, 3, N). Output: (B, 3, N*up_rate).
    """
    if normal:
        sparse_pts, centroid, furthest_distance = normalize_point_cloud(sparse_pts)

    pts_num = sparse_pts.shape[-1]
    up_pts_num = int(pts_num * up_rate)
    k = int(2 * up_rate)
    knn_pts = get_knn_pts(k, sparse_pts, sparse_pts)
    repeat_pts = repeat(sparse_pts, 'b c n -> b c n k', k=k)
    mid_pts = (knn_pts + repeat_pts) / 2.0
    mid_pts = rearrange(mid_pts, 'b c n k -> b c (n k)')
    interpolated_pts = FPS(mid_pts, up_pts_num)

    if normal:
        interpolated_pts = centroid + interpolated_pts * furthest_distance
    return interpolated_pts


def get_rate_list(R=4, base=4):
    """Decompose upsampling rate into a list of stages."""
    ls = []
    l = math.floor(math.log(R, base))
    if l >= 1:
        ls = [4] * l
    if R - np.power(base, l) > 0:
        ls.append(2)
    return ls


def get_interpolate(point, R=4, base=4):
    """Hierarchical midpoint interpolation for a given upsampling rate.
    point: (B, N, 3). Returns: (B, N*R, 3).
    """
    ls = get_rate_list(R, base)
    i = point.permute(0, 2, 1)
    for r in ls:
        i = midpoint_interpolate(i, up_rate=r, normal=True)
    return i.permute(0, 2, 1)


def load_h5_data(h5_file_path, num_points=256, R=4):
    """Load and normalize point cloud data from HDF5 file."""
    num_out_points = int(num_points * R)
    with h5py.File(h5_file_path, 'r') as f:
        input = f['poisson_%d' % num_points][:]
        gt = f['poisson_%d' % num_out_points][:]

    assert input.shape[0] == gt.shape[0]
    input_centroid = np.mean(input, axis=1, keepdims=True)
    input = input - input_centroid
    input_furthest_distance = np.amax(
        np.sqrt(np.sum(input ** 2, axis=-1)), axis=1, keepdims=True
    )
    input = input / np.expand_dims(input_furthest_distance, axis=-1)
    gt = gt - input_centroid
    gt = gt / np.expand_dims(input_furthest_distance, axis=-1)
    return input, gt


def numpy_to_pc(points):
    """Convert numpy array to Open3D PointCloud."""
    import open3d
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(points)
    return pc
