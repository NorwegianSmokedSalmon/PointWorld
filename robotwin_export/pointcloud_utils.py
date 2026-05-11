"""
Point cloud utilities: depth-to-pointcloud backprojection and random downsampling.
"""

import numpy as np


def depth_to_pointcloud(depth, intrinsic, rgb=None, min_depth=0.01, max_depth=4.0):
    """
    Backproject a depth map into a 3D point cloud.

    Args:
        depth: (H, W) float32, depth in meters.
        intrinsic: (3, 3) camera intrinsic matrix.
        rgb: optional (H, W, 3) uint8 RGB image.  When provided the
             returned array has 6 columns (XYZ + RGB 0-255).
        min_depth: discard points closer than this (meters).
        max_depth: discard points farther than this (meters).

    Returns:
        points: (N, 3) or (N, 6) float32.
    """
    H, W = depth.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    u, v = np.meshgrid(u, v)

    valid = (depth > min_depth) & (depth < max_depth) & np.isfinite(depth)

    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy

    points = np.stack([x, y, z], axis=-1).astype(np.float32)

    if rgb is not None:
        colors = rgb[valid].astype(np.float32)  # 0-255
        points = np.concatenate([points, colors], axis=-1)

    return points


def random_downsample(points, n_points, rng=None):
    """
    Randomly downsample a point cloud to *n_points*.

    Args:
        points: (N, D) point cloud.
        n_points: target number of points.
        rng: numpy RandomState (for reproducibility).

    Returns:
        (n_points, D) float32.
    """
    if rng is None:
        rng = np.random.RandomState()

    N = points.shape[0]

    if N == 0:
        D = points.shape[1] if points.ndim > 1 else 6
        return np.zeros((n_points, D), dtype=np.float32)

    if N >= n_points:
        indices = rng.choice(N, n_points, replace=False)
    else:
        indices = rng.choice(N, n_points, replace=True)

    return points[indices].astype(np.float32)
