#!/usr/bin/env python3
"""
Convert DROID scenes (processed by the PointWorld data pipeline) into
RoboTwin-compatible HDF5 episodes.

Prerequisites
-------------
* The PointWorld ``compute_depth.py`` stage must have been run so that
  ``<pointworld_output_dir>/depth/{uuid}_depth.h5`` files exist.
* The ZED Python SDK (``pyzed``) must be installed so that ``gather_data_dict``
  can read raw SVO files for the full RGB sequence.
* ``gsutil`` must be configured if DROID scenes are on GCS.

Usage
-----
::

    python robotwin_export/convert_droid_to_robotwin.py \\
        --input real/droid_paths.txt \\
        --pointworld_output_dir /path/to/pointworld/processed \\
        --output_dir /path/to/robotwin_hdf5 \\
        --image_height 480 --image_width 640 \\
        --num_points 2048 \\
        --time_skip_ratio 2 \\
        --rank 0 --world_size 1
"""

import argparse
import os
import sys
import random

import cv2
import h5py
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Make the PointWorld repo root importable regardless of CWD
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from real.droid_utils import (
    gather_data_dict,
    gather_trajectory,
    filter_by_timestamps,
    get_uuid,
)
from real.gcs_utils import enforce_gcs_cache_policy
from robotwin_export.pointcloud_utils import depth_to_pointcloud, random_downsample
from robotwin_export.robotwin_h5_writer import write_episode_hdf5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_scene_paths(txt_path):
    with open(txt_path, "r") as f:
        paths = [line.strip() for line in f if line.strip()]
    return paths


def _load_depth_from_h5(depth_h5_path, cam_group_name, target_timestamps):
    """Load depth frames from the PointWorld depth H5 and align to timestamps.

    Returns (T, H, W) float32 in metres and the native intrinsic is NOT
    returned here (the caller should use the intrinsic from ``gather_data_dict``).
    """
    with h5py.File(depth_h5_path, "r") as f:
        g = f[cam_group_name]
        depth_ds = g["depth"]  # (N, H, W) uint16 mm
        depth_ts = np.array(g["timestamps"])

        # Nearest-neighbour alignment
        idxs = np.array(
            [int(np.argmin(np.abs(depth_ts - ts))) for ts in target_timestamps],
            dtype=np.int64,
        )
        depth_uint16 = depth_ds[idxs]  # (T, H, W)

    return depth_uint16.astype(np.float32) / 1000.0  # → metres


def _scale_intrinsic(intrinsic, src_hw, dst_hw):
    """Scale a 3×3 pinhole intrinsic from *src_hw* to *dst_hw*."""
    K = intrinsic.astype(np.float64).copy()
    sy = dst_hw[0] / src_hw[0]
    sx = dst_hw[1] / src_hw[1]
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K.astype(np.float32)


def _pad_to_14(arr_1arm):
    """Pad a (T, D) single-arm array to (T, 14) by zero-filling the right arm."""
    T, D = arr_1arm.shape
    out = np.zeros((T, 14), dtype=np.float32)
    out[:, :D] = arr_1arm
    return out


# ---------------------------------------------------------------------------
# Per-scene conversion
# ---------------------------------------------------------------------------

def convert_one_scene(
    scene_path,
    pointworld_output_dir,
    output_path,
    downscale_ratio,
    time_skip_ratio,
    image_height,
    image_width,
    num_points,
    rng,
):
    uuid = get_uuid(scene_path)
    print(f"\n{'='*60}")
    print(f"Converting scene {uuid}")
    print(f"  source: {scene_path}")
    print(f"  output: {output_path}")

    # ---- Check depth H5 exists ----
    depth_h5_path = os.path.join(pointworld_output_dir, "depth", f"{uuid}_depth.h5")
    if not os.path.exists(depth_h5_path):
        print(f"  [SKIP] Depth H5 not found: {depth_h5_path}")
        return False

    # ---- 1. RGB + intrinsics from raw SVO (via ZED SDK) ----
    print("  Extracting RGB from SVO ...")
    data_dict = gather_data_dict(
        scene_path,
        downscale_ratio=downscale_ratio,
        include_stereo=False,
        include_depth=False,
    )

    # ---- 2. Proprioception ----
    print("  Extracting proprioception ...")
    proprio_dict = gather_trajectory(scene_path)

    # ---- 3. Canonical timestamps (from first camera, with time skip) ----
    first_cam_serial = list(data_dict.keys())[0]
    canonical_ts = data_dict[first_cam_serial]["timestamps"]
    if time_skip_ratio > 1:
        canonical_ts = canonical_ts[::time_skip_ratio]
    T = len(canonical_ts)
    print(f"  Canonical frames: {T}  (time_skip_ratio={time_skip_ratio})")

    # ---- 4. Align camera data & proprio to canonical timestamps ----
    data_dict = filter_by_timestamps(
        data_dict, canonical_ts, scene_path, is_proprio=False
    )
    proprio_dict = filter_by_timestamps(
        proprio_dict, canonical_ts, scene_path, is_proprio=True
    )

    # ---- 5. Pick the camera to use as "head_camera" (first ext camera) ----
    cam_serial = first_cam_serial
    cam_data = data_dict[cam_serial]
    rgb_all = cam_data["rgb"]  # (T, H_raw, W_raw, 3) uint8
    intrinsic_raw = cam_data["intrinsic"]  # (3, 3)
    H_raw, W_raw = rgb_all.shape[1], rgb_all.shape[2]

    # Determine depth camera group name (matches compute_depth.py convention)
    cam_group_name = f"{cam_serial}+ext"

    # ---- 6. Load depth from PointWorld depth H5 ----
    print("  Loading depth ...")
    depth_all = _load_depth_from_h5(depth_h5_path, cam_group_name, canonical_ts)
    # depth_all: (T, H_depth, W_depth) float32 metres
    H_depth, W_depth = depth_all.shape[1], depth_all.shape[2]

    # ---- 7. Build per-frame outputs ----
    print("  Generating per-frame outputs ...")
    rgb_frames_out = []
    depth_frames_out = np.empty((T, image_height, image_width), dtype=np.float32)
    pointclouds_out = np.empty((T, num_points, 6), dtype=np.float32)

    # Intrinsic at depth resolution (for pointcloud backprojection)
    intrinsic_depth = _scale_intrinsic(intrinsic_raw, (H_raw, W_raw), (H_depth, W_depth))

    for t in range(T):
        # --- RGB: resize to target ---
        rgb_t = rgb_all[t]  # (H_raw, W_raw, 3) uint8, RGB order
        rgb_resized = cv2.resize(
            rgb_t, (image_width, image_height), interpolation=cv2.INTER_AREA
        )
        rgb_frames_out.append(rgb_resized)

        # --- Depth: resize to target ---
        d_t = depth_all[t]  # (H_depth, W_depth) float32 metres
        d_resized = cv2.resize(
            d_t, (image_width, image_height), interpolation=cv2.INTER_NEAREST
        )
        depth_frames_out[t] = d_resized

        # --- Pointcloud: backproject at depth native res, then downsample ---
        # Use depth-res RGB for colour (resize rgb to depth res)
        rgb_for_pcd = cv2.resize(
            rgb_t, (W_depth, H_depth), interpolation=cv2.INTER_AREA
        )
        pcd = depth_to_pointcloud(
            d_t, intrinsic_depth, rgb=rgb_for_pcd,
            min_depth=0.01, max_depth=4.0,
        )
        pointclouds_out[t] = random_downsample(pcd, num_points, rng=rng)

    # ---- 8. Construct joint_state / endpose (pad to 14 dims) ----
    # DROID Franka: joint_positions (T, 7), gripper_positions (T,)
    jp = proprio_dict["joint_positions"]  # (T, 7)
    gp = proprio_dict["gripper_positions"]  # (T,) mapped 0-0.725
    # Concatenate joints + gripper → (T, 8), then pad → (T, 14)
    jp_with_gripper = np.concatenate([jp, gp[:, None]], axis=1).astype(np.float32)
    # But 8 > 7 so left arm = first 7 dims, pad the rest
    # We store all 7 joints in [0:7], then zeros in [7:14]
    joint_positions_14 = _pad_to_14(jp)  # (T, 14)  — pure joint angles

    # endpose: gripper_pose (T, 7) = xyz + quaternion → pad to 14
    endpose_14 = _pad_to_14(proprio_dict["gripper_pose"])  # (T, 14)

    # ---- 9. Write HDF5 ----
    print("  Writing HDF5 ...")
    write_episode_hdf5(
        output_path=output_path,
        rgb_frames=rgb_frames_out,
        depth_frames=depth_frames_out,
        joint_positions=joint_positions_14,
        endpose=endpose_14,
        pointclouds=pointclouds_out,
    )
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert DROID scenes to RoboTwin HDF5 format."
    )
    parser.add_argument(
        "--input", required=True,
        help="Text file with one DROID scene path per line (e.g. real/droid_paths.txt).",
    )
    parser.add_argument(
        "--pointworld_output_dir", required=True,
        help="Root of PointWorld processed outputs (must contain depth/ subdir).",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for RoboTwin HDF5 files.",
    )
    parser.add_argument("--downscale_ratio", type=float, default=0.5,
                        help="Downscale ratio used when extracting RGB from SVO.")
    parser.add_argument("--time_skip_ratio", type=int, default=2,
                        help="Temporal down-sampling (take every N-th camera frame).")
    parser.add_argument("--image_height", type=int, default=480)
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--num_points", type=int, default=2048,
                        help="Number of points per frame in the output pointcloud.")
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument(
        "--allow_gcs_streaming", action="store_true",
        help="Bypass GCS cache enforcement.",
    )
    args = parser.parse_args()

    # Read & partition scene paths
    scene_paths = _read_scene_paths(args.input)
    enforce_gcs_cache_policy(
        scene_paths,
        stage_name="convert_droid_to_robotwin",
        require_cache=True,
        allow_streaming=args.allow_gcs_streaming,
    )

    random.seed(args.seed)
    random.shuffle(scene_paths)
    if args.max_scenes is not None:
        scene_paths = scene_paths[: args.max_scenes]

    assert 0 <= args.rank < args.world_size
    worker_scenes = scene_paths[args.rank :: args.world_size]
    print(
        f"Worker {args.rank}/{args.world_size}: "
        f"{len(worker_scenes)}/{len(scene_paths)} scenes"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    success, fail = 0, 0
    pbar = tqdm(worker_scenes, desc=f"rank {args.rank}", unit="scene")
    for idx, scene_path in enumerate(pbar):
        try:
            uuid = get_uuid(scene_path)
        except Exception as e:
            print(f"  [ERROR] Cannot get UUID for {scene_path}: {e}")
            fail += 1
            continue

        out_path = os.path.join(args.output_dir, "data", f"episode{idx}.hdf5")

        try:
            ok = convert_one_scene(
                scene_path=scene_path,
                pointworld_output_dir=args.pointworld_output_dir,
                output_path=out_path,
                downscale_ratio=args.downscale_ratio,
                time_skip_ratio=args.time_skip_ratio,
                image_height=args.image_height,
                image_width=args.image_width,
                num_points=args.num_points,
                rng=rng,
            )
            if ok:
                success += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  [ERROR] {uuid}: {e}")
            fail += 1

    print(f"\nDone — success: {success}, failed/skipped: {fail}")


if __name__ == "__main__":
    main()
