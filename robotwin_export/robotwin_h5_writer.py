"""
RoboTwin HDF5 writer — writes one episode in the target format.

Target schema (matching RoboTwin / read_hdf5.py):

    endpose:                    (T, 14)   float32
    joint_action/vector:        (T, 14)   float32  — Action[t] = State[t+1]
    joint_state/effort:         (T, 14)   float32  — zeros (unused)
    joint_state/position:       (T, 14)   float32
    joint_state/vector:         (T, 14)   float32  — same as position
    joint_state/velocity:       (T, 14)   float32  — zeros (unused)
    observation/head_camera/depth: (T, H, W) float32
    observation/head_camera/rgb:   (T,)      bytes (JPEG-encoded)
    pointcloud:                 (T, N, 6) float32
"""

import os
import h5py
import numpy as np
import cv2


def _encode_rgb_jpeg(rgb_image, quality=95):
    """Encode an RGB uint8 image to JPEG bytes."""
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def write_episode_hdf5(
    output_path,
    rgb_frames,       # list[np.ndarray]  len=T, each (H, W, 3) uint8 RGB
    depth_frames,      # (T, H_out, W_out) float32  meters
    joint_positions,   # (T, 14) float32
    endpose,           # (T, 14) float32
    pointclouds,       # (T, n_points, 6) float32
    jpeg_quality=95,
):
    """
    Write a single episode HDF5 file in RoboTwin format.

    ``joint_action/vector`` is derived automatically as the next-frame
    absolute joint position:  action[t] = position[t+1], action[-1] = position[-1].

    ``joint_state/effort`` and ``joint_state/velocity`` are filled with zeros
    because DROID does not provide usable torque/velocity data in this context.
    """
    T = len(rgb_frames)
    assert depth_frames.shape[0] == T, "depth_frames length mismatch"
    assert joint_positions.shape == (T, 14), f"joint_positions shape {joint_positions.shape}"
    assert endpose.shape == (T, 14), f"endpose shape {endpose.shape}"
    assert pointclouds.shape[0] == T, "pointclouds length mismatch"

    # Action = next-frame state (absolute position control)
    joint_action = np.zeros_like(joint_positions)
    joint_action[:-1] = joint_positions[1:]
    joint_action[-1] = joint_positions[-1]

    # Zeros for unused fields
    zeros_14 = np.zeros((T, 14), dtype=np.float32)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # ---- endpose ----
        f.create_dataset("endpose", data=endpose.astype(np.float32))

        # ---- joint_action ----
        ja = f.create_group("joint_action")
        ja.create_dataset("vector", data=joint_action.astype(np.float32))

        # ---- joint_state ----
        js = f.create_group("joint_state")
        js.create_dataset("effort", data=zeros_14)
        js.create_dataset("position", data=joint_positions.astype(np.float32))
        js.create_dataset("vector", data=joint_positions.astype(np.float32))
        js.create_dataset("velocity", data=zeros_14)

        # ---- observation/head_camera ----
        obs = f.create_group("observation")
        cam = obs.create_group("head_camera")

        # RGB: JPEG-encoded byte strings (matching RoboTwin convention)
        jpeg_list = []
        max_len = 0
        for i in range(T):
            jbytes = _encode_rgb_jpeg(rgb_frames[i], quality=jpeg_quality)
            jpeg_list.append(jbytes)
            max_len = max(max_len, len(jbytes))

        # Pad to fixed length so the dataset is a regular S{max_len} array
        padded = [jb.ljust(max_len, b"\x00") for jb in jpeg_list]
        cam.create_dataset("rgb", data=padded, dtype=f"S{max_len}")

        # Depth
        cam.create_dataset("depth", data=depth_frames.astype(np.float32))

        # ---- pointcloud ----
        f.create_dataset("pointcloud", data=pointclouds.astype(np.float32))

    print(f"  Wrote {output_path}  T={T}")
