from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from stageA.train_stage1_vqvae import (
    TARGET_FPS,
    BVHSkeleton,
    build_active_joint_map,
    convert_bvh_to_6d_with_channel_order,
    load_bvh_channels,
    resample_motion_linear,
    rotation_6d_to_matrix,
    unwrap_bvh_angles_degrees,
)


def read_bvh_header_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    try:
        motion_idx = lines.index("MOTION")
    except ValueError:
        return lines
    return lines[: motion_idx + 1]


def write_bvh(path_out: Path, header_lines: list[str], motion: np.ndarray, frame_time: float) -> None:
    path_out.parent.mkdir(parents=True, exist_ok=True)
    lines = list(header_lines)
    lines.append(f"Frames: {int(motion.shape[0])}")
    lines.append(f"Frame Time: {float(frame_time):.6f}")
    for row in motion:
        lines.append(" ".join(f"{float(v):.6f}" for v in row))
    with path_out.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def load_full_motion_from_bvh(
    bvh_path: Path,
    *,
    skel: Optional[BVHSkeleton] = None,
    active_joint_map=None,
    fps: int = TARGET_FPS,
) -> np.ndarray:
    if skel is None:
        skel = BVHSkeleton.from_bvh(bvh_path)
    if active_joint_map is None:
        active_joint_map = build_active_joint_map(skel)

    raw_motion, frame_time = load_bvh_channels(bvh_path)
    if raw_motion is None or raw_motion.ndim != 2:
        raise RuntimeError(f"Failed to parse BVH motion: {bvh_path}")

    raw_motion = unwrap_bvh_angles_degrees(raw_motion, pos_dims=3)
    raw_motion = resample_motion_linear(raw_motion, frame_time, int(fps))
    full_motion = convert_bvh_to_6d_with_channel_order(raw_motion, skel, active_joint_map).astype(np.float32)
    if not np.isfinite(full_motion).all():
        raise RuntimeError(f"Non-finite canonical motion from BVH: {bvh_path}")
    return full_motion


def full_motion_to_bvh_channels(
    full_motion: np.ndarray,
    skel: BVHSkeleton,
    active_joint_map,
) -> np.ndarray:
    T = int(full_motion.shape[0])
    out = np.zeros((T, len(skel.channel_names)), dtype=np.float32)

    for axis_idx, ch_idx in enumerate(active_joint_map.root_pos_indices):
        if ch_idx is not None:
            out[:, int(ch_idx)] = full_motion[:, axis_idx]

    rot_full = full_motion[:, 3:].reshape(T, active_joint_map.n_active, 6)
    rot_mats = rotation_6d_to_matrix(torch.from_numpy(rot_full)).cpu().numpy()

    for order_channels, channel_indices, mats in zip(
        active_joint_map.active_rot_orders,
        active_joint_map.active_rot_channel_indices,
        rot_mats.transpose(1, 0, 2, 3),
    ):
        order = "".join(ch[0].upper() for ch in order_channels)
        euler_deg = R.from_matrix(mats).as_euler(order, degrees=True).astype(np.float32)
        euler_rad = np.unwrap(np.deg2rad(euler_deg), axis=0)
        euler_deg = np.rad2deg(euler_rad).astype(np.float32)
        for local_idx, ch_idx in enumerate(channel_indices):
            out[:, int(ch_idx)] = euler_deg[:, local_idx]

    return out


def export_canonical_full_motion_to_bvh(
    full_motion: np.ndarray,
    *,
    ref_bvh: Path,
    output_path: Path,
    header_source: Optional[Path] = None,
    fps: int = TARGET_FPS,
) -> Tuple[np.ndarray, Path]:
    skel = BVHSkeleton.from_bvh(ref_bvh)
    active_joint_map = build_active_joint_map(skel)
    channels = full_motion_to_bvh_channels(full_motion, skel, active_joint_map)
    header_lines = read_bvh_header_lines(header_source or ref_bvh)
    write_bvh(output_path, header_lines, channels, frame_time=1.0 / float(fps))
    return channels, output_path


def load_root_track_from_bvh(
    bvh_path: Path,
    *,
    fps: int = TARGET_FPS,
    target_len: Optional[int] = None,
) -> np.ndarray:
    raw_motion, frame_time = load_bvh_channels(bvh_path)
    if raw_motion is None or raw_motion.ndim != 2:
        raise RuntimeError(f"Failed to parse root track from BVH: {bvh_path}")

    raw_motion = unwrap_bvh_angles_degrees(raw_motion, pos_dims=3)
    root_xyz = resample_motion_linear(raw_motion[:, :3], frame_time, int(fps)).astype(np.float32)
    if target_len is None or int(target_len) == int(root_xyz.shape[0]):
        return root_xyz

    if int(target_len) <= 1 or root_xyz.shape[0] <= 1:
        return np.repeat(root_xyz[:1], int(target_len), axis=0).astype(np.float32)

    src_t = np.linspace(0.0, 1.0, num=root_xyz.shape[0], dtype=np.float32)
    dst_t = np.linspace(0.0, 1.0, num=int(target_len), dtype=np.float32)
    out = np.zeros((int(target_len), 3), dtype=np.float32)
    for axis in range(3):
        out[:, axis] = np.interp(dst_t, src_t, root_xyz[:, axis]).astype(np.float32)
    return out
