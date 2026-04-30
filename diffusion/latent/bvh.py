from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from scipy.spatial.transform import Rotation as R
except Exception:
    R = None

from diffusion.feature_layout import MotionFeatureLayout


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def rotation_6d_to_matrix_col_safe(d6: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    a1_norm = torch.linalg.norm(a1, dim=-1, keepdim=True)
    b1 = a1 / torch.clamp(a1_norm, min=eps)
    proj = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = a2 - proj * b1
    b2_norm = torch.linalg.norm(b2, dim=-1, keepdim=True)
    b2 = b2 / torch.clamp(b2_norm, min=eps)
    b3 = torch.cross(b1, b2, dim=-1)
    b3_norm = torch.linalg.norm(b3, dim=-1, keepdim=True)
    mat = torch.stack((b1, b2, b3), dim=-1)
    bad = (a1_norm.squeeze(-1) < eps) | (b2_norm.squeeze(-1) < eps) | (b3_norm.squeeze(-1) < eps)
    bad = bad | (~torch.isfinite(mat).all(dim=(-1, -2)))
    if bad.any():
        eye = torch.eye(3, device=d6.device, dtype=d6.dtype)
        eye = eye.view(*([1] * (mat.ndim - 2)), 3, 3).expand_as(mat)
        mat = torch.where(bad[..., None, None], eye, mat)
    return mat


def parse_bvh_channel_blocks(ref_bvh_path: str):
    with open(ref_bvh_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    header_lines: List[str] = []
    blocks: List[Dict[str, Any]] = []
    current_joint: Optional[str] = None
    for line in lines:
        s = line.strip()
        header_lines.append(line)
        if s == "MOTION":
            break
        parts = s.split()
        if not parts:
            continue
        if parts[0] in ("ROOT", "JOINT"):
            current_joint = parts[1]
        elif parts[0] == "CHANNELS" and current_joint is not None:
            n = int(parts[1])
            chans = parts[2 : 2 + n]
            blocks.append(
                {
                    "name": current_joint,
                    "channels": chans,
                    "has_pos": any(c.endswith("position") for c in chans),
                    "has_rot": any(c.endswith("rotation") for c in chans),
                }
            )
    return header_lines, blocks


def count_rot_joints_in_bvh(ref_bvh_path: str) -> int:
    _, blocks = parse_bvh_channel_blocks(ref_bvh_path)
    return int(sum(1 for block in blocks if block["has_rot"]))


def read_ref_root_first_frame_xyz(ref_bvh_path: str) -> Optional[Tuple[float, float, float]]:
    with open(ref_bvh_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    try:
        motion_idx = lines.index("MOTION")
    except ValueError:
        return None
    first = None
    for line in lines[motion_idx + 3 :]:
        if line.strip():
            first = line.strip()
            break
    if first is None:
        return None
    values = [float(v) for v in first.split()]
    _, blocks = parse_bvh_channel_blocks(ref_bvh_path)
    cursor = 0
    for block in blocks:
        if block["has_pos"]:
            xyz = {"Xposition": 0.0, "Yposition": 0.0, "Zposition": 0.0}
            for channel in block["channels"]:
                if channel in xyz and cursor < len(values):
                    xyz[channel] = values[cursor]
                cursor += 1
            return (xyz["Xposition"], xyz["Yposition"], xyz["Zposition"])
        cursor += len(block["channels"])
    return None


def _make_quat_continuous(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    for idx in range(1, out.shape[0]):
        if float(np.dot(out[idx - 1], out[idx])) < 0.0:
            out[idx] *= -1.0
    return out


def decode_motion_to_bvh(
    motion_denorm: torch.Tensor,
    *,
    layout: MotionFeatureLayout,
    fps: int,
    motion_dim: int,
    root_init_xz: Optional[Tuple[float, float]] = None,
    euler_order: str = "XYZ",
    unwrap_euler: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if R is None:
        raise RuntimeError("scipy is required for BVH rotation decoding")
    assert motion_denorm.ndim == 2 and motion_denorm.shape[1] == motion_dim
    dt = 1.0 / float(fps)
    root_pos = layout.decode_root_pos(motion_denorm, dt)
    if root_init_xz is not None:
        offset_x = float(root_init_xz[0]) - float(root_pos[0, 0].item())
        offset_z = float(root_init_xz[1]) - float(root_pos[0, 2].item())
        root_pos = root_pos.clone()
        root_pos[:, 0] += offset_x
        root_pos[:, 2] += offset_z

    rot_data = motion_denorm[:, layout.rot6d_start :]
    if rot_data.shape[1] % 6 != 0:
        raise ValueError(f"Invalid rot slice shape: {rot_data.shape}")
    joints = rot_data.shape[1] // 6
    rot6d = rot_data.view(motion_denorm.shape[0], joints, 6)
    rot_mats = rotation_6d_to_matrix_col_safe(rot6d)
    rot_np = _to_numpy(rot_mats).reshape(-1, 3, 3)
    quat = R.from_matrix(rot_np).as_quat().reshape(motion_denorm.shape[0], joints, 4)
    for joint_idx in range(joints):
        quat[:, joint_idx] = _make_quat_continuous(quat[:, joint_idx])
    euler = R.from_quat(quat.reshape(-1, 4)).as_euler(euler_order, degrees=True).reshape(motion_denorm.shape[0], joints, 3)
    if unwrap_euler:
        euler = np.rad2deg(np.unwrap(np.deg2rad(euler), axis=0))
    return _to_numpy(root_pos), euler


def save_bvh_remapped(
    root_pos: np.ndarray,
    euler_rots: np.ndarray,
    *,
    ref_bvh_path: str,
    output_path: str,
    fps: int,
) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header_lines, blocks = parse_bvh_channel_blocks(ref_bvh_path)
    frames = int(root_pos.shape[0])
    frame_time = 1.0 / float(fps)

    lines = list(header_lines)
    lines.append(f"Frames: {frames}")
    lines.append(f"Frame Time: {frame_time:.6f}")

    rot_cursor = 0
    for frame_idx in range(frames):
        values: List[float] = []
        rot_cursor = 0
        for block in blocks:
            for channel in block["channels"]:
                if channel == "Xposition":
                    values.append(float(root_pos[frame_idx, 0]))
                elif channel == "Yposition":
                    values.append(float(root_pos[frame_idx, 1]))
                elif channel == "Zposition":
                    values.append(float(root_pos[frame_idx, 2]))
                elif channel.endswith("rotation"):
                    axis = "XYZ".index(channel[0].upper())
                    values.append(float(euler_rots[frame_idx, rot_cursor, axis]))
            if block["has_rot"]:
                rot_cursor += 1
        lines.append(" ".join(f"{value:.6f}" for value in values))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
