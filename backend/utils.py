# backend/utils.py
# 使用 BEAT 75 关节参考骨架，将 6D local rotation 转为 BVH
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

# 参考 BVH（仅用于层级/offset），与训练用数据一致
REF_BVH_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "beat/beat_english_v0.2.1/24/24_kexin_0_3_3.bvh",
)


def _parse_bvh_header(path):
    lines = open(path, "r", errors="ignore").read().splitlines()
    try:
        motion_idx = lines.index("MOTION")
    except ValueError as e:
        raise RuntimeError(f"[BVH Gen] No MOTION section in {path}") from e

    header = lines[:motion_idx]  # 不含 MOTION
    joint_names = []
    parents = []
    offsets = []
    stack = []
    last_joint_idx = -1

    for ln in header:
        t = ln.strip()
        if t.startswith("ROOT") or t.startswith("JOINT"):
            name = t.split()[1]
            joint_idx = len(joint_names)
            joint_names.append(name)
            parent = stack[-1] if stack else -1
            parents.append(parent)
            last_joint_idx = joint_idx
        if t.startswith("OFFSET"):
            vals = [float(x) for x in t.split()[1:]]
            offsets.append(tuple(vals))
        if t.endswith("{"):
            if last_joint_idx >= 0:
                stack.append(last_joint_idx)
        if t == "}":
            if stack:
                stack.pop()
    return header, joint_names, parents, offsets


REF_HEADER_LINES, REF_JOINT_NAMES, REF_PARENTS, REF_OFFSETS = _parse_bvh_header(REF_BVH_PATH)


def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-1)


def motion_to_bvh_string(motion_data):
    print("[BVH Gen] Using BEAT 75-joint hierarchy (local rotations, intrinsic XYZ)")
    motion_data = np.nan_to_num(motion_data, nan=0.0)
    T = len(motion_data)
    num_joints = len(REF_JOINT_NAMES)
    expected_rot_dim = num_joints * 6

    if motion_data.shape[1] == expected_rot_dim:
        root_pos = np.zeros((T, 3))
        rot_flat = motion_data.reshape(-1, 6)
    elif motion_data.shape[1] == expected_rot_dim + 3:
        root_pos = motion_data[:, :3]
        rot_flat = motion_data[:, 3:].reshape(-1, 6)
    else:
        clipped = min(motion_data.shape[1], expected_rot_dim)
        padded = np.zeros((T, expected_rot_dim))
        padded[:, :clipped] = motion_data[:, :clipped]
        root_pos = np.zeros((T, 3))
        rot_flat = padded.reshape(-1, 6)

    mats = rotation_6d_to_matrix(rot_flat).reshape(T, num_joints, 3, 3)

    # 如果输入是 global rotation，改为 True 可转换为 local
    ROT_INPUT_IS_GLOBAL = False
    if ROT_INPUT_IS_GLOBAL:
        local_mats = np.empty_like(mats)
        for t in range(T):
            for j in range(num_joints):
                p = REF_PARENTS[j]
                if p < 0:
                    local_mats[t, j] = mats[t, j]
                else:
                    local_mats[t, j] = np.linalg.inv(mats[t, p]) @ mats[t, j]
    else:
        local_mats = mats

    eulers = R.from_matrix(local_mats.reshape(-1, 3, 3)).as_euler("XYZ", degrees=True)
    eulers = eulers.reshape(T, num_joints, 3)

    lines = []
    lines.extend(REF_HEADER_LINES)
    lines.append("MOTION")
    lines.append(f"Frames: {T}")
    lines.append("Frame Time: 0.066667")

    for t in range(T):
        row = []
        row.extend(root_pos[t])
        row.extend(eulers[t].reshape(-1))
        lines.append(" ".join(f"{x:.6f}" for x in row))

    return "\n".join(lines)
