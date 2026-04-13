"""BVH forward kinematics — global-rotation variant (Kimodo-style).

Kimodo stores **global** joint rotations (each joint's rotation is expressed in
world frame, not relative to parent). Given global rotations, joint global
positions can be computed by:

    pos[j] = pos[parent] + R_global[parent] @ offset_local[j]

(where offset_local[j] is the joint's rest-pose offset from its parent, as stored
in the BVH hierarchy).

This module:
- builds a skeleton struct from a BEAT BVH file
- converts between local and global rotations
- computes global joint positions from either local or global rotations
- provides a self-test comparing against the existing local-rot FK at
  stageA/train_stage1_vqvae.py:1303

Conventions:
- rotations: 6D continuous representation (Zhou et al. 2019)
- tensor shapes: [B, T, ...] leading batch + time; rotations are [..., J, 6]
- offsets: [J, 3] float32, parents: [J] int64 with root = -1
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 6D rotation helpers (copied from stageA/train_stage1_vqvae.py for independence)
# ---------------------------------------------------------------------------


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert continuous 6D rotation (Zhou+2019) to a 3x3 rotation matrix.

    d6 shape [..., 6] -> [..., 3, 3].
    """
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rotation_6d(mat: torch.Tensor) -> torch.Tensor:
    """Inverse of rotation_6d_to_matrix.

    Stores the first two COLUMNS of `mat` as a flat 6D vector [col0, col1].
    Matches stageA/train_stage1_vqvae.py:89 convention (transpose-then-flatten).
    """
    return mat[..., :2].transpose(-1, -2).flatten(-2)


# ---------------------------------------------------------------------------
# Skeleton info
# ---------------------------------------------------------------------------


@dataclass
class SkeletonInfo:
    """Lightweight skeleton container: parents, offsets, active-joint mask.

    active_indices: indices of joints that carry rotation data in the packed
    motion representation (non-active joints are treated as identity rotation).
    """

    parents: np.ndarray          # [J] int64, root = -1
    offsets: np.ndarray          # [J, 3] float32, local offsets from parent
    joint_names: List[str]
    active_indices: np.ndarray   # [N_active] int64

    @property
    def n_joints(self) -> int:
        return int(self.offsets.shape[0])

    @property
    def n_active(self) -> int:
        return int(self.active_indices.shape[0])

    def to_torch(self, device: torch.device | str = "cpu") -> "SkeletonInfoT":
        return SkeletonInfoT(
            parents=torch.from_numpy(self.parents).to(device=device, dtype=torch.long),
            offsets=torch.from_numpy(self.offsets).to(device=device, dtype=torch.float32),
            active_indices=torch.from_numpy(self.active_indices).to(device=device, dtype=torch.long),
            joint_names=self.joint_names,
        )


@dataclass
class SkeletonInfoT:
    parents: torch.Tensor
    offsets: torch.Tensor
    active_indices: torch.Tensor
    joint_names: List[str]

    @property
    def n_joints(self) -> int:
        return int(self.offsets.shape[0])

    @property
    def n_active(self) -> int:
        return int(self.active_indices.shape[0])


def build_skeleton_info(bvh_path: Path | str) -> SkeletonInfo:
    """Parse a BVH and return (parents, offsets, active joint indices).

    Active = joints that declare rotation channels. This matches the convention
    used by the existing StageA pipeline (see train_stage1_vqvae.py:1281-1289).
    """
    # reuse existing BVH parser
    from stageA.train_stage1_vqvae import BVHSkeleton

    skel = BVHSkeleton.from_bvh(Path(bvh_path))
    parents = np.array([j.parent for j in skel.joints], dtype=np.int64)
    offsets = np.stack([j.offset for j in skel.joints], axis=0).astype(np.float32)
    joint_names = [j.name for j in skel.joints]
    active_indices = np.array(
        [i for i, j in enumerate(skel.joints)
         if any("rotation" in ch.lower() for ch in j.channels)],
        dtype=np.int64,
    )
    return SkeletonInfo(
        parents=parents,
        offsets=offsets,
        joint_names=joint_names,
        active_indices=active_indices,
    )


# ---------------------------------------------------------------------------
# Local / global rotation conversion
# ---------------------------------------------------------------------------


def local_rotmat_to_global(local_rotmat: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    """Propagate local rotations into world frame.

    local_rotmat: [..., J, 3, 3], parents: [J] int64 (root = -1).
    Returns global rotations [..., J, 3, 3].
    """
    J = local_rotmat.shape[-3]
    out_list: List[torch.Tensor] = []
    parents_py = parents.detach().cpu().tolist()
    for j in range(J):
        p = int(parents_py[j])
        R_loc = local_rotmat[..., j, :, :]
        if p == -1:
            out_list.append(R_loc)
        else:
            out_list.append(out_list[p] @ R_loc)
    return torch.stack(out_list, dim=-3)


def global_rotmat_to_local(global_rotmat: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    """Inverse of local_rotmat_to_global.

    R_local[j] = R_global[parent].T @ R_global[j]
    """
    J = global_rotmat.shape[-3]
    out_list: List[torch.Tensor] = []
    parents_py = parents.detach().cpu().tolist()
    for j in range(J):
        p = int(parents_py[j])
        R_g = global_rotmat[..., j, :, :]
        if p == -1:
            out_list.append(R_g)
        else:
            R_par = global_rotmat[..., p, :, :]
            out_list.append(R_par.transpose(-1, -2) @ R_g)
    return torch.stack(out_list, dim=-3)


# ---------------------------------------------------------------------------
# Forward kinematics (positions)
# ---------------------------------------------------------------------------


def fk_positions_from_global_rot(
    global_rot_6d: torch.Tensor,   # [B, T, J_all, 6] -- dense (all joints, identity for inactive)
    root_pos: torch.Tensor,        # [B, T, 3]
    skel: SkeletonInfoT,
) -> torch.Tensor:
    """Compute global joint positions given **global** rotations.

    pos[j] = pos[parent] + R_global[parent] @ offset_local[j]

    Root position is `root_pos` shifted by the root joint's offset if non-zero.
    """
    assert global_rot_6d.dim() == 4 and global_rot_6d.shape[-1] == 6, global_rot_6d.shape
    assert root_pos.dim() == 3 and root_pos.shape[-1] == 3, root_pos.shape

    B, T, J, _ = global_rot_6d.shape
    assert J == skel.n_joints, f"{J} vs {skel.n_joints}"

    global_rot_6d = global_rot_6d.to(dtype=torch.float32)
    root_pos = root_pos.to(dtype=torch.float32)

    R_g = rotation_6d_to_matrix(global_rot_6d)  # [B, T, J, 3, 3]
    offsets = skel.offsets.to(device=global_rot_6d.device)
    parents_py = skel.parents.detach().cpu().tolist()

    pos_list: List[torch.Tensor] = []
    for j in range(J):
        p = int(parents_py[j])
        off_j = offsets[j].view(1, 1, 3).expand(B, T, 3)
        if p == -1:
            # root — place it at root_pos + its own offset
            pos_list.append(root_pos + off_j)
        else:
            # parent_pos + R_g[parent] @ offset
            R_par = R_g[:, :, p]  # [B, T, 3, 3]
            t_par = pos_list[p]
            # R_par @ off_j  (treat off_j as column vector)
            rotated = torch.einsum("btij,btj->bti", R_par, off_j)
            pos_list.append(t_par + rotated)
    return torch.stack(pos_list, dim=2)  # [B, T, J, 3]


def fk_positions_from_local_rot(
    local_rot_6d_active: torch.Tensor,  # [B, T, N_active, 6]
    root_pos: torch.Tensor,             # [B, T, 3]
    skel: SkeletonInfoT,
) -> torch.Tensor:
    """FK from **local** rotations (matches the existing BVHFkTorch path).

    Useful for sanity-checking the global-rot FK against the known-correct
    local-rot FK.
    """
    B, T, N_active, _ = local_rot_6d_active.shape
    assert N_active == skel.n_active

    # pack active rotations into a dense [B, T, J, 6], identity elsewhere
    device = local_rot_6d_active.device
    identity_6d = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        device=device,
        dtype=local_rot_6d_active.dtype,
    ).view(1, 1, 1, 6)
    J = skel.n_joints
    rot6d_full = identity_6d.expand(B, T, J, 6).clone()
    active_idx = skel.active_indices.to(device)
    rot6d_full[:, :, active_idx, :] = local_rot_6d_active

    rot_mat_local = rotation_6d_to_matrix(rot6d_full)  # [B,T,J,3,3]
    parents_py = skel.parents.detach().cpu().tolist()
    offsets = skel.offsets.to(device=device)

    R_list: List[torch.Tensor] = []
    pos_list: List[torch.Tensor] = []
    for j in range(J):
        p = int(parents_py[j])
        off_j = offsets[j].view(1, 1, 3).expand(B, T, 3)
        R_loc = rot_mat_local[:, :, j]
        if p == -1:
            # root: global rotation = local rotation, position = root_pos + own offset
            R_list.append(R_loc)
            pos_list.append(root_pos + off_j)
        else:
            R_par = R_list[p]
            R_g = R_par @ R_loc
            R_list.append(R_g)
            rotated = torch.einsum("btij,btj->bti", R_par, off_j)
            pos_list.append(pos_list[p] + rotated)
    return torch.stack(pos_list, dim=2)


def active_local_6d_to_global_6d(
    local_rot_6d_active: torch.Tensor,  # [B, T, N_active, 6]
    skel: SkeletonInfoT,
) -> torch.Tensor:
    """Convert active-joint local 6D rotations to full-skeleton global 6D rotations.

    Non-active joints (no rotation channels in BVH) are treated as identity,
    which is consistent with the existing pipeline.

    Returns [B, T, J_all, 6].
    """
    B, T, N_active, _ = local_rot_6d_active.shape
    device = local_rot_6d_active.device
    identity_6d = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        device=device,
        dtype=local_rot_6d_active.dtype,
    ).view(1, 1, 1, 6)
    J = skel.n_joints
    rot6d_full_local = identity_6d.expand(B, T, J, 6).clone()
    active_idx = skel.active_indices.to(device)
    rot6d_full_local[:, :, active_idx, :] = local_rot_6d_active

    rot_mat_local = rotation_6d_to_matrix(rot6d_full_local)
    parents_t = skel.parents.to(device)
    rot_mat_global = local_rotmat_to_global(rot_mat_local, parents_t)
    return matrix_to_rotation_6d(rot_mat_global)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test(bvh_path: Path, n_frames: int = 32, device: str = "cpu") -> None:
    """Load a BEAT BVH, run:
      1) existing local-rot FK (BVHFkTorch.fk_positions)
      2) our local-rot FK (sanity: same formula, should match exactly)
      3) our global-rot FK (via local→global 6D conversion)
    All three should give the same joint positions (to float32 precision).
    """
    import time

    from stageA.train_stage1_vqvae import BVHFkTorch
    from stageB.common.bvh_io import load_full_motion_from_bvh

    print(f"[fk.test] loading BVH: {bvh_path}")
    t0 = time.time()
    skel_info = build_skeleton_info(bvh_path)
    print(f"[fk.test] parsed skeleton: J={skel_info.n_joints}, N_active={skel_info.n_active} ({time.time()-t0:.2f}s)")

    full_motion = load_full_motion_from_bvh(Path(bvh_path))[:n_frames]  # [T, 3 + N_active*6]
    print(f"[fk.test] loaded motion: shape={full_motion.shape}")

    # existing FK
    from stageA.train_stage1_vqvae import BVHSkeleton
    skel = BVHSkeleton.from_bvh(Path(bvh_path))
    fk_old = BVHFkTorch(skel, drop_root_pos=False)
    fk_old.to_device_tensors(torch.device(device))

    motion_t = torch.from_numpy(full_motion).unsqueeze(0).to(device=device, dtype=torch.float32)
    pos_old = fk_old.fk_positions(motion_t)  # [1, T, J, 3]
    print(f"[fk.test] existing FK output: {tuple(pos_old.shape)}")

    # our local FK
    skel_t = skel_info.to_torch(device)
    root_pos = motion_t[:, :, :3]
    local6d_active = motion_t[:, :, 3:].view(1, n_frames, skel_info.n_active, 6)
    pos_local_ours = fk_positions_from_local_rot(local6d_active, root_pos, skel_t)
    err_local = (pos_old - pos_local_ours).abs().max().item()
    print(f"[fk.test] local-FK max err vs existing: {err_local:.2e}")

    # our global FK (via conversion)
    global6d_full = active_local_6d_to_global_6d(local6d_active, skel_t)
    pos_global_ours = fk_positions_from_global_rot(global6d_full, root_pos, skel_t)
    err_global = (pos_old - pos_global_ours).abs().max().item()
    print(f"[fk.test] global-FK max err vs existing: {err_global:.2e}")

    # summary
    tol = 1e-3  # mm-scale; we're in BVH units (usually cm for BEAT)
    ok = err_local < tol and err_global < tol
    print(f"[fk.test] tolerance={tol:.0e}  PASS" if ok else f"[fk.test] tolerance={tol:.0e}  FAIL")
    if not ok:
        raise AssertionError(f"FK self-test failed: local_err={err_local:.2e}, global_err={err_global:.2e}")


def _cli():
    import argparse
    import sys

    if __package__ is None or __package__ == "":
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument(
        "--bvh",
        type=str,
        default="beat/beat_english_v0.2.1/14/14_zhang_0_96_96.bvh",
    )
    ap.add_argument("--n_frames", type=int, default=32)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    bvh_path = Path(args.bvh)
    if not bvh_path.is_absolute():
        root = Path(__file__).resolve().parents[2]
        bvh_path = root / bvh_path

    if args.test:
        _self_test(bvh_path, n_frames=args.n_frames, device=args.device)
    else:
        ap.print_help()


if __name__ == "__main__":
    _cli()
