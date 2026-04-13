"""Kimodo-style cache builder.

Builds a cache with the 6-item Kimodo motion representation:
  r_p  [3]    smoothed global root position (xz smoothed, y global)
  r_a  [2]    heading as [cos ψ, sin ψ]
  j_p  [3J]   global joint positions (xz relative to smoothed root, y global)
  j_v  [3J]   global joint velocities (frame-to-frame delta of j_p)
  j_a  [6J]   global joint 6D rotations (all J joints, identity for inactive)
  f    [n_foot]  binary foot contact flags

Only pos/vel/foot are z-score normalized; 6D rotations are on the unit sphere
and are NOT normalized (preserves geometry).

Also stores skeleton offsets + parents + active-joint indices so that training
can run FK during loss computation.

Output: a torch `.pt` payload with:
  {
    "segments": [ { "segment_id", "src_bvh", "r_p", "r_a", "j_p", "j_v",
                    "j_a", "f", "text" (optional) }, ... ],
    "mean": {"r_p": [...], "j_p": [...], "j_v": [...], "f": [...]},
    "std":  {...},
    "skeleton": {"parents": [J], "offsets": [J,3],
                 "active_indices": [N_active], "joint_names": [...],
                 "foot_joint_indices": [n_foot]},
    "fps": 30,
    "block_size": int,
  }
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusion_learn.kimodo_lite.fk import (
    SkeletonInfo,
    active_local_6d_to_global_6d,
    build_skeleton_info,
    fk_positions_from_local_rot,
)
from stageB.common.bvh_io import load_full_motion_from_bvh


TARGET_FPS = 30
FOOT_JOINT_KEYWORDS = ("LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase")
FOOT_VEL_THRESHOLD = 0.02  # cm per frame; tuned for BEAT units


def _gaussian_kernel_1d(sigma: float, radius: int | None = None) -> np.ndarray:
    if radius is None:
        radius = int(max(1, math.ceil(sigma * 3.0)))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (xs / sigma) ** 2)
    k = k / k.sum()
    return k


def _smooth_1d(x: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """Apply a 1D Gaussian filter along the time axis with reflect padding."""
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    else:
        squeeze = False
    k = _gaussian_kernel_1d(sigma)
    r = (k.shape[0] - 1) // 2
    pad = np.pad(x, ((r, r), (0, 0)), mode="edge")
    out = np.empty_like(x)
    for d in range(x.shape[1]):
        out[:, d] = np.convolve(pad[:, d], k, mode="valid")
    return out[:, 0] if squeeze else out


def _compute_heading_from_rot_mat(root_rotmat_global: np.ndarray) -> np.ndarray:
    """Compute 2D heading [cos ψ, sin ψ] from root global rotation matrix.

    Convention: forward axis is local +Z (standard BVH). Project forward onto xz
    plane, compute yaw angle ψ = atan2(forward_x, forward_z).
    """
    # root_rotmat_global: [T, 3, 3]. forward in world = R @ [0,0,1]^T = R[:, :, 2].
    forward = root_rotmat_global[..., :, 2]  # [T, 3]
    # project to xz plane
    yaw = np.arctan2(forward[..., 0], forward[..., 2])  # [T]
    return np.stack([np.cos(yaw), np.sin(yaw)], axis=-1).astype(np.float32)


def _find_foot_joint_indices(info: SkeletonInfo) -> np.ndarray:
    idxs = []
    for kw in FOOT_JOINT_KEYWORDS:
        for i, name in enumerate(info.joint_names):
            if name == kw:
                idxs.append(i)
                break
    if len(idxs) != len(FOOT_JOINT_KEYWORDS):
        found = [info.joint_names[i] for i in idxs]
        raise RuntimeError(
            f"Could not locate all foot joints. wanted {FOOT_JOINT_KEYWORDS}, found {found}"
        )
    return np.array(idxs, dtype=np.int64)


def compute_kimodo_representation(
    full_motion: np.ndarray,  # [T, 3 + N_active*6]  (root_pos + local 6D per active joint)
    info: SkeletonInfo,
    smooth_sigma: float = 15.0,
    foot_vel_threshold: float = FOOT_VEL_THRESHOLD,
) -> Dict[str, np.ndarray]:
    """Compute the Kimodo 6-item representation from one motion clip.

    Returns a dict with keys r_p, r_a, j_p, j_v, j_a, f (all numpy float32).
    """
    T = int(full_motion.shape[0])
    root_pos = full_motion[:, :3].astype(np.float32)  # [T, 3]
    local6d_active = full_motion[:, 3:].reshape(T, info.n_active, 6).astype(np.float32)

    skel_t = info.to_torch("cpu")
    mot_t = torch.from_numpy(full_motion).unsqueeze(0).float()  # [1, T, D]
    root_pos_t = torch.from_numpy(root_pos).unsqueeze(0).float()
    local6d_t = torch.from_numpy(local6d_active).unsqueeze(0).float()

    # j_p (global joint positions) via existing local-FK (same result)
    j_p_global = fk_positions_from_local_rot(local6d_t, root_pos_t, skel_t).squeeze(0).numpy()
    # shape [T, J, 3]

    # j_a (global joint 6D rotations, all J joints)
    j_a_global = active_local_6d_to_global_6d(local6d_t, skel_t).squeeze(0).numpy()
    # shape [T, J, 6]

    # compute root rotation matrix globally for heading
    from diffusion_learn.kimodo_lite.fk import rotation_6d_to_matrix
    root_rotmat = rotation_6d_to_matrix(torch.from_numpy(j_a_global[:, 0:1, :])).squeeze(1).numpy()
    heading = _compute_heading_from_rot_mat(root_rotmat)  # [T, 2]

    # smooth xz of root position; keep y
    r_p_smoothed = root_pos.copy()
    r_p_smoothed[:, 0] = _smooth_1d(root_pos[:, 0], sigma=smooth_sigma)
    r_p_smoothed[:, 2] = _smooth_1d(root_pos[:, 2], sigma=smooth_sigma)

    # subtract smoothed root xz from joint positions (xz only), keep y global
    j_p_rel = j_p_global.copy()
    j_p_rel[..., 0] -= r_p_smoothed[:, None, 0]
    j_p_rel[..., 2] -= r_p_smoothed[:, None, 2]

    # joint velocities (finite differences, zero-pad last frame)
    j_v = np.zeros_like(j_p_global)
    j_v[:-1] = j_p_global[1:] - j_p_global[:-1]

    # foot contact: velocity magnitude below threshold
    foot_idx = _find_foot_joint_indices(info)
    foot_vel_mag = np.linalg.norm(j_v[:, foot_idx, :], axis=-1)  # [T, n_foot]
    f = (foot_vel_mag < foot_vel_threshold).astype(np.float32)

    return {
        "r_p": r_p_smoothed.astype(np.float32),          # [T, 3]
        "r_a": heading.astype(np.float32),               # [T, 2]
        "j_p": j_p_rel.reshape(T, -1).astype(np.float32),  # [T, 3J]
        "j_v": j_v.reshape(T, -1).astype(np.float32),      # [T, 3J]
        "j_a": j_a_global.reshape(T, -1).astype(np.float32),  # [T, 6J]
        "f": f.astype(np.float32),                        # [T, n_foot]
    }


def _running_stats(accum: Dict[str, Dict[str, float]], arr: np.ndarray, key: str) -> None:
    """Accumulate sum / sum_sq / count per feature-dim."""
    arr64 = arr.astype(np.float64, copy=False)
    cur_sum = arr64.sum(axis=0)
    cur_sq = np.square(arr64).sum(axis=0)
    if key not in accum:
        accum[key] = {"sum": cur_sum, "sq": cur_sq, "n": int(arr.shape[0])}
    else:
        accum[key]["sum"] += cur_sum
        accum[key]["sq"] += cur_sq
        accum[key]["n"] += int(arr.shape[0])


def _finalize_stats(accum: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    means, stds = {}, {}
    for k, s in accum.items():
        n = max(1, int(s["n"]))
        mean = s["sum"] / n
        var = np.maximum(s["sq"] / n - np.square(mean), 0.0)
        std = np.sqrt(var + 1e-6).astype(np.float32)
        std = np.where(std < 1e-4, 1.0, std)  # guard degenerate dims
        means[k] = mean.astype(np.float32)
        stds[k] = std.astype(np.float32)
    return means, stds


def build_cache(
    manifest: Path,
    output: Path,
    block_size: int = 96,
    limit: Optional[int] = None,
    smooth_sigma: float = 15.0,
) -> None:
    with manifest.open("r", encoding="utf-8") as f:
        lines = [json.loads(ln) for ln in f if ln.strip()]
    if limit is not None:
        lines = lines[:limit]
    print(f"[kimodo_cache] processing {len(lines)} manifest entries, block_size={block_size}")

    # build skeleton info from the first BVH; assume all BVH share the same skeleton (BEAT does)
    root_dir = Path(".").resolve()
    first_bvh = (root_dir / lines[0]["bvh"]).resolve()
    info = build_skeleton_info(first_bvh)
    foot_idx = _find_foot_joint_indices(info)
    print(f"[kimodo_cache] skeleton: J={info.n_joints}, N_active={info.n_active}, foot_joints={foot_idx.tolist()}")

    segments: List[Dict[str, Any]] = []
    accum: Dict[str, Dict[str, float]] = {}
    total_frames = 0
    kept = 0

    for i, item in enumerate(lines):
        bvh_path = (root_dir / item["bvh"]).resolve()
        try:
            full_motion = load_full_motion_from_bvh(bvh_path)
        except Exception as e:
            print(f"[kimodo_cache] WARN {item.get('id', '?')}: load failed {e}")
            continue

        if full_motion.shape[0] < block_size:
            continue

        # take as many non-overlapping blocks as possible
        n_blocks = full_motion.shape[0] // block_size
        for b in range(n_blocks):
            clip = full_motion[b * block_size : (b + 1) * block_size]
            rep = compute_kimodo_representation(clip, info, smooth_sigma=smooth_sigma)
            seg = {
                "segment_id": f"{item.get('id', f'item{i}')}_blk{b}",
                "src_bvh": str(item["bvh"]),
                **rep,
                "text": str(item.get("text", "")),
            }
            segments.append(seg)
            # accumulate stats only for z-score features
            for k in ("r_p", "j_p", "j_v", "f"):
                _running_stats(accum, rep[k], k)
            total_frames += block_size
            kept += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(lines):
            print(f"[kimodo_cache] {i+1}/{len(lines)} manifest entries → {kept} segments, {total_frames} frames")

    if not segments:
        raise RuntimeError("no segments produced; check manifest & block_size")

    means, stds = _finalize_stats(accum)
    # unnormalized: r_a, j_a (they're on unit sphere)
    print(
        "[kimodo_cache] stats computed: "
        f"r_p std range [{stds['r_p'].min():.3f}, {stds['r_p'].max():.3f}], "
        f"j_p std range [{stds['j_p'].min():.3f}, {stds['j_p'].max():.3f}], "
        f"j_v std range [{stds['j_v'].min():.3f}, {stds['j_v'].max():.3f}]"
    )

    payload = {
        "segments": segments,
        "mean": means,
        "std": stds,
        "skeleton": {
            "parents": info.parents.astype(np.int64),
            "offsets": info.offsets.astype(np.float32),
            "active_indices": info.active_indices.astype(np.int64),
            "joint_names": info.joint_names,
            "foot_joint_indices": foot_idx.astype(np.int64),
        },
        "fps": TARGET_FPS,
        "block_size": int(block_size),
        "smooth_sigma": float(smooth_sigma),
        "foot_vel_threshold": float(FOOT_VEL_THRESHOLD),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(output))
    print(f"[kimodo_cache] saved {len(segments)} segments → {output}")


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--block_size", type=int, default=96)
    ap.add_argument("--limit", type=int, default=None, help="limit number of manifest entries (for small caches)")
    ap.add_argument("--smooth_sigma", type=float, default=15.0)
    args = ap.parse_args()
    build_cache(args.manifest, args.output, block_size=args.block_size, limit=args.limit, smooth_sigma=args.smooth_sigma)


if __name__ == "__main__":
    _cli()
