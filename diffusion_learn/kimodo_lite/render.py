"""Render Kimodo-lite samples and GT training clips back to BVH.

Inverse transform:
  1. Take j_a (global 6D, all J joints) → global rotmat → local rotmat via
     global_rotmat_to_local → local 6D → keep only active joints.
  2. Concatenate root_pos (r_p, smoothed) + local 6D per active joint into the
     canonical full_motion format ([T, 3 + N_active*6]) used by the existing
     BVH exporter.
  3. Call export_canonical_full_motion_to_bvh with a reference BVH header.

Outputs one BVH per sample and per GT clip under --out_dir.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from stageB.common.bvh_io import export_canonical_full_motion_to_bvh

from .diffusion import ddim_sample, make_ddpm_schedule
from .feature_pack import FeatureLayout, denormalize, unpack
from .fk import (
    SkeletonInfoT,
    global_rotmat_to_local,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from .loss import KimodoMotion
from .model import TwoStageConfig, TwoStageDenoiser
from .train import KimodoClipDataset, skeleton_info_from_payload


def _gaussian_kernel_1d(sigma: float, radius: int) -> torch.Tensor:
    k = torch.arange(-radius, radius + 1, dtype=torch.float32)
    w = torch.exp(-0.5 * (k / sigma) ** 2)
    return w / w.sum()


def _build_admm_system(T: int, pos_w: float, vel_w: float, acc_w: float, dtype, device) -> torch.Tensor:
    """Return Cholesky factor L of H = pos_w*I + vel_w*V'V + acc_w*A'A where
    V is (T-1,T) first-diff and A is (T-2,T) second-diff.
    """
    import numpy as np
    I = np.eye(T, dtype=np.float64)
    V = np.zeros((T - 1, T), dtype=np.float64)
    for i in range(T - 1):
        V[i, i] = -1.0
        V[i, i + 1] = 1.0
    A = np.zeros((T - 2, T), dtype=np.float64)
    for i in range(T - 2):
        A[i, i] = 1.0
        A[i, i + 1] = -2.0
        A[i, i + 2] = 1.0
    H = pos_w * I + vel_w * V.T @ V + acc_w * A.T @ A
    H_t = torch.from_numpy(H).to(device=device, dtype=dtype)
    L = torch.linalg.cholesky(H_t)
    return L


def _admm_smooth_time(x: torch.Tensor, pos_w: float, vel_w: float, acc_w: float) -> torch.Tensor:
    """Solve per-channel (pos_w I + vel_w V'V + acc_w A'A) x_sm = pos_w x0.
    x: [B, T, ...]  along axis=1 (time).
    """
    B, T = x.shape[0], x.shape[1]
    flat = x.reshape(B, T, -1)  # [B, T, C]
    L = _build_admm_system(T, pos_w, vel_w, acc_w, dtype=flat.dtype, device=flat.device)
    rhs = pos_w * flat  # [B, T, C]
    sm = torch.cholesky_solve(rhs, L)  # solves H x = rhs for each [B, C] column
    return sm.reshape(x.shape)


def smooth_kimodo_motion(
    motion: KimodoMotion,
    sigma: float = 0.0,
    method: str = "admm",
    admm_weights: tuple = (0.001, 1.0, 10.0),
) -> KimodoMotion:
    """Temporal smoothing of r_p (xz) and j_a (global 6D) before BVH export.

    method="admm": Kimodo-style ADMM 2nd-order — pos/vel/acc weighted.
                   Heavily penalizes acceleration; sigma is ignored.
    method="gaussian": simple Gaussian with given sigma (must be > 0).

    j_a smoothed element-wise on the 6D values; rotation_6d_to_matrix re-orthogonalizes.
    r_p.y kept unchanged; only xz smoothed.
    """
    if method == "gaussian":
        if sigma <= 0:
            return motion
        radius = max(1, int(round(3 * sigma)))
        kernel = _gaussian_kernel_1d(sigma, radius).to(motion.r_p.device)

        def _smooth_time(x: torch.Tensor) -> torch.Tensor:
            B, T = x.shape[0], x.shape[1]
            flat = x.reshape(B, T, -1).permute(0, 2, 1)
            C = flat.shape[1]
            k = kernel.view(1, 1, -1).expand(C, 1, -1)
            pad = torch.nn.functional.pad(flat, (radius, radius), mode="replicate")
            out = torch.nn.functional.conv1d(pad, k, groups=C)
            return out.permute(0, 2, 1).reshape(x.shape)
    elif method == "admm":
        pos_w, vel_w, acc_w = admm_weights

        def _smooth_time(x: torch.Tensor) -> torch.Tensor:
            return _admm_smooth_time(x, pos_w, vel_w, acc_w)
    else:
        raise ValueError(f"unknown smooth method: {method}")

    r_p_sm = motion.r_p.clone()
    r_p_sm_xz = _smooth_time(torch.stack([motion.r_p[..., 0], motion.r_p[..., 2]], dim=-1))
    r_p_sm[..., 0] = r_p_sm_xz[..., 0]
    r_p_sm[..., 2] = r_p_sm_xz[..., 1]
    j_a_sm = _smooth_time(motion.j_a)
    return KimodoMotion(
        r_p=r_p_sm, r_a=motion.r_a, j_p=motion.j_p, j_v=motion.j_v,
        j_a=j_a_sm, f=motion.f,
    )


def kimodo_to_full_motion(
    motion: KimodoMotion,
    skel: SkeletonInfoT,
) -> np.ndarray:
    """Convert a Kimodo-style motion to BVH canonical full_motion [T, 3+N_active*6].

    Uses r_p as root position (smoothed is fine visually) and converts j_a
    (global 6D, all J joints) back to local 6D per active joint.
    """
    # motion fields shapes: r_p [B,T,3], j_a [B,T,J*6]
    assert motion.r_p.dim() == 3 and motion.r_p.shape[0] == 1, motion.r_p.shape
    T = motion.r_p.shape[1]
    J = skel.n_joints
    device = motion.j_a.device

    j_a_global_6d = motion.j_a.view(1, T, J, 6)
    R_global = rotation_6d_to_matrix(j_a_global_6d)   # [1,T,J,3,3]
    R_local = global_rotmat_to_local(R_global, skel.parents)  # [1,T,J,3,3]
    local_6d_full = matrix_to_rotation_6d(R_local)    # [1,T,J,6]
    # pick only active joints
    active_idx = skel.active_indices.to(device)
    local_6d_active = local_6d_full[:, :, active_idx, :]  # [1,T,N_active,6]

    root_pos = motion.r_p.squeeze(0)  # [T, 3]
    rot = local_6d_active.squeeze(0).reshape(T, -1)  # [T, N_active*6]
    full_motion = torch.cat([root_pos.to(rot.device), rot], dim=-1)
    return full_motion.detach().cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--ref_bvh", type=Path, default=Path("beat/beat_english_v0.2.1/14/14_zhang_0_96_96.bvh"))
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--n_gt", type=int, default=4, help="how many GT clips to export (0 = skip GT)")
    ap.add_argument("--ddim_steps", type=int, default=100)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--use_ema", action="store_true", default=True)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[render] loading ckpt: {args.ckpt}")
    ckpt = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    payload = torch.load(str(args.cache), map_location="cpu", weights_only=False)
    device = args.device

    skel = skeleton_info_from_payload(payload, device)
    layout = FeatureLayout(n_joints=ckpt["layout"]["n_joints"], n_foot=ckpt["layout"]["n_foot"])
    overfit_n = int(ckpt["args"]["overfit_n"])
    ds = KimodoClipDataset(payload, layout, overfit_n=overfit_n)
    stats = ds.stats.to(device)

    cfg = TwoStageConfig(**ckpt["cfg"])
    model = TwoStageDenoiser(layout, cfg).to(device)
    model.load_state_dict(ckpt["model_ema"] if args.use_ema else ckpt["model_raw"])
    model.eval()

    sched = make_ddpm_schedule(args.T).to(device)

    T_frames = ds.samples[0]["x0_packed"].shape[0]

    # ---- export GT clips first ----
    n_gt = min(args.n_gt, len(ds.samples))
    print(f"[render] exporting {n_gt} GT clips (of {len(ds.samples)} total)")
    for i, s in enumerate(ds.samples[:n_gt]):
        gt_motion = KimodoMotion(
            r_p=s["raw_r_p"].unsqueeze(0).to(device),
            r_a=s["raw_r_a"].unsqueeze(0).to(device),
            j_p=s["raw_j_p"].unsqueeze(0).to(device),
            j_v=s["raw_j_v"].unsqueeze(0).to(device),
            j_a=s["raw_j_a"].unsqueeze(0).to(device),
            f=s["raw_f"].unsqueeze(0).to(device),
        )
        full_motion = kimodo_to_full_motion(gt_motion, skel)
        out_path = args.out_dir / f"gt_clip{i}.bvh"
        export_canonical_full_motion_to_bvh(full_motion, ref_bvh=args.ref_bvh, output_path=out_path)
        print(f"  wrote {out_path} ({full_motion.shape[0]} frames)")

    # ---- sample and export ----
    print(f"[render] generating {args.n_samples} DDIM samples (steps={args.ddim_steps})")

    def x0_fn(x, tt):
        return model(x, tt)

    for k in range(args.n_samples):
        torch.manual_seed(args.seed + k)
        with torch.no_grad():
            shape = (1, T_frames, layout.total_dim)
            x0_samp = ddim_sample(x0_fn, shape, sched, n_steps=args.ddim_steps, device=device)
            samp_norm = unpack(x0_samp, layout)
            samp = denormalize(samp_norm, stats)

        full_motion = kimodo_to_full_motion(samp, skel)
        out_path = args.out_dir / f"sample{k:02d}.bvh"
        export_canonical_full_motion_to_bvh(full_motion, ref_bvh=args.ref_bvh, output_path=out_path)
        # quick MAE to nearest GT
        best = None
        for i, s in enumerate(ds.samples):
            r_mae = float((samp.r_p.squeeze(0).cpu() - s["raw_r_p"]).abs().mean())
            jp_mae = float((samp.j_p.squeeze(0).cpu() - s["raw_j_p"]).abs().mean())
            score = r_mae + jp_mae
            if best is None or score < best[0]:
                best = (score, i, r_mae, jp_mae)
        print(f"  wrote {out_path}  nearest=gt_clip{best[1]} r_p_MAE={best[2]:.3f}cm j_p_MAE={best[3]:.3f}cm")

    print(f"\n[render] all BVH files in: {args.out_dir}")


if __name__ == "__main__":
    main()
