"""Post-training diagnostic for Kimodo-lite overfit.

Answers two questions:
1) Does the model memorize the training clips? — DDIM-sample N times, compute
   MAE to the nearest training clip (not just clip0). A well-overfitted model
   should give very small min-MAE for every sample.
2) Is the DDPM x0-prediction loss floor consistent with expectation? — Evaluate
   loss at fixed t values (t=10, t=500, t=990) on training data. Low t should
   be ~0; high t inherits data variance.

Usage:
    python -m diffusion_learn.kimodo_lite.diagnose \
        --ckpt diffusion_learn/runs/kimodo_overfit4/final.pt \
        --cache cache/kimodo_tiny.pt --n_samples 8
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from .diffusion import DDPMSchedule, ddim_sample, make_ddpm_schedule, q_sample
from .feature_pack import (
    FeatureLayout,
    build_norm_stats,
    denormalize,
    normalize,
    pack,
    unpack,
)
from .loss import KimodoLossWeights, KimodoMotion, kimodo_loss
from .model import TwoStageConfig, TwoStageDenoiser
from .train import KimodoClipDataset, skeleton_info_from_payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--use_ema", action="store_true", default=True)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--ddim_steps", type=int, default=100)
    ap.add_argument("--diag_n", type=int, default=32, help="max clips to use for fixed-t loss eval (avoids OOM)")
    args = ap.parse_args()

    device = args.device
    print(f"[diag] loading ckpt: {args.ckpt}")
    ckpt = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    print(f"[diag] ckpt cfg: {ckpt['cfg']}, layout: {ckpt['layout']}")

    payload = torch.load(str(args.cache), map_location="cpu", weights_only=False)
    skel = skeleton_info_from_payload(payload, device)
    n_foot = int(payload["skeleton"]["foot_joint_indices"].shape[0])
    layout = FeatureLayout(n_joints=ckpt["layout"]["n_joints"], n_foot=ckpt["layout"]["n_foot"])

    # load overfit dataset (same as training)
    overfit_n = int(ckpt["args"]["overfit_n"])
    ds = KimodoClipDataset(payload, layout, overfit_n=overfit_n)
    stats = ds.stats.to(device)

    cfg = TwoStageConfig(**ckpt["cfg"])
    model = TwoStageDenoiser(layout, cfg).to(device)
    sd = ckpt["model_ema"] if args.use_ema else ckpt["model_raw"]
    model.load_state_dict(sd)
    model.eval()

    sched = make_ddpm_schedule(args.T).to(device)

    # ---- (1) Fixed-t loss evaluation ----
    print("\n[diag] === Fixed-t loss (should be near 0 at low t) ===")
    weights = KimodoLossWeights()
    # limit to diag_n clips to avoid OOM on large datasets
    diag_samples = ds.samples[: args.diag_n]
    print(f"[diag] using {len(diag_samples)} clips for fixed-t eval")
    batch_x0 = torch.stack([s["x0_packed"] for s in diag_samples], dim=0).to(device)
    batch_raw = {
        k: torch.stack([s[f"raw_{k}"] for s in diag_samples], dim=0).to(device)
        for k in ("r_p", "r_a", "j_p", "j_v", "j_a", "f")
    }
    j_p_world = torch.stack([s["j_p_world"] for s in diag_samples], dim=0).to(device)
    r_p_world = torch.stack([s["r_p_world"] for s in diag_samples], dim=0).to(device)
    target_motion = KimodoMotion(**batch_raw)

    B = batch_x0.shape[0]
    with torch.no_grad():
        for t_val in [0, 10, 100, 500, 900, 990]:
            t = torch.full((B,), int(t_val), device=device, dtype=torch.long)
            x_t = q_sample(batch_x0, t, sched)
            x0_pred_norm = model(x_t, t)
            pred_motion_norm = unpack(x0_pred_norm, layout)
            pred_motion = denormalize(pred_motion_norm, stats)
            total, terms = kimodo_loss(
                pred_motion, target_motion,
                skel=skel, j_p_target_for_fk=j_p_world, root_pos_target_world=r_p_world,
                weights=weights,
            )
            print(
                f"  t={t_val:4d}  total={float(total):6.3f}  r_p={terms['root_pos']:5.3f}  "
                f"jp={terms['joint_pos']:5.3f}  ja={terms['joint_rot']:5.3f}  "
                f"fk={terms['fk']:5.3f}"
            )

    # ---- (2) DDIM sampling: match to nearest training clip ----
    print(f"\n[diag] === DDIM samples vs nearest training clip ({args.n_samples} samples) ===")

    def x0_fn(x, tt):
        return model(x, tt)

    # precompute unnormalized training clips (for comparison; limit to diag_n)
    raw_train_clips = []
    for s in diag_samples:
        raw_train_clips.append({
            "r_p": s["raw_r_p"].to(device),
            "j_p": s["raw_j_p"].to(device),
            "j_a": s["raw_j_a"].to(device),
        })

    with torch.no_grad():
        for k in range(args.n_samples):
            torch.manual_seed(1000 + k)
            T_frames = ds.samples[0]["x0_packed"].shape[0]
            shape = (1, T_frames, layout.total_dim)
            x0_samp = ddim_sample(x0_fn, shape, sched, n_steps=args.ddim_steps, device=device)
            samp_norm = unpack(x0_samp, layout)
            samp = denormalize(samp_norm, stats)

            # compute MAE to each clip on r_p / j_p / j_a
            clip_scores = []
            for ci, clip in enumerate(raw_train_clips):
                rp_mae = float((samp.r_p - clip["r_p"].unsqueeze(0)).abs().mean())
                jp_mae = float((samp.j_p - clip["j_p"].unsqueeze(0)).abs().mean())
                ja_mae = float((samp.j_a - clip["j_a"].unsqueeze(0)).abs().mean())
                clip_scores.append((ci, rp_mae, jp_mae, ja_mae, rp_mae + jp_mae + ja_mae))
            clip_scores.sort(key=lambda x: x[4])
            best = clip_scores[0]
            print(
                f"  sample {k}: nearest=clip{best[0]}  "
                f"rp={best[1]:.3f}cm  jp={best[2]:.3f}cm  ja={best[3]:.4f}  "
                f"(2nd-best clip{clip_scores[1][0]} sum={clip_scores[1][4]:.2f})"
            )


if __name__ == "__main__":
    main()
