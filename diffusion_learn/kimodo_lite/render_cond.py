"""Render conditional Kimodo samples to BVH.

Generates one sample per training clip's condition; each should look like
that specific clip (if Phase 7 conditioning works).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from stageB.common.bvh_io import export_canonical_full_motion_to_bvh

from .diffusion import make_ddpm_schedule
from .feature_pack import FeatureLayout, denormalize, unpack
from .model_cond import CondConfig, ConditionalTwoStageDenoiser
from .model_concat import ConcatConfig, ConcatTwoStageDenoiser
from .render import kimodo_to_full_motion, smooth_kimodo_motion
from .train import skeleton_info_from_payload
from .train_cond import KimodoCondDataset, ddim_sample_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--ref_bvh", type=Path, default=Path("beat/beat_english_v0.2.1/14/14_zhang_0_96_96.bvh"))
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_cond", type=int, default=4)
    ap.add_argument("--cfg_w", type=float, default=0.0)
    ap.add_argument("--ddim_steps", type=int, default=100)
    ap.add_argument("--smooth_sigma", type=float, default=1.0,
                    help="smooth r_p/j_a before BVH export (0=off). Gaussian method only.")
    ap.add_argument("--smooth_method", choices=["admm", "gaussian", "none"], default="gaussian")
    ap.add_argument("--admm_pos", type=float, default=0.001)
    ap.add_argument("--admm_vel", type=float, default=1.0)
    ap.add_argument("--admm_acc", type=float, default=10.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--use_ema", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    ckpt = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    payload = torch.load(str(args.cache), map_location="cpu", weights_only=False)
    skel = skeleton_info_from_payload(payload, device)
    n_foot = int(payload["skeleton"]["foot_joint_indices"].shape[0])
    layout = FeatureLayout(n_joints=ckpt["layout"]["n_joints"], n_foot=ckpt["layout"]["n_foot"])
    overfit_n = int(ckpt["args"]["overfit_n"])
    ds = KimodoCondDataset(payload, layout, overfit_n=overfit_n)
    stats = ds.stats.to(device)

    arch = ckpt["args"].get("arch", "cross")
    if arch == "concat":
        cfg_dict = dict(ckpt["cfg"])
        cfg_dict.setdefault("support_anchor_mask", False)
        cfg = ConcatConfig(**cfg_dict)
        model = ConcatTwoStageDenoiser(layout, cfg).to(device)
    else:
        cfg = CondConfig(**ckpt["cfg"])
        model = ConditionalTwoStageDenoiser(layout, cfg).to(device)
    model.load_state_dict(ckpt["model_ema"] if args.use_ema else ckpt["model_raw"])
    model.eval()

    sched = make_ddpm_schedule(1000).to(device)
    T_frames = ds.samples[0]["x0_packed"].shape[0]
    shape = (1, T_frames, layout.total_dim)

    n_cond = min(args.n_cond, len(ds.samples))
    print(f"[render_cond] generating {n_cond} samples, one per clip's condition (cfg_w={args.cfg_w})")

    # Export GT clips for reference
    for i in range(n_cond):
        s = ds.samples[i]
        from .loss import KimodoMotion
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
        print(f"  gt wrote {out_path}")

    with torch.no_grad():
        for i in range(n_cond):
            torch.manual_seed(args.seed)
            s = ds.samples[i]
            audio = s["audio"].unsqueeze(0).to(device)
            text = s["text_emb"].unsqueeze(0).to(device)
            x0_samp = ddim_sample_cfg(model, shape, sched, audio, text,
                                       w_cfg=args.cfg_w, n_steps=args.ddim_steps, device=device)
            samp = denormalize(unpack(x0_samp, layout), stats)
            if args.smooth_method == "admm":
                samp = smooth_kimodo_motion(
                    samp, method="admm",
                    admm_weights=(args.admm_pos, args.admm_vel, args.admm_acc),
                )
                tag = f"admm_{args.admm_acc:g}"
            elif args.smooth_method == "gaussian" and args.smooth_sigma > 0:
                samp = smooth_kimodo_motion(samp, sigma=args.smooth_sigma, method="gaussian")
                tag = f"gauss_{args.smooth_sigma}"
            else:
                tag = "raw"
            full_motion = kimodo_to_full_motion(samp, skel)
            out_path = args.out_dir / f"cond_clip{i}_w{args.cfg_w}_{tag}.bvh"
            export_canonical_full_motion_to_bvh(full_motion, ref_bvh=args.ref_bvh, output_path=out_path)
            # quick MAE to this clip
            jp_mae = float((samp.j_p.cpu() - s["raw_j_p"].unsqueeze(0)).abs().mean())
            print(f"  cond_clip{i} jp_MAE={jp_mae:.3f}cm → {out_path}")

    print(f"\n[render_cond] all BVH in: {args.out_dir}")


if __name__ == "__main__":
    main()
