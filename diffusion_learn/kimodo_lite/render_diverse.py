"""Render a diverse set of samples for visual inspection.

Two test grids:
  A) Same condition × N seeds  — did we collapse to single mode per cond?
  B) N widely-spaced conditions × single seed — do different conds make
     different motions?

Writes BVH + MP4 (σ=1 Gaussian motion smooth baked in) to out_dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from stageB.common.bvh_io import export_canonical_full_motion_to_bvh

from .diffusion import make_ddpm_schedule
from .feature_pack import FeatureLayout, denormalize, unpack
from .loss import KimodoMotion
from .model_concat import ConcatConfig, ConcatTwoStageDenoiser
from .model_cond import CondConfig, ConditionalTwoStageDenoiser
from .render import kimodo_to_full_motion, smooth_kimodo_motion
from .train import skeleton_info_from_payload
from .train_cond import KimodoCondDataset, ddim_sample_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--ref_bvh", type=Path, default=Path("beat/beat_english_v0.2.1/14/14_zhang_0_96_96.bvh"))
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--cond_indices", type=int, nargs="*", default=None,
                    help="indices in dataset for grid B (diverse conditions). Default: 8 evenly spaced.")
    ap.add_argument("--same_cond_idx", type=int, default=0, help="cond index for grid A (same cond × seeds)")
    ap.add_argument("--n_seeds", type=int, default=4)
    ap.add_argument("--cfg_w", type=float, default=2.0)
    ap.add_argument("--ddim_steps", type=int, default=100)
    ap.add_argument("--smooth_sigma", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--use_ema", action="store_true", default=True)
    args = ap.parse_args()

    device = args.device
    args.out_dir.mkdir(parents=True, exist_ok=True)
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
        cfg = ConcatConfig(**ckpt["cfg"])
        model = ConcatTwoStageDenoiser(layout, cfg).to(device)
    else:
        cfg = CondConfig(**ckpt["cfg"])
        model = ConditionalTwoStageDenoiser(layout, cfg).to(device)
    model.load_state_dict(ckpt["model_ema"] if args.use_ema else ckpt["model_raw"])
    model.eval()

    sched = make_ddpm_schedule(1000).to(device)
    T_frames = ds.samples[0]["x0_packed"].shape[0]
    shape = (1, T_frames, layout.total_dim)

    N = len(ds.samples)
    if args.cond_indices is None:
        args.cond_indices = [int(i * N / 8) for i in range(8)]
    print(f"[render_diverse] arch={arch} | N_total={N} | grid B indices={args.cond_indices}")

    def _export_gt(i: int, out_path: Path):
        s = ds.samples[i]
        gt = KimodoMotion(
            r_p=s["raw_r_p"].unsqueeze(0).to(device),
            r_a=s["raw_r_a"].unsqueeze(0).to(device),
            j_p=s["raw_j_p"].unsqueeze(0).to(device),
            j_v=s["raw_j_v"].unsqueeze(0).to(device),
            j_a=s["raw_j_a"].unsqueeze(0).to(device),
            f=s["raw_f"].unsqueeze(0).to(device),
        )
        full = kimodo_to_full_motion(gt, skel)
        export_canonical_full_motion_to_bvh(full, ref_bvh=args.ref_bvh, output_path=out_path)

    def _sample_and_export(i: int, seed: int, out_path: Path):
        s = ds.samples[i]
        audio = s["audio"].unsqueeze(0).to(device)
        text = s["text_emb"].unsqueeze(0).to(device)
        torch.manual_seed(seed)
        with torch.no_grad():
            x0 = ddim_sample_cfg(model, shape, sched, audio, text,
                                 w_cfg=args.cfg_w, n_steps=args.ddim_steps, device=device)
            samp = denormalize(unpack(x0, layout), stats)
            if args.smooth_sigma > 0:
                samp = smooth_kimodo_motion(samp, sigma=args.smooth_sigma, method="gaussian")
            full = kimodo_to_full_motion(samp, skel)
        export_canonical_full_motion_to_bvh(full, ref_bvh=args.ref_bvh, output_path=out_path)
        jp_mae = float((samp.j_p.cpu() - s["raw_j_p"].unsqueeze(0)).abs().mean())
        return jp_mae

    # Grid A: same cond × N seeds
    print(f"\n=== Grid A: cond=clip{args.same_cond_idx} × {args.n_seeds} seeds ===")
    a_dir = args.out_dir / "A_same_cond_seeds"
    a_dir.mkdir(exist_ok=True)
    _export_gt(args.same_cond_idx, a_dir / f"gt_clip{args.same_cond_idx}.bvh")
    for s in range(args.n_seeds):
        mae = _sample_and_export(args.same_cond_idx, 100 + s, a_dir / f"seed{s}.bvh")
        print(f"  seed{s} jp_MAE={mae:.2f}cm")

    # Grid B: diverse conditions × single seed
    print(f"\n=== Grid B: {len(args.cond_indices)} diverse conditions ===")
    b_dir = args.out_dir / "B_diverse_conds"
    b_dir.mkdir(exist_ok=True)
    for i in args.cond_indices:
        _export_gt(i, b_dir / f"gt_clip{i}.bvh")
        mae = _sample_and_export(i, 42, b_dir / f"cond_clip{i}.bvh")
        print(f"  cond={i} jp_MAE={mae:.2f}cm")

    print(f"\n[render_diverse] BVH written to {args.out_dir}")


if __name__ == "__main__":
    main()
