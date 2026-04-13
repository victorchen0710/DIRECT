"""Motion inpainting / continuation demo.

Given N anchor frames + audio + text, generate the remaining (T-N) frames.

Three tests:
  1. Continuation: anchor = first 16 frames of clip_i → continue 80 frames
  2. Different anchors, same audio/text: pick anchors from clip_j (same audio/text
     from clip_i) → shows how anchors shape the output beyond audio/text alone
  3. Inbetween: anchor = first 8 + last 8 frames → fill middle 80
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
from .train_cond import KimodoCondDataset, ddim_sample_cfg_inpaint, ddim_sample_cfg_inpaint_native


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--ref_bvh", type=Path, default=Path("beat/beat_english_v0.2.1/14/14_zhang_0_96_96.bvh"))
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--clip_audio_text", type=int, default=0, help="audio+text come from this clip")
    ap.add_argument("--anchor_clips", type=int, nargs="*", default=[0, 1, 2, 3],
                    help="which clips' first-N-frames to use as anchors (all with the same audio/text)")
    ap.add_argument("--n_anchor", type=int, default=16, help="# initial frames to use as anchor")
    ap.add_argument("--mode", choices=["prefix", "inbetween"], default="prefix",
                    help="prefix = anchor first N frames; inbetween = anchor first N/2 + last N/2")
    ap.add_argument("--cfg_w", type=float, default=2.0)
    ap.add_argument("--ddim_steps", type=int, default=100)
    ap.add_argument("--smooth_sigma", type=float, default=1.0)
    ap.add_argument("--fade", type=int, default=4, help="soft-mask fade frames at anchor boundary (RePaint mode only)")
    ap.add_argument("--anchor_steps_frac", type=float, default=0.7,
                    help="only anchor in first fraction of DDIM steps (RePaint mode only)")
    ap.add_argument("--inpaint_mode", choices=["repaint", "native"], default="native",
                    help="native = use anchor_mask channel (requires mask-trained model); repaint = RePaint-style replacement (works with any model)")
    ap.add_argument("--resample_jumps", type=int, default=0,
                    help="RePaint-style resampling: N back-and-forth jumps during native inpainting (0=off)")
    ap.add_argument("--resample_n", type=int, default=3,
                    help="# re-denoise iterations per jump")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--use_ema", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
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

    # Build mask
    mask = torch.zeros(T_frames, dtype=torch.bool)
    if args.mode == "prefix":
        mask[: args.n_anchor] = True
    else:  # inbetween
        h = args.n_anchor // 2
        mask[:h] = True
        mask[-h:] = True
    n_known = int(mask.sum())
    print(f"[inpaint] mode={args.mode}, n_known={n_known}/{T_frames}, arch={arch}")
    print(f"[inpaint] audio+text from clip{args.clip_audio_text}, anchors from clips {args.anchor_clips}")

    # Audio + text from the chosen clip
    s_at = ds.samples[args.clip_audio_text]
    audio = s_at["audio"].unsqueeze(0).to(device)
    text = s_at["text_emb"].unsqueeze(0).to(device)

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

    # GT reference
    _export_gt(args.clip_audio_text, args.out_dir / f"gt_audio_text_clip{args.clip_audio_text}.bvh")

    shape = (1, T_frames, layout.total_dim)
    for anchor_i in args.anchor_clips:
        s_anchor = ds.samples[anchor_i]
        x0_anchor = s_anchor["x0_packed"].unsqueeze(0).to(device)
        _export_gt(anchor_i, args.out_dir / f"gt_anchor_clip{anchor_i}.bvh")
        torch.manual_seed(args.seed)
        with torch.no_grad():
            if args.inpaint_mode == "native":
                x0_samp = ddim_sample_cfg_inpaint_native(
                    model, shape, sched, audio, text,
                    w_cfg=args.cfg_w, n_steps=args.ddim_steps, device=device,
                    x0_known=x0_anchor, known_mask=mask,
                    resample_jumps=args.resample_jumps, resample_n=args.resample_n,
                )
            else:
                x0_samp = ddim_sample_cfg_inpaint(
                    model, shape, sched, audio, text,
                    w_cfg=args.cfg_w, n_steps=args.ddim_steps, device=device,
                    x0_known=x0_anchor, known_mask=mask,
                    fade=args.fade, anchor_steps_frac=args.anchor_steps_frac,
                )
            samp = denormalize(unpack(x0_samp, layout), stats)
            if args.smooth_sigma > 0:
                samp = smooth_kimodo_motion(samp, sigma=args.smooth_sigma, method="gaussian")
            full = kimodo_to_full_motion(samp, skel)
        out_path = args.out_dir / f"inpaint_audio{args.clip_audio_text}_anchor{anchor_i}.bvh"
        export_canonical_full_motion_to_bvh(full, ref_bvh=args.ref_bvh, output_path=out_path)
        jp_mae_to_anchor = float((samp.j_p.cpu() - s_anchor["raw_j_p"].unsqueeze(0)).abs().mean())
        jp_mae_to_at = float((samp.j_p.cpu() - s_at["raw_j_p"].unsqueeze(0)).abs().mean())
        print(f"  anchor={anchor_i}: MAE vs anchor={jp_mae_to_anchor:.2f}cm, vs audio-text={jp_mae_to_at:.2f}cm → {out_path.name}")

    print(f"\n[inpaint] BVH in {args.out_dir}")


if __name__ == "__main__":
    main()
