"""Conditional diagnostic for kimodo_cond.

Key questions:
 1) Does the model reconstruct each training clip when given its own audio+text
    at t=0? (tests conditional mapping, no CFG)
 2) Does the model produce *different* outputs when given different conditions?
    (the core sanity check for conditioning — if swapping conditions makes no
    difference, the conditioning is broken.)
 3) What CFG weight gives best quality? Scan w ∈ {0, 1, 2, 3}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .diffusion import make_ddpm_schedule, q_sample
from .feature_pack import FeatureLayout, denormalize, unpack
from .loss import KimodoLossWeights, KimodoMotion, kimodo_loss
from .model_cond import CondConfig, ConditionalTwoStageDenoiser
from .model_concat import ConcatConfig, ConcatTwoStageDenoiser
from .train import skeleton_info_from_payload
from .train_cond import KimodoCondDataset, ddim_sample_cfg


def _build_model_from_ckpt(ckpt, layout, device):
    arch = ckpt["args"].get("arch", "cross")
    if arch == "concat":
        cfg_dict = dict(ckpt["cfg"])
        # backward compat: old ckpts lack this key and were trained without mask
        cfg_dict.setdefault("support_anchor_mask", False)
        cfg = ConcatConfig(**cfg_dict)
        return ConcatTwoStageDenoiser(layout, cfg).to(device)
    cfg = CondConfig(**ckpt["cfg"])
    return ConditionalTwoStageDenoiser(layout, cfg).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--use_ema", action="store_true", default=True)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--ddim_steps", type=int, default=100)
    ap.add_argument("--diag_n", type=int, default=4)
    args = ap.parse_args()

    device = args.device
    ckpt = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    payload = torch.load(str(args.cache), map_location="cpu", weights_only=False)
    skel = skeleton_info_from_payload(payload, device)
    n_foot = int(payload["skeleton"]["foot_joint_indices"].shape[0])
    layout = FeatureLayout(n_joints=ckpt["layout"]["n_joints"], n_foot=ckpt["layout"]["n_foot"])
    overfit_n = int(ckpt["args"]["overfit_n"])
    ds = KimodoCondDataset(payload, layout, overfit_n=overfit_n)
    stats = ds.stats.to(device)

    model = _build_model_from_ckpt(ckpt, layout, device)
    sd = ckpt["model_ema"] if args.use_ema else ckpt["model_raw"]
    model.load_state_dict(sd)
    model.eval()

    sched = make_ddpm_schedule(args.T).to(device)
    weights = KimodoLossWeights()

    samples = ds.samples[: args.diag_n]
    N = len(samples)
    print(f"[cond_diag] using {N} clips")

    # ---- (1) Fixed-t loss with true conditions, no dropout ----
    print("\n=== Fixed-t conditional loss (no CFG, no drop) ===")
    batch_x0 = torch.stack([s["x0_packed"] for s in samples]).to(device)
    batch_audio = torch.stack([s["audio"] for s in samples]).to(device)
    batch_text = torch.stack([s["text_emb"] for s in samples]).to(device)
    batch_raw = {k: torch.stack([s[f"raw_{k}"] for s in samples]).to(device)
                 for k in ("r_p", "r_a", "j_p", "j_v", "j_a", "f")}
    j_p_world = torch.stack([s["j_p_world"] for s in samples]).to(device)
    r_p_world = torch.stack([s["r_p_world"] for s in samples]).to(device)
    target = KimodoMotion(**batch_raw)
    B = batch_x0.shape[0]

    with torch.no_grad():
        for t_val in [0, 10, 100, 500, 900, 990]:
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            x_t = q_sample(batch_x0, t, sched)
            x0_pred = model(x_t, t, audio=batch_audio, text=batch_text, drop_mask=torch.zeros(B, device=device))
            pred = denormalize(unpack(x0_pred, layout), stats)
            total, terms = kimodo_loss(pred, target, skel=skel,
                                        j_p_target_for_fk=j_p_world,
                                        root_pos_target_world=r_p_world,
                                        weights=weights)
            print(f"  t={t_val:4d} total={float(total):6.3f} "
                  f"r_p={terms['root_pos']:5.3f} jp={terms['joint_pos']:5.3f} "
                  f"ja={terms['joint_rot']:5.3f} fk={terms['fk']:5.3f}")

    # ---- (2) Condition swap: sample clip0's motion using clip_i's condition ----
    print("\n=== Condition-swap test (does swapping cond change output?) ===")
    print("Sample using clip_i's cond, compare MAE to each training clip.")
    T_frames = samples[0]["x0_packed"].shape[0]
    shape = (1, T_frames, layout.total_dim)

    with torch.no_grad():
        for src_i in range(N):
            torch.manual_seed(42)
            audio_i = samples[src_i]["audio"].unsqueeze(0).to(device)
            text_i = samples[src_i]["text_emb"].unsqueeze(0).to(device)
            x0_samp = ddim_sample_cfg(model, shape, sched, audio_i, text_i,
                                       w_cfg=0.0, n_steps=args.ddim_steps, device=device)
            samp = denormalize(unpack(x0_samp, layout), stats)
            maes = []
            for j in range(N):
                jp_mae = float((samp.j_p - target.j_p[j:j+1]).abs().mean())
                maes.append(jp_mae)
            # format: cond=i → MAE to clip0, clip1, clip2, clip3
            best = int(min(range(N), key=lambda k: maes[k]))
            mark = "✓" if best == src_i else "✗"
            maes_str = "  ".join(f"[{k}]={maes[k]:5.2f}" for k in range(N))
            print(f"  cond=clip{src_i} {mark} → j_p_MAE: {maes_str}  (best match=clip{best})")

    # ---- (3) CFG weight scan for clip0 ----
    print("\n=== CFG weight scan (using clip0 cond) ===")
    audio0 = samples[0]["audio"].unsqueeze(0).to(device)
    text0 = samples[0]["text_emb"].unsqueeze(0).to(device)
    with torch.no_grad():
        for w in [0.0, 0.5, 1.0, 2.0, 3.0]:
            torch.manual_seed(42)
            x0_samp = ddim_sample_cfg(model, shape, sched, audio0, text0,
                                       w_cfg=w, n_steps=args.ddim_steps, device=device)
            samp = denormalize(unpack(x0_samp, layout), stats)
            r_p_mae = float((samp.r_p - target.r_p[0:1]).abs().mean())
            j_p_mae = float((samp.j_p - target.j_p[0:1]).abs().mean())
            j_a_mae = float((samp.j_a - target.j_a[0:1]).abs().mean())
            print(f"  w={w:.1f}  r_p_MAE={r_p_mae:.3f}  j_p_MAE={j_p_mae:.3f}  j_a_MAE={j_a_mae:.4f}")


if __name__ == "__main__":
    main()
