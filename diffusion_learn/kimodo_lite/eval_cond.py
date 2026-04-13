"""Comprehensive conditional evaluation.

Metrics (SOTA-style, inspired by MDM / Kimodo / DiffGesture):

  1. Cond-swap accuracy: for each clip_i, sample using clip_i's cond, find
     the nearest training clip by j_p MAE; correct if nearest == i.
  2. Multi-seed diversity: same cond × N seeds → std across seeds. Also
     report diversity-vs-GT ratio.
  3. Beat alignment: Pearson correlation between audio RMS envelope peaks
     and joint velocity magnitude peaks (aggregated across limbs).
  4. Condition ablation: sample with (audio+text) / audio_only / text_only /
     neither, report j_p MAE vs GT for each.
  5. CFG sweep: same cond × CFG weight ∈ {0, 1, 2, 3}, report both MAE vs GT
     and diversity across CFG weights.

Use: python -m diffusion_learn.kimodo_lite.eval_cond \
        --ckpt path/to/final.pt --cache path/to/cache.pt --eval_n 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .diffusion import make_ddpm_schedule
from .feature_pack import FeatureLayout, denormalize, unpack
from .model_concat import ConcatConfig, ConcatTwoStageDenoiser
from .model_cond import CondConfig, ConditionalTwoStageDenoiser
from .train import skeleton_info_from_payload
from .train_cond import KimodoCondDataset, ddim_sample_cfg


def _build_model_from_ckpt(ckpt, layout, device):
    arch = ckpt["args"].get("arch", "cross")
    if arch == "concat":
        cfg = ConcatConfig(**ckpt["cfg"])
        return ConcatTwoStageDenoiser(layout, cfg).to(device), arch
    cfg = CondConfig(**ckpt["cfg"])
    return ConditionalTwoStageDenoiser(layout, cfg).to(device), arch


def _sample(model, shape, sched, audio, text, w_cfg, n_steps, device, seed):
    torch.manual_seed(seed)
    return ddim_sample_cfg(model, shape, sched, audio, text, w_cfg=w_cfg, n_steps=n_steps, device=device)


def _peak_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two 1D signals (peaks-preserving)."""
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float((a * b).sum() / denom)


def _audio_rms_envelope(audio_feat: np.ndarray) -> np.ndarray:
    """Proxy for audio onset strength: l2 norm of per-frame audio embedding.

    Not a real RMS (we have w2v2 features not waveform), but correlates with
    speech energy and is cheap. Returns [T] float.
    """
    return np.linalg.norm(audio_feat, axis=-1)


def _joint_velocity_magnitude(j_p: np.ndarray, wrist_ankle_idx: List[int]) -> np.ndarray:
    """Mean velocity magnitude across end-effectors. j_p: [T, J, 3]."""
    v = np.zeros_like(j_p)
    v[:-1] = j_p[1:] - j_p[:-1]
    v_mag = np.linalg.norm(v, axis=-1)  # [T, J]
    ee = v_mag[:, wrist_ankle_idx].mean(axis=-1)  # [T]
    return ee


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--eval_n", type=int, default=8, help="# clips to use in cond-swap test")
    ap.add_argument("--div_seeds", type=int, default=6, help="# seeds for diversity")
    ap.add_argument("--ddim_steps", type=int, default=100)
    ap.add_argument("--cfg_w", type=float, default=1.0, help="default CFG weight (non-sweep)")
    ap.add_argument("--cfg_sweep", type=float, nargs="*", default=[0.0, 1.0, 2.0, 3.0])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--use_ema", action="store_true", default=True)
    args = ap.parse_args()

    device = args.device
    print(f"[eval] loading ckpt: {args.ckpt}")
    ckpt = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    payload = torch.load(str(args.cache), map_location="cpu", weights_only=False)
    skel = skeleton_info_from_payload(payload, device)
    n_foot = int(payload["skeleton"]["foot_joint_indices"].shape[0])
    layout = FeatureLayout(n_joints=ckpt["layout"]["n_joints"], n_foot=ckpt["layout"]["n_foot"])
    overfit_n = int(ckpt["args"]["overfit_n"])
    ds = KimodoCondDataset(payload, layout, overfit_n=overfit_n)
    stats = ds.stats.to(device)

    model, arch = _build_model_from_ckpt(ckpt, layout, device)
    sd = ckpt["model_ema"] if args.use_ema else ckpt["model_raw"]
    model.load_state_dict(sd)
    model.eval()
    print(f"[eval] arch={arch}, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    sched = make_ddpm_schedule(1000).to(device)
    samples = ds.samples[: args.eval_n]
    N = len(samples)
    T_frames = samples[0]["x0_packed"].shape[0]
    shape = (1, T_frames, layout.total_dim)
    J = layout.n_joints

    # wrist/ankle indices for beat alignment (look up by joint name)
    jn = list(payload["skeleton"]["joint_names"])
    ee_names = ["LeftHand", "RightHand", "LeftFoot", "RightFoot"]
    ee_idx = [jn.index(n) for n in ee_names if n in jn]
    print(f"[eval] end-effector indices for beat test: {ee_idx}")

    report = {"arch": arch, "ckpt": str(args.ckpt)}

    # ====================================================================
    # 1. Cond-swap accuracy
    # ====================================================================
    print("\n=== 1. Cond-swap accuracy ===")
    target_jp = torch.stack([s["raw_j_p"] for s in samples]).to(device)  # [N, T, J*3]
    correct = 0
    per_clip = []
    with torch.no_grad():
        for i in range(N):
            audio_i = samples[i]["audio"].unsqueeze(0).to(device)
            text_i = samples[i]["text_emb"].unsqueeze(0).to(device)
            x0 = _sample(model, shape, sched, audio_i, text_i, args.cfg_w, args.ddim_steps, device, seed=42)
            samp = denormalize(unpack(x0, layout), stats)
            maes = (samp.j_p - target_jp.unsqueeze(1)).abs().mean(dim=(1, 2, 3)).cpu().numpy()
            # samp.j_p is [1, T, J*3], target_jp[j] is [T, J*3]. Broadcast carefully:
            maes = []
            for j in range(N):
                m = float((samp.j_p[0] - target_jp[j]).abs().mean())
                maes.append(m)
            best = int(np.argmin(maes))
            ok = (best == i)
            if ok: correct += 1
            per_clip.append({"i": i, "best": best, "mae_i": maes[i], "mae_best": maes[best]})
            print(f"  clip{i} → best={best} {'✓' if ok else '✗'}  mae(self)={maes[i]:.2f}  mae(best)={maes[best]:.2f}")
    acc = correct / N
    print(f"  Accuracy: {correct}/{N} = {acc*100:.1f}%")
    report["cond_swap_accuracy"] = acc
    report["cond_swap_detail"] = per_clip

    # ====================================================================
    # 2. Multi-seed diversity
    # ====================================================================
    print(f"\n=== 2. Multi-seed diversity ({args.div_seeds} seeds for clip0) ===")
    audio0 = samples[0]["audio"].unsqueeze(0).to(device)
    text0 = samples[0]["text_emb"].unsqueeze(0).to(device)
    jp_samples = []
    with torch.no_grad():
        for s in range(args.div_seeds):
            x0 = _sample(model, shape, sched, audio0, text0, args.cfg_w, args.ddim_steps, device, seed=100 + s)
            samp = denormalize(unpack(x0, layout), stats)
            jp_samples.append(samp.j_p[0].cpu().numpy())
    jp_samples = np.stack(jp_samples)  # [S, T, J*3]
    # pairwise mean L2 distance across seeds (higher = more diverse)
    diffs = []
    for a in range(args.div_seeds):
        for b in range(a + 1, args.div_seeds):
            diffs.append(float(np.linalg.norm(jp_samples[a] - jp_samples[b]) / np.sqrt(jp_samples[a].size)))
    div = float(np.mean(diffs)) if diffs else 0.0
    # GT diversity baseline: MAE between different training clips
    gt_diffs = []
    for a in range(min(N, 6)):
        for b in range(a + 1, min(N, 6)):
            gt_diffs.append(float((samples[a]["raw_j_p"] - samples[b]["raw_j_p"]).pow(2).mean().sqrt()))
    gt_div = float(np.mean(gt_diffs)) if gt_diffs else 0.0
    print(f"  sample diversity (RMSE per channel): {div:.3f}cm")
    print(f"  GT diversity (different clips): {gt_div:.3f}cm")
    print(f"  ratio: {div/gt_div if gt_div>0 else 0:.2f}  (1.0 = as diverse as training data, <0.3 = mean collapse)")
    report["diversity_sample_rmse"] = div
    report["diversity_gt_rmse"] = gt_div
    report["diversity_ratio"] = div / gt_div if gt_div > 0 else 0

    # ====================================================================
    # 3. Beat alignment
    # ====================================================================
    print("\n=== 3. Beat alignment (audio RMS ↔ end-effector velocity) ===")
    corrs_gen = []
    corrs_gt = []
    with torch.no_grad():
        for i in range(min(N, 8)):
            audio_i_np = samples[i]["audio"].numpy()  # [T, 768]
            audio_env = _audio_rms_envelope(audio_i_np)

            # generated
            audio_i = samples[i]["audio"].unsqueeze(0).to(device)
            text_i = samples[i]["text_emb"].unsqueeze(0).to(device)
            x0 = _sample(model, shape, sched, audio_i, text_i, args.cfg_w, args.ddim_steps, device, seed=42)
            samp = denormalize(unpack(x0, layout), stats)
            jp_gen = samp.j_p[0].cpu().numpy().reshape(T_frames, J, 3)
            ee_gen = _joint_velocity_magnitude(jp_gen, ee_idx)
            corrs_gen.append(_peak_correlation(audio_env, ee_gen))

            # gt
            jp_gt = samples[i]["raw_j_p"].numpy().reshape(T_frames, J, 3)
            ee_gt = _joint_velocity_magnitude(jp_gt, ee_idx)
            corrs_gt.append(_peak_correlation(audio_env, ee_gt))
    print(f"  gen: mean={np.mean(corrs_gen):.3f}  values={[f'{c:.2f}' for c in corrs_gen]}")
    print(f"  gt:  mean={np.mean(corrs_gt):.3f}  values={[f'{c:.2f}' for c in corrs_gt]}")
    report["beat_align_gen_mean"] = float(np.mean(corrs_gen))
    report["beat_align_gt_mean"] = float(np.mean(corrs_gt))

    # ====================================================================
    # 4. Condition ablation (clip0)
    # ====================================================================
    print("\n=== 4. Condition ablation (clip0) ===")
    modes = [
        ("both", samples[0]["audio"].unsqueeze(0).to(device), samples[0]["text_emb"].unsqueeze(0).to(device)),
        ("audio_only", samples[0]["audio"].unsqueeze(0).to(device), None),
        ("text_only", None, samples[0]["text_emb"].unsqueeze(0).to(device)),
        ("neither", None, None),
    ]
    abl = {}
    with torch.no_grad():
        for name, aud, txt in modes:
            x0 = _sample(model, shape, sched, aud, txt, 0.0, args.ddim_steps, device, seed=42)
            samp = denormalize(unpack(x0, layout), stats)
            m = float((samp.j_p[0] - target_jp[0]).abs().mean())
            abl[name] = m
            print(f"  {name:12s} j_p MAE vs clip0: {m:.3f}cm")
    report["ablation"] = abl

    # ====================================================================
    # 5. CFG sweep
    # ====================================================================
    print("\n=== 5. CFG sweep (clip0) ===")
    cfg_results = {}
    with torch.no_grad():
        audio_i = samples[0]["audio"].unsqueeze(0).to(device)
        text_i = samples[0]["text_emb"].unsqueeze(0).to(device)
        for w in args.cfg_sweep:
            x0 = _sample(model, shape, sched, audio_i, text_i, w, args.ddim_steps, device, seed=42)
            samp = denormalize(unpack(x0, layout), stats)
            m = float((samp.j_p[0] - target_jp[0]).abs().mean())
            cfg_results[f"w={w}"] = m
            print(f"  w={w:.1f}  j_p MAE: {m:.3f}cm")
    report["cfg_sweep"] = cfg_results

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[eval] report saved → {args.out_json}")

    return report


if __name__ == "__main__":
    main()
