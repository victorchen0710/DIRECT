"""Kimodo-lite training script.

Phase 5 sanity gate: 4-clip unconditional overfit.

Usage:
    python -m diffusion_learn.kimodo_lite.train \
        --cache cache/kimodo_tiny.pt --overfit_n 4 --steps 20000 \
        --save_dir diffusion_learn/runs/kimodo_overfit4

Logs loss per term every --log_every steps. Saves EMA checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .diffusion import DDPMSchedule, ddim_sample, make_ddpm_schedule, q_sample
from .feature_pack import (
    FeatureLayout,
    NormStats,
    build_norm_stats,
    denormalize,
    normalize,
    pack,
    unpack,
)
from .fk import SkeletonInfo, SkeletonInfoT
from .loss import KimodoLossWeights, KimodoMotion, kimodo_loss
from .model import TwoStageConfig, TwoStageDenoiser


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class KimodoClipDataset(Dataset):
    """In-memory dataset of packed, normalized motion clips.

    Each sample is a dict with:
      - x0_packed: [T, total_dim] (normalized + packed)
      - j_p_world: [T, J, 3]  (target global joint positions in world frame for FK loss)
      - r_p_world: [T, 3]     (target smoothed root in world frame)
      - raw: the full KimodoMotion unnormalized (for debug / sanity)
    """

    def __init__(self, payload: dict, layout: FeatureLayout, overfit_n: Optional[int] = None):
        self.layout = layout
        segs = payload["segments"]
        if overfit_n is not None:
            segs = segs[:overfit_n]
        self.segs = segs
        self.stats = build_norm_stats(payload)

        # Preconvert to tensors
        self.samples = []
        for seg in segs:
            r_p = torch.from_numpy(np.asarray(seg["r_p"])).float()
            r_a = torch.from_numpy(np.asarray(seg["r_a"])).float()
            j_p = torch.from_numpy(np.asarray(seg["j_p"])).float()
            j_v = torch.from_numpy(np.asarray(seg["j_v"])).float()
            j_a = torch.from_numpy(np.asarray(seg["j_a"])).float()
            f = torch.from_numpy(np.asarray(seg["f"])).float()

            T = r_p.shape[0]
            # j_p in cache is xz-relative to smoothed root; reconstruct world-frame j_p for FK target
            J = layout.n_joints
            j_p_rel = j_p.view(T, J, 3)
            j_p_world = j_p_rel.clone()
            j_p_world[..., 0] += r_p[:, 0:1]
            j_p_world[..., 2] += r_p[:, 2:3]

            motion = KimodoMotion(r_p=r_p.unsqueeze(0), r_a=r_a.unsqueeze(0),
                                  j_p=j_p.unsqueeze(0), j_v=j_v.unsqueeze(0),
                                  j_a=j_a.unsqueeze(0), f=f.unsqueeze(0))
            motion_norm = normalize(motion, self.stats)
            x0_packed = pack(motion_norm, layout).squeeze(0)  # [T, total_dim]

            self.samples.append({
                "x0_packed": x0_packed,
                "j_p_world": j_p_world,
                "r_p_world": r_p,
                "raw_r_p": r_p, "raw_r_a": r_a, "raw_j_p": j_p,
                "raw_j_v": j_v, "raw_j_a": j_a, "raw_f": f,
                "segment_id": seg["segment_id"],
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate(batch):
    keys = ["x0_packed", "j_p_world", "r_p_world", "raw_r_p", "raw_r_a", "raw_j_p", "raw_j_v", "raw_j_a", "raw_f"]
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    out["segment_id"] = [b["segment_id"] for b in batch]
    return out


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])


# ---------------------------------------------------------------------------
# Build skeleton tensor from payload
# ---------------------------------------------------------------------------


def skeleton_info_from_payload(payload: dict, device: str = "cpu") -> SkeletonInfoT:
    sk = payload["skeleton"]
    return SkeletonInfoT(
        parents=torch.from_numpy(np.asarray(sk["parents"])).long().to(device),
        offsets=torch.from_numpy(np.asarray(sk["offsets"])).float().to(device),
        active_indices=torch.from_numpy(np.asarray(sk["active_indices"])).long().to(device),
        joint_names=list(sk["joint_names"]),
    )


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def run_training(args):
    t_start = time.time()
    device = args.device
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] loading cache: {args.cache}")
    payload = torch.load(str(args.cache), map_location="cpu", weights_only=False)
    skel_cpu = skeleton_info_from_payload(payload, "cpu")
    skel = skeleton_info_from_payload(payload, device)
    n_foot = int(payload["skeleton"]["foot_joint_indices"].shape[0])
    layout = FeatureLayout(n_joints=skel.n_joints, n_foot=n_foot)
    print(f"[train] layout: total_dim={layout.total_dim}, root_dim={layout.root_dim}, body_dim={layout.body_dim}")

    ds = KimodoClipDataset(payload, layout, overfit_n=args.overfit_n)
    print(f"[train] dataset: {len(ds)} clips, T={ds.samples[0]['x0_packed'].shape[0]}")

    # stats to device for de-normalization in loss
    stats = ds.stats.to(device)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate, drop_last=False)

    cfg = TwoStageConfig(
        root_d=args.root_d, root_layers=args.root_layers, root_heads=args.root_heads,
        body_d=args.body_d, body_layers=args.body_layers, body_heads=args.body_heads,
    )
    model = TwoStageDenoiser(layout, cfg).to(device)
    rp, bp = model.num_params()
    print(f"[train] model params: root={rp/1e6:.2f}M body={bp/1e6:.2f}M total={(rp+bp)/1e6:.2f}M")

    sched = make_ddpm_schedule(args.T).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ema = EMA(model, decay=args.ema_decay)

    weights = KimodoLossWeights(
        root_pos=args.w_root_pos, root_heading=args.w_root_head,
        joint_pos=args.w_joint_pos, joint_vel=args.w_joint_vel,
        joint_rot=args.w_joint_rot, foot=args.w_foot, fk=args.w_fk,
    )
    print(f"[train] loss weights: {asdict(weights)}")

    step = 0
    running: Dict[str, float] = {}

    def _step_decode_for_loss(x0_pred_packed_norm: torch.Tensor) -> KimodoMotion:
        """Unpack + denormalize predicted x0 so it's in original unit space,
        matching the target motion for loss computation."""
        motion_norm = unpack(x0_pred_packed_norm, layout)
        return denormalize(motion_norm, stats)

    data_iter = iter(dl)
    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)
        step += 1

        x0 = batch["x0_packed"].to(device)  # [B, T, total_dim] normalized
        B, T, _ = x0.shape

        # sample diffusion step
        t = torch.randint(0, sched.T, (B,), device=device)
        noise = torch.randn_like(x0)
        x_t = q_sample(x0, t, sched, noise)

        x0_pred = model(x_t, t)  # [B, T, total_dim] normalized

        # Build prediction KimodoMotion in original units for loss
        pred_motion = _step_decode_for_loss(x0_pred)

        # Target motion: use the raw (unnormalized) values from batch
        target_motion = KimodoMotion(
            r_p=batch["raw_r_p"].to(device),
            r_a=batch["raw_r_a"].to(device),
            j_p=batch["raw_j_p"].to(device),
            j_v=batch["raw_j_v"].to(device),
            j_a=batch["raw_j_a"].to(device),
            f=batch["raw_f"].to(device),
        )
        j_p_world = batch["j_p_world"].to(device)
        r_p_world = batch["r_p_world"].to(device)

        total_loss, term_dict = kimodo_loss(
            pred_motion, target_motion,
            skel=skel, j_p_target_for_fk=j_p_world, root_pos_target_world=r_p_world,
            weights=weights,
        )

        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.ema_every == 0:
            ema.update(model)

        # logging
        term_dict["total"] = float(total_loss.detach().item())
        for k, v in term_dict.items():
            running[k] = running.get(k, 0.0) * 0.95 + v * 0.05
        if step == 1 or step % args.log_every == 0:
            elapsed = time.time() - t_start
            msg = (f"[train] step {step}/{args.steps} ({elapsed:.1f}s) "
                   f"total={running['total']:.3f} "
                   f"root_p={running['root_pos']:.3f} "
                   f"root_h={running['root_heading']:.3f} "
                   f"jp={running['joint_pos']:.3f} "
                   f"jv={running['joint_vel']:.3f} "
                   f"ja={running['joint_rot']:.3f} "
                   f"foot={running['foot']:.3f} "
                   f"fk={running['fk']:.3f}")
            print(msg, flush=True)

        # periodic sanity sample (teacher-free rollout)
        if step % args.sample_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                # copy EMA weights into a snapshot model
                ema_model = copy.deepcopy(model)
                ema.copy_to(ema_model)
                ema_model.eval()
                def x0_fn(x, tt):
                    return ema_model(x, tt)
                shape = (1, T, layout.total_dim)
                x0_sampled = ddim_sample(x0_fn, shape, sched, n_steps=args.sample_steps, device=device)
                sampled = _step_decode_for_loss(x0_sampled)
                # compute per-term distance to clip 0
                target0 = KimodoMotion(
                    r_p=target_motion.r_p[0:1], r_a=target_motion.r_a[0:1],
                    j_p=target_motion.j_p[0:1], j_v=target_motion.j_v[0:1],
                    j_a=target_motion.j_a[0:1], f=target_motion.f[0:1],
                )
                sample_err = {
                    "r_p": float((sampled.r_p - target0.r_p).abs().mean()),
                    "j_p": float((sampled.j_p - target0.j_p).abs().mean()),
                    "j_a": float((sampled.j_a - target0.j_a).abs().mean()),
                }
                print(f"[sample] step {step} free-rollout MAE vs clip0: {sample_err}", flush=True)
                # save last sampled motion
                torch.save({
                    "r_p": sampled.r_p.cpu(), "r_a": sampled.r_a.cpu(),
                    "j_p": sampled.j_p.cpu(), "j_v": sampled.j_v.cpu(),
                    "j_a": sampled.j_a.cpu(), "f": sampled.f.cpu(),
                }, str(save_dir / f"sample_step{step}.pt"))
            model.train()

    # final save
    ema_model = copy.deepcopy(model)
    ema.copy_to(ema_model)
    torch.save({
        "model_raw": model.state_dict(),
        "model_ema": ema_model.state_dict(),
        "cfg": asdict(cfg),
        "layout": {"n_joints": layout.n_joints, "n_foot": layout.n_foot},
        "args": vars(args),
    }, str(save_dir / "final.pt"))
    print(f"[train] saved final checkpoint to {save_dir/'final.pt'}")


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--save_dir", type=Path, required=True)
    ap.add_argument("--overfit_n", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--ema_decay", type=float, default=0.995)
    ap.add_argument("--ema_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--sample_every", type=int, default=2000)
    ap.add_argument("--sample_steps", type=int, default=100)
    # model config
    ap.add_argument("--root_d", type=int, default=192)
    ap.add_argument("--root_layers", type=int, default=3)
    ap.add_argument("--root_heads", type=int, default=4)
    ap.add_argument("--body_d", type=int, default=256)
    ap.add_argument("--body_layers", type=int, default=4)
    ap.add_argument("--body_heads", type=int, default=4)
    # loss weights (match Kimodo)
    ap.add_argument("--w_root_pos", type=float, default=10.0)
    ap.add_argument("--w_root_head", type=float, default=2.0)
    ap.add_argument("--w_joint_pos", type=float, default=10.0)
    ap.add_argument("--w_joint_vel", type=float, default=3.0)
    ap.add_argument("--w_joint_rot", type=float, default=10.0)
    ap.add_argument("--w_foot", type=float, default=4.0)
    ap.add_argument("--w_fk", type=float, default=5.0)
    args = ap.parse_args()
    run_training(args)


if __name__ == "__main__":
    _cli()
