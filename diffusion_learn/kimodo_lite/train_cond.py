"""Conditional Kimodo-lite training with audio+text cross-attention + CFG.

Extends train.py for the conditional cache produced by tools/kimodo_cache_cond.py.
Key differences vs unconditional training:
  - Dataset yields audio [T, audio_dim] and text [text_dim] per segment
  - Forward pass receives audio + text + drop_mask for classifier-free guidance
  - Sampling uses CFG: x0 = (1 + w)*x0_cond - w*x0_uncond

Phase 7d sanity gate usage (4-clip text-only conditional overfit):
    python -m diffusion_learn.kimodo_lite.train_cond \
        --cache cache/kimodo_cond_tiny.pt --overfit_n 4 --steps 20000 \
        --save_dir diffusion_learn/runs/kimodo_cond_overfit4 --cfg_drop 0.1
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .diffusion import ddim_sample, make_ddpm_schedule, q_sample
from .feature_pack import FeatureLayout, denormalize, normalize, pack, unpack
from .loss import KimodoLossWeights, KimodoMotion, kimodo_loss
from .model_cond import CondConfig, ConditionalTwoStageDenoiser
from .model_concat import ConcatConfig, ConcatTwoStageDenoiser
from .train import EMA, skeleton_info_from_payload


class KimodoCondDataset(Dataset):
    """Conditional dataset: motion + audio + text per segment."""

    def __init__(self, payload: dict, layout: FeatureLayout, overfit_n: Optional[int] = None):
        self.layout = layout
        segs = payload["segments"]
        if overfit_n is not None:
            segs = segs[:overfit_n]
        self.segs = segs
        # build stats without audio (audio is fed raw to model)
        from .feature_pack import build_norm_stats
        self.stats = build_norm_stats(payload)

        self.samples = []
        for seg in segs:
            r_p = torch.from_numpy(np.asarray(seg["r_p"])).float()
            r_a = torch.from_numpy(np.asarray(seg["r_a"])).float()
            j_p = torch.from_numpy(np.asarray(seg["j_p"])).float()
            j_v = torch.from_numpy(np.asarray(seg["j_v"])).float()
            j_a = torch.from_numpy(np.asarray(seg["j_a"])).float()
            f = torch.from_numpy(np.asarray(seg["f"])).float()
            audio = torch.from_numpy(np.asarray(seg["audio"])).float()   # [T, audio_dim]
            if "text_frame" in seg:
                text_emb = torch.from_numpy(np.asarray(seg["text_frame"])).float()  # [T, text_dim]
            else:
                text_emb = torch.from_numpy(np.asarray(seg["text_emb"])).float()    # [text_dim]

            T = r_p.shape[0]
            J = layout.n_joints
            j_p_rel = j_p.view(T, J, 3)
            j_p_world = j_p_rel.clone()
            j_p_world[..., 0] += r_p[:, 0:1]
            j_p_world[..., 2] += r_p[:, 2:3]

            motion = KimodoMotion(
                r_p=r_p.unsqueeze(0), r_a=r_a.unsqueeze(0),
                j_p=j_p.unsqueeze(0), j_v=j_v.unsqueeze(0),
                j_a=j_a.unsqueeze(0), f=f.unsqueeze(0),
            )
            motion_norm = normalize(motion, self.stats)
            x0_packed = pack(motion_norm, layout).squeeze(0)

            self.samples.append({
                "x0_packed": x0_packed,
                "j_p_world": j_p_world,
                "r_p_world": r_p,
                "raw_r_p": r_p, "raw_r_a": r_a, "raw_j_p": j_p,
                "raw_j_v": j_v, "raw_j_a": j_a, "raw_f": f,
                "audio": audio,
                "text_emb": text_emb,
                "segment_id": seg["segment_id"],
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def cond_collate(batch):
    keys = ["x0_packed", "j_p_world", "r_p_world", "raw_r_p", "raw_r_a",
            "raw_j_p", "raw_j_v", "raw_j_a", "raw_f", "audio", "text_emb"]
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    out["segment_id"] = [b["segment_id"] for b in batch]
    return out


def ddim_sample_cfg(model, shape, sched, audio, text, w_cfg: float, n_steps: int, device):
    """DDIM sampling with classifier-free guidance.

    x0 = (1+w)*x0_cond - w*x0_uncond at each step.
    """
    B = shape[0]
    drop_ones = torch.ones(B, device=device)
    drop_zeros = torch.zeros(B, device=device)

    def cfg_fn(x, tt):
        if w_cfg == 0.0:
            return model(x, tt, audio=audio, text=text, drop_mask=drop_zeros)
        # concat batch for efficiency
        x_cat = torch.cat([x, x], dim=0)
        t_cat = torch.cat([tt, tt], dim=0)
        audio_cat = torch.cat([audio, audio], dim=0) if audio is not None else None
        text_cat = torch.cat([text, text], dim=0) if text is not None else None
        drop_cat = torch.cat([drop_zeros, drop_ones], dim=0)
        x0_cat = model(x_cat, t_cat, audio=audio_cat, text=text_cat, drop_mask=drop_cat)
        x0_cond, x0_uncond = x0_cat[:B], x0_cat[B:]
        return (1.0 + w_cfg) * x0_cond - w_cfg * x0_uncond

    return ddim_sample(cfg_fn, shape, sched, n_steps=n_steps, device=device)


def _soft_mask_1d(T: int, anchor_ranges, fade: int = 4, device="cpu") -> torch.Tensor:
    """Build a [T] float mask in [0, 1]. 1.0 inside anchor, linear fade to 0
    over `fade` frames at each boundary, 0.0 elsewhere.

    anchor_ranges: list of (start, end) inclusive-exclusive ranges.
    """
    m = torch.zeros(T, dtype=torch.float32, device=device)
    for (s, e) in anchor_ranges:
        m[s:e] = 1.0
    if fade <= 0:
        return m
    out = m.clone()
    for i in range(T):
        if m[i] > 0:
            continue
        # distance to nearest anchor frame
        left = -1
        right = T
        for j in range(i - 1, -1, -1):
            if m[j] > 0:
                left = j
                break
        for j in range(i + 1, T):
            if m[j] > 0:
                right = j
                break
        d = min(i - left if left >= 0 else fade + 1,
                right - i if right < T else fade + 1)
        if d <= fade:
            out[i] = 1.0 - d / (fade + 1)
    return out


@torch.no_grad()
def ddim_sample_cfg_inpaint(
    model, shape, sched, audio, text,
    *, w_cfg: float, n_steps: int, device,
    x0_known: torch.Tensor, known_mask: torch.Tensor,
    fade: int = 4, anchor_steps_frac: float = 0.7,
):
    """DDIM + CFG + inpainting with soft boundary and early-anchor schedule.

    Args:
      x0_known: [B, T, D] in PACKED+NORMALIZED space.
      known_mask: [T] bool or [T] float. If bool, a soft mask with `fade`-frame
        linear transition is built automatically. If float, used as-is.
      fade: frames over which to linearly fade the anchor at its boundary.
      anchor_steps_frac: only apply anchoring in the first `frac` of DDIM steps;
        after that, let the model run free so the boundary can smooth out.
        Set to 1.0 to anchor throughout.
    """
    from .diffusion import q_sample
    B, T, D = shape
    drop_ones = torch.ones(B, device=device)
    drop_zeros = torch.zeros(B, device=device)

    # Build soft mask in [0,1]
    if known_mask.dtype == torch.bool:
        # convert True-runs to anchor_ranges then soft-fade
        bm = known_mask.cpu().numpy()
        ranges = []
        in_run = False
        s = 0
        for i, v in enumerate(bm):
            if v and not in_run:
                s = i; in_run = True
            elif not v and in_run:
                ranges.append((s, i)); in_run = False
        if in_run:
            ranges.append((s, T))
        soft = _soft_mask_1d(T, ranges, fade=fade, device=device)
    else:
        soft = known_mask.to(device).float()
    # expand to [B, T, D]
    soft_exp = soft.view(1, T, 1).expand(B, T, D)

    def cfg_predict(x, tt):
        if w_cfg == 0.0:
            return model(x, tt, audio=audio, text=text, drop_mask=drop_zeros)
        x_cat = torch.cat([x, x], dim=0)
        t_cat = torch.cat([tt, tt], dim=0)
        audio_cat = torch.cat([audio, audio], dim=0) if audio is not None else None
        text_cat = torch.cat([text, text], dim=0) if text is not None else None
        drop_cat = torch.cat([drop_zeros, drop_ones], dim=0)
        x0_cat = model(x_cat, t_cat, audio=audio_cat, text=text_cat, drop_mask=drop_cat)
        x0_c, x0_u = x0_cat[:B], x0_cat[B:]
        return (1.0 + w_cfg) * x0_c - w_cfg * x0_u

    T_sched = sched.T
    ts = torch.linspace(T_sched - 1, 0, n_steps + 1).long().to(device)
    x = torch.randn(*shape, device=device)
    noise_known = torch.randn_like(x0_known)
    anchor_cutoff = int(n_steps * anchor_steps_frac)

    # init x: blend pure noise with noised anchor
    t0 = ts[0]
    x_known_t0 = q_sample(x0_known, torch.full((B,), int(t0), device=device, dtype=torch.long), sched, noise=noise_known)
    x = soft_exp * x_known_t0 + (1 - soft_exp) * x

    for i in range(n_steps):
        t = ts[i]
        t_next = ts[i + 1]
        t_batch = torch.full((B,), int(t), device=device, dtype=torch.long)

        x0_pred = cfg_predict(x, t_batch)
        # Blend GT into x0 prediction, weighted by soft mask AND only while anchoring active
        if i < anchor_cutoff:
            w = soft_exp
        else:
            w = torch.zeros_like(soft_exp)
        x0_pred = w * x0_known + (1 - w) * x0_pred

        ab_t = sched.alphas_bar[t]
        ab_next = sched.alphas_bar[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)
        noise_est = (x - torch.sqrt(ab_t) * x0_pred) / torch.sqrt(1.0 - ab_t).clamp_min(1e-8)
        dir_xt = torch.sqrt((1.0 - ab_next).clamp_min(0.0)) * noise_est
        x = torch.sqrt(ab_next) * x0_pred + dir_xt

        # re-anchor x_t at t_next only while anchoring active
        if i < anchor_cutoff:
            if t_next > 0:
                tn_batch = torch.full((B,), int(t_next), device=device, dtype=torch.long)
                x_known_tn = q_sample(x0_known, tn_batch, sched, noise=noise_known)
                x = w * x_known_tn + (1 - w) * x
            else:
                x = w * x0_known + (1 - w) * x

    return x


@torch.no_grad()
def ddim_sample_cfg_inpaint_native(
    model, shape, sched, audio, text,
    *, w_cfg: float, n_steps: int, device,
    x0_known: torch.Tensor, known_mask: torch.Tensor,
    resample_jumps: int = 0, resample_n: int = 3,
):
    """Inpaint for mask-aware models (trained with --mask_prob > 0).

    Unlike RePaint-style, we pass CLEAN x0 at anchored frames throughout
    denoising (training-time distribution), and pass anchor_mask to the model
    so it knows which frames are given.

    Optional RePaint-style resampling:
      At each of `resample_jumps` evenly-spaced steps, noise back to current t
      and re-denoise `resample_n` times. Gives the model more chances to
      reconcile the anchor with the surrounding generated motion.
    """
    B, T, D = shape
    drop_ones = torch.ones(B, device=device)
    drop_zeros = torch.zeros(B, device=device)
    if known_mask.dtype == torch.bool:
        anchor_mask_1d = known_mask.to(device).float()
    else:
        anchor_mask_1d = known_mask.to(device).float()
    anchor_mask_bt = anchor_mask_1d.view(1, T, 1).expand(B, T, 1).contiguous()
    anchor_mask_btd = anchor_mask_1d.view(1, T, 1).expand(B, T, D)

    def cfg_predict(x, tt):
        if w_cfg == 0.0:
            return model(x, tt, audio=audio, text=text, drop_mask=drop_zeros,
                         anchor_mask=anchor_mask_bt)
        x_cat = torch.cat([x, x], dim=0)
        t_cat = torch.cat([tt, tt], dim=0)
        audio_cat = torch.cat([audio, audio], dim=0) if audio is not None else None
        text_cat = torch.cat([text, text], dim=0) if text is not None else None
        drop_cat = torch.cat([drop_zeros, drop_ones], dim=0)
        am_cat = torch.cat([anchor_mask_bt, anchor_mask_bt], dim=0)
        x0_cat = model(x_cat, t_cat, audio=audio_cat, text=text_cat,
                       drop_mask=drop_cat, anchor_mask=am_cat)
        x0_c, x0_u = x0_cat[:B], x0_cat[B:]
        return (1.0 + w_cfg) * x0_c - w_cfg * x0_u

    T_sched = sched.T
    ts = torch.linspace(T_sched - 1, 0, n_steps + 1).long().to(device)
    x = torch.randn(*shape, device=device)
    x = torch.where(anchor_mask_btd.bool(), x0_known, x)

    # evenly-spaced indices where we do resampling (avoid very early/late steps)
    resample_at = set()
    if resample_jumps > 0:
        for j in range(resample_jumps):
            idx = int((j + 1) * n_steps / (resample_jumps + 1))
            resample_at.add(idx)

    def _one_step(x, i):
        t = ts[i]
        t_next = ts[i + 1]
        t_batch = torch.full((B,), int(t), device=device, dtype=torch.long)
        x0_pred = cfg_predict(x, t_batch)
        x0_pred = torch.where(anchor_mask_btd.bool(), x0_known, x0_pred)
        ab_t = sched.alphas_bar[t]
        ab_next = sched.alphas_bar[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)
        noise_est = (x - torch.sqrt(ab_t) * x0_pred) / torch.sqrt(1.0 - ab_t).clamp_min(1e-8)
        dir_xt = torch.sqrt((1.0 - ab_next).clamp_min(0.0)) * noise_est
        x_prev = torch.sqrt(ab_next) * x0_pred + dir_xt
        x_prev = torch.where(anchor_mask_btd.bool(), x0_known, x_prev)
        return x_prev

    for i in range(n_steps):
        x = _one_step(x, i)
        # RePaint-style: at marked steps, noise back up then re-denoise
        if i in resample_at and i + 1 < n_steps:
            t_curr = ts[i + 1]  # the level we just arrived at
            t_prev_level = ts[i]  # what we will re-noise UP to
            for _ in range(resample_n):
                # noise x from level t_curr back to level t_prev_level
                ab_curr = sched.alphas_bar[t_curr]
                ab_prev = sched.alphas_bar[t_prev_level]
                # forward-diffuse the DELTA: x_{t_prev} = sqrt(ab_prev/ab_curr)*x + sqrt(1-ab_prev/ab_curr)*eps
                step_up = ab_prev / ab_curr.clamp_min(1e-8)
                eps = torch.randn_like(x)
                x_up = torch.sqrt(step_up) * x + torch.sqrt((1.0 - step_up).clamp_min(0.0)) * eps
                x_up = torch.where(anchor_mask_btd.bool(), x0_known, x_up)
                # denoise from t_prev_level back down to t_curr (one DDIM step)
                x = _one_step(x_up, i)

    return x


def _setup_ddp():
    """Detect DDP context (torchrun). Returns (is_ddp, rank, local_rank, world_size)."""
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 0, 1
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def run_training(args):
    t_start = time.time()
    is_ddp, rank, local_rank, world_size = _setup_ddp()
    is_main = (rank == 0)
    if is_ddp:
        device = f"cuda:{local_rank}"
    else:
        device = args.device
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    save_dir = Path(args.save_dir)
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        if is_main:
            print(msg, flush=True)

    log(f"[cond] DDP={is_ddp} world_size={world_size} rank={rank} device={device}")
    log(f"[cond] loading cache: {args.cache}")
    payload = torch.load(str(args.cache), map_location="cpu", weights_only=False)
    skel = skeleton_info_from_payload(payload, device)
    n_foot = int(payload["skeleton"]["foot_joint_indices"].shape[0])
    layout = FeatureLayout(n_joints=skel.n_joints, n_foot=n_foot)
    audio_dim = int(payload.get("audio_dim", 768))
    text_dim = int(payload.get("text_dim", 768))
    log(f"[cond] layout total={layout.total_dim}, audio_dim={audio_dim}, text_dim={text_dim}")

    ds = KimodoCondDataset(payload, layout, overfit_n=args.overfit_n)
    log(f"[cond] dataset: {len(ds)} clips, T={ds.samples[0]['x0_packed'].shape[0]}")

    stats = ds.stats.to(device)
    if is_ddp:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        dl = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                        num_workers=0, collate_fn=cond_collate, drop_last=True)
    else:
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=cond_collate, drop_last=False)

    if args.arch == "concat":
        cfg = ConcatConfig(
            root_d=args.root_d, root_layers=args.root_layers, root_heads=args.root_heads,
            body_d=args.body_d, body_layers=args.body_layers, body_heads=args.body_heads,
            audio_dim=audio_dim, text_dim=text_dim,
        )
        model_raw = ConcatTwoStageDenoiser(layout, cfg).to(device)
    else:
        cfg = CondConfig(
            root_d=args.root_d, root_layers=args.root_layers, root_heads=args.root_heads,
            body_d=args.body_d, body_layers=args.body_layers, body_heads=args.body_heads,
            audio_dim=audio_dim, text_dim=text_dim,
        )
        model_raw = ConditionalTwoStageDenoiser(layout, cfg).to(device)
    rp, bp = model_raw.num_params()
    log(f"[cond] model params: root={rp/1e6:.2f}M body={bp/1e6:.2f}M total={(rp+bp)/1e6:.2f}M")

    if is_ddp:
        model = DDP(model_raw, device_ids=[local_rank], find_unused_parameters=False)
    else:
        model = model_raw

    sched = make_ddpm_schedule(args.T).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ema = EMA(model_raw, decay=args.ema_decay)  # EMA tracks the un-wrapped model params

    weights = KimodoLossWeights(
        root_pos=args.w_root_pos, root_heading=args.w_root_head,
        joint_pos=args.w_joint_pos, joint_vel=args.w_joint_vel,
        joint_rot=args.w_joint_rot, foot=args.w_foot, fk=args.w_fk,
    )

    step = 0
    running: Dict[str, float] = {}

    def _decode(x0_pred_packed_norm):
        motion_norm = unpack(x0_pred_packed_norm, layout)
        return denormalize(motion_norm, stats)

    data_iter = iter(dl)
    epoch = 0
    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if is_ddp:
                dl.sampler.set_epoch(epoch)
            data_iter = iter(dl)
            batch = next(data_iter)
        step += 1

        x0 = batch["x0_packed"].to(device)
        B, T, _ = x0.shape
        audio = batch["audio"].to(device)
        text = batch["text_emb"].to(device)

        t = torch.randint(0, sched.T, (B,), device=device)
        noise = torch.randn_like(x0)
        x_t = q_sample(x0, t, sched, noise)

        # CFG dropout mask — independent per-sample
        drop_mask = (torch.rand(B, device=device) < args.cfg_drop).float()

        # Random anchor-mask training (Kimodo-style inpainting support).
        # For each sample, with prob args.mask_prob pick a mask strategy and
        # replace x_t at masked frames with CLEAN x0 (zero-noise). Pass
        # anchor_mask [B,T,1] so model knows where the clean info lives.
        anchor_mask = None
        if args.arch == "concat" and args.mask_prob > 0.0:
            anchor_mask = torch.zeros(B, T, device=device)
            strategies = torch.rand(B, device="cpu")
            for b in range(B):
                if strategies[b].item() > args.mask_prob:
                    continue  # no mask
                mode = torch.randint(0, 3, (1,)).item()
                if mode == 0:  # prefix
                    k = int(torch.randint(4, max(5, T // 2), (1,)).item())
                    anchor_mask[b, :k] = 1.0
                elif mode == 1:  # suffix
                    k = int(torch.randint(4, max(5, T // 2), (1,)).item())
                    anchor_mask[b, -k:] = 1.0
                else:  # scatter
                    ratio = float(torch.empty(1).uniform_(0.1, 0.4).item())
                    n = int(T * ratio)
                    idx = torch.randperm(T)[:n]
                    anchor_mask[b, idx] = 1.0
            # Replace x_t at masked frames with clean x0 (zero noise)
            mask_exp = anchor_mask.unsqueeze(-1).to(x_t.dtype)
            x_t = mask_exp * x0 + (1.0 - mask_exp) * x_t

        model_kwargs = dict(audio=audio, text=text, drop_mask=drop_mask)
        if anchor_mask is not None:
            model_kwargs["anchor_mask"] = anchor_mask
        x0_pred = model(x_t, t, **model_kwargs)

        pred_motion = _decode(x0_pred)
        target_motion = KimodoMotion(
            r_p=batch["raw_r_p"].to(device), r_a=batch["raw_r_a"].to(device),
            j_p=batch["raw_j_p"].to(device), j_v=batch["raw_j_v"].to(device),
            j_a=batch["raw_j_a"].to(device), f=batch["raw_f"].to(device),
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

        if step % args.ema_every == 0 and is_main:
            ema.update(model_raw)

        term_dict["total"] = float(total_loss.detach().item())
        for k, v in term_dict.items():
            running[k] = running.get(k, 0.0) * 0.95 + v * 0.05
        if (step == 1 or step % args.log_every == 0) and is_main:
            elapsed = time.time() - t_start
            print(
                f"[cond] step {step}/{args.steps} ({elapsed:.1f}s) "
                f"total={running['total']:.3f} r_p={running['root_pos']:.3f} "
                f"jp={running['joint_pos']:.3f} ja={running['joint_rot']:.3f} "
                f"fk={running['fk']:.3f}",
                flush=True,
            )

        # periodic checkpoint + CFG sample (rank 0 only)
        if (step % args.sample_every == 0 or step == args.steps) and is_main:
            model_raw.eval()
            with torch.no_grad():
                ema_model = copy.deepcopy(model_raw)
                ema.copy_to(ema_model)
                ema_model.eval()
                s0 = ds.samples[0]
                audio0 = s0["audio"].unsqueeze(0).to(device)
                text0 = s0["text_emb"].unsqueeze(0).to(device)
                T_frames = s0["x0_packed"].shape[0]
                shape = (1, T_frames, layout.total_dim)
                x0_samp = ddim_sample_cfg(
                    ema_model, shape, sched, audio0, text0,
                    w_cfg=args.cfg_w, n_steps=args.sample_steps, device=device,
                )
                sampled = _decode(x0_samp)
                err = {
                    "r_p": float((sampled.r_p - target_motion.r_p[0:1]).abs().mean()),
                    "j_p": float((sampled.j_p - target_motion.j_p[0:1]).abs().mean()),
                    "j_a": float((sampled.j_a - target_motion.j_a[0:1]).abs().mean()),
                }
                print(f"[cond] step {step} CFG(w={args.cfg_w}) sample MAE vs clip0: {err}", flush=True)
                # periodic save (safety net for overnight runs)
                torch.save({
                    "model_raw": model_raw.state_dict(),
                    "model_ema": ema_model.state_dict(),
                    "cfg": asdict(cfg),
                    "layout": {"n_joints": layout.n_joints, "n_foot": layout.n_foot},
                    "args": vars(args),
                    "step": step,
                }, str(save_dir / f"ckpt_step{step}.pt"))
            model_raw.train()

    if is_main:
        ema_model = copy.deepcopy(model_raw)
        ema.copy_to(ema_model)
        torch.save({
            "model_raw": model_raw.state_dict(),
            "model_ema": ema_model.state_dict(),
            "cfg": asdict(cfg),
            "layout": {"n_joints": layout.n_joints, "n_foot": layout.n_foot},
            "args": vars(args),
        }, str(save_dir / "final.pt"))
        log(f"[cond] saved final ckpt → {save_dir/'final.pt'}")

    if is_ddp:
        dist.destroy_process_group()


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
    ap.add_argument("--cfg_drop", type=float, default=0.1, help="CFG train-time drop prob")
    ap.add_argument("--cfg_w", type=float, default=2.0, help="CFG guidance scale at sample time")
    ap.add_argument("--arch", choices=["cross", "concat"], default="cross",
                    help="conditioning architecture: cross-attn (model_cond) or MDM-style input concat (model_concat)")
    ap.add_argument("--mask_prob", type=float, default=0.0,
                    help="per-sample prob of applying random anchor mask during training (concat arch only; 0=off)")
    # model
    ap.add_argument("--root_d", type=int, default=384)
    ap.add_argument("--root_layers", type=int, default=6)
    ap.add_argument("--root_heads", type=int, default=8)
    ap.add_argument("--body_d", type=int, default=512)
    ap.add_argument("--body_layers", type=int, default=8)
    ap.add_argument("--body_heads", type=int, default=8)
    # loss
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
