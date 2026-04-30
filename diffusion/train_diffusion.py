# diffusion/train_diffusion.py
import argparse
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from feature_layout import MotionFeatureLayout
except Exception:
    from diffusion.feature_layout import MotionFeatureLayout

try:
    from diffusion_policy import (
        DDPMScheduler,
        MotionDiffusionTransformer,
        rot6d_to_matrix,
        so3_log_map,
        so3_relative,
    )
except Exception:
    from diffusion.diffusion_policy import (
        DDPMScheduler,
        MotionDiffusionTransformer,
        rot6d_to_matrix,
        so3_log_map,
        so3_relative,
    )

# -----------------------------------------------------------------------------
# 1. Helper Classes
# -----------------------------------------------------------------------------
class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self._init_from(model)
    def _init_from(self, model: torch.nn.Module):
        self.shadow = {}
        for k, v in model.state_dict().items():
            if torch.is_floating_point(v):
                self.shadow[k] = v.detach().clone().float().cpu()
    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in msd.items():
            if not torch.is_floating_point(v): continue
            new = v.detach().float().cpu()
            if k not in self.shadow: self.shadow[k] = new.clone()
            else: self.shadow[k].mul_(d).add_(new, alpha=(1.0 - d))
    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}
    def load_state_dict(self, sd: Dict[str, Any]):
        self.decay = float(sd.get("decay", self.decay))
        shadow = sd.get("shadow", {})
        if isinstance(shadow, dict):
            self.shadow = {k: v.detach().clone().float().cpu() for k, v in shadow.items()}

class SkeletonFK(nn.Module):
    """
    智能 FK 模块：处理 Skeleton(88) 到 Feature(75) 的映射
    """
    def __init__(self, offsets: List[List[float]], parents: List[int],
                 all_names: List[str], feature_names: List[str]):
        super().__init__()
        offsets_np = np.asarray(offsets, dtype=np.float32)
        bone_len = np.linalg.norm(offsets_np, axis=1)
        non_zero = bone_len[bone_len > 1e-6]
        self.offset_scale = 1.0
        self.offset_stats = None
        if non_zero.size > 0:
            median_len = float(np.median(non_zero))
            max_len = float(np.max(non_zero))
            self.offset_stats = {"median": median_len, "max": max_len}
            # BVH offsets are often stored in centimeters while root motion here is near meters.
            # When the skeleton has implausibly large bone lengths, scale offsets to meters.
            if max_len > 10.0 and median_len > 1.0:
                self.offset_scale = 0.01
                offsets_np = offsets_np * self.offset_scale

        self.register_buffer("offsets", torch.tensor(offsets_np, dtype=torch.float32))
        self.register_buffer("parents", torch.tensor(parents, dtype=torch.long))

        self.J_all = len(all_names)     # 88
        self.J_feat = len(feature_names) # 75

        # 建立映射: Skeleton Index -> Feature Index
        skel_to_feat = []
        feat_map = {name: i for i, name in enumerate(feature_names)}

        for name in all_names:
            if name in feat_map:
                skel_to_feat.append(feat_map[name])
            else:
                # 尝试去掉 _End 后缀匹配
                clean = name.replace("_End", "")
                if clean in feat_map:
                    skel_to_feat.append(feat_map[clean])
                else:
                    skel_to_feat.append(-1)

        self.register_buffer("map_s2f", torch.tensor(skel_to_feat, dtype=torch.long))
        self.all_names = all_names

    def forward(self, rot_mats_feat: torch.Tensor, root_pos: torch.Tensor):
        B, T, _, _, _ = rot_mats_feat.shape
        identity = torch.eye(3, device=rot_mats_feat.device).view(1, 1, 3, 3).expand(B, T, 3, 3)

        global_rots = [None] * self.J_all
        global_pos = [None] * self.J_all

        for i in range(self.J_all):
            parent = self.parents[i].item()
            offset = self.offsets[i]

            feat_idx = self.map_s2f[i].item()
            if feat_idx >= 0:
                local_r = rot_mats_feat[:, :, feat_idx]
            else:
                local_r = identity

            if parent == -1:
                global_rots[i] = local_r
                global_pos[i] = root_pos
            else:
                parent_r = global_rots[parent]
                parent_p = global_pos[parent]
                global_rots[i] = torch.matmul(parent_r, local_r)
                off_rotated = torch.matmul(parent_r, offset.view(1, 1, 3, 1)).squeeze(-1)
                global_pos[i] = parent_p + off_rotated

        return torch.stack(global_pos, dim=2)

# -----------------------------------------------------------------------------
# Utils & Dataset
# -----------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_dist() -> bool: return dist.is_available() and dist.is_initialized()
def get_rank() -> int: return dist.get_rank() if is_dist() else 0
def get_world_size() -> int: return dist.get_world_size() if is_dist() else 1
def is_main() -> bool: return get_rank() == 0
def ddp_barrier(device=None):
    if not is_dist(): return
    if device and device.type == 'cuda': dist.barrier(device_ids=[device.index])
    else: dist.barrier()

def ddp_all_reduce_mean(x, device):
    if not is_dist(): return float(x)
    t = torch.tensor([float(x)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / get_world_size())

def make_autocast(device):
    try: return torch.amp.autocast("cuda" if device.type=="cuda" else "cpu")
    except: return torch.cuda.amp.autocast()
def make_grad_scaler(enabled):
    try: return torch.amp.GradScaler("cuda", enabled=enabled)
    except: return torch.cuda.amp.GradScaler(enabled=enabled)

class MotionAudioWordDataset(Dataset):
    def __init__(self, data_path: str, normalize_audio=True):
        print(f"[Dataset] Loading {data_path} ...")
        payload = torch.load(data_path, map_location="cpu", weights_only=False)
        self.data = payload["data"]
        self.mean = np.asarray(payload["mean"], dtype=np.float32)
        self.std = np.asarray(payload["std"], dtype=np.float32)
        self.std[self.std < 1e-6] = 1.0
        self.mean_t = torch.from_numpy(self.mean).float()
        self.std_t = torch.from_numpy(self.std).float()
        self.fps = int(payload.get("fps", 30))

        self.meta = payload.get("meta", {})
        self.motion_dim = int(self.mean.shape[0])
        self.layout = MotionFeatureLayout.from_metadata(self.meta, motion_dim=self.motion_dim)
        self.rot6d_start = int(self.layout.rot6d_start)
        self.contact_indices = list(self.layout.contact_indices)
        self.J = int((self.motion_dim - self.rot6d_start) // 6)

        self.audio_mean_t = None
        if normalize_audio and payload.get("audio_mean") is not None:
            self.audio_mean_t = torch.from_numpy(payload["audio_mean"]).float()
            self.audio_std_t = torch.from_numpy(payload["audio_std"]).float()
            self.audio_std_t[self.audio_std_t < 1e-6] = 1.0
        self.audio_mean = payload.get("audio_mean")
        self.audio_std = payload.get("audio_std")

        s0 = self.data[0]
        self.audio_dim = s0["audio"].shape[1]

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        mot = (torch.from_numpy(item["motion"]).float() - self.mean_t) / self.std_t
        aud = torch.from_numpy(item["audio"]).float()
        if self.audio_mean_t is not None: aud = (aud - self.audio_mean_t) / self.audio_std_t

        if item.get("word_times") is not None:
            w_arr = np.asarray(item["word_times"], dtype=np.float32).reshape(-1, 2)
            w_t = torch.from_numpy(w_arr).float()
        else:
            wt = []
            for w in item.get("words", []):
                try:
                    wt.append([float(w["start"]), float(w["end"])])
                except Exception:
                    pass
            w_t = torch.tensor(wt, dtype=torch.float32) if wt else torch.zeros((0, 2))

        return {
            "motion": mot, "text_ids": torch.tensor(item["text_ids"]).long(),
            "text_mask": torch.tensor(item["text_mask"]).long(),
            "audio": aud, "word_times": w_t,
            "motion_len": mot.shape[0], "audio_len": aud.shape[0], "word_len": w_t.shape[0]
        }

def collate_fn(batch):
    B = len(batch)
    max_m = max(b["motion_len"] for b in batch)
    max_a = max(b["audio_len"] for b in batch)
    max_w = max(b["word_len"] for b in batch)

    mot = torch.zeros(B, max_m, batch[0]["motion"].shape[1])
    mot_mask = torch.zeros(B, max_m, dtype=torch.bool)
    aud = torch.zeros(B, max_a, batch[0]["audio"].shape[1])
    aud_mask = torch.zeros(B, max_a, dtype=torch.bool)
    w_t = torch.zeros(B, max_w, 2)
    w_mask = torch.zeros(B, max_w, dtype=torch.bool)
    text_ids_lst, text_mask_lst = [], []

    for i, b in enumerate(batch):
        mot[i, :b["motion_len"]] = b["motion"]
        mot_mask[i, :b["motion_len"]] = True
        aud[i, :b["audio_len"]] = b["audio"]
        aud_mask[i, :b["audio_len"]] = True
        if b["word_len"] > 0:
            w_t[i, :b["word_len"]] = b["word_times"]
            w_mask[i, :b["word_len"]] = True
        text_ids_lst.append(b["text_ids"])
        text_mask_lst.append(b["text_mask"])

    return {
        "motion": mot, "motion_mask": mot_mask,
        "text_ids": pad_sequence(text_ids_lst, batch_first=True),
        "text_mask": pad_sequence(text_mask_lst, batch_first=True),
        "audio": aud, "audio_mask": aud_mask,
        "word_times": w_t, "word_mask": w_mask
    }


def resolve_joint_index(all_names: List[str], target: str) -> int:
    if target in all_names:
        return all_names.index(target)

    target_low = target.lower()
    for i, name in enumerate(all_names):
        low = name.lower()
        if low.endswith("_end") or "end site" in low:
            continue
        if low == target_low:
            return i

    for i, name in enumerate(all_names):
        low = name.lower()
        if low.endswith("_end") or "end site" in low:
            continue
        if target_low in low:
            return i

    clean_target = target_low.replace("_end", "")
    for i, name in enumerate(all_names):
        low = name.lower().replace("_end", "")
        if clean_target == low or clean_target in low:
            return i

    return -1


def serialize_args(args: argparse.Namespace, ema_obj: Optional[EMA]) -> Dict[str, Any]:
    out = {}
    for k, v in vars(args).items():
        if k == "ema":
            out[k] = bool(ema_obj is not None)
        elif isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def build_checkpoint_payload(
    *,
    model,
    optimizer,
    scaler,
    ema_obj: Optional[EMA],
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    dataset: MotionAudioWordDataset,
):
    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    layout_meta = dict(dataset.meta)
    layout_meta.update(dataset.layout.to_metadata())
    return {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": ema_obj.state_dict() if ema_obj is not None else None,
        "args": serialize_args(args, ema_obj),
        "mean": dataset.mean.astype(np.float32),
        "std": dataset.std.astype(np.float32),
        "audio_mean": None if dataset.audio_mean is None else np.asarray(dataset.audio_mean, dtype=np.float32),
        "audio_std": None if dataset.audio_std is None else np.asarray(dataset.audio_std, dtype=np.float32),
        "fps": int(dataset.fps),
        "audio_dim": int(dataset.audio_dim),
        "motion_dim": int(dataset.motion_dim),
        "rot6d_start": int(dataset.rot6d_start),
        "J": int(dataset.J),
        "contact_indices": list(dataset.contact_indices),
        "layout_meta": layout_meta,
        "meta": dataset.meta,
    }

# -----------------------------------------------------------------------------
# Epoch Runner
# -----------------------------------------------------------------------------
def run_epoch(
    model, loader, optimizer, ddpm, device,
    motion_mean, motion_std, fps, motion_layout, J,
    scaler, autocast_ctx, amp_enabled, epoch, grad_clip,
    args, fk_module, foot_indices,
    base_lr, warmup_steps, total_steps, global_step
):
    model.train()
    dt = 1.0 / max(1, fps)
    Dm = int(motion_mean.numel())
    rot6d_start = int(motion_layout.rot6d_start)

    # Feature mask
    dim_mask = torch.zeros(Dm, device=device, dtype=torch.bool)
    if args.dyn_dim_mode == "pre_rot6d": dim_mask[:rot6d_start] = True
    elif args.dyn_dim_mode == "all_no_contact": dim_mask[:] = True
    for ci in motion_layout.contact_indices:
        if ci < Dm: dim_mask[ci] = False

    loss_acc = {
        k: 0.0
        for k in ["loss", "main", "skate", "gnd", "vel", "acc", "rot", "ang_vel", "ang_acc"]
    }
    batches = 0
    lr = 0.0

    cur_skate_w = args.skate_w
    cur_height_w = args.height_w
    physics_scale = 1.0
    if epoch <= args.physics_start_epoch:
        physics_scale = 0.0
    elif args.physics_ramp_epochs > 0:
        physics_scale = min(1.0, float(epoch - args.physics_start_epoch) / float(args.physics_ramp_epochs))
    cur_skate_w *= physics_scale
    cur_height_w *= physics_scale

    pbar = tqdm(loader, desc=f"Ep {epoch}", disable=not is_main())
    optimizer.zero_grad(set_to_none=True)
    accum_counter = 0

    for it, batch in enumerate(pbar):
        x0 = batch["motion"].to(device, non_blocking=True)
        m_mask = batch["motion_mask"].to(device, non_blocking=True)
        B, T, _ = x0.shape

        with autocast_ctx:
            t = torch.randint(0, ddpm.num_timesteps, (B,), device=device).long()
            eps = torch.randn_like(x0)
            x_t = ddpm.add_noise(x0, eps, t)

            pred = model(x_t, t, batch["text_ids"].to(device), batch["text_mask"].to(device),
                         m_mask, batch["audio"].to(device), batch["audio_mask"].to(device),
                         batch["word_times"].to(device), batch["word_mask"].to(device))

            # --- Manual Target Calculation ---
            alpha = ddpm.sqrt_alphas_cumprod[t].view(-1, 1, 1)
            sigma = ddpm.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

            if args.pred_type == "v":
                target = alpha * eps - sigma * x0
                main_loss = (pred - target).square()
                x0_pred = alpha * x_t - sigma * pred
            else:
                target = eps
                main_loss = (pred - target).square()
                x0_pred = (x_t - sigma * pred) / (alpha + 1e-8)

            main_loss = (main_loss * m_mask.unsqueeze(-1)).sum() / (m_mask.sum() * x0.shape[-1])

            x0_true_feat = x0 * motion_std + motion_mean
            x0_pred_feat = x0_pred * motion_std + motion_mean

            # Vel/Acc Loss
            vel_loss = acc_loss = torch.tensor(0.0, device=device)
            if args.vel_w > 0 and T >= 2:
                v_true = x0_true_feat[:, 1:] - x0_true_feat[:, :-1]
                v_pred = x0_pred_feat[:, 1:] - x0_pred_feat[:, :-1]
                v_true, v_pred = v_true[..., dim_mask], v_pred[..., dim_mask]
                mm = m_mask[:, 1:] & m_mask[:, :-1]
                vel_loss = ((v_pred - v_true).square() * mm.unsqueeze(-1)).sum() / (mm.sum() * v_true.shape[-1] + 1e-6)

            if args.acc_w > 0 and T >= 3:
                a_true = x0_true_feat[:, 2:] - 2 * x0_true_feat[:, 1:-1] + x0_true_feat[:, :-2]
                a_pred = x0_pred_feat[:, 2:] - 2 * x0_pred_feat[:, 1:-1] + x0_pred_feat[:, :-2]
                a_true, a_pred = a_true[..., dim_mask], a_pred[..., dim_mask]
                mm = m_mask[:, 2:] & m_mask[:, 1:-1] & m_mask[:, :-2]
                acc_loss = ((a_pred - a_true).square() * mm.unsqueeze(-1)).sum() / (mm.sum() * a_true.shape[-1] + 1e-6)

            # Rot Reg
            rot6_true = x0_true_feat[..., rot6d_start:].view(B, T, J, 6)
            rot6_pred = x0_pred_feat[..., rot6d_start:].view(B, T, J, 6)
            rot_reg = torch.tensor(0.0, device=device)
            if args.rot_w > 0:
                rot_reg = (
                    (rot6_pred[..., :3].norm(dim=-1) - 1).square()
                    + (rot6_pred[..., 3:].norm(dim=-1) - 1).square()
                ).mean()

            ang_vel_loss = ang_acc_loss = torch.tensor(0.0, device=device)
            if (args.ang_vel_w > 0 or args.ang_acc_w > 0) and T >= 2:
                r_true = rot6d_to_matrix(rot6_true)
                r_pred = rot6d_to_matrix(rot6_pred)
                rel_true = so3_relative(r_true[:, :-1], r_true[:, 1:])
                rel_pred = so3_relative(r_pred[:, :-1], r_pred[:, 1:])
                w_true = so3_log_map(rel_true) / dt
                w_pred = so3_log_map(rel_pred) / dt
                mm = m_mask[:, 1:] & m_mask[:, :-1]

                if args.ang_vel_w > 0:
                    ang_vel_loss = (
                        ((w_pred - w_true).square() * mm.unsqueeze(-1).unsqueeze(-1)).sum()
                        / (mm.sum() * J * 3 + 1e-6)
                    )

                if args.ang_acc_w > 0 and T >= 3:
                    aw_true = w_true[:, 1:] - w_true[:, :-1]
                    aw_pred = w_pred[:, 1:] - w_pred[:, :-1]
                    mm2 = mm[:, 1:] & mm[:, :-1]
                    ang_acc_loss = (
                        ((aw_pred - aw_true).square() * mm2.unsqueeze(-1).unsqueeze(-1)).sum()
                        / (mm2.sum() * J * 3 + 1e-6)
                    )

            # Physics Loss
            skate_loss = torch.tensor(0.0, device=device)
            height_loss = torch.tensor(0.0, device=device)

            if (
                (cur_skate_w > 0 or cur_height_w > 0)
                and fk_module is not None
                and foot_indices is not None
                and foot_indices.numel() > 0
                and len(motion_layout.contact_indices) > 0
                and T > 1
            ):
                root_pos = motion_layout.decode_root_pos(x0_pred_feat, dt)
                r_pred = rot6d_to_matrix(rot6_pred)

                n_match = min(int(foot_indices.numel()), len(motion_layout.contact_indices))
                feet_use = foot_indices[:n_match]
                c_indices = list(motion_layout.contact_indices[:n_match])

                global_pos = fk_module(r_pred, root_pos)
                feet_pos = global_pos[:, :, feet_use]
                c_pred = x0_pred_feat[..., c_indices].sigmoid()

                if cur_skate_w > 0 and n_match > 0:
                    f_vel = (feet_pos[:, 1:] - feet_pos[:, :-1]).norm(dim=-1) / dt
                    skate_w = c_pred[:, :-1] * (m_mask[:, 1:] & m_mask[:, :-1]).unsqueeze(-1).float()
                    skate_loss = (skate_w * f_vel).sum() / (skate_w.sum() + 1e-6)

                if cur_height_w > 0 and n_match > 0:
                    feet_y = feet_pos[..., 1]
                    feet_valid = m_mask.unsqueeze(-1).expand_as(feet_y)
                    seq_ground = feet_y.masked_fill(~feet_valid, float("inf")).amin(dim=(1, 2), keepdim=True)
                    seq_ground = torch.where(torch.isfinite(seq_ground), seq_ground, torch.zeros_like(seq_ground)).detach()
                    feet_clearance = feet_y - seq_ground
                    height_w = c_pred * feet_valid.float()
                    height_loss = (height_w * feet_clearance.square()).sum() / (height_w.sum() + 1e-6)

            total_loss = (
                main_loss
                + args.rot_w * rot_reg
                + args.vel_w * vel_loss
                + args.acc_w * acc_loss
                + args.ang_vel_w * ang_vel_loss
                + args.ang_acc_w * ang_acc_loss
                + cur_skate_w * skate_loss
                + cur_height_w * height_loss
            )

        loss_acc["loss"] += float(total_loss.item())
        loss_acc["main"] += float(main_loss.item())
        loss_acc["vel"] += float(vel_loss.item())
        loss_acc["acc"] += float(acc_loss.item())
        loss_acc["rot"] += float(rot_reg.item())
        loss_acc["ang_vel"] += float(ang_vel_loss.item())
        loss_acc["ang_acc"] += float(ang_acc_loss.item())
        loss_acc["skate"] += float(skate_loss.item())
        loss_acc["gnd"] += float(height_loss.item())
        batches += 1

        scaled_loss = total_loss / args.grad_accum
        if amp_enabled:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        accum_counter += 1
        should_step = accum_counter >= args.grad_accum or (it + 1) == len(loader)
        if should_step:
            if amp_enabled:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if total_steps > 0:
                    p = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                    p = min(max(p, 0), 1)
                    lr_mult = args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))
                    if global_step < warmup_steps and warmup_steps > 0:
                        lr_mult = (global_step + 1) / warmup_steps
                    lr = base_lr * lr_mult
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if total_steps > 0:
                    p = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                    p = min(max(p, 0), 1)
                    lr_mult = args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))
                    if global_step < warmup_steps and warmup_steps > 0:
                        lr_mult = (global_step + 1) / warmup_steps
                    lr = base_lr * lr_mult
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            accum_counter = 0
            if args.ema:
                args.ema.update(model.module if isinstance(model, DDP) else model)

            global_step += 1

            if is_main():
                pbar.set_postfix(
                    loss=total_loss.item(),
                    sk=skate_loss.item(),
                    gd=height_loss.item(),
                    lr=lr,
                )
                if args.log_every > 0 and global_step % args.log_every == 0:
                    denom = max(1, batches)
                    print(
                        f"Step {global_step} Loss: {loss_acc['loss']/denom:.4f} "
                        f"Skate: {loss_acc['skate']/denom:.4f} "
                        f"Gnd: {loss_acc['gnd']/denom:.4f}"
                    )

    stats = {k: v / max(1, batches) for k, v in loss_acc.items()}
    stats["lr"] = lr
    return stats, global_step

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def legacy_main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="diffusion/cache/train_diffusion.pt")
    parser.add_argument("--save_dir", default="diffusion/checkpoints", type=Path)
    parser.add_argument("--bert_path", default="models/bert")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--no_audio_norm", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--cond_drop_prob", type=float, default=0.1)
    parser.add_argument("--skate_w", type=float, default=10.0)
    parser.add_argument("--height_w", type=float, default=10.0)
    parser.add_argument("--physics_start_epoch", type=int, default=5)
    parser.add_argument("--physics_ramp_epochs", type=int, default=10)
    parser.add_argument("--rot_w", type=float, default=0.02)
    parser.add_argument("--vel_w", type=float, default=0.05)
    parser.add_argument("--acc_w", type=float, default=0.02)
    parser.add_argument("--ang_vel_w", type=float, default=0.0)
    parser.add_argument("--ang_acc_w", type=float, default=0.0)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--pred_type", default="v")
    parser.add_argument("--dyn_dim_mode", default="pre_rot6d")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--debug_first_step", action="store_true")

    parser.set_defaults(amp=True)
    args, _ = parser.parse_known_args(argv)

    is_distributed = "LOCAL_RANK" in os.environ
    if is_distributed:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
        device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed + get_rank())

    ds = MotionAudioWordDataset(args.data, normalize_audio=(not args.no_audio_norm))
    sampler = DistributedSampler(ds) if is_distributed else None
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(not is_distributed),
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        persistent_workers=bool(args.persistent_workers and args.num_workers > 0),
    )

    if is_main():
        print(f"Motion Dim: {ds.motion_dim} (J={ds.J})")
        print(f"Feature Layout: {ds.layout.describe()}")
        if "joint_names" in ds.meta:
            print(f"Found Metadata for Skeleton! Joints: {len(ds.meta['joint_names'])}")
        else:
            print("WARNING: No skeleton meta found in cache.")

    fk_module = None
    foot_indices = None
    if "joint_names" in ds.meta:
        fk_module = SkeletonFK(
            ds.meta["skeleton_offsets"],
            ds.meta["skeleton_parents"],
            ds.meta["all_joint_names"],
            ds.meta["joint_names"]
        ).to(device)
        if is_main() and getattr(fk_module, "offset_scale", 1.0) != 1.0:
            stats = getattr(fk_module, "offset_stats", {}) or {}
            print(
                "[FK] Auto-scaled skeleton offsets by "
                f"{fk_module.offset_scale:g} "
                f"(median_bone={stats.get('median', 0.0):.3f}, max_bone={stats.get('max', 0.0):.3f})"
            )

        all_names = ds.meta["all_joint_names"]
        target_feet = list(ds.layout.foot_names) if ds.layout.foot_names else ["RightFoot", "LeftFoot", "RightToe", "LeftToe"]
        feet_indices = []

        for target in target_feet:
            found = resolve_joint_index(all_names, target)
            if found != -1:
                feet_indices.append(found)
            else:
                if is_main():
                    print(f"[WARN] Could not find foot joint: {target}")

        if feet_indices:
            foot_indices = torch.tensor(feet_indices, device=device).long()
        if is_main():
            print(f"Foot joints used for physics: {target_feet}")
            print(f"Resolved foot indices: {feet_indices}")

    model = MotionDiffusionTransformer(
        motion_dim=ds.motion_dim,
        audio_dim=ds.audio_dim,
        bert_path=args.bert_path,
        motion_fps=float(ds.fps),
        hidden=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        cond_drop_prob=args.cond_drop_prob,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = make_grad_scaler(amp_enabled)
    ddpm = DDPMScheduler(num_timesteps=1000, device=device)

    ema_obj = EMA(model, decay=args.ema_decay) if args.ema else None

    start_epoch = 1
    global_step = 0
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=True)
            if ckpt.get("optimizer") is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
            if ckpt.get("scaler") is not None:
                scaler.load_state_dict(ckpt["scaler"])
            if ema_obj is not None and ckpt.get("ema") is not None:
                ema_obj.load_state_dict(ckpt["ema"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            global_step = int(ckpt.get("global_step", 0))
            if is_main():
                print(f"[Resume] Loaded full checkpoint: {args.resume} (epoch={start_epoch - 1}, step={global_step})")
        else:
            model.load_state_dict(ckpt, strict=True)
            if is_main():
                print(f"[Resume] Loaded model weights only: {args.resume}")

    if is_distributed:
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])
        if ema_obj is not None:
            ema_obj = EMA(model.module, decay=args.ema_decay)
            if args.resume and isinstance(ckpt, dict) and ckpt.get("ema") is not None:
                ema_obj.load_state_dict(ckpt["ema"])

    args.ema = ema_obj

    args.save_dir.mkdir(parents=True, exist_ok=True)
    total_steps = max(1, len(loader) * max(0, args.epochs - start_epoch + 1) // max(1, args.grad_accum))

    for epoch in range(start_epoch, args.epochs + 1):
        if is_distributed:
            loader.sampler.set_epoch(epoch)
        stats, global_step = run_epoch(
            model, loader, optimizer, ddpm, device,
            ds.mean_t.to(device), ds.std_t.to(device), ds.fps, ds.layout, ds.J,
            scaler, make_autocast(device), amp_enabled, epoch, args.grad_clip,
            args, fk_module, foot_indices,
            args.lr, args.warmup_steps, total_steps, global_step
        )

        if is_main():
            print(
                f"[Epoch {epoch}] loss={stats['loss']:.4f} main={stats['main']:.4f} "
                f"vel={stats['vel']:.4f} acc={stats['acc']:.4f} "
                f"ang_vel={stats['ang_vel']:.4f} ang_acc={stats['ang_acc']:.4f} "
                f"sk={stats['skate']:.4f} gd={stats['gnd']:.4f} lr={stats['lr']:.6f}"
            )

        if is_main() and (epoch % args.save_every == 0):
            full_ckpt = build_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                ema_obj=args.ema,
                epoch=epoch,
                global_step=global_step,
                args=args,
                dataset=ds,
            )
            torch.save(full_ckpt, args.save_dir / "diffusion.pt")
            torch.save(model.module.state_dict() if is_distributed else model.state_dict(), args.save_dir / f"diffusion_ep{epoch}.pt")
            print(f"Saved epoch {epoch}")

def main(argv: Optional[List[str]] = None):
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--legacy", action="store_true")
    wrapper.add_argument("--stage", choices=["vae", "flow"], default=None)
    known, remaining = wrapper.parse_known_args(argv)

    if known.legacy:
        return legacy_main(remaining)

    stage = known.stage
    if stage is None:
        stage = "flow" if any(arg == "--vae_ckpt" for arg in remaining) else "vae"

    if stage == "vae":
        from diffusion.train_motion_vae import main as latent_main

        return latent_main(remaining)

    from diffusion.train_latent_flow import main as latent_main

    return latent_main(remaining)


if __name__ == "__main__":
    main()
