from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cache(cache_path: str | Path) -> Dict[str, Any]:
    payload = torch.load(str(cache_path), map_location="cpu", weights_only=False)
    required = {"mean", "std", "segments", "motion_contract", "fps"}
    missing = sorted(required.difference(payload.keys()))
    if missing:
        raise RuntimeError(f"cache missing keys: {missing}")
    return payload


@dataclass
class ClipRecord:
    segment_id: str
    src_bvh: str
    start: int
    end: int
    motion: np.ndarray
    root_anchor: Optional[np.ndarray] = None
    text: str = ""
    global_text: Optional[np.ndarray] = None
    text_ids: Optional[np.ndarray] = None
    text_mask: Optional[np.ndarray] = None
    word_times: Optional[np.ndarray] = None


@dataclass(frozen=True)
class RootVelocityReconstructionSpec:
    root_x_index: int
    root_y_index: int
    root_z_index: int
    yaw_index: int
    local_vel_x_index: int
    local_vel_z_index: int
    yaw_vel_index: int
    fps: float
    root_relative_first_frame: bool


def build_fixed_clips(
    payload: Dict[str, Any],
    *,
    clip_len: int,
    overfit_n: int | None = None,
    root_indices: Optional[List[int]] = None,
    root_relative_first_frame: bool = False,
) -> List[ClipRecord]:
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    root_idx = None if root_indices is None else np.asarray(root_indices, dtype=np.int64)
    if root_relative_first_frame and (root_idx is None or root_idx.size <= 0):
        raise RuntimeError("root_relative_first_frame requires non-empty root_indices")

    clips: List[ClipRecord] = []
    for seg in payload["segments"]:
        motion = np.asarray(seg["motion"], dtype=np.float32)
        if motion.shape[0] < clip_len:
            continue
        clip = motion[:clip_len].copy()
        clip_norm = (clip - mean) / std
        root_anchor = None
        if root_relative_first_frame:
            root_anchor = clip[0, root_idx].copy()
            clip_norm[:, root_idx] -= clip_norm[:1, root_idx]
        clips.append(
            ClipRecord(
                segment_id=str(seg["segment_id"]),
                src_bvh=str(seg["src_bvh"]),
                start=0,
                end=int(clip_len),
                motion=clip_norm.astype(np.float32, copy=False),
                root_anchor=root_anchor.astype(np.float32, copy=False) if root_anchor is not None else None,
                text=str(seg.get("text", "")),
                global_text=np.asarray(seg["global_text"], dtype=np.float32).copy() if "global_text" in seg else None,
                text_ids=np.asarray(seg["text_ids"], dtype=np.int64).copy() if "text_ids" in seg else None,
                text_mask=np.asarray(seg["text_mask"], dtype=np.int64).copy() if "text_mask" in seg else None,
                word_times=np.asarray(seg["word_times"], dtype=np.float32).copy() if "word_times" in seg else None,
            )
        )
        if overfit_n is not None and len(clips) >= int(overfit_n):
            break
    if not clips:
        raise RuntimeError(f"no segments with length >= clip_len={clip_len}")
    return clips


class FixedClipDataset(Dataset):
    def __init__(self, clips: List[ClipRecord]):
        self.clips = clips
        text_dim = 0
        text_token_len = 0
        word_len = 0
        for rec in clips:
            if rec.global_text is not None:
                text_dim = int(np.asarray(rec.global_text).shape[0])
                break
        self.text_dim = text_dim
        for rec in clips:
            if rec.text_ids is not None:
                text_token_len = int(np.asarray(rec.text_ids).shape[0])
                break
        self.text_token_len = text_token_len
        for rec in clips:
            if rec.word_times is not None:
                word_len = int(np.asarray(rec.word_times).shape[0])
                break
        self.word_len = word_len

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.clips[idx]
        word_times = torch.zeros((self.word_len, 2), dtype=torch.float32)
        word_mask = torch.zeros(self.word_len, dtype=torch.bool)
        if rec.word_times is not None:
            n_words = min(self.word_len, int(np.asarray(rec.word_times).shape[0]))
            if n_words > 0:
                word_times[:n_words] = torch.from_numpy(rec.word_times[:n_words]).float()
                word_mask[:n_words] = True
        return {
            "motion": torch.from_numpy(rec.motion).float(),
            "segment_id": rec.segment_id,
            "src_bvh": rec.src_bvh,
            "start": rec.start,
            "end": rec.end,
            "text": rec.text,
            "global_text": (
                torch.from_numpy(rec.global_text).float()
                if rec.global_text is not None
                else torch.zeros(self.text_dim, dtype=torch.float32)
            ),
            "text_ids": (
                torch.from_numpy(rec.text_ids).long()
                if rec.text_ids is not None
                else torch.zeros(self.text_token_len, dtype=torch.long)
            ),
            "text_mask": (
                torch.from_numpy(rec.text_mask).long()
                if rec.text_mask is not None
                else torch.zeros(self.text_token_len, dtype=torch.long)
            ),
            "word_times": word_times,
            "word_mask": word_mask,
        }


class FrozenBertTextEncoder:
    def __init__(self, bert_path: str | Path, motion_fps: float):
        self.bert_path = str(Path(bert_path))
        self.motion_fps = float(motion_fps)
        from transformers import BertModel, BertTokenizerFast

        try:
            self.tokenizer = BertTokenizerFast.from_pretrained(self.bert_path, local_files_only=True)
        except TypeError:
            self.tokenizer = BertTokenizerFast.from_pretrained(self.bert_path)
        except Exception:
            self.tokenizer = BertTokenizerFast.from_pretrained(self.bert_path)

        try:
            self.bert = BertModel.from_pretrained(self.bert_path, local_files_only=True)
        except TypeError:
            self.bert = BertModel.from_pretrained(self.bert_path)
        except Exception:
            self.bert = BertModel.from_pretrained(self.bert_path)

        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert.eval()
        self.hidden_size = int(self.bert.config.hidden_size)
        self._word_cache: Dict[str, torch.Tensor] = {}

    def to(self, device: torch.device) -> "FrozenBertTextEncoder":
        self.bert.to(device)
        self.bert.eval()
        return self

    @torch.no_grad()
    def encode(self, text_ids: torch.Tensor, text_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        ids = text_ids.to(device=device, dtype=torch.long)
        mask = text_mask.to(device=device, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        out = self.bert(input_ids=ids, attention_mask=mask)
        h = out.last_hidden_state
        m = (mask > 0).to(h.dtype).unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1.0)
        pooled = (h * m).sum(dim=1) / denom
        return pooled.to(torch.float32)

    @torch.no_grad()
    def encode_aligned_words(
        self,
        texts: List[str],
        word_times: torch.Tensor,
        word_mask: torch.Tensor,
        clip_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        bsz = len(texts)
        feats = torch.zeros((bsz, clip_len, self.hidden_size + 5), device=device, dtype=torch.float32)
        tol = 0.5 / max(self.motion_fps, 1e-6)
        t = torch.arange(clip_len, device=device, dtype=torch.float32) / max(self.motion_fps, 1e-6)

        for i, text in enumerate(texts):
            n_words = int(word_mask[i].sum().item())
            if n_words <= 0:
                continue
            word_emb = self._encode_words(text, device)
            wt = word_times[i, :n_words].to(device=device, dtype=torch.float32)
            n_use = min(int(word_emb.shape[0]), int(wt.shape[0]))
            if n_use <= 0:
                continue
            word_emb = word_emb[:n_use]
            wt = wt[:n_use]
            starts = wt[:, 0]
            ends = wt[:, 1]
            valid = ends > (starts + 1e-5)
            if int(valid.sum().item()) <= 0:
                continue
            starts = starts[valid]
            ends = ends[valid]
            word_emb = word_emb[valid]
            tcol = t[:, None]
            in_word = (tcol >= starts[None, :]) & (tcol < ends[None, :])
            has = in_word.any(dim=1)
            if int(has.sum().item()) <= 0:
                continue
            idx = in_word.to(torch.float32).argmax(dim=1)
            s_sel = starts[idx]
            e_sel = ends[idx]
            emb_sel = word_emb[idx]
            dur = (e_sel - s_sel).clamp(min=1e-4)
            prog = ((t - s_sel) / dur).clamp(0.0, 1.0)
            is_word = has.to(torch.float32)
            prog = torch.where(has, prog, torch.zeros_like(prog))
            dur = torch.where(has, dur, torch.zeros_like(dur))
            b_start = (has & (torch.abs(t - s_sel) <= tol)).to(torch.float32)
            b_end = (has & (torch.abs(t - e_sel) <= tol)).to(torch.float32)
            emb_sel = emb_sel * has.to(torch.float32).unsqueeze(-1)
            time_feat = torch.stack([is_word, prog, dur, b_start, b_end], dim=-1)
            feats[i] = torch.cat([emb_sel, time_feat], dim=-1)
        return feats

    @torch.no_grad()
    def _encode_words(self, text: str, device: torch.device) -> torch.Tensor:
        cached = self._word_cache.get(text)
        if cached is not None:
            return cached.to(device)
        words = text.strip().split()
        if not words:
            empty = torch.zeros((0, self.hidden_size), dtype=torch.float32)
            self._word_cache[text] = empty
            return empty.to(device)
        encoded = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=False,
        )
        word_ids = encoded.word_ids(batch_index=0)
        input_ids = encoded["input_ids"].to(device=device, dtype=torch.long)
        attention_mask = encoded["attention_mask"].to(device=device, dtype=torch.long)
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state[0]
        pooled_words: List[torch.Tensor] = []
        zero = torch.zeros(self.hidden_size, device=device, dtype=torch.float32)
        for word_idx in range(len(words)):
            token_idx = [j for j, wid in enumerate(word_ids) if wid == word_idx]
            if not token_idx:
                pooled_words.append(zero)
            else:
                pooled_words.append(h[token_idx].mean(dim=0).to(torch.float32))
        pooled = torch.stack(pooled_words, dim=0) if pooled_words else torch.zeros((0, self.hidden_size), device=device)
        self._word_cache[text] = pooled.detach().cpu()
        return pooled


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(1, half - 1)
    )
    angles = t[:, None] * freqs[None]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class TinyFlowTransformer(nn.Module):
    def __init__(
        self,
        *,
        motion_dim: int,
        clip_len: int,
        text_dim: int = 0,
        word_cond_dim: int = 0,
        hidden_dim: int = 128,
        layers: int = 4,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.clip_len = int(clip_len)
        self.text_dim = int(text_dim)
        self.word_cond_dim = int(word_cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.in_proj = nn.Linear(motion_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.text_proj = (
            nn.Sequential(
                nn.Linear(self.text_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if self.text_dim > 0
            else None
        )
        self.word_proj = (
            nn.Sequential(
                nn.Linear(self.word_cond_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if self.word_cond_dim > 0
            else None
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, clip_len, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.out_proj = nn.Linear(hidden_dim, motion_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        global_text: Optional[torch.Tensor] = None,
        word_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.in_proj(x_t) + self.pos_embed[:, : x_t.shape[1]]
        h = h + self.time_mlp(timestep_embedding(t, self.hidden_dim)).unsqueeze(1)
        if self.text_proj is not None:
            if global_text is None:
                raise RuntimeError("global_text is required when text_dim > 0")
            if global_text.ndim == 1:
                global_text = global_text.unsqueeze(0)
            h = h + self.text_proj(global_text).unsqueeze(1)
        if self.word_proj is not None:
            if word_cond is None:
                raise RuntimeError("word_cond is required when word_cond_dim > 0")
            h = h + self.word_proj(word_cond)
        h = self.encoder(h)
        return self.out_proj(h)


def denorm_motion(
    motion_norm: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    root_indices: List[int],
    root_anchor: Optional[np.ndarray],
) -> np.ndarray:
    motion = motion_norm * std + mean
    if root_anchor is None:
        return motion
    motion = motion.copy()
    root_idx = np.asarray(root_indices, dtype=np.int64)
    motion[:, root_idx] += root_anchor[None, :] - mean[root_idx][None, :]
    return motion


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :2].transpose(-1, -2).flatten(-2)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def canonicalize_rot6d_block(rot6d: torch.Tensor) -> torch.Tensor:
    if rot6d.shape[-1] <= 0 or rot6d.shape[-1] % 6 != 0:
        return rot6d
    rot_view = rot6d.reshape(*rot6d.shape[:-1], -1, 6)
    mats = rotation_6d_to_matrix(rot_view)
    return matrix_to_rotation_6d(mats).reshape_as(rot6d)


def project_motion_rot6d(motion: torch.Tensor, rot6d_start: int) -> torch.Tensor:
    if motion.shape[-1] <= rot6d_start:
        return motion
    return torch.cat([motion[:, :, :rot6d_start], canonicalize_rot6d_block(motion[:, :, rot6d_start:])], dim=-1)


def temporal_binomial_smooth(motion: torch.Tensor, passes: int) -> torch.Tensor:
    passes = max(0, int(passes))
    if passes <= 0 or motion.shape[1] <= 2:
        return motion
    kernel = motion.new_tensor([1.0, 2.0, 1.0], dtype=motion.dtype)
    kernel = kernel / kernel.sum()
    y = motion.transpose(1, 2)
    weight = kernel.view(1, 1, -1).repeat(y.shape[1], 1, 1)
    for _ in range(passes):
        y = F.pad(y, (1, 1), mode="replicate")
        y = F.conv1d(y, weight, groups=y.shape[1])
    return y.transpose(1, 2)


def replace_feature_indices(base: torch.Tensor, indices: List[int], values: torch.Tensor) -> torch.Tensor:
    if not indices:
        return base
    index = torch.as_tensor(indices, device=base.device, dtype=torch.long)
    if values.shape[:-1] != base.shape[:-1] or values.shape[-1] != index.numel():
        raise RuntimeError("replace_feature_indices got mismatched shapes")
    keep_mask = torch.ones(base.shape[-1], device=base.device, dtype=base.dtype)
    keep_mask.index_fill_(0, index, 0.0)
    out = base * keep_mask.view(*([1] * (base.ndim - 1)), -1)
    add = torch.zeros_like(base)
    add[..., index] = values
    return out + add


def sample_timesteps(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
    fixed_t: Optional[float],
    t_min: float,
    t_max: float,
    target_mode: str,
    sample_mode: str,
    low_bias_power: float,
    low_bias_mix: float,
) -> torch.Tensor:
    if fixed_t is not None:
        return torch.full((batch_size,), float(fixed_t), device=device, dtype=dtype)

    if sample_mode == "auto":
        sample_mode = "low_mix" if target_mode == "x0" else "uniform"

    u = torch.rand(batch_size, generator=generator, device=device, dtype=dtype)
    power = max(float(low_bias_power), 1e-3)
    low_bias_mix = min(max(float(low_bias_mix), 0.0), 1.0)

    if sample_mode == "uniform":
        base = u
    elif sample_mode == "low_bias":
        base = u.pow(power)
    elif sample_mode == "low_mix":
        use_low = torch.rand(batch_size, generator=generator, device=device, dtype=dtype) < low_bias_mix
        base = torch.where(use_low, u.pow(power), u)
    else:
        raise ValueError(f"unknown t sampling mode: {sample_mode}")

    return t_min + (t_max - t_min) * base


# def make_noisy_batch(
#     x0: torch.Tensor,
#     *,
#     target_mode: str,
#     generator: Optional[torch.Generator] = None,
# ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     z = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
#     t = torch.rand(x0.shape[0], generator=generator, device=x0.device, dtype=x0.dtype)
#     x_t = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * z
#     if target_mode == "x0":
#         target = x0
#     else:
#         target = z - x0
#     return x_t, t, z, target

def make_noisy_batch(
    x0: torch.Tensor,
    *,
    target_mode: str,
    generator: Optional[torch.Generator] = None,
    fixed_t: Optional[float] = None,
    t_min: float = 0.0,
    t_max: float = 1.0,
    t_sample_mode: str = "auto",
    t_low_bias_power: float = 2.0,
    t_low_bias_mix: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
    t = sample_timesteps(
        x0.shape[0],
        device=x0.device,
        dtype=x0.dtype,
        generator=generator,
        fixed_t=fixed_t,
        t_min=t_min,
        t_max=t_max,
        target_mode=target_mode,
        sample_mode=t_sample_mode,
        low_bias_power=t_low_bias_power,
        low_bias_mix=t_low_bias_mix,
    )

    x_t = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * z
    if target_mode == "x0":
        target = x0
    else:
        target = z - x0
    return x_t, t, z, target

def model_output_to_velocity(
    x_t: torch.Tensor,
    t: torch.Tensor,
    model_out: torch.Tensor,
    *,
    target_mode: str,
) -> torch.Tensor:
    if target_mode == "velocity":
        return model_out
    t_safe = t[:, None, None].clamp_min(1e-3)
    return (x_t - model_out) / t_safe


def model_output_to_x0(
    x_t: torch.Tensor,
    t: torch.Tensor,
    model_out: torch.Tensor,
    *,
    target_mode: str,
) -> torch.Tensor:
    if target_mode == "x0":
        return model_out
    return x_t - t[:, None, None] * model_out


def weighted_feature_mse(pred: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    per_sample = (pred - target).pow(2).flatten(1).mean(dim=1)
    weighted = per_sample * sample_weight
    return weighted.mean()


def temporal_smoothness_losses(
    pred_x0: torch.Tensor,
    target_x0: torch.Tensor,
    sample_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred_x0.shape[1] <= 1:
        zero = pred_x0.new_zeros(())
        return zero, zero
    pred_vel = pred_x0[:, 1:] - pred_x0[:, :-1]
    target_vel = target_x0[:, 1:] - target_x0[:, :-1]
    vel_loss = weighted_feature_mse(pred_vel, target_vel, sample_weight)
    if pred_vel.shape[1] <= 1:
        return vel_loss, pred_x0.new_zeros(())
    pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
    target_acc = target_vel[:, 1:] - target_vel[:, :-1]
    acc_loss = weighted_feature_mse(pred_acc, target_acc, sample_weight)
    return vel_loss, acc_loss


def root_velocity_loss(
    pred_x0: torch.Tensor,
    target_x0: torch.Tensor,
    root_indices: List[int],
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    if pred_x0.shape[1] <= 1:
        return pred_x0.new_zeros(())
    root_idx = torch.as_tensor(root_indices, device=pred_x0.device, dtype=torch.long)
    pred_root = pred_x0.index_select(dim=-1, index=root_idx)
    target_root = target_x0.index_select(dim=-1, index=root_idx)
    pred_step = pred_root[:, 1:] - pred_root[:, :-1]
    target_step = target_root[:, 1:] - target_root[:, :-1]
    return weighted_feature_mse(pred_step, target_step, sample_weight)


def root_acceleration_loss(
    pred_x0: torch.Tensor,
    target_x0: torch.Tensor,
    root_indices: List[int],
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    if pred_x0.shape[1] <= 2:
        return pred_x0.new_zeros(())
    root_idx = torch.as_tensor(root_indices, device=pred_x0.device, dtype=torch.long)
    pred_root = pred_x0.index_select(dim=-1, index=root_idx)
    target_root = target_x0.index_select(dim=-1, index=root_idx)
    pred_vel = pred_root[:, 1:] - pred_root[:, :-1]
    target_vel = target_root[:, 1:] - target_root[:, :-1]
    pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
    target_acc = target_vel[:, 1:] - target_vel[:, :-1]
    return weighted_feature_mse(pred_acc, target_acc, sample_weight)


def rotation_acceleration_loss(
    pred_x0: torch.Tensor,
    target_x0: torch.Tensor,
    rot6d_start: int,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    if pred_x0.shape[1] <= 2 or pred_x0.shape[-1] <= rot6d_start:
        return pred_x0.new_zeros(())
    pred_rot = canonicalize_rot6d_block(pred_x0[:, :, rot6d_start:])
    target_rot = canonicalize_rot6d_block(target_x0[:, :, rot6d_start:])
    pred_vel = pred_rot[:, 1:] - pred_rot[:, :-1]
    target_vel = target_rot[:, 1:] - target_rot[:, :-1]
    pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
    target_acc = target_vel[:, 1:] - target_vel[:, :-1]
    return weighted_feature_mse(pred_acc, target_acc, sample_weight)


def rotation_velocity_loss(
    pred_x0: torch.Tensor,
    target_x0: torch.Tensor,
    rot6d_start: int,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    if pred_x0.shape[1] <= 1 or pred_x0.shape[-1] <= rot6d_start:
        return pred_x0.new_zeros(())
    pred_rot = canonicalize_rot6d_block(pred_x0[:, :, rot6d_start:])
    target_rot = canonicalize_rot6d_block(target_x0[:, :, rot6d_start:])
    pred_vel = pred_rot[:, 1:] - pred_rot[:, :-1]
    target_vel = target_rot[:, 1:] - target_rot[:, :-1]
    return weighted_feature_mse(pred_vel, target_vel, sample_weight)


def root_position_loss(
    pred_x0: torch.Tensor,
    target_x0: torch.Tensor,
    root_indices: List[int],
) -> torch.Tensor:
    root_idx = torch.as_tensor(root_indices, device=pred_x0.device, dtype=torch.long)
    pred_root = pred_x0.index_select(dim=-1, index=root_idx)
    target_root = target_x0.index_select(dim=-1, index=root_idx)
    return F.mse_loss(pred_root, target_root)


def build_root_velocity_reconstruction_spec(
    motion_contract: Dict[str, Any],
    *,
    root_relative_first_frame: bool,
) -> Optional[RootVelocityReconstructionSpec]:
    layout_meta = dict(motion_contract.get("layout_meta", {}))
    root_idx = list(layout_meta.get("root_pos_indices") or [])
    local_vel_xz = layout_meta.get("local_vel_xz_indices")
    yaw_index = layout_meta.get("yaw_index")
    yaw_vel_index = layout_meta.get("yaw_vel_index")
    if len(root_idx) < 3 or local_vel_xz is None or yaw_index is None or yaw_vel_index is None:
        return None
    return RootVelocityReconstructionSpec(
        root_x_index=int(root_idx[0]),
        root_y_index=int(root_idx[1]),
        root_z_index=int(root_idx[2]),
        yaw_index=int(yaw_index),
        local_vel_x_index=int(local_vel_xz[0]),
        local_vel_z_index=int(local_vel_xz[1]),
        yaw_vel_index=int(yaw_vel_index),
        fps=float(motion_contract["fps"]),
        root_relative_first_frame=bool(root_relative_first_frame),
    )


def reconstruct_root_from_velocity_norm(
    motion_norm: torch.Tensor,
    *,
    anchor_norm: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    spec: Optional[RootVelocityReconstructionSpec],
) -> torch.Tensor:
    if spec is None or motion_norm.shape[1] <= 1:
        return motion_norm

    if anchor_norm.ndim == 2:
        anchor_norm = anchor_norm.unsqueeze(0)
    if anchor_norm.shape[0] != motion_norm.shape[0]:
        if anchor_norm.shape[0] == 1:
            anchor_norm = anchor_norm.expand(motion_norm.shape[0], -1, -1)
        else:
            raise RuntimeError("anchor_norm batch size does not match motion batch size")

    out = motion_norm.clone()
    dt = 1.0 / max(float(spec.fps), 1e-6)

    x_idx = int(spec.root_x_index)
    z_idx = int(spec.root_z_index)
    yaw_idx = int(spec.yaw_index)
    lvx_idx = int(spec.local_vel_x_index)
    lvz_idx = int(spec.local_vel_z_index)
    yv_idx = int(spec.yaw_vel_index)

    mean_x = mean[x_idx]
    mean_z = mean[z_idx]
    mean_yaw = mean[yaw_idx]
    mean_lvx = mean[lvx_idx]
    mean_lvz = mean[lvz_idx]
    mean_yv = mean[yv_idx]

    std_x = std[x_idx].clamp_min(1e-6)
    std_z = std[z_idx].clamp_min(1e-6)
    std_yaw = std[yaw_idx].clamp_min(1e-6)
    std_lvx = std[lvx_idx].clamp_min(1e-6)
    std_lvz = std[lvz_idx].clamp_min(1e-6)
    std_yv = std[yv_idx].clamp_min(1e-6)

    # de-normalize the quantities involved in root reconstruction
    yaw0 = anchor_norm[:, 0, yaw_idx] * std_yaw + mean_yaw
    local_vx = motion_norm[:, :, lvx_idx] * std_lvx + mean_lvx
    local_vz = motion_norm[:, :, lvz_idx] * std_lvz + mean_lvz
    yaw_vel = motion_norm[:, :, yv_idx] * std_yv + mean_yv

    T = motion_norm.shape[1]

    yaw_abs = torch.zeros_like(motion_norm[:, :, yaw_idx])
    yaw_abs[:, 0] = yaw0

    x_track = torch.zeros_like(motion_norm[:, :, x_idx])
    z_track = torch.zeros_like(motion_norm[:, :, z_idx])

    if spec.root_relative_first_frame:
        x_track[:, 0] = 0.0
        z_track[:, 0] = 0.0
    else:
        x_track[:, 0] = anchor_norm[:, 0, x_idx] * std_x + mean_x
        z_track[:, 0] = anchor_norm[:, 0, z_idx] * std_z + mean_z

    # Integrate root translation with midpoint yaw to reduce zig-zag jitter.
    # Crucially, the step from (t-1) -> t uses yaw_vel[t-1] and local_v[t-1].
    for t in range(1, T):
        yaw_prev = yaw_abs[:, t - 1]
        yaw_delta = yaw_vel[:, t - 1] * dt
        yaw_curr = yaw_prev + yaw_delta
        yaw_mid = yaw_prev + 0.5 * yaw_delta

        c = torch.cos(yaw_mid)
        s = torch.sin(yaw_mid)

        dx = (c * local_vx[:, t - 1] - s * local_vz[:, t - 1]) * dt
        dz = (s * local_vx[:, t - 1] + c * local_vz[:, t - 1]) * dt

        x_track[:, t] = x_track[:, t - 1] + dx
        z_track[:, t] = z_track[:, t - 1] + dz
        yaw_abs[:, t] = yaw_curr

    if spec.root_relative_first_frame:
        # build_fixed_clips() makes root position relative as (clip - anchor) / std
        out[:, :, x_idx] = x_track / std_x
        out[:, :, z_idx] = z_track / std_z
    else:
        out[:, :, x_idx] = (x_track - mean_x) / std_x
        out[:, :, z_idx] = (z_track - mean_z) / std_z

    out[:, :, yaw_idx] = (yaw_abs - mean_yaw) / std_yaw
    return out


def postprocess_motion_prediction_norm(
    motion_norm: torch.Tensor,
    *,
    rot6d_start: int,
    smooth_passes: int,
    anchor_norm: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    root_velocity_spec: Optional[RootVelocityReconstructionSpec],
) -> torch.Tensor:
    out = project_motion_rot6d(motion_norm, rot6d_start)

    smooth_passes = max(0, int(smooth_passes))
    if smooth_passes > 0 and out.shape[1] > 2:
        if root_velocity_spec is not None:
            driver_idx = [
                int(root_velocity_spec.local_vel_x_index),
                int(root_velocity_spec.local_vel_z_index),
                int(root_velocity_spec.yaw_vel_index),
            ]
            driver_smooth = temporal_binomial_smooth(out[:, :, driver_idx], smooth_passes)
            out = replace_feature_indices(out, driver_idx, driver_smooth)
        if out.shape[-1] > rot6d_start:
            rot_smooth = temporal_binomial_smooth(out[:, :, rot6d_start:], smooth_passes)
            out = torch.cat([out[:, :, :rot6d_start], canonicalize_rot6d_block(rot_smooth)], dim=-1)

    out = reconstruct_root_from_velocity_norm(
        out,
        anchor_norm=anchor_norm,
        mean=mean,
        std=std,
        spec=root_velocity_spec,
    )
    return out

def compute_direct_metrics(
    pred_norm: np.ndarray,
    ref_norm: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    root_indices: List[int],
    rot6d_start: int,
    root_anchor: Optional[np.ndarray],
) -> Dict[str, float]:
    pred = denorm_motion(pred_norm, mean=mean, std=std, root_indices=root_indices, root_anchor=root_anchor)
    ref = denorm_motion(ref_norm, mean=mean, std=std, root_indices=root_indices, root_anchor=root_anchor)
    root_idx = np.asarray(root_indices, dtype=np.int64)
    pred_root = pred[:, root_idx]
    ref_root = ref[:, root_idx]
    pred_step = np.diff(pred, axis=0)
    ref_step = np.diff(ref, axis=0)

    def _safe_mean(arr: np.ndarray) -> float:
        if arr.size <= 0:
            return 0.0
        return float(np.mean(arr))

    pred_root_vel = np.diff(pred_root, axis=0)
    ref_root_vel = np.diff(ref_root, axis=0)
    pred_root_step = np.linalg.norm(pred_root_vel, axis=-1)
    ref_root_step = np.linalg.norm(ref_root_vel, axis=-1)
    pred_root_acc = np.linalg.norm(np.diff(pred_root_vel, axis=0), axis=-1)
    ref_root_acc = np.linalg.norm(np.diff(ref_root_vel, axis=0), axis=-1)

    pred_rot_vel = np.diff(pred[:, rot6d_start:], axis=0)
    ref_rot_vel = np.diff(ref[:, rot6d_start:], axis=0)
    pred_rot_step = _safe_mean(np.abs(pred_rot_vel))
    ref_rot_step = _safe_mean(np.abs(ref_rot_vel))
    pred_rot_acc = _safe_mean(np.abs(np.diff(pred_rot_vel, axis=0)))
    ref_rot_acc = _safe_mean(np.abs(np.diff(ref_rot_vel, axis=0)))
    eps = 1e-8
    return {
        "mse_to_ref": float(np.mean((pred_norm - ref_norm) ** 2)),
        "root_err_mean": float(np.mean(np.linalg.norm(pred_root - ref_root, axis=-1))),
        "root_err_final": float(np.linalg.norm(pred_root[-1] - ref_root[-1])),
        "frame_diff_ratio": float(np.mean(np.abs(pred_step)) / (np.mean(np.abs(ref_step)) + eps)),
        "root_step_ratio": float(pred_root_step.mean() / (ref_root_step.mean() + eps)),
        "rot_step_ratio": float(pred_rot_step / (ref_rot_step + eps)),
        "root_acc_ratio": float(_safe_mean(pred_root_acc) / (_safe_mean(ref_root_acc) + eps)),
        "rot_acc_ratio": float(pred_rot_acc / (ref_rot_acc + eps)),
        "pred_std": float(np.std(pred)),
        "ref_std": float(np.std(ref)),
    }


def encode_clip_text_condition(
    rec: ClipRecord,
    *,
    text_encoder: Optional[FrozenBertTextEncoder],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if text_encoder is not None:
        if rec.text_ids is None or rec.text_mask is None:
            raise RuntimeError("clip is missing text_ids/text_mask but BERT text conditioning was requested")
        return text_encoder.encode(
            torch.from_numpy(rec.text_ids).long(),
            torch.from_numpy(rec.text_mask).long(),
            device,
        )[0]
    if rec.global_text is not None:
        return torch.from_numpy(rec.global_text).float().to(device)
    return None


def encode_clip_word_condition(
    rec: ClipRecord,
    *,
    text_encoder: Optional[FrozenBertTextEncoder],
    clip_len: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if text_encoder is None:
        return None
    if rec.word_times is None:
        raise RuntimeError("clip is missing word_times but BERT word conditioning was requested")
    word_times = torch.from_numpy(rec.word_times).float().unsqueeze(0)
    word_mask = torch.ones((1, int(rec.word_times.shape[0])), dtype=torch.bool)
    return text_encoder.encode_aligned_words([rec.text], word_times, word_mask, clip_len, device)[0]


def evaluate_free_preview(
    model: TinyFlowTransformer,
    *,
    clips: List[ClipRecord],
    text_encoder: Optional[FrozenBertTextEncoder],
    mean_torch: torch.Tensor,
    std_torch: torch.Tensor,
    root_velocity_spec: Optional[RootVelocityReconstructionSpec],
    mean: np.ndarray,
    std: np.ndarray,
    root_indices: List[int],
    rot6d_start: int,
    preview_index: int,
    preview_seed: int,
    preview_num_steps: int,
    preview_solver: str,
    target_mode: str,
    pred_smooth_passes: int,
    device: torch.device,
) -> Dict[str, Any]:
    if preview_index < 0 or preview_index >= len(clips):
        raise IndexError(f"preview_index={preview_index} out of range for {len(clips)} clips")
    ref = clips[preview_index]
    global_text = None
    word_cond = None
    if model.text_dim > 0:
        global_text = encode_clip_text_condition(ref, text_encoder=text_encoder, device=device)
        if global_text is None:
            raise RuntimeError("clip is missing text conditioning but model expects it")
    if model.word_cond_dim > 0:
        word_cond = encode_clip_word_condition(ref, text_encoder=text_encoder, clip_len=ref.motion.shape[0], device=device)
    pred_norm = sample_motion(
        model,
        clip_len=ref.motion.shape[0],
        motion_dim=ref.motion.shape[1],
        num_steps=preview_num_steps,
        solver=preview_solver,
        target_mode=target_mode,
        device=device,
        seed=preview_seed,
        global_text=global_text,
        word_cond=word_cond,
        anchor_motion=torch.from_numpy(ref.motion).float(),
        mean=mean_torch,
        std=std_torch,
        root_velocity_spec=root_velocity_spec,
        rot6d_start=rot6d_start,
        pred_smooth_passes=pred_smooth_passes,
    ).detach().cpu().numpy()
    direct = compute_direct_metrics(
        pred_norm,
        ref.motion,
        mean=mean,
        std=std,
        root_indices=root_indices,
        rot6d_start=rot6d_start,
        root_anchor=ref.root_anchor,
    )
    all_mse = [float(np.mean((pred_norm - clip.motion) ** 2)) for clip in clips]
    nearest_idx = int(np.argmin(np.asarray(all_mse, dtype=np.float64)))
    nearest = clips[nearest_idx]
    return {
        "mode": "free",
        "preview_index": int(preview_index),
        "preview_segment_id": ref.segment_id,
        "nearest_index": nearest_idx,
        "nearest_segment_id": nearest.segment_id,
        "nearest_mse": float(all_mse[nearest_idx]),
        **direct,
    }


@torch.no_grad()
def teacher_forced_reconstruct(
    model: TinyFlowTransformer,
    *,
    ref_motion: torch.Tensor,
    global_text: Optional[torch.Tensor],
    word_cond: Optional[torch.Tensor],
    mean: torch.Tensor,
    std: torch.Tensor,
    root_velocity_spec: Optional[RootVelocityReconstructionSpec],
    rot6d_start: int,
    pred_smooth_passes: int,
    target_mode: str,
    device: torch.device,
    seed: int,
    fixed_t: Optional[float] = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    x0 = ref_motion.unsqueeze(0).to(device)
    x_t, t, z, target = make_noisy_batch(
        x0,
        target_mode=target_mode,
        generator=generator,
        fixed_t=fixed_t,
        t_sample_mode="uniform",
    )
    cond = None
    if global_text is not None:
        cond = global_text.to(device)
        if cond.ndim == 1:
            cond = cond.unsqueeze(0)
    word = None
    if word_cond is not None:
        word = word_cond.to(device)
        if word.ndim == 2:
            word = word.unsqueeze(0)
    model_out = model(x_t, t, cond, word)
    x0_hat = model_output_to_x0(x_t, t, model_out, target_mode=target_mode)
    x0_hat = postprocess_motion_prediction_norm(
        x0_hat,
        rot6d_start=rot6d_start,
        smooth_passes=pred_smooth_passes,
        anchor_norm=x0,
        mean=mean,
        std=std,
        root_velocity_spec=root_velocity_spec,
    )
    if target_mode == "x0":
        target_mse = F.mse_loss(x0_hat, x0)
    else:
        target_mse = F.mse_loss(model_out, z - x0)
    return x0_hat[0].detach().cpu(), {
        "teacher_t": float(t[0].item()),
        "teacher_target_mse": float(target_mse.item()),
    }


def evaluate_teacher_preview(
    model: TinyFlowTransformer,
    *,
    clips: List[ClipRecord],
    text_encoder: Optional[FrozenBertTextEncoder],
    mean_torch: torch.Tensor,
    std_torch: torch.Tensor,
    root_velocity_spec: Optional[RootVelocityReconstructionSpec],
    mean: np.ndarray,
    std: np.ndarray,
    root_indices: List[int],
    rot6d_start: int,
    preview_index: int,
    preview_seed: int,
    target_mode: str,
    pred_smooth_passes: int,
    device: torch.device,
) -> Dict[str, Any]:
    if preview_index < 0 or preview_index >= len(clips):
        raise IndexError(f"preview_index={preview_index} out of range for {len(clips)} clips")
    ref = clips[preview_index]
    global_text = None
    word_cond = None
    if model.text_dim > 0:
        global_text = encode_clip_text_condition(ref, text_encoder=text_encoder, device=device).cpu()
    if model.word_cond_dim > 0:
        word_cond = encode_clip_word_condition(ref, text_encoder=text_encoder, clip_len=ref.motion.shape[0], device=device).cpu()
    pred_norm, aux = teacher_forced_reconstruct(
        model,
        ref_motion=torch.from_numpy(ref.motion).float(),
        global_text=global_text,
        word_cond=word_cond,
        mean=mean_torch,
        std=std_torch,
        root_velocity_spec=root_velocity_spec,
        rot6d_start=rot6d_start,
        pred_smooth_passes=pred_smooth_passes,
        target_mode=target_mode,
        device=device,
        seed=preview_seed,
    )
    direct = compute_direct_metrics(
        pred_norm.numpy(),
        ref.motion,
        mean=mean,
        std=std,
        root_indices=root_indices,
        rot6d_start=rot6d_start,
        root_anchor=ref.root_anchor,
    )
    return {
        "mode": "teacher",
        "preview_index": int(preview_index),
        "preview_segment_id": ref.segment_id,
        **aux,
        **direct,
    }


def train_one_epoch(
    model: TinyFlowTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    text_encoder: Optional[FrozenBertTextEncoder],
    mean: torch.Tensor,
    std: torch.Tensor,
    root_velocity_spec: Optional[RootVelocityReconstructionSpec],
    device: torch.device,
    target_mode: str,
    root_indices: List[int],
    rot6d_start: int,
    lambda_vel: float,
    lambda_acc: float,
    lambda_rot_vel: float,
    lambda_rot_acc: float,
    lambda_root_vel: float,
    lambda_root_acc: float,
    lambda_root_pos: float,
    pred_smooth_passes: int,
    smooth_weight_floor: float,
    t_sample_mode: str,
    t_low_bias_power: float,
    t_low_bias_mix: float,
) -> tuple[Dict[str, float], int]:
    model.train()
    totals = {
        "loss": 0.0,
        "loss_target": 0.0,
        "loss_vel": 0.0,
        "loss_acc": 0.0,
        "loss_rot_vel": 0.0,
        "loss_rot_acc": 0.0,
        "loss_root_vel": 0.0,
        "loss_root_acc": 0.0,
        "loss_root_pos": 0.0,
    }
    count = 0
    steps = 0
    for batch in loader:
        x0 = batch["motion"].to(device)
        global_text = None
        word_cond = None
        if text_encoder is not None and model.text_dim > 0:
            global_text = text_encoder.encode(batch["text_ids"], batch["text_mask"], device)
        elif model.text_dim > 0:
            global_text = batch["global_text"].to(device)
        if text_encoder is not None and model.word_cond_dim > 0:
            word_cond = text_encoder.encode_aligned_words(
                list(batch["text"]),
                batch["word_times"],
                batch["word_mask"],
                int(x0.shape[1]),
                device,
            )
        x_t, t, _z, target = make_noisy_batch(
            x0,
            target_mode=target_mode,
            t_sample_mode=t_sample_mode,
            t_low_bias_power=t_low_bias_power,
            t_low_bias_mix=t_low_bias_mix,
        )
        pred = model(x_t, t, global_text, word_cond)
        pred_x0 = model_output_to_x0(x_t, t, pred, target_mode=target_mode)
        if target_mode == "x0":
            pred_x0 = postprocess_motion_prediction_norm(
                pred_x0,
                rot6d_start=rot6d_start,
                smooth_passes=pred_smooth_passes,
                anchor_norm=x0,
                mean=mean,
                std=std,
                root_velocity_spec=root_velocity_spec,
            )
        if target_mode == "x0":
            loss_target = F.mse_loss(pred_x0, x0)
        else:
            loss_target = F.mse_loss(pred, target)
        smooth_weight_floor = min(max(float(smooth_weight_floor), 0.0), 1.0)
        smooth_weight = smooth_weight_floor + (1.0 - smooth_weight_floor) * (1.0 - t).pow(2)
        loss_vel, loss_acc = temporal_smoothness_losses(pred_x0, x0, smooth_weight)
        loss_rot_vel = rotation_velocity_loss(pred_x0, x0, rot6d_start, smooth_weight)
        loss_rot_acc = rotation_acceleration_loss(pred_x0, x0, rot6d_start, smooth_weight)
        loss_root_vel = root_velocity_loss(pred_x0, x0, root_indices, smooth_weight)
        loss_root_acc = root_acceleration_loss(pred_x0, x0, root_indices, smooth_weight)
        loss_root_pos = root_position_loss(pred_x0, x0, root_indices)
        loss = (
            loss_target
            + float(lambda_vel) * loss_vel
            + float(lambda_acc) * loss_acc
            + float(lambda_rot_vel) * loss_rot_vel
            + float(lambda_rot_acc) * loss_rot_acc
            + float(lambda_root_vel) * loss_root_vel
            + float(lambda_root_acc) * loss_root_acc
            + float(lambda_root_pos) * loss_root_pos
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        totals["loss"] += float(loss.item())
        totals["loss_target"] += float(loss_target.item())
        totals["loss_vel"] += float(loss_vel.item())
        totals["loss_acc"] += float(loss_acc.item())
        totals["loss_rot_vel"] += float(loss_rot_vel.item())
        totals["loss_rot_acc"] += float(loss_rot_acc.item())
        totals["loss_root_vel"] += float(loss_root_vel.item())
        totals["loss_root_acc"] += float(loss_root_acc.item())
        totals["loss_root_pos"] += float(loss_root_pos.item())
        count += 1
        steps += 1
    denom = max(1, count)
    return {key: value / denom for key, value in totals.items()}, steps


@torch.no_grad()
def sample_motion(
    model: TinyFlowTransformer,
    *,
    clip_len: int,
    motion_dim: int,
    num_steps: int,
    solver: str,
    target_mode: str,
    device: torch.device,
    seed: int,
    global_text: Optional[torch.Tensor],
    word_cond: Optional[torch.Tensor],
    anchor_motion: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    root_velocity_spec: Optional[RootVelocityReconstructionSpec],
    rot6d_start: int,
    pred_smooth_passes: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    x = torch.randn((1, clip_len, motion_dim), generator=generator, device=device)

    cond = None
    if global_text is not None:
        cond = global_text.to(device)
        if cond.ndim == 1:
            cond = cond.unsqueeze(0)

    word = None
    if word_cond is not None:
        word = word_cond.to(device)
        if word.ndim == 2:
            word = word.unsqueeze(0)

    anchor = anchor_motion.to(device)
    if anchor.ndim == 2:
        anchor = anchor.unsqueeze(0)

    num_steps = max(1, int(num_steps))

    # For x0-prediction, do not integrate all the way to t=0.
    # Stop at a small positive t and do one final x0 projection.
    if target_mode == "x0":
        t_min = 0.02
    else:
        t_min = 0.0

    # Heun的第二步在 t_next 低于这个阈值时退化成Euler，避免 (x_t - x0) / t_next 爆炸
    HEUN_T_MIN = 0.05

    schedule = torch.linspace(1.0, t_min, num_steps + 1, device=device, dtype=x.dtype)

    for idx in range(num_steps):
        t_cur = schedule[idx].expand(1)
        t_next = schedule[idx + 1].expand(1)

        model_out = model(x, t_cur, cond, word)
        if target_mode == "x0":
            model_out = postprocess_motion_prediction_norm(
                model_out,
                rot6d_start=rot6d_start,
                smooth_passes=pred_smooth_passes,
                anchor_norm=anchor,
                mean=mean,
                std=std,
                root_velocity_spec=root_velocity_spec,
            )

        v = model_output_to_velocity(x, t_cur, model_out, target_mode=target_mode)
        dt = float((schedule[idx + 1] - schedule[idx]).item())

        if solver == "heun":
            x_euler = x + dt * v

            # ↓↓↓ 核心改动：t_next 太小时 Heun 第二步的除法会爆炸，退化成 Euler ↓↓↓
            if target_mode == "x0" and float(t_next[0].item()) < HEUN_T_MIN:
                x = x_euler
            else:
                model_out_next = model(x_euler, t_next, cond, word)
                if target_mode == "x0":
                    model_out_next = postprocess_motion_prediction_norm(
                        model_out_next,
                        rot6d_start=rot6d_start,
                        smooth_passes=pred_smooth_passes,
                        anchor_norm=anchor,
                        mean=mean,
                        std=std,
                        root_velocity_spec=root_velocity_spec,
                    )

                v_next = model_output_to_velocity(
                    x_euler,
                    t_next,
                    model_out_next,
                    target_mode=target_mode,
                )
                x = x + dt * 0.5 * (v + v_next)

        elif solver == "euler":
            x = x + dt * v

        else:
            raise ValueError(f"unknown solver: {solver}")

    # For x0-prediction, end with one explicit projection from x_{t_min} -> x0.
    if target_mode == "x0":
        t_final = schedule[-1].expand(1)
        x0_final = model(x, t_final, cond, word)
        x0_final = postprocess_motion_prediction_norm(
            x0_final,
            rot6d_start=rot6d_start,
            smooth_passes=pred_smooth_passes,
            anchor_norm=anchor,
            mean=mean,
            std=std,
            root_velocity_spec=root_velocity_spec,
        )
        return x0_final[0]

    # velocity mode keeps the old behavior
    final = reconstruct_root_from_velocity_norm(
        x,
        anchor_norm=anchor,
        mean=mean,
        std=std,
        spec=root_velocity_spec,
    )
    return final[0]

def cmd_inspect(args: argparse.Namespace) -> None:
    payload = load_cache(args.cache)
    motion_contract = dict(payload["motion_contract"])
    root_indices = list(motion_contract["layout_meta"]["root_pos_indices"])
    clips = build_fixed_clips(
        payload,
        clip_len=args.clip_len,
        overfit_n=None,
        root_indices=root_indices,
        root_relative_first_frame=bool(args.root_relative_first_frame),
    )
    print(json.dumps({"cache": str(Path(args.cache).resolve()), "clip_len": args.clip_len, "clips": len(clips)}, ensure_ascii=False))
    for idx, rec in enumerate(clips[: args.limit]):
        print(
            json.dumps(
                {
                    "index": idx,
                    "segment_id": rec.segment_id,
                    "src_bvh": rec.src_bvh,
                    "start": rec.start,
                    "end": rec.end,
                    "motion_shape": list(rec.motion.shape),
                },
                ensure_ascii=False,
            )
        )


def cmd_train(args: argparse.Namespace) -> None:
    payload = load_cache(args.cache)
    motion_contract = dict(payload["motion_contract"])
    root_indices = list(motion_contract["layout_meta"]["root_pos_indices"])
    rot6d_start = int(motion_contract["rot6d_start"])
    clips = build_fixed_clips(
        payload,
        clip_len=args.clip_len,
        overfit_n=args.overfit_n,
        root_indices=root_indices,
        root_relative_first_frame=bool(args.root_relative_first_frame),
    )
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    ds = FixedClipDataset(clips)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    mean_torch = torch.from_numpy(mean).float().to(device)
    std_torch = torch.from_numpy(std).float().to(device)
    root_velocity_spec = (
        build_root_velocity_reconstruction_spec(
            motion_contract,
            root_relative_first_frame=bool(args.root_relative_first_frame),
        )
        if bool(args.reconstruct_root_from_vel)
        else None
    )
    if bool(args.reconstruct_root_from_vel) and str(args.target_mode) != "x0":
        raise RuntimeError("reconstruct_root_from_vel currently requires --target_mode x0")
    if bool(args.use_global_text) and bool(args.use_bert_text):
        raise RuntimeError("use_global_text and use_bert_text are mutually exclusive")
    if bool(args.use_bert_words) and not (bool(args.use_bert_text) or bool(args.use_global_text)):
        # Word-aligned semantics are still useful without a global condition; keep this allowed.
        pass
    text_encoder = None
    text_dim = 0
    word_cond_dim = 0
    if bool(args.use_bert_text) or bool(args.use_bert_words):
        if ds.text_token_len <= 0:
            raise RuntimeError("BERT text conditioning requires cache clips to contain text_ids/text_mask")
        text_encoder = FrozenBertTextEncoder(args.bert_path, float(payload["fps"])).to(device)
        if bool(args.use_bert_text):
            text_dim = int(text_encoder.hidden_size)
        if bool(args.use_bert_words):
            word_cond_dim = int(text_encoder.hidden_size + 5)
    elif bool(args.use_global_text):
        text_dim = ds.text_dim
        if text_dim <= 0:
            raise RuntimeError("use_global_text=True but cache clips do not contain global_text")
    model = TinyFlowTransformer(
        motion_dim=clips[0].motion.shape[1],
        clip_len=args.clip_len,
        text_dim=text_dim,
        word_cond_dim=word_cond_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        losses, step_count = train_one_epoch(
            model,
            loader,
            optimizer,
            text_encoder=text_encoder,
            mean=mean_torch,
            std=std_torch,
            root_velocity_spec=root_velocity_spec,
            device=device,
            target_mode=str(args.target_mode),
            root_indices=root_indices,
            rot6d_start=rot6d_start,
            lambda_vel=float(args.lambda_vel),
            lambda_acc=float(args.lambda_acc),
            lambda_rot_vel=float(args.lambda_rot_vel),
            lambda_rot_acc=float(args.lambda_rot_acc),
            lambda_root_vel=float(args.lambda_root_vel),
            lambda_root_acc=float(args.lambda_root_acc),
            lambda_root_pos=float(args.lambda_root_pos),
            pred_smooth_passes=int(args.pred_smooth_passes),
            smooth_weight_floor=float(args.smooth_weight_floor),
            t_sample_mode=str(args.t_sample_mode),
            t_low_bias_power=float(args.t_low_bias_power),
            t_low_bias_mix=float(args.t_low_bias_mix),
        )
        global_step += int(step_count)
        log: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "loss": float(losses["loss"]),
            "loss_target": float(losses["loss_target"]),
            "loss_vel": float(losses["loss_vel"]),
            "loss_acc": float(losses["loss_acc"]),
            "loss_rot_vel": float(losses["loss_rot_vel"]),
            "loss_rot_acc": float(losses["loss_rot_acc"]),
            "loss_root_vel": float(losses["loss_root_vel"]),
            "loss_root_acc": float(losses["loss_root_acc"]),
            "loss_root_pos": float(losses["loss_root_pos"]),
            "clips": len(clips),
            "target_mode": str(args.target_mode),
        }
        if int(args.preview_every) > 0 and (epoch % int(args.preview_every) == 0 or epoch == args.epochs):
            if str(args.preview_mode) in {"free", "both"}:
                log["preview_free"] = evaluate_free_preview(
                    model,
                    clips=clips,
                    text_encoder=text_encoder,
                    mean_torch=mean_torch,
                    std_torch=std_torch,
                    root_velocity_spec=root_velocity_spec,
                    mean=mean,
                    std=std,
                    root_indices=root_indices,
                    rot6d_start=rot6d_start,
                    preview_index=int(args.preview_index),
                    preview_seed=int(args.preview_seed),
                    preview_num_steps=int(args.preview_num_steps),
                    preview_solver=str(args.preview_solver),
                    target_mode=str(args.target_mode),
                    pred_smooth_passes=int(args.pred_smooth_passes),
                    device=device,
                )
            if str(args.preview_mode) in {"teacher", "both"}:
                log["preview_teacher"] = evaluate_teacher_preview(
                    model,
                    clips=clips,
                    text_encoder=text_encoder,
                    mean_torch=mean_torch,
                    std_torch=std_torch,
                    root_velocity_spec=root_velocity_spec,
                    mean=mean,
                    std=std,
                    root_indices=root_indices,
                    rot6d_start=rot6d_start,
                    preview_index=int(args.preview_index),
                    preview_seed=int(args.preview_seed),
                    target_mode=str(args.target_mode),
                    pred_smooth_passes=int(args.pred_smooth_passes),
                    device=device,
                )
        print(json.dumps(log, ensure_ascii=False))
        payload_to_save = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_spec": {
                "motion_dim": clips[0].motion.shape[1],
                "clip_len": int(args.clip_len),
                "text_dim": int(text_dim),
                "word_cond_dim": int(word_cond_dim),
                "hidden_dim": int(args.hidden_dim),
                "layers": int(args.layers),
                "heads": int(args.heads),
                "dropout": float(args.dropout),
                "target_mode": str(args.target_mode),
                "use_bert_text": bool(args.use_bert_text),
                "use_bert_words": bool(args.use_bert_words),
                "reconstruct_root_from_vel": bool(args.reconstruct_root_from_vel),
                "bert_path": str(Path(args.bert_path).resolve())
                if (bool(args.use_bert_text) or bool(args.use_bert_words))
                else None,
            },
            "train_spec": {
                "cache": str(Path(args.cache).resolve()),
                "clip_len": int(args.clip_len),
                "overfit_n": None if args.overfit_n is None else int(args.overfit_n),
                "target_mode": str(args.target_mode),
                "root_relative_first_frame": bool(args.root_relative_first_frame),
                "use_global_text": bool(args.use_global_text),
                "use_bert_text": bool(args.use_bert_text),
                "use_bert_words": bool(args.use_bert_words),
                "reconstruct_root_from_vel": bool(args.reconstruct_root_from_vel),
                "bert_path": str(Path(args.bert_path).resolve())
                if (bool(args.use_bert_text) or bool(args.use_bert_words))
                else None,
                "lambda_vel": float(args.lambda_vel),
                "lambda_acc": float(args.lambda_acc),
                "lambda_rot_vel": float(args.lambda_rot_vel),
                "lambda_rot_acc": float(args.lambda_rot_acc),
                "lambda_root_vel": float(args.lambda_root_vel),
                "lambda_root_acc": float(args.lambda_root_acc),
                "lambda_root_pos": float(args.lambda_root_pos),
                "pred_smooth_passes": int(args.pred_smooth_passes),
                "smooth_weight_floor": float(args.smooth_weight_floor),
                "t_sample_mode": str(args.t_sample_mode),
                "t_low_bias_power": float(args.t_low_bias_power),
                "t_low_bias_mix": float(args.t_low_bias_mix),
            },
            "loss": float(losses["loss"]),
            "loss_target": float(losses["loss_target"]),
            "loss_vel": float(losses["loss_vel"]),
            "loss_acc": float(losses["loss_acc"]),
            "loss_rot_vel": float(losses["loss_rot_vel"]),
            "loss_rot_acc": float(losses["loss_rot_acc"]),
            "loss_root_vel": float(losses["loss_root_vel"]),
            "loss_root_acc": float(losses["loss_root_acc"]),
            "loss_root_pos": float(losses["loss_root_pos"]),
        }
        if "preview_free" in log:
            payload_to_save["preview_free"] = log["preview_free"]
        if "preview_teacher" in log:
            payload_to_save["preview_teacher"] = log["preview_teacher"]
        torch.save(payload_to_save, save_dir / "last.pt")
        if float(losses["loss"]) < best:
            best = float(losses["loss"])
            torch.save(payload_to_save, save_dir / "best.pt")


def load_model_from_checkpoint(ckpt_path: str | Path, device: torch.device) -> tuple[TinyFlowTransformer, Dict[str, Any]]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    spec = dict(ckpt["model_spec"])
    model = TinyFlowTransformer(
        motion_dim=int(spec["motion_dim"]),
        clip_len=int(spec["clip_len"]),
        text_dim=int(spec.get("text_dim", 0)),
        word_cond_dim=int(spec.get("word_cond_dim", 0)),
        hidden_dim=int(spec["hidden_dim"]),
        layers=int(spec["layers"]),
        heads=int(spec["heads"]),
        dropout=float(spec["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt


def cmd_sample(args: argparse.Namespace) -> None:
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model, ckpt = load_model_from_checkpoint(args.checkpoint, device)
    clip_len = int(ckpt["model_spec"]["clip_len"])
    payload = load_cache(args.cache)
    payload_motion_contract = dict(payload["motion_contract"])
    root_indices = list(payload_motion_contract["layout_meta"]["root_pos_indices"])
    rot6d_start = int(payload_motion_contract["rot6d_start"])
    target_mode = str(ckpt.get("train_spec", {}).get("target_mode", ckpt["model_spec"].get("target_mode", "velocity")))
    root_relative_first_frame = bool(ckpt.get("train_spec", {}).get("root_relative_first_frame", False))
    pred_smooth_passes = (
        int(args.pred_smooth_passes)
        if args.pred_smooth_passes is not None
        else int(ckpt.get("train_spec", {}).get("pred_smooth_passes", 1 if target_mode == "x0" else 0))
    )
    use_global_text = bool(ckpt.get("train_spec", {}).get("use_global_text", False))
    use_bert_text = bool(
        ckpt.get("train_spec", {}).get("use_bert_text", ckpt.get("model_spec", {}).get("use_bert_text", False))
    )
    use_bert_words = bool(
        ckpt.get("train_spec", {}).get("use_bert_words", ckpt.get("model_spec", {}).get("use_bert_words", False))
    )
    reconstruct_root_from_vel = bool(
        ckpt.get("train_spec", {}).get(
            "reconstruct_root_from_vel",
            ckpt.get("model_spec", {}).get("reconstruct_root_from_vel", False),
        )
    )
    bert_path = ckpt.get("train_spec", {}).get("bert_path", ckpt.get("model_spec", {}).get("bert_path", None))
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    mean_torch = torch.from_numpy(mean).float().to(device)
    std_torch = torch.from_numpy(std).float().to(device)
    root_velocity_spec = (
        build_root_velocity_reconstruction_spec(
            payload_motion_contract,
            root_relative_first_frame=root_relative_first_frame,
        )
        if reconstruct_root_from_vel
        else None
    )
    clips = build_fixed_clips(
        payload,
        clip_len=clip_len,
        overfit_n=None,
        root_indices=root_indices,
        root_relative_first_frame=root_relative_first_frame,
    )
    if args.index < 0 or args.index >= len(clips):
        raise IndexError(f"--index {args.index} out of range for {len(clips)} clips")
    rec = clips[args.index]
    text_encoder = None
    if use_bert_text or use_bert_words:
        if not bert_path:
            raise RuntimeError("checkpoint expects BERT text conditioning but does not record bert_path")
        text_encoder = FrozenBertTextEncoder(str(bert_path), float(payload["fps"])).to(device)
    global_text = None
    word_cond = None
    if model.text_dim > 0:
        global_text = encode_clip_text_condition(rec, text_encoder=text_encoder, device=device)
        if global_text is None and use_global_text:
            raise RuntimeError("checkpoint expects global_text conditioning but selected clip has no global_text")
    if model.word_cond_dim > 0:
        word_cond = encode_clip_word_condition(rec, text_encoder=text_encoder, clip_len=clip_len, device=device)
    if str(args.sample_mode) == "teacher":
        pred_norm, teacher_aux = teacher_forced_reconstruct(
            model,
            ref_motion=torch.from_numpy(rec.motion).float(),
            global_text=global_text,
            word_cond=word_cond,
            mean=mean_torch,
            std=std_torch,
            root_velocity_spec=root_velocity_spec,
            rot6d_start=rot6d_start,
            pred_smooth_passes=pred_smooth_passes,
            target_mode=target_mode,
            device=device,
            seed=args.seed,
            fixed_t=args.teacher_t,
        )
        pred_norm = pred_norm.numpy()
    else:
        teacher_aux = None
        pred_norm = sample_motion(
            model,
            clip_len=clip_len,
            motion_dim=int(ckpt["model_spec"]["motion_dim"]),
            num_steps=args.num_steps,
            solver=args.solver,
            target_mode=target_mode,
            device=device,
            seed=args.seed,
            global_text=global_text,
            word_cond=word_cond,
            anchor_motion=torch.from_numpy(rec.motion).float(),
            mean=mean_torch,
            std=std_torch,
            root_velocity_spec=root_velocity_spec,
            rot6d_start=rot6d_start,
            pred_smooth_passes=pred_smooth_passes,
        ).detach().cpu().numpy()
    pred_motion = denorm_motion(
        pred_norm,
        mean=mean,
        std=std,
        root_indices=root_indices,
        root_anchor=rec.root_anchor,
    )
    gt_motion = denorm_motion(
        rec.motion,
        mean=mean,
        std=std,
        root_indices=root_indices,
        root_anchor=rec.root_anchor,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "pred_motion.npy", pred_motion.astype(np.float32))
    np.save(out_dir / "gt_motion.npy", gt_motion.astype(np.float32))

    try:
        from diffusion.latent.bvh import decode_motion_to_bvh, save_bvh_remapped
        from diffusion.latent.contracts import load_cache_contract

        cache_loaded = load_cache_contract(args.cache, require_word_times=False)
        motion_contract = cache_loaded["motion_contract"]
        pred_root, pred_euler = decode_motion_to_bvh(
            torch.from_numpy(pred_motion).float(),
            layout=motion_contract.feature_layout,
            fps=motion_contract.fps,
            motion_dim=motion_contract.motion_dim,
        )
        gt_root, gt_euler = decode_motion_to_bvh(
            torch.from_numpy(gt_motion).float(),
            layout=motion_contract.feature_layout,
            fps=motion_contract.fps,
            motion_dim=motion_contract.motion_dim,
        )
        save_bvh_remapped(pred_root, pred_euler, ref_bvh_path=rec.src_bvh, output_path=str(out_dir / "pred.bvh"), fps=motion_contract.fps)
        save_bvh_remapped(gt_root, gt_euler, ref_bvh_path=rec.src_bvh, output_path=str(out_dir / "gt.bvh"), fps=motion_contract.fps)
    except Exception as exc:
        (out_dir / "bvh_export_error.txt").write_text(str(exc), encoding="utf-8")

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "cache": str(Path(args.cache).resolve()),
        "index": int(args.index),
        "segment_id": rec.segment_id,
        "src_bvh": rec.src_bvh,
        "clip_len": clip_len,
        "num_steps": int(args.num_steps),
        "solver": str(args.solver),
        "sample_mode": str(args.sample_mode),
        "target_mode": target_mode,
        "root_relative_first_frame": root_relative_first_frame,
        "use_global_text": use_global_text,
        "use_bert_text": use_bert_text,
        "use_bert_words": use_bert_words,
        "reconstruct_root_from_vel": reconstruct_root_from_vel,
        "pred_smooth_passes": pred_smooth_passes,
    }
    summary["metrics"] = compute_direct_metrics(
        pred_norm,
        rec.motion,
        mean=mean,
        std=std,
        root_indices=root_indices,
        rot6d_start=rot6d_start,
        root_anchor=rec.root_anchor,
    )
    if teacher_aux is not None:
        summary["teacher"] = teacher_aux
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    inspect_p = sub.add_parser("inspect")
    inspect_p.add_argument("--cache", type=str, required=True)
    inspect_p.add_argument("--clip_len", type=int, default=96)
    inspect_p.add_argument("--limit", type=int, default=8)
    inspect_p.add_argument("--root_relative_first_frame", action="store_true")
    inspect_p.set_defaults(func=cmd_inspect)

    train_p = sub.add_parser("train")
    train_p.add_argument("--cache", type=str, required=True)
    train_p.add_argument("--save_dir", type=str, required=True)
    train_p.add_argument("--clip_len", type=int, default=96)
    train_p.add_argument("--overfit_n", type=int, default=4)
    train_p.add_argument("--epochs", type=int, default=200)
    train_p.add_argument("--batch_size", type=int, default=4)
    train_p.add_argument("--hidden_dim", type=int, default=512)
    train_p.add_argument("--layers", type=int, default=4)
    train_p.add_argument("--heads", type=int, default=4)
    train_p.add_argument("--dropout", type=float, default=0.0)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--weight_decay", type=float, default=0.0)
    train_p.add_argument("--target_mode", choices=["velocity", "x0"], default="velocity")
    train_p.add_argument("--use_global_text", action="store_true")
    train_p.add_argument("--use_bert_text", action="store_true")
    train_p.add_argument("--use_bert_words", action="store_true")
    train_p.add_argument("--bert_path", type=str, default="models/bert")
    train_p.add_argument("--lambda_vel", type=float, default=0.0)
    train_p.add_argument("--lambda_acc", type=float, default=0.0)
    train_p.add_argument("--lambda_rot_vel", type=float, default=2.0)
    train_p.add_argument("--lambda_rot_acc", type=float, default=0.0)
    train_p.add_argument("--lambda_root_vel", type=float, default=0.0)
    train_p.add_argument("--lambda_root_acc", type=float, default=0.0)
    train_p.add_argument("--lambda_root_pos", type=float, default=0.0)
    train_p.add_argument("--pred_smooth_passes", type=int, default=1)
    train_p.add_argument("--smooth_weight_floor", type=float, default=0.25)
    train_p.add_argument("--t_sample_mode", choices=["auto", "uniform", "low_bias", "low_mix"], default="auto")
    train_p.add_argument("--t_low_bias_power", type=float, default=2.0)
    train_p.add_argument("--t_low_bias_mix", type=float, default=0.7)
    train_p.add_argument("--reconstruct_root_from_vel", action="store_true")
    train_p.add_argument("--root_relative_first_frame", action="store_true")
    train_p.add_argument("--device", type=str, default="cuda")
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--preview_every", type=int, default=20)
    train_p.add_argument("--preview_mode", choices=["free", "teacher", "both"], default="both")
    train_p.add_argument("--preview_index", type=int, default=0)
    train_p.add_argument("--preview_seed", type=int, default=123)
    train_p.add_argument("--preview_num_steps", type=int, default=32)
    train_p.add_argument("--preview_solver", choices=["euler", "heun"], default="heun")
    train_p.set_defaults(func=cmd_train)

    sample_p = sub.add_parser("sample")
    sample_p.add_argument("--checkpoint", type=str, required=True)
    sample_p.add_argument("--cache", type=str, required=True)
    sample_p.add_argument("--index", type=int, default=0)
    sample_p.add_argument("--out_dir", type=str, required=True)
    sample_p.add_argument("--num_steps", type=int, default=64)
    sample_p.add_argument("--solver", choices=["euler", "heun"], default="heun")
    sample_p.add_argument("--sample_mode", choices=["free", "teacher"], default="free")
    sample_p.add_argument("--device", type=str, default="cuda")
    sample_p.add_argument("--seed", type=int, default=42)
    sample_p.set_defaults(func=cmd_sample)
    sample_p.add_argument("--teacher_t", type=float, default=None)
    sample_p.add_argument("--pred_smooth_passes", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    set_seed(getattr(args, "seed", 42))
    args.func(args)


if __name__ == "__main__":
    main()
