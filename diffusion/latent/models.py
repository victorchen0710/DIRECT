from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import sinusoidal_encoding


PARTITION_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    # Mirrors the stageA split bias without importing the stageA training stack here.
    "upper": ("spine", "spine1", "spine2", "spine3", "neck", "head", "shoulder", "arm", "clavicle"),
    "hand": ("forearm", "hand", "thumb", "index", "middle", "ring", "pinky", "finger"),
    "lower": ("hips", "upleg", "leg", "foot", "toe", "toebase"),
}
PARTITION_PRIORITY: Tuple[str, ...] = ("hand", "lower", "upper")


def build_motion_partitions(
    *,
    joint_names: List[str],
    motion_dim: int,
    rot6d_start: int,
) -> Dict[str, List[int]]:
    if rot6d_start < 0 or rot6d_start > motion_dim:
        raise ValueError(f"Invalid rot6d_start={rot6d_start} for motion_dim={motion_dim}")
    if (motion_dim - rot6d_start) % 6 != 0:
        raise ValueError(f"Expected rot6d payload after index {rot6d_start}, got motion_dim={motion_dim}")

    partitions: Dict[str, List[int]] = {
        "root": list(range(rot6d_start)),
        "upper": [],
        "hand": [],
        "lower": [],
    }
    joint_count = (motion_dim - rot6d_start) // 6
    if len(joint_names) != joint_count:
        raise ValueError(
            f"joint_names length {len(joint_names)} does not match rot6d joint count {joint_count}"
        )

    for joint_idx, raw_name in enumerate(joint_names):
        name = str(raw_name).lower()
        part = "upper"
        for candidate in PARTITION_PRIORITY:
            if any(token in name for token in PARTITION_KEYWORDS[candidate]):
                part = candidate
                break
        start = rot6d_start + joint_idx * 6
        partitions[part].extend(range(start, start + 6))
    return partitions


def _select_features(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.index_select(x, dim=-1, index=indices)


def _downsample_mask(mask: torch.Tensor, stride: int) -> torch.Tensor:
    pooled = F.max_pool1d(mask.float().unsqueeze(1), kernel_size=stride, stride=stride, ceil_mode=True)
    return pooled.squeeze(1) > 0.5


def _resample_sequence(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.shape[1] == target_len:
        return x
    if x.shape[1] <= 1:
        return x[:, :1].expand(-1, target_len, -1)
    y = x.transpose(1, 2)
    y = F.interpolate(y, size=target_len, mode="linear", align_corners=False)
    return y.transpose(1, 2)


def _resample_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    if mask.shape[1] == target_len:
        return mask
    y = F.interpolate(mask.float().unsqueeze(1), size=target_len, mode="nearest")
    return y.squeeze(1) > 0.5


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    y = torch.stack((-x2, x1), dim=-1)
    return y.flatten(-2)


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
) -> torch.Tensor:
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=False,
    )
    return out.transpose(1, 2).contiguous().view(q.shape[0], q.shape[2], -1)


class MotionVAE(nn.Module):
    def __init__(
        self,
        motion_dim: int,
        *,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        latent_stride: int = 4,
        dropout: float = 0.1,
        deterministic_ae: bool = False,
        part_aware: bool = False,
        joint_names: Optional[List[str]] = None,
        rot6d_start: int = 15,
        part_layout: str = "legacy_root_v1",
        part_feature_indices: Optional[Dict[str, List[int]]] = None,
        slot_packed_latent: bool = False,
    ):
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_stride = int(latent_stride)
        self.deterministic_ae = bool(deterministic_ae)
        self.part_aware = bool(part_aware)
        self.rot6d_start = int(rot6d_start)
        self.part_layout = str(part_layout)
        self.part_names: Tuple[str, ...] = ("root", "upper", "hand", "lower")
        self.explicit_part_groups = bool(self.part_aware and part_feature_indices)
        self.slot_packed_latent = bool(slot_packed_latent and self.explicit_part_groups)
        self.factorized_part_slots = bool(
            self.explicit_part_groups
            and (self.part_layout == "upper_hand_lower_slots_v2" or self.slot_packed_latent)
        )
        self.slot_latent_dim = int(self.latent_dim)

        if self.part_aware:
            if self.explicit_part_groups:
                self.part_names = tuple(name for name in ("root", "upper", "hand", "lower") if name in part_feature_indices)
                self.part_in_proj = nn.ModuleDict()
                self.part_out_proj = nn.ModuleDict()
                for name in self.part_names:
                    indices = sorted(set(int(v) for v in (part_feature_indices or {}).get(name, ())))
                    if not indices:
                        continue
                    self.register_buffer(f"{name}_feature_indices", torch.tensor(indices, dtype=torch.long), persistent=False)
                    self.part_in_proj[name] = nn.Linear(len(indices), self.hidden_dim)
                    self.part_out_proj[name] = nn.Linear(self.hidden_dim, len(indices))
                if self.factorized_part_slots:
                    self.part_embed = nn.Parameter(torch.randn(len(self.part_names), self.hidden_dim) * 0.02)
                    if self.slot_packed_latent:
                        if self.latent_dim % max(1, len(self.part_names)) != 0:
                            raise ValueError(
                                f"slot_packed_latent requires latent_dim divisible by part count: "
                                f"latent_dim={self.latent_dim}, parts={len(self.part_names)}"
                            )
                        self.slot_latent_dim = self.latent_dim // len(self.part_names)
                        self.slot_mu_proj = nn.ModuleDict(
                            {name: nn.Linear(self.hidden_dim, self.slot_latent_dim) for name in self.part_names}
                        )
                        self.slot_logvar_proj = nn.ModuleDict(
                            {name: nn.Linear(self.hidden_dim, self.slot_latent_dim) for name in self.part_names}
                        )
                        self.slot_mem_proj = nn.ModuleDict(
                            {name: nn.Linear(self.slot_latent_dim, self.hidden_dim) for name in self.part_names}
                        )
                    else:
                        self.latent_fuse = nn.Sequential(
                            nn.Linear(len(self.part_names) * self.hidden_dim, self.hidden_dim * 2),
                            nn.GELU(),
                            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                            nn.LayerNorm(self.hidden_dim),
                        )
            else:
                if not joint_names:
                    raise ValueError("part_aware MotionVAE requires joint_names")
                partitions = build_motion_partitions(
                    joint_names=list(joint_names),
                    motion_dim=self.motion_dim,
                    rot6d_start=self.rot6d_start,
                )
                self.register_buffer("root_feature_indices", torch.tensor(partitions["root"], dtype=torch.long), persistent=False)
                self.register_buffer("upper_feature_indices", torch.tensor(partitions["upper"], dtype=torch.long), persistent=False)
                self.register_buffer("hand_feature_indices", torch.tensor(partitions["hand"], dtype=torch.long), persistent=False)
                self.register_buffer("lower_feature_indices", torch.tensor(partitions["lower"], dtype=torch.long), persistent=False)
                self.root_in_proj = nn.Linear(len(partitions["root"]), self.hidden_dim)
                self.root_out_proj = nn.Linear(self.hidden_dim, len(partitions["root"]))
                self.part_in_proj = nn.ModuleDict(
                    {
                        name: nn.Linear(len(partitions[name]), self.hidden_dim)
                        for name in self.part_names
                    }
                )
                self.part_out_proj = nn.ModuleDict(
                    {
                        name: nn.Linear(self.hidden_dim, len(partitions[name]))
                        for name in self.part_names
                    }
                )
        else:
            self.in_proj = nn.Linear(self.motion_dim, self.hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.mu_proj = nn.Linear(self.hidden_dim, self.latent_dim)
        self.logvar_proj = nn.Linear(self.hidden_dim, self.latent_dim)

        self.mem_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        self.out_ln = nn.LayerNorm(self.hidden_dim)
        if not self.part_aware:
            self.out_proj = nn.Linear(self.hidden_dim, self.motion_dim)

    def _pack_slot_latents(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected slot latents with dim=4, got shape={tuple(x.shape)}")
        return x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

    def _unpack_slot_latents(self, x: torch.Tensor) -> torch.Tensor:
        part_count = len(self.part_names)
        expected_dim = part_count * self.slot_latent_dim
        if x.shape[-1] != expected_dim:
            raise ValueError(
                f"Packed slot latents expect last dim {expected_dim}, got {x.shape[-1]}"
            )
        return x.view(x.shape[0], x.shape[1], part_count, self.slot_latent_dim)

    def _build_input_tokens(self, motion_norm: torch.Tensor) -> torch.Tensor:
        if not self.part_aware:
            return self.in_proj(motion_norm)
        if self.explicit_part_groups:
            if self.factorized_part_slots:
                tokens = []
                for part_idx, name in enumerate(self.part_names):
                    idx = getattr(self, f"{name}_feature_indices")
                    part = self.part_in_proj[name](_select_features(motion_norm, idx))
                    part = part + self.part_embed[part_idx].view(1, 1, -1)
                    tokens.append(part)
                return torch.stack(tokens, dim=1)
            x = motion_norm.new_zeros((motion_norm.shape[0], motion_norm.shape[1], self.hidden_dim))
            for name in self.part_names:
                idx = getattr(self, f"{name}_feature_indices")
                x = x + self.part_in_proj[name](_select_features(motion_norm, idx))
            return x
        x = self.root_in_proj(_select_features(motion_norm, self.root_feature_indices))
        for name in self.part_names:
            idx = getattr(self, f"{name}_feature_indices")
            x = x + self.part_in_proj[name](_select_features(motion_norm, idx))
        return x

    def _project_output(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.part_aware:
            return self.out_proj(hidden)
        if self.explicit_part_groups:
            if hidden.dim() == 4:
                out = hidden.new_zeros((hidden.shape[0], hidden.shape[2], self.motion_dim))
                for part_idx, name in enumerate(self.part_names):
                    idx = getattr(self, f"{name}_feature_indices")
                    out[..., idx] = self.part_out_proj[name](hidden[:, part_idx])
                return out
            out = hidden.new_zeros((hidden.shape[0], hidden.shape[1], self.motion_dim))
            for name in self.part_names:
                idx = getattr(self, f"{name}_feature_indices")
                out[..., idx] = self.part_out_proj[name](hidden)
            return out
        out = hidden.new_zeros((hidden.shape[0], hidden.shape[1], self.motion_dim))
        out[..., self.root_feature_indices] = self.root_out_proj(hidden)
        for name in self.part_names:
            idx = getattr(self, f"{name}_feature_indices")
            out[..., idx] = self.part_out_proj[name](hidden)
        return out

    def encode(self, motion_norm: torch.Tensor, motion_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._build_input_tokens(motion_norm)
        latent_mask = _downsample_mask(motion_mask, self.latent_stride)
        if self.factorized_part_slots:
            bsz, part_count, seq_len, _ = x.shape
            pos = sinusoidal_encoding(seq_len, self.hidden_dim, x.device, x.dtype).view(1, 1, seq_len, self.hidden_dim)
            x = x + pos
            x_flat = x.reshape(bsz * part_count, seq_len, self.hidden_dim)
            mask_flat = motion_mask.unsqueeze(1).expand(-1, part_count, -1).reshape(bsz * part_count, seq_len)
            x_flat = self.encoder(x_flat, src_key_padding_mask=~mask_flat)
            valid = mask_flat.unsqueeze(-1).to(dtype=x_flat.dtype)
            x_flat = x_flat * valid
            pooled_sum = F.avg_pool1d(
                x_flat.transpose(1, 2),
                kernel_size=self.latent_stride,
                stride=self.latent_stride,
                ceil_mode=True,
            ) * float(self.latent_stride)
            pooled_count = F.avg_pool1d(
                mask_flat.float().unsqueeze(1),
                kernel_size=self.latent_stride,
                stride=self.latent_stride,
                ceil_mode=True,
            ) * float(self.latent_stride)
            pooled = (pooled_sum / pooled_count.clamp_min(1e-6)).transpose(1, 2)
            pooled = pooled.view(bsz, part_count, pooled.shape[1], self.hidden_dim).permute(0, 2, 1, 3)
            if self.slot_packed_latent:
                mu_slots = []
                logvar_slots = []
                for part_idx, name in enumerate(self.part_names):
                    part_pooled = pooled[:, :, part_idx, :]
                    mu_slots.append(self.slot_mu_proj[name](part_pooled))
                    logvar_slots.append(self.slot_logvar_proj[name](part_pooled))
                mu = torch.stack(mu_slots, dim=2)
                logvar = torch.stack(logvar_slots, dim=2).clamp(min=-10.0, max=10.0)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                use_stochastic = self.training and (not self.deterministic_ae)
                z = mu + eps * std if use_stochastic else mu
                return (
                    self._pack_slot_latents(z),
                    latent_mask,
                    self._pack_slot_latents(mu),
                    self._pack_slot_latents(logvar),
                )
            pooled = pooled.reshape(bsz, pooled.shape[1], part_count * self.hidden_dim)
            pooled = self.latent_fuse(pooled)
        else:
            x = x + sinusoidal_encoding(x.shape[1], x.shape[2], x.device, x.dtype).unsqueeze(0)
            x = self.encoder(x, src_key_padding_mask=~motion_mask)
            valid = motion_mask.unsqueeze(-1).to(dtype=x.dtype)
            x = x * valid
            pooled_sum = F.avg_pool1d(
                x.transpose(1, 2),
                kernel_size=self.latent_stride,
                stride=self.latent_stride,
                ceil_mode=True,
            ) * float(self.latent_stride)
            pooled_count = F.avg_pool1d(
                motion_mask.float().unsqueeze(1),
                kernel_size=self.latent_stride,
                stride=self.latent_stride,
                ceil_mode=True,
            ) * float(self.latent_stride)
            pooled = (pooled_sum / pooled_count.clamp_min(1e-6)).transpose(1, 2)
        mu = self.mu_proj(pooled)
        logvar = self.logvar_proj(pooled).clamp(min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        use_stochastic = self.training and (not self.deterministic_ae)
        z = mu + eps * std if use_stochastic else mu
        return z, latent_mask, mu, logvar

    def decode(self, latents: torch.Tensor, latent_mask: torch.Tensor, target_len: int) -> torch.Tensor:
        if self.slot_packed_latent:
            slot_latents = self._unpack_slot_latents(latents)
            bsz, latent_len, part_count, _ = slot_latents.shape
            mem_slots = []
            pos = sinusoidal_encoding(latent_len, self.hidden_dim, latents.device, latents.dtype).unsqueeze(0)
            for part_idx, name in enumerate(self.part_names):
                part_mem = self.slot_mem_proj[name](slot_latents[:, :, part_idx, :])
                part_mem = part_mem + pos + self.part_embed[part_idx].view(1, 1, -1)
                mem_slots.append(part_mem)
            mem = torch.stack(mem_slots, dim=1)
        else:
            mem = self.mem_proj(latents)
            mem = mem + sinusoidal_encoding(mem.shape[1], mem.shape[2], mem.device, mem.dtype).unsqueeze(0)
        if self.factorized_part_slots:
            bsz = latents.shape[0]
            part_count = len(self.part_names)
            time_pos = sinusoidal_encoding(target_len, self.hidden_dim, latents.device, latents.dtype).view(1, 1, target_len, self.hidden_dim)
            queries = self.part_embed.to(dtype=latents.dtype, device=latents.device).view(1, part_count, 1, self.hidden_dim)
            queries = queries.expand(bsz, part_count, target_len, self.hidden_dim) + time_pos
            q_flat = queries.reshape(bsz * part_count, target_len, self.hidden_dim)
            if self.slot_packed_latent:
                mem_all = mem.permute(0, 2, 1, 3).reshape(bsz, mem.shape[2] * part_count, self.hidden_dim)
                mem_flat = mem_all.unsqueeze(1).expand(bsz, part_count, mem_all.shape[1], self.hidden_dim).reshape(
                    bsz * part_count, mem_all.shape[1], self.hidden_dim
                )
                mask_all = latent_mask.unsqueeze(-1).expand(bsz, latent_mask.shape[1], part_count).reshape(
                    bsz, latent_mask.shape[1] * part_count
                )
                mask_flat = mask_all.unsqueeze(1).expand(bsz, part_count, mask_all.shape[1]).reshape(
                    bsz * part_count, mask_all.shape[1]
                )
            else:
                mem_flat = mem.unsqueeze(1).expand(bsz, part_count, mem.shape[1], self.hidden_dim).reshape(
                    bsz * part_count, mem.shape[1], self.hidden_dim
                )
                mask_flat = latent_mask.unsqueeze(1).expand(bsz, part_count, latent_mask.shape[1]).reshape(
                    bsz * part_count, latent_mask.shape[1]
                )
            out = self.decoder(
                tgt=q_flat,
                memory=mem_flat,
                memory_key_padding_mask=~mask_flat,
            )
            out = self.out_ln(out).reshape(bsz, part_count, target_len, self.hidden_dim)
            return self._project_output(out)
        queries = torch.zeros((latents.shape[0], target_len, self.hidden_dim), device=latents.device, dtype=latents.dtype)
        queries = queries + sinusoidal_encoding(target_len, self.hidden_dim, latents.device, latents.dtype).unsqueeze(0)
        out = self.decoder(
            tgt=queries,
            memory=mem,
            memory_key_padding_mask=~latent_mask,
        )
        out = self.out_ln(out)
        return self._project_output(out)

    def forward(self, motion_norm: torch.Tensor, motion_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        z, latent_mask, mu, logvar = self.encode(motion_norm, motion_mask)
        recon = self.decode(z, latent_mask, target_len=motion_norm.shape[1])
        return {
            "recon_motion_norm": recon,
            "latents": z,
            "latent_mask": latent_mask,
            "mu": mu,
            "logvar": logvar,
        }


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = max(1, self.dim // 2)
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(1, half - 1))
        )
        angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb)


class RotaryEmbedding1D(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RotaryEmbedding1D requires an even dim, got {dim}")
        self.dim = int(dim)
        self.base = float(base)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / float(self.dim)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def apply(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = positions.to(device=q.device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.to(q.device))
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        cos = emb.cos().to(dtype=q.dtype).view(1, 1, q.shape[2], self.dim)
        sin = emb.sin().to(dtype=q.dtype).view(1, 1, q.shape[2], self.dim)
        q_rot = (q * cos) + (_rotate_half(q) * sin)
        k_rot = (k * cos) + (_rotate_half(k) * sin)
        return q_rot, k_rot


class AdaLNModulation(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6),
        )

    def forward(self, cond_vec: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.net(cond_vec).chunk(6, dim=-1)
        return tuple(value.unsqueeze(1) for value in (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp))


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextConditionRefiner(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.num_layers = int(max(0, num_layers))
        if self.num_layers <= 0:
            self.encoder = None
        else:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        cond_vec: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.encoder is None:
            return x
        x = x + self.context_proj(cond_vec).unsqueeze(1)
        return self.encoder(x, src_key_padding_mask=None if mask is None else ~mask)


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        num_heads: int,
        dropout: float,
        use_rope: bool,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.dropout = float(dropout)
        self.use_rope = bool(use_rope)

        self.motion_norm1 = nn.LayerNorm(self.hidden_dim)
        self.cond_norm1 = nn.LayerNorm(self.hidden_dim)
        self.motion_norm2 = nn.LayerNorm(self.hidden_dim)
        self.cond_norm2 = nn.LayerNorm(self.hidden_dim)
        self.motion_mod = AdaLNModulation(self.hidden_dim)
        self.cond_mod = AdaLNModulation(self.hidden_dim)

        self.motion_qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)
        self.cond_qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)
        self.motion_out = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.cond_out = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.motion_ff = FeedForward(self.hidden_dim, self.dropout)
        self.cond_ff = FeedForward(self.hidden_dim, self.dropout)
        self.rotary = RotaryEmbedding1D(self.head_dim) if self.use_rope else None

    def _split_qkv(self, qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = qkv.chunk(3, dim=-1)
        shape = (q.shape[0], q.shape[1], self.num_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        return q, k, v

    def forward(
        self,
        motion: torch.Tensor,
        cond: torch.Tensor,
        *,
        cond_vec: torch.Tensor,
        attn_mask: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m_shift_attn, m_scale_attn, m_gate_attn, m_shift_mlp, m_scale_mlp, m_gate_mlp = self.motion_mod(cond_vec)
        c_shift_attn, c_scale_attn, c_gate_attn, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.cond_mod(cond_vec)

        motion_attn_in = _modulate(self.motion_norm1(motion), m_shift_attn, m_scale_attn)
        cond_attn_in = _modulate(self.cond_norm1(cond), c_shift_attn, c_scale_attn)

        mq, mk, mv = self._split_qkv(self.motion_qkv(motion_attn_in))
        cq, ck, cv = self._split_qkv(self.cond_qkv(cond_attn_in))
        q = torch.cat([mq, cq], dim=2)
        k = torch.cat([mk, ck], dim=2)
        v = torch.cat([mv, cv], dim=2)
        if self.rotary is not None:
            q, k = self.rotary.apply(q, k, positions)
        attn_out = _attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
        motion_out, cond_out = torch.split(attn_out, [motion.shape[1], cond.shape[1]], dim=1)
        motion = motion + torch.tanh(m_gate_attn) * self.motion_out(motion_out)
        cond = cond + torch.tanh(c_gate_attn) * self.cond_out(cond_out)

        motion_ff_in = _modulate(self.motion_norm2(motion), m_shift_mlp, m_scale_mlp)
        cond_ff_in = _modulate(self.cond_norm2(cond), c_shift_mlp, c_scale_mlp)
        motion = motion + torch.tanh(m_gate_mlp) * self.motion_ff(motion_ff_in)
        cond = cond + torch.tanh(c_gate_mlp) * self.cond_ff(cond_ff_in)
        return motion, cond


class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        num_heads: int,
        dropout: float,
        use_rope: bool,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.dropout = float(dropout)
        self.use_rope = bool(use_rope)

        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.mod = AdaLNModulation(self.hidden_dim)
        self.qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)
        self.out = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ff = FeedForward(self.hidden_dim, self.dropout)
        self.rotary = RotaryEmbedding1D(self.head_dim) if self.use_rope else None

    def forward(
        self,
        x: torch.Tensor,
        *,
        cond_vec: torch.Tensor,
        attn_mask: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.mod(cond_vec)
        attn_in = _modulate(self.norm1(x), shift_attn, scale_attn)
        q, k, v = self.qkv(attn_in).chunk(3, dim=-1)
        shape = (x.shape[0], x.shape[1], self.num_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        if self.rotary is not None:
            q, k = self.rotary.apply(q, k, positions)
        attn_out = _attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
        x = x + torch.tanh(gate_attn) * self.out(attn_out)
        ff_in = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + torch.tanh(gate_mlp) * self.ff(ff_in)
        return x


class LatentRectifiedFlowTransformer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        *,
        audio_dim: int,
        lexical_dim: int,
        global_text_dim: Optional[int] = None,
        word_dim: int = 5,
        hidden_dim: int = 768,
        num_layers: int = 8,
        num_double_layers: int = 2,
        token_refiner_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        audio_drop_prob: float = 0.1,
        text_drop_prob: float = 0.1,
        word_drop_prob: float = 0.1,
        global_text_drop_prob: float = 0.1,
        local_attn_window: int = 15,
        use_rope: bool = True,
        slot_tokenized: bool = False,
        slot_part_names: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.audio_dim = int(audio_dim)
        self.lexical_dim = int(lexical_dim)
        self.global_text_dim = int(global_text_dim if global_text_dim is not None else lexical_dim)
        self.word_dim = int(word_dim)
        self.hidden_dim = int(hidden_dim)
        self.audio_drop_prob = float(audio_drop_prob)
        self.text_drop_prob = float(text_drop_prob)
        self.word_drop_prob = float(word_drop_prob)
        self.global_text_drop_prob = float(global_text_drop_prob)
        self.local_attn_window = int(local_attn_window)
        self.use_rope = bool(use_rope)
        self.num_double_layers = int(max(1, num_double_layers))
        self.num_single_layers = int(max(1, num_layers))
        self.slot_tokenized = bool(slot_tokenized)
        self.slot_part_names = tuple(str(name) for name in (slot_part_names or ()))
        self.slot_part_count = int(len(self.slot_part_names))
        if self.slot_tokenized:
            if self.slot_part_count <= 0:
                raise ValueError("slot_tokenized flow requires non-empty slot_part_names")
            if self.latent_dim % self.slot_part_count != 0:
                raise ValueError(
                    f"slot_tokenized flow requires latent_dim divisible by part count: "
                    f"latent_dim={self.latent_dim}, parts={self.slot_part_count}"
                )
            self.slot_latent_dim = self.latent_dim // self.slot_part_count
            self.slot_latent_proj = nn.Linear(self.slot_latent_dim, self.hidden_dim)
            self.slot_prefix_proj = nn.Sequential(
                nn.Linear(self.slot_latent_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            self.motion_part_embed = nn.Parameter(torch.randn(self.slot_part_count, self.hidden_dim) * 0.02)
        else:
            self.slot_latent_dim = self.latent_dim
            self.latent_proj = nn.Linear(self.latent_dim, self.hidden_dim)
            self.prefix_proj = nn.Sequential(nn.Linear(self.latent_dim, self.hidden_dim), nn.SiLU(), nn.Dropout(dropout))
        self.audio_proj = nn.Sequential(nn.Linear(self.audio_dim, self.hidden_dim), nn.SiLU(), nn.Dropout(dropout))
        self.lexical_proj = nn.Sequential(nn.Linear(self.lexical_dim, self.hidden_dim), nn.SiLU(), nn.Dropout(dropout))
        self.word_proj = nn.Sequential(nn.Linear(self.word_dim, self.hidden_dim), nn.SiLU(), nn.Dropout(dropout))
        self.global_text_proj = nn.Sequential(nn.Linear(self.global_text_dim, self.hidden_dim), nn.SiLU(), nn.Dropout(dropout))
        self.time_embed = SinusoidalTimeEmbedding(self.hidden_dim)
        self.text_refiner = TextConditionRefiner(
            self.hidden_dim,
            num_layers=token_refiner_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.task_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)
        self.type_embed = nn.Embedding(6, self.hidden_dim)
        self.null_audio = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)
        self.null_text = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)
        self.null_word = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)
        self.null_global_text = nn.Parameter(torch.randn(1, self.hidden_dim) * 0.02)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_rope=self.use_rope,
                )
                for _ in range(self.num_double_layers)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_rope=self.use_rope,
                )
                for _ in range(self.num_single_layers)
            ]
        )
        self.out_ln = nn.LayerNorm(self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.slot_latent_dim if self.slot_tokenized else self.latent_dim)

    def _project_motion_tokens(
        self,
        x_t: torch.Tensor,
        latent_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.slot_tokenized:
            positions = torch.arange(x_t.shape[1], device=x_t.device, dtype=torch.long)
            return self.latent_proj(x_t), latent_mask, positions

        if x_t.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected latent dim {self.latent_dim}, got {x_t.shape[-1]}")
        bsz, latent_len, _ = x_t.shape
        slots = x_t.view(bsz, latent_len, self.slot_part_count, self.slot_latent_dim)
        motion = self.slot_latent_proj(slots)
        motion = motion + self.motion_part_embed.view(1, 1, self.slot_part_count, self.hidden_dim)
        motion = motion.reshape(bsz, latent_len * self.slot_part_count, self.hidden_dim)
        motion_mask = latent_mask.unsqueeze(-1).expand(-1, -1, self.slot_part_count).reshape(
            bsz, latent_len * self.slot_part_count
        )
        positions = torch.arange(latent_len, device=x_t.device, dtype=torch.long).repeat_interleave(self.slot_part_count)
        return motion, motion_mask, positions

    def _sample_drop_mask(self, batch_size: int, device: torch.device, drop_prob: float, force_drop: bool) -> torch.Tensor:
        if force_drop:
            return torch.ones((batch_size,), device=device, dtype=torch.bool)
        if not self.training or drop_prob <= 0.0:
            return torch.zeros((batch_size,), device=device, dtype=torch.bool)
        return torch.rand((batch_size,), device=device) < drop_prob

    def _apply_null_sequence(
        self,
        x: torch.Tensor,
        *,
        null_token: torch.Tensor,
        drop_prob: float,
        force_drop: bool,
    ) -> torch.Tensor:
        drop_mask = self._sample_drop_mask(x.shape[0], x.device, drop_prob, force_drop)
        if not bool(drop_mask.any().item()):
            return x
        null_value = null_token.to(dtype=x.dtype, device=x.device).expand(x.shape[0], x.shape[1], -1)
        return torch.where(drop_mask.view(-1, 1, 1), null_value, x)

    def _apply_null_vector(
        self,
        x: torch.Tensor,
        *,
        null_vector: torch.Tensor,
        drop_prob: float,
        force_drop: bool,
    ) -> torch.Tensor:
        drop_mask = self._sample_drop_mask(x.shape[0], x.device, drop_prob, force_drop)
        if not bool(drop_mask.any().item()):
            return x
        null_value = null_vector.to(dtype=x.dtype, device=x.device).expand(x.shape[0], -1)
        return torch.where(drop_mask.view(-1, 1), null_value, x)

    def _build_attention_mask(
        self,
        *,
        motion_mask: torch.Tensor,
        cond_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = motion_mask.shape[0]
        motion_len = motion_mask.shape[1]
        cond_len = cond_mask.shape[1]
        total_len = motion_len + cond_len
        device = motion_mask.device

        allowed = torch.zeros((total_len, total_len), device=device, dtype=torch.bool)
        if self.local_attn_window <= 0:
            allowed[:motion_len, :motion_len] = True
        else:
            idx = torch.arange(motion_len, device=device)
            if self.slot_tokenized:
                idx = torch.div(idx, self.slot_part_count, rounding_mode="floor")
            dist = (idx[:, None] - idx[None, :]).abs()
            allowed[:motion_len, :motion_len] = dist <= int(self.local_attn_window)
        allowed[:motion_len, motion_len:] = True
        allowed[motion_len:, motion_len:] = True

        total_mask = torch.cat([motion_mask, cond_mask], dim=1)
        allowed = allowed.view(1, 1, total_len, total_len).expand(batch_size, 1, total_len, total_len).clone()
        allowed = allowed & total_mask.view(batch_size, 1, 1, total_len)
        allowed = allowed & total_mask.view(batch_size, 1, total_len, 1)

        diag = torch.eye(total_len, device=device, dtype=torch.bool).view(1, 1, total_len, total_len).expand(batch_size, 1, total_len, total_len)
        all_false = ~allowed.any(dim=-1, keepdim=True)
        allowed = torch.where(all_false, diag, allowed)
        return torch.where(
            allowed,
            torch.zeros((), device=device, dtype=torch.float32),
            torch.full((), float("-inf"), device=device, dtype=torch.float32),
        )

    def _project_condition_streams(
        self,
        *,
        target_len: int,
        timesteps: torch.Tensor,
        audio: torch.Tensor,
        audio_mask: Optional[torch.Tensor],
        lexical_frame: torch.Tensor,
        lexical_mask: Optional[torch.Tensor],
        word_frame: torch.Tensor,
        word_mask: Optional[torch.Tensor],
        global_text: Optional[torch.Tensor],
        prefix_latent: Optional[torch.Tensor],
        prefix_mask: Optional[torch.Tensor],
        force_drop_all: bool,
        force_drop_audio: bool,
        force_drop_text: bool,
        force_drop_word: bool,
        force_drop_global_text: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = audio.device
        audio_rs = _resample_sequence(audio, target_len)
        lexical_rs = _resample_sequence(lexical_frame, target_len)
        word_rs = _resample_sequence(word_frame, target_len)
        audio_rs_mask = _resample_mask(
            audio_mask if audio_mask is not None else torch.ones(audio.shape[:2], device=device, dtype=torch.bool),
            target_len,
        )
        lexical_rs_mask = _resample_mask(
            lexical_mask if lexical_mask is not None else torch.ones(lexical_frame.shape[:2], device=device, dtype=torch.bool),
            target_len,
        )
        word_rs_mask = _resample_mask(
            word_mask if word_mask is not None else torch.ones(word_frame.shape[:2], device=device, dtype=torch.bool),
            target_len,
        )

        audio_h = self.audio_proj(audio_rs)
        lexical_h = self.lexical_proj(lexical_rs)
        word_h = self.word_proj(word_rs)
        if global_text is None:
            global_text_h = self.null_global_text.to(device=device, dtype=audio_h.dtype).expand(audio.shape[0], -1)
        else:
            global_text_h = self.global_text_proj(global_text)

        global_text_h = self._apply_null_vector(
            global_text_h,
            null_vector=self.null_global_text,
            drop_prob=self.global_text_drop_prob,
            force_drop=force_drop_all or force_drop_global_text,
        )
        cond_vec = self.time_embed(timesteps) + global_text_h

        lexical_h = self.text_refiner(lexical_h, cond_vec=cond_vec, mask=lexical_rs_mask)
        audio_h = self._apply_null_sequence(
            audio_h,
            null_token=self.null_audio,
            drop_prob=self.audio_drop_prob,
            force_drop=force_drop_all or force_drop_audio,
        )
        lexical_h = self._apply_null_sequence(
            lexical_h,
            null_token=self.null_text,
            drop_prob=self.text_drop_prob,
            force_drop=force_drop_all or force_drop_text,
        )
        word_h = self._apply_null_sequence(
            word_h,
            null_token=self.null_word,
            drop_prob=self.word_drop_prob,
            force_drop=force_drop_all or force_drop_word,
        )

        task_token = self.task_token.expand(audio.shape[0], -1, -1) + self.type_embed.weight[0].view(1, 1, -1)
        global_token = global_text_h.unsqueeze(1) + self.type_embed.weight[4].view(1, 1, -1)
        streams = [
            task_token,
            global_token,
            audio_h + self.type_embed.weight[1].view(1, 1, -1),
            lexical_h + self.type_embed.weight[2].view(1, 1, -1),
            word_h + self.type_embed.weight[3].view(1, 1, -1),
        ]
        masks = [
            torch.ones((audio.shape[0], 1), device=device, dtype=torch.bool),
            torch.ones((audio.shape[0], 1), device=device, dtype=torch.bool),
            audio_rs_mask,
            lexical_rs_mask,
            word_rs_mask,
        ]

        if prefix_latent is not None and prefix_latent.numel() > 0:
            if self.slot_tokenized:
                bsz, prefix_len, _ = prefix_latent.shape
                prefix_slots = prefix_latent.view(bsz, prefix_len, self.slot_part_count, self.slot_latent_dim)
                prefix_h = self.slot_prefix_proj(prefix_slots)
                prefix_h = prefix_h + self.motion_part_embed.view(1, 1, self.slot_part_count, self.hidden_dim)
                prefix_h = prefix_h.reshape(bsz, prefix_len * self.slot_part_count, self.hidden_dim)
            else:
                prefix_h = self.prefix_proj(prefix_latent)
            prefix_h = prefix_h + self.type_embed.weight[5].view(1, 1, -1)
            streams.append(prefix_h)
            if prefix_mask is None:
                masks.append(torch.ones((audio.shape[0], prefix_h.shape[1]), device=device, dtype=torch.bool))
            else:
                if self.slot_tokenized:
                    prefix_mask = prefix_mask.unsqueeze(-1).expand(-1, -1, self.slot_part_count).reshape(
                        prefix_mask.shape[0], prefix_mask.shape[1] * self.slot_part_count
                    )
                masks.append(prefix_mask.to(device=device, dtype=torch.bool))

        cond = torch.cat(streams, dim=1)
        cond_mask = torch.cat(masks, dim=1)
        return cond, cond_mask, cond_vec

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        latent_mask: torch.Tensor,
        *,
        audio: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        lexical_frame: torch.Tensor,
        lexical_mask: Optional[torch.Tensor] = None,
        word_frame: torch.Tensor,
        word_mask: Optional[torch.Tensor] = None,
        global_text: Optional[torch.Tensor] = None,
        prefix_latent: Optional[torch.Tensor] = None,
        prefix_mask: Optional[torch.Tensor] = None,
        force_drop_all: bool = False,
        force_drop_audio: bool = False,
        force_drop_text: bool = False,
        force_drop_word: bool = False,
        force_drop_global_text: bool = False,
    ) -> torch.Tensor:
        target_len = x_t.shape[1]
        motion, motion_mask, motion_positions = self._project_motion_tokens(x_t, latent_mask)
        cond, cond_mask, cond_vec = self._project_condition_streams(
            target_len=target_len,
            timesteps=timesteps,
            audio=audio,
            audio_mask=audio_mask,
            lexical_frame=lexical_frame,
            lexical_mask=lexical_mask,
            word_frame=word_frame,
            word_mask=word_mask,
            global_text=global_text,
            prefix_latent=prefix_latent,
            prefix_mask=prefix_mask,
            force_drop_all=force_drop_all,
            force_drop_audio=force_drop_audio,
            force_drop_text=force_drop_text,
            force_drop_word=force_drop_word,
            force_drop_global_text=force_drop_global_text,
        )
        attn_mask = self._build_attention_mask(motion_mask=motion_mask, cond_mask=cond_mask).to(device=x_t.device, dtype=x_t.dtype)
        cond_positions = target_len + torch.arange(cond.shape[1], device=x_t.device, dtype=torch.long)
        positions = torch.cat([motion_positions, cond_positions], dim=0)
        for block in self.double_blocks:
            motion, cond = block(
                motion,
                cond,
                cond_vec=cond_vec,
                attn_mask=attn_mask,
                positions=positions,
            )

        seq = torch.cat([motion, cond], dim=1)
        for block in self.single_blocks:
            seq = block(
                seq,
                cond_vec=cond_vec,
                attn_mask=attn_mask,
                positions=positions,
            )
        out = self.out_ln(seq[:, :motion.shape[1]])
        out = self.out_proj(out)
        if self.slot_tokenized:
            out = out.view(x_t.shape[0], target_len, self.slot_part_count, self.slot_latent_dim)
            out = out.reshape(x_t.shape[0], target_len, self.latent_dim)
        return out * latent_mask.unsqueeze(-1).to(dtype=out.dtype)


class MotionRefiner(nn.Module):
    def __init__(
        self,
        motion_dim: int,
        *,
        audio_dim: int,
        lexical_dim: int,
        global_text_dim: Optional[int] = None,
        word_dim: int = 5,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.audio_dim = int(audio_dim)
        self.lexical_dim = int(lexical_dim)
        self.global_text_dim = int(global_text_dim if global_text_dim is not None else lexical_dim)
        self.word_dim = int(word_dim)
        self.hidden_dim = int(hidden_dim)

        self.motion_proj = nn.Linear(self.motion_dim, self.hidden_dim)
        self.audio_proj = nn.Sequential(nn.Linear(self.audio_dim, self.hidden_dim), nn.SiLU(), nn.Dropout(dropout))
        self.lexical_proj = nn.Sequential(nn.Linear(self.lexical_dim, self.hidden_dim), nn.SiLU(), nn.Dropout(dropout))
        self.word_proj = nn.Sequential(nn.Linear(self.word_dim, self.hidden_dim), nn.SiLU(), nn.Dropout(dropout))
        self.global_text_proj = nn.Sequential(nn.Linear(self.global_text_dim, self.hidden_dim), nn.SiLU(), nn.Dropout(dropout))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(max(1, num_layers)))
        self.out_ln = nn.LayerNorm(self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.motion_dim)

    def forward(
        self,
        motion_norm: torch.Tensor,
        motion_mask: torch.Tensor,
        *,
        audio: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        lexical_frame: torch.Tensor,
        lexical_mask: Optional[torch.Tensor] = None,
        word_frame: torch.Tensor,
        word_mask: Optional[torch.Tensor] = None,
        global_text: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del audio_mask, lexical_mask, word_mask
        target_len = motion_norm.shape[1]
        x = self.motion_proj(motion_norm)
        x = x + self.audio_proj(_resample_sequence(audio, target_len))
        x = x + self.lexical_proj(_resample_sequence(lexical_frame, target_len))
        x = x + self.word_proj(_resample_sequence(word_frame, target_len))
        if global_text is not None:
            x = x + self.global_text_proj(global_text).unsqueeze(1)
        x = x + sinusoidal_encoding(target_len, self.hidden_dim, motion_norm.device, motion_norm.dtype).unsqueeze(0)
        x = self.encoder(x, src_key_padding_mask=~motion_mask)
        residual = self.out_proj(self.out_ln(x))
        refined = motion_norm + residual
        return refined * motion_mask.unsqueeze(-1).to(dtype=refined.dtype)
