import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


def rot6d_to_matrix(d6: Tensor, eps: float = 1e-8) -> Tensor:
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]

    b1 = a1 / torch.linalg.norm(a1, dim=-1, keepdim=True).clamp_min(eps)
    proj = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = a2 - proj * b1
    b2 = b2 / torch.linalg.norm(b2, dim=-1, keepdim=True).clamp_min(eps)
    b3 = torch.cross(b1, b2, dim=-1)

    rm = torch.stack([b1, b2, b3], dim=-1)

    bad = ~torch.isfinite(rm).all(dim=(-1, -2))
    if bad.any():
        eye = torch.eye(3, device=d6.device, dtype=d6.dtype)
        eye = eye.view(*([1] * (rm.ndim - 2)), 3, 3).expand_as(rm)
        rm = torch.where(bad[..., None, None], eye, rm)

    return rm


def so3_relative(r_prev: Tensor, r_cur: Tensor) -> Tensor:
    return r_prev.transpose(-1, -2) @ r_cur


def so3_log_map(rm: Tensor, eps: float = 1e-8) -> Tensor:
    w = torch.stack(
        [
            rm[..., 2, 1] - rm[..., 1, 2],
            rm[..., 0, 2] - rm[..., 2, 0],
            rm[..., 1, 0] - rm[..., 0, 1],
        ],
        dim=-1,
    )

    sin_theta = 0.5 * torch.linalg.norm(w, dim=-1)
    tr = rm[..., 0, 0] + rm[..., 1, 1] + rm[..., 2, 2]
    cos_theta = (0.5 * (tr - 1.0)).clamp(-1.0, 1.0)
    theta = torch.atan2(sin_theta, cos_theta)

    denom = (2.0 * sin_theta).clamp_min(eps)
    scale = theta / denom
    scale = torch.where(sin_theta < 1e-4, torch.full_like(scale, 0.5), scale)

    omega = scale.unsqueeze(-1) * w
    omega = torch.where(torch.isfinite(omega), omega, torch.zeros_like(omega))
    return omega


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = int(d_model)
        pe = self._build_pe(max_len, device=torch.device("cpu"))
        self.register_buffer("pe", pe, persistent=True)

    def _build_pe(self, length: int, device: torch.device) -> Tensor:
        pe = torch.zeros(length, self.d_model, device=device, dtype=torch.float32)
        position = torch.arange(0, length, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / float(self.d_model))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        t = int(x.shape[1])
        if t > int(self.pe.shape[1]) or self.pe.device != x.device:
            self.pe = self._build_pe(t, device=x.device)
        return self.pe[:, :t, :]


class DDPMScheduler:
    """
    Notation:
      x_t = a * x0 + s * eps
      a = sqrt(alpha_bar), s = sqrt(1 - alpha_bar)

    v-parameterization:
      v = a * eps - s * x0

    Then:
      x0 = a * x_t - s * v
      eps= s * x_t + a * v
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: Union[torch.device, str] = "cuda",
    ):
        self.num_timesteps = int(num_timesteps)
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, device=device, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def _get_a_s(self, timesteps: Tensor) -> Tuple[Tensor, Tensor]:
        a = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1)
        s = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1)
        return a, s

    def add_noise(self, x0: Tensor, eps: Tensor, timesteps: Tensor) -> Tensor:
        a, s = self._get_a_s(timesteps)
        return a * x0 + s * eps

    def v_target(self, x0: Tensor, eps: Tensor, timesteps: Tensor) -> Tensor:
        a, s = self._get_a_s(timesteps)
        return a * eps - s * x0

    def x0_from_v(self, x_t: Tensor, v: Tensor, timesteps: Tensor) -> Tensor:
        a, s = self._get_a_s(timesteps)
        return a * x_t - s * v

    def eps_from_v(self, x_t: Tensor, v: Tensor, timesteps: Tensor) -> Tensor:
        a, s = self._get_a_s(timesteps)
        return s * x_t + a * v


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        if half_dim < 2:
            raise ValueError(f"SinusoidalPosEmb dim too small: dim={self.dim}")
        emb_scale = math.log(10000.0) / float(half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb_scale)
        args = x.to(torch.float32)[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class MotionDiffusionTransformer(nn.Module):
    def __init__(
        self,
        motion_dim: int,
        audio_dim: int,
        bert_path: str,
        motion_fps: float = 30.0,
        hidden: int = 768,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        cond_drop_prob: float = 0.1,
        text_drop_prob: float = 0.0,
        audio_drop_prob: float = 0.0,
        word_drop_prob: float = 0.0,
    ):
        super().__init__()

        self.motion_dim = int(motion_dim)
        self.audio_dim = int(audio_dim)
        self.motion_fps = float(motion_fps)
        self.cond_drop_prob = float(cond_drop_prob)
        self.text_drop_prob = float(text_drop_prob)
        self.audio_drop_prob = float(audio_drop_prob)
        self.word_drop_prob = float(word_drop_prob)

        print(f"[Model] Loading BERT from {bert_path}...")
        from transformers import BertModel

        try:
            self.bert = BertModel.from_pretrained(bert_path, local_files_only=True)
        except TypeError:
            self.bert = BertModel.from_pretrained(bert_path)
        except Exception:
            self.bert = BertModel.from_pretrained(bert_path)

        for p in self.bert.parameters():
            p.requires_grad = False
        self.bert.eval()

        self.text_global_proj = nn.Sequential(
            nn.Linear(768, hidden),
            nn.LayerNorm(hidden),
        )

        if self.audio_dim <= 0:
            raise ValueError("audio_dim must be > 0, got %d" % self.audio_dim)
        self.audio_proj = nn.Sequential(
            nn.Linear(self.audio_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.word_proj = nn.Sequential(
            nn.Linear(5, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.cond_log_scale = nn.Parameter(torch.zeros(3))
        self.cond_fuse = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.cond_pos = PositionalEncoding(hidden)
        self.cond_ln = nn.LayerNorm(hidden)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.motion_proj = nn.Linear(self.motion_dim, hidden)
        self.motion_pos = PositionalEncoding(hidden)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.pre_out_ln = nn.LayerNorm(hidden)
        self.final_proj = nn.Linear(hidden, self.motion_dim)

        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.bert.eval()
        return self

    def _encode_text_global(self, text_input_ids: Tensor, text_mask: Tensor) -> Tensor:
        out = self.bert(input_ids=text_input_ids, attention_mask=text_mask)
        h = out.last_hidden_state
        m = (text_mask > 0).to(h.dtype).unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1.0)
        pooled = (h * m).sum(dim=1) / denom
        return pooled

    def _resample_audio_to_len(
        self,
        audio: Tensor,
        audio_mask: Tensor,
        target_lens: Tensor,
        t_pad: int,
    ) -> Tensor:
        bsz, _, da = audio.shape
        out = audio.new_zeros((bsz, t_pad, da), dtype=torch.float32)

        for i in range(bsz):
            ti = int(target_lens[i].item())
            if ti <= 0:
                continue
            ai = int(audio_mask[i].sum().item())
            if ai <= 0:
                continue

            a = audio[i, :ai].to(torch.float32)
            if ai == 1:
                a_rs = a.repeat(ti, 1)
            else:
                a_ = a.transpose(0, 1).unsqueeze(0)
                a_rs = F.interpolate(a_, size=ti, mode="linear", align_corners=False)[0].transpose(0, 1)
            out[i, :ti] = a_rs

        return out

    def _build_word_time_feats(
        self,
        word_times: Tensor,
        word_mask: Tensor,
        target_lens: Tensor,
        t_pad: int,
        device: torch.device,
    ) -> Tensor:
        bsz, _, _ = word_times.shape
        feats = torch.zeros(bsz, t_pad, 5, device=device, dtype=torch.float32)

        tol = 0.5 / max(self.motion_fps, 1e-6)

        for i in range(bsz):
            ti = int(target_lens[i].item())
            if ti <= 0:
                continue

            ni = int(word_mask[i].sum().item())
            if ni <= 0:
                continue

            wt = word_times[i, :ni].to(torch.float32)
            starts = wt[:, 0]
            ends = wt[:, 1]
            valid = ends > (starts + 1e-5)
            if int(valid.sum().item()) == 0:
                continue

            starts = starts[valid]
            ends = ends[valid]
            if starts.numel() == 0:
                continue

            t = torch.arange(ti, device=device, dtype=torch.float32) / float(self.motion_fps)
            tcol = t[:, None]

            in_word = (tcol >= starts[None, :]) & (tcol < ends[None, :])
            has = in_word.any(dim=1)
            if int(has.sum().item()) == 0:
                continue

            idx = in_word.to(torch.float32).argmax(dim=1)
            s_sel = starts[idx]
            e_sel = ends[idx]

            dur = (e_sel - s_sel).clamp(min=1e-4)
            prog = ((t - s_sel) / dur).clamp(0.0, 1.0)

            is_word = has.to(torch.float32)
            prog = torch.where(has, prog, torch.zeros_like(prog))
            dur = torch.where(has, dur, torch.zeros_like(dur))

            b_start = (has & (torch.abs(t - s_sel) <= tol)).to(torch.float32)
            b_end = (has & (torch.abs(t - e_sel) <= tol)).to(torch.float32)

            feats[i, :ti] = torch.stack([is_word, prog, dur, b_start, b_end], dim=-1)

        return feats

    def _apply_cond_drop(
        self,
        text_g: Tensor,
        audio_h: Tensor,
        word_h: Tensor,
        force_drop_all: bool = False,
        force_drop_text: bool = False,
        force_drop_audio: bool = False,
        force_drop_word: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        device = text_g.device
        bsz = int(text_g.shape[0])

        keep_all = torch.ones((bsz, 1, 1), device=device, dtype=torch.float32)
        keep_text = torch.ones((bsz, 1, 1), device=device, dtype=torch.float32)
        keep_audio = torch.ones((bsz, 1, 1), device=device, dtype=torch.float32)
        keep_word = torch.ones((bsz, 1, 1), device=device, dtype=torch.float32)

        if self.training:
            if self.cond_drop_prob > 0:
                keep_all = (torch.rand((bsz, 1, 1), device=device) > self.cond_drop_prob).to(torch.float32)
            if self.text_drop_prob > 0:
                keep_text = (torch.rand((bsz, 1, 1), device=device) > self.text_drop_prob).to(torch.float32)
            if self.audio_drop_prob > 0:
                keep_audio = (torch.rand((bsz, 1, 1), device=device) > self.audio_drop_prob).to(torch.float32)
            if self.word_drop_prob > 0:
                keep_word = (torch.rand((bsz, 1, 1), device=device) > self.word_drop_prob).to(torch.float32)

        if force_drop_all:
            keep_all = torch.zeros_like(keep_all)
        if force_drop_text:
            keep_text = torch.zeros_like(keep_text)
        if force_drop_audio:
            keep_audio = torch.zeros_like(keep_audio)
        if force_drop_word:
            keep_word = torch.zeros_like(keep_word)

        text_g = text_g * keep_all * keep_text
        audio_h = audio_h * keep_all * keep_audio
        word_h = word_h * keep_all * keep_word
        return text_g, audio_h, word_h

    def forward(
        self,
        x_noisy: Tensor,
        timesteps: Tensor,
        text_input_ids: Tensor,
        text_mask: Tensor,
        motion_mask: Optional[Tensor] = None,
        audio: Optional[Tensor] = None,
        audio_mask: Optional[Tensor] = None,
        word_times: Optional[Tensor] = None,
        word_mask: Optional[Tensor] = None,
        force_drop_all: bool = False,
        force_drop_text: bool = False,
        force_drop_audio: bool = False,
        force_drop_word: bool = False,
    ) -> Tensor:
        device = x_noisy.device
        bsz, t_pad, _ = x_noisy.shape

        if motion_mask is None:
            motion_mask_bool = torch.ones((bsz, t_pad), device=device, dtype=torch.bool)
        else:
            motion_mask_bool = motion_mask.to(torch.bool)
        target_lens = motion_mask_bool.sum(dim=1).to(torch.long).clamp(min=0)

        with torch.no_grad():
            text_global = self._encode_text_global(text_input_ids, text_mask)
        text_g = self.text_global_proj(text_global).to(torch.float32)
        text_g = text_g.unsqueeze(1).expand(bsz, t_pad, -1)

        if audio is None:
            audio_rs = torch.zeros((bsz, t_pad, self.audio_dim), device=device, dtype=torch.float32)
        else:
            if audio_mask is None:
                audio_mask = torch.ones((audio.shape[0], audio.shape[1]), device=audio.device, dtype=torch.bool)
            audio_rs = self._resample_audio_to_len(audio, audio_mask, target_lens, t_pad)
        audio_h = self.audio_proj(audio_rs)

        if word_times is None or word_mask is None:
            word_feat = torch.zeros((bsz, t_pad, 5), device=device, dtype=torch.float32)
        else:
            word_feat = self._build_word_time_feats(word_times, word_mask, target_lens, t_pad, device)
        word_h = self.word_proj(word_feat)

        text_g, audio_h, word_h = self._apply_cond_drop(
            text_g,
            audio_h,
            word_h,
            force_drop_all=bool(force_drop_all),
            force_drop_text=bool(force_drop_text),
            force_drop_audio=bool(force_drop_audio),
            force_drop_word=bool(force_drop_word),
        )

        scales = torch.exp(self.cond_log_scale).clamp(0.1, 10.0)
        sa, sw, st = scales[0], scales[1], scales[2]
        cond = torch.cat([sa * audio_h, sw * word_h, st * text_g], dim=-1)
        cond = self.cond_fuse(cond)
        cond = cond + self.cond_pos(cond)
        cond = self.cond_ln(cond)

        x = self.motion_proj(x_noisy).to(torch.float32)
        x = x + self.motion_pos(x)

        t_emb = self.time_mlp(timesteps).to(torch.float32)
        x = x + t_emb.unsqueeze(1)

        key_pad = ~motion_mask_bool
        out = self.transformer(
            tgt=x,
            memory=cond,
            tgt_key_padding_mask=key_pad,
            memory_key_padding_mask=key_pad,
        )
        out = self.pre_out_ln(out)
        return self.final_proj(out)


class SkeletonFK(nn.Module):
    def __init__(self, offsets, parents, all_names, feature_names):
        super().__init__()
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.float32))
        self.register_buffer("parents", torch.tensor(parents, dtype=torch.long))

        self.j_all = len(all_names)
        self.j_feat = len(feature_names)

        feat_map = {name: i for i, name in enumerate(feature_names)}
        skel_to_feat = []
        for name in all_names:
            if name in feat_map:
                skel_to_feat.append(feat_map[name])
            else:
                clean = name.replace("_End", "")
                skel_to_feat.append(feat_map.get(clean, -1))
        self.register_buffer("map_s2f", torch.tensor(skel_to_feat, dtype=torch.long))
        self.all_names = list(all_names)

    def forward(self, rot_mats_feat: Tensor, root_pos: Tensor) -> Tensor:
        bsz, steps, _, _, _ = rot_mats_feat.shape
        identity = torch.eye(3, device=rot_mats_feat.device).view(1, 1, 3, 3).expand(bsz, steps, 3, 3)

        global_rots = [None] * self.j_all
        global_pos = [None] * self.j_all

        for i in range(self.j_all):
            parent = self.parents[i].item()
            offset = self.offsets[i]

            feat_idx = self.map_s2f[i].item()
            local_r = rot_mats_feat[:, :, feat_idx] if feat_idx >= 0 else identity

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


__all__ = [
    "DDPMScheduler",
    "MotionDiffusionTransformer",
    "PositionalEncoding",
    "SinusoidalPosEmb",
    "SkeletonFK",
    "rot6d_to_matrix",
    "so3_log_map",
    "so3_relative",
]
