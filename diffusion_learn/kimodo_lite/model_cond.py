"""Conditional two-stage denoiser with audio+text cross-attention (Method A).

Adds cross-attention conditioning only to BodyDenoiser; RootDenoiser stays
unconditional. Cross-attention K/V tokens are formed by:
  - per-frame audio (W2V2 768D) projected to d_model → T tokens
  - global text (BERT 768D) projected to d_model → 1 token
Concatenated as [text_token, audio_t0, audio_t1, ...] of length 1+T.

For CFG, learned null tokens replace conditions when `drop_mask` is True.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .feature_pack import FeatureLayout
from .model import PositionalEncoding, RootDenoiser, TimeMLP


class ConditionalTransformerLayer(nn.Module):
    """Pre-LN decoder-style layer: self-attn → cross-attn → FFN."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.norm2_q = nn.LayerNorm(d_model)
        self.norm2_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )

    def forward(self, x: torch.Tensor, cond_kv: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        q = self.norm2_q(x)
        kv = self.norm2_kv(cond_kv)
        x = x + self.cross_attn(q, kv, kv, need_weights=False)[0]
        x = x + self.mlp(self.norm3(x))
        return x


class ConditionalCore(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.posenc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [ConditionalTransformerLayer(d_model, n_heads) for _ in range(n_layers)]
        )

    def forward(self, tokens: torch.Tensor, t_emb: torch.Tensor, cond_kv: torch.Tensor) -> torch.Tensor:
        tokens = tokens + t_emb.unsqueeze(1)
        tokens = self.posenc(tokens)
        for layer in self.layers:
            tokens = layer(tokens, cond_kv)
        return tokens


class ConditionalBodyDenoiser(nn.Module):
    def __init__(
        self,
        layout: FeatureLayout,
        audio_dim: int,
        text_dim: int,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
    ):
        super().__init__()
        self.layout = layout
        self.d_model = d_model
        in_dim = layout.total_dim + layout.root_dim
        self.input_proj = nn.Linear(in_dim, d_model)
        self.time_mlp = TimeMLP(d_model)
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        # learned null embeddings (one per position role); used when CFG drops cond
        self.null_audio_tok = nn.Parameter(torch.zeros(1, 1, d_model))
        self.null_text_tok = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.null_audio_tok, std=0.02)
        nn.init.normal_(self.null_text_tok, std=0.02)
        self.cond_posenc = PositionalEncoding(d_model)
        self.core = ConditionalCore(d_model, n_layers, n_heads)
        self.out_proj = nn.Linear(d_model, layout.body_dim)

    def forward(
        self,
        x_t_full: torch.Tensor,
        root_pred: torch.Tensor,
        t: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x_t_full.shape
        x = torch.cat([x_t_full, root_pred], dim=-1)
        tokens = self.input_proj(x)
        t_emb = self.time_mlp(t, self.d_model)

        if audio is not None:
            audio_kv = self.audio_proj(audio)  # [B, T, d]
        else:
            audio_kv = self.null_audio_tok.expand(B, T, -1)
        # text can be global [B, D] (→ 1 token) or per-frame [B, T, D] (→ T tokens)
        if text is not None:
            if text.dim() == 2:
                text_kv = self.text_proj(text).unsqueeze(1)  # [B, 1, d]
                text_T = 1
            else:
                text_kv = self.text_proj(text)  # [B, T, d]
                text_T = T
        else:
            text_kv = self.null_text_tok.expand(B, 1, -1)
            text_T = 1

        if drop_mask is not None:
            m = drop_mask.view(B, 1, 1).to(tokens.dtype)
            audio_kv = m * self.null_audio_tok.expand(B, T, -1) + (1 - m) * audio_kv
            text_kv = m * self.null_text_tok.expand(B, text_T, -1) + (1 - m) * text_kv

        cond = torch.cat([text_kv, audio_kv], dim=1)  # [B, text_T+T, d]
        cond = self.cond_posenc(cond)
        h = self.core(tokens, t_emb, cond)
        return self.out_proj(h)


@dataclass
class CondConfig:
    root_d: int = 384
    root_layers: int = 6
    root_heads: int = 8
    body_d: int = 512
    body_layers: int = 8
    body_heads: int = 8
    audio_dim: int = 768
    text_dim: int = 768


class ConditionalTwoStageDenoiser(nn.Module):
    def __init__(self, layout: FeatureLayout, cfg: CondConfig = CondConfig()):
        super().__init__()
        self.layout = layout
        self.cfg = cfg
        self.root = RootDenoiser(layout, cfg.root_d, cfg.root_layers, cfg.root_heads)
        self.body = ConditionalBodyDenoiser(
            layout,
            cfg.audio_dim,
            cfg.text_dim,
            cfg.body_d,
            cfg.body_layers,
            cfg.body_heads,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        root_pred = self.root(x_t, t)
        body_pred = self.body(x_t, root_pred, t, audio=audio, text=text, drop_mask=drop_mask)
        return torch.cat([root_pred, body_pred], dim=-1)

    def num_params(self) -> Tuple[int, int]:
        rp = sum(p.numel() for p in self.root.parameters())
        bp = sum(p.numel() for p in self.body.parameters())
        return rp, bp
