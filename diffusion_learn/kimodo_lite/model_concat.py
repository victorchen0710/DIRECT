"""MDM-style input-concatenation conditional denoiser.

Alternative to cross-attention (which collapsed to mean on full-data). Here the
condition is concatenated to the noisy motion at the INPUT, so every token is
forced to see the condition before any attention.

BodyDenoiser input:
  concat(x_t, root_pred, audio, text_frame)  → input_proj → d_model
    [B, T, total_dim + root_dim + audio_dim + text_dim]  →  [B, T, d]

Root stays unconditional. CFG is implemented by swapping audio/text with
learned null vectors when drop_mask is True.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .feature_pack import FeatureLayout
from .model import RootDenoiser, TimeMLP, TransformerDenoiserCore


class ConcatBodyDenoiser(nn.Module):
    def __init__(
        self,
        layout: FeatureLayout,
        audio_dim: int,
        text_dim: int,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        support_anchor_mask: bool = True,
    ):
        super().__init__()
        self.layout = layout
        self.d_model = d_model
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.support_anchor_mask = support_anchor_mask
        # input: x_t + root_pred + audio + text (+ anchor_mask channel if supported)
        in_dim = layout.total_dim + layout.root_dim + audio_dim + text_dim
        if support_anchor_mask:
            in_dim += 1  # extra channel [B, T, 1] that is 1.0 where frame is anchored
        self.input_proj = nn.Linear(in_dim, d_model)
        self.time_mlp = TimeMLP(d_model)
        self.null_audio = nn.Parameter(torch.zeros(1, 1, audio_dim))
        self.null_text = nn.Parameter(torch.zeros(1, 1, text_dim))
        nn.init.normal_(self.null_audio, std=0.02)
        nn.init.normal_(self.null_text, std=0.02)
        self.core = TransformerDenoiserCore(d_model, n_layers, n_heads)
        self.out_proj = nn.Linear(d_model, layout.body_dim)

    def forward(
        self,
        x_t_full: torch.Tensor,
        root_pred: torch.Tensor,
        t: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
        anchor_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x_t_full.shape

        if audio is None:
            audio = self.null_audio.expand(B, T, -1)
        if text is None:
            text = self.null_text.expand(B, T, -1)
        elif text.dim() == 2:
            text = text.unsqueeze(1).expand(-1, T, -1)

        if drop_mask is not None:
            m = drop_mask.view(B, 1, 1).to(x_t_full.dtype)
            audio = m * self.null_audio.expand(B, T, -1) + (1 - m) * audio
            text = m * self.null_text.expand(B, T, -1) + (1 - m) * text

        parts = [x_t_full, root_pred, audio, text]
        if self.support_anchor_mask:
            if anchor_mask is None:
                anchor_mask = torch.zeros(B, T, 1, device=x_t_full.device, dtype=x_t_full.dtype)
            elif anchor_mask.dim() == 2:
                anchor_mask = anchor_mask.unsqueeze(-1)
            parts.append(anchor_mask.to(x_t_full.dtype))
        x = torch.cat(parts, dim=-1)
        tokens = self.input_proj(x)
        t_emb = self.time_mlp(t, self.d_model)
        h = self.core(tokens, t_emb)
        return self.out_proj(h)


@dataclass
class ConcatConfig:
    root_d: int = 384
    root_layers: int = 6
    root_heads: int = 8
    body_d: int = 512
    body_layers: int = 8
    body_heads: int = 8
    audio_dim: int = 768
    text_dim: int = 768
    support_anchor_mask: bool = True


class ConcatTwoStageDenoiser(nn.Module):
    def __init__(self, layout: FeatureLayout, cfg: ConcatConfig = ConcatConfig()):
        super().__init__()
        self.layout = layout
        self.cfg = cfg
        self.root = RootDenoiser(layout, cfg.root_d, cfg.root_layers, cfg.root_heads)
        self.body = ConcatBodyDenoiser(
            layout, cfg.audio_dim, cfg.text_dim,
            cfg.body_d, cfg.body_layers, cfg.body_heads,
            support_anchor_mask=cfg.support_anchor_mask,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
        anchor_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        root_pred = self.root(x_t, t)
        body_pred = self.body(
            x_t, root_pred, t,
            audio=audio, text=text, drop_mask=drop_mask, anchor_mask=anchor_mask,
        )
        return torch.cat([root_pred, body_pred], dim=-1)

    def num_params(self) -> Tuple[int, int]:
        rp = sum(p.numel() for p in self.root.parameters())
        bp = sum(p.numel() for p in self.body.parameters())
        return rp, bp
