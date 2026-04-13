"""Two-stage denoiser: RootDenoiser + BodyDenoiser (Kimodo-style).

Each denoiser is a small transformer encoder over the time dimension, with
sinusoidal timestep conditioning added as a learned token + additive embedding.

Both stages run at every diffusion step:
    r_hat = RootDenoiser(x_t, t)                        # predicts clean r_p, r_a
    b_hat = BodyDenoiser(x_t, r_hat, t)                 # predicts clean j_p, j_v, j_a, f
    x0_hat = concat(r_hat, b_hat)

For Phase 5 (unconditional 4-clip overfit) we keep models small: root 4L/d256,
body 6L/d384. These are orders of magnitude smaller than Kimodo's 282M model —
intentional for the sanity gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from .feature_pack import FeatureLayout


# ---------------------------------------------------------------------------
# Common modules
# ---------------------------------------------------------------------------


def sinusoidal_timestep_emb(t: torch.Tensor, d: int) -> torch.Tensor:
    """Standard transformer-style sinusoidal embedding for integer timesteps.

    t shape [B], returns [B, d].
    """
    half = d // 2
    device = t.device
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if d % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TimeMLP(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, t: torch.Tensor, d_model: int) -> torch.Tensor:
        emb = sinusoidal_timestep_emb(t, d_model)
        return self.net(emb)


class TransformerDenoiserCore(nn.Module):
    """Time-axis transformer encoder with additive timestep conditioning.

    Given token sequence [B, T, d_model] and timestep embedding [B, d_model],
    adds timestep to every token and passes through a TransformerEncoder.
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.posenc = PositionalEncoding(d_model)

    def forward(self, tokens: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, d], t_emb: [B, d]
        tokens = tokens + t_emb.unsqueeze(1)
        tokens = self.posenc(tokens)
        return self.encoder(tokens)


# ---------------------------------------------------------------------------
# Stage 1: RootDenoiser
# ---------------------------------------------------------------------------


class RootDenoiser(nn.Module):
    """Predicts clean [r_p, r_a] from noisy full motion x_t.

    The full noisy motion serves as context, but the output is only the root
    subspace. Kimodo uses the full motion as context so that body motion can
    inform root prediction (e.g., swing of arms correlates with walk direction).
    """

    def __init__(self, layout: FeatureLayout, d_model: int = 256, n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        self.layout = layout
        self.input_proj = nn.Linear(layout.total_dim, d_model)
        self.time_mlp = TimeMLP(d_model)
        self.core = TransformerDenoiserCore(d_model, n_layers, n_heads)
        self.out_proj = nn.Linear(d_model, layout.root_dim)
        self.d_model = d_model

    def forward(self, x_t_full: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x_t_full: [B, T, total_dim]
        tokens = self.input_proj(x_t_full)
        t_emb = self.time_mlp(t, self.d_model)
        h = self.core(tokens, t_emb)
        return self.out_proj(h)  # [B, T, root_dim]


# ---------------------------------------------------------------------------
# Stage 2: BodyDenoiser
# ---------------------------------------------------------------------------


class BodyDenoiser(nn.Module):
    """Predicts clean [j_p, j_v, j_a, f] given noisy full motion + predicted root.

    Inputs: noisy full motion x_t (total_dim) + predicted clean root (root_dim).
    Output: body_dim.
    """

    def __init__(self, layout: FeatureLayout, d_model: int = 384, n_layers: int = 6, n_heads: int = 8):
        super().__init__()
        self.layout = layout
        in_dim = layout.total_dim + layout.root_dim
        self.input_proj = nn.Linear(in_dim, d_model)
        self.time_mlp = TimeMLP(d_model)
        self.core = TransformerDenoiserCore(d_model, n_layers, n_heads)
        self.out_proj = nn.Linear(d_model, layout.body_dim)
        self.d_model = d_model

    def forward(
        self,
        x_t_full: torch.Tensor,
        root_pred: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # x_t_full: [B, T, total_dim], root_pred: [B, T, root_dim]
        x = torch.cat([x_t_full, root_pred], dim=-1)
        tokens = self.input_proj(x)
        t_emb = self.time_mlp(t, self.d_model)
        h = self.core(tokens, t_emb)
        return self.out_proj(h)  # [B, T, body_dim]


# ---------------------------------------------------------------------------
# Combined two-stage wrapper
# ---------------------------------------------------------------------------


@dataclass
class TwoStageConfig:
    root_d: int = 256
    root_layers: int = 4
    root_heads: int = 4
    body_d: int = 384
    body_layers: int = 6
    body_heads: int = 8


class TwoStageDenoiser(nn.Module):
    def __init__(self, layout: FeatureLayout, cfg: TwoStageConfig = TwoStageConfig()):
        super().__init__()
        self.layout = layout
        self.cfg = cfg
        self.root = RootDenoiser(layout, cfg.root_d, cfg.root_layers, cfg.root_heads)
        self.body = BodyDenoiser(layout, cfg.body_d, cfg.body_layers, cfg.body_heads)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return predicted clean x0 concatenated [root_pred, body_pred]."""
        root_pred = self.root(x_t, t)
        body_pred = self.body(x_t, root_pred, t)
        return torch.cat([root_pred, body_pred], dim=-1)

    def num_params(self) -> Tuple[int, int]:
        rp = sum(p.numel() for p in self.root.parameters())
        bp = sum(p.numel() for p in self.body.parameters())
        return rp, bp
