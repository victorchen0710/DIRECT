"""DDPM x0-prediction wrapper + cosine noise schedule.

Follows Kimodo: T=1000 training, DDIM 100-step inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


def _cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    t = torch.linspace(0, T, steps, dtype=torch.float64) / T
    alphas_bar = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return betas.clamp(1e-5, 0.999).float()


@dataclass
class DDPMSchedule:
    T: int
    betas: torch.Tensor             # [T]
    alphas: torch.Tensor            # [T]
    alphas_bar: torch.Tensor        # [T]
    sqrt_alphas_bar: torch.Tensor   # [T]
    sqrt_one_minus_alphas_bar: torch.Tensor   # [T]

    def to(self, device) -> "DDPMSchedule":
        return DDPMSchedule(
            T=self.T,
            betas=self.betas.to(device),
            alphas=self.alphas.to(device),
            alphas_bar=self.alphas_bar.to(device),
            sqrt_alphas_bar=self.sqrt_alphas_bar.to(device),
            sqrt_one_minus_alphas_bar=self.sqrt_one_minus_alphas_bar.to(device),
        )


def make_ddpm_schedule(T: int = 1000) -> DDPMSchedule:
    betas = _cosine_beta_schedule(T)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return DDPMSchedule(
        T=T,
        betas=betas,
        alphas=alphas,
        alphas_bar=alphas_bar,
        sqrt_alphas_bar=torch.sqrt(alphas_bar),
        sqrt_one_minus_alphas_bar=torch.sqrt(1.0 - alphas_bar),
    )


def q_sample(x0: torch.Tensor, t: torch.Tensor, sched: DDPMSchedule, noise: torch.Tensor | None = None) -> torch.Tensor:
    """x_t = sqrt(alphas_bar[t]) * x0 + sqrt(1 - alphas_bar[t]) * noise.

    x0 shape [B, T, D]; t shape [B] int64.
    """
    if noise is None:
        noise = torch.randn_like(x0)
    sab = sched.sqrt_alphas_bar[t].view(-1, 1, 1)
    s1mab = sched.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1)
    return sab * x0 + s1mab * noise


@torch.no_grad()
def ddim_sample(
    model_x0_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    shape: tuple,
    sched: DDPMSchedule,
    n_steps: int = 100,
    device: str = "cuda",
    eta: float = 0.0,
) -> torch.Tensor:
    """Run DDIM sampling with x0-prediction.

    model_x0_fn(x_t, t) -> x0_hat.  x_t and x0 have shape `shape`.
    Returns x_0 at last step.
    """
    T = sched.T
    ts = torch.linspace(T - 1, 0, n_steps + 1).long().to(device)
    x = torch.randn(*shape, device=device)
    for i in range(n_steps):
        t = ts[i]
        t_next = ts[i + 1]
        t_batch = torch.full((shape[0],), int(t), device=device, dtype=torch.long)
        x0_hat = model_x0_fn(x, t_batch)

        ab_t = sched.alphas_bar[t]
        ab_next = sched.alphas_bar[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)
        # derive noise estimate from x0_hat
        noise_est = (x - torch.sqrt(ab_t) * x0_hat) / torch.sqrt(1.0 - ab_t).clamp_min(1e-8)
        # DDIM deterministic step (eta=0)
        dir_xt = torch.sqrt((1.0 - ab_next).clamp_min(0.0)) * noise_est
        x = torch.sqrt(ab_next) * x0_hat + dir_xt
    return x
