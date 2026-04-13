"""Pack / unpack Kimodo's 6-item representation into a flat per-frame vector.

Layout (in order): r_p(3) | r_a(2) | j_p(3J) | j_v(3J) | j_a(6J) | f(n_foot)

Normalization:
- r_p / j_p / j_v / f : z-score with mean/std from cache
- r_a / j_a : not normalized (on unit sphere)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

from .loss import KimodoMotion


@dataclass
class FeatureLayout:
    n_joints: int
    n_foot: int

    @property
    def r_p_dim(self) -> int: return 3
    @property
    def r_a_dim(self) -> int: return 2
    @property
    def j_p_dim(self) -> int: return 3 * self.n_joints
    @property
    def j_v_dim(self) -> int: return 3 * self.n_joints
    @property
    def j_a_dim(self) -> int: return 6 * self.n_joints
    @property
    def f_dim(self) -> int: return self.n_foot

    @property
    def total_dim(self) -> int:
        return self.r_p_dim + self.r_a_dim + self.j_p_dim + self.j_v_dim + self.j_a_dim + self.f_dim

    @property
    def root_dim(self) -> int:
        """Subspace handled by RootDenoiser: r_p + r_a."""
        return self.r_p_dim + self.r_a_dim

    @property
    def body_dim(self) -> int:
        return self.total_dim - self.root_dim

    def slices(self) -> Dict[str, slice]:
        o = 0
        out: Dict[str, slice] = {}
        for name, d in [("r_p", self.r_p_dim), ("r_a", self.r_a_dim),
                        ("j_p", self.j_p_dim), ("j_v", self.j_v_dim),
                        ("j_a", self.j_a_dim), ("f", self.f_dim)]:
            out[name] = slice(o, o + d)
            o += d
        return out


@dataclass
class NormStats:
    mean: Dict[str, torch.Tensor]   # keys: r_p, j_p, j_v, f
    std: Dict[str, torch.Tensor]    # same keys

    def to(self, device) -> "NormStats":
        return NormStats(
            mean={k: v.to(device) for k, v in self.mean.items()},
            std={k: v.to(device) for k, v in self.std.items()},
        )


def normalize(motion: KimodoMotion, stats: NormStats) -> KimodoMotion:
    return KimodoMotion(
        r_p=(motion.r_p - stats.mean["r_p"]) / stats.std["r_p"],
        r_a=motion.r_a,  # not normalized
        j_p=(motion.j_p - stats.mean["j_p"]) / stats.std["j_p"],
        j_v=(motion.j_v - stats.mean["j_v"]) / stats.std["j_v"],
        j_a=motion.j_a,  # not normalized
        f=(motion.f - stats.mean["f"]) / stats.std["f"],
    )


def denormalize(motion: KimodoMotion, stats: NormStats) -> KimodoMotion:
    return KimodoMotion(
        r_p=motion.r_p * stats.std["r_p"] + stats.mean["r_p"],
        r_a=motion.r_a,
        j_p=motion.j_p * stats.std["j_p"] + stats.mean["j_p"],
        j_v=motion.j_v * stats.std["j_v"] + stats.mean["j_v"],
        j_a=motion.j_a,
        f=motion.f * stats.std["f"] + stats.mean["f"],
    )


def pack(motion: KimodoMotion, layout: FeatureLayout) -> torch.Tensor:
    """Concatenate fields into [B, T, total_dim]."""
    return torch.cat([motion.r_p, motion.r_a, motion.j_p, motion.j_v, motion.j_a, motion.f], dim=-1)


def unpack(x: torch.Tensor, layout: FeatureLayout) -> KimodoMotion:
    s = layout.slices()
    return KimodoMotion(
        r_p=x[..., s["r_p"]],
        r_a=x[..., s["r_a"]],
        j_p=x[..., s["j_p"]],
        j_v=x[..., s["j_v"]],
        j_a=x[..., s["j_a"]],
        f=x[..., s["f"]],
    )


def build_norm_stats(cache_payload: dict) -> NormStats:
    mean_in = cache_payload["mean"]
    std_in = cache_payload["std"]
    def tcvt(d):
        return {k: torch.from_numpy(np.asarray(v)).float() for k, v in d.items()}
    return NormStats(mean=tcvt(mean_in), std=tcvt(std_in))
