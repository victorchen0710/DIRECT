"""Kimodo-style 7-term loss.

Weights (from Kimodo paper, Table / loss def):
  root_pos      γ1 = 10
  root_heading  γ2 =  2
  joint_pos     γ3 = 10
  joint_vel     γ4 =  3
  joint_rot     γ5 = 10
  foot_contact  γ6 =  4
  fk_consistency γ7 = 5  (||FK(ĵ_a, r̂_p) - j_p_target||)

All terms use Smooth L1. x₀-prediction: `pred` and `target` are the noiseless
signal reconstructions, not noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .fk import SkeletonInfoT, fk_positions_from_global_rot


@dataclass
class KimodoMotion:
    """Bundle of the 6 per-frame feature tensors plus j_p (used by FK loss target)."""

    r_p: torch.Tensor   # [B, T, 3]
    r_a: torch.Tensor   # [B, T, 2]
    j_p: torch.Tensor   # [B, T, J*3]   (xz relative to smoothed root, y global)
    j_v: torch.Tensor   # [B, T, J*3]
    j_a: torch.Tensor   # [B, T, J*6]
    f: torch.Tensor     # [B, T, n_foot]


@dataclass
class KimodoLossWeights:
    root_pos: float = 10.0
    root_heading: float = 2.0
    joint_pos: float = 10.0
    joint_vel: float = 3.0
    joint_rot: float = 10.0
    foot: float = 4.0
    fk: float = 5.0


def _smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Smooth L1 with optional per-frame mask.

    mask shape [B, T] or [B, T, 1]; if provided, elementwise loss is multiplied
    by mask then averaged over unmasked elements.
    """
    elem = F.smooth_l1_loss(pred, target, reduction="none", beta=1.0)
    if mask is None:
        return elem.mean()
    while mask.dim() < elem.dim():
        mask = mask.unsqueeze(-1)
    weighted = elem * mask
    denom = mask.expand_as(elem).sum().clamp_min(1.0)
    return weighted.sum() / denom


def kimodo_loss(
    pred: KimodoMotion,
    target: KimodoMotion,
    *,
    skel: SkeletonInfoT,
    j_p_target_for_fk: torch.Tensor,   # [B, T, J, 3] — target global joint positions in world frame
    root_pos_target_world: torch.Tensor,   # [B, T, 3] — world-frame smoothed root (since j_p stored is xz-relative)
    weights: KimodoLossWeights = KimodoLossWeights(),
    frame_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Compute Kimodo 7-term loss.

    Returns (total_loss, per_term_dict).

    Note on FK term: j_p stored in cache is xz-relative to smoothed root, with y
    global. The FK we run operates in world frame (on absolute root position +
    global rotations). To compare FK output to the target joint positions, we
    either (a) remap FK output back to the same xz-relative frame, or (b) use
    the absolute target in world frame. We take (b) and pass `j_p_target_for_fk`
    explicitly (world-frame target).
    """
    B, T, _ = pred.r_p.shape
    J = skel.n_joints

    losses: Dict[str, torch.Tensor] = {}

    losses["root_pos"] = weights.root_pos * _smooth_l1(pred.r_p, target.r_p, frame_mask)
    losses["root_heading"] = weights.root_heading * _smooth_l1(pred.r_a, target.r_a, frame_mask)
    losses["joint_pos"] = weights.joint_pos * _smooth_l1(pred.j_p, target.j_p, frame_mask)
    losses["joint_vel"] = weights.joint_vel * _smooth_l1(pred.j_v, target.j_v, frame_mask)
    losses["joint_rot"] = weights.joint_rot * _smooth_l1(pred.j_a, target.j_a, frame_mask)
    losses["foot"] = weights.foot * _smooth_l1(pred.f, target.f, frame_mask)

    if weights.fk > 0.0:
        # reshape pred rotations and root to the shapes FK wants
        pred_j_a = pred.j_a.view(B, T, J, 6)
        fk_pos = fk_positions_from_global_rot(pred_j_a, root_pos_target_world, skel)
        # Note: we intentionally condition FK on the *target* root position here,
        # so that rotation error alone drives the FK term — this matches Kimodo's
        # intent of enforcing rotation/position consistency, not compounding root
        # error. If you want end-to-end FK you can swap to pred.r_p.
        losses["fk"] = weights.fk * _smooth_l1(fk_pos, j_p_target_for_fk, frame_mask)

    total = sum(losses.values())
    return total, {k: float(v.detach().item()) for k, v in losses.items()}
