from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.diffusion_policy import rot6d_to_matrix
from diffusion.feature_layout import MotionFeatureLayout


def resolve_joint_indices(all_names: Sequence[str], keywords: Iterable[str]) -> List[int]:
    found: List[int] = []
    for idx, name in enumerate(all_names):
        low = str(name).lower()
        if low.endswith("_end") or "endsite" in low:
            continue
        if any(key.lower() in low for key in keywords):
            found.append(idx)
    return found


class SkeletonFK(nn.Module):
    def __init__(
        self,
        offsets: Sequence[Sequence[float]],
        parents: Sequence[int],
        all_names: Sequence[str],
        feature_names: Sequence[str],
    ):
        super().__init__()
        offsets_np = np.asarray(offsets, dtype=np.float32)
        bone_len = np.linalg.norm(offsets_np, axis=1)
        non_zero = bone_len[bone_len > 1e-6]
        self.offset_scale = 1.0
        if non_zero.size > 0:
            if float(np.max(non_zero)) > 10.0 and float(np.median(non_zero)) > 1.0:
                self.offset_scale = 0.01
                offsets_np = offsets_np * self.offset_scale
        self.register_buffer("offsets", torch.tensor(offsets_np, dtype=torch.float32))
        self.register_buffer("parents", torch.tensor(parents, dtype=torch.long))
        feat_map = {str(name): idx for idx, name in enumerate(feature_names)}
        mapping = []
        for name in all_names:
            if name in feat_map:
                mapping.append(feat_map[name])
            else:
                mapping.append(feat_map.get(str(name).replace("_End", ""), -1))
        self.register_buffer("map_s2f", torch.tensor(mapping, dtype=torch.long))
        self.all_names = list(all_names)

    def forward(self, rot_mats_feat: torch.Tensor, root_pos: torch.Tensor) -> torch.Tensor:
        bsz, steps, _, _, _ = rot_mats_feat.shape
        identity = torch.eye(3, device=rot_mats_feat.device, dtype=rot_mats_feat.dtype)
        identity = identity.view(1, 1, 3, 3).expand(bsz, steps, 3, 3)

        global_rots = [None] * int(self.parents.numel())
        global_pos = [None] * int(self.parents.numel())
        for idx in range(int(self.parents.numel())):
            parent = int(self.parents[idx].item())
            feat_idx = int(self.map_s2f[idx].item())
            local_r = rot_mats_feat[:, :, feat_idx] if feat_idx >= 0 else identity
            if parent == -1:
                global_rots[idx] = local_r
                global_pos[idx] = root_pos
            else:
                parent_r = global_rots[parent]
                parent_p = global_pos[parent]
                global_rots[idx] = parent_r @ local_r
                offset = self.offsets[idx].view(1, 1, 3, 1)
                offset_world = (parent_r @ offset).squeeze(-1)
                global_pos[idx] = parent_p + offset_world
        return torch.stack(global_pos, dim=2)


@dataclass
class MotionLossContext:
    layout: MotionFeatureLayout
    motion_mean: torch.Tensor
    motion_std: torch.Tensor
    fk_module: Optional[SkeletonFK]
    foot_indices: Optional[torch.Tensor]
    hand_indices: Optional[torch.Tensor]
    upper_body_indices: Optional[torch.Tensor]
    fps: int


def build_motion_loss_context(
    *,
    motion_contract,
    mean: Sequence[float] | np.ndarray,
    std: Sequence[float] | np.ndarray,
    device: torch.device,
) -> MotionLossContext:
    meta = motion_contract.layout_meta
    fk_module = None
    foot_indices = None
    hand_indices = None
    upper_body_indices = None
    joint_names = list(meta.get("joint_names") or [])
    if joint_names:
        upper_body_list = resolve_joint_indices(
            joint_names,
            [
                "Spine",
                "Neck",
                "Head",
                "Shoulder",
                "Arm",
                "ForeArm",
                "Hand",
                "Thumb",
                "Index",
                "Middle",
                "Ring",
                "Pinky",
                "Finger",
                "Wrist",
                "Clav",
            ],
        )
        if upper_body_list:
            upper_body_indices = torch.tensor(sorted(set(upper_body_list)), device=device, dtype=torch.long)
    if meta.get("joint_names") and meta.get("all_joint_names") and meta.get("skeleton_offsets") is not None:
        fk_module = SkeletonFK(
            offsets=meta["skeleton_offsets"],
            parents=meta["skeleton_parents"],
            all_names=meta["all_joint_names"],
            feature_names=meta["joint_names"],
        ).to(device)
        all_names = meta["all_joint_names"]
        foot_keys = list(motion_contract.foot_names) or ["RightFoot", "LeftFoot", "RightToe", "LeftToe"]
        foot_list = resolve_joint_indices(all_names, foot_keys)
        hand_list = resolve_joint_indices(
            all_names,
            ["Hand", "ForeArm", "Thumb", "Index", "Middle", "Ring", "Pinky", "Finger"],
        )
        if foot_list:
            foot_indices = torch.tensor(sorted(set(foot_list)), device=device, dtype=torch.long)
        if hand_list:
            hand_indices = torch.tensor(sorted(set(hand_list)), device=device, dtype=torch.long)
    return MotionLossContext(
        layout=motion_contract.feature_layout,
        motion_mean=torch.as_tensor(mean, dtype=torch.float32, device=device),
        motion_std=torch.as_tensor(std, dtype=torch.float32, device=device),
        fk_module=fk_module,
        foot_indices=foot_indices,
        hand_indices=hand_indices,
        upper_body_indices=upper_body_indices,
        fps=motion_contract.fps,
    )


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.float()
    while weight.ndim < pred.ndim:
        weight = weight.unsqueeze(-1)
    denom = weight.sum() * max(1, int(np.prod(pred.shape[mask.ndim :])))
    return (torch.abs(pred - target) * weight).sum() / (denom + 1e-6)


def _masked_bce_with_logits(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    weight = mask.float()
    while weight.ndim < pred.ndim:
        weight = weight.unsqueeze(-1)
    denom = weight.sum() * max(1, int(np.prod(pred.shape[mask.ndim :])))
    return (loss * weight).sum() / (denom + 1e-6)


def _root_relative_to_first_valid(root_xyz: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if root_xyz.ndim != 3:
        raise ValueError(f"Expected root_xyz with shape [B, T, C], got {tuple(root_xyz.shape)}")
    if mask.ndim != 2:
        raise ValueError(f"Expected mask with shape [B, T], got {tuple(mask.shape)}")

    first_valid = mask.float().argmax(dim=1)
    batch_idx = torch.arange(root_xyz.shape[0], device=root_xyz.device)
    anchor = root_xyz[batch_idx, first_valid].unsqueeze(1)
    return root_xyz - anchor


def compute_motion_recon_losses(
    *,
    pred_motion_norm: torch.Tensor,
    motion_denorm: torch.Tensor,
    motion_mask: torch.Tensor,
    context: MotionLossContext,
    recon_weight: float = 1.0,
    root_weight: float = 1.0,
    root_relative_first_frame: bool = False,
    contact_weight: float = 1.0,
    rot_weight: float = 0.0,
    upper_rot_weight: float = 0.0,
    upper_rot_vel_weight: float = 0.0,
    upper_rot_speed_weight: float = 0.0,
    upper_rot_acc_weight: float = 0.0,
    upper_rot_jerk_weight: float = 0.0,
    vel_weight: float = 0.1,
    acc_weight: float = 0.05,
    foot_fk_weight: float = 0.05,
    hand_fk_weight: float = 0.05,
) -> Dict[str, torch.Tensor]:
    pred_denorm = pred_motion_norm * context.motion_std.view(1, 1, -1) + context.motion_mean.view(1, 1, -1)
    total = pred_motion_norm.new_tensor(0.0)
    losses: Dict[str, torch.Tensor] = {}

    recon = _masked_l1(pred_denorm, motion_denorm, motion_mask)
    total = total + float(recon_weight) * recon
    losses["recon"] = recon

    root_idx = list(context.layout.root_pos_indices)
    pred_root = pred_denorm[..., root_idx]
    true_root = motion_denorm[..., root_idx]
    if root_relative_first_frame:
        pred_root = _root_relative_to_first_valid(pred_root, motion_mask)
        true_root = _root_relative_to_first_valid(true_root, motion_mask)
    root_recon = _masked_l1(pred_root, true_root, motion_mask)
    total = total + float(root_weight) * root_recon
    losses["root"] = root_recon

    if context.layout.contact_indices:
        contact_gt = motion_denorm[..., list(context.layout.contact_indices)].clamp(0.0, 1.0)
        contact_pd = pred_denorm[..., list(context.layout.contact_indices)]
        contact = _masked_bce_with_logits(contact_pd, contact_gt, motion_mask)
    else:
        contact = total.new_tensor(0.0)
    total = total + float(contact_weight) * contact
    losses["contact"] = contact

    rot = total.new_tensor(0.0)
    upper_rot = total.new_tensor(0.0)
    rot6d_start = int(context.layout.rot6d_start)
    joints = (pred_denorm.shape[-1] - rot6d_start) // 6
    pred_rot = pred_denorm[..., rot6d_start:].view(pred_denorm.shape[0], pred_denorm.shape[1], joints, 6)
    true_rot = motion_denorm[..., rot6d_start:].view(motion_denorm.shape[0], motion_denorm.shape[1], joints, 6)
    if joints > 0:
        rot = _masked_l1(pred_rot, true_rot, motion_mask)
        total = total + float(rot_weight) * rot
        if context.upper_body_indices is not None and context.upper_body_indices.numel() > 0:
            upper_rot = _masked_l1(
                pred_rot[:, :, context.upper_body_indices],
                true_rot[:, :, context.upper_body_indices],
                motion_mask,
            )
            total = total + float(upper_rot_weight) * upper_rot
    losses["rot"] = rot
    losses["upper_rot"] = upper_rot

    upper_rot_vel = total.new_tensor(0.0)
    upper_rot_speed = total.new_tensor(0.0)
    upper_rot_acc = total.new_tensor(0.0)
    upper_rot_jerk = total.new_tensor(0.0)
    if (
        joints > 0
        and motion_denorm.shape[1] >= 2
        and context.upper_body_indices is not None
        and context.upper_body_indices.numel() > 0
    ):
        pred_upper_vel = pred_rot[:, 1:, context.upper_body_indices] - pred_rot[:, :-1, context.upper_body_indices]
        true_upper_vel = true_rot[:, 1:, context.upper_body_indices] - true_rot[:, :-1, context.upper_body_indices]
        upper_vel_mask = motion_mask[:, 1:] & motion_mask[:, :-1]
        upper_rot_vel = _masked_l1(pred_upper_vel, true_upper_vel, upper_vel_mask)
        total = total + float(upper_rot_vel_weight) * upper_rot_vel
        pred_upper_speed = torch.linalg.norm(pred_upper_vel, dim=-1)
        true_upper_speed = torch.linalg.norm(true_upper_vel, dim=-1)
        upper_rot_speed = _masked_l1(pred_upper_speed, true_upper_speed, upper_vel_mask)
        total = total + float(upper_rot_speed_weight) * upper_rot_speed
        if motion_denorm.shape[1] >= 3:
            pred_upper_acc = pred_upper_vel[:, 1:] - pred_upper_vel[:, :-1]
            true_upper_acc = true_upper_vel[:, 1:] - true_upper_vel[:, :-1]
            upper_acc_mask = upper_vel_mask[:, 1:] & upper_vel_mask[:, :-1]
            upper_rot_acc = _masked_l1(pred_upper_acc, true_upper_acc, upper_acc_mask)
            total = total + float(upper_rot_acc_weight) * upper_rot_acc
            if motion_denorm.shape[1] >= 4:
                pred_upper_jerk = pred_upper_acc[:, 1:] - pred_upper_acc[:, :-1]
                true_upper_jerk = true_upper_acc[:, 1:] - true_upper_acc[:, :-1]
                upper_jerk_mask = upper_acc_mask[:, 1:] & upper_acc_mask[:, :-1]
                upper_rot_jerk = _masked_l1(pred_upper_jerk, true_upper_jerk, upper_jerk_mask)
                total = total + float(upper_rot_jerk_weight) * upper_rot_jerk
    losses["upper_rot_vel"] = upper_rot_vel
    losses["upper_rot_speed"] = upper_rot_speed
    losses["upper_rot_acc"] = upper_rot_acc
    losses["upper_rot_jerk"] = upper_rot_jerk

    vel = total.new_tensor(0.0)
    if motion_denorm.shape[1] >= 2:
        pred_vel = pred_denorm[:, 1:] - pred_denorm[:, :-1]
        true_vel = motion_denorm[:, 1:] - motion_denorm[:, :-1]
        vel_mask = motion_mask[:, 1:] & motion_mask[:, :-1]
        vel = _masked_l1(pred_vel, true_vel, vel_mask)
        total = total + float(vel_weight) * vel
    losses["vel"] = vel

    acc = total.new_tensor(0.0)
    if motion_denorm.shape[1] >= 3:
        pred_acc = pred_denorm[:, 2:] - 2.0 * pred_denorm[:, 1:-1] + pred_denorm[:, :-2]
        true_acc = motion_denorm[:, 2:] - 2.0 * motion_denorm[:, 1:-1] + motion_denorm[:, :-2]
        acc_mask = motion_mask[:, 2:] & motion_mask[:, 1:-1] & motion_mask[:, :-2]
        acc = _masked_l1(pred_acc, true_acc, acc_mask)
        total = total + float(acc_weight) * acc
    losses["acc"] = acc

    foot_fk = total.new_tensor(0.0)
    hand_fk = total.new_tensor(0.0)
    if context.fk_module is not None:
        pred_root_world = context.layout.decode_root_pos(pred_denorm, 1.0 / max(1, context.fps))
        true_root_world = context.layout.decode_root_pos(motion_denorm, 1.0 / max(1, context.fps))
        pred_pos = context.fk_module(rot6d_to_matrix(pred_rot), pred_root_world)
        true_pos = context.fk_module(rot6d_to_matrix(true_rot), true_root_world)

        if context.foot_indices is not None and context.foot_indices.numel() > 0:
            foot_fk = _masked_l1(pred_pos[:, :, context.foot_indices], true_pos[:, :, context.foot_indices], motion_mask)
            total = total + float(foot_fk_weight) * foot_fk
        if context.hand_indices is not None and context.hand_indices.numel() > 0:
            hand_fk = _masked_l1(pred_pos[:, :, context.hand_indices], true_pos[:, :, context.hand_indices], motion_mask)
            total = total + float(hand_fk_weight) * hand_fk
    losses["foot_fk"] = foot_fk
    losses["hand_fk"] = hand_fk
    losses["loss"] = total
    return losses


def compute_vae_losses(
    *,
    recon_motion_norm: torch.Tensor,
    motion_norm: torch.Tensor,
    motion_denorm: torch.Tensor,
    motion_mask: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    context: MotionLossContext,
    kl_weight: float,
    recon_weight: float = 1.0,
    root_weight: float = 1.0,
    root_relative_first_frame: bool = False,
    contact_weight: float = 1.0,
    rot_weight: float = 0.0,
    upper_rot_weight: float = 0.0,
    upper_rot_vel_weight: float = 0.0,
    upper_rot_speed_weight: float = 0.0,
    upper_rot_acc_weight: float = 0.0,
    upper_rot_jerk_weight: float = 0.0,
    vel_weight: float = 0.1,
    acc_weight: float = 0.05,
    foot_fk_weight: float = 0.05,
    hand_fk_weight: float = 0.05,
) -> Dict[str, torch.Tensor]:
    losses = compute_motion_recon_losses(
        pred_motion_norm=recon_motion_norm,
        motion_denorm=motion_denorm,
        motion_mask=motion_mask,
        context=context,
        recon_weight=recon_weight,
        root_weight=root_weight,
        root_relative_first_frame=root_relative_first_frame,
        contact_weight=contact_weight,
        rot_weight=rot_weight,
        upper_rot_weight=upper_rot_weight,
        upper_rot_vel_weight=upper_rot_vel_weight,
        upper_rot_speed_weight=upper_rot_speed_weight,
        upper_rot_acc_weight=upper_rot_acc_weight,
        upper_rot_jerk_weight=upper_rot_jerk_weight,
        vel_weight=vel_weight,
        acc_weight=acc_weight,
        foot_fk_weight=foot_fk_weight,
        hand_fk_weight=hand_fk_weight,
    )
    total = losses["loss"]
    kl = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
    total = total + float(kl_weight) * kl
    losses["kl"] = kl
    losses["loss"] = total
    return losses
