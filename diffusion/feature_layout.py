from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch


def _tuple_ints(values: Optional[Sequence[int]]) -> Tuple[int, ...]:
    if values is None:
        return ()
    return tuple(int(v) for v in values)


@dataclass(frozen=True)
class MotionFeatureLayout:
    layout_version: str
    root_pos_mode: str
    rot6d_start: int
    contact_indices: Tuple[int, ...]
    foot_names: Tuple[str, ...] = ()
    root_pos_indices: Tuple[int, int, int] = (0, 1, 2)
    legacy_root_indices: Tuple[int, int, int] = (0, 1, 2)
    yaw_index: Optional[int] = None
    local_vel_xz_indices: Optional[Tuple[int, int]] = None
    yaw_vel_index: Optional[int] = None

    @property
    def contact_dim(self) -> int:
        return len(self.contact_indices)

    def decode_root_pos(self, feat: torch.Tensor, dt: float) -> torch.Tensor:
        if feat.ndim < 2:
            raise ValueError(f"Expected motion features with ndim >= 2, got {feat.shape}")

        if self.root_pos_mode == "absolute_xyz":
            return feat[..., list(self.root_pos_indices)]

        vel_x = feat[..., self.legacy_root_indices[0]]
        pos_y = feat[..., self.legacy_root_indices[1]]
        vel_z = feat[..., self.legacy_root_indices[2]]
        pos_x = torch.cumsum(vel_x * float(dt), dim=-1)
        pos_z = torch.cumsum(vel_z * float(dt), dim=-1)
        return torch.stack([pos_x, pos_y, pos_z], dim=-1)

    def describe(self) -> str:
        return (
            f"{self.layout_version} root={self.root_pos_mode} "
            f"rot6d_start={self.rot6d_start} contact_dim={self.contact_dim}"
        )

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "layout_version": self.layout_version,
            "root_pos_mode": self.root_pos_mode,
            "rot6d_start": int(self.rot6d_start),
            "contact_indices": list(self.contact_indices),
            "foot_names": list(self.foot_names),
            "root_pos_indices": list(self.root_pos_indices),
            "legacy_root_indices": list(self.legacy_root_indices),
            "yaw_index": None if self.yaw_index is None else int(self.yaw_index),
            "local_vel_xz_indices": None if self.local_vel_xz_indices is None else list(self.local_vel_xz_indices),
            "yaw_vel_index": None if self.yaw_vel_index is None else int(self.yaw_vel_index),
        }

    @classmethod
    def from_metadata(
        cls,
        meta: Optional[Dict[str, Any]],
        *,
        motion_dim: int,
        rot6d_start: Optional[int] = None,
        contact_indices: Optional[Sequence[int]] = None,
    ) -> "MotionFeatureLayout":
        meta = dict(meta or {})

        rot6d = int(meta.get("rot6d_start", rot6d_start if rot6d_start is not None else 8))
        if rot6d <= 0 or rot6d >= int(motion_dim):
            raise ValueError(f"Invalid rot6d_start={rot6d} for motion_dim={motion_dim}")

        contact_idx = _tuple_ints(meta.get("contact_indices", contact_indices))
        if not contact_idx:
            if "contact_dim" in meta:
                contact_dim = max(0, int(meta.get("contact_dim", 0)))
            else:
                root_dim = int(meta.get("root_dim", 0) or 0)
                if root_dim == 6 and bool(meta.get("has_yaw_vel", False)):
                    contact_dim = max(0, rot6d - 7)
                else:
                    contact_dim = 4 if rot6d >= 8 else max(0, rot6d - 4)
            start = max(0, rot6d - contact_dim)
            contact_idx = tuple(range(start, start + contact_dim))

        layout_version = str(meta.get("layout_version", "") or "")
        root_pos_mode = str(meta.get("root_pos_mode", "") or "")

        if not root_pos_mode:
            root_dim = int(meta.get("root_dim", 0) or 0)
            has_yaw_vel = bool(meta.get("has_yaw_vel", False))
            if root_dim == 6 and has_yaw_vel and rot6d == (7 + len(contact_idx)):
                root_pos_mode = "absolute_xyz"
                if not layout_version:
                    layout_version = "root_pos_abs_v2"
            else:
                root_pos_mode = "legacy_vel_y"
                if not layout_version:
                    layout_version = "legacy_v1"

        root_pos_indices = meta.get("root_pos_indices")
        if root_pos_indices is None:
            root_pos_indices = (0, 1, 2)

        legacy_root_indices = meta.get("legacy_root_indices")
        if legacy_root_indices is None:
            legacy_root_indices = (0, 1, 2)

        local_vel_xz_indices = meta.get("local_vel_xz_indices")
        if local_vel_xz_indices is None and root_pos_mode == "absolute_xyz":
            local_vel_xz_indices = (4, 5)

        yaw_index = meta.get("yaw_index")
        if yaw_index is None and root_pos_mode == "absolute_xyz":
            yaw_index = 3

        yaw_vel_index = meta.get("yaw_vel_index")
        if yaw_vel_index is None and root_pos_mode == "absolute_xyz":
            yaw_vel_index = 6

        foot_names = tuple(str(x) for x in meta.get("foot_names", []) or [])

        return cls(
            layout_version=layout_version,
            root_pos_mode=root_pos_mode,
            rot6d_start=rot6d,
            contact_indices=tuple(int(v) for v in contact_idx),
            foot_names=foot_names,
            root_pos_indices=tuple(int(v) for v in root_pos_indices),
            legacy_root_indices=tuple(int(v) for v in legacy_root_indices),
            yaw_index=None if yaw_index is None else int(yaw_index),
            local_vel_xz_indices=None if local_vel_xz_indices is None else tuple(int(v) for v in local_vel_xz_indices),
            yaw_vel_index=None if yaw_vel_index is None else int(yaw_vel_index),
        )
