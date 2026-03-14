import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import nullcontext
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R


TARGET_FPS = 30
IDENTITY_6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
LEGACY_PART_MODEL_DEFAULTS = {
    "upper": {"code_dim": 256, "n_codes": 512, "fk_hand_pos_w": 1.0, "fk_hand_vel_w": 0.5},
    "hand": {"code_dim": 256, "n_codes": 512, "fk_hand_pos_w": 1.0, "fk_hand_vel_w": 0.5},
    "lower": {"code_dim": 128, "n_codes": 512, "fk_foot_lock_w": 1.0},
    "global": {"hidden": 256, "code_dim": 256, "n_codes": 256, "n_downsample": 0, "global_ae_layers": 4},
}
LOM_OFFICIAL_PART_MODEL_DEFAULTS = {
    "upper": {"hidden": 256, "code_dim": 256, "n_codes": 256, "fk_hand_pos_w": 1.0, "fk_hand_vel_w": 0.5},
    "hand": {"hidden": 256, "code_dim": 256, "n_codes": 256, "fk_hand_pos_w": 1.0, "fk_hand_vel_w": 0.5},
    "lower": {"hidden": 256, "code_dim": 256, "n_codes": 256, "fk_foot_lock_w": 1.0},
    "global": {"hidden": 256, "code_dim": 256, "n_codes": 256, "n_downsample": 0, "global_ae_layers": 4},
}
LOM_README_PART_MODEL_DEFAULTS = {
    "upper": {"hidden": 256, "code_dim": 256, "n_codes": 512, "fk_hand_pos_w": 1.0, "fk_hand_vel_w": 0.5},
    "hand": {"hidden": 256, "code_dim": 256, "n_codes": 512, "fk_hand_pos_w": 1.0, "fk_hand_vel_w": 0.5},
    "lower": {"hidden": 256, "code_dim": 128, "n_codes": 512, "fk_foot_lock_w": 1.0},
    "global": {"hidden": 256, "code_dim": 256, "n_codes": 256, "n_downsample": 0, "global_ae_layers": 4},
}
CONFIG_PROFILE_DEFAULTS = {
    "legacy": {},
    "lom_official": {
        "block_size": 64,
        "batch_size": 32,
        "num_workers": 4,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "n_downsample": 2,
        "hidden": 256,
        "vq_usage_entropy_w": 0.0,
        "vq_seq_usage_w": 0.0,
    },
    "lom_readme": {
        "block_size": 64,
        "batch_size": 32,
        "num_workers": 4,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "n_downsample": 2,
        "hidden": 256,
        "vq_usage_entropy_w": 0.0,
        "vq_seq_usage_w": 0.0,
    },
}
PROFILE_PART_MODEL_DEFAULTS = {
    "legacy": LEGACY_PART_MODEL_DEFAULTS,
    "lom_official": LOM_OFFICIAL_PART_MODEL_DEFAULTS,
    "lom_readme": LOM_README_PART_MODEL_DEFAULTS,
}
DEFAULT_PART_KEYWORDS = {
    # Keep upper-body articulation in the upper branch while leaving forearm
    # and hand/fingers together so the distal arm-hand chain still co-varies.
    "upper": ["spine", "spine1", "spine2", "spine3", "neck", "head", "shoulder", "arm", "clavicle"],
    "hand": ["forearm", "hand", "thumb", "index", "middle", "ring", "pinky", "finger"],
    "lower": ["hips", "upleg", "leg", "foot", "toe", "toebase"],
}
DEFAULT_FOOT_CONTACT_KEYWORDS = ["RightFoot", "LeftFoot", "RightToeBase", "LeftToeBase"]

# -----------------------------
# 6D Rotation Helpers
# -----------------------------
def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Input: [..., 3, 3]
    Output: [..., 6]
    修正版：[Col1, Col2] 格式
    """
    return matrix[..., :2].transpose(-1, -2).flatten(-2)

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Input: [..., 6]
    Output: [..., 3, 3]
    标准版：因为输入已经是标准的 [Col1, Col2] 格式，
    所以直接切分前3和后3即可，不需要之前的 0::2 补丁了。
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)

# -----------------------------
# AMP compatibility helpers
# -----------------------------
def make_autocast(device: torch.device):
    """
    Returns autocast context manager factory compatible with:
    - old: torch.cuda.amp.autocast(enabled=...)
    - new: torch.amp.autocast(device_type="cuda", enabled=...)
    """
    try:
        from torch.amp import autocast as amp_autocast  # type: ignore

        def _ctx(enabled: bool):
            if not enabled:
                return nullcontext()
            dev_type = "cuda" if device.type == "cuda" else "cpu"
            return amp_autocast(device_type=dev_type, enabled=True)

        return _ctx
    except Exception:
        from torch.cuda.amp import autocast as cuda_autocast  # type: ignore

        def _ctx(enabled: bool):
            return cuda_autocast(enabled=enabled)

        return _ctx


def make_grad_scaler(device: torch.device, enabled: bool):
    """
    GradScaler compatibility:
    - old: torch.cuda.amp.GradScaler(enabled=...)
    - new: torch.amp.GradScaler("cuda", enabled=...)
    """
    try:
        from torch.amp import GradScaler  # type: ignore
        dev_type = "cuda" if device.type == "cuda" else "cpu"
        return GradScaler(dev_type, enabled=enabled)
    except Exception:
        from torch.cuda.amp import GradScaler  # type: ignore
        return GradScaler(enabled=enabled)


# -----------------------------
# Repro
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# BVH utils
# -----------------------------
def load_bvh_channels(path: Path) -> tuple[np.ndarray, float]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    try:
        motion_idx = lines.index("MOTION")
    except ValueError:
        return None, None

    frame_time = float(lines[motion_idx + 2].split(":")[1].strip())
    data = []
    for ln in lines[motion_idx + 3:]:
        if not ln.strip():
            continue
        try:
            data.append([float(x) for x in ln.strip().split()])
        except ValueError:
            continue
    return np.asarray(data, dtype=np.float32), frame_time


def unwrap_bvh_angles_degrees(motion: np.ndarray, pos_dims: int = 3) -> np.ndarray:
    """
    BVH motion is typically:
      first 3 dims: root XYZ position (linear)
      remaining dims: Euler angles in degrees (circular, wrap at +/-180)
    We unwrap angle dims along time to eliminate 360-degree jumps.
    """
    if motion.ndim != 2 or motion.shape[0] < 2:
        return motion.astype(np.float32)

    m = motion.astype(np.float32, copy=True)
    D = m.shape[1]
    if D <= pos_dims:
        return m

    ang = m[:, pos_dims:]
    rad = np.deg2rad(ang)
    rad = np.unwrap(rad, axis=0)
    ang_unwrapped = np.rad2deg(rad).astype(np.float32)
    m[:, pos_dims:] = ang_unwrapped
    return m


def resample_motion_linear(motion: np.ndarray, frame_time: float, target_fps: int) -> np.ndarray:
    """
    Re-sample (not subsample) to avoid aliasing artifacts.
    Linear interpolation acts like a mild low-pass (triangular kernel).
    """
    if motion.ndim != 2 or len(motion) < 2:
        return motion.astype(np.float32)

    src_fps = 1.0 / (frame_time + 1e-6)
    T_src = motion.shape[0]
    t_src = np.arange(T_src, dtype=np.float32) / src_fps

    duration = float(t_src[-1])
    T_tgt = int(round(duration * float(target_fps))) + 1
    if T_tgt < 2:
        return motion[:1].astype(np.float32)

    t_tgt = np.arange(T_tgt, dtype=np.float32) / float(target_fps)

    out = np.empty((T_tgt, motion.shape[1]), dtype=np.float32)
    for d in range(motion.shape[1]):
        out[:, d] = np.interp(t_tgt, t_src, motion[:, d].astype(np.float32))
    return out


def _resolve(base_dir: Path, p: Union[str, Path]) -> Path:
    p = Path(p)

    # 1) absolute path
    if p.is_absolute():
        return p

    # 2) relative to current working directory
    cand = p
    if cand.exists():
        return cand

    # 3) relative to manifest directory
    cand = base_dir / p
    if cand.exists():
        return cand

    # 4) relative to parent of manifest directory
    cand = base_dir.parent / p
    if cand.exists():
        return cand

    # 5) fallback
    return base_dir / p


# -----------------------------
# TextGrid parsing (minimal, robust)
# -----------------------------
def _strip_quoted(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


def parse_textgrid_interval_tiers(path: Path) -> Dict[str, List[Tuple[float, float, str]]]:
    """
    Parse Praat TextGrid (long text format) and return:
      {tier_name: [(xmin, xmax, text), ...]} for IntervalTier only.
    """
    tiers: Dict[str, List[Tuple[float, float, str]]] = {}
    if path is None or (not path.exists()):
        return tiers

    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return tiers

    in_item = False
    is_interval_tier = False
    tier_name: Optional[str] = None

    in_interval = False
    cur_xmin: Optional[float] = None
    cur_xmax: Optional[float] = None
    cur_text: Optional[str] = None

    def _flush_interval():
        nonlocal cur_xmin, cur_xmax, cur_text, in_interval, tier_name
        if not is_interval_tier or tier_name is None:
            cur_xmin = cur_xmax = None
            cur_text = None
            in_interval = False
            return
        if cur_xmin is not None and cur_xmax is not None and cur_text is not None:
            tiers.setdefault(tier_name, []).append((float(cur_xmin), float(cur_xmax), str(cur_text)))
        cur_xmin = cur_xmax = None
        cur_text = None
        in_interval = False

    for raw in lines:
        ln = raw.strip()

        if ln.startswith("item [") and ln.endswith("]:"):
            if in_interval:
                _flush_interval()
            in_item = True
            is_interval_tier = False
            tier_name = None
            continue

        if not in_item:
            continue

        if ln.startswith("class") and "IntervalTier" in ln:
            is_interval_tier = True
            continue

        if ln.startswith("name") and "=" in ln:
            _, rhs = ln.split("=", 1)
            tier_name = _strip_quoted(rhs.strip())
            continue

        if is_interval_tier and ln.startswith("intervals [") and ln.endswith("]:"):
            if in_interval:
                _flush_interval()
            in_interval = True
            cur_xmin = cur_xmax = None
            cur_text = None
            continue

        if is_interval_tier and in_interval and ln.startswith("xmin") and "=" in ln:
            _, rhs = ln.split("=", 1)
            try:
                cur_xmin = float(rhs.strip())
            except Exception:
                cur_xmin = None
            continue

        if is_interval_tier and in_interval and ln.startswith("xmax") and "=" in ln:
            _, rhs = ln.split("=", 1)
            try:
                cur_xmax = float(rhs.strip())
            except Exception:
                cur_xmax = None
            continue

        if is_interval_tier and in_interval and ln.startswith("text") and "=" in ln:
            _, rhs = ln.split("=", 1)
            cur_text = _strip_quoted(rhs.strip())
            _flush_interval()
            continue

    if in_interval:
        _flush_interval()

    return tiers


def choose_best_tier(
    tiers: Dict[str, List[Tuple[float, float, str]]],
    prefer: Optional[str] = None
) -> Optional[str]:
    """
    Heuristic tier chooser:
    1) if prefer specified and exists -> use it
    2) tier name contains 'word' -> use it
    3) tier name contains 'phone' -> use it
    4) tier with max non-empty intervals
    """
    if not tiers:
        return None

    if prefer and prefer in tiers:
        return prefer

    names = list(tiers.keys())
    for key in names:
        if "word" in key.lower():
            return key
    for key in names:
        if "phone" in key.lower():
            return key

    best_name = None
    best_score = -1
    for name, intervals in tiers.items():
        non_empty = sum(1 for _, _, t in intervals if (t.strip() != ""))
        if non_empty > best_score:
            best_score = non_empty
            best_name = name
    return best_name


def build_speech_segments_from_intervals(
    intervals: List[Tuple[float, float, str]],
    fps: int,
    T_frames: int,
    merge_silence_s: float = 0.2,
    pad_s: float = 0.3,
) -> List[Tuple[int, int]]:
    """
    Build speech segments from IntervalTier intervals by treating text=="" as silence.

    - merge_silence_s: silence shorter than this will NOT break a segment
    - pad_s: expand each segment by +-pad_s seconds (clamped to [0,T])
    """
    if not intervals:
        return []

    merge_sil = float(merge_silence_s)
    pad = int(round(float(pad_s) * fps))

    segs: List[Tuple[int, int]] = []
    cur_a: Optional[int] = None
    cur_b: Optional[int] = None

    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))

    def _to_a(t: float) -> int:
        return int(math.floor(t * fps + 1e-6))

    def _to_b(t: float) -> int:
        return int(math.ceil(t * fps - 1e-6))

    def _close():
        nonlocal cur_a, cur_b
        if cur_a is not None and cur_b is not None and cur_b > cur_a:
            a = max(0, cur_a - pad)
            b = min(T_frames, cur_b + pad)
            if b > a:
                segs.append((a, b))
        cur_a = None
        cur_b = None

    for xmin, xmax, text in intervals:
        t = (text or "").strip()
        a = max(0, _to_a(xmin))
        b = min(T_frames, _to_b(xmax))
        if b <= a:
            continue

        is_speech = (t != "")
        dur_s = float(xmax - xmin)

        if is_speech:
            if cur_a is None:
                cur_a, cur_b = a, b
            else:
                cur_b = max(cur_b, b)
        else:
            if cur_a is None:
                continue
            if dur_s <= merge_sil:
                cur_b = max(cur_b, b)
            else:
                _close()

    _close()

    if not segs:
        return []
    segs.sort()
    merged = [segs[0]]
    for a, b in segs[1:]:
        pa, pb = merged[-1]
        if a <= pb:
            merged[-1] = (pa, max(pb, b))
        else:
            merged.append((a, b))
    return merged


# -----------------------------
# BVH Skeleton parsing + differentiable FK
# -----------------------------
@dataclass
class _JointDef:
    name: str
    parent: int
    offset: np.ndarray  # [3]
    channels: List[str]
    ch_start: int = 0   # start index in full motion vector


@dataclass
class BVHSkeleton:
    joints: List[_JointDef]
    channel_names: List[str]  # full motion channels order

    @staticmethod
    def from_bvh(path: Path) -> "BVHSkeleton":
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        joints: List[_JointDef] = []
        stack: List[int] = []
        cur: Optional[int] = None
        in_hierarchy = False

        def _add_joint(name: str, parent: int) -> int:
            j = _JointDef(name=name, parent=parent, offset=np.zeros(3, np.float32), channels=[])
            joints.append(j)
            return len(joints) - 1

        for raw in lines:
            ln = raw.strip()
            if ln == "HIERARCHY":
                in_hierarchy = True
                continue
            if not in_hierarchy:
                continue
            if ln == "MOTION":
                break

            toks = ln.split()
            if not toks:
                continue

            if toks[0] == "ROOT" or toks[0] == "JOINT":
                name = toks[1]
                parent = stack[-1] if stack else -1
                cur = _add_joint(name, parent)
                continue

            if toks[0] == "End" and len(toks) >= 2 and toks[1] == "Site":
                parent = stack[-1] if stack else -1
                base = joints[parent].name if parent >= 0 else "End"
                name = f"{base}_EndSite_{len(joints)}"
                cur = _add_joint(name, parent)
                continue

            if toks[0] == "{":
                if cur is None:
                    continue
                stack.append(cur)
                continue

            if toks[0] == "}":
                if stack:
                    stack.pop()
                cur = stack[-1] if stack else None
                continue

            if toks[0] == "OFFSET" and cur is not None and len(toks) >= 4:
                joints[cur].offset = np.array([float(toks[1]), float(toks[2]), float(toks[3])], dtype=np.float32)
                continue

            if toks[0] == "CHANNELS" and cur is not None:
                n = int(toks[1])
                ch = toks[2:2 + n]
                joints[cur].channels = ch
                continue

        channel_names: List[str] = []
        cursor = 0
        for j in joints:
            j.ch_start = cursor
            channel_names.extend(j.channels)
            cursor += len(j.channels)

        return BVHSkeleton(joints=joints, channel_names=channel_names)

    def joint_index(self, name: str) -> int:
        for i, j in enumerate(self.joints):
            if j.name == name:
                return i
        return -1

    def find_joints_by_keywords(self, keywords: List[str]) -> List[int]:
        ks = [k.lower() for k in keywords]
        out = []
        for i, j in enumerate(self.joints):
            nm = j.name.lower()
            if any(k in nm for k in ks):
                out.append(i)
        return out


@dataclass
class ActiveJointMap:
    root_joint_id: int
    root_pos_indices: List[Optional[int]]
    active_joint_ids: List[int]
    active_joint_names: List[str]
    active_rot_orders: List[List[str]]
    active_rot_channel_indices: List[List[int]]
    joint_id_to_active_idx: Dict[int, int]

    @property
    def n_active(self) -> int:
        return len(self.active_joint_ids)


@dataclass
class MotionPartSpec:
    part: str
    skel: BVHSkeleton
    active_joint_map: ActiveJointMap
    part_joint_ids: List[int]
    part_joint_names: List[str]
    part_active_indices: List[int]
    part_rot_dim_indices: List[int]
    drop_root_pos_full: bool
    lower_include_root: bool
    lower_include_foot_contact: bool
    foot_contact_joint_ids: List[int]
    foot_contact_joint_names: List[str]

    @property
    def full_canonical_dim(self) -> int:
        return 3 + self.active_joint_map.n_active * 6

    @property
    def rot_feature_dim(self) -> int:
        return len(self.part_rot_dim_indices)

    @property
    def include_root_translation(self) -> bool:
        return self.part == "global" or (self.part == "lower" and self.lower_include_root)

    @property
    def include_foot_contact(self) -> bool:
        return self.part == "global" or (self.part == "lower" and self.lower_include_foot_contact)

    @property
    def uses_vq(self) -> bool:
        return self.part != "global"

    @property
    def root_feature_slice(self) -> Optional[slice]:
        if not self.include_root_translation:
            return None
        if self.part == "global":
            start = self.rot_feature_dim
        elif self.part == "lower":
            start = 0
        else:
            return None
        return slice(start, start + 3)

    @property
    def foot_contact_slice(self) -> Optional[slice]:
        if not self.include_foot_contact:
            return None
        if self.part == "global":
            start = self.rot_feature_dim + 3
        elif self.part == "lower":
            start = self.rot_feature_dim + (3 if self.lower_include_root else 0)
        else:
            return None
        return slice(start, start + 4)

    @property
    def model_dim(self) -> int:
        if self.part == "full":
            return self.full_canonical_dim - (3 if self.drop_root_pos_full else 0)

        dim = self.rot_feature_dim
        if self.include_root_translation:
            dim += 3
        if self.include_foot_contact:
            dim += 4
        return dim

    @property
    def needs_full_gt(self) -> bool:
        return self.part != "full"


def _split_csv_keywords(raw: Optional[str], fallback: List[str]) -> List[str]:
    if raw is None:
        return list(fallback)
    out = [x.strip() for x in str(raw).split(",") if x.strip()]
    return out if out else list(fallback)


def _build_rot_feature_indices(active_indices: List[int]) -> List[int]:
    out: List[int] = []
    for idx in active_indices:
        base = int(idx) * 6
        out.extend(range(base, base + 6))
    return out


def build_active_joint_map(skel: BVHSkeleton) -> ActiveJointMap:
    root_joint_id = -1
    for i, j in enumerate(skel.joints):
        if j.parent == -1:
            root_joint_id = i
            break
    if root_joint_id < 0:
        raise RuntimeError("Failed to find BVH root joint.")

    root_joint = skel.joints[root_joint_id]
    root_pos_indices: List[Optional[int]] = []
    for axis in ("X", "Y", "Z"):
        found = None
        for local_idx, ch in enumerate(root_joint.channels):
            if ch.lower() == f"{axis.lower()}position":
                found = root_joint.ch_start + local_idx
                break
        root_pos_indices.append(found)

    active_joint_ids: List[int] = []
    active_joint_names: List[str] = []
    active_rot_orders: List[List[str]] = []
    active_rot_channel_indices: List[List[int]] = []
    joint_id_to_active_idx: Dict[int, int] = {}

    for jid, joint in enumerate(skel.joints):
        rot_order: List[str] = []
        rot_indices: List[int] = []
        for local_idx, ch in enumerate(joint.channels):
            if "rotation" not in ch.lower():
                continue
            rot_order.append(ch)
            rot_indices.append(joint.ch_start + local_idx)

        if not rot_order:
            continue
        if len(rot_order) != 3:
            print(f"[WARN] Skip joint with non-3 rotation channels: {joint.name} -> {joint.channels}")
            continue

        joint_id_to_active_idx[jid] = len(active_joint_ids)
        active_joint_ids.append(jid)
        active_joint_names.append(joint.name)
        active_rot_orders.append(rot_order)
        active_rot_channel_indices.append(rot_indices)

    if len(active_joint_ids) == 0:
        raise RuntimeError("No active rotation joints found in BVH skeleton.")

    return ActiveJointMap(
        root_joint_id=root_joint_id,
        root_pos_indices=root_pos_indices,
        active_joint_ids=active_joint_ids,
        active_joint_names=active_joint_names,
        active_rot_orders=active_rot_orders,
        active_rot_channel_indices=active_rot_channel_indices,
        joint_id_to_active_idx=joint_id_to_active_idx,
    )


def select_part_joint_ids(
    skel: BVHSkeleton,
    active_joint_map: ActiveJointMap,
    part: str,
    keyword_map: Dict[str, List[str]],
) -> List[int]:
    if part == "full":
        return list(active_joint_map.active_joint_ids)
    if part == "global":
        part = "lower"

    if part not in keyword_map:
        raise ValueError(f"Unknown part '{part}' for keyword selection.")

    priorities = ("hand", "lower", "upper")
    selected: List[int] = []

    for joint_id in active_joint_map.active_joint_ids:
        name = skel.joints[joint_id].name.lower()
        assigned = None
        for group in priorities:
            kws = [k.lower() for k in keyword_map.get(group, [])]
            if any(kw in name for kw in kws):
                assigned = group
                break
        if assigned == part:
            selected.append(joint_id)

    return selected


def _select_joint_ids_by_keywords(
    skel: BVHSkeleton,
    keywords: List[str],
    *,
    require_active: bool = False,
    active_joint_map: Optional[ActiveJointMap] = None,
    limit: Optional[int] = None,
) -> List[int]:
    active_set = set(active_joint_map.active_joint_ids) if (require_active and active_joint_map is not None) else None
    out: List[int] = []
    used: set[int] = set()

    for keyword in keywords:
        kw = keyword.lower()
        match = None
        for jid, joint in enumerate(skel.joints):
            name = joint.name.lower()
            if "endsite" in name or name.endswith("_end"):
                continue
            if active_set is not None and jid not in active_set:
                continue
            if kw in name:
                match = jid
                break
        if match is not None and match not in used:
            out.append(match)
            used.add(match)
        if limit is not None and len(out) >= limit:
            break

    return out


def _rot_x(a: torch.Tensor) -> torch.Tensor:
    ca, sa = torch.cos(a), torch.sin(a)
    z = torch.zeros_like(a); o = torch.ones_like(a)
    R = torch.stack([
        torch.stack([o, z, z], dim=-1),
        torch.stack([z, ca, -sa], dim=-1),
        torch.stack([z, sa, ca], dim=-1),
    ], dim=-2)
    return R


def _rot_y(a: torch.Tensor) -> torch.Tensor:
    ca, sa = torch.cos(a), torch.sin(a)
    z = torch.zeros_like(a); o = torch.ones_like(a)
    R = torch.stack([
        torch.stack([ca, z, sa], dim=-1),
        torch.stack([z, o, z], dim=-1),
        torch.stack([-sa, z, ca], dim=-1),
    ], dim=-2)
    return R


def _rot_z(a: torch.Tensor) -> torch.Tensor:
    ca, sa = torch.cos(a), torch.sin(a)
    z = torch.zeros_like(a); o = torch.ones_like(a)
    R = torch.stack([
        torch.stack([ca, -sa, z], dim=-1),
        torch.stack([sa, ca, z], dim=-1),
        torch.stack([z, z, o], dim=-1),
    ], dim=-2)
    return R


def euler_channels_to_matrix_deg(x_deg: torch.Tensor, order: List[str]) -> torch.Tensor:
    """
    x_deg: [..., 3] degrees values aligned to axes in order list (e.g. ['Xrotation','Yrotation','Zrotation'])
    Returns [..., 3, 3]
    """
    a = torch.deg2rad(x_deg)
    R = torch.eye(3, device=x_deg.device, dtype=x_deg.dtype).expand(*a.shape[:-1], 3, 3)
    for i, ch in enumerate(order):
        axis = ch[0].upper()
        ai = a[..., i]
        if axis == "X":
            Ri = _rot_x(ai)
        elif axis == "Y":
            Ri = _rot_y(ai)
        else:
            Ri = _rot_z(ai)
        R = R @ Ri
    return R


def convert_bvh_to_6d_with_channel_order(
    motion_deg: np.ndarray,
    skel: BVHSkeleton,
    active_joint_map: ActiveJointMap,
) -> np.ndarray:
    """
    Convert resampled BVH motion channels to a canonical full-body representation:
      [root_xyz(3), active_joint_rot6d(J_active * 6)]

    Euler rotation order is read per joint from the BVH CHANNELS definition.
    """
    if motion_deg.ndim != 2:
        raise ValueError(f"Expected motion_deg [T, D], got shape={motion_deg.shape}")

    T = int(motion_deg.shape[0])
    root_pos = np.zeros((T, 3), dtype=np.float32)
    for axis_id, src_idx in enumerate(active_joint_map.root_pos_indices):
        if src_idx is not None and 0 <= int(src_idx) < motion_deg.shape[1]:
            root_pos[:, axis_id] = motion_deg[:, int(src_idx)]

    rot_parts: List[np.ndarray] = []
    for chan_indices, order in zip(active_joint_map.active_rot_channel_indices, active_joint_map.active_rot_orders):
        x_deg = torch.from_numpy(motion_deg[:, chan_indices].astype(np.float32))
        mats = euler_channels_to_matrix_deg(x_deg, order)
        rot6d = matrix_to_rotation_6d(mats).cpu().numpy().astype(np.float32)
        rot_parts.append(rot6d)

    if len(rot_parts) == 0:
        raise RuntimeError("No active rotation joints available for 6D conversion.")

    rot_6d = np.concatenate(rot_parts, axis=1).astype(np.float32)
    return np.concatenate([root_pos, rot_6d], axis=1).astype(np.float32)


def compute_foot_contacts(
    full_motion: np.ndarray,
    motion_spec: MotionPartSpec,
    fps: int = TARGET_FPS,
    vel_threshold: float = 2.0,
    height_margin: float = 5.0,
    ground_percentile: float = 5.0,
) -> np.ndarray:
    """
    Compute up to 4 foot-contact channels from canonical full motion.
    Missing slots are zero padded to keep lower-part dim stable.
    """
    T = int(full_motion.shape[0])
    out = np.zeros((T, 4), dtype=np.float32)

    if T <= 1 or len(motion_spec.foot_contact_joint_ids) == 0:
        return out

    fk_cpu = BVHFkTorch(motion_spec.skel, drop_root_pos=False).to_device_tensors(torch.device("cpu"))
    with torch.no_grad():
        pos = fk_cpu.fk_positions(torch.from_numpy(full_motion[None]).float()).squeeze(0).cpu().numpy()

    for k, joint_id in enumerate(motion_spec.foot_contact_joint_ids[:4]):
        p = pos[:, joint_id, :]
        vel = np.linalg.norm(np.gradient(p, axis=0), axis=1) * float(fps)
        h = p[:, 1]
        gnd = float(np.percentile(h, ground_percentile))
        out[:, k] = ((vel < float(vel_threshold)) & (h < (gnd + float(height_margin)))).astype(np.float32)

    return out


def split_motion_into_parts(
    full_motion: np.ndarray,
    motion_spec: MotionPartSpec,
    fps: int = TARGET_FPS,
) -> np.ndarray:
    """
    Slice canonical full-body motion into full / upper / hand / lower training features.
    """
    if full_motion.ndim != 2:
        raise ValueError(f"Expected full_motion [T, D], got shape={full_motion.shape}")

    if motion_spec.part == "full":
        return full_motion[:, 3:] if motion_spec.drop_root_pos_full else full_motion

    chunks: List[np.ndarray] = []
    rot_full = full_motion[:, 3:]

    if motion_spec.part == "global":
        if motion_spec.rot_feature_dim > 0:
            chunks.append(rot_full[:, motion_spec.part_rot_dim_indices])
        chunks.append(full_motion[:, :3])
        chunks.append(compute_foot_contacts(full_motion, motion_spec, fps=fps))
    else:
        if motion_spec.part == "lower" and motion_spec.lower_include_root:
            chunks.append(full_motion[:, :3])

        if motion_spec.rot_feature_dim > 0:
            chunks.append(rot_full[:, motion_spec.part_rot_dim_indices])

        if motion_spec.part == "lower" and motion_spec.lower_include_foot_contact:
            chunks.append(compute_foot_contacts(full_motion, motion_spec, fps=fps))

    if len(chunks) == 0:
        raise RuntimeError(f"No motion features selected for part={motion_spec.part}")

    return np.concatenate(chunks, axis=1).astype(np.float32)


def estimate_linear_velocity(data_seq: torch.Tensor, dt: float) -> torch.Tensor:
    """
    LoM/EMAGE-style finite-difference velocity estimate:
    forward diff at the first frame, central diff in the middle, backward diff at the end.
    """
    if data_seq.ndim != 3:
        raise ValueError(f"Expected [B, T, C], got {tuple(data_seq.shape)}")
    if data_seq.shape[1] <= 1:
        return torch.zeros_like(data_seq)
    if data_seq.shape[1] == 2:
        v = (data_seq[:, 1:] - data_seq[:, :1]) / float(dt)
        return torch.cat([v, v], dim=1)

    init_vel = (data_seq[:, 1:2] - data_seq[:, :1]) / float(dt)
    middle_vel = (data_seq[:, 2:] - data_seq[:, :-2]) / float(2.0 * dt)
    final_vel = (data_seq[:, -1:] - data_seq[:, -2:-1]) / float(dt)
    return torch.cat([init_vel, middle_vel, final_vel], dim=1)


def velocity2position(
    data_seq: Union[np.ndarray, torch.Tensor],
    dt: float,
    init_pos: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    if torch.is_tensor(data_seq):
        out = []
        for i in range(data_seq.shape[1]):
            if i == 0:
                out.append(init_pos.unsqueeze(1))
            else:
                out.append(data_seq[:, i - 1:i] * float(dt) + out[-1])
        return torch.cat(out, dim=1)

    out_np = []
    for i in range(data_seq.shape[1]):
        if i == 0:
            out_np.append(np.expand_dims(init_pos, axis=1))
        else:
            out_np.append(data_seq[:, i - 1:i] * float(dt) + out_np[-1])
    return np.concatenate(out_np, axis=1)


def prepare_global_condition_input(
    part_motion: Union[np.ndarray, torch.Tensor],
    motion_spec: MotionPartSpec,
) -> Union[np.ndarray, torch.Tensor]:
    if motion_spec.part != "global":
        return part_motion

    cond = part_motion.clone() if torch.is_tensor(part_motion) else np.array(part_motion, dtype=np.float32, copy=True)
    cond[..., motion_spec.rot_feature_dim:] = 0.0
    return cond


def recover_global_root_translation(
    part_motion: Union[np.ndarray, torch.Tensor],
    motion_spec: MotionPartSpec,
    gt_root: Optional[Union[np.ndarray, torch.Tensor]] = None,
    fps: int = TARGET_FPS,
) -> Union[np.ndarray, torch.Tensor]:
    if motion_spec.part != "global":
        raise ValueError(f"recover_global_root_translation expects part=global, got {motion_spec.part}")

    root_slice = motion_spec.root_feature_slice
    if root_slice is None:
        raise RuntimeError("Global motion_spec must expose a root feature slice.")

    is_torch = torch.is_tensor(part_motion)
    x = part_motion
    squeeze = False
    if x.ndim == 2:
        x = x.unsqueeze(0) if is_torch else x[None, ...]
        squeeze = True
    if x.ndim != 3:
        raise ValueError(f"Expected part_motion [T, D] or [B, T, D], got shape={tuple(x.shape)}")

    trans_signal = x[..., root_slice]
    if gt_root is not None:
        g = gt_root[..., :3]
        if g.ndim == 2:
            g = g.unsqueeze(0) if torch.is_tensor(g) else g[None, ...]
        if tuple(g.shape[:2]) != tuple(x.shape[:2]):
            raise ValueError(
                f"gt_root shape mismatch: expected first dims {tuple(x.shape[:2])}, got {tuple(g.shape[:2])}"
            )
        init_x = g[:, 0, 0:1]
        init_z = g[:, 0, 2:3]
    else:
        if is_torch:
            init_x = trans_signal.new_zeros((trans_signal.shape[0], 1))
            init_z = trans_signal.new_zeros((trans_signal.shape[0], 1))
        else:
            init_x = np.zeros((trans_signal.shape[0], 1), dtype=np.float32)
            init_z = np.zeros((trans_signal.shape[0], 1), dtype=np.float32)

    dt = 1.0 / float(fps)
    rec_x = velocity2position(trans_signal[..., 0:1], dt, init_x)
    rec_y = trans_signal[..., 1:2]
    rec_z = velocity2position(trans_signal[..., 2:3], dt, init_z)

    if is_torch:
        root_xyz = torch.cat([rec_x, rec_y, rec_z], dim=-1)
    else:
        root_xyz = np.concatenate([rec_x, rec_y, rec_z], axis=-1).astype(np.float32)

    if squeeze:
        root_xyz = root_xyz[0]
    return root_xyz


def global_motion_loss(
    pred_part_motion: torch.Tensor,
    target_part_motion: torch.Tensor,
    motion_spec: MotionPartSpec,
    fps: int = TARGET_FPS,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if motion_spec.part != "global":
        raise ValueError(f"global_motion_loss expects part=global, got {motion_spec.part}")

    root_slice = motion_spec.root_feature_slice
    contact_slice = motion_spec.foot_contact_slice
    if root_slice is None or contact_slice is None:
        raise RuntimeError("Global motion_spec must include root and foot-contact slices.")

    rec_trans = pred_part_motion[..., root_slice]
    tar_root = target_part_motion[..., root_slice]
    rec_root = recover_global_root_translation(pred_part_motion, motion_spec, gt_root=tar_root, fps=fps)

    dt = 1.0 / float(fps)
    tar_vel_x = estimate_linear_velocity(tar_root[..., 0:1], dt=dt)
    tar_vel_z = estimate_linear_velocity(tar_root[..., 2:3], dt=dt)

    rec_contact = pred_part_motion[..., contact_slice]
    tar_contact = target_part_motion[..., contact_slice]
    loss_contact = F.mse_loss(rec_contact, tar_contact)

    loss_trans_vel = (
        F.l1_loss(rec_trans[..., 0:1], tar_vel_x)
        + F.l1_loss(rec_trans[..., 2:3], tar_vel_z)
    )

    zero = pred_part_motion.new_tensor(0.0)
    if pred_part_motion.shape[1] >= 2:
        v3 = (
            F.l1_loss(rec_trans[:, 1:, 0:1] - rec_trans[:, :-1, 0:1], tar_vel_x[:, 1:] - tar_vel_x[:, :-1])
            + F.l1_loss(rec_trans[:, 1:, 2:3] - rec_trans[:, :-1, 2:3], tar_vel_z[:, 1:] - tar_vel_z[:, :-1])
        )
        v2 = F.l1_loss(rec_root[:, 1:] - rec_root[:, :-1], tar_root[:, 1:] - tar_root[:, :-1])
    else:
        v3 = zero
        v2 = zero

    if pred_part_motion.shape[1] >= 3:
        a3 = (
            F.l1_loss(
                rec_trans[:, 2:, 0:1] + rec_trans[:, :-2, 0:1] - 2.0 * rec_trans[:, 1:-1, 0:1],
                tar_vel_x[:, 2:] + tar_vel_x[:, :-2] - 2.0 * tar_vel_x[:, 1:-1],
            )
            + F.l1_loss(
                rec_trans[:, 2:, 2:3] + rec_trans[:, :-2, 2:3] - 2.0 * rec_trans[:, 1:-1, 2:3],
                tar_vel_z[:, 2:] + tar_vel_z[:, :-2] - 2.0 * tar_vel_z[:, 1:-1],
            )
        )
        a2 = F.l1_loss(
            rec_root[:, 2:] + rec_root[:, :-2] - 2.0 * rec_root[:, 1:-1],
            tar_root[:, 2:] + tar_root[:, :-2] - 2.0 * tar_root[:, 1:-1],
        )
    else:
        a3 = zero
        a2 = zero

    loss_trans = F.l1_loss(rec_root, tar_root)
    smooth = 5.0 * v3 + 5.0 * a3 + 5.0 * v2 + 5.0 * a2
    total = loss_contact + loss_trans_vel + loss_trans + smooth

    metrics = {
        "root_l1": loss_trans,
        "trans_vel_l1": loss_trans_vel,
        "contact_mse": loss_contact,
        "smooth_l1": smooth,
    }
    return total, metrics


def merge_parts_back_to_full(
    part_motion: Union[np.ndarray, torch.Tensor],
    motion_spec: MotionPartSpec,
    gt_full: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Merge a part-wise motion tensor back into canonical full-body motion.
    Outside-part joints are copied from GT when available, otherwise filled with identity 6D.
    """
    is_torch = torch.is_tensor(part_motion)
    base = gt_full

    if base is not None:
        out = base.clone() if torch.is_tensor(base) else np.array(base, dtype=np.float32, copy=True)
    else:
        shape = tuple(part_motion.shape[:-1]) + (motion_spec.full_canonical_dim,)
        if is_torch:
            out = part_motion.new_zeros(shape)
            rot_identity = torch.as_tensor(
                np.tile(IDENTITY_6D, motion_spec.active_joint_map.n_active),
                device=part_motion.device,
                dtype=part_motion.dtype,
            )
            out[..., 3:] = rot_identity
        else:
            out = np.zeros(shape, dtype=np.float32)
            out[..., 3:] = np.tile(IDENTITY_6D, motion_spec.active_joint_map.n_active)

    if motion_spec.part == "full":
        if motion_spec.drop_root_pos_full:
            out[..., 3:] = part_motion
        else:
            out[...] = part_motion
        return out

    if motion_spec.part == "global":
        gt_root = None if gt_full is None else gt_full[..., :3]
        out[..., :3] = recover_global_root_translation(part_motion, motion_spec, gt_root=gt_root)
        return out

    offset = 0
    if motion_spec.part == "lower" and motion_spec.lower_include_root:
        out[..., :3] = part_motion[..., :3]
        offset += 3

    rot_len = motion_spec.rot_feature_dim
    if rot_len > 0:
        if is_torch:
            rot_idx = torch.as_tensor(motion_spec.part_rot_dim_indices, device=part_motion.device, dtype=torch.long) + 3
        else:
            rot_idx = 3 + np.asarray(motion_spec.part_rot_dim_indices, dtype=np.int64)
        out[..., rot_idx] = part_motion[..., offset:offset + rot_len]
        offset += rot_len

    # lower foot-contact dims are auxiliary and do not map back to full-body rotation channels.
    return out


def build_motion_part_spec(
    skel: BVHSkeleton,
    *,
    part: str,
    drop_root_pos: bool,
    lower_include_root: bool,
    lower_include_foot_contact: bool,
    part_keywords: Dict[str, List[str]],
    foot_contact_keywords: List[str],
) -> MotionPartSpec:
    active_joint_map = build_active_joint_map(skel)
    part_joint_ids = select_part_joint_ids(skel, active_joint_map, part, part_keywords)
    part_joint_names = [skel.joints[jid].name for jid in part_joint_ids]
    part_active_indices = [active_joint_map.joint_id_to_active_idx[jid] for jid in part_joint_ids]
    part_active_indices = sorted(part_active_indices)
    part_rot_dim_indices = _build_rot_feature_indices(part_active_indices)

    if part != "full" and len(part_joint_ids) == 0:
        raise RuntimeError(f"No active joints matched part={part}. Please adjust the part keywords.")

    foot_contact_joint_ids = []
    if lower_include_foot_contact or part == "global":
        foot_contact_joint_ids = _select_joint_ids_by_keywords(
            skel,
            foot_contact_keywords,
            require_active=False,
            active_joint_map=active_joint_map,
            limit=4,
        )

    foot_contact_joint_names = [skel.joints[jid].name for jid in foot_contact_joint_ids]

    return MotionPartSpec(
        part=part,
        skel=skel,
        active_joint_map=active_joint_map,
        part_joint_ids=part_joint_ids,
        part_joint_names=part_joint_names,
        part_active_indices=part_active_indices,
        part_rot_dim_indices=part_rot_dim_indices,
        drop_root_pos_full=bool(drop_root_pos),
        lower_include_root=bool(lower_include_root),
        lower_include_foot_contact=bool(lower_include_foot_contact),
        foot_contact_joint_ids=foot_contact_joint_ids,
        foot_contact_joint_names=foot_contact_joint_names,
    )


class BVHFkTorch:
    """
    Differentiable FK producing joint positions.
    Includes fixes for:
    1. Active Joints mismatch (88 joints vs 75 data inputs)
    2. Float32 precision enforcement (prevents AMP crashes)
    """
    def __init__(self, skel: BVHSkeleton, drop_root_pos: bool):
        self.skel = skel
        self.drop_root_pos = bool(drop_root_pos)

        self.parents = [j.parent for j in skel.joints]
        self.offsets = np.stack([j.offset for j in skel.joints], axis=0).astype(np.float32)

        # ---------------------------------------------------------
        # 1. 自动识别哪些关节有旋转数据 (Active Joints)
        # ---------------------------------------------------------
        self.joint_has_rot_list = []
        active_indices = []
        
        for i, j in enumerate(skel.joints):
            # 只要包含 "rotation" 的通道就算活跃关节
            has_rot = any("rotation" in ch.lower() for ch in j.channels)
            self.joint_has_rot_list.append(has_rot)
            if has_rot:
                active_indices.append(i)
        
        # 记录活跃关节的索引，用于将压缩的输入映射回完整骨骼
        self.active_rot_indices_np = np.array(active_indices, dtype=np.int64)

    def to_device_tensors(self, device: torch.device):
        self.parents_t = torch.tensor(self.parents, device=device, dtype=torch.long)
        self.offsets_t = torch.from_numpy(self.offsets).to(device=device, dtype=torch.float32)
        
        self.active_rot_idx_t = torch.from_numpy(self.active_rot_indices_np).to(device=device, dtype=torch.long)
        
        # 6D Identity Vector: [1, 0, 0, 0, 1, 0] -> Column1=[1,0,0], Column2=[0,1,0]
        # 这是标准的 Identity，在 rotation_6d_to_matrix 计算后就是单位矩阵
        self.identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device, dtype=torch.float32).view(1, 1, 6)
        
        return self

    def fk_positions(self, motion_keep: torch.Tensor) -> torch.Tensor:
        """
        motion_keep: [B,T,D] 
        """
        # ==============================================================
        # 2. 强制转为 Float32
        # - 解决 AMP 下的 Float/Half 赋值报错
        # - 保证 FK 连乘的精度，防止末端抖动
        # ==============================================================
        motion_keep = motion_keep.to(dtype=torch.float32)

        B, T, D = motion_keep.shape
        J = int(self.offsets_t.shape[0]) 
        BT = B * T
        x = motion_keep.reshape(BT, D)

        R_list: List[torch.Tensor] = []
        t_list: List[torch.Tensor] = []

        # 解析 Root Pos
        offset_idx = 0
        if not self.drop_root_pos:
            root_p = x[:, :3]
            offset_idx = 3
        
        # 解析旋转数据 (Dense)
        rot_dense = x[:, offset_idx:] # [BT, N_active * 6]
        
        # 校验维度
        n_active = len(self.active_rot_indices_np)
        if rot_dense.shape[1] != n_active * 6:
            raise RuntimeError(
                f"FK Dimension mismatch! Input has {rot_dense.shape[1]} rot-dims ({rot_dense.shape[1]//6} joints), "
                f"but Skeleton expects {n_active} active joints. "
                f"Please check if your cache data matches the skeleton."
            )

        rot_dense = rot_dense.view(BT, n_active, 6)

        # 填充完整骨骼 (Sparse)
        # 默认填 Identity (不旋转)，然后把有数据的关节填进去
        rot_full = self.identity_6d.expand(BT, J, 6).clone()
        rot_full[:, self.active_rot_idx_t, :] = rot_dense

        # 统一转矩阵
        all_rots = rotation_6d_to_matrix(rot_full) # [BT, J, 3, 3]

        # FK 循环
        for j in range(J):
            parent = int(self.parents[j])
            t_loc = self.offsets_t[j].view(1, 3).expand(BT, 3)

            if parent == -1 and (not self.drop_root_pos):
                t_loc = t_loc + root_p

            R_loc = all_rots[:, j] 

            if parent == -1:
                Rg = R_loc
                tg = t_loc
            else:
                Rp = R_list[parent]
                tp = t_list[parent]
                Rg = Rp @ R_loc
                tg = (Rp @ t_loc.unsqueeze(-1)).squeeze(-1) + tp

            R_list.append(Rg)
            t_list.append(tg)

        pos = torch.stack(t_list, dim=1).view(B, T, J, 3)
        return pos

def _to_cpu_float_tensor(x: Any) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        out = x.detach().cpu()
        return out.float() if out.dtype != torch.float32 else out
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        return None
    return torch.from_numpy(arr).float()


def _load_stats_npz(path: Path, std_floor: float) -> Tuple[np.ndarray, np.ndarray, int]:
    stats = np.load(path)
    mean = stats["mean"].astype(np.float32)
    std = np.maximum(stats["std"].astype(np.float32), float(std_floor)).astype(np.float32)
    kd = stats["keep_dim"]
    keep_dim = int(kd) if np.ndim(kd) == 0 else int(np.asarray(kd).reshape(-1)[0])
    return mean, std, keep_dim


def _save_stats_npz(path: Path, mean: np.ndarray, std: np.ndarray, keep_dim: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=mean.astype(np.float32), std=std.astype(np.float32), keep_dim=np.array([keep_dim], dtype=np.int32))


def _compute_mean_std_from_accumulators(
    sum_vec: np.ndarray,
    sumsq_vec: np.ndarray,
    total_frames: int,
    std_floor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = (sum_vec / total_frames).astype(np.float32)
    var = (sumsq_vec / total_frames) - (mean.astype(np.float64) ** 2)
    std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
    std = np.maximum(std, std_floor).astype(np.float32)
    return mean, std


def _collate_motion_like(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    if isinstance(batch[0], dict):
        out = {}
        for k in batch[0].keys():
            vals = [b[k] for b in batch]
            out[k] = torch.stack(vals, dim=0) if torch.is_tensor(vals[0]) else vals
        return out
    return torch.stack(batch, dim=0)


class MotionWindowCachedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_pt: Path,
        block_size: int = 256,
        is_train: bool = True,
        drop_root_pos: bool = True,
        fixed_crop: bool = False,
        overfit_n: int = 0,
        std_floor: float = 1e-4,
        normalize_in_dataset: bool = True,
        use_speaking: bool = False,
        speech_prob: float = 0.8,
        min_speech_len: int = 16,
        pad_frames: int = 4,
        merge_silence: int = 3,
        stats_path: str = "checkpoints/motion_stats_vqvae.npz",
        motion_spec: Optional[MotionPartSpec] = None,
    ):
        super().__init__()
        self.cache_pt = Path(cache_pt)
        self.block_size = int(block_size)
        self.is_train = bool(is_train)
        self.drop_root_pos = bool(drop_root_pos)
        self.fixed_crop = bool(fixed_crop) or (overfit_n > 0)
        self.std_floor = float(std_floor)
        self.normalize_in_dataset = bool(normalize_in_dataset)
        self.use_speaking = bool(use_speaking)
        self.speech_prob = float(speech_prob)
        self.min_speech_len = int(min_speech_len)
        self.pad_frames = int(pad_frames)
        self.merge_silence = int(merge_silence)
        self.stats_path = Path(stats_path)
        self.motion_spec = motion_spec
        self.return_full_gt = bool(motion_spec is not None and motion_spec.needs_full_gt)

        pack = torch.load(self.cache_pt, map_location="cpu", weights_only=False)
        samples = pack.get("samples", None)
        if samples is None or not isinstance(samples, list) or len(samples) == 0:
            raise RuntimeError(f"Bad cache_pt: missing 'samples' or empty: {self.cache_pt}")

        if overfit_n and overfit_n > 0:
            samples = samples[: int(overfit_n)]
            print(f"[INFO] Overfit mode: using first {len(samples)} cached samples only.")

        motions: List[torch.Tensor] = []
        full_motions: List[Optional[torch.Tensor]] = []
        speakings: List[Optional[torch.Tensor]] = []
        valid_ids: List[int] = []
        seg_cache: List[Optional[List[Tuple[int, int]]]] = []

        sum_vec = None
        sumsq_vec = None
        total_frames = 0
        keep_dim = None

        for sample in samples:
            if not isinstance(sample, dict):
                continue
            y_t = _to_cpu_float_tensor(sample.get("y", None))
            full_store_t = _to_cpu_float_tensor(sample.get("full_y", None))

            full_t: Optional[torch.Tensor] = None
            part_t: Optional[torch.Tensor] = None

            if self.motion_spec is not None:
                if full_store_t is not None:
                    if full_store_t.shape[1] != self.motion_spec.full_canonical_dim:
                        continue
                    full_t = full_store_t

                if y_t is not None:
                    if y_t.shape[1] == self.motion_spec.model_dim:
                        part_t = y_t
                    elif y_t.shape[1] == self.motion_spec.full_canonical_dim:
                        full_t = y_t

                if full_t is None and part_t is None:
                    continue

                if part_t is None:
                    part_np = split_motion_into_parts(full_t.numpy(), self.motion_spec, fps=TARGET_FPS)
                    part_t = torch.from_numpy(part_np).float()

                if self.return_full_gt and full_t is None:
                    continue
            else:
                base_t = full_store_t if full_store_t is not None else y_t
                if base_t is None:
                    continue
                full_t = base_t
                if self.drop_root_pos:
                    if base_t.shape[1] < 3:
                        continue
                    part_t = base_t[:, 3:]
                else:
                    part_t = base_t

            if part_t is None or part_t.ndim != 2:
                continue

            if keep_dim is None:
                keep_dim = int(part_t.shape[1])
                sum_vec = np.zeros((keep_dim,), dtype=np.float64)
                sumsq_vec = np.zeros((keep_dim,), dtype=np.float64)
            elif int(part_t.shape[1]) != int(keep_dim):
                continue

            T = int(part_t.shape[0])
            if T < self.block_size:
                continue

            part_np = part_t.numpy().astype(np.float32, copy=False)
            if not np.isfinite(part_np).all():
                continue

            motions.append(part_t)
            full_motions.append(full_t if self.return_full_gt else None)

            sp = None
            if sample.get("speaking", None) is not None:
                sp_np = np.asarray(sample["speaking"], dtype=np.uint8)
                if sp_np.ndim == 1 and sp_np.shape[0] >= T:
                    sp = torch.from_numpy(sp_np[:T].copy()).to(dtype=torch.uint8)
            speakings.append(sp)

            valid_ids.append(len(motions) - 1)
            seg_cache.append(None)
            sum_vec += part_np.sum(axis=0, dtype=np.float64)
            sumsq_vec += (part_np.astype(np.float64) ** 2).sum(axis=0)
            total_frames += T

        if len(motions) == 0 or keep_dim is None or total_frames <= 0:
            raise RuntimeError(f"No valid motions in cache: {self.cache_pt}")

        recompute_stats = True
        if self.stats_path.exists():
            mean, std, stats_keep_dim = _load_stats_npz(self.stats_path, self.std_floor)
            if int(stats_keep_dim) == int(keep_dim):
                self.mean, self.std, self.keep_dim = mean, std, int(stats_keep_dim)
                recompute_stats = False
                print(f"[INFO] Loaded stats: dim={self.keep_dim} from {self.stats_path}")
            else:
                print(
                    f"[WARN] stats keep_dim mismatch for {self.stats_path}: "
                    f"stats={stats_keep_dim}, cache={keep_dim}. Recomputing stats for the current split."
                )

        if recompute_stats:
            mean, std = _compute_mean_std_from_accumulators(sum_vec, sumsq_vec, total_frames, self.std_floor)
            _save_stats_npz(self.stats_path, mean, std, int(keep_dim))
            self.mean, self.std, self.keep_dim = mean, std, int(keep_dim)
            print(f"[INFO] Saved stats to {self.stats_path} (dim={self.keep_dim})")

        self.mean_t = torch.from_numpy(self.mean).float()
        self.std_t = torch.from_numpy(self.std).float()
        self.motions = motions
        self.full_motions = full_motions
        self.speakings = speakings
        self.valid_ids = valid_ids
        self._speech_segs = seg_cache

        print(
            f"[INFO] Cached dataset loaded: {len(self.motions)} items, "
            f"dim(keep)={self.keep_dim}, block={self.block_size}, "
            f"normalize_in_dataset={self.normalize_in_dataset}, use_speaking={self.use_speaking}, "
            f"part={(self.motion_spec.part if self.motion_spec is not None else 'full')}"
        )

    def __len__(self):
        return len(self.valid_ids)

    def _build_speech_segments(self, sp: torch.Tensor) -> List[Tuple[int, int]]:
        T = int(sp.shape[0])
        if T <= 0:
            return []

        segs: List[Tuple[int, int]] = []
        i = 0
        while i < T:
            if int(sp[i].item()) == 0:
                i += 1
                continue
            j = i + 1
            while j < T and int(sp[j].item()) == 1:
                j += 1
            segs.append((i, j))
            i = j

        if not segs:
            return []

        merged = [segs[0]]
        for a, b in segs[1:]:
            pa, pb = merged[-1]
            if (a - pb) <= self.merge_silence:
                merged[-1] = (pa, b)
            else:
                merged.append((a, b))

        out: List[Tuple[int, int]] = []
        for a, b in merged:
            a2 = max(0, a - self.pad_frames)
            b2 = min(T, b + self.pad_frames)
            if (b2 - a2) >= max(self.min_speech_len, self.block_size):
                out.append((a2, b2))
        return out

    def __getitem__(self, idx: int):
        sid = self.valid_ids[int(idx)]
        y = self.motions[sid]
        T = int(y.shape[0])

        if self.is_train and (not self.fixed_crop):
            use_speech = self.use_speaking and (random.random() < self.speech_prob)
            if use_speech and self.speakings[sid] is not None:
                if self._speech_segs[sid] is None:
                    self._speech_segs[sid] = self._build_speech_segments(self.speakings[sid])
                segs = self._speech_segs[sid] or []
                if len(segs) > 0:
                    a, b = random.choice(segs)
                    start = random.randint(a, max(a, b - self.block_size))
                else:
                    start = random.randint(0, T - self.block_size)
            else:
                start = random.randint(0, T - self.block_size)
        else:
            start = 0

        w = y.narrow(0, start, self.block_size)
        if self.normalize_in_dataset:
            w = (w - self.mean_t) / self.std_t

        if self.return_full_gt:
            return {
                "motion": w,
                "full_motion": self.full_motions[sid].narrow(0, start, self.block_size),
            }
        return w


def collate_motion_tensor(batch):
    return _collate_motion_like(batch)


class MotionWindowDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        block_size: int = 256,
        is_train: bool = True,
        stats_path: str = "checkpoints/motion_stats_vqvae.npz",
        approx_stats_n: int = 500,
        drop_root_pos: bool = True,
        overfit_n: int = 0,
        fixed_crop: bool = False,
        std_floor: float = 1e-4,
        use_textgrid: bool = True,
        speech_prob: float = 0.8,
        merge_silence_s: float = 0.2,
        pad_s: float = 0.3,
        tier_name: Optional[str] = None,
        cache_in_mem: bool = False,
        cache_preload: bool = False,
        motion_spec: Optional[MotionPartSpec] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.base_dir = self.manifest_path.parent
        self.block_size = int(block_size)
        self.is_train = bool(is_train)
        self.drop_root_pos = bool(drop_root_pos)
        self.fixed_crop = bool(fixed_crop) or (overfit_n > 0)
        self.std_floor = float(std_floor)
        self.use_textgrid = bool(use_textgrid)
        self.speech_prob = float(speech_prob)
        self.merge_silence_s = float(merge_silence_s)
        self.pad_s = float(pad_s)
        self.tier_name = tier_name
        self.cache_in_mem = bool(cache_in_mem)
        self.cache_preload = bool(cache_preload)
        self.motion_spec = motion_spec
        self.return_full_gt = bool(motion_spec is not None and motion_spec.needs_full_gt)
        self.stats_path = Path(stats_path)
        self._motion_cache: Dict[int, Dict[str, np.ndarray]] = {}

        raw = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw.append(json.loads(line))
        if overfit_n and overfit_n > 0:
            raw = raw[: int(overfit_n)]
            print(f"[INFO] Overfit mode: using first {len(raw)} items only.")

        self.items = raw
        self._seg_cache: Dict[str, List[Tuple[int, int]]] = {}
        print(f"[INFO] Indexed {len(self.items)} items from {self.manifest_path}")

        expected_keep_dim = int(self.motion_spec.model_dim) if self.motion_spec is not None else None
        recompute_stats = True
        if self.stats_path.exists():
            mean, std, stats_keep_dim = _load_stats_npz(self.stats_path, self.std_floor)
            if expected_keep_dim is None or int(stats_keep_dim) == int(expected_keep_dim):
                self.mean, self.std, self.keep_dim = mean, std, int(stats_keep_dim)
                recompute_stats = False
                print(f"[INFO] Loaded stats: dim={self.keep_dim} from {self.stats_path}")
            else:
                print(
                    f"[WARN] stats keep_dim mismatch for {self.stats_path}: "
                    f"stats={stats_keep_dim}, expected={expected_keep_dim}. Recomputing stats for the current split."
                )

        if recompute_stats:
            print(f"[WARN] Stats not found. Computing approx stats from first {approx_stats_n} valid motions...")
            feats: List[np.ndarray] = []
            for idx in tqdm(range(len(self.items)), desc="stats"):
                item = self._load_motion_preprocessed(idx, min_frames=32)
                if item is None:
                    continue
                feats.append(item["motion"])
                if len(feats) >= int(approx_stats_n):
                    break
            if len(feats) == 0:
                raise RuntimeError("No valid motions for stats.")
            all_m = np.concatenate(feats, axis=0).astype(np.float32)
            mean = np.mean(all_m, axis=0).astype(np.float32)
            std = np.maximum(np.std(all_m, axis=0).astype(np.float32), self.std_floor).astype(np.float32)
            self.keep_dim = int(all_m.shape[1])
            self.mean, self.std = mean, std
            _save_stats_npz(self.stats_path, self.mean, self.std, self.keep_dim)
            print(f"[INFO] Saved stats to {self.stats_path} (dim={self.keep_dim})")

        if self.cache_in_mem and self.cache_preload:
            ok = 0
            for i in range(len(self.items)):
                mm = self._load_motion_preprocessed(i)
                if mm is not None:
                    self._motion_cache[i] = mm
                    ok += 1
            print(f"[INFO] Preloaded motions into RAM: {ok}/{len(self.items)}")

    def __len__(self):
        return len(self.items)

    def _resolve_motion_path(self, item: dict) -> Optional[Path]:
        bvh_rel = item.get("bvh", None) or item.get("motion", None) or item.get("motion_path", None)
        if not bvh_rel:
            return None
        bvh = _resolve(self.base_dir, bvh_rel)
        return bvh if bvh.exists() else None

    def _load_motion_preprocessed(self, idx: int, min_frames: Optional[int] = None) -> Optional[Dict[str, np.ndarray]]:
        item = self.items[idx]
        bvh = self._resolve_motion_path(item)
        if bvh is None:
            return None

        raw_motion, frame_time = load_bvh_channels(bvh)
        if raw_motion is None or raw_motion.ndim != 2:
            return None

        raw_motion = unwrap_bvh_angles_degrees(raw_motion, pos_dims=3)
        raw_motion = resample_motion_linear(raw_motion, frame_time, TARGET_FPS)

        req_frames = self.block_size if min_frames is None else int(min_frames)
        if len(raw_motion) < req_frames:
            return None

        if self.motion_spec is None:
            raise RuntimeError("MotionWindowDataset now requires a motion_spec for shared BVH conversion.")

        full_motion = convert_bvh_to_6d_with_channel_order(
            raw_motion,
            self.motion_spec.skel,
            self.motion_spec.active_joint_map,
        ).astype(np.float32)
        motion = split_motion_into_parts(full_motion, self.motion_spec, fps=TARGET_FPS).astype(np.float32)

        if not np.isfinite(full_motion).all() or not np.isfinite(motion).all():
            return None
        if motion.shape[1] <= 0:
            return None
        return {"motion": motion, "full_motion": full_motion}

    def _get_speech_segments(self, item: dict, T_frames: int) -> List[Tuple[int, int]]:
        if not self.use_textgrid:
            return []
        tg_rel = item.get("textgrid", None) or item.get("TextGrid", None)
        if not tg_rel:
            return []

        tg_path = _resolve(self.base_dir, tg_rel)
        key = str(tg_path.resolve()) if tg_path.exists() else str(tg_path)
        if key in self._seg_cache:
            segs = self._seg_cache[key]
            return [(max(0, a), min(T_frames, b)) for a, b in segs if min(T_frames, b) > max(0, a)]

        if not tg_path.exists():
            self._seg_cache[key] = []
            return []

        tiers = parse_textgrid_interval_tiers(tg_path)
        tier = choose_best_tier(tiers, prefer=self.tier_name)
        if tier is None:
            self._seg_cache[key] = []
            return []

        segs = build_speech_segments_from_intervals(
            intervals=tiers.get(tier, []),
            fps=TARGET_FPS,
            T_frames=T_frames,
            merge_silence_s=self.merge_silence_s,
            pad_s=self.pad_s,
        )
        self._seg_cache[key] = segs
        return segs

    def __getitem__(self, idx):
        try:
            item = self._motion_cache.get(idx, None) if self.cache_in_mem else None
            if item is None:
                item = self._load_motion_preprocessed(idx)
                if item is None:
                    return None
                if self.cache_in_mem:
                    self._motion_cache[idx] = item

            motion = item["motion"]
            full_motion = item["full_motion"]
            T = int(motion.shape[0])

            if self.is_train and (not self.fixed_crop):
                use_speech = random.random() < self.speech_prob
                if use_speech:
                    segs = [(a, b) for a, b in self._get_speech_segments(self.items[idx], T) if (b - a) >= self.block_size]
                    if len(segs) > 0:
                        a, b = random.choice(segs)
                        start = random.randint(a, max(a, b - self.block_size))
                    else:
                        start = random.randint(0, T - self.block_size)
                else:
                    start = random.randint(0, T - self.block_size)
            else:
                start = 0

            end = start + self.block_size
            w = motion[start:end].astype(np.float32, copy=False)
            w = (w - self.mean) / self.std
            if not np.isfinite(w).all():
                return None

            if self.return_full_gt:
                return {
                    "motion": torch.from_numpy(w),
                    "full_motion": torch.from_numpy(full_motion[start:end].astype(np.float32, copy=False)),
                }
            return torch.from_numpy(w)
        except Exception:
            return None


def collate_motion(batch):
    return _collate_motion_like(batch)


# -----------------------------
# EMA Vector Quantizer (stable)
# -----------------------------

# -----------------------------
# Spherical EMA Vector Quantizer (L2-Normalized)
# -----------------------------
class VectorQuantizerEMA(nn.Module):
    """
    Spherical DDP-safe EMA VQ
    + L2 Normalization (Anti-collapse & AMP safe)
    + soft usage entropy bonus
    + dead code revival
    """
    def __init__(
        self,
        n_codes: int,
        code_dim: int,
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        usage_entropy_w: float = 0.0,
        usage_temp: float = 0.5,
        revive_threshold: float = 1.0,
    ):
        super().__init__()
        self.n_codes = int(n_codes)
        self.code_dim = int(code_dim)
        self.beta = float(beta)
        self.decay = float(decay)
        self.eps = float(eps)

        self.usage_entropy_w = float(usage_entropy_w)
        self.usage_temp = float(usage_temp)
        self.revive_threshold = float(revive_threshold)

        # 初始化时直接让 Codebook 落在单位球面上
        self.codebook = nn.Embedding(self.n_codes, self.code_dim)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=1.0)
        with torch.no_grad():
            self.codebook.weight.data = F.normalize(self.codebook.weight.data, p=2, dim=-1)

        self.register_buffer("ema_cluster_size", torch.zeros(self.n_codes))
        self.register_buffer("ema_w", self.codebook.weight.data.clone())

        self.codebook.weight.requires_grad_(False)

        # runtime stats
        self.last_usage_loss = torch.tensor(0.0)
        self.last_usage_entropy = 0.0
        self.last_active_codes = 0.0

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, codes: torch.Tensor):
        # z 这里接收到的已经是归一化后的特征
        K = self.n_codes
        one_hot = F.one_hot(codes, K).type(z.dtype)   # [N, K]
        cluster_size = one_hot.sum(dim=0)             # [K]
        dw = one_hot.t() @ z                          # [K, C]

        if dist.is_initialized():
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)
            dist.all_reduce(dw, op=dist.ReduceOp.SUM)

        self.ema_cluster_size.mul_(self.decay).add_(cluster_size * (1.0 - self.decay))
        self.ema_w.mul_(self.decay).add_(dw * (1.0 - self.decay))

        n = self.ema_cluster_size.sum()
        smoothed_cluster_size = (self.ema_cluster_size + self.eps) / (n + K * self.eps) * n
        embed = self.ema_w / smoothed_cluster_size.unsqueeze(1)
        
        # 🔑 关键修改 1：EMA 更新后，再次将密码本投影回单位球面
        embed_normalized = F.normalize(embed, p=2, dim=1)
        self.codebook.weight.data.copy_(embed_normalized)

        # dead code revival
        dead_codes = smoothed_cluster_size < self.revive_threshold
        if dead_codes.any():
            num_dead = int(dead_codes.sum().item())

            if (not dist.is_initialized()) or dist.get_rank() == 0:
                if z.shape[0] >= num_dead:
                    rand_idx = torch.randperm(z.shape[0], device=z.device)[:num_dead]
                else:
                    rand_idx = torch.randint(0, z.shape[0], (num_dead,), device=z.device)
                new_embeddings = z[rand_idx].detach()
            else:
                new_embeddings = torch.empty((num_dead, z.shape[1]), device=z.device, dtype=z.dtype)

            if dist.is_initialized():
                dist.broadcast(new_embeddings, src=0)

            # 复活的也是归一化后的 z，所以直接赋值即可
            self.codebook.weight.data[dead_codes] = new_embeddings
            self.ema_w.data[dead_codes] = new_embeddings * self.eps
            self.ema_cluster_size.data[dead_codes] = self.eps

    def forward(self, z_e: torch.Tensor):
        B, C, T = z_e.shape
        
        # 🔑 关键修改 2：强制隔离 AMP，防止 FP16 计算点积时溢出
        device_type = "cuda" if z_e.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            z_e_fp32 = z_e.float()
            
            # 🔑 关键修改 3：对 Encoder 提取的特征进行 L2 归一化
            z_e_norm = F.normalize(z_e_fp32, p=2, dim=1)  # 沿着 Channel 维度 [B, C, T]
            
            z = z_e_norm.permute(0, 2, 1).contiguous().view(-1, C)   # [N, C]
            
            # 对 Codebook 也进行归一化（虽然 EMA 里归一化过了，这里是为了计算万无一失）
            e_norm = F.normalize(self.codebook.weight.float(), p=2, dim=1) # [K, C]

            # 在球面上计算距离: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e 
            z2 = (z ** 2).sum(dim=1, keepdim=True)                   
            e2 = (e_norm ** 2).sum(dim=1).unsqueeze(0)                    
            ze = z @ e_norm.t()                                           
            distances = z2 + e2 - 2.0 * ze                           

        # 找最近的聚类中心
        codes = torch.argmin(distances, dim=1)                   # [N]
        
        # 提取量化后的特征 (必须是从 e_norm 里取)
        z_q = e_norm[codes].view(B, T, C).permute(0, 2, 1).contiguous()
        z_q_fp32 = z_q.float()

        # VQ loss (基于归一化后的特征计算)
        vq_loss = self.beta * F.mse_loss(z_e_norm, z_q_fp32.detach())
        
        # Straight-Through Estimator：让 Decoder 接收归一化后的特征
        z_q_st = z_e_norm + (z_q_fp32 - z_e_norm).detach()
        z_q_st = z_q_st.to(z_e.dtype)

        # EMA update
        if self.training:
            self._ema_update(z, codes)

        # hard perplexity
        with torch.no_grad():
            one_hot = F.one_hot(codes, self.n_codes).float()
            avg_probs = one_hot.mean(dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            active_codes = float((avg_probs > 1e-5).sum().item())

        # soft usage entropy bonus (防坍缩)
        if self.usage_entropy_w > 0.0:
            # 在球面上，最大距离是 4，默认 usage_temp=0.5 非常合适
            p_soft = torch.softmax(-distances / max(self.usage_temp, 1e-6), dim=1)   
            avg_p_soft = p_soft.mean(dim=0)                                          
            usage_entropy = -torch.sum(avg_p_soft * torch.log(avg_p_soft + 1e-10))
            usage_entropy_norm = usage_entropy / math.log(self.n_codes)
            usage_loss = -self.usage_entropy_w * usage_entropy_norm
            self.last_usage_loss = usage_loss.to(z_e.dtype)
            self.last_usage_entropy = float(usage_entropy_norm.detach().item())
        else:
            self.last_usage_loss = z_e.new_tensor(0.0)
            self.last_usage_entropy = 0.0

        self.last_active_codes = active_codes

        codes = codes.view(B, T)
        return z_q_st, vq_loss.to(z_e.dtype), codes, perplexity

'''
class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        n_codes: int,
        code_dim: int,
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        usage_entropy_w: float = 0.0,
        usage_temp: float = 0.5,
        revive_threshold: float = 1.0,
    ):
        super().__init__()
        self.n_codes = int(n_codes)
        self.code_dim = int(code_dim)
        self.beta = float(beta)
        self.decay = float(decay)
        self.eps = float(eps)

        self.usage_entropy_w = float(usage_entropy_w)
        self.usage_temp = float(usage_temp)
        self.revive_threshold = float(revive_threshold)

        self.codebook = nn.Embedding(self.n_codes, self.code_dim)
        bound = 1.0 / math.sqrt(self.code_dim)
        nn.init.uniform_(self.codebook.weight, -bound, bound)

        self.register_buffer("ema_cluster_size", torch.zeros(self.n_codes))
        self.register_buffer("ema_w", self.codebook.weight.data.clone())

        self.codebook.weight.requires_grad_(False)

        # runtime stats (not in state_dict)
        self.last_usage_loss = torch.tensor(0.0)
        self.last_usage_entropy = 0.0
        self.last_active_codes = 0.0

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, codes: torch.Tensor):
        K = self.n_codes
        one_hot = F.one_hot(codes, K).type(z.dtype)   # [N, K]
        cluster_size = one_hot.sum(dim=0)             # [K]
        dw = one_hot.t() @ z                          # [K, C]

        if dist.is_initialized():
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)
            dist.all_reduce(dw, op=dist.ReduceOp.SUM)

        self.ema_cluster_size.mul_(self.decay).add_(cluster_size * (1.0 - self.decay))
        self.ema_w.mul_(self.decay).add_(dw * (1.0 - self.decay))

        n = self.ema_cluster_size.sum()
        smoothed_cluster_size = (self.ema_cluster_size + self.eps) / (n + K * self.eps) * n
        embed = self.ema_w / smoothed_cluster_size.unsqueeze(1)
        self.codebook.weight.data.copy_(embed)

        # dead code revival
        dead_codes = smoothed_cluster_size < self.revive_threshold
        if dead_codes.any():
            num_dead = int(dead_codes.sum().item())

            if (not dist.is_initialized()) or dist.get_rank() == 0:
                if z.shape[0] >= num_dead:
                    rand_idx = torch.randperm(z.shape[0], device=z.device)[:num_dead]
                else:
                    rand_idx = torch.randint(0, z.shape[0], (num_dead,), device=z.device)
                new_embeddings = z[rand_idx].detach()
            else:
                new_embeddings = torch.empty((num_dead, z.shape[1]), device=z.device, dtype=z.dtype)

            if dist.is_initialized():
                dist.broadcast(new_embeddings, src=0)

            self.codebook.weight.data[dead_codes] = new_embeddings
            self.ema_w.data[dead_codes] = new_embeddings * self.eps
            self.ema_cluster_size.data[dead_codes] = self.eps

    def forward(self, z_e: torch.Tensor):
        """
        z_e: [B, C, T]
        return:
            z_q_st: [B, C, T]
            vq_loss
            codes: [B, T]
            perplexity
        """
        B, C, T = z_e.shape
        z_e_fp32 = z_e.float()
        z = z_e_fp32.permute(0, 2, 1).contiguous().view(-1, C)   # [N, C]
        e = self.codebook.weight.float()                         # [K, C]

        device_type = "cuda" if z_e.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # 再次确保变量是 float32（防止外部 autocast 污染）
            z32 = z.float()
            e32 = e.float()
            z2 = (z32 ** 2).sum(dim=1, keepdim=True)
            e2 = (e32 ** 2).sum(dim=1).unsqueeze(0)
            ze = z32 @ e32.t()
            distances = z2 + e2 - 2.0 * ze

        codes = torch.argmin(distances, dim=1)                   # [N]
        z_q = self.codebook(codes).view(B, T, C).permute(0, 2, 1).contiguous()
        z_q_fp32 = z_q.float()

        # standard VQ loss
        vq_loss = self.beta * F.mse_loss(z_e_fp32, z_q_fp32.detach())
        z_q_st = z_e + (z_q.to(z_e.dtype) - z_e).detach()

        # EMA update
        if self.training:
            self._ema_update(z, codes)

        # hard perplexity
        with torch.no_grad():
            one_hot = F.one_hot(codes, self.n_codes).float()
            avg_probs = one_hot.mean(dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            active_codes = float((avg_probs > 1e-5).sum().item())

        # soft usage entropy bonus (anti-collapse)
        if self.usage_entropy_w > 0.0:
            p_soft = torch.softmax(-distances / max(self.usage_temp, 1e-6), dim=1)   # [N, K]
            avg_p_soft = p_soft.mean(dim=0)                                           # [K]
            usage_entropy = -torch.sum(avg_p_soft * torch.log(avg_p_soft + 1e-10))
            usage_entropy_norm = usage_entropy / math.log(self.n_codes)
            usage_loss = -self.usage_entropy_w * usage_entropy_norm
            self.last_usage_loss = usage_loss.to(z_e.dtype)
            self.last_usage_entropy = float(usage_entropy_norm.detach().item())
        else:
            self.last_usage_loss = z_e.new_tensor(0.0)
            self.last_usage_entropy = 0.0

        self.last_active_codes = active_codes

        codes = codes.view(B, T)
        return z_q_st, vq_loss.to(z_e.dtype), codes, perplexity

'''

# -----------------------------
# VQ-VAE model (1D conv) with slower token rate and less upsampling artifacts
# -----------------------------
class UpConv1d(nn.Module):
    """
    Upsample (linear) + Conv1d to avoid ConvTranspose1d periodic artifacts.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))

class ResBlock1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.block(x)


class GlobalResBlock1d(nn.Module):
    """
    LoM-style residual block used by the global AE:
    Conv1d -> LeakyReLU -> Conv1d + residual, without temporal downsampling.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


def init_conv1d_xavier(m: nn.Module):
    if isinstance(m, (nn.Conv1d, nn.Linear, nn.ConvTranspose1d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class MotionVQVAE(nn.Module):
    def __init__(
        self,
        motion_dim: int,
        hidden: int = 512,
        code_dim: int = 256,
        n_codes: int = 1024,
        beta: float = 0.25,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        n_downsample: int = 2,
        vq_usage_entropy_w: float = 0.0,
        vq_usage_temp: float = 0.5,
        vq_revive_threshold: float = 1.0,
    ):
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.hidden = int(hidden)
        self.code_dim = int(code_dim)
        self.n_downsample = int(n_downsample)

        # -------- Encoder --------
        enc_layers: List[nn.Module] = []
        enc_layers.append(nn.Conv1d(self.motion_dim, hidden, kernel_size=3, stride=1, padding=1))
        enc_layers.append(nn.ReLU(inplace=True))

        for _ in range(self.n_downsample):
            enc_layers.append(nn.Conv1d(hidden, hidden, kernel_size=4, stride=2, padding=1))
            enc_layers.append(nn.ReLU(inplace=True))
            enc_layers.append(ResBlock1d(hidden))
            enc_layers.append(ResBlock1d(hidden))

        enc_layers.append(nn.Conv1d(hidden, code_dim, kernel_size=3, stride=1, padding=1))
        self.enc = nn.Sequential(*enc_layers)

        # -------- Quantizer --------
        self.vq = VectorQuantizerEMA(
            n_codes=n_codes,
            code_dim=code_dim,
            beta=beta,
            decay=ema_decay,
            eps=ema_eps,
            usage_entropy_w=vq_usage_entropy_w,
            usage_temp=vq_usage_temp,
            revive_threshold=vq_revive_threshold,
        )

        # -------- Decoder --------
        dec_layers: List[nn.Module] = []
        dec_layers.append(nn.Conv1d(code_dim, hidden, kernel_size=3, stride=1, padding=1))
        dec_layers.append(nn.ReLU(inplace=True))

        for _ in range(self.n_downsample):
            dec_layers.append(ResBlock1d(hidden))
            dec_layers.append(ResBlock1d(hidden))
            dec_layers.append(UpConv1d(hidden, hidden))
            dec_layers.append(nn.ReLU(inplace=True))

        dec_layers.append(nn.Conv1d(hidden, self.motion_dim, kernel_size=3, stride=1, padding=1))
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor, use_vq: bool = True):
        x1 = x.transpose(1, 2).contiguous()   # [B, D, T]
        z_e = self.enc(x1)

        if use_vq:
            z_q, vq_loss, codes, ppl = self.vq(z_e)
        else:
            z_q = z_e
            vq_loss = z_e.new_tensor(0.0)
            codes = torch.zeros((x.shape[0], z_e.shape[-1]), device=x.device, dtype=torch.long)
            ppl = z_e.new_tensor(0.0)
            self.vq.last_usage_loss = z_e.new_tensor(0.0)
            self.vq.last_usage_entropy = 0.0
            self.vq.last_active_codes = 0.0

        x_hat = self.dec(z_q).transpose(1, 2).contiguous()
        return x_hat, vq_loss, codes, ppl, z_e


class MotionGlobalAE(nn.Module):
    """
    LoM-like global branch:
    - no vector quantization
    - no temporal downsampling
    - Conv1d + residual blocks over the full frame rate
    """
    def __init__(self, motion_dim: int, hidden: int = 256, n_layers: int = 4):
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.hidden = int(hidden)
        self.n_layers = int(n_layers)

        enc_layers: List[nn.Module] = [
            nn.Conv1d(self.motion_dim, self.hidden, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            GlobalResBlock1d(self.hidden),
        ]
        for _ in range(1, self.n_layers):
            enc_layers.extend([
                nn.Conv1d(self.hidden, self.hidden, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                GlobalResBlock1d(self.hidden),
            ])
        self.enc = nn.Sequential(*enc_layers)

        dec_layers: List[nn.Module] = [
            GlobalResBlock1d(self.hidden),
            GlobalResBlock1d(self.hidden),
        ]
        for _ in range(max(0, self.n_layers - 1)):
            dec_layers.extend([
                nn.Conv1d(self.hidden, self.hidden, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        dec_layers.extend([
            nn.Conv1d(self.hidden, self.motion_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.motion_dim, self.motion_dim, kernel_size=3, stride=1, padding=1),
        ])
        self.dec = nn.Sequential(*dec_layers)

        self.apply(init_conv1d_xavier)

    def forward(self, x: torch.Tensor, use_vq: bool = False):
        x1 = x.transpose(1, 2).contiguous()
        z_e = self.enc(x1)
        x_hat = self.dec(z_e).transpose(1, 2).contiguous()
        codes = torch.zeros((x.shape[0], x.shape[1]), device=x.device, dtype=torch.long)
        vq_loss = x.new_tensor(0.0)
        ppl = x.new_tensor(0.0)
        return x_hat, vq_loss, codes, ppl, z_e
# -----------------------------
# Losses
# -----------------------------
def velocity_loss(x_hat: torch.Tensor, x: torch.Tensor, w: float = 1.0):
    if w <= 0:
        return x_hat.new_tensor(0.0)
    vx = x[:, 1:] - x[:, :-1]
    vhat = x_hat[:, 1:] - x_hat[:, :-1]
    return w * F.smooth_l1_loss(vhat, vx)


def acceleration_loss(x_hat: torch.Tensor, x: torch.Tensor, w: float = 0.05):
    if w <= 0:
        return x_hat.new_tensor(0.0)
    ax = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    ah = x_hat[:, 2:] - 2 * x_hat[:, 1:-1] + x_hat[:, :-2]
    return w * F.smooth_l1_loss(ah, ax)


def latent_smoothness_loss(z_e: torch.Tensor, w: float = 0.0):
    if w <= 0:
        return z_e.new_tensor(0.0)
    dz = z_e[:, :, 1:] - z_e[:, :, :-1]
    return w * torch.mean(dz ** 2)

def fk_joint_losses(
    pos_gt: torch.Tensor,
    pos_pd: torch.Tensor,
    joint_ids: Optional[Union[List[int], torch.Tensor]],
    pos_w: float = 0.0,
    vel_w: float = 0.0,
    acc_w: float = 0.0,
):
    """
    pos_gt / pos_pd: [B, T, J, 3]
    joint_ids: selected joints, e.g. hands (treated as end-effectors here)
    """
    zero = pos_pd.new_tensor(0.0)

    if joint_ids is None:
        return zero, zero, zero
    if isinstance(joint_ids, list) and len(joint_ids) == 0:
        return zero, zero, zero

    if not torch.is_tensor(joint_ids):
        joint_ids = torch.as_tensor(joint_ids, device=pos_pd.device, dtype=torch.long)
    else:
        joint_ids = joint_ids.to(device=pos_pd.device, dtype=torch.long)

    gt = pos_gt[:, :, joint_ids, :]   # [B, T, K, 3]
    pd = pos_pd[:, :, joint_ids, :]

    pos_loss = zero
    vel_loss = zero
    acc_loss = zero

    if pos_w > 0.0:
        pos_loss = pos_w * F.smooth_l1_loss(pd, gt)

    if vel_w > 0.0 and gt.shape[1] >= 2:
        gt_v = gt[:, 1:] - gt[:, :-1]
        pd_v = pd[:, 1:] - pd[:, :-1]
        vel_loss = vel_w * F.smooth_l1_loss(pd_v, gt_v)

    if acc_w > 0.0 and gt.shape[1] >= 3:
        gt_a = gt[:, 2:] - 2.0 * gt[:, 1:-1] + gt[:, :-2]
        pd_a = pd[:, 2:] - 2.0 * pd[:, 1:-1] + pd[:, :-2]
        acc_loss = acc_w * F.smooth_l1_loss(pd_a, gt_a)

    return pos_loss, vel_loss, acc_loss

# -----------------------------
# Resume compatibility helpers (关键修复：处理 *.conv.* 命名差异)
# -----------------------------
def remap_strip_conv_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    dec.2.conv.weight -> dec.2.weight
    dec.2.conv.bias   -> dec.2.bias
    and any '*.conv.*' -> '*.*'
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k
        nk = nk.replace(".conv.weight", ".weight")
        nk = nk.replace(".conv.bias", ".bias")
        nk = nk.replace(".conv.", ".")
        out[nk] = v
    return out


def remap_add_conv_keys_if_needed(state_dict: Dict[str, torch.Tensor], model_keys: set) -> Dict[str, torch.Tensor]:
    """
    If model expects ".conv.weight" but ckpt provides ".weight", rewrite accordingly.
    Only rewrites when the rewritten key exists in model_keys.
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k
        if k.endswith(".weight"):
            cand = k[:-7] + ".conv.weight"
            if cand in model_keys:
                nk = cand
        elif k.endswith(".bias"):
            cand = k[:-5] + ".conv.bias"
            if cand in model_keys:
                nk = cand
        out[nk] = v
    return out


def load_state_dict_compat(model: nn.Module, state_dict: Dict[str, torch.Tensor], strict: bool = True):
    """
    Try loading as-is. If fails due to legacy naming differences, try remaps:
      1) strip ".conv."
      2) add ".conv." when model expects it
    """
    try:
        model.load_state_dict(state_dict, strict=strict)
        print("[INFO] Loaded checkpoint state_dict (as-is).")
        return

    except RuntimeError as e1:
        msg1 = str(e1)
        # try strip conv keys
        try:
            sd2 = remap_strip_conv_keys(state_dict)
            model.load_state_dict(sd2, strict=strict)
            print("[WARN] Loaded checkpoint after stripping '*.conv.*' keys for compatibility.")
            return
        except RuntimeError as e2:
            # try add conv keys
            try:
                mk = set(model.state_dict().keys())
                sd3 = remap_add_conv_keys_if_needed(state_dict, mk)
                model.load_state_dict(sd3, strict=strict)
                print("[WARN] Loaded checkpoint after adding '*.conv.*' keys for compatibility.")
                return
            except RuntimeError as e3:
                print("[ERROR] Failed to load checkpoint with compatibility remaps.")
                print("---- First error (as-is) ----")
                print(msg1)
                print("---- Second error (strip conv) ----")
                print(str(e2))
                print("---- Third error (add conv) ----")
                print(str(e3))
                raise


# -----------------------------
# Train
# -----------------------------
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    is_train: bool,
    recon_w: float,
    part_recon_norm_w: float,
    part_recon_denorm_w: float,
    vel_w: float,
    acc_w: float,
    vq_w_eff: float,
    z_smooth_w: float,
    amp: bool,
    scaler,
    autocast_ctx,
    norm_mean_t=None,
    norm_std_t=None,

    fk: Optional[BVHFkTorch] = None,
    foot_joint_ids: Optional[List[int]] = None,
    fk_foot_pos_w: float = 0.0,
    fk_foot_lock_w: float = 0.0,
    fk_contact_vel_th: float = 0.5,

    vq_seq_usage_w: float = 0.0,

    hand_joint_ids: Optional[List[int]] = None,
    fk_hand_pos_w: float = 0.0,
    fk_hand_vel_w: float = 0.0,
    fk_hand_acc_w: float = 0.0,

    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    motion_spec: Optional[MotionPartSpec] = None,
):
    model.train() if is_train else model.eval()

    rank = dist.get_rank() if dist.is_initialized() else 0
    pbar_disable = dist.is_initialized() and rank != 0

    raw_model = model.module if hasattr(model, "module") else model

    tot = 0.0
    tot_recon = 0.0
    tot_recon_norm = 0.0
    tot_recon_denorm = 0.0
    tot_vel = 0.0
    tot_acc = 0.0
    tot_vq = 0.0
    tot_zsm = 0.0
    tot_ppl = 0.0

    tot_usage = 0.0
    tot_usage_ent = 0.0
    tot_active_codes = 0.0

    tot_footpos = 0.0
    tot_footlock = 0.0

    tot_handpos = 0.0
    tot_handvel = 0.0
    tot_handacc = 0.0
    tot_global_contact = 0.0

    tot_seq_usage = 0.0
    tot_seq_ent = 0.0
    tot_seq_distinct = 0.0
    tot_seq_ppl = 0.0
    n = 0

    foot_active = (
        (fk is not None)
        and (foot_joint_ids is not None)
        and (len(foot_joint_ids) > 0)
        and ((fk_foot_pos_w > 0.0) or (fk_foot_lock_w > 0.0))
    )

    hand_active = (
        (fk is not None)
        and (hand_joint_ids is not None)
        and (len(hand_joint_ids) > 0)
        and ((fk_hand_pos_w > 0.0) or (fk_hand_vel_w > 0.0) or (fk_hand_acc_w > 0.0))
    )

    any_fk = foot_active or hand_active

    mean_t = None
    std_t = None
    foot_sel = None
    hand_sel = None
    global_active = bool(motion_spec is not None and motion_spec.part == "global")

    if any_fk or global_active:
        if mean is None or std is None:
            raise RuntimeError("Global/FK branches require mean/std in run_epoch.")
        mean_t = torch.as_tensor(mean, device=device).view(1, 1, -1)
        std_t = torch.as_tensor(std, device=device).view(1, 1, -1)

        if foot_active:
            foot_sel = torch.as_tensor(foot_joint_ids, device=device, dtype=torch.long)
        if hand_active:
            hand_sel = torch.as_tensor(hand_joint_ids, device=device, dtype=torch.long)

    if is_train:
        optim.zero_grad(set_to_none=True)

    for batch in tqdm(loader, desc="train" if is_train else "val", disable=pbar_disable):
        if batch is None:
            continue

        full_motion_gt = None
        if isinstance(batch, dict):
            x = batch["motion"].to(device, non_blocking=True)
            if "full_motion" in batch and torch.is_tensor(batch["full_motion"]):
                full_motion_gt = batch["full_motion"].to(device, non_blocking=True)
        else:
            x = batch.to(device, non_blocking=True)  # [B, T, D]
        if (norm_mean_t is not None) and (norm_std_t is not None):
            x = (x - norm_mean_t) / norm_std_t

        x_model = x
        x_den_for_aux = None
        if global_active:
            if mean_t is None or std_t is None:
                raise RuntimeError("Global training requires mean/std tensors.")
            x_den_for_aux = x * std_t.to(dtype=x.dtype) + mean_t.to(dtype=x.dtype)
            x_model_raw = prepare_global_condition_input(x_den_for_aux, motion_spec)
            x_model = (x_model_raw - mean_t.to(dtype=x.dtype)) / std_t.to(dtype=x.dtype)

        with autocast_ctx(amp):
            use_vq = bool((vq_w_eff > 0.0) and (motion_spec is None or motion_spec.uses_vq))
            x_hat, vq_loss, codes, ppl, z_e = model(x_model, use_vq=use_vq)

            global_contact_loss = x_hat.new_tensor(0.0)
            x_den = None
            xh_den = None
            recon_norm_metric = x_hat.new_tensor(0.0)
            recon_denorm_metric = x_hat.new_tensor(0.0)
            if global_active:
                if x_den_for_aux is None or mean_t is None or std_t is None:
                    raise RuntimeError("Global branch failed to prepare denormalized targets.")
                x_den = x_den_for_aux.to(dtype=x_hat.dtype)
                xh_den = x_hat * std_t.to(dtype=x_hat.dtype) + mean_t.to(dtype=x_hat.dtype)
                global_total, global_metrics = global_motion_loss(xh_den, x_den, motion_spec, fps=TARGET_FPS)
                recon = global_metrics["root_l1"].to(device=x_hat.device, dtype=x_hat.dtype)
                recon_denorm_metric = recon
                vel = global_metrics["trans_vel_l1"].to(device=x_hat.device, dtype=x_hat.dtype)
                acc = global_metrics["smooth_l1"].to(device=x_hat.device, dtype=x_hat.dtype)
                global_contact_loss = global_metrics["contact_mse"].to(device=x_hat.device, dtype=x_hat.dtype)
            else:
                if (
                    (motion_spec is not None)
                    and (motion_spec.part != "full")
                    and (mean_t is not None)
                    and (std_t is not None)
                ):
                    # Part-wise channels can have extremely small std after the split.
                    # Compute reconstruction/dynamics losses in denormalized space so
                    # the model is optimized for actual motion error instead of being
                    # dominated by tiny-variance channels.
                    recon_norm_metric = F.smooth_l1_loss(x_hat, x)
                    x_den = x * std_t.to(dtype=x_hat.dtype) + mean_t.to(dtype=x_hat.dtype)
                    xh_den = x_hat * std_t.to(dtype=x_hat.dtype) + mean_t.to(dtype=x_hat.dtype)
                    recon_denorm_metric = F.smooth_l1_loss(xh_den, x_den)
                    recon = (part_recon_norm_w * recon_norm_metric) + (part_recon_denorm_w * recon_denorm_metric)
                    vel = velocity_loss(xh_den, x_den, w=vel_w)
                    acc = acceleration_loss(xh_den, x_den, w=acc_w)
                else:
                    recon = F.smooth_l1_loss(x_hat, x)
                    recon_norm_metric = recon
                    vel = velocity_loss(x_hat, x, w=vel_w)
                    acc = acceleration_loss(x_hat, x, w=acc_w)
            zsm = latent_smoothness_loss(z_e, w=z_smooth_w)

            usage_loss = x_hat.new_tensor(0.0)
            usage_ent_scalar = 0.0
            active_codes_scalar = 0.0

            if use_vq and hasattr(raw_model, "vq"):
                usage_loss = raw_model.vq.last_usage_loss.to(device=x_hat.device, dtype=x_hat.dtype)
                usage_ent_scalar = float(getattr(raw_model.vq, "last_usage_entropy", 0.0))
                active_codes_scalar = float(getattr(raw_model.vq, "last_active_codes", 0.0))
            
            seq_usage_loss = x_hat.new_tensor(0.0)
            seq_usage_ent_scalar = 0.0
            seq_distinct_scalar = 0.0
            seq_ppl_scalar = 0.0

            if use_vq:
                seq_usage_loss, seq_usage_ent_scalar, seq_distinct_scalar, seq_ppl_scalar = \
                    sequence_usage_entropy_loss(
                        codes=codes,
                        n_codes=raw_model.vq.n_codes,
                        w=vq_seq_usage_w,
                    )
                seq_usage_loss = seq_usage_loss.to(device=x_hat.device, dtype=x_hat.dtype)

            foot_pos_loss = x_hat.new_tensor(0.0)
            foot_lock_loss = x_hat.new_tensor(0.0)
            hand_pos_loss = x_hat.new_tensor(0.0)
            hand_vel_loss = x_hat.new_tensor(0.0)
            hand_acc_loss = x_hat.new_tensor(0.0)

            if any_fk:
                if x_den is None or xh_den is None:
                    x_den = x * std_t.to(dtype=x_hat.dtype) + mean_t.to(dtype=x_hat.dtype)
                    xh_den = x_hat * std_t.to(dtype=x_hat.dtype) + mean_t.to(dtype=x_hat.dtype)

                if (motion_spec is not None) and (motion_spec.part != "full") and (full_motion_gt is not None):
                    full_gt = full_motion_gt.to(dtype=x_hat.dtype)
                    full_pd = merge_parts_back_to_full(xh_den, motion_spec, gt_full=full_gt)
                    pos_gt = fk.fk_positions(full_gt)   # [B, T, J, 3]
                    pos_pd = fk.fk_positions(full_pd)   # [B, T, J, 3]
                else:
                    pos_gt = fk.fk_positions(x_den)   # [B, T, J, 3]
                    pos_pd = fk.fk_positions(xh_den)  # [B, T, J, 3]

                if foot_active and (foot_sel is not None):
                    gt_f = pos_gt[:, :, foot_sel, :]
                    pd_f = pos_pd[:, :, foot_sel, :]

                    if fk_foot_pos_w > 0.0:
                        foot_pos_loss = fk_foot_pos_w * F.smooth_l1_loss(pd_f, gt_f)

                    if fk_foot_lock_w > 0.0 and gt_f.shape[1] >= 2:
                        v_gt = gt_f[:, 1:] - gt_f[:, :-1]
                        gt_speed = torch.linalg.norm(v_gt, dim=-1)
                        mask = (gt_speed < float(fk_contact_vel_th)).to(x_hat.dtype)

                        v_pd = pd_f[:, 1:] - pd_f[:, :-1]
                        l = F.smooth_l1_loss(v_pd, torch.zeros_like(v_pd), reduction="none").mean(dim=-1)
                        denom = mask.sum() + 1e-6
                        foot_lock_loss = fk_foot_lock_w * (l * mask).sum() / denom

                if hand_active and (hand_sel is not None):
                    hand_pos_loss, hand_vel_loss, hand_acc_loss = fk_joint_losses(
                        pos_gt=pos_gt,
                        pos_pd=pos_pd,
                        joint_ids=hand_sel,
                        pos_w=fk_hand_pos_w,
                        vel_w=fk_hand_vel_w,
                        acc_w=fk_hand_acc_w,
                    )

            if global_active:
                loss = (
                    recon_w * (recon + vel + acc + global_contact_loss)
                    + zsm
                    + foot_pos_loss
                    + foot_lock_loss
                    + hand_pos_loss
                    + hand_vel_loss
                    + hand_acc_loss
                )
            else:
                loss = (
                    recon_w * recon
                    + vq_w_eff * vq_loss
                    + vel
                    + acc
                    + zsm
                    + usage_loss
                    + seq_usage_loss
                    + foot_pos_loss
                    + foot_lock_loss
                    + hand_pos_loss
                    + hand_vel_loss
                    + hand_acc_loss
                )

# ----- DDP-safe finite check -----
        debug_terms = {
            "loss": loss,
            "recon": recon,
            "recon_norm": recon_norm_metric,
            "recon_denorm": recon_denorm_metric,
            "vel": vel,
            "acc": acc,
            "vq": vq_loss,
            "zsm": zsm,
            "usage": usage_loss,
            "global_contact": global_contact_loss,
            "foot_pos": foot_pos_loss,
            "foot_lock": foot_lock_loss,
            "hand_pos": hand_pos_loss,
            "hand_vel": hand_vel_loss,
            "hand_acc": hand_acc_loss,
        }

        all_finite = ddp_all_finite_scalar_dict(debug_terms, device)

        if not all_finite:
            if is_train:
                optim.zero_grad(set_to_none=True)

            rank = dist.get_rank() if dist.is_initialized() else 0

            local_msg = []
            for k, v in debug_terms.items():
                if torch.is_tensor(v):
                    finite = bool(torch.isfinite(v).all().item())
                    try:
                        val = float(v.detach().float().cpu())
                    except Exception:
                        val = None
                else:
                    finite = np.isfinite(v)
                    val = float(v)
                local_msg.append(f"{k}={val} finite={finite}")

            print(f"[RANK {rank}] Non-finite step: " + " | ".join(local_msg), flush=True)

            continue
        if is_train:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

        tot += float(loss.item())
        tot_recon += float(recon.item())
        tot_recon_norm += float(recon_norm_metric.item())
        tot_recon_denorm += float(recon_denorm_metric.item())
        tot_vel += float(vel.item())
        tot_acc += float(acc.item())
        tot_vq += float(vq_loss.item())
        tot_zsm += float(zsm.item())
        tot_ppl += float(ppl.item())

        tot_usage += float(usage_loss.item())
        tot_usage_ent += float(usage_ent_scalar)
        tot_active_codes += float(active_codes_scalar)

        tot_seq_usage += float(seq_usage_loss.item())
        tot_seq_ent += float(seq_usage_ent_scalar)
        tot_seq_distinct += float(seq_distinct_scalar)
        tot_seq_ppl += float(seq_ppl_scalar)

        tot_footpos += float(foot_pos_loss.item())
        tot_footlock += float(foot_lock_loss.item())

        tot_handpos += float(hand_pos_loss.item())
        tot_handvel += float(hand_vel_loss.item())
        tot_handacc += float(hand_acc_loss.item())
        tot_global_contact += float(global_contact_loss.item())

        n += 1

    if n == 0:
        return {}

    return {
        "loss": tot / n,
        "recon": tot_recon / n,
        "recon_norm": tot_recon_norm / n,
        "recon_denorm": tot_recon_denorm / n,
        "vel": tot_vel / n,
        "acc": tot_acc / n,
        "vq": tot_vq / n,
        "zsm": tot_zsm / n,
        "ppl": tot_ppl / n,

        "usage": tot_usage / n,
        "usage_ent": tot_usage_ent / n,
        "active_codes": tot_active_codes / n,

        "seq_usage": tot_seq_usage / n,
        "seq_usage_ent": tot_seq_ent / n,
        "seq_distinct": tot_seq_distinct / n,
        "seq_ppl": tot_seq_ppl / n,

        "footpos": tot_footpos / n,
        "footlock": tot_footlock / n,

        "handpos": tot_handpos / n,
        "handvel": tot_handvel / n,
        "handacc": tot_handacc / n,
        "global_contact": tot_global_contact / n,


        "n": n,
    }

# -----------------------------
# Checkpointing
# -----------------------------
def save_ckpt(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_ckpt(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)

def ddp_any_true(flag: bool, device: torch.device) -> bool:
    """
    DDP-safe boolean OR across all ranks.
    If any rank reports True, returns True on all ranks.
    """
    if not dist.is_initialized():
        return flag
    x = torch.tensor([1 if flag else 0], device=device, dtype=torch.int32)
    dist.all_reduce(x, op=dist.ReduceOp.MAX)
    return bool(x.item())

def ddp_all_finite_scalar_dict(stats: dict, device: torch.device):
    """
    Return:
      all_finite: bool
      flags_per_rank: optional gathered flags on rank0
    """
    keys = sorted(stats.keys())
    vals = []
    for k in keys:
        v = stats[k]
        if torch.is_tensor(v):
            vv = v.detach()
            ok = torch.isfinite(vv).all()
        else:
            vv = torch.tensor(float(v), device=device)
            ok = torch.isfinite(vv).all()
        vals.append(1 if ok else 0)

    flag = torch.tensor([min(vals)], device=device, dtype=torch.int32)

    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)

    return bool(flag.item())

def ddp_all_finite(loss: torch.Tensor) -> bool:
    """
    Return True only if ALL ranks have finite loss.
    """
    flag = torch.tensor(
        [1 if torch.isfinite(loss).all() else 0],
        device=loss.device,
        dtype=torch.int32,
    )
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def ddp_weighted_mean_dict(stats: dict, device: torch.device):
    if not dist.is_initialized() or not stats:
        return stats
    if "n" not in stats:
        raise KeyError("stats must contain 'n' for weighted DDP reduction.")

    local_n = float(stats["n"])
    n_tensor = torch.tensor([local_n], device=device, dtype=torch.float64)
    dist.all_reduce(n_tensor, op=dist.ReduceOp.SUM)
    total_n = float(n_tensor.item())

    out = {"n": total_n}
    for k, v in stats.items():
        if k == "n":
            continue
        vv = torch.tensor([float(v) * local_n], device=device, dtype=torch.float64)
        dist.all_reduce(vv, op=dist.ReduceOp.SUM)
        out[k] = float(vv.item() / max(total_n, 1e-12))
    return out


def sequence_usage_entropy_loss(
    codes: torch.Tensor,
    n_codes: int,
    w: float = 0.0,
):
    """
    codes: [B, Tq] long
    返回:
      loss: 标量，直接加到总 loss
      ent_mean: 平均归一化熵，方便日志
      distinct_mean: 每条序列平均用了多少个不同 code
      ppl_seq_mean: 每条序列平均 perplexity
    """
    if w <= 0.0 or codes is None or codes.numel() == 0:
        z = codes.new_tensor(0.0, dtype=torch.float32) if torch.is_tensor(codes) else torch.tensor(0.0)
        return z, 0.0, 0.0, 0.0

    one_hot = F.one_hot(codes, num_classes=n_codes).float()   # [B, Tq, K]
    p_seq = one_hot.mean(dim=1)                               # [B, K]

    ent = -(p_seq * torch.log(p_seq + 1e-10)).sum(dim=-1)     # [B]
    ent_norm = ent / math.log(n_codes)                        # [B], 0~1
    ppl_seq = torch.exp(ent)                                  # [B]
    distinct = (p_seq > 1e-5).float().sum(dim=-1)             # [B]

    loss = -w * ent_norm.mean()
    return loss, float(ent_norm.mean().detach().item()), \
           float(distinct.mean().detach().item()), \
           float(ppl_seq.mean().detach().item())


def _with_part_suffix(path: Path, part: str) -> Path:
    if part == "full":
        return path
    if part in path.stem:
        return path
    return path.with_name(f"{path.stem}_{part}{path.suffix}")


def resolve_part_output_paths(args):
    if args.part == "full":
        return

    stats_path = Path(args.stats_path)
    save_path = Path(args.save)

    if stats_path == Path("checkpoints/motion_stats_vqvae.npz"):
        args.stats_path = str(stats_path.with_name(f"stats_{args.part}.npz"))
    else:
        args.stats_path = str(_with_part_suffix(stats_path, args.part))

    if save_path == Path("checkpoints/stage1_vqvae.pt"):
        args.save = save_path.parent / f"ckpt_{args.part}" / save_path.name
    else:
        args.save = _with_part_suffix(save_path, args.part)

    if args.recon_dump_dir is None:
        args.recon_dump_dir = str(args.save.parent / "recon_debug")


def _collect_explicit_cli_flags(argv: List[str]) -> set[str]:
    out: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        out.add(token.split("=", 1)[0])
    return out


def _apply_profile_default(args, explicit_flags: set[str], attr: str, value, flag_name: Optional[str] = None):
    cli_flag = flag_name or f"--{attr}"
    if cli_flag in explicit_flags:
        return
    setattr(args, attr, value)


def resolve_training_config_profile(args, explicit_flags: set[str]) -> str:
    profile = args.config_profile
    if profile == "auto":
        profile = "legacy" if args.part == "full" else "lom_official"

    preset = CONFIG_PROFILE_DEFAULTS.get(profile, {})
    for attr, value in preset.items():
        _apply_profile_default(args, explicit_flags, attr, value)

    part_defaults = PROFILE_PART_MODEL_DEFAULTS.get(profile, LEGACY_PART_MODEL_DEFAULTS)
    if args.part in part_defaults:
        defaults = part_defaults[args.part]
        for attr, value in defaults.items():
            if attr in ("code_dim", "n_codes"):
                if getattr(args, attr) is None and f"--{attr}" not in explicit_flags:
                    setattr(args, attr, value)
            else:
                _apply_profile_default(args, explicit_flags, attr, value)
    else:
        if args.code_dim is None:
            args.code_dim = 256
        if args.n_codes is None:
            args.n_codes = 1024

    return profile


def resolve_reference_bvh(
    explicit_ref_bvh: Optional[str],
    fallback_ref_bvh: Optional[str],
    manifest_path: Optional[Path],
) -> Optional[Path]:
    for cand in (explicit_ref_bvh, fallback_ref_bvh):
        if cand is None:
            continue
        p = Path(cand)
        if p.exists():
            return p

    if manifest_path is None:
        return None

    base_dir = manifest_path.parent
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                bvh_rel = item.get("bvh", None) or item.get("motion", None) or item.get("motion_path", None)
                if not bvh_rel:
                    continue
                p = _resolve(base_dir, bvh_rel)
                if p.exists():
                    return p
    except Exception:
        return None

    return None


def print_part_joint_report(rank: int, motion_spec: MotionPartSpec, part_keywords: Dict[str, List[str]]):
    if rank != 0:
        return
    assigned = set()
    for part_name in ("upper", "hand", "lower"):
        joint_ids = select_part_joint_ids(motion_spec.skel, motion_spec.active_joint_map, part_name, part_keywords)
        joint_names = [motion_spec.skel.joints[jid].name for jid in joint_ids]
        assigned.update(joint_ids)
        print(f"[INFO] Part[{part_name}] joints ({len(joint_names)}): {joint_names}")

    unmatched = [
        motion_spec.skel.joints[jid].name
        for jid in motion_spec.active_joint_map.active_joint_ids
        if jid not in assigned
    ]
    if unmatched:
        print(f"[INFO] Part[unmatched] active joints ({len(unmatched)}): {unmatched}")

    print(
        f"[INFO] Training part={motion_spec.part}, model_dim={motion_spec.model_dim}, "
        f"full_dim={motion_spec.full_canonical_dim}, lower_root={motion_spec.lower_include_root}, "
        f"lower_contact={motion_spec.lower_include_foot_contact}, use_vq={motion_spec.uses_vq}"
    )
    if motion_spec.part_joint_names:
        print(f"[INFO] Selected {motion_spec.part} joints: {motion_spec.part_joint_names}")
    if motion_spec.foot_contact_joint_names:
        label = "Global" if motion_spec.part == "global" else "Lower"
        print(f"[INFO] {label} foot-contact joints: {motion_spec.foot_contact_joint_names}")
    if motion_spec.part == "global":
        print("[INFO] Global branch follows LoM/EMAGE semantics: lower-rotation condition -> predict root/contact.")


def save_reconstruction_debug_npz(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    mean: np.ndarray,
    std: np.ndarray,
    motion_spec: MotionPartSpec,
    out_path: Path,
    sample_is_normalized: bool,
):
    raw_model = model.module if hasattr(model, "module") else model
    sample = None
    for i in range(min(len(dataset), 16)):
        sample = dataset[i]
        if sample is not None:
            break
    if sample is None:
        return

    if not isinstance(sample, dict) or "full_motion" not in sample:
        return

    mean_t = torch.as_tensor(mean, device=device).view(1, 1, -1)
    std_t = torch.as_tensor(std, device=device).view(1, 1, -1)

    x = sample["motion"].unsqueeze(0).to(device)
    full_gt = sample["full_motion"].unsqueeze(0).to(device)
    x_den = x * std_t + mean_t if sample_is_normalized else x
    if motion_spec.part == "global":
        x_model_raw = prepare_global_condition_input(x_den, motion_spec)
        x_model = (x_model_raw - mean_t) / std_t
    else:
        x_model = x if sample_is_normalized else (x - mean_t) / std_t

    was_training = raw_model.training
    raw_model.eval()
    with torch.no_grad():
        x_hat, _, _, _, _ = raw_model(x_model, use_vq=motion_spec.uses_vq)
    if was_training:
        raw_model.train()

    xh_den = x_hat * std_t + mean_t
    merged = merge_parts_back_to_full(xh_den, motion_spec, gt_full=full_gt)

    extra_payload = {}
    if motion_spec.part == "global":
        extra_payload["gt_root_xyz"] = full_gt[0, :, :3].detach().cpu().numpy()
        extra_payload["recon_root_xyz"] = recover_global_root_translation(
            xh_den[0], motion_spec, gt_root=full_gt[0, :, :3]
        ).detach().cpu().numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        gt_part=x_den[0].detach().cpu().numpy(),
        recon_part=xh_den[0].detach().cpu().numpy(),
        gt_full=full_gt[0].detach().cpu().numpy(),
        recon_full_merged=merged[0].detach().cpu().numpy(),
        part=np.array([motion_spec.part], dtype=object),
        part_joint_names=np.array(motion_spec.part_joint_names, dtype=object),
        active_joint_names=np.array(motion_spec.active_joint_map.active_joint_names, dtype=object),
        **extra_payload,
    )

# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser()

    # IO / Data
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--val_manifest", type=Path, default=None)

    p.add_argument("--block_size", type=int, default=256, help="window length in frames")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--epochs", type=int, default=500)

    p.add_argument("--stats_path", type=str, default="checkpoints/motion_stats_vqvae.npz")
    p.add_argument("--approx_stats_n", type=int, default=500)
    p.add_argument("--std_floor", type=float, default=1e-4)

    p.add_argument("--drop_root_pos", action="store_true")
    p.add_argument("--overfit_n", type=int, default=0)
    p.add_argument("--fixed_crop", action="store_true")

    p.add_argument("--save", type=Path, default=Path("checkpoints/stage1_vqvae.pt"))
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--recon_dump_dir", type=str, default=None, help="optional directory to dump part reconstruction npz")

    # Model
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--code_dim", type=int, default=None)
    p.add_argument("--n_codes", type=int, default=None)
    p.add_argument("--n_downsample", type=int, default=3)
    p.add_argument("--global_ae_layers", type=int, default=4, help="LoM-like global AE depth (used when --part global)")

    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--ema_decay", type=float, default=0.99)
    p.add_argument("--ema_eps", type=float, default=1e-5)

    # Optim
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)

    # Loss weights
    p.add_argument("--recon_w", type=float, default=1.0)
    p.add_argument("--part_recon_norm_w", type=float, default=1.0,
                   help="weight for normalized reconstruction loss in part training")
    p.add_argument("--part_recon_denorm_w", type=float, default=0.25,
                   help="weight for denormalized reconstruction loss in part training")
    p.add_argument("--vel_w", type=float, default=1.0)
    p.add_argument("--acc_w", type=float, default=0.5)
    p.add_argument("--z_smooth_w", type=float, default=0.0)

    # VQ schedule
    p.add_argument("--vq_w", type=float, default=1.0)
    p.add_argument("--vq_freeze_epochs", type=int, default=0)
    p.add_argument("--vq_warmup_epochs", type=int, default=0)

    # AMP / seed
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=1234)

    # TextGrid sampling (optional)
    p.add_argument("--use_textgrid", action="store_true")
    p.add_argument("--speech_prob", type=float, default=0.8)
    p.add_argument("--merge_silence_s", type=float, default=0.2)
    p.add_argument("--pad_s", type=float, default=0.3)
    p.add_argument("--tier_name", type=str, default=None)

    # FK losses
    p.add_argument("--fk_foot_pos_w", type=float, default=0.0, help="FK foot position loss weight")
    p.add_argument("--fk_foot_lock_w", type=float, default=0.0, help="Foot lock loss weight during GT contact")
    p.add_argument("--fk_contact_vel_th", type=float, default=0.1, help="GT foot speed threshold for contact mask")
    p.add_argument("--fk_foot_keywords", type=str, default="RightFoot,LeftFoot,RightToeBase,LeftToeBase",
                   help="comma keywords to select foot joints (names)")
    p.add_argument("--fk_ref_bvh", type=str, default=None, help="optional reference BVH path to parse skeleton")

    p.add_argument("--fk_hand_pos_w", type=float, default=0.0, help="FK hand position loss weight")
    p.add_argument("--fk_hand_keywords", type=str, default="RightHand,LeftHand",
               help="comma keywords to select hand joints (names)")

    p.add_argument("--vq_usage_entropy_w", type=float, default=1e-3)
    p.add_argument("--vq_usage_temp", type=float, default=0.5)
    p.add_argument("--vq_revive_threshold", type=float, default=1.0)
    p.add_argument("--vq_seq_usage_w", type=float, default=1e-3)

    p.add_argument("--fk_hand_vel_w", type=float, default=0.0)
    p.add_argument("--fk_hand_acc_w", type=float, default=0.0)

    p.add_argument("--cache_in_mem", action="store_true")
    p.add_argument("--cache_preload", action="store_true")

    p.add_argument("--cache_pt", type=Path, default=None, help="if set, load motions from cache .pt")
    p.add_argument("--cache_use_speaking", action="store_true", help="use speaking mask to bias crops (optional)")
    p.add_argument("--cache_speaking_prob", type=float, default=0.8)
    p.add_argument("--ref_bvh", type=str, default=None, help="reference BVH for channel-order parsing / part split / merge")
    p.add_argument("--part", type=str, choices=["full", "upper", "hand", "lower", "global"], default="full")
    p.add_argument(
        "--config_profile",
        type=str,
        choices=["auto", "legacy", "lom_official", "lom_readme"],
        default="auto",
        help="auto keeps legacy defaults for full-body and applies LoM-like stage1 defaults for part training",
    )
    p.add_argument("--upper_keywords", type=str, default=",".join(DEFAULT_PART_KEYWORDS["upper"]))
    p.add_argument("--hand_keywords", type=str, default=",".join(DEFAULT_PART_KEYWORDS["hand"]))
    p.add_argument("--lower_keywords", type=str, default=",".join(DEFAULT_PART_KEYWORDS["lower"]))
    p.add_argument("--foot_contact_keywords", type=str, default=",".join(DEFAULT_FOOT_CONTACT_KEYWORDS))
    p.add_argument("--lower_include_root", action="store_true")
    p.add_argument("--lower_include_foot_contact", action="store_true")
    
    args = p.parse_args()
    if args.cache_pt is None and args.manifest is None:
        raise ValueError("Either --cache_pt or --manifest must be provided.")

    explicit_flags = _collect_explicit_cli_flags(sys.argv[1:])
    args.config_profile_resolved = resolve_training_config_profile(args, explicit_flags)

    resolve_part_output_paths(args)

    part_keywords = {
        "upper": _split_csv_keywords(args.upper_keywords, DEFAULT_PART_KEYWORDS["upper"]),
        "hand": _split_csv_keywords(args.hand_keywords, DEFAULT_PART_KEYWORDS["hand"]),
        "lower": _split_csv_keywords(args.lower_keywords, DEFAULT_PART_KEYWORDS["lower"]),
    }
    foot_contact_keywords = _split_csv_keywords(args.foot_contact_keywords, DEFAULT_FOOT_CONTACT_KEYWORDS)

    # ---------------- DDP 初始化 (关键修改) ----------------
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        is_distributed = True
        rank = dist.get_rank()
        if rank == 0:
            print(f"[INFO] DDP initialized: world size {dist.get_world_size()}")
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_distributed = False
        rank = 0
        print("[INFO] Running in Single-GPU mode.")

    # 设置种子 (多卡时偏移种子以避免数据完全一样)
    set_seed(args.seed + local_rank)
    if rank == 0 and args.part == "global":
        print("[INFO] part=global uses fixed [lower_rot, root_xyz, foot_contact4] features and ignores lower_include_* toggles.")

    # 准备 Scaler
    autocast_ctx = make_autocast(device)
    scaler = make_grad_scaler(device, enabled=args.amp)

    fk_enabled = (
        (args.fk_foot_pos_w > 0.0)
        or (args.fk_foot_lock_w > 0.0)
        or (args.fk_hand_pos_w > 0.0)
        or (args.fk_hand_vel_w > 0.0)
        or (args.fk_hand_acc_w > 0.0)
    )

    ref_manifest = args.manifest if args.manifest is not None else args.val_manifest
    ref_bvh = resolve_reference_bvh(args.ref_bvh, args.fk_ref_bvh, ref_manifest)
    need_motion_spec = (
        (args.manifest is not None)
        or (args.val_manifest is not None)
        or (args.part != "full")
        or args.lower_include_foot_contact
        or fk_enabled
    )
    motion_spec = None
    if need_motion_spec:
        if ref_bvh is None:
            raise RuntimeError(
                "Failed to resolve reference BVH. Please provide --ref_bvh (or --fk_ref_bvh), "
                "especially when using --cache_pt or part mode."
            )
        skel = BVHSkeleton.from_bvh(ref_bvh)
        motion_spec = build_motion_part_spec(
            skel,
            part=args.part,
            drop_root_pos=args.drop_root_pos,
            lower_include_root=args.lower_include_root,
            lower_include_foot_contact=args.lower_include_foot_contact,
            part_keywords=part_keywords,
            foot_contact_keywords=foot_contact_keywords,
        )
        print_part_joint_report(rank, motion_spec, part_keywords)

    if rank == 0:
        print(
            "[INFO] Config profile="
            f"{args.config_profile_resolved}, part={args.part}, block={args.block_size}, "
            f"batch={args.batch_size}, workers={args.num_workers}, lr={args.lr}, wd={args.weight_decay}, "
            f"hidden={args.hidden}, code_dim={args.code_dim}, n_codes={args.n_codes}, "
            f"downsample={args.n_downsample}, global_ae_layers={args.global_ae_layers}, lower_root={args.lower_include_root}, "
            f"lower_contact={args.lower_include_foot_contact}, "
            f"part_recon_norm_w={args.part_recon_norm_w}, part_recon_denorm_w={args.part_recon_denorm_w}, "
            f"fk_foot_pos_w={args.fk_foot_pos_w}, fk_foot_lock_w={args.fk_foot_lock_w}, "
            f"fk_hand_pos_w={args.fk_hand_pos_w}, fk_hand_vel_w={args.fk_hand_vel_w}, "
            f"use_vq={(motion_spec.uses_vq if motion_spec is not None else True)}, "
            f"vq_usage_entropy_w={args.vq_usage_entropy_w}, vq_seq_usage_w={args.vq_seq_usage_w}"
        )

    # Block size 检查
    token_stride = 2 ** int(args.n_downsample)
    if args.block_size % token_stride != 0:
        if rank == 0:
            print(f"[WARN] block_size={args.block_size} is not divisible by token_stride={token_stride}. "
                  f"Consider using a multiple to avoid boundary artifacts.")

    # ---------------- Dataset / Loader (DDP Sampler修改) ----------------
    train_ds = None
    if args.cache_pt is not None:
        train_ds = MotionWindowCachedDataset(
            cache_pt=args.cache_pt,
            block_size=args.block_size,
            is_train=True,
            drop_root_pos=args.drop_root_pos,
            fixed_crop=args.fixed_crop,
            overfit_n=args.overfit_n,
            std_floor=args.std_floor,
            normalize_in_dataset=False,
            use_speaking=args.cache_use_speaking,
            speech_prob=args.cache_speaking_prob,
            stats_path=args.stats_path,
            motion_spec=motion_spec,
        )
    else:
        train_ds = MotionWindowDataset(
            manifest_path=args.manifest,
            block_size=args.block_size,
            is_train=True,
            stats_path=args.stats_path,
            approx_stats_n=args.approx_stats_n,
            drop_root_pos=args.drop_root_pos,
            overfit_n=args.overfit_n,
            fixed_crop=args.fixed_crop,
            std_floor=args.std_floor,
            use_textgrid=args.use_textgrid,
            speech_prob=args.speech_prob,
            merge_silence_s=args.merge_silence_s,
            pad_s=args.pad_s,
            tier_name=args.tier_name,
            cache_in_mem=args.cache_in_mem,
            cache_preload=args.cache_preload,
            motion_spec=motion_spec,
        )

    # DDP Sampler
    train_sampler = None
    train_shuffle = True
    if is_distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        train_shuffle = False # Sampler handles shuffling

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size, # 这是单张卡的 batch size
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_motion_tensor if args.cache_pt else collate_motion,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # 获取归一化参数
    if args.cache_pt is not None:
        norm_mean_t = torch.as_tensor(train_ds.mean, device=device).view(1, 1, -1)
        norm_std_t  = torch.as_tensor(train_ds.std,  device=device).view(1, 1, -1)
    else:
        norm_mean_t = None
        norm_std_t = None

    # Validation Set
    val_loader = None
    val_ds = None
    if args.val_manifest is not None:
        val_ds = MotionWindowDataset(
            manifest_path=args.val_manifest,
            block_size=args.block_size,
            is_train=False,
            stats_path=args.stats_path,
            approx_stats_n=args.approx_stats_n,
            drop_root_pos=args.drop_root_pos,
            overfit_n=0,
            fixed_crop=True,
            std_floor=args.std_floor,
            use_textgrid=args.use_textgrid,
            speech_prob=0.0,
            merge_silence_s=args.merge_silence_s,
            pad_s=args.pad_s,
            tier_name=args.tier_name,
            motion_spec=motion_spec,
        )
        # Validation sampler for DDP
        val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed else None
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_motion,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )

    # ---------------- Model & Optim (DDP Wrapper) ----------------
    model_arch = "motion_global_ae" if (motion_spec is not None and motion_spec.part == "global") else "motion_vqvae"
    if model_arch == "motion_global_ae":
        model = MotionGlobalAE(
            motion_dim=train_ds.keep_dim,
            hidden=args.hidden,
            n_layers=args.global_ae_layers,
        ).to(device)
    else:
        model = MotionVQVAE(
            motion_dim=train_ds.keep_dim,
            hidden=args.hidden,
            code_dim=args.code_dim,
            n_codes=args.n_codes,
            beta=args.beta,
            ema_decay=args.ema_decay,
            ema_eps=args.ema_eps,
            n_downsample=args.n_downsample,
            vq_usage_entropy_w=args.vq_usage_entropy_w,
            vq_usage_temp=args.vq_usage_temp,
            vq_revive_threshold=args.vq_revive_threshold,
        ).to(device)

    # Wrap DDP
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optim = torch.optim.AdamW(
        [pp for pp in model.parameters() if pp.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ---------------- FK Setup ----------------
    fk = None
    foot_joint_ids = None
    hand_joint_ids = None

    if fk_enabled:
        if motion_spec is None:
            if rank == 0:
                print("[WARN] FK enabled but no valid reference BVH resolved. Disable FK losses.")
            fk = None
        else:
            try:
                fk_drop_root = args.drop_root_pos if args.part == "full" else False
                fk = BVHFkTorch(motion_spec.skel, drop_root_pos=fk_drop_root).to_device_tensors(device)

                # joints lookup (only needed log on rank 0)
                foot_joint_ids = None
                if (args.fk_foot_pos_w > 0.0) or (args.fk_foot_lock_w > 0.0):
                    fkeys = _split_csv_keywords(args.fk_foot_keywords, DEFAULT_FOOT_CONTACT_KEYWORDS)
                    fids = fk.skel.find_joints_by_keywords(fkeys)
                    fids = [i for i in fids if (len(fk.skel.joints[i].channels) > 0) and ("EndSite" not in fk.skel.joints[i].name) and (not fk.skel.joints[i].name.endswith("End"))]
                    if len(fids) > 0:
                        foot_joint_ids = fids
                        if rank == 0:
                            print("[INFO] FK foot joints:", ", ".join([f"{i}:{fk.skel.joints[i].name}" for i in foot_joint_ids]))
                    else:
                        if rank == 0:
                            print(f"[WARN] FK foot keywords matched nothing: {fkeys}. Foot FK disabled.")

                hand_joint_ids = None
                if (args.fk_hand_pos_w > 0.0) or (args.fk_hand_vel_w > 0.0) or (args.fk_hand_acc_w > 0.0):
                    hkeys = _split_csv_keywords(args.fk_hand_keywords, ["RightHand", "LeftHand"])
                    hids = fk.skel.find_joints_by_keywords(hkeys)
                    hids = [i for i in hids if (len(fk.skel.joints[i].channels) > 0) and ("EndSite" not in fk.skel.joints[i].name) and (not fk.skel.joints[i].name.endswith("End"))]
                    if len(hids) > 0:
                        hand_joint_ids = hids
                        if rank == 0:
                            print("[INFO] FK hand joints:", ", ".join([f"{i}:{fk.skel.joints[i].name}" for i in hand_joint_ids]))
                    else:
                        if rank == 0:
                            print(f"[WARN] FK hand keywords matched nothing: {hkeys}. Hand FK disabled.")
                
                if rank == 0:
                    print(f"[INFO] FK enabled. ref_bvh={ref_bvh}")

            except Exception as e:
                if rank == 0:
                    print(f"[WARN] FK initialization failed: {repr(e)}. Disable FK losses.")
                fk = None

    # ---------------- Resume ----------------
    start_epoch = 1
    best = float("inf")

    if args.resume is not None and Path(args.resume).exists():
        resume_path = Path(args.resume)
        # 加上 weights_only=False
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt

        # DDP 权重兼容性处理: 去掉或添加 module. 前缀
        new_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                k = k[7:]
            if is_distributed:
                k = "module." + k
            new_state[k] = v
        
        load_state_dict_compat(model, new_state, strict=True)

        if isinstance(ckpt, dict) and "optim" in ckpt:
            try:
                optim.load_state_dict(ckpt["optim"])
            except Exception:
                if rank == 0:
                    print("[WARN] Failed to load optimizer state. Will re-init optimizer.")

        if isinstance(ckpt, dict) and ("scaler" in ckpt) and ckpt["scaler"] is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                if rank == 0:
                    print("[WARN] Failed to load GradScaler state. Continue without it.")

        start_epoch = int(ckpt.get("epoch", 0)) + 1 if isinstance(ckpt, dict) else 1
        best = float(ckpt.get("best", float("inf"))) if isinstance(ckpt, dict) else float("inf")
        if rank == 0:
            print(f"[INFO] Resumed from {resume_path} (start_epoch={start_epoch}, best={best})")

    # ---------------- Train loop ----------------
    for ep in range(start_epoch, args.epochs + 1):
        if is_distributed:
            train_loader.sampler.set_epoch(ep)

        if args.vq_freeze_epochs and ep <= int(args.vq_freeze_epochs):
            vq_w_eff = 0.0
        else:
            if args.vq_warmup_epochs and args.vq_warmup_epochs > 0:
                base_ep = max(1, ep - int(args.vq_freeze_epochs))
                t = min(1.0, float(base_ep) / float(args.vq_warmup_epochs))
                vq_w_eff = float(args.vq_w) * t
            else:
                vq_w_eff = float(args.vq_w)

            tr = run_epoch(
                model=model,
                loader=train_loader,
                optim=optim,
                device=device,
                is_train=True,
                recon_w=args.recon_w,
                part_recon_norm_w=args.part_recon_norm_w,
                part_recon_denorm_w=args.part_recon_denorm_w,
                vel_w=args.vel_w,
                acc_w=args.acc_w,
                vq_w_eff=vq_w_eff,
            z_smooth_w=args.z_smooth_w,
            amp=args.amp,
            scaler=scaler,
            autocast_ctx=autocast_ctx,
            norm_mean_t=norm_mean_t,
            norm_std_t=norm_std_t,
            
            vq_seq_usage_w=args.vq_seq_usage_w,
            fk=fk,
            foot_joint_ids=foot_joint_ids,
            fk_foot_pos_w=args.fk_foot_pos_w,
            fk_foot_lock_w=args.fk_foot_lock_w,
            fk_contact_vel_th=args.fk_contact_vel_th,

            hand_joint_ids=hand_joint_ids,
            fk_hand_pos_w=args.fk_hand_pos_w,
            fk_hand_vel_w=args.fk_hand_vel_w,
            fk_hand_acc_w=args.fk_hand_acc_w,

            mean=train_ds.mean,
            std=train_ds.std,
            motion_spec=motion_spec,
        )
        tr = ddp_weighted_mean_dict(tr, device)

        if tr["n"] <= 0:
            if rank == 0:
                print("[WARN] No valid batches this epoch.")
            continue

        # Logging (Rank 0 only)
        if rank == 0:
            fkpos = tr.get("footpos", 0.0)
            fklk = tr.get("footlock", 0.0)
            handpos = tr.get("handpos", 0.0)
            handvel = tr.get("handvel", 0.0)
            if args.part == "global":
                log = (
                    f"[Epoch {ep}] "
                    f"Train loss={tr['loss']:.4f} "
                    f"(root_l1={tr['recon']:.4f}, trans_vel={tr['vel']:.4f}, smooth={tr['acc']:.4f}, "
                    f"contact={tr['global_contact']:.4f}, zsm={tr['zsm']:.4f}, "
                    f"fkpos={fkpos:.4f}, fklk={fklk:.4f}, hand_p={handpos:.4f}, hand_vel={handvel:.4f}) "
                    f"| use_vq=False"
                )
            else:
                log = (f"[Epoch {ep}] "
                       f"Train loss={tr['loss']:.4f} "
                       f"(recon={tr['recon']:.4f}, recon_n={tr.get('recon_norm', 0.0):.4f}, "
                       f"recon_d={tr.get('recon_denorm', 0.0):.4f}, vel={tr['vel']:.4f}, acc={tr['acc']:.4f}, "
                       f"vq={tr['vq']:.4f}, zsm={tr['zsm']:.4f}, ppl={tr['ppl']:.2f}, "
                       f"usage={tr['usage']:.4f}, usage_ent={tr['usage_ent']:.4f}, active_codes={tr['active_codes']:.1f}, "
                       f"seq_ppl={tr['seq_ppl']:.2f}, seq_ent={tr['seq_usage_ent']:.4f}, seq_dist={tr['seq_distinct']:.1f}, "
                       f"fkpos={fkpos:.4f}, fklk={fklk:.4f}, hand_p={handpos:.4f}, hand_vel={handvel:.4f}) "
                       f"| vq_w={vq_w_eff:.4f}")

        score = tr["loss"]
        if val_loader is not None:
            va = run_epoch(
                model=model,
                loader=val_loader,
                optim=optim,
                device=device,
                is_train=False,
                recon_w=args.recon_w,
                part_recon_norm_w=args.part_recon_norm_w,
                part_recon_denorm_w=args.part_recon_denorm_w,
                vel_w=args.vel_w,
                acc_w=args.acc_w,
                vq_w_eff=vq_w_eff,
                z_smooth_w=args.z_smooth_w,
                amp=False,
                scaler=scaler,
                autocast_ctx=autocast_ctx,

                fk=fk,
                foot_joint_ids=foot_joint_ids,
                fk_foot_pos_w=args.fk_foot_pos_w,
                fk_foot_lock_w=args.fk_foot_lock_w,
                fk_contact_vel_th=args.fk_contact_vel_th,

                hand_joint_ids=hand_joint_ids,
                fk_hand_pos_w=args.fk_hand_pos_w,
                fk_hand_vel_w=args.fk_hand_vel_w,
                fk_hand_acc_w=args.fk_hand_acc_w,

                mean=(val_ds.mean if val_ds is not None else train_ds.mean),
                std=(val_ds.std if val_ds is not None else train_ds.std),
                motion_spec=motion_spec,
            )
            if va:
                va = ddp_weighted_mean_dict(va, device)

            if va.get("n", 0) > 0:
                score = va["loss"]
                if rank == 0:
                    if args.part == "global":
                        log = (
                            f"[Epoch {ep}] "
                            f"Train loss={tr['loss']:.4f} "
                            f"(root_l1={tr['recon']:.4f}, trans_vel={tr['vel']:.4f}, smooth={tr['acc']:.4f}, "
                            f"contact={tr['global_contact']:.4f}, footpos={tr['footpos']:.4f}, "
                            f"footlock={tr['footlock']:.4f}, handpos={tr['handpos']:.4f}, "
                            f"handvel={tr['handvel']:.4f}, handacc={tr['handacc']:.4f}) "
                            f"| use_vq=False"
                        )
                    else:
                        log = (
                            f"[Epoch {ep}] "
                            f"Train loss={tr['loss']:.4f} "
                            f"(recon={tr['recon']:.4f}, recon_n={tr.get('recon_norm', 0.0):.4f}, "
                            f"recon_d={tr.get('recon_denorm', 0.0):.4f}, vel={tr['vel']:.4f}, acc={tr['acc']:.4f}, "
                            f"vq={tr['vq']:.4f}, zsm={tr['zsm']:.4f}, ppl={tr['ppl']:.2f}, "
                            f"usage={tr['usage']:.4f}, usage_ent={tr['usage_ent']:.4f}, active_codes={tr['active_codes']:.1f}, "
                            f"footpos={tr['footpos']:.4f}, footlock={tr['footlock']:.4f}, "
                            f"handpos={tr['handpos']:.4f}, handvel={tr['handvel']:.4f}, handacc={tr['handacc']:.4f}) "
                            f"| vq_w={vq_w_eff:.4f}"
                        )
            else:
                if rank == 0:
                    log += " || Val skipped(no valid batches)"

        improved = (score < best)
        if improved:
            best = score

        # 保存和打印只在 Rank 0 进行
        if rank == 0:
            # 获取原始 model (去掉 DDP wrapper 的 module.)
            raw_model = model.module if is_distributed else model
            
            ckpt_payload = {
                "epoch": ep,
                "best": best,
                "model": raw_model.state_dict(),
                "optim": optim.state_dict(),
                "scaler": scaler.state_dict() if args.amp else None,

                "mean": train_ds.mean,
                "std": train_ds.std,
                "keep_dim": train_ds.keep_dim,
                "drop_root_pos": args.drop_root_pos,
                "block_size": args.block_size,
                "part": args.part,
                "config_profile": args.config_profile,
                "config_profile_resolved": args.config_profile_resolved,
                "ref_bvh": str(ref_bvh) if ref_bvh is not None else None,
                "lower_include_root": args.lower_include_root,
                "lower_include_foot_contact": args.lower_include_foot_contact,
                "use_vq": (motion_spec.uses_vq if motion_spec is not None else True),
                "part_keywords": part_keywords,
                "foot_contact_keywords": foot_contact_keywords,
                "model_arch": model_arch,
                "global_ae_layers": args.global_ae_layers,

                "n_codes": args.n_codes,
                "code_dim": args.code_dim,
                "hidden": args.hidden,
                "beta": args.beta,
                "ema_decay": args.ema_decay,
                "ema_eps": args.ema_eps,
                "n_downsample": args.n_downsample,
                "token_stride": token_stride,
                # "target_fps": TARGET_FPS, # TARGET_FPS 是全局变量，如果 args 里没存就没法存

                "use_textgrid": args.use_textgrid,
                "speech_prob": args.speech_prob,
                "merge_silence_s": args.merge_silence_s,
                "pad_s": args.pad_s,
                "tier_name": args.tier_name,

                "recon_w": args.recon_w,
                "part_recon_norm_w": args.part_recon_norm_w,
                "part_recon_denorm_w": args.part_recon_denorm_w,
                "vel_w": args.vel_w,
                "acc_w": args.acc_w,
                "z_smooth_w": args.z_smooth_w,

                "fk_foot_pos_w": args.fk_foot_pos_w,
                "fk_foot_lock_w": args.fk_foot_lock_w,
                "fk_contact_vel_th": args.fk_contact_vel_th,
                "fk_foot_keywords": args.fk_foot_keywords,
                "fk_ref_bvh": args.fk_ref_bvh,

                "fk_hand_pos_w": args.fk_hand_pos_w,
                "vq_usage_entropy_w": args.vq_usage_entropy_w,
                "vq_usage_temp": args.vq_usage_temp,
                "vq_revive_threshold": args.vq_revive_threshold,
                "fk_hand_vel_w": args.fk_hand_vel_w,
                "fk_hand_acc_w": args.fk_hand_acc_w,

                "args": vars(args),
            }
            save_ckpt(args.save, ckpt_payload)
            if improved:
                best_path = args.save.with_name(args.save.stem + "_best.pt")
                save_ckpt(best_path, ckpt_payload)
                if (args.part != "full") and (args.recon_dump_dir is not None) and (motion_spec is not None):
                    debug_ds = val_ds if val_ds is not None else train_ds
                    dump_path = Path(args.recon_dump_dir) / f"recon_{args.part}_best_ep{ep}.npz"
                    save_reconstruction_debug_npz(
                        model=model,
                        dataset=debug_ds,
                        device=device,
                        mean=debug_ds.mean,
                        std=debug_ds.std,
                        motion_spec=motion_spec,
                        out_path=dump_path,
                        sample_is_normalized=isinstance(debug_ds, MotionWindowDataset),
                    )
                log += " [Saved(best)]"
            
            print(log)

    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
