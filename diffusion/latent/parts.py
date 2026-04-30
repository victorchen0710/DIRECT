from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .contracts import MotionContract


DEFAULT_PART_KEYWORDS: Dict[str, List[str]] = {
    # Match the existing stageA semantics: upper keeps torso/proximal arms, while
    # forearm+hand stay together so distal articulation can co-vary.
    "upper": ["spine", "spine1", "spine2", "spine3", "neck", "head", "shoulder", "arm", "clavicle"],
    "hand": ["forearm", "hand", "thumb", "index", "middle", "ring", "pinky", "finger"],
    "lower": ["hips", "upleg", "leg", "foot", "toe", "toebase"],
}
PART_ASSIGNMENT_PRIORITY: Sequence[str] = ("hand", "lower", "upper")


def _match_keywords(name: str, keywords: Iterable[str]) -> bool:
    low = str(name).lower()
    return any(keyword.lower() in low for keyword in keywords)


def resolve_part_joint_indices(
    joint_names: Sequence[str],
    *,
    keyword_map: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, List[int]]:
    keywords = dict(DEFAULT_PART_KEYWORDS if keyword_map is None else keyword_map)
    out: Dict[str, List[int]] = {name: [] for name in ("upper", "hand", "lower")}
    for joint_idx, joint_name in enumerate(joint_names):
        low = str(joint_name).lower()
        if low.endswith("_end") or "endsite" in low:
            continue
        assigned = None
        for part_name in PART_ASSIGNMENT_PRIORITY:
            if _match_keywords(joint_name, keywords.get(part_name, ())):
                assigned = part_name
                break
        out[assigned or "upper"].append(int(joint_idx))
    return out


def build_part_feature_indices(
    motion_contract: MotionContract,
    *,
    keyword_map: Optional[Mapping[str, Sequence[str]]] = None,
    include_root_slot: bool = False,
) -> Dict[str, List[int]]:
    layout = motion_contract.feature_layout
    meta = motion_contract.layout_meta
    joint_names = list(meta.get("joint_names") or [])
    if not joint_names:
        raise RuntimeError("motion_contract.layout_meta.joint_names is required for part-aware VAE")

    part_joint_ids = resolve_part_joint_indices(joint_names, keyword_map=keyword_map)
    rot6d_start = int(layout.rot6d_start)

    def rot_indices_for_joints(joint_ids: Sequence[int]) -> List[int]:
        out: List[int] = []
        for joint_idx in joint_ids:
            start = rot6d_start + int(joint_idx) * 6
            out.extend(range(start, start + 6))
        return out

    lower_extra = list(layout.root_pos_indices)
    yaw_idx = meta.get("yaw_index")
    if yaw_idx is not None:
        lower_extra.append(int(yaw_idx))
    lower_extra.extend(int(v) for v in meta.get("local_vel_xz_indices", ()))
    yaw_vel_idx = meta.get("yaw_vel_index")
    if yaw_vel_idx is not None:
        lower_extra.append(int(yaw_vel_idx))
    lower_extra.extend(int(v) for v in layout.contact_indices)

    groups = {
        "upper": rot_indices_for_joints(part_joint_ids["upper"]),
        "hand": rot_indices_for_joints(part_joint_ids["hand"]),
        "lower": (rot_indices_for_joints(part_joint_ids["lower"]) if include_root_slot else lower_extra + rot_indices_for_joints(part_joint_ids["lower"])),
    }
    if include_root_slot:
        groups["root"] = list(lower_extra)
    return {name: sorted(set(int(v) for v in values)) for name, values in groups.items()}


def resolve_part_feature_indices(motion_contract: MotionContract, *, part_layout: str) -> Optional[Dict[str, List[int]]]:
    layout = str(part_layout)
    if not layout.startswith("upper_hand_lower_") and layout != "root_upper_hand_lower_slots_v3":
        return None
    return build_part_feature_indices(
        motion_contract,
        include_root_slot=(layout == "root_upper_hand_lower_slots_v3"),
    )


def resolve_joint_rot_feature_indices(
    motion_contract: MotionContract,
    *,
    include_keywords: Sequence[str],
    exclude_keywords: Optional[Sequence[str]] = None,
) -> List[int]:
    joint_names = list(motion_contract.layout_meta.get("joint_names") or [])
    if not joint_names:
        raise RuntimeError("motion_contract.layout_meta.joint_names is required for joint-keyword selection")
    include = [str(v).lower() for v in include_keywords if str(v).strip()]
    exclude = [str(v).lower() for v in (exclude_keywords or ()) if str(v).strip()]
    if not include:
        return []
    rot6d_start = int(motion_contract.rot6d_start)
    out: List[int] = []
    for joint_idx, joint_name in enumerate(joint_names):
        low = str(joint_name).lower()
        if not any(keyword in low for keyword in include):
            continue
        if any(keyword in low for keyword in exclude):
            continue
        start = rot6d_start + int(joint_idx) * 6
        out.extend(range(start, start + 6))
    return sorted(set(int(v) for v in out))
