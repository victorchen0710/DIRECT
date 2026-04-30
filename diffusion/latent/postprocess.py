from __future__ import annotations

import numpy as np

from diffusion.latent.parts import resolve_part_feature_indices


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if int(window) <= 1:
        return x.copy()
    win = int(window)
    kernel = np.ones(win, dtype=np.float32) / float(win)
    pad_left = (win - 1) // 2
    pad_right = win // 2
    xpad = np.pad(x, ((pad_left, pad_right), (0, 0)), mode="edge")
    out = np.empty_like(x)
    for feat_idx in range(x.shape[1]):
        conv = np.convolve(xpad[:, feat_idx], kernel, mode="valid")
        out[:, feat_idx] = conv[: x.shape[0]]
    return out


def apply_temporal_smoothing(
    motion: np.ndarray,
    *,
    motion_contract,
    part_layout: str = "root_upper_hand_lower_slots_v3",
    root_window: int = 0,
    upper_hand_rot_window: int = 0,
    all_rot_window: int = 0,
) -> np.ndarray:
    out = np.asarray(motion, dtype=np.float32).copy()

    if int(root_window) > 1:
        root_idx = np.array(list(motion_contract.feature_layout.root_pos_indices), dtype=np.int64)
        out[:, root_idx] = _moving_average(out[:, root_idx], int(root_window))

    if int(upper_hand_rot_window) > 1:
        part_idx = resolve_part_feature_indices(motion_contract, part_layout=part_layout)
        upper_idx = np.array(part_idx.get("upper", ()), dtype=np.int64)
        hand_idx = np.array(part_idx.get("hand", ()), dtype=np.int64)
        upper_hand_rot = np.array(
            sorted(
                {
                    int(idx)
                    for idx in np.concatenate([upper_idx, hand_idx], axis=0)
                    if int(idx) >= int(motion_contract.rot6d_start)
                }
            ),
            dtype=np.int64,
        )
        if upper_hand_rot.size > 0:
            out[:, upper_hand_rot] = _moving_average(out[:, upper_hand_rot], int(upper_hand_rot_window))

    if int(all_rot_window) > 1:
        rot_idx = np.arange(int(motion_contract.rot6d_start), int(motion_contract.motion_dim), dtype=np.int64)
        out[:, rot_idx] = _moving_average(out[:, rot_idx], int(all_rot_window))

    return out


def blend_motion_parts(
    base_motion: np.ndarray,
    source_motion: np.ndarray,
    *,
    motion_contract,
    part_layout: str = "root_upper_hand_lower_slots_v3",
    source_parts: tuple[str, ...] = ("upper", "hand"),
) -> np.ndarray:
    base = np.asarray(base_motion, dtype=np.float32)
    source = np.asarray(source_motion, dtype=np.float32)
    if base.shape != source.shape:
        raise ValueError(f"Motion shape mismatch for blending: {base.shape} vs {source.shape}")
    if base.ndim != 2:
        raise ValueError(f"Expected [T, D] motion arrays, got {base.shape}")
    if len(source_parts) == 0:
        return base.copy()

    part_idx = resolve_part_feature_indices(motion_contract, part_layout=part_layout)
    if not part_idx:
        raise RuntimeError(f"part_layout does not support part blending: {part_layout}")

    feature_indices = []
    for part_name in source_parts:
        if part_name not in part_idx:
            raise KeyError(f"Unknown blend part '{part_name}' for layout {part_layout}")
        feature_indices.extend(int(v) for v in part_idx[part_name])
    feature_indices = np.array(sorted(set(feature_indices)), dtype=np.int64)
    if feature_indices.size == 0:
        return base.copy()

    out = base.copy()
    out[:, feature_indices] = source[:, feature_indices]
    return out


def blend_motion_features(
    base_motion: np.ndarray,
    source_motion: np.ndarray,
    *,
    feature_indices: np.ndarray,
) -> np.ndarray:
    base = np.asarray(base_motion, dtype=np.float32)
    source = np.asarray(source_motion, dtype=np.float32)
    if base.shape != source.shape:
        raise ValueError(f"Motion shape mismatch for blending: {base.shape} vs {source.shape}")
    indices = np.asarray(feature_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return base.copy()
    out = base.copy()
    out[:, indices] = source[:, indices]
    return out
