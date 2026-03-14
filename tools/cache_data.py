import argparse
import json
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stageA.train_stage1_vqvae import (
    TARGET_FPS,
    DEFAULT_FOOT_CONTACT_KEYWORDS,
    DEFAULT_PART_KEYWORDS,
    BVHSkeleton,
    MotionPartSpec,
    build_motion_part_spec,
    convert_bvh_to_6d_with_channel_order,
    load_bvh_channels,
    print_part_joint_report,
    resample_motion_linear,
    resolve_reference_bvh,
    split_motion_into_parts,
    unwrap_bvh_angles_degrees,
    _resolve,
    _split_csv_keywords,
)


SIL_SET = {"", "sil", "sp", "silence"}
DEFAULT_OUTPUT = Path("output/train_stage1_vqvae.pt")
DEFAULT_STATS = Path("checkpoints/motion_stats_vqvae.npz")

_WORKER_BASE_DIR: Optional[Path] = None
_WORKER_FPS: int = TARGET_FPS
_WORKER_MOTION_SPEC: Optional[MotionPartSpec] = None
_WORKER_BUILD_SPEAKING: bool = True


def _with_part_suffix(path: Path, part: str) -> Path:
    if part == "full":
        return path
    return path.with_name(f"{path.stem}_{part}{path.suffix}")


def _resolve_cache_output_paths(args):
    if args.part == "full":
        return

    if args.stats_out == DEFAULT_STATS:
        args.stats_out = DEFAULT_STATS.with_name(f"stats_{args.part}.npz")
    else:
        args.stats_out = _with_part_suffix(args.stats_out, args.part)

    if args.output == DEFAULT_OUTPUT:
        args.output = DEFAULT_OUTPUT.with_name(f"{DEFAULT_OUTPUT.stem}_{args.part}{DEFAULT_OUTPUT.suffix}")
    else:
        args.output = _with_part_suffix(args.output, args.part)


def _sec_to_start_frame(s: float, fps: int) -> int:
    return int(np.floor(s * fps + 1e-6))


def _sec_to_end_frame(e: float, fps: int) -> int:
    return int(np.ceil(e * fps - 1e-6))


def build_speaking_mask_from_words(word_timestamps, T: int, fps: int):
    speaking = np.zeros((T,), dtype=np.uint8)
    for it in (word_timestamps or []):
        lab = (it.get("word") or "").strip().lower()
        if lab in SIL_SET:
            continue
        s = float(it.get("start", 0.0))
        e = float(it.get("end", 0.0))
        if e <= s:
            continue
        a = max(0, _sec_to_start_frame(s, fps))
        b = min(T, _sec_to_end_frame(e, fps))
        if b > a:
            speaking[a:b] = 1
    return speaking


def build_speaking_mask(align_path: Path, T: int, fps: int):
    if (align_path is None) or (not align_path.exists()):
        return np.zeros((T,), dtype=np.uint8)

    try:
        with align_path.open("r", encoding="utf-8") as f:
            align = json.load(f)
    except Exception:
        return np.zeros((T,), dtype=np.uint8)

    words = align.get("words", []) or []
    phones = align.get("phones", []) or []
    speaking = np.zeros((T,), dtype=np.uint8)

    if len(phones) > 0:
        for it in phones:
            lab = (it.get("lab") or "").strip().lower()
            if lab in SIL_SET:
                continue
            s = float(it.get("s", 0.0))
            e = float(it.get("e", 0.0))
            if e <= s:
                continue
            a = max(0, _sec_to_start_frame(s, fps))
            b = min(T, _sec_to_end_frame(e, fps))
            if b > a:
                speaking[a:b] = 1
    else:
        for it in words:
            lab = (it.get("lab") or "").strip().lower()
            if lab in SIL_SET:
                continue
            s = float(it.get("s", 0.0))
            e = float(it.get("e", 0.0))
            if e <= s:
                continue
            a = max(0, _sec_to_start_frame(s, fps))
            b = min(T, _sec_to_end_frame(e, fps))
            if b > a:
                speaking[a:b] = 1

    return speaking


def init_worker(
    base_dir_str: str,
    fps: int,
    ref_bvh_str: str,
    part: str,
    lower_include_root: bool,
    lower_include_foot_contact: bool,
    part_keywords_json: str,
    foot_contact_keywords_json: str,
    build_speaking: bool,
):
    global _WORKER_BASE_DIR, _WORKER_FPS, _WORKER_MOTION_SPEC, _WORKER_BUILD_SPEAKING
    _WORKER_BASE_DIR = Path(base_dir_str)
    _WORKER_FPS = int(fps)
    _WORKER_BUILD_SPEAKING = bool(build_speaking)

    part_keywords = json.loads(part_keywords_json)
    foot_contact_keywords = json.loads(foot_contact_keywords_json)
    skel = BVHSkeleton.from_bvh(Path(ref_bvh_str))
    _WORKER_MOTION_SPEC = build_motion_part_spec(
        skel,
        part=part,
        drop_root_pos=False,
        lower_include_root=bool(lower_include_root),
        lower_include_foot_contact=bool(lower_include_foot_contact),
        part_keywords=part_keywords,
        foot_contact_keywords=foot_contact_keywords,
    )


def process_single_item(item: dict):
    assert _WORKER_BASE_DIR is not None
    assert _WORKER_MOTION_SPEC is not None

    try:
        bvh_rel = item.get("bvh", None) or item.get("motion", None) or item.get("motion_path", None)
        if not bvh_rel:
            return None

        bvh_path = _resolve(_WORKER_BASE_DIR, bvh_rel)
        if not bvh_path.exists():
            return None

        raw_motion, frame_time = load_bvh_channels(bvh_path)
        if raw_motion is None or frame_time is None or frame_time <= 0 or raw_motion.ndim != 2:
            return None

        motion = unwrap_bvh_angles_degrees(raw_motion, pos_dims=3)
        motion = resample_motion_linear(motion, frame_time, _WORKER_FPS)

        full_motion = convert_bvh_to_6d_with_channel_order(
            motion,
            _WORKER_MOTION_SPEC.skel,
            _WORKER_MOTION_SPEC.active_joint_map,
        ).astype(np.float32)
        y = split_motion_into_parts(full_motion, _WORKER_MOTION_SPEC, fps=_WORKER_FPS).astype(np.float32)

        if not np.isfinite(full_motion).all() or not np.isfinite(y).all():
            return None

        T_m = int(y.shape[0])
        speaking = None
        if _WORKER_BUILD_SPEAKING:
            align_path = bvh_path.with_suffix(".align.json")
            if align_path.exists():
                speaking = build_speaking_mask(align_path, T_m, _WORKER_FPS)
            else:
                speaking = build_speaking_mask_from_words(item.get("word_timestamps"), T_m, _WORKER_FPS)

        sum_vec = y.sum(axis=0, dtype=np.float64)
        sumsq_vec = (y.astype(np.float64) ** 2).sum(axis=0)

        sample = {
            "y": y,
            "speaking": speaking,
            "id": item.get("id", str(bvh_path.stem)),
            "text": item.get("text", ""),
            "bvh": str(bvh_path),
        }
        if _WORKER_MOTION_SPEC.part != "full":
            sample["full_y"] = full_motion

        return sample, sum_vec, sumsq_vec, T_m
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output .pt file")
    parser.add_argument("--stats_out", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--fps", type=int, default=TARGET_FPS, help="Resample target FPS")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--ref_bvh", type=str, default=None, help="reference BVH for channel-order parsing / part split")
    parser.add_argument("--part", type=str, choices=["full", "upper", "hand", "lower", "global"], default="full")
    parser.add_argument("--upper_keywords", type=str, default=",".join(DEFAULT_PART_KEYWORDS["upper"]))
    parser.add_argument("--hand_keywords", type=str, default=",".join(DEFAULT_PART_KEYWORDS["hand"]))
    parser.add_argument("--lower_keywords", type=str, default=",".join(DEFAULT_PART_KEYWORDS["lower"]))
    parser.add_argument("--foot_contact_keywords", type=str, default=",".join(DEFAULT_FOOT_CONTACT_KEYWORDS))
    parser.add_argument("--lower_include_root", action="store_true")
    parser.add_argument("--lower_include_foot_contact", action="store_true")
    parser.add_argument("--skip_speaking", action="store_true", help="do not build/store speaking masks in cache")
    args = parser.parse_args()

    _resolve_cache_output_paths(args)

    part_keywords: Dict[str, List[str]] = {
        "upper": _split_csv_keywords(args.upper_keywords, DEFAULT_PART_KEYWORDS["upper"]),
        "hand": _split_csv_keywords(args.hand_keywords, DEFAULT_PART_KEYWORDS["hand"]),
        "lower": _split_csv_keywords(args.lower_keywords, DEFAULT_PART_KEYWORDS["lower"]),
    }
    foot_contact_keywords = _split_csv_keywords(args.foot_contact_keywords, DEFAULT_FOOT_CONTACT_KEYWORDS)

    ref_bvh = resolve_reference_bvh(args.ref_bvh, None, args.manifest)
    if ref_bvh is None:
        raise RuntimeError("Failed to resolve reference BVH. Please provide --ref_bvh or ensure manifest contains valid BVH paths.")

    motion_spec = build_motion_part_spec(
        BVHSkeleton.from_bvh(ref_bvh),
        part=args.part,
        drop_root_pos=False,
        lower_include_root=args.lower_include_root,
        lower_include_foot_contact=args.lower_include_foot_contact,
        part_keywords=part_keywords,
        foot_contact_keywords=foot_contact_keywords,
    )
    print_part_joint_report(rank=0, motion_spec=motion_spec, part_keywords=part_keywords)
    if args.part == "global":
        print("[INFO] part=global cache stores [lower_rot, root_xyz, foot_contact4] targets for the LoM-style global branch.")

    num_workers = min(args.workers, cpu_count())
    print(f"[INFO] Caching data at {args.fps} FPS using {num_workers} workers. part={args.part}")

    data_list = []
    with args.manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data_list.append(json.loads(line))

    cached_samples = []
    total_sum = None
    total_sumsq = None
    total_frames = 0
    out_dim = None

    with Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(
            str(args.manifest.parent),
            int(args.fps),
            str(ref_bvh),
            args.part,
            bool(args.lower_include_root),
            bool(args.lower_include_foot_contact),
            json.dumps(part_keywords),
            json.dumps(foot_contact_keywords),
            not args.skip_speaking,
        ),
    ) as pool:
        for res in tqdm(pool.imap_unordered(process_single_item, data_list), total=len(data_list)):
            if res is None:
                continue

            sample, s_vec, sq_vec, cnt = res
            curr_dim = int(sample["y"].shape[1])

            if out_dim is None:
                out_dim = curr_dim
                total_sum = np.zeros((out_dim,), dtype=np.float64)
                total_sumsq = np.zeros((out_dim,), dtype=np.float64)
            elif curr_dim != out_dim:
                continue

            cached_samples.append(sample)
            total_sum += s_vec
            total_sumsq += sq_vec
            total_frames += cnt

    print(f"[INFO] Processed {len(cached_samples)} valid samples.")
    if total_frames == 0 or out_dim is None:
        raise RuntimeError("No valid data found!")

    mean = (total_sum / total_frames).astype(np.float32)
    var = (total_sumsq / total_frames) - (mean.astype(np.float64) ** 2)
    std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)

    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.stats_out, mean=mean, std=std, keep_dim=np.array([out_dim], dtype=np.int32))
    print(f"[INFO] Saved stats to {args.stats_out} (dim={out_dim})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "samples": cached_samples,
            "mean": mean,
            "std": std,
            "fps": int(args.fps),
            "out_dim": int(out_dim),
            "part": args.part,
            "representation": "full_canonical" if args.part == "full" else "part_with_full_gt",
            "full_canonical_dim": int(motion_spec.full_canonical_dim),
            "ref_bvh": str(ref_bvh),
            "use_vq": bool(motion_spec.uses_vq),
            "has_speaking": not args.skip_speaking,
            "lower_include_root": bool(args.lower_include_root),
            "lower_include_foot_contact": bool(args.lower_include_foot_contact),
            "part_keywords": part_keywords,
            "foot_contact_keywords": foot_contact_keywords,
        },
        args.output,
    )
    print(f"[INFO] Saved cache to {args.output}")


if __name__ == "__main__":
    main()
