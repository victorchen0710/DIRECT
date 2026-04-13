"""Per-frame BERT conditional Kimodo cache.

Key difference vs kimodo_cache_cond.py: text is [T, 768] per-frame (aligned to
the currently-spoken word via TextGrid) instead of a single [768] pooled vector.

For each source BVH:
  1. Load TextGrid → list of (start, end, word) intervals
  2. align_words_to_frames: for each frame, which word is active
  3. build_dense_interval_bert_embeddings: each word's contextual BERT vector
  4. align_dense_interval_embeddings_to_frames: produce [T_full, 768]
  5. For each 96-frame segment, slice [start:start+96] of above → [96, 768]

Silence frames get zero vectors (by the alignment fn).

Output key: segments[i]["text_frame"] float32 [96, 768]  (replaces text_emb)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusion_learn.kimodo_lite.fk import build_skeleton_info
from stageB.common.bvh_io import load_full_motion_from_bvh
from stageB.common.text_bert import (
    align_dense_interval_embeddings_to_frames,
    build_dense_interval_bert_embeddings,
    load_bert_word_encoder,
)
from stageB.common.textgrid_utils import align_words_to_frames, load_word_intervals_from_item
from tools.kimodo_cache import (
    TARGET_FPS,
    _find_foot_joint_indices,
    _finalize_stats,
    _running_stats,
    compute_kimodo_representation,
)
from tools.kimodo_cache_cond import _load_audio_feat


def _build_per_frame_bert(
    item: dict,
    base_dir: Path,
    num_frames: int,
    encoder,
    fps: int = TARGET_FPS,
) -> np.ndarray:
    """Return [num_frames, 768] per-frame BERT; silence frames are zero."""
    intervals, tg_path, tier = load_word_intervals_from_item(item, base_dir)
    wf = align_words_to_frames(intervals, num_frames=num_frames, fps=fps)
    if not wf.words:
        return np.zeros((num_frames, encoder.hidden_size), dtype=np.float32)
    dense_bert = build_dense_interval_bert_embeddings(wf.words, encoder=encoder)
    frame_bert = align_dense_interval_embeddings_to_frames(
        wf.words, dense_bert, num_frames=num_frames
    )
    return frame_bert.astype(np.float32)


def build_perframe_cache(
    manifest: Path,
    output: Path,
    bert_model_dir: Path,
    block_size: int = 96,
    limit: Optional[int] = None,
    smooth_sigma: float = 15.0,
    device: str = "cpu",
) -> None:
    with manifest.open("r", encoding="utf-8") as f:
        lines = [json.loads(ln) for ln in f if ln.strip()]
    if limit is not None:
        lines = lines[:limit]
    print(f"[pf_cache] processing {len(lines)} manifest entries, block_size={block_size}")

    root_dir = Path(".").resolve()
    first_bvh = (root_dir / lines[0]["bvh"]).resolve()
    info = build_skeleton_info(first_bvh)
    foot_idx = _find_foot_joint_indices(info)
    print(f"[pf_cache] skeleton J={info.n_joints}, N_active={info.n_active}")

    print(f"[pf_cache] loading BERT encoder on {device}")
    encoder = load_bert_word_encoder(bert_model_dir, device=device)
    text_dim = int(encoder.hidden_size)

    segments: List[Dict[str, Any]] = []
    accum: Dict[str, Dict[str, float]] = {}
    total_frames = 0
    kept = 0
    audio_dim: Optional[int] = None
    coverage_sum = 0.0
    coverage_n = 0

    for i, item in enumerate(lines):
        if not item.get("bvh") or not item.get("feature"):
            continue  # skip manifest entries missing bvh or audio feature
        bvh_path = (root_dir / item["bvh"]).resolve()
        feat_path = (root_dir / item["feature"]).resolve()
        try:
            full_motion = load_full_motion_from_bvh(bvh_path)
        except Exception as e:
            print(f"[pf_cache] WARN {item.get('id', '?')} bvh: {e}")
            continue
        try:
            audio_feat = _load_audio_feat(feat_path)
        except Exception as e:
            print(f"[pf_cache] WARN {item.get('id', '?')} audio: {e}")
            continue
        if audio_dim is None:
            audio_dim = int(audio_feat.shape[1])

        T_motion = full_motion.shape[0]
        T_aud = audio_feat.shape[0]
        T_common = min(T_motion, T_aud)
        if T_common < block_size:
            continue

        # Per-frame BERT for the entire source file
        frame_bert_full = _build_per_frame_bert(item, root_dir, T_common, encoder, fps=TARGET_FPS)
        # coverage diagnostics
        nonzero_mask = (np.abs(frame_bert_full).sum(axis=1) > 0).astype(np.float32)
        coverage_sum += float(nonzero_mask.sum())
        coverage_n += int(T_common)

        n_blocks = T_common // block_size
        for b in range(n_blocks):
            start = b * block_size
            end = start + block_size
            clip = full_motion[start:end]
            rep = compute_kimodo_representation(clip, info, smooth_sigma=smooth_sigma)
            audio_clip = audio_feat[start:end].astype(np.float32)
            text_frame = frame_bert_full[start:end].astype(np.float32)  # [96, 768]
            seg = {
                "segment_id": f"{item.get('id', f'item{i}')}_blk{b}",
                "src_bvh": str(item["bvh"]),
                "start_frame": int(start),
                **rep,
                "audio": audio_clip,
                "text_frame": text_frame,
                "text": str(item.get("text", "")),
            }
            segments.append(seg)
            for k in ("r_p", "j_p", "j_v", "f"):
                _running_stats(accum, rep[k], k)
            _running_stats(accum, audio_clip, "audio")
            total_frames += block_size
            kept += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(lines):
            cov = coverage_sum / max(1, coverage_n)
            print(
                f"[pf_cache] {i+1}/{len(lines)} → {kept} segs, {total_frames} frames, "
                f"BERT coverage={cov*100:.1f}% frames have a word"
            )

    if not segments:
        raise RuntimeError("no segments produced")

    means, stds = _finalize_stats(accum)
    print(
        f"[pf_cache] stats r_p [{stds['r_p'].min():.3f}, {stds['r_p'].max():.3f}], "
        f"audio [{stds['audio'].min():.3f}, {stds['audio'].max():.3f}]"
    )

    payload = {
        "segments": segments,
        "mean": means,
        "std": stds,
        "skeleton": {
            "parents": info.parents.astype(np.int64),
            "offsets": info.offsets.astype(np.float32),
            "active_indices": info.active_indices.astype(np.int64),
            "joint_names": info.joint_names,
            "foot_joint_indices": foot_idx.astype(np.int64),
        },
        "fps": TARGET_FPS,
        "block_size": int(block_size),
        "smooth_sigma": float(smooth_sigma),
        "audio_dim": audio_dim,
        "text_dim": text_dim,
        "text_is_per_frame": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(output))
    print(f"[pf_cache] saved {len(segments)} segments → {output}")


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bert_model_dir", type=Path, default=Path("models/bert"))
    ap.add_argument("--block_size", type=int, default=96)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smooth_sigma", type=float, default=15.0)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    build_perframe_cache(
        args.manifest, args.output, args.bert_model_dir,
        block_size=args.block_size, limit=args.limit,
        smooth_sigma=args.smooth_sigma, device=args.device,
    )


if __name__ == "__main__":
    _cli()
