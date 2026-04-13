"""Conditional Kimodo cache: motion + per-frame W2V2 audio + global BERT text.

Extends `tools/kimodo_cache.py` with two conditioning features per segment:
  audio  [T, 768]   per-frame W2V2 (sliced from pre-computed .npz aligned to 30fps)
  text   [768]      pooled BERT CLS of the utterance text for the source BVH

Simpler than per-frame BERT alignment — sufficient for Phase 7 sanity gate.
Per-frame text can be added later by swapping the text extractor.

Output payload adds:
  segments[i]["audio"]  float32 [T, 768]
  segments[i]["text_emb"] float32 [768]
  segments[i]["start_frame"]  int   (frame index of block start in source file)
  "audio_dim": 768, "text_dim": 768
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
from tools.kimodo_cache import (
    TARGET_FPS,
    _find_foot_joint_indices,
    _finalize_stats,
    _running_stats,
    compute_kimodo_representation,
)


def _load_audio_feat(feature_path: Path) -> np.ndarray:
    """Load pre-computed W2V2 features at 30fps. Returns [T, 768] float32."""
    d = np.load(str(feature_path))
    feat = d["w2v2_30fps"].astype(np.float32)  # [T, 768], was float16
    return feat


def _encode_text_bert(texts: List[str], bert_model_dir: Path, device: str = "cpu") -> np.ndarray:
    """Encode each text string with BERT and return pooled CLS embedding.

    Returns [N, hidden_dim] float32 array.
    """
    from transformers import BertModel, BertTokenizer

    tok = BertTokenizer.from_pretrained(str(bert_model_dir))
    model = BertModel.from_pretrained(str(bert_model_dir)).to(device).eval()

    out = []
    with torch.no_grad():
        for text in texts:
            text = text.strip() if text else "[empty]"
            enc = tok(
                text, return_tensors="pt", truncation=True, max_length=256, padding=True
            ).to(device)
            h = model(**enc).last_hidden_state  # [1, L, D]
            # mean-pool over tokens (excluding padding) — more stable than CLS alone
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            out.append(pooled.squeeze(0).cpu().numpy().astype(np.float32))
    return np.stack(out, axis=0)


def build_cond_cache(
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
    print(f"[cond_cache] processing {len(lines)} manifest entries, block_size={block_size}")

    root_dir = Path(".").resolve()
    first_bvh = (root_dir / lines[0]["bvh"]).resolve()
    info = build_skeleton_info(first_bvh)
    foot_idx = _find_foot_joint_indices(info)
    print(f"[cond_cache] skeleton: J={info.n_joints}, N_active={info.n_active}")

    # Encode all utterance texts at once (much faster than per-segment)
    texts = [str(item.get("text", "")) for item in lines]
    print(f"[cond_cache] encoding {len(texts)} utterances with BERT on {device}")
    text_embs = _encode_text_bert(texts, bert_model_dir, device=device)  # [N_items, D]
    text_dim = int(text_embs.shape[1])
    print(f"[cond_cache] text_dim={text_dim}")

    segments: List[Dict[str, Any]] = []
    accum: Dict[str, Dict[str, float]] = {}
    total_frames = 0
    kept = 0
    audio_dim: Optional[int] = None

    for i, item in enumerate(lines):
        bvh_path = (root_dir / item["bvh"]).resolve()
        feat_path = (root_dir / item["feature"]).resolve()
        try:
            full_motion = load_full_motion_from_bvh(bvh_path)
        except Exception as e:
            print(f"[cond_cache] WARN {item.get('id', '?')} bvh: {e}")
            continue
        try:
            audio_feat = _load_audio_feat(feat_path)  # [T_aud, 768]
        except Exception as e:
            print(f"[cond_cache] WARN {item.get('id', '?')} audio: {e}")
            continue

        if audio_dim is None:
            audio_dim = int(audio_feat.shape[1])

        T_motion = full_motion.shape[0]
        T_aud = audio_feat.shape[0]
        T_common = min(T_motion, T_aud)
        if T_common < block_size:
            continue

        n_blocks = T_common // block_size
        for b in range(n_blocks):
            start = b * block_size
            end = start + block_size
            clip = full_motion[start:end]
            rep = compute_kimodo_representation(clip, info, smooth_sigma=smooth_sigma)
            audio_clip = audio_feat[start:end].astype(np.float32)  # [T, 768]
            seg = {
                "segment_id": f"{item.get('id', f'item{i}')}_blk{b}",
                "src_bvh": str(item["bvh"]),
                "start_frame": int(start),
                **rep,
                "audio": audio_clip,
                "text_emb": text_embs[i].copy(),
                "text": str(item.get("text", "")),
            }
            segments.append(seg)
            for k in ("r_p", "j_p", "j_v", "f"):
                _running_stats(accum, rep[k], k)
            # also z-score audio
            _running_stats(accum, audio_clip, "audio")
            total_frames += block_size
            kept += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(lines):
            print(f"[cond_cache] {i+1}/{len(lines)} items → {kept} segments, {total_frames} frames")

    if not segments:
        raise RuntimeError("no segments produced")

    means, stds = _finalize_stats(accum)
    print(
        "[cond_cache] stats: "
        f"r_p std [{stds['r_p'].min():.3f}, {stds['r_p'].max():.3f}], "
        f"audio std [{stds['audio'].min():.3f}, {stds['audio'].max():.3f}]"
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
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(output))
    print(f"[cond_cache] saved {len(segments)} segments → {output}")


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
    build_cond_cache(
        args.manifest, args.output, args.bert_model_dir,
        block_size=args.block_size, limit=args.limit,
        smooth_sigma=args.smooth_sigma, device=args.device,
    )


if __name__ == "__main__":
    _cli()
