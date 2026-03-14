from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stageB.common.motion_codec import load_motion_codec_bundle
from stageB.common.bvh_io import load_full_motion_from_bvh
from stageB.common.prosody import extract_prosody_features
from stageB.common.text_bert import (
    align_dense_interval_embeddings_to_frames,
    build_dense_interval_bert_embeddings,
    load_bert_word_encoder,
)
from stageB.common.textgrid_utils import (
    align_words_to_frames,
    build_word_vocab,
    load_word_intervals_from_item,
    pool_frame_features_to_tokens,
    pool_frame_ids_to_tokens,
)
from stageA.train_stage1_vqvae import TARGET_FPS, _resolve


DESIGN_SUMMARY = {
    "lom": [
        "Keep the StageA/StageB split: frozen StageA motion tokenizer/decoder, StageB predicts StageA discrete codes.",
        "Support the current project's compositional tokenizer setup via multiple heads (upper / hand / lower).",
        "Keep audio and text as explicit conditions instead of asking StageA to absorb semantics.",
    ],
    "emage": [
        "Use content-rhythm dual conditioning: prosody is encoded separately from TextGrid-derived lexical/boundary features.",
        "Fuse text content and rhythm with an explicit gate so debug outputs can show content-vs-rhythm preference over time.",
        "Train with token CE as the main loss and motion-space auxiliary supervision via the frozen StageA decoder.",
    ],
    "simplifications": [
        "No HuBERT/T5 multimodal LM pretraining and no SMPL-X/FLAME/BEAT2 dependency. The pipeline stays on the current BEAT+Bvh stack.",
        "No face/body/hands/global full EMAGE stack. The minimum loop is wav + TextGrid -> token predictor -> StageA decoder -> BVH.",
        "The optional root/global branch is left as a later extension; the first version keeps StageA part decoders as the main path and uses inference-time root fallback.",
    ],
}


def _load_manifest(manifest_path: Path, limit: int = 0) -> List[dict]:
    items: List[dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    if int(limit) > 0:
        items = items[: int(limit)]
    return items


def _resolve_audio_path(item: dict, base_dir: Path) -> Path:
    audio_rel = item.get("wav") or item.get("audio") or item.get("audio_path")
    if not audio_rel:
        raise RuntimeError(f"Manifest item has no audio path: {item.get('id')}")
    audio_path = _resolve(base_dir, audio_rel)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    return audio_path


def _resolve_bvh_path(item: dict, base_dir: Path) -> Path:
    bvh_rel = item.get("bvh") or item.get("motion") or item.get("motion_path")
    if not bvh_rel:
        raise RuntimeError(f"Manifest item has no BVH path: {item.get('id')}")
    bvh_path = _resolve(base_dir, bvh_rel)
    if not bvh_path.exists():
        raise FileNotFoundError(f"BVH not found: {bvh_path}")
    return bvh_path


def _infer_speaker(item: dict, bvh_path: Path) -> str:
    if item.get("speaker"):
        return str(item["speaker"])
    stem_parts = bvh_path.stem.split("_")
    if len(stem_parts) >= 2:
        return stem_parts[1]
    return bvh_path.parent.name


def _window_starts(total_frames: int, block_size: int, hop: int, include_tail: bool) -> list[int]:
    if total_frames < block_size:
        return []
    starts = list(range(0, total_frames - block_size + 1, hop))
    if include_tail:
        tail_start = total_frames - block_size
        if tail_start >= 0 and (len(starts) == 0 or starts[-1] != tail_start):
            starts.append(tail_start)
    return sorted(set(starts))


def build_cache(args) -> dict:
    manifest_path = Path(args.manifest)
    base_dir = manifest_path.parent
    items = _load_manifest(manifest_path, limit=args.limit)
    if not items:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")

    device = torch.device(args.device)
    codec_bundle = load_motion_codec_bundle(
        device=device,
        ref_bvh=Path(args.ref_bvh) if args.ref_bvh else None,
        stage1_ckpt=Path(args.stage1_ckpt) if args.stage1_ckpt else None,
        upper_ckpt=Path(args.upper_ckpt) if args.upper_ckpt else None,
        hand_ckpt=Path(args.hand_ckpt) if args.hand_ckpt else None,
        lower_ckpt=Path(args.lower_ckpt) if args.lower_ckpt else None,
        global_ckpt=Path(args.global_ckpt) if args.global_ckpt else None,
    )

    block_size = int(args.block_size)
    if block_size % codec_bundle.token_stride != 0:
        raise ValueError(
            f"block_size={block_size} must be divisible by Stage1 token_stride={codec_bundle.token_stride}"
        )
    window_hop = int(args.window_hop) if int(args.window_hop) > 0 else block_size
    bert_enabled = not bool(args.disable_bert_text)
    bert_encoder = None
    bert_feature_dim = 0
    if bert_enabled:
        bert_device = args.bert_device if args.bert_device else args.device
        bert_encoder = load_bert_word_encoder(args.bert_model_dir, device=bert_device)
        bert_feature_dim = int(bert_encoder.hidden_size)

    vocab = build_word_vocab(items, base_dir=base_dir, prefer_tier=args.tier_name, min_freq=args.min_word_freq)
    samples: list[dict] = []
    prosody_feature_names = None
    text_scalar_feature_names = None
    skipped = {
        "missing_bvh": 0,
        "missing_audio": 0,
        "load_error": 0,
        "too_short": 0,
        "no_windows": 0,
    }

    for item in tqdm(items, desc="stage2_cache"):
        try:
            bvh_path = _resolve_bvh_path(item, base_dir)
        except (RuntimeError, FileNotFoundError):
            skipped["missing_bvh"] += 1
            continue

        try:
            audio_path = _resolve_audio_path(item, base_dir)
        except (RuntimeError, FileNotFoundError):
            skipped["missing_audio"] += 1
            continue

        try:
            full_motion = load_full_motion_from_bvh(
                bvh_path,
                skel=codec_bundle.skel,
                active_joint_map=codec_bundle.active_joint_map,
                fps=TARGET_FPS,
            )
        except Exception:
            skipped["load_error"] += 1
            continue

        T = int(full_motion.shape[0])
        if T < block_size:
            skipped["too_short"] += 1
            continue

        try:
            prosody_pack = extract_prosody_features(
                audio_path,
                fps=TARGET_FPS,
                target_frames=T,
                mel_bins=args.mel_bins,
                mfcc_dim=args.mfcc_dim,
            )
        except Exception:
            skipped["load_error"] += 1
            continue
        prosody_frame = prosody_pack["frame_features"].astype(np.float32)
        prosody_feature_names = list(prosody_pack["feature_names"])

        try:
            word_intervals, tg_path, tier_name = load_word_intervals_from_item(
                item,
                base_dir,
                prefer_tier=args.tier_name,
                allow_manifest_fallback=True,
            )
        except Exception:
            skipped["load_error"] += 1
            continue
        text_frame = align_words_to_frames(word_intervals, num_frames=T, fps=TARGET_FPS, vocab=vocab)
        text_scalar_feature_names = list(text_frame.feature_names)
        if bert_encoder is not None:
            try:
                dense_bert = build_dense_interval_bert_embeddings(
                    text_frame.words,
                    encoder=bert_encoder,
                    max_words_per_chunk=args.bert_max_words,
                    overlap_words=args.bert_overlap_words,
                )
                frame_text_bert = align_dense_interval_embeddings_to_frames(
                    text_frame.words,
                    dense_bert,
                    num_frames=T,
                    dtype=np.float32,
                )
            except Exception:
                skipped["load_error"] += 1
                continue
        else:
            frame_text_bert = None

        starts = _window_starts(T, block_size, window_hop, include_tail=args.include_tail)
        if not starts:
            skipped["no_windows"] += 1
            continue

        for start in starts:
            end = start + block_size
            window_full = full_motion[start:end].astype(np.float32, copy=False)
            part_motion = codec_bundle.full_motion_to_parts(window_full)
            codes = {
                name: codec_bundle.parts[name]
                .encode_motion(part_motion[name])[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
                for name in codec_bundle.primary_part_names
            }

            window_prosody = prosody_frame[start:end].astype(np.float32, copy=False)
            window_word_ids = text_frame.word_ids[start:end].astype(np.int64, copy=False)
            window_text_scalar = text_frame.scalar[start:end].astype(np.float32, copy=False)
            if frame_text_bert is not None:
                window_text_bert = frame_text_bert[start:end].astype(np.float32, copy=False)
            else:
                window_text_bert = np.zeros((block_size, bert_feature_dim), dtype=np.float32)

            token_prosody = pool_frame_features_to_tokens(window_prosody, codec_bundle.token_stride, mode="mean")
            token_text_scalar = pool_frame_features_to_tokens(window_text_scalar, codec_bundle.token_stride, mode="mean")
            token_word_ids = pool_frame_ids_to_tokens(
                window_word_ids,
                window_text_scalar[:, 0],
                codec_bundle.token_stride,
                blank_id=int(vocab["<blank>"]),
            )
            token_text_bert = pool_frame_features_to_tokens(window_text_bert, codec_bundle.token_stride, mode="mean").astype(np.float16)

            word_meta = []
            win_start_s = start / float(TARGET_FPS)
            win_end_s = end / float(TARGET_FPS)
            for info in text_frame.words:
                if float(info["end"]) <= win_start_s or float(info["start"]) >= win_end_s:
                    continue
                word_meta.append(
                    {
                        **info,
                        "rel_start": float(info["start"]) - win_start_s,
                        "rel_end": float(info["end"]) - win_start_s,
                    }
                )

            sample = {
                "full_motion": window_full,
                "part_motion": {name: motion.astype(np.float32) for name, motion in part_motion.items() if name in codec_bundle.primary_part_names},
                "codes": codes,
                "prosody_frame": window_prosody,
                "text_word_ids_frame": window_word_ids,
                "text_scalar_frame": window_text_scalar,
                "token_prosody": token_prosody,
                "token_word_ids": token_word_ids,
                "token_text_scalar": token_text_scalar,
                "token_text_bert": token_text_bert,
                "meta": {
                    "utt_id": item.get("id", bvh_path.stem),
                    "speaker": _infer_speaker(item, bvh_path),
                    "bvh_path": str(bvh_path),
                    "audio_path": str(audio_path),
                    "textgrid_path": str(tg_path) if tg_path is not None else None,
                    "tier_name": tier_name,
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "start_time": win_start_s,
                    "end_time": win_end_s,
                    "words": word_meta,
                    "text": item.get("text", ""),
                },
            }
            samples.append(sample)

    return {
        "samples": samples,
        "vocab": vocab,
        "index_to_word": {idx: word for word, idx in vocab.items()},
        "meta": {
            "manifest": str(manifest_path),
            "fps": TARGET_FPS,
            "window_hop": window_hop,
            "num_items": len(items),
            "num_windows": len(samples),
            "skipped": skipped,
            "bert_enabled": bert_enabled,
            "bert_model_dir": str(args.bert_model_dir) if bert_enabled else None,
        },
        "stage1": codec_bundle.describe(),
        "token_stride": codec_bundle.token_stride,
        "block_size": block_size,
        "prosody_feature_names": prosody_feature_names or [],
        "text_scalar_feature_names": text_scalar_feature_names or [],
        "bert_feature_dim": int(bert_feature_dim),
        "design_summary": DESIGN_SUMMARY,
    }


def main():
    parser = argparse.ArgumentParser(description="Build Stage2 semantic cache: wav + TextGrid + Stage1 codes.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ref_bvh", type=str, default=None)
    parser.add_argument("--stage1_ckpt", type=str, default=None, help="single-stream Stage1 checkpoint")
    parser.add_argument("--upper_ckpt", type=str, default="checkpoints/ckpt_upper/stage1_vqvae_best.pt")
    parser.add_argument("--hand_ckpt", type=str, default="checkpoints/ckpt_hand/stage1_vqvae_best.pt")
    parser.add_argument("--lower_ckpt", type=str, default="checkpoints/ckpt_lower/stage1_vqvae_best.pt")
    parser.add_argument("--global_ckpt", type=str, default=None)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--window_hop", type=int, default=128)
    parser.add_argument("--include_tail", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tier_name", type=str, default=None)
    parser.add_argument("--min_word_freq", type=int, default=1)
    parser.add_argument("--mel_bins", type=int, default=0)
    parser.add_argument("--mfcc_dim", type=int, default=0)
    parser.add_argument("--disable_bert_text", action="store_true")
    parser.add_argument("--bert_model_dir", type=str, default="models/bert")
    parser.add_argument("--bert_device", type=str, default=None)
    parser.add_argument("--bert_max_words", type=int, default=128)
    parser.add_argument("--bert_overlap_words", type=int, default=16)
    args = parser.parse_args()

    pack = build_cache(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "num_windows": pack["meta"]["num_windows"],
                "token_stride": pack["token_stride"],
                "block_size": pack["block_size"],
                "vocab_size": len(pack["vocab"]),
                "bert_feature_dim": int(pack.get("bert_feature_dim", 0)),
                "bert_enabled": bool(pack["meta"].get("bert_enabled", False)),
                "stage1_parts": list(pack["stage1"]["parts"].keys()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
