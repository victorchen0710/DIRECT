import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusion.latent.contracts import (
    AUDIO_SPEC_VERSION,
    CACHE_VERSION,
    TEXT_SPEC_VERSION,
    AudioFeatureSpec,
    MotionContract,
    TextTokenSpec,
    validate_cache_payload,
)
from diffusion.latent.runtime import build_global_text_embedding
legacy_cache = None
_WORD_CACHE: Dict[str, Any] = {}


def _get_legacy_cache():
    global legacy_cache
    if legacy_cache is None:
        from tools import diffusion_cache as legacy_cache_mod

        legacy_cache = legacy_cache_mod
    return legacy_cache


def _load_word_source(path: Path) -> Any:
    cache_key = str(path.resolve())
    cached = _WORD_CACHE.get(cache_key)
    if cached is not None:
        return cached
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    _WORD_CACHE[cache_key] = obj
    return obj


def _flatten_word_entries(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [dict(x) for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        if isinstance(obj.get("words"), list):
            return [dict(x) for x in obj["words"] if isinstance(x, dict)]
        if isinstance(obj.get("segments"), list):
            out: List[Dict[str, Any]] = []
            for seg in obj["segments"]:
                words = seg.get("words", [])
                if isinstance(words, list):
                    out.extend(dict(x) for x in words if isinstance(x, dict))
            return out
    return []


def _load_segment_words(item: Dict[str, Any], base_dir: Path) -> List[Dict[str, Any]]:
    word_entries: List[Dict[str, Any]] = []
    words_json = item.get("words_json")
    if words_json:
        path = base_dir / str(words_json)
        if path.exists():
            word_entries = _flatten_word_entries(_load_word_source(path))
    if not word_entries and isinstance(item.get("word_timestamps"), list):
        word_entries = [dict(x) for x in item["word_timestamps"] if isinstance(x, dict)]

    words_range = item.get("words_range")
    if isinstance(words_range, (list, tuple)) and len(words_range) == 2:
        try:
            start_idx = max(0, int(words_range[0]))
            end_idx = max(start_idx, int(words_range[1]))
            word_entries = word_entries[start_idx:end_idx]
        except Exception:
            pass

    start_s = 0.0 if item.get("start") is None else float(item["start"])
    end_s = float("inf") if item.get("end") is None else float(item["end"])

    clipped: List[Dict[str, Any]] = []
    for word in word_entries:
        token = str(word.get("word", word.get("text", ""))).strip()
        if not token:
            continue
        try:
            word_start = float(word.get("start"))
            word_end = float(word.get("end"))
        except Exception:
            continue
        if not np.isfinite(word_start) or not np.isfinite(word_end) or word_end <= word_start:
            continue
        if word_end <= start_s or word_start >= end_s:
            continue
        rel_start = max(word_start, start_s) - start_s
        rel_end = min(word_end, end_s) - start_s
        if rel_end <= rel_start + 1e-4:
            continue
        clipped.append(
            {
                "word": token,
                "start": float(rel_start),
                "end": float(rel_end),
            }
        )
    return clipped


def _build_word_frame_features(word_times: np.ndarray, fps: int, frames: int) -> np.ndarray:
    feats = np.zeros((frames, 5), dtype=np.float32)
    if word_times.size == 0:
        return feats
    tol = 0.5 / max(float(fps), 1e-6)
    frame_ts = np.arange(frames, dtype=np.float32) / float(fps)
    for idx, t_sec in enumerate(frame_ts):
        for start, end in word_times:
            if end <= start:
                continue
            if start <= t_sec < end:
                dur = max(end - start, 1e-4)
                feats[idx, 0] = 1.0
                feats[idx, 1] = np.clip((t_sec - start) / dur, 0.0, 1.0)
                feats[idx, 2] = dur
                feats[idx, 3] = 1.0 if abs(t_sec - start) <= tol else 0.0
                feats[idx, 4] = 1.0 if abs(t_sec - end) <= tol else 0.0
                break
    return feats


def _load_audio_feature_array(path: Path) -> Tuple[np.ndarray, str, float]:
    npz = np.load(str(path), allow_pickle=True)
    feature_key = None
    for key in ("w2v2_30fps", "feat", "features", "feats", "hidden", "arr_0"):
        if key in npz:
            feature_key = key
            break
    if feature_key is None:
        raise RuntimeError(f"Could not find audio feature array in {path}")
    feat = np.asarray(npz[feature_key], dtype=np.float32)
    if feat.ndim != 2:
        raise RuntimeError(f"Audio features must be [T,D], got {feat.shape} from {path}")

    fps = None
    for key in ("fps", "frame_rate", "feature_fps"):
        if key in npz:
            fps = float(np.asarray(npz[key]).reshape(-1)[0])
            break
    if fps is None and feature_key == "w2v2_30fps":
        fps = 30.0
    if fps is None:
        raise RuntimeError(f"Audio feature npz missing fps metadata: {path}")
    return feat.astype(np.float32, copy=False), str(feature_key), float(fps)


def _slice_audio_clip(
    item: Dict[str, Any],
    *,
    base_dir: Path,
    target_len: int,
) -> Tuple[np.ndarray, str, float, int]:
    legacy_cache = _get_legacy_cache()
    feature_rel = item.get("feature")
    if not feature_rel:
        raise RuntimeError(f"Missing feature path for segment {item.get('seg_id') or item.get('id')}")
    feature_path = base_dir / str(feature_rel)
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing audio feature npz: {feature_path}")
    feat, feature_key, feature_fps = _load_audio_feature_array(feature_path)
    if abs(float(feature_fps) - float(legacy_cache.TARGET_FPS)) > 1e-4:
        raise RuntimeError(
            f"Audio feature fps mismatch for {feature_path}: got {feature_fps}, expected {legacy_cache.TARGET_FPS}"
        )

    start_s = item.get("start")
    end_s = item.get("end")
    start_idx = 0 if start_s is None else max(0, int(math.floor(float(start_s) * feature_fps)))
    end_idx = feat.shape[0] if end_s is None else min(feat.shape[0], int(math.ceil(float(end_s) * feature_fps)))
    if end_idx <= start_idx:
        raise RuntimeError(f"Invalid audio slice for segment {item.get('seg_id') or item.get('id')}")
    clip = feat[start_idx:end_idx]
    if clip.shape[0] < target_len:
        out = np.zeros((target_len, clip.shape[1]), dtype=np.float32)
        out[: clip.shape[0]] = clip
        clip = out
    elif clip.shape[0] > target_len:
        clip = clip[:target_len]
    return clip.astype(np.float32, copy=False), feature_key, feature_fps, int(feat.shape[1])


def _build_lexical_features(
    *,
    words: Sequence[Dict[str, Any]],
    frames: int,
    fps: int,
    bert_encoder,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    from stageB.common.text_bert import align_dense_interval_embeddings_to_frames, extract_contextual_word_embeddings

    word_texts = [str(word["word"]) for word in words]
    word_times = (
        np.asarray([[float(word["start"]), float(word["end"])] for word in words], dtype=np.float32).reshape(-1, 2)
        if words
        else np.zeros((0, 2), dtype=np.float32)
    )
    lexical = np.zeros((frames, int(bert_encoder.hidden_size)), dtype=np.float32)
    if words:
        words_meta: List[Dict[str, Any]] = []
        for word in words:
            frame_start = max(0, min(frames, int(math.floor(float(word["start"]) * fps))))
            frame_end = max(frame_start, min(frames, int(math.ceil(float(word["end"]) * fps))))
            if frame_end <= frame_start:
                frame_end = min(frames, frame_start + 1)
            words_meta.append(
                {
                    "word": str(word["word"]),
                    "frame_start": int(frame_start),
                    "frame_end": int(frame_end),
                    "blank": False,
                }
            )
        lexical_words = extract_contextual_word_embeddings(bert_encoder, word_texts)
        lexical = align_dense_interval_embeddings_to_frames(words_meta, lexical_words, num_frames=frames)
    word_frame = _build_word_frame_features(word_times, fps=fps, frames=frames)
    return lexical.astype(np.float32), word_frame.astype(np.float32), word_texts


def _process_item(args: Tuple[Dict[str, Any], Path]) -> Dict[str, Any]:
    item, base_dir = args
    seg_key = str(item.get("seg_id") or item.get("id") or "")
    legacy_cache = _get_legacy_cache()
    try:
        clips = legacy_cache.process_item((item, base_dir))
        if not clips:
            raise RuntimeError("legacy motion extractor returned no clips")
        if len(clips) != 1:
            raise RuntimeError(f"Expected exactly one clip, got {len(clips)}")
        clip = dict(clips[0])
        words = _load_segment_words(item, base_dir)
        if not words:
            raise RuntimeError(f"Segment {seg_key} has no usable word timestamps")

        audio_clip, feature_key, feature_fps, audio_dim = _slice_audio_clip(
            item,
            base_dir=base_dir,
            target_len=int(clip["motion_len"]),
        )

        clip["segment_id"] = seg_key
        clip["text"] = str(item.get("text", "")).strip()
        clip["word_items"] = words
        clip["word_times"] = np.asarray(
            [[float(word["start"]), float(word["end"])] for word in words],
            dtype=np.float32,
        ).reshape(-1, 2)
        clip["audio"] = audio_clip
        clip["audio_len"] = int(audio_clip.shape[0])
        clip["audio_dim"] = int(audio_dim)
        clip["audio_feature_key"] = feature_key
        clip["audio_feature_fps"] = float(feature_fps)
        return {"ok": clip}
    except Exception as exc:
        return {"error": f"{seg_key}: {exc}"}


def _compute_feature_stats(samples: Sequence[Dict[str, Any]], key: str) -> Tuple[np.ndarray, np.ndarray]:
    total = None
    total_sq = None
    count = 0
    for sample in samples:
        arr = np.asarray(sample[key], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.size == 0:
            continue
        arr64 = arr.astype(np.float64, copy=False)
        cur_sum = arr64.sum(axis=0)
        cur_sq = np.square(arr64).sum(axis=0)
        if total is None:
            total = cur_sum
            total_sq = cur_sq
        else:
            total += cur_sum
            total_sq += cur_sq
        count += int(arr.shape[0])
    if total is None or total_sq is None or count <= 0:
        raise RuntimeError(f"Could not compute stats for key={key}")
    mean = total / float(count)
    var = np.maximum(total_sq / float(count) - np.square(mean), 0.0)
    std = np.sqrt(var + 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def _build_motion_contract_meta(
    *,
    skeleton: Dict[str, Any],
    foot_names: Sequence[str],
    contact_dim: int,
) -> Dict[str, Any]:
    legacy_cache = _get_legacy_cache()
    root_dim = 6
    has_yaw_vel = 1
    rot6d_start = root_dim + has_yaw_vel + int(contact_dim)
    return {
        "joint_names": skeleton["joint_names"],
        "all_joint_names": skeleton["all_joint_names"],
        "skeleton_parents": skeleton["parents"],
        "skeleton_offsets": skeleton["offsets"],
        "fps": legacy_cache.TARGET_FPS,
        "root_dim": root_dim,
        "has_yaw_vel": True,
        "contact_dim": int(contact_dim),
        "contact_indices": list(range(root_dim + has_yaw_vel, rot6d_start)),
        "foot_names": list(foot_names),
        "layout_version": "root_pos_abs_v2",
        "root_pos_mode": "absolute_xyz",
        "root_pos_indices": [0, 1, 2],
        "yaw_index": 3,
        "local_vel_xz_indices": [4, 5],
        "yaw_vel_index": 6,
        "rot6d_start": rot6d_start,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="manifests/train_whisperx_split.jsonl")
    parser.add_argument("--output", type=str, default="diffusion/cache/train_diffusion_v3.pt")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bert_model_dir", type=str, default="models/bert")
    parser.add_argument("--bert_device", type=str, default="cpu")
    parser.add_argument("--tokenizer_name", type=str, default="models/bert")
    parser.add_argument("--audio_model", type=str, default="w2v2_base")
    parser.add_argument("--skip_invalid", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    base_dir = manifest_path.parent.parent
    with manifest_path.open("r", encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    if args.limit and int(args.limit) > 0:
        items = items[: int(args.limit)]
    if not items:
        raise RuntimeError(f"No items found in manifest: {manifest_path}")
    legacy_cache = _get_legacy_cache()

    first_bvh = items[0].get("bvh") or items[0].get("motion")
    if not first_bvh:
        raise RuntimeError("Manifest does not contain a resolvable first BVH path")
    skeleton = legacy_cache.extract_skeleton_meta(base_dir / str(first_bvh))
    if skeleton is None:
        raise RuntimeError(f"Failed to extract skeleton metadata from {first_bvh}")

    if args.workers > 0:
        with Pool(int(args.workers), initializer=legacy_cache.init_worker) as pool:
            results = list(tqdm(pool.imap_unordered(_process_item, [(item, base_dir) for item in items]), total=len(items)))
    else:
        legacy_cache.init_worker()
        results = [_process_item((item, base_dir)) for item in tqdm(items)]

    failures = [str(result["error"]) for result in results if "error" in result]
    clips = [result["ok"] for result in results if "ok" in result]

    if failures and not args.skip_invalid:
        preview = "\n".join(failures[:20])
        raise RuntimeError(f"diffusion_cache_v3 strict build failed on {len(failures)} items:\n{preview}")
    if not clips:
        raise RuntimeError("No valid clips were produced")

    from stageB.common.text_bert import load_bert_word_encoder

    bert_encoder = load_bert_word_encoder(args.bert_model_dir, device=args.bert_device)
    samples: List[Dict[str, Any]] = []
    ref_audio_key = str(clips[0]["audio_feature_key"])
    ref_audio_dim = int(clips[0]["audio_dim"])
    ref_audio_fps = float(clips[0]["audio_feature_fps"])
    ref_foot_names = list(clips[0].get("foot_names", []))
    for clip in tqdm(clips, desc="bert", disable=False):
        if str(clip["audio_feature_key"]) != ref_audio_key:
            raise RuntimeError(
                f"Mixed audio feature keys are not allowed: {clip['audio_feature_key']} vs {ref_audio_key}"
            )
        if int(clip["audio_dim"]) != ref_audio_dim:
            raise RuntimeError(f"Mixed audio feature dims are not allowed: {clip['audio_dim']} vs {ref_audio_dim}")
        if abs(float(clip["audio_feature_fps"]) - ref_audio_fps) > 1e-4:
            raise RuntimeError(
                f"Mixed audio feature fps are not allowed: {clip['audio_feature_fps']} vs {ref_audio_fps}"
            )
        if list(clip.get("foot_names", [])) != ref_foot_names:
            raise RuntimeError("Mixed foot_names across segments are not allowed for the strict latent backend")
        lexical_frame, word_frame, word_texts = _build_lexical_features(
            words=clip["word_items"],
            frames=int(clip["motion_len"]),
            fps=legacy_cache.TARGET_FPS,
            bert_encoder=bert_encoder,
        )
        global_text = build_global_text_embedding(
            clip["text"],
            bert_model_dir=args.bert_model_dir,
            bert_device=args.bert_device,
            bert_encoder=bert_encoder,
        )
        sample = {
            "segment_id": str(clip["segment_id"]),
            "motion": np.asarray(clip["motion"], dtype=np.float32),
            "audio": np.asarray(clip["audio"], dtype=np.float32),
            "text": str(clip["text"]),
            "text_ids": np.asarray(clip["text_ids"], dtype=np.int64),
            "text_mask": np.asarray(clip["text_mask"], dtype=np.int64),
            "global_text": np.asarray(global_text, dtype=np.float32),
            "word_times": np.asarray(clip["word_times"], dtype=np.float32),
            "word_texts": list(word_texts),
            "lexical_frame": lexical_frame,
            "word_frame": word_frame,
            "motion_len": int(clip["motion_len"]),
            "audio_len": int(clip["audio_len"]),
            "src_bvh": str(clip.get("src_bvh", "")),
            "foot_names": list(clip.get("foot_names", [])),
        }
        samples.append(sample)

    if args.debug:
        print(f"[DEBUG] built {len(samples)} segments")
        for sample in samples[:5]:
            print(
                sample["segment_id"],
                sample["motion"].shape,
                sample["audio"].shape,
                sample["lexical_frame"].shape,
                sample["word_times"].shape,
            )

    motion_mean, motion_std = _compute_feature_stats(samples, "motion")
    audio_mean, audio_std = _compute_feature_stats(samples, "audio")

    first = clips[0]
    audio_spec = AudioFeatureSpec(
        spec_version=AUDIO_SPEC_VERSION,
        model=str(args.audio_model),
        layer=str(first["audio_feature_key"]),
        fps=float(first["audio_feature_fps"]),
        dim=int(first["audio_dim"]),
        normalized=False,
    )
    text_spec = TextTokenSpec(
        spec_version=TEXT_SPEC_VERSION,
        tokenizer_name=str(args.tokenizer_name),
        lexical_model=str(args.bert_model_dir),
        lexical_dim=int(bert_encoder.hidden_size),
        global_text_dim=int(bert_encoder.hidden_size),
        global_text_pooling="cls",
        frame_aligned=True,
        word_timestamps_required=True,
    )

    motion_meta = _build_motion_contract_meta(
        skeleton=skeleton,
        foot_names=list(first.get("foot_names", [])),
        contact_dim=int(first.get("contact_dim", 0)),
    )
    motion_contract = MotionContract.from_cache_meta(
        motion_meta,
        motion_dim=int(motion_mean.shape[0]),
        fps=legacy_cache.TARGET_FPS,
    )

    payload = {
        "cache_version": CACHE_VERSION,
        "fps": legacy_cache.TARGET_FPS,
        "mean": motion_mean,
        "std": motion_std,
        "audio_mean": audio_mean,
        "audio_std": audio_std,
        "motion_contract": motion_contract.to_dict(),
        "audio_feature_spec": audio_spec.to_dict(),
        "text_token_spec": text_spec.to_dict(),
        "segments": samples,
    }
    validate_cache_payload(payload, require_word_times=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(out_path))

    print(f"[INFO] Generated {len(samples)} segments")
    print(
        f"[INFO] motion_dim={motion_mean.shape[0]} "
        f"rot6d_start={motion_contract.rot6d_start} "
        f"contact_dim={len(motion_contract.contact_indices)}"
    )
    print(
        f"[INFO] audio_spec model={audio_spec.model} layer={audio_spec.layer} "
        f"fps={audio_spec.fps} dim={audio_spec.dim}"
    )
    print(f"[INFO] Saved strict v3 cache to {out_path}")


if __name__ == "__main__":
    main()
