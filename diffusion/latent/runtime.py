from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def load_w2v2_feature(
    npz_path: str | Path,
    *,
    expected_dim: int,
    expected_fps: float,
) -> tuple[np.ndarray, float]:
    npz = np.load(str(npz_path), allow_pickle=True)
    feature_key = None
    for key in ("w2v2_30fps", "features", "feat", "feats", "hidden", "arr_0"):
        if key in npz:
            feature_key = key
            break
    if feature_key is None:
        raise RuntimeError(f"Could not find feature array in {npz_path}")
    feat = np.asarray(npz[feature_key], dtype=np.float32)
    if feat.ndim != 2:
        raise RuntimeError(f"Expected [T,D] audio features in {npz_path}, got {feat.shape}")
    fps = None
    for key in ("fps", "frame_rate", "feature_fps"):
        if key in npz:
            fps = float(np.asarray(npz[key]).reshape(-1)[0])
            break
    if fps is None and feature_key == "w2v2_30fps":
        fps = 30.0
    if fps is None:
        raise RuntimeError(f"Audio feature npz is missing fps metadata: {npz_path}")
    if feat.shape[1] != int(expected_dim):
        raise RuntimeError(f"Audio feature dim mismatch: got {feat.shape[1]}, expected {expected_dim}")
    if abs(float(fps) - float(expected_fps)) > 1e-4:
        raise RuntimeError(f"Audio feature fps mismatch: got {fps}, expected {expected_fps}")
    return feat.astype(np.float32), float(fps)


def _whisper_cache_key(audio_path: str | Path, model_name: str, language: Optional[str], with_word_ts: bool) -> str:
    path = Path(audio_path).resolve()
    stat = path.stat()
    raw = f"{path}|{stat.st_size}|{int(stat.st_mtime)}|{model_name}|{language}|{int(with_word_ts)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_or_create_whisper_segments(
    *,
    audio_path: str | Path,
    cache_dir: str | Path,
    model_name: str,
    language: Optional[str],
    with_word_ts: bool,
) -> Dict[str, Any]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _whisper_cache_key(audio_path, model_name, language, with_word_ts)
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        expected_audio = str(Path(audio_path).resolve())
        cached_audio = str(cached.get("audio_path", ""))
        if cached_audio and Path(cached_audio).resolve() != Path(expected_audio).resolve():
            raise RuntimeError(
                f"Whisper cache audio_path mismatch for {cache_path}: got {cached_audio}, expected {expected_audio}"
            )
        if cached.get("model_name") not in (None, model_name):
            raise RuntimeError(
                f"Whisper cache model mismatch for {cache_path}: got {cached.get('model_name')}, expected {model_name}"
            )
        if cached.get("language") not in (None, language):
            raise RuntimeError(
                f"Whisper cache language mismatch for {cache_path}: got {cached.get('language')}, expected {language}"
            )
        if cached.get("with_word_ts") not in (None, bool(with_word_ts)):
            raise RuntimeError(
                f"Whisper cache word timestamp mismatch for {cache_path}: got {cached.get('with_word_ts')}, expected {bool(with_word_ts)}"
            )
        return cached

    try:
        import whisper
    except Exception as exc:
        raise RuntimeError("whisper is required for inference when no cache file exists") from exc

    model = whisper.load_model(model_name)
    result = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=with_word_ts,
        verbose=False,
    )
    result["audio_path"] = str(Path(audio_path).resolve())
    result["model_name"] = model_name
    result["language"] = language
    result["with_word_ts"] = bool(with_word_ts)
    cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def merge_transcript_segments(
    whisper_result: Dict[str, Any],
    *,
    max_seg_dur: float,
    gap_th: float,
) -> List[Dict[str, Any]]:
    base_segments = whisper_result.get("segments", [])
    merged: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for seg in base_segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = float(seg["start"])
        end = float(seg["end"])
        words = [dict(w) for w in seg.get("words", []) if "start" in w and "end" in w]
        if current is None:
            current = {"start": start, "end": end, "text": text, "words": words}
            continue
        gap = start - float(current["end"])
        new_dur = end - float(current["start"])
        if gap <= float(gap_th) and new_dur <= float(max_seg_dur):
            current["end"] = end
            current["text"] = (str(current["text"]).rstrip() + " " + text.lstrip()).strip()
            current["words"].extend(words)
        else:
            merged.append(current)
            current = {"start": start, "end": end, "text": text, "words": words}
    if current is not None:
        merged.append(current)
    return merged


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


def build_global_text_embedding(
    text: str,
    *,
    bert_model_dir: str,
    bert_device: str,
    bert_encoder=None,
) -> np.ndarray:
    if bert_encoder is None:
        from stageB.common.text_bert import load_bert_word_encoder

        encoder = load_bert_word_encoder(bert_model_dir, device=bert_device)
    else:
        encoder = bert_encoder
    encoded = encoder.tokenizer(
        str(text).strip(),
        return_tensors="pt",
        truncation=True,
        max_length=min(512, int(getattr(encoder.model.config, "max_position_embeddings", 512))),
    )
    encoded = {key: value.to(encoder.device) for key, value in encoded.items()}
    with torch.no_grad():
        hidden = encoder.model(**encoded).last_hidden_state[0, 0]
    return hidden.detach().cpu().numpy().astype(np.float32)


def build_segment_condition_frames(
    *,
    text: str,
    words: Sequence[Dict[str, Any]],
    segment_start: float,
    segment_end: float,
    fps: int,
    bert_model_dir: str,
    bert_device: str,
    bert_encoder=None,
) -> Dict[str, Any]:
    from stageB.common.text_bert import (
        align_dense_interval_embeddings_to_frames,
        extract_contextual_word_embeddings,
        load_bert_word_encoder,
    )

    duration = max(1e-3, float(segment_end) - float(segment_start))
    frames = max(2, int(round(duration * float(fps))))

    word_texts: List[str] = []
    word_times: List[List[float]] = []
    words_meta: List[Dict[str, Any]] = []

    for word in words:
        if "word" not in word or "start" not in word or "end" not in word:
            continue
        start = max(float(word["start"]), float(segment_start))
        end = min(float(word["end"]), float(segment_end))
        if end <= start + 1e-4:
            continue
        rel_start = start - float(segment_start)
        rel_end = end - float(segment_start)
        word_texts.append(str(word["word"]).strip())
        word_times.append([rel_start, rel_end])
        words_meta.append(
            {
                "word": str(word["word"]).strip(),
                "frame_start": int(max(0, min(frames, math.floor(rel_start * fps)))),
                "frame_end": int(max(0, min(frames, math.ceil(rel_end * fps)))),
                "blank": False,
            }
        )

    word_times_np = np.asarray(word_times, dtype=np.float32).reshape(-1, 2) if word_times else np.zeros((0, 2), dtype=np.float32)
    word_frame = _build_word_frame_features(word_times_np, fps=fps, frames=frames)

    encoder = bert_encoder or load_bert_word_encoder(bert_model_dir, device=bert_device)
    lexical_word = extract_contextual_word_embeddings(encoder, word_texts)
    lexical_frame = align_dense_interval_embeddings_to_frames(words_meta, lexical_word, num_frames=frames)
    global_text = build_global_text_embedding(
        text,
        bert_model_dir=bert_model_dir,
        bert_device=bert_device,
        bert_encoder=encoder,
    )

    return {
        "text": text,
        "word_texts": word_texts,
        "word_times": word_times_np,
        "word_frame": word_frame.astype(np.float32),
        "lexical_frame": lexical_frame.astype(np.float32),
        "global_text": global_text.astype(np.float32),
        "frames": int(frames),
    }
