from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from stageA.train_stage1_vqvae import (
    TARGET_FPS,
    _resolve,
    build_speech_segments_from_intervals,
    choose_best_tier,
    parse_textgrid_interval_tiers,
)


BLANK_TOKEN = "<blank>"
UNK_TOKEN = "<unk>"


@dataclass
class WordFrameFeatures:
    word_ids: np.ndarray
    scalar: np.ndarray
    feature_names: list[str]
    words: list[dict]


def normalize_word(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _intervals_from_word_timestamps(word_timestamps: Sequence[dict]) -> list[tuple[float, float, str]]:
    intervals: list[tuple[float, float, str]] = []
    for item in word_timestamps or []:
        word = item.get("word", "")
        start = float(item.get("start", 0.0))
        end = float(item.get("end", 0.0))
        if end <= start:
            continue
        intervals.append((start, end, str(word or "")))
    return sorted(intervals, key=lambda x: (x[0], x[1]))


def load_word_intervals_from_item(
    item: dict,
    base_dir: Path,
    *,
    prefer_tier: Optional[str] = None,
    allow_manifest_fallback: bool = True,
) -> tuple[list[tuple[float, float, str]], Optional[Path], Optional[str]]:
    tg_rel = item.get("textgrid") or item.get("TextGrid")
    tg_path = None
    if tg_rel:
        tg_path = _resolve(base_dir, tg_rel)
        if tg_path.exists():
            tiers = parse_textgrid_interval_tiers(tg_path)
            tier = choose_best_tier(tiers, prefer=prefer_tier)
            if tier is not None:
                return sorted(tiers.get(tier, []), key=lambda x: (x[0], x[1])), tg_path, tier

    if allow_manifest_fallback and item.get("word_timestamps"):
        return _intervals_from_word_timestamps(item["word_timestamps"]), tg_path, None

    return [], tg_path, None


def build_word_vocab(
    items: Sequence[dict],
    *,
    base_dir: Path,
    prefer_tier: Optional[str] = None,
    min_freq: int = 1,
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in items:
        intervals, _, _ = load_word_intervals_from_item(item, base_dir, prefer_tier=prefer_tier)
        for _, _, text in intervals:
            word = normalize_word(text)
            if word:
                counter[word] += 1

    vocab = {BLANK_TOKEN: 0, UNK_TOKEN: 1}
    for word, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        if int(freq) >= int(min_freq) and word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def _ensure_blank_intervals(
    intervals: Sequence[tuple[float, float, str]],
    duration_s: float,
) -> list[tuple[float, float, str]]:
    duration_s = max(0.0, float(duration_s))
    if not intervals:
        return [(0.0, duration_s, "")]

    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    out: list[tuple[float, float, str]] = []
    cursor = 0.0
    for start, end, text in intervals:
        start = max(0.0, float(start))
        end = min(duration_s, float(end))
        if end <= start:
            continue
        if start > cursor:
            out.append((cursor, start, ""))
        out.append((start, end, str(text or "")))
        cursor = end
    if cursor < duration_s:
        out.append((cursor, duration_s, ""))
    if not out:
        out.append((0.0, duration_s, ""))
    return out


def align_words_to_frames(
    intervals: Sequence[tuple[float, float, str]],
    *,
    num_frames: int,
    fps: int = TARGET_FPS,
    vocab: Optional[dict[str, int]] = None,
) -> WordFrameFeatures:
    blank_id = 0 if vocab is None else int(vocab.get(BLANK_TOKEN, 0))
    unk_id = 1 if vocab is None else int(vocab.get(UNK_TOKEN, 1))
    duration_s = max(0.0, float(num_frames) / float(fps))
    dense_intervals = _ensure_blank_intervals(intervals, duration_s)

    word_ids = np.full((num_frames,), blank_id, dtype=np.int64)
    blank_mask = np.ones((num_frames,), dtype=np.float32)
    word_start = np.zeros((num_frames,), dtype=np.float32)
    word_end = np.zeros((num_frames,), dtype=np.float32)
    rel_pos = np.zeros((num_frames,), dtype=np.float32)
    word_duration = np.zeros((num_frames,), dtype=np.float32)
    prev_boundary = np.zeros((num_frames,), dtype=np.float32)
    next_boundary = np.zeros((num_frames,), dtype=np.float32)

    frame_centers = (np.arange(num_frames, dtype=np.float32) + 0.5) / float(fps)
    words_meta: list[dict] = []
    cursor = 0

    for start, end, text in dense_intervals:
        word = normalize_word(text)
        start_idx = int(np.floor(start * fps + 1e-6))
        end_idx = int(np.ceil(end * fps - 1e-6))
        start_idx = max(0, min(num_frames, start_idx))
        end_idx = max(start_idx, min(num_frames, end_idx))
        if end_idx <= start_idx and num_frames > 0:
            nearest = min(num_frames - 1, max(0, int(round(start * fps))))
            start_idx = nearest
            end_idx = min(num_frames, nearest + 1)
        if end_idx <= start_idx:
            continue

        token_id = blank_id
        if word:
            token_id = unk_id if vocab is None else int(vocab.get(word, unk_id))

        word_ids[start_idx:end_idx] = token_id
        blank_mask[start_idx:end_idx] = 1.0 if word == "" else 0.0
        word_start[start_idx] = 1.0
        word_end[end_idx - 1] = 1.0

        dur = max(float(end - start), 1.0 / float(fps))
        centers = frame_centers[start_idx:end_idx]
        rel_pos[start_idx:end_idx] = np.clip((centers - float(start)) / dur, 0.0, 1.0)
        word_duration[start_idx:end_idx] = dur
        prev_boundary[start_idx:end_idx] = np.clip(centers - float(start), 0.0, None)
        next_boundary[start_idx:end_idx] = np.clip(float(end) - centers, 0.0, None)

        words_meta.append(
            {
                "index": cursor,
                "word": word,
                "start": float(start),
                "end": float(end),
                "blank": bool(word == ""),
                "frame_start": int(start_idx),
                "frame_end": int(end_idx),
            }
        )
        cursor += 1

    scalar = np.stack(
        [
            blank_mask,
            word_start,
            word_end,
            rel_pos,
            word_duration,
            prev_boundary,
            next_boundary,
        ],
        axis=-1,
    ).astype(np.float32)
    return WordFrameFeatures(
        word_ids=word_ids,
        scalar=scalar,
        feature_names=[
            "blank",
            "word_start",
            "word_end",
            "rel_pos",
            "word_duration_s",
            "prev_boundary_s",
            "next_boundary_s",
        ],
        words=words_meta,
    )


def pool_frame_features_to_tokens(frame_feat: np.ndarray, token_stride: int, mode: str = "mean") -> np.ndarray:
    if frame_feat.ndim == 1:
        frame_feat = frame_feat[:, None]
    T, C = frame_feat.shape
    if T % int(token_stride) != 0:
        raise ValueError(f"Frame length {T} must be divisible by token_stride={token_stride}")
    x = frame_feat.reshape(T // int(token_stride), int(token_stride), C)
    if mode == "max":
        return x.max(axis=1).astype(np.float32)
    if mode == "last":
        return x[:, -1].astype(np.float32)
    return x.mean(axis=1).astype(np.float32)


def pool_frame_ids_to_tokens(
    frame_ids: np.ndarray,
    frame_blank_mask: np.ndarray,
    token_stride: int,
    *,
    blank_id: int = 0,
) -> np.ndarray:
    T = int(frame_ids.shape[0])
    if T % int(token_stride) != 0:
        raise ValueError(f"Frame length {T} must be divisible by token_stride={token_stride}")

    token_ids = np.full((T // int(token_stride),), int(blank_id), dtype=np.int64)
    ids = frame_ids.reshape(-1, int(token_stride))
    blanks = frame_blank_mask.reshape(-1, int(token_stride))
    for i in range(ids.shape[0]):
        non_blank = ids[i][blanks[i] < 0.5]
        if non_blank.size == 0:
            token_ids[i] = int(blank_id)
            continue
        values, counts = np.unique(non_blank, return_counts=True)
        token_ids[i] = int(values[np.argmax(counts)])
    return token_ids


def dump_vocab_json(path: Path, vocab: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
