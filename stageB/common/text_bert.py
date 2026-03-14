from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class BertWordEncoder:
    model_dir: Path
    device: torch.device
    tokenizer: object
    model: torch.nn.Module
    hidden_size: int


def load_bert_word_encoder(
    model_dir: str | Path = "models/bert",
    *,
    device: str | torch.device = "cpu",
) -> BertWordEncoder:
    model_path = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
    model.eval()
    dev = torch.device(device)
    model.to(dev)
    for param in model.parameters():
        param.requires_grad_(False)
    hidden_size = int(getattr(model.config, "hidden_size", 768))
    return BertWordEncoder(
        model_dir=model_path,
        device=dev,
        tokenizer=tokenizer,
        model=model,
        hidden_size=hidden_size,
    )


def _encode_word_chunk(encoder: BertWordEncoder, words: Sequence[str]) -> np.ndarray:
    if not words:
        return np.zeros((0, encoder.hidden_size), dtype=np.float32)
    enc = encoder.tokenizer(
        list(words),
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=min(512, int(getattr(encoder.model.config, "max_position_embeddings", 512))),
    )
    word_ids = enc.word_ids()
    enc = {k: v.to(encoder.device) for k, v in enc.items()}
    with torch.no_grad():
        hidden = encoder.model(**enc).last_hidden_state[0]

    out = torch.zeros((len(words), hidden.shape[-1]), device=hidden.device, dtype=hidden.dtype)
    counts = torch.zeros((len(words),), device=hidden.device, dtype=hidden.dtype)
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx < 0 or word_idx >= len(words):
            continue
        out[word_idx] = out[word_idx] + hidden[token_idx]
        counts[word_idx] = counts[word_idx] + 1.0
    counts = counts.clamp_min_(1.0).unsqueeze(-1)
    out = out / counts
    return out.detach().cpu().numpy().astype(np.float32)


def extract_contextual_word_embeddings(
    encoder: BertWordEncoder,
    words: Sequence[str],
    *,
    max_words_per_chunk: int = 128,
    overlap_words: int = 16,
) -> np.ndarray:
    if not words:
        return np.zeros((0, encoder.hidden_size), dtype=np.float32)

    max_words_per_chunk = max(1, int(max_words_per_chunk))
    overlap_words = max(0, int(overlap_words))
    if max_words_per_chunk <= overlap_words:
        overlap_words = max(0, max_words_per_chunk // 4)
    step = max(1, max_words_per_chunk - overlap_words)

    n_words = len(words)
    accum = np.zeros((n_words, encoder.hidden_size), dtype=np.float32)
    counts = np.zeros((n_words, 1), dtype=np.float32)

    for start in range(0, n_words, step):
        end = min(n_words, start + max_words_per_chunk)
        chunk_vec = _encode_word_chunk(encoder, words[start:end])
        accum[start:end] += chunk_vec
        counts[start:end] += 1.0
        if end >= n_words:
            break

    counts = np.maximum(counts, 1.0)
    return accum / counts


def build_dense_interval_bert_embeddings(
    words_meta: Sequence[dict],
    *,
    encoder: BertWordEncoder,
    max_words_per_chunk: int = 128,
    overlap_words: int = 16,
) -> np.ndarray:
    nonblank_words = [str(info.get("word", "") or "") for info in words_meta if not bool(info.get("blank", False))]
    nonblank_vecs = extract_contextual_word_embeddings(
        encoder,
        nonblank_words,
        max_words_per_chunk=max_words_per_chunk,
        overlap_words=overlap_words,
    )

    dense = np.zeros((len(words_meta), encoder.hidden_size), dtype=np.float32)
    cursor = 0
    for idx, info in enumerate(words_meta):
        if bool(info.get("blank", False)):
            continue
        if cursor < nonblank_vecs.shape[0]:
            dense[idx] = nonblank_vecs[cursor]
        cursor += 1
    return dense


def align_dense_interval_embeddings_to_frames(
    words_meta: Sequence[dict],
    interval_embeddings: np.ndarray,
    *,
    num_frames: int,
    dtype=np.float32,
) -> np.ndarray:
    interval_embeddings = np.asarray(interval_embeddings, dtype=np.float32)
    if interval_embeddings.ndim != 2:
        raise ValueError(f"Expected [N, D] interval embeddings, got {tuple(interval_embeddings.shape)}")
    out = np.zeros((int(num_frames), int(interval_embeddings.shape[1])), dtype=np.float32)
    for idx, info in enumerate(words_meta):
        if idx >= interval_embeddings.shape[0]:
            break
        start = int(info.get("frame_start", 0))
        end = int(info.get("frame_end", start))
        start = max(0, min(int(num_frames), start))
        end = max(start, min(int(num_frames), end))
        if end <= start:
            continue
        out[start:end] = interval_embeddings[idx]
    return out.astype(dtype, copy=False)
