from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _to_tensor(x, *, dtype=None):
    if torch.is_tensor(x):
        out = x.clone().detach()
    else:
        out = torch.from_numpy(np.asarray(x))
    return out.to(dtype=dtype) if dtype is not None else out


class Stage2SemanticDataset(Dataset):
    def __init__(self, cache_path: Path):
        self.cache_path = Path(cache_path)
        pack = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        self.samples = pack["samples"]
        self.meta = pack.get("meta", {})
        self.stage1 = pack.get("stage1", {})
        self.vocab = pack.get("vocab", {})
        self.prosody_feature_names = list(pack.get("prosody_feature_names", []))
        self.text_scalar_feature_names = list(pack.get("text_scalar_feature_names", []))
        self.bert_feature_dim = int(pack.get("bert_feature_dim", 0))
        self.token_stride = int(pack.get("token_stride", 1))
        self.block_size = int(pack.get("block_size", 0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[int(idx)]
        token_word_ids = np.asarray(sample["token_word_ids"])
        if "token_text_bert" in sample:
            token_text_bert = sample["token_text_bert"]
        else:
            token_text_bert = np.zeros((int(token_word_ids.shape[0]), int(self.bert_feature_dim)), dtype=np.float32)
        batch = {
            "full_motion": _to_tensor(sample["full_motion"], dtype=torch.float32),
            "prosody_frame": _to_tensor(sample["prosody_frame"], dtype=torch.float32),
            "text_word_ids_frame": _to_tensor(sample["text_word_ids_frame"], dtype=torch.long),
            "text_scalar_frame": _to_tensor(sample["text_scalar_frame"], dtype=torch.float32),
            "token_prosody": _to_tensor(sample["token_prosody"], dtype=torch.float32),
            "token_word_ids": _to_tensor(token_word_ids, dtype=torch.long),
            "token_text_scalar": _to_tensor(sample["token_text_scalar"], dtype=torch.float32),
            "token_text_bert": _to_tensor(token_text_bert, dtype=torch.float32),
            "meta": sample["meta"],
        }
        batch["part_motion"] = {
            name: _to_tensor(value, dtype=torch.float32)
            for name, value in sample["part_motion"].items()
        }
        batch["codes"] = {
            name: _to_tensor(value, dtype=torch.long)
            for name, value in sample["codes"].items()
        }
        return batch


def collate_stage2_semantic(batch: list[dict]) -> Optional[dict]:
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    out = {
        "full_motion": torch.stack([item["full_motion"] for item in batch], dim=0),
        "prosody_frame": torch.stack([item["prosody_frame"] for item in batch], dim=0),
        "text_word_ids_frame": torch.stack([item["text_word_ids_frame"] for item in batch], dim=0),
        "text_scalar_frame": torch.stack([item["text_scalar_frame"] for item in batch], dim=0),
        "token_prosody": torch.stack([item["token_prosody"] for item in batch], dim=0),
        "token_word_ids": torch.stack([item["token_word_ids"] for item in batch], dim=0),
        "token_text_scalar": torch.stack([item["token_text_scalar"] for item in batch], dim=0),
        "token_text_bert": torch.stack([item["token_text_bert"] for item in batch], dim=0),
        "part_motion": {},
        "codes": {},
        "meta": [item["meta"] for item in batch],
    }

    for name in batch[0]["part_motion"].keys():
        out["part_motion"][name] = torch.stack([item["part_motion"][name] for item in batch], dim=0)
    for name in batch[0]["codes"].keys():
        out["codes"][name] = torch.stack([item["codes"][name] for item in batch], dim=0)
    return out


def build_all_mask_code_inputs(
    code_shape: tuple[int, int],
    codebook_sizes: Dict[str, int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch_size, seq_len = int(code_shape[0]), int(code_shape[1])
    return {
        name: torch.full((batch_size, seq_len), int(size), dtype=torch.long, device=device)
        for name, size in codebook_sizes.items()
    }


def _expand_token_mask(mask: torch.Tensor, span: int) -> torch.Tensor:
    span = max(1, int(span))
    if span <= 1:
        return mask
    if span % 2 == 0:
        span += 1
    expanded = F.max_pool1d(mask.float().unsqueeze(1), kernel_size=span, stride=1, padding=span // 2)
    return expanded.squeeze(1) > 0.5


def sample_masked_code_inputs(
    code_targets: Dict[str, torch.Tensor],
    codebook_sizes: Dict[str, int],
    *,
    mask_ratio: float = 0.35,
    mask_span: int = 3,
    random_replace_prob: float = 0.1,
    keep_original_prob: float = 0.1,
    batch_drop_prob: float = 0.25,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if not code_targets:
        raise ValueError("code_targets must not be empty")

    ref = next(iter(code_targets.values()))
    if ref.ndim != 2:
        raise ValueError(f"Expected code_targets tensors with shape [B, L], got {tuple(ref.shape)}")

    device = ref.device
    batch_size, seq_len = int(ref.shape[0]), int(ref.shape[1])
    mask = torch.rand((batch_size, seq_len), device=device) < float(mask_ratio)
    mask = _expand_token_mask(mask, mask_span)

    if float(batch_drop_prob) > 0.0:
        batch_drop = torch.rand((batch_size, 1), device=device) < float(batch_drop_prob)
        mask = torch.where(batch_drop, torch.ones_like(mask), mask)

    code_inputs: dict[str, torch.Tensor] = {}
    for name, target in code_targets.items():
        n_codes = int(codebook_sizes[name])
        input_ids = target.clone()
        replace_draw = torch.rand_like(target, dtype=torch.float32)
        random_codes = torch.randint(0, n_codes, target.shape, device=device)

        use_mask_token = mask & (replace_draw < 1.0 - float(random_replace_prob) - float(keep_original_prob))
        use_random = mask & (replace_draw >= 1.0 - float(random_replace_prob) - float(keep_original_prob)) & (
            replace_draw < 1.0 - float(keep_original_prob)
        )

        input_ids = torch.where(use_mask_token, torch.full_like(input_ids, n_codes), input_ids)
        input_ids = torch.where(use_random, random_codes, input_ids)
        code_inputs[name] = input_ids
    return code_inputs, mask


def code_usage_regularization(logits: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    if weights is None:
        marginal = probs.mean(dim=(0, 1))
    else:
        norm_w = weights / (weights.sum() + 1e-6)
        marginal = (probs * norm_w.unsqueeze(-1)).sum(dim=(0, 1))
    entropy = -(marginal * torch.log(marginal + 1e-8)).sum()
    max_entropy = np.log(float(logits.shape[-1])) if int(logits.shape[-1]) > 0 else 0.0
    return logits.new_tensor(float(max_entropy)) - entropy


def compute_token_emphasis_weights(
    token_prosody: torch.Tensor,
    token_text_scalar: torch.Tensor,
    *,
    onset_index: int = 2,
    blank_index: int = 0,
    word_start_index: int = 1,
    word_end_index: int = 2,
    speech_boost: float = 0.25,
    boundary_boost: float = 0.25,
    onset_boost: float = 0.25,
) -> torch.Tensor:
    onset = token_prosody[..., onset_index]
    onset = onset - onset.amin(dim=1, keepdim=True)
    onset = onset / (onset.amax(dim=1, keepdim=True) + 1e-6)

    non_blank = 1.0 - token_text_scalar[..., blank_index]
    boundaries = torch.clamp(token_text_scalar[..., word_start_index] + token_text_scalar[..., word_end_index], 0.0, 1.0)

    weights = 1.0 + speech_boost * non_blank + boundary_boost * boundaries + onset_boost * onset
    return weights.detach()


def compute_frame_emphasis_weights(
    prosody_frame: torch.Tensor,
    text_scalar_frame: torch.Tensor,
    *,
    onset_index: int = 2,
    blank_index: int = 0,
    word_start_index: int = 1,
    word_end_index: int = 2,
    speech_boost: float = 0.25,
    boundary_boost: float = 0.25,
    onset_boost: float = 0.25,
) -> torch.Tensor:
    onset = prosody_frame[..., onset_index]
    onset = onset - onset.amin(dim=1, keepdim=True)
    onset = onset / (onset.amax(dim=1, keepdim=True) + 1e-6)

    non_blank = 1.0 - text_scalar_frame[..., blank_index]
    boundaries = torch.clamp(text_scalar_frame[..., word_start_index] + text_scalar_frame[..., word_end_index], 0.0, 1.0)
    weights = 1.0 + speech_boost * non_blank + boundary_boost * boundaries + onset_boost * onset
    return weights.detach()


def weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    if weights is None:
        return loss.mean()
    while weights.ndim < loss.ndim:
        weights = weights.unsqueeze(-1)
    return (loss * weights).sum() / (weights.sum() * loss.shape[-1] + 1e-6)


def weighted_velocity_loss(pred: torch.Tensor, target: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    pred_v = pred[:, 1:] - pred[:, :-1]
    target_v = target[:, 1:] - target[:, :-1]
    w = None if weights is None else 0.5 * (weights[:, 1:] + weights[:, :-1])
    return weighted_smooth_l1(pred_v, target_v, w)


def weighted_acceleration_loss(pred: torch.Tensor, target: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    if pred.shape[1] < 3:
        return pred.new_tensor(0.0)
    pred_a = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
    target_a = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
    w = None if weights is None else (weights[:, 2:] + weights[:, 1:-1] + weights[:, :-2]) / 3.0
    return weighted_smooth_l1(pred_a, target_a, w)
