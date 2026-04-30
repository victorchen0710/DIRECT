from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    return torch.distributed.get_rank() if is_dist() else 0


def is_main() -> bool:
    return get_rank() == 0


def make_autocast(device: torch.device, enabled: bool):
    if not enabled:
        return torch.autocast(device_type="cpu", enabled=False)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    return torch.autocast(device_type=device_type, enabled=enabled)


def make_grad_scaler(device: torch.device, enabled: bool):
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self._init_from(model)

    def _init_from(self, model: torch.nn.Module) -> None:
        self.shadow = {}
        for key, value in model.state_dict().items():
            if torch.is_floating_point(value):
                self.shadow[key] = value.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for key, value in state.items():
            if not torch.is_floating_point(value):
                continue
            new_value = value.detach().float().cpu()
            if key not in self.shadow:
                self.shadow[key] = new_value.clone()
            else:
                self.shadow[key].mul_(self.decay).add_(new_value, alpha=(1.0 - self.decay))

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.decay = float(state_dict.get("decay", self.decay))
        self.shadow = {
            key: value.detach().clone().float().cpu()
            for key, value in dict(state_dict.get("shadow", {})).items()
        }


def sinusoidal_encoding(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    half = max(1, dim // 2)
    scales = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(1, half - 1)))
    angles = positions * scales.unsqueeze(0)
    emb = torch.zeros((length, dim), device=device, dtype=torch.float32)
    emb[:, 0::2] = torch.sin(angles[:, : emb[:, 0::2].shape[1]])
    emb[:, 1::2] = torch.cos(angles[:, : emb[:, 1::2].shape[1]])
    return emb.to(dtype=dtype)


@dataclass
class PaddedBatch:
    motion_norm: torch.Tensor
    motion_denorm: torch.Tensor
    motion_mask: torch.Tensor
    audio: torch.Tensor
    audio_mask: torch.Tensor
    lexical_frame: torch.Tensor
    lexical_mask: torch.Tensor
    global_text: torch.Tensor
    word_frame: torch.Tensor
    word_mask: torch.Tensor
    text_ids: torch.Tensor
    text_mask: torch.Tensor
    segment_ids: List[str]
    continuation_mask: Optional[torch.Tensor] = None


def _pad_float_sequences(items: List[torch.Tensor], pad_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(x.shape[0]) for x in items)
    batch = torch.zeros((len(items), max_len, pad_dim), dtype=items[0].dtype)
    mask = torch.zeros((len(items), max_len), dtype=torch.bool)
    for idx, value in enumerate(items):
        batch[idx, : value.shape[0]] = value
        mask[idx, : value.shape[0]] = True
    return batch, mask


def collate_motion_batch(batch: List[Dict[str, Any]]) -> PaddedBatch:
    motion_norm, motion_denorm, audio, lexical, global_text, word_frame = [], [], [], [], [], []
    text_ids, text_mask = [], []
    segment_ids: List[str] = []
    continuation_rows: List[Optional[torch.Tensor]] = []

    for item in batch:
        motion_norm.append(item["motion_norm"])
        motion_denorm.append(item["motion_denorm"])
        audio.append(item["audio"])
        lexical.append(item["lexical_frame"])
        global_text.append(item["global_text"])
        word_frame.append(item["word_frame"])
        text_ids.append(item["text_ids"])
        text_mask.append(item["text_mask"])
        segment_ids.append(str(item["segment_id"]))
        continuation_rows.append(item.get("continuation_mask"))

    motion_norm_t, motion_mask = _pad_float_sequences(motion_norm, motion_norm[0].shape[1])
    motion_denorm_t, _ = _pad_float_sequences(motion_denorm, motion_denorm[0].shape[1])
    audio_t, audio_mask = _pad_float_sequences(audio, audio[0].shape[1])
    lexical_t, lexical_mask = _pad_float_sequences(lexical, lexical[0].shape[1])
    global_text_t = torch.stack(global_text, dim=0)
    word_t, word_mask = _pad_float_sequences(word_frame, word_frame[0].shape[1])

    text_ids_t = pad_sequence(text_ids, batch_first=True)
    text_mask_t = pad_sequence(text_mask, batch_first=True)

    continuation_mask = None
    if any(value is not None for value in continuation_rows):
        max_len = motion_norm_t.shape[1]
        continuation_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
        for idx, value in enumerate(continuation_rows):
            if value is None:
                continue
            continuation_mask[idx, : value.shape[0]] = value

    return PaddedBatch(
        motion_norm=motion_norm_t,
        motion_denorm=motion_denorm_t,
        motion_mask=motion_mask,
        audio=audio_t,
        audio_mask=audio_mask,
        lexical_frame=lexical_t,
        lexical_mask=lexical_mask,
        global_text=global_text_t,
        word_frame=word_t,
        word_mask=word_mask,
        text_ids=text_ids_t,
        text_mask=text_mask_t,
        segment_ids=segment_ids,
        continuation_mask=continuation_mask,
    )
