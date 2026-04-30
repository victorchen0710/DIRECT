from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import CacheContractError, load_cache_contract, validate_sample


class MotionCacheV3Dataset(Dataset):
    def __init__(
        self,
        cache_path: str | Path,
        *,
        require_word_times: bool = True,
        continuation_prob: float = 0.0,
        max_prefix_ratio: float = 0.25,
    ):
        loaded = load_cache_contract(cache_path, require_word_times=require_word_times)
        self.payload = loaded["payload"]
        self.motion_contract = loaded["motion_contract"]
        self.audio_feature_spec = loaded["audio_feature_spec"]
        self.text_token_spec = loaded["text_token_spec"]
        self.samples = list(self.payload["segments"])
        self.mean = np.asarray(self.payload["mean"], dtype=np.float32)
        self.std = np.asarray(self.payload["std"], dtype=np.float32)
        self.std[self.std < 1e-6] = 1.0
        self.audio_mean = np.asarray(self.payload["audio_mean"], dtype=np.float32)
        self.audio_std = np.asarray(self.payload["audio_std"], dtype=np.float32)
        self.audio_std[self.audio_std < 1e-6] = 1.0
        self.require_word_times = bool(require_word_times)
        self.continuation_prob = float(max(0.0, continuation_prob))
        self.max_prefix_ratio = float(max(0.0, min(0.95, max_prefix_ratio)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        validate_sample(
            sample,
            motion_contract=self.motion_contract,
            audio_spec=self.audio_feature_spec,
            text_spec=self.text_token_spec,
            require_word_times=self.require_word_times,
        )

        motion = np.asarray(sample["motion"], dtype=np.float32)
        audio = np.asarray(sample["audio"], dtype=np.float32)
        lexical = np.asarray(sample["lexical_frame"], dtype=np.float32)
        global_text = np.asarray(sample["global_text"], dtype=np.float32)
        word_frame = np.asarray(sample["word_frame"], dtype=np.float32)

        motion_norm = (motion - self.mean) / self.std
        audio_norm = (audio - self.audio_mean) / self.audio_std

        out = {
            "segment_id": str(sample["segment_id"]),
            "motion_denorm": torch.from_numpy(motion).float(),
            "motion_norm": torch.from_numpy(motion_norm).float(),
            "audio": torch.from_numpy(audio_norm).float(),
            "lexical_frame": torch.from_numpy(lexical).float(),
            "global_text": torch.from_numpy(global_text).float(),
            "word_frame": torch.from_numpy(word_frame).float(),
            "text_ids": torch.as_tensor(sample["text_ids"]).long(),
            "text_mask": torch.as_tensor(sample["text_mask"]).long(),
        }

        if self.continuation_prob > 0.0 and random.random() < self.continuation_prob and motion.shape[0] >= 8:
            max_prefix = max(1, int(round(motion.shape[0] * self.max_prefix_ratio)))
            prefix = random.randint(1, max_prefix)
            continuation_mask = np.ones((motion.shape[0],), dtype=np.bool_)
            continuation_mask[:prefix] = False
            out["continuation_mask"] = torch.from_numpy(continuation_mask)
        else:
            out["continuation_mask"] = None
        return out
