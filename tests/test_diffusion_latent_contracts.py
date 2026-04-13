import copy

import numpy as np
import pytest
import torch

from diffusion.latent.contracts import CacheContractError, validate_cache_payload
from diffusion.latent.runtime import build_global_text_embedding


def _make_payload() -> dict:
    motion_dim = 465
    audio_dim = 768
    lexical_dim = 768
    sample = {
        "segment_id": "seg0",
        "motion": np.zeros((16, motion_dim), dtype=np.float32),
        "audio": np.zeros((16, audio_dim), dtype=np.float32),
        "text": "hello world",
        "text_ids": np.zeros((8,), dtype=np.int64),
        "text_mask": np.ones((8,), dtype=np.int64),
        "global_text": np.zeros((lexical_dim,), dtype=np.float32),
        "word_times": np.asarray([[0.0, 0.3], [0.4, 0.8]], dtype=np.float32),
        "word_texts": ["hello", "world"],
        "lexical_frame": np.zeros((16, lexical_dim), dtype=np.float32),
        "word_frame": np.zeros((16, 5), dtype=np.float32),
        "motion_len": 16,
        "audio_len": 16,
    }
    return {
        "cache_version": "diffusion_cache_v3",
        "fps": 30,
        "mean": np.zeros((motion_dim,), dtype=np.float32),
        "std": np.ones((motion_dim,), dtype=np.float32),
        "audio_mean": np.zeros((audio_dim,), dtype=np.float32),
        "audio_std": np.ones((audio_dim,), dtype=np.float32),
        "motion_contract": {
            "layout_version": "root_pos_abs_v2",
            "motion_dim": motion_dim,
            "rot6d_start": 15,
            "contact_indices": list(range(7, 15)),
            "foot_names": [
                "RightFoot",
                "RightForeFoot",
                "RightToeBase",
                "RightToeBaseEnd",
                "LeftFoot",
                "LeftForeFoot",
                "LeftToeBase",
                "LeftToeBaseEnd",
            ],
            "fps": 30,
            "layout_meta": {
                "layout_version": "root_pos_abs_v2",
                "root_pos_mode": "absolute_xyz",
                "root_pos_indices": [0, 1, 2],
                "yaw_index": 3,
                "local_vel_xz_indices": [4, 5],
                "yaw_vel_index": 6,
                "rot6d_start": 15,
                "contact_indices": list(range(7, 15)),
                "foot_names": [
                    "RightFoot",
                    "RightForeFoot",
                    "RightToeBase",
                    "RightToeBaseEnd",
                    "LeftFoot",
                    "LeftForeFoot",
                    "LeftToeBase",
                    "LeftToeBaseEnd",
                ],
                "joint_names": [f"Joint{i}" for i in range(75)],
                "all_joint_names": [f"Joint{i}" for i in range(88)],
                "skeleton_offsets": [[0.0, 0.0, 0.0] for _ in range(88)],
                "skeleton_parents": [-1] + [0] * 87,
            },
        },
        "audio_feature_spec": {
            "spec_version": "audio_spec_v1",
            "model": "w2v2_base",
            "layer": "w2v2_30fps",
            "fps": 30.0,
            "dim": audio_dim,
            "normalized": False,
        },
        "text_token_spec": {
            "spec_version": "text_spec_v1",
            "tokenizer_name": "models/bert",
            "lexical_model": "models/bert",
            "lexical_dim": lexical_dim,
            "global_text_dim": lexical_dim,
            "global_text_pooling": "cls",
            "frame_aligned": True,
            "word_timestamps_required": True,
        },
        "segments": [sample],
    }


def test_validate_cache_payload_accepts_strict_v3() -> None:
    payload = _make_payload()
    motion_contract, audio_spec, text_spec = validate_cache_payload(payload, require_word_times=True)
    assert motion_contract.motion_dim == 465
    assert motion_contract.rot6d_start == 15
    assert audio_spec.dim == 768
    assert text_spec.lexical_dim == 768
    assert text_spec.global_text_dim == 768


def test_validate_cache_payload_rejects_missing_global_text() -> None:
    payload = _make_payload()
    bad = copy.deepcopy(payload)
    del bad["segments"][0]["global_text"]
    with pytest.raises(CacheContractError):
        validate_cache_payload(bad, require_word_times=True)


def test_validate_cache_payload_rejects_empty_word_times() -> None:
    payload = _make_payload()
    bad = copy.deepcopy(payload)
    bad["segments"][0]["word_times"] = np.zeros((0, 2), dtype=np.float32)
    bad["segments"][0]["word_texts"] = []
    with pytest.raises(CacheContractError):
        validate_cache_payload(bad, require_word_times=True)


def test_build_global_text_embedding_uses_cls_token() -> None:
    class _FakeTokenizer:
        def __call__(self, *args, **kwargs):
            return {
                "input_ids": torch.tensor([[101, 11, 12, 102]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            }

    class _FakeModel:
        def __init__(self):
            self.config = type("Config", (), {"max_position_embeddings": 512})()

        def __call__(self, **kwargs):
            hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
            return type("Out", (), {"last_hidden_state": hidden})()

    fake_encoder = type(
        "Encoder",
        (),
        {
            "tokenizer": _FakeTokenizer(),
            "model": _FakeModel(),
            "device": torch.device("cpu"),
            "hidden_size": 2,
        },
    )()

    global_text = build_global_text_embedding("hello world", bert_model_dir="unused", bert_device="cpu", bert_encoder=fake_encoder)
    assert global_text.shape == (2,)
    assert np.allclose(global_text, np.asarray([1.0, 2.0], dtype=np.float32))
