from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from diffusion.feature_layout import MotionFeatureLayout


CACHE_VERSION = "diffusion_cache_v3"
AUDIO_SPEC_VERSION = "audio_spec_v1"
TEXT_SPEC_VERSION = "text_spec_v1"


class CacheContractError(RuntimeError):
    pass


def _require_keys(obj: Mapping[str, Any], keys: Iterable[str], *, prefix: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise CacheContractError(f"{prefix} missing required keys: {missing}")


def _as_list_str(values: Sequence[Any]) -> List[str]:
    return [str(v) for v in values]


@dataclass(frozen=True)
class AudioFeatureSpec:
    spec_version: str
    model: str
    layer: str
    fps: float
    dim: int
    normalized: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "model": self.model,
            "layer": self.layer,
            "fps": float(self.fps),
            "dim": int(self.dim),
            "normalized": bool(self.normalized),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AudioFeatureSpec":
        _require_keys(data, ["spec_version", "model", "layer", "fps", "dim", "normalized"], prefix="audio_feature_spec")
        spec = cls(
            spec_version=str(data["spec_version"]),
            model=str(data["model"]),
            layer=str(data["layer"]),
            fps=float(data["fps"]),
            dim=int(data["dim"]),
            normalized=bool(data["normalized"]),
        )
        if spec.spec_version != AUDIO_SPEC_VERSION:
            raise CacheContractError(
                f"Unsupported audio_feature_spec version: {spec.spec_version} (expected {AUDIO_SPEC_VERSION})"
            )
        if spec.fps <= 0 or spec.dim <= 0:
            raise CacheContractError(f"Invalid audio_feature_spec values: fps={spec.fps}, dim={spec.dim}")
        return spec


@dataclass(frozen=True)
class TextTokenSpec:
    spec_version: str
    tokenizer_name: str
    lexical_model: str
    lexical_dim: int
    global_text_dim: int
    global_text_pooling: str
    frame_aligned: bool
    word_timestamps_required: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "tokenizer_name": self.tokenizer_name,
            "lexical_model": self.lexical_model,
            "lexical_dim": int(self.lexical_dim),
            "global_text_dim": int(self.global_text_dim),
            "global_text_pooling": self.global_text_pooling,
            "frame_aligned": bool(self.frame_aligned),
            "word_timestamps_required": bool(self.word_timestamps_required),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TextTokenSpec":
        _require_keys(
            data,
            [
                "spec_version",
                "tokenizer_name",
                "lexical_model",
                "lexical_dim",
                "global_text_dim",
                "global_text_pooling",
                "frame_aligned",
                "word_timestamps_required",
            ],
            prefix="text_token_spec",
        )
        spec = cls(
            spec_version=str(data["spec_version"]),
            tokenizer_name=str(data["tokenizer_name"]),
            lexical_model=str(data["lexical_model"]),
            lexical_dim=int(data["lexical_dim"]),
            global_text_dim=int(data["global_text_dim"]),
            global_text_pooling=str(data["global_text_pooling"]),
            frame_aligned=bool(data["frame_aligned"]),
            word_timestamps_required=bool(data["word_timestamps_required"]),
        )
        if spec.spec_version != TEXT_SPEC_VERSION:
            raise CacheContractError(
                f"Unsupported text_token_spec version: {spec.spec_version} (expected {TEXT_SPEC_VERSION})"
            )
        if spec.lexical_dim <= 0:
            raise CacheContractError(f"Invalid lexical_dim={spec.lexical_dim}")
        if spec.global_text_dim <= 0:
            raise CacheContractError(f"Invalid global_text_dim={spec.global_text_dim}")
        if spec.global_text_pooling != "cls":
            raise CacheContractError(
                f"Unsupported global_text_pooling={spec.global_text_pooling}; expected cls"
            )
        return spec


@dataclass(frozen=True)
class MotionContract:
    layout_version: str
    motion_dim: int
    rot6d_start: int
    contact_indices: Tuple[int, ...]
    foot_names: Tuple[str, ...]
    fps: int
    layout_meta: Dict[str, Any]

    @property
    def feature_layout(self) -> MotionFeatureLayout:
        return MotionFeatureLayout.from_metadata(
            self.layout_meta,
            motion_dim=self.motion_dim,
            rot6d_start=self.rot6d_start,
            contact_indices=self.contact_indices,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layout_version": self.layout_version,
            "motion_dim": int(self.motion_dim),
            "rot6d_start": int(self.rot6d_start),
            "contact_indices": list(self.contact_indices),
            "foot_names": list(self.foot_names),
            "fps": int(self.fps),
            "layout_meta": dict(self.layout_meta),
        }

    @classmethod
    def from_cache_meta(
        cls,
        meta: Mapping[str, Any],
        *,
        motion_dim: int,
        fps: int,
    ) -> "MotionContract":
        layout = MotionFeatureLayout.from_metadata(meta, motion_dim=motion_dim)
        if layout.layout_version != "root_pos_abs_v2":
            raise CacheContractError(
                f"Only root_pos_abs_v2 is supported by the latent backend, got {layout.layout_version}"
            )
        if motion_dim != 465:
            raise CacheContractError(f"Only motion_dim=465 is supported by the latent backend, got {motion_dim}")
        if int(layout.rot6d_start) != 15:
            raise CacheContractError(f"Only rot6d_start=15 is supported by the latent backend, got {layout.rot6d_start}")
        if not layout.contact_indices:
            raise CacheContractError("Motion contract requires explicit contact_indices")
        if not meta.get("joint_names") or not meta.get("all_joint_names"):
            raise CacheContractError("Motion contract requires skeleton joint metadata")
        if meta.get("skeleton_offsets") is None or meta.get("skeleton_parents") is None:
            raise CacheContractError("Motion contract requires skeleton_offsets and skeleton_parents")
        layout_meta = dict(meta)
        layout_meta.update(layout.to_metadata())
        return cls(
            layout_version=layout.layout_version,
            motion_dim=int(motion_dim),
            rot6d_start=int(layout.rot6d_start),
            contact_indices=tuple(int(v) for v in layout.contact_indices),
            foot_names=tuple(str(v) for v in layout.foot_names),
            fps=int(fps),
            layout_meta=layout_meta,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MotionContract":
        _require_keys(
            data,
            ["layout_version", "motion_dim", "rot6d_start", "contact_indices", "foot_names", "fps", "layout_meta"],
            prefix="motion_contract",
        )
        contract = cls(
            layout_version=str(data["layout_version"]),
            motion_dim=int(data["motion_dim"]),
            rot6d_start=int(data["rot6d_start"]),
            contact_indices=tuple(int(v) for v in data["contact_indices"]),
            foot_names=tuple(str(v) for v in data["foot_names"]),
            fps=int(data["fps"]),
            layout_meta=dict(data["layout_meta"]),
        )
        if contract.layout_version != "root_pos_abs_v2":
            raise CacheContractError(f"Unsupported layout_version={contract.layout_version}")
        if contract.motion_dim != 465 or contract.rot6d_start != 15:
            raise CacheContractError(
                f"Unsupported motion contract: motion_dim={contract.motion_dim}, rot6d_start={contract.rot6d_start}"
            )
        _ = contract.feature_layout
        return contract


def validate_sample(
    sample: Mapping[str, Any],
    *,
    motion_contract: MotionContract,
    audio_spec: AudioFeatureSpec,
    text_spec: TextTokenSpec,
    require_word_times: bool = True,
) -> None:
    _require_keys(
        sample,
        [
            "segment_id",
            "motion",
            "audio",
            "text",
            "text_ids",
            "text_mask",
            "global_text",
            "word_times",
            "word_texts",
            "lexical_frame",
            "word_frame",
            "motion_len",
            "audio_len",
        ],
        prefix="sample",
    )

    motion = np.asarray(sample["motion"], dtype=np.float32)
    audio = np.asarray(sample["audio"], dtype=np.float32)
    lexical = np.asarray(sample["lexical_frame"], dtype=np.float32)
    global_text = np.asarray(sample["global_text"], dtype=np.float32)
    word_frame = np.asarray(sample["word_frame"], dtype=np.float32)
    word_times = np.asarray(sample["word_times"], dtype=np.float32)
    word_texts = list(sample["word_texts"])

    if motion.ndim != 2 or motion.shape[1] != motion_contract.motion_dim:
        raise CacheContractError(f"Invalid motion shape {motion.shape}; expected [T,{motion_contract.motion_dim}]")
    if audio.ndim != 2 or audio.shape[1] != audio_spec.dim:
        raise CacheContractError(f"Invalid audio shape {audio.shape}; expected [Ta,{audio_spec.dim}]")
    if lexical.ndim != 2 or lexical.shape != (motion.shape[0], text_spec.lexical_dim):
        raise CacheContractError(
            f"Invalid lexical_frame shape {lexical.shape}; expected {(motion.shape[0], text_spec.lexical_dim)}"
        )
    if global_text.ndim != 1 or global_text.shape[0] != text_spec.global_text_dim:
        raise CacheContractError(
            f"Invalid global_text shape {global_text.shape}; expected {(text_spec.global_text_dim,)}"
        )
    if word_frame.ndim != 2 or word_frame.shape != (motion.shape[0], 5):
        raise CacheContractError(f"Invalid word_frame shape {word_frame.shape}; expected {(motion.shape[0], 5)}")
    if word_times.ndim != 2 or word_times.shape[1] != 2:
        raise CacheContractError(f"Invalid word_times shape {word_times.shape}; expected [W,2]")
    if len(word_texts) != word_times.shape[0]:
        raise CacheContractError(
            f"word_texts length {len(word_texts)} does not match word_times length {word_times.shape[0]}"
        )
    if require_word_times and word_times.shape[0] == 0:
        raise CacheContractError(f"Sample {sample.get('segment_id')} has empty word_times")
    if int(sample["motion_len"]) != motion.shape[0]:
        raise CacheContractError("motion_len does not match motion.shape[0]")
    if int(sample["audio_len"]) != audio.shape[0]:
        raise CacheContractError("audio_len does not match audio.shape[0]")
    if not np.isfinite(motion).all():
        raise CacheContractError(f"Sample {sample.get('segment_id')} motion contains non-finite values")
    if not np.isfinite(audio).all():
        raise CacheContractError(f"Sample {sample.get('segment_id')} audio contains non-finite values")
    if not np.isfinite(global_text).all():
        raise CacheContractError(f"Sample {sample.get('segment_id')} global_text contains non-finite values")


def validate_cache_payload(
    payload: Mapping[str, Any],
    *,
    require_word_times: bool = True,
) -> Tuple[MotionContract, AudioFeatureSpec, TextTokenSpec]:
    _require_keys(
        payload,
        [
            "cache_version",
            "fps",
            "mean",
            "std",
            "audio_mean",
            "audio_std",
            "motion_contract",
            "audio_feature_spec",
            "text_token_spec",
            "segments",
        ],
        prefix="payload",
    )

    cache_version = str(payload["cache_version"])
    if cache_version != CACHE_VERSION:
        raise CacheContractError(f"Unsupported cache_version={cache_version}; expected {CACHE_VERSION}")

    segments = payload["segments"]
    if not isinstance(segments, list) or not segments:
        raise CacheContractError("Payload requires a non-empty 'segments' list")

    motion_contract = MotionContract.from_dict(payload["motion_contract"])
    audio_spec = AudioFeatureSpec.from_dict(payload["audio_feature_spec"])
    text_spec = TextTokenSpec.from_dict(payload["text_token_spec"])

    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    audio_mean = np.asarray(payload["audio_mean"], dtype=np.float32)
    audio_std = np.asarray(payload["audio_std"], dtype=np.float32)

    if mean.shape != (motion_contract.motion_dim,) or std.shape != (motion_contract.motion_dim,):
        raise CacheContractError("mean/std shape does not match motion contract")
    if audio_mean.shape != (audio_spec.dim,) or audio_std.shape != (audio_spec.dim,):
        raise CacheContractError("audio_mean/audio_std shape does not match audio_feature_spec")

    for sample in segments[: min(8, len(segments))]:
        validate_sample(
            sample,
            motion_contract=motion_contract,
            audio_spec=audio_spec,
            text_spec=text_spec,
            require_word_times=require_word_times,
        )

    return motion_contract, audio_spec, text_spec


def load_cache_contract(path: str | Path, *, require_word_times: bool = True) -> Dict[str, Any]:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    motion_contract, audio_spec, text_spec = validate_cache_payload(payload, require_word_times=require_word_times)
    return {
        "payload": payload,
        "motion_contract": motion_contract,
        "audio_feature_spec": audio_spec,
        "text_token_spec": text_spec,
    }


def load_checkpoint_contract(path: str | Path) -> Dict[str, Any]:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    _require_keys(ckpt, ["contract"], prefix="checkpoint")
    contract = ckpt["contract"]
    _require_keys(contract, ["motion_contract", "audio_feature_spec", "text_token_spec"], prefix="checkpoint.contract")
    motion_contract = MotionContract.from_dict(contract["motion_contract"])
    audio_spec = AudioFeatureSpec.from_dict(contract["audio_feature_spec"])
    text_spec = TextTokenSpec.from_dict(contract["text_token_spec"])
    return {
        "checkpoint": ckpt,
        "motion_contract": motion_contract,
        "audio_feature_spec": audio_spec,
        "text_token_spec": text_spec,
    }
