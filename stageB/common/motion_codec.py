from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from stageA.train_stage1_vqvae import (
    DEFAULT_FOOT_CONTACT_KEYWORDS,
    DEFAULT_PART_KEYWORDS,
    BVHSkeleton,
    MotionGlobalAE,
    MotionPartSpec,
    MotionVQVAE,
    _split_csv_keywords,
    build_active_joint_map,
    build_motion_part_spec,
    load_ckpt,
    load_state_dict_compat,
    merge_parts_back_to_full,
    prepare_global_condition_input,
    split_motion_into_parts,
)


def _to_numpy_f32(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


@dataclass
class MotionCodecPart:
    name: str
    ckpt_path: Path
    ckpt: dict
    motion_spec: MotionPartSpec
    model: torch.nn.Module
    mean: np.ndarray
    std: np.ndarray
    block_size: int
    token_stride: int
    use_vq: bool

    def to(self, device: torch.device) -> "MotionCodecPart":
        self.model.to(device)
        return self

    @property
    def keep_dim(self) -> int:
        return int(self.mean.shape[0])

    @property
    def n_codes(self) -> int:
        if not self.use_vq:
            return 0
        return int(self.ckpt.get("n_codes", self.ckpt.get("args", {}).get("n_codes", 0)))

    def prepare_part_motion(self, part_motion) -> torch.Tensor:
        x = torch.as_tensor(_to_numpy_f32(part_motion), dtype=torch.float32, device=next(self.model.parameters()).device)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(f"Expected part motion [T, D] or [B, T, D], got {tuple(x.shape)}")
        return x

    def _normalize(self, part_motion: torch.Tensor) -> torch.Tensor:
        mean_t = torch.as_tensor(self.mean, device=part_motion.device, dtype=part_motion.dtype).view(1, 1, -1)
        std_t = torch.as_tensor(self.std, device=part_motion.device, dtype=part_motion.dtype).view(1, 1, -1)
        x = part_motion
        if self.motion_spec.part == "global":
            x = prepare_global_condition_input(x, self.motion_spec)
        return (x - mean_t) / std_t

    def _denormalize(self, part_motion_norm: torch.Tensor) -> torch.Tensor:
        mean_t = torch.as_tensor(self.mean, device=part_motion_norm.device, dtype=part_motion_norm.dtype).view(1, 1, -1)
        std_t = torch.as_tensor(self.std, device=part_motion_norm.device, dtype=part_motion_norm.dtype).view(1, 1, -1)
        return part_motion_norm * std_t + mean_t

    @torch.no_grad()
    def encode_motion(self, part_motion) -> torch.Tensor:
        if not self.use_vq:
            raise RuntimeError(f"Part '{self.name}' has no VQ codes to encode.")
        self.model.eval()
        x = self.prepare_part_motion(part_motion)
        x_norm = self._normalize(x)
        _, _, codes, _, _ = self.model(x_norm, use_vq=True)
        return codes

    @torch.no_grad()
    def encode_motion_batch(self, part_motion, batch_size: int = 32) -> torch.Tensor:
        if not self.use_vq:
            raise RuntimeError(f"Part '{self.name}' has no VQ codes to encode.")
        self.model.eval()
        x = self.prepare_part_motion(part_motion)
        batch_size = max(1, int(batch_size))
        if x.shape[0] <= batch_size:
            x_norm = self._normalize(x)
            _, _, codes, _, _ = self.model(x_norm, use_vq=True)
            return codes

        out = []
        for start in range(0, x.shape[0], batch_size):
            chunk = x[start : start + batch_size]
            chunk_norm = self._normalize(chunk)
            _, _, codes, _, _ = self.model(chunk_norm, use_vq=True)
            out.append(codes)
        return torch.cat(out, dim=0)

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if not self.use_vq:
            raise RuntimeError(f"Part '{self.name}' has no VQ decoder.")
        if codes.ndim == 1:
            codes = codes.unsqueeze(0)
        codebook = self.model.vq.codebook(codes.long())
        z_q = codebook.permute(0, 2, 1).contiguous()
        x_norm = self.model.dec(z_q).transpose(1, 2).contiguous()
        return self._denormalize(x_norm)

    def soft_decode_logits(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        if not self.use_vq:
            raise RuntimeError(f"Part '{self.name}' has no VQ decoder.")
        probs = torch.softmax(logits / max(float(temperature), 1e-4), dim=-1)
        codebook = self.model.vq.codebook.weight.to(dtype=logits.dtype, device=logits.device)
        z_q = torch.matmul(probs, codebook).transpose(1, 2).contiguous()
        x_norm = self.model.dec(z_q).transpose(1, 2).contiguous()
        return self._denormalize(x_norm)

    def predict_codes(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1)

    def set_decoder_trainable(self, enabled: bool) -> None:
        enabled = bool(enabled)
        for param in self.model.parameters():
            param.requires_grad_(False)
        if enabled:
            for param in self.model.dec.parameters():
                param.requires_grad_(True)

    def decoder_state_dict(self) -> dict:
        return self.model.state_dict()


@dataclass
class MotionCodecBundle:
    ref_bvh: Path
    skel: BVHSkeleton
    active_joint_map: any
    parts: Dict[str, MotionCodecPart]

    @property
    def full_dim(self) -> int:
        return 3 + int(self.active_joint_map.n_active) * 6

    @property
    def vq_parts(self) -> dict[str, MotionCodecPart]:
        return {name: part for name, part in self.parts.items() if part.use_vq}

    @property
    def primary_part_names(self) -> list[str]:
        return list(self.vq_parts.keys())

    @property
    def token_stride(self) -> int:
        strides = {part.token_stride for part in self.vq_parts.values()}
        if len(strides) != 1:
            raise RuntimeError(f"Stage1 parts expose mismatched token strides: {sorted(strides)}")
        return int(next(iter(strides)))

    def full_motion_to_parts(self, full_motion: np.ndarray) -> dict[str, np.ndarray]:
        return {
            name: split_motion_into_parts(full_motion, part.motion_spec).astype(np.float32)
            for name, part in self.parts.items()
        }

    def encode_full_motion(self, full_motion: np.ndarray) -> dict[str, torch.Tensor]:
        part_motions = self.full_motion_to_parts(full_motion)
        return {
            name: self.parts[name].encode_motion(part_motions[name])
            for name in self.primary_part_names
        }

    def merge_decoded_parts(
        self,
        decoded_parts: dict[str, np.ndarray | torch.Tensor],
        *,
        gt_full: Optional[np.ndarray | torch.Tensor] = None,
    ) -> np.ndarray | torch.Tensor:
        out = gt_full
        for name in self.primary_part_names:
            if name not in decoded_parts:
                continue
            out = merge_parts_back_to_full(decoded_parts[name], self.parts[name].motion_spec, gt_full=out)
        if out is None:
            raise RuntimeError("Cannot merge parts without decoded inputs.")
        return out

    def apply_stage1_overrides(self, overrides: Optional[dict[str, dict]]) -> None:
        if not overrides:
            return
        for name, state_dict in overrides.items():
            if name in self.parts and state_dict is not None:
                load_state_dict_compat(self.parts[name].model, state_dict, strict=False)

    def set_decoder_trainable(self, enabled: bool) -> None:
        for part in self.vq_parts.values():
            part.set_decoder_trainable(enabled)

    def iter_trainable_decoder_parameters(self) -> Iterable[torch.nn.Parameter]:
        for part in self.vq_parts.values():
            for param in part.model.parameters():
                if param.requires_grad:
                    yield param

    def describe(self) -> dict:
        return {
            "ref_bvh": str(self.ref_bvh),
            "token_stride": self.token_stride,
            "parts": {
                name: {
                    "ckpt_path": str(part.ckpt_path),
                    "part": part.motion_spec.part,
                    "keep_dim": part.keep_dim,
                    "token_stride": part.token_stride,
                    "block_size": part.block_size,
                    "n_codes": part.n_codes,
                    "use_vq": part.use_vq,
                }
                for name, part in self.parts.items()
            },
        }


def _build_part(
    name: str,
    ckpt_path: Path,
    skel: BVHSkeleton,
    device: torch.device,
) -> MotionCodecPart:
    ckpt = load_ckpt(ckpt_path)
    part_name = ckpt.get("part") or ("full" if name == "motion" else name)
    part_keywords = ckpt.get("part_keywords") or DEFAULT_PART_KEYWORDS
    foot_contact_keywords = ckpt.get("foot_contact_keywords") or DEFAULT_FOOT_CONTACT_KEYWORDS

    motion_spec = build_motion_part_spec(
        skel,
        part=part_name,
        drop_root_pos=bool(ckpt.get("drop_root_pos", False)),
        lower_include_root=bool(ckpt.get("lower_include_root", False)),
        lower_include_foot_contact=bool(ckpt.get("lower_include_foot_contact", False)),
        part_keywords=part_keywords,
        foot_contact_keywords=foot_contact_keywords,
    )

    model_arch = ckpt.get("model_arch")
    if model_arch is None and part_name == "global" and not bool(ckpt.get("use_vq", True)):
        model_arch = "motion_global_ae"

    keep_dim = int(ckpt["keep_dim"])
    if keep_dim != int(motion_spec.model_dim):
        raise RuntimeError(
            f"keep_dim mismatch for {name}: ckpt={keep_dim}, motion_spec={motion_spec.model_dim}, ckpt={ckpt_path}"
        )

    if model_arch == "motion_global_ae":
        model = MotionGlobalAE(
            motion_dim=keep_dim,
            hidden=int(ckpt.get("hidden", ckpt.get("args", {}).get("hidden", 256))),
            n_layers=int(ckpt.get("global_ae_layers", ckpt.get("args", {}).get("global_ae_layers", 4))),
        )
    else:
        model = MotionVQVAE(
            motion_dim=keep_dim,
            hidden=int(ckpt.get("hidden", ckpt.get("args", {}).get("hidden", 512))),
            code_dim=int(ckpt.get("code_dim", ckpt.get("args", {}).get("code_dim", 256))),
            n_codes=int(ckpt.get("n_codes", ckpt.get("args", {}).get("n_codes", 1024))),
            beta=float(ckpt.get("beta", ckpt.get("args", {}).get("beta", 0.25))),
            ema_decay=float(ckpt.get("ema_decay", ckpt.get("args", {}).get("ema_decay", 0.99))),
            ema_eps=float(ckpt.get("ema_eps", ckpt.get("args", {}).get("ema_eps", 1e-5))),
            n_downsample=int(ckpt.get("n_downsample", ckpt.get("args", {}).get("n_downsample", 2))),
            vq_usage_entropy_w=float(ckpt.get("vq_usage_entropy_w", 0.0)),
            vq_usage_temp=float(ckpt.get("vq_usage_temp", 0.5)),
            vq_revive_threshold=float(ckpt.get("vq_revive_threshold", 1.0)),
        )

    load_state_dict_compat(model, ckpt["model"], strict=False)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    mean = _to_numpy_f32(ckpt["mean"])
    std = np.maximum(_to_numpy_f32(ckpt["std"]), 1e-6)

    token_stride = int(ckpt.get("token_stride", 1 if model_arch == "motion_global_ae" else 2 ** int(ckpt.get("n_downsample", 2))))
    return MotionCodecPart(
        name=name,
        ckpt_path=ckpt_path,
        ckpt=ckpt,
        motion_spec=motion_spec,
        model=model,
        mean=mean,
        std=std,
        block_size=int(ckpt.get("block_size", 64)),
        token_stride=token_stride,
        use_vq=bool(ckpt.get("use_vq", part_name != "global")),
    )


def load_motion_codec_bundle(
    *,
    device: torch.device,
    ref_bvh: Optional[Path] = None,
    stage1_ckpt: Optional[Path] = None,
    upper_ckpt: Optional[Path] = None,
    hand_ckpt: Optional[Path] = None,
    lower_ckpt: Optional[Path] = None,
    global_ckpt: Optional[Path] = None,
) -> MotionCodecBundle:
    path_map: dict[str, Path] = {}
    if stage1_ckpt is not None:
        path_map["motion"] = Path(stage1_ckpt)
    else:
        for name, p in {
            "upper": upper_ckpt,
            "hand": hand_ckpt,
            "lower": lower_ckpt,
            "global": global_ckpt,
        }.items():
            if p is not None:
                path_map[name] = Path(p)

    if not path_map:
        raise ValueError("No Stage1 checkpoints provided for MotionCodecBundle.")

    first_ckpt_path = next(iter(path_map.values()))
    first_ckpt = load_ckpt(first_ckpt_path)
    resolved_ref_bvh = Path(ref_bvh or first_ckpt.get("ref_bvh") or "")
    if not resolved_ref_bvh.is_absolute():
        repo_root = Path(__file__).resolve().parents[1]
        for cand in (Path.cwd() / resolved_ref_bvh, repo_root / resolved_ref_bvh):
            if cand.exists():
                resolved_ref_bvh = cand
                break
    if not resolved_ref_bvh.exists():
        raise FileNotFoundError(
            f"Failed to resolve reference BVH for Stage1 bundle. Got ref_bvh={resolved_ref_bvh}"
        )

    skel = BVHSkeleton.from_bvh(resolved_ref_bvh)
    active_joint_map = build_active_joint_map(skel)
    parts = {name: _build_part(name, path, skel, device) for name, path in path_map.items()}
    return MotionCodecBundle(
        ref_bvh=resolved_ref_bvh,
        skel=skel,
        active_joint_map=active_joint_map,
        parts=parts,
    )
