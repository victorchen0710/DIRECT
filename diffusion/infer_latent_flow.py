import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusion.latent.bvh import count_rot_joints_in_bvh, decode_motion_to_bvh, read_ref_root_first_frame_xyz, save_bvh_remapped
from diffusion.latent.contracts import load_checkpoint_contract
from diffusion.latent.models import LatentRectifiedFlowTransformer, MotionRefiner, MotionVAE
from diffusion.latent.parts import resolve_part_feature_indices, resolve_joint_rot_feature_indices
from diffusion.latent.postprocess import apply_temporal_smoothing, blend_motion_features, blend_motion_parts
from diffusion.latent.runtime import build_segment_condition_frames, load_or_create_whisper_segments, load_w2v2_feature, merge_transcript_segments


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _apply_ema_if_available(model: torch.nn.Module, ckpt: Dict[str, Any], use_ema: bool) -> None:
    if not use_ema:
        model.load_state_dict(ckpt["model"], strict=True)
        return
    ema = ckpt.get("ema")
    if not isinstance(ema, dict) or "shadow" not in ema:
        model.load_state_dict(ckpt["model"], strict=True)
        return
    state = dict(model.state_dict())
    for key, value in ema["shadow"].items():
        if key in state:
            state[key] = value.to(dtype=state[key].dtype)
    model.load_state_dict(state, strict=False)


def _slice_audio_by_time(audio_feat: np.ndarray, audio_fps: float, start_s: float, end_s: float) -> np.ndarray:
    start = max(0, int(math.floor(start_s * audio_fps)))
    end = max(start + 1, int(math.ceil(end_s * audio_fps)))
    end = min(end, audio_feat.shape[0])
    return audio_feat[start:end].astype(np.float32, copy=False)


def _window_starts(total_frames: int, chunk_frames: int, hop_frames: int) -> List[int]:
    if total_frames <= chunk_frames:
        return [0]
    starts = list(range(0, total_frames - chunk_frames + 1, max(1, hop_frames)))
    tail = total_frames - chunk_frames
    if starts[-1] != tail:
        starts.append(tail)
    return starts


def _predict_velocity(
    model: LatentRectifiedFlowTransformer,
    *,
    cfg_scale: float,
    cfg_audio_scale: Optional[float],
    cfg_text_scale: Optional[float],
    cfg_word_scale: Optional[float],
    cfg_global_text_scale: Optional[float],
    x: torch.Tensor,
    t_cur: torch.Tensor,
    latent_mask: torch.Tensor,
    audio: torch.Tensor,
    lexical: torch.Tensor,
    global_text: torch.Tensor,
    word_frame: torch.Tensor,
    prefix_latent: Optional[torch.Tensor],
    prefix_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    text_scale = float(cfg_scale if cfg_text_scale is None else cfg_text_scale)
    audio_scale = float(cfg_scale if cfg_audio_scale is None else cfg_audio_scale)
    word_scale = float(text_scale if cfg_word_scale is None else cfg_word_scale)
    global_text_scale = float(text_scale if cfg_global_text_scale is None else cfg_global_text_scale)
    use_guidance = (
        (float(cfg_scale) != 1.0)
        or (audio_scale != 1.0)
        or (text_scale != 1.0)
        or (word_scale != text_scale)
        or (global_text_scale != text_scale)
    )
    if not use_guidance:
        return model(
            x,
            t_cur,
            latent_mask,
            audio=audio,
            lexical_frame=lexical,
            global_text=global_text,
            word_frame=word_frame,
            prefix_latent=prefix_latent,
            prefix_mask=prefix_mask,
        )

    v_uncond = model(
        x,
        t_cur,
        latent_mask,
        audio=audio,
        lexical_frame=lexical,
        global_text=global_text,
        word_frame=word_frame,
        prefix_latent=prefix_latent,
        prefix_mask=prefix_mask,
        force_drop_all=True,
    )
    v_full = model(
        x,
        t_cur,
        latent_mask,
        audio=audio,
        lexical_frame=lexical,
        global_text=global_text,
        word_frame=word_frame,
        prefix_latent=prefix_latent,
        prefix_mask=prefix_mask,
    )
    velocity = v_uncond + float(cfg_scale) * (v_full - v_uncond)
    if audio_scale != float(cfg_scale):
        v_no_audio = model(
            x,
            t_cur,
            latent_mask,
            audio=audio,
            lexical_frame=lexical,
            global_text=global_text,
            word_frame=word_frame,
            prefix_latent=prefix_latent,
            prefix_mask=prefix_mask,
            force_drop_audio=True,
        )
        velocity = velocity + (audio_scale - float(cfg_scale)) * (v_full - v_no_audio)
    if text_scale != float(cfg_scale):
        v_no_text = model(
            x,
            t_cur,
            latent_mask,
            audio=audio,
            lexical_frame=lexical,
            global_text=global_text,
            word_frame=word_frame,
            prefix_latent=prefix_latent,
            prefix_mask=prefix_mask,
            force_drop_text=True,
            force_drop_word=True,
            force_drop_global_text=True,
        )
        velocity = velocity + (text_scale - float(cfg_scale)) * (v_full - v_no_text)
    if word_scale != text_scale:
        v_no_word = model(
            x,
            t_cur,
            latent_mask,
            audio=audio,
            lexical_frame=lexical,
            global_text=global_text,
            word_frame=word_frame,
            prefix_latent=prefix_latent,
            prefix_mask=prefix_mask,
            force_drop_word=True,
        )
        velocity = velocity + (word_scale - text_scale) * (v_full - v_no_word)
    if global_text_scale != text_scale:
        v_no_global = model(
            x,
            t_cur,
            latent_mask,
            audio=audio,
            lexical_frame=lexical,
            global_text=global_text,
            word_frame=word_frame,
            prefix_latent=prefix_latent,
            prefix_mask=prefix_mask,
            force_drop_global_text=True,
        )
        velocity = velocity + (global_text_scale - text_scale) * (v_full - v_no_global)
    return velocity


@torch.no_grad()
def sample_latent_chunk(
    model: LatentRectifiedFlowTransformer,
    *,
    latent_len: int,
    latent_dim: int,
    num_steps: int,
    cfg_scale: float,
    cfg_audio_scale: Optional[float],
    cfg_text_scale: Optional[float],
    cfg_word_scale: Optional[float],
    cfg_global_text_scale: Optional[float],
    solver: str,
    audio: torch.Tensor,
    lexical: torch.Tensor,
    global_text: torch.Tensor,
    word_frame: torch.Tensor,
    prefix_latent: Optional[torch.Tensor],
    prefix_mask: Optional[torch.Tensor],
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    latent_mask = torch.ones((1, latent_len), device=device, dtype=torch.bool)
    x = torch.randn((1, latent_len, latent_dim), device=device, generator=generator, dtype=torch.float32)
    schedule = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=torch.float32)
    for idx in range(num_steps):
        t_cur = schedule[idx].view(1)
        velocity = _predict_velocity(
            model,
            cfg_scale=cfg_scale,
            cfg_audio_scale=cfg_audio_scale,
            cfg_text_scale=cfg_text_scale,
            cfg_word_scale=cfg_word_scale,
            cfg_global_text_scale=cfg_global_text_scale,
            x=x,
            t_cur=t_cur,
            latent_mask=latent_mask,
            audio=audio,
            lexical=lexical,
            global_text=global_text,
            word_frame=word_frame,
            prefix_latent=prefix_latent,
            prefix_mask=prefix_mask,
        )
        dt = float(schedule[idx + 1] - schedule[idx])
        if solver == "heun":
            x_euler = x + dt * velocity
            velocity_next = _predict_velocity(
                model,
                cfg_scale=cfg_scale,
                cfg_audio_scale=cfg_audio_scale,
                cfg_text_scale=cfg_text_scale,
                cfg_word_scale=cfg_word_scale,
                cfg_global_text_scale=cfg_global_text_scale,
                x=x_euler,
                t_cur=schedule[idx + 1].view(1),
                latent_mask=latent_mask,
                audio=audio,
                lexical=lexical,
                global_text=global_text,
                word_frame=word_frame,
                prefix_latent=prefix_latent,
                prefix_mask=prefix_mask,
            )
            x = x + dt * 0.5 * (velocity + velocity_next)
        else:
            x = x + dt * velocity
    return x


def load_vae(vae_ckpt: str | Path, device: torch.device) -> tuple[MotionVAE, Dict[str, Any]]:
    loaded = load_checkpoint_contract(vae_ckpt)
    ckpt = loaded["checkpoint"]
    cfg = dict(ckpt["vae_spec"])
    motion_contract = loaded["motion_contract"]
    part_layout = str(cfg.get("part_layout", "legacy_root_v1" if cfg.get("part_aware", False) else "legacy_root_v1"))
    part_feature_indices = resolve_part_feature_indices(motion_contract, part_layout=part_layout)
    model = MotionVAE(
        motion_dim=motion_contract.motion_dim,
        latent_dim=int(cfg["latent_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_encoder_layers=int(cfg["num_encoder_layers"]),
        num_decoder_layers=int(cfg["num_decoder_layers"]),
        num_heads=int(cfg["num_heads"]),
        latent_stride=int(cfg["latent_stride"]),
        dropout=float(cfg["dropout"]),
        deterministic_ae=bool(cfg.get("deterministic_ae", False)),
        part_aware=bool(cfg.get("part_aware", False)),
        joint_names=list(motion_contract.layout_meta.get("joint_names") or []),
        rot6d_start=int(cfg.get("rot6d_start", motion_contract.rot6d_start)),
        part_layout=part_layout,
        part_feature_indices=part_feature_indices,
        slot_packed_latent=bool(cfg.get("slot_packed_latent", False)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, loaded


def load_flow_bundle(
    checkpoint: str | Path,
    *,
    vae_ckpt: Optional[str | Path],
    device: torch.device,
    use_ema: bool,
) -> Dict[str, Any]:
    flow_loaded = load_checkpoint_contract(checkpoint)
    flow_ckpt = flow_loaded["checkpoint"]
    resolved_vae_ckpt = str(vae_ckpt or Path(checkpoint).with_name("motion_vae.pt"))
    vae, vae_loaded = load_vae(resolved_vae_ckpt, device)

    if flow_loaded["motion_contract"].to_dict() != vae_loaded["motion_contract"].to_dict():
        raise RuntimeError("Flow checkpoint contract does not match VAE checkpoint contract")
    if flow_loaded["audio_feature_spec"].to_dict() != vae_loaded["audio_feature_spec"].to_dict():
        raise RuntimeError("Flow and VAE audio specs do not match")
    if flow_loaded["text_token_spec"].to_dict() != vae_loaded["text_token_spec"].to_dict():
        raise RuntimeError("Flow and VAE text specs do not match")

    motion_contract = flow_loaded["motion_contract"]
    audio_spec = flow_loaded["audio_feature_spec"]
    text_spec = flow_loaded["text_token_spec"]
    vae_payload = vae_loaded["checkpoint"]
    flow_cfg = dict(flow_ckpt["flow_spec"])
    model = LatentRectifiedFlowTransformer(
        latent_dim=int(flow_cfg["latent_dim"]),
        audio_dim=audio_spec.dim,
        lexical_dim=text_spec.lexical_dim,
        global_text_dim=int(flow_cfg.get("global_text_dim", text_spec.global_text_dim)),
        hidden_dim=int(flow_cfg["hidden_dim"]),
        num_layers=int(flow_cfg["num_layers"]),
        num_double_layers=int(flow_cfg.get("num_double_layers", 2)),
        token_refiner_layers=int(flow_cfg.get("token_refiner_layers", 2)),
        num_heads=int(flow_cfg["num_heads"]),
        dropout=float(flow_cfg["dropout"]),
        audio_drop_prob=float(flow_cfg.get("audio_drop_prob", 0.1)),
        text_drop_prob=float(flow_cfg.get("text_drop_prob", 0.1)),
        word_drop_prob=float(flow_cfg.get("word_drop_prob", 0.1)),
        global_text_drop_prob=float(flow_cfg.get("global_text_drop_prob", 0.1)),
        local_attn_window=int(flow_cfg.get("local_attn_window", 15)),
        use_rope=bool(flow_cfg.get("use_rope", True)),
        slot_tokenized=bool(flow_cfg.get("slot_tokenized_flow", False)),
        slot_part_names=list(flow_cfg.get("slot_part_names") or vae_payload.get("vae_spec", {}).get("slot_part_names") or ()),
    ).to(device)
    _apply_ema_if_available(model, flow_ckpt, use_ema=use_ema)
    model.eval()

    motion_refiner = None
    if bool(flow_cfg.get("use_motion_refiner", False)):
        motion_refiner = MotionRefiner(
            motion_dim=motion_contract.motion_dim,
            audio_dim=audio_spec.dim,
            lexical_dim=text_spec.lexical_dim,
            global_text_dim=int(flow_cfg.get("global_text_dim", text_spec.global_text_dim)),
            hidden_dim=int(flow_cfg.get("refiner_hidden_dim", 256)),
            num_layers=int(flow_cfg.get("refiner_layers", 3)),
            num_heads=int(flow_cfg.get("refiner_heads", 4)),
            dropout=float(flow_cfg.get("refiner_dropout", 0.0)),
        ).to(device)
        refiner_state = flow_ckpt.get("motion_refiner")
        if isinstance(refiner_state, dict):
            motion_refiner.load_state_dict(refiner_state, strict=True)
        motion_refiner.eval()

    motion_mean = torch.from_numpy(np.asarray(vae_payload["motion_mean"], dtype=np.float32)).to(device)
    motion_std = torch.from_numpy(np.asarray(vae_payload["motion_std"], dtype=np.float32)).to(device)
    return {
        "checkpoint_path": str(Path(checkpoint).resolve()),
        "vae_ckpt_path": str(Path(resolved_vae_ckpt).resolve()),
        "flow_loaded": flow_loaded,
        "flow_ckpt": flow_ckpt,
        "vae_loaded": vae_loaded,
        "motion_contract": motion_contract,
        "audio_spec": audio_spec,
        "text_token_spec": text_spec,
        "vae_payload": vae_payload,
        "part_layout": str(vae_payload.get("vae_spec", {}).get("part_layout", "root_upper_hand_lower_slots_v3")),
        "flow_cfg": flow_cfg,
        "vae": vae,
        "model": model,
        "motion_refiner": motion_refiner,
        "motion_mean": motion_mean,
        "motion_std": motion_std,
    }


@torch.no_grad()
def sample_motion_chunk_from_bundle(
    bundle: Dict[str, Any],
    *,
    frame_len: int,
    num_steps: int,
    cfg_scale: float,
    cfg_audio_scale: Optional[float],
    cfg_text_scale: Optional[float],
    cfg_word_scale: Optional[float],
    cfg_global_text_scale: Optional[float],
    solver: str,
    audio_t: torch.Tensor,
    lexical_t: torch.Tensor,
    global_text_t: torch.Tensor,
    word_t: torch.Tensor,
    prefix_latent: Optional[torch.Tensor],
    prefix_mask: Optional[torch.Tensor],
    device: torch.device,
    generator: torch.Generator,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    vae = bundle["vae"]
    model = bundle["model"]
    motion_refiner = bundle["motion_refiner"]
    latent_len = int(math.ceil(frame_len / float(vae.latent_stride)))
    chunk_latents = sample_latent_chunk(
        model,
        latent_len=latent_len,
        latent_dim=vae.latent_dim,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
        cfg_audio_scale=cfg_audio_scale,
        cfg_text_scale=cfg_text_scale,
        cfg_word_scale=cfg_word_scale,
        cfg_global_text_scale=cfg_global_text_scale,
        solver=solver,
        audio=audio_t,
        lexical=lexical_t,
        global_text=global_text_t,
        word_frame=word_t,
        prefix_latent=prefix_latent,
        prefix_mask=prefix_mask,
        device=device,
        generator=generator,
    )
    latent_mask = torch.ones((1, latent_len), device=device, dtype=torch.bool)
    motion_norm = vae.decode(chunk_latents, latent_mask, target_len=frame_len)
    if motion_refiner is not None:
        motion_mask = torch.ones((1, frame_len), device=device, dtype=torch.bool)
        motion_norm = motion_refiner(
            motion_norm,
            motion_mask,
            audio=audio_t,
            lexical_frame=lexical_t,
            word_frame=word_t,
            global_text=global_text_t,
        )
    motion_denorm = motion_norm * bundle["motion_std"].view(1, 1, -1) + bundle["motion_mean"].view(1, 1, -1)
    motion_np = motion_denorm[0].detach().cpu().numpy().astype(np.float32)
    return motion_np, chunk_latents.detach(), torch.ones((1, chunk_latents.shape[1]), device=device, dtype=torch.bool)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="gesture_flow checkpoint")
    parser.add_argument("--vae_ckpt", type=str, default=None)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--w2v2_npz", type=str, required=True)
    parser.add_argument("--ref_bvh", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--out_npy", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--cfg_audio_scale", type=float, default=None)
    parser.add_argument("--cfg_text_scale", type=float, default=None)
    parser.add_argument("--cfg_word_scale", type=float, default=None)
    parser.add_argument("--cfg_global_text_scale", type=float, default=None)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--preview_steps", type=int, default=16)
    parser.add_argument("--solver", choices=["euler", "heun"], default="heun")
    parser.add_argument("--chunk_frames", type=int, default=240)
    parser.add_argument("--prefix_frames", type=int, default=48)
    parser.add_argument("--smooth_root_window", type=int, default=0)
    parser.add_argument("--smooth_upper_hand_rot_window", type=int, default=0)
    parser.add_argument("--smooth_all_rot_window", type=int, default=0)
    parser.add_argument("--hybrid_secondary_checkpoint", type=str, default=None)
    parser.add_argument("--hybrid_secondary_vae_ckpt", type=str, default=None)
    parser.add_argument("--hybrid_secondary_cfg_scale", type=float, default=None)
    parser.add_argument("--hybrid_secondary_cfg_audio_scale", type=float, default=None)
    parser.add_argument("--hybrid_secondary_cfg_text_scale", type=float, default=None)
    parser.add_argument("--hybrid_secondary_cfg_word_scale", type=float, default=None)
    parser.add_argument("--hybrid_secondary_cfg_global_text_scale", type=float, default=None)
    parser.add_argument("--hybrid_secondary_parts", type=str, default="upper,hand")
    parser.add_argument("--hybrid_secondary_joint_keywords", type=str, default=None)
    parser.add_argument("--hybrid_secondary_exclude_joint_keywords", type=str, default=None)
    parser.add_argument("--hybrid_secondary_seed_offset", type=int, default=9973)
    parser.add_argument("--hybrid_secondary_smooth_root_window", type=int, default=0)
    parser.add_argument("--hybrid_secondary_smooth_upper_hand_rot_window", type=int, default=0)
    parser.add_argument("--hybrid_secondary_smooth_all_rot_window", type=int, default=0)
    parser.add_argument("--max_seg_dur", type=float, default=8.0)
    parser.add_argument("--gap_th", type=float, default=0.35)
    parser.add_argument("--whisper_model", "--whisper", dest="whisper_model", type=str, default="medium")
    parser.add_argument("--whisper_cache_dir", type=str, default="diffusion/whisper_cache_v2")
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--no_word_ts", action="store_true")
    parser.add_argument("--bert_model_dir", type=str, default="models/bert")
    parser.add_argument("--bert_device", type=str, default="cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--root_xz_from_ref", action="store_true")
    parser.add_argument("--no_ema", action="store_true")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    primary_bundle = load_flow_bundle(
        args.checkpoint,
        vae_ckpt=args.vae_ckpt,
        device=device,
        use_ema=not args.no_ema,
    )
    motion_contract = primary_bundle["motion_contract"]
    audio_spec = primary_bundle["audio_spec"]
    text_spec = primary_bundle["text_token_spec"]
    vae_payload = primary_bundle["vae_payload"]
    vae = primary_bundle["vae"]

    secondary_bundle = None
    hybrid_parts = tuple(part.strip() for part in str(args.hybrid_secondary_parts).split(",") if part.strip())
    hybrid_joint_keywords = tuple(part.strip() for part in str(args.hybrid_secondary_joint_keywords or "").split(",") if part.strip())
    hybrid_exclude_joint_keywords = tuple(part.strip() for part in str(args.hybrid_secondary_exclude_joint_keywords or "").split(",") if part.strip())
    hybrid_feature_indices = None
    if args.hybrid_secondary_checkpoint:
        secondary_bundle = load_flow_bundle(
            args.hybrid_secondary_checkpoint,
            vae_ckpt=args.hybrid_secondary_vae_ckpt,
            device=device,
            use_ema=not args.no_ema,
        )
        if secondary_bundle["motion_contract"].to_dict() != motion_contract.to_dict():
            raise RuntimeError("Hybrid secondary checkpoint contract does not match primary contract")
        if secondary_bundle["audio_spec"].to_dict() != audio_spec.to_dict():
            raise RuntimeError("Hybrid secondary audio_feature_spec does not match primary")
        if secondary_bundle["text_token_spec"].to_dict() != text_spec.to_dict():
            raise RuntimeError("Hybrid secondary text_token_spec does not match primary")
        if hybrid_joint_keywords:
            hybrid_feature_indices = np.asarray(
                resolve_joint_rot_feature_indices(
                    motion_contract,
                    include_keywords=hybrid_joint_keywords,
                    exclude_keywords=hybrid_exclude_joint_keywords,
                ),
                dtype=np.int64,
            )

    audio_feat, audio_fps = load_w2v2_feature(
        args.w2v2_npz,
        expected_dim=audio_spec.dim,
        expected_fps=audio_spec.fps,
    )
    audio_mean = np.asarray(vae_payload["audio_mean"], dtype=np.float32)
    audio_std = np.asarray(vae_payload["audio_std"], dtype=np.float32)
    audio_std = np.where(audio_std < 1e-6, 1.0, audio_std)
    audio_feat = (audio_feat - audio_mean) / audio_std

    whisper_result = load_or_create_whisper_segments(
        audio_path=args.audio,
        cache_dir=args.whisper_cache_dir,
        model_name=args.whisper_model,
        language=args.language,
        with_word_ts=not args.no_word_ts,
    )
    segments = merge_transcript_segments(
        whisper_result,
        max_seg_dur=args.max_seg_dur,
        gap_th=args.gap_th,
    )
    if not segments:
        raise RuntimeError("No transcript segments available for inference")

    ref_root = read_ref_root_first_frame_xyz(args.ref_bvh) if args.root_xz_from_ref else None
    if count_rot_joints_in_bvh(args.ref_bvh) != (motion_contract.motion_dim - motion_contract.rot6d_start) // 6:
        raise RuntimeError("Reference BVH joint count does not match motion contract")

    from stageB.common.text_bert import load_bert_word_encoder

    bert_encoder = load_bert_word_encoder(args.bert_model_dir, device=args.bert_device)
    full_motion_chunks = []
    full_motion_chunks_secondary = [] if secondary_bundle is not None else None
    prev_prefix_latent = None
    prev_prefix_mask = None
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))
    prev_prefix_latent_secondary = None
    prev_prefix_mask_secondary = None
    secondary_generator = None
    if secondary_bundle is not None:
        secondary_generator = torch.Generator(device=device)
        secondary_generator.manual_seed(int(args.seed) + int(args.hybrid_secondary_seed_offset))

    for seg in segments:
        cond = build_segment_condition_frames(
            text=str(seg["text"]),
            words=seg.get("words", []),
            segment_start=float(seg["start"]),
            segment_end=float(seg["end"]),
            fps=motion_contract.fps,
            bert_model_dir=args.bert_model_dir,
            bert_device=args.bert_device,
            bert_encoder=bert_encoder,
        )
        if cond["word_times"].shape[0] == 0:
            raise RuntimeError(f"Segment '{seg['text']}' has no word timestamps; strict latent inference refuses fallback")

        segment_frames = int(cond["frames"])
        segment_audio = _slice_audio_by_time(audio_feat, audio_fps, float(seg["start"]), float(seg["end"]))
        starts = _window_starts(segment_frames, args.chunk_frames, max(1, args.chunk_frames - args.prefix_frames))
        for chunk_idx, frame_start in enumerate(starts):
            frame_end = min(segment_frames, frame_start + args.chunk_frames)
            frame_len = int(frame_end - frame_start)
            chunk_audio = _slice_audio_by_time(
                segment_audio,
                audio_fps,
                frame_start / float(motion_contract.fps),
                frame_end / float(motion_contract.fps),
            )
            lexical = cond["lexical_frame"][frame_start:frame_end]
            global_text = cond["global_text"]
            word_frame = cond["word_frame"][frame_start:frame_end]

            audio_t = torch.from_numpy(chunk_audio).float().unsqueeze(0).to(device)
            lexical_t = torch.from_numpy(lexical).float().unsqueeze(0).to(device)
            global_text_t = torch.from_numpy(np.asarray(global_text, dtype=np.float32)).float().unsqueeze(0).to(device)
            word_t = torch.from_numpy(word_frame).float().unsqueeze(0).to(device)
            motion_np, next_prefix_latent, next_prefix_mask = sample_motion_chunk_from_bundle(
                primary_bundle,
                frame_len=frame_len,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                cfg_audio_scale=args.cfg_audio_scale,
                cfg_text_scale=args.cfg_text_scale,
                cfg_word_scale=args.cfg_word_scale,
                cfg_global_text_scale=args.cfg_global_text_scale,
                solver=args.solver,
                audio_t=audio_t,
                lexical_t=lexical_t,
                global_text_t=global_text_t,
                word_t=word_t,
                prefix_latent=prev_prefix_latent,
                prefix_mask=prev_prefix_mask,
                device=device,
                generator=generator,
            )
            prefix_tokens = min(next_prefix_latent.shape[1], int(math.ceil(args.prefix_frames / float(vae.latent_stride))))
            prev_prefix_latent = next_prefix_latent[:, -prefix_tokens:].detach()
            prev_prefix_mask = next_prefix_mask[:, -prefix_tokens:].detach()

            if full_motion_chunks and (frame_start > 0 or chunk_idx > 0):
                full_motion_chunks.append(motion_np[args.prefix_frames :])
            else:
                full_motion_chunks.append(motion_np)

            if secondary_bundle is not None:
                secondary_motion_np, next_prefix_latent_secondary, next_prefix_mask_secondary = sample_motion_chunk_from_bundle(
                    secondary_bundle,
                    frame_len=frame_len,
                    num_steps=args.num_steps,
                    cfg_scale=(args.cfg_scale if args.hybrid_secondary_cfg_scale is None else args.hybrid_secondary_cfg_scale),
                    cfg_audio_scale=(args.cfg_audio_scale if args.hybrid_secondary_cfg_audio_scale is None else args.hybrid_secondary_cfg_audio_scale),
                    cfg_text_scale=(args.cfg_text_scale if args.hybrid_secondary_cfg_text_scale is None else args.hybrid_secondary_cfg_text_scale),
                    cfg_word_scale=(args.cfg_word_scale if args.hybrid_secondary_cfg_word_scale is None else args.hybrid_secondary_cfg_word_scale),
                    cfg_global_text_scale=(args.cfg_global_text_scale if args.hybrid_secondary_cfg_global_text_scale is None else args.hybrid_secondary_cfg_global_text_scale),
                    solver=args.solver,
                    audio_t=audio_t,
                    lexical_t=lexical_t,
                    global_text_t=global_text_t,
                    word_t=word_t,
                    prefix_latent=prev_prefix_latent_secondary,
                    prefix_mask=prev_prefix_mask_secondary,
                    device=device,
                    generator=secondary_generator,
                )
                secondary_vae = secondary_bundle["vae"]
                prefix_tokens_secondary = min(next_prefix_latent_secondary.shape[1], int(math.ceil(args.prefix_frames / float(secondary_vae.latent_stride))))
                prev_prefix_latent_secondary = next_prefix_latent_secondary[:, -prefix_tokens_secondary:].detach()
                prev_prefix_mask_secondary = next_prefix_mask_secondary[:, -prefix_tokens_secondary:].detach()
                if full_motion_chunks_secondary and (frame_start > 0 or chunk_idx > 0):
                    full_motion_chunks_secondary.append(secondary_motion_np[args.prefix_frames :])
                else:
                    full_motion_chunks_secondary.append(secondary_motion_np)

    full_motion = np.concatenate(full_motion_chunks, axis=0).astype(np.float32)
    if secondary_bundle is not None:
        secondary_motion = np.concatenate(full_motion_chunks_secondary, axis=0).astype(np.float32)
        secondary_motion = apply_temporal_smoothing(
            secondary_motion,
            motion_contract=motion_contract,
            part_layout=secondary_bundle["part_layout"],
            root_window=args.hybrid_secondary_smooth_root_window,
            upper_hand_rot_window=args.hybrid_secondary_smooth_upper_hand_rot_window,
            all_rot_window=args.hybrid_secondary_smooth_all_rot_window,
        )
        if hybrid_feature_indices is not None:
            full_motion = blend_motion_features(
                full_motion,
                secondary_motion,
                feature_indices=hybrid_feature_indices,
            )
        else:
            full_motion = blend_motion_parts(
                full_motion,
                secondary_motion,
                motion_contract=motion_contract,
                part_layout=secondary_bundle["part_layout"],
                source_parts=hybrid_parts,
            )
    full_motion = apply_temporal_smoothing(
        full_motion,
        motion_contract=motion_contract,
        part_layout=primary_bundle["part_layout"],
        root_window=args.smooth_root_window,
        upper_hand_rot_window=args.smooth_upper_hand_rot_window,
        all_rot_window=args.smooth_all_rot_window,
    )
    full_motion_t = torch.from_numpy(full_motion).float()
    root_pos, euler = decode_motion_to_bvh(
        full_motion_t,
        layout=motion_contract.feature_layout,
        fps=motion_contract.fps,
        motion_dim=motion_contract.motion_dim,
        root_init_xz=None if ref_root is None else (ref_root[0], ref_root[2]),
    )
    ensure_dir(Path(args.out).parent)
    save_bvh_remapped(
        root_pos,
        euler,
        ref_bvh_path=args.ref_bvh,
        output_path=args.out,
        fps=motion_contract.fps,
    )
    if args.out_npy:
        np.save(args.out_npy, full_motion)
    summary = {
        "checkpoint": args.checkpoint,
        "vae_ckpt": primary_bundle["vae_ckpt_path"],
        "cfg_scale": float(args.cfg_scale),
        "cfg_audio_scale": None if args.cfg_audio_scale is None else float(args.cfg_audio_scale),
        "cfg_text_scale": None if args.cfg_text_scale is None else float(args.cfg_text_scale),
        "cfg_word_scale": None if args.cfg_word_scale is None else float(args.cfg_word_scale),
        "cfg_global_text_scale": None if args.cfg_global_text_scale is None else float(args.cfg_global_text_scale),
        "num_steps": int(args.num_steps),
        "solver": str(args.solver),
        "smooth_root_window": int(args.smooth_root_window),
        "smooth_upper_hand_rot_window": int(args.smooth_upper_hand_rot_window),
        "smooth_all_rot_window": int(args.smooth_all_rot_window),
        "segments": int(len(segments)),
        "frames": int(full_motion.shape[0]),
    }
    if secondary_bundle is not None:
        summary["hybrid_secondary_checkpoint"] = secondary_bundle["checkpoint_path"]
        summary["hybrid_secondary_vae_ckpt"] = secondary_bundle["vae_ckpt_path"]
        summary["hybrid_secondary_parts"] = list(hybrid_parts)
        if hybrid_joint_keywords:
            summary["hybrid_secondary_joint_keywords"] = list(hybrid_joint_keywords)
            summary["hybrid_secondary_exclude_joint_keywords"] = list(hybrid_exclude_joint_keywords)
        summary["hybrid_secondary_cfg_scale"] = float(args.cfg_scale if args.hybrid_secondary_cfg_scale is None else args.hybrid_secondary_cfg_scale)
        summary["hybrid_secondary_cfg_audio_scale"] = None if (args.hybrid_secondary_cfg_audio_scale is None and args.cfg_audio_scale is None) else float(args.cfg_audio_scale if args.hybrid_secondary_cfg_audio_scale is None else args.hybrid_secondary_cfg_audio_scale)
        summary["hybrid_secondary_cfg_text_scale"] = None if (args.hybrid_secondary_cfg_text_scale is None and args.cfg_text_scale is None) else float(args.cfg_text_scale if args.hybrid_secondary_cfg_text_scale is None else args.hybrid_secondary_cfg_text_scale)
        summary["hybrid_secondary_cfg_word_scale"] = None if (args.hybrid_secondary_cfg_word_scale is None and args.cfg_word_scale is None) else float(args.cfg_word_scale if args.hybrid_secondary_cfg_word_scale is None else args.hybrid_secondary_cfg_word_scale)
        summary["hybrid_secondary_cfg_global_text_scale"] = None if (args.hybrid_secondary_cfg_global_text_scale is None and args.cfg_global_text_scale is None) else float(args.cfg_global_text_scale if args.hybrid_secondary_cfg_global_text_scale is None else args.hybrid_secondary_cfg_global_text_scale)
        summary["hybrid_secondary_smooth_root_window"] = int(args.hybrid_secondary_smooth_root_window)
        summary["hybrid_secondary_smooth_upper_hand_rot_window"] = int(args.hybrid_secondary_smooth_upper_hand_rot_window)
        summary["hybrid_secondary_smooth_all_rot_window"] = int(args.hybrid_secondary_smooth_all_rot_window)
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
