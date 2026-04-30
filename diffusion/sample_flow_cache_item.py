import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusion.infer_latent_flow import load_flow_bundle, sample_motion_chunk_from_bundle
from diffusion.latent.bvh import decode_motion_to_bvh, save_bvh_remapped
from diffusion.latent.contracts import load_cache_contract, load_checkpoint_contract
from diffusion.latent.models import MotionVAE
from diffusion.latent.parts import resolve_part_feature_indices, resolve_joint_rot_feature_indices
from diffusion.latent.postprocess import apply_temporal_smoothing, blend_motion_features, blend_motion_parts


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _load_vae(vae_ckpt: str | Path, device: torch.device) -> tuple[MotionVAE, dict]:
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


def _apply_ema_if_available(model: torch.nn.Module, ckpt: dict, use_ema: bool) -> None:
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


def _select_segment(segments: list[dict], *, index: Optional[int], segment_id: Optional[str]) -> tuple[int, dict]:
    if segment_id is not None:
        for idx, sample in enumerate(segments):
            if str(sample.get("segment_id")) == str(segment_id):
                return idx, sample
        raise KeyError(f"segment_id not found in cache: {segment_id}")
    idx = int(index or 0)
    if idx < 0 or idx >= len(segments):
        raise IndexError(f"index {idx} out of range for {len(segments)} segments")
    return idx, segments[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="gesture_flow checkpoint")
    parser.add_argument("--vae_ckpt", type=str, default=None)
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--segment_id", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--mode", choices=["flow", "vae_recon"], default="flow")
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--cfg_scale", type=float, default=1.2)
    parser.add_argument("--cfg_audio_scale", type=float, default=None)
    parser.add_argument("--cfg_text_scale", type=float, default=None)
    parser.add_argument("--cfg_word_scale", type=float, default=None)
    parser.add_argument("--cfg_global_text_scale", type=float, default=None)
    parser.add_argument("--solver", choices=["euler", "heun"], default="heun")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_ema", action="store_true")
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
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    cache_loaded = load_cache_contract(args.cache, require_word_times=True)
    payload = cache_loaded["payload"]
    motion_contract = cache_loaded["motion_contract"]
    audio_spec = cache_loaded["audio_feature_spec"]
    text_spec = cache_loaded["text_token_spec"]
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    std[std < 1e-6] = 1.0
    audio_mean = np.asarray(payload["audio_mean"], dtype=np.float32)
    audio_std = np.asarray(payload["audio_std"], dtype=np.float32)
    audio_std[audio_std < 1e-6] = 1.0

    seg_index, sample = _select_segment(payload["segments"], index=args.index, segment_id=args.segment_id)

    if args.mode == "flow" and not args.checkpoint:
        raise RuntimeError("--checkpoint is required in flow mode")
    if args.mode == "vae_recon" and not args.vae_ckpt:
        raise RuntimeError("--vae_ckpt is required in vae_recon mode")

    vae_ckpt = args.vae_ckpt or (str(Path(args.checkpoint).with_name("motion_vae.pt")) if args.checkpoint else None)
    vae, vae_loaded = _load_vae(vae_ckpt, device)
    if vae_loaded["motion_contract"].to_dict() != motion_contract.to_dict():
        raise RuntimeError("VAE checkpoint and cache motion contract do not match")

    primary_bundle = None
    secondary_bundle = None
    hybrid_parts = tuple(part.strip() for part in str(args.hybrid_secondary_parts).split(",") if part.strip())
    hybrid_joint_keywords = tuple(part.strip() for part in str(args.hybrid_secondary_joint_keywords or "").split(",") if part.strip())
    hybrid_exclude_joint_keywords = tuple(part.strip() for part in str(args.hybrid_secondary_exclude_joint_keywords or "").split(",") if part.strip())
    hybrid_feature_indices = None
    if args.mode == "flow":
        primary_bundle = load_flow_bundle(
            args.checkpoint,
            vae_ckpt=args.vae_ckpt,
            device=device,
            use_ema=not args.no_ema,
        )
        if primary_bundle["motion_contract"].to_dict() != motion_contract.to_dict():
            raise RuntimeError("Flow checkpoint and cache motion contract do not match")
        if primary_bundle["audio_spec"].to_dict() != audio_spec.to_dict():
            raise RuntimeError("Flow checkpoint and cache audio_feature_spec do not match")
        if primary_bundle["text_token_spec"].to_dict() != text_spec.to_dict():
            raise RuntimeError("Flow checkpoint and cache text_token_spec do not match")
        if args.hybrid_secondary_checkpoint:
            secondary_bundle = load_flow_bundle(
                args.hybrid_secondary_checkpoint,
                vae_ckpt=args.hybrid_secondary_vae_ckpt,
                device=device,
                use_ema=not args.no_ema,
            )
            if secondary_bundle["motion_contract"].to_dict() != motion_contract.to_dict():
                raise RuntimeError("Hybrid secondary checkpoint and cache motion contract do not match")
            if secondary_bundle["audio_spec"].to_dict() != audio_spec.to_dict():
                raise RuntimeError("Hybrid secondary checkpoint and cache audio_feature_spec do not match")
            if secondary_bundle["text_token_spec"].to_dict() != text_spec.to_dict():
                raise RuntimeError("Hybrid secondary checkpoint and cache text_token_spec do not match")
            if hybrid_joint_keywords:
                hybrid_feature_indices = np.asarray(
                    resolve_joint_rot_feature_indices(
                        motion_contract,
                        include_keywords=hybrid_joint_keywords,
                        exclude_keywords=hybrid_exclude_joint_keywords,
                    ),
                    dtype=np.int64,
                )

    motion = np.asarray(sample["motion"], dtype=np.float32)
    motion_norm = ((motion - mean) / std).astype(np.float32, copy=False)
    audio = np.asarray(sample["audio"], dtype=np.float32)
    audio_norm = ((audio - audio_mean) / audio_std).astype(np.float32, copy=False)
    lexical = np.asarray(sample["lexical_frame"], dtype=np.float32)
    global_text = np.asarray(sample["global_text"], dtype=np.float32)
    word_frame = np.asarray(sample["word_frame"], dtype=np.float32)

    motion_len = int(motion.shape[0])
    audio_t = torch.from_numpy(audio_norm).float().unsqueeze(0).to(device)
    lexical_t = torch.from_numpy(lexical).float().unsqueeze(0).to(device)
    global_text_t = torch.from_numpy(global_text).float().unsqueeze(0).to(device)
    word_t = torch.from_numpy(word_frame).float().unsqueeze(0).to(device)
    motion_norm_t = torch.from_numpy(motion_norm).float().unsqueeze(0).to(device)
    motion_mask_t = torch.ones((1, motion_len), device=device, dtype=torch.bool)

    with torch.no_grad():
        if args.mode == "flow":
            generator = torch.Generator(device=device)
            generator.manual_seed(int(args.seed))
            pred_motion, _, _ = sample_motion_chunk_from_bundle(
                primary_bundle,
                frame_len=motion_len,
                num_steps=int(args.num_steps),
                cfg_scale=float(args.cfg_scale),
                cfg_audio_scale=args.cfg_audio_scale,
                cfg_text_scale=args.cfg_text_scale,
                cfg_word_scale=args.cfg_word_scale,
                cfg_global_text_scale=args.cfg_global_text_scale,
                solver=args.solver,
                audio_t=audio_t,
                lexical_t=lexical_t,
                global_text_t=global_text_t,
                word_t=word_t,
                prefix_latent=None,
                prefix_mask=None,
                device=device,
                generator=generator,
            )
            if secondary_bundle is not None:
                secondary_generator = torch.Generator(device=device)
                secondary_generator.manual_seed(int(args.seed) + int(args.hybrid_secondary_seed_offset))
                secondary_motion, _, _ = sample_motion_chunk_from_bundle(
                    secondary_bundle,
                    frame_len=motion_len,
                    num_steps=int(args.num_steps),
                    cfg_scale=float(args.cfg_scale if args.hybrid_secondary_cfg_scale is None else args.hybrid_secondary_cfg_scale),
                    cfg_audio_scale=(args.cfg_audio_scale if args.hybrid_secondary_cfg_audio_scale is None else args.hybrid_secondary_cfg_audio_scale),
                    cfg_text_scale=(args.cfg_text_scale if args.hybrid_secondary_cfg_text_scale is None else args.hybrid_secondary_cfg_text_scale),
                    cfg_word_scale=(args.cfg_word_scale if args.hybrid_secondary_cfg_word_scale is None else args.hybrid_secondary_cfg_word_scale),
                    cfg_global_text_scale=(args.cfg_global_text_scale if args.hybrid_secondary_cfg_global_text_scale is None else args.hybrid_secondary_cfg_global_text_scale),
                    solver=args.solver,
                    audio_t=audio_t,
                    lexical_t=lexical_t,
                    global_text_t=global_text_t,
                    word_t=word_t,
                    prefix_latent=None,
                    prefix_mask=None,
                    device=device,
                    generator=secondary_generator,
                )
                secondary_motion = apply_temporal_smoothing(
                    secondary_motion,
                    motion_contract=motion_contract,
                    part_layout=secondary_bundle["part_layout"],
                    root_window=args.hybrid_secondary_smooth_root_window,
                    upper_hand_rot_window=args.hybrid_secondary_smooth_upper_hand_rot_window,
                    all_rot_window=args.hybrid_secondary_smooth_all_rot_window,
                )
                if hybrid_feature_indices is not None:
                    pred_motion = blend_motion_features(
                        pred_motion,
                        secondary_motion,
                        feature_indices=hybrid_feature_indices,
                    )
                else:
                    pred_motion = blend_motion_parts(
                        pred_motion,
                        secondary_motion,
                        motion_contract=motion_contract,
                        part_layout=secondary_bundle["part_layout"],
                        source_parts=hybrid_parts,
                    )
        else:
            pred_motion_norm = vae(motion_norm_t, motion_mask_t)["recon_motion_norm"][0]
            pred_motion = pred_motion_norm.detach().cpu().numpy() * std + mean
    pred_motion = apply_temporal_smoothing(
        pred_motion,
        motion_contract=motion_contract,
        part_layout=(primary_bundle["part_layout"] if primary_bundle is not None else str(vae_loaded["checkpoint"]["vae_spec"].get("part_layout", "root_upper_hand_lower_slots_v3"))),
        root_window=args.smooth_root_window,
        upper_hand_rot_window=args.smooth_upper_hand_rot_window,
        all_rot_window=args.smooth_all_rot_window,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_bvh = _resolve_project_path(sample.get("src_bvh"))
    pred_root, pred_euler = decode_motion_to_bvh(
        torch.from_numpy(pred_motion).float(),
        layout=motion_contract.feature_layout,
        fps=motion_contract.fps,
        motion_dim=motion_contract.motion_dim,
    )
    gt_root, gt_euler = decode_motion_to_bvh(
        torch.from_numpy(motion).float(),
        layout=motion_contract.feature_layout,
        fps=motion_contract.fps,
        motion_dim=motion_contract.motion_dim,
    )
    pred_bvh = out_dir / "pred.bvh"
    gt_bvh = out_dir / "gt.bvh"
    save_bvh_remapped(pred_root, pred_euler, ref_bvh_path=str(src_bvh), output_path=str(pred_bvh), fps=motion_contract.fps)
    save_bvh_remapped(gt_root, gt_euler, ref_bvh_path=str(src_bvh), output_path=str(gt_bvh), fps=motion_contract.fps)

    np.save(out_dir / "pred_motion.npy", pred_motion)
    np.save(out_dir / "gt_motion.npy", motion)
    summary = {
        "mode": args.mode,
        "segment_index": int(seg_index),
        "segment_id": str(sample["segment_id"]),
        "text": str(sample["text"]),
        "motion_len": int(motion_len),
        "audio_len": int(audio.shape[0]),
        "src_bvh": str(src_bvh),
        "pred_bvh": str(pred_bvh),
        "gt_bvh": str(gt_bvh),
        "checkpoint": None if args.checkpoint is None else str(Path(args.checkpoint).resolve()),
        "vae_ckpt": None if vae_ckpt is None else str(Path(vae_ckpt).resolve()),
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
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
