from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stageB.common.bvh_io import export_canonical_full_motion_to_bvh, load_root_track_from_bvh
from stageB.common.motion_codec import load_motion_codec_bundle
from stageB.common.prosody import extract_prosody_features
from stageB.common.text_bert import (
    align_dense_interval_embeddings_to_frames,
    build_dense_interval_bert_embeddings,
    load_bert_word_encoder,
)
from stageB.data_semantic import build_all_mask_code_inputs
from stageB.common.textgrid_utils import (
    align_words_to_frames,
    parse_textgrid_interval_tiers,
    choose_best_tier,
    pool_frame_features_to_tokens,
    pool_frame_ids_to_tokens,
)
from stageB.debug_vis import save_debug_plot
from stageB.models.audio_text_token_predictor import AudioTextTokenPredictor
from stageA.train_stage1_vqvae import TARGET_FPS


def resolve_stage1_kwargs(args, stage2_ckpt: dict) -> dict:
    stage1 = stage2_ckpt.get("stage1", {})
    part_meta = stage1.get("parts", {})
    kwargs = {
        "ref_bvh": Path(args.ref_bvh) if args.ref_bvh else (Path(stage1["ref_bvh"]) if stage1.get("ref_bvh") else None),
        "stage1_ckpt": Path(args.stage1_ckpt) if args.stage1_ckpt else None,
        "upper_ckpt": Path(args.upper_ckpt) if args.upper_ckpt else None,
        "hand_ckpt": Path(args.hand_ckpt) if args.hand_ckpt else None,
        "lower_ckpt": Path(args.lower_ckpt) if args.lower_ckpt else None,
        "global_ckpt": Path(args.global_ckpt) if args.global_ckpt else None,
    }
    if kwargs["stage1_ckpt"] is None and "motion" in part_meta:
        kwargs["stage1_ckpt"] = Path(part_meta["motion"]["ckpt_path"])
    for name in ("upper", "hand", "lower", "global"):
        key = f"{name}_ckpt"
        if kwargs[key] is None and name in part_meta:
            kwargs[key] = Path(part_meta[name]["ckpt_path"])
    return kwargs


def load_word_intervals(textgrid_path: Path, tier_name: str | None) -> tuple[list[tuple[float, float, str]], str | None]:
    tiers = parse_textgrid_interval_tiers(textgrid_path)
    tier = choose_best_tier(tiers, prefer=tier_name)
    if tier is None:
        return [], None
    return tiers.get(tier, []), tier


def apply_root_fallback(full_motion: np.ndarray, *, ref_bvh: Path, root_mode: str) -> np.ndarray:
    out = np.array(full_motion, dtype=np.float32, copy=True)
    if root_mode == "zero":
        out[:, :3] = 0.0
        return out

    ref_root = load_root_track_from_bvh(ref_bvh, fps=TARGET_FPS, target_len=out.shape[0])
    if root_mode == "hold":
        out[:, :3] = ref_root[:1]
    else:
        out[:, :3] = ref_root
    return out


def build_refine_code_inputs(
    pred_codes: dict[str, torch.Tensor],
    pred_logits: dict[str, torch.Tensor],
    codebook_sizes: dict[str, int],
    *,
    confidence_threshold: float,
) -> dict[str, torch.Tensor]:
    code_inputs: dict[str, torch.Tensor] = {}
    for name, codes in pred_codes.items():
        conf = torch.softmax(pred_logits[name], dim=-1).amax(dim=-1)
        mask_id = int(codebook_sizes[name])
        ids = codes.clone()
        if float(confidence_threshold) > 0.0:
            ids = torch.where(conf >= float(confidence_threshold), ids, torch.full_like(ids, mask_id))
        code_inputs[name] = ids
    return code_inputs


def sample_semantic_codes(
    logits: torch.Tensor,
    *,
    temperature: float,
    topk: int,
) -> torch.Tensor:
    if int(topk) <= 1 or float(temperature) <= 0.0:
        return torch.argmax(logits, dim=-1)
    topk = min(int(topk), int(logits.shape[-1]))
    scaled = logits / max(float(temperature), 1e-4)
    top_vals, top_idx = torch.topk(scaled, k=topk, dim=-1)
    probs = torch.softmax(top_vals, dim=-1)
    sampled = torch.multinomial(probs.reshape(-1, topk), num_samples=1).view(*logits.shape[:-1], 1)
    return torch.gather(top_idx, dim=-1, index=sampled).squeeze(-1)


def summarize_predicted_codes(pred_codes: dict[str, torch.Tensor]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for name, codes in pred_codes.items():
        flat = codes.detach().cpu().reshape(-1)
        if flat.numel() <= 0:
            summary[name] = {"unique_codes": 0, "top1_ratio": 0.0}
            continue
        hist = torch.bincount(flat)
        summary[name] = {
            "unique_codes": int((hist > 0).sum().item()),
            "top1_ratio": float(hist.max().item() / max(1, flat.numel())),
        }
    return summary


def compute_motion_energy(full_motion: np.ndarray) -> float:
    motion = np.asarray(full_motion, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[0] < 2:
        return 0.0
    return float(np.abs(motion[1:] - motion[:-1]).mean())


def _window_starts(total_steps: int, block_steps: int, hop_steps: int) -> list[int]:
    if total_steps <= block_steps:
        return [0]
    starts = list(range(0, total_steps - block_steps + 1, max(1, hop_steps)))
    tail_start = total_steps - block_steps
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


def main():
    parser = argparse.ArgumentParser(description="Infer semantic StageB motion and export BVH.")
    parser.add_argument("--wav", type=str, required=True)
    parser.add_argument("--textgrid", type=str, required=True)
    parser.add_argument("--stage2_ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage2_infer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ref_bvh", type=str, default=None)
    parser.add_argument("--tier_name", type=str, default=None)
    parser.add_argument("--root_mode", choices=["zero", "ref", "hold"], default="hold")
    parser.add_argument("--refine_iters", type=int, default=1)
    parser.add_argument("--refine_confidence", type=float, default=0.6)
    parser.add_argument("--semantic_topk", type=int, default=8)
    parser.add_argument("--semantic_temperature", type=float, default=1.1)
    parser.add_argument("--infer_block_size_frames", type=int, default=0)
    parser.add_argument("--infer_hop_frames", type=int, default=0)
    parser.add_argument("--bert_model_dir", type=str, default=None)
    parser.add_argument("--bert_device", type=str, default=None)
    parser.add_argument("--bert_max_words", type=int, default=128)
    parser.add_argument("--bert_overlap_words", type=int, default=16)
    parser.add_argument("--stage1_ckpt", type=str, default=None)
    parser.add_argument("--upper_ckpt", type=str, default=None)
    parser.add_argument("--hand_ckpt", type=str, default=None)
    parser.add_argument("--lower_ckpt", type=str, default=None)
    parser.add_argument("--global_ckpt", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    stage2_ckpt = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=False)
    stage1_kwargs = resolve_stage1_kwargs(args, stage2_ckpt)
    codec_bundle = load_motion_codec_bundle(device=device, **stage1_kwargs)
    codec_bundle.apply_stage1_overrides(stage2_ckpt.get("stage1_decoder_overrides"))

    model = AudioTextTokenPredictor(
        prosody_dim=len(stage2_ckpt["prosody_feature_names"]),
        text_scalar_dim=len(stage2_ckpt["text_scalar_feature_names"]),
        bert_text_dim=int(stage2_ckpt.get("bert_feature_dim", 0)),
        vocab_size=len(stage2_ckpt["vocab"]),
        codebook_sizes={name: part.n_codes for name, part in codec_bundle.vq_parts.items()},
        d_model=int(stage2_ckpt["args"]["d_model"]),
        nhead=int(stage2_ckpt["args"]["nhead"]),
        num_layers=int(stage2_ckpt["args"]["num_layers"]),
        dropout=float(stage2_ckpt["args"]["dropout"]),
        use_text=(stage2_ckpt["args"]["cond_mode"] == "audio_text"),
        use_code_hints=not bool(stage2_ckpt["args"].get("disable_code_hints", False)),
        code_hint_dim=int(stage2_ckpt["args"].get("code_hint_dim", 128)),
        spatial_code_dim=int(stage2_ckpt["args"].get("spatial_code_dim", 128)),
        num_spatial_tasks=int(stage2_ckpt.get("spatial_num_tasks", 4)),
    ).to(device)
    model.load_state_dict(stage2_ckpt["model"], strict=False)
    model.eval()

    wav_path = Path(args.wav)
    tg_path = Path(args.textgrid)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV not found: {wav_path}")
    if not tg_path.exists():
        raise FileNotFoundError(f"TextGrid not found: {tg_path}")

    vocab = stage2_ckpt["vocab"]
    audio_duration = float(sf.info(str(wav_path)).duration)
    word_intervals, tier = load_word_intervals(tg_path, args.tier_name)
    text_end = max((float(end) for _, end, _ in word_intervals), default=0.0)
    total_duration = max(audio_duration, text_end)
    target_frames = max(1, int(math.ceil(total_duration * TARGET_FPS)))

    pad_frames = (-target_frames) % codec_bundle.token_stride
    target_frames += pad_frames

    prosody_pack = extract_prosody_features(wav_path, fps=TARGET_FPS, target_frames=target_frames)
    text_frame = align_words_to_frames(word_intervals, num_frames=target_frames, fps=TARGET_FPS, vocab=vocab)

    token_prosody = pool_frame_features_to_tokens(prosody_pack["frame_features"], codec_bundle.token_stride, mode="mean")
    token_text_scalar = pool_frame_features_to_tokens(text_frame.scalar, codec_bundle.token_stride, mode="mean")
    token_word_ids = pool_frame_ids_to_tokens(
        text_frame.word_ids,
        text_frame.scalar[:, 0],
        codec_bundle.token_stride,
        blank_id=int(vocab["<blank>"]),
    )
    bert_feature_dim = int(stage2_ckpt.get("bert_feature_dim", 0))
    if bert_feature_dim > 0:
        bert_model_dir = args.bert_model_dir or stage2_ckpt["args"].get("bert_model_dir") or "models/bert"
        bert_device = args.bert_device or args.device
        bert_encoder = load_bert_word_encoder(bert_model_dir, device=bert_device)
        dense_bert = build_dense_interval_bert_embeddings(
            text_frame.words,
            encoder=bert_encoder,
            max_words_per_chunk=args.bert_max_words,
            overlap_words=args.bert_overlap_words,
        )
        frame_text_bert = align_dense_interval_embeddings_to_frames(
            text_frame.words,
            dense_bert,
            num_frames=target_frames,
            dtype=np.float32,
        )
        token_text_bert = pool_frame_features_to_tokens(frame_text_bert, codec_bundle.token_stride, mode="mean")
    else:
        token_text_bert = np.zeros((int(token_word_ids.shape[0]), 0), dtype=np.float32)

    token_prosody_t = torch.from_numpy(token_prosody).unsqueeze(0).to(device)
    token_word_ids_t = torch.from_numpy(token_word_ids).unsqueeze(0).to(device)
    token_text_scalar_t = torch.from_numpy(token_text_scalar).unsqueeze(0).to(device)
    token_text_bert_t = torch.from_numpy(token_text_bert).unsqueeze(0).to(device)

    codebook_sizes = {name: part.n_codes for name, part in codec_bundle.vq_parts.items()}
    ckpt_block_frames = int(stage2_ckpt.get("block_size", 0))
    infer_block_frames = int(args.infer_block_size_frames) if int(args.infer_block_size_frames) > 0 else ckpt_block_frames
    if infer_block_frames <= 0:
        infer_block_frames = int(token_prosody_t.shape[1]) * int(codec_bundle.token_stride)
    if infer_block_frames % int(codec_bundle.token_stride) != 0:
        raise ValueError(
            f"infer_block_size_frames={infer_block_frames} must be divisible by token_stride={codec_bundle.token_stride}"
        )
    infer_hop_frames = int(args.infer_hop_frames) if int(args.infer_hop_frames) > 0 else infer_block_frames
    if infer_hop_frames % int(codec_bundle.token_stride) != 0:
        raise ValueError(
            f"infer_hop_frames={infer_hop_frames} must be divisible by token_stride={codec_bundle.token_stride}"
        )
    block_tokens = max(1, int(infer_block_frames // codec_bundle.token_stride))
    hop_tokens = max(1, int(infer_hop_frames // codec_bundle.token_stride))
    total_tokens = int(token_prosody_t.shape[1])
    starts = _window_starts(total_tokens, block_tokens, hop_tokens)

    with torch.no_grad():
        logit_sums = {
            name: torch.zeros((1, total_tokens, int(part.n_codes)), device=device, dtype=torch.float32)
            for name, part in codec_bundle.vq_parts.items()
        }
        fusion_gate_sum = torch.zeros((1, total_tokens, 1), device=device, dtype=torch.float32)
        hint_gate_sum = torch.zeros((1, total_tokens, 1), device=device, dtype=torch.float32)
        counts = torch.zeros((1, total_tokens, 1), device=device, dtype=torch.float32)

        for start in starts:
            end = min(start + block_tokens, total_tokens)
            chunk_prosody = token_prosody_t[:, start:end]
            chunk_word_ids = token_word_ids_t[:, start:end]
            chunk_text_scalar = token_text_scalar_t[:, start:end]
            chunk_text_bert = token_text_bert_t[:, start:end]

            token_code_inputs = None
            if model.use_code_hints:
                token_code_inputs = build_all_mask_code_inputs(
                    (int(chunk_prosody.shape[0]), int(chunk_prosody.shape[1])),
                    codebook_sizes,
                    device,
                )

            outputs = None
            pred_codes = None
            for refine_idx in range(max(1, int(args.refine_iters))):
                outputs = model(
                    token_prosody=chunk_prosody,
                    token_word_ids=chunk_word_ids,
                    token_text_scalar=chunk_text_scalar,
                    token_text_bert=chunk_text_bert,
                    token_code_inputs=token_code_inputs,
                )
                pred_codes = {}
                for name, logits in outputs["logits"].items():
                    if name in {"upper", "hand"}:
                        pred_codes[name] = sample_semantic_codes(
                            logits,
                            temperature=args.semantic_temperature,
                            topk=args.semantic_topk,
                        )
                    else:
                        pred_codes[name] = torch.argmax(logits, dim=-1)
                if not model.use_code_hints or refine_idx >= max(1, int(args.refine_iters)) - 1:
                    break
                token_code_inputs = build_refine_code_inputs(
                    pred_codes,
                    outputs["logits"],
                    codebook_sizes,
                    confidence_threshold=args.refine_confidence,
                )

            for name, logits in outputs["logits"].items():
                logit_sums[name][:, start:end] += logits.float()
            fusion_gate_sum[:, start:end] += outputs["fusion_gate"].float()
            hint_gate_sum[:, start:end] += outputs["hint_gate"].float()
            counts[:, start:end] += 1.0

        counts = counts.clamp_min(1.0)
        outputs = {
            "logits": {name: values / counts for name, values in logit_sums.items()},
            "fusion_gate": fusion_gate_sum / counts,
            "hint_gate": hint_gate_sum / counts,
        }
        pred_codes = {}
        for name, logits in outputs["logits"].items():
            if name in {"upper", "hand"}:
                pred_codes[name] = sample_semantic_codes(
                    logits,
                    temperature=args.semantic_temperature,
                    topk=args.semantic_topk,
                )
            else:
                pred_codes[name] = torch.argmax(logits, dim=-1)

    decoded_parts = {
        name: codec_bundle.parts[name].decode_codes(code)[0].detach().cpu().numpy().astype(np.float32)
        for name, code in pred_codes.items()
    }
    full_motion = codec_bundle.merge_decoded_parts(decoded_parts, gt_full=None)
    full_motion = np.asarray(full_motion, dtype=np.float32)
    full_motion = apply_root_fallback(full_motion, ref_bvh=codec_bundle.ref_bvh, root_mode=args.root_mode)
    code_stats = summarize_predicted_codes(pred_codes)
    motion_energy = compute_motion_energy(full_motion)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    motion_path = out_dir / "predicted_full_motion.npy"
    code_path = out_dir / "predicted_codes.pt"
    bvh_path = out_dir / "output.bvh"
    debug_png = out_dir / "debug.png"
    meta_path = out_dir / "summary.json"

    np.save(motion_path, full_motion)
    torch.save({name: code.detach().cpu() for name, code in pred_codes.items()}, code_path)
    export_canonical_full_motion_to_bvh(
        full_motion,
        ref_bvh=codec_bundle.ref_bvh,
        output_path=bvh_path,
    )
    save_debug_plot(
        debug_png,
        prosody_frame=prosody_pack["frame_features"],
        token_gate=outputs["fusion_gate"][0, :, 0].detach().cpu().numpy(),
        token_codes={name: code.detach().cpu().numpy()[0] for name, code in pred_codes.items()},
        token_preds={name: code.detach().cpu().numpy()[0] for name, code in pred_codes.items()},
        words=text_frame.words,
        fps=TARGET_FPS,
        token_stride=codec_bundle.token_stride,
        part_stats=code_stats,
    )

    summary = {
        "wav": str(wav_path),
        "textgrid": str(tg_path),
        "tier_name": tier,
        "stage2_ckpt": str(args.stage2_ckpt),
        "stage1_parts": list(codec_bundle.parts.keys()),
        "token_stride": codec_bundle.token_stride,
        "target_frames": target_frames,
        "infer_block_size_frames": int(infer_block_frames),
        "infer_hop_frames": int(infer_hop_frames),
        "num_blocks": int(len(starts)),
        "root_mode": args.root_mode,
        "refine_iters": int(args.refine_iters),
        "refine_confidence": float(args.refine_confidence),
        "semantic_topk": int(args.semantic_topk),
        "semantic_temperature": float(args.semantic_temperature),
        "bert_feature_dim": bert_feature_dim,
        "code_stats": code_stats,
        "motion_energy": motion_energy,
        "outputs": {
            "motion_npy": str(motion_path),
            "codes_pt": str(code_path),
            "bvh": str(bvh_path),
            "debug_png": str(debug_png),
        },
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
