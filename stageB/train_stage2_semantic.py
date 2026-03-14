from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stageB.common.motion_codec import load_motion_codec_bundle
from stageB.data_semantic import (
    Stage2SemanticDataset,
    build_all_mask_code_inputs,
    collate_stage2_semantic,
    code_usage_regularization,
    compute_frame_emphasis_weights,
    compute_token_emphasis_weights,
    sample_masked_code_inputs,
    weighted_acceleration_loss,
    weighted_smooth_l1,
    weighted_velocity_loss,
)
from stageB.debug_vis import save_debug_plot
from stageB.models.audio_text_token_predictor import AudioTextTokenPredictor
from stageB.models.spatial_tasks import (
    SpatialTaskSpec,
    build_spatial_tasks,
    parse_spatial_task_weights,
    sample_spatial_task,
    task_name_list,
    task_to_part_masks,
)
from stageA.train_stage1_vqvae import BVHFkTorch, fk_joint_losses


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_stage1_kwargs(args, ds: Stage2SemanticDataset) -> dict:
    stage1_meta = ds.stage1 or {}
    part_meta = stage1_meta.get("parts", {})
    kwargs = {
        "ref_bvh": Path(args.ref_bvh) if args.ref_bvh else (Path(stage1_meta["ref_bvh"]) if stage1_meta.get("ref_bvh") else None),
        "stage1_ckpt": Path(args.stage1_ckpt) if args.stage1_ckpt else None,
        "upper_ckpt": Path(args.upper_ckpt) if args.upper_ckpt else None,
        "hand_ckpt": Path(args.hand_ckpt) if args.hand_ckpt else None,
        "lower_ckpt": Path(args.lower_ckpt) if args.lower_ckpt else None,
        "global_ckpt": Path(args.global_ckpt) if args.global_ckpt else None,
    }
    if kwargs["stage1_ckpt"] is None and args.stage1_ckpt is None and "motion" in part_meta:
        kwargs["stage1_ckpt"] = Path(part_meta["motion"]["ckpt_path"])
    for name in ("upper", "hand", "lower", "global"):
        key = f"{name}_ckpt"
        if kwargs[key] is None and name in part_meta:
            kwargs[key] = Path(part_meta[name]["ckpt_path"])
    return kwargs


def weighted_accuracy(pred: torch.Tensor, target: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    correct = (pred == target).float()
    if weights is None:
        return correct.mean()
    return (correct * weights).sum() / (weights.sum() + 1e-6)


def code_usage_metrics(pred_codes: torch.Tensor, n_codes: int) -> tuple[float, float]:
    if pred_codes.numel() == 0 or int(n_codes) <= 0:
        return 0.0, 0.0
    hist = torch.bincount(pred_codes.reshape(-1), minlength=int(n_codes)).float()
    probs = hist / (hist.sum() + 1e-6)
    perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum()).item()
    active = float((hist > 0).sum().item())
    return float(perplexity), active


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def parse_gpu_ids(gpu_ids: Optional[str]) -> list[int]:
    if not gpu_ids:
        return []
    return [int(x.strip()) for x in str(gpu_ids).split(",") if x.strip()]


def get_amp_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name == "bf16":
        return torch.bfloat16
    return torch.float16


def save_stage2_ckpt(
    path: Path,
    *,
    model: AudioTextTokenPredictor,
    optimizer: torch.optim.Optimizer,
    args,
    train_ds: Stage2SemanticDataset,
    epoch: int,
    best_metric: float,
    codec_bundle,
    finetuned_decoder: bool,
    extra_state: Optional[dict] = None,
) -> None:
    model_to_save = unwrap_model(model)
    payload = {
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "stage1": train_ds.stage1,
        "vocab": train_ds.vocab,
        "prosody_feature_names": train_ds.prosody_feature_names,
        "text_scalar_feature_names": train_ds.text_scalar_feature_names,
        "bert_feature_dim": int(train_ds.bert_feature_dim),
        "token_stride": train_ds.token_stride,
        "block_size": train_ds.block_size,
        "stage1_decoder_overrides": (
            {name: part.decoder_state_dict() for name, part in codec_bundle.vq_parts.items()}
            if finetuned_decoder
            else None
        ),
    }
    if extra_state:
        payload.update(extra_state)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_spatial_pretrain_weights(model: AudioTextTokenPredictor, ckpt_path: str | Path) -> dict[str, list[str]]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    keep_prefixes = ("temporal.", "heads.", "positional_encoding.")
    filtered = {key: value for key, value in state.items() if key.startswith(keep_prefixes)}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return {"missing": list(missing), "unexpected": list(unexpected)}


def _branch_metric_name(part_name: str, metric: str, branch: Optional[str]) -> str:
    return f"{part_name}_{metric}" if not branch else f"{part_name}_{branch}_{metric}"


def compute_branch_objective(
    *,
    outputs: dict,
    code_targets: Dict[str, torch.Tensor],
    token_weights: torch.Tensor,
    frame_weights: torch.Tensor,
    part_motion: Dict[str, torch.Tensor],
    codec_bundle,
    args,
    branch: Optional[str],
    masked_token_mask: Optional[torch.Tensor] = None,
    compute_motion_losses: bool = True,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    ref = next(iter(code_targets.values()))
    branch_loss = ref.new_tensor(0.0, dtype=torch.float32)
    batch_metrics: dict[str, float] = {}
    decoded_parts: dict[str, torch.Tensor] = {}

    for name in codec_bundle.primary_part_names:
        logits = outputs["logits"][name]
        target = code_targets[name]
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            reduction="none",
            label_smoothing=float(args.label_smoothing),
        ).view_as(target)
        ce_loss = (ce * token_weights).sum() / (token_weights.sum() + 1e-6)

        masked_ce = logits.new_tensor(0.0)
        if masked_token_mask is not None and float(args.masked_ce_w) > 0.0:
            masked_weights = token_weights * masked_token_mask.float()
            if float(masked_weights.sum().detach().item()) > 0.0:
                masked_ce = (ce * masked_weights).sum() / (masked_weights.sum() + 1e-6)

        usage_loss = logits.new_tensor(0.0)
        if float(args.usage_w) > 0.0:
            usage_loss = code_usage_regularization(logits, token_weights)

        pred_codes = torch.argmax(logits, dim=-1)
        part_acc = weighted_accuracy(pred_codes, target, token_weights)
        ppl, active = code_usage_metrics(pred_codes.detach(), codec_bundle.parts[name].n_codes)

        part_loss = args.ce_w * ce_loss + args.masked_ce_w * masked_ce + args.usage_w * usage_loss

        recon = logits.new_tensor(0.0)
        vel = logits.new_tensor(0.0)
        accel = logits.new_tensor(0.0)
        if compute_motion_losses:
            decoded = codec_bundle.parts[name].soft_decode_logits(logits, temperature=args.decoder_softmax_temp)
            decoded_parts[name] = decoded
            recon = weighted_smooth_l1(decoded, part_motion[name], frame_weights)
            vel = weighted_velocity_loss(decoded, part_motion[name], frame_weights)
            accel = weighted_acceleration_loss(decoded, part_motion[name], frame_weights)
            part_loss = part_loss + args.recon_w * recon + args.vel_w * vel + args.acc_w * accel

        branch_loss = branch_loss + part_loss
        batch_metrics[_branch_metric_name(name, "ce", branch)] = float(ce_loss.detach().item())
        batch_metrics[_branch_metric_name(name, "masked_ce", branch)] = float(masked_ce.detach().item())
        batch_metrics[_branch_metric_name(name, "usage", branch)] = float(usage_loss.detach().item())
        batch_metrics[_branch_metric_name(name, "acc", branch)] = float(part_acc.detach().item())
        batch_metrics[_branch_metric_name(name, "ppl", branch)] = float(ppl)
        batch_metrics[_branch_metric_name(name, "active", branch)] = float(active)
        if compute_motion_losses:
            batch_metrics[_branch_metric_name(name, "recon", branch)] = float(recon.detach().item())
            batch_metrics[_branch_metric_name(name, "vel", branch)] = float(vel.detach().item())
            batch_metrics[_branch_metric_name(name, "accel", branch)] = float(accel.detach().item())

    return branch_loss, batch_metrics, decoded_parts


def compute_spatial_objective(
    *,
    outputs: dict,
    code_targets: Dict[str, torch.Tensor],
    part_motion: Dict[str, torch.Tensor],
    codec_bundle,
    args,
    task: SpatialTaskSpec,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    loss = next(iter(code_targets.values())).new_tensor(0.0, dtype=torch.float32)
    metrics: dict[str, float] = {}
    decoded_parts: dict[str, torch.Tensor] = {}

    task_ce = 0.0
    task_acc = 0.0
    task_ppl = 0.0
    task_active = 0.0
    n_targets = 0

    for name in task.target_parts:
        logits = outputs["logits"][name]
        target = code_targets[name]
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            reduction="mean",
            label_smoothing=float(args.label_smoothing),
        )
        usage_loss = logits.new_tensor(0.0)
        if float(args.spatial_usage_w) > 0.0:
            usage_loss = code_usage_regularization(logits)

        part_loss = args.ce_w * ce_loss + args.spatial_usage_w * usage_loss

        recon = logits.new_tensor(0.0)
        if float(args.spatial_recon_w) > 0.0:
            decoded = codec_bundle.parts[name].soft_decode_logits(logits, temperature=args.decoder_softmax_temp)
            decoded_parts[name] = decoded
            recon = F.smooth_l1_loss(decoded, part_motion[name], reduction="mean")
            part_loss = part_loss + args.spatial_recon_w * recon

        pred_codes = torch.argmax(logits, dim=-1)
        acc = weighted_accuracy(pred_codes, target, None)
        ppl, active = code_usage_metrics(pred_codes.detach(), codec_bundle.parts[name].n_codes)

        loss = loss + part_loss
        metrics[f"{task.name}_{name}_ce"] = float(ce_loss.detach().item())
        metrics[f"{task.name}_{name}_usage"] = float(usage_loss.detach().item())
        metrics[f"{task.name}_{name}_acc"] = float(acc.detach().item())
        metrics[f"{task.name}_{name}_ppl"] = float(ppl)
        metrics[f"{task.name}_{name}_active"] = float(active)
        metrics[f"{task.name}_{name}_recon"] = float(recon.detach().item())
        task_ce += float(ce_loss.detach().item())
        task_acc += float(acc.detach().item())
        task_ppl += float(ppl)
        task_active += float(active)
        n_targets += 1

    denom = max(1, n_targets)
    metrics[f"{task.name}_loss"] = float(loss.detach().item())
    metrics[f"{task.name}_ce"] = task_ce / denom
    metrics[f"{task.name}_acc"] = task_acc / denom
    metrics[f"{task.name}_ppl"] = task_ppl / denom
    metrics[f"{task.name}_active"] = task_active / denom
    return loss, metrics, decoded_parts


def run_epoch_spatial_single(
    *,
    model: AudioTextTokenPredictor,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler,
    codec_bundle,
    args,
    task: Optional[SpatialTaskSpec],
    task_pool: Sequence[SpatialTaskSpec],
    task_weights: Sequence[float],
    split: str,
    epoch: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    rng: random.Random,
) -> tuple[dict, Optional[dict]]:
    is_train = optimizer is not None
    model.train(is_train)
    model_ref = unwrap_model(model)
    non_blocking = device.type == "cuda"
    progress = tqdm(
        loader,
        desc=(f"{split} {epoch} {task.name}" if task is not None else f"{split} {epoch}"),
        leave=True,
        dynamic_ncols=True,
        file=sys.stdout,
        mininterval=0.5,
    )

    totals: Dict[str, float] = {"loss": 0.0, "n": 0.0}
    task_counts = {spec.name: 0.0 for spec in task_pool}
    debug_payload = None
    epoch_start = time.time()
    tokens_seen = 0

    for batch in progress:
        if batch is None:
            continue
        full_motion = batch["full_motion"].to(device, non_blocking=non_blocking)
        part_motion = {name: tensor.to(device, non_blocking=non_blocking) for name, tensor in batch["part_motion"].items()}
        code_targets = {name: tensor.to(device, non_blocking=non_blocking) for name, tensor in batch["codes"].items()}
        batch_size, seq_len = next(iter(code_targets.values())).shape

        batch_task = task if task is not None else sample_spatial_task(task_pool, task_weights, rng)
        source_mask, target_mask = task_to_part_masks(
            batch_task,
            model_ref.part_names,
            batch_size=int(batch_size),
            device=device,
        )
        task_ids = torch.full((int(batch_size),), int(batch_task.task_id), dtype=torch.long, device=device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model_ref.forward_spatial(
                code_targets=code_targets,
                source_part_mask=source_mask,
                target_part_mask=target_mask,
                task_ids=task_ids,
            )
            loss, batch_metrics, decoded_parts = compute_spatial_objective(
                outputs=outputs,
                code_targets=code_targets,
                part_motion=part_motion,
                codec_bundle=codec_bundle,
                args=args,
                task=batch_task,
            )

            if float(args.spatial_fk_hand_pos_w) > 0.0 and "hand" in batch_task.target_parts and "hand" in decoded_parts:
                pred_full = codec_bundle.merge_decoded_parts(decoded_parts, gt_full=full_motion)
                pos_gt = args._spatial_fk.fk_positions(full_motion)
                pos_pd = args._spatial_fk.fk_positions(pred_full)
                hand_pos, _, _ = fk_joint_losses(
                    pos_gt=pos_gt,
                    pos_pd=pos_pd,
                    joint_ids=args._spatial_hand_joint_ids,
                    pos_w=args.spatial_fk_hand_pos_w,
                    vel_w=0.0,
                    acc_w=0.0,
                )
                loss = loss + hand_pos
                batch_metrics[f"{batch_task.name}_fk_hand_pos"] = float(hand_pos.detach().item())

        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        totals["loss"] += float(loss.detach().item())
        totals["n"] += 1.0
        task_counts[batch_task.name] += 1.0
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)

        tokens_seen += int(batch_size) * int(seq_len)
        elapsed = max(time.time() - epoch_start, 1e-6)
        progress.set_postfix(
            task=batch_task.name,
            loss=f"{float(loss.detach().item()):.4f}",
            acc=f"{batch_metrics.get(f'{batch_task.name}_acc', 0.0):.3f}",
            tok_s=f"{tokens_seen / elapsed:.0f}",
        )

        if debug_payload is None:
            pred_codes = {name: torch.argmax(logits, dim=-1) for name, logits in outputs["logits"].items()}
            debug_payload = {
                "task": batch_task.name,
                "token_gt": {name: code_targets[name][0].detach().cpu().numpy() for name in batch_task.target_parts},
                "token_pred": {name: pred_codes[name][0].detach().cpu().numpy() for name in batch_task.target_parts},
                "words": batch["meta"][0].get("words", []),
            }

    progress.close()
    if totals["n"] <= 0:
        return {}, debug_payload

    mean_metrics: Dict[str, float] = {"loss": totals["loss"] / totals["n"], "n": totals["n"]}
    for key, value in totals.items():
        if key in {"loss", "n"}:
            continue
        denom = totals["n"]
        for task_name, count in task_counts.items():
            if key == task_name or key.startswith(task_name + "_"):
                denom = count
                break
        mean_metrics[key] = value / max(1.0, denom)
    total_batches = max(1.0, totals["n"])
    if task is not None:
        mean_metrics[f"{task.name}_sample_freq"] = 1.0
    else:
        for task_name, count in task_counts.items():
            mean_metrics[f"{task_name}_sample_freq"] = count / total_batches
    return mean_metrics, debug_payload


def run_epoch_spatial(
    *,
    model: AudioTextTokenPredictor,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler,
    codec_bundle,
    args,
    split: str,
    epoch: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    spatial_tasks: Sequence[SpatialTaskSpec],
    spatial_task_weights: Sequence[float],
    rng: random.Random,
) -> tuple[dict, Optional[dict]]:
    if optimizer is not None or len(spatial_tasks) <= 1:
        return run_epoch_spatial_single(
            model=model,
            loader=loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            codec_bundle=codec_bundle,
            args=args,
            task=None,
            task_pool=spatial_tasks,
            task_weights=spatial_task_weights,
            split=split,
            epoch=epoch,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            rng=rng,
        )

    merged: dict[str, float] = {"loss": 0.0, "n": 0.0}
    debug_payload = None
    for task in spatial_tasks:
        task_metrics, task_debug = run_epoch_spatial_single(
            model=model,
            loader=loader,
            device=device,
            optimizer=None,
            scaler=None,
            codec_bundle=codec_bundle,
            args=args,
            task=task,
            task_pool=spatial_tasks,
            task_weights=spatial_task_weights,
            split=split,
            epoch=epoch,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            rng=rng,
        )
        if not task_metrics:
            continue
        merged["loss"] += float(task_metrics["loss"])
        merged["n"] += 1.0
        for key, value in task_metrics.items():
            if key in {"loss", "n"}:
                continue
            merged[key] = value
        if debug_payload is None and task_debug is not None:
            debug_payload = task_debug
    if merged["n"] > 0:
        merged["loss"] /= merged["n"]
    return merged, debug_payload


def run_epoch(
    *,
    model: AudioTextTokenPredictor,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler,
    codec_bundle,
    args,
    fk: Optional[BVHFkTorch],
    hand_joint_ids: Optional[list[int]],
    split: str,
    epoch: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[dict, Optional[dict]]:
    is_train = optimizer is not None
    model.train(is_train)
    model_ref = unwrap_model(model)
    codebook_sizes = {name: part.n_codes for name, part in codec_bundle.vq_parts.items()}
    non_blocking = device.type == "cuda"

    totals: Dict[str, float] = {"loss": 0.0, "n": 0.0}
    for name in codec_bundle.primary_part_names:
        totals[f"{name}_ce"] = 0.0
        totals[f"{name}_masked_ce"] = 0.0
        totals[f"{name}_usage"] = 0.0
        totals[f"{name}_acc"] = 0.0
        totals[f"{name}_ppl"] = 0.0
        totals[f"{name}_active"] = 0.0
        totals[f"{name}_recon"] = 0.0
        totals[f"{name}_vel"] = 0.0
        totals[f"{name}_accel"] = 0.0
        totals[f"{name}_hint_ce"] = 0.0
        totals[f"{name}_hint_masked_ce"] = 0.0
        totals[f"{name}_hint_usage"] = 0.0
        totals[f"{name}_hint_acc"] = 0.0
        totals[f"{name}_hint_ppl"] = 0.0
        totals[f"{name}_hint_active"] = 0.0
    totals["gen_loss"] = 0.0
    totals["hint_loss"] = 0.0
    totals["fk_hand_pos"] = 0.0
    totals["fk_hand_vel"] = 0.0
    totals["fk_hand_acc"] = 0.0

    debug_payload = None

    progress = tqdm(
        loader,
        desc=f"{split} {epoch}",
        leave=True,
        dynamic_ncols=True,
        file=sys.stdout,
        mininterval=0.5,
    )
    epoch_start = time.time()
    tokens_seen = 0
    samples_seen = 0
    for step, batch in enumerate(progress, start=1):
        if batch is None:
            continue
        full_motion = batch["full_motion"].to(device, non_blocking=non_blocking)
        token_prosody = batch["token_prosody"].to(device, non_blocking=non_blocking)
        token_word_ids = batch["token_word_ids"].to(device, non_blocking=non_blocking)
        token_text_scalar = batch["token_text_scalar"].to(device, non_blocking=non_blocking)
        token_text_bert = batch["token_text_bert"].to(device, non_blocking=non_blocking)
        text_scalar_frame = batch["text_scalar_frame"].to(device, non_blocking=non_blocking)
        prosody_frame = batch["prosody_frame"].to(device, non_blocking=non_blocking)
        part_motion = {name: tensor.to(device, non_blocking=non_blocking) for name, tensor in batch["part_motion"].items()}
        code_targets = {name: tensor.to(device, non_blocking=non_blocking) for name, tensor in batch["codes"].items()}

        if args.cond_mode == "prosody_only":
            token_word_ids = torch.zeros_like(token_word_ids)
            token_text_scalar = torch.zeros_like(token_text_scalar)
            token_text_bert = torch.zeros_like(token_text_bert)

        token_shape = next(iter(code_targets.values())).shape
        gen_code_inputs = None
        hint_code_inputs = None
        hint_token_mask = None
        if getattr(model_ref, "use_code_hints", False):
            gen_code_inputs = build_all_mask_code_inputs((token_shape[0], token_shape[1]), codebook_sizes, device)
            if is_train and not args.disable_dual_branch and float(args.hint_branch_w) > 0.0:
                hint_code_inputs, hint_token_mask = sample_masked_code_inputs(
                    code_targets,
                    codebook_sizes,
                    mask_ratio=args.code_hint_mask_ratio,
                    mask_span=args.code_hint_mask_span,
                    random_replace_prob=args.code_hint_random_replace_prob,
                    keep_original_prob=args.code_hint_keep_prob,
                    batch_drop_prob=args.code_hint_batch_drop_prob,
                )

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            gen_outputs = model(
                token_prosody=token_prosody,
                token_word_ids=token_word_ids,
                token_text_scalar=token_text_scalar,
                token_text_bert=token_text_bert,
                token_code_inputs=gen_code_inputs,
            )

            token_weights = compute_token_emphasis_weights(
                token_prosody,
                batch["token_text_scalar"].to(device, non_blocking=non_blocking),
                speech_boost=args.speech_boost,
                boundary_boost=args.boundary_boost,
                onset_boost=args.onset_boost,
            )
            frame_weights = compute_frame_emphasis_weights(
                prosody_frame,
                text_scalar_frame,
                speech_boost=args.speech_boost,
                boundary_boost=args.boundary_boost,
                onset_boost=args.onset_boost,
            )

            gen_loss, batch_metrics, decoded_parts = compute_branch_objective(
                outputs=gen_outputs,
                code_targets=code_targets,
                token_weights=token_weights,
                frame_weights=frame_weights,
                part_motion=part_motion,
                codec_bundle=codec_bundle,
                args=args,
                branch=None,
                masked_token_mask=None,
                compute_motion_losses=True,
            )

            fk_hand_pos = full_motion.new_tensor(0.0)
            fk_hand_vel = full_motion.new_tensor(0.0)
            fk_hand_acc = full_motion.new_tensor(0.0)
            if fk is not None and hand_joint_ids:
                pred_full = codec_bundle.merge_decoded_parts(decoded_parts, gt_full=full_motion)
                pos_gt = fk.fk_positions(full_motion)
                pos_pd = fk.fk_positions(pred_full)
                fk_hand_pos, fk_hand_vel, fk_hand_acc = fk_joint_losses(
                    pos_gt=pos_gt,
                    pos_pd=pos_pd,
                    joint_ids=hand_joint_ids,
                    pos_w=args.fk_hand_pos_w,
                    vel_w=args.fk_hand_vel_w,
                    acc_w=args.fk_hand_acc_w,
                )
                gen_loss = gen_loss + fk_hand_pos + fk_hand_vel + fk_hand_acc

            hint_loss = full_motion.new_tensor(0.0)
            if hint_code_inputs is not None:
                hint_outputs = model(
                    token_prosody=token_prosody,
                    token_word_ids=token_word_ids,
                    token_text_scalar=token_text_scalar,
                    token_text_bert=token_text_bert,
                    token_code_inputs=hint_code_inputs,
                )
                hint_loss, hint_metrics, _ = compute_branch_objective(
                    outputs=hint_outputs,
                    code_targets=code_targets,
                    token_weights=token_weights,
                    frame_weights=frame_weights,
                    part_motion=part_motion,
                    codec_bundle=codec_bundle,
                    args=args,
                    branch="hint",
                    masked_token_mask=hint_token_mask,
                    compute_motion_losses=False,
                )
                batch_metrics.update(hint_metrics)

            total_loss = args.gen_branch_w * gen_loss + args.hint_branch_w * hint_loss

        if is_train:
            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        totals["loss"] += float(total_loss.detach().item())
        totals["gen_loss"] += float(gen_loss.detach().item())
        totals["hint_loss"] += float(hint_loss.detach().item())
        totals["n"] += 1.0
        totals["fk_hand_pos"] += float(fk_hand_pos.detach().item())
        totals["fk_hand_vel"] += float(fk_hand_vel.detach().item())
        totals["fk_hand_acc"] += float(fk_hand_acc.detach().item())
        for key, value in batch_metrics.items():
            totals[key] += float(value)

        samples_seen += int(full_motion.shape[0])
        tokens_seen += int(full_motion.shape[0]) * int(next(iter(code_targets.values())).shape[1])
        elapsed = max(time.time() - epoch_start, 1e-6)

        progress.set_postfix(
            loss=f"{float(total_loss.detach().item()):.4f}",
            upper_acc=f"{batch_metrics.get('upper_acc', 0.0):.3f}",
            hand_acc=f"{batch_metrics.get('hand_acc', 0.0):.3f}",
            lower_acc=f"{batch_metrics.get('lower_acc', 0.0):.3f}",
            hint=f"{float(hint_loss.detach().item()):.2f}",
            tok_s=f"{tokens_seen / elapsed:.0f}",
        )

        if debug_payload is None:
            debug_payload = {
                "prosody_frame": batch["prosody_frame"][0].detach().cpu().numpy(),
                "fusion_gate": gen_outputs["fusion_gate"][0, :, 0].detach().cpu().numpy(),
                "hint_gate": gen_outputs["hint_gate"][0, :, 0].detach().cpu().numpy(),
                "token_gt": {name: code_targets[name][0].detach().cpu().numpy() for name in codec_bundle.primary_part_names},
                "token_pred": {name: torch.argmax(gen_outputs["logits"][name][0], dim=-1).detach().cpu().numpy() for name in codec_bundle.primary_part_names},
                "words": batch["meta"][0]["words"],
            }

    progress.close()
    if totals["n"] <= 0:
        return {}, debug_payload
    for key in list(totals.keys()):
        if key != "n":
            totals[key] /= totals["n"]
    return totals, debug_payload


def main():
    parser = argparse.ArgumentParser(description="Train semantic StageB on top of frozen StageA codecs.")
    parser.add_argument("--train_cache", type=str, required=True)
    parser.add_argument("--val_cache", type=str, default=None)
    parser.add_argument("--save", type=str, default="checkpoints/stage2_semantic.pt")
    parser.add_argument("--train_mode", choices=["spatial_pretrain", "stage2_finetune"], default="stage2_finetune")
    parser.add_argument("--load_spatial_pretrain", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--cond_mode", choices=["prosody_only", "audio_text"], default="audio_text")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--code_hint_dim", type=int, default=128)
    parser.add_argument("--disable_code_hints", action="store_true")
    parser.add_argument("--code_hint_mask_ratio", type=float, default=0.2)
    parser.add_argument("--code_hint_mask_span", type=int, default=3)
    parser.add_argument("--code_hint_random_replace_prob", type=float, default=0.1)
    parser.add_argument("--code_hint_keep_prob", type=float, default=0.1)
    parser.add_argument("--code_hint_batch_drop_prob", type=float, default=0.15)
    parser.add_argument("--disable_dual_branch", action="store_true")
    parser.add_argument("--gen_branch_w", type=float, default=1.0)
    parser.add_argument("--hint_branch_w", type=float, default=0.5)
    parser.add_argument("--decoder_softmax_temp", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--masked_ce_w", type=float, default=0.5)
    parser.add_argument("--usage_w", type=float, default=0.02)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--amp_dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--spatial_tasks", type=str, default="upper_to_hand,upper_to_lower,hand_to_upper,upper_hand_to_lower")
    parser.add_argument("--spatial_task_weights", type=str, default=None)
    parser.add_argument("--spatial_code_dim", type=int, default=128)
    parser.add_argument("--spatial_recon_w", type=float, default=0.0)
    parser.add_argument("--spatial_usage_w", type=float, default=0.01)
    parser.add_argument("--spatial_fk_hand_pos_w", type=float, default=0.0)

    parser.add_argument("--ce_w", type=float, default=1.0)
    parser.add_argument("--recon_w", type=float, default=0.5)
    parser.add_argument("--vel_w", type=float, default=0.1)
    parser.add_argument("--acc_w", type=float, default=0.05)
    parser.add_argument("--fk_hand_pos_w", type=float, default=0.0)
    parser.add_argument("--fk_hand_vel_w", type=float, default=0.0)
    parser.add_argument("--fk_hand_acc_w", type=float, default=0.0)

    parser.add_argument("--speech_boost", type=float, default=0.25)
    parser.add_argument("--boundary_boost", type=float, default=0.25)
    parser.add_argument("--onset_boost", type=float, default=0.25)

    parser.add_argument("--finetune_stage1_decoder", action="store_true")
    parser.add_argument("--debug_dir", type=str, default="outputs/stage2_debug")

    parser.add_argument("--ref_bvh", type=str, default=None)
    parser.add_argument("--stage1_ckpt", type=str, default=None)
    parser.add_argument("--upper_ckpt", type=str, default=None)
    parser.add_argument("--hand_ckpt", type=str, default=None)
    parser.add_argument("--lower_ckpt", type=str, default=None)
    parser.add_argument("--global_ckpt", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    device = torch.device(f"cuda:{gpu_ids[0]}") if (args.device.startswith("cuda") and gpu_ids) else torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    amp_enabled = device.type == "cuda" and (not args.no_amp)
    amp_dtype = get_amp_dtype(args.amp_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    spatial_tasks = build_spatial_tasks(args.spatial_tasks)
    spatial_task_weights = parse_spatial_task_weights(args.spatial_task_weights, spatial_tasks)
    spatial_rng = random.Random(args.seed + 12345)

    train_ds = Stage2SemanticDataset(Path(args.train_cache))
    val_ds = Stage2SemanticDataset(Path(args.val_cache)) if args.val_cache else None

    stage1_kwargs = resolve_stage1_kwargs(args, train_ds)
    codec_bundle = load_motion_codec_bundle(device=device, **stage1_kwargs)
    if args.finetune_stage1_decoder:
        codec_bundle.set_decoder_trainable(True)

    model = AudioTextTokenPredictor(
        prosody_dim=len(train_ds.prosody_feature_names),
        text_scalar_dim=len(train_ds.text_scalar_feature_names),
        bert_text_dim=int(train_ds.bert_feature_dim),
        vocab_size=len(train_ds.vocab),
        codebook_sizes={name: part.n_codes for name, part in codec_bundle.vq_parts.items()},
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_text=(args.cond_mode == "audio_text"),
        use_code_hints=(not args.disable_code_hints),
        code_hint_dim=args.code_hint_dim,
        spatial_code_dim=args.spatial_code_dim,
        num_spatial_tasks=max(task.task_id for task in spatial_tasks) + 1,
    ).to(device)

    data_parallel_enabled = bool(args.data_parallel and device.type == "cuda" and args.train_mode != "spatial_pretrain")
    if data_parallel_enabled:
        if not gpu_ids:
            gpu_ids = list(range(torch.cuda.device_count()))
        if len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])

    spatial_load_info = None
    if args.load_spatial_pretrain:
        spatial_load_info = load_spatial_pretrain_weights(unwrap_model(model), args.load_spatial_pretrain)

    params = list(model.parameters())
    if args.finetune_stage1_decoder:
        params.extend(list(codec_bundle.iter_trainable_decoder_parameters()))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=bool(args.num_workers > 0),
        collate_fn=collate_stage2_semantic,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=bool(args.num_workers > 0),
            collate_fn=collate_stage2_semantic,
        )

    fk = None
    hand_joint_ids = None
    if args.fk_hand_pos_w > 0.0 or args.fk_hand_vel_w > 0.0 or args.fk_hand_acc_w > 0.0:
        fk = BVHFkTorch(codec_bundle.skel, drop_root_pos=False).to_device_tensors(device)
        hand_joint_ids = codec_bundle.skel.find_joints_by_keywords(["RightHand", "LeftHand", "Hand"])
        hand_joint_ids = [idx for idx in hand_joint_ids if "End" not in codec_bundle.skel.joints[idx].name]

    args._spatial_fk = None
    args._spatial_hand_joint_ids = None
    if args.spatial_fk_hand_pos_w > 0.0:
        args._spatial_fk = BVHFkTorch(codec_bundle.skel, drop_root_pos=False).to_device_tensors(device)
        args._spatial_hand_joint_ids = codec_bundle.skel.find_joints_by_keywords(["RightHand", "LeftHand", "Hand"])
        args._spatial_hand_joint_ids = [
            idx for idx in args._spatial_hand_joint_ids if "End" not in codec_bundle.skel.joints[idx].name
        ]

    best_metric = float("inf")
    save_path = Path(args.save)
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    train_cache_size_gb = Path(args.train_cache).stat().st_size / float(1024 ** 3)
    print(
        json.dumps(
            {
                "event": "train_start",
                "train_mode": args.train_mode,
                "train_samples": len(train_ds),
                "val_samples": (len(val_ds) if val_ds is not None else 0),
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "device": str(device),
                "amp": amp_enabled,
                "amp_dtype": args.amp_dtype,
                "use_code_hints": (not args.disable_code_hints),
                "bert_feature_dim": int(train_ds.bert_feature_dim),
                "dual_branch": (not args.disable_dual_branch),
                "gen_branch_w": args.gen_branch_w,
                "hint_branch_w": args.hint_branch_w,
                "data_parallel": bool(data_parallel_enabled and len(gpu_ids) > 1),
                "gpu_ids": gpu_ids,
                "train_cache_gb": round(train_cache_size_gb, 2),
                "spatial_tasks": task_name_list(spatial_tasks),
                "spatial_task_weights": {task.name: weight for task, weight in zip(spatial_tasks, spatial_task_weights)},
                "load_spatial_pretrain": args.load_spatial_pretrain,
                "spatial_load_info": spatial_load_info,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        if args.train_mode == "spatial_pretrain":
            train_metrics, _ = run_epoch_spatial(
                model=model,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                codec_bundle=codec_bundle,
                args=args,
                split="train",
                epoch=epoch,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                spatial_tasks=spatial_tasks,
                spatial_task_weights=spatial_task_weights,
                rng=spatial_rng,
            )
        else:
            train_metrics, _ = run_epoch(
                model=model,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                codec_bundle=codec_bundle,
                args=args,
                fk=fk,
                hand_joint_ids=hand_joint_ids,
                split="train",
                epoch=epoch,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
        if not train_metrics:
            raise RuntimeError("Training produced no valid batches.")

        log = {"epoch": epoch, "train": train_metrics}
        score = train_metrics["loss"]

        if val_loader is not None:
            with torch.no_grad():
                if args.train_mode == "spatial_pretrain":
                    val_metrics, debug_payload = run_epoch_spatial(
                        model=model,
                        loader=val_loader,
                        device=device,
                        optimizer=None,
                        scaler=None,
                        codec_bundle=codec_bundle,
                        args=args,
                        split="val",
                        epoch=epoch,
                        amp_enabled=amp_enabled,
                        amp_dtype=amp_dtype,
                        spatial_tasks=spatial_tasks,
                        spatial_task_weights=spatial_task_weights,
                        rng=spatial_rng,
                    )
                else:
                    val_metrics, debug_payload = run_epoch(
                        model=model,
                        loader=val_loader,
                        device=device,
                        optimizer=None,
                        scaler=None,
                        codec_bundle=codec_bundle,
                        args=args,
                        fk=fk,
                        hand_joint_ids=hand_joint_ids,
                        split="val",
                        epoch=epoch,
                        amp_enabled=amp_enabled,
                        amp_dtype=amp_dtype,
                    )
            if val_metrics:
                log["val"] = val_metrics
                score = val_metrics["loss"]
                if debug_payload is not None:
                    if args.train_mode == "spatial_pretrain":
                        torch.save(debug_payload, debug_dir / f"epoch_{epoch:03d}_spatial_debug.pt")
                    else:
                        save_debug_plot(
                            debug_dir / f"epoch_{epoch:03d}.png",
                            prosody_frame=debug_payload["prosody_frame"],
                            token_gate=debug_payload["fusion_gate"],
                            token_codes=debug_payload["token_gt"],
                            token_preds=debug_payload["token_pred"],
                            words=debug_payload["words"],
                            fps=train_ds.meta.get("fps", 15),
                            token_stride=train_ds.token_stride,
                        )

        print(json.dumps(log, ensure_ascii=False))

        save_stage2_ckpt(
            save_path,
            model=model,
            optimizer=optimizer,
            args=args,
            train_ds=train_ds,
            epoch=epoch,
            best_metric=min(best_metric, score),
            codec_bundle=codec_bundle,
            finetuned_decoder=args.finetune_stage1_decoder,
            extra_state={
                "train_mode": args.train_mode,
                "spatial_tasks": task_name_list(spatial_tasks),
                "spatial_num_tasks": max(task.task_id for task in spatial_tasks) + 1,
            },
        )
        if score < best_metric:
            best_metric = score
            best_path = save_path.with_name(save_path.stem + "_best.pt")
            save_stage2_ckpt(
                best_path,
                model=model,
                optimizer=optimizer,
                args=args,
                train_ds=train_ds,
                epoch=epoch,
                best_metric=best_metric,
                codec_bundle=codec_bundle,
                finetuned_decoder=args.finetune_stage1_decoder,
                extra_state={
                    "train_mode": args.train_mode,
                    "spatial_tasks": task_name_list(spatial_tasks),
                    "spatial_num_tasks": max(task.task_id for task in spatial_tasks) + 1,
                },
            )


if __name__ == "__main__":
    main()
