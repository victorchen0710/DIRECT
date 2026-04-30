import argparse
import math
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusion.latent.common import EMA, collate_motion_batch, get_rank, is_dist, is_main, make_autocast, make_grad_scaler, set_seed
from diffusion.latent.contracts import load_checkpoint_contract
from diffusion.latent.data import MotionCacheV3Dataset
from diffusion.latent.io import save_checkpoint, serialize_args_dict
from diffusion.latent.losses import build_motion_loss_context, compute_motion_recon_losses
from diffusion.latent.models import LatentRectifiedFlowTransformer, MotionRefiner, MotionVAE
from diffusion.latent.parts import resolve_part_feature_indices


def maybe_init_distributed() -> torch.device:
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ddp_reduce_scalar_dict(metrics: Dict[str, float], device: torch.device) -> Dict[str, float]:
    if not is_dist():
        return metrics
    out = {}
    for key, value in metrics.items():
        t = torch.tensor([float(value)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        out[key] = float(t.item() / dist.get_world_size())
    return out


def load_vae(path: str | Path, device: torch.device) -> Tuple[MotionVAE, Dict[str, object]]:
    loaded = load_checkpoint_contract(path)
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
    for param in model.parameters():
        param.requires_grad_(False)
    return model, loaded


def extract_prefix_latents(
    latents: torch.Tensor,
    latent_mask: torch.Tensor,
    continuation_mask: Optional[torch.Tensor],
    latent_stride: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    if continuation_mask is None:
        return None, None, latent_mask

    prefix_counts = []
    for row in continuation_mask:
        if bool(row.any().item()):
            prefix_frames = int((~row).sum().item())
        else:
            prefix_frames = 0
        prefix_counts.append(int(math.ceil(prefix_frames / float(latent_stride))))

    max_prefix = max(prefix_counts)
    if max_prefix <= 0:
        return None, None, latent_mask

    prefix_latent = latents.new_zeros((latents.shape[0], max_prefix, latents.shape[-1]))
    prefix_mask = torch.zeros((latents.shape[0], max_prefix), device=latents.device, dtype=torch.bool)
    target_mask = latent_mask.clone()
    for batch_idx, count in enumerate(prefix_counts):
        if count <= 0:
            continue
        prefix_latent[batch_idx, :count] = latents[batch_idx, :count]
        prefix_mask[batch_idx, :count] = True
        target_mask[batch_idx, :count] = False
    return prefix_latent, prefix_mask, target_mask


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.unsqueeze(-1).float()
    return ((pred - target).square() * weight).sum() / (weight.sum() * pred.shape[-1] + 1e-6)


def masked_cosine(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    cosine = torch.nn.functional.cosine_similarity(pred, target, dim=-1)
    weight = mask.float()
    return (cosine * weight).sum() / (weight.sum() + 1e-6)


def run_epoch(
    model: LatentRectifiedFlowTransformer,
    loader: DataLoader,
    *,
    vae: MotionVAE,
    motion_refiner: Optional[MotionRefiner],
    loss_context,
    optimizer,
    scaler,
    ema: Optional[EMA],
    device: torch.device,
    amp_enabled: bool,
    beta_alpha: float,
    beta_beta: float,
    latent_loss_weight: float,
    motion_recon_weight: float,
    motion_root_weight: float,
    motion_root_relative_first_frame: bool,
    motion_rot_weight: float,
    motion_upper_rot_weight: float,
    motion_upper_rot_vel_weight: float,
    motion_upper_rot_speed_weight: float,
    motion_upper_rot_acc_weight: float,
    motion_upper_rot_jerk_weight: float,
    motion_vel_weight: float,
    motion_acc_weight: float,
    train: bool,
) -> Dict[str, float]:
    model.train(train)
    if motion_refiner is not None:
        motion_refiner.train(train)
    totals = {
        "loss": 0.0,
        "latent_loss": 0.0,
        "motion_loss": 0.0,
        "motion_recon": 0.0,
        "motion_root": 0.0,
        "motion_rot": 0.0,
        "motion_upper_rot": 0.0,
        "motion_upper_rot_vel": 0.0,
        "motion_upper_rot_speed": 0.0,
        "motion_upper_rot_acc": 0.0,
        "motion_upper_rot_jerk": 0.0,
        "motion_vel": 0.0,
        "motion_acc": 0.0,
        "pred_rms": 0.0,
        "target_rms": 0.0,
        "cosine": 0.0,
    }
    steps = 0
    beta_dist = torch.distributions.Beta(beta_alpha, beta_beta)
    iterator = tqdm(loader, disable=not is_main(), desc="train" if train else "val")
    aux_enabled = any(
        weight > 0.0
        for weight in (
            motion_recon_weight,
            motion_root_weight,
            motion_rot_weight,
            motion_upper_rot_weight,
            motion_upper_rot_vel_weight,
            motion_upper_rot_speed_weight,
            motion_upper_rot_acc_weight,
            motion_upper_rot_jerk_weight,
            motion_vel_weight,
            motion_acc_weight,
        )
    )

    for batch in iterator:
        motion_norm = batch.motion_norm.to(device, non_blocking=True)
        motion_denorm = batch.motion_denorm.to(device, non_blocking=True)
        motion_mask = batch.motion_mask.to(device, non_blocking=True)
        audio = batch.audio.to(device, non_blocking=True)
        audio_mask = batch.audio_mask.to(device, non_blocking=True)
        lexical = batch.lexical_frame.to(device, non_blocking=True)
        lexical_mask = batch.lexical_mask.to(device, non_blocking=True)
        global_text = batch.global_text.to(device, non_blocking=True)
        word_frame = batch.word_frame.to(device, non_blocking=True)
        word_mask = batch.word_mask.to(device, non_blocking=True)
        cont_mask = batch.continuation_mask.to(device, non_blocking=True) if batch.continuation_mask is not None else None

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            vae_outputs = vae(motion_norm, motion_mask)
            clean_latents = vae_outputs["latents"].detach()
            latent_mask = vae_outputs["latent_mask"].detach()

        prefix_latent, prefix_mask, target_mask = extract_prefix_latents(
            clean_latents,
            latent_mask,
            cont_mask,
            vae.latent_stride,
        )

        t = beta_dist.sample((clean_latents.shape[0],)).to(device=device, dtype=clean_latents.dtype)
        t_view = t.view(-1, 1, 1)
        noise = torch.randn_like(clean_latents)
        x_t = (1.0 - t_view) * clean_latents + t_view * noise
        velocity_target = noise - clean_latents

        with make_autocast(device, amp_enabled):
            pred = model(
                x_t,
                t,
                latent_mask,
                audio=audio,
                audio_mask=audio_mask,
                lexical_frame=lexical,
                lexical_mask=lexical_mask,
                global_text=global_text,
                word_frame=word_frame,
                word_mask=word_mask,
                prefix_latent=prefix_latent,
                prefix_mask=prefix_mask,
            )
            latent_loss = masked_mse(pred, velocity_target, target_mask)
            motion_loss = latent_loss.new_tensor(0.0)
            motion_terms = None
            if aux_enabled:
                pred_clean_latents = x_t - t_view * pred
                pred_motion_norm = vae.decode(pred_clean_latents, latent_mask, target_len=motion_norm.shape[1])
                if motion_refiner is not None:
                    pred_motion_norm = motion_refiner(
                        pred_motion_norm,
                        motion_mask,
                        audio=audio,
                        audio_mask=audio_mask,
                        lexical_frame=lexical,
                        lexical_mask=lexical_mask,
                        word_frame=word_frame,
                        word_mask=word_mask,
                        global_text=global_text,
                    )
                motion_target_mask = (cont_mask & motion_mask) if cont_mask is not None else motion_mask
                motion_terms = compute_motion_recon_losses(
                    pred_motion_norm=pred_motion_norm,
                    motion_denorm=motion_denorm,
                    motion_mask=motion_target_mask,
                    context=loss_context,
                    recon_weight=motion_recon_weight,
                    root_weight=motion_root_weight,
                    root_relative_first_frame=motion_root_relative_first_frame,
                    contact_weight=0.0,
                    rot_weight=motion_rot_weight,
                    upper_rot_weight=motion_upper_rot_weight,
                    upper_rot_vel_weight=motion_upper_rot_vel_weight,
                    upper_rot_speed_weight=motion_upper_rot_speed_weight,
                    upper_rot_acc_weight=motion_upper_rot_acc_weight,
                    upper_rot_jerk_weight=motion_upper_rot_jerk_weight,
                    vel_weight=motion_vel_weight,
                    acc_weight=motion_acc_weight,
                    foot_fk_weight=0.0,
                    hand_fk_weight=0.0,
                )
                motion_loss = motion_terms["loss"]
            loss = float(latent_loss_weight) * latent_loss + motion_loss
            pred_rms = pred.square().mean().sqrt()
            target_rms = velocity_target.square().mean().sqrt()
            cosine = masked_cosine(pred, velocity_target, target_mask)

        if train:
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if ema is not None:
                ema.update(model.module if isinstance(model, DDP) else model)

        totals["loss"] += float(loss.item())
        totals["latent_loss"] += float(latent_loss.item())
        totals["motion_loss"] += float(motion_loss.item())
        totals["motion_recon"] += 0.0 if motion_terms is None else float(motion_terms["recon"].item())
        totals["motion_root"] += 0.0 if motion_terms is None else float(motion_terms["root"].item())
        totals["motion_rot"] += 0.0 if motion_terms is None else float(motion_terms["rot"].item())
        totals["motion_upper_rot"] += 0.0 if motion_terms is None else float(motion_terms["upper_rot"].item())
        totals["motion_upper_rot_vel"] += 0.0 if motion_terms is None else float(motion_terms["upper_rot_vel"].item())
        totals["motion_upper_rot_speed"] += 0.0 if motion_terms is None else float(motion_terms["upper_rot_speed"].item())
        totals["motion_upper_rot_acc"] += 0.0 if motion_terms is None else float(motion_terms["upper_rot_acc"].item())
        totals["motion_upper_rot_jerk"] += 0.0 if motion_terms is None else float(motion_terms["upper_rot_jerk"].item())
        totals["motion_vel"] += 0.0 if motion_terms is None else float(motion_terms["vel"].item())
        totals["motion_acc"] += 0.0 if motion_terms is None else float(motion_terms["acc"].item())
        totals["pred_rms"] += float(pred_rms.item())
        totals["target_rms"] += float(target_rms.item())
        totals["cosine"] += float(cosine.item())
        steps += 1
        if is_main():
            iterator.set_postfix(
                loss=float(loss.item()),
                lat=float(latent_loss.item()),
                aux=float(motion_loss.item()),
                cos=float(cosine.item()),
            )

    metrics = {key: value / max(1, steps) for key, value in totals.items()}
    return ddp_reduce_scalar_dict(metrics, device)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", "--data", dest="cache", type=str, required=True)
    parser.add_argument("--val_cache", "--val_data", dest="val_cache", type=str, default=None)
    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--save_dir", type=Path, default=Path("diffusion/checkpoints/latent"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=8, help="Number of single-stream MMDiT blocks")
    parser.add_argument("--num_double_layers", type=int, default=2, help="Number of dual-stream motion/condition blocks")
    parser.add_argument("--token_refiner_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--audio_drop_prob", type=float, default=0.1)
    parser.add_argument("--text_drop_prob", type=float, default=0.1)
    parser.add_argument("--word_drop_prob", type=float, default=0.1)
    parser.add_argument("--global_text_drop_prob", type=float, default=0.1)
    parser.add_argument("--beta_alpha", type=float, default=2.0)
    parser.add_argument("--beta_beta", type=float, default=1.2)
    parser.add_argument("--latent_loss_weight", type=float, default=1.0)
    parser.add_argument("--motion_recon_weight", type=float, default=0.0)
    parser.add_argument("--motion_root_weight", type=float, default=0.0)
    parser.add_argument("--motion_root_relative_first_frame", action="store_true")
    parser.add_argument("--motion_rot_weight", type=float, default=0.0)
    parser.add_argument("--motion_upper_rot_weight", type=float, default=0.0)
    parser.add_argument("--motion_upper_rot_vel_weight", type=float, default=0.0)
    parser.add_argument("--motion_upper_rot_speed_weight", type=float, default=0.0)
    parser.add_argument("--motion_upper_rot_acc_weight", type=float, default=0.0)
    parser.add_argument("--motion_upper_rot_jerk_weight", type=float, default=0.0)
    parser.add_argument("--motion_vel_weight", type=float, default=0.0)
    parser.add_argument("--motion_acc_weight", type=float, default=0.0)
    parser.add_argument("--use_motion_refiner", action="store_true")
    parser.add_argument("--refiner_hidden_dim", type=int, default=256)
    parser.add_argument("--refiner_layers", type=int, default=3)
    parser.add_argument("--refiner_heads", type=int, default=4)
    parser.add_argument("--refiner_dropout", type=float, default=0.0)
    parser.add_argument("--continuation_prob", type=float, default=0.5)
    parser.add_argument("--max_prefix_ratio", type=float, default=0.25)
    parser.add_argument("--local_attn_window", type=int, default=15, help="Local motion attention window in latent tokens; <=0 disables")
    parser.add_argument("--no_rope", action="store_true")
    parser.add_argument("--slot_tokenized_flow", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--overfit_n", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args(argv)

    device = maybe_init_distributed()
    amp_enabled = bool(args.amp and device.type == "cuda")
    set_seed(args.seed + get_rank())

    vae, vae_loaded = load_vae(args.vae_ckpt, device)
    motion_contract = vae_loaded["motion_contract"]
    audio_spec = vae_loaded["audio_feature_spec"]
    text_spec = vae_loaded["text_token_spec"]
    if args.slot_tokenized_flow and not bool(getattr(vae, "slot_packed_latent", False)):
        raise RuntimeError("--slot_tokenized_flow requires a VAE checkpoint trained with slot_packed_latent")

    train_ds = MotionCacheV3Dataset(
        args.cache,
        continuation_prob=args.continuation_prob,
        max_prefix_ratio=args.max_prefix_ratio,
    )
    if args.overfit_n > 0:
        train_ds = Subset(train_ds, list(range(min(len(train_ds), int(args.overfit_n)))))
    raw_train_ds = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    if raw_train_ds.motion_contract.to_dict() != motion_contract.to_dict():
        raise RuntimeError("Cache motion contract does not match VAE checkpoint contract")
    if raw_train_ds.audio_feature_spec.to_dict() != audio_spec.to_dict():
        raise RuntimeError("Cache audio_feature_spec does not match VAE checkpoint contract")
    if raw_train_ds.text_token_spec.to_dict() != text_spec.to_dict():
        raise RuntimeError("Cache text_token_spec does not match VAE checkpoint contract")
    loss_context = build_motion_loss_context(
        motion_contract=raw_train_ds.motion_contract,
        mean=raw_train_ds.mean,
        std=raw_train_ds.std,
        device=device,
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_dist() else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_motion_batch,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = None
    val_loss_context = None
    if args.val_cache:
        val_ds = MotionCacheV3Dataset(args.val_cache, continuation_prob=0.0)
        if val_ds.motion_contract.to_dict() != motion_contract.to_dict():
            raise RuntimeError("Validation cache motion contract does not match VAE checkpoint contract")
        if val_ds.audio_feature_spec.to_dict() != audio_spec.to_dict():
            raise RuntimeError("Validation cache audio_feature_spec does not match VAE checkpoint contract")
        if val_ds.text_token_spec.to_dict() != text_spec.to_dict():
            raise RuntimeError("Validation cache text_token_spec does not match VAE checkpoint contract")
        val_loss_context = build_motion_loss_context(
            motion_contract=val_ds.motion_contract,
            mean=val_ds.mean,
            std=val_ds.std,
            device=device,
        )
        val_sampler = DistributedSampler(val_ds, shuffle=False) if is_dist() else None
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_motion_batch,
            persistent_workers=args.num_workers > 0,
        )

    model = LatentRectifiedFlowTransformer(
        latent_dim=vae.latent_dim,
        audio_dim=audio_spec.dim,
        lexical_dim=text_spec.lexical_dim,
        global_text_dim=text_spec.global_text_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_double_layers=args.num_double_layers,
        token_refiner_layers=args.token_refiner_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        audio_drop_prob=args.audio_drop_prob,
        text_drop_prob=args.text_drop_prob,
        word_drop_prob=args.word_drop_prob,
        global_text_drop_prob=args.global_text_drop_prob,
        local_attn_window=args.local_attn_window,
        use_rope=not args.no_rope,
        slot_tokenized=args.slot_tokenized_flow,
        slot_part_names=list(getattr(vae, "part_names", ())) if args.slot_tokenized_flow else None,
    ).to(device)
    motion_refiner = None
    if args.use_motion_refiner:
        motion_refiner = MotionRefiner(
            motion_dim=motion_contract.motion_dim,
            audio_dim=audio_spec.dim,
            lexical_dim=text_spec.lexical_dim,
            global_text_dim=text_spec.global_text_dim,
            hidden_dim=args.refiner_hidden_dim,
            num_layers=args.refiner_layers,
            num_heads=args.refiner_heads,
            dropout=args.refiner_dropout,
        ).to(device)
    if is_dist():
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
        if motion_refiner is not None:
            motion_refiner = DDP(motion_refiner, device_ids=[device.index] if device.type == "cuda" else None)

    params = list(model.parameters())
    if motion_refiner is not None:
        params.extend(list(motion_refiner.parameters()))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(device, amp_enabled)
    ema = EMA(model.module if isinstance(model, DDP) else model, decay=args.ema_decay) if args.ema else None

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            vae=vae,
            motion_refiner=motion_refiner,
            loss_context=loss_context,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            device=device,
            amp_enabled=amp_enabled,
            beta_alpha=args.beta_alpha,
            beta_beta=args.beta_beta,
            latent_loss_weight=args.latent_loss_weight,
            motion_recon_weight=args.motion_recon_weight,
            motion_root_weight=args.motion_root_weight,
            motion_root_relative_first_frame=args.motion_root_relative_first_frame,
            motion_rot_weight=args.motion_rot_weight,
            motion_upper_rot_weight=args.motion_upper_rot_weight,
            motion_upper_rot_vel_weight=args.motion_upper_rot_vel_weight,
            motion_upper_rot_speed_weight=args.motion_upper_rot_speed_weight,
            motion_upper_rot_acc_weight=args.motion_upper_rot_acc_weight,
            motion_upper_rot_jerk_weight=args.motion_upper_rot_jerk_weight,
            motion_vel_weight=args.motion_vel_weight,
            motion_acc_weight=args.motion_acc_weight,
            train=True,
        )
        if is_main():
            print(f"[Epoch {epoch}] train {train_metrics}")

        val_metrics = None
        if val_loader is not None:
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                vae=vae,
                motion_refiner=motion_refiner,
                loss_context=val_loss_context if val_loss_context is not None else loss_context,
                optimizer=optimizer,
                scaler=scaler,
                ema=None,
                device=device,
                amp_enabled=False,
                beta_alpha=args.beta_alpha,
                beta_beta=args.beta_beta,
                latent_loss_weight=args.latent_loss_weight,
                motion_recon_weight=args.motion_recon_weight,
                motion_root_weight=args.motion_root_weight,
                motion_root_relative_first_frame=args.motion_root_relative_first_frame,
                motion_rot_weight=args.motion_rot_weight,
                motion_upper_rot_weight=args.motion_upper_rot_weight,
                motion_upper_rot_vel_weight=args.motion_upper_rot_vel_weight,
                motion_upper_rot_speed_weight=args.motion_upper_rot_speed_weight,
                motion_upper_rot_acc_weight=args.motion_upper_rot_acc_weight,
                motion_upper_rot_jerk_weight=args.motion_upper_rot_jerk_weight,
                motion_vel_weight=args.motion_vel_weight,
                motion_acc_weight=args.motion_acc_weight,
                train=False,
            )
            if is_main():
                print(f"[Epoch {epoch}] val {val_metrics}")

        score = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
        raw_model = model.module if isinstance(model, DDP) else model
        raw_refiner = None if motion_refiner is None else (motion_refiner.module if isinstance(motion_refiner, DDP) else motion_refiner)
        payload = {
            "epoch": int(epoch),
            "model": raw_model.state_dict(),
            "motion_refiner": None if raw_refiner is None else raw_refiner.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "ema": None if ema is None else ema.state_dict(),
            "config": serialize_args_dict(args),
            "contract": {
                "motion_contract": motion_contract.to_dict(),
                "audio_feature_spec": audio_spec.to_dict(),
                "text_token_spec": text_spec.to_dict(),
            },
            "vae_checkpoint": str(args.vae_ckpt),
            "vae_spec": vae_loaded["checkpoint"]["vae_spec"],
            "flow_spec": {
                "latent_dim": vae.latent_dim,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "num_double_layers": args.num_double_layers,
                "token_refiner_layers": args.token_refiner_layers,
                "num_heads": args.num_heads,
                "dropout": args.dropout,
                "beta_alpha": args.beta_alpha,
                "beta_beta": args.beta_beta,
                "latent_loss_weight": args.latent_loss_weight,
                "motion_recon_weight": args.motion_recon_weight,
                "motion_root_weight": args.motion_root_weight,
                "motion_root_relative_first_frame": args.motion_root_relative_first_frame,
                "motion_rot_weight": args.motion_rot_weight,
                "motion_upper_rot_weight": args.motion_upper_rot_weight,
                "motion_upper_rot_vel_weight": args.motion_upper_rot_vel_weight,
                "motion_upper_rot_speed_weight": args.motion_upper_rot_speed_weight,
                "motion_upper_rot_acc_weight": args.motion_upper_rot_acc_weight,
                "motion_upper_rot_jerk_weight": args.motion_upper_rot_jerk_weight,
                "motion_vel_weight": args.motion_vel_weight,
                "motion_acc_weight": args.motion_acc_weight,
                "global_text_dim": text_spec.global_text_dim,
                "audio_drop_prob": args.audio_drop_prob,
                "text_drop_prob": args.text_drop_prob,
                "word_drop_prob": args.word_drop_prob,
                "global_text_drop_prob": args.global_text_drop_prob,
                "local_attn_window": args.local_attn_window,
                "use_rope": not args.no_rope,
                "slot_tokenized_flow": args.slot_tokenized_flow,
                "slot_part_names": list(getattr(vae, "part_names", ())) if args.slot_tokenized_flow else None,
                "slot_part_count": len(getattr(vae, "part_names", ())) if args.slot_tokenized_flow else 0,
                "slot_latent_dim": getattr(vae, "slot_latent_dim", vae.latent_dim) if args.slot_tokenized_flow else vae.latent_dim,
                "use_motion_refiner": bool(args.use_motion_refiner),
                "refiner_hidden_dim": args.refiner_hidden_dim,
                "refiner_layers": args.refiner_layers,
                "refiner_heads": args.refiner_heads,
                "refiner_dropout": args.refiner_dropout,
            },
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        if is_main():
            save_checkpoint(args.save_dir / "gesture_flow.pt", payload)
            if score < best_val:
                best_val = score
                save_checkpoint(args.save_dir / "gesture_flow_best.pt", payload)
            if epoch % max(1, args.save_every) == 0:
                save_checkpoint(args.save_dir / f"gesture_flow_ep{epoch}.pt", payload)

    if is_dist():
        dist.barrier()


if __name__ == "__main__":
    main()
