import argparse
import math
import os
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusion.latent.common import collate_motion_batch, get_rank, is_dist, is_main, make_autocast, make_grad_scaler, set_seed
from diffusion.latent.contracts import AudioFeatureSpec, MotionContract, TextTokenSpec
from diffusion.latent.data import MotionCacheV3Dataset
from diffusion.latent.io import save_checkpoint, serialize_args_dict
from diffusion.latent.losses import MotionLossContext, build_motion_loss_context, compute_vae_losses
from diffusion.latent.models import MotionVAE
from diffusion.latent.parts import resolve_part_feature_indices


def ddp_reduce_scalar_dict(metrics: Dict[str, float], device: torch.device) -> Dict[str, float]:
    if not is_dist():
        return metrics
    out: Dict[str, float] = {}
    for key, value in metrics.items():
        tensor = torch.tensor([float(value)], device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        out[key] = float(tensor.item() / dist.get_world_size())
    return out


def maybe_init_distributed() -> torch.device:
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loss_context(dataset: MotionCacheV3Dataset, device: torch.device) -> MotionLossContext:
    return build_motion_loss_context(
        motion_contract=dataset.motion_contract,
        mean=dataset.mean,
        std=dataset.std,
        device=device,
    )


def resolve_part_layout(part_aware: bool) -> str:
    return "root_upper_hand_lower_slots_v3" if part_aware else "legacy_root_v1"


def resolve_slot_packed_latent(part_aware: bool, part_layout: str, enabled: bool) -> bool:
    return bool(enabled and part_aware and part_layout == "root_upper_hand_lower_slots_v3")


def run_epoch(
    model: MotionVAE,
    loader: DataLoader,
    *,
    optimizer,
    scaler,
    context: MotionLossContext,
    device: torch.device,
    amp_enabled: bool,
    train: bool,
    global_step: int,
    kl_max: float,
    kl_warmup_steps: int,
    recon_weight: float,
    root_weight: float,
    root_relative_first_frame: bool,
    contact_weight: float,
    rot_weight: float,
    upper_rot_weight: float,
    upper_rot_vel_weight: float,
    upper_rot_speed_weight: float,
    vel_weight: float,
    acc_weight: float,
    foot_fk_weight: float,
    hand_fk_weight: float,
) -> tuple[Dict[str, float], int]:
    model.train(train)
    totals = {
        k: 0.0
        for k in [
            "loss",
            "recon",
            "root",
            "contact",
            "rot",
            "upper_rot",
            "upper_rot_vel",
            "upper_rot_speed",
            "vel",
            "acc",
            "foot_fk",
            "hand_fk",
            "kl",
            "kl_contrib",
            "mu_rms",
            "std_mean",
        ]
    }
    steps = 0
    iterator = tqdm(loader, disable=not is_main(), desc="train" if train else "val")

    for batch in iterator:
        motion_norm = batch.motion_norm.to(device, non_blocking=True)
        motion_denorm = batch.motion_denorm.to(device, non_blocking=True)
        motion_mask = batch.motion_mask.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with make_autocast(device, amp_enabled):
            outputs = model(motion_norm, motion_mask)
            if train and kl_warmup_steps > 0:
                kl_weight = float(kl_max) * min(1.0, float(global_step + 1) / float(kl_warmup_steps))
            else:
                kl_weight = float(kl_max)
            losses = compute_vae_losses(
                recon_motion_norm=outputs["recon_motion_norm"],
                motion_norm=motion_norm,
                motion_denorm=motion_denorm,
                motion_mask=motion_mask,
                mu=outputs["mu"],
                logvar=outputs["logvar"],
                context=context,
                kl_weight=kl_weight,
                recon_weight=recon_weight,
                root_weight=root_weight,
                root_relative_first_frame=root_relative_first_frame,
                contact_weight=contact_weight,
                rot_weight=rot_weight,
                upper_rot_weight=upper_rot_weight,
                upper_rot_vel_weight=upper_rot_vel_weight,
                upper_rot_speed_weight=upper_rot_speed_weight,
                vel_weight=vel_weight,
                acc_weight=acc_weight,
                foot_fk_weight=foot_fk_weight,
                hand_fk_weight=hand_fk_weight,
            )
            loss = losses["loss"]
            kl_contrib = losses["kl"] * float(kl_weight)
            mu_rms = outputs["mu"].square().mean().sqrt()
            std_mean = torch.exp(0.5 * outputs["logvar"]).mean()

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
            global_step += 1

        for key in totals:
            if key == "kl_contrib":
                totals[key] += float(kl_contrib.item())
            elif key == "mu_rms":
                totals[key] += float(mu_rms.item())
            elif key == "std_mean":
                totals[key] += float(std_mean.item())
            else:
                totals[key] += float(losses[key].item())
        steps += 1

        if is_main():
            iterator.set_postfix(
                loss=float(loss.item()),
                kl=float(losses["kl"].item()),
                klc=float(kl_contrib.item()),
                mu=float(mu_rms.item()),
                std=float(std_mean.item()),
            )

    metrics = {key: value / max(1, steps) for key, value in totals.items()}
    metrics = ddp_reduce_scalar_dict(metrics, device)
    return metrics, global_step


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", "--data", dest="cache", type=str, required=True)
    parser.add_argument("--val_cache", "--val_data", dest="val_cache", type=str, default=None)
    parser.add_argument("--save_dir", type=Path, default=Path("diffusion/checkpoints/latent"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--latent_stride", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--deterministic_ae", action="store_true")
    parser.add_argument("--part_aware", action="store_true")
    parser.add_argument("--slot_packed_latent", action="store_true")
    parser.add_argument("--kl_max", type=float, default=1e-4)
    parser.add_argument("--kl_warmup_steps", type=int, default=5000)
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--root_weight", type=float, default=1.0)
    parser.add_argument("--root_relative_first_frame", action="store_true")
    parser.add_argument("--contact_weight", type=float, default=1.0)
    parser.add_argument("--rot_weight", type=float, default=0.0)
    parser.add_argument("--upper_rot_weight", type=float, default=0.0)
    parser.add_argument("--upper_rot_vel_weight", type=float, default=0.0)
    parser.add_argument("--upper_rot_speed_weight", type=float, default=0.0)
    parser.add_argument("--vel_weight", type=float, default=0.1)
    parser.add_argument("--acc_weight", type=float, default=0.05)
    parser.add_argument("--foot_fk_weight", type=float, default=0.05)
    parser.add_argument("--hand_fk_weight", type=float, default=0.05)
    parser.add_argument("--overfit_n", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args(argv)

    device = maybe_init_distributed()
    amp_enabled = bool(args.amp and device.type == "cuda")
    set_seed(args.seed + get_rank())

    train_ds = MotionCacheV3Dataset(args.cache, continuation_prob=0.0)
    if args.overfit_n > 0:
        train_ds = Subset(train_ds, list(range(min(len(train_ds), int(args.overfit_n)))))
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
    raw_train_ds = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    val_loss_context = None
    if args.val_cache:
        val_ds = MotionCacheV3Dataset(args.val_cache, continuation_prob=0.0)
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
        val_loss_context = build_loss_context(val_ds, device)

    part_layout = resolve_part_layout(args.part_aware)
    slot_packed_latent = resolve_slot_packed_latent(args.part_aware, part_layout, args.slot_packed_latent)
    part_feature_indices = (
        resolve_part_feature_indices(raw_train_ds.motion_contract, part_layout=part_layout)
        if args.part_aware
        else None
    )
    model = MotionVAE(
        motion_dim=raw_train_ds.motion_contract.motion_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        latent_stride=args.latent_stride,
        dropout=args.dropout,
        deterministic_ae=args.deterministic_ae,
        part_aware=args.part_aware,
        joint_names=list(raw_train_ds.motion_contract.layout_meta.get("joint_names") or []),
        rot6d_start=int(raw_train_ds.motion_contract.rot6d_start),
        part_layout=part_layout,
        part_feature_indices=part_feature_indices,
        slot_packed_latent=slot_packed_latent,
    ).to(device)
    if is_dist():
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(device, amp_enabled)
    loss_context = build_loss_context(raw_train_ds, device)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics, global_step = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            context=loss_context,
            device=device,
            amp_enabled=amp_enabled,
            train=True,
            global_step=global_step,
            kl_max=args.kl_max,
            kl_warmup_steps=args.kl_warmup_steps,
            recon_weight=args.recon_weight,
            root_weight=args.root_weight,
            root_relative_first_frame=args.root_relative_first_frame,
            contact_weight=args.contact_weight,
            rot_weight=args.rot_weight,
            upper_rot_weight=args.upper_rot_weight,
            upper_rot_vel_weight=args.upper_rot_vel_weight,
            upper_rot_speed_weight=args.upper_rot_speed_weight,
            vel_weight=args.vel_weight,
            acc_weight=args.acc_weight,
            foot_fk_weight=args.foot_fk_weight,
            hand_fk_weight=args.hand_fk_weight,
        )

        if is_main():
            print(f"[Epoch {epoch}] train {train_metrics}")

        val_metrics = None
        if val_loader is not None:
            val_metrics, _ = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                scaler=scaler,
                context=val_loss_context or loss_context,
                device=device,
                amp_enabled=False,
                train=False,
                global_step=global_step,
                kl_max=args.kl_max,
                kl_warmup_steps=args.kl_warmup_steps,
                recon_weight=args.recon_weight,
                root_weight=args.root_weight,
                root_relative_first_frame=args.root_relative_first_frame,
                contact_weight=args.contact_weight,
                rot_weight=args.rot_weight,
                upper_rot_weight=args.upper_rot_weight,
                upper_rot_vel_weight=args.upper_rot_vel_weight,
                upper_rot_speed_weight=args.upper_rot_speed_weight,
                vel_weight=args.vel_weight,
                acc_weight=args.acc_weight,
                foot_fk_weight=args.foot_fk_weight,
                hand_fk_weight=args.hand_fk_weight,
            )
            if is_main():
                print(f"[Epoch {epoch}] val {val_metrics}")

        score = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
        raw_model = model.module if isinstance(model, DDP) else model
        payload = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "config": serialize_args_dict(args),
            "contract": {
                "motion_contract": raw_train_ds.motion_contract.to_dict(),
                "audio_feature_spec": raw_train_ds.audio_feature_spec.to_dict(),
                "text_token_spec": raw_train_ds.text_token_spec.to_dict(),
            },
            "motion_mean": raw_train_ds.mean,
            "motion_std": raw_train_ds.std,
            "audio_mean": raw_train_ds.audio_mean,
            "audio_std": raw_train_ds.audio_std,
            "vae_spec": {
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
                "num_encoder_layers": args.num_encoder_layers,
                "num_decoder_layers": args.num_decoder_layers,
                "num_heads": args.num_heads,
                "latent_stride": args.latent_stride,
                "dropout": args.dropout,
                "deterministic_ae": bool(args.deterministic_ae),
                "part_aware": bool(args.part_aware),
                "rot6d_start": int(raw_train_ds.motion_contract.rot6d_start),
                "part_layout": part_layout,
                "slot_packed_latent": bool(slot_packed_latent),
                "slot_part_names": list(getattr(raw_model, "part_names", ()) if slot_packed_latent else ()),
                "slot_part_count": int(len(getattr(raw_model, "part_names", ()))) if slot_packed_latent else 0,
                "slot_latent_dim": int(getattr(raw_model, "slot_latent_dim", 0)) if slot_packed_latent else 0,
            },
            "vae_loss_spec": {
                "recon_weight": args.recon_weight,
                "root_weight": args.root_weight,
                "root_relative_first_frame": bool(args.root_relative_first_frame),
                "contact_weight": args.contact_weight,
                "rot_weight": args.rot_weight,
                "upper_rot_weight": args.upper_rot_weight,
                "upper_rot_vel_weight": args.upper_rot_vel_weight,
                "upper_rot_speed_weight": args.upper_rot_speed_weight,
                "vel_weight": args.vel_weight,
                "acc_weight": args.acc_weight,
                "foot_fk_weight": args.foot_fk_weight,
                "hand_fk_weight": args.hand_fk_weight,
                "kl_max": args.kl_max,
            },
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        if is_main():
            save_checkpoint(args.save_dir / "motion_vae.pt", payload)
            if score < best_val:
                best_val = score
                save_checkpoint(args.save_dir / "motion_vae_best.pt", payload)
            if epoch % max(1, args.save_every) == 0:
                save_checkpoint(args.save_dir / f"motion_vae_ep{epoch}.pt", payload)

    if is_dist():
        dist.barrier()


if __name__ == "__main__":
    main()
