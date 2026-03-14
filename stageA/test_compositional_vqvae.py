import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from stageA.train_stage1_vqvae import (
    TARGET_FPS,
    DEFAULT_FOOT_CONTACT_KEYWORDS,
    DEFAULT_PART_KEYWORDS,
    BVHSkeleton,
    MotionGlobalAE,
    MotionPartSpec,
    MotionVQVAE,
    _resolve,
    _split_csv_keywords,
    build_active_joint_map,
    build_motion_part_spec,
    convert_bvh_to_6d_with_channel_order,
    load_bvh_channels,
    load_ckpt,
    load_state_dict_compat,
    merge_parts_back_to_full,
    prepare_global_condition_input,
    resample_motion_linear,
    resolve_reference_bvh,
    rotation_6d_to_matrix,
    split_motion_into_parts,
    unwrap_bvh_angles_degrees,
)


@dataclass
class PartBundle:
    name: str
    ckpt_path: Path
    ckpt: dict
    motion_spec: MotionPartSpec
    model: MotionVQVAE
    mean: np.ndarray
    std: np.ndarray
    block_size: int
    token_stride: int
    use_vq: bool


def read_bvh_header_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    try:
        motion_idx = lines.index("MOTION")
    except ValueError:
        return lines
    return lines[: motion_idx + 1]


def write_bvh(path_out: Path, header_lines: List[str], motion: np.ndarray, frame_time: float):
    path_out.parent.mkdir(parents=True, exist_ok=True)
    lines = list(header_lines)
    lines.append(f"Frames: {int(motion.shape[0])}")
    lines.append(f"Frame Time: {frame_time:.6f}")
    for row in motion:
        lines.append(" ".join(f"{float(v):.6f}" for v in row))
    with path_out.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def to_numpy_f32(x) -> np.ndarray:
    if x is None:
        raise ValueError("Expected array-like value, got None")
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def infer_source_bvh_from_manifest(manifest_path: Path, sample_idx: int) -> Tuple[Path, dict]:
    items: List[dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    if not items:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")
    if sample_idx < 0 or sample_idx >= len(items):
        raise IndexError(f"sample_idx out of range: idx={sample_idx}, total={len(items)}")

    item = items[sample_idx]
    bvh_rel = item.get("bvh", None) or item.get("motion", None) or item.get("motion_path", None)
    if not bvh_rel:
        raise RuntimeError(f"Manifest item {sample_idx} has no bvh/motion path field.")
    bvh = _resolve(manifest_path.parent, bvh_rel)
    if not bvh.exists():
        raise FileNotFoundError(f"BVH not found for manifest item {sample_idx}: {bvh}")
    return bvh, item


def load_full_motion_from_bvh(bvh_path: Path, skel: BVHSkeleton, active_joint_map) -> np.ndarray:
    raw_motion, frame_time = load_bvh_channels(bvh_path)
    if raw_motion is None or raw_motion.ndim != 2:
        raise RuntimeError(f"Failed to load BVH channels from {bvh_path}")

    raw_motion = unwrap_bvh_angles_degrees(raw_motion, pos_dims=3)
    raw_motion = resample_motion_linear(raw_motion, frame_time, TARGET_FPS)
    full_motion = convert_bvh_to_6d_with_channel_order(raw_motion, skel, active_joint_map).astype(np.float32)
    if not np.isfinite(full_motion).all():
        raise RuntimeError(f"Non-finite full motion after conversion: {bvh_path}")
    return full_motion


def load_full_motion_from_cache(cache_pt: Path, sample_idx: int, full_dim: int) -> Tuple[np.ndarray, Optional[Path], str]:
    pack = torch.load(cache_pt, map_location="cpu", weights_only=False)
    samples = pack.get("samples", None)
    if not isinstance(samples, list) or len(samples) == 0:
        raise RuntimeError(f"Cache has no valid 'samples' list: {cache_pt}")
    if sample_idx < 0 or sample_idx >= len(samples):
        raise IndexError(f"sample_idx out of range: idx={sample_idx}, total={len(samples)}")

    sample = samples[sample_idx]
    sample_id = str(sample.get("id", f"sample_{sample_idx}"))
    bvh_path = None
    if sample.get("bvh"):
        cand = Path(sample["bvh"])
        if cand.exists():
            bvh_path = cand

    full_motion = sample.get("full_y", None)
    if full_motion is None:
        y = sample.get("y", None)
        if y is None:
            raise RuntimeError(f"Cache sample has neither 'y' nor 'full_y': {cache_pt} idx={sample_idx}")
        y_np = to_numpy_f32(y)
        if y_np.ndim != 2:
            raise RuntimeError(f"Cache sample y must be [T, D], got {y_np.shape}")
        if y_np.shape[1] != full_dim:
            raise RuntimeError(
                f"Cache sample does not carry full-body canonical motion. "
                f"Need dim={full_dim}, got={y_np.shape[1]}. Re-cache with part cache that keeps full_y or use full cache."
            )
        full_motion = y_np
    else:
        full_motion = to_numpy_f32(full_motion)

    if full_motion.ndim != 2 or full_motion.shape[1] != full_dim:
        raise RuntimeError(f"Expected full motion [T, {full_dim}], got {full_motion.shape}")
    return full_motion, bvh_path, sample_id


def full_motion_to_bvh_channels(
    full_motion: np.ndarray,
    skel: BVHSkeleton,
    active_joint_map,
) -> np.ndarray:
    T = int(full_motion.shape[0])
    out = np.zeros((T, len(skel.channel_names)), dtype=np.float32)

    for axis_idx, ch_idx in enumerate(active_joint_map.root_pos_indices):
        if ch_idx is not None:
            out[:, ch_idx] = full_motion[:, axis_idx]

    rot_full = full_motion[:, 3:].reshape(T, active_joint_map.n_active, 6)
    rot_mats = rotation_6d_to_matrix(torch.from_numpy(rot_full)).cpu().numpy()

    for active_idx, (order_channels, channel_indices) in enumerate(
        zip(active_joint_map.active_rot_orders, active_joint_map.active_rot_channel_indices)
    ):
        order = "".join(ch[0].upper() for ch in order_channels)
        mats = rot_mats[:, active_idx]
        euler_deg = R.from_matrix(mats).as_euler(order, degrees=True).astype(np.float32)
        euler_rad = np.unwrap(np.deg2rad(euler_deg), axis=0)
        euler_deg = np.rad2deg(euler_rad).astype(np.float32)
        for local_idx, ch_idx in enumerate(channel_indices):
            out[:, ch_idx] = euler_deg[:, local_idx]

    return out


def build_part_bundle(
    part_name: str,
    ckpt_path: Path,
    skel: BVHSkeleton,
    device: torch.device,
) -> PartBundle:
    ckpt = load_ckpt(ckpt_path)
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

    keep_dim = int(ckpt["keep_dim"])
    if keep_dim != motion_spec.model_dim:
        raise RuntimeError(
            f"{part_name} keep_dim mismatch: ckpt={keep_dim}, motion_spec={motion_spec.model_dim}, ckpt={ckpt_path}"
        )

    model_arch = ckpt.get("model_arch", None)
    if model_arch is None and part_name == "global" and not bool(ckpt.get("use_vq", True)) and int(ckpt.get("n_downsample", 0)) == 0:
        model_arch = "motion_global_ae"
    if model_arch == "motion_global_ae":
        model = MotionGlobalAE(
            motion_dim=keep_dim,
            hidden=int(ckpt.get("hidden", 256)),
            n_layers=int(ckpt.get("global_ae_layers", 4)),
        ).to(device)
    else:
        model = MotionVQVAE(
            motion_dim=keep_dim,
            hidden=int(ckpt.get("hidden", 512)),
            code_dim=int(ckpt.get("code_dim", 256)),
            n_codes=int(ckpt.get("n_codes", 1024)),
            beta=float(ckpt.get("beta", 0.25)),
            ema_decay=float(ckpt.get("ema_decay", 0.99)),
            ema_eps=float(ckpt.get("ema_eps", 1e-5)),
            n_downsample=int(ckpt.get("n_downsample", 2)),
            vq_usage_entropy_w=float(ckpt.get("vq_usage_entropy_w", 0.0)),
            vq_usage_temp=float(ckpt.get("vq_usage_temp", 0.5)),
            vq_revive_threshold=float(ckpt.get("vq_revive_threshold", 1.0)),
        ).to(device)
    load_state_dict_compat(model, ckpt["model"], strict=True)
    model.eval()

    mean = to_numpy_f32(ckpt["mean"])
    std = np.maximum(to_numpy_f32(ckpt["std"]), 1e-6)
    if mean.shape[0] != keep_dim or std.shape[0] != keep_dim:
        raise RuntimeError(f"{part_name} mean/std dim mismatch in ckpt: {ckpt_path}")

    return PartBundle(
        name=part_name,
        ckpt_path=ckpt_path,
        ckpt=ckpt,
        motion_spec=motion_spec,
        model=model,
        mean=mean,
        std=std,
        block_size=int(ckpt.get("block_size", 64)),
        token_stride=(1 if model_arch == "motion_global_ae" else 2 ** int(ckpt.get("n_downsample", 2))),
        use_vq=bool(ckpt.get("use_vq", part_name != "global")),
    )


def infer_part_reconstruction(bundle: PartBundle, window_full: np.ndarray, device: torch.device) -> Dict[str, np.ndarray]:
    gt_part = split_motion_into_parts(window_full, bundle.motion_spec, fps=TARGET_FPS).astype(np.float32)
    target_len = int(gt_part.shape[0])
    pad_len = (-target_len) % int(bundle.token_stride)
    if pad_len > 0:
        gt_part_model = np.pad(gt_part, ((0, pad_len), (0, 0)), mode="edge")
    else:
        gt_part_model = gt_part

    mean_t = torch.from_numpy(bundle.mean).to(device=device, dtype=torch.float32).view(1, 1, -1)
    std_t = torch.from_numpy(bundle.std).to(device=device, dtype=torch.float32).view(1, 1, -1)
    x_raw = torch.from_numpy(gt_part_model).to(device=device, dtype=torch.float32).unsqueeze(0)
    if bundle.motion_spec.part == "global":
        x_model_raw = prepare_global_condition_input(x_raw, bundle.motion_spec)
    else:
        x_model_raw = x_raw
    x_norm = (x_model_raw - mean_t) / std_t

    with torch.no_grad():
        x_hat_norm, _, codes, ppl, _ = bundle.model(x_norm, use_vq=bundle.use_vq)
    x_hat = x_hat_norm * std_t + mean_t

    recon_part = x_hat[0, :target_len].detach().cpu().numpy().astype(np.float32)
    codes_np = codes[0].detach().cpu().numpy().astype(np.int64)

    return {
        "gt_part": gt_part,
        "recon_part": recon_part,
        "codes": codes_np,
        "ppl": np.array([float(ppl.item())], dtype=np.float32),
        "pad_len": np.array([pad_len], dtype=np.int32),
    }


def default_output_tag(source_name: str, sample_idx: int, start_frame: int, window_len: int) -> str:
    stem = Path(source_name).stem if source_name else f"sample_{sample_idx}"
    return f"{stem}_idx{sample_idx}_s{start_frame}_l{window_len}"


def main():
    parser = argparse.ArgumentParser(description="Minimal compositional tokenizer test: upper + hand + lower (+ optional global) -> merged full-body.")
    parser.add_argument("--upper_ckpt", type=Path, default=Path("checkpoints/ckpt_upper/stage1_vqvae_best.pt"))
    parser.add_argument("--hand_ckpt", type=Path, default=Path("checkpoints/ckpt_hand/stage1_vqvae_best.pt"))
    parser.add_argument("--lower_ckpt", type=Path, default=Path("checkpoints/ckpt_lower/stage1_vqvae_best.pt"))
    parser.add_argument("--global_ckpt", type=Path, default=None, help="optional global AE checkpoint; omitted by default because a weak global branch can dominate merged quality")

    parser.add_argument("--input_bvh", type=Path, default=Path("beat/beat_english_v0.2.1/1/1_wayne_0_1_1.bvh"), help="direct BVH file to test")
    parser.add_argument("--manifest", type=Path, default=None, help="manifest to pick one sample from")
    parser.add_argument("--cache_pt", type=Path, default=None, help="cache file to pick one sample from")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--window_len", type=int, default=-1, help="defaults to checkpoint block_size; set <=0 to use full sequence")
    parser.add_argument("--ref_bvh", type=str, default=Path("beat/beat_english_v0.2.1/1/1_wayne_0_1_1.bvh"), help="override reference BVH skeleton")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu; default picks cuda when available")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/"))
    args = parser.parse_args()

    data_sources = [args.input_bvh is not None, args.manifest is not None, args.cache_pt is not None]
    if sum(data_sources) != 1:
        raise ValueError("Provide exactly one data source: --input_bvh or --manifest or --cache_pt")

    sample_meta: dict = {}
    source_bvh: Optional[Path] = None
    source_name = ""
    sample_id = f"sample_{args.sample_idx}"

    if args.input_bvh is not None:
        source_bvh = args.input_bvh
        if not source_bvh.exists():
            raise FileNotFoundError(f"input_bvh not found: {source_bvh}")
        source_name = str(source_bvh)
        sample_id = source_bvh.stem
    elif args.manifest is not None:
        source_bvh, sample_meta = infer_source_bvh_from_manifest(args.manifest, args.sample_idx)
        source_name = str(source_bvh)
        sample_id = str(sample_meta.get("id", source_bvh.stem))

    ckpt_refs = []
    ckpt_paths = [args.upper_ckpt, args.hand_ckpt, args.lower_ckpt]
    if args.global_ckpt is not None:
        ckpt_paths.append(args.global_ckpt)
    for ckpt_path in ckpt_paths:
        ckpt = load_ckpt(ckpt_path)
        ckpt_refs.append(ckpt.get("ref_bvh"))

    fallback_ref = next((x for x in ckpt_refs if x), None)
    ref_bvh = resolve_reference_bvh(args.ref_bvh, fallback_ref, args.manifest)
    if ref_bvh is None and source_bvh is not None:
        ref_bvh = source_bvh
    if ref_bvh is None:
        raise RuntimeError("Failed to resolve reference BVH. Pass --ref_bvh explicitly.")

    skel = BVHSkeleton.from_bvh(ref_bvh)
    active_joint_map = build_active_joint_map(skel)
    full_dim = 3 + active_joint_map.n_active * 6

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Reference BVH: {ref_bvh}")

    bundles = {
        "upper": build_part_bundle("upper", args.upper_ckpt, skel, device),
        "hand": build_part_bundle("hand", args.hand_ckpt, skel, device),
        "lower": build_part_bundle("lower", args.lower_ckpt, skel, device),
    }
    if args.global_ckpt is not None:
        bundles["global"] = build_part_bundle("global", args.global_ckpt, skel, device)

    block_sizes = {name: bundle.block_size for name, bundle in bundles.items()}
    if len(set(block_sizes.values())) != 1:
        raise RuntimeError(f"Checkpoint block_size mismatch: {block_sizes}")

    for part_name, bundle in bundles.items():
        print(
            f"[INFO] Part[{part_name}] dim={bundle.motion_spec.model_dim}, "
            f"joints={bundle.motion_spec.part_joint_names}, ckpt={bundle.ckpt_path}"
        )

    if args.cache_pt is not None:
        full_motion, cache_bvh, sample_id = load_full_motion_from_cache(args.cache_pt, args.sample_idx, full_dim)
        if source_bvh is None:
            source_bvh = cache_bvh
        source_name = str(source_bvh) if source_bvh is not None else f"cache_idx{args.sample_idx}"
    else:
        if source_bvh is None:
            raise RuntimeError("Expected a BVH source path from --input_bvh or --manifest.")
        full_motion = load_full_motion_from_bvh(source_bvh, skel, active_joint_map)

    window_len = block_sizes["upper"] if args.window_len is None else int(args.window_len)
    if window_len <= 0:
        window_len = int(full_motion.shape[0])
    if full_motion.shape[0] < window_len:
        raise RuntimeError(f"Motion too short: frames={full_motion.shape[0]}, requested={window_len}")

    start_frame = int(args.start_frame)
    if start_frame < 0 or (start_frame + window_len) > int(full_motion.shape[0]):
        raise RuntimeError(
            f"Invalid crop: start={start_frame}, len={window_len}, total={full_motion.shape[0]}"
        )
    window_full = full_motion[start_frame:start_frame + window_len].astype(np.float32, copy=False)

    merged_full = window_full.copy()
    outputs: Dict[str, Dict[str, np.ndarray]] = {}
    metrics: Dict[str, dict] = {}

    part_order = ["upper", "hand", "lower"] + (["global"] if "global" in bundles else [])
    for part_name in part_order:
        bundle = bundles[part_name]
        out = infer_part_reconstruction(bundle, window_full, device)
        merged_full = merge_parts_back_to_full(out["recon_part"], bundle.motion_spec, gt_full=merged_full)
        outputs[part_name] = out
        if bundle.motion_spec.part == "global":
            global_full = merge_parts_back_to_full(out["recon_part"], bundle.motion_spec, gt_full=window_full)
            root_l1 = float(np.mean(np.abs(global_full[:, :3] - window_full[:, :3])))
            contact_slice = bundle.motion_spec.foot_contact_slice
            contact_l1 = 0.0
            if contact_slice is not None:
                contact_l1 = float(np.mean(np.abs(out["recon_part"][:, contact_slice] - out["gt_part"][:, contact_slice])))
            part_l1 = root_l1
        else:
            root_l1 = None
            contact_l1 = None
            part_l1 = float(np.mean(np.abs(out["recon_part"] - out["gt_part"])))
        metrics[part_name] = {
            "ppl": float(out["ppl"][0]),
            "active_codes": int(np.unique(out["codes"]).size) if bundle.use_vq else 0,
            "token_len": int(out["codes"].shape[0]),
            "pad_len": int(out["pad_len"][0]),
            "l1": part_l1,
            "dim": int(out["gt_part"].shape[1]),
            "joint_names": bundle.motion_spec.part_joint_names,
        }
        if root_l1 is not None:
            metrics[part_name]["root_l1"] = root_l1
            metrics[part_name]["contact_l1"] = contact_l1

    metrics["merged"] = {
        "full_l1": float(np.mean(np.abs(merged_full - window_full))),
        "rot_l1": float(np.mean(np.abs(merged_full[:, 3:] - window_full[:, 3:]))),
    }

    tag = default_output_tag(source_name, args.sample_idx, start_frame, window_len)
    out_dir = args.out_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "compositional_recon.npz"
    npz_payload = dict(
        gt_full=window_full,
        recon_full_merged=merged_full,
        gt_upper=outputs["upper"]["gt_part"],
        recon_upper=outputs["upper"]["recon_part"],
        codes_upper=outputs["upper"]["codes"],
        gt_hand=outputs["hand"]["gt_part"],
        recon_hand=outputs["hand"]["recon_part"],
        codes_hand=outputs["hand"]["codes"],
        gt_lower=outputs["lower"]["gt_part"],
        recon_lower=outputs["lower"]["recon_part"],
        codes_lower=outputs["lower"]["codes"],
        active_joint_names=np.asarray(active_joint_map.active_joint_names, dtype=str),
        upper_joint_names=np.asarray(bundles["upper"].motion_spec.part_joint_names, dtype=str),
        hand_joint_names=np.asarray(bundles["hand"].motion_spec.part_joint_names, dtype=str),
        lower_joint_names=np.asarray(bundles["lower"].motion_spec.part_joint_names, dtype=str),
    )
    if "global" in outputs:
        global_root_recon = merge_parts_back_to_full(outputs["global"]["recon_part"], bundles["global"].motion_spec, gt_full=window_full)
        npz_payload.update(
            gt_global=outputs["global"]["gt_part"],
            recon_global=outputs["global"]["recon_part"],
            codes_global=outputs["global"]["codes"],
            gt_root_xyz=window_full[:, :3],
            recon_root_xyz=global_root_recon[:, :3],
            global_joint_names=np.asarray(bundles["global"].motion_spec.part_joint_names, dtype=str),
        )
    np.savez(npz_path, **npz_payload)

    header_source = source_bvh if source_bvh is not None and source_bvh.exists() else ref_bvh
    header_lines = read_bvh_header_lines(header_source)
    gt_bvh_motion = full_motion_to_bvh_channels(window_full, skel, active_joint_map)
    rec_bvh_motion = full_motion_to_bvh_channels(merged_full, skel, active_joint_map)
    gt_bvh_path = out_dir / "gt_window.bvh"
    rec_bvh_path = out_dir / "recon_merged.bvh"
    write_bvh(gt_bvh_path, header_lines, gt_bvh_motion, frame_time=1.0 / float(TARGET_FPS))
    write_bvh(rec_bvh_path, header_lines, rec_bvh_motion, frame_time=1.0 / float(TARGET_FPS))

    summary = {
        "sample_id": sample_id,
        "source_bvh": str(source_bvh) if source_bvh is not None else None,
        "ref_bvh": str(ref_bvh),
        "window": {"start_frame": start_frame, "window_len": window_len, "fps": TARGET_FPS},
        "metrics": metrics,
        "outputs": {
            "npz": str(npz_path),
            "gt_bvh": str(gt_bvh_path),
            "recon_bvh": str(rec_bvh_path),
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved compositional test outputs to {out_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
