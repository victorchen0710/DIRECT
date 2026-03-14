from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stageB.common.motion_codec import load_motion_codec_bundle
from stageB.data_semantic import Stage2SemanticDataset
from stageB.models.audio_text_token_predictor import AudioTextTokenPredictor
from stageB.models.spatial_tasks import build_spatial_tasks, task_to_part_masks


def resolve_stage1_kwargs(args, ckpt: dict, ds: Stage2SemanticDataset) -> dict:
    stage1 = ckpt.get("stage1", ds.stage1 or {})
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check LoM-lite spatial pretraining on cached StageB codes.")
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--spatial_ckpt", type=str, required=True)
    parser.add_argument("--task", type=str, default="upper_to_hand")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="outputs/stageB_spatial_check")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ref_bvh", type=str, default=None)
    parser.add_argument("--stage1_ckpt", type=str, default=None)
    parser.add_argument("--upper_ckpt", type=str, default=None)
    parser.add_argument("--hand_ckpt", type=str, default=None)
    parser.add_argument("--lower_ckpt", type=str, default=None)
    parser.add_argument("--global_ckpt", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    ds = Stage2SemanticDataset(Path(args.cache))
    ckpt = torch.load(args.spatial_ckpt, map_location="cpu", weights_only=False)
    task = build_spatial_tasks(args.task)[0]

    codec_bundle = load_motion_codec_bundle(device=device, **resolve_stage1_kwargs(args, ckpt, ds))
    model = AudioTextTokenPredictor(
        prosody_dim=len(ds.prosody_feature_names),
        text_scalar_dim=len(ds.text_scalar_feature_names),
        bert_text_dim=int(ds.bert_feature_dim),
        vocab_size=len(ds.vocab),
        codebook_sizes={name: part.n_codes for name, part in codec_bundle.vq_parts.items()},
        d_model=int(ckpt["args"]["d_model"]),
        nhead=int(ckpt["args"]["nhead"]),
        num_layers=int(ckpt["args"]["num_layers"]),
        dropout=float(ckpt["args"]["dropout"]),
        use_text=(ckpt["args"].get("cond_mode", "audio_text") == "audio_text"),
        use_code_hints=not bool(ckpt["args"].get("disable_code_hints", False)),
        code_hint_dim=int(ckpt["args"].get("code_hint_dim", 128)),
        spatial_code_dim=int(ckpt["args"].get("spatial_code_dim", 128)),
        num_spatial_tasks=int(ckpt.get("spatial_num_tasks", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    sample = ds[int(args.sample_idx)]
    code_targets = {name: tensor.unsqueeze(0).to(device) for name, tensor in sample["codes"].items()}
    source_mask, target_mask = task_to_part_masks(task, model.part_names, batch_size=1, device=device)
    task_ids = torch.tensor([int(task.task_id)], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model.forward_spatial(
            code_targets=code_targets,
            source_part_mask=source_mask,
            target_part_mask=target_mask,
            task_ids=task_ids,
        )

    pred_codes = {name: torch.argmax(outputs["logits"][name], dim=-1) for name in task.target_parts}
    decoded_target = {
        name: codec_bundle.parts[name].decode_codes(pred_codes[name])[0].detach().cpu().numpy().astype(np.float32)
        for name in task.target_parts
    }
    gt_target = {
        name: sample["part_motion"][name].detach().cpu().numpy().astype(np.float32)
        for name in task.target_parts
    }
    full_motion = sample["full_motion"].detach().cpu().numpy().astype(np.float32)
    pred_full = codec_bundle.merge_decoded_parts(decoded_target, gt_full=full_motion).astype(np.float32)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "task": task.name,
            "sample_idx": int(args.sample_idx),
            "source_parts": task.source_parts,
            "target_parts": task.target_parts,
            "gt_codes": {name: code_targets[name].detach().cpu()[0] for name in task.target_parts},
            "pred_codes": {name: pred_codes[name].detach().cpu()[0] for name in task.target_parts},
            "meta": sample["meta"],
        },
        out_dir / "codes.pt",
    )
    for name in task.target_parts:
        np.save(out_dir / f"{name}_gt_motion.npy", gt_target[name])
        np.save(out_dir / f"{name}_pred_motion.npy", decoded_target[name])
    np.save(out_dir / "full_motion_gt.npy", full_motion)
    np.save(out_dir / "full_motion_pred.npy", pred_full)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "task": task.name,
                "sample_idx": int(args.sample_idx),
                "source_parts": task.source_parts,
                "target_parts": task.target_parts,
                "out_dir": str(out_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
