from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.render import render_motion_to_mp4
from stageA.train_stage1_vqvae import BVHFkTorch, BVHSkeleton, TARGET_FPS, build_active_joint_map
from stageB.common.bvh_io import load_full_motion_from_bvh


def render_bvh_to_mp4(
    input_bvh: Path,
    output_mp4: Path,
    *,
    fps: int = TARGET_FPS,
    max_frames: int | None = None,
) -> dict:
    skel = BVHSkeleton.from_bvh(input_bvh)
    active_joint_map = build_active_joint_map(skel)
    full_motion = load_full_motion_from_bvh(
        input_bvh,
        skel=skel,
        active_joint_map=active_joint_map,
        fps=int(fps),
    )
    if max_frames is not None:
        full_motion = full_motion[: int(max_frames)]
    fk = BVHFkTorch(skel, drop_root_pos=False).to_device_tensors(torch.device("cpu"))
    joint_pos = fk.fk_positions(torch.from_numpy(full_motion).unsqueeze(0).float())[0].cpu().numpy()
    render_motion_to_mp4(joint_pos, str(output_mp4), fps=int(fps), title=input_bvh.stem)
    return {
        "input_bvh": str(input_bvh.resolve()),
        "output_mp4": str(output_mp4.resolve()),
        "frames": int(joint_pos.shape[0]),
        "joints": int(joint_pos.shape[1]),
        "fps": int(fps),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a BVH file to an MP4 preview.")
    parser.add_argument("--input_bvh", type=str, required=True)
    parser.add_argument("--output_mp4", type=str, required=True)
    parser.add_argument("--fps", type=int, default=TARGET_FPS)
    parser.add_argument("--max_frames", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = render_bvh_to_mp4(
        Path(args.input_bvh),
        Path(args.output_mp4),
        fps=int(args.fps),
        max_frames=args.max_frames,
    )
    print(summary)


if __name__ == "__main__":
    main()
