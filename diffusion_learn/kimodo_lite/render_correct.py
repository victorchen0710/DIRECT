"""Render BVH motion to MP4 using the BVH file's REAL parent topology.

src/render.py hard-codes a 15-joint simplified skeleton which mis-connects
BEAT's 88-joint rig. This module reads the true parents from the BVH hierarchy
and connects joints correctly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from stageA.train_stage1_vqvae import BVHFkTorch, BVHSkeleton, TARGET_FPS
from stageB.common.bvh_io import load_full_motion_from_bvh


def temporal_gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """1D Gaussian smooth along the time axis (axis=0). Reflection-padded."""
    if sigma <= 0:
        return x
    radius = max(1, int(round(3 * sigma)))
    k = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(k ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    T = x.shape[0]
    # reflection pad along time
    pad_hi = x[-2 : -radius - 2 : -1] if radius > 0 else x[:0]
    pad_lo = x[1 : radius + 1][::-1]
    padded = np.concatenate([pad_lo, x, pad_hi], axis=0)
    out = np.empty_like(x)
    # convolve along time axis
    for t in range(T):
        window = padded[t : t + 2 * radius + 1]
        out[t] = (window * kernel.reshape((-1,) + (1,) * (x.ndim - 1))).sum(axis=0)
    return out


def compute_joint_positions(bvh_path: Path, max_frames: int | None = None) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Returns (joint_pos [T,J,3], parent_child_pairs)."""
    skel = BVHSkeleton.from_bvh(bvh_path)
    full_motion = load_full_motion_from_bvh(bvh_path)
    if max_frames is not None:
        full_motion = full_motion[: int(max_frames)]
    fk = BVHFkTorch(skel, drop_root_pos=False).to_device_tensors(torch.device("cpu"))
    pos = fk.fk_positions(torch.from_numpy(full_motion).unsqueeze(0).float())[0].cpu().numpy()
    pairs = [(j.parent, i) for i, j in enumerate(skel.joints) if j.parent >= 0]
    return pos, pairs


def render_motion(
    joint_pos: np.ndarray,
    parent_pairs: List[Tuple[int, int]],
    save_path: Path,
    *,
    fps: int = 30,
    title: str = "",
) -> None:
    T, J, _ = joint_pos.shape

    # Matplotlib default is Z-up; BVH is Y-up. Swap so character stands upright.
    # xyz (Y-up) → (x, -z, y) for Z-up rendering
    pts = np.stack([joint_pos[..., 0], -joint_pos[..., 2], joint_pos[..., 1]], axis=-1)

    pad = 20.0
    xs, ys, zs = pts[..., 0], pts[..., 1], pts[..., 2]
    x_mid, y_mid, z_mid = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, (zs.min() + zs.max()) / 2
    max_range = max(xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()) / 2 + pad

    fig = plt.figure(figsize=(7, 7), dpi=110)
    fig.patch.set_facecolor("#0b0f19")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0b0f19")
    ax.grid(False)
    ax.set_axis_off()
    ax.view_init(elev=12, azim=60)
    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)
    if title:
        fig.suptitle(title, color="#cccccc", fontsize=11)

    scat = ax.scatter([], [], [], c="#ff3366", s=10)
    bones = [ax.plot([], [], [], color="#00f2ea", linewidth=1.5)[0] for _ in parent_pairs]

    def update(t):
        scat._offsets3d = (pts[t, :, 0], pts[t, :, 1], pts[t, :, 2])
        for (p, c), ln in zip(parent_pairs, bones):
            ln.set_data([pts[t, p, 0], pts[t, c, 0]], [pts[t, p, 1], pts[t, c, 1]])
            ln.set_3d_properties([pts[t, p, 2], pts[t, c, 2]])
        return [scat, *bones]

    print(f"[render] writing {save_path} ({T} frames, {J} joints, {len(parent_pairs)} bones)")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    anim = animation.FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=1500, codec="libx264")
    anim.save(str(save_path), writer=writer, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_bvh", type=Path, required=True)
    ap.add_argument("--output_mp4", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=TARGET_FPS)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--smooth_sigma", type=float, default=0.0,
                    help="temporal Gaussian smooth sigma in frames on joint positions (0 = off)")
    args = ap.parse_args()
    pos, pairs = compute_joint_positions(args.input_bvh, args.max_frames)
    if args.smooth_sigma > 0:
        pos = temporal_gaussian_smooth(pos, args.smooth_sigma)
        print(f"[render] applied temporal Gaussian smooth sigma={args.smooth_sigma}")
    render_motion(pos, pairs, args.output_mp4, fps=args.fps, title=args.input_bvh.stem)


if __name__ == "__main__":
    main()
