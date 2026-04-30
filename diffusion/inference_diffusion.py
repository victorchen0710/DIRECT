import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Optional: smooth
try:
    from scipy.ndimage import gaussian_filter1d
except Exception:
    gaussian_filter1d = None

# Rotation utils
try:
    from scipy.spatial.transform import Rotation as R
except Exception:
    R = None

# Whisper (OpenAI whisper)
try:
    import whisper
except Exception:
    whisper = None

# BERT tokenizer
try:
    from transformers import BertTokenizer
except Exception:
    BertTokenizer = None

# Safe import model
try:
    from diffusion.feature_layout import MotionFeatureLayout
except Exception:
    from feature_layout import MotionFeatureLayout

try:
    from diffusion.diffusion_policy import MotionDiffusionTransformer, DDPMScheduler
except Exception:
    from diffusion_policy import MotionDiffusionTransformer, DDPMScheduler


# =============================================================================
# Utils
# =============================================================================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _is_finite_np(x: np.ndarray) -> bool:
    return np.isfinite(x).all()


def randn_like_gen(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    # 兼容老 torch：randn_like 不支持 generator
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)

# =============================================================================
# Rotation: 6D(col0+col1) -> matrix (robust) + optional SO(3) projection
# =============================================================================
def rotation_6d_to_matrix_col_safe(d6: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    6D rotation (col0+col1) -> 3x3 matrix (columns), with degenerate fallback.
    Input:  [..., 6]
    Output: [..., 3, 3]
    """
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]

    a1_norm = torch.linalg.norm(a1, dim=-1, keepdim=True)
    b1 = a1 / torch.clamp(a1_norm, min=eps)

    proj = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = a2 - proj * b1
    b2_norm = torch.linalg.norm(b2, dim=-1, keepdim=True)
    b2 = b2 / torch.clamp(b2_norm, min=eps)

    b3 = torch.cross(b1, b2, dim=-1)
    b3_norm = torch.linalg.norm(b3, dim=-1, keepdim=True)

    mat = torch.stack((b1, b2, b3), dim=-1)  # [...,3,3]

    bad = (a1_norm.squeeze(-1) < eps) | (b2_norm.squeeze(-1) < eps) | (b3_norm.squeeze(-1) < eps)
    bad = bad | (~torch.isfinite(mat).all(dim=(-1, -2)))

    if bad.any():
        I = torch.eye(3, device=d6.device, dtype=d6.dtype)
        expand_shape = list(mat.shape)
        I = I.view(*([1] * (len(expand_shape) - 2)), 3, 3).expand(*expand_shape)
        mat = torch.where(bad[..., None, None], I, mat)

    return mat


def project_matrix_to_so3(mat: torch.Tensor) -> torch.Tensor:
    """
    Project arbitrary 3x3 to nearest rotation matrix via SVD (polar decomposition).
    mat: [...,3,3]
    return: [...,3,3] with det=+1
    """
    U, _, Vh = torch.linalg.svd(mat)
    Rm = U @ Vh
    det = torch.det(Rm)

    if (det < 0).any():
        Vh_fix = Vh.clone()
        mask = det < 0
        # torch.linalg.svd supports batching, so mask indexes first batch dim(s)
        Vh_fix[mask, 2, :] *= -1.0
        Rm = U @ Vh_fix

    return Rm


def project_rot6d_mean_to_valid(
    mean_vec: np.ndarray,
    motion_dim: int,
    rot6d_start: int,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Projects mean rotation-6D slice to a valid rotation (SO(3)) per joint.
    mean_vec: [D]
    layout: [velx, posy, velz, yaw_vel, contact4, rot6d...]
    => rot6d_start should be 8 for your cache.
    """
    assert mean_vec.shape[0] == motion_dim
    if motion_dim <= rot6d_start:
        return mean_vec

    rot_dim = motion_dim - rot6d_start
    if rot_dim % 6 != 0:
        return mean_vec

    J = rot_dim // 6
    mean_out = mean_vec.copy()
    rot6 = mean_out[rot6d_start:].reshape(J, 6).astype(np.float32)

    d6 = torch.from_numpy(rot6)  # [J,6]
    mat = rotation_6d_to_matrix_col_safe(d6, eps=eps)  # [J,3,3]
    mat = project_matrix_to_so3(mat)                   # [J,3,3]

    col0 = mat[:, :, 0]
    col1 = mat[:, :, 1]
    rot6_proj = torch.cat([col0, col1], dim=-1).cpu().numpy().astype(np.float32)  # [J,6]
    mean_out[rot6d_start:] = rot6_proj.reshape(-1)
    return mean_out


# =============================================================================
# BVH: strict channel remap
# =============================================================================
def parse_bvh_channel_blocks(ref_bvh_path: str):
    with open(ref_bvh_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    header_lines: List[str] = []
    blocks: List[Dict[str, Any]] = []
    curr_joint: Optional[str] = None

    for ln in lines:
        s = ln.strip()
        header_lines.append(ln)
        if s == "MOTION":
            break

        parts = s.split()
        if not parts:
            continue
        if parts[0] in ("ROOT", "JOINT"):
            curr_joint = parts[1]
        elif parts[0] == "CHANNELS" and curr_joint is not None:
            n = int(parts[1])
            chans = parts[2: 2 + n]
            pos_ch = [c for c in chans if c.endswith("position")]
            rot_ch = [c for c in chans if c.endswith("rotation")]
            blocks.append(
                dict(
                    name=curr_joint,
                    channels=chans,
                    has_pos=(len(pos_ch) > 0),
                    has_rot=(len(rot_ch) > 0),
                )
            )

    return header_lines, blocks


def count_rot_joints_in_bvh(ref_bvh_path: str) -> int:
    _, blocks = parse_bvh_channel_blocks(ref_bvh_path)
    return int(sum(1 for b in blocks if b.get("has_rot", False)))


def read_ref_root_first_frame_xyz(ref_bvh_path: str) -> Optional[Tuple[float, float, float]]:
    try:
        with open(ref_bvh_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
        motion_idx = lines.index("MOTION")

        first_data = None
        for ln in lines[motion_idx + 3:]:
            if ln.strip():
                first_data = ln.strip()
                break
        if first_data is None:
            return None

        _, blocks = parse_bvh_channel_blocks(ref_bvh_path)

        vals = [float(x) for x in first_data.split()]
        cursor = 0
        for b in blocks:
            if b["has_pos"]:
                xyz = {"Xposition": 0.0, "Yposition": 0.0, "Zposition": 0.0}
                for ch in b["channels"]:
                    if cursor >= len(vals):
                        break
                    if ch in xyz:
                        xyz[ch] = float(vals[cursor])
                    cursor += 1
                return (xyz["Xposition"], xyz["Yposition"], xyz["Zposition"])
            else:
                cursor += len(b["channels"])
        return None
    except Exception:
        return None


def save_bvh_remapped(
    root_pos: np.ndarray,          # (T,3) [X,Y,Z]
    euler_rots: np.ndarray,        # (T,J,3) [X,Y,Z] degrees
    ref_bvh_path: str,
    output_path: str,
    fps: int,
    strict_bvh: bool = True,
):
    print(f"[INFO] Saving BVH -> {output_path}")
    out_p = Path(output_path)
    ensure_dir(out_p.parent)

    T = int(root_pos.shape[0])
    header_lines, blocks = parse_bvh_channel_blocks(ref_bvh_path)
    frame_time = 1.0 / float(fps)

    used_joints = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for ln in header_lines:
            f.write(ln.rstrip("\n") + "\n")

        f.write(f"Frames: {T}\n")
        f.write(f"Frame Time: {frame_time:.8f}\n")

        for t in range(T):
            rot_ptr = 0
            line_vals: List[float] = []

            for b in blocks:
                if b["has_pos"]:
                    cur_pos = root_pos[t]
                else:
                    cur_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                if b["has_rot"]:
                    if rot_ptr < euler_rots.shape[1]:
                        cur_euler = euler_rots[t, rot_ptr]
                        rot_ptr += 1
                    else:
                        cur_euler = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                else:
                    cur_euler = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                for ch in b["channels"]:
                    if ch == "Xposition":
                        line_vals.append(float(cur_pos[0]))
                    elif ch == "Yposition":
                        line_vals.append(float(cur_pos[1]))
                    elif ch == "Zposition":
                        line_vals.append(float(cur_pos[2]))
                    elif ch == "Xrotation":
                        line_vals.append(float(cur_euler[0]))
                    elif ch == "Yrotation":
                        line_vals.append(float(cur_euler[1]))
                    elif ch == "Zrotation":
                        line_vals.append(float(cur_euler[2]))
                    else:
                        line_vals.append(0.0 if strict_bvh else 0.0)

            used_joints = max(used_joints, min(rot_ptr, euler_rots.shape[1]))
            f.write(" ".join(f"{v:.6f}" for v in line_vals) + "\n")

    print(f"[INFO] BVH saved. Rot joints used per frame = {used_joints}/{euler_rots.shape[1]}")


# =============================================================================
# Whisper cache (JSON)
# =============================================================================
def whisper_cache_path(cache_dir: Path, audio_path: str, whisper_model: str) -> Path:
    ap = Path(audio_path)
    name = f"{ap.stem}__whisper_{whisper_model}.json"
    return cache_dir / name


def load_or_run_whisper(
    audio_path: str,
    whisper_model: str,
    cache_dir: Path,
    language: Optional[str] = None,
    word_timestamps: bool = True,
) -> Dict[str, Any]:
    ensure_dir(cache_dir)
    cpath = whisper_cache_path(cache_dir, audio_path, whisper_model)

    if cpath.exists():
        try:
            obj = json.loads(cpath.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and "segments" in obj:
                print(f"[INFO] Loaded Whisper cache: {cpath}")
                return obj
        except Exception:
            pass

    print(f"[INFO] Running Whisper ({whisper_model}) on {audio_path}")
    if whisper is None:
        raise ImportError("whisper is required for inference segmentation. Please install openai-whisper in the inference environment.")
    asr = whisper.load_model(whisper_model)

    kwargs = {}
    if language:
        kwargs["language"] = language

    try:
        result = asr.transcribe(audio_path, word_timestamps=bool(word_timestamps), **kwargs)
    except TypeError:
        result = asr.transcribe(audio_path, **kwargs)

    segments = result.get("segments", [])
    payload = {
        "audio_path": str(audio_path),
        "whisper_model": str(whisper_model),
        "language": language,
        "segments": segments,
    }
    cpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Whisper cache saved: {cpath}")
    return payload


def build_segments_from_whisper(
    whisper_obj: Dict[str, Any],
    max_duration: float = 8.0,
    gap_threshold: float = 0.35,
) -> List[Dict[str, Any]]:
    segments = whisper_obj.get("segments", [])
    if not segments:
        return [{"start": 0.0, "end": 3.0, "text": "Hello world.", "words": []}]

    all_words = []
    for seg in segments:
        ws = seg.get("words", None)
        if isinstance(ws, list) and len(ws) > 0:
            for w in ws:
                if "start" in w and "end" in w:
                    all_words.append(w)

    # Prefer word-level regrouping if available
    if len(all_words) > 0:
        out = []
        cur_words = []
        cur_start = float(all_words[0]["start"])
        last_end = float(all_words[0]["end"])

        def flush(end_t: float):
            nonlocal cur_words, cur_start
            if cur_words:
                text = "".join([str(cw.get("word", "")) for cw in cur_words]).strip()
                text = " ".join(text.split())
                if len(text) > 0 and end_t > cur_start:
                    out.append({"start": float(cur_start), "end": float(end_t), "text": text, "words": list(cur_words)})
            cur_words = []

        for w in all_words:
            w_start = float(w["start"])
            w_end = float(w["end"])
            word_str = str(w.get("word", "")).strip()

            gap = w_start - last_end
            dur = w_end - cur_start
            is_break = bool(word_str) and word_str.endswith((".", "?", "!", "。", "？", "！"))

            if not cur_words:
                cur_start = w_start

            cur_words.append(w)
            last_end = w_end

            if (gap > gap_threshold) or (dur >= max_duration) or is_break:
                flush(last_end)
                cur_start = w_end

        flush(last_end)
        out = [s for s in out if (s["end"] - s["start"]) >= 0.3 and len(s["text"]) > 0]
        if len(out) > 0:
            return out

    # fallback: segment-level
    out = []
    for seg in segments:
        try:
            st = float(seg.get("start", 0.0))
            ed = float(seg.get("end", st + 1.0))
            tx = str(seg.get("text", "")).strip()
            words = seg.get("words", []) if isinstance(seg.get("words", []), list) else []
            if ed > st and len(tx) > 0:
                out.append({"start": st, "end": ed, "text": tx, "words": words})
        except Exception:
            continue

    if len(out) == 0:
        out = [{"start": 0.0, "end": 3.0, "text": "Hello world.", "words": []}]
    return out


# =============================================================================
# wav2vec2 feature loader (npz) + optional normalize
# =============================================================================
def load_w2v2_feature(npz_path: Path):
    npz_path = Path(npz_path)
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            audio_fps = None
            if "fps" in z:
                try:
                    audio_fps = float(np.asarray(z["fps"]).reshape(-1)[0])
                except Exception:
                    audio_fps = None

            duration = None
            if "duration" in z:
                try:
                    duration = float(np.asarray(z["duration"]).reshape(-1)[0])
                except Exception:
                    duration = None

            feat = None
            candidate_keys = ["feat", "feats", "features", "hidden_states", "last_hidden_state", "x", "emb", "embedding"]
            for k in candidate_keys:
                if k in z:
                    try:
                        feat = np.asarray(z[k], dtype=np.float32)
                        break
                    except Exception:
                        feat = None

            if feat is None:
                for k in z.files:
                    if k in ("fps", "duration"):
                        continue
                    try:
                        arr = z[k]
                    except Exception:
                        continue
                    if hasattr(arr, "ndim") and arr.ndim >= 2:
                        try:
                            feat = np.asarray(arr, dtype=np.float32)
                            break
                        except Exception:
                            feat = None

            if feat is None:
                return None, None, None

            if feat.ndim == 3 and feat.shape[0] == 1:
                feat = feat[0]
            if feat.ndim != 2:
                return None, None, None

            if audio_fps is None and duration is not None and duration > 0 and feat.shape[0] > 1:
                audio_fps = float(feat.shape[0]) / float(duration)

            if audio_fps is None or audio_fps <= 0:
                return None, None, None

            if not _is_finite_np(feat):
                return None, None, None

            return feat, float(audio_fps), duration
    except Exception:
        return None, None, None


def maybe_normalize_audio(
    w2v2_feat: np.ndarray,
    audio_mean: Optional[np.ndarray],
    audio_std: Optional[np.ndarray],
) -> np.ndarray:
    if audio_mean is None or audio_std is None:
        return w2v2_feat
    audio_mean = np.asarray(audio_mean, dtype=np.float32).reshape(-1)
    audio_std = np.asarray(audio_std, dtype=np.float32).reshape(-1)
    audio_std = np.where(audio_std < 1e-6, 1.0, audio_std).astype(np.float32)

    if w2v2_feat.shape[1] != audio_mean.shape[0] or w2v2_feat.shape[1] != audio_std.shape[0]:
        raise ValueError(
            f"audio stats dim mismatch: feat_dim={w2v2_feat.shape[1]}, "
            f"mean_dim={audio_mean.shape[0]}, std_dim={audio_std.shape[0]}"
        )

    return (w2v2_feat - audio_mean[None, :]) / audio_std[None, :]


# =============================================================================
# Build per-segment word_times tensor (relative seconds)
# =============================================================================
def build_word_times_for_segment(seg: Dict[str, Any], seg_start: float, seg_end: float) -> np.ndarray:
    """
    From Whisper word list (absolute seconds), build relative times in [0, seg_dur].
    Output: (N,2) float32
    """
    words = seg.get("words", [])
    if not isinstance(words, list) or len(words) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    out = []
    for w in words:
        try:
            st = float(w.get("start", None))
            ed = float(w.get("end", None))
            if not (np.isfinite(st) and np.isfinite(ed)):
                continue
            if ed <= seg_start or st >= seg_end:
                continue
            st = max(st, seg_start)
            ed = min(ed, seg_end)
            if ed <= st + 1e-4:
                continue
            out.append([st - seg_start, ed - seg_start])
        except Exception:
            continue

    if len(out) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    return np.asarray(out, dtype=np.float32)


# =============================================================================
# V-parameterization formulas
# =============================================================================
@torch.no_grad()
def predict_x0_eps_from_v(
    scheduler: DDPMScheduler,
    x_t: torch.Tensor,
    v: torch.Tensor,
    t_index: torch.Tensor,  # [B]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    v-parameterization:
      x0 = sqrt(ab) * x_t - sqrt(1-ab) * v
      eps= sqrt(1-ab)* x_t + sqrt(ab) * v
    """
    sqrt_ab = scheduler.sqrt_alphas_cumprod[t_index].view(-1, 1, 1)
    sqrt_1mab = scheduler.sqrt_one_minus_alphas_cumprod[t_index].view(-1, 1, 1)
    x0 = sqrt_ab * x_t - sqrt_1mab * v
    eps = sqrt_1mab * x_t + sqrt_ab * v
    return x0, eps


# =============================================================================
# Sampling (DDIM / DDPM) with v-prediction + overlap stitching
# =============================================================================
def build_timestep_schedule(num_train_steps: int, num_steps: int) -> List[int]:
    """
    Return a descending list of timestep indices, inclusive ends.
    If num_steps >= num_train_steps => full [T-1..0]
    else uniform subsample with endpoints.
    """
    N = int(num_train_steps)
    S = int(num_steps)
    if S >= N:
        return list(range(N - 1, -1, -1))

    idx = np.linspace(0, N - 1, S, dtype=np.int64)
    idx = np.unique(idx)
    ts = idx.tolist()[::-1]
    if ts[0] != N - 1:
        ts = [N - 1] + ts
    if ts[-1] != 0:
        ts = ts + [0]
    return ts


@torch.no_grad()
def sample_one_segment_v(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    *,
    text_ids: torch.Tensor,
    text_mask: torch.Tensor,
    motion_mask: torch.Tensor,
    audio: Optional[torch.Tensor],
    audio_mask: Optional[torch.Tensor],
    word_times: Optional[torch.Tensor],
    word_mask: Optional[torch.Tensor],
    n_frames: int,
    motion_dim: int,
    sampler: str,
    num_steps: int,
    eta: float,
    temperature: float,
    overlap_prev_x0: Optional[torch.Tensor],
    overlap: int,
    generator: torch.Generator,
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    """
    Returns x0 in normalized space: [1, n_frames, D]
    overlap_prev_x0: previous segment final x0_norm (clean), [1, Tprev, D]
    """
    assert sampler in ("ddim", "ddpm")
    x = torch.randn((1, n_frames, motion_dim), device=device, generator=generator, dtype=torch.float32)

    # timestep schedule
    ts = build_timestep_schedule(scheduler.num_timesteps, num_steps)

    # prepare autocast
    autocast_ctx = None
    if amp and device.type == "cuda":
        try:
            from torch.amp import autocast
            autocast_ctx = autocast(device_type="cuda", enabled=True)
        except Exception:
            from torch.cuda.amp import autocast
            autocast_ctx = autocast(enabled=True)

    def model_forward(x_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
        if autocast_ctx is not None:
            with autocast_ctx:
                v_pred = model(
                    x_in, t_in, text_ids, text_mask,
                    motion_mask, audio, audio_mask,
                    word_times, word_mask
                )
        else:
            v_pred = model(
                x_in, t_in, text_ids, text_mask,
                motion_mask, audio, audio_mask,
                word_times, word_mask
            )
        return v_pred.to(torch.float32)

    # overlap helper: replace first L frames of x_t using prev x0 (re-noised to current t)
    def apply_overlap(x_t: torch.Tensor, t_scalar: int):
        if overlap_prev_x0 is None or overlap <= 0:
            return x_t
        L = min(int(overlap), int(n_frames), int(overlap_prev_x0.shape[1]))
        if L <= 0:
            return x_t
        t_idx = torch.full((1,), int(t_scalar), device=device, dtype=torch.long)
        prev_clean = overlap_prev_x0[:, -L:]  # [1,L,D]
        noise = randn_like_gen(prev_clean, generator)
        # q(x_t | x0)
        x_prev_noisy = scheduler.add_noise(prev_clean, noise, t_idx)
        x_t[:, :L] = x_prev_noisy
        return x_t

    # main loop
    for i in range(len(ts) - 1):
        t_cur = int(ts[i])
        t_next = int(ts[i + 1])

        # enforce overlap at current t before prediction (important for consistency)
        x = apply_overlap(x, t_cur)

        t = torch.full((1,), t_cur, device=device, dtype=torch.long)
        v_pred = model_forward(x, t)
        x0, eps = predict_x0_eps_from_v(scheduler, x, v_pred, t)

        if sampler == "ddim":
            t_prev = torch.full((1,), t_next, device=device, dtype=torch.long)
            ab_t = scheduler.alphas_cumprod[t].view(-1, 1, 1)
            ab_prev = scheduler.alphas_cumprod[t_prev].view(-1, 1, 1)

            if float(eta) > 0:
                sigma = (
                    float(eta)
                    * torch.sqrt((1 - ab_prev) / (1 - ab_t))
                    * torch.sqrt(1 - (ab_t / ab_prev).clamp(max=1.0))
                )
            else:
                sigma = torch.zeros_like(ab_t)

            if t_next == 0:
                noise = torch.zeros_like(x)
            else:
                noise = randn_like_gen(x, generator)

            x = torch.sqrt(ab_prev) * x0 + torch.sqrt((1 - ab_prev) - sigma ** 2) * eps + sigma * noise

        else:
            # DDPM step using eps
            beta_t = scheduler.betas[t].view(-1, 1, 1)
            alpha_t = scheduler.alphas[t].view(-1, 1, 1)
            ab_t = scheduler.alphas_cumprod[t].view(-1, 1, 1)

            mean = (1.0 / torch.sqrt(alpha_t)) * (x - beta_t / torch.sqrt(1 - ab_t + 1e-8) * eps)

            if t_next == 0:
                x = mean
            else:
                t_prev = torch.full((1,), t_next, device=device, dtype=torch.long)
                ab_prev = scheduler.alphas_cumprod[t_prev].view(-1, 1, 1)
                var = beta_t * (1 - ab_prev) / (1 - ab_t + 1e-8)
                noise = randn_like_gen(x, generator) * float(temperature)
                x = mean + torch.sqrt(var.clamp(min=1e-20)) * noise

    # one more predict at t=0 to get final x0
    x = apply_overlap(x, 0)
    t0 = torch.zeros((1,), device=device, dtype=torch.long)
    v0 = model_forward(x, t0)
    x0, _ = predict_x0_eps_from_v(scheduler, x, v0, t0)
    return x0


# =============================================================================
# Decode motion -> BVH root + Euler
# =============================================================================
def _make_quat_continuous(q: np.ndarray) -> np.ndarray:
    q = q.copy()
    for t in range(1, q.shape[0]):
        if float(np.dot(q[t - 1], q[t])) < 0.0:
            q[t] *= -1.0
    return q


def decode_motion_to_bvh(
    motion_denorm: torch.Tensor,  # [T, D] denormed (feature space)
    motion_dim: int,
    fps: int,
    rot6d_start: int,
    layout: Optional[MotionFeatureLayout] = None,
    euler_order: str = "XYZ",
    vel_deadzone: float = 0.0,
    root_init_xz: Optional[Tuple[float, float]] = None,
    unwrap_euler: bool = True,
):
    """
    Supports both:
      legacy_v1: [vel_x, pos_y, vel_z, yaw_vel, contact..., rot6d...]
      root_pos_abs_v2: [root_x, root_y, root_z, yaw, lvx, lvz, yaw_vel, contact..., rot6d...]
    """
    if R is None:
        raise ImportError("scipy is required for BVH rotation decoding. Please install scipy in the inference environment.")
    assert motion_denorm.ndim == 2 and motion_denorm.shape[1] == motion_dim
    layout = layout or MotionFeatureLayout.from_metadata({}, motion_dim=motion_dim, rot6d_start=rot6d_start)
    rot6d_start = int(layout.rot6d_start)
    assert rot6d_start >= 0 and rot6d_start < motion_dim

    T = motion_denorm.shape[0]
    dt = 1.0 / float(fps)

    if layout.root_pos_mode == "legacy_vel_y":
        vel_x = motion_denorm[:, layout.legacy_root_indices[0]].clone()
        pos_y = motion_denorm[:, layout.legacy_root_indices[1]].clone()
        vel_z = motion_denorm[:, layout.legacy_root_indices[2]].clone()

        if vel_deadzone > 0:
            vel_x = torch.where(torch.abs(vel_x) < vel_deadzone, torch.zeros_like(vel_x), vel_x)
            vel_z = torch.where(torch.abs(vel_z) < vel_deadzone, torch.zeros_like(vel_z), vel_z)

        pos_x = torch.cumsum(vel_x * dt, dim=0)
        pos_z = torch.cumsum(vel_z * dt, dim=0)
        root_pos = torch.stack([pos_x, pos_y, pos_z], dim=-1)

        if root_init_xz is not None:
            root_pos[:, 0] = root_pos[:, 0] + float(root_init_xz[0])
            root_pos[:, 2] = root_pos[:, 2] + float(root_init_xz[1])
    else:
        root_pos = layout.decode_root_pos(motion_denorm, dt)
        if root_init_xz is not None:
            offset_x = float(root_init_xz[0]) - float(root_pos[0, 0].item())
            offset_z = float(root_init_xz[1]) - float(root_pos[0, 2].item())
            root_pos = root_pos.clone()
            root_pos[:, 0] = root_pos[:, 0] + offset_x
            root_pos[:, 2] = root_pos[:, 2] + offset_z

    rot_data = motion_denorm[:, rot6d_start:]
    if rot_data.shape[1] % 6 != 0:
        raise ValueError(f"rot dims not divisible by 6: got={rot_data.shape[1]} (rot6d_start={rot6d_start})")
    J = rot_data.shape[1] // 6

    rot6d = rot_data.view(T, J, 6)
    rot_mats = rotation_6d_to_matrix_col_safe(rot6d, eps=1e-8)  # [T,J,3,3]

    rot_mats_np = _to_numpy(rot_mats).reshape(-1, 3, 3)
    r = R.from_matrix(rot_mats_np)
    quat = r.as_quat().reshape(T, J, 4)  # [x,y,z,w]

    for j in range(J):
        quat[:, j, :] = _make_quat_continuous(quat[:, j, :])

    r2 = R.from_quat(quat.reshape(-1, 4))
    euler_deg = r2.as_euler(euler_order, degrees=True).reshape(T, J, 3)

    if unwrap_euler:
        euler_rad = np.deg2rad(euler_deg)
        euler_rad = np.unwrap(euler_rad, axis=0)
        euler_deg = np.rad2deg(euler_rad)

    return _to_numpy(root_pos), euler_deg


# =============================================================================
# Full generation with segmentation + overlap stitching
# =============================================================================
@torch.no_grad()
def generate_motion_full(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    tokenizer: BertTokenizer,
    *,
    segments: List[Dict[str, Any]],
    motion_dim: int,
    fps: int,
    device: torch.device,
    sampler: str,
    num_steps: int,
    eta: float,
    temperature: float,
    overlap: int,
    amp: bool,
    w2v2_feat: Optional[np.ndarray],
    w2v2_fps: Optional[float],
    require_audio: bool,
    require_word_ts: bool,
    debug_cond: bool,
    seed: int,
) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    full_list = []
    prev_x0 = None

    for i, seg in enumerate(segments):
        text = str(seg["text"])
        start_sec = float(seg["start"])
        end_sec = float(seg["end"])
        dur = max(0.3, end_sec - start_sec)

        n_frames = int(round(dur * fps))
        n_frames = max(n_frames, 2)

        tokens = tokenizer(
            [text],
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt",
        )
        text_ids = tokens.input_ids.to(device)
        text_mask = tokens.attention_mask.to(device)

        motion_mask = torch.ones((1, n_frames), device=device, dtype=torch.bool)

        # audio slice (absolute time -> wav2vec2 frames)
        audio_t = None
        audio_mask_t = None
        a0 = a1 = 0
        if (w2v2_feat is not None) and (w2v2_fps is not None):
            a0 = int(np.floor(start_sec * float(w2v2_fps)))
            a1 = int(np.ceil(end_sec * float(w2v2_fps)))
            a0 = max(0, min(a0, w2v2_feat.shape[0]))
            a1 = max(0, min(a1, w2v2_feat.shape[0]))
            if a1 > a0:
                a_clip = torch.from_numpy(w2v2_feat[a0:a1]).float().to(device)  # [Ta,Da]
                audio_t = a_clip.unsqueeze(0)  # [1,Ta,Da]
                audio_mask_t = torch.ones((1, audio_t.shape[1]), device=device, dtype=torch.bool)

        # word times relative to segment start
        wt_np = build_word_times_for_segment(seg, start_sec, end_sec)
        if wt_np.shape[0] > 0:
            word_times_t = torch.from_numpy(wt_np).float().to(device).unsqueeze(0)
            word_mask_t = torch.ones((1, wt_np.shape[0]), device=device, dtype=torch.bool)
        else:
            word_times_t = None
            word_mask_t = None

        if debug_cond:
            msg = f"[DEBUG] seg#{i} sec=({start_sec:.3f},{end_sec:.3f}) frames={n_frames} text_len={len(text)}"
            if (w2v2_feat is not None) and (w2v2_fps is not None):
                Ta = max(0, a1 - a0)
                msg += f" | audio a0={a0} a1={a1} Ta={Ta}"
                if audio_t is not None:
                    msg += f" mean={float(audio_t.mean().item()):.4f} std={float(audio_t.std().item()):.4f}"
                else:
                    msg += " (audio=None)"
            msg += f" | words={int(wt_np.shape[0])}"
            print(msg)

        if require_audio and audio_t is None:
            raise RuntimeError("require_audio=True but audio slice is None. Provide --w2v2_npz and valid audio range.")
        if require_word_ts and word_times_t is None:
            raise RuntimeError("require_word_ts=True but segment has no words (try enabling whisper word_timestamps).")

        x0 = sample_one_segment_v(
            model=model,
            scheduler=scheduler,
            text_ids=text_ids,
            text_mask=text_mask,
            motion_mask=motion_mask,
            audio=audio_t,
            audio_mask=audio_mask_t,
            word_times=word_times_t,
            word_mask=word_mask_t,
            n_frames=n_frames,
            motion_dim=motion_dim,
            sampler=sampler,
            num_steps=num_steps,
            eta=eta,
            temperature=temperature,
            overlap_prev_x0=prev_x0,
            overlap=overlap,
            generator=g,
            device=device,
            amp=amp,
        )  # [1,n_frames,D]

        prev_x0 = x0

        if i == 0:
            full_list.append(x0)
        else:
            L = min(int(overlap), int(n_frames))
            full_list.append(x0[:, L:])

    full = torch.cat(full_list, dim=1)  # [1,T,D]
    return full


# =============================================================================
# Main
# =============================================================================
def legacy_main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument("--checkpoint", type=str, required=True, help="trained checkpoint (full diffusion.pt or raw state_dict)")
    parser.add_argument("--audio", type=str, required=True, help="raw audio path for Whisper segmentation")
    parser.add_argument("--ref_bvh", type=str, required=True, help="template BVH for strict channel remap")
    parser.add_argument("--out", type=str, required=True, help="output bvh path")
    parser.add_argument("--out_npy", type=str, default=None, help="also save denorm feature as .npy")

    # runtime
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")

    # whisper
    parser.add_argument("--whisper", type=str, default="medium")
    parser.add_argument("--whisper_cache_dir", type=str, default="diffusion/whisper_cache")
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--no_word_ts", action="store_true")

    # segmentation
    parser.add_argument("--max_seg_dur", type=float, default=8.0)
    parser.add_argument("--gap_th", type=float, default=0.35)

    # diffusion sampling
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--num_steps", type=int, default=50, help="sampling steps (DDIM sub-sample); if =1000 same as train steps")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta (0 deterministic)")
    parser.add_argument("--temperature", type=float, default=1.0, help="DDPM noise scale (only used when sampler=ddpm)")
    parser.add_argument("--overlap", type=int, default=10)

    # postprocess / decode
    parser.add_argument("--smooth_sigma", type=float, default=0.0)
    parser.add_argument("--vel_deadzone", type=float, default=0.0)
    parser.add_argument("--anti_drift", action="store_true")
    parser.add_argument("--euler_order", type=str, default="XYZ")
    parser.add_argument("--strict_bvh", action="store_true")
    parser.add_argument("--root_xz_from_ref", action="store_true")
    parser.add_argument("--no_unwrap_euler", action="store_true")

    # wav2vec2 feature (npz)
    parser.add_argument("--w2v2_npz", type=str, default=None, help="wav2vec2 npz feature (recommended)")

    # audio stats normalize
    parser.add_argument("--audio_stats_from", type=str, default=None, help="optional: load audio_mean/std from cache pt")
    parser.add_argument("--no_audio_norm", action="store_true")
    parser.add_argument("--checkpoint_meta", type=str, default=None, help="metadata/full checkpoint to pair with a raw state_dict checkpoint")

    # strict checks
    parser.add_argument("--strict_skeleton", action="store_true")
    parser.add_argument("--require_audio", action="store_true")
    parser.add_argument("--require_word_ts", action="store_true")
    parser.add_argument("--debug_cond", action="store_true")

    # stabilize mean rotation (recommended)
    parser.add_argument("--project_mean_rot", action="store_true",
                        help="project ckpt mean rot6d onto valid SO(3)")

    # ===== NEW: EMA control =====
    parser.add_argument("--no_ema", action="store_true",
                        help="disable EMA at inference (default: use EMA if present)")

    args = parser.parse_args(argv)

    set_seed(int(args.seed))
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device={device}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    raw_obj = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    loaded_raw_state = False
    ckpt = raw_obj
    if not (isinstance(ckpt, dict) and "model" in ckpt):
        loaded_raw_state = True
        meta_path = Path(args.checkpoint_meta) if args.checkpoint_meta else Path(args.checkpoint).with_name("diffusion.pt")
        if not meta_path.exists():
            raise RuntimeError(
                "[ERR] raw state_dict checkpoint provided but no metadata checkpoint found. "
                "Use --checkpoint_meta or keep diffusion.pt in the same folder."
            )
        meta_ckpt = torch.load(meta_path, map_location="cpu", weights_only=False)
        if "model" not in meta_ckpt:
            raise RuntimeError(f"[ERR] metadata checkpoint missing key 'model': {meta_path}")
        ckpt = dict(meta_ckpt)
        ckpt["model"] = raw_obj
        print(f"[INFO] Loaded raw weights from {args.checkpoint}")
        print(f"[INFO] Using metadata from {meta_path}")

    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    fps = int(float(ckpt.get("fps", 30)))
    motion_dim = int(mean.shape[0])

    layout_meta: Dict[str, Any] = {}
    if isinstance(ckpt.get("meta"), dict):
        layout_meta.update(ckpt["meta"])
    if isinstance(ckpt.get("layout_meta"), dict):
        layout_meta.update(ckpt["layout_meta"])
    if ckpt.get("rot6d_start") is not None:
        layout_meta["rot6d_start"] = ckpt["rot6d_start"]
    if ckpt.get("contact_indices") is not None:
        layout_meta["contact_indices"] = ckpt["contact_indices"]

    layout = MotionFeatureLayout.from_metadata(
        layout_meta,
        motion_dim=motion_dim,
        rot6d_start=int(ckpt.get("rot6d_start", layout_meta.get("rot6d_start", 8))),
        contact_indices=ckpt.get("contact_indices"),
    )
    rot6d_start = int(layout.rot6d_start)
    if rot6d_start < 0 or rot6d_start >= motion_dim:
        raise RuntimeError(f"[ERR] invalid rot6d_start={rot6d_start} for motion_dim={motion_dim}")

    rot_dim = motion_dim - rot6d_start
    if rot_dim <= 0 or rot_dim % 6 != 0:
        raise RuntimeError(f"[ERR] invalid rot slice: motion_dim={motion_dim}, rot6d_start={rot6d_start}")
    J = rot_dim // 6

    J_ref = count_rot_joints_in_bvh(args.ref_bvh)
    if J_ref != J:
        msg = f"[WARN] ref_bvh rot joints = {J_ref}, but motion expects J = {J} (motion_dim={motion_dim}, rot6d_start={rot6d_start})."
        if args.strict_skeleton:
            raise RuntimeError(msg + " Use matching ref_bvh or disable --strict_skeleton.")
        print(msg)

    # model config
    t_args = ckpt.get("args", {}) or {}
    hidden = int(t_args.get("hidden_dim", 768))
    num_layers = int(t_args.get("num_layers", 8))
    num_heads = int(t_args.get("num_heads", 8))
    cond_drop_prob = float(t_args.get("cond_drop_prob", 0.0))

    # BERT path
    bert_path = str(t_args.get("bert_path", "models/bert"))
    if not Path(bert_path).exists() and Path("models/bert").exists():
        bert_path = "models/bert"

    print(f"[INFO] ckpt fps={fps}, motion_dim={motion_dim}, rot6d_start={rot6d_start}, J={J}")
    print(f"[INFO] feature layout: {layout.describe()}")
    print(f"[INFO] model: hidden={hidden}, layers={num_layers}, heads={num_heads}, cond_drop_prob={cond_drop_prob}")
    print(f"[INFO] Using BERT path: {bert_path}")
    print(f"[INFO] sampler={args.sampler} num_steps={args.num_steps} eta={args.eta} overlap={args.overlap}")

    # wav2vec2 feature
    w2v2_feat = None
    w2v2_fps = None
    audio_dim = None
    if args.w2v2_npz is not None:
        npz_path = Path(args.w2v2_npz)
        if not npz_path.exists():
            raise FileNotFoundError(f"--w2v2_npz not found: {npz_path}")
        feat, afps, _dur = load_w2v2_feature(npz_path)
        if feat is None or afps is None:
            raise RuntimeError(f"Failed to read wav2vec2 npz: {npz_path}")
        w2v2_feat = feat
        w2v2_fps = afps
        audio_dim = int(w2v2_feat.shape[1])
        print(f"[INFO] w2v2: shape={w2v2_feat.shape}, fps={w2v2_fps}, audio_dim={audio_dim}")

    # audio stats normalize
    audio_mean = ckpt.get("audio_mean", None)
    audio_std = ckpt.get("audio_std", None)
    if (audio_mean is None or audio_std is None) and (args.audio_stats_from is not None):
        p = Path(args.audio_stats_from)
        if not p.exists():
            raise FileNotFoundError(f"--audio_stats_from not found: {p}")
        cache_obj = torch.load(str(p), map_location="cpu", weights_only=False)
        audio_mean = cache_obj.get("audio_mean", None)
        audio_std = cache_obj.get("audio_std", None)

    if (w2v2_feat is not None) and (not args.no_audio_norm):
        if audio_mean is not None and audio_std is not None:
            w2v2_feat = maybe_normalize_audio(w2v2_feat, audio_mean, audio_std)
            print("[INFO] Applied audio normalization using audio_mean/audio_std.")
        else:
            print("[WARN] No audio_mean/audio_std found. Audio will be unnormalized.")

    # build model
    if audio_dim is None:
        audio_dim = int(ckpt.get("audio_dim", t_args.get("audio_dim", 768)))

    model = MotionDiffusionTransformer(
        motion_dim=motion_dim,
        audio_dim=audio_dim,
        bert_path=bert_path,
        motion_fps=float(fps),
        hidden=hidden,
        num_layers=num_layers,
        num_heads=num_heads,
        cond_drop_prob=float(cond_drop_prob),
    ).to(device)

    # ===== NEW: choose EMA or raw weights =====
    state = None
    if not args.no_ema and not loaded_raw_state:
        ema = ckpt.get("ema", None)
        if isinstance(ema, dict):
            shadow = ema.get("shadow", None)
            if isinstance(shadow, dict) and len(shadow) > 0:
                state = shadow
                print(f"[INFO] Using EMA shadow weights (n={len(state)}).")
            else:
                print("[WARN] EMA exists but shadow is empty. Fallback to raw ckpt['model'].")
        else:
            print("[WARN] No EMA found in ckpt. Fallback to raw ckpt['model'].")

    if state is None:
        state = ckpt["model"]
        print(f"[INFO] Using raw model weights (n={len(state)}).")

    # strip DDP prefix if any
    if isinstance(state, dict) and len(state) > 0:
        first_k = next(iter(state.keys()))
        if first_k.startswith("module."):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

    # strict load first; if it fails, show info then fallback to strict=False (prevents hard crash)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print("[WARN] strict=True load failed:", str(e).split("\n")[0])
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[WARN] strict=False loaded. missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 0:
            print("  missing examples:", missing[:10])
        if len(unexpected) > 0:
            print("  unexpected examples:", unexpected[:10])

    model.eval()

    # scheduler: train steps fixed 1000
    scheduler = DDPMScheduler(num_timesteps=1000, device=device)

    # tokenizer
    print(f"[INFO] Loading Tokenizer from: {bert_path}")
    if BertTokenizer is None:
        raise ImportError("transformers is required for tokenizer loading. Please install a compatible transformers package.")
    try:
        tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)
    except Exception:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # whisper -> segments
    whisper_obj = load_or_run_whisper(
        audio_path=args.audio,
        whisper_model=args.whisper,
        cache_dir=Path(args.whisper_cache_dir),
        language=args.language,
        word_timestamps=(not args.no_word_ts),
    )
    segments = build_segments_from_whisper(
        whisper_obj,
        max_duration=float(args.max_seg_dur),
        gap_threshold=float(args.gap_th),
    )
    print(f"[INFO] Segments: {len(segments)}")

    # generate (normalized x0)
    full_norm = generate_motion_full(
        model=model,
        scheduler=scheduler,
        tokenizer=tokenizer,
        segments=segments,
        motion_dim=motion_dim,
        fps=fps,
        device=device,
        sampler=str(args.sampler),
        num_steps=int(args.num_steps),
        eta=float(args.eta),
        temperature=float(args.temperature),
        overlap=int(args.overlap),
        amp=bool(args.amp),
        w2v2_feat=w2v2_feat,
        w2v2_fps=w2v2_fps,
        require_audio=bool(args.require_audio),
        require_word_ts=bool(args.require_word_ts),
        debug_cond=bool(args.debug_cond),
        seed=int(args.seed),
    )  # [1,T,D]

    full_norm_np = full_norm.squeeze(0).detach().cpu().numpy().astype(np.float32)  # [T,D]

    # smooth in normalized space
    if float(args.smooth_sigma) > 0:
        if gaussian_filter1d is None:
            print("[WARN] scipy.ndimage.gaussian_filter1d not available, skip smoothing.")
        else:
            full_norm_np = gaussian_filter1d(full_norm_np, sigma=float(args.smooth_sigma), axis=0)

    # denorm (with optional projected mean for rotations)
    mean_eff = mean.copy()
    if args.project_mean_rot:
        mean_eff = project_rot6d_mean_to_valid(mean_eff, motion_dim=motion_dim, rot6d_start=rot6d_start, eps=1e-8)
        print("[INFO] Projected mean rot6d -> valid SO(3).")

    if bool(args.anti_drift):
        if layout.root_pos_mode == "legacy_vel_y":
            mean_eff[layout.legacy_root_indices[0]] = 0.0
            mean_eff[layout.legacy_root_indices[2]] = 0.0
        elif layout.local_vel_xz_indices is not None:
            mean_eff[layout.local_vel_xz_indices[0]] = 0.0
            mean_eff[layout.local_vel_xz_indices[1]] = 0.0

    motion_denorm = torch.from_numpy(full_norm_np).to(device).float()
    mean_t = torch.from_numpy(mean_eff).to(device).float()
    std_t = torch.from_numpy(std).to(device).float()
    std_t = torch.where(std_t < 1e-6, torch.ones_like(std_t), std_t)
    motion_denorm = motion_denorm * std_t + mean_t  # [T,D]

    # optionally save npy (denorm feature)
    if args.out_npy is not None:
        out_npy_p = Path(args.out_npy)
        ensure_dir(out_npy_p.parent)
        np.save(str(out_npy_p), motion_denorm.detach().cpu().numpy().astype(np.float32))
        print(f"[INFO] Saved feature npy -> {out_npy_p}")

    # root init from ref
    root_init_xz = None
    if bool(args.root_xz_from_ref):
        xyz = read_ref_root_first_frame_xyz(args.ref_bvh)
        if xyz is not None:
            root_init_xz = (float(xyz[0]), float(xyz[2]))
            print(f"[INFO] root_init_xz from ref: {root_init_xz}")

    # decode
    root_pos, euler_rots = decode_motion_to_bvh(
        motion_denorm=motion_denorm,
        motion_dim=motion_dim,
        fps=fps,
        rot6d_start=rot6d_start,
        layout=layout,
        euler_order=str(args.euler_order),
        vel_deadzone=float(args.vel_deadzone),
        root_init_xz=root_init_xz,
        unwrap_euler=(not bool(args.no_unwrap_euler)),
    )

    # save BVH
    save_bvh_remapped(
        root_pos=root_pos,
        euler_rots=euler_rots,
        ref_bvh_path=args.ref_bvh,
        output_path=args.out,
        fps=fps,
        strict_bvh=bool(args.strict_bvh),
    )



def main(argv: Optional[List[str]] = None):
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--legacy", action="store_true")
    known, remaining = wrapper.parse_known_args(argv)
    if known.legacy:
        return legacy_main(remaining)

    from diffusion.infer_latent_flow import main as latent_main

    return latent_main(remaining)


if __name__ == "__main__":
    main()
