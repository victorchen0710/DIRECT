import argparse
import json
import math
import random
from pathlib import Path
from typing import Union, Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

TARGET_FPS_DEFAULT = 30


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :2].transpose(-1, -2).flatten(-2)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_bvh_channels(path: Path) -> Tuple[Optional[np.ndarray], Optional[float]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    try:
        motion_idx = lines.index("MOTION")
        frame_time = float(lines[motion_idx + 2].split(":")[1].strip())
    except Exception:
        return None, None

    data = []
    for ln in lines[motion_idx + 3:]:
        if not ln.strip():
            continue
        try:
            data.append([float(x) for x in ln.strip().split()])
        except ValueError:
            continue

    if not data or frame_time <= 0:
        return None, None

    return np.asarray(data, dtype=np.float32), frame_time


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
    T = int(motion.shape[0])

    lines = []
    lines.extend(header_lines)
    lines.append(f"Frames: {T}")
    lines.append(f"Frame Time: {frame_time:.6f}")
    for t in range(T):
        row = motion[t]
        lines.append(" ".join(f"{float(v):.6f}" for v in row))

    with path_out.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _resolve(base_dir: Path, p: Union[str, Path]) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    cand = base_dir / p
    if cand.exists():
        return cand
    cand = base_dir.parent / p
    if cand.exists():
        return cand
    return base_dir / p


def unwrap_bvh_angles_degrees(motion: np.ndarray, pos_dims: int = 3) -> np.ndarray:
    if motion.ndim != 2 or motion.shape[0] < 2:
        return motion.astype(np.float32)
    m = motion.astype(np.float32, copy=True)
    if m.shape[1] <= pos_dims:
        return m
    ang = m[:, pos_dims:]
    rad = np.deg2rad(ang)
    rad = np.unwrap(rad, axis=0)
    m[:, pos_dims:] = np.rad2deg(rad).astype(np.float32)
    return m


def wrap_angles_degrees(motion: np.ndarray, pos_dims: int = 3) -> np.ndarray:
    if motion.ndim != 2:
        return motion
    m = motion.astype(np.float32, copy=True)
    if m.shape[1] <= pos_dims:
        return m
    ang = m[:, pos_dims:]
    ang = ((ang + 180.0) % 360.0) - 180.0
    m[:, pos_dims:] = ang
    return m


def resample_motion_linear(motion: np.ndarray, frame_time: float, target_fps: int) -> np.ndarray:
    if motion.ndim != 2 or len(motion) < 2:
        return motion.astype(np.float32)

    src_fps = 1.0 / (frame_time + 1e-6)
    T_src = motion.shape[0]
    t_src = np.arange(T_src, dtype=np.float32) / src_fps

    duration = float(t_src[-1])
    T_tgt = int(round(duration * float(target_fps))) + 1
    if T_tgt < 2:
        return motion[:1].astype(np.float32)

    t_tgt = np.arange(T_tgt, dtype=np.float32) / float(target_fps)

    out = np.empty((T_tgt, motion.shape[1]), dtype=np.float32)
    for d in range(motion.shape[1]):
        out[:, d] = np.interp(t_tgt, t_src, motion[:, d].astype(np.float32))
    return out


class VectorQuantizerEMA(nn.Module):
    """
    Spherical DDP-safe EMA VQ (Eval Version)
    """
    def __init__(
        self,
        n_codes: int,
        code_dim: int,
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        # 兼容 eval 代码中没传 entropy 相关参数的情况
        usage_entropy_w: float = 0.0,
        usage_temp: float = 0.5,
        revive_threshold: float = 1.0,
    ):
        super().__init__()
        self.n_codes = int(n_codes)
        self.code_dim = int(code_dim)
        self.beta = float(beta)
        self.decay = float(decay)
        self.eps = float(eps)

        self.usage_entropy_w = float(usage_entropy_w)
        self.usage_temp = float(usage_temp)
        self.revive_threshold = float(revive_threshold)

        self.codebook = nn.Embedding(self.n_codes, self.code_dim)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=1.0)
        with torch.no_grad():
            self.codebook.weight.data = F.normalize(self.codebook.weight.data, p=2, dim=-1)

        self.register_buffer("ema_cluster_size", torch.zeros(self.n_codes))
        self.register_buffer("ema_w", self.codebook.weight.data.clone())

        self.codebook.weight.requires_grad_(False)

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, codes: torch.Tensor):
        # Eval 阶段其实不会调用这里，但为了代码完整性保留
        pass

    def forward(self, z_e: torch.Tensor):
        B, C, T = z_e.shape
        
        # 隔离 AMP
        device_type = "cuda" if z_e.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            z_e_fp32 = z_e.float()
            
            # 【核心】对输入的特征进行 L2 归一化
            z_e_norm = F.normalize(z_e_fp32, p=2, dim=1)
            
            z = z_e_norm.permute(0, 2, 1).contiguous().view(-1, C)
            
            # 【核心】对 Codebook 进行 L2 归一化
            e_norm = F.normalize(self.codebook.weight.float(), p=2, dim=1)

            z2 = (z ** 2).sum(dim=1, keepdim=True)                   
            e2 = (e_norm ** 2).sum(dim=1).unsqueeze(0)                    
            ze = z @ e_norm.t()                                           
            distances = z2 + e2 - 2.0 * ze                           

        codes = torch.argmin(distances, dim=1)
        
        # 提取量化后的特征 (必须是从 e_norm 里取)
        z_q = e_norm[codes].view(B, T, C).permute(0, 2, 1).contiguous()
        z_q_fp32 = z_q.float()

        vq_loss = self.beta * F.mse_loss(z_e_norm, z_q_fp32.detach())
        z_q_st = z_e_norm + (z_q_fp32 - z_e_norm).detach()
        z_q_st = z_q_st.to(z_e.dtype)

        with torch.no_grad():
            one_hot = F.one_hot(codes, self.n_codes).float()
            avg_probs = one_hot.mean(dim=0)
            ppl = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        codes = codes.view(B, T)
        return z_q_st, vq_loss.to(z_e.dtype), codes, ppl

'''
class VectorQuantizerEMA(nn.Module):
    def __init__(self, n_codes: int, code_dim: int, beta: float = 0.25, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.n_codes = int(n_codes)
        self.code_dim = int(code_dim)
        self.beta = float(beta)
        self.decay = float(decay)
        self.eps = float(eps)

        self.codebook = nn.Embedding(self.n_codes, self.code_dim)
        bound = 1.0 / math.sqrt(self.code_dim)
        nn.init.uniform_(self.codebook.weight, -bound, bound)

        self.register_buffer("ema_cluster_size", torch.zeros(self.n_codes))
        self.register_buffer("ema_w", self.codebook.weight.data.clone())
        self.codebook.weight.requires_grad_(False)

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, codes: torch.Tensor):
        K = self.n_codes
        one_hot = F.one_hot(codes, K).type(z.dtype)
        cluster_size = one_hot.sum(dim=0)
        dw = one_hot.t() @ z
        self.ema_cluster_size.mul_(self.decay).add_(cluster_size * (1.0 - self.decay))
        self.ema_w.mul_(self.decay).add_(dw * (1.0 - self.decay))
        n = self.ema_cluster_size.sum()
        cluster_size = (self.ema_cluster_size + self.eps) / (n + K * self.eps) * n
        embed = self.ema_w / cluster_size.unsqueeze(1)
        self.codebook.weight.data.copy_(embed)

    def forward(self, z_e: torch.Tensor):
        B, C, T = z_e.shape
        z_e_fp32 = z_e.float()
        z = z_e_fp32.permute(0, 2, 1).contiguous().view(-1, C)
        e = self.codebook.weight.float()

        z2 = (z ** 2).sum(dim=1, keepdim=True)
        e2 = (e ** 2).sum(dim=1).unsqueeze(0)
        ze = z @ e.t()
        dist = z2 + e2 - 2.0 * ze

        codes = torch.argmin(dist, dim=1)
        z_q = self.codebook(codes).view(B, T, C).permute(0, 2, 1).contiguous()
        z_q_fp32 = z_q.float()

        vq_loss = self.beta * F.mse_loss(z_e_fp32, z_q_fp32.detach())
        z_q_st = z_e + (z_q.to(z_e.dtype) - z_e).detach()

        with torch.no_grad():
            one_hot = F.one_hot(codes, self.n_codes).float()
            avg_probs = one_hot.mean(dim=0)
            ppl = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        codes = codes.view(B, T)
        return z_q_st, vq_loss.to(z_e.dtype), codes, ppl
'''

class UpConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class ResBlock1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.block(x)


class MotionVQVAE(nn.Module):
    def __init__(
        self,
        motion_dim: int,
        hidden: int = 512,
        code_dim: int = 256,
        n_codes: int = 1024,
        beta: float = 0.25,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        n_downsample: int = 2,
    ):
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.hidden = int(hidden)
        self.code_dim = int(code_dim)
        self.n_downsample = int(n_downsample)

        enc_layers: List[nn.Module] = []
        enc_layers.append(nn.Conv1d(self.motion_dim, hidden, kernel_size=3, stride=1, padding=1))
        enc_layers.append(nn.ReLU(inplace=True))
        for _ in range(self.n_downsample):
            enc_layers.append(nn.Conv1d(hidden, hidden, kernel_size=4, stride=2, padding=1))
            enc_layers.append(nn.ReLU(inplace=True))
            enc_layers.append(ResBlock1d(hidden))
            enc_layers.append(ResBlock1d(hidden))
        enc_layers.append(nn.Conv1d(hidden, code_dim, kernel_size=3, stride=1, padding=1))
        self.enc = nn.Sequential(*enc_layers)

        self.vq = VectorQuantizerEMA(
            n_codes=n_codes,
            code_dim=code_dim,
            beta=beta,
            decay=ema_decay,
            eps=ema_eps,
        )

        dec_layers: List[nn.Module] = []
        dec_layers.append(nn.Conv1d(code_dim, hidden, kernel_size=3, stride=1, padding=1))
        dec_layers.append(nn.ReLU(inplace=True))
        for _ in range(self.n_downsample):
            dec_layers.append(ResBlock1d(hidden))
            dec_layers.append(ResBlock1d(hidden))
            dec_layers.append(UpConv1d(hidden, hidden))
            dec_layers.append(nn.ReLU(inplace=True))
        dec_layers.append(nn.Conv1d(hidden, self.motion_dim, kernel_size=3, stride=1, padding=1))
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor, use_vq: bool = True):
        x1 = x.transpose(1, 2).contiguous()
        z_e = self.enc(x1)
        if use_vq:
            z_q, vq_loss, codes, ppl = self.vq(z_e)
        else:
            z_q = z_e
            vq_loss = z_e.new_tensor(0.0)
            codes = torch.zeros((x.shape[0], z_e.shape[-1]), device=x.device, dtype=torch.long)
            ppl = z_e.new_tensor(0.0)
        x_hat = self.dec(z_q).transpose(1, 2).contiguous()
        return x_hat, vq_loss, codes, ppl, z_e


def remap_strip_conv_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k.replace(".conv.weight", ".weight").replace(".conv.bias", ".bias").replace(".conv.", ".")
        out[nk] = v
    return out


def remap_add_conv_keys_if_needed(state_dict: Dict[str, torch.Tensor], model_keys: set) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k
        if k.endswith(".weight"):
            cand = k[:-7] + ".conv.weight"
            if cand in model_keys:
                nk = cand
        elif k.endswith(".bias"):
            cand = k[:-5] + ".conv.bias"
            if cand in model_keys:
                nk = cand
        out[nk] = v
    return out


def load_state_dict_compat(model: nn.Module, state_dict: Dict[str, torch.Tensor], strict: bool = True):
    try:
        model.load_state_dict(state_dict, strict=strict)
        print("[INFO] Loaded checkpoint state_dict (as-is).")
        return
    except RuntimeError:
        pass

    try:
        sd2 = remap_strip_conv_keys(state_dict)
        model.load_state_dict(sd2, strict=strict)
        print("[WARN] Loaded checkpoint after stripping '*.conv.*' keys.")
        return
    except RuntimeError:
        pass

    mk = set(model.state_dict().keys())
    sd3 = remap_add_conv_keys_if_needed(state_dict, mk)
    model.load_state_dict(sd3, strict=strict)
    print("[WARN] Loaded checkpoint after adding '*.conv.*' keys.")


@torch.no_grad()
def eval_one_window(
    model: MotionVQVAE,
    motion_full_6d: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    block_size: int,
    drop_root_pos: bool,
    device: torch.device,
    start: int,
    use_vq: bool,
) -> Dict:
    T, _ = motion_full_6d.shape
    start = int(max(0, min(start, T - block_size)))
    end = start + block_size
    gt_full = motion_full_6d[start:end].astype(np.float32)

    if drop_root_pos:
        root = gt_full[:, :3]
        gt_keep = gt_full[:, 3:]
    else:
        root = None
        gt_keep = gt_full

    gt_norm = (gt_keep - mean) / std
    x = torch.from_numpy(gt_norm).unsqueeze(0).to(device)

    x_hat, vq_loss, codes, ppl, _ = model(x, use_vq=use_vq)
    recon = F.smooth_l1_loss(x_hat, x).item()

    x_hat_np = x_hat.squeeze(0).cpu().numpy().astype(np.float32)
    x_hat_den = x_hat_np * std + mean

    if drop_root_pos:
        recon_full_6d = np.concatenate([root, x_hat_den], axis=1)
    else:
        recon_full_6d = x_hat_den

    return {
        "gt_full_6d": gt_full,
        "recon_full_6d": recon_full_6d.astype(np.float32),
        "codes": codes.squeeze(0).cpu().numpy(),
        "ppl": float(ppl.item()),
        "vq": float(vq_loss.item()),
        "recon": float(recon),
        "start": start,
        "end": end,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("outputs/vqvae_recon"))
    ap.add_argument("--num_samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--fixed_start", type=int, default=0)
    ap.add_argument("--random_start", action="store_true")
    ap.add_argument("--block_size", type=int, default=None)
    ap.add_argument("--use_vq", action="store_true")
    ap.add_argument("--no_use_vq", action="store_true")
    ap.add_argument("--wrap_angles", action="store_true")
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (torch.cuda.is_available() and args.device.startswith("cuda")) else "cpu")
    print(f"[INFO] Device: {device}")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    mean = ckpt.get("mean", None)
    std = ckpt.get("std", None)
    keep_dim = int(ckpt.get("keep_dim", 0))
    drop_root_pos = bool(ckpt.get("drop_root_pos", False))
    ckpt_block_size = int(ckpt.get("block_size", 256))
    target_fps = TARGET_FPS_DEFAULT

    n_codes = int(ckpt.get("n_codes", 1024))
    code_dim = int(ckpt.get("code_dim", 256))
    beta = float(ckpt.get("beta", 0.25))
    ema_decay = float(ckpt.get("ema_decay", 0.99))
    ema_eps = float(ckpt.get("ema_eps", 1e-5))
    hidden = int(ckpt.get("hidden", 512))
    n_downsample = int(ckpt.get("n_downsample", 3))

    if mean is None or std is None or keep_dim <= 0:
        raise RuntimeError("Checkpoint missing mean/std/keep_dim.")

    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    block_size = int(args.block_size) if args.block_size is not None else ckpt_block_size

    print(f"[INFO] target_fps={target_fps} block_size={block_size} drop_root_pos={drop_root_pos} keep_dim={keep_dim}")

    model = MotionVQVAE(
        motion_dim=keep_dim,
        hidden=hidden,
        code_dim=code_dim,
        n_codes=n_codes,
        beta=beta,
        ema_decay=ema_decay,
        ema_eps=ema_eps,
        n_downsample=n_downsample,
    ).to(device)

    load_state_dict_compat(model, state, strict=True)
    model.eval()

    use_vq = True
    if args.no_use_vq:
        use_vq = False
    elif args.use_vq:
        use_vq = True

    items = []
    base_dir = args.manifest.parent
    with args.manifest.open("r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                items.append(json.loads(ln))
    print(f"[INFO] Loaded {len(items)} items from {args.manifest}")

    candidates = []
    for it in items:
        bvh_rel = it.get("bvh", None) or it.get("motion", None) or it.get("motion_path", None)
        if not bvh_rel:
            continue
        bvh = _resolve(base_dir, bvh_rel)
        if not bvh.exists():
            continue
        candidates.append(it)

    if len(candidates) == 0:
        raise RuntimeError("No valid BVH paths found in manifest.")

    if args.shuffle:
        random.shuffle(candidates)

    candidates = candidates[: max(1, int(args.num_samples))]

    recon_list, vq_list, ppl_list = [], [], []
    code_hist = np.zeros((n_codes,), dtype=np.int64)

    for it in tqdm(candidates, desc="eval"):
        bvh_rel = it.get("bvh", None) or it.get("motion", None) or it.get("motion_path", None)
        bvh = _resolve(base_dir, bvh_rel)

        header = read_bvh_header_lines(bvh)
        motion_raw, ft = load_bvh_channels(bvh)
        if motion_raw is None or ft is None or motion_raw.ndim != 2 or motion_raw.shape[0] < 2:
            continue

        motion = unwrap_bvh_angles_degrees(motion_raw, pos_dims=3)
        motion = resample_motion_linear(motion, ft, target_fps)

        if motion.shape[0] < block_size:
            continue

        root_pos = motion[:, :3]
        rot_euler = motion[:, 3:]
        if rot_euler.shape[1] % 3 != 0:
            print(f"[WARN] skip bad channels: {rot_euler.shape}")
            continue

        rot_flat = rot_euler.reshape(-1, 3)
        r_obj = R.from_euler("XYZ", rot_flat, degrees=True)
        mats = r_obj.as_matrix()

        mats_t = torch.from_numpy(mats).float()
        rot_6d = matrix_to_rotation_6d(mats_t).numpy().reshape(motion.shape[0], -1)
        motion_6d = np.concatenate([root_pos, rot_6d], axis=1).astype(np.float32)

        D_full_6d = motion_6d.shape[1]
        if drop_root_pos:
            if D_full_6d - 3 != keep_dim:
                print(f"[WARN] skip dim mismatch: {bvh} full_6d={D_full_6d} keep_dim={keep_dim}")
                continue
        else:
            if D_full_6d != keep_dim:
                print(f"[WARN] skip dim mismatch: {bvh} full_6d={D_full_6d} keep_dim={keep_dim}")
                continue

        if args.random_start:
            start = random.randint(0, motion_6d.shape[0] - block_size)
        else:
            start = int(args.fixed_start)

        out = eval_one_window(
            model=model,
            motion_full_6d=motion_6d,
            mean=mean,
            std=std,
            block_size=block_size,
            drop_root_pos=drop_root_pos,
            device=device,
            start=start,
            use_vq=use_vq,
        )

        recon_list.append(out["recon"])
        vq_list.append(out["vq"])
        ppl_list.append(out["ppl"])

        codes = out["codes"].reshape(-1)
        for c in codes:
            ci = int(c)
            if 0 <= ci < n_codes:
                code_hist[ci] += 1

        stem = (it.get("id", None) or bvh.stem).replace("/", "_")
        gt_path = args.out_dir / f"{stem}_GT.bvh"
        rec_path = args.out_dir / f"{stem}_REC.bvh"

        def to_euler_motion(motion_6d_in: np.ndarray) -> np.ndarray:
            r_pos = motion_6d_in[:, :3]
            r_6d = motion_6d_in[:, 3:]
            T_ = r_6d.shape[0]
            J_ = r_6d.shape[1] // 6

            r_6d_t = torch.from_numpy(r_6d).float().reshape(T_ * J_, 6)
            mats_out = rotation_6d_to_matrix(r_6d_t).numpy()
            r_obj_out = R.from_matrix(mats_out)
            euler_out = r_obj_out.as_euler("XYZ", degrees=True).reshape(T_, J_ * 3)
            return np.concatenate([r_pos, euler_out], axis=1)

        gt_6d = out["gt_full_6d"]
        rec_6d = out["recon_full_6d"]

        gt_euler = to_euler_motion(gt_6d)
        rec_euler = to_euler_motion(rec_6d)

        if args.wrap_angles:
            gt_euler = wrap_angles_degrees(gt_euler, pos_dims=3)
            rec_euler = wrap_angles_degrees(rec_euler, pos_dims=3)

        frame_time = 1.0 / float(target_fps)
        write_bvh(gt_path, header, gt_euler, frame_time=frame_time)
        write_bvh(rec_path, header, rec_euler, frame_time=frame_time)

    if len(recon_list) == 0:
        print("[ERROR] No samples exported.")
        return

    used = int((code_hist > 0).sum())
    topk = np.argsort(-code_hist)[:10]
    topk_str = ", ".join([f"{i}:{int(code_hist[i])}" for i in topk if code_hist[i] > 0])

    print("\n========== VQ-VAE Recon Eval Summary ==========")
    print(f"[INFO] Export dir: {args.out_dir}")
    print(f"[METRIC] recon(smoothL1): mean={np.mean(recon_list):.6f}")
    print(f"[METRIC] vq_loss:         mean={np.mean(vq_list):.6f}")
    print(f"[METRIC] ppl:             mean={np.mean(ppl_list):.2f}")
    print(f"[CODE] used_codes={used}/{n_codes}")
    print(f"[CODE] top10 codes: {topk_str}")
    print("=============================================\n")


if __name__ == "__main__":
    main()