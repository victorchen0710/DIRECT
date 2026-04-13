import argparse
import json
from pathlib import Path
from multiprocessing import Pool
import warnings
import re

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

warnings.filterwarnings("ignore")


# =============================================================================
# Config
# =============================================================================
TARGET_FPS = 30
TEXT_MAX_LEN = 128
MIN_MOTION_FRAMES = 15

# “脚相关”关键词：会把所有匹配到的关节都加入 contact（foot lock）
FOOT_KEYWORDS = ["Foot", "Toe"]

# =============================================================================
# Math Helpers
# =============================================================================

def _rle_runs(x01: np.ndarray):
    """
    x01: [T] float/0-1
    return: list of (val, s, e) where e is exclusive
    """
    x = (x01 > 0.5).astype(np.int32)
    T = x.shape[0]
    if T == 0:
        return []
    runs = []
    s = 0
    cur = x[0]
    for t in range(1, T):
        if x[t] != cur:
            runs.append((cur, s, t))
            s = t
            cur = x[t]
    runs.append((cur, s, T))
    return runs

def debounce_contact_runs(contact: np.ndarray, min_on=8, min_off=6):
    """
    contact: [T, K] float (0/1 or prob)
    - ON-run 长度 < min_on  -> 改成 0
    - OFF-run长度 < min_off -> 改成 1
    返回: 去抖后的 contact [T,K] float32 0/1
    """
    c = (contact > 0.5).astype(np.int32)
    T, K = c.shape
    out = c.copy()

    for k in range(K):
        runs = _rle_runs(out[:, k])
        for val, s, e in runs:
            L = e - s
            if val == 1 and L < min_on:
                out[s:e, k] = 0
            elif val == 0 and L < min_off:
                out[s:e, k] = 1

    return out.astype(np.float32)


def matrix_to_rotation_6d(mat: torch.Tensor) -> torch.Tensor:
    # mat: [...,3,3] -> [...,6]
    return torch.cat([mat[..., :, 0], mat[..., :, 1]], dim=-1)

def extract_yaw_from_matrix_y_up(rot_mats: np.ndarray) -> np.ndarray:
    """
    更稳的 yaw 提取（Y-up, Z-forward）
    rot_mats: [T, 3, 3]，local->world
    用 local forward(0,0,1) 在世界的方向投影到 XZ 平面，求 yaw
    """
    fwd = rot_mats[:, :, 2]  # 第三列 = local Z 轴在 world 的方向
    fwd_x = fwd[:, 0]
    fwd_z = fwd[:, 2]
    yaw = np.arctan2(fwd_x, fwd_z)  # yaw=0 时 forward 指向世界 +Z
    return yaw

def world_to_local_velocity(world_vel: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """
    world -> local：用 -yaw（逆旋转）
    world_vel: [T,3] (vx,vy,vz)
    yaw: [T]
    return: [T,3] (side, up, forward)
    """
    c = np.cos(yaw)
    s = np.sin(yaw)
    v_x, v_y, v_z = world_vel[:, 0], world_vel[:, 1], world_vel[:, 2]
    local_x =  c * v_x - s * v_z
    local_z =  s * v_x + c * v_z
    return np.stack([local_x, v_y, local_z], axis=-1)

# =============================================================================
# BVH Structure & FK
# =============================================================================
class BVHStructure:
    def __init__(self, lines):
        self.joints = []
        self.names = []
        self.parents = []
        self.offsets = []
        self.parse(lines)

    def _ensure_offsets_len(self, idx: int):
        if len(self.offsets) <= idx:
            self.offsets.extend([np.zeros(3, dtype=np.float32) for _ in range(idx - len(self.offsets) + 1)])

    def parse(self, lines):
        stack = []
        for line in lines:
            s = line.strip()
            if s.startswith("ROOT") or s.startswith("JOINT"):
                name = s.split()[1]
                parent_idx = stack[-1] if stack else -1
                self.names.append(name)
                self.parents.append(parent_idx)
                self.joints.append({"name": name, "parent": parent_idx})
                stack.append(len(self.joints) - 1)
            elif s.startswith("End Site"):
                parent_idx = stack[-1] if stack else -1
                end_name = f"{self.names[parent_idx]}_End" if parent_idx >= 0 else "EndSite"
                self.names.append(end_name)
                self.parents.append(parent_idx)
                self.joints.append({"name": end_name, "parent": parent_idx})
                stack.append(len(self.joints) - 1)
            elif s.startswith("OFFSET"):
                parts = s.split()
                if len(parts) >= 4 and stack:
                    off = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
                    curr_idx = stack[-1]
                    self._ensure_offsets_len(curr_idx)
                    self.offsets[curr_idx] = off
            elif s.startswith("}"):
                if stack:
                    stack.pop()
            elif s.upper() == "MOTION":
                break

        while len(self.offsets) < len(self.joints):
            self.offsets.append(np.zeros(3, dtype=np.float32))
        self.offsets = np.asarray(self.offsets, dtype=np.float32)

    def forward_kinematics(self, rot_mats_anim, root_pos, rot_name_map):
        """
        rot_mats_anim: [T, J_rot, 3,3]
        root_pos: [T,3]
        rot_name_map: joint_name -> index in rot_mats_anim
        return global_pos: [T, J_all, 3]
        """
        T = rot_mats_anim.shape[0]
        J_all = len(self.joints)
        global_rots = np.zeros((T, J_all, 3, 3), dtype=np.float32)
        global_pos = np.zeros((T, J_all, 3), dtype=np.float32)
        identity = np.eye(3, dtype=np.float32)[None, :, :].repeat(T, axis=0)

        local_rots = []
        for i in range(J_all):
            name = self.joints[i]["name"]
            clean_name = name.replace("_End", "")
            if clean_name in rot_name_map:
                idx = rot_name_map[clean_name]
                local_rots.append(rot_mats_anim[:, idx])
            elif name in rot_name_map:
                idx = rot_name_map[name]
                local_rots.append(rot_mats_anim[:, idx])
            else:
                local_rots.append(identity)

        for i in range(J_all):
            parent = self.parents[i]
            local_r = local_rots[i]
            offset = self.offsets[i]
            if parent == -1:
                global_rots[:, i] = local_r
                global_pos[:, i] = root_pos.astype(np.float32)
            else:
                global_rots[:, i] = np.matmul(global_rots[:, parent], local_r)
                off_rot = np.matmul(global_rots[:, parent], offset)
                global_pos[:, i] = global_pos[:, parent] + off_rot
        return global_pos

# =============================================================================
# BVH Parsing
# =============================================================================
def load_bvh_lines(path: Path):
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except:
        return None


def compute_contact_and_locks(
    gpos: np.ndarray,               # [T, J_all, 3]
    struct_names,                   # struct.names
    foot_names,                     # list[str]
    fps: float,
    gnd_pct: float =3.0,
    h_th: float = 3.0,
    v_th: float = 1.5,              # 3D speed threshold for contact
    v_xz_th: float = 2.0,           # horizontal speed threshold for lock_xz
    y_lock_th: float = 2.0,         # |h-gnd| < y_lock_th => lock_y
    use_lock_y: bool = False,
    gap_fill: int = 2
):
    """
    改动点（不删你原逻辑，只增强/更稳）：
    1) ground 用“全脚点的每帧最低点 min_h”，再取 percentile => 一个 global_ground
    2) contact：低速 + 低高度
    3) lock_xz：contact 门控 + 水平速度低
    4) lock_y：可选（默认关），避免脚尖翘起/滚脚被锁死
    """
    # --- 找脚点 idx ---
    def find_idx(name):
        q = name.lower()
        for i, n in enumerate(struct_names):
            if n.lower() == q:
                return i
        for i, n in enumerate(struct_names):
            if q in n.lower():
                return i
        return -1

    foot_idxs = [find_idx(nm) for nm in foot_names]
    valid = [i for i in foot_idxs if i != -1]
    K = len(foot_idxs)
    T = gpos.shape[0]

    contact = np.zeros((T, K), dtype=np.float32)
    lock_xz = np.zeros((T, K), dtype=np.float32)
    lock_y  = np.zeros((T, K), dtype=np.float32)
    gnd = np.zeros((K,), dtype=np.float32)

    # -------- 统一 ground：全脚点一起估（每帧最低点 -> percentile）--------
    if len(valid) == 0:
        global_g = 0.0
    else:
        h_all = np.stack([gpos[:, idx, 1] for idx in valid], axis=1)  # [T, K_valid]
        min_h = np.min(h_all, axis=1)                                 # [T]
        global_g = float(np.percentile(min_h, gnd_pct))

    gnd[:] = global_g  # 所有脚点共享同一 ground

    # -------- per-foot contact/locks --------
    for k, idx in enumerate(foot_idxs):
        if idx == -1:
            continue

        p = gpos[:, idx, :]                     # [T,3]
        dp = np.gradient(p, axis=0) * fps       # [T,3] m/s (BVH unit/s)
        v3 = np.linalg.norm(dp, axis=1)         # [T]
        vxz = np.linalg.norm(dp[:, [0, 2]], axis=1)
        h = p[:, 1]
        g = global_g

        c = (v3 < v_th) & (h < (g + h_th))
        lxz = c & (vxz < v_xz_th)

        if use_lock_y:
            ly = c & (np.abs(h - g) < y_lock_th)
        else:
            ly = np.zeros_like(c)

        contact[:, k] = c.astype(np.float32)
        lock_xz[:, k] = lxz.astype(np.float32)
        lock_y[:, k]  = ly.astype(np.float32)

    # --- 去噪：填补短 0 洞 ---
    def fill_short_gaps(x, max_gap):
        if max_gap <= 0:
            return x
        x = x.copy()
        t = 0
        while t < len(x):
            if x[t] > 0.5:
                t += 1
                continue
            t0 = t
            while t < len(x) and x[t] < 0.5:
                t += 1
            t1 = t
            if (t1 - t0) <= max_gap:
                left_ok = (t0 - 1 >= 0 and x[t0 - 1] > 0.5)
                right_ok = (t1 < len(x) and x[t1] > 0.5)
                if left_ok and right_ok:
                    x[t0:t1] = 1.0
        return x

    for k in range(K):
        contact[:, k] = fill_short_gaps(contact[:, k], gap_fill)
        lock_xz[:, k] = fill_short_gaps(lock_xz[:, k], gap_fill)
        if use_lock_y:
            lock_y[:, k] = fill_short_gaps(lock_y[:, k], gap_fill)

    return foot_idxs, gnd, contact, lock_xz, lock_y



def parse_bvh_channel_blocks(lines):
    blocks = []
    curr_joint = None
    channel_cursor = 0
    for line in lines:
        s = line.strip()
        if s.upper() == "MOTION":
            break
        parts = s.split()
        if not parts:
            continue
        if parts[0] in ("ROOT", "JOINT"):
            curr_joint = parts[1]
        elif parts[0] == "CHANNELS":
            if curr_joint is None:
                continue
            n = int(parts[1])
            chans = parts[2:2 + n]
            idxs = list(range(channel_cursor, channel_cursor + n))
            channel_cursor += n

            rot_dims = []
            rot_indices = {}
            for c, gi in zip(chans, idxs):
                if "rotation" in c.lower():
                    axis = c[0].upper()
                    rot_dims.append(axis)
                    rot_indices[axis] = gi

            blocks.append({
                "name": curr_joint,
                "has_pos": any("position" in c.lower() for c in chans),
                "has_rot": len(rot_dims) == 3,
                "rot_order": "".join(rot_dims),  # e.g. "XYZ"
                "rot_indices": rot_indices,
                "pos_indices": {c[0].upper(): i for c, i in zip(chans, idxs) if "position" in c.lower()}
            })
    return blocks, channel_cursor

def load_bvh_data(path: Path):
    lines = load_bvh_lines(path)
    if not lines:
        return None, None, None, None

    motion_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().upper() == "MOTION":
            motion_idx = i
            break
    if motion_idx is None:
        return None, None, None, None

    header = lines[:motion_idx]
    blocks, n_channels = parse_bvh_channel_blocks(header)

    # Frame Time
    try:
        frame_time = float(lines[motion_idx + 2].split(":")[1].strip())
    except:
        frame_time = 0.033333

    data_lines = lines[motion_idx + 3:]
    data = []
    for ln in data_lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            v = [float(x) for x in ln.split()]
        except:
            continue
        if len(v) == n_channels:
            data.append(v)

    if not data:
        return None, None, None, None

    return np.array(data, dtype=np.float32), frame_time, blocks, header

def extract_skeleton_meta(bvh_path: Path):
    raw, ft, blocks, header = load_bvh_data(bvh_path)
    if raw is None:
        return None
    struct = BVHStructure(header)
    rot_names = [b["name"] for b in blocks if b["has_rot"]]
    return {
        "all_joint_names": struct.names,
        "parents": struct.parents,
        "offsets": struct.offsets.tolist(),
        "joint_names": rot_names
    }

def resample_pose(root_pos, rot_mats, t_src, t_tgt):
    out_pos = np.zeros((len(t_tgt), 3), dtype=np.float32)
    for i in range(3):
        out_pos[:, i] = np.interp(t_tgt, t_src, root_pos[:, i])

    T_src, J, _, _ = rot_mats.shape
    out_rot = np.zeros((len(t_tgt), J, 3, 3), dtype=np.float32)
    t_src_f = t_src.astype(np.float64)
    t_tgt_f = np.clip(t_tgt.astype(np.float64), t_src_f[0], t_src_f[-1])

    for j in range(J):
        slerp = Slerp(t_src_f, R.from_matrix(rot_mats[:, j]))
        out_rot[:, j] = slerp(t_tgt_f).as_matrix()

    return out_pos, out_rot

# =============================================================================
# Foot joints: take ALL matching joints
# =============================================================================
def find_all_foot_joint_indices(all_names, keywords):
    idxs = []
    for i, name in enumerate(all_names):
        low = name.lower()
        if "end site" in low:
            continue
        # BVHStructure 会把 End Site 变成 *_End，这些一般不需要
        if low.endswith("_end"):
            continue
        for kw in keywords:
            if kw.lower() in low:
                idxs.append(i)
                break
    # 去重保持顺序
    seen = set()
    out = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out

# =============================================================================
# Worker
# =============================================================================
tokenizer = None
words_json_cache = {}

def init_worker():
    global tokenizer, words_json_cache
    words_json_cache = {}
    try:
        tokenizer = BertTokenizer.from_pretrained("models/bert", local_files_only=True)
    except:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def _clean_text(txt):
    if txt is None:
        return None
    if isinstance(txt, (list, tuple)):
        txt = " ".join([str(x) for x in txt if x is not None]).strip()
    else:
        txt = str(txt).strip()
    if len(txt) == 0:
        return None
    return txt

def _slice_by_time(arr, start_s, end_s, fps):
    """
    arr: [T,...]
    start_s/end_s: seconds
    returns sliced arr and (s_idx,e_idx)
    """
    if start_s is None or end_s is None:
        return arr, 0, arr.shape[0]
    try:
        st = float(start_s)
        ed = float(end_s)
    except:
        return arr, 0, arr.shape[0]

    # 允许少量重叠：这里用 floor/ceil 会更“包住”句子
    s_idx = int(np.floor(st * fps))
    e_idx = int(np.ceil(ed * fps))  # 不含 e_idx
    s_idx = max(0, s_idx)
    e_idx = min(arr.shape[0], e_idx)
    if e_idx <= s_idx:
        return None, s_idx, e_idx
    return arr[s_idx:e_idx], s_idx, e_idx


def load_segment_word_times(item, base_dir, start_s, end_s):
    words_json = item.get("words_json", None)
    if not words_json:
        return np.zeros((0, 2), dtype=np.float32)

    path = base_dir / words_json
    if not path.exists():
        return np.zeros((0, 2), dtype=np.float32)

    cache_key = str(path)
    obj = words_json_cache.get(cache_key)
    if obj is None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            obj = None
        words_json_cache[cache_key] = obj

    if not isinstance(obj, dict):
        return np.zeros((0, 2), dtype=np.float32)

    segments = obj.get("segments", [])
    all_words = []
    for seg in segments:
        ws = seg.get("words", [])
        if isinstance(ws, list):
            all_words.extend(ws)

    if not all_words:
        return np.zeros((0, 2), dtype=np.float32)

    wr = item.get("words_range", None)
    if isinstance(wr, (list, tuple)) and len(wr) == 2:
        try:
            a = max(0, int(wr[0]))
            b = max(a, int(wr[1]))
            all_words = all_words[a:b]
        except Exception:
            pass

    start_s = 0.0 if start_s is None else float(start_s)
    end_s = float("inf") if end_s is None else float(end_s)

    out = []
    for w in all_words:
        try:
            st = float(w.get("start", None))
            ed = float(w.get("end", None))
        except Exception:
            continue
        if not np.isfinite(st) or not np.isfinite(ed):
            continue
        if ed <= start_s or st >= end_s:
            continue
        st = max(st, start_s)
        ed = min(ed, end_s)
        if ed <= st + 1e-4:
            continue
        out.append([st - start_s, ed - start_s])

    if not out:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(out, dtype=np.float32)

def process_item(args):
    item, base_dir = args

    filename = item.get("bvh") or item.get("motion")
    if not filename:
        return []

    bvh_path = base_dir / filename
    if not bvh_path.exists():
        return []

    raw, frame_time, blocks, header = load_bvh_data(bvh_path)
    if raw is None:
        return []

    # ---- text 必须非空 ----
    txt = _clean_text(item.get("text", None))
    if txt is None:
        return []

    # 1) Parse rotations
    T = raw.shape[0]
    rot_blocks = [b for b in blocks if b["has_rot"]]
    J = len(rot_blocks)
    if J == 0:
        return []

    rot_mats = np.zeros((T, J, 3, 3), dtype=np.float32)
    rot_map = {}
    for j, b in enumerate(rot_blocks):
        rot_map[b["name"]] = j
        order = b["rot_order"]
        # euler in detected order
        euler = np.stack([raw[:, b["rot_indices"][ax]] for ax in order], axis=-1)
        rot_mats[:, j] = R.from_euler(order, euler, degrees=True).as_matrix()

    # 2) Parse root position joint
    pos_blocks = [b for b in blocks if b["has_pos"]]
    if not pos_blocks:
        return []
    rb = pos_blocks[0]
    root_name = rb["name"]
    root_pos = np.stack([raw[:, rb["pos_indices"][ax]] for ax in ["X", "Y", "Z"]], axis=-1)

    # 3) Resample full clip to TARGET_FPS
    if T <= 1:
        return []
    duration = (T - 1) * frame_time
    T_tgt_full = int(round(duration * TARGET_FPS)) + 1

    t_src = np.arange(T) * frame_time
    t_tgt = np.arange(T_tgt_full) / TARGET_FPS
    root_pos_tgt, rot_mats_tgt = resample_pose(root_pos, rot_mats, t_src, t_tgt)

    # 4) Compute yaw + yaw_vel (on full clip)
    root_rot_idx = rot_map.get(root_name, None)
    if root_rot_idx is None:
        root_rot_idx = 0
    yaw = extract_yaw_from_matrix_y_up(rot_mats_tgt[:, root_rot_idx])
    yaw_unwrap = np.unwrap(yaw)
    yaw_vel = np.gradient(yaw_unwrap) * TARGET_FPS  # rad/s

    # 5) Compute world vel + local vel (on full clip)
    world_vel = np.gradient(root_pos_tgt, axis=0) * TARGET_FPS
    local_vel = world_to_local_velocity(world_vel, yaw)

    # 6) Contacts (foot lock) on full clip
    struct = BVHStructure(header)
    gpos = struct.forward_kinematics(rot_mats_tgt, root_pos_tgt, rot_map)

    foot_idxs = find_all_foot_joint_indices(struct.names, FOOT_KEYWORDS)
    if len(foot_idxs) == 0:
        return []

    foot_names = [struct.names[i] for i in foot_idxs]

    # 用统一 ground + 更稳 contact/lock
    _foot_idxs2, gnd, contact_full, lock_xz_full, lock_y_full = compute_contact_and_locks(
        gpos=gpos,
        struct_names=struct.names,
        foot_names=foot_names,
        fps=float(TARGET_FPS),
        gnd_pct=5.0,
        h_th=5.0,
        v_th=2.0,
        v_xz_th=2.0,
        y_lock_th=2.0,
        use_lock_y=False,   # 先关，避免滚脚被锁死（你想开再开）
        gap_fill=2
    )

    contacts_full = contact_full.astype(np.float32)

    for k, idx in enumerate(foot_idxs):
        fp = gpos[:, idx, :]
        v = np.linalg.norm(np.gradient(fp, axis=0), axis=1) * TARGET_FPS
        h = fp[:, 1]
        gnd = np.percentile(h, 5)
        c_v = (v < 2.0).astype(np.float32)
        c_h = (h < (gnd + 5.0)).astype(np.float32)
        contacts_full[:, k] = c_v * c_h

    contacts_full = debounce_contact_runs(contacts_full, min_on=8, min_off=6)
    
    # 7) Rot6D on full clip
    rot6d_full = matrix_to_rotation_6d(torch.from_numpy(rot_mats_tgt)).numpy().reshape(T_tgt_full, -1)

    # 8) Slice by sentence start/end (seconds)
    start_s = item.get("start", None)
    end_s = item.get("end", None)

    root_pos_clip, s_idx, e_idx = _slice_by_time(root_pos_tgt, start_s, end_s, TARGET_FPS)
    if root_pos_clip is None:
        return []

    rot6d_clip = rot6d_full[s_idx:e_idx]
    yaw_unwrap_clip = yaw_unwrap[s_idx:e_idx]
    yaw_vel_clip = yaw_vel[s_idx:e_idx]
    local_vel_clip = local_vel[s_idx:e_idx]
    contacts_clip = contacts_full[s_idx:e_idx]

    T_clip = root_pos_clip.shape[0]
    if T_clip < MIN_MOTION_FRAMES:
        return []

    # 9) Build root feature (可复原 + 保留 yaw_vel)
    # root_pos_xyz: 世界位置（可复原）
    # yaw_unwrap:   绝对朝向（可复原方向）
    # local_vel_xz: 仍然保留，训练更稳（对坐标变换更鲁棒）
    root_x = root_pos_clip[:, 0]
    root_y = root_pos_clip[:, 1]
    root_z = root_pos_clip[:, 2]
    lvx = local_vel_clip[:, 0]
    lvz = local_vel_clip[:, 2]

    feat_root = np.stack([root_x, root_y, root_z, yaw_unwrap_clip, lvx, lvz], axis=-1)  # [T,6]

    # motion_feat: [root(6), yaw_vel(1), contacts(K), rot6d]
    motion_feat = np.concatenate(
        [feat_root, yaw_vel_clip[:, None], contacts_clip, rot6d_clip],
        axis=-1
    ).astype(np.float32)

    word_times = load_segment_word_times(item, base_dir, start_s, end_s)

    # 10) Text tokens
    tokens = tokenizer(
        text=txt,
        padding="max_length",
        max_length=TEXT_MAX_LEN,
        truncation=True,
        return_tensors="pt"
    )

    # 11) Audio feature slice & align to T_clip (not wav, only npz feat)
    audio = np.zeros((T_clip, 768), dtype=np.float32)
    if "feature" in item:
        ap = base_dir / item["feature"]
        if ap.exists():
            try:
                feat = np.load(ap)["feat"].astype(np.float32)
                # 先对齐到 full length（T_tgt_full），再按 s/e 切
                if feat.shape[0] >= T_tgt_full:
                    feat_full = feat[:T_tgt_full]
                else:
                    pad = np.zeros((T_tgt_full, feat.shape[1]), dtype=np.float32)
                    pad[:feat.shape[0]] = feat
                    feat_full = pad

                feat_clip = feat_full[s_idx:e_idx]
                # 最终保证维度一致
                if feat_clip.shape[0] == T_clip and feat_clip.shape[1] == 768:
                    audio = feat_clip
                else:
                    # 兜底：补齐/裁剪
                    tmp = np.zeros((T_clip, 768), dtype=np.float32)
                    tmin = min(T_clip, feat_clip.shape[0])
                    dmin = min(768, feat_clip.shape[1])
                    tmp[:tmin, :dmin] = feat_clip[:tmin, :dmin]
                    audio = tmp
            except:
                pass

    clip = {
        "motion": motion_feat,
        "text_ids": tokens.input_ids[0].numpy(),
        "text_mask": tokens.attention_mask[0].numpy(),
        "audio": audio,
        "word_times": word_times,
        "motion_len": int(T_clip),
        "audio_len": int(audio.shape[0]),
        "src_bvh": str(bvh_path),
        "contact_dim": int(contacts_clip.shape[1]),
        "foot_names": foot_names,
        "dbg": {
            "id": item.get("id", ""),
            "seg_id": item.get("seg_id", ""),
            "start": start_s,
            "end": end_s,
            "bvh_frames_src": int(T),
            "frame_time": float(frame_time),
            "tgt_full": int(T_tgt_full),
            "slice_len": int(T_clip),
            "K": int(contacts_clip.shape[1]),
            "motion_dim": int(motion_feat.shape[1]),
        }
    }
    return [clip]


def compute_feature_stats(clips, key):
    total = None
    total_sq = None
    count = 0
    for clip in clips:
        arr = np.asarray(clip[key], dtype=np.float32)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        arr64 = arr.astype(np.float64, copy=False)
        s = arr64.sum(axis=0)
        ss = np.square(arr64).sum(axis=0)
        if total is None:
            total = s
            total_sq = ss
        else:
            total += s
            total_sq += ss
        count += int(arr.shape[0])

    if total is None or total_sq is None or count <= 0:
        raise RuntimeError(f"No valid data found while computing stats for key={key}")

    mean = total / float(count)
    var = np.maximum(total_sq / float(count) - np.square(mean), 0.0)
    std = np.sqrt(var + 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="manifests/train_whisperx_split.jsonl")
    parser.add_argument("--output", type=str, default="diffusion/cache/train_diffusion.pt")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    base = Path(args.manifest).parent.parent
    with open(args.manifest, "r", encoding="utf-8") as f:
        items = [json.loads(x) for x in f if x.strip()]

    if args.limit and args.limit > 0:
        items = items[:args.limit]

    # Skeleton meta from first file (best-effort)
    skel = None
    first_bvh = items[0].get("bvh") or items[0].get("motion") if items else None
    if first_bvh:
        skel = extract_skeleton_meta(base / first_bvh)

    clips = []
    with Pool(args.workers, initializer=init_worker) as p:
        for r in tqdm(p.imap_unordered(process_item, [(it, base) for it in items]), total=len(items)):
            if r:
                clips.extend(r)

    if not clips:
        raise RuntimeError("No clips generated!")

    # Stats
    mean, std = compute_feature_stats(clips, "motion")
    audio_mean, audio_std = compute_feature_stats(clips, "audio")

    # Meta
    root_dim = 6
    has_yaw_vel = 1
    contact_dim = int(clips[0]["contact_dim"])
    rot6d_start = root_dim + has_yaw_vel + contact_dim
    contact_indices = list(range(root_dim + has_yaw_vel, rot6d_start))

    meta = {
        "joint_names": skel["joint_names"] if skel else None,
        "all_joint_names": skel["all_joint_names"] if skel else None,
        "skeleton_parents": skel["parents"] if skel else None,
        "skeleton_offsets": skel["offsets"] if skel else None,
        "fps": TARGET_FPS,
        "root_dim": root_dim,
        "has_yaw_vel": True,
        "contact_dim": contact_dim,
        "contact_indices": contact_indices,
        "foot_names": clips[0]["foot_names"],
        "layout_version": "root_pos_abs_v2",
        "root_pos_mode": "absolute_xyz",
        "root_pos_indices": [0, 1, 2],
        "yaw_index": 3,
        "local_vel_xz_indices": [4, 5],
        "yaw_vel_index": 6,
        "rot6d_start": rot6d_start,
    }

    if args.debug:
        print("\n[DEBUG] show first 10 clips:")
        for c in clips[:10]:
            d = c.get("dbg", {})
            key = d.get("seg_id") or d.get("id")
            print(
                key,
                "start/end=", d.get("start"), d.get("end"),
                "src=", d.get("bvh_frames_src"), "@", d.get("frame_time"),
                "tgt_full=", d.get("tgt_full"),
                "slice_len=", d.get("slice_len"),
                "K=", d.get("K"),
                "motion_dim=", d.get("motion_dim"),
            )
        print(f"[INFO] contact_dim(K)={contact_dim}, rot6d_start={rot6d_start}")

    torch.save({
        "data": clips,
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "audio_mean": audio_mean.astype(np.float32),
        "audio_std": audio_std.astype(np.float32),
        "fps": TARGET_FPS,
        "meta": meta,
    }, args.output)

    print(f"[INFO] Generated {len(clips)} clips. Feature Dim: {mean.shape[0]}")
    print(f"[INFO] contact_dim(K)={contact_dim}, rot6d_start={rot6d_start}")
    print(f"[INFO] Saved to {args.output}")

if __name__ == "__main__":
    main()
