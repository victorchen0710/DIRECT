import os, sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.multiprocessing as mp
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# ---- config ----
MODEL_NAME = "/home2/chenwq/projects/direct/models/wav2vec2"
TARGET_SR = 16000
TARGET_FPS = 30
CHUNK_SEC = 20.0          # 长音频分块推理，避免显存爆
CHUNK_OVERLAP_SEC = 0.2   # 小重叠，拼接更平滑
SAVE_DTYPE = np.float16   # 节省空间

def list_wavs(root: Path):
    return sorted(root.rglob("*.wav"))

def ensure_16k(wav: torch.Tensor, sr: int) -> torch.Tensor:
    if sr == TARGET_SR:
        return wav
    resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
    return resampler(wav)

@torch.no_grad()
def extract_w2v2_features(model, processor, wav_16k: torch.Tensor, device: str):
    # wav_16k: (T,) float32
    # 采用分块提特征，然后拼起来（去掉 overlap 重叠区）
    total_len = wav_16k.shape[0]
    chunk_len = int(CHUNK_SEC * TARGET_SR)
    overlap = int(CHUNK_OVERLAP_SEC * TARGET_SR)
    step = chunk_len - overlap

    feats = []
    start = 0
    while start < total_len:
        end = min(start + chunk_len, total_len)
        chunk = wav_16k[start:end]

        inputs = processor(chunk.numpy(), sampling_rate=TARGET_SR, return_tensors="pt")
        input_values = inputs.input_values.to(device)

        out = model(input_values)
        h = out.last_hidden_state.squeeze(0)  # (T_feat, 768)

        # 丢掉除首块外的前 overlap 对应的特征（近似按比例丢）
        if start > 0 and overlap > 0:
            # w2v2 输出帧数与输入采样数不严格线性，但近似可用比例映射
            drop = int(h.shape[0] * (overlap / (end - start)))
            h = h[drop:, :]

        feats.append(h.cpu())
        start += step

    feats = torch.cat(feats, dim=0)  # (T_feat_total, 768)
    return feats

def pool_to_fps(feats: torch.Tensor, duration_sec: float, fps: int):
    """
    feats: (T_feat, D)
    用“时间中心点”把 w2v2 特征池化到 motion fps
    """
    T_feat = feats.shape[0]
    D = feats.shape[1]
    # w2v2 每帧中心时间（均匀假设）
    feat_times = (torch.arange(T_feat) + 0.5) * (duration_sec / T_feat)

    T_out = int(np.round(duration_sec * fps))
    out = torch.zeros((T_out, D), dtype=feats.dtype)
    out_times = (torch.arange(T_out) + 0.5) / fps

    # 双指针聚合：把落在同一个 motion frame 时间窗里的 feat 求均值
    left = 0
    for i in range(T_out):
        t_center = out_times[i].item()
        t0 = t_center - 0.5 / fps
        t1 = t_center + 0.5 / fps

        # 移动 left 到 t0
        while left < T_feat and feat_times[left].item() < t0:
            left += 1
        right = left
        while right < T_feat and feat_times[right].item() < t1:
            right += 1

        if right > left:
            out[i] = feats[left:right].mean(dim=0)
        else:
            # 如果窗口里没有任何 feat（极少发生），就用最近的一个
            idx = min(max(left, 0), T_feat - 1)
            out[i] = feats[idx]

    return out  # (T_out, D)

def _split_list(lst, n):
    n = max(1, n)
    if len(lst) == 0:
        return []
    size = (len(lst) + n - 1) // n
    return [lst[i * size : (i + 1) * size] for i in range(n) if lst[i * size : (i + 1) * size]]

def _worker(wavs, beat_root: Path, out_root: Path, device: str, worker_idx: int):
    if len(wavs) == 0:
        return
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    desc = f"gpu {device}" if device.startswith("cuda") else "cpu"
    for wav_path in tqdm(wavs, desc=desc, position=worker_idx):
        rel = wav_path.relative_to(beat_root)
        save_path = out_root / "w2v2_base" / (str(rel) + ".npz")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists():
            continue

        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # 转单声道
        wav = torch.from_numpy(audio)

        wav = ensure_16k(wav, sr)
        duration_sec = wav.shape[0] / TARGET_SR

        feats = extract_w2v2_features(model, processor, wav, device=device)  # (T_feat, 768)
        pooled = pool_to_fps(feats, duration_sec, TARGET_FPS)               # (T_out, 768)

        np.savez_compressed(
            save_path,
            w2v2_30fps=pooled.numpy().astype(SAVE_DTYPE),
            duration=np.array([duration_sec], dtype=np.float32),
            fps=np.array([TARGET_FPS], dtype=np.int32),
        )

def main(beat_root: str, out_root: str, device: str = None, num_workers: int = None):
    beat_root = Path(beat_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    use_multi_gpu = device.startswith("cuda") and torch.cuda.is_available()
    if num_workers is None:
        num_workers = torch.cuda.device_count() if use_multi_gpu else 1
    num_workers = max(1, num_workers)

    print(f"[INFO] device={device}, workers={num_workers}")
    wavs = list_wavs(beat_root)
    print(f"[INFO] found {len(wavs)} wav files")

    if num_workers == 1:
        _worker(wavs, beat_root, out_root, device, worker_idx=0)
        return

    if not use_multi_gpu:
        print("[WARN] num_workers>1 but cuda not available, fallback to single worker on CPU")
        _worker(wavs, beat_root, out_root, device, worker_idx=0)
        return

    # 多 GPU 并行：每个进程绑一个 GPU，分片处理文件
    shards = _split_list(wavs, num_workers)
    devices = [f"cuda:{i}" for i in range(len(shards))]
    mp.set_start_method("spawn", force=True)
    procs = []
    for idx, (dev, shard) in enumerate(zip(devices, shards)):
        p = mp.Process(target=_worker, args=(shard, beat_root, out_root, dev, idx))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_w2v2.py <BEAT_ROOT> <OUT_ROOT> [cuda|cpu|cuda:0] [num_workers]")
        sys.exit(1)
    dev = sys.argv[3] if len(sys.argv) >= 4 else None
    workers = int(sys.argv[4]) if len(sys.argv) >= 5 else None
    main(sys.argv[1], sys.argv[2], device=dev, num_workers=workers)
