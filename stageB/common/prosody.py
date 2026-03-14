from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from stageA.train_stage1_vqvae import TARGET_FPS


def _resample_to_target(times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] <= 1 or times.shape[0] <= 1:
        return np.repeat(values[:1], len(target_times), axis=0).astype(np.float32)

    out = np.zeros((len(target_times), values.shape[1]), dtype=np.float32)
    for i in range(values.shape[1]):
        out[:, i] = np.interp(target_times, times, values[:, i]).astype(np.float32)
    return out


def _nearest_binary_to_target(times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0 or times.size == 0:
        return np.zeros((len(target_times), 1), dtype=np.float32)
    idx = np.searchsorted(times, target_times, side="left")
    idx = np.clip(idx, 0, len(times) - 1)
    return values[idx][:, None].astype(np.float32)


def _load_audio(audio_path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(audio_path), always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if int(sr) != int(sample_rate):
        y = scipy.signal.resample_poly(y, sample_rate, sr).astype(np.float32)
        sr = int(sample_rate)
    return y.astype(np.float32), int(sr)


def _frame_audio(y: np.ndarray, frame_length: int, hop_length: int) -> torch.Tensor:
    wave = torch.from_numpy(y).float()
    pad = frame_length // 2
    wave = F.pad(wave[None, None], (pad, pad), mode="constant", value=0.0).squeeze(0).squeeze(0)
    if wave.numel() < frame_length:
        wave = F.pad(wave, (0, frame_length - int(wave.numel())), mode="constant", value=0.0)
    return wave.unfold(0, frame_length, hop_length)


def extract_prosody_features(
    audio_path: Path,
    *,
    fps: int = TARGET_FPS,
    target_frames: Optional[int] = None,
    sample_rate: int = 16000,
    mel_bins: int = 0,
    mfcc_dim: int = 0,
) -> dict[str, np.ndarray]:
    y, sr = _load_audio(audio_path, sample_rate)
    if target_frames is None:
        duration = len(y) / float(sr)
        target_frames = max(1, int(math.ceil(duration * float(fps))))
    target_frames = int(target_frames)
    target_times = (np.arange(target_frames, dtype=np.float32) + 0.5) / float(fps)

    hop_length = max(1, int(round(sr / float(fps))))
    frame_length = max(1024, 2 ** int(np.ceil(np.log2(max(hop_length * 4, 1024)))))
    frames = _frame_audio(y, frame_length, hop_length)
    frame_times = ((np.arange(frames.shape[0], dtype=np.float32) + 0.5) * hop_length) / float(sr)

    rms = torch.sqrt(torch.mean(frames**2, dim=-1) + 1e-8).cpu().numpy().astype(np.float32)
    log_energy = np.log(np.maximum(rms**2, 1e-8)).astype(np.float32)
    delta_energy = np.concatenate([[0.0], np.diff(log_energy)]).astype(np.float32)

    waveform = torch.from_numpy(y).float().unsqueeze(0)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=frame_length,
        hop_length=hop_length,
        n_mels=max(int(mel_bins), 40),
        power=2.0,
    )
    mel_spec = mel_transform(waveform).squeeze(0)  # [M, F]
    log_mel = torch.log1p(mel_spec)
    mel_times = ((np.arange(log_mel.shape[-1], dtype=np.float32) + 0.5) * hop_length) / float(sr)

    onset = torch.relu(log_mel[:, 1:] - log_mel[:, :-1]).mean(dim=0)
    onset = torch.cat([torch.zeros(1), onset], dim=0).cpu().numpy().astype(np.float32)

    f0 = np.zeros((len(frame_times),), dtype=np.float32)
    voiced_flag = np.zeros((len(frame_times),), dtype=np.float32)
    try:
        pitch = torchaudio.functional.detect_pitch_frequency(
            waveform,
            sample_rate=sr,
            frame_time=float(hop_length) / float(sr),
        ).squeeze(0)
        pitch = pitch.cpu().numpy().astype(np.float32)
        pitch = pitch[: len(frame_times)]
        f0[: len(pitch)] = pitch
        voiced_flag[: len(pitch)] = (pitch > 1.0).astype(np.float32)
    except Exception:
        pass

    f0_mask = (f0 > 1.0).astype(np.float32)
    log_f0 = np.log(np.maximum(f0, 1.0)).astype(np.float32)

    non_silent = rms[rms > 0.0]
    silence_threshold = max(1e-5, float(np.percentile(non_silent, 20)) * 0.5) if non_silent.size > 0 else 1e-4
    silence = (rms <= silence_threshold).astype(np.float32)
    speaking = 1.0 - silence

    feat_list = [
        _resample_to_target(frame_times, log_energy, target_times),
        _resample_to_target(frame_times, delta_energy, target_times),
        _resample_to_target(mel_times, onset, target_times),
        _nearest_binary_to_target(frame_times, voiced_flag, target_times),
        _resample_to_target(frame_times, log_f0, target_times),
        _nearest_binary_to_target(frame_times, f0_mask, target_times),
        _nearest_binary_to_target(frame_times, silence, target_times),
        _nearest_binary_to_target(frame_times, speaking, target_times),
    ]
    feature_names = [
        "log_energy",
        "delta_energy",
        "onset_strength",
        "voiced",
        "log_f0",
        "f0_mask",
        "silence",
        "speaking",
    ]

    if int(mel_bins) > 0:
        mel_small = log_mel[: int(mel_bins)].transpose(0, 1).cpu().numpy().astype(np.float32)
        feat_list.append(_resample_to_target(mel_times[: mel_small.shape[0]], mel_small, target_times))
        feature_names.extend([f"log_mel_{i:02d}" for i in range(int(mel_bins))])

    if int(mfcc_dim) > 0:
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=int(mfcc_dim),
            melkwargs={"n_fft": frame_length, "hop_length": hop_length},
        )
        mfcc = mfcc_transform(waveform).squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)
        feat_list.append(_resample_to_target(frame_times[: mfcc.shape[0]], mfcc, target_times))
        feature_names.extend([f"mfcc_{i:02d}" for i in range(int(mfcc_dim))])

    frame_feat = np.concatenate(feat_list, axis=-1).astype(np.float32)
    return {
        "frame_features": frame_feat,
        "feature_names": np.asarray(feature_names, dtype=object),
        "sample_rate": np.asarray([sr], dtype=np.int32),
    }
