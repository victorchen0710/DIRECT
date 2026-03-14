"""
Build JSONL manifests for BEAT audio→skeleton training.

Each line contains:
{
    "id": "beat_english_v0.2.1/1/1_wayne_0_1_1",
    "audio": "beat/beat_english_v0.2.1/1/1_wayne_0_1_1.wav",
    "feature": "w2v2_base/w2v2_base/beat_english_v0.2.1/1/1_wayne_0_1_1.wav.npz",
    "bvh": "beat/beat_english_v0.2.1/1/1_wayne_0_1_1.bvh",
    "textgrid": "beat/beat_english_v0.2.1/1/1_wayne_0_1_1.TextGrid",
    "gesture_txt": "beat/beat_english_v0.2.1/1/1_wayne_0_1_1.txt",
    "emotion_csv": "beat/beat_english_v0.2.1/1/1_wayne_0_1_1.csv",
    "text": "the first thing i like to do on weekends is relaxing ...",
    "word_timestamps": [
        {"word": "the", "start": 1.35, "end": 1.46},
        {"word": "first", "start": 1.46, "end": 1.80}
    ],
    "phone_timestamps": [
        {"phone": "DH", "start": 1.35, "end": 1.41},
        {"phone": "AH0", "start": 1.41, "end": 1.46}
    ],
    "style_bone_lengths": [...],
    "duration": 69.0,
    "raw_fps": 120,
    "fps": 120
}
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf


def discover_samples(beat_root: Path, feat_root: Path) -> List[Dict]:
    wavs = sorted(beat_root.rglob("*.wav"))
    samples: List[Dict] = []
    missing_feat = 0

    for i, wav in enumerate(wavs, 1):
        rel = wav.relative_to(beat_root)
        feat_path = feat_root / "w2v2_base" / rel.with_suffix(".wav.npz")
        if not feat_path.exists():
            missing_feat += 1
            continue

        bvh_path = beat_root / rel.with_suffix(".bvh")
        textgrid = beat_root / rel.with_suffix(".TextGrid")
        gesture_txt = beat_root / rel.with_suffix(".txt")
        emotion_csv = beat_root / rel.with_suffix(".csv")

        tg = (
            parse_textgrid_alignment(textgrid)
            if textgrid.exists()
            else {
                "text": None,
                "word_timestamps": None,
            }
        )

        style_bones = parse_bvh_bone_lengths(bvh_path) if bvh_path.exists() else None
        duration = load_duration(wav)
        raw_fps = parse_bvh_fps(bvh_path) if bvh_path.exists() else None
        raw_fps = raw_fps if raw_fps is not None else 120

        # 你当前用的是原始 BEAT motion，所以训练/生成 fps 也先写 120
        fps = raw_fps

        samples.append(
            {
                "id": str(rel.with_suffix("")),
                "audio": str(wav),
                "feature": str(feat_path),
                "bvh": str(bvh_path) if bvh_path.exists() else None,
                "textgrid": str(textgrid) if textgrid.exists() else None,
                "gesture_txt": str(gesture_txt) if gesture_txt.exists() else None,
                "emotion_csv": str(emotion_csv) if emotion_csv.exists() else None,
                "text": tg["text"],
                "word_timestamps": tg["word_timestamps"],
                "style_bone_lengths": style_bones,
                "duration": float(duration),
                "raw_fps": int(raw_fps),
                "fps": int(fps),
            }
        )

        if i % 200 == 0:
            print(f"[INFO] processed {i}/{len(wavs)}")

    if missing_feat:
        print(f"[WARN] skipped {missing_feat} wavs without extracted features")
    print(f"[INFO] collected {len(samples)} samples")
    return samples


def load_duration(wav_path: Path) -> float:
    info = sf.info(str(wav_path))
    return float(info.frames) / float(info.samplerate)


def parse_textgrid_alignment(path: Path) -> Dict[str, Optional[List[Dict]]]:
    """
    Parse Praat TextGrid and only keep the words tier.
    Return:
    {
        "text": str | None,
        "word_timestamps": [{"word": str, "start": float, "end": float}, ...] | None,
    }
    """
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return {
            "text": None,
            "word_timestamps": None,
        }

    current_tier_name = None
    current_tier_class = None
    current_xmin = None
    current_xmax = None

    word_timestamps: List[Dict] = []

    for ln in lines:
        s = ln.strip()

        m_class = re.match(r'class *= *"(.*)"', s)
        if m_class:
            current_tier_class = m_class.group(1).strip()
            continue

        m_name = re.match(r'name *= *"(.*)"', s)
        if m_name:
            current_tier_name = m_name.group(1).strip().lower()
            continue

        m_xmin = re.match(r'xmin *= *([0-9.]+)', s)
        if m_xmin:
            current_xmin = float(m_xmin.group(1))
            continue

        m_xmax = re.match(r'xmax *= *([0-9.]+)', s)
        if m_xmax:
            current_xmax = float(m_xmax.group(1))
            continue

        m_text = re.match(r'text *= *"(.*)"', s)
        if not m_text:
            continue

        if current_tier_class != "IntervalTier" or current_tier_name != "words":
            continue
        if current_xmin is None or current_xmax is None:
            continue

        label = m_text.group(1).strip()
        if not label:
            continue

        word_timestamps.append(
            {
                "word": label,
                "start": float(current_xmin),
                "end": float(current_xmax),
            }
        )

    text = " ".join(x["word"] for x in word_timestamps) if word_timestamps else None

    return {
        "text": text,
        "word_timestamps": word_timestamps or None,
    }

def parse_bvh_fps(path: Path) -> Optional[int]:
    """
    Parse BVH 'Frame Time:' and convert to fps.
    Example:
        Frame Time: 0.00833333 -> 120 fps
    """
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None

    for ln in lines:
        s = ln.strip()
        if s.lower().startswith("frame time:"):
            try:
                frame_time = float(s.split(":")[1].strip())
            except Exception:
                return None
            if frame_time > 0:
                return int(round(1.0 / frame_time))
    return None


def parse_bvh_bone_lengths(path: Path) -> Optional[List[float]]:
    """
    Parse BVH hierarchy offsets to derive bone length vector (style code).
    """
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None

    offsets = []
    for ln in lines:
        ln = ln.strip()
        if ln.startswith("OFFSET"):
            parts = ln.split()
            if len(parts) == 4:
                x, y, z = map(float, parts[1:])
                length = float((x * x + y * y + z * z) ** 0.5)
                offsets.append(length)
        if ln.upper() == "MOTION":
            break

    return offsets if offsets else None


def split_samples(samples: List[Dict], train_ratio: float, val_ratio: float, seed: int):
    random.Random(seed).shuffle(samples)
    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]
    return train, val, test


def save_jsonl(samples: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[INFO] wrote {len(samples)} rows -> {path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("beat_root", type=Path, help="Root of BEAT data (contains beat_english_v0.2.1)")
    ap.add_argument("feature_root", type=Path, help="Root where w2v2 features were written")
    ap.add_argument("--out_dir", type=Path, default=Path("manifests"), help="Where to write JSONL splits")
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    samples = discover_samples(args.beat_root, args.feature_root)
    train, val, test = split_samples(samples, args.train_ratio, args.val_ratio, args.seed)
    save_jsonl(train, args.out_dir / "train.jsonl")
    save_jsonl(val, args.out_dir / "val.jsonl")
    save_jsonl(test, args.out_dir / "test.jsonl")


if __name__ == "__main__":
    main()