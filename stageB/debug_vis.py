from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def _word_spans(words: Sequence[object], fps: int, n_frames: int) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for item in words or []:
        if isinstance(item, Mapping):
            start = float(item.get("start", 0.0))
            end = float(item.get("end", start))
            text = str(item.get("text", "")).strip()
        elif isinstance(item, (tuple, list)) and len(item) >= 3:
            start = float(item[0])
            end = float(item[1])
            text = str(item[2]).strip()
        else:
            continue
        s = max(0, min(n_frames, int(round(start * float(fps)))))
        e = max(s + 1, min(n_frames, int(round(end * float(fps)))))
        spans.append((s, e, text))
    return spans


def save_debug_plot(
    path: str | Path,
    *,
    prosody_frame: np.ndarray,
    token_gate: np.ndarray,
    token_codes: Mapping[str, np.ndarray],
    token_preds: Mapping[str, np.ndarray],
    words: Sequence[object],
    fps: int,
    token_stride: int,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        out_path.with_suffix(".txt").write_text("debug plot skipped: matplotlib unavailable\n", encoding="utf-8")
        return

    prosody = np.asarray(prosody_frame, dtype=np.float32)
    if prosody.ndim != 2 or prosody.shape[0] <= 0:
        prosody = np.zeros((1, 1), dtype=np.float32)
    gate = np.asarray(token_gate, dtype=np.float32).reshape(-1)

    n_frames = int(prosody.shape[0])
    time_frame = np.arange(n_frames, dtype=np.float32) / max(float(fps), 1.0)
    n_tokens = max(1, int(gate.shape[0]))
    time_token = (np.arange(n_tokens, dtype=np.float32) + 0.5) * float(token_stride) / max(float(fps), 1.0)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False, constrained_layout=True)

    energy = prosody[:, 0] if prosody.shape[1] >= 1 else np.zeros((n_frames,), dtype=np.float32)
    onset = prosody[:, 2] if prosody.shape[1] >= 3 else np.zeros((n_frames,), dtype=np.float32)
    axes[0].plot(time_frame, energy, label="log_energy", linewidth=1.2)
    axes[0].plot(time_frame, onset, label="onset_strength", linewidth=1.2, alpha=0.8)
    axes[0].set_title("Prosody")
    axes[0].legend(loc="upper right")

    spans = _word_spans(words, fps=fps, n_frames=n_frames)
    for idx, (s, e, text) in enumerate(spans):
        x0 = s / max(float(fps), 1.0)
        x1 = e / max(float(fps), 1.0)
        axes[0].axvspan(x0, x1, color=("#eef5e8" if idx % 2 == 0 else "#e8f1f8"), alpha=0.35, linewidth=0.0)
        if text:
            axes[0].text((x0 + x1) * 0.5, axes[0].get_ylim()[1], text, ha="center", va="bottom", fontsize=8, rotation=0)

    axes[1].plot(time_token, gate[:n_tokens], color="#c45508", linewidth=1.4)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Fusion Gate (content preference)")

    all_parts = list(token_codes.keys())
    axes[2].set_title("Ground Truth Tokens")
    axes[3].set_title("Predicted Tokens")
    for ax_idx, source in ((2, token_codes), (3, token_preds)):
        ax = axes[ax_idx]
        for part_idx, name in enumerate(all_parts):
            codes = np.asarray(source[name]).reshape(-1)
            if codes.size == 0:
                continue
            t = (np.arange(codes.size, dtype=np.float32) + 0.5) * float(token_stride) / max(float(fps), 1.0)
            offset = float(part_idx) * 0.15
            ax.step(t, codes.astype(np.float32) + offset, where="mid", linewidth=1.0, label=name)
        ax.legend(loc="upper right")

    axes[3].set_xlabel("Time (s)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
