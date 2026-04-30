from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def serialize_args_dict(args: Any) -> Dict[str, Any]:
    if isinstance(args, dict):
        src = args
    else:
        src = vars(args)
    return {key: _serialize_value(value) for key, value in src.items()}


def save_checkpoint(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def to_numpy_f32(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return arr.astype(np.float32, copy=False)
