from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Iterable, Sequence

import torch


@dataclass(frozen=True)
class SpatialTaskSpec:
    task_id: int
    name: str
    source_parts: tuple[str, ...]
    target_parts: tuple[str, ...]


DEFAULT_SPATIAL_TASK_SPECS: tuple[SpatialTaskSpec, ...] = (
    SpatialTaskSpec(0, "upper_to_hand", ("upper",), ("hand",)),
    SpatialTaskSpec(1, "upper_to_lower", ("upper",), ("lower",)),
    SpatialTaskSpec(2, "hand_to_upper", ("hand",), ("upper",)),
    SpatialTaskSpec(3, "upper_hand_to_lower", ("upper", "hand"), ("lower",)),
)

DEFAULT_SPATIAL_TASK_WEIGHTS: Dict[str, float] = {
    "upper_to_hand": 0.30,
    "upper_to_lower": 0.25,
    "hand_to_upper": 0.20,
    "upper_hand_to_lower": 0.25,
}


def get_default_task_registry() -> dict[str, SpatialTaskSpec]:
    return {task.name: task for task in DEFAULT_SPATIAL_TASK_SPECS}


def parse_spatial_task_names(task_names: str | None) -> list[str]:
    if not task_names:
        return [task.name for task in DEFAULT_SPATIAL_TASK_SPECS]
    return [name.strip() for name in str(task_names).split(",") if name.strip()]


def build_spatial_tasks(task_names: str | None) -> list[SpatialTaskSpec]:
    registry = get_default_task_registry()
    tasks: list[SpatialTaskSpec] = []
    for name in parse_spatial_task_names(task_names):
        if name not in registry:
            raise ValueError(f"Unknown spatial task '{name}'. Available: {sorted(registry)}")
        tasks.append(registry[name])
    if not tasks:
        raise ValueError("At least one spatial task must be enabled.")
    return tasks


def parse_spatial_task_weights(task_weights: str | None, tasks: Sequence[SpatialTaskSpec]) -> list[float]:
    weights = {name: float(value) for name, value in DEFAULT_SPATIAL_TASK_WEIGHTS.items()}
    if task_weights:
        for item in str(task_weights).split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"Invalid spatial task weight '{item}', expected name=value")
            name, value = item.split("=", 1)
            weights[name.strip()] = float(value.strip())

    out = [max(0.0, float(weights.get(task.name, 0.0))) for task in tasks]
    total = sum(out)
    if total <= 0.0:
        out = [1.0 for _ in tasks]
        total = float(len(out))
    return [value / total for value in out]


def sample_spatial_task(tasks: Sequence[SpatialTaskSpec], weights: Sequence[float], rng: random.Random) -> SpatialTaskSpec:
    if len(tasks) != len(weights):
        raise ValueError("tasks and weights must have the same length")
    return rng.choices(list(tasks), weights=list(weights), k=1)[0]


def task_to_part_masks(
    task: SpatialTaskSpec,
    part_names: Sequence[str],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.zeros((int(batch_size), len(part_names)), dtype=torch.bool, device=device)
    target = torch.zeros((int(batch_size), len(part_names)), dtype=torch.bool, device=device)
    for idx, name in enumerate(part_names):
        if name in task.source_parts:
            source[:, idx] = True
        if name in task.target_parts:
            target[:, idx] = True
    return source, target


def task_name_list(tasks: Iterable[SpatialTaskSpec]) -> list[str]:
    return [task.name for task in tasks]
