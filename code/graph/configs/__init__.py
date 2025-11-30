from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml


@dataclass
class DatasetConfig:
    """Dataset-specific hyperparameters."""

    lr: float
    dropout: float
    weight_decay: float
    curvature: float
    epochs: int = 2000
    batch_size: int = 128
    hidden_dim: int = 16
    inner_act: str = "none"
    outer_act: str = "tangent"
    patience: int = 200


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    datasets: List[str]
    model_types: List[str]
    data_root: Path
    runs: int = 5
    base_seed: int = 40
    time_test: bool = False
    frechet_iters: Optional[Iterable[int]] = None
    override_epochs: Optional[int] = None
    override_batch_size: Optional[int] = None
    override_lr: Optional[float] = None
    override_dropout: Optional[float] = None
    override_weight_decay: Optional[float] = None
    override_curvature: Optional[float] = None


def load_dataset_configs(path: Path) -> Dict[str, DatasetConfig]:
    """Load dataset presets from a YAML file."""

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    configs: Dict[str, DatasetConfig] = {}
    for name, params in payload.items():
        configs[name.lower()] = DatasetConfig(**params)
    return configs


def apply_overrides(ds_cfg: DatasetConfig, exp_cfg: ExperimentConfig) -> DatasetConfig:
    """Return a copy of the dataset config after applying CLI overrides."""

    data = ds_cfg.__dict__.copy()
    if exp_cfg.override_epochs is not None:
        data["epochs"] = exp_cfg.override_epochs
    if exp_cfg.override_batch_size is not None:
        data["batch_size"] = exp_cfg.override_batch_size
    if exp_cfg.override_lr is not None:
        data["lr"] = exp_cfg.override_lr
    if exp_cfg.override_dropout is not None:
        data["dropout"] = exp_cfg.override_dropout
    if exp_cfg.override_weight_decay is not None:
        data["weight_decay"] = exp_cfg.override_weight_decay
    if exp_cfg.override_curvature is not None:
        data["curvature"] = exp_cfg.override_curvature
    return DatasetConfig(**data)

