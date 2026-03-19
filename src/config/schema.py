"""Typed config helpers for the runtime-critical Hydra surface."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf


@dataclass
class ModeConfig:
    """Fields shared by the built-in runtime modes."""

    name: str = "train"
    n_folds: int = 1


@dataclass
class PathsConfig:
    """Paths consumed directly by the framework runtime."""

    data_dir: str = "data"
    output_dir: str = "outputs"


@dataclass
class AppConfig:
    """Top-level config contract validated before the framework runs."""

    project_name: str = "deeplab"
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    seed: int = 42
    monitor: str = "val/loss"
    test_after_train: bool = False
    resume_ckpt: Optional[str] = None
    mode: ModeConfig = field(default_factory=ModeConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: Dict[str, Any] = field(default_factory=dict)
    datamodule: Dict[str, Any] = field(default_factory=dict)
    callbacks: Dict[str, Any] = field(default_factory=dict)
    logger: Dict[str, Any] = field(default_factory=dict)
    trainer: Dict[str, Any] = field(default_factory=dict)
    hydra: Dict[str, Any] = field(default_factory=dict)
    hpo: Optional[Dict[str, Any]] = None


def validate_app_config(cfg: DictConfig) -> DictConfig:
    """Validate runtime-critical top-level fields while keeping plugin configs flexible."""

    schema = OmegaConf.structured(AppConfig)
    known_fields = list(AppConfig.__dataclass_fields__.keys())
    candidate = OmegaConf.create({name: cfg.get(name) for name in known_fields if name in cfg})
    validated = OmegaConf.merge(schema, candidate)

    mode_name = validated.mode.name
    if mode_name not in {"train", "cv"}:
        raise ValueError(f"Unsupported mode '{mode_name}'. Only 'train' and 'cv' are allowed.")

    if mode_name == "cv" and int(validated.mode.n_folds) < 2:
        raise ValueError("cv mode requires mode.n_folds >= 2")

    return cfg
