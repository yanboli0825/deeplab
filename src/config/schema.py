"""Structured config helpers for the framework-owned Hydra surface."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf, open_dict


@dataclass
class ModeConfig:
    """Built-in execution mode config shared by train and cv."""

    name: str = "train"
    n_folds: Optional[int] = None


@dataclass
class PathsConfig:
    """Filesystem paths consumed by the framework runtime."""

    output_dir: str = "outputs"


@dataclass
class RuntimeConfig:
    """Resolved runtime fields injected by bootstrap or the runner."""

    mode: Optional[str] = None
    fold: Optional[int] = None
    parent_output_dir: Optional[str] = None
    output_dir: Optional[str] = None
    run_name: Optional[str] = None
    experiment_name: Optional[str] = None
    resolved_config_path: Optional[str] = None
    summary_path: Optional[str] = None
    artifact_index_path: Optional[str] = None


@dataclass
class ArtifactConfig:
    """Runtime artifact toggles and canonical file names."""

    enabled: bool = True
    summary_name: str = "run_summary.json"
    index_name: str = "artifacts.json"
    workflow_summary_name: str = "workflow_summary.json"
    workflow_index_name: str = "workflow_artifacts.json"
    split_manifest_name: str = "split_manifest.yaml"


@dataclass
class SplitConfig:
    """Framework-level split policy configuration."""

    method: Optional[str] = None
    data_file: Optional[str] = None
    group_id_column: Optional[str] = None
    label_column: Optional[str] = None
    n_folds: Optional[int] = None
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42


@dataclass
class PluginConfig:
    """Hydra object config with explicit plugin argument storage."""

    _target_: Optional[str] = None
    _recursive_: Optional[bool] = None
    init_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoggerCollectionConfig:
    """Logger config surface. `items` keeps the door open for multi-logger setups."""

    items: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CallbackCollectionConfig:
    """Callback collection config surface."""

    items: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class WorkflowConfig:
    """Workflow-level defaults used by Python orchestration modules."""

    root_dir: str = "outputs/workflows"


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
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    model: PluginConfig = field(default_factory=PluginConfig)
    datamodule: PluginConfig = field(default_factory=PluginConfig)
    callbacks: CallbackCollectionConfig = field(default_factory=CallbackCollectionConfig)
    logger: LoggerCollectionConfig = field(default_factory=LoggerCollectionConfig)
    trainer: PluginConfig = field(default_factory=PluginConfig)
    hydra: Dict[str, Any] = field(default_factory=dict)
    hpo: Optional[Dict[str, Any]] = None


def _normalize_object_config(cfg: DictConfig) -> DictConfig:
    """Normalize a plugin object config to `_target_ + init_args` form.

    Args:
        cfg: Raw Hydra object config.

    Returns:
        DictConfig: Normalized object config.

    Raises:
        TypeError: Raised when the config does not resolve to a mapping.
    """

    if not cfg:
        return OmegaConf.structured(PluginConfig)

    payload = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(payload, dict):
        raise TypeError("Object config must resolve to a mapping")

    normalized = dict(payload)
    if "init_args" not in normalized:
        normalized["init_args"] = {
            key: value
            for key, value in payload.items()
            if not str(key).startswith("_")
        }
        for key in list(normalized["init_args"].keys()):
            normalized.pop(key, None)
    return OmegaConf.merge(OmegaConf.structured(PluginConfig), OmegaConf.create(normalized))


def _normalize_logger_config(cfg: DictConfig) -> DictConfig:
    """Normalize logger config to the framework `items` collection format.

    Args:
        cfg: Raw logger config.

    Returns:
        DictConfig: Normalized logger collection config.

    Raises:
        TypeError: Raised when the config shape is not supported.
    """

    if not cfg:
        return OmegaConf.structured(LoggerCollectionConfig)

    payload = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(payload, dict):
        raise TypeError("logger config must resolve to a mapping")

    if "items" in payload:
        items = payload["items"]
    else:
        items = [payload]

    if not isinstance(items, list):
        raise TypeError("logger.items must be a list")

    return OmegaConf.merge(
        OmegaConf.structured(LoggerCollectionConfig),
        OmegaConf.create({"items": items}),
    )


def _normalize_callback_config(cfg: DictConfig) -> DictConfig:
    """Normalize callback config to the framework `items` collection format.

    Args:
        cfg: Raw callback config.

    Returns:
        DictConfig: Normalized callback collection config.

    Raises:
        TypeError: Raised when the config shape is not supported.
    """

    if not cfg:
        return OmegaConf.structured(CallbackCollectionConfig)

    payload = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(payload, dict):
        raise TypeError("callbacks config must resolve to a mapping")

    if "items" in payload and isinstance(payload["items"], dict):
        items = payload["items"]
    else:
        items = payload

    return OmegaConf.merge(
        OmegaConf.structured(CallbackCollectionConfig),
        OmegaConf.create({"items": items}),
    )


def _validate_mode(mode_cfg: DictConfig) -> None:
    """Validate the built-in runtime mode configuration.

    Args:
        mode_cfg: Mode configuration resolved from Hydra.

    Returns:
        None: The function raises on unsupported or invalid mode settings.

    Raises:
        ValueError: Raised when the mode is unsupported or incomplete.
    """

    mode_name = str(mode_cfg.get("name", "train"))
    validated = OmegaConf.merge(OmegaConf.structured(ModeConfig), mode_cfg)
    if mode_name == "train":
        return
    if mode_name == "cv":
        if validated.n_folds is None or int(validated.n_folds) < 2:
            raise ValueError("cv mode requires mode.n_folds >= 2")
        return
    raise ValueError(f"Unsupported mode '{mode_name}'. Only 'train' and 'cv' are allowed.")


def validate_app_config(cfg: DictConfig) -> DictConfig:
    """Validate the framework-owned config surface while preserving plugin extensibility.

    Args:
        cfg: Hydra-composed application config.

    Returns:
        DictConfig: Normalized config with framework-owned sections rewritten.

    Raises:
        ValueError: Raised when required framework fields are missing or invalid.
    """

    known_fields = list(AppConfig.__dataclass_fields__.keys())
    candidate = OmegaConf.create({name: cfg.get(name) for name in known_fields if name in cfg})
    validated = OmegaConf.merge(OmegaConf.structured(AppConfig), candidate)

    _validate_mode(validated.mode)
    validated.model = _normalize_object_config(validated.model)
    validated.datamodule = _normalize_object_config(validated.datamodule)
    validated.trainer = _normalize_object_config(validated.trainer)
    validated.logger = _normalize_logger_config(validated.logger)
    validated.callbacks = _normalize_callback_config(validated.callbacks)

    if validated.model.get("_target_") is None:
        raise ValueError("model._target_ is required")
    if validated.datamodule.get("_target_") is None:
        raise ValueError("datamodule._target_ is required")
    if validated.trainer.get("_target_") is None:
        raise ValueError("trainer._target_ is required")

    output = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(output):
        for name in known_fields:
            output[name] = getattr(validated, name)
    return output
