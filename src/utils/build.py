from typing import Any, List, Optional, TYPE_CHECKING

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from src.datamodules.split import (
    GroupHoldoutSplitProvider,
    GroupKFoldSplitProvider,
    HoldoutSplitProvider,
    KFoldSplitProvider,
    SplitIndices,
    SplitProvider,
)

if TYPE_CHECKING:
    from lightning.pytorch.loggers import Logger


def build_model(cfg: DictConfig) -> L.LightningModule:
    """Instantiate the model object defined by Hydra config.

    Args:
        cfg: Model config in `_target_ + init_args` form.

    Returns:
        L.LightningModule: Instantiated task model.
    """

    materialized_cfg = _materialize_object_config(cfg)
    return hydra.utils.instantiate(materialized_cfg)


def _materialize_object_config(cfg: DictConfig) -> DictConfig:
    """Flatten `init_args` into a Hydra object config before instantiation.

    Args:
        cfg: Framework-normalized object config.

    Returns:
        DictConfig: Hydra-compatible config with all constructor arguments at top level.

    Raises:
        TypeError: Raised when `cfg` does not resolve to a mapping.
    """

    payload = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(payload, dict):
        raise TypeError("object config must resolve to a mapping")

    output = {
        key: value
        for key, value in payload.items()
        if key != "init_args"
    }
    init_args = payload.get("init_args", {})
    if isinstance(init_args, dict):
        output.update(init_args)
    return OmegaConf.create(output)


def build_split_provider(cfg: Optional[DictConfig], fold: Optional[int] = None) -> Optional[SplitProvider]:
    """Build a split provider from top-level split config.

    Args:
        cfg: Optional split configuration.
        fold: Fold index injected by the runner for CV execution.

    Returns:
        Optional[SplitProvider]: Concrete split provider or `None` when split config is disabled.

    Raises:
        ValueError: Raised when `split.method` is not supported.
    """

    if not cfg or not cfg.get("method"):
        return None

    method = str(cfg.method)
    num_samples = cfg.get("num_samples")
    if num_samples is None:
        return None

    common_kwargs = {
        "num_samples": int(num_samples),
        "seed": int(cfg.get("seed", 42)),
        "candidate_indices": cfg.get("candidate_indices"),
    }

    if method == "holdout":
        return HoldoutSplitProvider(
            val_ratio=float(cfg.get("val_ratio", 0.2)),
            **common_kwargs,
        )
    if method == "group_holdout":
        return GroupHoldoutSplitProvider(
            group_ids=list(cfg.get("group_ids") or []),
            val_ratio=float(cfg.get("val_ratio", 0.2)),
            **common_kwargs,
        )
    if method == "kfold":
        return KFoldSplitProvider(
            n_splits=int(cfg.get("n_splits", 5)),
            fold=int(fold if fold is not None else cfg.get("fold", 0)),
            **common_kwargs,
        )
    if method == "group_kfold":
        return GroupKFoldSplitProvider(
            group_ids=list(cfg.get("group_ids") or []),
            n_splits=int(cfg.get("n_splits", 5)),
            fold=int(fold if fold is not None else cfg.get("fold", 0)),
            **common_kwargs,
        )
    raise ValueError(f"Unsupported split.method '{method}'")


def build_datamodule(
    cfg: DictConfig,
    split_indices: Optional[SplitIndices] = None,
    split_provider: Optional[SplitProvider] = None,
) -> L.LightningDataModule:
    """Instantiate a datamodule and inject resolved split inputs.

    Args:
        cfg: Datamodule config in `_target_ + init_args` form.
        split_indices: Explicit split indices provided by the caller.
        split_provider: Optional provider that can generate split indices lazily.

    Returns:
        L.LightningDataModule: Instantiated datamodule.
    """

    materialized_cfg = _materialize_object_config(cfg)
    return hydra.utils.instantiate(
        materialized_cfg,
        split_indices=split_indices,
        split_provider=split_provider,
    )


def build_logger(cfg: DictConfig) -> Any:
    """Instantiate one or more Lightning loggers.

    Args:
        cfg: Logger config with `items`.

    Returns:
        Any: A single logger, a list of loggers, or `False` when logging is disabled.
    """

    items = cfg.get("items", [])
    if not items:
        return False

    loggers = [hydra.utils.instantiate(item) for item in items]
    for logger in loggers:
        try:
            if isinstance(logger, L.pytorch.loggers.MLFlowLogger):
                import mlflow

                mlflow.enable_system_metrics_logging()
        except Exception:
            pass

    if len(loggers) == 1:
        return loggers[0]
    return loggers


def build_callbacks(cfg: DictConfig) -> List[L.Callback]:
    """Instantiate configured Lightning callbacks.

    Args:
        cfg: Callback config with `items`.

    Returns:
        List[L.Callback]: Instantiated callbacks in config iteration order.
    """

    callbacks_list: List[L.Callback] = []
    items = cfg.get("items", cfg)
    for _, cb_conf in items.items():
        if cb_conf is not None:
            callbacks_list.append(hydra.utils.instantiate(cb_conf))
    return callbacks_list


def build_trainer(
    cfg: DictConfig,
    logger: "Logger",
    callbacks: List[L.Callback],
) -> L.Trainer:
    """Instantiate the Lightning trainer with logger and callback dependencies.

    Args:
        cfg: Trainer config in `_target_ + init_args` form.
        logger: Logger object attached to the trainer.
        callbacks: Callback list attached to the trainer.

    Returns:
        L.Trainer: Instantiated trainer.
    """

    return hydra.utils.instantiate(
        _materialize_object_config(cfg),
        logger=logger,
        callbacks=callbacks,
    )
