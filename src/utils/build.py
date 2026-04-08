import csv
from typing import Any, List, Optional, TYPE_CHECKING

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from src.datamodules.split import (
    DevTestKFoldSplitProvider,
    GroupDevTestKFoldSplitProvider,
    GroupHoldoutSplitProvider,
    GroupKFoldSplitProvider,
    GroupTrainValTestHoldoutSplitProvider,
    HoldoutSplitProvider,
    KFoldSplitProvider,
    SplitProvider,
    StratifiedDevTestKFoldSplitProvider,
    StratifiedGroupDevTestKFoldSplitProvider,
    StratifiedGroupHoldoutSplitProvider,
    StratifiedGroupKFoldSplitProvider,
    StratifiedGroupTrainValTestHoldoutSplitProvider,
    StratifiedHoldoutSplitProvider,
    StratifiedKFoldSplitProvider,
    StratifiedTrainValTestHoldoutSplitProvider,
    TrainValTestHoldoutSplitProvider,
)

if TYPE_CHECKING:
    from lightning.pytorch.loggers import Logger


GROUP_AWARE_METHODS = {
    "group_holdout",
    "group_kfold",
    "group_train_val_test_holdout",
    "group_dev_test_kfold",
    "stratified_group_holdout",
    "stratified_group_kfold",
    "stratified_group_train_val_test_holdout",
    "stratified_group_dev_test_kfold",
}

KFOLD_METHODS = {
    "kfold",
    "group_kfold",
    "dev_test_kfold",
    "group_dev_test_kfold",
    "stratified_kfold",
    "stratified_group_kfold",
    "stratified_dev_test_kfold",
    "stratified_group_dev_test_kfold",
}

STRATIFIED_METHODS = {
    "stratified_holdout",
    "stratified_kfold",
    "stratified_train_val_test_holdout",
    "stratified_dev_test_kfold",
    "stratified_group_holdout",
    "stratified_group_kfold",
    "stratified_group_train_val_test_holdout",
    "stratified_group_dev_test_kfold",
}


def _load_split_records(data_file: str) -> List[dict[str, Any]]:
    """Load manifest records used to infer split metadata.

    Args:
        data_file: CSV manifest path referenced by split config.

    Returns:
        List[dict[str, Any]]: Parsed manifest rows.
    """

    with open(data_file, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_num_samples(cfg: DictConfig) -> int:
    """Resolve the sample count for split providers.

    Args:
        cfg: Top-level split configuration.

    Returns:
        int: Number of samples available to the split provider.

    Raises:
        ValueError: Raised when `data_file` is missing.
    """

    data_file = cfg.get("data_file")
    if not data_file:
        raise ValueError("split.data_file is required")

    return len(_load_split_records(str(data_file)))


def _resolve_group_ids(cfg: DictConfig, num_samples: int) -> List[Any]:
    """Resolve group identifiers for group-aware split providers.

    Args:
        cfg: Top-level split configuration.
        num_samples: Expected number of samples.

    Returns:
        List[Any]: Group identifiers aligned with sample order.

    Raises:
        ValueError: Raised when required configuration or manifest columns are missing.
    """

    data_file = cfg.get("data_file")
    if not data_file:
        raise ValueError("group-aware split requires split.data_file")

    group_id_column = cfg.get("group_id_column")
    if not group_id_column:
        raise ValueError("group-aware split requires split.group_id_column")

    rows = _load_split_records(str(data_file))
    if len(rows) != num_samples:
        raise ValueError("split.data_file row count does not match inferred sample count")
    if not rows:
        raise ValueError("split.data_file is empty")
    if str(group_id_column) not in rows[0]:
        raise ValueError(f"Column '{group_id_column}' not found in split.data_file")

    return [row[str(group_id_column)] for row in rows]


def _resolve_labels(cfg: DictConfig, num_samples: int) -> List[Any]:
    """Resolve label values for stratified split providers."""

    data_file = cfg.get("data_file")
    if not data_file:
        raise ValueError("stratified split requires split.data_file")

    label_column = cfg.get("label_column")
    if not label_column:
        raise ValueError("stratified split requires split.label_column")

    rows = _load_split_records(str(data_file))
    if len(rows) != num_samples:
        raise ValueError("split.data_file row count does not match inferred sample count")
    if not rows:
        raise ValueError("split.data_file is empty")
    if str(label_column) not in rows[0]:
        raise ValueError(f"Column '{label_column}' not found in split.data_file")

    return [row[str(label_column)] for row in rows]


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
        ValueError: Raised when `split.method` is not supported or a required K-fold
            `fold` argument is missing.
    """

    if not cfg or not cfg.get("method"):
        return None

    method = str(cfg.method)
    num_samples = _resolve_num_samples(cfg)
    group_ids = None
    labels = None
    if method in GROUP_AWARE_METHODS:
        group_ids = _resolve_group_ids(cfg, num_samples)
    if method in STRATIFIED_METHODS:
        labels = _resolve_labels(cfg, num_samples)
    if method in KFOLD_METHODS and fold is None:
        raise ValueError(f"split.method '{method}' requires a runtime fold value")
    if method in KFOLD_METHODS and cfg.get("n_folds") is None:
        raise ValueError(f"split.method '{method}' requires split.n_folds")

    common_kwargs = {
        "num_samples": int(num_samples),
        "seed": int(cfg.get("seed", 42)),
    }

    if method == "holdout":
        return HoldoutSplitProvider(
            val_ratio=float(cfg.get("val_ratio", 0.2)),
            **common_kwargs,
        )
    if method == "stratified_holdout":
        return StratifiedHoldoutSplitProvider(
            labels=labels,
            val_ratio=float(cfg.get("val_ratio", 0.2)),
            **common_kwargs,
        )
    if method == "group_holdout":
        return GroupHoldoutSplitProvider(
            group_ids=group_ids,
            val_ratio=float(cfg.get("val_ratio", 0.2)),
            **common_kwargs,
        )
    if method == "stratified_group_holdout":
        return StratifiedGroupHoldoutSplitProvider(
            group_ids=group_ids,
            labels=labels,
            val_ratio=float(cfg.get("val_ratio", 0.2)),
            **common_kwargs,
        )
    if method == "train_val_test_holdout":
        return TrainValTestHoldoutSplitProvider(
            val_ratio=float(cfg.get("val_ratio", 0.2)),
            test_ratio=float(cfg.get("test_ratio", 0.2)),
            **common_kwargs,
        )
    if method == "stratified_train_val_test_holdout":
        return StratifiedTrainValTestHoldoutSplitProvider(
            labels=labels,
            val_ratio=float(cfg.get("val_ratio", 0.2)),
            test_ratio=float(cfg.get("test_ratio", 0.2)),
            **common_kwargs,
        )
    if method == "group_train_val_test_holdout":
        return GroupTrainValTestHoldoutSplitProvider(
            group_ids=group_ids,
            val_ratio=float(cfg.get("val_ratio", 0.2)),
            test_ratio=float(cfg.get("test_ratio", 0.2)),
            **common_kwargs,
        )
    if method == "stratified_group_train_val_test_holdout":
        return StratifiedGroupTrainValTestHoldoutSplitProvider(
            group_ids=group_ids,
            labels=labels,
            val_ratio=float(cfg.get("val_ratio", 0.2)),
            test_ratio=float(cfg.get("test_ratio", 0.2)),
            **common_kwargs,
        )
    if method == "kfold":
        return KFoldSplitProvider(
            n_splits=int(cfg.get("n_folds")),
            fold=int(fold),
            **common_kwargs,
        )
    if method == "stratified_kfold":
        return StratifiedKFoldSplitProvider(
            labels=labels,
            n_splits=int(cfg.get("n_folds")),
            fold=int(fold),
            **common_kwargs,
        )
    if method == "group_kfold":
        return GroupKFoldSplitProvider(
            group_ids=group_ids,
            n_splits=int(cfg.get("n_folds")),
            fold=int(fold),
            **common_kwargs,
        )
    if method == "stratified_group_kfold":
        return StratifiedGroupKFoldSplitProvider(
            group_ids=group_ids,
            labels=labels,
            n_splits=int(cfg.get("n_folds")),
            fold=int(fold),
            **common_kwargs,
        )
    if method == "dev_test_kfold":
        return DevTestKFoldSplitProvider(
            n_splits=int(cfg.get("n_folds")),
            fold=int(fold),
            test_ratio=float(cfg.get("test_ratio", 0.2)),
            **common_kwargs,
        )
    if method == "stratified_dev_test_kfold":
        return StratifiedDevTestKFoldSplitProvider(
            labels=labels,
            n_splits=int(cfg.get("n_folds")),
            fold=int(fold),
            test_ratio=float(cfg.get("test_ratio", 0.2)),
            **common_kwargs,
        )
    if method == "group_dev_test_kfold":
        return GroupDevTestKFoldSplitProvider(
            group_ids=group_ids,
            n_splits=int(cfg.get("n_folds")),
            fold=int(fold),
            test_ratio=float(cfg.get("test_ratio", 0.2)),
            **common_kwargs,
        )
    if method == "stratified_group_dev_test_kfold":
        return StratifiedGroupDevTestKFoldSplitProvider(
            group_ids=group_ids,
            labels=labels,
            n_splits=int(cfg.get("n_folds")),
            fold=int(fold),
            test_ratio=float(cfg.get("test_ratio", 0.2)),
            **common_kwargs,
        )
    raise ValueError(f"Unsupported split.method '{method}'")


def build_datamodule(
    cfg: DictConfig,
    split_provider: Optional[SplitProvider] = None,
) -> L.LightningDataModule:
    """Instantiate a datamodule and inject resolved split inputs.

    Args:
        cfg: Datamodule config in `_target_ + init_args` form.
        split_provider: Optional provider that can generate split indices lazily.

    Returns:
        L.LightningDataModule: Instantiated datamodule.
    """

    materialized_cfg = _materialize_object_config(cfg)
    datamodule = hydra.utils.instantiate(materialized_cfg)
    datamodule._split_provider = split_provider
    return datamodule


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
