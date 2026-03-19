from typing import List, Optional, TYPE_CHECKING

import hydra
import lightning as L
from omegaconf import DictConfig

from src.datamodules.split import SplitIndices

if TYPE_CHECKING:
    from lightning.pytorch.loggers import Logger


def build_model(cfg: DictConfig) -> L.LightningModule:
    return hydra.utils.instantiate(cfg)


def build_datamodule(
    cfg: DictConfig,
    split_indices: Optional[SplitIndices] = None,
) -> L.LightningDataModule:
    return hydra.utils.instantiate(
        cfg,
        split_indices=split_indices,
    )


def build_logger(cfg: DictConfig) -> "Logger":
    logger = hydra.utils.instantiate(cfg)

    try:
        if isinstance(logger, L.pytorch.loggers.MLFlowLogger):
            import mlflow
            mlflow.enable_system_metrics_logging()
    except Exception:
        # 不让 system metrics logging 失败阻断主训练
        pass

    return logger


def build_callbacks(cfg: DictConfig) -> List[L.Callback]:
    callbacks_list: List[L.Callback] = []

    for _, cb_conf in cfg.items():
        if cb_conf is not None:
            callbacks_list.append(hydra.utils.instantiate(cb_conf))

    return callbacks_list


def build_trainer(
    cfg: DictConfig,
    logger: "Logger",
    callbacks: List[L.Callback],
) -> L.Trainer:
    return hydra.utils.instantiate(
        cfg,
        logger=logger,
        callbacks=callbacks,
    )
