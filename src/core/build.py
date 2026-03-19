from typing import Any, List, Optional, TYPE_CHECKING
from omegaconf import DictConfig
import lightning as L
import hydra

if TYPE_CHECKING:
    from lightning.pytorch.loggers import Logger


def build_model(cfg: DictConfig) -> L.LightningModule:
    """构建模型

    Args:
        cfg: 模型配置字典，必须包含 _target_ 指向模型类

    Returns:
        LightningModule 实例
    """
    return hydra.utils.instantiate(cfg)


def build_datamodule(
    cfg: DictConfig,
    fold: Optional[int] = None
) -> L.LightningDataModule:
    """构建数据模块

    Args:
        cfg: 数据配置字典
        fold: 交叉验证折数，None 表示不使用交叉验证

    Returns:
        LightningDataModule 实例
    """
    return hydra.utils.instantiate(
        cfg,
        fold=fold
    )


def build_logger(cfg: DictConfig) -> "Logger":
    """构建日志记录器

    Args:
        cfg: 日志配置，支持 mlflow 或 wandb

    Returns:
        Logger 实例
    """
    import mlflow
    mlflow.enable_system_metrics_logging()
    return hydra.utils.instantiate(cfg)


def build_callbacks(cfg: DictConfig) -> List[L.Callback]:
    """构建回调函数列表

    Args:
        cfg: 回调配置字典

    Returns:
        Callback 对象列表
    """
    callbacks_list: List[L.Callback] = []

    for cb_name, cb_conf in cfg.items():
        if cb_conf is not None:
            cb: L.Callback = hydra.utils.instantiate(cb_conf)
            callbacks_list.append(cb)
    return callbacks_list


def build_trainer(
    cfg: DictConfig,
    logger: "Logger",
    callbacks: List[L.Callback]
) -> L.Trainer:
    """构建训练器

    Args:
        cfg: 训练器配置
        logger: 日志记录器
        callbacks: 回调函数列表

    Returns:
        Trainer 实例
    """
    return hydra.utils.instantiate(
        cfg,
        logger=logger,
        callbacks=callbacks
    )
