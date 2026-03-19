import numpy as np
import pytorch_lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import os
from typing import Any
from omegaconf import DictConfig
from src.core.build import (
    build_model,
    build_datamodule,
    build_logger,
    build_callbacks,
    build_trainer,
)
from src.utils.utils import log_hyperparameters


def cv_loop(cfg: DictConfig) -> float:
    val_scores = []

    # =================================================================================
    # 交叉验证
    # =================================================================================
    n_splits = cfg.mode.get('n_folds', 5)
    for fold in range(n_splits):
        fold_cfg = cfg.copy()

        # 构建组件
        model = build_model(fold_cfg.model)
        datamodule = build_datamodule(fold_cfg.datamodule, fold)

        logger_cfg = fold_cfg.logger.copy()
        if logger_cfg._target_.endswith('MLFlowLogger'):
            run_name = logger_cfg.get('run_name', '')
            logger_cfg.run_name = f"{run_name}_fold{fold}" if run_name else f"fold{fold}"
        elif logger_cfg._target_.endswith('WandbLogger'):
            name = logger_cfg.get('name', '')
            logger_cfg.name = f"{name}_fold{fold}" if name else f"fold{fold}"

        logger = build_logger(logger_cfg)

        # 构建 callbacks，并修改 ModelCheckpoint 的路径
        callbacks = build_callbacks(fold_cfg.callbacks)
        for cb in callbacks:
            if isinstance(cb, ModelCheckpoint):
                cb.dirpath = os.path.join(fold_cfg.paths.output_dir, f"fold_{fold}_checkpoints")
                os.makedirs(cb.dirpath, exist_ok=True)

        trainer = build_trainer(cfg.trainer, logger, callbacks)

        log_hyperparameters(fold_cfg, model, trainer)

        trainer.fit(model, datamodule)

        monitor = cfg.get("monitor", "val/loss")
        score = trainer.callback_metrics.get(monitor)

        val_scores.append(score)

    return float(np.mean(val_scores))