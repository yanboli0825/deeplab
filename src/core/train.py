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


def train_loop(cfg: DictConfig) -> float:
    # =================================================================================
    # 构建lightning核心组件
    # =================================================================================
    model = build_model(cfg.model)
    datamodule = build_datamodule(cfg.datamodule)
    logger = build_logger(cfg.logger)
    callbacks = build_callbacks(cfg.callbacks)
    trainer = build_trainer(cfg.trainer, logger, callbacks)

    log_hyperparameters(cfg, model, trainer)

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=cfg.resume_ckpt
    )

    monitor = cfg.get("monitor", "val/loss")
    val_score = trainer.callback_metrics.get(monitor)

    # 如果测试，则返回测试集上的performance
    if cfg.get('test_after_train', False):
        ckpt_path = trainer.checkpoint_callback.best_model_path  if trainer.checkpoint_callback else None
        test_results = trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )
        test_score = test_results[0].get(monitor.replace("val/", "test/"))
        return test_score

    return val_score