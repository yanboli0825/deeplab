import os
import shutil
from typing import Any, Dict
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import Logger
import lightning as L


def load_dotenv() -> None:
    """加载.env环境变量"""
    try:
        from dotenv import load_dotenv
        from dotenv import find_dotenv
        load_dotenv(find_dotenv())
    except ImportError:
        print("Warning: python-dotenv not installed. Skipping .env loading.")


def log_hyperparameters(
    cfg: DictConfig,
    model: L.LightningModule,
    trainer: L.Trainer
) -> None:
    """将 Hydra 配置和模型参数量记录到 logger"""
    hparams = OmegaConf.to_container(cfg, resolve=True)
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def snapshot_code(exp_dir: str, code_dir: str = "code") -> None:
    """保存代码快照"""
    code_full_dir = os.path.join(exp_dir, code_dir)

    os.makedirs(code_full_dir, exist_ok=True)

    exclude_dirs = {"outputs", ".git", "__pycache__", "data"}
    exclude_files = {".env", ".DS_Store"}

    for item in os.listdir("."):
        if item in exclude_dirs or item in exclude_files:
            continue

        src = os.path.join(".", item)
        dst = os.path.join(code_full_dir, item)

        if os.path.isdir(src):
            shutil.copytree(
                src, dst,
                ignore=shutil.ignore_patterns('*.pyc', '__pycache__', '.git')
            )
        elif os.path.isfile(src) and src.endswith('.py'):
            shutil.copy2(src, dst)
