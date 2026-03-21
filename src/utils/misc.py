import json
import os
import shutil
from typing import Any, Dict

import lightning as L
from omegaconf import DictConfig, OmegaConf
import yaml


def load_dotenv() -> None:
    """Load environment variables from the nearest `.env` file when available.

    Returns:
        None: The function is used for its side effect of loading environment variables.
    """

    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv())
    except ImportError:
        print("Warning: python-dotenv not installed. Skipping .env loading.")


def log_hyperparameters(
    cfg: DictConfig,
    model: L.LightningModule,
    trainer: L.Trainer,
) -> None:
    """Log the resolved config and parameter counts to every configured logger.

    Args:
        cfg: Fully resolved runtime configuration.
        model: Instantiated Lightning model used to count parameters.
        trainer: Trainer that owns the configured loggers.

    Returns:
        None: The function is used for its side effect of logging hyperparameters.
    """

    hparams = OmegaConf.to_container(cfg, resolve=True)
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def save_json(payload: Dict[str, Any], path: str) -> None:
    """Write a JSON artifact with UTF-8 encoding.

    Args:
        payload: Serializable mapping written to disk.
        path: Destination path for the JSON file.

    Returns:
        None: The function writes a file and does not return a value.
    """

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_yaml(payload: Dict[str, Any], path: str) -> None:
    """Write a YAML artifact with UTF-8 encoding.

    Args:
        payload: Serializable mapping written to disk.
        path: Destination path for the YAML file.

    Returns:
        None: The function writes a file and does not return a value.
    """

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def write_resolved_config(cfg: DictConfig, path: str) -> None:
    """Persist a fully resolved Hydra config snapshot.

    Args:
        cfg: Runtime configuration to resolve and write.
        path: Destination path for the config snapshot.

    Returns:
        None: The function writes a file and does not return a value.
    """

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    OmegaConf.save(config=cfg, f=path, resolve=True)


def snapshot_code(exp_dir: str, code_dir: str = "code") -> None:
    """Copy Python source files into the run directory for reproducibility.

    Args:
        exp_dir: Root directory of the current run or experiment.
        code_dir: Relative directory name used to store the copied source tree.

    Returns:
        None: The function copies files and directories for reproducibility.
    """

    code_full_dir = os.path.join(exp_dir, code_dir)
    os.makedirs(code_full_dir, exist_ok=True)

    exclude_dirs = {"outputs", ".git", "__pycache__", "scripts"}
    exclude_files = {".env", ".DS_Store"}

    for item in os.listdir("."):
        if item in exclude_dirs or item in exclude_files:
            continue

        src = os.path.join(".", item)
        dst = os.path.join(code_full_dir, item)

        if os.path.isdir(src):
            shutil.copytree(
                src,
                dst,
                ignore=shutil.ignore_patterns("*.pyc", "__pycache__", ".git"),
            )
        elif os.path.isfile(src) and src.endswith(".py"):
            shutil.copy2(src, dst)
