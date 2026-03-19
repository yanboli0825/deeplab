"""Single-run runtime implementation shared by train and cv modes."""

import os
from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf, open_dict

from src.core.contracts import RunContext, RunSummary
from src.datamodules.split import SplitIndices
from src.utils.build import (
    build_callbacks,
    build_datamodule,
    build_logger,
    build_model,
    build_trainer,
)
from src.utils.misc import log_hyperparameters, save_json, write_resolved_config


@dataclass
class RunResult:
    """In-memory result returned to the caller after a concrete run finishes."""

    context: RunContext
    summary: RunSummary
    model: Optional[Any] = None
    datamodule: Optional[Any] = None


def _clone_cfg(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def _to_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _resolve_suffix(
    fold: Optional[int] = None,
    run_name_suffix: Optional[str] = None,
) -> Optional[str]:
    if run_name_suffix:
        return run_name_suffix
    if fold is not None:
        return f"fold{fold}"
    return None


def _resolve_experiment_name(cfg: DictConfig) -> str:
    experiment_name = cfg.get("experiment_name", "default")
    return str(experiment_name)


def _resolve_run_name(cfg: DictConfig, suffix: Optional[str]) -> str:
    base_name = cfg.get("run_name") or cfg.mode.name
    if suffix is None:
        return str(base_name)
    return f"{base_name}_{suffix}"


def _build_run_context(
    cfg: DictConfig,
    fold: Optional[int] = None,
    run_name_suffix: Optional[str] = None,
) -> RunContext:
    suffix = _resolve_suffix(fold=fold, run_name_suffix=run_name_suffix)
    experiment_name = _resolve_experiment_name(cfg)
    run_name = _resolve_run_name(cfg, suffix=suffix)

    parent_output_dir = str(cfg.paths.output_dir)
    output_dir = parent_output_dir
    if suffix is not None:
        output_dir = os.path.join(parent_output_dir, "runs", run_name)

    return RunContext(
        mode=str(cfg.mode.name),
        experiment_name=experiment_name,
        run_name=run_name,
        output_dir=output_dir,
        resolved_config_path=os.path.join(output_dir, "config.yaml"),
        summary_path=os.path.join(output_dir, "run_summary.json"),
        fold=fold,
        parent_output_dir=parent_output_dir,
    )


def _with_runtime_context(cfg: DictConfig, context: RunContext) -> DictConfig:
    local_cfg = _clone_cfg(cfg)
    with open_dict(local_cfg):
        local_cfg.experiment_name = context.experiment_name
        local_cfg.run_name = context.run_name
        local_cfg.paths.output_dir = context.output_dir
        local_cfg.runtime = {
            "mode": context.mode,
            "fold": context.fold,
            "parent_output_dir": context.parent_output_dir,
            "output_dir": context.output_dir,
            "run_name": context.run_name,
            "experiment_name": context.experiment_name,
            "resolved_config_path": context.resolved_config_path,
            "summary_path": context.summary_path,
        }
    return local_cfg


def run_experiment(
    cfg: DictConfig,
    fold: Optional[int] = None,
    ckpt_path: Optional[str] = None,
    split_indices: Optional[SplitIndices] = None,
    run_name_suffix: Optional[str] = None,
    keep_artifacts: bool = False,
) -> RunResult:
    """Execute one concrete training unit and persist its standard summary artifact."""

    context = _build_run_context(cfg, fold=fold, run_name_suffix=run_name_suffix)
    local_cfg = _with_runtime_context(cfg, context)
    os.makedirs(context.output_dir, exist_ok=True)
    write_resolved_config(local_cfg, context.resolved_config_path)

    model = build_model(local_cfg.model)
    datamodule = build_datamodule(local_cfg.datamodule, split_indices=split_indices)
    logger = build_logger(local_cfg.logger)
    callbacks = build_callbacks(local_cfg.callbacks)
    trainer = build_trainer(local_cfg.trainer, logger, callbacks)

    log_hyperparameters(local_cfg, model, trainer)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path if ckpt_path is not None else local_cfg.get("resume_ckpt", None),
    )

    monitor = str(local_cfg.get("monitor", "val/loss"))
    val_score = _to_float(trainer.callback_metrics.get(monitor))

    best_ckpt_path = None
    checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
    if checkpoint_callback is not None:
        best_ckpt_path = checkpoint_callback.best_model_path or None

    test_score = None
    if local_cfg.get("test_after_train", False):
        test_results = trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=best_ckpt_path,
        )
        test_monitor = monitor.replace("val/", "test/")
        if test_results:
            test_score = _to_float(test_results[0].get(test_monitor))

    summary = RunSummary(
        mode=context.mode,
        experiment_name=context.experiment_name,
        run_name=context.run_name,
        output_dir=context.output_dir,
        resolved_config_path=context.resolved_config_path,
        summary_path=context.summary_path,
        monitor=monitor,
        val_score=val_score,
        test_score=test_score,
        best_ckpt_path=best_ckpt_path,
        fold=fold,
    )
    save_json(summary.to_dict(), context.summary_path)

    return RunResult(
        context=context,
        summary=summary,
        model=model if keep_artifacts else None,
        datamodule=datamodule if keep_artifacts else None,
    )
