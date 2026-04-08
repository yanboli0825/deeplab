"""Single-run runtime implementation shared by train and cv modes."""

import os
from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf, open_dict

from src.core.contracts import ArtifactIndex, RunContext, RunSummary
from src.utils.build import (
    build_callbacks,
    build_datamodule,
    build_logger,
    build_model,
    build_split_provider,
    build_trainer,
)
from src.utils.misc import log_hyperparameters, save_json, write_resolved_config

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


@dataclass
class RunResult:
    """In-memory result returned to the caller after a concrete run finishes."""

    context: RunContext
    summary: RunSummary
    artifact_index: ArtifactIndex
    model: Optional[Any] = None
    datamodule: Optional[Any] = None


def _clone_cfg(cfg: DictConfig) -> DictConfig:
    """Create a mutable copy of the input config without resolving interpolations.

    Args:
        cfg: Original runtime configuration.

    Returns:
        DictConfig: A cloned config object safe to mutate locally.
    """

    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def _to_float(value: Any, default: float = float("nan")) -> float:
    """Convert Lightning outputs to a Python float.

    Args:
        value: Metric value that may be `None`, a tensor, or a Python scalar.
        default: Fallback value used when `value` is `None`.

    Returns:
        float: The converted metric value.
    """

    if value is None:
        return default
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _resolve_suffix(
    fold: Optional[int] = None,
    run_name_suffix: Optional[str] = None,
) -> Optional[str]:
    """Build the suffix appended to run names for nested execution units.

    Args:
        fold: Fold index for CV-style execution.
        run_name_suffix: Explicit suffix supplied by the caller.

    Returns:
        Optional[str]: The selected suffix or `None` for a plain run name.
    """

    if run_name_suffix:
        return run_name_suffix
    if fold is not None:
        return f"fold{fold}"
    return None


def _flatten_best_metrics(best_metrics: dict[str, Any]) -> dict[str, float]:
    """Flatten structured best-metric payload into MLflow scalar keys."""

    if not best_metrics:
        return {}

    flattened: dict[str, float] = {}
    epoch = best_metrics.get("epoch")
    if epoch is not None:
        flattened["best/epoch"] = float(epoch)
    monitor_value = best_metrics.get("monitor_value")
    if monitor_value is not None:
        flattened["best/monitor_value"] = float(monitor_value)

    for stage in ("val", "test"):
        payload = best_metrics.get(stage)
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            flattened[f"best/{stage}_{key}"] = _to_float(value)
    return flattened


def _log_best_metrics_to_mlflow(logger: Any, best_metrics: dict[str, Any]) -> None:
    """Log final best metrics once to any configured MLflow logger."""

    if not best_metrics or logger in (None, False):
        return

    flattened = _flatten_best_metrics(best_metrics)
    if not flattened:
        return

    loggers = logger if isinstance(logger, list) else [logger]
    for item in loggers:
        if type(item).__name__ != "MLFlowLogger":
            continue
        client = getattr(item, "experiment", None)
        run_id = getattr(item, "run_id", None)
        if client is None or run_id is None:
            continue
        for key, value in flattened.items():
            client.log_metric(run_id, key, value)
        monitor = best_metrics.get("monitor")
        if monitor is not None:
            client.set_tag(run_id, "best/monitor", str(monitor))


def _resolve_experiment_name(cfg: DictConfig) -> str:
    """Resolve the experiment name visible to runtime artifacts.

    Args:
        cfg: Runtime configuration that may define `experiment_name`.

    Returns:
        str: Concrete experiment name used for output layout and logging.
    """

    experiment_name = cfg.get("experiment_name", "default")
    return str(experiment_name)


def _resolve_run_name(cfg: DictConfig, suffix: Optional[str]) -> str:
    """Resolve the run name for one concrete execution unit.

    Args:
        cfg: Runtime configuration that may define a base `run_name`.
        suffix: Optional fold or caller-defined suffix.

    Returns:
        str: Concrete run name used by loggers and output directories.
    """

    base_name = cfg.get("run_name") or cfg.mode.name
    if suffix is None:
        return str(base_name)
    return f"{base_name}_{suffix}"


def _build_run_context(
    cfg: DictConfig,
    fold: Optional[int] = None,
    run_name_suffix: Optional[str] = None,
) -> RunContext:
    """Build the immutable runtime context for one training unit.

    Args:
        cfg: Runtime configuration for the current invocation.
        fold: Optional fold index when the run belongs to CV-like repetition.
        run_name_suffix: Optional explicit suffix for nested orchestration.

    Returns:
        RunContext: Standardized metadata describing the run and its artifact paths.
    """

    suffix = _resolve_suffix(fold=fold, run_name_suffix=run_name_suffix)
    experiment_name = _resolve_experiment_name(cfg)
    run_name = _resolve_run_name(cfg, suffix=suffix)

    parent_output_dir = str(cfg.paths.output_dir)
    output_dir = parent_output_dir
    if suffix is not None:
        output_dir = os.path.join(parent_output_dir, "runs", run_name)

    summary_name = str(cfg.artifacts.get("summary_name", "run_summary.json"))
    artifact_name = str(cfg.artifacts.get("index_name", "artifacts.json"))

    return RunContext(
        mode=str(cfg.mode.name),
        experiment_name=experiment_name,
        run_name=run_name,
        output_dir=output_dir,
        resolved_config_path=os.path.join(output_dir, "config.yaml"),
        summary_path=os.path.join(output_dir, summary_name),
        artifact_index_path=os.path.join(output_dir, artifact_name),
        fold=fold,
        parent_output_dir=parent_output_dir,
    )


def _with_runtime_context(cfg: DictConfig, context: RunContext) -> DictConfig:
    """Inject runtime-only fields into a local config copy.

    Args:
        cfg: Original runtime configuration.
        context: Concrete run context computed for this execution unit.

    Returns:
        DictConfig: Local config with runtime paths and datamodule artifact settings.
    """

    local_cfg = _clone_cfg(cfg)
    with open_dict(local_cfg):
        local_cfg.experiment_name = context.experiment_name
        local_cfg.run_name = context.run_name
        local_cfg.paths.output_dir = context.output_dir
    with open_dict(local_cfg.runtime):
        local_cfg.runtime.mode = context.mode
        local_cfg.runtime.fold = context.fold
        local_cfg.runtime.parent_output_dir = context.parent_output_dir
        local_cfg.runtime.output_dir = context.output_dir
        local_cfg.runtime.run_name = context.run_name
        local_cfg.runtime.experiment_name = context.experiment_name
        local_cfg.runtime.resolved_config_path = context.resolved_config_path
        local_cfg.runtime.summary_path = context.summary_path
        local_cfg.runtime.artifact_index_path = context.artifact_index_path
    with open_dict(local_cfg.datamodule.init_args.data_cfg):
        local_cfg.datamodule.init_args.data_cfg.runtime = {
            "output_dir": context.output_dir,
            "artifacts": OmegaConf.to_container(local_cfg.artifacts, resolve=True),
        }
    return local_cfg


def _resolve_runtime_config(cfg: DictConfig) -> DictConfig:
    """Resolve interpolations after runtime fields have been injected.

    Args:
        cfg: Runtime-injected config that still contains OmegaConf interpolations.

    Returns:
        DictConfig: Fully resolved config snapshot used by the execution path.
    """

    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _collect_logger_artifacts(logger: Any) -> dict[str, Any]:
    """Extract backend-specific identifiers from configured Lightning loggers.

    Args:
        logger: Single logger instance, logger list, or falsy value when logging is disabled.

    Returns:
        dict[str, Any]: Serializable logger metadata for the artifact index.
    """

    if logger in (None, False):
        return {
            "mlflow_run_id": None,
            "wandb_run_id": None,
            "logger_types": [],
        }

    loggers = logger if isinstance(logger, list) else [logger]
    payload = {
        "mlflow_run_id": None,
        "wandb_run_id": None,
        "logger_types": [],
    }

    for item in loggers:
        payload["logger_types"].append(type(item).__name__)
        if hasattr(item, "run_id"):
            payload["mlflow_run_id"] = getattr(item, "run_id")
        experiment = getattr(item, "experiment", None)
        if experiment is not None and hasattr(experiment, "id"):
            payload["wandb_run_id"] = getattr(experiment, "id")
    return payload


def _build_artifact_index(
    context: RunContext,
    summary: RunSummary,
    datamodule: Optional[Any],
    logger: Any,
    checkpoint_callback: Optional[Any],
) -> ArtifactIndex:
    """Build the artifact index written after a concrete run.

    Args:
        context: Immutable run metadata.
        summary: High-frequency summary for the completed run.
        datamodule: Built datamodule, used to query data artifact metadata.
        logger: Logger instance or list of logger instances attached to the trainer.
        checkpoint_callback: Model checkpoint callback from the trainer when available.

    Returns:
        ArtifactIndex: Serializable index of local and remote run artifacts.
    """

    data_artifacts = datamodule.data_artifacts() if datamodule is not None and hasattr(datamodule, "data_artifacts") else {}
    logger_artifacts = _collect_logger_artifacts(logger)
    figure_dir = os.path.join(context.output_dir, "confusion_matrices")
    figures = []
    if os.path.isdir(figure_dir):
        figures = sorted(
            os.path.join(figure_dir, name)
            for name in os.listdir(figure_dir)
            if name.endswith(".png")
        )

    checkpoint_dir = None
    all_ckpts = []
    if checkpoint_callback is not None:
        checkpoint_dir = getattr(checkpoint_callback, "dirpath", None)
        if checkpoint_dir and os.path.isdir(checkpoint_dir):
            all_ckpts = sorted(
                os.path.join(checkpoint_dir, name)
                for name in os.listdir(checkpoint_dir)
                if name.endswith(".ckpt")
            )

    return ArtifactIndex(
        run={
            "mode": context.mode,
            "experiment_name": context.experiment_name,
            "run_name": context.run_name,
            "fold": context.fold,
            "output_dir": context.output_dir,
        },
        config={
            "resolved_config": context.resolved_config_path,
            "hydra_output_dir": context.output_dir,
        },
        checkpoints={
            "best": summary.best_ckpt_path,
            "last": next((path for path in all_ckpts if path.endswith("last.ckpt")), None),
            "all": all_ckpts,
            "dir": checkpoint_dir,
        },
        metrics={
            "monitor": summary.monitor,
            "val_score": summary.val_score,
            "last_val_score": summary.last_val_score,
            "test_score": summary.test_score,
            "summary_path": context.summary_path,
        },
        data=data_artifacts,
        figures={
            "confusion_matrix_dir": figure_dir,
            "files": figures,
        },
        logs=logger_artifacts,
    )


def run_experiment(
    cfg: DictConfig,
    fold: Optional[int] = None,
    ckpt_path: Optional[str] = None,
    run_name_suffix: Optional[str] = None,
    keep_artifacts: bool = False,
) -> RunResult:
    """Execute one concrete training unit and persist its standard artifacts.

    Args:
        cfg: Resolved runtime configuration.
        fold: Optional fold index for CV-style execution.
        ckpt_path: Optional checkpoint path used to resume or test.
        run_name_suffix: Optional suffix appended to the base run name.
        keep_artifacts: Whether to keep in-memory model and datamodule references.

    Returns:
        RunResult: In-memory result object containing summary, context, and artifact index.
    """

    split_method = cfg.get("split", {}).get("method")
    if split_method in KFOLD_METHODS and fold is None:
        fold = 0

    context = _build_run_context(cfg, fold=fold, run_name_suffix=run_name_suffix)
    local_cfg = _with_runtime_context(cfg, context)

    # resolve configurations and save it
    resolved_cfg = _resolve_runtime_config(local_cfg)
    os.makedirs(context.output_dir, exist_ok=True)
    write_resolved_config(resolved_cfg, context.resolved_config_path)

    provider = build_split_provider(resolved_cfg.get("split"), fold=fold)
    model = build_model(resolved_cfg.model)
    datamodule = build_datamodule(
        resolved_cfg.datamodule,
        split_provider=provider,
    )
    logger = build_logger(resolved_cfg.logger)
    callbacks = build_callbacks(resolved_cfg.callbacks)
    trainer = build_trainer(resolved_cfg.trainer, logger, callbacks)

    log_hyperparameters(resolved_cfg, model, trainer)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path if ckpt_path is not None else resolved_cfg.get("resume_ckpt", None),
    )

    monitor = str(resolved_cfg.get("monitor", "val/loss"))
    last_val_score = _to_float(trainer.callback_metrics.get(monitor))

    val_score = last_val_score
    best_ckpt_path = None
    checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
    if checkpoint_callback is not None:
        best_ckpt_path = checkpoint_callback.best_model_path or None
        best_model_score = getattr(checkpoint_callback, "best_model_score", None)
        if best_model_score is not None:
            val_score = _to_float(best_model_score)

    test_score = None
    if resolved_cfg.get("test_after_train", False):
        test_results = trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=best_ckpt_path,
            weights_only=False,
        )
        test_monitor = monitor.replace("val/", "test/")
        if test_results:
            test_score = _to_float(test_results[0].get(test_monitor))

    best_metrics = getattr(model, "best_metrics", {}) or {}
    _log_best_metrics_to_mlflow(logger, best_metrics)

    summary = RunSummary(
        mode=context.mode,
        experiment_name=context.experiment_name,
        run_name=context.run_name,
        output_dir=context.output_dir,
        resolved_config_path=context.resolved_config_path,
        summary_path=context.summary_path,
        artifact_index_path=context.artifact_index_path,
        monitor=monitor,
        val_score=val_score,
        last_val_score=last_val_score,
        test_score=test_score,
        best_ckpt_path=best_ckpt_path,
        best_metrics=best_metrics,
        fold=fold,
    )
    save_json(summary.to_dict(), context.summary_path)

    artifact_index = _build_artifact_index(
        context=context,
        summary=summary,
        datamodule=datamodule,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
    )
    save_json(artifact_index.to_dict(), context.artifact_index_path)

    return RunResult(
        context=context,
        summary=summary,
        artifact_index=artifact_index,
        model=model if keep_artifacts else None,
        datamodule=datamodule if keep_artifacts else None,
    )
