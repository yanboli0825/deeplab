import os

import numpy as np
from omegaconf import DictConfig

from src.core.contracts import ArtifactIndex, CvSummary
from src.core.runner import run_experiment
from src.utils.misc import save_json


def cv_loop(cfg: DictConfig) -> float:
    """Run repeated training units across folds and persist an aggregate summary.

    Args:
        cfg: Resolved runtime configuration with `mode.n_folds`.

    Returns:
        float: Mean validation score across all valid folds, or `nan` if none exist.
    """

    n_splits = int(cfg.mode.get("n_folds", 5))
    results = []

    for fold in range(n_splits):
        result = run_experiment(cfg, fold=fold)
        results.append(result)

    scores = [item.summary.val_score for item in results]
    valid_scores = [score for score in scores if not np.isnan(score)]
    if not valid_scores:
        return float("nan")

    test_scores = [
        item.summary.test_score
        for item in results
        if item.summary.test_score is not None and not np.isnan(item.summary.test_score)
    ]
    summary_path = os.path.join(str(cfg.paths.output_dir), str(cfg.artifacts.workflow_summary_name))
    artifact_index_path = os.path.join(str(cfg.paths.output_dir), str(cfg.artifacts.workflow_index_name))
    summary = CvSummary(
        mode="cv",
        experiment_name=str(cfg.get("experiment_name", "default")),
        run_name=str(cfg.get("run_name", "default")),
        output_dir=str(cfg.paths.output_dir),
        resolved_config_path=(
            str(getattr(cfg.runtime, "resolved_config_path", ""))
            if cfg.get("runtime")
            else os.path.join(str(cfg.paths.output_dir), "config.yaml")
        ),
        summary_path=summary_path,
        artifact_index_path=artifact_index_path,
        monitor=str(cfg.get("monitor", "val/loss")),
        val_score=float(np.mean(valid_scores)),
        test_score=float(np.mean(test_scores)) if test_scores else None,
        n_folds=n_splits,
        fold_summary_paths=[item.context.summary_path for item in results],
        fold_artifact_paths=[item.context.artifact_index_path for item in results],
        fold_scores=scores,
        fold_test_scores=[item.summary.test_score for item in results],
    )
    save_json(summary.to_dict(), summary.summary_path)
    artifact_index = ArtifactIndex(
        run={
            "mode": "cv",
            "experiment_name": summary.experiment_name,
            "run_name": summary.run_name,
            "fold": None,
            "output_dir": summary.output_dir,
        },
        config={
            "resolved_config": summary.resolved_config_path,
            "hydra_output_dir": summary.output_dir,
        },
        checkpoints={},
        metrics={
            "monitor": summary.monitor,
            "val_score": summary.val_score,
            "test_score": summary.test_score,
            "summary_path": summary.summary_path,
        },
        data={},
        figures={},
        logs={},
        children=summary.fold_artifact_paths,
    )
    save_json(artifact_index.to_dict(), summary.artifact_index_path)
    return summary.val_score
