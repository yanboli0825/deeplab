from __future__ import annotations

import argparse
import os
from statistics import fmean

from src.workflows.common import latest_json, latest_path, run_main, workflow_output_dir, write_workflow_outputs


def main() -> int:
    """Run flat cross-validation through repeated `train` executions.

    Returns:
        int: Process exit code, `0` on success.
    """

    parser = argparse.ArgumentParser(description="Flat CV workflow launcher.")
    parser.add_argument("--experiment-name", default="flat_cv_demo")
    parser.add_argument("--run-prefix", default="flatcv")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fixed-test-ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    summary_paths = []
    artifact_paths = []
    scores = []
    test_scores = []

    for fold in range(args.n_folds):
        run_name = f"{args.run_prefix}_fold{fold}"
        run_main(
            [
                "mode=train",
                f"experiment_name={args.experiment_name}",
                f"run_name={run_name}",
                "test_after_train=true",
                f"datamodule.init_args.data_cfg.test_ratio={args.fixed_test_ratio}",
                f"model.init_args.optimizer.lr={args.lr}",
                f"model.init_args.optimizer.weight_decay={args.weight_decay}",
                *args.overrides,
            ]
        )
        summary_pattern = os.path.join("outputs", args.experiment_name, run_name, "**", "run_summary.json")
        artifact_pattern = os.path.join("outputs", args.experiment_name, run_name, "**", "artifacts.json")
        payload = latest_json(summary_pattern)
        summary_paths.append(latest_path(summary_pattern))
        artifact_paths.append(latest_path(artifact_pattern))
        scores.append(float(payload["val_score"]))
        if payload.get("test_score") is not None:
            test_scores.append(float(payload["test_score"]))

    output_dir = workflow_output_dir("outputs/workflows", args.experiment_name, "flat_cv")
    os.makedirs(output_dir, exist_ok=True)
    write_workflow_outputs(
        output_dir=output_dir,
        workflow_name="flat_cv",
        experiment_name=args.experiment_name,
        primary_metric="val_score",
        primary_score=fmean(scores) if scores else None,
        child_summary_paths=summary_paths,
        child_artifact_paths=artifact_paths,
        extra={
            "n_folds": args.n_folds,
            "val_score_mean": fmean(scores) if scores else None,
            "test_score_mean": fmean(test_scores) if test_scores else None,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
