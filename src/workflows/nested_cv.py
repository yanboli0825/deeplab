from __future__ import annotations

import argparse
import os
from statistics import fmean

from src.workflows.common import latest_json, latest_path, run_main, workflow_output_dir, write_workflow_outputs


INNER_TO_REFIT_METHOD = {
    "dev_test_kfold": "train_val_test_holdout",
    "group_dev_test_kfold": "group_train_val_test_holdout",
}


def _parse_candidate(candidate: str) -> list[str]:
    """Split one candidate override string into individual Hydra overrides.

    Args:
        candidate: Space-separated Hydra override string.

    Returns:
        list[str]: Individual override tokens.
    """

    return [item for item in candidate.split(" ") if item]


def main() -> int:
    """Run a provider-driven nested-style CV workflow.

    The workflow performs repeated outer runs. Each outer run uses a distinct
    split seed to derive a fixed test partition and then evaluates candidates by
    running `mode=cv` on the remaining development partition.

    Returns:
        int: Process exit code, `0` on success.
    """

    parser = argparse.ArgumentParser(description="Provider-driven nested CV workflow launcher.")
    parser.add_argument("--experiment-name", default="nested_demo")
    parser.add_argument("--run-prefix", default="nested")
    parser.add_argument("--outer-runs", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument(
        "--split-method",
        choices=sorted(INNER_TO_REFIT_METHOD.keys()),
        default="dev_test_kfold",
        help="Provider used for inner candidate selection.",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-id-column", default=None)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[
            "model.init_args.optimizer.lr=3e-4 model.init_args.optimizer.weight_decay=1e-4",
            "model.init_args.optimizer.lr=1e-4 model.init_args.optimizer.weight_decay=1e-4",
        ],
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    summary_paths = []
    artifact_paths = []
    selections = []
    refit_method = INNER_TO_REFIT_METHOD[args.split_method]

    for outer_idx in range(args.outer_runs):
        outer_name = f"outer_{outer_idx}"
        outer_seed = args.seed + outer_idx
        best_score = None
        best_override = None

        for candidate in args.candidate:
            run_name = f"{args.run_prefix}_{outer_name}_search"
            run_main(
                [
                    "mode=cv",
                    "test_after_train=false",
                    f"mode.n_folds={args.inner_folds}",
                    f"experiment_name={args.experiment_name}",
                    f"run_name={run_name}",
                    f"split.method={args.split_method}",
                    f"split.seed={outer_seed}",
                    *([f"split.group_id_column={args.group_id_column}"] if args.group_id_column else []),
                    f"split.test_ratio={args.test_ratio}",
                    *(_parse_candidate(candidate)),
                    *args.overrides,
                ]
            )
            payload = latest_json(
                os.path.join("outputs", args.experiment_name, run_name, "**", "workflow_summary.json")
            )
            score = float(payload["primary_score"])
            if best_score is None or score < best_score:
                best_score = score
                best_override = candidate

        refit_run_name = f"{args.run_prefix}_{outer_name}_refit"
        run_main(
            [
                "mode=train",
                "test_after_train=true",
                f"experiment_name={args.experiment_name}",
                f"run_name={refit_run_name}",
                f"split.method={refit_method}",
                f"split.seed={outer_seed}",
                f"split.test_ratio={args.test_ratio}",
                *([f"split.group_id_column={args.group_id_column}"] if args.group_id_column else []),
                *(_parse_candidate(best_override or "")),
                *args.overrides,
            ]
        )
        summary_pattern = os.path.join("outputs", args.experiment_name, refit_run_name, "**", "run_summary.json")
        artifact_pattern = os.path.join("outputs", args.experiment_name, refit_run_name, "**", "artifacts.json")
        summary_paths.append(latest_path(summary_pattern))
        artifact_paths.append(latest_path(artifact_pattern))
        selections.append(
            {
                "outer": outer_name,
                "outer_seed": outer_seed,
                "split_method": args.split_method,
                "refit_method": refit_method,
                "best_override": best_override,
                "best_score": best_score,
            }
        )

    output_dir = workflow_output_dir("outputs/workflows", args.experiment_name, "nested_cv")
    os.makedirs(output_dir, exist_ok=True)
    write_workflow_outputs(
        output_dir=output_dir,
        workflow_name="nested_cv",
        experiment_name=args.experiment_name,
        primary_metric="val_score",
        primary_score=fmean(item["best_score"] for item in selections if item["best_score"] is not None) if selections else None,
        child_summary_paths=summary_paths,
        child_artifact_paths=artifact_paths,
        extra={
            "outer_runs": selections,
            "inner_folds": args.inner_folds,
            "test_ratio": args.test_ratio,
            "split_method": args.split_method,
            "refit_method": refit_method,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
