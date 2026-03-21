from __future__ import annotations

import argparse
import os
from statistics import fmean

from src.workflows.common import latest_json, latest_path, run_main, workflow_output_dir, write_workflow_outputs


def _parse_candidate(candidate: str) -> list[str]:
    """Split one candidate override string into individual Hydra overrides.

    Args:
        candidate: Space-separated Hydra override string.

    Returns:
        list[str]: Individual override tokens.
    """

    return [item for item in candidate.split(" ") if item]


def main() -> int:
    """Run nested cross-validation with candidate selection and refit.

    Returns:
        int: Process exit code, `0` on success.
    """

    parser = argparse.ArgumentParser(description="Nested CV workflow launcher.")
    parser.add_argument("--experiment-name", default="nested_demo")
    parser.add_argument("--run-prefix", default="nested")
    parser.add_argument(
        "--split-root",
        default="splits/nested",
        help="Directory containing outer_x/inner_y.yaml and refit.yaml manifests.",
    )
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

    for outer_name in sorted(name for name in os.listdir(args.split_root) if name.startswith("outer_")):
        outer_dir = os.path.join(args.split_root, outer_name)
        best_score = None
        best_override = None

        for candidate in args.candidate:
            inner_scores = []
            for split_name in sorted(name for name in os.listdir(outer_dir) if name.startswith("inner_") and name.endswith(".yaml")):
                split_path = os.path.join(outer_dir, split_name)
                run_name = f"{args.run_prefix}_{outer_name}_{split_name[:-5]}"
                run_main(
                    [
                        "mode=train",
                        f"experiment_name={args.experiment_name}",
                        f"run_name={run_name}",
                        "test_after_train=false",
                        f"datamodule.init_args.data_cfg.split_file={split_path}",
                        *_parse_candidate(candidate),
                        *args.overrides,
                    ]
                )
                payload = latest_json(os.path.join("outputs", args.experiment_name, run_name, "**", "run_summary.json"))
                inner_scores.append(float(payload["val_score"]))

            score = fmean(inner_scores) if inner_scores else None
            if best_score is None or (score is not None and score < best_score):
                best_score = score
                best_override = candidate

        refit_run_name = f"{args.run_prefix}_{outer_name}_refit"
        refit_split = os.path.join(outer_dir, "refit.yaml")
        run_main(
            [
                "mode=train",
                f"experiment_name={args.experiment_name}",
                f"run_name={refit_run_name}",
                "test_after_train=true",
                f"datamodule.init_args.data_cfg.split_file={refit_split}",
                *(_parse_candidate(best_override or "")),
                *args.overrides,
            ]
        )
        summary_pattern = os.path.join("outputs", args.experiment_name, refit_run_name, "**", "run_summary.json")
        artifact_pattern = os.path.join("outputs", args.experiment_name, refit_run_name, "**", "artifacts.json")
        summary_paths.append(latest_path(summary_pattern))
        artifact_paths.append(latest_path(artifact_pattern))
        selections.append({"outer": outer_name, "best_override": best_override, "best_score": best_score})

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
        extra={"outer_folds": selections},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
