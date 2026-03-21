from __future__ import annotations

import argparse
import os
from statistics import fmean

from src.workflows.common import latest_json, latest_path, run_main, workflow_output_dir, write_workflow_outputs


def main() -> int:
    """Run simple CV-based candidate search followed by one refit run.

    Returns:
        int: Process exit code, `0` on success.
    """

    parser = argparse.ArgumentParser(description="Simple HPO + refit workflow launcher.")
    parser.add_argument("--experiment-name", default="hpo_demo")
    parser.add_argument("--search-prefix", default="trial")
    parser.add_argument("--refit-run-name", default="refit")
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

    scored_candidates = []
    child_summary_paths = []
    child_artifact_paths = []

    for index, candidate in enumerate(args.candidate):
        run_name = f"{args.search_prefix}_{index:02d}"
        run_main(
            [
                "mode=cv",
                "test_after_train=false",
                f"experiment_name={args.experiment_name}",
                f"run_name={run_name}",
                *[item for item in candidate.split(" ") if item],
                *args.overrides,
            ]
        )
        summary_pattern = os.path.join("outputs", args.experiment_name, run_name, "**", "workflow_summary.json")
        artifact_pattern = os.path.join("outputs", args.experiment_name, run_name, "**", "workflow_artifacts.json")
        payload = latest_json(summary_pattern)
        child_summary_paths.append(latest_path(summary_pattern))
        child_artifact_paths.append(latest_path(artifact_pattern))
        scored_candidates.append({"override": candidate, "score": float(payload["val_score"])})

    best = min(scored_candidates, key=lambda item: item["score"])
    run_main(
        [
            "mode=train",
            "test_after_train=true",
            f"experiment_name={args.experiment_name}",
            f"run_name={args.refit_run_name}",
            *[item for item in best["override"].split(" ") if item],
            *args.overrides,
        ]
    )

    refit_summary_pattern = os.path.join("outputs", args.experiment_name, args.refit_run_name, "**", "run_summary.json")
    refit_artifact_pattern = os.path.join("outputs", args.experiment_name, args.refit_run_name, "**", "artifacts.json")
    child_summary_paths.append(latest_path(refit_summary_pattern))
    child_artifact_paths.append(latest_path(refit_artifact_pattern))

    output_dir = workflow_output_dir("outputs/workflows", args.experiment_name, "hpo_refit")
    os.makedirs(output_dir, exist_ok=True)
    write_workflow_outputs(
        output_dir=output_dir,
        workflow_name="hpo_refit",
        experiment_name=args.experiment_name,
        primary_metric="val_score",
        primary_score=fmean(item["score"] for item in scored_candidates) if scored_candidates else None,
        child_summary_paths=child_summary_paths,
        child_artifact_paths=child_artifact_paths,
        extra={"best_candidate": best, "candidates": scored_candidates},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
