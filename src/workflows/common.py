"""Shared helpers for Python workflow orchestration."""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from src.core.contracts import ArtifactIndex, WorkflowSummary
from src.utils.misc import save_json


def workflow_output_dir(root_dir: str, experiment_name: str, workflow_name: str) -> str:
    """Build a timestamped output directory for one workflow run.

    Args:
        root_dir: Root directory used for workflow outputs.
        experiment_name: Experiment namespace.
        workflow_name: Workflow identifier such as `flat_cv`.

    Returns:
        str: Timestamped workflow output directory.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(root_dir, experiment_name, workflow_name, timestamp)


def run_main(overrides: Iterable[str], cwd: Optional[str] = None) -> None:
    """Launch `main.py` as a subprocess with Hydra overrides.

    Args:
        overrides: Command-line overrides forwarded to Hydra.
        cwd: Optional working directory for the subprocess.

    Returns:
        None: The function runs a subprocess and raises on failure.
    """

    cmd = [sys.executable, "main.py", *list(overrides)]
    subprocess.run(cmd, check=True, cwd=cwd)


def latest_json(pattern: str) -> Dict[str, Any]:
    """Load the newest JSON file that matches a glob pattern.

    Args:
        pattern: Recursive glob pattern.

    Returns:
        Dict[str, Any]: Parsed JSON payload from the latest matching file.

    Raises:
        FileNotFoundError: Raised when the pattern matches no files.
    """

    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(f"No JSON file matched pattern: {pattern}")
    with open(matches[-1], "r", encoding="utf-8") as f:
        return json.load(f)


def latest_path(pattern: str) -> str:
    """Return the newest path that matches a glob pattern.

    Args:
        pattern: Recursive glob pattern.

    Returns:
        str: Latest matching path.

    Raises:
        FileNotFoundError: Raised when the pattern matches no files.
    """

    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(f"No file matched pattern: {pattern}")
    return matches[-1]


def write_workflow_outputs(
    output_dir: str,
    workflow_name: str,
    experiment_name: str,
    primary_metric: str,
    primary_score: Optional[float],
    child_summary_paths: List[str],
    child_artifact_paths: List[str],
    extra: Optional[Dict[str, Any]] = None,
) -> WorkflowSummary:
    """Write workflow summary artifacts for a higher-level orchestration run.

    Args:
        output_dir: Workflow output directory.
        workflow_name: Workflow identifier.
        experiment_name: Experiment namespace.
        primary_metric: Metric name used as the workflow's headline score.
        primary_score: Headline score value.
        child_summary_paths: Run summary paths produced by child runs.
        child_artifact_paths: Artifact index paths produced by child runs.
        extra: Optional extra metadata stored in the workflow summary.

    Returns:
        WorkflowSummary: Serializable summary object written to disk.
    """

    summary_path = os.path.join(output_dir, "workflow_summary.json")
    artifact_index_path = os.path.join(output_dir, "workflow_artifacts.json")
    summary = WorkflowSummary(
        workflow_name=workflow_name,
        experiment_name=experiment_name,
        output_dir=output_dir,
        summary_path=summary_path,
        artifact_index_path=artifact_index_path,
        primary_metric=primary_metric,
        primary_score=primary_score,
        child_summary_paths=child_summary_paths,
        child_artifact_paths=child_artifact_paths,
        extra=extra or {},
    )
    save_json(summary.to_dict(), summary_path)

    artifact_index = ArtifactIndex(
        run={
            "mode": "workflow",
            "experiment_name": experiment_name,
            "run_name": workflow_name,
            "fold": None,
            "output_dir": output_dir,
        },
        config={},
        checkpoints={},
        metrics={
            "monitor": primary_metric,
            "primary_score": primary_score,
            "summary_path": summary_path,
        },
        data={},
        figures={},
        logs={},
        children=child_artifact_paths,
        extra={"child_summary_paths": child_summary_paths},
    )
    save_json(artifact_index.to_dict(), artifact_index_path)
    return summary
