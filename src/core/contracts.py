"""Runtime and workflow contracts shared across execution layers."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RunContext:
    """Immutable metadata for one concrete execution unit."""

    mode: str
    experiment_name: str
    run_name: str
    output_dir: str
    resolved_config_path: str
    summary_path: str
    artifact_index_path: str
    fold: Optional[int] = None
    parent_output_dir: Optional[str] = None


@dataclass
class ArtifactIndex:
    """Stable index of files and remote handles produced by one run."""

    run: Dict[str, Any]
    config: Dict[str, Any]
    checkpoints: Dict[str, Any]
    metrics: Dict[str, Any]
    data: Dict[str, Any]
    figures: Dict[str, Any]
    logs: Dict[str, Any]
    children: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        extra = payload.pop("extra", {})
        payload.update(extra)
        return payload


@dataclass
class RunSummary:
    """High-frequency summary consumed by scripts and workflow aggregators.

    `val_score` is the best score selected by the checkpoint monitor, not the
    final validation metric from the last epoch. The last observed validation
    value is preserved in `last_val_score` for debugging and diagnostics.
    """

    mode: str
    experiment_name: str
    run_name: str
    output_dir: str
    resolved_config_path: str
    summary_path: str
    artifact_index_path: str
    monitor: str
    val_score: float
    last_val_score: Optional[float]
    test_score: Optional[float]
    best_ckpt_path: Optional[str]
    best_metrics: Dict[str, Any] = field(default_factory=dict)
    fold: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        extra = payload.pop("extra", {})
        payload.update(extra)
        return payload


@dataclass
class CvSummary:
    """Aggregate contract for cv mode."""

    mode: str
    experiment_name: str
    run_name: str
    output_dir: str
    resolved_config_path: str
    summary_path: str
    artifact_index_path: str
    monitor: str
    val_score: float
    test_score: Optional[float]
    n_folds: int
    fold_summary_paths: List[str]
    fold_artifact_paths: List[str]
    fold_scores: List[float]
    fold_test_scores: List[Optional[float]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowSummary:
    """Aggregate contract for Python workflow orchestration."""

    workflow_name: str
    experiment_name: str
    output_dir: str
    summary_path: str
    artifact_index_path: str
    primary_metric: str
    primary_score: Optional[float]
    child_summary_paths: List[str]
    child_artifact_paths: List[str]
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        extra = payload.pop("extra", {})
        payload.update(extra)
        return payload
