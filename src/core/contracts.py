"""Runtime contracts shared by train and cv execution paths."""

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
    fold: Optional[int] = None
    parent_output_dir: Optional[str] = None


@dataclass
class RunSummary:
    """Stable artifact contract consumed by scripts and aggregators."""

    mode: str
    experiment_name: str
    run_name: str
    output_dir: str
    resolved_config_path: str
    summary_path: str
    monitor: str
    val_score: float
    test_score: Optional[float]
    best_ckpt_path: Optional[str]
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
    monitor: str
    val_score: float
    test_score: Optional[float]
    n_folds: int
    fold_summary_paths: List[str]
    fold_scores: List[float]
    fold_test_scores: List[Optional[float]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
