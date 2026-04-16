"""Runtime settings for the HTTP training service."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class ServiceSettings:
    """Filesystem and execution settings for the service layer."""

    repo_root: Path
    runtime_dir: Path
    jobs_dir: Path
    logs_dir: Path
    generated_configs_dir: Path
    index_dir: Path
    python_executable: str
    main_script: Path
    max_concurrent_jobs: int
    default_job_timeout_hours: int


@lru_cache(maxsize=1)
def get_settings() -> ServiceSettings:
    """Resolve service settings once per process."""

    repo_root = Path(__file__).resolve().parents[2]
    runtime_dir = repo_root / "service" / "runtime"
    return ServiceSettings(
        repo_root=repo_root,
        runtime_dir=runtime_dir,
        jobs_dir=runtime_dir / "jobs",
        logs_dir=runtime_dir / "logs",
        generated_configs_dir=runtime_dir / "generated_configs",
        index_dir=runtime_dir / "index",
        python_executable=os.getenv("DEEPLAB_PYTHON", sys.executable),
        main_script=repo_root / "main.py",
        max_concurrent_jobs=int(os.getenv("DEEPLAB_MAX_CONCURRENT_JOBS", "1")),
        default_job_timeout_hours=int(os.getenv("DEEPLAB_JOB_TIMEOUT_HOURS", "48")),
    )
