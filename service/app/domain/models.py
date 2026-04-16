"""Shared service-layer data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from service.app.domain.status import JobStatus


@dataclass
class JobRecord:
    """Persisted metadata for one submitted job."""

    job_id: str
    job_type: str
    status: JobStatus
    request_payload: Dict[str, Any]
    created_at: str
    updated_at: str
    normalized_config_path: Optional[str] = None
    log_path: Optional[str] = None
    output_dir: Optional[str] = None
    summary_path: Optional[str] = None
    artifact_index_path: Optional[str] = None
    error_message: Optional[str] = None
    runtime: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
