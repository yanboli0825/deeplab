"""HTTP response schema for job APIs."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class JobCreateResponse(BaseModel):
    """Response returned immediately after job submission."""

    job_id: str
    job_type: str
    status: str
    submitted_at: str


class JobDetailResponse(BaseModel):
    """Current state and metadata for one job."""

    job_id: str
    job_type: str
    status: str
    request_payload: Dict[str, Any]
    created_at: str
    updated_at: str
    normalized_config_path: Optional[str] = None
    log_path: Optional[str] = None
    output_dir: Optional[str] = None
    summary_path: Optional[str] = None
    artifact_index_path: Optional[str] = None
    error_message: Optional[str] = None
    runtime: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
