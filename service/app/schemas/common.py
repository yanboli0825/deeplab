"""Common request and response schema components."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ResourceRequest(BaseModel):
    """Requested local resources for a training job."""

    gpu_count: int = 0
    gpu_ids: List[int] = Field(default_factory=list)


class PlatformPayload(BaseModel):
    """Platform-facing fields orthogonal to the training framework."""

    project_name: str = "deeplab-service"
    submitted_by: Optional[str] = None
    priority: str = "normal"
    resource_request: ResourceRequest = Field(default_factory=ResourceRequest)
    tags: Dict[str, str] = Field(default_factory=dict)


class DatasetPayload(BaseModel):
    """Optional manifest information resolved by the service layer."""

    manifest_path: Optional[str] = None
    manifest_content: Optional[str] = None
    manifest_filename: str = "manifest.csv"
