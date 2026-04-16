"""HTTP request schema for job creation."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from service.app.schemas.common import DatasetPayload, PlatformPayload


class JobRequest(BaseModel):
    """Request body accepted by train/cv/hpo endpoints."""

    platform: PlatformPayload = Field(default_factory=PlatformPayload)
    framework: Dict[str, Any]
    dataset: Optional[DatasetPayload] = None
