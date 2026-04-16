"""Persistence interfaces for service metadata."""

from __future__ import annotations

from typing import Any, Dict, Protocol


class JobStore(Protocol):
    """Metadata persistence contract."""

    def create_job(self, job_type: str, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create and persist a new job record."""

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Load a persisted job record."""

    def update_job(self, job_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Patch a persisted job record."""
