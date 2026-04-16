"""Build normalized framework config from HTTP payloads."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from service.app.engine.config_adapter import ConfigAdapter


class ConfigService:
    """Thin wrapper around the engine-facing config adapter."""

    def __init__(self, adapter: ConfigAdapter) -> None:
        self._adapter = adapter

    def prepare_job_config(
        self,
        *,
        job_id: str,
        job_type: str,
        request_payload: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str], str]:
        return self._adapter.prepare(job_id=job_id, job_type=job_type, request_payload=request_payload)
