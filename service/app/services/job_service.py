"""High-level job orchestration service."""

from __future__ import annotations

from typing import Any, Dict

from service.app.services.config_service import ConfigService
from service.app.services.execution_service import ExecutionService
from service.app.storage.file_job_store import FileJobStore


class JobService:
    """Coordinate request persistence, config generation, and execution."""

    def __init__(
        self,
        *,
        job_store: FileJobStore,
        config_service: ConfigService,
        execution_service: ExecutionService,
    ) -> None:
        self._job_store = job_store
        self._config_service = config_service
        self._execution_service = execution_service

    def submit_job(self, *, job_type: str, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        record = self._job_store.create_job(job_type=job_type, request_payload=request_payload)
        normalized_cfg, overrides, config_path = self._config_service.prepare_job_config(
            job_id=record["job_id"],
            job_type=job_type,
            request_payload=request_payload,
        )
        record = self._job_store.update_job(
            record["job_id"],
            {
                "normalized_config_path": config_path,
                "runtime": {"normalized_config": normalized_cfg},
            },
        )
        self._execution_service.submit(
            job_id=record["job_id"],
            job_type=job_type,
            overrides=overrides,
            platform_cfg=request_payload.get("platform", {}),
        )
        return record

    def get_job(self, job_id: str) -> Dict[str, Any]:
        return self._job_store.get_job(job_id)

    def get_summary(self, job_id: str) -> Dict[str, Any]:
        record = self._job_store.get_job(job_id)
        return record.get("result") or {}

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        self._execution_service.cancel(job_id)
        return self._job_store.get_job(job_id)
