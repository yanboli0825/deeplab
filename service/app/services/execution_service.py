"""Submit normalized jobs to the local execution backend."""

from __future__ import annotations

from typing import Dict, List

from service.app.engine.deeplab_runner import DeeplabRunner
from service.app.engine.process_manager import JobExecutionRequest, LocalProcessManager


class ExecutionService:
    """Prepare command/env and dispatch to the background executor."""

    def __init__(self, runner: DeeplabRunner, process_manager: LocalProcessManager) -> None:
        self._runner = runner
        self._process_manager = process_manager

    def submit(
        self,
        *,
        job_id: str,
        job_type: str,
        overrides: List[str],
        platform_cfg: Dict[str, object],
    ) -> None:
        resource_request = platform_cfg.get("resource_request", {})
        command = self._runner.build_command(job_type=job_type, overrides=overrides)
        env_overrides = self._runner.build_environment(resource_request=resource_request)
        self._process_manager.submit(
            JobExecutionRequest(
                job_id=job_id,
                job_type=job_type,
                command=command,
                env_overrides=env_overrides,
            )
        )

    def cancel(self, job_id: str) -> None:
        self._process_manager.cancel(job_id)
