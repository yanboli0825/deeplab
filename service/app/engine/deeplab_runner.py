"""Build subprocess commands for the training engine."""

from __future__ import annotations

from typing import Dict, List, Tuple

from service.app.settings import ServiceSettings


class DeeplabRunner:
    """Translate job metadata into a subprocess command."""

    def __init__(self, settings: ServiceSettings) -> None:
        self._settings = settings

    def build_command(
        self,
        *,
        job_type: str,
        overrides: List[str],
    ) -> List[str]:
        """Return the exact command line used to invoke the framework."""

        command = [self._settings.python_executable, str(self._settings.main_script)]
        if job_type == "hpo":
            command.append("-m")
        command.extend(overrides)
        if job_type == "hpo":
            command.append("hpo=optuna")
        return command

    def build_environment(self, *, resource_request: Dict[str, object]) -> Dict[str, str]:
        """Return process environment customizations derived from platform metadata."""

        env: Dict[str, str] = {}
        gpu_ids = resource_request.get("gpu_ids", [])
        if isinstance(gpu_ids, list) and gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in gpu_ids)
        return env
