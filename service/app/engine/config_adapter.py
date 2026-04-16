"""Adapt service-layer job payloads into framework execution inputs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from service.app.domain.exceptions import InvalidJobRequestError
from service.app.settings import ServiceSettings


class ConfigAdapter:
    """Translate HTTP job payloads into framework config and CLI overrides."""

    def __init__(self, settings: ServiceSettings) -> None:
        self._settings = settings

    def prepare(
        self,
        *,
        job_id: str,
        job_type: str,
        request_payload: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str], str]:
        """Normalize request payload, build CLI overrides, and persist config snapshot."""

        platform_cfg = copy.deepcopy(request_payload.get("platform", {}))
        framework_cfg = copy.deepcopy(request_payload.get("framework", {}))
        dataset_cfg = copy.deepcopy(request_payload.get("dataset", {})) or None

        if not isinstance(framework_cfg, dict):
            raise InvalidJobRequestError("framework payload must be a mapping")

        manifest_path = self._resolve_manifest(job_id=job_id, dataset_cfg=dataset_cfg)
        normalized_cfg = self._normalize_framework_config(
            job_id=job_id,
            job_type=job_type,
            platform_cfg=platform_cfg,
            framework_cfg=framework_cfg,
            manifest_path=manifest_path,
        )
        config_path = self._settings.generated_configs_dir / f"{job_id}.json"
        config_path.write_text(json.dumps(normalized_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        return normalized_cfg, self._build_overrides(normalized_cfg), str(config_path)

    def _normalize_framework_config(
        self,
        *,
        job_id: str,
        job_type: str,
        platform_cfg: Dict[str, Any],
        framework_cfg: Dict[str, Any],
        manifest_path: Optional[str],
    ) -> Dict[str, Any]:
        project_name = str(framework_cfg.get("project_name") or platform_cfg.get("project_name") or "deeplab")
        framework_cfg["project_name"] = project_name
        framework_cfg["experiment_name"] = str(framework_cfg.get("experiment_name") or project_name)
        framework_cfg["run_name"] = str(framework_cfg.get("run_name") or job_id)
        framework_cfg.setdefault("seed", 42)
        framework_cfg.setdefault("monitor", "val/loss")

        if job_type == "train":
            framework_cfg["mode"] = {"name": "train"}
        elif job_type == "cv":
            framework_cfg.setdefault("mode", {})
            framework_cfg["mode"]["name"] = "cv"
        elif job_type == "hpo":
            framework_cfg["mode"] = {"name": "cv", **framework_cfg.get("mode", {})}
            framework_cfg["mode"]["name"] = "cv"
            framework_cfg["test_after_train"] = False
        else:
            raise InvalidJobRequestError(f"Unsupported job type '{job_type}'")

        if manifest_path is not None:
            datamodule = framework_cfg.setdefault("datamodule", {})
            dm_args = datamodule.setdefault("init_args", {})
            data_cfg = dm_args.setdefault("data_cfg", {})
            data_cfg["data_file"] = manifest_path
            split_cfg = framework_cfg.setdefault("split", {})
            split_cfg["data_file"] = manifest_path

        job_root = self._settings.jobs_dir / job_id
        framework_cfg.setdefault("hydra", {})
        framework_cfg["hydra"]["run"] = {"dir": str(job_root / "run")}
        framework_cfg["hydra"]["sweep"] = {"dir": str(job_root / "sweep")}
        return framework_cfg

    def _resolve_manifest(self, *, job_id: str, dataset_cfg: Optional[Dict[str, Any]]) -> Optional[str]:
        if not dataset_cfg:
            return None

        manifest_path = dataset_cfg.get("manifest_path")
        manifest_content = dataset_cfg.get("manifest_content")
        manifest_filename = str(dataset_cfg.get("manifest_filename") or "manifest.csv")

        if manifest_path and manifest_content:
            raise InvalidJobRequestError("Provide either dataset.manifest_path or dataset.manifest_content, not both")
        if manifest_path:
            path = Path(str(manifest_path)).expanduser().resolve()
            if not path.exists():
                raise InvalidJobRequestError(f"Manifest path does not exist: {path}")
            return str(path)
        if manifest_content:
            inputs_dir = self._settings.jobs_dir / job_id / "inputs"
            inputs_dir.mkdir(parents=True, exist_ok=True)
            path = inputs_dir / manifest_filename
            path.write_text(str(manifest_content), encoding="utf-8")
            return str(path)
        return None

    def _build_overrides(self, normalized_cfg: Dict[str, Any]) -> List[str]:
        overrides: List[str] = []
        for key, value in normalized_cfg.items():
            if key == "hpo":
                continue
            overrides.extend(self._flatten_override(key, value))
        return overrides

    def _flatten_override(self, prefix: str, value: Any) -> List[str]:
        if isinstance(value, dict):
            overrides: List[str] = []
            for key, nested in value.items():
                nested_prefix = f"{prefix}.{key}" if prefix else str(key)
                overrides.extend(self._flatten_override(nested_prefix, nested))
            return overrides
        return [f"{prefix}={self._encode_value(value)}"]

    @staticmethod
    def _encode_value(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        return json.dumps(value, ensure_ascii=False)
