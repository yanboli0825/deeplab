"""File-backed job metadata store."""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from service.app.domain.exceptions import JobNotFoundError
from service.app.domain.status import JobStatus
from service.app.settings import ServiceSettings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class FileJobStore:
    """Persist job metadata as JSON under `service/runtime`."""

    def __init__(self, settings: ServiceSettings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._index_path = settings.index_dir / "jobs.json"
        for path in (
            settings.runtime_dir,
            settings.jobs_dir,
            settings.logs_dir,
            settings.generated_configs_dir,
            settings.index_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        if not self._index_path.exists():
            self._write_json(self._index_path, {"jobs": []})

    def create_job(self, job_type: str, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create the on-disk layout for a new job."""

        with self._lock:
            job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
            created_at = _utc_now()
            job_dir = self._job_dir(job_id)
            job_dir.mkdir(parents=True, exist_ok=True)
            record: Dict[str, Any] = {
                "job_id": job_id,
                "job_type": job_type,
                "status": JobStatus.QUEUED.value,
                "request_payload": deepcopy(request_payload),
                "created_at": created_at,
                "updated_at": created_at,
                "normalized_config_path": None,
                "log_path": str(self._settings.logs_dir / f"{job_id}.log"),
                "output_dir": None,
                "summary_path": None,
                "artifact_index_path": None,
                "error_message": None,
                "runtime": {},
                "result": None,
            }
            self._write_json(job_dir / "request.json", request_payload)
            self._write_json(job_dir / "metadata.json", record)
            self._append_index_entry(record)
            return record

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Return the latest metadata for a job."""

        metadata_path = self._job_dir(job_id) / "metadata.json"
        if not metadata_path.exists():
            raise JobNotFoundError(f"Job '{job_id}' does not exist")
        return self._read_json(metadata_path)

    def update_job(self, job_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Patch and persist a job record."""

        with self._lock:
            record = self.get_job(job_id)
            record.update(deepcopy(updates))
            record["updated_at"] = _utc_now()
            metadata_path = self._job_dir(job_id) / "metadata.json"
            self._write_json(metadata_path, record)
            self._rewrite_index_entry(record)
            return record

    def _job_dir(self, job_id: str) -> Path:
        return self._settings.jobs_dir / job_id

    def _append_index_entry(self, record: Dict[str, Any]) -> None:
        index = self._read_json(self._index_path)
        index.setdefault("jobs", []).append(
            {
                "job_id": record["job_id"],
                "job_type": record["job_type"],
                "status": record["status"],
                "created_at": record["created_at"],
                "updated_at": record["updated_at"],
            }
        )
        self._write_json(self._index_path, index)

    def _rewrite_index_entry(self, record: Dict[str, Any]) -> None:
        index = self._read_json(self._index_path)
        jobs = []
        for item in index.get("jobs", []):
            if item["job_id"] == record["job_id"]:
                jobs.append(
                    {
                        "job_id": record["job_id"],
                        "job_type": record["job_type"],
                        "status": record["status"],
                        "created_at": record["created_at"],
                        "updated_at": record["updated_at"],
                    }
                )
            else:
                jobs.append(item)
        index["jobs"] = jobs
        self._write_json(self._index_path, index)

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
