"""Read job logs from the local runtime directory."""

from __future__ import annotations

from pathlib import Path

from service.app.storage.file_job_store import FileJobStore


class LogService:
    """Expose persisted job logs."""

    def __init__(self, job_store: FileJobStore) -> None:
        self._job_store = job_store

    def read_job_log(self, job_id: str) -> str:
        record = self._job_store.get_job(job_id)
        log_path = record.get("log_path")
        if log_path is None:
            return ""
        path = Path(log_path)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")
