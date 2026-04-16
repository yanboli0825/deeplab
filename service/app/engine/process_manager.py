"""Local background executor for training jobs."""

from __future__ import annotations

import os
import queue
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from service.app.domain.status import JobStatus
from service.app.engine.result_parser import ResultParser
from service.app.storage.file_job_store import FileJobStore
from service.app.settings import ServiceSettings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobExecutionRequest:
    """Execution payload consumed by the background runner."""

    job_id: str
    job_type: str
    command: list[str]
    env_overrides: Dict[str, str]


class LocalProcessManager:
    """Simple file-backed local executor with cooperative cancellation."""

    def __init__(
        self,
        *,
        settings: ServiceSettings,
        job_store: FileJobStore,
        result_parser: ResultParser,
    ) -> None:
        self._settings = settings
        self._job_store = job_store
        self._result_parser = result_parser
        self._queue: "queue.Queue[JobExecutionRequest]" = queue.Queue()
        self._running: Dict[str, subprocess.Popen[str]] = {}
        self._running_lock = threading.Lock()
        self._semaphore = threading.Semaphore(settings.max_concurrent_jobs)
        self._dispatcher = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._dispatcher.start()

    def submit(self, request: JobExecutionRequest) -> None:
        """Queue a job for execution."""

        self._queue.put(request)

    def cancel(self, job_id: str) -> None:
        """Cancel a queued or running job."""

        record = self._job_store.get_job(job_id)
        if record["status"] in {JobStatus.SUCCEEDED.value, JobStatus.FAILED.value, JobStatus.CANCELED.value}:
            return
        self._job_store.update_job(job_id, {"status": JobStatus.CANCEL_REQUESTED.value})
        with self._running_lock:
            process = self._running.get(job_id)
        if process is not None:
            process.terminate()

    def _dispatch_loop(self) -> None:
        while True:
            request = self._queue.get()
            self._semaphore.acquire()
            worker = threading.Thread(target=self._run_job, args=(request,), daemon=True)
            worker.start()

    def _run_job(self, request: JobExecutionRequest) -> None:
        try:
            record = self._job_store.get_job(request.job_id)
            if record["status"] == JobStatus.CANCEL_REQUESTED.value:
                self._job_store.update_job(request.job_id, {"status": JobStatus.CANCELED.value})
                return

            log_path = Path(record["log_path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(request.env_overrides)
            with log_path.open("w", encoding="utf-8") as log_handle:
                process = subprocess.Popen(
                    request.command,
                    cwd=str(self._settings.repo_root),
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                with self._running_lock:
                    self._running[request.job_id] = process
                self._job_store.update_job(
                    request.job_id,
                    {
                        "status": JobStatus.RUNNING.value,
                        "runtime": {
                            "pid": process.pid,
                            "started_at": _utc_now(),
                            "command": request.command,
                        },
                    },
                )
                timeout_seconds = self._settings.default_job_timeout_hours * 3600
                return_code = process.wait(timeout=timeout_seconds)

            final_status = JobStatus.SUCCEEDED.value if return_code == 0 else JobStatus.FAILED.value
            updated = self._job_store.get_job(request.job_id)
            if updated["status"] == JobStatus.CANCEL_REQUESTED.value:
                final_status = JobStatus.CANCELED.value
            payload: Dict[str, Any] = {
                "status": final_status,
                "runtime": {
                    **updated.get("runtime", {}),
                    "finished_at": _utc_now(),
                    "return_code": return_code,
                },
            }
            if final_status == JobStatus.SUCCEEDED.value:
                result = self._result_parser.parse(
                    job_type=request.job_type,
                    job_root=self._settings.jobs_dir / request.job_id,
                )
                payload.update(
                    {
                        "result": result,
                        "output_dir": result.get("output_dir"),
                        "summary_path": result.get("summary_path"),
                        "artifact_index_path": result.get("artifact_index_path"),
                    }
                )
            elif final_status == JobStatus.FAILED.value:
                payload["error_message"] = f"Training process exited with code {return_code}"
            self._job_store.update_job(request.job_id, payload)
        except subprocess.TimeoutExpired:
            self._job_store.update_job(
                request.job_id,
                {
                    "status": JobStatus.FAILED.value,
                    "error_message": "Training process timed out",
                },
            )
        except Exception as exc:  # pragma: no cover - final guardrail
            self._job_store.update_job(
                request.job_id,
                {
                    "status": JobStatus.FAILED.value,
                    "error_message": str(exc),
                },
            )
        finally:
            with self._running_lock:
                self._running.pop(request.job_id, None)
            self._semaphore.release()
