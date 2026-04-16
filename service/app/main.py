"""FastAPI entrypoint for the deeplab training service."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from fastapi import FastAPI

from service.app.api.routes_health import router as health_router
from service.app.api.routes_jobs import router as jobs_router
from service.app.engine.config_adapter import ConfigAdapter
from service.app.engine.deeplab_runner import DeeplabRunner
from service.app.engine.process_manager import LocalProcessManager
from service.app.engine.result_parser import ResultParser
from service.app.services.config_service import ConfigService
from service.app.services.execution_service import ExecutionService
from service.app.services.job_service import JobService
from service.app.services.log_service import LogService
from service.app.settings import get_settings
from service.app.storage.file_job_store import FileJobStore


@dataclass(frozen=True)
class ServiceContainer:
    """Singleton service graph used by API dependencies."""

    job_store: FileJobStore
    job_service: JobService
    log_service: LogService


@lru_cache(maxsize=1)
def get_container() -> ServiceContainer:
    settings = get_settings()
    job_store = FileJobStore(settings)
    result_parser = ResultParser()
    runner = DeeplabRunner(settings)
    process_manager = LocalProcessManager(
        settings=settings,
        job_store=job_store,
        result_parser=result_parser,
    )
    config_service = ConfigService(ConfigAdapter(settings))
    execution_service = ExecutionService(runner, process_manager)
    job_service = JobService(
        job_store=job_store,
        config_service=config_service,
        execution_service=execution_service,
    )
    log_service = LogService(job_store)
    return ServiceContainer(
        job_store=job_store,
        job_service=job_service,
        log_service=log_service,
    )


def create_app() -> FastAPI:
    """Build the FastAPI application."""

    app = FastAPI(title="deeplab-training-service", version="0.1.0")
    app.include_router(health_router)
    app.include_router(jobs_router)
    return app


app = create_app()
