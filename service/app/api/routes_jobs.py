"""Job management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response

from service.app.domain.exceptions import JobNotFoundError, ServiceError
from service.app.schemas.job_requests import JobRequest
from service.app.schemas.job_responses import JobCreateResponse, JobDetailResponse
from service.app.services.job_service import JobService
from service.app.services.log_service import LogService

router = APIRouter(prefix="/jobs", tags=["jobs"])


def get_job_service() -> JobService:
    from service.app.main import get_container

    return get_container().job_service


def get_log_service() -> LogService:
    from service.app.main import get_container

    return get_container().log_service


@router.post("/train", response_model=JobCreateResponse)
def submit_train_job(
    request: JobRequest,
    job_service: JobService = Depends(get_job_service),
) -> JobCreateResponse:
    return _submit(job_type="train", request=request, job_service=job_service)


@router.post("/cv", response_model=JobCreateResponse)
def submit_cv_job(
    request: JobRequest,
    job_service: JobService = Depends(get_job_service),
) -> JobCreateResponse:
    return _submit(job_type="cv", request=request, job_service=job_service)


@router.post("/hpo", response_model=JobCreateResponse)
def submit_hpo_job(
    request: JobRequest,
    job_service: JobService = Depends(get_job_service),
) -> JobCreateResponse:
    return _submit(job_type="hpo", request=request, job_service=job_service)


@router.get("/{job_id}", response_model=JobDetailResponse)
def get_job(
    job_id: str,
    job_service: JobService = Depends(get_job_service),
) -> JobDetailResponse:
    try:
        return JobDetailResponse(**job_service.get_job(job_id))
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{job_id}/summary")
def get_job_summary(
    job_id: str,
    job_service: JobService = Depends(get_job_service),
) -> dict:
    try:
        return job_service.get_summary(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{job_id}/logs")
def get_job_logs(
    job_id: str,
    log_service: LogService = Depends(get_log_service),
) -> Response:
    try:
        content = log_service.read_job_log(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(content=content, media_type="text/plain; charset=utf-8")


@router.post("/{job_id}/cancel", response_model=JobDetailResponse)
def cancel_job(
    job_id: str,
    job_service: JobService = Depends(get_job_service),
) -> JobDetailResponse:
    try:
        return JobDetailResponse(**job_service.cancel_job(job_id))
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _submit(*, job_type: str, request: JobRequest, job_service: JobService) -> JobCreateResponse:
    try:
        payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        record = job_service.submit_job(job_type=job_type, request_payload=payload)
    except ServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JobCreateResponse(
        job_id=record["job_id"],
        job_type=record["job_type"],
        status=record["status"],
        submitted_at=record["created_at"],
    )
