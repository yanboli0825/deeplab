"""Job status model."""

from enum import Enum


class JobStatus(str, Enum):
    """Service-visible lifecycle for submitted jobs."""

    PENDING = "PENDING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
