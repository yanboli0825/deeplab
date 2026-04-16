"""Service-layer exceptions."""


class ServiceError(RuntimeError):
    """Base class for service errors."""


class JobNotFoundError(ServiceError):
    """Raised when a requested job does not exist."""


class InvalidJobRequestError(ServiceError):
    """Raised when the service cannot adapt a job request."""
