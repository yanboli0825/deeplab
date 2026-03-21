from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseLoggerHandler(ABC):
    """Backend-specific logging adapter used by models and runtime code."""

    def __init__(self, logger: Any) -> None:
        """Store the backend-specific logger instance.

        Args:
            logger: Concrete logger object from Lightning.

        Returns:
            None: The constructor stores the logger for later use.
        """

        self.logger = logger

    @abstractmethod
    def log_figure(
        self,
        figure: Any,
        file_path: str,
        caption: str = "",
        step: Optional[int] = None,
    ) -> None:
        """Log a figure through a concrete backend adapter.

        Args:
            figure: Matplotlib figure or similar object accepted by the backend.
            file_path: Artifact-relative path used by the backend.
            caption: Optional human-readable caption.
            step: Optional global step or epoch value.

        Returns:
            None: The function is implemented for backend side effects.
        """

        raise NotImplementedError


class NullLoggerHandler(BaseLoggerHandler):
    """No-op adapter used when no supported logger backend is attached."""

    def __init__(self) -> None:
        """Initialize the no-op logger handler.

        Returns:
            None: The constructor binds a `None` logger.
        """

        super().__init__(logger=None)

    def log_figure(
        self,
        figure: Any,
        file_path: str,
        caption: str = "",
        step: Optional[int] = None,
    ) -> None:
        """Ignore figure logging when no supported backend is attached.

        Args:
            figure: Ignored figure object.
            file_path: Ignored artifact-relative path.
            caption: Ignored caption text.
            step: Ignored step value.

        Returns:
            None: The function intentionally does nothing.
        """

        return
