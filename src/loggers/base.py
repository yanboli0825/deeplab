from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseLoggerHandler(ABC):
    """Backend-specific logging adapter used by models and runtime code."""

    def __init__(self, logger: Any) -> None:
        self.logger = logger

    @abstractmethod
    def log_figure(
        self,
        figure: Any,
        file_path: str,
        caption: str = "",
        step: Optional[int] = None,
    ) -> None:
        raise NotImplementedError


class NullLoggerHandler(BaseLoggerHandler):
    """No-op adapter used when no supported logger backend is attached."""

    def __init__(self) -> None:
        super().__init__(logger=None)

    def log_figure(
        self,
        figure: Any,
        file_path: str,
        caption: str = "",
        step: Optional[int] = None,
    ) -> None:
        return
