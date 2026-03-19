from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import lightning as L


class BaseLoggerHandler(ABC):
    """日志处理器抽象基类。"""

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
    """空实现，避免业务层到处判空。"""

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
