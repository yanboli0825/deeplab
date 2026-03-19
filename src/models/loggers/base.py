from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import lightning as L


class BaseLoggerHandler(ABC):
    """日志处理器基类

    使用策略模式处理不同类型的 logger
    """

    @abstractmethod
    def log(self, artifacts: Dict[str, Any], stage: str = "val") -> None:
        """记录工制品

        Args:
            artifacts: 工制品字典，包含图像、图表等
            stage: 阶段名称
        """
        pass

    @abstractmethod
    def log_figure(self, figure: Any, file_path: str, caption: str = "") -> None:
        """记录图像

        Args:
            figure: matplotlib 图像对象
            file_path: 文件路径
            caption: 图像标题
        """
        pass
