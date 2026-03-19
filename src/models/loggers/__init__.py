from typing import Optional
import lightning as L
from .base import BaseLoggerHandler
from .wandb_handler import WandBLoggerHandler
from .mlflow_handler import MLFlowLoggerHandler


class LoggerFactory:
    """日志处理器工厂"""

    _handlers: dict = {
        L.pytorch.loggers.WandbLogger: WandBLoggerHandler,
        L.pytorch.loggers.MLFlowLogger: MLFlowLoggerHandler,
    }

    @classmethod
    def create(cls, logger: Any) -> Optional[BaseLoggerHandler]:
        """根据 logger 类型创建对应的处理器

        Args:
            logger: Lightning logger 实例

        Returns:
            BaseLoggerHandler 实例，如果类型不支持则返回 None
        """
        for logger_type, handler_class in cls._handlers.items():
            if isinstance(logger, logger_type):
                return handler_class(logger)
        return None

    @classmethod
    def register_handler(cls, logger_type: type, handler_class: type) -> None:
        """注册新的处理器类型

        Args:
            logger_type: logger 类型
            handler_class: 处理器类
        """
        cls._handlers[logger_type] = handler_class


__all__ = ['BaseLoggerHandler', 'WandBLoggerHandler', 'MLFlowLoggerHandler', 'LoggerFactory']
