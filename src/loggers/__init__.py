from typing import Any, List, Tuple, Type

import lightning as L

from .base import BaseLoggerHandler, NullLoggerHandler
from .mlflow_handler import MLFlowLoggerHandler
from .wandb_handler import WandBLoggerHandler


class LoggerFactory:
    """根据 Lightning logger 类型返回对应 handler。"""

    # logger 注册表
    _registry: List[Tuple[Type[Any], Type[BaseLoggerHandler]]] = [
        (L.pytorch.loggers.WandbLogger, WandBLoggerHandler),
        (L.pytorch.loggers.MLFlowLogger, MLFlowLoggerHandler),
    ]

    @classmethod
    def create(cls, logger: Any) -> BaseLoggerHandler:
        if logger is None:
            return NullLoggerHandler()

        for logger_type, handler_type in cls._registry:
            if isinstance(logger, logger_type):
                return handler_type(logger)

        return NullLoggerHandler()

    @classmethod
    def register(cls, logger_type: Type[Any], handler_type: Type[BaseLoggerHandler]) -> None:
        cls._registry.append((logger_type, handler_type))
