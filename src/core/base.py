from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type


def _resolve_metric_class(metric_name: str, **kwargs) -> Type:
    """解析指标类

    Args:
        metric_name: 指标名称
        **kwargs: 指标构造参数

    Returns:
        torchmetrics 指标类
    """
    import torchmetrics
    return getattr(torchmetrics, metric_name)


class BaseComponent(ABC):
    """所有可配置组件的基类"""
    pass
