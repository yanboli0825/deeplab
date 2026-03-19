from typing import Any, Dict
from .base import BaseLoggerHandler


class WandBLoggerHandler(BaseLoggerHandler):
    """WandB 日志处理器"""

    def __init__(self, logger):
        self.logger = logger

    def log(self, artifacts: Dict[str, Any], stage: str = "val") -> None:
        """记录工制品"""
        self.logger.experiment.log(artifacts)

    def log_figure(self, figure: Any, file_path: str, caption: str = "") -> None:
        """记录图像"""
        import wandb
        self.logger.experiment.log({
            file_path: wandb.Image(figure, caption=caption)
        })
