import os
import tempfile
from typing import Any, Dict
from .base import BaseLoggerHandler


class MLFlowLoggerHandler(BaseLoggerHandler):
    """MLFlow 日志处理器"""

    def __init__(self, logger):
        self.logger = logger

    def log(self, artifacts: Dict[str, Any], stage: str = "val") -> None:
        """记录工制品"""
        client = self.logger.experiment
        run_id = self.logger.run_id

        for key, value in artifacts.items():
            if hasattr(value, 'savefig'):
                # 是 matplotlib 图像
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    value.savefig(tmp.name)
                    client.log_artifact(run_id, tmp.name, artifact_path=key)
                os.unlink(tmp.name)
            else:
                client.log_param(run_id, key, value)

    def log_figure(self, figure: Any, file_path: str, caption: str = "") -> None:
        """记录图像"""
        client = self.logger.experiment
        run_id = self.logger.run_id

        try:
            client.log_figure(run_id, figure, file_path)
        except Exception:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                figure.savefig(tmp.name)
                client.log_artifact(run_id, tmp.name, artifact_path="confusion_matrices")
            os.unlink(tmp.name)
