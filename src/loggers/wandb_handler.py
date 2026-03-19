from typing import Any, Optional

from .base import BaseLoggerHandler


class WandBLoggerHandler(BaseLoggerHandler):
    """Weights & Biases adapter."""

    def log_figure(
        self,
        figure: Any,
        file_path: str,
        caption: str = "",
        step: Optional[int] = None,
    ) -> None:
        import wandb

        payload = {file_path: wandb.Image(figure, caption=caption)}
        if step is not None:
            self.logger.experiment.log(payload, step=step)
        else:
            self.logger.experiment.log(payload)
