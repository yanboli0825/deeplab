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
        """Log a figure to Weights & Biases.

        Args:
            figure: Matplotlib figure to log.
            file_path: Artifact-relative name shown in WandB.
            caption: Caption displayed with the image.
            step: Optional step value used by WandB.

        Returns:
            None: The function logs the image through WandB side effects.
        """

        import wandb

        payload = {file_path: wandb.Image(figure, caption=caption)}
        if step is not None:
            self.logger.experiment.log(payload, step=step)
        else:
            self.logger.experiment.log(payload)
