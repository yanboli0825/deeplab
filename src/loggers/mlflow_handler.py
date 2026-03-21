import os
import tempfile
from typing import Any, Optional

from .base import BaseLoggerHandler


class MLFlowLoggerHandler(BaseLoggerHandler):
    """MLflow adapter."""

    def log_figure(
        self,
        figure: Any,
        file_path: str,
        caption: str = "",
        step: Optional[int] = None,
    ) -> None:
        """Log a figure to MLflow, falling back to artifact upload when needed.

        Args:
            figure: Matplotlib figure to log.
            file_path: Artifact-relative path inside MLflow.
            caption: Unused caption placeholder kept for interface compatibility.
            step: Unused step placeholder kept for interface compatibility.

        Returns:
            None: The function logs the figure through MLflow side effects.
        """

        client = self.logger.experiment
        run_id = self.logger.run_id

        try:
            client.log_figure(run_id, figure, file_path)
        except Exception:
            artifact_path = os.path.dirname(file_path) or ""
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                figure.savefig(tmp.name)
                client.log_artifact(run_id, tmp.name, artifact_path=artifact_path)
            os.unlink(tmp.name)
