from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchmetrics

from src.loggers import LoggerFactory


class BaseModel(L.LightningModule, ABC):
    """Shared LightningModule scaffold for task logic and metric reporting."""

    def __init__(
        self,
        model_cfg: Dict[str, Any],
        metrics_cfg: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.num_classes = model_cfg.get("num_classes", 2)
        self.metrics_cfg = metrics_cfg or self._default_metrics()

        self._setup_metrics()
        self.save_hyperparameters()

        self.monitor = "val/loss"
        self.best_metric = float("inf") if self.monitor.startswith("loss") else float("-inf")
        self.best_epoch = -1

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Subclasses implement task-specific forward logic."""
        raise NotImplementedError

    def _default_metrics(self) -> Dict[str, Any]:
        return {
            "Accuracy": {"task": "multiclass", "num_classes": self.num_classes},
            "Precision": {"task": "multiclass", "num_classes": self.num_classes},
            "Recall": {"task": "multiclass", "num_classes": self.num_classes},
            "F1Score": {"task": "multiclass", "num_classes": self.num_classes},
            "AUROC": {"task": "multiclass", "num_classes": self.num_classes},
        }

    def _setup_metrics(self) -> None:
        base_metrics = {}
        for name, metric_kwargs in self.metrics_cfg.items():
            metric_class = getattr(torchmetrics, name)
            base_metrics[name] = metric_class(**metric_kwargs)

        metrics = torchmetrics.MetricCollection(base_metrics)
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")
        self.val_cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        self.test_cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)

    def on_train_start(self) -> None:
        if self.trainer and self.trainer.checkpoint_callback:
            self.monitor = self.trainer.checkpoint_callback.monitor or self.monitor
            self.best_metric = float("inf") if "loss" in self.monitor.lower() else float("-inf")

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_metrics.update(logits, y)
        return loss

    def on_train_epoch_end(self) -> None:
        output = self.train_metrics.compute()
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=True)
        self.train_metrics.reset()

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)

        self.val_metrics.update(logits, y)
        self.val_cm.update(logits, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        output = self.val_metrics.compute()
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=True)
        self.val_metrics.reset()

        current_metric = output.get(self.monitor, None)
        if current_metric is None:
            current_metric = self.trainer.callback_metrics.get(self.monitor)

        if current_metric is not None:
            current_value = current_metric.item() if hasattr(current_metric, "item") else current_metric
            is_better = (
                current_value < self.best_metric
                if "loss" in self.monitor.lower()
                else current_value > self.best_metric
            )
            if is_better:
                self.best_metric = current_value
                self.best_epoch = self.current_epoch
                self._log_confusion_matrix(self.val_cm.compute().cpu().numpy(), epoch=self.current_epoch, stage="val")
                self.log("best_metric", self.best_metric, on_step=False, on_epoch=True)
                self.log("best_epoch", self.best_epoch, on_step=False, on_epoch=True)

        self.val_cm.reset()

    def test_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)

        self.test_metrics.update(logits, y)
        self.test_cm.update(logits, y)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        output = self.test_metrics.compute()
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=True)
        self.test_metrics.reset()
        self._log_confusion_matrix(self.test_cm.compute().cpu().numpy(), epoch=self.current_epoch, stage="test")
        self.test_cm.reset()

    def _log_confusion_matrix(self, cm: Any, epoch: int, stage: str = "val") -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        if stage == "test":
            title = "Confusion Matrix (Test Set)"
            file_name = "confusion_matrices/test_cm.png"
        else:
            title = f"Confusion Matrix (Val Set, Epoch {epoch})"
            file_name = "confusion_matrices/val_cm.png"

        ax.set_title(title)

        if self.logger is None:
            plt.close(fig)
            return

        LoggerFactory.create(self.logger).log_figure(
            figure=fig,
            file_path=file_name,
            caption=title,
            step=epoch,
        )
        plt.close(fig)

    def configure_optimizers(self) -> Any:
        from hydra.utils import instantiate

        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
        scheduler = instantiate(self.hparams.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]
