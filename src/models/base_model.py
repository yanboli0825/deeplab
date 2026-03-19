from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import lightning as L
import torch
import torchmetrics
import matplotlib.pyplot as plt
import seaborn as sns


class BaseModel(L.LightningModule, ABC):
    """所有模型的基类，封装通用逻辑

    子类只需实现:
        - __init__(model_cfg, ...): 初始化模型结构
        - forward(x): 前向传播

    基类提供:
        - 指标管理
        - 通用训练/验证/测试流程
        - 最佳模型追踪
        - 混淆矩阵记录
    """

    def __init__(
        self,
        model_cfg: Dict[str, Any],
        metrics_cfg: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__()

        # 关键：在 save_hyperparameters() 之前先设置 model_cfg
        # 因为 save_hyperparameters() 可能会触发其他方法调用
        self.model_cfg = model_cfg
        self.num_classes = model_cfg.get('num_classes', 2)
        self.metrics_cfg = metrics_cfg or self._default_metrics()


        # 设置指标
        self._setup_metrics()

        # 现在可以安全地保存超参数
        self.save_hyperparameters()

        # # 设置指标
        # self._setup_metrics()

        # 追踪最佳模型
        self.monitor: str = "val/loss"  # 将在 on_train_start 中更新
        self.best_metric: float = float('inf') if self.monitor.startswith('loss') else float('-inf')
        self.best_epoch: int = -1


    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """子类必须实现的前向传播

        Args:
            x: 输入张量

        Returns:
            输出张量 (通常是 logits)
        """
        pass

    def _default_metrics(self) -> Dict[str, Any]:
        """可覆写的默认指标配置"""
        return {
            'Accuracy': {'task': 'multiclass', 'num_classes': self.num_classes},
            'Precision': {'task': 'multiclass', 'num_classes': self.num_classes},
            'Recall': {'task': 'multiclass', 'num_classes': self.num_classes},
            'F1Score': {'task': 'multiclass', 'num_classes': self.num_classes},
            'AUROC': {'task': 'multiclass', 'num_classes': self.num_classes},
        }

    def _setup_metrics(self) -> None:
        """根据配置设置指标"""
        base_metrics = {}
        for name, kwargs in self.metrics_cfg.items():
            metric_class = getattr(torchmetrics, name)
            base_metrics[name] = metric_class(**kwargs)

        metrics = torchmetrics.MetricCollection(base_metrics)
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # 混淆矩阵
        self.val_cm = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=self.num_classes
        )
        self.test_cm = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=self.num_classes
        )

    def on_train_start(self) -> None:
        """训练开始时获取监控指标"""
        if self.trainer and self.trainer.checkpoint_callback:
            self.monitor = self.trainer.checkpoint_callback.monitor or self.monitor
            # 初始化最佳指标方向
            if 'loss' in self.monitor.lower():
                self.best_metric = float('inf')
            else:
                self.best_metric = float('-inf')

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """训练步骤 - 子类可覆写"""
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_metrics.update(logits, y)
        return loss

    def on_train_epoch_end(self) -> None:
        """训练 epoch 结束"""
        output = self.train_metrics.compute()
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=True)
        self.train_metrics.reset()

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """验证步骤"""
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)

        self.val_metrics.update(logits, y)
        self.val_cm.update(logits, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """验证 epoch 结束"""
        output = self.val_metrics.compute()
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=True)
        self.val_metrics.reset()

        # 获取当前监控指标
        current_metric = output.get(self.monitor, None)
        if current_metric is None:
            current_metric = self.trainer.callback_metrics.get(self.monitor)

        if current_metric is not None:
            current_value = current_metric.item() if hasattr(current_metric, 'item') else current_metric
            is_better = current_value < self.best_metric if 'loss' in self.monitor.lower() else current_value > self.best_metric

            if is_better:
                self.best_metric = current_value
                self.best_epoch = self.current_epoch

                # 记录混淆矩阵
                cm = self.val_cm.compute().cpu().numpy()
                self._log_confusion_matrix(cm, epoch=self.current_epoch, stage='val')

                self.log("best_metric", self.best_metric, on_step=False, on_epoch=True)
                self.log("best_epoch", self.best_epoch, on_step=False, on_epoch=True)

        self.val_cm.reset()

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """测试步骤"""
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)

        self.test_metrics.update(logits, y)
        self.test_cm.update(logits, y)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """测试 epoch 结束"""
        output = self.test_metrics.compute()
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=True)
        self.test_metrics.reset()

        cm = self.test_cm.compute().cpu().numpy()
        self._log_confusion_matrix(cm, epoch=self.current_epoch, stage="test")

        self.test_cm.reset()

    def _log_confusion_matrix(self, cm: Any, epoch: int, stage: str = "val") -> None:
        """记录混淆矩阵图像到 logger

        Args:
            cm: 混淆矩阵数组
            epoch: 当前 epoch
            stage: 阶段名称 ('val' 或 'test')
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        if stage == "test":
            title = 'Confusion Matrix (Test Set)'
            file_name = "confusion_matrices/test_cm.png"
        else:
            title = f'Confusion Matrix (Val Set, Epoch {epoch})'
            file_name = f"confusion_matrices/val_cm.png"

        ax.set_title(title)

        if self.logger is None:
            plt.close(fig)
            return

        # WandB 记录
        if isinstance(self.logger, L.pytorch.loggers.WandbLogger):
            import wandb
            self.logger.experiment.log({
                f"{stage}_confusion_matrix": wandb.Image(fig, caption=title)
            }, step=epoch)

        # MLflow 记录
        elif isinstance(self.logger, L.pytorch.loggers.MLFlowLogger):
            import os, tempfile
            client = self.logger.experiment
            run_id = self.logger.run_id

            try:
                client.log_figure(run_id, fig, file_name)
            except Exception as e:
                print(f"MLflow log_figure failed: {e}. Using fallback...")
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    fig.savefig(tmp.name)
                    client.log_artifact(run_id, tmp.name, artifact_path="confusion_matrices")
                os.unlink(tmp.name)

        plt.close(fig)

    def configure_optimizers(self) -> Any:
        """配置优化器和学习率调度器 - 子类可覆写"""
        from hydra.utils import instantiate
        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
        scheduler = instantiate(self.hparams.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]
