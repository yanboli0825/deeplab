import torch
import torch.nn as nn
from typing import Any, Dict
from .base_model import BaseModel


class DummyModel(BaseModel):
    """模板模型类

    继承 BaseModel 后，只需实现:
        - __init__: 初始化模型结构
        - forward: 前向传播

    其余逻辑由基类处理
    """
    def __init__(self, model_cfg: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        # 初始化分类器 - 必须在 super().__init__() 之前设置为 None
        # 因为 super().__init__() 会调用 save_hyperparameters()，
        # 而 classifier 是 nn.Module，需要先声明占位符
        self.classifier = None

        # 调用基类初始化 (设置指标、追踪逻辑等)
        # 这里会调用 save_hyperparameters()，注册所有参数
        super().__init__(model_cfg, *args, **kwargs)

        # 设置分类器 - 必须在 super().__init__() 之后
        # 这样 classifier 才能被正确注册为子模块
        self.classifier = nn.Linear(512, model_cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量，shape (batch_size, bag_size, feature_dim)

        Returns:
            输出 logits，shape (batch_size, num_classes)
        """
        bag_feature = torch.mean(x, dim=1)
        logits = self.classifier(bag_feature)
        return logits

    # 不需要覆写: training_step, on_train_epoch_end,
    #           validation_step, on_validation_epoch_end,
    #           test_step, on_test_epoch_end, _log_confusion_matrix
    # 这些方法由 BaseModel 提供

    # def configure_optimizers(self) -> Any:
    #     """配置优化器和调度器 (从配置文件读取)"""
    #     from hydra.utils import instantiate
    #     optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
    #     scheduler = instantiate(self.hparams.scheduler, optimizer=optimizer)
    #     return [optimizer], [scheduler]
