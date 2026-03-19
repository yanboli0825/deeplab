from typing import Any, Dict

import torch
import torch.nn as nn

from .base_model import BaseModel


class DummyModel(BaseModel):
    """Minimal reference model used by smoke tests and example configs."""

    def __init__(self, model_cfg: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        # Declare the module attribute before `save_hyperparameters()` runs in BaseModel.
        self.classifier = None
        super().__init__(model_cfg, *args, **kwargs)
        self.classifier = nn.Linear(512, model_cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool a bag of embeddings and classify the pooled feature."""

        bag_feature = torch.mean(x, dim=1)
        return self.classifier(bag_feature)
