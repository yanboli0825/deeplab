from typing import Any, Dict

import torch
import torch.nn as nn

from .base_model import BaseModel


class DummyModel(BaseModel):
    """Minimal reference model used by smoke tests and example configs."""

    def __init__(self, model_cfg: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        """Initialize the dummy classification head.

        Args:
            model_cfg: Model settings including `num_classes`.
            *args: Extra positional arguments forwarded to `BaseModel`.
            **kwargs: Extra keyword arguments forwarded to `BaseModel`.

        Returns:
            None: The constructor initializes the linear classifier.
        """

        # Declare the module attribute before `save_hyperparameters()` runs in BaseModel.
        self.classifier = None
        super().__init__(model_cfg, *args, **kwargs)
        self.classifier = nn.Linear(512, int(model_cfg.get("num_classes", 2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool a bag of embeddings and classify the pooled feature.

        Args:
            x: Tensor shaped like `(batch, instances, feature_dim)`.

        Returns:
            torch.Tensor: Classification logits for each batch element.
        """

        bag_feature = torch.mean(x, dim=1)
        return self.classifier(bag_feature)
