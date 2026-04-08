from __future__ import annotations

from typing import Any, Dict, Iterable

import torch

from src.models.backbones.tabular_mlp import TabularMLPBackbone
from src.models.base_model import BaseModel
from src.models.heads.classification_head import ClassificationHead


class TabularClassificationModel(BaseModel):
    """Reference task model showing the recommended backbone/head/task split."""

    def __init__(self, model_cfg: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        """Initialize the reference tabular classification task.

        Args:
            model_cfg: Task-specific settings such as input size and class count.
            *args: Extra positional arguments forwarded to `BaseModel`.
            **kwargs: Extra keyword arguments forwarded to `BaseModel`.

        Returns:
            None: The constructor builds the backbone and classification head.
        """

        self.backbone = None
        self.head = None
        super().__init__(model_cfg, *args, **kwargs)

        hidden_dims = model_cfg.get("hidden_dims", [128, 64])
        self.backbone = TabularMLPBackbone(
            input_dim=int(model_cfg["input_dim"]),
            hidden_dims=[int(dim) for dim in hidden_dims],
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
        self.head = ClassificationHead(
            in_features=self.backbone.output_dim,
            num_classes=int(model_cfg.get("num_classes", 2)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the tabular backbone and classification head.

        Args:
            x: Batched tabular feature tensor.

        Returns:
            torch.Tensor: Classification logits.
        """

        features = self.backbone(x)
        return self.head(features)
