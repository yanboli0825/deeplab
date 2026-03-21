from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Linear classification head."""

    def __init__(self, in_features: int, num_classes: int) -> None:
        """Initialize the linear classification head.

        Args:
            in_features: Feature dimension produced by the backbone.
            num_classes: Number of output classes.

        Returns:
            None: The constructor builds the linear projection.
        """

        super().__init__()
        self.classifier = nn.Linear(int(in_features), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map backbone features to class logits.

        Args:
            x: Backbone feature tensor.

        Returns:
            torch.Tensor: Classification logits.
        """

        return self.classifier(x)
