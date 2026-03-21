from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class TabularMLPBackbone(nn.Module):
    """Simple MLP backbone for tabular features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        """Initialize the tabular MLP backbone.

        Args:
            input_dim: Number of input features.
            hidden_dims: Hidden layer widths in order.
            dropout: Dropout probability applied after each hidden layer.

        Returns:
            None: The constructor builds the MLP layers.
        """

        super().__init__()
        dims = [int(input_dim), *[int(dim) for dim in hidden_dims]]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                ]
            )

        self.network = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode tabular features into a learned representation.

        Args:
            x: Batched tabular input tensor.

        Returns:
            torch.Tensor: Encoded feature tensor.
        """

        return self.network(x)
