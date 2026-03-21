from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """Dataset backed by row mappings and explicit feature/label columns."""

    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        feature_columns: Sequence[str],
        label_column: str,
    ) -> None:
        """Store manifest rows and column definitions for tabular access.

        Args:
            rows: Row mappings read from a manifest file.
            feature_columns: Ordered feature column names.
            label_column: Column name containing the label.

        Returns:
            None: The constructor stores dataset metadata.
        """

        self.rows = list(rows)
        self.feature_columns = list(feature_columns)
        self.label_column = str(label_column)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: Dataset size.
        """

        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read one tabular sample and convert it to tensors.

        Args:
            idx: Zero-based sample index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Feature tensor and label tensor.
        """

        row = self.rows[idx]
        features = torch.tensor(
            [float(row[column]) for column in self.feature_columns],
            dtype=torch.float32,
        )
        label = torch.tensor(int(row[self.label_column]), dtype=torch.long)
        return features, label
