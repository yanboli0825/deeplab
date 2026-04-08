"""Reference datamodule for manifest-driven tabular classification."""

from __future__ import annotations

import csv
from typing import Any, Dict, List, Optional

import numpy as np
from torch.utils.data import DataLoader

from src.datamodules.base_dm import BaseDataModule
from src.datamodules.datasets.tabular_ds import TabularDataset
from src.datamodules.split import SplitIndices, SplitProvider


class TabularClassificationDataModule(BaseDataModule):
    """Reference implementation for CSV-backed classification datasets.

    The datamodule expects a manifest file and explicit feature/label column
    definitions. Split policy remains external and is injected via split inputs.
    """

    def __init__(
        self,
        data_cfg: Dict[str, Any],
        split_provider: Optional[SplitProvider] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the manifest-driven tabular datamodule.

        Args:
            data_cfg: Datamodule configuration including manifest and feature columns.
            split_provider: Optional split provider supplied by the runtime.
            *args: Extra positional arguments kept for compatibility.
            **kwargs: Extra keyword arguments kept for compatibility.

        Returns:
            None: The constructor initializes in-memory dataset holders.
        """

        super().__init__(data_cfg=data_cfg, split_provider=split_provider, *args, **kwargs)
        self.rows: List[Dict[str, Any]] = []
        self.train_dataset: Optional[TabularDataset] = None
        self.val_dataset: Optional[TabularDataset] = None
        self.test_dataset: Optional[TabularDataset] = None

    def prepare_data(self) -> None:
        """Validate the presence of the configured manifest path.

        Returns:
            None: The function validates configuration and raises on missing input.

        Raises:
            ValueError: Raised when `data_cfg.data_file` is missing.
        """

        data_file = self.hparams.data_cfg.get("data_file")
        if not data_file:
            raise ValueError("TabularClassificationDataModule requires data_cfg.data_file")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load the manifest and build split-specific datasets.

        Args:
            stage: Optional Lightning stage, unused by this implementation.

        Returns:
            None: The function populates dataset attributes and emits artifacts.
        """

        if not self.rows:
            with open(self.hparams.data_cfg.data_file, "r", encoding="utf-8", newline="") as f:
                self.rows = list(csv.DictReader(f))

        split = self.resolve_split()
        self._active_split = split

        feature_columns = list(self.hparams.data_cfg.feature_columns)
        label_column = str(self.hparams.data_cfg.label_column)

        self.train_dataset = TabularDataset(
            [self.rows[idx] for idx in split.train.tolist()],
            feature_columns,
            label_column,
        )
        self.val_dataset = TabularDataset(
            [self.rows[idx] for idx in split.val.tolist()],
            feature_columns,
            label_column,
        )
        self.test_dataset = TabularDataset(
            [self.rows[idx] for idx in split.test.tolist()],
            feature_columns,
            label_column,
        )
        self.emit_data_artifacts()

    def _default_split(self) -> SplitIndices:
        """Create a simple fallback split from manifest order.

        Returns:
            SplitIndices: Default split built from the loaded manifest rows.

        Raises:
            ValueError: Raised when the manifest is empty.
        """

        total_samples = len(self.rows)
        if total_samples == 0:
            raise ValueError("Tabular dataset is empty")

        val_ratio = 0.2
        test_ratio = 0.1
        indices = np.arange(total_samples, dtype=int)
        n_test = int(round(total_samples * test_ratio))
        n_val = int(round(total_samples * val_ratio))

        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]
        return SplitIndices(train=train_idx, val=val_idx, test=test_idx)

    def train_dataloader(self) -> DataLoader:
        """Build the training dataloader for tabular data.

        Returns:
            DataLoader: Training dataloader for the current split.
        """

        if self.train_dataset is None:
            self.setup()
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.data_cfg.batch_size,
            num_workers=self.hparams.data_cfg.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Build the validation dataloader for tabular data.

        Returns:
            DataLoader: Validation dataloader for the current split.
        """

        if self.val_dataset is None:
            self.setup()
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.data_cfg.batch_size,
            num_workers=self.hparams.data_cfg.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Build the test dataloader for tabular data.

        Returns:
            DataLoader: Test dataloader for the current split.
        """

        if self.test_dataset is None:
            self.setup()
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.data_cfg.batch_size,
            num_workers=self.hparams.data_cfg.num_workers,
            shuffle=False,
        )
