from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.datamodules.base_dm import BaseDataModule
from src.datamodules.split import SplitIndices, SplitProvider


class DummyDataset(Dataset):
    """Deterministic toy dataset keyed by sample ids."""

    def __init__(self, sample_ids: Sequence[int]) -> None:
        """Store sample identifiers used to generate deterministic dummy data.

        Args:
            sample_ids: Sample identifiers that seed random feature generation.

        Returns:
            None: The constructor stores dataset indices.
        """

        self.sample_ids = list(sample_ids)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: Dataset size.
        """

        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate one deterministic dummy sample from its sample id.

        Args:
            idx: Position of the sample in the local dataset view.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Feature tensor and integer label tensor.
        """

        sample_id = int(self.sample_ids[idx])
        generator = torch.Generator().manual_seed(sample_id)
        bag = torch.randn(10, 512, generator=generator)
        label = torch.tensor(sample_id % 2, dtype=torch.long)
        return bag, label


class DummyDataModule(BaseDataModule):
    """Reference datamodule that consumes split artifacts."""

    def __init__(
        self,
        data_cfg: Dict[str, Any],
        split_provider: Optional[SplitProvider] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the reference datamodule.

        Args:
            data_cfg: Datamodule configuration such as batch size and dummy sample count.
            split_provider: Optional split provider supplied by the runtime.
            *args: Extra positional arguments kept for compatibility.
            **kwargs: Extra keyword arguments kept for compatibility.

        Returns:
            None: The constructor initializes dataset holders.
        """

        super().__init__(data_cfg=data_cfg, split_provider=split_provider, *args, **kwargs)
        self.train_dataset: Optional[DummyDataset] = None
        self.val_dataset: Optional[DummyDataset] = None
        self.test_dataset: Optional[DummyDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Build train/val/test datasets from the resolved split.

        Args:
            stage: Optional Lightning stage, unused by this reference implementation.

        Returns:
            None: The function populates dataset attributes and emits artifacts.
        """

        split = self.resolve_split()
        self._active_split = split

        self.train_dataset = DummyDataset(split.train.tolist())
        self.val_dataset = DummyDataset(split.val.tolist())
        self.test_dataset = DummyDataset(split.test.tolist())
        self.emit_data_artifacts()

    def _default_split(self) -> SplitIndices:
        """Create a simple deterministic split from a fixed local policy.

        Returns:
            SplitIndices: Default train/val/test split for the dummy dataset.
        """

        total_samples = int(self.hparams.data_cfg.get("num_samples", 120))
        val_ratio = 0.2
        test_ratio = 0.1

        all_indices = np.arange(total_samples, dtype=int)
        n_test = int(round(total_samples * test_ratio))
        n_val = int(round(total_samples * val_ratio))

        test_idx = all_indices[:n_test]
        val_idx = all_indices[n_test:n_test + n_val]
        train_idx = all_indices[n_test + n_val:]

        return SplitIndices(train=train_idx, val=val_idx, test=test_idx)

    def train_dataloader(self) -> DataLoader:
        """Build the training dataloader.

        Returns:
            DataLoader: Training dataloader for the dummy dataset.
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
        """Build the validation dataloader.

        Returns:
            DataLoader: Validation dataloader for the dummy dataset.
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
        """Build the test dataloader.

        Returns:
            DataLoader: Test dataloader for the dummy dataset.
        """

        if self.test_dataset is None:
            self.setup()
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.data_cfg.batch_size,
            num_workers=self.hparams.data_cfg.num_workers,
            shuffle=False,
        )
