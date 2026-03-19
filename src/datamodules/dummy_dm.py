from typing import Any, Dict, Optional, Sequence, Tuple

import lightning as L
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.datamodules.split import SplitIndices


class DummyDataset(Dataset):
    """Deterministic toy dataset keyed by sample ids."""

    def __init__(self, sample_ids: Sequence[int]) -> None:
        self.sample_ids = list(sample_ids)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_id = int(self.sample_ids[idx])
        generator = torch.Generator().manual_seed(sample_id)
        bag = torch.randn(10, 512, generator=generator)
        label = torch.tensor(sample_id % 2, dtype=torch.long)
        return bag, label


class DummyDataModule(L.LightningDataModule):
    """Template datamodule that consumes split artifacts instead of owning split policy."""

    def __init__(
        self,
        data_cfg: Dict[str, Any],
        split_indices: Optional[SplitIndices] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset: Optional[DummyDataset] = None
        self.val_dataset: Optional[DummyDataset] = None
        self.test_dataset: Optional[DummyDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        split = self.hparams.split_indices or self._load_split_manifest()
        if split is None:
            split = self._default_split()

        self.train_dataset = DummyDataset(split.train.tolist())
        self.val_dataset = DummyDataset(split.val.tolist())
        self.test_dataset = DummyDataset(split.test.tolist())

    def _load_split_manifest(self) -> Optional[SplitIndices]:
        split_file = self.hparams.data_cfg.get("split_file")
        if not split_file:
            return None

        with open(split_file, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}

        return SplitIndices(
            train=np.asarray(payload.get("train", []), dtype=int),
            val=np.asarray(payload.get("val", []), dtype=int),
            test=np.asarray(payload.get("test", []), dtype=int),
        )

    def _default_split(self) -> SplitIndices:
        total_samples = int(self.hparams.data_cfg.get("total_samples", 120))
        val_ratio = float(self.hparams.data_cfg.get("val_ratio", 0.2))
        test_ratio = float(self.hparams.data_cfg.get("test_ratio", 0.1))

        all_indices = np.arange(total_samples, dtype=int)
        n_test = int(round(total_samples * test_ratio))
        n_val = int(round(total_samples * val_ratio))

        test_idx = all_indices[:n_test]
        val_idx = all_indices[n_test:n_test + n_val]
        train_idx = all_indices[n_test + n_val:]

        return SplitIndices(train=train_idx, val=val_idx, test=test_idx)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup()
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.data_cfg.batch_size,
            num_workers=self.hparams.data_cfg.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup()
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.data_cfg.batch_size,
            num_workers=self.hparams.data_cfg.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup()
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.data_cfg.batch_size,
            num_workers=self.hparams.data_cfg.num_workers,
            shuffle=False,
        )
