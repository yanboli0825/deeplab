import numpy as np
from torch.utils.data import DataLoader

from src.datamodules.split import SplitIndices, SplitProvider
from src.datamodules.datasets.frozen_section_ds import CtranspathFrozenSectionDataset, Resnet50FrozenSectionDataset
from src.datamodules.base_dm import BaseDataModule
from typing import Any, Dict, Optional


class FrozenSectionDataModule(BaseDataModule):
    def __init__(self, data_cfg: Dict[str, Any], split_provider: Optional[SplitProvider] = None,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(data_cfg, split_provider=split_provider, *args, **kwargs)
        self.train_dataset: Optional[CtranspathFrozenSectionDataset] = None
        self.val_dataset: Optional[CtranspathFrozenSectionDataset] = None
        self.test_dataset: Optional[CtranspathFrozenSectionDataset] = None

        self.data: list[dict[str, Any]] = []

    def prepare_data(self) -> None:
        if not self.data:
            import csv
            with open(self.hparams.data_cfg.data_file) as f:
                reader = list(csv.DictReader(f))
            self.data = reader

    def setup(self, stage: Optional[str] = None) -> None:
        self.prepare_data()

        split = self.resolve_split()
        self._active_split = split

        self.train_dataset = CtranspathFrozenSectionDataset(
            [self.data[idx] for idx in split.train.tolist()],
        )
        self.val_dataset = CtranspathFrozenSectionDataset(
            [self.data[idx] for idx in split.val.tolist()],
        )
        self.test_dataset = CtranspathFrozenSectionDataset(
            [self.data[idx] for idx in split.test.tolist()],
        )
        self.emit_data_artifacts()


    def _default_split(self) -> SplitIndices:
        total_samples = len(self.data)
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

class FrozenSectionDataModule_Resnet50(BaseDataModule):
    def __init__(self, data_cfg: Dict[str, Any], split_provider: Optional[SplitProvider] = None,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(data_cfg, split_provider=split_provider, *args, **kwargs)
        self.train_dataset: Optional[Resnet50FrozenSectionDataset] = None
        self.val_dataset: Optional[Resnet50FrozenSectionDataset] = None
        self.test_dataset: Optional[Resnet50FrozenSectionDataset] = None

        self.data: list[dict[str, Any]] = []

    def prepare_data(self) -> None:
        if not self.data:
            import csv
            with open(self.hparams.data_cfg.data_file) as f:
                reader = list(csv.DictReader(f))
            self.data = reader

    def setup(self, stage: Optional[str] = None) -> None:
        self.prepare_data()

        split = self.resolve_split()
        self._active_split = split

        self.train_dataset = Resnet50FrozenSectionDataset(
            [self.data[idx] for idx in split.train.tolist()],
        )
        self.val_dataset = Resnet50FrozenSectionDataset(
            [self.data[idx] for idx in split.val.tolist()],
        )
        self.test_dataset = Resnet50FrozenSectionDataset(
            [self.data[idx] for idx in split.test.tolist()],
        )
        self.emit_data_artifacts()


    def _default_split(self) -> SplitIndices:
        total_samples = len(self.data)
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

if __name__ == "__main__":
    raise SystemExit("Instantiate FrozenSectionDataModule through Hydra config")
