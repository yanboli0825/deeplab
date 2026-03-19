import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, Tuple, Optional, Dict
import lightning as L
import numpy as np


class DummyDataset(Dataset):
    """模拟 WSI 数据集: 每个样本是一个 'bag'，包含 10 个图像块 (tiles)"""

    def __init__(self, num_samples: int = 20) -> None:
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        bag = torch.randn(10, 512)
        label = torch.tensor(np.random.randint(0, 2))
        return bag, label


class DummyDataModule(L.LightningDataModule):
    """ The template datamodule class in this training framework.
    """

    def __init__(
        self,
        data_cfg: Dict[str, Any],
        fold: Optional[int] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.fold = fold

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            DummyDataset(700),
            batch_size=self.hparams.data_cfg.batch_size,
            num_workers=self.hparams.data_cfg.num_workers,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            DummyDataset(500),
            batch_size=self.hparams.data_cfg.batch_size,
            num_workers=self.hparams.data_cfg.num_workers,
            shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            DummyDataset(200),
            batch_size=self.hparams.data_cfg.batch_size,
            num_workers=self.hparams.data_cfg.num_workers,
            shuffle=False
        )
