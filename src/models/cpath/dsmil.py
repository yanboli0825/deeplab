import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import BaseModel


class FCLayer(nn.Module):
    def __init__(self, in_size: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_size, num_classes)

    def forward(self, feats: torch.Tensor):
        """
        feats: [N, D]
        returns:
            feats: [N, D]
            ins_prediction: [N, C]
        """
        ins_prediction = self.fc(feats)
        return feats, ins_prediction


class BClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        q_dim: int = 128,
        dropout_v: float = 0.0,
        nonlinear: bool = True,
        passing_v: bool = False,
    ):
        super().__init__()

        if nonlinear:
            self.q = nn.Sequential(
                nn.Linear(input_size, q_dim),
                nn.ReLU(),
                nn.Linear(q_dim, q_dim),
                nn.Tanh(),
            )
        else:
            self.q = nn.Linear(input_size, q_dim)

        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU(),
            )
        else:
            self.v = nn.Identity()

        # input: [1, C, D] -> output: [1, C, 1]
        self.fcc = nn.Conv1d(num_classes, num_classes, kernel_size=input_size)

    def forward(self, feats: torch.Tensor, c: torch.Tensor):
        """
        feats: [N, D]
        c: [N, C]

        returns:
            bag_prediction: [1, C]
            A: [N, C]
            B: [1, C, D]
        """
        v = self.v(feats)
        q = self.q(feats)

        _, m_indices = torch.sort(c, dim=0, descending=True)
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])
        q_max = self.q(m_feats)

        a = torch.mm(q, q_max.transpose(0, 1))
        a = F.softmax(a / math.sqrt(q.shape[1]), dim=0)

        b = torch.mm(a.transpose(0, 1), v)
        b = b.unsqueeze(0)
        bag_prediction = self.fcc(b).view(1, -1)

        return bag_prediction, a, b


class DSMIL(BaseModel):
    def __init__(self, model_cfg: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(model_cfg, *args, **kwargs)
        self.num_classes = model_cfg.get("num_classes", 2)

        self.i_classifier = FCLayer(
            in_size=model_cfg.get("in_size", 768),
            num_classes=model_cfg.get("num_classes", 2),
        )
        self.b_classifier = BClassifier(
            input_size=model_cfg.get("in_size", 768),
            num_classes=model_cfg.get("num_classes", 2),
            q_dim=model_cfg.get("q_dim", 128),
            dropout_v=model_cfg.get("dropout_v", 0.0),
            nonlinear=model_cfg.get("nonlinear", True),
            passing_v=model_cfg.get("passing_v", False),
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B, N, D], and here B is assumed to be 1
        returns:
            ins_prediction: [N, C]
            bag_prediction: [1, C]
            A: [N, C]
            B: [1, C, D]
        """
        x = x.squeeze(0)

        feats, ins_prediction = self.i_classifier(x)
        bag_prediction, a, b = self.b_classifier(feats, ins_prediction)

        return ins_prediction, bag_prediction, a, b

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        """Use DSMIL's mixed bag/max loss while reporting bag-level metrics."""

        x, y = batch
        ins_prediction, bag_prediction, _, _ = self(x)

        max_prediction, _ = torch.max(ins_prediction, dim=0)
        bag_logits = bag_prediction.view(1, -1)
        max_logits = max_prediction.view(1, -1)

        criterion = nn.CrossEntropyLoss()
        bag_loss = criterion(bag_logits, y)
        max_loss = criterion(max_logits, y)
        loss = 0.5 * bag_loss + 0.5 * max_loss

        if stage == "train":
            self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.train_metrics.update(bag_logits, y)
        elif stage == "val":
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.val_metrics.update(bag_logits, y)
            self.val_cm.update(bag_logits, y)
        elif stage == "test":
            self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.test_metrics.update(bag_logits, y)
            self.test_cm.update(bag_logits, y)
        else:
            raise ValueError(f"Unsupported stage: {stage}")

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, stage="val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, stage="test")
