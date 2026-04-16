import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_model import BaseModel
from typing import Any, Dict


class ABMIL(BaseModel):
    def __init__(self, model_cfg: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(model_cfg, *args, **kwargs)
        self.M = int(model_cfg.get("M", 768))
        self.L = 128
        self.ATTENTION_BRANCHES = 1
        self.nClasses =  int(model_cfg.get("num_classes", 2))

        self.attention_V = nn.Sequential(
          nn.Linear(self.M, self.L), # matrix V
          nn.Tanh()
        )

        self.attention_U = nn.Sequential(
          nn.Linear(self.M, self.L), # matrix U
          nn.Sigmoid()
        )
        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
          nn.Linear(self.M * self.ATTENTION_BRANCHES, self.nClasses),
          # nn.Sigmoid()
        )
    
    # def forward(self, x, y=None, instance_mask=None):
    def forward(self, x):
        x = x.squeeze(0) # B, N, M => N, M

        A_V = self.attention_V(x) # N, L
        A_U = self.attention_U(x) # N, L
        A = self.attention_w(A_V * A_U) # element-wise multiplication # N, ATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0) # ATTENTION_BRANCHES, N
        A = F.softmax(A, dim=1) # softmax over N

        Z = torch.mm(A, x) # ATTENTION_BRANCHES, M (slide embedding)
        
        logits = self.classifier(Z)
        return logits

        if not self.training:
            Y_hat = torch.argmax(logits, dim=1)
            Y_prob = F.softmax(logits, dim=1)
            return Y_prob, Y_hat, A, None
        else:
            ceLossFn = nn.CrossEntropyLoss()
            ceLoss = ceLossFn(logits, y)
            return {"ceLoss": ceLoss.float()}



