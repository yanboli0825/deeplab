import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict

from src.models.cpath.nystrom_attention import NystromAttention
from src.models.base_model import BaseModel



class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, drop_rate=0):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim//8,
            heads=8,
            num_landmarks=dim//2,
            pinv_iterations=6,
            residual=True,
            dropout=drop_rate,

        )
    def forward(self, x, mask):
        h, attn = self.attn(self.norm(x), mask=mask, return_attn=True)
        x = x + h
        return x, attn

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1,2).view(B,C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2) # .flatten(2)表示从第二维开始   将后续维度展平为1维
        # 我认为这里可以加一层全连接层来将生硬拼接的位置信息进行更好地融合
        x = self.fc(x) # 我加的一层
        x = torch.cat([cls_token.unsqueeze(1), x], dim=1)
        return x

class TransMIL(BaseModel):
    # def __init__(self, n_classes, drop_rate=0):
    #     super().__init__()
    def __init__(self, model_cfg: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(model_cfg, *args, **kwargs)
        embed_dim = int(model_cfg.get("embed_dim", 2048))

        self._fc1 = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 512))
        self.layer1 = TransLayer(dim=512, drop_rate=model_cfg.get("drop_rate", 0.0))
        self.layer2 = TransLayer(dim=512, drop_rate=model_cfg.get("drop_rate", 0.0))
        self.layer3 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, model_cfg.get("num_classes", 2))
        self.pos_layer = PPEG(dim=512)

    def forward(self, x, instance_mask=None):
        h = self._fc1(x) # [B, n, 512]

        # Squaring
        H = h.shape[1]
        B = h.shape[0]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H

        if instance_mask is None:
            instance_mask = torch.ones(B, H, device=h.device, dtype=torch.bool)

        if add_length > 0:
            h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]
            instance_mask = torch.cat([instance_mask,torch.zeros(B, add_length, dtype=torch.bool, device=h.device)], dim=1)

        # cls_token
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1) # [B, N+1, 512]

        # 扩展mask以包含cls_token
        mask = torch.cat([torch.ones(B,1,device=h.device, dtype=torch.bool), instance_mask], dim=1) # [B, N+1]

        # Translayer x1
        h, attn1 = self.layer1(h, mask) # [B, N+1, 512]

        # PPEG
        h = self.pos_layer(h, _H, _W) # [B, N+1, 512]

        # Translayer x2
        h, _ = self.layer2(h, mask) # [B, N+1, 512]

        h, _ = self.layer3(h, mask)

        # cls_token
        h = self.norm(h)[:, 0]

        logits = self._fc2(h) # [B, n_classes]

        return logits

        # Historical branch kept for reference only. Under the current BaseModel
        # contract, forward() should return logits and the train/val/test steps
        # are handled by BaseModel.
        # if not self.training:
        #     Y_hat = torch.argmax(logits, dim=1)
        #     Y_prob = torch.softmax(logits, dim=1)
        #     return Y_prob, Y_hat, attn1, attn2
        # else:
        #     ce_loss_fn = nn.CrossEntropyLoss()
        #     ce_loss = ce_loss_fn(logits, y)
        #     return {"ceLoss": ce_loss.float()}




if __name__ == '__main__':
    print()
    # data = torch.randn(6,300,768).cuda()
    # model = VFT(6).cuda()
    # result, _, r3 = model(data)
    # print(r3)

