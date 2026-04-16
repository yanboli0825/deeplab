from src.models.base_model import BaseModel
from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
import math



class GatedAttention(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int = 1024) -> None:
        super().__init__()

        self.attention_V = nn.Sequential(
          nn.Linear(dim_in, dim_hidden), # matrix V
          nn.Tanh()
        )

        self.attention_U = nn.Sequential(
          nn.Linear(dim_in, dim_hidden), # matrix U
          nn.Sigmoid()
        )

        self.attention_w = nn.Linear(dim_hidden, 1)

        self.norm = nn.LayerNorm(dim_in)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x.shape = [B, N, dim_in], mask.shape = [B, N]
        a_u = self.attention_U(x)  # [B, N, dim_hidden]
        a_v = self.attention_V(x)  # [B, N, dim_hidden]
        a = self.attention_w(a_v * a_u)  # [B, N, 1]
        a = torch.transpose(a, 2, 1)  # [B, 1, N]

        mask = mask.unsqueeze(1)
        a = a.masked_fill(mask, -1e9)

        a = F.softmax(a, dim=-1)  # softmax over N
        z = torch.bmm(a, x)  # [B, 1, N] × (B, N, dim_in) => (B, 1, dim_in)
        return self.norm(z.squeeze(1))  # [B, D]

class MLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_layers: int = 1, dim_hidden: int = 1024, dropout: float = 0.0,
                 act_func: nn.Module = nn.GELU, use_layernorm: bool = False) -> None:
        super().__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(dim_in, dim_hidden))
            layers.append(act_func())
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(dim_hidden, dim_out))
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(dim_out) if use_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))

class Transformer(nn.Module):
    def __init__(self, dim_in: int, num_heads: int = 8, atten_dropout: float = 0.1, proj_dropout: float = 0.1) -> None:
        super().__init__()
        assert dim_in % num_heads == 0, "dim_in must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim_in // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim_in, dim_in * 3)
        self.out_proj = nn.Linear(dim_in, dim_in)
        self.atten_dropout = nn.Dropout(p=atten_dropout)
        self.proj_dropout = nn.Dropout(p=proj_dropout)

    def forward(self,x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, F, D = x.shape
        assert B == 1, "B must be 1"

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, F, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, F, Hd]
        atten_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, F, F]
        atten = torch.softmax(atten_logits, dim=-1)
        atten = self.atten_dropout(atten)
        out = torch.matmul(atten, v)  # [B, H, F, Hd]
        out = out.transpose(1, 2).contiguous().reshape(B, F, D)
        out = self.out_proj(out)
        out = self.proj_dropout(out)
        return out, atten

# class OrderMagAwareTransformer(Transformer):
#     def __init__(self, dim_in: int, num_heads: int = 8, num_order_buckets:int = 8, num_mags: int = 5,
#                  atten_dropout: float = 0.1, proj_dropout: float = 0.1) -> None:
#         super().__init__(dim_in, num_heads, atten_dropout, proj_dropout)
#         self.order_bias = nn.Embedding(num_order_buckets, num_heads)
#
#
#     def forward(self, x: torch.Tensor, order: torch.Tensor, meta: torch.Tensor, mag_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         B, F, D = x.shape
#
#         qkv = self.qkv(x) # [B, F, 3D]
#         qkv = qkv.reshape(B, F, 3, self.num_heads, self.head_dim)
#         qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, F, Hd]
#         q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, F, Hd]
#
#         atten_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale # [B, H, F, F]
#
#         order_dist_map = (order.unsqueeze(2) - order.unsqueeze(1)).abs()
#
#     @staticmethod
#     def _bucketize_order_distance(
#         order_dist: torch.Tensor,
#         num_buckets: int,
#     ) -> torch.Tensor:
#         """
#         对 retained-order 相对距离 |i-j| 做离散化，供 relative bias embedding 使用。
#
#         使用简单的 log bucket：
#             bucket = floor(log2(dist + 1))
#
#         参数
#         ----
#         order_dist: LongTensor[B, N, N]
#             pairwise retained-order 距离
#         num_buckets: int
#             bucket 数
#
#         返回
#         ----
#         bucket_ids: LongTensor[B, N, N]
#         """
#         bucket_ids = torch.floor(torch.log2(order_dist.float() + 1.0)).long()
#         bucket_ids = bucket_ids.clamp(min=0, max=num_buckets - 1)
#         return bucket_ids
#
#
# class OrderMagAwareSelfAttention(nn.Module):
#     """
#     packet-level 多头自注意力。
#
#     与旧版的差别
#     ------------
#     不再使用任何 gap bias。
#     attention logits 中只显式加入：
#     1) retained-order relative bias
#     2) magnification pair bias
#
#     解释
#     ----
#     - retained-order relative bias：表示在保留序列中“前后邻近”
#     - magnification pair bias：表示不同倍率之间的交互模式
#     """
#
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int,
#         num_order_buckets: int,
#         num_scales: int,
#         attn_dropout: float = 0.1,
#         proj_dropout: float = 0.1,
#     ) -> None:
#         super().__init__()
#         assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
#
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3)
#         self.out_proj = nn.Linear(dim, dim)
#
#         self.order_bias = nn.Embedding(num_order_buckets, num_heads)
#         self.mag_pair_bias = nn.Parameter(
#             torch.zeros(num_scales, num_scales, num_heads)
#         )
#
#         self.attn_dropout = nn.Dropout(attn_dropout)
#         self.proj_dropout = nn.Dropout(proj_dropout)
#
#         nn.init.trunc_normal_(self.order_bias.weight, std=0.02)
#         nn.init.zeros_(self.mag_pair_bias)
#
#     def forward(
#         self,
#         x: torch.Tensor,          # [B, N, D]
#         order_rank: torch.Tensor, # [B, N]
#         mag_id: torch.Tensor,     # [B, N]
#         mask: torch.Tensor,       # [B, N], True=有效
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         B, N, D = x.shape
#
#         qkv = self.qkv(x)  # [B, N, 3D]
#         qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
#         qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, Hd]
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
#
#         # --------------------------------------------------
#         # 1) retained-order relative bias
#         # --------------------------------------------------
#         order_dist = (order_rank.unsqueeze(2) - order_rank.unsqueeze(1)).abs()  # [B, N, N]
#         order_bucket = bucketize_order_distance(
#             order_dist,
#             self.order_bias.num_embeddings
#         )  # [B, N, N]
#
#         order_bias = self.order_bias(order_bucket)  # [B, N, N, H]
#         order_bias = order_bias.permute(0, 3, 1, 2)  # [B, H, N, N]
#
#         # --------------------------------------------------
#         # 2) magnification pair bias
#         # --------------------------------------------------
#         mag_i = mag_id.unsqueeze(2).expand(B, N, N)
#         mag_j = mag_id.unsqueeze(1).expand(B, N, N)
#         mag_bias = self.mag_pair_bias[mag_i, mag_j]  # [B, N, N, H]
#         mag_bias = mag_bias.permute(0, 3, 1, 2)      # [B, H, N, N]
#
#         attn_logits = attn_logits + order_bias + mag_bias
#
#         # key mask：无效 key 不允许被关注
#         key_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
#         attn_logits = attn_logits.masked_fill(~key_mask, -1e9)
#
#         attn = torch.softmax(attn_logits, dim=-1)
#         attn = self.attn_dropout(attn)
#
#         out = torch.matmul(attn, v)  # [B, H, N, Hd]
#         out = out.transpose(1, 2).contiguous().reshape(B, N, D)
#         out = self.out_proj(out)
#         out = self.proj_dropout(out)
#
#         # 无效 query 输出清零
#         out = out * mask.unsqueeze(-1).float()
#         return out, attn
#
#

class MultiClassTopKEvidencePool(nn.Module):
    def __init__(
        self,
        dim_in: int,
        num_classes: int,
        topk_ratio: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.topk_ratio = topk_ratio

        # 对每个 frame token 输出 C 个类别分数
        self.score_head = nn.Linear(dim_in, num_classes)

    def forward(
        self,
        x: torch.Tensor,   # [B, N, D], 默认 B=1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"x 应该是 [B, N, D]，实际 shape={tuple(x.shape)}")

        B, N, D = x.shape
        C = self.num_classes

        if B != 1:
            raise ValueError(
                f"当前这个简化版本默认 B=1，但实际 B={B}。"
                "如果你以后要支持 B>1，需要恢复 batch 维循环或向量化实现。"
            )

        if N == 0:
            raise ValueError("当前样本没有任何 frame，无法做 evidence pooling")

        # --------------------------------------------------
        # 1) 每个 frame 对每个类别打分
        # raw_scores: [1, N, C]
        # --------------------------------------------------
        raw_scores = self.score_head(x)

        # 去掉 batch 维，后面按单样本处理
        # scores_0: [N, C]
        scores_0 = raw_scores[0]

        # top-k 个数，至少为 1
        k = max(1, math.ceil(N * self.topk_ratio))

        class_vecs = []
        evidence_alpha_0 = torch.zeros(N, C, device=x.device, dtype=x.dtype)

        # --------------------------------------------------
        # 2) 对每个类别单独做 top-k pooling
        # --------------------------------------------------
        for c in range(C):
            # 第 c 类在所有 frame 上的分数: [N]
            scores_c = scores_0[:, c]

            # 取该类别 top-k frame
            topk_scores, topk_idx = torch.topk(scores_c, k=k, dim=0)

            # 在 top-k 内做 softmax，得到该类别的 frame 权重
            alpha_c = torch.softmax(topk_scores, dim=0)  # [k]

            # 聚合该类别的 evidence vector
            # x[0, topk_idx]: [k, D]
            vec_c = torch.sum(alpha_c.unsqueeze(-1) * x[0, topk_idx], dim=0)  # [D]

            class_vecs.append(vec_c)
            evidence_alpha_0[topk_idx, c] = alpha_c

        # [C, D]
        class_vecs = torch.stack(class_vecs, dim=0)

        # 恢复 batch 维
        evidence_vecs = class_vecs.unsqueeze(0)         # [1, C, D]
        evidence_alpha = evidence_alpha_0.unsqueeze(0) # [1, N, C]

        return evidence_vecs, raw_scores, evidence_alpha

if __name__ == "__main__":
    model = MultiClassTopKEvidencePool(dim_in=10, num_classes=2, topk_ratio=0.2)
    input = torch.randn(1, 64, 10)
    y = model(input)
    print(y)

class MOHPMIL_v0(BaseModel):
    def __init__(self, model_cfg: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(model_cfg, *args, **kwargs)
        self.dim_in: int = model_cfg.get("dim_in", 768)
        self.dim_hidden: int = model_cfg.get("dim_hidden", 1024)
        self.meta_dim: int = model_cfg.get("meta_dim", 3)
        self.mag_dim: int = model_cfg.get("mag_dim", 5)
        self.num_classes: int = model_cfg.get("num_classes", 2)

        # model architecture
        self.patch_fusion = GatedAttention(dim_in=self.dim_in, dim_hidden=self.dim_hidden)

        self.mag_embed = nn.Embedding(self.mag_dim, self.dim_in)

        self.meta_embed = MLP(self.meta_dim, self.dim_in)

        self.frame_fusion = Transformer(self.dim_in)

        self.evidence_branch = None
        self.context_branch = None

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.dim_in),
            nn.Linear(self.dim_in, self.dim_hidden),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.dim_hidden, self.num_classes),
        )

    def forward(self, x: torch.Tensor, order: torch.Tensor, meta: torch.Tensor, mag_id: torch.Tensor, patch_mask: torch.Tensor = None) -> torch.Tensor:
        # x.shape = [B, F. P, D]
        # order.shape = [B, F]
        # meta.shape = [B, F, 3]
        # mag.shape = [B, F]
        # patch_mask.shape = [B, F, P]
        B, F, P, D = x.shape

        assert B == 1 # currently only support batchsize = 1

        x = x.view(B * F, P, D)  # [F, P, D]

        frame_tokens = self.patch_fusion(x, patch_mask).view(B, F, D) # [B, F, D]
        emb_tokens = self.meta_embed(meta)
        mag_tokens = self.mag_embed(mag_id)

        frame_tokens = frame_tokens + emb_tokens + mag_tokens

        pass



# """
# MHO-PMIL
# ========
# Magnification-aware Hierarchical Ordered Packet MIL
#
# 最终版设计：
#     patch -> frame -> ordered packet
#
# 核心原则
# --------
# 1. frame_index 仅用于恢复 retained frame 的顺序，不进入模型做 gap 建模
# 2. packet-level 只建模 retained order，不建模原始时间间隔
# 3. magnification 是核心条件变量（scale condition）
# 4. quality metadata 只做 reliability，不直接当类别证据
# 5. 每个 frame 内先做 patch-level spatial aggregation，再做 packet-level ordered aggregation
#
# ------------------------------------------------------------
# 期望 DataLoader 输出的 batch 结构
# ------------------------------------------------------------
# batch = {
#     "patches": Tensor[B, N, M, 3, H, W],
#     "patch_coords": Tensor[B, N, M, 2],     # 建议归一化到 [0, 1]
#     "patch_mask": BoolTensor[B, N, M],      # True=有效 patch
#     "frame_mask": BoolTensor[B, N],         # True=有效 frame
#
#     "frame_index": LongTensor[B, N],        # 只用于排序/对齐，不进入模型
#     "magnification": LongTensor[B, N],      # magnification id，如 2x->0,10x->1,20x->2,40x->3
#     "sharpness": FloatTensor[B, N],
#     "tissue_area": FloatTensor[B, N],
#     "label": FloatTensor[B],                # 二分类 0/1
# }
#
# ------------------------------------------------------------
# 依赖
# ------------------------------------------------------------
# pip install timm lightning torchmetrics
#
# 如果你用自己的 backbone，也可以把 PatchBackbone 替换掉。
# """
#
#
#
# # -*- coding: utf-8 -*-
# from __future__ import annotations
#
# import math
#
# from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
# import torch
#
# import torch.nn as nn
# import torch.nn.functional as F
#
# import lightning.pytorch as pl
#
# try:
#     import timm
# except ImportError:
#     timm = None
#
# try:
#     from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score
#     _HAS_TORCHMETRICS = True
# except ImportError:
#     _HAS_TORCHMETRICS = False
#
#
# # ============================================================
# # 1. 基础工具函数
# # ============================================================
#
# def masked_softmax(
#     logits: torch.Tensor,
#     mask: torch.Tensor,
#     dim: int = -1,
#     eps: float = 1e-8,
# ) -> torch.Tensor:
#     """
#     对带 mask 的 logits 做 softmax。
#
#     参数
#     ----
#     logits: Tensor[..., L]
#         未归一化分数
#     mask: BoolTensor[..., L]
#         True 表示有效位置，False 表示 padding
#     dim:
#         softmax 维度
#
#     返回
#     ----
#     probs: Tensor[..., L]
#         在有效位置上归一化后的概率，padding 位置为 0
#     """
#     masked_logits = logits.masked_fill(~mask, -1e9)
#     probs = torch.softmax(masked_logits, dim=dim)
#     probs = probs * mask.to(probs.dtype)
#     probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(eps)
#     return probs
#
#
# def compute_retained_order_features(
#     frame_mask: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     根据 retained frame 在输入张量中的顺序，计算：
#     1) order_rank: 每个有效 frame 在 retained 序列中的编号，从 0 开始
#     2) norm_pos  : 每个有效 frame 在 retained packet 内的归一化位置 [0, 1]
#
#     重要说明
#     --------
#     这里不再使用 frame_index 的差值。
#     我们只使用“保留后的顺序”，因为这才是当前数据中最安全、最可解释的顺序信息。
#
#     参数
#     ----
#     frame_mask: BoolTensor[B, N]
#         True=有效 frame, False=padding frame
#
#     返回
#     ----
#     order_rank: LongTensor[B, N]
#     norm_pos: FloatTensor[B, N]
#     """
#     # cumsum 后减 1，可以得到有效 frame 的顺序编号
#     # 例如 [1,1,1,0,0] -> cumsum=[1,2,3,3,3] -> rank=[0,1,2,2,2]
#     order_rank = torch.cumsum(frame_mask.long(), dim=1) - 1
#     order_rank = torch.clamp(order_rank, min=0)
#
#     # padding 位置统一置 0
#     order_rank = torch.where(frame_mask, order_rank, torch.zeros_like(order_rank))
#
#     valid_len = frame_mask.sum(dim=1, keepdim=True).clamp_min(1)
#     denom = (valid_len - 1).clamp_min(1)
#     norm_pos = order_rank.float() / denom.float()
#     norm_pos = norm_pos * frame_mask.float()
#
#     return order_rank, norm_pos
#
#
# def bucketize_order_distance(
#     order_dist: torch.Tensor,
#     num_buckets: int,
# ) -> torch.Tensor:
#     """
#     对 retained-order 相对距离 |i-j| 做离散化，供 relative bias embedding 使用。
#
#     使用简单的 log bucket：
#         bucket = floor(log2(dist + 1))
#
#     参数
#     ----
#     order_dist: LongTensor[B, N, N]
#         pairwise retained-order 距离
#     num_buckets: int
#         bucket 数
#
#     返回
#     ----
#     bucket_ids: LongTensor[B, N, N]
#     """
#     bucket_ids = torch.floor(torch.log2(order_dist.float() + 1.0)).long()
#     bucket_ids = bucket_ids.clamp(min=0, max=num_buckets - 1)
#     return bucket_ids
#
#
# def build_2d_sincos_pos_embed(
#     coords: torch.Tensor,
#     dim: int,
#     temperature: float = 10000.0,
# ) -> torch.Tensor:
#     """
#     根据二维连续坐标构造 2D sine-cos positional encoding。
#
#     参数
#     ----
#     coords: Tensor[..., 2]
#         坐标，建议已归一化到 [0, 1]
#         coords[..., 0] = x, coords[..., 1] = y
#     dim: int
#         输出维度，要求能被 4 整除
#     temperature: float
#         正余弦编码温度
#
#     返回
#     ----
#     pos_emb: Tensor[..., dim]
#
#     说明
#     ----
#     我们不再用简单的 coord MLP -> D。
#     这里改成更标准的 2D sine-cos positional encoding。
#     """
#     if dim % 4 != 0:
#         raise ValueError(f"2D sine-cos positional encoding 要求 dim % 4 == 0, 但当前 dim={dim}")
#
#     quarter_dim = dim // 4
#     device = coords.device
#     dtype = coords.dtype
#
#     # 频率项
#     omega = torch.arange(quarter_dim, device=device, dtype=dtype)
#     omega = 1.0 / (temperature ** (omega / quarter_dim))  # [quarter_dim]
#
#     # 将归一化坐标映射到 [0, 2pi]
#     x = coords[..., 0:1] * (2.0 * math.pi)
#     y = coords[..., 1:2] * (2.0 * math.pi)
#
#     x_proj = x * omega   # [..., quarter_dim]
#     y_proj = y * omega
#
#     x_emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # [..., dim/2]
#     y_emb = torch.cat([torch.sin(y_proj), torch.cos(y_proj)], dim=-1)  # [..., dim/2]
#
#     pos_emb = torch.cat([x_emb, y_emb], dim=-1)  # [..., dim]
#     return pos_emb
#
#
# class MLP(nn.Module):
#     """
#     常用两层 MLP。
#     """
#
#     def __init__(
#         self,
#         in_dim: int,
#         hidden_dim: int,
#         out_dim: int,
#         dropout: float = 0.0,
#         act_layer: nn.Module = nn.GELU,
#         use_layernorm: bool = False,
#     ) -> None:
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             act_layer(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, out_dim),
#         )
#         self.norm = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.norm(self.net(x))
#
#
# # ============================================================
# # 2. patch backbone
# # ============================================================
#
# class PatchBackbone(nn.Module):
#     """
#     patch 级图像编码 backbone。
#
#     默认使用 timm backbone，并移除分类头，只输出 global pooled feature。
#     你可以把这里替换成自己的 pathology backbone（例如 CTransPath）。
#     """
#
#     def __init__(
#         self,
#         backbone_name: str = "resnet50",
#         pretrained: bool = True,
#         out_dim: int = 512,
#     ) -> None:
#         super().__init__()
#
#         if timm is None:
#             raise ImportError("需要安装 timm：pip install timm")
#
#         self.backbone = timm.create_model(
#             backbone_name,
#             pretrained=pretrained,
#             num_classes=0,
#             global_pool="avg",
#         )
#
#         in_dim = getattr(self.backbone, "num_features", None)
#         if in_dim is None:
#             raise ValueError(f"无法从 backbone={backbone_name} 读取 num_features")
#
#         self.proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
#         self.out_dim = out_dim
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: Tensor[B, 3, H, W]
#         return: Tensor[B, out_dim]
#         """
#         feat = self.backbone(x)
#         if feat.dim() > 2:
#             feat = feat.flatten(1)
#         feat = self.proj(feat)
#         return feat
#
#
# # ============================================================
# # 3. frame-level spatial encoder
# # ============================================================
#
# class SpatialFrameEncoder(nn.Module):
#     """
#     单帧内部的 patch 聚合模块。
#
#     输入
#     ----
#     patch_tokens: Tensor[BN, M, D]
#     patch_coords: Tensor[BN, M, 2]
#     patch_mask  : BoolTensor[BN, M]
#     mag_id      : LongTensor[BN]
#
#     输出
#     ----
#     frame_token : Tensor[BN, D]
#
#     设计
#     ----
#     对每个 patch token，加上：
#     1) 2D sine-cos position embedding
#     2) magnification embedding
#
#     然后用轻量 Transformer 在单帧内部聚合 patch，
#     最终用 CLS token 作为 frame representation。
#     """
#
#     def __init__(
#         self,
#         dim: int,
#         num_scales: int,
#         num_layers: int = 2,
#         num_heads: int = 8,
#         mlp_ratio: float = 4.0,
#         dropout: float = 0.1,
#     ) -> None:
#         super().__init__()
#
#         self.dim = dim
#         self.mag_embed = nn.Embedding(num_scales, dim)
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
#
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=dim,
#             nhead=num_heads,
#             dim_feedforward=int(dim * mlp_ratio),
#             dropout=dropout,
#             activation="gelu",
#             batch_first=True,
#             norm_first=True,
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.norm = nn.LayerNorm(dim)
#
#         nn.init.trunc_normal_(self.cls_token, std=0.02)
#
#     def forward(
#         self,
#         patch_tokens: torch.Tensor,   # [BN, M, D]
#         patch_coords: torch.Tensor,   # [BN, M, 2]
#         patch_mask: torch.Tensor,     # [BN, M]
#         mag_id: torch.Tensor,         # [BN]
#     ) -> torch.Tensor:
#         BN, M, D = patch_tokens.shape
#
#         pos_emb = build_2d_sincos_pos_embed(patch_coords, dim=D)     # [BN, M, D]
#         mag_emb = self.mag_embed(mag_id).unsqueeze(1)                # [BN, 1, D]
#
#         # patch token = visual token + 2D position encoding + magnification embedding
#         x = patch_tokens + pos_emb + mag_emb
#
#         # CLS token 也加 magnification embedding
#         cls = self.cls_token.expand(BN, -1, -1) + mag_emb           # [BN, 1, D]
#         x = torch.cat([cls, x], dim=1)                              # [BN, 1+M, D]
#
#         # Transformer 的 src_key_padding_mask 中，True 表示“忽略”
#         cls_valid = torch.ones(BN, 1, dtype=torch.bool, device=x.device)
#         valid_mask = torch.cat([cls_valid, patch_mask], dim=1)
#         key_padding_mask = ~valid_mask
#
#         x = self.encoder(x, src_key_padding_mask=key_padding_mask)
#         frame_token = self.norm(x[:, 0])  # 取 CLS token
#         return frame_token
#
#
# # ============================================================
# # 4. metadata encoder + reliability gate
# # ============================================================
#
# class FrameMetadataEncoder(nn.Module):
#     """
#     frame-level metadata 编码器。
#
#     当前只使用 3 维 metadata：
#         [log1p(sharpness), log1p(tissue_area), norm_pos]
#
#     说明
#     ----
#     - 不再使用任何 gap / index difference
#     - norm_pos 表示 retained packet 内的绝对顺序位置
#     """
#
#     def __init__(
#         self,
#         in_dim: int = 3,
#         out_dim: int = 256,
#         dropout: float = 0.1,
#     ) -> None:
#         super().__init__()
#         self.mlp = MLP(
#             in_dim=in_dim,
#             hidden_dim=out_dim,
#             out_dim=out_dim,
#             dropout=dropout,
#             use_layernorm=True,
#         )
#
#     def forward(self, raw_meta: torch.Tensor) -> torch.Tensor:
#         return self.mlp(raw_meta)
#
#
# class ReliabilityGate(nn.Module):
#     """
#     利用 metadata + magnification 生成 frame reliability gate。
#
#     设计思想
#     --------
#     sharpness / tissue_area / norm_pos 以及 magnification，
#     不直接决定“癌 / 非癌”，而是决定：
#         “这个 frame 的表征有多可信”
#     """
#
#     def __init__(
#         self,
#         token_dim: int,
#         meta_dim: int,
#         num_scales: int,
#         dropout: float = 0.1,
#     ) -> None:
#         super().__init__()
#         self.mag_embed = nn.Embedding(num_scales, meta_dim)
#
#         gate_in_dim = meta_dim + meta_dim
#         self.gate_mlp = MLP(
#             in_dim=gate_in_dim,
#             hidden_dim=meta_dim,
#             out_dim=1,
#             dropout=dropout,
#         )
#         self.residual_proj = MLP(
#             in_dim=gate_in_dim,
#             hidden_dim=token_dim,
#             out_dim=token_dim,
#             dropout=dropout,
#             use_layernorm=True,
#         )
#
#     def forward(
#         self,
#         frame_tokens: torch.Tensor,  # [B, N, D]
#         meta_emb: torch.Tensor,      # [B, N, meta_dim]
#         mag_id: torch.Tensor,        # [B, N]
#         frame_mask: torch.Tensor,    # [B, N]
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         mag_emb = self.mag_embed(mag_id)               # [B, N, meta_dim]
#         gate_input = torch.cat([meta_emb, mag_emb], dim=-1)
#
#         gate = torch.sigmoid(self.gate_mlp(gate_input))    # [B, N, 1]
#         residual = self.residual_proj(gate_input)          # [B, N, D]
#
#         out = frame_tokens + gate * residual
#         out = out * frame_mask.unsqueeze(-1).float()
#
#         return out, gate.squeeze(-1)
#
#
# # ============================================================
# # 5. packet-level retained-order / magnification-aware encoder
# # ============================================================
#
# class OrderMagAwareSelfAttention(nn.Module):
#     """
#     packet-level 多头自注意力。
#
#     与旧版的差别
#     ------------
#     不再使用任何 gap bias。
#     attention logits 中只显式加入：
#     1) retained-order relative bias
#     2) magnification pair bias
#
#     解释
#     ----
#     - retained-order relative bias：表示在保留序列中“前后邻近”
#     - magnification pair bias：表示不同倍率之间的交互模式
#     """
#
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int,
#         num_order_buckets: int,
#         num_scales: int,
#         attn_dropout: float = 0.1,
#         proj_dropout: float = 0.1,
#     ) -> None:
#         super().__init__()
#         assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
#
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3)
#         self.out_proj = nn.Linear(dim, dim)
#
#         self.order_bias = nn.Embedding(num_order_buckets, num_heads)
#         self.mag_pair_bias = nn.Parameter(
#             torch.zeros(num_scales, num_scales, num_heads)
#         )
#
#         self.attn_dropout = nn.Dropout(attn_dropout)
#         self.proj_dropout = nn.Dropout(proj_dropout)
#
#         nn.init.trunc_normal_(self.order_bias.weight, std=0.02)
#         nn.init.zeros_(self.mag_pair_bias)
#
#     def forward(
#         self,
#         x: torch.Tensor,          # [B, N, D]
#         order_rank: torch.Tensor, # [B, N]
#         mag_id: torch.Tensor,     # [B, N]
#         mask: torch.Tensor,       # [B, N], True=有效
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         B, N, D = x.shape
#
#         qkv = self.qkv(x)  # [B, N, 3D]
#         qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
#         qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, Hd]
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
#
#         # --------------------------------------------------
#         # 1) retained-order relative bias
#         # --------------------------------------------------
#         order_dist = (order_rank.unsqueeze(2) - order_rank.unsqueeze(1)).abs()  # [B, N, N]
#         order_bucket = bucketize_order_distance(
#             order_dist,
#             self.order_bias.num_embeddings
#         )  # [B, N, N]
#
#         order_bias = self.order_bias(order_bucket)  # [B, N, N, H]
#         order_bias = order_bias.permute(0, 3, 1, 2)  # [B, H, N, N]
#
#         # --------------------------------------------------
#         # 2) magnification pair bias
#         # --------------------------------------------------
#         mag_i = mag_id.unsqueeze(2).expand(B, N, N)
#         mag_j = mag_id.unsqueeze(1).expand(B, N, N)
#         mag_bias = self.mag_pair_bias[mag_i, mag_j]  # [B, N, N, H]
#         mag_bias = mag_bias.permute(0, 3, 1, 2)      # [B, H, N, N]
#
#         attn_logits = attn_logits + order_bias + mag_bias
#
#         # key mask：无效 key 不允许被关注
#         key_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
#         attn_logits = attn_logits.masked_fill(~key_mask, -1e9)
#
#         attn = torch.softmax(attn_logits, dim=-1)
#         attn = self.attn_dropout(attn)
#
#         out = torch.matmul(attn, v)  # [B, H, N, Hd]
#         out = out.transpose(1, 2).contiguous().reshape(B, N, D)
#         out = self.out_proj(out)
#         out = self.proj_dropout(out)
#
#         # 无效 query 输出清零
#         out = out * mask.unsqueeze(-1).float()
#         return out, attn
#
#
# class OrderMagAwareBlock(nn.Module):
#     """
#     packet-level Transformer block：
#         LN -> order/mag aware attention -> residual
#         LN -> FFN                      -> residual
#     """
#
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int,
#         num_order_buckets: int,
#         num_scales: int,
#         mlp_ratio: float = 4.0,
#         dropout: float = 0.1,
#     ) -> None:
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = OrderMagAwareSelfAttention(
#             dim=dim,
#             num_heads=num_heads,
#             num_order_buckets=num_order_buckets,
#             num_scales=num_scales,
#             attn_dropout=dropout,
#             proj_dropout=dropout,
#         )
#         self.norm2 = nn.LayerNorm(dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, int(dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(int(dim * mlp_ratio), dim),
#             nn.Dropout(dropout),
#         )
#
#     def forward(
#         self,
#         x: torch.Tensor,
#         order_rank: torch.Tensor,
#         mag_id: torch.Tensor,
#         mask: torch.Tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         attn_out, attn = self.attn(
#             self.norm1(x),
#             order_rank=order_rank,
#             mag_id=mag_id,
#             mask=mask,
#         )
#         x = x + attn_out
#         x = x + self.ffn(self.norm2(x))
#         x = x * mask.unsqueeze(-1).float()
#         return x, attn
#
#
# # ============================================================
# # 6. pooling 模块
# # ============================================================
#
# class ScaleAwarePool(nn.Module):
#     """
#     先按 magnification 分组，再做组内 attention pooling。
#     每个倍率得到一个 scale token。
#
#     输出
#     ----
#     scale_tokens: [B, S, D]
#     scale_mask  : [B, S]
#     """
#
#     def __init__(self, dim: int, num_scales: int) -> None:
#         super().__init__()
#         self.num_scales = num_scales
#         self.scale_queries = nn.Parameter(torch.randn(num_scales, dim) * 0.02)
#
#     def forward(
#         self,
#         frame_tokens: torch.Tensor,  # [B, N, D]
#         mag_id: torch.Tensor,        # [B, N]
#         frame_mask: torch.Tensor,    # [B, N]
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         B, N, D = frame_tokens.shape
#         scale_tokens = []
#         scale_mask = []
#
#         for s in range(self.num_scales):
#             cur_mask = frame_mask & (mag_id == s)           # [B, N]
#             scale_mask.append(cur_mask.any(dim=1))          # [B]
#
#             query = self.scale_queries[s].view(1, 1, D)     # [1, 1, D]
#             logits = (frame_tokens * query).sum(dim=-1) / math.sqrt(D)  # [B, N]
#
#             alpha = masked_softmax(logits, cur_mask, dim=1)
#             token = torch.sum(alpha.unsqueeze(-1) * frame_tokens, dim=1)  # [B, D]
#             scale_tokens.append(token)
#
#         scale_tokens = torch.stack(scale_tokens, dim=1)  # [B, S, D]
#         scale_mask = torch.stack(scale_mask, dim=1)      # [B, S]
#         return scale_tokens, scale_mask
#
#
# class TopKEvidencePool(nn.Module):
#     """
#     evidence pooling：
#     偏向找少数最强癌证据 frame。
#     """
#
#     def __init__(self, dim: int, topk_ratio: float = 0.2) -> None:
#         super().__init__()
#         self.score_head = nn.Linear(dim, 1)
#         self.topk_ratio = topk_ratio
#
#     def forward(
#         self,
#         x: torch.Tensor,        # [B, N, D]
#         mask: torch.Tensor,     # [B, N]
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         B, N, D = x.shape
#         raw_scores = self.score_head(x).squeeze(-1)           # [B, N]
#         raw_scores = raw_scores.masked_fill(~mask, -1e9)
#
#         pooled = []
#         all_alpha = []
#
#         for b in range(B):
#             valid_idx = torch.where(mask[b])[0]
#             if len(valid_idx) == 0:
#                 pooled.append(torch.zeros(D, device=x.device, dtype=x.dtype))
#                 all_alpha.append(torch.zeros(N, device=x.device, dtype=x.dtype))
#                 continue
#
#             k = max(1, math.ceil(len(valid_idx) * self.topk_ratio))
#             scores_b = raw_scores[b, valid_idx]
#             topk_scores, topk_pos = torch.topk(scores_b, k=k, dim=0)
#             topk_idx = valid_idx[topk_pos]
#
#             alpha = torch.softmax(topk_scores, dim=0)
#             vec = torch.sum(alpha.unsqueeze(-1) * x[b, topk_idx], dim=0)
#
#             full_alpha = torch.zeros(N, device=x.device, dtype=x.dtype)
#             full_alpha[topk_idx] = alpha
#
#             pooled.append(vec)
#             all_alpha.append(full_alpha)
#
#         pooled = torch.stack(pooled, dim=0)       # [B, D]
#         all_alpha = torch.stack(all_alpha, dim=0) # [B, N]
#         return pooled, raw_scores, all_alpha
#
#
# class GatedAttentionPool(nn.Module):
#     """
#     context pooling：
#     用 gated attention 建模整体上下文。
#     """
#
#     def __init__(self, dim: int, hidden_dim: Optional[int] = None) -> None:
#         super().__init__()
#         hidden_dim = hidden_dim or dim
#         self.v = nn.Linear(dim, hidden_dim)
#         self.u = nn.Linear(dim, hidden_dim)
#         self.w = nn.Linear(hidden_dim, 1)
#
#     def forward(
#         self,
#         x: torch.Tensor,        # [B, L, D]
#         mask: torch.Tensor,     # [B, L]
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         a = torch.tanh(self.v(x)) * torch.sigmoid(self.u(x))
#         logits = self.w(a).squeeze(-1)                        # [B, L]
#         alpha = masked_softmax(logits, mask, dim=1)
#         pooled = torch.sum(alpha.unsqueeze(-1) * x, dim=1)   # [B, D]
#         return pooled, logits, alpha
#
#
# # ============================================================
# # 7. 主模型
# # ============================================================
#
# class MagnificationAwareOrderedPacketMIL(nn.Module):
#     """
#     主模型：MHO-PMIL
#
#     流程
#     ----
#     1) patch backbone 编 patch
#     2) frame-level spatial encoder 聚合 patch -> frame token
#     3) metadata encoder + reliability gate
#     4) packet-level retained-order / magnification-aware blocks
#     5) scale-aware pooling 产生每个倍率的 scale token
#     6) cross-scale encoder 建模倍率级上下文
#     7) evidence/context pooling
#     8) classifier 输出 bag-level logit
#     """
#
#     def __init__(
#         self,
#         backbone_name: str = "resnet50",
#         patch_embed_dim: int = 512,
#         model_dim: int = 512,
#         meta_dim: int = 256,
#         num_scales: int = 4,
#         frame_encoder_layers: int = 2,
#         frame_encoder_heads: int = 8,
#         packet_layers: int = 2,
#         packet_heads: int = 8,
#         num_order_buckets: int = 16,
#         scale_encoder_layers: int = 1,
#         dropout: float = 0.1,
#         topk_ratio: float = 0.2,
#         pretrained_backbone: bool = True,
#         patch_encode_chunk_size: int = 2048,
#     ) -> None:
#         super().__init__()
#
#         self.num_scales = num_scales
#         self.model_dim = model_dim
#         self.patch_encode_chunk_size = patch_encode_chunk_size
#
#         # ----------------------------
#         # 1) patch backbone
#         # ----------------------------
#         self.patch_backbone = PatchBackbone(
#             backbone_name=backbone_name,
#             pretrained=pretrained_backbone,
#             out_dim=patch_embed_dim,
#         )
#         self.patch_proj = (
#             nn.Identity()
#             if patch_embed_dim == model_dim
#             else nn.Linear(patch_embed_dim, model_dim)
#         )
#
#         # ----------------------------
#         # 2) frame-level spatial encoder
#         # ----------------------------
#         self.frame_encoder = SpatialFrameEncoder(
#             dim=model_dim,
#             num_scales=num_scales,
#             num_layers=frame_encoder_layers,
#             num_heads=frame_encoder_heads,
#             dropout=dropout,
#         )
#
#         # ----------------------------
#         # 3) metadata + reliability
#         # ----------------------------
#         self.meta_encoder = FrameMetadataEncoder(
#             in_dim=3,
#             out_dim=meta_dim,
#             dropout=dropout,
#         )
#         self.meta_to_model = MLP(
#             in_dim=meta_dim,
#             hidden_dim=model_dim,
#             out_dim=model_dim,
#             dropout=dropout,
#             use_layernorm=True,
#         )
#         self.frame_mag_embed = nn.Embedding(num_scales, model_dim)
#
#         # retained absolute order position embedding
#         self.order_pos_proj = MLP(
#             in_dim=1,
#             hidden_dim=model_dim,
#             out_dim=model_dim,
#             dropout=dropout,
#             use_layernorm=True,
#         )
#
#         self.reliability_gate = ReliabilityGate(
#             token_dim=model_dim,
#             meta_dim=meta_dim,
#             num_scales=num_scales,
#             dropout=dropout,
#         )
#
#         # ----------------------------
#         # 4) packet-level ordered encoder
#         # ----------------------------
#         self.packet_blocks = nn.ModuleList([
#             OrderMagAwareBlock(
#                 dim=model_dim,
#                 num_heads=packet_heads,
#                 num_order_buckets=num_order_buckets,
#                 num_scales=num_scales,
#                 dropout=dropout,
#             )
#             for _ in range(packet_layers)
#         ])
#         self.packet_norm = nn.LayerNorm(model_dim)
#
#         # ----------------------------
#         # 5) scale-aware pooling + cross-scale encoder
#         # ----------------------------
#         self.scale_pool = ScaleAwarePool(dim=model_dim, num_scales=num_scales)
#
#         scale_encoder_layer = nn.TransformerEncoderLayer(
#             d_model=model_dim,
#             nhead=min(packet_heads, 8),
#             dim_feedforward=model_dim * 4,
#             dropout=dropout,
#             activation="gelu",
#             batch_first=True,
#             norm_first=True,
#         )
#         self.scale_encoder = nn.TransformerEncoder(
#             scale_encoder_layer,
#             num_layers=scale_encoder_layers,
#         )
#
#         # ----------------------------
#         # 6) dual pooling
#         # ----------------------------
#         self.evidence_pool = TopKEvidencePool(
#             dim=model_dim,
#             topk_ratio=topk_ratio,
#         )
#         self.context_pool = GatedAttentionPool(dim=model_dim)
#
#         # ----------------------------
#         # 7) classifier
#         # ----------------------------
#         classifier_in_dim = model_dim * 3
#         self.classifier = nn.Sequential(
#             nn.LayerNorm(classifier_in_dim),
#             nn.Linear(classifier_in_dim, model_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(model_dim, 1),
#         )
#
#     def _encode_valid_patches(
#         self,
#         patches: torch.Tensor,      # [B, N, M, 3, H, W]
#         patch_mask: torch.Tensor,   # [B, N, M]
#     ) -> torch.Tensor:
#         """
#         只对有效 patch 做 backbone 编码，避免把大量 padding patch 送进 CNN。
#
#         返回
#         ----
#         patch_tokens: Tensor[B, N, M, D]
#         """
#         B, N, M, C, H, W = patches.shape
#         total = B * N * M
#
#         flat_patches = patches.reshape(total, C, H, W)
#         flat_mask = patch_mask.reshape(total)
#
#         flat_tokens = torch.zeros(
#             total,
#             self.model_dim,
#             device=patches.device,
#             dtype=patches.dtype,
#         )
#
#         valid_idx = torch.where(flat_mask)[0]
#         if len(valid_idx) > 0:
#             valid_patches = flat_patches[valid_idx]
#             encoded_chunks = []
#
#             for start in range(0, len(valid_idx), self.patch_encode_chunk_size):
#                 end = min(start + self.patch_encode_chunk_size, len(valid_idx))
#                 chunk = valid_patches[start:end]
#                 feat = self.patch_backbone(chunk)
#                 feat = self.patch_proj(feat)
#                 encoded_chunks.append(feat)
#
#             valid_tokens = torch.cat(encoded_chunks, dim=0)
#             flat_tokens[valid_idx] = valid_tokens
#
#         patch_tokens = flat_tokens.view(B, N, M, self.model_dim)
#         return patch_tokens
#
#     def _build_raw_meta(
#         self,
#         sharpness: torch.Tensor,   # [B, N]
#         tissue_area: torch.Tensor, # [B, N]
#         frame_mask: torch.Tensor,  # [B, N]
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         组装 raw metadata：
#             [log1p(sharpness), log1p(tissue_area), norm_pos]
#
#         同时返回：
#             order_rank: retained 顺序编号，用于 packet-level relative bias
#         """
#         order_rank, norm_pos = compute_retained_order_features(frame_mask)
#
#         raw_meta = torch.stack(
#             [
#                 torch.log1p(sharpness.clamp_min(0.0)),
#                 torch.log1p(tissue_area.clamp_min(0.0)),
#                 norm_pos,
#             ],
#             dim=-1,
#         )  # [B, N, 3]
#
#         raw_meta = raw_meta * frame_mask.unsqueeze(-1).float()
#         return raw_meta, order_rank
#
#     def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """
#         返回一个字典，包含：
#         - logits
#         - frame_tokens
#         - scale_tokens
#         - reliability_gate
#         - frame_scores
#         - evidence_alpha
#         - context_alpha
#         - attn_maps
#         """
#         patches = batch["patches"]               # [B, N, M, 3, H, W]
#         patch_coords = batch["patch_coords"]     # [B, N, M, 2]
#         patch_mask = batch["patch_mask"]         # [B, N, M]
#         frame_mask = batch["frame_mask"]         # [B, N]
#         mag_id = batch["magnification"]          # [B, N]
#         sharpness = batch["sharpness"]           # [B, N]
#         tissue_area = batch["tissue_area"]       # [B, N]
#
#         B, N, M, C, H, W = patches.shape
#
#         # ====================================================
#         # Step 1. patch backbone
#         # ====================================================
#         patch_tokens = self._encode_valid_patches(patches, patch_mask)  # [B, N, M, D]
#
#         # reshape 成 frame encoder 需要的形式
#         patch_tokens_bn = patch_tokens.view(B * N, M, self.model_dim)
#         patch_coords_bn = patch_coords.view(B * N, M, 2)
#         patch_mask_bn = patch_mask.view(B * N, M)
#         mag_id_bn = mag_id.view(B * N)
#
#         # ====================================================
#         # Step 2. frame-level spatial aggregation
#         # ====================================================
#         frame_tokens_bn = self.frame_encoder(
#             patch_tokens=patch_tokens_bn,
#             patch_coords=patch_coords_bn,
#             patch_mask=patch_mask_bn,
#             mag_id=mag_id_bn,
#         )  # [B*N, D]
#
#         frame_tokens = frame_tokens_bn.view(B, N, self.model_dim)
#         frame_tokens = frame_tokens * frame_mask.unsqueeze(-1).float()
#
#         # ====================================================
#         # Step 3. metadata + magnification + retained order pos
#         # ====================================================
#         raw_meta, order_rank = self._build_raw_meta(
#             sharpness=sharpness,
#             tissue_area=tissue_area,
#             frame_mask=frame_mask,
#         )  # [B, N, 3], [B, N]
#
#         meta_emb = self.meta_encoder(raw_meta)            # [B, N, meta_dim]
#         meta_token = self.meta_to_model(meta_emb)         # [B, N, D]
#         mag_token = self.frame_mag_embed(mag_id)          # [B, N, D]
#
#         norm_pos = raw_meta[..., 2:3]                     # [B, N, 1]
#         order_pos_token = self.order_pos_proj(norm_pos)   # [B, N, D]
#
#         frame_tokens = frame_tokens + meta_token + mag_token + order_pos_token
#         frame_tokens = frame_tokens * frame_mask.unsqueeze(-1).float()
#
#         frame_tokens, reliability_gate = self.reliability_gate(
#             frame_tokens=frame_tokens,
#             meta_emb=meta_emb,
#             mag_id=mag_id,
#             frame_mask=frame_mask,
#         )
#
#         # ====================================================
#         # Step 4. packet-level retained-order / magnification-aware encoding
#         # ====================================================
#         x = frame_tokens
#         attn_maps = []
#
#         for block in self.packet_blocks:
#             x, attn = block(
#                 x=x,
#                 order_rank=order_rank,
#                 mag_id=mag_id,
#                 mask=frame_mask,
#             )
#             attn_maps.append(attn)
#
#         x = self.packet_norm(x)
#         x = x * frame_mask.unsqueeze(-1).float()   # [B, N, D]
#
#         # ====================================================
#         # Step 5. scale-aware pooling + cross-scale encoder
#         # ====================================================
#         scale_tokens, scale_mask = self.scale_pool(
#             frame_tokens=x,
#             mag_id=mag_id,
#             frame_mask=frame_mask,
#         )  # [B, S, D], [B, S]
#
#         scale_tokens = self.scale_encoder(
#             scale_tokens,
#             src_key_padding_mask=~scale_mask,
#         )
#         scale_tokens = scale_tokens * scale_mask.unsqueeze(-1).float()
#
#         # ====================================================
#         # Step 6. dual pooling
#         # ====================================================
#         evidence_vec, frame_scores, evidence_alpha = self.evidence_pool(
#             x=x,
#             mask=frame_mask,
#         )  # [B, D], [B, N], [B, N]
#
#         context_vec, context_logits, context_alpha = self.context_pool(
#             x=scale_tokens,
#             mask=scale_mask,
#         )  # [B, D], [B, S], [B, S]
#
#         scale_alpha = scale_mask.float()
#         scale_alpha = scale_alpha / scale_alpha.sum(dim=1, keepdim=True).clamp_min(1.0)
#         global_scale_vec = torch.sum(scale_alpha.unsqueeze(-1) * scale_tokens, dim=1)
#
#         # ====================================================
#         # Step 7. classifier
#         # ====================================================
#         fused = torch.cat([evidence_vec, context_vec, global_scale_vec], dim=-1)
#         logits = self.classifier(fused).squeeze(-1)  # [B]
#
#         return {
#             "logits": logits,
#             "frame_tokens": x,
#             "scale_tokens": scale_tokens,
#             "reliability_gate": reliability_gate,
#             "frame_scores": frame_scores,
#             "evidence_alpha": evidence_alpha,
#             "context_alpha": context_alpha,
#             "attn_maps": attn_maps,
#         }
#
#
# # ============================================================
# # 8. LightningModule
# # ============================================================
#
# class LitMagnificationAwareOrderedPacketMIL(pl.LightningModule):
#     """
#     PyTorch Lightning 封装。
#
#     当前损失
#     --------
#     仅使用 bag-level BCEWithLogitsLoss。
#     不再使用任何基于 gap 的 consistency loss。
#     """
#
#     def __init__(
#         self,
#         model: MagnificationAwareOrderedPacketMIL,
#         lr: float = 1e-4,
#         weight_decay: float = 1e-4,
#         pos_weight: Optional[float] = None,
#         scheduler_tmax: int = 50,
#     ) -> None:
#         super().__init__()
#         self.model = model
#
#         self.lr = lr
#         self.weight_decay = weight_decay
#         self.scheduler_tmax = scheduler_tmax
#
#         if pos_weight is None:
#             self.criterion = nn.BCEWithLogitsLoss()
#         else:
#             self.criterion = nn.BCEWithLogitsLoss(
#                 pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
#             )
#
#         if _HAS_TORCHMETRICS:
#             self.train_acc = BinaryAccuracy()
#             self.val_acc = BinaryAccuracy()
#             self.test_acc = BinaryAccuracy()
#
#             self.val_auc = BinaryAUROC()
#             self.test_auc = BinaryAUROC()
#
#             self.val_f1 = BinaryF1Score()
#             self.test_f1 = BinaryF1Score()
#
#         self.save_hyperparameters(ignore=["model"])
#
#     def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         return self.model(batch)
#
#     def _shared_step(
#         self,
#         batch: Dict[str, torch.Tensor],
#         stage: str,
#     ) -> torch.Tensor:
#         out = self(batch)
#         logits = out["logits"]
#
#         label = batch["label"].float()
#         loss = self.criterion(logits, label)
#
#         probs = torch.sigmoid(logits)
#
#         self.log(f"{stage}/loss", loss, prog_bar=(stage != "train"), on_step=False, on_epoch=True)
#
#         if _HAS_TORCHMETRICS:
#             if stage == "train":
#                 self.train_acc.update(probs, label.int())
#                 self.log(f"{stage}/acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
#
#             elif stage == "val":
#                 self.val_acc.update(probs, label.int())
#                 self.val_auc.update(probs, label.int())
#                 self.val_f1.update(probs, label.int())
#
#                 self.log(f"{stage}/acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
#                 self.log(f"{stage}/auc", self.val_auc, prog_bar=True, on_step=False, on_epoch=True)
#                 self.log(f"{stage}/f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True)
#
#             elif stage == "test":
#                 self.test_acc.update(probs, label.int())
#                 self.test_auc.update(probs, label.int())
#                 self.test_f1.update(probs, label.int())
#
#                 self.log(f"{stage}/acc", self.test_acc, prog_bar=True, on_step=False, on_epoch=True)
#                 self.log(f"{stage}/auc", self.test_auc, prog_bar=True, on_step=False, on_epoch=True)
#                 self.log(f"{stage}/f1", self.test_f1, prog_bar=True, on_step=False, on_epoch=True)
#
#         return loss
#
#     def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
#         return self._shared_step(batch, stage="train")
#
#     def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
#         self._shared_step(batch, stage="val")
#
#     def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
#         self._shared_step(batch, stage="test")
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(
#             self.parameters(),
#             lr=self.lr,
#             weight_decay=self.weight_decay,
#         )
#
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer,
#             T_max=self.scheduler_tmax,
#         )
#
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,
#                 "interval": "epoch",
#                 "frequency": 1,
#             },
#         }
#
#
# # ============================================================
# # 9. collate_fn
# # ============================================================
#
# def _normalize_single_sample(
#     sample: Dict[str, Any],
# ) -> Dict[str, torch.Tensor]:
#     """
#     把单个 sample 规范化成统一张量格式，并按 frame_index 升序排序。
#
#     支持两种 patches 组织方式：
#
#     方式 A：
#         sample["patches"] = Tensor[Ni, Mi, 3, H, W]
#         sample["patch_coords"] = Tensor[Ni, Mi, 2]
#         sample["patch_mask"] = BoolTensor[Ni, Mi]  (可选)
#
#     方式 B：
#         sample["patches"] = List[Tensor[Mi_i, 3, H, W]]，长度为 Ni
#         sample["patch_coords"] = List[Tensor[Mi_i, 2]]，长度为 Ni
#         sample["patch_mask"] 可省略
#
#     返回统一格式：
#         {
#             "patches": Tensor[Ni, Mi_max, 3, H, W],
#             "patch_coords": Tensor[Ni, Mi_max, 2],
#             "patch_mask": BoolTensor[Ni, Mi_max],
#             "frame_index": LongTensor[Ni],
#             "magnification": LongTensor[Ni],
#             "sharpness": FloatTensor[Ni],
#             "tissue_area": FloatTensor[Ni],
#             "label": FloatTensor[]
#         }
#     """
#     patches = sample["patches"]
#     patch_coords = sample["patch_coords"]
#
#     frame_index = torch.as_tensor(sample["frame_index"], dtype=torch.long)
#     magnification = torch.as_tensor(sample["magnification"], dtype=torch.long)
#     sharpness = torch.as_tensor(sample["sharpness"], dtype=torch.float32)
#     tissue_area = torch.as_tensor(sample["tissue_area"], dtype=torch.float32)
#     label = torch.as_tensor(sample["label"], dtype=torch.float32).reshape(())
#
#     # --------------------------------------------------------
#     # 情况 1：patches 已经是 Tensor[Ni, Mi, 3, H, W]
#     # --------------------------------------------------------
#     if isinstance(patches, torch.Tensor):
#         if patches.ndim != 5:
#             raise ValueError(
#                 f"当 sample['patches'] 为 Tensor 时，期望 shape=[Ni, Mi, 3, H, W]，实际 ndim={patches.ndim}"
#             )
#
#         Ni, Mi, C, H, W = patches.shape
#         patch_coords = torch.as_tensor(patch_coords, dtype=torch.float32)
#
#         if patch_coords.shape != (Ni, Mi, 2):
#             raise ValueError(
#                 f"patch_coords shape 不匹配，期望={(Ni, Mi, 2)}，实际={tuple(patch_coords.shape)}"
#             )
#
#         if "patch_mask" in sample and sample["patch_mask"] is not None:
#             patch_mask = torch.as_tensor(sample["patch_mask"], dtype=torch.bool)
#             if patch_mask.shape != (Ni, Mi):
#                 raise ValueError(
#                     f"patch_mask shape 不匹配，期望={(Ni, Mi)}，实际={tuple(patch_mask.shape)}"
#                 )
#         else:
#             patch_mask = torch.ones(Ni, Mi, dtype=torch.bool)
#
#     # --------------------------------------------------------
#     # 情况 2：patches 是 List[Tensor[Mi_i, 3, H, W]]
#     # --------------------------------------------------------
#     elif isinstance(patches, (list, tuple)):
#         if len(patches) == 0:
#             raise ValueError("sample['patches'] 为空 list，无法构造 sample")
#
#         if not isinstance(patch_coords, (list, tuple)):
#             raise ValueError("当 sample['patches'] 是 list 时，sample['patch_coords'] 也必须是 list/tuple")
#
#         Ni = len(patches)
#         if len(patch_coords) != Ni:
#             raise ValueError(f"patches 长度={Ni}，但 patch_coords 长度={len(patch_coords)}，不一致")
#
#         # 找到单个 sample 内的最大 patch 数
#         Mi = max(p.shape[0] for p in patches)
#         C, H, W = patches[0].shape[1:]
#
#         patches_padded = torch.zeros(Ni, Mi, C, H, W, dtype=patches[0].dtype)
#         coords_padded = torch.zeros(Ni, Mi, 2, dtype=torch.float32)
#         mask_padded = torch.zeros(Ni, Mi, dtype=torch.bool)
#
#         for i, (p_i, c_i) in enumerate(zip(patches, patch_coords)):
#             p_i = torch.as_tensor(p_i)
#             c_i = torch.as_tensor(c_i, dtype=torch.float32)
#
#             if p_i.ndim != 4:
#                 raise ValueError(f"第 {i} 个 frame 的 patch tensor 期望 ndim=4，实际={p_i.ndim}")
#             if c_i.ndim != 2 or c_i.shape[1] != 2:
#                 raise ValueError(f"第 {i} 个 frame 的 patch_coords 期望 shape=[Mi_i, 2]，实际={tuple(c_i.shape)}")
#
#             Mi_i = p_i.shape[0]
#             if c_i.shape[0] != Mi_i:
#                 raise ValueError(f"第 {i} 个 frame 的 patch 数与坐标数不一致")
#
#             patches_padded[i, :Mi_i] = p_i
#             coords_padded[i, :Mi_i] = c_i
#             mask_padded[i, :Mi_i] = True
#
#         patches = patches_padded
#         patch_coords = coords_padded
#         patch_mask = mask_padded
#
#     else:
#         raise TypeError("sample['patches'] 必须是 Tensor 或 list/tuple")
#
#     # 基本长度检查
#     Ni = patches.shape[0]
#     if frame_index.shape[0] != Ni:
#         raise ValueError(f"frame_index 长度={frame_index.shape[0]} 与 frame 数={Ni} 不一致")
#     if magnification.shape[0] != Ni:
#         raise ValueError(f"magnification 长度={magnification.shape[0]} 与 frame 数={Ni} 不一致")
#     if sharpness.shape[0] != Ni:
#         raise ValueError(f"sharpness 长度={sharpness.shape[0]} 与 frame 数={Ni} 不一致")
#     if tissue_area.shape[0] != Ni:
#         raise ValueError(f"tissue_area 长度={tissue_area.shape[0]} 与 frame 数={Ni} 不一致")
#
#     # --------------------------------------------------------
#     # 关键：按 frame_index 升序排序
#     # --------------------------------------------------------
#     order = torch.argsort(frame_index)
#     patches = patches[order]
#     patch_coords = patch_coords[order]
#     patch_mask = patch_mask[order]
#     frame_index = frame_index[order]
#     magnification = magnification[order]
#     sharpness = sharpness[order]
#     tissue_area = tissue_area[order]
#
#     return {
#         "patches": patches,
#         "patch_coords": patch_coords,
#         "patch_mask": patch_mask,
#         "frame_index": frame_index,
#         "magnification": magnification,
#         "sharpness": sharpness,
#         "tissue_area": tissue_area,
#         "label": label,
#     }
#
#
# def packet_mil_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
#     """
#     将一个 batch 的变长 packet 样本整理成可直接输入模型的张量。
#
#     这个 collate_fn 做了两件关键事：
#     1) 每个 sample 内先按 frame_index 升序排序
#     2) 再对 batch 内不同样本做 padding
#
#     输出
#     ----
#     {
#         "patches": Tensor[B, N, M, 3, H, W],
#         "patch_coords": Tensor[B, N, M, 2],
#         "patch_mask": BoolTensor[B, N, M],
#         "frame_mask": BoolTensor[B, N],
#         "frame_index": LongTensor[B, N],   # 仅保留，模型不做 gap 使用
#         "magnification": LongTensor[B, N],
#         "sharpness": FloatTensor[B, N],
#         "tissue_area": FloatTensor[B, N],
#         "label": FloatTensor[B],
#     }
#     """
#     if len(batch) == 0:
#         raise ValueError("batch 为空")
#
#     # 先把每个 sample 规范化并排序
#     batch = [_normalize_single_sample(sample) for sample in batch]
#
#     batch_size = len(batch)
#     max_num_frames = max(sample["patches"].shape[0] for sample in batch)
#     max_num_patches = max(sample["patches"].shape[1] for sample in batch)
#
#     _, _, C, H, W = batch[0]["patches"].shape
#
#     patches = torch.zeros(
#         batch_size, max_num_frames, max_num_patches, C, H, W,
#         dtype=batch[0]["patches"].dtype,
#     )
#     patch_coords = torch.zeros(
#         batch_size, max_num_frames, max_num_patches, 2,
#         dtype=torch.float32,
#     )
#     patch_mask = torch.zeros(
#         batch_size, max_num_frames, max_num_patches,
#         dtype=torch.bool,
#     )
#     frame_mask = torch.zeros(
#         batch_size, max_num_frames,
#         dtype=torch.bool,
#     )
#
#     frame_index = torch.zeros(
#         batch_size, max_num_frames,
#         dtype=torch.long,
#     )
#     magnification = torch.zeros(
#         batch_size, max_num_frames,
#         dtype=torch.long,
#     )
#     sharpness = torch.zeros(
#         batch_size, max_num_frames,
#         dtype=torch.float32,
#     )
#     tissue_area = torch.zeros(
#         batch_size, max_num_frames,
#         dtype=torch.float32,
#     )
#     label = torch.zeros(
#         batch_size,
#         dtype=torch.float32,
#     )
#
#     for b_idx, sample in enumerate(batch):
#         sample_patches = sample["patches"]               # [Ni, Mi, 3, H, W]
#         sample_patch_coords = sample["patch_coords"]     # [Ni, Mi, 2]
#         sample_patch_mask = sample["patch_mask"]         # [Ni, Mi]
#         sample_frame_index = sample["frame_index"]       # [Ni]
#         sample_magnification = sample["magnification"]   # [Ni]
#         sample_sharpness = sample["sharpness"]           # [Ni]
#         sample_tissue_area = sample["tissue_area"]       # [Ni]
#         sample_label = sample["label"]
#
#         Ni, Mi = sample_patches.shape[:2]
#
#         patches[b_idx, :Ni, :Mi] = sample_patches
#         patch_coords[b_idx, :Ni, :Mi] = sample_patch_coords
#         patch_mask[b_idx, :Ni, :Mi] = sample_patch_mask
#
#         frame_mask[b_idx, :Ni] = True
#         frame_index[b_idx, :Ni] = sample_frame_index
#         magnification[b_idx, :Ni] = sample_magnification
#         sharpness[b_idx, :Ni] = sample_sharpness
#         tissue_area[b_idx, :Ni] = sample_tissue_area
#
#         label[b_idx] = sample_label.float()
#
#     return {
#         "patches": patches,
#         "patch_coords": patch_coords,
#         "patch_mask": patch_mask,
#         "frame_mask": frame_mask,
#         "frame_index": frame_index,       # 仅保留，不用于 gap
#         "magnification": magnification,
#         "sharpness": sharpness,
#         "tissue_area": tissue_area,
#         "label": label,
#     }
#
#
# # ============================================================
# # 10. 最小初始化示例
# # ============================================================
#
# if __name__ == "__main__":
#     """
#     这里只演示如何初始化模型。
#     """
#
#     model = MagnificationAwareOrderedPacketMIL(
#         backbone_name="resnet50",
#         patch_embed_dim=512,
#         model_dim=512,
#         meta_dim=256,
#         num_scales=4,                # 2x / 10x / 20x / 40x
#         frame_encoder_layers=2,
#         frame_encoder_heads=8,
#         packet_layers=2,
#         packet_heads=8,
#         num_order_buckets=16,
#         scale_encoder_layers=1,
#         dropout=0.1,
#         topk_ratio=0.2,
#         pretrained_backbone=True,
#         patch_encode_chunk_size=1024,
#     )
#
#     lit_model = LitMagnificationAwareOrderedPacketMIL(
#         model=model,
#         lr=1e-4,
#         weight_decay=1e-4,
#         pos_weight=None,
#         scheduler_tmax=50,
#     )
#
#     print(lit_model)