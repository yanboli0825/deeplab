from src.models.base_model import BaseModel
from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as torchF
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

        a = torchF.softmax(a, dim=-1)  # softmax over N
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
    def __init__(
        self,
        dim_in: int,
        num_heads: int = 8,
        atten_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        assert dim_in % num_heads == 0, "dim_in must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim_in // num_heads
        self.scale = self.head_dim ** -0.5
        hidden_dim = int(dim_in * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim_in)
        self.qkv = nn.Linear(dim_in, dim_in * 3)
        self.out_proj = nn.Linear(dim_in, dim_in)
        self.atten_dropout = nn.Dropout(p=atten_dropout)
        self.proj_dropout = nn.Dropout(p=proj_dropout)
        self.norm2 = nn.LayerNorm(dim_in)
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=proj_dropout),
            nn.Linear(hidden_dim, dim_in),
            nn.Dropout(p=proj_dropout),
        )

    def forward(self,x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, F, D = x.shape
        assert B == 1, "B must be 1"

        residual = x
        x = self.norm1(x)
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
        x = residual + out

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        return x, atten


def bucketize_order_distance(
    order_dist: torch.Tensor,
    num_buckets: int,
) -> torch.Tensor:
    bucket_ids = torch.floor(torch.log2(order_dist.float() + 1.0)).long()
    bucket_ids = bucket_ids.clamp(min=0, max=num_buckets - 1)
    return bucket_ids


class OrderMagAwareTransformerBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        num_heads: int = 8,
        num_order_buckets: int = 8,
        num_mag: int = 5,
        atten_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        assert dim_in % num_heads == 0, "dim_in must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim_in // num_heads
        self.scale = self.head_dim ** -0.5
        hidden_dim = int(dim_in * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim_in)
        self.qkv = nn.Linear(dim_in, dim_in * 3)
        self.out_proj = nn.Linear(dim_in, dim_in)
        self.atten_dropout = nn.Dropout(p=atten_dropout)
        self.proj_dropout = nn.Dropout(p=proj_dropout)
        self.norm2 = nn.LayerNorm(dim_in)
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=proj_dropout),
            nn.Linear(hidden_dim, dim_in),
            nn.Dropout(p=proj_dropout),
        )

        self.order_bias = nn.Embedding(num_order_buckets, num_heads)
        self.mag_pair_bias = nn.Parameter(torch.zeros(num_mag, num_mag, num_heads))
        nn.init.trunc_normal_(self.order_bias.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        order_rank: torch.Tensor,
        mag_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, F, D = x.shape
        assert B == 1, "B must be 1"

        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, F, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, F, Hd]

        atten_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, F, F]

        order_dist = (order_rank.unsqueeze(2) - order_rank.unsqueeze(1)).abs()
        order_bucket = bucketize_order_distance(order_dist, self.order_bias.num_embeddings)
        order_bias = self.order_bias(order_bucket).permute(0, 3, 1, 2)  # [B, H, F, F]

        mag_i = mag_id.unsqueeze(2).expand(B, F, F)
        mag_j = mag_id.unsqueeze(1).expand(B, F, F)
        mag_bias = self.mag_pair_bias[mag_i, mag_j].permute(0, 3, 1, 2)  # [B, H, F, F]

        atten_logits = atten_logits + order_bias + mag_bias

        atten = torch.softmax(atten_logits, dim=-1)
        atten = self.atten_dropout(atten)
        out = torch.matmul(atten, v)  # [B, H, F, Hd]
        out = out.transpose(1, 2).contiguous().reshape(B, F, D)
        out = self.out_proj(out)
        out = self.proj_dropout(out)
        x = residual + out

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        return x, atten



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


def masked_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    对带 mask 的 logits 做 softmax。

    参数
    ----
    logits: Tensor[..., L]
    mask:   BoolTensor[..., L]
    dim:    softmax 维度
    """
    masked_logits = logits.masked_fill(~mask, -1e9)
    probs = torch.softmax(masked_logits, dim=dim)
    probs = probs * mask.to(probs.dtype)
    probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(eps)
    return probs


class ScaleAwarePool(nn.Module):
    """
    默认 B=1 的 scale-aware pooling。

    作用
    ----
    把 frame token 按 magnification 分组，
    每个倍率聚成一个 scale token。

    输入
    ----
    frame_tokens: Tensor[B, N, D]   (默认 B=1)
    mag_id:       Tensor[B, N]
        magnification id，例如 4x->0, 10x->1, 20x->2, 40x->3

    输出
    ----
    scale_tokens: Tensor[B, S, D]
        S = num_scales，每个倍率一个 token

    scale_mask: Tensor[B, S]
        表示当前样本是否包含该倍率
    """

    def __init__(
        self,
        dim: int,
        num_scales: int,
    ) -> None:
        super().__init__()
        self.num_scales = num_scales

        # 每个倍率一个 query，用来从该倍率对应的 frame 中聚合出 scale token
        self.scale_queries = nn.Parameter(torch.randn(num_scales, dim) * 0.02)

    def forward(
        self,
        frame_tokens: torch.Tensor,  # [B, N, D], 默认 B=1
        mag_id: torch.Tensor,        # [B, N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if frame_tokens.ndim != 3:
            raise ValueError(f"frame_tokens 应为 [B, N, D]，实际={tuple(frame_tokens.shape)}")
        if mag_id.ndim != 2:
            raise ValueError(f"mag_id 应为 [B, N]，实际={tuple(mag_id.shape)}")

        B, N, D = frame_tokens.shape
        if B != 1:
            raise ValueError(
                f"当前简化版 ScaleAwarePool 默认 B=1，但实际 B={B}。"
            )

        scale_tokens = []
        scale_mask = []

        # 只处理当前这一条样本：frame_tokens[0], mag_id[0]
        x = frame_tokens[0]   # [N, D]
        m = mag_id[0]         # [N]

        for s in range(self.num_scales):
            # 当前倍率对应的 frame
            cur_mask = (m == s)  # [N]
            scale_mask.append(cur_mask.any())

            if cur_mask.any():
                # query_s: [D]
                query_s = self.scale_queries[s]

                # 当前倍率下每个 frame 的打分: [N]
                logits = (x * query_s.unsqueeze(0)).sum(dim=-1) / math.sqrt(D)

                # 只在当前倍率对应的 frame 上做 softmax
                alpha = masked_softmax(logits, cur_mask, dim=0)  # [N]

                # 聚成一个 scale token: [D]
                token = torch.sum(alpha.unsqueeze(-1) * x, dim=0)
            else:
                # 如果该倍率不存在，返回零向量
                token = torch.zeros(D, device=x.device, dtype=x.dtype)

            scale_tokens.append(token)

        # [S, D] -> [1, S, D]
        scale_tokens = torch.stack(scale_tokens, dim=0).unsqueeze(0)
        scale_mask = torch.stack(scale_mask, dim=0).unsqueeze(0)  # [1, S]

        return scale_tokens, scale_mask


class GatedAttentionPool(nn.Module):
    """
    默认 B=1 的 gated attention pooling。

    作用
    ----
    对 scale tokens 做 gated attention，
    得到一个 context vector。

    输入
    ----
    x:    Tensor[B, S, D]   (默认 B=1)
    mask: BoolTensor[B, S]
        表示哪些倍率 token 是有效的

    输出
    ----
    pooled: Tensor[B, D]
        context vector

    logits: Tensor[B, S]
        每个倍率 token 的原始分数

    alpha: Tensor[B, S]
        每个倍率 token 的 attention 权重
    """

    def __init__(
        self,
        dim: int,
        hidden_dim = None,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim

        self.v = nn.Linear(dim, hidden_dim)
        self.u = nn.Linear(dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,     # [B, S, D], 默认 B=1
        mask: torch.Tensor,  # [B, S]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"x 应为 [B, S, D]，实际={tuple(x.shape)}")
        if mask.ndim != 2:
            raise ValueError(f"mask 应为 [B, S]，实际={tuple(mask.shape)}")

        B, S, D = x.shape
        if B != 1:
            raise ValueError(
                f"当前简化版 GatedAttentionPool 默认 B=1，但实际 B={B}。"
            )

        # gated attention
        a = torch.tanh(self.v(x)) * torch.sigmoid(self.u(x))  # [1, S, H]
        logits = self.w(a).squeeze(-1)                        # [1, S]

        alpha = masked_softmax(logits, mask, dim=1)          # [1, S]
        pooled = torch.sum(alpha.unsqueeze(-1) * x, dim=1)   # [1, D]

        return pooled, logits, alpha


class ContextBranch(nn.Module):
    """
    默认 B=1 的完整 context branch。

    流程
    ----
    frame tokens
        -> 按倍率聚合成 scale tokens
        -> gated attention pooling 得到 context_vec
        -> 同时输出 global_scale_vec（对所有有效倍率 token 做简单平均）

    输入
    ----
    frame_tokens: Tensor[B, N, D]
    mag_id:       Tensor[B, N]

    输出
    ----
    context_vec: Tensor[B, D]
        context 分支输出

    context_logits: Tensor[B, S]
        每个倍率 token 的 gated attention 原始分数

    context_alpha: Tensor[B, S]
        每个倍率 token 的注意力权重

    scale_tokens: Tensor[B, S, D]
        每个倍率一个 token

    scale_mask: Tensor[B, S]
        哪些倍率在当前样本中存在

    global_scale_vec: Tensor[B, D]
        所有有效倍率 token 的简单平均向量
    """

    def __init__(
        self,
        dim: int,
        num_scales: int,
    ) -> None:
        super().__init__()
        self.scale_pool = ScaleAwarePool(dim=dim, num_scales=num_scales)
        self.context_pool = GatedAttentionPool(dim=dim)

    def forward(
        self,
        frame_tokens: torch.Tensor,  # [B, N, D], 默认 B=1
        mag_id: torch.Tensor,        # [B, N]
    ):
        # Step 1: frame -> scale
        scale_tokens, scale_mask = self.scale_pool(
            frame_tokens=frame_tokens,
            mag_id=mag_id,
        )  # [1, S, D], [1, S]

        # Step 2: scale -> context
        context_vec, context_logits, context_alpha = self.context_pool(
            x=scale_tokens,
            mask=scale_mask,
        )  # [1, D], [1, S], [1, S]

        # Step 3: simple global summary over valid scales
        scale_alpha = scale_mask.float()  # [1, S]
        scale_alpha = scale_alpha / scale_alpha.sum(dim=1, keepdim=True).clamp_min(1.0)
        global_scale_vec = torch.sum(scale_alpha.unsqueeze(-1) * scale_tokens, dim=1)  # [1, D]

        # return {
        #     "context_vec": context_vec,             # [1, D]
        #     "context_logits": context_logits,       # [1, S]
        #     "context_alpha": context_alpha,         # [1, S]
        #     "scale_tokens": scale_tokens,           # [1, S, D]
        #     "scale_mask": scale_mask,               # [1, S]
        #     "global_scale_vec": global_scale_vec,   # [1, D]
        # }, context_logits, context_alpha, scale_alpha, global_scale_vec
        return context_vec, global_scale_vec


class MOHPMIL_v0(BaseModel):
    def __init__(self, model_cfg: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(model_cfg, *args, **kwargs)
        self.dim_in: int = model_cfg.get("dim_in", 768)
        self.dim_hidden: int = model_cfg.get("dim_hidden", 1024)
        self.meta_dim: int = model_cfg.get("meta_dim", 3)
        self.num_mag: int = model_cfg.get("num_mag", 5)
        self.num_classes: int = model_cfg.get("num_classes", 2)
        self.topk_ratio: int = model_cfg.get("topk_ratio", 0.2)


        # model architecture
        self.patch_fusion = GatedAttention(dim_in=self.dim_in, dim_hidden=self.dim_hidden)

        self.mag_embed = nn.Embedding(self.num_mag, self.dim_in)

        self.meta_embed = MLP(self.meta_dim, self.dim_in)

        self.frame_fusion = Transformer(self.dim_in)

        self.evidence_branch = MultiClassTopKEvidencePool(self.dim_in, self.num_classes, self.topk_ratio)
        self.context_branch = ContextBranch(self.dim_in, self.num_mag)

        self.classifier = nn.Sequential(
            nn.Linear(self.dim_in * 3, self.dim_hidden),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.dim_hidden, 1),
        )

    def forward(self, x: torch.Tensor, order: torch.Tensor, meta: torch.Tensor, mag_id: torch.Tensor, patch_mask: torch.Tensor = None) -> torch.Tensor:
        # x.shape = [B, F. P, D]
        # order.shape = [B, F]
        # meta.shape = [B, F, 3]
        # mag.shape = [B, F]
        # patch_mask.shape = [B, F, P]
        B, F, P, D = x.shape
        C = self.num_classes

        assert B == 1 # currently only support batchsize = 1

        x = x.view(B * F, P, D)  # [F, P, D]
        if patch_mask is not None:
            patch_mask = patch_mask.view(B * F, P)  # [F, P]

        frame_tokens = self.patch_fusion(x, patch_mask).view(B, F, D) # [B, F, D]
        mag_tokens = self.mag_embed(mag_id)

        frame_tokens = frame_tokens + mag_tokens
        frame_tokens, _ = self.frame_fusion(frame_tokens)

        evidence_vecs, raw_scores, evidence_alpha = self.evidence_branch(frame_tokens)
        context_vec, global_scale_vec = self.context_branch(frame_tokens, mag_id)

        context_vec = context_vec.unsqueeze(1).expand(B, C, D)
        global_scale_vec = global_scale_vec.unsqueeze(1).expand(B, C, D)

        fused = torch.concat([evidence_vecs, context_vec, global_scale_vec], dim=-1)
        logits = self.classifier(fused).squeeze(-1)  # [B, C]
        return logits

    def _shared_step(self, batch: Tuple[torch.Tensor, ...], stage: str) -> torch.Tensor:
        x, y, order, meta, mag_id, patch_mask = batch
        logits = self(x, order, meta, mag_id, patch_mask)
        loss = torchF.cross_entropy(logits, y)

        if stage == "train":
            self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.train_metrics.update(logits, y)
        elif stage == "val":
            self.val_metrics.update(logits, y)
            self.val_cm.update(logits, y)
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        elif stage == "test":
            self.test_metrics.update(logits, y)
            self.test_cm.update(logits, y)
            self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        else:
            raise ValueError(f"Unsupported stage '{stage}'")

        return loss

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        self._shared_step(batch, stage="val")

    def test_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        self._shared_step(batch, stage="test")


class MOHPMIL_v1(BaseModel):
    def __init__(self, model_cfg: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(model_cfg, *args, **kwargs)
        self.dim_in: int = model_cfg.get("dim_in", 768)
        self.dim_hidden: int = model_cfg.get("dim_hidden", 1024)
        self.meta_dim: int = model_cfg.get("meta_dim", 3)
        self.num_mag: int = model_cfg.get("num_mag", 5)
        self.num_classes: int = model_cfg.get("num_classes", 2)
        self.topk_ratio: float = model_cfg.get("topk_ratio", 0.2)
        self.num_order_buckets: int = model_cfg.get("num_order_buckets", 8)
        self.num_heads: int = model_cfg.get("num_heads", 8)

        self.patch_fusion = GatedAttention(dim_in=self.dim_in, dim_hidden=self.dim_hidden)
        self.mag_embed = nn.Embedding(self.num_mag, self.dim_in)
        self.frame_fusion = OrderMagAwareTransformerBlock(
            dim_in=self.dim_in,
            num_heads=self.num_heads,
            num_order_buckets=self.num_order_buckets,
            num_mag=self.num_mag,
        )
        self.evidence_branch = MultiClassTopKEvidencePool(self.dim_in, self.num_classes, self.topk_ratio)
        self.context_branch = ContextBranch(self.dim_in, self.num_mag)
        self.classifier = nn.Sequential(
            nn.Linear(self.dim_in * 3, self.dim_hidden),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.dim_hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        order: torch.Tensor,
        meta: torch.Tensor,
        mag_id: torch.Tensor,
        patch_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, F, P, D = x.shape
        C = self.num_classes

        assert B == 1, "currently only support batchsize = 1"

        x = x.view(B * F, P, D)  # [F, P, D]
        if patch_mask is not None:
            patch_mask = patch_mask.view(B * F, P)  # [F, P]

        frame_tokens = self.patch_fusion(x, patch_mask).view(B, F, D)
        mag_tokens = self.mag_embed(mag_id)
        frame_tokens = frame_tokens + mag_tokens
        frame_tokens, _ = self.frame_fusion(frame_tokens, order, mag_id)

        evidence_vecs, _, _ = self.evidence_branch(frame_tokens)
        context_vec, global_scale_vec = self.context_branch(frame_tokens, mag_id)

        context_vec = context_vec.unsqueeze(1).expand(B, C, D)
        global_scale_vec = global_scale_vec.unsqueeze(1).expand(B, C, D)

        fused = torch.concat([evidence_vecs, context_vec, global_scale_vec], dim=-1)
        logits = self.classifier(fused).squeeze(-1)
        return logits

    def _shared_step(self, batch: Tuple[torch.Tensor, ...], stage: str) -> torch.Tensor:
        x, y, order, meta, mag_id, patch_mask = batch
        logits = self(x, order, meta, mag_id, patch_mask)
        loss = torchF.cross_entropy(logits, y)

        if stage == "train":
            self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.train_metrics.update(logits, y)
        elif stage == "val":
            self.val_metrics.update(logits, y)
            self.val_cm.update(logits, y)
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        elif stage == "test":
            self.test_metrics.update(logits, y)
            self.test_cm.update(logits, y)
            self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        else:
            raise ValueError(f"Unsupported stage '{stage}'")

        return loss

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        self._shared_step(batch, stage="val")

    def test_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        self._shared_step(batch, stage="test")


class ScaleMemoryContextBranch(nn.Module):
    def __init__(self, dim: int, num_scales: int) -> None:
        super().__init__()
        self.scale_pool = ScaleAwarePool(dim=dim, num_scales=num_scales)

    def forward(
        self,
        frame_tokens: torch.Tensor,
        mag_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_tokens, scale_mask = self.scale_pool(
            frame_tokens=frame_tokens,
            mag_id=mag_id,
        )  # [B, S, D], [B, S]
        scale_alpha = scale_mask.float()
        scale_alpha = scale_alpha / scale_alpha.sum(dim=1, keepdim=True).clamp_min(1.0)
        global_scale_vec = torch.sum(scale_alpha.unsqueeze(-1) * scale_tokens, dim=1)  # [B, D]
        return scale_tokens, scale_mask, global_scale_vec


class ClassConditionedContextRefiner(nn.Module):
    def __init__(
        self,
        dim_in: int,
        num_heads: int = 8,
        atten_dropout: float = 0.1,
        proj_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert dim_in % num_heads == 0, "dim_in must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim_in // num_heads
        self.scale = self.head_dim ** -0.5
        self.query_norm = nn.LayerNorm(dim_in)
        self.memory_norm = nn.LayerNorm(dim_in)
        self.q_proj = nn.Linear(dim_in, dim_in)
        self.k_proj = nn.Linear(dim_in, dim_in)
        self.v_proj = nn.Linear(dim_in, dim_in)
        self.out_proj = nn.Linear(dim_in, dim_in)
        self.atten_dropout = nn.Dropout(p=atten_dropout)
        self.proj_dropout = nn.Dropout(p=proj_dropout)

    def forward(
        self,
        evidence_vecs: torch.Tensor,
        scale_tokens: torch.Tensor,
        scale_mask: torch.Tensor,
    ) -> torch.Tensor:
        if evidence_vecs.ndim != 3:
            raise ValueError(f"evidence_vecs should be [B, C, D], got {tuple(evidence_vecs.shape)}")
        if scale_tokens.ndim != 3:
            raise ValueError(f"scale_tokens should be [B, S, D], got {tuple(scale_tokens.shape)}")
        if scale_mask.ndim != 2:
            raise ValueError(f"scale_mask should be [B, S], got {tuple(scale_mask.shape)}")

        B, C, D = evidence_vecs.shape
        _, S, _ = scale_tokens.shape

        q = self.q_proj(self.query_norm(evidence_vecs))
        k = self.k_proj(self.memory_norm(scale_tokens))
        v = self.v_proj(self.memory_norm(scale_tokens))

        q = q.reshape(B, C, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, C, Hd]
        k = k.reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, S, Hd]
        v = v.reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, S, Hd]

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, C, S]
        key_mask = ~scale_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
        attn_logits = attn_logits.masked_fill(key_mask, -1e9)

        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.atten_dropout(attn)
        out = torch.matmul(attn, v)  # [B, H, C, Hd]
        out = out.transpose(1, 2).contiguous().reshape(B, C, D)
        out = self.out_proj(out)
        out = self.proj_dropout(out)
        return out


class MOHPMIL_v2(BaseModel):
    def __init__(self, model_cfg: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(model_cfg, *args, **kwargs)
        self.dim_in: int = model_cfg.get("dim_in", 768)
        self.dim_hidden: int = model_cfg.get("dim_hidden", 1024)
        self.meta_dim: int = model_cfg.get("meta_dim", 3)
        self.num_mag: int = model_cfg.get("num_mag", 5)
        self.num_classes: int = model_cfg.get("num_classes", 2)
        self.topk_ratio: float = model_cfg.get("topk_ratio", 0.2)
        self.num_order_buckets: int = model_cfg.get("num_order_buckets", 8)
        self.num_heads: int = model_cfg.get("num_heads", 8)
        self.context_num_heads: int = model_cfg.get("context_num_heads", self.num_heads)

        self.patch_fusion = GatedAttention(dim_in=self.dim_in, dim_hidden=self.dim_hidden)
        self.mag_embed = nn.Embedding(self.num_mag, self.dim_in)
        self.frame_fusion = OrderMagAwareTransformerBlock(
            dim_in=self.dim_in,
            num_heads=self.num_heads,
            num_order_buckets=self.num_order_buckets,
            num_mag=self.num_mag,
        )
        self.evidence_branch = MultiClassTopKEvidencePool(self.dim_in, self.num_classes, self.topk_ratio)
        self.context_branch = ScaleMemoryContextBranch(self.dim_in, self.num_mag)
        self.context_refiner = ClassConditionedContextRefiner(
            dim_in=self.dim_in,
            num_heads=self.context_num_heads,
        )
        self.gate_proj = nn.Sequential(
            nn.LayerNorm(self.dim_in * 2),
            nn.Linear(self.dim_in * 2, self.dim_in),
            nn.GELU(),
            nn.Linear(self.dim_in, self.dim_in),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.dim_in * 2, self.dim_hidden),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.dim_hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        order: torch.Tensor,
        meta: torch.Tensor,
        mag_id: torch.Tensor,
        patch_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, F, P, D = x.shape
        C = self.num_classes

        assert B == 1, "currently only support batchsize = 1"

        x = x.view(B * F, P, D)
        if patch_mask is not None:
            patch_mask = patch_mask.view(B * F, P)

        frame_tokens = self.patch_fusion(x, patch_mask).view(B, F, D)
        mag_tokens = self.mag_embed(mag_id)
        frame_tokens = frame_tokens + mag_tokens
        frame_tokens, _ = self.frame_fusion(frame_tokens, order, mag_id)

        evidence_vecs, _, _ = self.evidence_branch(frame_tokens)  # [B, C, D]
        scale_tokens, scale_mask, global_scale_vec = self.context_branch(frame_tokens, mag_id)  # [B, S, D], [B, S], [B, D]
        class_context_vecs = self.context_refiner(evidence_vecs, scale_tokens, scale_mask)  # [B, C, D]

        gate = self.gate_proj(torch.cat([evidence_vecs, class_context_vecs], dim=-1))
        refined_evidence = evidence_vecs + gate * class_context_vecs

        global_scale_vec = global_scale_vec.unsqueeze(1).expand(B, C, D)
        fused = torch.cat([refined_evidence, global_scale_vec], dim=-1)
        logits = self.classifier(fused).squeeze(-1)
        return logits

    def _shared_step(self, batch: Tuple[torch.Tensor, ...], stage: str) -> torch.Tensor:
        x, y, order, meta, mag_id, patch_mask = batch
        logits = self(x, order, meta, mag_id, patch_mask)
        loss = torchF.cross_entropy(logits, y)

        if stage == "train":
            self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.train_metrics.update(logits, y)
        elif stage == "val":
            self.val_metrics.update(logits, y)
            self.val_cm.update(logits, y)
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        elif stage == "test":
            self.test_metrics.update(logits, y)
            self.test_cm.update(logits, y)
            self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        else:
            raise ValueError(f"Unsupported stage '{stage}'")

        return loss

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        self._shared_step(batch, stage="val")

    def test_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        self._shared_step(batch, stage="test")

if __name__ == "__main__":
    model_cfg = {"dim_in": 768, "dim_hidden": 1024, "meta_dim": 3, "num_mag": 5, "num_classes": 2, "topk_ratio": 0.2}

    B, F, P, D = 1, 7, 10, 768

    # patch features
    x = torch.randn(B, F, P, D)

    # retained order（当前版本虽然没用，但先补齐）
    order = torch.arange(F).unsqueeze(0)  # [1, 7]

    # frame-level metadata，按你现在的设定是 3 维
    meta = torch.randn(B, F, 3)

    # magnification id，范围必须在 [0, mag_dim-1]
    mag_id = torch.tensor([[0, 1, 1, 2, 3, 4, 0]], dtype=torch.long)

    # patch_mask: True=无效 patch, False=有效 patch
    # 这里先全部设为有效
    patch_mask = torch.zeros(B, F, P, dtype=torch.bool)

    model = MOHPMIL_v0(model_cfg)
    with torch.no_grad():
        out = model(x, order, meta, mag_id, patch_mask)
        print(torchF.softmax(out))

