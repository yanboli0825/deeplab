import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.models.base_model import BaseModel
from typing import Any, Dict

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM_SB(BaseModel):
    # def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
    #     subtyping=False, embed_dim=1024):
    #     super().__init__()
    def __init__(self, model_cfg: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(model_cfg, *args, **kwargs)
        self.size_dict = {"small": [model_cfg.get("embed_dim", 768), 512, 256], "big": [model_cfg.get("embed_dim", 768), 512, 384]}
        size = self.size_dict[model_cfg.get("size_arg", "small")]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(model_cfg.get("dropout", 0.0))]
        if model_cfg.get("gate", True):
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = model_cfg.get("dropout", 0.0), n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = model_cfg.get("dropout", 0.0), n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], model_cfg.get("n_classes", 2))
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(model_cfg.get("n_classes", 2))]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = model_cfg.get("k_sample", 8)
        self.instance_loss_fn = nn.CrossEntropyLoss()
        self.n_classes = model_cfg.get("n_classes", 2)
        self.subtyping = model_cfg.get("subtyping", False)
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    # 仅接受batch_size=1
    def forward(self, h, label=None, instance_eval=True, return_features=False, attention_only=False):
        h = h.squeeze(0)
        A, h = self.attention_net(h)  # NxK。 这里返回的h已经不是原始，是输入过了一个非线性层后的输出
        A = torch.transpose(A, 1, 0)  # KxN （1xN）
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})

        return logits, results_dict
        if self.training:
            loss_fn = nn.CrossEntropyLoss()
            ceLoss = loss_fn(logits, label)
            return {'ce_loss': ceLoss.float(), 'inst_loss': total_inst_loss}
        else:
            return logits, Y_prob, Y_hat, A_raw


    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits, results_dict = self(x, y)
        loss = torch.nn.functional.cross_entropy(logits, y) + results_dict["instance_loss"]

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_metrics.update(logits, y)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        logits, results_dict = self(x, y)
        loss = torch.nn.functional.cross_entropy(logits, y) + results_dict["instance_loss"]

        self.val_metrics.update(logits, y)
        self.val_cm.update(logits, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        logits, results_dict = self(x, y)
        loss = torch.nn.functional.cross_entropy(logits, y) + results_dict["instance_loss"]

        self.test_metrics.update(logits, y)
        self.test_cm.update(logits, y)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

class CLAM_MB(CLAM_SB):
    # def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
    #     instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
    #     nn.Module.__init__(self)
    def __init__(self, model_cfg: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(model_cfg, *args, **kwargs)
        self.size_dict = {"small": [model_cfg.get("embed_dim", 768), 512, 256], "big": [model_cfg.get("embed_dim", 768), 512, 384]}
        size = self.size_dict[model_cfg.get("size_arg", "small")]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(model_cfg.get("dropout", 0.0))]
        if model_cfg.get("gate", True):
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = model_cfg.get("dropout", 0.0), n_classes = model_cfg.get("n_classes", 2))
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = model_cfg.get("dropout", 0.0), n_classes = model_cfg.get("n_classes", 2))
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(model_cfg.get("n_classes", 2))] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(model_cfg.get("n_classes", 2))]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = model_cfg.get("k_sample", 8)
        self.instance_loss_fn = nn.CrossEntropyLoss()
        self.n_classes = model_cfg.get("n_classes", 2)
        self.subtyping = model_cfg.get("subtyping", False)

    def forward(self, h, label=None, instance_mask=None, instance_eval=True, return_features=False, attention_only=False):
        h = h.squeeze(0)
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h) 

        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})

        return logits, results_dict

        if self.training:
            loss_fn = nn.CrossEntropyLoss()
            ceLoss = loss_fn(logits, label)
            return {'ce_loss': ceLoss.float(), 'inst_loss': total_inst_loss}
        else:
            return Y_prob, Y_hat, logits, A_raw
