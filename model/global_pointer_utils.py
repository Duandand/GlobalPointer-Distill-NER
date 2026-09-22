# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

class MetricsCalculator(object):
    def __init__(self):
        super().__init__()

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    def get_evaluate_fpr(self, y_pred, y_true, theta=0):
        """单个 batch 的实体级 P/R/F1（用于训练过程中的日志展示）。"""
        X, Y, Z = 1e-10, 1e-10, 1e-10
        y_pred = y_pred.data.cpu().numpy()
        y_true = y_true.data.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > theta)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))
        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)
        if Y == 0 or Z == 0:
            return 0, 0, 0
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall

    def get_evaluate_counts(self, y_pred, y_true, theta=0):
        """单个 batch 的实体级命中计数 (X, Y, Z)。

        X = 预测实体 ∩ 真实实体（边界+类型完全匹配）, Y = 预测实体数, Z = 真实实体数。
        跨 batch 累加后再统一计算 P/R/F1，即为 corpus 级指标，可与论文/公开基准直接对比。
        """
        y_pred = y_pred.data.cpu().numpy()
        y_true = y_true.data.cpu().numpy()
        pred = {(b, l, s, e) for b, l, s, e in zip(*np.where(y_pred > theta))}
        true = {(b, l, s, e) for b, l, s, e in zip(*np.where(y_true > 0))}
        return len(pred & true), len(pred), len(true)

    def get_evaluate_counts_by_class(self, y_pred, y_true, theta=0):
        """单个 batch 内每个实体类型的 (X, Y, Z) 命中计数，key 为类型 id。

        与 get_evaluate_counts 配合使用，跨 batch 累加后可得到分类别的 corpus 级指标。
        """
        y_pred = y_pred.data.cpu().numpy()
        y_true = y_true.data.cpu().numpy()
        pred_by, true_by = {}, {}
        for b, l, s, e in zip(*np.where(y_pred > theta)):
            pred_by.setdefault(int(l), set()).add((b, l, s, e))
        for b, l, s, e in zip(*np.where(y_true > 0)):
            true_by.setdefault(int(l), set()).add((b, l, s, e))
        res = {}
        for label in set(pred_by) | set(true_by):
            p_set = pred_by.get(label, set())
            t_set = true_by.get(label, set())
            res[label] = (len(p_set & t_set), len(p_set), len(t_set))
        return res

    @staticmethod
    def counts_to_fpr(x, y, z):
        """由累计计数 (X, Y, Z) 计算 corpus 级 (f1, precision, recall)。"""
        precision = x / y if y else 0.0
        recall = x / z if z else 0.0
        f1 = 2 * x / (y + z) if (y + z) else 0.0
        return f1, precision, recall


class SinusoidalPositionEmbedding(nn.Module):
    """定义Sin-Cos位置Embedding"""

    def __init__(self, output_dim, merge_mode="add", custom_position_ids=False):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        if self.custom_position_ids:
            seq_len = inputs.shape[1]
            inputs, position_ids = inputs
            position_ids = position_ids.type(torch.float)
        else:
            input_shape = inputs.shape
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum("bn,d->bnd", position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        if self.merge_mode == "add":
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == "mul":
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == "zero":
            return embeddings.to(inputs.device)

def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12 # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

def gp_loss_func(y_pred, y_true):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_true.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1) # (batch_size*ent_type_size, max_len*max_len)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1) # (batch_size*ent_type_size, max_len*max_len)
    loss = multilabel_categorical_crossentropy(y_pred, y_true)
    return loss


def tril_onnx(
    inputs: torch.FloatTensor, diagonal: Optional[int] = 0
) -> torch.FloatTensor:
    """
    caveat to export an tril-based operator with ONNX
    https://github.com/pytorch/pytorch/issues/34129

    inputs: Input tensor.
    diagonal: Value of diagonal

    Returns:
        (torch.FloatTensor):Output tensor.

    examples:
    a = torch.randn((1,4,4,4))
    b = torch.tril(a, diagonal=-1)
    c = tril_onnx(a, diagonal=-1)
    print((b-c).sum())

    """
    arange = torch.arange(inputs.size(2), device=inputs.device)  # dim need to change
    arange2 = torch.arange(inputs.size(3), device=inputs.device)  # dim need to change

    mask = arange.unsqueeze(-1).expand(-1, inputs.size(2)) >= (
        arange2 - diagonal
    )  # dim need to change

    return inputs.masked_fill(mask == 0, 0)
