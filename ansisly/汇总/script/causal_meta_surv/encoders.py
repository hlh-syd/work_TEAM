"""encoders.py — Meta/Causal/Clinical/TaskAdapter 编码器（LayerNorm，无 BatchNorm）。

提供：
    - GeneProjection [B,P] -> [B, 64]
    - MetaEncoder [B, 64] -> z_meta[B, 24]（64-48-24）
    - CausalInvariantEncoder [B, 64+Q] -> z_inv[B, 16]（48-24-16）
    - ClinicalEncoder [B, Q+mask] -> z_clin[B, 8]（16-8）
    - TaskAdapter rank 4-8 低秩残差
    - GradientReversalFunction / gradient_reversal() for DANN
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# 梯度反转层（DANN）
# ──────────────────────────────────────────────────────────────────────
class GradientReversalFunction(torch.autograd.Function):
    """梯度反转函数（DANN 核心组件）。"""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.lambda_, None


def gradient_reversal(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """应用梯度反转层。"""
    return GradientReversalFunction.apply(x, lambda_)


# ──────────────────────────────────────────────────────────────────────
# 基因投影
# ──────────────────────────────────────────────────────────────────────
class GeneProjection(nn.Module):
    """[B, P] -> [B, 64] low-rank 基因投影。

    PPI/pathway 正则作用点（简化版不实现 group sparsity）。
    """

    def __init__(self, input_dim: int, output_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.norm(h)
        h = F.gelu(h)
        h = self.dropout(h)
        return h


# ──────────────────────────────────────────────────────────────────────
# MetaEncoder（承载 Reptile 跨癌种初始化）
# ──────────────────────────────────────────────────────────────────────
class MetaEncoder(nn.Module):
    """[B, 64] -> z_meta[B, 24]（64-48-24）。

    所有隐藏层使用 LayerNorm（禁用 BatchNorm）。
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 48,
        output_dim: int = 24,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.norm1(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.norm2(h)
        return h


# ──────────────────────────────────────────────────────────────────────
# CausalInvariantEncoder
# ──────────────────────────────────────────────────────────────────────
class CausalInvariantEncoder(nn.Module):
    """[B, 64+Q] -> z_inv[B, 16]（48-24-16）。

    学习稳定/混杂相关表示，输出经 GRL 进入 DomainHead。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 16,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.norm1(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.norm2(h)
        return h


# ──────────────────────────────────────────────────────────────────────
# ClinicalEncoder
# ──────────────────────────────────────────────────────────────────────
class ClinicalEncoder(nn.Module):
    """[B, Q+mask] -> z_clin[B, 8]（16-8）。

    保留强临床信号（stage, age, sex），防止域对齐误删。
    输入：临床变量 + missing indicator 拼接。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        output_dim: int = 8,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.norm1(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.norm2(h)
        return h


# ──────────────────────────────────────────────────────────────────────
# TaskAdapter（rank 4-8 低秩残差）
# ──────────────────────────────────────────────────────────────────────
class TaskAdapter(nn.Module):
    """z_meta -> z_task rank 4-8 低秩残差 + task-specific bias（P2-4）。

    CRC/癌种特异低秩适配，避免全参数微调过拟合。
    公式（P2-4 升级）：z_task = z_meta + A @ B @ z_meta + bias + task_bias[task_id]
        其中 A: [rank, dim], B: [dim, rank], task_bias: nn.Embedding[n_tasks, dim]

    向后兼容：task_id=None 时仅用原 bias（等价于 P2-4 前行为）。
    """

    def __init__(self, dim: int, rank: int = 4, n_tasks: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.n_tasks = n_tasks
        # 低秩矩阵 A @ B
        self.A = nn.Parameter(torch.zeros(rank, dim))
        self.B = nn.Parameter(torch.zeros(dim, rank))
        # 初始化为小值
        nn.init.normal_(self.A, std=0.01)
        nn.init.normal_(self.B, std=0.01)
        # 任务偏置
        self.bias = nn.Parameter(torch.zeros(dim))
        # P2-4: task-specific bias
        self.task_bias: Optional[nn.Embedding] = None
        if n_tasks is not None and n_tasks > 0:
            self.task_bias = nn.Embedding(n_tasks, dim)
            nn.init.zeros_(self.task_bias.weight)  # 零初始化保证向后兼容

    def forward(self, z_meta: torch.Tensor, task_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """z_task = z_meta + (z_meta @ B) @ A + bias + task_bias[task_id]。

        Args:
            z_meta: [B, dim]
            task_id: [B] long tensor 或 None。None 时退化为原行为。
        """
        # z_meta: [B, dim]
        h = z_meta @ self.B  # [B, rank]
        h = h @ self.A  # [B, dim]
        out = z_meta + h + self.bias
        # P2-4: task-specific bias
        if self.task_bias is not None and task_id is not None:
            # task_id: [B] long
            tb = self.task_bias(task_id.to(z_meta.device))  # [B, dim]
            out = out + tb
        return out
