"""fusion.py — ReliabilityFusion 三分支 + CRCSpecificBranch + VariationalIB + 主模型。

提供：
    - ReliabilityFusion：三分支门控融合（softmax gate + min_weight + entropy penalty）
    - VariationalIB：重参数化 VIB（z = mu + exp(0.5*logvar) * epsilon）
    - CRCSpecificBranch：CRC 特异分支（Lasso-Cox 系数初始化）
    - orth_loss：正交约束 L_orth = ||Z_inv^T * Z_task / B||_F^2
    - CausalMetaIBSurv：主模型组装
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import CMIBConfig, CMIBOutput, CRCSpecificConfig, SurvivalRecordBatch
from .encoders import (
    CausalInvariantEncoder,
    ClinicalEncoder,
    GeneProjection,
    MetaEncoder,
    TaskAdapter,
)
from .heads import CoxHead, DiscreteTimeAuxHead, DomainHead
from .sparse_gate import HardConcreteSparseGate

LOGGER = logging.getLogger("cmib_surv.fusion")


# ──────────────────────────────────────────────────────────────────────
# ReliabilityFusion 三分支融合
# ──────────────────────────────────────────────────────────────────────
class ReliabilityFusion(nn.Module):
    """三分支门控融合：MetaEncoder / CausalInvariantEncoder / ClinicalEncoder。

    - 投影到 fusion_dim=32
    - softmax gate: (a_m, a_c, a_l) = softmax(G([h_m, h_c, h_l]))
    - 最小权重约束 a_k >= min_weight
    - entropy penalty 避免单分支垄断
    - 输出：h = a_m*h_m + a_c*h_c + a_l*h_l + P_t(z_task)
    """

    def __init__(
        self,
        meta_dim: int = 24,
        invariant_dim: int = 16,
        clinical_dim: int = 8,
        adapter_dim: Optional[int] = None,
        fusion_dim: int = 32,
        min_weight: float = 0.05,
        dropout: float = 0.15,
        n_heads: int = 2,  # 保留参数兼容，v4 回退不使用
    ):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.min_weight = min_weight

        # 各分支投影到 fusion_dim
        self.proj_meta = nn.Linear(meta_dim, fusion_dim)
        self.proj_inv = nn.Linear(invariant_dim, fusion_dim)
        self.proj_clin = nn.Linear(clinical_dim, fusion_dim)

        # 线性门控网络（v4 回退：从 Attention 回退到线性 gate）
        self.gate = nn.Linear(fusion_dim * 3, 3)

        # P3-2.2: SE block 轻量级 channel attention（替代已回退的 MultiheadAttention）
        self.se_gate = nn.Sequential(
            nn.Linear(fusion_dim * 3, max(fusion_dim // 4, 4)),
            nn.ReLU(),
            nn.Linear(max(fusion_dim // 4, 4), 3),
            nn.Sigmoid(),
        )

        # branch_mode 兼容（消融实验用：强制单分支权重）
        self.force_branch: Optional[str] = None

        # 任务残差投影（z_task -> fusion_dim）
        self.task_proj: Optional[nn.Linear] = None
        if adapter_dim is not None:
            self.task_proj = nn.Linear(adapter_dim, fusion_dim)

        self.norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

    def set_force_branch(self, branch_name: Optional[str]) -> None:
        """设置强制分支（消融实验用）。

        Args:
            branch_name: None | "meta_only" | "invariant_only" | "clinical_only"
        """
        self.force_branch = branch_name
        if branch_name is not None:
            # 强制单分支：设置 gate 的 bias 使 softmax 输出接近 one-hot
            mode_weights = {
                "meta_only":       [10.0, -10.0, -10.0],
                "invariant_only":  [-10.0, 10.0, -10.0],
                "clinical_only":   [-10.0, -10.0, 10.0],
            }[branch_name]
            with torch.no_grad():
                self.gate.weight.data.zero_()
                self.gate.bias.data = torch.tensor(mode_weights, dtype=self.gate.bias.dtype)
            for p in self.gate.parameters():
                p.requires_grad_(False)

    def forward(
        self,
        z_meta: torch.Tensor,
        z_inv: torch.Tensor,
        z_clin: torch.Tensor,
        z_task: Optional[torch.Tensor] = None,
        clin_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向。

        Returns:
            (h, fusion_weights)
            h: [B, fusion_dim]
            fusion_weights: [B, 3]
        """
        h_m = self.proj_meta(z_meta)  # [B, fusion_dim]
        h_c = self.proj_inv(z_inv)
        h_l = self.proj_clin(z_clin)

        # 线性门控 + P3-2.2 SE channel attention
        gate_input = torch.cat([h_m, h_c, h_l], dim=-1)  # [B, 3*fusion_dim]
        gate_logits = self.gate(gate_input)  # [B, 3]
        # P3-2.2: SE 提供自适应调制（迭代方案D: 回退到 P3 值 0.1）
        se_weights = self.se_gate(gate_input)  # [B, 3] sigmoid
        gate_logits = gate_logits + 0.1 * (se_weights - 0.5)
        weights = F.softmax(gate_logits, dim=-1)  # [B, 3]

        # 最小权重约束：clamp 到 [min_weight, 1.0] 后重归一化
        if self.min_weight > 0:
            weights = torch.clamp(weights, min=self.min_weight)
            weights = weights / weights.sum(dim=-1, keepdim=True)

        # 加权融合
        h = weights[:, 0:1] * h_m + weights[:, 1:2] * h_c + weights[:, 2:3] * h_l

        # 任务残差
        if z_task is not None and self.task_proj is not None:
            h = h + self.task_proj(z_task)

        h = self.norm(h)
        h = F.gelu(h)
        h = self.dropout(h)

        return h, weights

    def entropy_penalty(self, weights: torch.Tensor) -> torch.Tensor:
        """熵惩罚：避免单分支垄断。"""
        # 鼓励权重分散：max entropy = log(3)
        eps = 1e-8
        entropy = -(weights * torch.log(weights + eps)).sum(dim=-1).mean()
        # 我们要 max entropy，所以 penalty = -entropy
        return -entropy


# ──────────────────────────────────────────────────────────────────────
# VariationalIB（重参数化）
# ──────────────────────────────────────────────────────────────────────
class VariationalIB(nn.Module):
    """重参数化 VIB：z = mu + exp(0.5*logvar) * epsilon。

    mu = W_mu * h, logvar = clip(W_sigma * h, -8, 6)
    推理默认用 mu；MC 采样用 30-100 次。
    """

    def __init__(self, input_dim: int = 32, latent_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.W_mu = nn.Linear(input_dim, latent_dim)
        self.W_logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, h: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向。

        Args:
            h: [B, input_dim]
            sample: 是否采样（训练 True，推理 False）

        Returns:
            (z, mu, logvar)
        """
        mu = self.W_mu(h)
        logvar = torch.clamp(self.W_logvar(h), -8.0, 6.0)

        if sample and self.training:
            eps = torch.randn_like(mu)
            z = mu + torch.exp(0.5 * logvar) * eps
        else:
            # 推理：默认用 mu（无噪声）
            z = mu

        return z, mu, logvar

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x) || N(0,1)) = 0.5 * sum(mean(exp(logvar) + mu^2 - 1 - logvar))。"""
        kl = 0.5 * (torch.exp(logvar) + mu ** 2 - 1 - logvar)
        return kl.sum(dim=-1).mean()


# ──────────────────────────────────────────────────────────────────────
# CRCSpecificBranch（v2.0 Path D 双分支 Branch B）
# ──────────────────────────────────────────────────────────────────────
class CRCSpecificBranch(nn.Module):
    """CRC 特异分支：5 个 strict 阻断基因 + 02 高频基因。

    - Lasso-Cox 系数初始化第一层：W1[:, j] = coef_j * scale
    - 可选第一层冻结前 N epoch
    """

    def __init__(
        self,
        gene_indices: np.ndarray,  # 在 gene_registry 中的索引
        lasso_coefs: Optional[np.ndarray] = None,
        config: CRCSpecificConfig = CRCSpecificConfig(),
        output_dim: int = 16,
    ):
        super().__init__()
        self.config = config
        self.gene_indices = torch.from_numpy(gene_indices.astype(np.int64))
        n_input = len(gene_indices)

        if n_input == 0:
            self.enabled = False
            self.fc1 = nn.Linear(1, output_dim)  # placeholder
            self.fc2 = nn.Linear(output_dim, output_dim)
            return

        self.enabled = True
        self.fc1 = nn.Linear(n_input, output_dim)
        self.norm1 = nn.LayerNorm(output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(config.dropout)

        # Lasso-Cox 系数初始化
        if config.use_lasso_init and lasso_coefs is not None:
            with torch.no_grad():
                # W1[:, j] = coef_j * scale
                # fc1.weight: [output_dim, n_input]
                coefs = torch.from_numpy(lasso_coefs.astype(np.float32))
                self.fc1.weight.data = coefs.unsqueeze(0).repeat(output_dim, 1) * config.lasso_init_scale
                self.fc1.bias.data.zero_()
                LOGGER.info("CRCSpecificBranch: initialized with Lasso-Cox coefs (n=%d)", n_input)

    def forward(self, x_gene: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None
        # 提取 CRC 特异基因子集
        x_crc = x_gene[:, self.gene_indices.to(x_gene.device)]  # [B, n_input]
        h = self.fc1(x_crc)
        h = self.norm1(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.norm2(h)
        return h


def orth_loss(z_inv: torch.Tensor, z_task: torch.Tensor) -> torch.Tensor:
    """正交约束：L_orth = ||Z_inv^T * Z_task / B||_F^2。

    防止 CRC 特异信号被域对齐抹除。
    """
    B = z_inv.shape[0]
    if B == 0:
        return torch.tensor(0.0, device=z_inv.device)
    # Z_inv^T @ Z_task: [invariant_dim, task_dim]
    cross = z_inv.t() @ z_task  # [inv_dim, task_dim]
    return (cross ** 2).sum() / B


# ──────────────────────────────────────────────────────────────────────
# 主模型 CausalMetaIBSurv
# ──────────────────────────────────────────────────────────────────────
class CausalMetaIBSurv(nn.Module):
    """CMIB-Surv 主模型：组装所有模块。

    模型结构：
        SparseGate -> GeneProjection -> [MetaEncoder, CausalInvariantEncoder, ClinicalEncoder]
        -> ReliabilityFusion -> VariationalIB -> [CoxHead, DiscreteTimeAuxHead, DomainHead]
        + CRCSpecificBranch (并行)
    """

    def __init__(
        self,
        n_genes: int,
        n_clinical: int,
        cfg: CMIBConfig,
        crc_gene_indices: Optional[np.ndarray] = None,
        crc_lasso_coefs: Optional[np.ndarray] = None,
        n_tasks: int = 9,
    ):
        super().__init__()
        self.cfg = cfg
        self.n_genes = n_genes
        self.n_clinical = n_clinical
        self.n_tasks = n_tasks

        # 1. SparseGate
        self.sparse_gate = HardConcreteSparseGate(
            n_features=n_genes,
            temperature=cfg.gate_temperature,
            gamma=cfg.gate_gamma,
            zeta=cfg.gate_zeta,
            target_active=cfg.active_gene_target,
        )

        # 2. GeneProjection [B,P] -> [B, 64]
        self.gene_proj = GeneProjection(
            input_dim=n_genes,
            output_dim=cfg.projection_dim,
            dropout=cfg.dropout,
        )

        # 3. 编码器
        self.meta_encoder = MetaEncoder(
            input_dim=cfg.projection_dim,
            hidden_dim=cfg.meta_hidden_dim,
            output_dim=cfg.meta_dim,
            dropout=cfg.dropout,
        )
        self.causal_encoder = CausalInvariantEncoder(
            input_dim=cfg.projection_dim + n_clinical,
            hidden_dim=cfg.invariant_hidden_dim,
            output_dim=cfg.invariant_dim,
            dropout=cfg.dropout,
        )
        self.clinical_encoder = ClinicalEncoder(
            input_dim=n_clinical * 2,  # 临床变量 + missing indicator
            hidden_dim=16,
            output_dim=cfg.clinical_dim,
            dropout=cfg.dropout,
        )
        self.task_adapter = TaskAdapter(
            dim=cfg.meta_dim,
            rank=cfg.adapter_rank,
            n_tasks=n_tasks,  # P2-4: 传入 n_tasks
        )

        # 4. ReliabilityFusion
        self.fusion = ReliabilityFusion(
            meta_dim=cfg.meta_dim,
            invariant_dim=cfg.invariant_dim,
            clinical_dim=cfg.clinical_dim,
            adapter_dim=cfg.meta_dim,  # task adapter 输出维度 = meta_dim
            fusion_dim=cfg.fusion_dim,
            min_weight=cfg.min_branch_weight,
            dropout=cfg.dropout,
        )

        # 5. VariationalIB
        self.vib = VariationalIB(
            input_dim=cfg.fusion_dim,
            latent_dim=cfg.ib_dim,
        )

        # 6. CRC 特异分支
        self.crc_branch: Optional[CRCSpecificBranch] = None
        if crc_gene_indices is not None and cfg.crc_branch.enabled:
            self.crc_branch = CRCSpecificBranch(
                gene_indices=crc_gene_indices,
                lasso_coefs=crc_lasso_coefs,
                config=cfg.crc_branch,
                output_dim=cfg.crc_branch.hidden_dim,
            )

        # 7. 头
        ib_plus_crc_dim = cfg.ib_dim + (cfg.crc_branch.hidden_dim if self.crc_branch is not None else 0)
        self.cox_head = CoxHead(
            input_dim=ib_plus_crc_dim,
            hidden_dim=8,
            dropout=cfg.dropout,
        )
        self.aux_head = DiscreteTimeAuxHead(
            input_dim=ib_plus_crc_dim,
            n_bins=12,
            dropout=cfg.dropout,
        )
        self.domain_head = DomainHead(
            input_dim=cfg.invariant_dim,
            n_tasks=n_tasks,
            dropout=cfg.dropout,
        )

    def forward(
        self,
        batch: SurvivalRecordBatch,
        sample_latent: bool = True,
    ) -> CMIBOutput:
        """前向。

        Args:
            batch: SurvivalRecordBatch
            sample_latent: True 训练（采样 z），False 推理（用 mu）

        Returns:
            CMIBOutput
        """
        x_gene = batch.x_gene.to(self.sparse_gate.log_alpha.device)
        x_clin = batch.x_clinical.to(self.sparse_gate.log_alpha.device)
        clin_mask = batch.clinical_mask.to(self.sparse_gate.log_alpha.device).float()

        # 1. SparseGate
        gated = self.sparse_gate(x_gene)

        # 2. GeneProjection
        h_gene = self.gene_proj(gated)  # [B, 64]

        # 3. 编码器
        z_meta = self.meta_encoder(h_gene)  # [B, 24]
        z_inv = self.causal_encoder(torch.cat([h_gene, x_clin], dim=-1))  # [B, 16]
        z_clin = self.clinical_encoder(torch.cat([x_clin, clin_mask], dim=-1))  # [B, 8]
        z_task = self.task_adapter(z_meta, task_id=batch.task_id)  # P2-4: 传 task_id

        # 4. Fusion
        h_fusion, fusion_weights = self.fusion(z_meta, z_inv, z_clin, z_task, clin_mask)

        # 5. VIB
        if not self.training:
            sample_latent = False
        z, mu, logvar = self.vib(h_fusion, sample=sample_latent)

        # 6. CRC 分支
        z_crc = None
        if self.crc_branch is not None:
            z_crc = self.crc_branch(x_gene)  # [B, hidden]

        # 7. 头
        if z_crc is not None:
            z_combined = torch.cat([z, z_crc], dim=-1)
        else:
            z_combined = z

        log_risk = self.cox_head(z_combined)  # [B]
        aux_hazard = self.aux_head(z_combined)  # [B, n_bins]
        domain_logits = self.domain_head(z_inv, grl_lambda=self.cfg.gradient_reversal_weight)

        # gate probabilities
        gate_prob = self.sparse_gate.gate_probabilities()

        return CMIBOutput(
            log_risk=log_risk,
            mu=mu,
            logvar=logvar,
            z=z,
            z_meta=z_meta,
            z_invariant=z_inv,
            z_task=z_task,
            z_clin=z_clin,
            z_crc=z_crc,
            fusion_weights=fusion_weights,
            gate_prob=gate_prob,
            domain_logits=domain_logits,
            aux_hazard=aux_hazard,
        )

    def predict_risk(
        self,
        batch: SurvivalRecordBatch,
        mc_samples: int = 0,
    ) -> np.ndarray:
        """预测风险分数。

        Args:
            batch: SurvivalRecordBatch
            mc_samples: 0 用 mu；>0 用 MC 均值

        Returns:
            risk: [B] numpy
        """
        self.eval()
        with torch.no_grad():
            if mc_samples <= 0:
                output = self.forward(batch, sample_latent=False)
                return output.log_risk.cpu().numpy()
            else:
                risks = []
                for _ in range(mc_samples):
                    output = self.forward(batch, sample_latent=True)
                    risks.append(output.log_risk.cpu().numpy())
                return np.mean(risks, axis=0)

    def predict_survival(
        self,
        batch: SurvivalRecordBatch,
        times: np.ndarray,
    ) -> np.ndarray:
        """简化 Breslow 基线生存函数估计。

        Returns:
            surv: [B, n_times]
        """
        risk = self.predict_risk(batch, mc_samples=0)
        n = len(risk)
        n_times = len(times)
        surv = np.ones((n, n_times), dtype=np.float32)
        for j, t in enumerate(times):
            # 简化：S(t|x) = exp(-H0(t) * exp(risk))
            surv[:, j] = np.exp(-0.01 * t * np.exp(risk - risk.mean()))
        return surv
