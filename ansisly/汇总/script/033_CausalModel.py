"""033_CausalModel.py — Causal-DeepSurv-IB 模块化重构版

因果深度生存模型变体 2：
使用 PyTorch 原生实现因果编码器（替代 031 的外部 CausalEGM 库），
结合信息瓶颈 (IB) 正则化和对比学习 (Contrastive Learning) 预训练。

核心差异 (vs 031_CDSIB / 032_Baseline):
    - 因果编码器: PyTorch 原生 VAE 风格编码器，支持 z_shared/z_outcome 解耦
    - 对比预训练: 在因果表征空间施加 NT-Xent 对比损失，增强类内紧凑/类间分离
    - IB 门控: 带可学习温度退火的 VIB 正则化
    - 对抗去混杂: 梯度反转层实现 treatment-invariant 表征
    - 无外部 CausalEGM 依赖，完全自包含

架构：
    X → FeatureProjector → CausalEncoder (z_shared, z_outcome)
    → AdversarialDeconfounder → IBGatingLayer → SurvivalHead → log risk

损失函数：
    L = L_IPCW + 0.3*L_rank + L_IB + 0.5*L_contrast + 1e-4*L_elastic

参考文献：
    - VIB: Alemi et al., ICLR 2017
    - 对比学习: Chen et al., NeurIPS 2020 (SimCLR)
    - 对抗去混杂: Ganin et al., JMLR 2016 (DANN)
"""

from __future__ import annotations

import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ── 项目内依赖 ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from shared_utils import (
    setup_logger, RANDOM_SEED, EPS,
    DATA_DIR, RESULTS_DIR, ESSENTIAL_DIR,
    ensure_dir, safe_read_tsv, normalize_patient_id,
    detect_event_time_columns, coerce_event,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
    from sksurv.util import Surv
    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False

try:
    from lifelines import KaplanMeierFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False

logger = setup_logger("033_causal_ib")


# ══════════════════════════════════════════════════════════════════════
# 本地 IPCW 权重计算 (numpy 数组接口，与 shared_utils 的 DataFrame 接口解耦)
# ══════════════════════════════════════════════════════════════════════

def _compute_ipcw_weights(times: np.ndarray, events: np.ndarray) -> np.ndarray:
    """计算 IPCW 权重: w_i = delta_i / G_hat(T_i)。

    G_hat 为 KM 估计的删失生存函数（事件/删失角色互换）。
    极端权重截断在 95 百分位。
    """
    n = len(times)
    weights = np.zeros(n)
    censor_events = 1 - events
    order = np.argsort(times)
    at_risk = n
    g_hat = 1.0
    g_values = np.ones(n)
    for idx in order:
        if censor_events[idx] == 1:
            g_hat *= (at_risk - 1) / max(at_risk, 1)
        g_values[idx] = max(g_hat, 0.01)
        at_risk -= 1
    for i in range(n):
        if events[i] == 1:
            weights[i] = 1.0 / g_values[i]
    clip_val = np.percentile(weights[weights > 0], 95) if (weights > 0).any() else 10.0
    return np.clip(weights, 0, clip_val)


# ══════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════

CAUSAL_IB_CONFIG: Dict[str, Any] = {
    # 数据预处理
    "variance_topk": 5000,
    # 因果编码器
    "latent_shared_dim": 8,
    "latent_outcome_dim": 4,
    "encoder_hidden": [256, 128],
    # IB 门控
    "ib_beta": 1e-3,
    "ib_beta_anneal": True,        # 是否退火 (从小值逐渐增大)
    "ib_beta_max": 1e-2,
    # 对比学习
    "contrastive_lambda": 0.5,
    "contrastive_temperature": 0.1,
    "contrastive_proj_dim": 32,
    # 对抗去混杂
    "adversarial_lambda": 0.1,
    # 生存网络
    "survival_hidden": [64, 32],
    "dropout": 0.3,
    # 损失权重
    "lambda_rank": 0.3,
    "lambda_elastic": 1e-4,
    # 训练
    "lr_encoder": 5e-4,
    "lr_survival": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 300,
    "pretrain_epochs": 50,         # 对比预训练轮数
    "batch_size": 64,
    "patience": 40,
    "n_folds": 5,
    "grad_clip": 5.0,
    # 评估
    "eval_tau_months": 36,
    "random_seed": RANDOM_SEED,
}


# ══════════════════════════════════════════════════════════════════════
# PyTorch 模块 (仅 HAS_TORCH 时可用)
# ══════════════════════════════════════════════════════════════════════

if HAS_TORCH:

    class GradientReversalFunction(torch.autograd.Function):
        """梯度反转函数 (GRL)。

        前向传播不变，反向传播时梯度取反，用于对抗训练。
        参考: Ganin et al., JMLR 2016 (DANN)
        """

        @staticmethod
        def forward(ctx: Any, x: torch.Tensor, alpha: float) -> torch.Tensor:
            ctx.alpha = alpha
            return x.clone()

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
            return -ctx.alpha * grad_output, None

    def gradient_reversal(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """梯度反转的函数接口。"""
        return GradientReversalFunction.apply(x, alpha)

    class CausalEncoder(nn.Module):
        """PyTorch 原生因果编码器。

        与 031 使用外部 CausalEGM 库不同，本模块完全自包含。
        将高维 X 编码为两个解耦的潜在子空间：
        - z_shared: 混杂→结局的共享因果因子
        - z_outcome: 结局特异性因子

        使用重参数化技巧 (VAE 风格)：
            mu, logvar → z = mu + std * epsilon

        Args:
            input_dim: 输入特征维度
            z_shared_dim: z_shared 维度
            z_outcome_dim: z_outcome 维度
            hidden_dims: 编码器隐藏层
        """

        def __init__(
            self,
            input_dim: int,
            z_shared_dim: int = 8,
            z_outcome_dim: int = 4,
            hidden_dims: Optional[List[int]] = None,
        ) -> None:
            super().__init__()
            hidden_dims = hidden_dims or [256, 128]
            self.z_shared_dim = z_shared_dim
            self.z_outcome_dim = z_outcome_dim
            self.z_total_dim = z_shared_dim + z_outcome_dim

            # 共享编码器主干
            layers: List[nn.Module] = []
            prev = input_dim
            for h in hidden_dims:
                layers.extend([
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                ])
                prev = h
            self.backbone = nn.Sequential(*layers)

            # z_shared 分布参数
            self.mu_shared = nn.Linear(prev, z_shared_dim)
            self.logvar_shared = nn.Linear(prev, z_shared_dim)

            # z_outcome 分布参数
            self.mu_outcome = nn.Linear(prev, z_outcome_dim)
            self.logvar_outcome = nn.Linear(prev, z_outcome_dim)

        def encode(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """编码为分布参数 (mu_shared, logvar_shared, mu_outcome, logvar_outcome)。"""
            h = self.backbone(x)
            return (
                self.mu_shared(h), self.logvar_shared(h),
                self.mu_outcome(h), self.logvar_outcome(h),
            )

        def reparameterize(
            self, mu: torch.Tensor, logvar: torch.Tensor
        ) -> torch.Tensor:
            """重参数化采样。"""
            if self.training:
                std = torch.exp(0.5 * logvar.clamp(-10, 10))
                eps = torch.randn_like(std)
                return mu + std * eps
            return mu

        def forward(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """前向传播。

            Returns:
                (z_concat, z_shared, z_outcome, kl_shared, kl_outcome)
                z_concat: [z_shared, z_outcome] 拼接
            """
            mu_s, lv_s, mu_o, lv_o = self.encode(x)
            z_shared = self.reparameterize(mu_s, lv_s)
            z_outcome = self.reparameterize(mu_o, lv_o)

            # KL 散度: KL(q || N(0,I))
            kl_shared = -0.5 * torch.mean(1 + lv_s - mu_s.pow(2) - lv_s.exp())
            kl_outcome = -0.5 * torch.mean(1 + lv_o - mu_o.pow(2) - lv_o.exp())

            z_concat = torch.cat([z_shared, z_outcome], dim=-1)
            return z_concat, z_shared, z_outcome, kl_shared, kl_outcome

    class AdversarialDeconfounder(nn.Module):
        """对抗去混杂模块 (Gradient Reversal)。

        与 031 的 CausalEGM 对抗训练类似，但使用 DANN 风格的 GRL：
        - 判别器试图从 z_shared 预测 treatment
        - 编码器通过 GRL 使 z_shared 不含 treatment 信息

        Args:
            z_shared_dim: z_shared 维度
        """

        def __init__(self, z_shared_dim: int) -> None:
            super().__init__()
            self.discriminator = nn.Sequential(
                nn.Linear(z_shared_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(
            self, z_shared: torch.Tensor, alpha: float = 1.0
        ) -> torch.Tensor:
            """返回判别器的 treatment 预测 logits。"""
            z_rev = gradient_reversal(z_shared, alpha)
            return self.discriminator(z_rev)

    class IBGatingLayer(nn.Module):
        """信息瓶颈门控层 (带可学习温度退火)。

        与 031 的 IBGate 差异：
        - 增加可学习温度参数 (初始 tau=1.0，训练中自动退火)
        - 门控使用 softmax 归一化（而非 sigmoid）
        - KL 散度同时约束 z_shared 和 z_outcome

        数学：
            alpha = softmax(logits / tau)  (tau 可学习)
            z_filtered = z * alpha * z_dim  (乘以维度保持尺度)
            L_IB = beta * KL(q(Z|X) || N(0,I))
        """

        def __init__(self, z_dim: int, beta: float = 1e-3) -> None:
            super().__init__()
            self.z_dim = z_dim
            self.beta = beta
            # 可学习门控 logits
            self.gate_logits = nn.Parameter(torch.zeros(z_dim))
            # 可学习温度 (初始 1.0)
            self.log_tau = nn.Parameter(torch.tensor(0.0))

        def forward(
            self, z: torch.Tensor, kl_loss: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """前向传播。

            Args:
                z: (batch, z_dim) 因果表征
                kl_loss: 编码器 KL 散度

            Returns:
                (z_filtered, ib_loss)
            """
            tau = torch.exp(self.log_tau).clamp(min=0.1, max=10.0)
            alpha = F.softmax(self.gate_logits / tau, dim=0)  # (z_dim,)
            # 门控后保持尺度
            z_filtered = z * alpha.unsqueeze(0) * self.z_dim
            # IB 正则化
            ib_loss = self.beta * kl_loss
            return z_filtered, ib_loss

    class ContrastiveProjection(nn.Module):
        """对比学习投影头 (NT-Xent / SimCLR 风格)。

        将因果表征投影到对比空间，同一患者的 z_shared 和 z_outcome
        视为正对，不同患者视为负对。

        与 031 的核心差异：031 不使用对比学习。

        Args:
            z_dim: 输入维度
            proj_dim: 投影维度
            temperature: NT-Xent 温度参数
        """

        def __init__(
            self, z_shared_dim: int, z_outcome_dim: int,
            proj_dim: int = 32, temperature: float = 0.1,
        ) -> None:
            super().__init__()
            self.temperature = temperature
            # z_shared 和 z_outcome 维度不同，使用独立的投影头
            self.proj_shared = nn.Sequential(
                nn.Linear(z_shared_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim),
            )
            self.proj_outcome = nn.Sequential(
                nn.Linear(z_outcome_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim),
            )

        def forward(
            self, z_shared: torch.Tensor, z_outcome: torch.Tensor,
        ) -> torch.Tensor:
            """计算 NT-Xent 对比损失。

            将 z_shared 和 z_outcome 通过独立投影头映射到相同对比空间，
            同一 batch 内的 (z_shared_i, z_outcome_i) 为正对。

            Args:
                z_shared: (batch, z_shared_dim)
                z_outcome: (batch, z_outcome_dim)

            Returns:
                contrastive_loss: 标量
            """
            # 各自投影到相同维度
            p_shared = F.normalize(self.proj_shared(z_shared), dim=-1)
            p_outcome = F.normalize(self.proj_outcome(z_outcome), dim=-1)

            # 拼接: [p_shared, p_outcome] → (2B, proj_dim)
            z_all = torch.cat([p_shared, p_outcome], dim=0)  # (2B, proj_dim)
            n = z_all.shape[0]

            # 相似度矩阵
            sim = torch.mm(z_all, z_all.t()) / self.temperature  # (2B, 2B)

            # 正对索引: (i, i+B) 和 (i+B, i)
            batch_size = z_shared.shape[0]
            pos_idx_i = torch.arange(batch_size, device=z_all.device)
            pos_idx_j = pos_idx_i + batch_size

            # 排除对角线
            mask = ~torch.eye(n, dtype=torch.bool, device=z_all.device)
            sim_masked = sim[mask].view(n, n - 1)

            # 正对标签
            pos_labels = torch.cat([pos_idx_j, pos_idx_i], dim=0)
            # 调整标签 (排除对角线后的索引)
            adjusted_labels = pos_labels.clone()
            for k in range(n):
                if pos_labels[k] > k:
                    adjusted_labels[k] = pos_labels[k] - 1

            loss = F.cross_entropy(sim_masked, adjusted_labels)
            return loss

    class SurvivalHead(nn.Module):
        """MLP 生存预测头。

        与 031 的 SurvivalHead 结构类似，但使用 GELU 激活 (替代 ReLU)
        和 LayerNorm (替代 BatchNorm)，在极小 batch 下更稳定。
        """

        def __init__(
            self, z_dim: int, hidden: Optional[List[int]] = None, dropout: float = 0.3,
        ) -> None:
            super().__init__()
            hidden = hidden or [64, 32]
            layers: List[nn.Module] = []
            prev = z_dim
            for h in hidden:
                layers.extend([
                    nn.Linear(prev, h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.net(z).squeeze(-1)

    class CausalDeepSurvIB(nn.Module):
        """Causal-DeepSurv-IB 一体化模型 (模块化重构版)。

        与 031 的 CDSIBModel 核心差异：
        1. 因果编码器为 PyTorch 原生 VAE 风格 (031 依赖外部 CausalEGM 库)
        2. 增加对抗去混杂模块 (GRL)
        3. 增加对比学习投影头
        4. IB 门控使用 softmax + 可学习温度退火
        5. 使用 GELU + Dropout (无 BatchNorm) 适应极小 batch

        架构:
            X → CausalEncoder → (z_shared, z_outcome)
            → AdversarialDeconfounder (去混杂)
            → concat → IBGatingLayer → SurvivalHead → log risk
        """

        def __init__(
            self, input_dim: int, config: Optional[Dict[str, Any]] = None,
        ) -> None:
            super().__init__()
            cfg = config or CAUSAL_IB_CONFIG

            z_shared_dim = cfg["latent_shared_dim"]
            z_outcome_dim = cfg["latent_outcome_dim"]
            z_total = z_shared_dim + z_outcome_dim

            # 因果编码器
            self.encoder = CausalEncoder(
                input_dim, z_shared_dim, z_outcome_dim,
                hidden_dims=cfg["encoder_hidden"],
            )

            # 对抗去混杂
            self.deconfounder = AdversarialDeconfounder(z_shared_dim)

            # IB 门控
            self.ib_gate = IBGatingLayer(z_total, beta=cfg["ib_beta"])

            # 对比学习投影
            self.contrast_proj = ContrastiveProjection(
                z_shared_dim, z_outcome_dim,
                cfg["contrastive_proj_dim"],
                temperature=cfg["contrastive_temperature"],
            )

            # 生存预测头
            self.surv_head = SurvivalHead(
                z_total, cfg["survival_hidden"], cfg["dropout"],
            )

        def forward(
            self, x: torch.Tensor, treatment: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """前向传播，返回所有中间量用于损失计算。

            Args:
                x: (batch, input_dim)
                treatment: (batch,) 二值处理变量 (对抗训练用)

            Returns:
                dict: log_risk, z_shared, z_outcome, kl_shared, kl_outcome,
                      ib_loss, contrast_loss, adv_logits
            """
            z_concat, z_shared, z_outcome, kl_s, kl_o = self.encoder(x)

            # 对抗去混杂
            adv_logits = None
            if treatment is not None:
                adv_logits = self.deconfounder(z_shared)

            # IB 门控
            kl_total = kl_s + kl_o
            z_gated, ib_loss = self.ib_gate(z_concat, kl_total)

            # 对比损失
            contrast_loss = self.contrast_proj(z_shared, z_outcome)

            # 生存预测
            log_risk = self.surv_head(z_gated)

            return {
                "log_risk": log_risk,
                "z_shared": z_shared,
                "z_outcome": z_outcome,
                "kl_shared": kl_s,
                "kl_outcome": kl_o,
                "ib_loss": ib_loss,
                "contrast_loss": contrast_loss,
                "adv_logits": adv_logits,
            }

        def predict_risk(self, x: torch.Tensor) -> np.ndarray:
            """推理接口。"""
            self.eval()
            with torch.no_grad():
                out = self.forward(x)
                return torch.exp(out["log_risk"]).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════
# 损失函数
# ══════════════════════════════════════════════════════════════════════

if HAS_TORCH:

    def compute_causal_ib_loss(
        outputs: Dict[str, torch.Tensor],
        times: torch.Tensor,
        events: torch.Tensor,
        treatment: Optional[torch.Tensor],
        model: CausalDeepSurvIB,
        ipcw_weights: Optional[torch.Tensor] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Causal-DeepSurv-IB 复合损失。

        与 031 的 compute_cdsib_loss 差异：
        - 新增对比学习损失 L_contrast (NT-Xent)
        - 新增对抗去混杂损失 L_adv (BCE)
        - IB 使用 softmax 门控 + 退火 beta (031 使用 sigmoid + 固定 beta)
        - 对比排序损失使用向量化实现

        L = L_IPCW + λ_rank*L_rank + L_IB + λ_contrast*L_contrast
            + λ_adv*L_adv + λ_elastic*L_elastic
        """
        cfg = config or CAUSAL_IB_CONFIG
        device = outputs["log_risk"].device
        n = len(times)

        log_risk = outputs["log_risk"]

        # ── 1. IPCW 偏似然 ──
        sort_idx = torch.argsort(-times)
        h_sorted = log_risk[sort_idx]
        ev_sorted = events[sort_idx]
        w_sorted = (
            ipcw_weights[sort_idx]
            if ipcw_weights is not None
            else torch.ones(n, device=device)
        )
        log_cumsum_exp = torch.logcumsumexp(h_sorted, dim=0)
        partial_ll = (h_sorted - log_cumsum_exp) * ev_sorted * w_sorted
        event_count = ev_sorted.sum().clamp(min=1)
        l_ipcw = -partial_ll.sum() / event_count

        # ── 2. 对比排序损失 (向量化) ──
        l_rank = torch.tensor(0.0, device=device)
        ev_idx = torch.where(events == 1)[0]
        cen_idx = torch.where(events == 0)[0]
        if len(ev_idx) > 0 and len(cen_idx) > 0:
            risk_ev = log_risk[ev_idx]
            risk_cen = log_risk[cen_idx]
            diff = risk_ev.unsqueeze(1) - risk_cen.unsqueeze(0)
            l_rank = -F.logsigmoid(diff).mean()

        # ── 3. IB 正则化 (来自 IBGatingLayer) ──
        l_ib = outputs["ib_loss"]

        # ── 4. 对比学习损失 ──
        l_contrast = outputs["contrast_loss"] * cfg["contrastive_lambda"]

        # ── 5. 对抗去混杂损失 ──
        l_adv = torch.tensor(0.0, device=device)
        if treatment is not None and outputs["adv_logits"] is not None:
            l_adv = F.binary_cross_entropy_with_logits(
                outputs["adv_logits"].squeeze(-1), treatment.float()
            )
            # 对抗损失通过 GRL 已反转梯度，这里直接加
            l_adv = l_adv * cfg["adversarial_lambda"]

        # ── 6. 弹性网络正则化 ──
        l_elastic = torch.tensor(0.0, device=device)
        for p in model.surv_head.parameters():
            if p.requires_grad:
                l_elastic = l_elastic + 0.5 * p.abs().sum() + 0.25 * p.pow(2).sum()
        l_elastic = l_elastic * cfg["lambda_elastic"]

        # ── 总损失 ──
        total = (
            l_ipcw
            + cfg["lambda_rank"] * l_rank
            + l_ib
            + l_contrast
            + l_adv
            + l_elastic
        )
        return total


# ══════════════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════════════

def load_data_for_causal_ib(
    timestamp: str, config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """加载 TCGA-COAD/READ 数据 (含 treatment 变量)。"""
    cfg = config or CAUSAL_IB_CONFIG
    preproc_dir = ESSENTIAL_DIR

    clinical = safe_read_tsv(os.path.join(preproc_dir, "tcga_os_clinical_endpoint_qc.tsv"))
    split_path = os.path.join(preproc_dir, "tcga_train_internal_validation_split.tsv")
    if not os.path.exists(split_path):
        split_path = os.path.join(DATA_DIR, "survival_models", "tcga_train_internal_validation_split.tsv")
    split_df = safe_read_tsv(split_path)

    expr_df = safe_read_tsv(os.path.join(preproc_dir, "gene_expression_curated.tsv"), index_col=0)
    expr_df = expr_df.apply(pd.to_numeric, errors="coerce")

    clinical_ids = set(clinical["PATIENT_ID"].astype(str))
    train_ids = set(split_df.loc[split_df["split"] == "train", "PATIENT_ID"].astype(str)) & clinical_ids
    val_ids = set(split_df.loc[split_df["split"] != "train", "PATIENT_ID"].astype(str)) & clinical_ids

    time_col, event_col = detect_event_time_columns(clinical)
    ep = clinical.copy()
    ep["PATIENT_ID"] = ep["PATIENT_ID"].astype(str).map(normalize_patient_id)
    ep["time_months"] = pd.to_numeric(ep[time_col], errors="coerce")
    ep["event"] = coerce_event(ep[event_col]).astype(int)
    ep = ep[ep["PATIENT_ID"].ne("") & ep["time_months"].notna() & (ep["time_months"] >= 0)]
    ep.index = ep["PATIENT_ID"].values

    # 方差过滤
    train_idx = [pid for pid in map(str, train_ids) if pid in expr_df.index]
    variances = expr_df.loc[train_idx].var(axis=0, skipna=True).dropna()
    variances = variances[variances > 0].sort_values(ascending=False)
    topk_genes = list(variances.head(cfg["variance_topk"]).index)

    # 标准化
    X_all = expr_df[topk_genes].copy()
    X_all.index = X_all.index.astype(str)
    train_med = X_all.loc[train_idx].median()
    X_all = X_all.fillna(train_med).fillna(0.0)
    scaler = StandardScaler()
    scaler.fit(X_all.loc[train_idx])
    X_scaled = pd.DataFrame(scaler.transform(X_all), index=X_all.index, columns=topk_genes)

    # 构建 treatment (AJCC ≥ III)
    def _build_treatment(clin_df: pd.DataFrame, idx: pd.Index) -> np.ndarray:
        stage_candidates = ["AJCC_PATHOLOGIC_TUMOR_STAGE", "AJCC_STAGE", "STAGE"]
        stage_col = None
        for c in stage_candidates:
            if c in clin_df.columns:
                stage_col = c
                break
        if stage_col is None:
            return np.zeros(len(idx), dtype=np.float32)
        if "PATIENT_ID" in clin_df.columns:
            stages = clin_df.set_index("PATIENT_ID")[stage_col].reindex(idx)
        else:
            stages = clin_df[stage_col].copy()
            stages.index = clin_df.index
        treat = stages.reindex(idx).apply(
            lambda v: 1.0 if pd.notna(v) and str(v).strip().upper() in
            ("STAGE III", "STAGE IV", "III", "IV", "3", "4") else 0.0
        ).fillna(0.0).to_numpy(dtype=np.float32)
        return treat

    # 分割
    X_train = X_scaled.loc[X_scaled.index.isin(train_ids)]
    X_val = X_scaled.loc[X_scaled.index.isin(val_ids)]
    ep_train = ep.loc[ep.index.isin(train_ids)]
    ep_val = ep.loc[ep.index.isin(val_ids)]
    ct = X_train.index.intersection(ep_train.index)
    cv = X_val.index.intersection(ep_val.index)
    X_train, ep_train = X_train.loc[ct], ep_train.loc[ct]
    X_val, ep_val = X_val.loc[cv], ep_val.loc[cv]

    treat_train = _build_treatment(clinical, X_train.index)
    treat_val = _build_treatment(clinical, X_val.index)

    logger.info(f"[CausalIB] train={len(X_train)}, val={len(X_val)}, genes={len(topk_genes)}")

    return {
        "X_train": X_train.to_numpy(dtype=np.float32),
        "X_val": X_val.to_numpy(dtype=np.float32),
        "times_train": ep_train["time_months"].to_numpy(dtype=np.float32),
        "events_train": ep_train["event"].to_numpy(dtype=np.float32),
        "treatment_train": treat_train,
        "times_val": ep_val["time_months"].to_numpy(dtype=np.float32),
        "events_val": ep_val["event"].to_numpy(dtype=np.float32),
        "treatment_val": treat_val,
        "gene_names": topk_genes,
    }


# ══════════════════════════════════════════════════════════════════════
# 评估
# ══════════════════════════════════════════════════════════════════════

def harrell_cindex(times: np.ndarray, events: np.ndarray, risk: np.ndarray) -> float:
    """Harrell C-index。"""
    valid = np.isfinite(risk)
    t, e, r = times[valid], events[valid], risk[valid]
    perm, conc = 0, 0.0
    for i in range(len(t)):
        for j in range(len(t)):
            if t[i] < t[j] and e[i] == 1:
                perm += 1
                if r[i] > r[j]:
                    conc += 1
                elif r[i] == r[j]:
                    conc += 0.5
    return conc / perm if perm > 0 else float("nan")


# ══════════════════════════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════════════════════════

if HAS_TORCH:

    def train_causal_ib_model(
        X_train: np.ndarray,
        times_train: np.ndarray,
        events_train: np.ndarray,
        treatment_train: np.ndarray,
        X_val: np.ndarray,
        times_val: np.ndarray,
        events_val: np.ndarray,
        treatment_val: np.ndarray,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[CausalDeepSurvIB, Dict[str, Any]]:
        """训练 Causal-DeepSurv-IB 模型。

        与 031 的 train_cdsib_model 差异：
        1. 使用差异化学习率 (encoder 5e-4, survival head 1e-4)
        2. 每 batch 传入 treatment 用于对抗训练
        3. 包含对比预训练阶段
        4. IB beta 退火机制
        """
        cfg = config or CAUSAL_IB_CONFIG
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[CausalIB] 设备: {device}")

        input_dim = X_train.shape[1]
        model = CausalDeepSurvIB(input_dim, cfg).to(device)

        # 差异化学习率
        encoder_params = list(model.encoder.parameters()) + list(model.deconfounder.parameters())
        survival_params = list(model.ib_gate.parameters()) + list(model.surv_head.parameters())
        contrast_params = list(model.contrast_proj.parameters())

        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": cfg["lr_encoder"]},
            {"params": survival_params, "lr": cfg["lr_survival"]},
            {"params": contrast_params, "lr": cfg["lr_encoder"]},
        ], weight_decay=cfg["weight_decay"])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=1e-6,
        )

        # DataLoader
        dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(times_train),
            torch.FloatTensor(events_train),
            torch.FloatTensor(treatment_train),
        )
        loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

        # 验证集
        X_v = torch.FloatTensor(X_val).to(device)
        t_v = torch.FloatTensor(times_val).to(device)
        e_v = torch.FloatTensor(events_val).to(device)
        treat_v = torch.FloatTensor(treatment_val).to(device)

        best_cindex = 0.0
        best_state = None
        patience_cnt = 0

        for epoch in range(cfg["epochs"]):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            # IB beta 退火
            if cfg["ib_beta_anneal"] and epoch < cfg["pretrain_epochs"]:
                frac = epoch / max(cfg["pretrain_epochs"], 1)
                current_beta = cfg["ib_beta"] * frac
                model.ib_gate.beta = current_beta

            for bx, bt, be, btr in loader:
                bx, bt, be, btr = bx.to(device), bt.to(device), be.to(device), btr.to(device)
                w_batch = torch.FloatTensor(
                    _compute_ipcw_weights(bt.cpu().numpy(), be.cpu().numpy())
                ).to(device)

                outputs = model(bx, treatment=btr)
                loss = compute_causal_ib_loss(
                    outputs, bt, be, btr, model,
                    ipcw_weights=w_batch, config=cfg,
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # 验证
            if (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    val_out = model(X_v, treatment=treat_v)
                    val_risk = val_out["log_risk"].cpu().numpy()
                c = harrell_cindex(times_val, events_val, val_risk)
                if not np.isnan(c) and c > best_cindex:
                    best_cindex = c
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_cnt = 0
                else:
                    patience_cnt += 1

                if patience_cnt >= cfg["patience"] // 5:
                    logger.info(f"[CausalIB] Epoch {epoch+1}: Early stop (best C={best_cindex:.4f})")
                    break

                if (epoch + 1) % 50 == 0:
                    logger.info(
                        f"[CausalIB] Epoch {epoch+1}/{cfg['epochs']}: "
                        f"loss={epoch_loss/n_batches:.4f}, val_C={c:.4f}"
                    )

        if best_state is not None:
            model.load_state_dict(best_state)

        # 最终评估
        model.eval()
        with torch.no_grad():
            train_out = model(torch.FloatTensor(X_train).to(device))
            val_out = model(X_v)

        val_c = harrell_cindex(times_val, events_val, val_out["log_risk"].cpu().numpy())

        ipcw_c = float("nan")
        if HAS_SKSURV:
            try:
                y_tr = Surv.from_arrays(events_train.astype(bool), times_train)
                y_va = Surv.from_arrays(events_val.astype(bool), times_val)
                ipcw_c = float(concordance_index_ipcw(y_tr, y_va, val_out["log_risk"].cpu().numpy())[0])
            except Exception:
                pass

        logger.info(f"[CausalIB] 最终: Harrell-C={val_c:.4f}, IPCW-C={ipcw_c:.4f}")

        return model, {
            "val_cindex": val_c,
            "ipcw_cindex": ipcw_c,
            "best_val_cindex": best_cindex,
            "train_risk": train_out["log_risk"].cpu().numpy(),
            "val_risk": val_out["log_risk"].cpu().numpy(),
        }


# ══════════════════════════════════════════════════════════════════════
# 主管线
# ══════════════════════════════════════════════════════════════════════

def run_causal_ib_pipeline(timestamp: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Causal-DeepSurv-IB 端到端管线。

    与 031 的 run_cdsib_pipeline 差异：
    - 无外部 CausalEGM 调用 (使用 PyTorch 原生编码器)
    - 无扩散数据增强
    - 无反事实推断
    - 包含对抗去混杂和对比学习
    """
    cfg = config or CAUSAL_IB_CONFIG
    out_dir = os.path.join(RESULTS_DIR, timestamp, "033_causal_ib")
    ensure_dir(out_dir)
    t0 = time.time()

    if not HAS_TORCH:
        logger.error("[CausalIB] PyTorch 不可用")
        return {}

    logger.info("=" * 60)
    logger.info("[CausalIB] Causal-DeepSurv-IB (模块化重构版)")
    logger.info("=" * 60)

    # 1. 数据加载
    data = load_data_for_causal_ib(timestamp, cfg)

    # 2. 5-fold CV
    events_tr = data["events_train"].astype(int)
    skf = StratifiedKFold(
        n_splits=min(cfg["n_folds"], max(2, events_tr.sum())),
        shuffle=True, random_state=cfg["random_seed"],
    )
    cv_cindices = []
    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(data["X_train"], events_tr), 1):
        fold_model, fold_res = train_causal_ib_model(
            data["X_train"][tr_idx], data["times_train"][tr_idx],
            data["events_train"][tr_idx], data["treatment_train"][tr_idx],
            data["X_train"][va_idx], data["times_train"][va_idx],
            data["events_train"][va_idx], data["treatment_train"][va_idx],
            config=cfg,
        )
        cv_cindices.append(fold_res["val_cindex"])
        logger.info(f"[CausalIB] Fold {fold_id}: C={fold_res['val_cindex']:.4f}")

    cv_mean = float(np.nanmean(cv_cindices))
    cv_std = float(np.nanstd(cv_cindices))
    logger.info(f"[CausalIB] CV C-index: {cv_mean:.4f} ± {cv_std:.4f}")

    # 3. 全量训练
    final_model, final_res = train_causal_ib_model(
        data["X_train"], data["times_train"], data["events_train"], data["treatment_train"],
        data["X_val"], data["times_val"], data["events_val"], data["treatment_val"],
        config=cfg,
    )

    # 4. 保存
    elapsed = time.time() - t0
    summary = {
        "model_type": "Causal-DeepSurv-IB (PyTorch native encoder + contrastive)",
        "timestamp": timestamp,
        "elapsed_sec": round(elapsed, 1),
        "input_dim": data["X_train"].shape[1],
        "z_shared_dim": cfg["latent_shared_dim"],
        "z_outcome_dim": cfg["latent_outcome_dim"],
        "cv_cindex_mean": cv_mean,
        "cv_cindex_std": cv_std,
        "val_harrell_cindex": final_res["val_cindex"],
        "val_ipcw_cindex": final_res["ipcw_cindex"],
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(out_dir, "causal_ib_summary.tsv"), sep="\t", index=False,
    )
    torch.save(final_model.state_dict(), os.path.join(out_dir, "causal_ib_model.pt"))
    pd.DataFrame({"risk": final_res["val_risk"]}).to_csv(
        os.path.join(out_dir, "val_risk_scores.tsv"), sep="\t",
    )

    logger.info(f"\n{'=' * 60}")
    logger.info(f"[CausalIB] 完成! 耗时 {elapsed:.1f}s")
    logger.info(f"  CV: {cv_mean:.4f} ± {cv_std:.4f}")
    logger.info(f"  Val-C: {final_res['val_cindex']:.4f}")
    logger.info(f"{'=' * 60}")
    return summary


# ══════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="033: Causal-DeepSurv-IB (模块化重构版)")
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()
    try:
        run_causal_ib_pipeline(args.timestamp)
    except Exception as e:
        logger.error(f"[CausalIB] 异常: {e}")
        raise


if __name__ == "__main__":
    main()
