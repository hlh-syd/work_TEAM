"""sparse_gate.py — Hard-Concrete L0 SparseFeatureGate。

实现：
    - HardConcreteSparseGate(nn.Module)：Hard-Concrete 分布的 L0 稀疏门控
        - forward: g_tilde = clamp[0,1](sigma((log u - log(1-u) + a)/tau)(zeta-gamma)+gamma)
        - l0_penalty(): sum P(g_tilde > 0)
        - gate_probabilities(): 每个基因被选概率
        - initialize_from_prior(frequencies): logit(clip(pi, 0.05, 0.95))
    - compute_gate_loss(): L0 + count + group 三项
    - jaccard_stability(): 跨 fold 基因选择稳定性
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


def _calibrate_log_alpha(
    n_features: int,
    target_active: int,
    temperature: float,
    gamma: float,
    zeta: float,
    tol: float = 1e-3,
    max_iter: int = 100,
) -> float:
    """二分搜索校准 log_alpha 初始值，使期望激活数 ≈ target_active。

    P(g > 0) = sigmoid(log_alpha - temperature * log(-gamma / zeta))
    期望激活数 = n_features * P(g > 0) = target_active
    => P(g > 0) = target_active / n_features
    => log_alpha = logit(target_active / n_features) + temperature * log(-gamma / zeta)

    这里用闭式解直接计算（比二分搜索更精确），但保留函数名以兼容 035 接口。
    """
    import math
    target_prob = float(target_active) / float(n_features)
    target_prob = min(max(target_prob, 1e-4), 1.0 - 1e-4)
    # logit(p) = log(p / (1-p))
    logit_p = math.log(target_prob / (1.0 - target_prob))
    # log_alpha = logit(p) + temperature * log(-gamma / zeta)
    if gamma >= 0:
        return logit_p
    ratio = -gamma / zeta
    log_ratio = math.log(max(ratio, 1e-12))
    return logit_p + temperature * log_ratio


class HardConcreteSparseGate(nn.Module):
    """Hard-Concrete L0 稀疏门控（非 sigmoid + top-k）。

    参考：Louizos et al. (2018) "Learning Sparse Neural Networks through L0 Regularization"

    与 035 对齐：支持 target_active 参数，通过二分搜索校准 log_alpha 初始值，
    使期望激活基因数 ≈ target_active（避免零初始化导致每基因被选概率≈0.5）。
    """

    def __init__(
        self,
        n_features: int,
        temperature: float = 2.0 / 3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
        target_active: Optional[int] = None,
        prior_log_alpha: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.n_features = n_features
        self.temperature = temperature
        self.gamma = gamma
        self.zeta = zeta
        # log_alpha 为可学习参数
        # 若提供 target_active，用二分搜索校准初始 log_alpha，
        # 使期望激活数 ≈ target_active；否则保持零初始化（每基因≈0.5）
        if target_active is not None and 0 < target_active < n_features:
            init_log_alpha = _calibrate_log_alpha(
                n_features=n_features,
                target_active=target_active,
                temperature=temperature,
                gamma=gamma,
                zeta=zeta,
            )
            self.log_alpha = nn.Parameter(torch.full((n_features,), float(init_log_alpha)))
        else:
            self.log_alpha = nn.Parameter(torch.zeros(n_features))

        # P1-1: 因果先验对齐目标（非可学习 buffer，仅用于 prior_alignment_loss）
        # 若提供 prior_log_alpha，训练中通过 gate_prior_alignment_loss 持续拉向该目标
        if prior_log_alpha is not None:
            self.register_buffer(
                "prior_log_alpha",
                prior_log_alpha.to(self.log_alpha.device).to(self.log_alpha.dtype),
            )
        else:
            self.register_buffer("prior_log_alpha", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：x'_j = x_j * g_tilde_j。

        Args:
            x: [B, P]

        Returns:
            [B, P] 门控后的输入
        """
        if self.training:
            # 训练：采样 u ~ Uniform(0, 1)
            u = torch.zeros_like(x).uniform_(1e-8, 1.0 - 1e-8)
            # s = sigmoid((log u - log(1-u) + log_alpha) / temperature)
            s = torch.sigmoid(
                (torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.temperature
            )
            # s_bar = s * (zeta - gamma) + gamma
            s_bar = s * (self.zeta - self.gamma) + self.gamma
            # clip 到 [0, 1]
            z = torch.clamp(s_bar, 0.0, 1.0)
        else:
            # 推理：使用确定性门控 P(g > 0)
            z = self._deterministic_gate().unsqueeze(0).expand_as(x)

        return x * z

    def _deterministic_gate(self) -> torch.Tensor:
        """推理时使用的确定性门控值。"""
        # P(g_tilde > 0) 的解析近似
        prob = self.gate_probabilities()
        return (prob > 0.5).float()

    def gate_probabilities(self) -> torch.Tensor:
        """返回每个基因被选概率 P(g_tilde > 0)。

        公式：P(g > 0) = sigmoid(log_alpha - temperature * log(-gamma / zeta))
        """
        # 当 gamma < 0, zeta > 1 时
        if self.gamma >= 0:
            return torch.sigmoid(self.log_alpha)
        ratio = -self.gamma / self.zeta
        log_ratio = torch.log(torch.tensor(ratio, dtype=torch.float, device=self.log_alpha.device))
        prob = torch.sigmoid(self.log_alpha - self.temperature * log_ratio)
        return torch.clamp(prob, 0.0, 1.0)

    def l0_penalty(self) -> torch.Tensor:
        """L0 正则：sum P(g_tilde > 0)。"""
        return self.gate_probabilities().sum()

    def prior_alignment_loss(self) -> torch.Tensor:
        """P1-1: 因果先验对齐损失。

        L_align = mean((sigmoid(log_alpha) - sigmoid(prior_log_alpha))^2)

        当 prior_log_alpha 为 None 时返回 0。
        正值 prior_log_alpha → 拉向高激活；负值 → 拉向低激活。
        """
        if self.prior_log_alpha is None:
            return torch.tensor(0.0, device=self.log_alpha.device)
        return (
            (torch.sigmoid(self.log_alpha) - torch.sigmoid(self.prior_log_alpha)) ** 2
        ).mean()

    def initialize_from_prior(self, frequencies: np.ndarray) -> None:
        """用频率向量初始化 log_alpha。

        Args:
            frequencies: [P] 数组，值域 [0, 1]，表示基因被选频率
        """
        with torch.no_grad():
            freq = np.clip(np.asarray(frequencies, dtype=np.float32), 0.05, 0.95)
            logit = np.log(freq / (1.0 - freq)).astype(np.float32)
            self.log_alpha.copy_(torch.from_numpy(logit))

    def active_count(self) -> int:
        """推理时激活的基因数。"""
        return int((self.gate_probabilities() > 0.5).sum().item())


def compute_gate_loss(
    gate: HardConcreteSparseGate,
    target_k: int,
    lambda_l0: float = 2e-4,
    lambda_count: float = 2e-3,
    lambda_group: float = 1e-4,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """计算门控损失 L_gate = lambda_L0 * L_L0/P + lambda_count * ((L_L0 - K)/P)^2 + lambda_group * sum_g ||W_g||_2。

    Args:
        gate: HardConcreteSparseGate
        target_k: 期望激活数 K
        lambda_l0, lambda_count, lambda_group: 权重

    Returns:
        (loss, components_dict)
    """
    P = gate.n_features
    l0 = gate.l0_penalty()

    # L0 项
    l0_term = lambda_l0 * l0 / max(P, 1)
    # count 项：惩罚激活数偏离 K
    count_term = lambda_count * ((l0 - target_k) / max(P, 1)) ** 2
    # group 项：简化为 0（无 group 结构时）
    group_term = torch.tensor(0.0, device=l0.device)

    total = l0_term + count_term + group_term

    components = {
        "l0_penalty": float(l0.item()),
        "l0_term": float(l0_term.item()),
        "count_term": float(count_term.item()),
        "active_prob_mean": float((l0 / max(P, 1)).item()),
    }
    return total, components


def jaccard_stability(
    selected_sets: Sequence[Sequence[str]],
) -> Dict[str, float]:
    """计算跨 fold 基因选择的 Jaccard 稳定性。

    Args:
        selected_sets: 每折选中的基因列表

    Returns:
        {"mean_jaccard": float, "std_jaccard": float, "pair_count": int}
    """
    if len(selected_sets) < 2:
        return {"mean_jaccard": 1.0, "std_jaccard": 0.0, "pair_count": 0}

    sets = [set(s) for s in selected_sets]
    jaccards: List[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            if len(union) == 0:
                continue
            inter = sets[i] & sets[j]
            jaccards.append(len(inter) / len(union))

    if not jaccards:
        return {"mean_jaccard": 1.0, "std_jaccard": 0.0, "pair_count": 0}

    return {
        "mean_jaccard": float(np.mean(jaccards)),
        "std_jaccard": float(np.std(jaccards)),
        "pair_count": len(jaccards),
    }


def fold_gate_prior(
    train_x: np.ndarray,
    train_event: np.ndarray,
    train_time: np.ndarray,
    n_bootstraps: int = 30,
    alpha: float = 0.02,
    max_iter: int = 2000,
    seed: int = 0,
    target_active: int = 128,
) -> np.ndarray:
    """在 inner-train 内 bootstrap 拟合 elastic-net Cox，记录基因被选频率。

    与 035 的 `_bootstrap_gate_prior` 实现完全对齐：
        1. 主路径：`CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=[alpha], max_iter=max_iter, tol=1e-3)`
        2. Fallback 1：`CoxPHSurvivalAnalysis(alpha=1.0, n_iter=200)`（保留删失结构）
        3. Fallback 2：单变量相关（pseudo-risk = event / max(time, 1)）
        4. 每个 bootstrap 选 top-K 基因（K=target_active），用平滑频率 (selected+0.5)/(successful+1)
        5. 所有样本均参与 Cox 拟合（保留删失结构），不再仅用事件样本

    Args:
        train_x: [N, P] 训练折表达矩阵
        train_event: [N] 事件标记
        train_time: [N] 生存时间
        n_bootstraps: bootstrap 次数
        alpha: Coxnet alpha（L1+L2 混合，单点 alpha 网格）
        max_iter: 最大迭代
        seed: 随机种子
        target_active: top-K 选基因数（与 035 的 active_gene_target 一致）

    Returns:
        frequencies: [P] 每个基因被选频率，值域 [0, 1]
    """
    import warnings
    from sksurv.util import Surv
    from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis

    rng = np.random.RandomState(seed)
    n, p = train_x.shape
    selected_count = np.zeros(p, dtype=np.float64)
    successful = 0

    for b in range(max(1, n_bootstraps)):
        # 患者级 bootstrap
        idx = rng.integers(0, n, size=n) if hasattr(rng, "integers") else rng.choice(n, size=n, replace=True)
        x_boot = train_x[idx]
        e_boot = train_event[idx].astype(bool)
        t_boot = train_time[idx]

        # 至少 2 个事件
        if e_boot.sum() < 2:
            continue

        coefficient: Optional[np.ndarray] = None
        outcome = Surv.from_arrays(e_boot, t_boot)

        # 主路径：CoxnetSurvivalAnalysis
        try:
            estimator = CoxnetSurvivalAnalysis(
                l1_ratio=0.5,
                alphas=[float(alpha)],
                max_iter=int(max_iter),
                tol=1e-3,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(x_boot, outcome)
            coef_arr = np.asarray(estimator.coef_[:, 0], dtype=float)
            if np.isfinite(coef_arr).all():
                coefficient = np.abs(coef_arr)
        except Exception:
            coefficient = None

        # Fallback 1：CoxPHSurvivalAnalysis（Ridge Cox）
        if coefficient is None or coefficient.max() == 0:
            try:
                ridge = CoxPHSurvivalAnalysis(alpha=1.0, n_iter=200)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ridge.fit(x_boot, outcome)
                ridge_coef = np.asarray(ridge.coef_, dtype=float)
                if np.isfinite(ridge_coef).all() and ridge_coef.max() > 0:
                    coefficient = np.abs(ridge_coef)
                else:
                    coefficient = None
            except Exception:
                coefficient = None

        # Fallback 2：单变量相关（pseudo-risk = event / max(time, 1)）
        if coefficient is None or coefficient.max() == 0:
            pseudo_risk = e_boot.astype(float) / np.maximum(t_boot, 1.0)
            centered_x = x_boot - x_boot.mean(axis=0, keepdims=True)
            centered_y = pseudo_risk - pseudo_risk.mean()
            coefficient = np.abs(centered_x.T @ centered_y)

        if coefficient is None or not np.isfinite(coefficient).all() or coefficient.max() == 0:
            continue

        # top-K 选择
        k = min(int(target_active), p)
        top_idx = np.argsort(-coefficient, kind="stable")[:k]
        selected_count[top_idx] += 1.0
        successful += 1

    if successful == 0:
        # 全失败时退化为均匀先验
        frequencies = np.full(p, target_active / p, dtype=np.float32)
    else:
        # 平滑频率：(selected + 0.5) / (successful + 1)
        frequencies = ((selected_count + 0.5) / (successful + 1.0)).astype(np.float32)

    return frequencies
