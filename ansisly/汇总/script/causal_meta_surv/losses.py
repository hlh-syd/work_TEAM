"""losses.py — compute_cmib_loss 统一损失接口。

提供 11 项损失：
    1. cox_efron_loss：处理并列事件的 Efron 部分似然
    2. rank_loss：仅可比较对 (delta_i=1 and t_i<t_j)
    3. ib_loss：free-bits kappa
    4. gate_loss：L0 + count + group
    5. elastic_loss：归一化 L1+L2（bias/norm 不惩罚）
    6. distill_loss：stop-gradient
    7. bal_loss：加权 MMD/CORAL
    8. domain_loss：CE with GRL
    9. orth_loss：Frobenius 正交
    10. cons_loss：一致性
    11. anchor_loss：锚定

统一接口：compute_cmib_loss(output, batch, config, model, theta0) -> CMIBLoss
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import CMIBConfig, CMIBLoss, CMIBOutput, SurvivalRecordBatch
from .fusion import orth_loss
from .sparse_gate import HardConcreteSparseGate, compute_gate_loss


# ──────────────────────────────────────────────────────────────────────
# 1. Cox-Efron 部分似然（完全可微 PyTorch 实现）
# ──────────────────────────────────────────────────────────────────────
def cox_efron_loss(
    eta: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    """Negative Cox partial log-likelihood with exact Efron tie handling.

    完全可微实现（参考 035 efron_cox_loss），使用 logcumsumexp 进行
    数值稳定的前缀风险计算，梯度正确回传到 eta。

    Args:
        eta: [B] 模型输出 log-risk
        time: [B] 生存时间（>0）
        event: [B] 事件标记（0/1）

    Returns:
        scalar tensor（可微，可 .backward()）
    """
    eta = eta.reshape(-1)
    observed_time = time.reshape(-1)
    observed_event = event.reshape(-1)

    if eta.numel() == 0:
        return torch.tensor(0.0, device=eta.device)

    observed_event = observed_event.bool()
    n_events = int(observed_event.sum().item())
    if n_events == 0:
        return torch.tensor(0.0, device=eta.device)

    # 按时间降序排序（大→小），确保风险集为前缀
    order = torch.argsort(observed_time, descending=True, stable=True)
    sorted_time = observed_time[order]
    sorted_event = observed_event[order]
    sorted_eta = eta[order]

    # logcumsumexp: log(sum_{j>=i} exp(eta_j)) 从前缀构建
    log_prefix_risk = torch.logcumsumexp(sorted_eta, dim=0)

    # 按时间分组处理并列事件
    _, counts = torch.unique_consecutive(sorted_time, return_counts=True)

    partial_log_likelihood = sorted_eta.sum() * 0.0  # 保持计算图
    start = 0
    one_minus_eps = 1.0 - torch.finfo(sorted_eta.dtype).eps

    for count in counts.tolist():
        stop = start + count
        # 当前时间组内的事件
        event_eta = sorted_eta[start:stop][sorted_event[start:stop]]
        tied_events = event_eta.numel()

        if tied_events > 0:
            # log(sum of risk set) = log_prefix_risk at this group's last index
            log_risk_set = log_prefix_risk[stop - 1]
            # log(sum of tied event risks)
            log_tied_risk = torch.logsumexp(event_eta, dim=0)

            # Efron fractions: l/d for l=0,...,d-1
            fractions = (
                torch.arange(tied_events, device=eta.device, dtype=eta.dtype)
                / float(tied_events)
            )
            # ratio = fraction * exp(log_tied - log_risk_set)，clamp for stability
            ratio = (
                fractions * torch.exp(log_tied_risk - log_risk_set)
            ).clamp(max=one_minus_eps)

            # denominator contribution: d * log_risk_set + sum log(1 - ratio)
            denominator = float(tied_events) * log_risk_set + torch.log1p(-ratio).sum()
            partial_log_likelihood = partial_log_likelihood + event_eta.sum() - denominator

        start = stop

    negative_pll = -partial_log_likelihood
    return negative_pll / float(n_events)


# ──────────────────────────────────────────────────────────────────────
# 2. Comparable-pair rank loss
# ──────────────────────────────────────────────────────────────────────
def rank_loss(
    eta: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    tau_r: float = 1.0,
) -> torch.Tensor:
    """仅可比较对 (delta_i=1 and t_i<t_j)。

    L_rank = (1/|P|) * sum log(1 + exp(-(eta_i - eta_j)/tau_r))

    禁止把所有 event 样本无条件排在所有 censored 样本前。
    """
    n = eta.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=eta.device)

    # 构造可比较对：delta_i=1 and t_i < t_j
    # 向量化：i 行, j 列
    eta_i = eta.unsqueeze(1)  # [n, 1]
    eta_j = eta.unsqueeze(0)  # [1, n]
    t_i = time.unsqueeze(1)
    t_j = time.unsqueeze(0)
    e_i = event.unsqueeze(1)

    # comparable: e_i=1 and t_i < t_j
    mask = (e_i == 1) & (t_i < t_j)
    n_pairs = mask.sum().item()
    if n_pairs == 0:
        return torch.tensor(0.0, device=eta.device)

    # log(1 + exp(-(eta_i - eta_j)/tau_r))
    diff = (eta_i - eta_j) / tau_r
    # 用 softplus: log(1+exp(-x)) = softplus(-x)
    losses = F.softplus(-diff)
    loss = (losses * mask.float()).sum() / max(n_pairs, 1)
    return loss


# ──────────────────────────────────────────────────────────────────────
# 3. IB loss with free-bits
# ──────────────────────────────────────────────────────────────────────
def ib_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1e-4,
    kappa: float = 0.01,
) -> torch.Tensor:
    """L_IB = beta * sum_k max(KL(q(z_k|h) || N(0,1)), kappa)。

    free-bits kappa 防止 posterior collapse。
    """
    # 逐维 KL
    kl_per_dim = 0.5 * (torch.exp(logvar) + mu ** 2 - 1 - logvar)  # [B, latent_dim]
    # free-bits：每维 KL 至少为 kappa
    kl_free = torch.clamp(kl_per_dim, min=kappa)
    # 对 batch 和 dim 求均值
    return beta * kl_free.mean()


# ──────────────────────────────────────────────────────────────────────
# 5. Elastic-Net loss (normalized L1+L2)
# ──────────────────────────────────────────────────────────────────────
def elastic_loss(
    model: nn.Module,
    lambda_l1: float = 2e-5,
    lambda_l2: float = 2e-5,
) -> torch.Tensor:
    """归一化 L1+L2，定向作用于关键层（gene_proj / meta_encoder / causal_encoder）。

    与 035 对齐：不对所有权重施加 elastic 约束，避免抑制 gate/fusion/CRC 分支等
    其他层的学习能力。bias 和 LayerNorm 始终跳过。
    """
    # 定向前缀：与 fusion.py 中模型属性名一致
    target_prefixes = ("gene_proj", "meta_encoder", "causal_encoder")
    l1 = 0.0
    l2 = 0.0
    n_params = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 跳过 bias 和 LayerNorm
        if name.endswith("bias") or "norm" in name.lower():
            continue
        # 仅惩罚关键层（模块属性名.参数名 形式，如 gene_proj.0.weight）
        if not (name.startswith(target_prefixes)):
            continue
        l1 = l1 + param.abs().sum()
        l2 = l2 + (param ** 2).sum()
        n_params += param.numel()

    if n_params == 0:
        return torch.tensor(0.0)

    # 归一化
    l1_norm = l1 / max(n_params, 1)
    l2_norm = l2 / max(n_params, 1)
    return lambda_l1 * l1_norm + lambda_l2 * l2_norm


# ──────────────────────────────────────────────────────────────────────
# 6. Distill loss (stop-gradient)
# ──────────────────────────────────────────────────────────────────────
def distill_loss(
    z_inv: torch.Tensor,
    z_teacher: Optional[torch.Tensor],
) -> torch.Tensor:
    """蒸馏损失，含 stop-gradient on teacher。"""
    if z_teacher is None:
        return torch.tensor(0.0, device=z_inv.device)
    # stop-gradient on teacher
    z_teacher_sg = z_teacher.detach()
    return F.mse_loss(z_inv, z_teacher_sg)


# ──────────────────────────────────────────────────────────────────────
# 7. Balance loss (MMD)
# ──────────────────────────────────────────────────────────────────────
def bal_loss(
    z_inv: torch.Tensor,
    treatment: Optional[torch.Tensor],
) -> torch.Tensor:
    """加权 MMD/CORAL（简化版：treatment=None 时返回 0）。"""
    if treatment is None:
        return torch.tensor(0.0, device=z_inv.device)
    # 简化：按 treatment 分组计算 MMD
    treated = z_inv[treatment == 1]
    control = z_inv[treatment == 0]
    if len(treated) == 0 or len(control) == 0:
        return torch.tensor(0.0, device=z_inv.device)
    # MMD^2 = ||mu_t - mu_c||^2
    return ((treated.mean(0) - control.mean(0)) ** 2).sum()


# ──────────────────────────────────────────────────────────────────────
# 8. Domain loss (CE with GRL)
# ──────────────────────────────────────────────────────────────────────
def domain_loss(
    domain_logits: torch.Tensor,
    task_id: torch.Tensor,
) -> torch.Tensor:
    """CE(D, C_domain(GRL(z_inv)))。"""
    return F.cross_entropy(domain_logits, task_id.long())


# ──────────────────────────────────────────────────────────────────────
# 10. Consistency loss
# ──────────────────────────────────────────────────────────────────────
def cons_loss(
    z_meta: torch.Tensor,
    z_meta_aug: Optional[torch.Tensor],
) -> torch.Tensor:
    """一致性损失：z_meta 与 z_meta_aug 的 MSE。"""
    if z_meta_aug is None:
        return torch.tensor(0.0, device=z_meta.device)
    return F.mse_loss(z_meta, z_meta_aug.detach())


# ──────────────────────────────────────────────────────────────────────
# 11. Anchor loss
# ──────────────────────────────────────────────────────────────────────
def anchor_loss(
    model: nn.Module,
    theta0: Optional[Dict[str, torch.Tensor]],
) -> torch.Tensor:
    """锚定损失：L_anchor = ||theta - theta0||^2。

    与 035 对齐：仅对 gene_proj / meta_encoder / causal_encoder 前缀的参数计算，
    防止元初始化参数漂移（避免对全部参数施加约束导致欠拟合）。

    注意：模型属性名已从 035 的 gene_projection 重命名为 gene_proj，
    invariant_encoder 重命名为 causal_encoder（见 fusion.py）。
    """
    if theta0 is None:
        return torch.tensor(0.0)
    # 定向前缀：与 fusion.py 中模型属性名一致
    target_prefixes = ("gene_proj", "meta_encoder", "causal_encoder")
    terms: List[torch.Tensor] = []
    for name, param in model.named_parameters():
        if name not in theta0:
            continue
        if not name.startswith(target_prefixes):
            continue
        terms.append(((param - theta0[name].to(param)) ** 2).mean())
    if not terms:
        return torch.tensor(0.0)
    return torch.stack(terms).mean()


# ──────────────────────────────────────────────────────────────────────
# 4. Gate loss wrapper
# ──────────────────────────────────────────────────────────────────────
def gate_loss(
    gate: HardConcreteSparseGate,
    target_k: int,
    cfg: CMIBConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """L_gate = lambda_L0 * L_L0/P + lambda_count * ((L_L0 - K)/P)^2 + lambda_group * sum_g ||W_g||_2。"""
    return compute_gate_loss(
        gate=gate,
        target_k=target_k,
        lambda_l0=cfg.gate_l0_weight,
        lambda_count=cfg.gate_count_weight,
        lambda_group=cfg.gate_group_weight,
    )


def gate_prior_alignment_loss(gate: HardConcreteSparseGate) -> torch.Tensor:
    """P1-1: 因果先验对齐损失（包装 gate 自身方法）。

    L_align = mean((sigmoid(log_alpha) - sigmoid(prior_log_alpha))^2)

    当 gate 未设置 prior_log_alpha 时返回 0。
    """
    return gate.prior_alignment_loss()


# ──────────────────────────────────────────────────────────────────────
# 统一接口 compute_cmib_loss
# ──────────────────────────────────────────────────────────────────────
def compute_cmib_loss(
    output: CMIBOutput,
    batch: SurvivalRecordBatch,
    config: CMIBConfig,
    model: nn.Module,
    theta0: Optional[Dict[str, torch.Tensor]] = None,
    theta0_state: Optional[Dict[str, torch.Tensor]] = None,
    beta: Optional[float] = None,
    z_teacher: Optional[torch.Tensor] = None,
    z_meta_aug: Optional[torch.Tensor] = None,
    current_epoch: int = 0,
    stage: str = "full",
) -> CMIBLoss:
    """统一损失接口。

    Args:
        output: CMIBOutput
        batch: SurvivalRecordBatch
        config: CMIBConfig
        model: CausalMetaIBSurv
        theta0: 元学习初始化参数（用于 anchor loss）
        theta0_state: alias for theta0（target_trainer 用）
        beta: 覆盖 beta 值（None 时用 warmup schedule）
        z_teacher: 因果教师表示（用于 distill loss）
        z_meta_aug: 增强后的 z_meta（用于 consistency loss）
        current_epoch: 当前 epoch（用于 beta warmup）
        stage: 训练阶段（"full" | "meta_warmup" | "ib_only" | "sparse_only"）

    Returns:
        CMIBLoss dataclass，字段为 torch.Tensor（可微），total 可直接 .backward()
    """
    # theta0_state alias
    if theta0 is None and theta0_state is not None:
        theta0 = theta0_state

    losses = CMIBLoss()

    # beta warmup
    if beta is None:
        warmup_epochs = max(config.kl_warmup_epochs, 1)
        if current_epoch < warmup_epochs:
            beta = config.kl_beta * (current_epoch / warmup_epochs)
        else:
            beta = config.kl_beta

    # 确定哪些 loss 参与（stage 控制）
    use_cox = stage in ("full", "meta_warmup")
    use_rank = stage in ("full", "meta_warmup")
    use_ib = stage in ("full", "ib_only", "meta_warmup")
    use_gate = stage in ("full", "sparse_only", "meta_warmup")
    use_elastic = stage in ("full", "meta_warmup")

    # 1. Cox loss
    cox_t = torch.tensor(0.0, device=output.log_risk.device)
    if use_cox:
        cox_t = cox_efron_loss(output.log_risk, batch.time, batch.event)
    losses.cox = cox_t

    # 2. Rank loss
    rank_t = torch.tensor(0.0, device=output.log_risk.device)
    if use_rank:
        rank_t = rank_loss(output.log_risk, batch.time, batch.event, tau_r=1.0)
    losses.rank = rank_t

    # 3. IB loss
    ib_t = torch.tensor(0.0, device=output.mu.device)
    if use_ib:
        ib_t = ib_loss(output.mu, output.logvar, beta=beta, kappa=config.kl_free_bits)
    losses.ib = ib_t

    # 4. Gate loss
    gate_t = torch.tensor(0.0, device=output.log_risk.device)
    if use_gate and hasattr(model, "sparse_gate"):
        gate_t, _ = gate_loss(model.sparse_gate, config.active_gene_target, config)
    losses.gate = gate_t

    # 5. Elastic loss
    el_t = torch.tensor(0.0, device=output.log_risk.device)
    if use_elastic:
        el_t = elastic_loss(model, config.elastic_l1, config.elastic_l2)
    losses.elastic = el_t

    # 6. Distill loss
    dl = distill_loss(output.z_invariant, z_teacher)
    losses.distill = dl

    # 7. Balance loss
    bl = bal_loss(output.z_invariant, batch.treatment)
    losses.bal = bl

    # 8. Domain loss
    doml = torch.tensor(0.0, device=output.log_risk.device)
    if output.domain_logits is not None:
        doml = domain_loss(output.domain_logits, batch.task_id)
    losses.domain = doml

    # 9. Orth loss
    ol = torch.tensor(0.0, device=output.log_risk.device)
    if output.z_task is not None and output.z_crc is not None:
        ol = orth_loss(output.z_invariant, output.z_task)
    losses.orth = ol

    # 10. Consistency loss
    cl = cons_loss(output.z_meta, z_meta_aug)
    losses.cons = cl

    # 11. Anchor loss (P2-2: 加入 warmup 系数)
    al = anchor_loss(model, theta0)
    losses.anchor = al
    # P2-2: anchor warmup — 前anchor_warmup_epochs个epoch线性增到1.0
    anchor_warmup = getattr(config, "anchor_warmup_epochs", 10)
    anchor_coef = min(1.0, float(current_epoch) / max(anchor_warmup, 1))
    anchor_weight_effective = config.anchor_weight * anchor_coef

    # P1-1: gate prior alignment loss
    gate_align_t = torch.tensor(0.0, device=output.log_risk.device)
    if use_gate and hasattr(model, "sparse_gate"):
        gate_align_t = gate_prior_alignment_loss(model.sparse_gate)

    # total（tensor，可微）
    total = (
        config.cox_weight * losses.cox
        + config.rank_weight * losses.rank
        + losses.ib
        + losses.gate
        + losses.elastic
        + config.causal_distill_weight * losses.distill
        + config.treatment_mmd_weight * losses.bal
        + config.domain_weight * losses.domain
        + config.crc_branch.orthogonality_weight * losses.orth
        + config.consistency_weight * losses.cons
        + anchor_weight_effective * losses.anchor
        + getattr(config, "gate_prior_alignment_weight", 0.0) * gate_align_t
    )
    losses.total = total

    return losses
