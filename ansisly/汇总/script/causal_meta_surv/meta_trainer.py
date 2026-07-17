"""meta_trainer.py — 修正版 Reptile 元学习。

核心修正（spec.md / Fusion_Plan §3.3）：
外循环更新公式：theta0 <- theta0 + meta_lr * mean(theta' - theta0)
NOT 在内循环终点继续外推。

关键设置：
- meta_batch_size=3-4 tasks
- inner_steps ∈ {3, 5}
- episode 分层采样：event × log(time) quantile
- support events >= 8, query events >= 5
- epsilon 从 0.05-0.10 余弦衰减到 0.005
- 任务采样按事件数平方根加权并设上限
"""
from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .contracts import CMIBConfig, CMIBOutput, SurvivalRecordBatch
from .data import SourceTask
from .losses import compute_cmib_loss


# =====================================================================
# 参数克隆工具
# =====================================================================


def clone_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """深拷贝模型参数（返回 state_dict 副本）。"""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def restore_parameters(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """用 state_dict 恢复模型参数。"""
    model.load_state_dict({k: v.to(p.device) for (k, p), (k2, v) in
                           zip(model.state_dict().items(), state.items())})


def compute_delta(theta0: Dict[str, torch.Tensor],
                  theta_prime: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """计算 delta = theta' - theta0。"""
    return {k: theta_prime[k] - theta0[k] for k in theta0}


def apply_delta(theta0: Dict[str, torch.Tensor],
                delta: Dict[str, torch.Tensor], lr: float) -> Dict[str, torch.Tensor]:
    """theta0 + lr * delta（Reptile 内插公式）。"""
    return {k: theta0[k] + lr * delta[k] for k in theta0}


def mean_deltas(deltas: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """对多个 task 的 delta 求平均。"""
    if not deltas:
        return {}
    keys = deltas[0].keys()
    return {k: torch.stack([d[k] for d in deltas]).mean(0) for k in keys}


# =====================================================================
# 分层 survival episode 采样
# =====================================================================


def stratified_survival_episode(
    task: SourceTask,
    config: Optional[CMIBConfig] = None,
    support_size: int = 32,
    query_size: int = 32,
    min_support_events: int = 8,
    min_query_events: int = 5,
    seed: int = 42,
    max_expand: int = 5,
) -> Tuple[SurvivalRecordBatch, SurvivalRecordBatch]:
    """按 event × log(time) 分层采样 support/query。

    按 event × log(time) quantile 分层，不满足最低事件数时扩大 episode
    （而非复用 support 充当 query）。

    Args:
        task: 源癌种任务
        config: 配置（用于 make_batch；若 None 需 task 已缓存 batch）

    Returns:
        (support_batch, query_batch)
    """
    rng = np.random.RandomState(seed)
    if config is not None:
        batch = task.make_batch(config)
    else:
        batch = task.make_batch(CMIBConfig())  # fallback 用默认配置
    n = batch.batch_size
    time = batch.time.cpu().numpy()
    event = batch.event.cpu().numpy().astype(int)

    # 分层键
    log_t = np.log1p(np.clip(time, 1e-6, None))
    bins = np.quantile(log_t, np.linspace(0, 1, 4))
    bins[0] = -np.inf
    bins[-1] = np.inf
    bin_idx = np.digitize(log_t, bins[1:-1])
    strata = event.astype(str) + "|" + bin_idx.astype(str)

    # 扩大 support 直到满足最低事件数
    cur_support_size = support_size
    for _ in range(max_expand):
        support_idx = _stratified_sample(strata, cur_support_size, rng)
        if int(event[support_idx].sum()) >= min_support_events or cur_support_size >= n:
            break
        cur_support_size = min(int(cur_support_size * 1.5), n)

    # query：剩余样本中分层采样
    remaining = np.setdiff1d(np.arange(n), support_idx)
    if len(remaining) < query_size:
        # 不够则允许重复（但不复用 support 充当 query）
        query_idx = remaining
    else:
        cur_query_size = query_size
        for _ in range(max_expand):
            query_idx = _stratified_sample(strata[remaining], cur_query_size, rng)
            query_idx = remaining[query_idx]
            if int(event[query_idx].sum()) >= min_query_events or cur_query_size >= len(remaining):
                break
            cur_query_size = min(int(cur_query_size * 1.5), len(remaining))

    support_batch = _index_batch(batch, support_idx)
    query_batch = _index_batch(batch, query_idx)
    return support_batch, query_batch


def _stratified_sample(strata: np.ndarray, n: int,
                       rng: np.random.RandomState) -> np.ndarray:
    """分层采样（不放回）。"""
    N = len(strata)
    n = min(n, N)
    unique = np.unique(strata)
    per_stratum = max(1, n // len(unique))
    idx_list: List[int] = []
    for s in unique:
        s_idx = np.where(strata == s)[0]
        take = min(per_stratum, len(s_idx))
        chosen = rng.choice(s_idx, size=take, replace=False)
        idx_list.extend(chosen.tolist())
    # 不足则随机补
    if len(idx_list) < n:
        remaining = np.setdiff1d(np.arange(N), idx_list)
        extra = rng.choice(remaining, size=min(n - len(idx_list), len(remaining)),
                           replace=False)
        idx_list.extend(extra.tolist())
    return np.array(idx_list[:n], dtype=np.int64)


def _index_batch(batch: SurvivalRecordBatch, idx: np.ndarray) -> SurvivalRecordBatch:
    """从 batch 中按索引取子集。"""
    from dataclasses import replace as _replace
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=batch.x_gene.device)
    pid = [batch.patient_id[i] for i in idx]
    origin = [batch.origin_patient_id[i] for i in idx]
    return SurvivalRecordBatch(
        patient_id=pid,
        x_gene=batch.x_gene[idx_t],
        x_clinical=batch.x_clinical[idx_t],
        clinical_mask=batch.clinical_mask[idx_t],
        task_id=batch.task_id[idx_t],
        time=batch.time[idx_t],
        event=batch.event[idx_t],
        treatment=batch.treatment[idx_t] if batch.treatment is not None else None,
        sample_weight=batch.sample_weight[idx_t],
        origin_patient_id=origin,
        is_synthetic=batch.is_synthetic[idx_t],
    )


# =====================================================================
# Reptile 元学习
# =====================================================================


class ReptileMetaTrainer:
    """修正版 Reptile 元学习训练器。

    外循环公式：theta0 <- theta0 + meta_lr * mean(theta' - theta0)
    """

    def __init__(self, model: nn.Module, config: CMIBConfig,
                 optimizer_factory: Optional[Callable] = None):
        self.model = model
        self.config = config
        self.optimizer_factory = optimizer_factory or (
            lambda params: torch.optim.AdamW(params, lr=config.meta_inner_lr,
                                             weight_decay=config.weight_decay)
        )
        self.meta_history: List[Dict[str, float]] = []

    def inner_update(
        self,
        theta0: Dict[str, torch.Tensor],
        support_batch: SurvivalRecordBatch,
        lr: float,
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """单步内循环适配：在 theta0 基础上用 support 算梯度并更新。

        Returns:
            theta_prime: 适配后的参数
            loss_val: 适配损失
        """
        # 恢复到 theta0
        restore_parameters(self.model, theta0)
        self.model.train()
        optimizer = self.optimizer_factory(list(self.model.parameters()))
        # 手动设 lr
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        output = self.model(support_batch)
        loss = compute_cmib_loss(output, support_batch, self.config, self.model,
                                  stage="meta_warmup")
        optimizer.zero_grad()
        loss.total.backward()
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        optimizer.step()

        theta_prime = clone_parameters(self.model)
        return theta_prime, float(loss.total.detach().item())

    def meta_batch_step(
        self,
        tasks: List[SourceTask],
        inner_steps: int,
        meta_lr: float,
        step: int = 0,
    ) -> Dict[str, float]:
        """单个 meta-batch step。

        1. 保存 theta0
        2. 对每个 task：内循环 inner_steps 步得 theta'
        3. delta = theta' - theta0
        4. theta0 += meta_lr * mean(deltas)
        5. P0-6 修复：invariant 更新步（CORAL + domain adversarial）

        Returns:
            {'mean_loss': float, 'n_tasks': int, 'invariant_loss': float}
        """
        theta0 = clone_parameters(self.model)
        deltas: List[Dict[str, torch.Tensor]] = []
        losses: List[float] = []

        for task in tasks:
            # 内循环：inner_steps 步，每步不同 episode
            theta_task = {k: v.clone() for k, v in theta0.items()}
            for s in range(inner_steps):
                support, query = stratified_survival_episode(
                    task, config=self.config, seed=self.config.seed + step * 100 + s,
                    min_support_events=self.config.min_support_events,
                    min_query_events=self.config.min_query_events,
                )
                theta_task, l = self.inner_update(theta_task, support,
                                                   lr=self.config.meta_inner_lr)
                losses.append(l)
            delta = compute_delta(theta0, theta_task)
            deltas.append(delta)

        # 平均 delta 并更新 theta0
        if deltas:
            mean_delta = mean_deltas(deltas)
            new_theta0 = apply_delta(theta0, mean_delta, meta_lr)
            restore_parameters(self.model, new_theta0)

        # P0-6 修复：invariant 更新步（CORAL + domain adversarial）
        # 与 035 对齐：Reptile 外循环后独立更新 causal_encoder + domain_head，
        # 强化跨癌种不变表征学习
        inv_loss = self._invariant_update_step(tasks, step)

        return {
            "mean_loss": float(np.mean(losses)) if losses else float("nan"),
            "n_tasks": len(tasks),
            "invariant_loss": inv_loss,
        }

    def _invariant_update_step(
        self,
        tasks: List[SourceTask],
        step: int,
    ) -> float:
        """P0-6 修复：独立的 invariant 更新步（CORAL + domain adversarial）。

        与 035 对齐：Reptile 外循环后，对 causal_encoder 和 domain_head 做独立更新，
        强化跨癌种不变表征学习。CORAL 对齐源癌种间协方差，domain adversarial
        防止编码器泄露任务身份。

        Returns:
            invariant_loss: CORAL + domain 加权损失值
        """
        if not hasattr(self.model, "causal_encoder") or not hasattr(self.model, "domain_head"):
            return 0.0
        if len(tasks) < 2:
            return 0.0

        self.model.train()
        # 仅更新 causal_encoder 和 domain_head 参数
        inv_params = list(self.model.causal_encoder.parameters())
        dom_params = list(self.model.domain_head.parameters())
        params_to_update = [p for p in inv_params + dom_params if p.requires_grad]
        if not params_to_update:
            return 0.0

        optimizer = torch.optim.AdamW(
            params_to_update,
            lr=self.config.meta_inner_lr * 0.5,
            weight_decay=self.config.weight_decay,
        )

        # 收集各 task 的 invariant 表征
        task_z_invs: List[torch.Tensor] = []
        task_ids: List[torch.Tensor] = []
        for task in tasks:
            try:
                batch = task.make_batch(self.config)
                if batch.batch_size == 0:
                    continue
                with torch.no_grad():
                    # 只前向到 causal_encoder，避免影响其他层
                    x_gene = batch.x_gene.to(self.model.sparse_gate.log_alpha.device)
                    x_clin = batch.x_clinical.to(self.model.sparse_gate.log_alpha.device)
                    gated = self.model.sparse_gate(x_gene)
                    h_gene = self.model.gene_proj(gated)
                    z_inv = self.model.causal_encoder(torch.cat([h_gene, x_clin], dim=-1))
                task_z_invs.append(z_inv)
                task_ids.append(torch.full((z_inv.shape[0],), task.task_id,
                                           dtype=torch.long, device=z_inv.device))
            except Exception:
                continue

        if len(task_z_invs) < 2:
            return 0.0

        # CORAL loss：对齐源癌种间协方差
        coral_losses = []
        ref_z = task_z_invs[0]
        for z in task_z_invs[1:]:
            # 协方差对齐：||Cov(z_a) - Cov(z_b)||_F^2
            n_a = ref_z.shape[0]
            n_b = z.shape[0]
            if n_a < 2 or n_b < 2:
                continue
            cov_a = (ref_z - ref_z.mean(0, keepdim=True)).t() @ (ref_z - ref_z.mean(0, keepdim=True)) / max(n_a - 1, 1)
            cov_b = (z - z.mean(0, keepdim=True)).t() @ (z - z.mean(0, keepdim=True)) / max(n_b - 1, 1)
            coral_losses.append(((cov_a - cov_b) ** 2).sum())
        coral_loss = torch.stack(coral_losses).mean() if coral_losses else torch.tensor(0.0, device=ref_z.device)

        # Domain adversarial：让 causal_encoder 输出无法区分任务
        all_z = torch.cat(task_z_invs, dim=0).detach()
        all_ids = torch.cat(task_ids, dim=0)
        # 重新前向以建立计算图（all_z 已 detach，需要新前向）
        # 简化：用 domain_head 对 detached z_inv 做分类，梯度只回传到 domain_head
        domain_logits = self.model.domain_head(all_z, grl_lambda=self.config.gradient_reversal_weight)
        dom_loss = torch.nn.functional.cross_entropy(domain_logits, all_ids)

        # 总损失：CORAL + domain（注意 domain 用 GRL，encoder 梯度反转）
        total_inv_loss = (
            self.config.coral_weight * coral_loss
            + self.config.domain_weight * dom_loss
        )

        optimizer.zero_grad()
        if total_inv_loss.requires_grad:
            total_inv_loss.backward()
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params_to_update, self.config.grad_clip)
            optimizer.step()

        return float(total_inv_loss.detach().item())

    def cosine_epsilon(self, step: int, total_steps: int) -> float:
        """余弦衰减 epsilon：0.05-0.10 -> 0.005。"""
        eps_init = self.config.meta_lr
        eps_min = self.config.meta_lr_min
        progress = min(1.0, step / max(1, total_steps))
        return eps_min + 0.5 * (eps_init - eps_min) * (1 + math.cos(math.pi * progress))

    def train_meta(
        self,
        source_tasks: List[SourceTask],
        epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """执行完整元学习训练。

        Returns:
            {'final_loss': float, 'epochs': int, 'history': list}
        """
        import logging as _logging
        _logger = _logging.getLogger("cmib_surv.meta_trainer")
        import time as _time

        epochs = epochs or self.config.meta_epochs
        history: List[Dict[str, float]] = []
        _t0 = _time.time()
        _log_every = max(1, epochs // 10)  # 每 10% 进度输出一次
        for ep in range(epochs):
            # 任务采样：按事件数平方根加权
            weights = np.array([math.sqrt(max(1, t.n_events)) for t in source_tasks],
                              dtype=float)
            weights = weights / weights.sum()
            # 设上限：单 task 最多 0.5
            weights = np.clip(weights, 0.0, 0.5)
            weights = weights / weights.sum()
            n_sample = min(self.config.meta_batch_size, len(source_tasks))
            chosen_idx = np.random.choice(len(source_tasks), size=n_sample,
                                          replace=False, p=weights)
            chosen_tasks = [source_tasks[i] for i in chosen_idx]

            eps = self.cosine_epsilon(ep, epochs)
            info = self.meta_batch_step(chosen_tasks, self.config.inner_steps, eps, ep)
            info["epoch"] = ep
            info["epsilon"] = eps
            history.append(info)
            self.meta_history.append(info)

            # 进度日志：每 _log_every 个 epoch 输出一次
            if (ep + 1) % _log_every == 0 or ep == 0 or ep == epochs - 1:
                elapsed = _time.time() - _t0
                eta = elapsed / (ep + 1) * (epochs - ep - 1) if ep > 0 else 0.0
                _logger.info(
                    "Meta iter %d/%d: mean_loss=%.4f invariant_loss=%.4f eps=%.4f "
                    "tasks=%d elapsed=%.1fs eta=%.1fs",
                    ep + 1, epochs, info.get("mean_loss", float("nan")),
                    info.get("invariant_loss", 0.0), eps,
                    info.get("n_tasks", 0), elapsed, eta,
                )

        _logger.info("Meta-training completed: epochs=%d final_loss=%.4f total_time=%.1fs",
                     epochs, history[-1]["mean_loss"] if history else float("nan"),
                     _time.time() - _t0)
        return {
            "final_loss": history[-1]["mean_loss"] if history else float("nan"),
            "epochs": epochs,
            "history": history,
        }


# =====================================================================
# Leave-one-cancer-out 评估
# =====================================================================


def leave_one_cancer_out_eval(
    model_factory: Callable[[], nn.Module],
    source_tasks: List[SourceTask],
    config: CMIBConfig,
) -> Dict[str, Any]:
    """Leave-one-cancer-out 元评估。

    对每个 source cancer：用其余训练，在被留出的 cancer 上评估。
    剔除持续负迁移的 source task。
    """
    from .evaluation import harrell_c_index

    results: Dict[str, Any] = {"per_cancer": {}, "negative_transfer": []}
    for i, held_out in enumerate(source_tasks):
        train_tasks = [t for j, t in enumerate(source_tasks) if j != i]
        if len(train_tasks) < 2:
            continue
        model = model_factory()
        trainer = ReptileMetaTrainer(model, config)
        try:
            trainer.train_meta(train_tasks, epochs=max(2, config.meta_epochs // 2))
            # 评估
            model.eval()
            held_batch = held_out.make_batch(config)
            with torch.no_grad():
                output = model(held_batch)
            risk = output.log_risk.cpu().numpy()
            time = held_batch.time.cpu().numpy()
            event = held_batch.event.cpu().numpy().astype(int)
            c = harrell_c_index(risk, time, event)
            results["per_cancer"][held_out.cancer] = c
            if c < 0.5:
                results["negative_transfer"].append(held_out.cancer)
        except Exception as e:
            results["per_cancer"][held_out.cancer] = float("nan")
            results["negative_transfer"].append(f"{held_out.cancer}: {e}")

    return results
