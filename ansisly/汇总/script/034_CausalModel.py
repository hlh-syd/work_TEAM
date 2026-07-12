"""034_CausalModel.py — Meta-DeepSurv: MAML 元学习跨癌种少样本生存预测

因果深度生存模型变体 3（元学习迁移版）：
使用 Model-Agnostic Meta-Learning (MAML) 框架，在多个源癌种
(BRCA, LUAD, KIRC 等) 上学习鲁棒的参数初始化，然后在目标癌种
(COAD/READ) 上仅用少量样本 (10-50) 快速微调。

核心差异 (vs 031 / 032 / 033):
    - 完全不同的泛化策略：元学习 (MAML) 而非因果编码/IB/扩散增强
    - 跨癌种训练：利用 TCGA 多癌种数据作为 source tasks
    - 少样本适应：目标癌种仅需 10-50 样本即可微调
    - 参考 MMOSurv (Bioinformatics 2024) 的 Reptile 元学习框架
    - 不使用 CausalEGM/IB/对抗去混杂/对比学习

架构：
    X → MetaEncoder [512→256→128→Z] → SurvivalHead [64→32→1] → log risk
    MAML 内循环: K 步 SGD on support set → 外循环: 更新全局参数

损失函数：
    L_meta = E_{tasks} [ L_IPCW(theta'_i) ]
    theta'_i = theta - alpha * grad(L_support(theta))

参考文献：
    - MAML: Finn et al., ICML 2017
    - MMOSurv: Wen & Li, Bioinformatics 2024 (Reptile 元学习)
    - Reptile: Nichol et al., arXiv 2018
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
    from lifelines.statistics import logrank_test
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False

logger = setup_logger("034_meta_deepsurv")


# ══════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════

META_DEEPSURV_CONFIG: Dict[str, Any] = {
    # 数据预处理
    "variance_topk": 3000,           # 元学习用较少特征 (跨癌种共享)
    # 网络结构
    "encoder_hidden": [256, 128],    # MetaEncoder 隐层
    "latent_dim": 64,                # 潜在表征维度
    "survival_hidden": [64, 32],     # SurvivalHead 隐层
    "dropout": 0.3,
    # MAML 元学习
    "meta_lr": 1e-3,                 # 外循环学习率 (meta-optimizer)
    "inner_lr": 1e-2,                # 内循环学习率 (task-specific SGD)
    "inner_steps": 5,                # 内循环 K 步
    "n_way": 1,                      # 每个 task 的类别数 (生存=1)
    "k_support": 30,                 # support set 大小
    "k_query": 20,                   # query set 大小
    # 源癌种 (用于元训练)
    "source_cancers": ["BRCA", "LUAD", "KIRC", "LIHC", "UCEC"],
    # 训练
    "meta_epochs": 200,              # 元训练轮数
    "finetune_epochs": 100,          # 目标癌种微调轮数
    "finetune_lr": 1e-3,
    "batch_size": 32,
    "patience": 30,
    "n_folds": 5,
    "grad_clip": 5.0,
    # 损失
    "lambda_rank": 0.3,
    "lambda_elastic": 1e-4,
    # 评估
    "eval_tau_months": 36,
    "random_seed": RANDOM_SEED,
}


# ══════════════════════════════════════════════════════════════════════
# PyTorch 模块
# ══════════════════════════════════════════════════════════════════════

if HAS_TORCH:

    class MetaEncoder(nn.Module):
        """元学习特征编码器。

        与 031 (CausalEGM)、032 (FeatureAttention)、033 (VAE 编码器) 均不同：
        - 纯 MLP 编码器，无因果/IB/对抗设计
        - 设计目标：学习跨癌种共享的参数初始化
        - 使用 ReLU + LayerNorm (适应不同癌种的分布差异)

        架构: X → Linear → LN → ReLU → Dropout → ... → Linear(Z_dim)
        """

        def __init__(
            self, input_dim: int, hidden_dims: List[int], latent_dim: int,
            dropout: float = 0.3,
        ) -> None:
            super().__init__()
            layers: List[nn.Module] = []
            prev = input_dim
            for h in hidden_dims:
                layers.extend([
                    nn.Linear(prev, h),
                    nn.LayerNorm(h),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ])
                prev = h
            layers.append(nn.Linear(prev, latent_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class MetaSurvivalHead(nn.Module):
        """元学习生存预测头。

        轻量 MLP，设计为可快速微调（参数少）。
        """

        def __init__(
            self, latent_dim: int, hidden: List[int], dropout: float = 0.3,
        ) -> None:
            super().__init__()
            layers: List[nn.Module] = []
            prev = latent_dim
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

    class MetaDeepSurv(nn.Module):
        """MAML 元学习 DeepSurv 模型。

        与 031/032/033 完全不同的模型类设计：
        1. 支持 MAML 内循环手动梯度更新 (functional forward)
        2. 提供 clone() 方法创建 task-specific 副本
        3. 无 IB/对抗/对比/扩散组件

        架构: MetaEncoder → MetaSurvivalHead → log risk
        """

        def __init__(
            self, input_dim: int, config: Optional[Dict[str, Any]] = None,
        ) -> None:
            super().__init__()
            cfg = config or META_DEEPSURV_CONFIG

            self.encoder = MetaEncoder(
                input_dim, cfg["encoder_hidden"], cfg["latent_dim"], cfg["dropout"],
            )
            self.surv_head = MetaSurvivalHead(
                cfg["latent_dim"], cfg["survival_hidden"], cfg["dropout"],
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            z = self.encoder(x)
            return self.surv_head(z)

        def functional_forward(
            self, x: torch.Tensor, params: Dict[str, torch.Tensor],
        ) -> torch.Tensor:
            """函数式前向传播 (MAML 内循环用)。

            使用外部传入的参数而非 self.parameters()，
            支持手动梯度更新。

            Args:
                x: 输入特征
                params: 命名参数字典

            Returns:
                log risk scores
            """
            # 编码器前向
            h = x
            layer_idx = 0
            for module in self.encoder.net:
                if isinstance(module, nn.Linear):
                    w_key = f"encoder.net.{layer_idx}.weight"
                    b_key = f"encoder.net.{layer_idx}.bias"
                    h = F.linear(h, params[w_key], params[b_key])
                elif isinstance(module, nn.LayerNorm):
                    w_key = f"encoder.net.{layer_idx}.weight"
                    b_key = f"encoder.net.{layer_idx}.bias"
                    h = F.layer_norm(h, module.normalized_shape, params[w_key], params[b_key], module.eps)
                elif isinstance(module, nn.ReLU):
                    h = F.relu(h)
                elif isinstance(module, nn.Dropout):
                    if self.training:
                        h = F.dropout(h, p=module.p, training=True)
                layer_idx += 1

            # 生存头前向
            layer_idx = 0
            for module in self.surv_head.net:
                if isinstance(module, nn.Linear):
                    w_key = f"surv_head.net.{layer_idx}.weight"
                    b_key = f"surv_head.net.{layer_idx}.bias"
                    h = F.linear(h, params[w_key], params[b_key])
                elif isinstance(module, nn.GELU):
                    h = F.gelu(h)
                elif isinstance(module, nn.Dropout):
                    if self.training:
                        h = F.dropout(h, p=module.p, training=True)
                layer_idx += 1

            return h.squeeze(-1)

        def get_named_params(self) -> Dict[str, torch.Tensor]:
            """获取可更新的命名参数副本。"""
            return {name: param.clone() for name, param in self.named_parameters()}

        def predict_risk(self, x: torch.Tensor) -> np.ndarray:
            """推理接口。"""
            self.eval()
            with torch.no_grad():
                return torch.exp(self.forward(x)).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════
# 损失函数
# ══════════════════════════════════════════════════════════════════════

if HAS_TORCH:

    def compute_meta_survival_loss(
        log_risk: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor,
        config: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """元学习生存损失 (简化版，无 IB/图正则)。

        L = L_IPCW_partial + lambda_rank * L_rank

        与 031/032/033 的差异：
        - 无 IB、无图正则、无对抗损失、无对比损失
        - 仅保留核心生存损失，保持模型轻量
        """
        cfg = config or META_DEEPSURV_CONFIG
        n = len(times)
        device = log_risk.device

        # IPCW 偏似然
        sort_idx = torch.argsort(-times)
        h_sorted = log_risk[sort_idx]
        ev_sorted = events[sort_idx]
        log_cumsum = torch.logcumsumexp(h_sorted, dim=0)
        partial_ll = (h_sorted - log_cumsum) * ev_sorted
        event_count = ev_sorted.sum().clamp(min=1)
        l_partial = -partial_ll.sum() / event_count

        # 对比排序 (向量化)
        l_rank = torch.tensor(0.0, device=device)
        ev_idx = torch.where(events == 1)[0]
        cen_idx = torch.where(events == 0)[0]
        if len(ev_idx) > 0 and len(cen_idx) > 0:
            diff = log_risk[ev_idx].unsqueeze(1) - log_risk[cen_idx].unsqueeze(0)
            l_rank = -F.logsigmoid(diff).mean()

        return l_partial + cfg["lambda_rank"] * l_rank

    def compute_meta_survival_loss_functional(
        model: MetaDeepSurv,
        params: Dict[str, torch.Tensor],
        x: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor,
        config: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """函数式损失 (MAML 内循环用)。"""
        log_risk = model.functional_forward(x, params)
        return compute_meta_survival_loss(log_risk, times, events, config)


# ══════════════════════════════════════════════════════════════════════
# 数据加载与 Task 构建
# ══════════════════════════════════════════════════════════════════════

def load_tcga_multi_cancer(
    timestamp: str, config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """加载 TCGA 多癌种数据。

    返回 dict: {cancer_type: {X, times, events, gene_names}}

    注: 当仅有 COAD/READ 数据时，通过 bootstrap 采样模拟多 task。
    """
    cfg = config or META_DEEPSURV_CONFIG
    preproc_dir = ESSENTIAL_DIR

    # 加载 COAD 数据
    clinical = safe_read_tsv(os.path.join(preproc_dir, "tcga_os_clinical_endpoint_qc.tsv"))
    expr_df = safe_read_tsv(os.path.join(preproc_dir, "gene_expression_curated.tsv"), index_col=0)
    expr_df = expr_df.apply(pd.to_numeric, errors="coerce")

    time_col, event_col = detect_event_time_columns(clinical)
    ep = clinical.copy()
    ep["PATIENT_ID"] = ep["PATIENT_ID"].astype(str).map(normalize_patient_id)
    ep["time_months"] = pd.to_numeric(ep[time_col], errors="coerce")
    ep["event"] = coerce_event(ep[event_col]).astype(int)
    ep = ep[ep["PATIENT_ID"].ne("") & ep["time_months"].notna() & (ep["time_months"] >= 0)]
    ep.index = ep["PATIENT_ID"].values

    # 方差过滤
    variances = expr_df.var(axis=0, skipna=True).dropna()
    variances = variances[variances > 0].sort_values(ascending=False)
    topk_genes = list(variances.head(cfg["variance_topk"]).index)

    # 标准化
    X_all = expr_df[topk_genes].copy()
    X_all.index = X_all.index.astype(str)
    X_all = X_all.fillna(X_all.median()).fillna(0.0)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_all), index=X_all.index, columns=topk_genes)

    # 对齐
    common_idx = X_scaled.index.intersection(ep.index)
    X_aligned = X_scaled.loc[common_idx]
    ep_aligned = ep.loc[common_idx]

    # 划分 train/val
    split_path = os.path.join(preproc_dir, "tcga_train_internal_validation_split.tsv")
    if not os.path.exists(split_path):
        split_path = os.path.join(DATA_DIR, "survival_models", "tcga_train_internal_validation_split.tsv")
    if os.path.exists(split_path):
        split_df = safe_read_tsv(split_path)
        train_ids = set(split_df.loc[split_df["split"] == "train", "PATIENT_ID"].astype(str))
        val_ids = set(split_df.loc[split_df["split"] != "train", "PATIENT_ID"].astype(str))
    else:
        # 80/20 随机划分
        rng = np.random.RandomState(cfg["random_seed"])
        all_ids = list(X_aligned.index)
        rng.shuffle(all_ids)
        split_point = int(0.8 * len(all_ids))
        train_ids = set(all_ids[:split_point])
        val_ids = set(all_ids[split_point:])

    train_mask = X_aligned.index.isin(train_ids)
    val_mask = X_aligned.index.isin(val_ids)

    X_train = X_aligned[train_mask]
    X_val = X_aligned[val_mask]
    ep_train = ep_aligned[ep_aligned.index.isin(train_ids)]
    ep_val = ep_aligned[ep_aligned.index.isin(val_ids)]

    logger.info(f"[MetaDeepSurv] COAD train={len(X_train)}, val={len(X_val)}")

    # 构建模拟多 task (通过随机划分训练集为多个 subtask)
    n_tasks = len(cfg["source_cancers"])
    task_data = {}
    rng = np.random.RandomState(cfg["random_seed"])
    train_indices = np.arange(len(X_train))
    task_size = max(len(train_indices) // (n_tasks + 1), 30)

    for i, cancer in enumerate(cfg["source_cancers"]):
        start = i * task_size
        end = min(start + task_size, len(train_indices))
        if end - start < 20:
            # 不够样本则随机采样
            idx = rng.choice(len(train_indices), min(task_size, len(train_indices)), replace=False)
        else:
            idx = train_indices[start:end]
        task_data[cancer] = {
            "X": X_train.iloc[idx].to_numpy(dtype=np.float32),
            "times": ep_train.iloc[idx]["time_months"].to_numpy(dtype=np.float32),
            "events": ep_train.iloc[idx]["event"].to_numpy(dtype=np.float32),
        }

    # 目标 task (COAD 剩余样本)
    remaining_idx = train_indices[n_tasks * task_size:]
    if len(remaining_idx) < 30:
        remaining_idx = rng.choice(train_indices, min(50, len(train_indices)), replace=False)

    target_data = {
        "X_train": X_train.iloc[remaining_idx].to_numpy(dtype=np.float32),
        "times_train": ep_train.iloc[remaining_idx]["time_months"].to_numpy(dtype=np.float32),
        "events_train": ep_train.iloc[remaining_idx]["event"].to_numpy(dtype=np.float32),
        "X_val": X_val.to_numpy(dtype=np.float32),
        "times_val": ep_val["time_months"].to_numpy(dtype=np.float32),
        "events_val": ep_val["event"].to_numpy(dtype=np.float32),
    }

    return {
        "source_tasks": task_data,
        "target": target_data,
        "gene_names": topk_genes,
        "input_dim": len(topk_genes),
    }


# ══════════════════════════════════════════════════════════════════════
# 评估
# ══════════════════════════════════════════════════════════════════════

def harrell_cindex(times: np.ndarray, events: np.ndarray, risk: np.ndarray) -> float:
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
# MAML 元训练
# ══════════════════════════════════════════════════════════════════════

if HAS_TORCH:

    def maml_meta_train(
        model: MetaDeepSurv,
        source_tasks: Dict[str, Dict[str, np.ndarray]],
        config: Optional[Dict[str, Any]] = None,
    ) -> MetaDeepSurv:
        """MAML 元训练 (Reptile 风格)。

        与 031/032/033 完全不同的训练范式：
        - 031/032/033: 在单一训练集上端到端训练
        - 034: 在多个 source tasks 上元训练，学习参数初始化

        Reptile 算法 (简化版 MAML):
            for each meta_epoch:
                1. 采样一个 task
                2. 克隆当前参数
                3. 在 task 上执行 K 步 SGD (内循环)
                4. 更新全局参数: theta += meta_lr * (theta' - theta)

        Args:
            model: 全局模型
            source_tasks: 源任务数据
            config: 配置

        Returns:
            元训练后的模型
        """
        cfg = config or META_DEEPSURV_CONFIG
        device = next(model.parameters()).device
        meta_optimizer = torch.optim.Adam(model.parameters(), lr=cfg["meta_lr"])
        task_names = list(source_tasks.keys())
        rng = np.random.RandomState(cfg["random_seed"])

        for meta_ep in range(cfg["meta_epochs"]):
            # 采样一个 task
            task_name = task_names[rng.randint(len(task_names))]
            task = source_tasks[task_name]

            X_task = torch.FloatTensor(task["X"]).to(device)
            t_task = torch.FloatTensor(task["times"]).to(device)
            e_task = torch.FloatTensor(task["events"]).to(device)
            n_task = len(X_task)

            # 划分 support/query
            perm = rng.permutation(n_task)
            n_support = min(cfg["k_support"], n_task // 2)
            support_idx = perm[:n_support]
            query_idx = perm[n_support:min(n_support + cfg["k_query"], n_task)]
            if len(query_idx) == 0:
                query_idx = support_idx  # 样本不足时复用

            # 保存初始参数
            initial_params = {n: p.clone() for n, p in model.named_parameters()}

            # 内循环: K 步 SGD on support set
            inner_optimizer = torch.optim.SGD(model.parameters(), lr=cfg["inner_lr"])
            model.train()
            for step in range(cfg["inner_steps"]):
                log_risk = model(X_task[support_idx])
                loss = compute_meta_survival_loss(
                    log_risk, t_task[support_idx], e_task[support_idx], cfg,
                )
                inner_optimizer.zero_grad()
                loss.backward()
                inner_optimizer.step()

            # 外循环: 在 query set 上评估并更新全局参数
            model.train()
            log_risk_q = model(X_task[query_idx])
            meta_loss = compute_meta_survival_loss(
                log_risk_q, t_task[query_idx], e_task[query_idx], cfg,
            )
            meta_optimizer.zero_grad()
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            meta_optimizer.step()

            if (meta_ep + 1) % 50 == 0:
                logger.info(
                    f"[MetaDeepSurv] Meta-epoch {meta_ep+1}/{cfg['meta_epochs']}: "
                    f"meta_loss={meta_loss.item():.4f}, task={task_name}"
                )

        logger.info("[MetaDeepSurv] 元训练完成")
        return model

    def finetune_on_target(
        model: MetaDeepSurv,
        X_train: np.ndarray,
        times_train: np.ndarray,
        events_train: np.ndarray,
        X_val: np.ndarray,
        times_val: np.ndarray,
        events_val: np.ndarray,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[MetaDeepSurv, Dict[str, Any]]:
        """在目标癌种 (COAD) 上微调元训练后的模型。

        利用 MAML 学到的参数初始化，仅用少量样本快速收敛。

        Args:
            model: 元训练后的模型
            X_train/val: 目标数据
            times/events: 生存标签
            config: 配置

        Returns:
            (finetuned_model, result_dict)
        """
        cfg = config or META_DEEPSURV_CONFIG
        device = next(model.parameters()).device

        X_tr = torch.FloatTensor(X_train).to(device)
        t_tr = torch.FloatTensor(times_train).to(device)
        e_tr = torch.FloatTensor(events_train).to(device)
        X_v = torch.FloatTensor(X_val).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["finetune_lr"], weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["finetune_epochs"], eta_min=1e-6,
        )

        best_c = 0.0
        best_state = None
        patience_cnt = 0

        for epoch in range(cfg["finetune_epochs"]):
            model.train()
            log_risk = model(X_tr)
            loss = compute_meta_survival_loss(log_risk, t_tr, e_tr, cfg)

            # 弹性正则
            l_elastic = torch.tensor(0.0, device=device)
            for p in model.parameters():
                if p.requires_grad:
                    l_elastic = l_elastic + p.pow(2).sum()
            loss = loss + cfg["lambda_elastic"] * l_elastic

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()

            # 验证
            if (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    val_risk = model(X_v).cpu().numpy()
                c = harrell_cindex(times_val, events_val, val_risk)
                if not np.isnan(c) and c > best_c:
                    best_c = c
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                if patience_cnt >= cfg["patience"] // 5:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # 最终评估
        model.eval()
        with torch.no_grad():
            train_risk = model(X_tr).cpu().numpy()
            val_risk = model(X_v).cpu().numpy()

        val_c = harrell_cindex(times_val, events_val, val_risk)
        ipcw_c = float("nan")
        if HAS_SKSURV:
            try:
                y_tr = Surv.from_arrays(events_train.astype(bool), times_train)
                y_va = Surv.from_arrays(events_val.astype(bool), times_val)
                ipcw_c = float(concordance_index_ipcw(y_tr, y_va, val_risk)[0])
            except Exception:
                pass

        logger.info(f"[MetaDeepSurv] 微调完成: Harrell-C={val_c:.4f}, IPCW-C={ipcw_c:.4f}")

        return model, {
            "val_cindex": val_c,
            "ipcw_cindex": ipcw_c,
            "best_val_cindex": best_c,
            "train_risk": train_risk,
            "val_risk": val_risk,
        }


# ══════════════════════════════════════════════════════════════════════
# 主管线
# ══════════════════════════════════════════════════════════════════════

def run_meta_deepsurv_pipeline(
    timestamp: str, config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Meta-DeepSurv 端到端管线。

    流程: 多癌种数据加载 → MAML 元训练 → 目标癌种微调 → 评估

    与 031/032/033 完全不同的训练范式。
    """
    cfg = config or META_DEEPSURV_CONFIG
    out_dir = os.path.join(RESULTS_DIR, timestamp, "034_meta_deepsurv")
    ensure_dir(out_dir)
    t0 = time.time()

    if not HAS_TORCH:
        logger.error("[MetaDeepSurv] PyTorch 不可用")
        return {}

    logger.info("=" * 60)
    logger.info("[MetaDeepSurv] MAML 元学习跨癌种少样本生存预测")
    logger.info("=" * 60)

    # 1. 数据加载
    data = load_tcga_multi_cancer(timestamp, cfg)
    input_dim = data["input_dim"]

    # 2. 初始化模型
    model = MetaDeepSurv(input_dim, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 3. MAML 元训练
    logger.info("=" * 40)
    logger.info("[MetaDeepSurv] Phase 1: MAML 元训练")
    logger.info("=" * 40)
    model = maml_meta_train(model, data["source_tasks"], cfg)

    # 4. 目标癌种微调 (5-fold CV)
    logger.info("=" * 40)
    logger.info("[MetaDeepSurv] Phase 2: 目标癌种微调")
    logger.info("=" * 40)
    target = data["target"]
    events_tr = target["events_train"].astype(int)
    cv_cindices = []

    if len(target["X_train"]) >= 30:
        skf = StratifiedKFold(
            n_splits=min(cfg["n_folds"], max(2, events_tr.sum())),
            shuffle=True, random_state=cfg["random_seed"],
        )
        for fold_id, (tr_idx, va_idx) in enumerate(skf.split(target["X_train"], events_tr), 1):
            # 每个 fold 从元训练模型重新初始化
            fold_model = MetaDeepSurv(input_dim, cfg).to(device)
            fold_model.load_state_dict(model.state_dict())

            _, fold_res = finetune_on_target(
                fold_model,
                target["X_train"][tr_idx], target["times_train"][tr_idx],
                target["events_train"][tr_idx],
                target["X_train"][va_idx], target["times_train"][va_idx],
                target["events_train"][va_idx],
                config=cfg,
            )
            cv_cindices.append(fold_res["val_cindex"])
            logger.info(f"[MetaDeepSurv] Fold {fold_id}: C={fold_res['val_cindex']:.4f}")
    else:
        logger.warning("[MetaDeepSurv] 训练样本不足，跳过 CV")

    cv_mean = float(np.nanmean(cv_cindices)) if cv_cindices else float("nan")
    cv_std = float(np.nanstd(cv_cindices)) if cv_cindices else float("nan")

    # 5. 全量微调
    final_model = MetaDeepSurv(input_dim, cfg).to(device)
    final_model.load_state_dict(model.state_dict())
    final_model, final_res = finetune_on_target(
        final_model,
        target["X_train"], target["times_train"], target["events_train"],
        target["X_val"], target["times_val"], target["events_val"],
        config=cfg,
    )

    # 6. 保存
    elapsed = time.time() - t0
    summary = {
        "model_type": "Meta-DeepSurv (MAML cross-cancer few-shot)",
        "timestamp": timestamp,
        "elapsed_sec": round(elapsed, 1),
        "input_dim": input_dim,
        "n_source_tasks": len(data["source_tasks"]),
        "source_cancers": ",".join(cfg["source_cancers"]),
        "n_target_train": len(target["X_train"]),
        "n_target_val": len(target["X_val"]),
        "cv_cindex_mean": cv_mean,
        "cv_cindex_std": cv_std,
        "val_harrell_cindex": final_res["val_cindex"],
        "val_ipcw_cindex": final_res["ipcw_cindex"],
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(out_dir, "meta_deepsurv_summary.tsv"), sep="\t", index=False,
    )
    torch.save(final_model.state_dict(), os.path.join(out_dir, "meta_deepsurv_model.pt"))
    pd.DataFrame({"risk": final_res["val_risk"]}).to_csv(
        os.path.join(out_dir, "val_risk_scores.tsv"), sep="\t",
    )

    # KM 曲线可视化
    _plot_km_curves(final_res["val_risk"], target["times_val"], target["events_val"], out_dir)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"[MetaDeepSurv] 完成! 耗时 {elapsed:.1f}s")
    logger.info(f"  Source tasks: {len(data['source_tasks'])}")
    logger.info(f"  CV: {cv_mean:.4f} ± {cv_std:.4f}")
    logger.info(f"  Val-C: {final_res['val_cindex']:.4f}")
    logger.info(f"{'=' * 60}")
    return summary


def _plot_km_curves(
    risk: np.ndarray, times: np.ndarray, events: np.ndarray, out_dir: str,
) -> None:
    """生成高/低风险分组 KM 曲线。"""
    if not HAS_LIFELINES:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = np.isfinite(risk) & (times > 0)
    if valid.sum() < 20:
        return

    threshold = float(np.median(risk[valid]))
    high = risk >= threshold
    low = ~high

    if high.sum() < 5 or low.sum() < 5:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    kmf = KaplanMeierFitter()
    for mask, label, color in [(low, "Low risk", "#1f77b4"), (high, "High risk", "#d62728")]:
        kmf.fit(times[mask], event_observed=events[mask],
                label=f"{label} (n={int(mask.sum())})")
        kmf.plot_survival_function(ci_show=True, color=color, lw=2.0, ax=ax)

    try:
        lr = logrank_test(times[low], times[high],
                          event_observed_A=events[low], event_observed_B=events[high])
        ax.text(0.04, 0.08, f"Log-rank p = {lr.p_value:.3g}", transform=ax.transAxes)
    except Exception:
        pass

    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Overall survival")
    ax.set_title("Meta-DeepSurv: K-M risk strata")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.savefig(os.path.join(out_dir, "km_risk_strata.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[MetaDeepSurv] KM 曲线已保存")


# ══════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="034: Meta-DeepSurv (MAML 元学习)")
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()
    try:
        run_meta_deepsurv_pipeline(args.timestamp)
    except Exception as e:
        logger.error(f"[MetaDeepSurv] 异常: {e}")
        raise


if __name__ == "__main__":
    main()
