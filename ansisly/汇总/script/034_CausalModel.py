"""034_CausalModel.py — Meta-DeepSurv: Reptile 元学习跨癌种少样本生存预测

因果深度生存模型变体 3（元学习迁移版）：
使用 Reptile 元学习框架，在多个源癌种
(BRCA, LUAD, KIRC, LIHC, UCEC, STAD, ESCA, PAAD) 上学习鲁棒的
参数初始化，然后在目标癌种 (COAD/READ) 上仅用少量样本快速微调。

核心差异 (vs 031 / 032 / 033):
    - 完全不同的泛化策略：元学习 (Reptile) 而非因果编码/IB/扩散增强
    - 跨癌种训练：利用 TCGA 多癌种真实数据作为 source tasks
    - 少样本适应：目标癌种仅需少量样本即可微调
    - 参考 MMOSurv (Bioinformatics 2024) 的 Reptile 元学习框架
    - 不使用 CausalEGM/IB/对抗去混杂/对比学习

架构：
    X → MetaEncoder [256→128→Z] → SurvivalHead [64→32→1] → log risk
    Reptile 内循环: K 步 SGD on support set → 外循环: 参数插值更新

Reptile 更新规则：
    theta = theta + meta_lr * (theta' - theta)
    theta' = fine-tuned params after K inner-loop SGD steps

参考文献：
    - Reptile: Nichol et al., arXiv 2018
    - MMOSurv: Wen & Li, Bioinformatics 2024 (Reptile 元学习)
    - MAML: Finn et al., ICML 2017
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
    # Reptile 元学习
    "meta_lr": 1e-3,                 # Reptile 外循环插值系数
    "inner_lr": 1e-2,                # 内循环学习率 (task-specific SGD)
    "inner_steps": 5,                # 内循环 K 步
    "n_way": 1,                      # 每个 task 的类别数 (生存=1)
    "k_support": 30,                 # support set 大小
    "k_query": 20,                   # query set 大小
    # 源癌种 (用于元训练)
    # 当 real_multicancer_dir 有数据时仅使用有文件的癌种;
    # 回退模拟模式时使用全部 8 个名称切分 COAD 数据
    "source_cancers": ["STAD", "ESCA", "PAAD",
                       "BRCA", "LUAD", "KIRC", "LIHC", "UCEC"],
    # 训练
    "meta_epochs": 300,              # 元训练轮数
    "finetune_epochs": 100,          # 目标癌种微调轮数
    "finetune_lr": 1e-3,
    "batch_size": 32,
    "patience": 30,
    "n_folds": 5,
    "grad_clip": 5.0,
    # 损失
    "lambda_rank": 0.3,
    "lambda_elastic": 1e-4,
    # 真实多癌种数据:
    #   real_multicancer_dir=None 时回退到 COAD 模拟切分
    #   use_real_multicancer=False 时强制模拟模式（即使目录存在）
    "real_multicancer_dir": os.path.join(
        os.path.dirname(os.path.dirname(_SCRIPT_DIR)), "rawData", "multicancer_preprocessed"
    ),
    "use_real_multicancer": True,   # 开关: True=启用真实数据, False=强制模拟
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
        """Reptile 元学习 DeepSurv 模型。

        与 031/032/033 完全不同的模型类设计：
        1. Reptile 元学习：内循环 SGD + 外循环参数插值
        2. 无 IB/对抗/对比/扩散组件

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

# ══════════════════════════════════════════════════════════════════════
# 数据加载与 Task 构建
# ══════════════════════════════════════════════════════════════════════

def _load_single_cancer_real(
    cancer_type: str, data_dir: str, shared_genes: List[str],
) -> Dict[str, np.ndarray]:
    """加载单个癌种的真实表达+生存数据，对齐到共享基因集。"""
    cancer_upper = cancer_type.upper()
    expr_file = os.path.join(data_dir, f"{cancer_upper}_gene_expression.tsv")
    clin_file = os.path.join(data_dir, f"{cancer_upper}_os_clinical.tsv")

    if not os.path.exists(expr_file) or not os.path.exists(clin_file):
        raise FileNotFoundError(f"{cancer_type}: 数据文件不存在")

    expr = pd.read_csv(expr_file, sep="\t", index_col=0)
    clin = pd.read_csv(clin_file, sep="\t")
    # 统一列名：部分文件用 patient_id，部分用 PATIENT_ID
    _id_col = "patient_id" if "patient_id" in clin.columns else [c for c in clin.columns if c.upper() == "PATIENT_ID"]
    if not isinstance(_id_col, str):
        _id_col = _id_col[0] if _id_col else "patient_id"
    clin.rename(columns={_id_col: "patient_id"}, inplace=True)

    # 对齐基因 (严格使用 shared_genes，缺失基因填 0)
    avail = [g for g in shared_genes if g in expr.columns]
    if len(avail) < 50:
        raise ValueError(f"{cancer_type}: 共享基因仅 {len(avail)} 个")
    expr = expr[avail]
    expr = expr.reindex(columns=shared_genes, fill_value=0.0)

    clin["patient_id"] = clin["patient_id"].astype(str).map(normalize_patient_id)
    clin["time_months"] = pd.to_numeric(clin["time_months"], errors="coerce")
    clin["event"] = coerce_event(clin["event"]).astype(int)
    clin = clin[clin["patient_id"].ne("") & clin["time_months"].notna() & (clin["time_months"] > 0)]
    clin.index = clin["patient_id"].values

    common = expr.index.intersection(clin.index)
    if len(common) < 20:
        raise ValueError(f"{cancer_type}: 对齐后仅 {len(common)} 样本")

    X = expr.loc[common].fillna(0.0)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=shared_genes)

    return {
        "X": X_scaled.to_numpy(dtype=np.float32),
        "times": clin.loc[common, "time_months"].to_numpy(dtype=np.float32),
        "events": clin.loc[common, "event"].to_numpy(dtype=np.float32),
        "n_samples": len(common),
        "n_genes": len(shared_genes),
    }


def load_tcga_multi_cancer(
    timestamp: str, config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """加载 TCGA 多癌种数据。

    支持两种模式:
    1. 真实多癌种: 当 real_multicancer_dir 存在且包含数据时
    2. 模拟多 task: 仅用 COAD 数据随机切分 (回退模式)
    """
    cfg = config or META_DEEPSURV_CONFIG
    real_dir = cfg.get("real_multicancer_dir")
    use_real = cfg.get("use_real_multicancer", True)

    # ── 尝试真实多癌种数据 ──
    if real_dir and os.path.isdir(real_dir) and use_real:
        logger.info(f"[MetaDeepSurv] 使用真实多癌种数据: {real_dir}")
        # 第一遍: 收集各癌种 top-k 基因
        gene_sets = {}
        for cancer in cfg["source_cancers"]:
            expr_f = os.path.join(real_dir, f"{cancer.upper()}_gene_expression.tsv")
            if os.path.exists(expr_f):
                df = pd.read_csv(expr_f, sep="\t", index_col=0, nrows=5)
                var_all = pd.read_csv(expr_f, sep="\t", index_col=0).var(axis=0, skipna=True)
                var_all = var_all[var_all > 0].sort_values(ascending=False)
                gene_sets[cancer] = set(var_all.head(cfg["variance_topk"]).index)
            else:
                logger.warning(f"[{cancer}] 数据文件不存在，跳过")

        # 共享基因: 至少 N-2 个癌种共有 (去重 + top-k)
        if gene_sets:
            n_required = max(len(gene_sets) - 2, 1)
            from collections import Counter
            gene_counts = Counter(g for gs in gene_sets.values() for g in gs)
            # 先收集满足条件的基因（保持方差排序顺序）
            all_candidate_genes = []
            seen = set()
            for gs in gene_sets.values():
                for g in gs:
                    if g not in seen and gene_counts[g] >= n_required:
                        all_candidate_genes.append(g)
                        seen.add(g)
            shared_genes = all_candidate_genes[:cfg["variance_topk"]]
            logger.info(f"  共享基因: {len(shared_genes)} (去重后, 至少{n_required}个癌种共有)")
        else:
            shared_genes = []

        # 第二遍: 加载数据
        task_data = {}
        for cancer in cfg["source_cancers"]:
            if cancer not in gene_sets:
                continue
            try:
                task_data[cancer] = _load_single_cancer_real(cancer, real_dir, shared_genes)
                logger.info(f"  [{cancer}] {task_data[cancer]['n_samples']} 样本, "
                            f"{task_data[cancer]['n_genes']} 基因")
            except Exception as e:
                logger.warning(f"  [{cancer}] 加载失败: {e}")

        if len(task_data) >= 3:
            # 目标 task: COAD，使用相同共享基因集
            target_data = _load_coad_target(cfg, shared_genes)
            return {
                "source_tasks": task_data,
                "target": target_data,
                "gene_names": shared_genes,
                "input_dim": len(shared_genes),
            }
        logger.warning("真实数据加载不足 3 个癌种，回退到模拟模式")

    # ── 回退: COAD 模拟多 task ──
    return _load_simulated_multitask(cfg)


def _load_coad_target(cfg: Dict[str, Any], shared_genes: Optional[List[str]] = None) -> Dict[str, Any]:
    """加载 COAD 目标 task (train/val)。

    Args:
        cfg: 配置
        shared_genes: 若提供，COAD 表达数据将过滤为仅这些基因（与 source tasks 对齐）
    """
    preproc_dir = ESSENTIAL_DIR
    clinical = safe_read_tsv(os.path.join(preproc_dir, "tcga_os_clinical_endpoint_qc.tsv"))
    time_col, event_col = detect_event_time_columns(clinical)
    ep = clinical.copy()
    ep["PATIENT_ID"] = ep["PATIENT_ID"].astype(str).map(normalize_patient_id)
    ep["time_months"] = pd.to_numeric(ep[time_col], errors="coerce")
    ep["event"] = coerce_event(ep[event_col]).astype(int)
    ep = ep[ep["PATIENT_ID"].ne("") & ep["time_months"].notna() & (ep["time_months"] >= 0)]
    ep.index = ep["PATIENT_ID"].values

    expr_df = safe_read_tsv(os.path.join(preproc_dir, "gene_expression_curated.tsv"), index_col=0)
    expr_df = expr_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # 过滤到共享基因集（与 source tasks 对齐）
    if shared_genes is not None:
        avail = [g for g in shared_genes if g in expr_df.columns]
        missing = [g for g in shared_genes if g not in expr_df.columns]
        expr_df = expr_df[avail]
        expr_df = expr_df.reindex(columns=shared_genes, fill_value=0.0)
        logger.info(f"[MetaDeepSurv] COAD target: 使用 {len(shared_genes)} 共享基因 "
                    f"({len(avail)} 可用, {len(missing)} 填零)")

    split_path = os.path.join(preproc_dir, "tcga_train_internal_validation_split.tsv")
    if not os.path.exists(split_path):
        split_path = os.path.join(DATA_DIR, "survival_models", "tcga_train_internal_validation_split.tsv")
    if os.path.exists(split_path):
        split_df = safe_read_tsv(split_path)
        train_ids = set(split_df.loc[split_df["split"] == "train", "PATIENT_ID"].astype(str))
        val_ids = set(split_df.loc[split_df["split"] != "train", "PATIENT_ID"].astype(str))
    else:
        rng = np.random.RandomState(cfg["random_seed"])
        all_ids = list(ep.index)
        rng.shuffle(all_ids)
        sp = int(0.8 * len(all_ids))
        train_ids, val_ids = set(all_ids[:sp]), set(all_ids[sp:])

    common_tr = expr_df.index.intersection(ep.index)
    X_all = expr_df.loc[common_tr].fillna(0.0)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_all), index=X_all.index, columns=X_all.columns)

    tr_mask = X_scaled.index.isin(train_ids)
    va_mask = X_scaled.index.isin(val_ids)
    ep_tr = ep.loc[ep.index.isin(train_ids)]
    ep_va = ep.loc[ep.index.isin(val_ids)]

    return {
        "X_train": X_scaled[tr_mask].to_numpy(dtype=np.float32),
        "times_train": ep_tr["time_months"].to_numpy(dtype=np.float32),
        "events_train": ep_tr["event"].to_numpy(dtype=np.float32),
        "X_val": X_scaled[va_mask].to_numpy(dtype=np.float32),
        "times_val": ep_va["time_months"].to_numpy(dtype=np.float32),
        "events_val": ep_va["event"].to_numpy(dtype=np.float32),
    }


def _load_simulated_multitask(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """回退模式: 仅用 COAD 数据，随机切分模拟多 source tasks。"""
    preproc_dir = ESSENTIAL_DIR
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

    variances = expr_df.var(axis=0, skipna=True).dropna()
    variances = variances[variances > 0].sort_values(ascending=False)
    topk_genes = list(variances.head(cfg["variance_topk"]).index)

    X_all = expr_df[topk_genes].copy()
    X_all.index = X_all.index.astype(str)
    X_all = X_all.fillna(X_all.median()).fillna(0.0)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_all), index=X_all.index, columns=topk_genes)

    common_idx = X_scaled.index.intersection(ep.index)
    X_aligned = X_scaled.loc[common_idx]
    ep_aligned = ep.loc[common_idx]

    split_path = os.path.join(preproc_dir, "tcga_train_internal_validation_split.tsv")
    if not os.path.exists(split_path):
        split_path = os.path.join(DATA_DIR, "survival_models", "tcga_train_internal_validation_split.tsv")
    if os.path.exists(split_path):
        split_df = safe_read_tsv(split_path)
        train_ids = set(split_df.loc[split_df["split"] == "train", "PATIENT_ID"].astype(str))
        val_ids = set(split_df.loc[split_df["split"] != "train", "PATIENT_ID"].astype(str))
    else:
        rng = np.random.RandomState(cfg["random_seed"])
        all_ids = list(X_aligned.index)
        rng.shuffle(all_ids)
        split_point = int(0.8 * len(all_ids))
        train_ids = set(all_ids[:split_point])
        val_ids = set(all_ids[split_point:])

    X_train = X_aligned[X_aligned.index.isin(train_ids)]
    X_val = X_aligned[X_aligned.index.isin(val_ids)]
    ep_train = ep_aligned[ep_aligned.index.isin(train_ids)]
    ep_val = ep_aligned[ep_aligned.index.isin(val_ids)]

    logger.info(f"[MetaDeepSurv] COAD(模拟) train={len(X_train)}, val={len(X_val)}")

    n_tasks = len(cfg["source_cancers"])
    task_data = {}
    rng = np.random.RandomState(cfg["random_seed"])
    train_indices = np.arange(len(X_train))
    task_size = max(len(train_indices) // (n_tasks + 1), 25)

    for i, cancer in enumerate(cfg["source_cancers"]):
        start = i * task_size
        end = min(start + task_size, len(train_indices))
        if end - start < 20:
            idx = rng.choice(len(train_indices), min(task_size, len(train_indices)), replace=False)
        else:
            idx = train_indices[start:end]
        task_data[cancer] = {
            "X": X_train.iloc[idx].to_numpy(dtype=np.float32),
            "times": ep_train.iloc[idx]["time_months"].to_numpy(dtype=np.float32),
            "events": ep_train.iloc[idx]["event"].to_numpy(dtype=np.float32),
        }

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
# Reptile 元训练
# ══════════════════════════════════════════════════════════════════════

if HAS_TORCH:

    def maml_meta_train(
        model: MetaDeepSurv,
        source_tasks: Dict[str, Dict[str, np.ndarray]],
        config: Optional[Dict[str, Any]] = None,
    ) -> MetaDeepSurv:
        """Reptile 元训练。

        与 031/032/033 完全不同的训练范式：
        - 031/032/033: 在单一训练集上端到端训练
        - 034: 在多个 source tasks 上元训练，学习跨任务通用初始化

        Reptile 算法:
            for each meta_epoch:
                1. 采样一个 task
                2. 保存当前参数 theta
                3. 在 task 上执行 K 步 SGD 得到 theta'
                4. 参数插值: theta += meta_lr * (theta' - theta)

        Args:
            model: 全局模型
            source_tasks: 源任务数据
            config: 配置

        Returns:
            元训练后的模型
        """
        cfg = config or META_DEEPSURV_CONFIG
        device = next(model.parameters()).device
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

            # ── Reptile 外循环: 参数插值 theta += meta_lr * (theta' - theta) ──
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.data.add_(
                        cfg["meta_lr"] * (param.data - initial_params[name])
                    )

            # query set 监控日志 (不参与梯度更新)
            if (meta_ep + 1) % 50 == 0:
                model.eval()
                with torch.no_grad():
                    log_risk_q = model(X_task[query_idx])
                    monitor_loss = compute_meta_survival_loss(
                        log_risk_q, t_task[query_idx], e_task[query_idx], cfg,
                    )
                logger.info(
                    f"[MetaDeepSurv] Meta-epoch {meta_ep+1}/{cfg['meta_epochs']}: "
                    f"monitor_loss={monitor_loss.item():.4f}, task={task_name}"
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

        利用 Reptile 学到的参数初始化，仅用少量样本快速收敛。

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

    流程: 多癌种数据加载 → Reptile 元训练 → 目标癌种微调 → 评估

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
    logger.info("[MetaDeepSurv] Reptile 元学习跨癌种少样本生存预测")
    logger.info("=" * 60)

    # 1. 数据加载
    data = load_tcga_multi_cancer(timestamp, cfg)
    input_dim = data["input_dim"]

    # 2. 初始化模型
    model = MetaDeepSurv(input_dim, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 3. Reptile 元训练
    logger.info("=" * 40)
    logger.info("[MetaDeepSurv] Phase 1: Reptile 元训练")
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
        "model_type": "Meta-DeepSurv (Reptile cross-cancer few-shot)",
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
    parser = argparse.ArgumentParser(description="034: Meta-DeepSurv (Reptile 元学习)")
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()
    try:
        run_meta_deepsurv_pipeline(args.timestamp)
    except Exception as e:
        logger.error(f"[MetaDeepSurv] 异常: {e}")
        raise


if __name__ == "__main__":
    main()
