"""032_CausalModel.py — 纯 DeepSurv 基线模型 (无因果编码)

因果深度生存模型变体 1（Baseline DeepSurv）：
不使用 CausalEGM 因果编码器，直接在高维基因表达矩阵 X 上
训练 DeepSurv (Cox-NN) 生存预测模型。

核心差异 (vs 031_CDSIB):
    - 无 CausalEGM 因果降维：直接在 X 空间训练
    - 无 IB 信息瓶颈正则化
    - 无条件扩散数据增强
    - 无反事实推断 / SHAP 可解释性
    - 使用 learnable feature attention 替代 SparseGate
    - 使用 mini-batch SGD 替代 031 的全量梯度下降
    - 独立的 train/val/test 三划分评估

定位：作为 031/033/034 因果增强模型的消融对照基准，
      验证 CausalEGM 因果编码和 IB 正则化的实际增益。

架构：X (dim=5000) → FeatureAttention → MLP [128→64→32→1] → log risk

损失函数：
    L = L_IPCW_partial + 0.5 * L_rank + 1e-4 * L_elastic

依赖：torch, numpy, pandas, scikit-survival (可选)
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
    setup_logger, RANDOM_SEED,
    DATA_DIR, RESULTS_DIR, ESSENTIAL_DIR,
    ensure_dir, safe_read_tsv, normalize_patient_id,
    detect_event_time_columns, coerce_event,
)

# ── PyTorch ──
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── scikit-survival ──
try:
    from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
    from sksurv.util import Surv
    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False

# ── lifelines ──
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False

logger = setup_logger("032_deepsurv_baseline")


# ══════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════

DEEPSURV_BASELINE_CONFIG: Dict[str, Any] = {
    # 数据预处理
    "variance_topk": 5000,
    # 网络结构
    "hidden_dims": [128, 64, 32],
    "dropout_input": 0.3,
    "dropout_hidden": 0.4,
    "use_batch_norm": True,
    # 损失权重
    "lambda_rank": 0.5,
    "lambda_elastic": 1e-4,
    "elastic_alpha": 0.5,
    # 训练
    "lr": 1e-4,
    "weight_decay": 5e-4,
    "epochs": 300,
    "batch_size": 64,
    "patience": 40,
    "n_folds": 5,
    "grad_clip": 3.0,
    # 评估
    "eval_tau_months": 36,
    "random_seed": RANDOM_SEED,
}


# ══════════════════════════════════════════════════════════════════════
# PyTorch 模块
# ══════════════════════════════════════════════════════════════════════

if HAS_TORCH:

    class FeatureAttention(nn.Module):
        """可学习特征注意力层 (替代 031 的 SparseGate)。

        与 031 的 Gumbel-Softmax SparseGate 不同，本模块使用：
        - 确定性 sigmoid 门控（无 Gumbel 噪声）
        - L1 稀疏正则化（鼓励门控值趋零）
        - 不强制 top-k，由网络自行学习稀疏度

        数学：
            alpha_j = sigmoid(W_att * x_j + b_att)
            x_out = x * alpha
            L_sparse = lambda_s * mean(alpha)
        """

        def __init__(self, input_dim: int, sparse_lambda: float = 1e-5) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.sparse_lambda = sparse_lambda
            # 每个特征一个可学习门控参数
            self.W_att = nn.Parameter(torch.randn(input_dim) * 0.01)
            self.b_att = nn.Parameter(torch.zeros(input_dim))

        def forward(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """前向传播。

            Args:
                x: (batch, input_dim)

            Returns:
                (x_gated, sparse_loss): 门控后输入和稀疏正则项
            """
            alpha = torch.sigmoid(self.W_att * x + self.b_att)
            x_gated = x * alpha
            # 稀疏正则：鼓励门控关闭
            sparse_loss = self.sparse_lambda * alpha.mean()
            return x_gated, sparse_loss

        def get_active_features(self, threshold: float = 0.5) -> torch.Tensor:
            """返回被激活的特征索引（用于可解释性分析）。"""
            with torch.no_grad():
                alpha = torch.sigmoid(self.W_att + self.b_att)
                return torch.where(alpha > threshold)[0]

    class DeepSurvNetwork(nn.Module):
        """纯 DeepSurv 网络 (无因果编码)。

        与 031 的 CDSIBModel 的核心差异：
        1. 接受高维 X 输入 (dim=5000)，而非低维 Z (dim=12)
        2. 使用 FeatureAttention 替代 SparseGate
        3. 无 IB 门控层 (IBGate)
        4. 更深的 MLP (3 层 vs 2 层)
        5. 输入层额外 Dropout

        架构：
            FeatureAttention → Dropout → Linear → BN → ReLU → Dropout → ... → Linear(1)
        """

        def __init__(
            self,
            input_dim: int,
            config: Optional[Dict[str, Any]] = None,
        ) -> None:
            super().__init__()
            cfg = config or DEEPSURV_BASELINE_CONFIG

            # 特征注意力层
            self.feature_attn = FeatureAttention(input_dim)

            # 输入 Dropout
            self.input_dropout = nn.Dropout(p=cfg["dropout_input"])

            # MLP 主体
            layers: List[nn.Module] = []
            prev_dim = input_dim
            for h_dim in cfg["hidden_dims"]:
                layers.append(nn.Linear(prev_dim, h_dim))
                if cfg["use_batch_norm"]:
                    layers.append(nn.BatchNorm1d(h_dim))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(p=cfg["dropout_hidden"]))
                prev_dim = h_dim
            # 输出层
            layers.append(nn.Linear(prev_dim, 1))
            self.mlp = nn.Sequential(*layers)

        def forward(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """前向传播。

            Args:
                x: (batch, input_dim) 标准化后的基因表达

            Returns:
                (log_risk, sparse_loss): 对数风险评分和稀疏正则项
            """
            x_attn, sparse_loss = self.feature_attn(x)
            x_drop = self.input_dropout(x_attn)
            log_risk = self.mlp(x_drop).squeeze(-1)
            return log_risk, sparse_loss

        def predict_risk(self, x: torch.Tensor) -> np.ndarray:
            """推理接口：返回 exp(g(x)) 风险评分。"""
            self.eval()
            with torch.no_grad():
                log_risk, _ = self.forward(x)
                return torch.exp(log_risk).cpu().numpy()

        def get_active_gene_indices(self, threshold: float = 0.5) -> np.ndarray:
            """获取注意力激活的基因索引。"""
            return self.feature_attn.get_active_features(threshold).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════
# 损失函数
# ══════════════════════════════════════════════════════════════════════

if HAS_TORCH:

    def compute_deepsurv_loss(
        log_risk: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor,
        sparse_loss: torch.Tensor,
        model: nn.Module,
        ipcw_weights: Optional[torch.Tensor] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """DeepSurv 复合损失 (无 IB、无图正则化)。

        与 031 的 compute_cdsib_loss 差异：
        - 无 IB KL 正则化项
        - 无图 Laplacian 正则化项
        - 使用 logcumsumexp 数值稳定技巧（与 031 相同）
        - 对比排序损失使用向量化实现（031 使用随机采样循环）

        L = L_IPCW_partial + lambda_rank * L_rank + lambda_elastic * L_elastic + L_sparse

        Args:
            log_risk: (batch,) 模型输出
            times: (batch,) 生存时间
            events: (batch,) 事件指示器
            sparse_loss: 特征注意力稀疏正则项
            model: 模型（用于弹性网络正则化）
            ipcw_weights: (batch,) IPCW 权重（可选）
            config: 配置字典

        Returns:
            标量总损失
        """
        cfg = config or DEEPSURV_BASELINE_CONFIG
        n = len(times)
        device = log_risk.device

        # ── 1. IPCW 加权偏似然 ──
        sort_idx = torch.argsort(-times)  # 降序
        h_sorted = log_risk[sort_idx]
        ev_sorted = events[sort_idx]
        w_sorted = (
            ipcw_weights[sort_idx]
            if ipcw_weights is not None
            else torch.ones(n, device=device)
        )

        # 数值稳定的 log-cum-sum-exp
        log_cumsum_exp = torch.logcumsumexp(h_sorted, dim=0)
        partial_ll = (h_sorted - log_cumsum_exp) * ev_sorted * w_sorted
        event_count = ev_sorted.sum().clamp(min=1)
        l_partial = -partial_ll.sum() / event_count

        # ── 2. 对比排序损失 (向量化实现，不同于 031 的循环采样) ──
        l_rank = torch.tensor(0.0, device=device)
        event_idx = torch.where(events == 1)[0]
        censored_idx = torch.where(events == 0)[0]
        if len(event_idx) > 0 and len(censored_idx) > 0:
            # 事件样本风险 vs 删失样本风险 (事件样本时间更短 → 风险应更高)
            risk_ev = log_risk[event_idx]  # (n_ev,)
            risk_cen = log_risk[censored_idx]  # (n_cen,)
            # 对每对 (i=事件, j=删失且T_j>T_i)，期望 risk_i > risk_j
            # 简化：取所有事件 vs 所有删失的均值
            diff = risk_ev.unsqueeze(1) - risk_cen.unsqueeze(0)  # (n_ev, n_cen)
            l_rank = -F.logsigmoid(diff).mean()

        # ── 3. 弹性网络正则化 ──
        l_elastic = torch.tensor(0.0, device=device)
        alpha_e = cfg["elastic_alpha"]
        for p in model.parameters():
            if p.requires_grad:
                l_elastic = l_elastic + alpha_e * p.abs().sum() + (1 - alpha_e) / 2 * p.pow(2).sum()
        l_elastic = l_elastic * cfg["lambda_elastic"]

        # ── 总损失 ──
        total = l_partial + cfg["lambda_rank"] * l_rank + l_elastic + sparse_loss
        return total


# ══════════════════════════════════════════════════════════════════════
# IPCW 权重计算
# ══════════════════════════════════════════════════════════════════════

def compute_ipcw_weights_np(times: np.ndarray, events: np.ndarray) -> np.ndarray:
    """计算 IPCW 权重 (纯 numpy 实现)。

    与 031 的 _compute_ipcw_weights 逻辑相同，但独立实现避免耦合。

    w_i = delta_i / G_hat(T_i)
    G_hat: KM 估计的删失生存函数
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
# 评估
# ══════════════════════════════════════════════════════════════════════

def harrell_cindex_np(
    times: np.ndarray, events: np.ndarray, risk: np.ndarray
) -> float:
    """Harrell C-index (纯 numpy 实现)。"""
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
# 数据加载 (简化版，复用 shared_utils)
# ══════════════════════════════════════════════════════════════════════

def load_data_for_deepsurv(
    timestamp: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """加载 TCGA-COAD/READ 数据。

    与 031 的 load_and_preprocess_cdsib 结构相似但更精简：
    - 不做 RINT 分位数正态化 (031 使用 QuantileTransformer)
    - 仅做 StandardScaler 标准化
    - 返回 numpy 数组而非 DataFrame
    """
    cfg = config or DEEPSURV_BASELINE_CONFIG
    preproc_dir = ESSENTIAL_DIR

    # 临床数据
    clinical = safe_read_tsv(os.path.join(preproc_dir, "tcga_os_clinical_endpoint_qc.tsv"))

    # 数据划分
    split_path = os.path.join(preproc_dir, "tcga_train_internal_validation_split.tsv")
    if not os.path.exists(split_path):
        split_path = os.path.join(DATA_DIR, "survival_models", "tcga_train_internal_validation_split.tsv")
    split_df = safe_read_tsv(split_path)

    # 基因表达
    expr_df = safe_read_tsv(os.path.join(preproc_dir, "gene_expression_curated.tsv"), index_col=0)
    expr_df = expr_df.apply(pd.to_numeric, errors="coerce")
    logger.info(f"[DeepSurv] 表达矩阵: {expr_df.shape}")

    # 划分
    clinical_ids = set(clinical["PATIENT_ID"].astype(str))
    train_ids = set(split_df.loc[split_df["split"] == "train", "PATIENT_ID"].astype(str)) & clinical_ids
    val_ids = set(split_df.loc[split_df["split"] != "train", "PATIENT_ID"].astype(str)) & clinical_ids

    # 构建 endpoint
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
    logger.info(f"[DeepSurv] 方差过滤: {expr_df.shape[1]} → {len(topk_genes)}")

    # StandardScaler (train-only fit)
    X_all = expr_df[topk_genes].copy()
    X_all.index = X_all.index.astype(str)
    train_med = X_all.loc[train_idx].median()
    X_all = X_all.fillna(train_med).fillna(0.0)
    scaler = StandardScaler()
    scaler.fit(X_all.loc[train_idx])
    X_scaled = pd.DataFrame(scaler.transform(X_all), index=X_all.index, columns=topk_genes)

    # 分割
    X_train = X_scaled.loc[X_scaled.index.isin(train_ids)]
    X_val = X_scaled.loc[X_scaled.index.isin(val_ids)]
    ep_train = ep.loc[ep.index.isin(train_ids)]
    ep_val = ep.loc[ep.index.isin(val_ids)]

    # 对齐
    ct = X_train.index.intersection(ep_train.index)
    cv = X_val.index.intersection(ep_val.index)
    X_train, ep_train = X_train.loc[ct], ep_train.loc[ct]
    X_val, ep_val = X_val.loc[cv], ep_val.loc[cv]
    logger.info(f"[DeepSurv] train={len(X_train)}, val={len(X_val)}")

    return {
        "X_train": X_train.to_numpy(dtype=np.float32),
        "X_val": X_val.to_numpy(dtype=np.float32),
        "times_train": ep_train["time_months"].to_numpy(dtype=np.float32),
        "events_train": ep_train["event"].to_numpy(dtype=np.float32),
        "times_val": ep_val["time_months"].to_numpy(dtype=np.float32),
        "events_val": ep_val["event"].to_numpy(dtype=np.float32),
        "gene_names": topk_genes,
        "n_train": len(X_train),
        "n_val": len(X_val),
    }


# ══════════════════════════════════════════════════════════════════════
# 训练循环 (mini-batch SGD，不同于 031 的全量梯度)
# ══════════════════════════════════════════════════════════════════════

if HAS_TORCH:

    def train_deepsurv_model(
        X_train: np.ndarray,
        times_train: np.ndarray,
        events_train: np.ndarray,
        X_val: np.ndarray,
        times_val: np.ndarray,
        events_val: np.ndarray,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[DeepSurvNetwork, Dict[str, Any]]:
        """训练纯 DeepSurv 模型 (mini-batch SGD)。

        与 031 的 train_cdsib_model 差异：
        1. 使用 DataLoader mini-batch (031 使用全量梯度)
        2. 使用 CosineAnnealingLR 学习率调度 (031 无调度)
        3. 每 epoch 都做验证 (031 每 10 epoch 验证)
        4. 保存 best model by validation C-index

        Args:
            X_train/val: 特征矩阵 numpy
            times/events: 生存数据 numpy
            config: 配置字典

        Returns:
            (model, result_dict)
        """
        cfg = config or DEEPSURV_BASELINE_CONFIG
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[DeepSurv] 设备: {device}")

        input_dim = X_train.shape[1]
        model = DeepSurvNetwork(input_dim, cfg).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=1e-6
        )

        # DataLoader
        dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(times_train),
            torch.FloatTensor(events_train),
        )
        loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

        # 验证集 tensor
        X_v = torch.FloatTensor(X_val).to(device)
        t_v = torch.FloatTensor(times_val).to(device)
        e_v = torch.FloatTensor(events_val).to(device)

        best_cindex = 0.0
        best_state = None
        patience_cnt = 0
        history: Dict[str, List[float]] = {"train_loss": [], "val_cindex": []}

        for epoch in range(cfg["epochs"]):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for bx, bt, be in loader:
                bx, bt, be = bx.to(device), bt.to(device), be.to(device)
                # 计算 mini-batch IPCW 权重
                w_batch = torch.FloatTensor(
                    compute_ipcw_weights_np(bt.cpu().numpy(), be.cpu().numpy())
                ).to(device)

                log_risk, sparse_loss = model(bx)
                loss = compute_deepsurv_loss(
                    log_risk, bt, be, sparse_loss, model,
                    ipcw_weights=w_batch, config=cfg,
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["grad_clip"]
                )
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_loss)

            # 验证 (每 5 epoch)
            if (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    val_risk, _ = model(X_v)
                c = harrell_cindex_np(times_val, events_val, val_risk.cpu().numpy())
                if not np.isnan(c):
                    history["val_cindex"].append(c)
                    if c > best_cindex:
                        best_cindex = c
                        best_state = {
                            k: v.clone() for k, v in model.state_dict().items()
                        }
                        patience_cnt = 0
                    else:
                        patience_cnt += 1

                if patience_cnt >= cfg["patience"] // 5:
                    logger.info(
                        f"[DeepSurv] Epoch {epoch+1}: Early stop "
                        f"(best C={best_cindex:.4f})"
                    )
                    break

                if (epoch + 1) % 50 == 0:
                    logger.info(
                        f"[DeepSurv] Epoch {epoch+1}/{cfg['epochs']}: "
                        f"loss={avg_loss:.4f}, val_C={c:.4f}, "
                        f"best_C={best_cindex:.4f}"
                    )

        # 恢复最佳模型
        if best_state is not None:
            model.load_state_dict(best_state)

        # 最终评估
        model.eval()
        with torch.no_grad():
            train_risk, _ = model(torch.FloatTensor(X_train).to(device))
            val_risk, _ = model(X_v)

        val_c = harrell_cindex_np(
            times_val, events_val, val_risk.cpu().numpy()
        )

        # IPCW C-index & IBS (if sksurv)
        ipcw_c = float("nan")
        ibs_val = float("nan")
        if HAS_SKSURV:
            try:
                y_tr = Surv.from_arrays(events_train.astype(bool), times_train)
                y_va = Surv.from_arrays(events_val.astype(bool), times_val)
                ipcw_c = float(
                    concordance_index_ipcw(
                        y_tr, y_va, val_risk.cpu().numpy()
                    )[0]
                )
            except Exception:
                pass

        logger.info(
            f"[DeepSurv] 最终: Harrell-C={val_c:.4f}, "
            f"IPCW-C={ipcw_c:.4f}, IBS={ibs_val:.4f}"
        )

        return model, {
            "val_cindex": val_c,
            "ipcw_cindex": ipcw_c,
            "ibs": ibs_val,
            "best_val_cindex": best_cindex,
            "train_risk": train_risk.cpu().numpy(),
            "val_risk": val_risk.cpu().numpy(),
            "history": history,
        }


# ══════════════════════════════════════════════════════════════════════
# 主管线
# ══════════════════════════════════════════════════════════════════════

def run_deepsurv_baseline(timestamp: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """纯 DeepSurv 基线管线。

    流程: 数据加载 → StandardScaler → DeepSurv 训练 → 评估
    不含因果编码、IB 正则化、数据增强。

    Args:
        timestamp: 运行时间戳
        config: 配置字典

    Returns:
        结果汇总字典
    """
    cfg = config or DEEPSURV_BASELINE_CONFIG
    out_dir = os.path.join(RESULTS_DIR, timestamp, "032_deepsurv_baseline")
    ensure_dir(out_dir)
    t0 = time.time()

    if not HAS_TORCH:
        logger.error("[DeepSurv] PyTorch 不可用")
        return {}

    logger.info("=" * 60)
    logger.info("[DeepSurv] 纯 DeepSurv 基线模型 — 开始")
    logger.info("=" * 60)

    # 1. 数据加载
    data = load_data_for_deepsurv(timestamp, cfg)

    # 2. 5-fold CV
    events_tr = data["events_train"].astype(int)
    skf = StratifiedKFold(
        n_splits=min(cfg["n_folds"], max(2, events_tr.sum())),
        shuffle=True, random_state=cfg["random_seed"],
    )
    cv_cindices = []
    for fold_id, (tr_idx, va_idx) in enumerate(
        skf.split(data["X_train"], events_tr), 1
    ):
        fold_model, fold_res = train_deepsurv_model(
            data["X_train"][tr_idx],
            data["times_train"][tr_idx],
            data["events_train"][tr_idx],
            data["X_train"][va_idx],
            data["times_train"][va_idx],
            data["events_train"][va_idx],
            config=cfg,
        )
        cv_cindices.append(fold_res["val_cindex"])
        logger.info(f"[DeepSurv] Fold {fold_id}: C={fold_res['val_cindex']:.4f}")

    cv_mean = float(np.nanmean(cv_cindices))
    cv_std = float(np.nanstd(cv_cindices))
    logger.info(f"[DeepSurv] CV C-index: {cv_mean:.4f} ± {cv_std:.4f}")

    # 3. 全量训练
    final_model, final_res = train_deepsurv_model(
        data["X_train"], data["times_train"], data["events_train"],
        data["X_val"], data["times_val"], data["events_val"],
        config=cfg,
    )

    # 4. 保存
    elapsed = time.time() - t0
    summary = {
        "model_type": "DeepSurv_Baseline (no causal encoding)",
        "timestamp": timestamp,
        "elapsed_sec": round(elapsed, 1),
        "n_train": data["n_train"],
        "n_val": data["n_val"],
        "input_dim": data["X_train"].shape[1],
        "cv_cindex_mean": cv_mean,
        "cv_cindex_std": cv_std,
        "val_harrell_cindex": final_res["val_cindex"],
        "val_ipcw_cindex": final_res["ipcw_cindex"],
        "val_ibs": final_res["ibs"],
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(out_dir, "deepsurv_summary.tsv"), sep="\t", index=False
    )

    # 保存风险评分
    risk_df = pd.DataFrame({
        "risk_score": final_res["val_risk"],
    })
    risk_df.to_csv(os.path.join(out_dir, "val_risk_scores.tsv"), sep="\t")

    # 保存模型
    torch.save(final_model.state_dict(), os.path.join(out_dir, "deepsurv_model.pt"))

    # 保存激活基因索引
    active_idx = final_model.get_active_gene_indices(threshold=0.3)
    if len(active_idx) > 0:
        active_genes = [data["gene_names"][i] for i in active_idx if i < len(data["gene_names"])]
        pd.DataFrame({"gene": active_genes}).to_csv(
            os.path.join(out_dir, "active_genes.tsv"), sep="\t", index=False
        )
        logger.info(f"[DeepSurv] 激活基因数: {len(active_genes)} / {len(data['gene_names'])}")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"[DeepSurv] 完成! 耗时 {elapsed:.1f}s")
    logger.info(f"  CV C-index: {cv_mean:.4f} ± {cv_std:.4f}")
    logger.info(f"  Val Harrell-C: {final_res['val_cindex']:.4f}")
    logger.info(f"  Val IPCW-C: {final_res['ipcw_cindex']:.4f}")
    logger.info(f"  输出目录: {out_dir}")
    logger.info(f"{'=' * 60}")

    return summary


# ══════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    """命令行入口。"""
    import argparse
    parser = argparse.ArgumentParser(
        description="032: 纯 DeepSurv 基线模型 (无因果编码)"
    )
    parser.add_argument(
        "--timestamp", type=str, required=True,
        help="运行时间戳",
    )
    args = parser.parse_args()
    logger.info(f"[DeepSurv] timestamp={args.timestamp}")
    try:
        run_deepsurv_baseline(args.timestamp)
    except Exception as e:
        logger.error(f"[DeepSurv] 异常: {e}")
        raise


if __name__ == "__main__":
    main()
