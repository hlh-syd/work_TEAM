"""031_Casual_model.py — Causal-DeepSurv-IB (CDSIB) 因果深度生存预测模型

融合因果表征学习 (CausalEGM)、信息瓶颈正则化 (IB) 与深度 Cox 生存模型 (DeepSurv)，
面向高维小样本 (p>>n) 结直肠癌预后预测。

架构: 输入层 → 稀疏门控 → 因果编码器 → IB门控 → 生存预测网络 → 临床输出

依赖: causal_egm_adapter.py, shared_utils.py, config_causal.py
参考文献: 03因果模型设计.txt (完整架构设计与数学推导)

用法:
    python 031_Casual_model.py --timestamp <run_timestamp>
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import eye as sparse_eye
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── 项目内依赖 ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from shared_utils import (
    setup_logger, RANDOM_SEED, EPS,
    DATA_DIR, RESULTS_DIR, ESSENTIAL_DIR,
    ensure_dir, safe_read_tsv, normalize_patient_id,
    detect_event_time_columns, coerce_event,
    kaplan_meier_survival_at, compute_ipcw_weights,
)

from causal_egm_adapter import (
    CausalEGMAdapter, _build_treatment_proxy, _check_causalegm_available,
)

# ── 可选依赖: PyTorch ──
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── 可选依赖: scikit-survival ──
try:
    from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
    from sksurv.util import Surv
    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False

# ── 可选依赖: lifelines ──
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False

# ── 可选依赖: SHAP ──
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

logger = setup_logger("031_cdsib")

# ══════════════════════════════════════════════════════════════════════
# 配置 (所有超参数集中于此，便于调优)
# ══════════════════════════════════════════════════════════════════════
CDSIB_CONFIG: Dict[str, Any] = {
    # 数据预处理
    "variance_topk": 5000,          # 方差过滤保留 top-K 高变异基因
    "sparse_gate_k": 500,           # 稀疏门控 top-k
    # 因果编码器
    "causal_latent_dim": 12,        # CausalEGM 潜在空间维度
    "causal_mode": "rmst_ate",      # CausalEGM 工作模式
    "causal_rmst_tau": 36.0,        # RMST 截断时间 (月)
    # 生存网络
    "hidden_dims": [64, 32],        # SurvivalHead 隐层维度
    "dropout_input": 0.2,           # 输入层 Dropout
    "dropout_hidden": 0.3,          # 隐层 Dropout
    # 正则化权重
    "ib_beta": 1e-3,                # 信息瓶颈 β (KL 正则系数)
    "rank_lambda": 0.3,             # 对比排序损失权重
    "graph_lambda": 0.01,           # 图 Laplacian 正则系数 (PPI 网络可用时启用)
    "elastic_lambda": 1e-4,         # 弹性网络正则系数
    "sparse_lambda": 1e-5,          # 稀疏门控正则系数
    # 训练
    "lr": 5e-4,                     # 学习率
    "weight_decay": 1e-4,           # Adam weight decay
    "epochs": 200,                  # 最大训练轮数
    "batch_size": 64,               # Mini-batch 大小
    "patience": 30,                 # Early stopping patience
    "n_folds": 5,                   # 交叉验证折数
    # 数据增强
    "diffusion_steps": 100,         # 扩散模型步数
    "augment_ratio": 1.0,           # 合成:真实 比例
    # 评估
    "eval_tau_months": 36,          # 固定时间点评估 (月)
}


# ══════════════════════════════════════════════════════════════════════
# PyTorch 模块 (仅在 HAS_TORCH=True 时可用)
# ══════════════════════════════════════════════════════════════════════

if HAS_TORCH:

    class SparseGate(nn.Module):
        """Gumbel-Softmax 可学习稀疏门控。

        训练时通过 Concrete 分布松弛采样实现可微 top-k；
        推理时确定性选取 top-k 特征。
        """

        def __init__(self, input_dim: int, k: int = 500, tau: float = 0.5):
            super().__init__()
            self.input_dim = input_dim
            self.k = min(k, input_dim)
            self.tau = tau
            self.gate_logits = nn.Parameter(torch.zeros(input_dim))

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """返回 (x_masked, sparse_loss)。"""
            if self.training:
                # Gumbel-Softmax 松弛: 连续近似 top-k
                gumbel = -torch.log(-torch.log(
                    torch.rand(self.input_dim, device=x.device) + EPS
                ) + EPS)
                scores = (self.gate_logits + gumbel) / self.tau
                m = torch.sigmoid(scores)
                # 稀疏正则: 鼓励门控值趋零
                sparse_loss = self.gate_logits.sigmoid().mean()
                return x * m.unsqueeze(0), sparse_loss
            else:
                # 确定性 top-k
                _, topk_idx = torch.topk(self.gate_logits, self.k)
                mask = torch.zeros(self.input_dim, device=x.device)
                mask[topk_idx] = 1.0
                return x * mask.unsqueeze(0), torch.tensor(0.0, device=x.device)

    class IBGate(nn.Module):
        """信息瓶颈门控层。

        可学习门控 α = σ(W·Z + b)，逐元素乘法过滤因果表征。
        输出 KL 散度作为正则化项。
        """

        def __init__(self, z_dim: int, beta_vib: float = 1e-3):
            super().__init__()
            self.gate_fc = nn.Linear(z_dim, z_dim)
            self.mu_fc = nn.Linear(z_dim, z_dim)
            self.logvar_fc = nn.Linear(z_dim, z_dim)
            self.beta = beta_vib

        def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """返回 (z_filtered, kl_loss)。"""
            alpha = torch.sigmoid(self.gate_fc(z))
            z_filtered = z * alpha
            # VIB: KL(q(Z|X) || N(0,I))
            mu = self.mu_fc(z)
            logvar = self.logvar_fc(z).clamp(-10, 10)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return z_filtered, self.beta * kl

    class SurvivalHead(nn.Module):
        """3层 MLP 生存预测网络 (DeepSurv 变体)。

        输出对数风险 g_θ(Z)，风险函数 h(t|x) = h₀(t) · exp(g_θ(Z))。
        """

        def __init__(self, z_dim: int, hidden: List[int] = None, dropout: float = 0.3):
            super().__init__()
            hidden = hidden or [64, 32]
            layers = []
            prev = z_dim
            for h in hidden:
                layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h),
                               nn.ReLU(), nn.Dropout(dropout)])
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.net(z).squeeze(-1)


    class CDSIBModel(nn.Module):
        """Causal-DeepSurv-IB 一体化模型。

        架构: SparseGate → (外部 CausalEGM 编码) → IBGate → SurvivalHead
        注: 因果编码在外部完成，本模型接受 Z 空间输入。
        """

        def __init__(self, z_dim: int, config: Dict[str, Any]):
            super().__init__()
            self.ib_gate = IBGate(z_dim, beta_vib=config["ib_beta"])
            self.surv_head = SurvivalHead(
                z_dim, hidden=config["hidden_dims"], dropout=config["dropout_hidden"]
            )

        def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """返回 (log_risk, ib_kl_loss)。"""
            z_filt, kl_loss = self.ib_gate(z)
            log_risk = self.surv_head(z_filt)
            return log_risk, kl_loss

        def predict_risk(self, z: torch.Tensor) -> torch.Tensor:
            """推理模式: 返回对数风险评分。"""
            self.eval()
            with torch.no_grad():
                log_risk, _ = self.forward(z)
            return log_risk


# ══════════════════════════════════════════════════════════════════════
# 损失函数
# ══════════════════════════════════════════════════════════════════════

def _compute_ipcw_weights(times: np.ndarray, events: np.ndarray) -> np.ndarray:
    """计算 IPCW 权重: w_i = δ_i / Ĝ(T_i)。

    Ĝ 为 Kaplan-Meier 估计的删失生存函数（将事件/删失角色互换）。
    极端权重截断在 95 百分位以增强稳定性。
    """
    n = len(times)
    weights = np.zeros(n)
    # 将删失视为"事件"来估计 Ĝ
    censor_times = times.copy()
    censor_events = 1 - events  # 删失指示器
    order = np.argsort(censor_times)
    at_risk = n
    g_hat = 1.0
    g_values = np.ones(n)
    for idx in order:
        if censor_events[idx] == 1:
            g_hat *= (at_risk - 1) / max(at_risk, 1)
        g_values[idx] = max(g_hat, 0.01)  # 防止除零
        at_risk -= 1
    for i in range(n):
        if events[i] == 1:
            weights[i] = 1.0 / g_values[i]
    # 截断极端权重
    clip_val = np.percentile(weights[weights > 0], 95) if (weights > 0).any() else 10.0
    weights = np.clip(weights, 0, clip_val)
    return weights


def _load_ppi_laplacian(gene_names: List[str]) -> Optional[np.ndarray]:
    """加载 STRING PPI 网络并构建图 Laplacian 矩阵。

    当 PPI 文件不存在时返回 None（跳过图正则化）。
    返回稀疏 Laplacian L = D - A，大小 = len(gene_names) × len(gene_names)。
    """
    ppi_path = os.path.join(DATA_DIR, "external", "string_ppi_coad.tsv")
    if not os.path.exists(ppi_path):
        logger.info("[CDSIB] PPI 网络文件不存在，跳过图 Laplacian 正则化")
        return None
    try:
        ppi = pd.read_csv(ppi_path, sep="\t")
        gene_set = set(g.upper() for g in gene_names)
        gene_idx = {g.upper(): i for i, g in enumerate(gene_names)}
        n = len(gene_names)
        A = np.zeros((n, n), dtype=np.float32)
        for _, row in ppi.iterrows():
            g1, g2 = str(row.iloc[0]).upper(), str(row.iloc[1]).upper()
            if g1 in gene_idx and g2 in gene_idx:
                i, j = gene_idx[g1], gene_idx[g2]
                w = float(row.iloc[2]) if len(row) > 2 else 1.0
                A[i, j] = w
                A[j, i] = w
        D = np.diag(A.sum(axis=1))
        L = D - A
        logger.info(f"[CDSIB] PPI 图 Laplacian 已构建: {n} 节点, {int(A.sum()/2)} 边")
        return L
    except Exception as e:
        logger.warning(f"[CDSIB] PPI 加载失败 ({e})，跳过图正则化")
        return None


if HAS_TORCH:
    def compute_cdsib_loss(
        log_risk: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor,
        ib_kl_loss: torch.Tensor,
        model: nn.Module,
        ipcw_weights: Optional[torch.Tensor] = None,
        config: Optional[Dict] = None,
        graph_laplacian: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """CDSIB 复合损失函数。

        L_total = L_IPCW + λ_rank·L_rank + L_IB + λ_graph·L_graph + λ_elastic·L_elastic
        """
        cfg = config or CDSIB_CONFIG
        n = len(times)
        device = log_risk.device

        # ── 1. IPCW 加权偏似然损失 ──
        sort_idx = torch.argsort(-times)
        h_sorted = log_risk[sort_idx]
        ev_sorted = events[sort_idx]
        w_sorted = ipcw_weights[sort_idx] if ipcw_weights is not None else torch.ones(n, device=device)

        log_cumsum_exp = torch.logcumsumexp(h_sorted, dim=0)
        partial_ll = (h_sorted - log_cumsum_exp) * ev_sorted * w_sorted
        event_count = ev_sorted.sum().clamp(min=1)
        l_ipcw = -partial_ll.sum() / event_count

        # ── 2. 对比排序损失 ──
        l_rank = torch.tensor(0.0, device=device)
        event_mask = (events == 1)
        if event_mask.sum() > 1 and (~event_mask).sum() > 0:
            ev_idx = torch.where(event_mask)[0]
            no_ev_idx = torch.where(~event_mask)[0]
            n_pairs = min(500, len(ev_idx) * len(no_ev_idx))
            for _ in range(n_pairs):
                i = ev_idx[torch.randint(len(ev_idx), (1,))]
                j_candidates = torch.cat([
                    ev_idx[log_risk[ev_idx] < log_risk[i]],
                    no_ev_idx[times[no_ev_idx] > times[i]],
                ])
                if len(j_candidates) == 0:
                    continue
                j = j_candidates[torch.randint(len(j_candidates), (1,))]
                l_rank = l_rank - torch.log(torch.sigmoid(log_risk[i] - log_risk[j]) + EPS)
            l_rank = l_rank / max(n_pairs, 1)

        # ── 3. IB 正则化 ──
        l_ib = ib_kl_loss

        # ── 4. 图 Laplacian 正则化 (可选) ──
        l_graph = torch.tensor(0.0, device=device)
        if graph_laplacian is not None:
            # 获取输入层权重 (SurvivalHead 第一层)
            first_layer_weight = None
            for module in model.modules():
                if isinstance(module, nn.Linear) and module.in_features > 10:
                    first_layer_weight = module.weight
                    break
            if first_layer_weight is not None:
                w = first_layer_weight  # (out_dim, in_dim)
                # L_graph = Σ_{(i,j)∈E} A_{ij} * ||w_i - w_j||^2
                # = w^T L w (对每列计算)
                l_graph = torch.trace(w @ graph_laplacian @ w.t())
                l_graph = l_graph * cfg.get("graph_lambda", 0.01)

        # ── 5. 弹性网络正则化 ──
        l_elastic = torch.tensor(0.0, device=device)
        for p in model.parameters():
            if p.requires_grad:
                l_elastic = l_elastic + 0.5 * p.abs().sum() + 0.25 * p.pow(2).sum()
        l_elastic = l_elastic * cfg["elastic_lambda"]

        # ── 总损失 ──
        loss = l_ipcw + cfg["rank_lambda"] * l_rank + l_ib + l_graph + l_elastic
        return loss


# ══════════════════════════════════════════════════════════════════════
# 数据加载与预处理
# ══════════════════════════════════════════════════════════════════════

def load_and_preprocess_cdsib(
    timestamp: str,
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """加载 TCGA-COAD/READ 数据并执行预处理。

    Returns:
        dict 包含: X_train, X_val, endpoint_train, endpoint_val,
                   clinical, gene_names, train_ids, val_ids
    """
    cfg = config or CDSIB_CONFIG
    results_base = os.path.join(RESULTS_DIR, timestamp)
    preproc_dir = ESSENTIAL_DIR

    # 读取临床数据
    clinical_path = os.path.join(preproc_dir, "tcga_os_clinical_endpoint_qc.tsv")
    clinical = safe_read_tsv(clinical_path)

    # 读取 train/val split
    split_path = os.path.join(preproc_dir, "tcga_train_internal_validation_split.tsv")
    if not os.path.exists(split_path):
        split_path = os.path.join(DATA_DIR, "survival_models", "tcga_train_internal_validation_split.tsv")
    split_df = safe_read_tsv(split_path)

    # 读取基因表达矩阵
    expr_path = os.path.join(preproc_dir, "gene_expression_curated.tsv")
    if not os.path.exists(expr_path):
        raise FileNotFoundError(f"基因表达矩阵不存在: {expr_path}")
    expr_df = safe_read_tsv(expr_path, index_col=0)
    expr_df = expr_df.apply(pd.to_numeric, errors="coerce")
    logger.info(f"[CDSIB] 基因表达矩阵: {expr_df.shape[0]} 样本 × {expr_df.shape[1]} 基因")

    # 构建 train/val ID 集合
    clinical_ids = set(clinical["PATIENT_ID"].astype(str))
    train_ids = set(split_df.loc[split_df["split"] == "train", "PATIENT_ID"].astype(str)) & clinical_ids
    val_ids = set(split_df.loc[split_df["split"] != "train", "PATIENT_ID"].astype(str)) & clinical_ids
    logger.info(f"[CDSIB] 样本划分: train={len(train_ids)}, val={len(val_ids)}")

    # 构建 survival endpoint
    time_col, event_col = detect_event_time_columns(clinical)
    ep = clinical.copy()
    ep["PATIENT_ID"] = ep["PATIENT_ID"].astype(str).map(normalize_patient_id)
    ep["time_months"] = pd.to_numeric(ep[time_col], errors="coerce")
    ep["event"] = coerce_event(ep[event_col]).astype(int)
    ep = ep[ep["PATIENT_ID"].ne("") & ep["time_months"].notna() & (ep["time_months"] >= 0)]
    ep.index = ep["PATIENT_ID"].values

    # 方差过滤: 保留 top-K 高变异基因 (仅在训练集上计算)
    train_idx = [pid for pid in map(str, train_ids) if pid in expr_df.index]
    train_expr = expr_df.loc[train_idx].replace([np.inf, -np.inf], np.nan)
    variances = train_expr.var(axis=0, skipna=True).dropna()
    variances = variances[variances > 0].sort_values(ascending=False)
    topk_genes = list(variances.head(cfg["variance_topk"]).index)
    logger.info(f"[CDSIB] 方差过滤: {expr_df.shape[1]} → {len(topk_genes)} 基因")

    # 标准化: train-only RINT (Rank Inverse Normal) + StandardScaler
    X_all = expr_df[topk_genes].copy()
    X_all.index = X_all.index.astype(str)
    # RINT: 对每列做分位数正态化 (train-only fit，防止验证泄漏)
    for col in X_all.columns:
        train_vals = X_all.loc[train_idx, col].dropna().to_numpy().reshape(-1, 1)
        if len(train_vals) < 3:
            continue
        n_q = min(1000, len(train_vals))
        qt = QuantileTransformer(n_quantiles=n_q, output_distribution="normal",
                                 random_state=RANDOM_SEED)
        qt.fit(train_vals)
        all_vals = X_all[col].to_numpy().reshape(-1, 1)
        valid = ~np.isnan(all_vals.ravel())
        if valid.any():
            transformed = np.full(all_vals.shape[0], np.nan)
            transformed[valid] = qt.transform(all_vals[valid]).ravel()
            X_all[col] = transformed
    # 中位数填充 + StandardScaler
    train_med = X_all.loc[train_idx].median()
    X_all = X_all.fillna(train_med).fillna(0.0)
    scaler = StandardScaler()
    scaler.fit(X_all.loc[train_idx])
    X_scaled = pd.DataFrame(
        scaler.transform(X_all),
        index=X_all.index, columns=topk_genes,
    )

    # 分割 train/val
    X_train = X_scaled.loc[X_scaled.index.isin(train_ids)]
    X_val = X_scaled.loc[X_scaled.index.isin(val_ids)]
    ep_train = ep.loc[ep.index.isin(train_ids)]
    ep_val = ep.loc[ep.index.isin(val_ids)]

    # 对齐
    common_train = X_train.index.intersection(ep_train.index)
    common_val = X_val.index.intersection(ep_val.index)
    X_train, ep_train = X_train.loc[common_train], ep_train.loc[common_train]
    X_val, ep_val = X_val.loc[common_val], ep_val.loc[common_val]
    logger.info(f"[CDSIB] 对齐后: train={len(X_train)}, val={len(X_val)}")

    return {
        "X_train": X_train, "X_val": X_val,
        "endpoint_train": ep_train, "endpoint_val": ep_val,
        "clinical": clinical, "gene_names": topk_genes,
        "train_ids": train_ids, "val_ids": val_ids,
    }


# ══════════════════════════════════════════════════════════════════════
# 因果编码器集成
# ══════════════════════════════════════════════════════════════════════

def fit_causal_encoder(
    X_train: pd.DataFrame,
    clinical_df: pd.DataFrame,
    endpoint_train: pd.DataFrame,
    config: Optional[Dict] = None,
) -> Tuple[Any, pd.DataFrame]:
    """训练因果编码器并提取低维表征 Z。

    优先使用 CausalEGM，不可用时回退到 PCA。

    Returns:
        (encoder_or_pca, Z_train_df)
    """
    cfg = config or CDSIB_CONFIG
    z_dim = cfg["causal_latent_dim"]

    if _check_causalegm_available():
        logger.info("[CDSIB] 使用 CausalEGM 进行因果降维")
        try:
            # 构建 treatment proxy (AJCC ≥ III)
            treatment = _build_treatment_proxy(clinical_df, X_train.index)
            treatment = treatment.reindex(X_train.index).fillna(0).astype(int)
            # outcome: RMST
            outcome = endpoint_train.loc[X_train.index, "time_months"].astype(float)
            event = endpoint_train.loc[X_train.index, "event"].astype(int)

            adapter = CausalEGMAdapter(
                latent_dim=z_dim, mode=cfg["causal_mode"],
                rmst_tau=cfg["causal_rmst_tau"],
            )
            adapter.fit(X_train, treatment, outcome, event)
            Z = adapter.transform(X_train)
            logger.info(f"[CDSIB] CausalEGM 编码完成: {X_train.shape} → {Z.shape}")
            return adapter, Z
        except Exception as e:
            logger.warning(f"[CDSIB] CausalEGM 失败 ({e})，回退到 PCA")

    # 回退: PCA 降维
    logger.info(f"[CDSIB] 使用 PCA(n_components={z_dim}) 作为降维回退方案")
    pca = PCA(n_components=z_dim, random_state=RANDOM_SEED)
    Z_np = pca.fit_transform(X_train.to_numpy(float))
    cols = [f"Z_{i}" for i in range(z_dim)]
    Z = pd.DataFrame(Z_np, index=X_train.index, columns=cols)
    logger.info(f"[CDSIB] PCA 解释方差比: {pca.explained_variance_ratio_.sum():.3f}")
    return pca, Z


def transform_causal(encoder: Any, X: pd.DataFrame) -> pd.DataFrame:
    """使用已训练的编码器转换新数据。"""
    if isinstance(encoder, CausalEGMAdapter):
        return encoder.transform(X)
    elif isinstance(encoder, PCA):
        Z_np = encoder.transform(X.to_numpy(float))
        cols = [f"Z_{i}" for i in range(Z_np.shape[1])]
        return pd.DataFrame(Z_np, index=X.index, columns=cols)
    raise TypeError(f"不支持的编码器类型: {type(encoder)}")


# ══════════════════════════════════════════════════════════════════════
# 条件扩散数据增强
# ══════════════════════════════════════════════════════════════════════

if HAS_TORCH:

    class _SinusoidalEmbedding(nn.Module):
        """正弦时间步嵌入。"""
        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim
        def forward(self, t: torch.Tensor) -> torch.Tensor:
            half = self.dim // 2
            freq = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(half, device=t.device) / half)
            emb = torch.cat([torch.sin(t.unsqueeze(-1) * freq), torch.cos(t.unsqueeze(-1) * freq)], dim=-1)
            return emb

    class _DenoiseNet(nn.Module):
        """扩散去噪网络: MLP + 时间嵌入。"""
        def __init__(self, z_dim: int, cond_dim: int, t_emb_dim: int = 32):
            super().__init__()
            self.t_emb = _SinusoidalEmbedding(t_emb_dim)
            in_dim = z_dim + cond_dim + t_emb_dim
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, z_dim),
            )
        def forward(self, z_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            t_embed = self.t_emb(t.float())
            return self.net(torch.cat([z_t, cond, t_embed], dim=-1))


class ConditionalDiffusionAugmenter:
    """在因果表征 Z 空间的条件扩散模型 (简化 TabDDPM)。

    在低维 Z 空间 (12 维) 进行扩散，避免模式崩溃。
    训练失败时自动回退到 SMOTE 噪声增强。
    """

    def __init__(self, z_dim: int = 12, cond_dim: int = 3,
                 T: int = 100, config: Optional[Dict] = None):
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.T = T
        self.cfg = config or CDSIB_CONFIG
        self._fitted = False
        self._device = torch.device("cpu")
        if HAS_TORCH:
            self.net = _DenoiseNet(z_dim, cond_dim).to(self._device)
            # 线性 beta schedule
            self.betas = torch.linspace(1e-4, 0.02, T, device=self._device)
            self.alphas = 1 - self.betas
            self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def fit(self, Z_real: np.ndarray, conditions: np.ndarray) -> "ConditionalDiffusionAugmenter":
        """训练扩散模型。"""
        if not HAS_TORCH or Z_real.shape[0] < 10:
            logger.warning("[CDSIB] 扩散模型: 数据不足或 torch 不可用")
            return self
        Z_t = torch.tensor(Z_real, dtype=torch.float32, device=self._device)
        C_t = torch.tensor(conditions, dtype=torch.float32, device=self._device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        n = Z_t.shape[0]
        self.net.train()
        for epoch in range(300):
            idx = torch.randint(0, n, (min(self.cfg["batch_size"], n),))
            z0 = Z_t[idx]
            c = C_t[idx]
            t = torch.randint(0, self.T, (len(idx),), device=self._device)
            # 前向扩散
            sqrt_alpha = torch.sqrt(self.alpha_cumprod[t]).unsqueeze(-1)
            sqrt_one_minus = torch.sqrt(1 - self.alpha_cumprod[t]).unsqueeze(-1)
            noise = torch.randn_like(z0)
            z_t_noisy = sqrt_alpha * z0 + sqrt_one_minus * noise
            # 去噪
            pred_noise = self.net(z_t_noisy, c, t)
            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self._fitted = True
        logger.info(f"[CDSIB] 扩散模型训练完成 (T={self.T}, n={n})")
        return self

    def sample(self, n_samples: int, conditions: np.ndarray) -> np.ndarray:
        """从扩散模型采样合成 Z。"""
        if not self._fitted or not HAS_TORCH:
            return np.random.randn(n_samples, self.z_dim).astype(np.float32) * 0.1
        self.net.eval()
        C_t = torch.tensor(conditions[:n_samples], dtype=torch.float32, device=self._device)
        if len(C_t) < n_samples:
            C_t = torch.cat([C_t, C_t[torch.randint(len(C_t), (n_samples - len(C_t),))]])
        z = torch.randn(n_samples, self.z_dim, device=self._device)
        with torch.no_grad():
            for t_step in reversed(range(self.T)):
                t = torch.full((n_samples,), t_step, device=self._device)
                pred_noise = self.net(z, C_t, t)
                alpha_t = self.alphas[t_step]
                alpha_cum = self.alpha_cumprod[t_step]
                z = (z - (1 - alpha_t) / torch.sqrt(1 - alpha_cum) * pred_noise) / torch.sqrt(alpha_t)
                if t_step > 0:
                    z += torch.sqrt(self.betas[t_step]) * torch.randn_like(z)
        return z.cpu().numpy()


def augment_z_space(
    Z_train: pd.DataFrame,
    endpoint_train: pd.DataFrame,
    clinical_df: pd.DataFrame,
    config: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """在 Z 空间进行条件扩散增强。

    Returns:
        (Z_augmented, endpoint_augmented)
    """
    cfg = config or CDSIB_CONFIG
    n_real = len(Z_train)
    n_syn = int(n_real * cfg["augment_ratio"])
    if n_syn == 0:
        return Z_train, endpoint_train

    # 构建条件变量: [stage_binary, sex, age_normalized]
    cond_df = _build_diffusion_conditions(clinical_df, Z_train.index)
    cond_np = cond_df.to_numpy(dtype=np.float32)

    # 尝试扩散模型
    try:
        augmenter = ConditionalDiffusionAugmenter(
            z_dim=Z_train.shape[1], cond_dim=cond_np.shape[1],
            T=cfg["diffusion_steps"], config=cfg,
        )
        augmenter.fit(Z_train.to_numpy(float), cond_np)
        Z_syn_np = augmenter.sample(n_syn, cond_np)
        Z_syn = pd.DataFrame(
            Z_syn_np, index=[f"DIFF_{i}" for i in range(n_syn)],
            columns=Z_train.columns,
        )
        logger.info(f"[CDSIB] 扩散增强: +{n_syn} 合成样本")
    except Exception as e:
        logger.warning(f"[CDSIB] 扩散增强失败 ({e})，回退 SMOTE")
        rng = np.random.default_rng(RANDOM_SEED)
        Z_syn_list = []
        for i in range(n_syn):
            src = rng.choice(n_real)
            z_row = Z_train.iloc[src].copy()
            noise = rng.normal(0, 0.05 * z_row.std(), size=len(z_row))
            Z_syn_list.append(z_row + noise)
        Z_syn = pd.DataFrame(Z_syn_list, index=[f"SMOTE_{i}" for i in range(n_syn)],
                             columns=Z_train.columns)

    # 构建合成 endpoint (复制最近邻真实样本的生存标签)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(Z_train.to_numpy(float))
    _, nn_idx = nn.kneighbors(Z_syn.to_numpy(float))
    nn_idx = nn_idx.ravel()
    ep_syn = endpoint_train.iloc[nn_idx].copy()
    ep_syn.index = Z_syn.index
    if "PATIENT_ID" in ep_syn.columns:
        ep_syn["PATIENT_ID"] = Z_syn.index.tolist()

    Z_aug = pd.concat([Z_train, Z_syn], axis=0)
    ep_aug = pd.concat([endpoint_train, ep_syn], axis=0)
    logger.info(f"[CDSIB] 增强后: {len(Z_train)} → {len(Z_aug)} 样本")
    return Z_aug, ep_aug


def _build_diffusion_conditions(clinical_df: pd.DataFrame, sample_index: pd.Index) -> pd.DataFrame:
    """构建扩散模型条件变量: [stage_binary, sex, age_normalized]。"""
    cond = pd.DataFrame(index=sample_index)
    # Stage
    try:
        treatment = _build_treatment_proxy(clinical_df, sample_index)
        cond["stage_binary"] = treatment.reindex(sample_index).fillna(0).astype(float)
    except Exception:
        cond["stage_binary"] = 0.0
    # Sex
    sex_col = None
    for c in ["SEX", "GENDER"]:
        if c in clinical_df.columns:
            sex_col = c
            break
    if sex_col:
        cond["sex"] = clinical_df.set_index("PATIENT_ID")[sex_col].reindex(sample_index).map(
            {"Male": 1.0, "Female": 0.0}).fillna(0.5)
    else:
        cond["sex"] = 0.5
    # Age
    age_col = None
    for c in ["AGE", "AGE_AT_DIAGNOSIS"]:
        if c in clinical_df.columns:
            age_col = c
            break
    if age_col:
        ages = pd.to_numeric(
            clinical_df.set_index("PATIENT_ID")[age_col].reindex(sample_index),
            errors="coerce").fillna(60)
        cond["age_norm"] = (ages - ages.mean()) / max(ages.std(), 1)
    else:
        cond["age_norm"] = 0.0
    return cond.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# 反事实生存预测
# ══════════════════════════════════════════════════════════════════════

def estimate_counterfactual_survival(
    model: Any,
    Z: pd.DataFrame,
    endpoint: pd.DataFrame,
    treatment_col: str = "stage_binary",
    config: Optional[Dict] = None,
) -> pd.DataFrame:
    """估计反事实生存曲线和个体处理效应 (ITE)。

    基于 CausalEGM 的因果解耦: 干预 treatment 维度，
    比较 do(a=1) vs do(a=0) 的生存曲线差异。

    Returns:
        DataFrame: [patient_id, risk_factual, risk_cf_a0, risk_cf_a1, ite_tau]
    """
    cfg = config or CDSIB_CONFIG
    if not HAS_TORCH:
        logger.warning("[CDSIB] torch 不可用，跳过反事实推断")
        return pd.DataFrame()

    z_dim = Z.shape[1]
    # 严格对齐 Z 与 endpoint 的索引 (取交集)
    common_idx = Z.index.intersection(endpoint.index)
    if len(common_idx) == 0:
        logger.warning("[CDSIB] Z 与 endpoint 无公共索引，跳过反事实推断")
        return pd.DataFrame()
    if len(common_idx) < len(Z):
        logger.warning(
            f"[CDSIB] Z({len(Z)}) 与 endpoint({len(endpoint)}) "
            f"索引不完全匹配，取交集 {len(common_idx)} 个样本"
        )
    Z_aligned = Z.loc[common_idx]
    ep_aligned = endpoint.loc[common_idx]
    Z_np = Z_aligned.to_numpy(dtype=np.float32)
    times = ep_aligned["time_months"].to_numpy(dtype=np.float64)
    events = ep_aligned["event"].to_numpy(dtype=np.int64)
    tau = cfg["eval_tau_months"]

    # Baseline KM 生存函数 S₀(t)
    eval_times = np.linspace(0, min(times.max(), tau * 1.5), 50)
    s0 = _baseline_km_survival(times, events, eval_times)

    # Factual 风险
    Z_tensor = torch.tensor(Z_np, device=torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        risk_factual, _ = model(Z_tensor)
    risk_factual = risk_factual.numpy()

    # 反事实: 干预最后几个维度 (treatment-related dims)
    # CausalEGM z_dims = [z_shared, z_treat, z_outcome, z_noise]
    # z_treat 维度约在 z_shared 之后的位置
    treat_dims = list(range(max(0, z_dim // 3), max(1, 2 * z_dim // 3)))
    # 使用 np.ix_ 避免 boolean + integer list 混合索引的广播问题
    mask_a0 = (events == 0)
    mask_a1 = (events == 1)
    treat_mean_a0 = (
        Z_np[np.ix_(mask_a0, treat_dims)].mean(axis=0)
        if mask_a0.any() else np.zeros(len(treat_dims), dtype=np.float32)
    )
    treat_mean_a1 = (
        Z_np[np.ix_(mask_a1, treat_dims)].mean(axis=0)
        if mask_a1.any() else np.ones(len(treat_dims), dtype=np.float32)
    )

    results = []
    for i in range(len(Z_np)):
        z_i = Z_np[i].copy()
        # Counterfactual: do(a=0)
        z_cf0 = z_i.copy()
        z_cf0[treat_dims] = treat_mean_a0
        # Counterfactual: do(a=1)
        z_cf1 = z_i.copy()
        z_cf1[treat_dims] = treat_mean_a1

        with torch.no_grad():
            out0 = model(torch.tensor(z_cf0.reshape(1, -1)))
            out1 = model(torch.tensor(z_cf1.reshape(1, -1)))
            r0 = out0[0].item() if isinstance(out0, tuple) else out0.item()
            r1 = out1[0].item() if isinstance(out1, tuple) else out1.item()

        # 反事实生存曲线: S(t) = S₀(t)^exp(r)
        s_cf0 = np.power(np.clip(s0, 0.001, 1.0), np.exp(r0))
        s_cf1 = np.power(np.clip(s0, 0.001, 1.0), np.exp(r1))

        # ITE(τ) = ∫₀^τ [S(t|do(1)) - S(t|do(0))] dt
        mask_tau = eval_times <= tau
        ite = float(np.trapz(s_cf1[mask_tau] - s_cf0[mask_tau], eval_times[mask_tau]))

        results.append({
            "patient_id": common_idx[i],
            "risk_factual": float(risk_factual[i]),
            "risk_cf_a0": float(r0),
            "risk_cf_a1": float(r1),
            "ite_tau": ite,
        })
    return pd.DataFrame(results)


def _baseline_km_survival(times: np.ndarray, events: np.ndarray,
                          eval_times: np.ndarray) -> np.ndarray:
    """计算 baseline Kaplan-Meier 生存函数。"""
    if HAS_LIFELINES:
        kmf = KaplanMeierFitter()
        kmf.fit(times, event_observed=events)
        return np.array([float(kmf.predict(t)) for t in eval_times])
    # 简单 Nelson-Aalen 估计
    unique_t = np.sort(np.unique(times))
    hazard = np.zeros(len(unique_t))
    for idx, t in enumerate(unique_t):
        at_risk = (times >= t).sum()
        d = ((times == t) & (events == 1)).sum()
        hazard[idx] = d / max(at_risk, 1)
    cum_hazard = np.cumsum(hazard)
    s0 = np.exp(-cum_hazard)
    # 插值到 eval_times
    return np.interp(eval_times, unique_t, s0, left=1.0, right=s0[-1])


# ══════════════════════════════════════════════════════════════════════
# Harrell C-index (纯 numpy 实现，无需 sksurv)
# ══════════════════════════════════════════════════════════════════════

def harrell_cindex(times: np.ndarray, events: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's concordance index。"""
    valid = np.isfinite(risk)
    t, e, r = times[valid], events[valid], risk[valid]
    permissible, concordant = 0, 0.0
    for i in range(len(t)):
        for j in range(len(t)):
            if t[i] < t[j] and e[i] == 1:
                permissible += 1
                if r[i] > r[j]:
                    concordant += 1
                elif r[i] == r[j]:
                    concordant += 0.5
    return concordant / permissible if permissible > 0 else float("nan")


# ══════════════════════════════════════════════════════════════════════
# 训练与评估流程
# ══════════════════════════════════════════════════════════════════════

def train_cdsib_model(
    Z_train: pd.DataFrame,
    endpoint_train: pd.DataFrame,
    Z_val: pd.DataFrame,
    endpoint_val: pd.DataFrame,
    config: Optional[Dict] = None,
    graph_laplacian_np: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """训练 CDSIB 模型 (5-fold CV + 最终模型)。

    Returns:
        dict: model, cv_metrics, val_metrics, Z_train_risk, Z_val_risk
    """
    cfg = config or CDSIB_CONFIG
    if not HAS_TORCH:
        logger.error("[CDSIB] PyTorch 不可用，无法训练深度模型")
        return {}

    z_dim = Z_train.shape[1]
    device = torch.device("cpu")
    # 图 Laplacian 转为 tensor (可选)
    graph_L = None
    if graph_laplacian_np is not None and HAS_TORCH:
        graph_L = torch.tensor(graph_laplacian_np, dtype=torch.float32, device=device)

    # ── 5-fold CV ──
    events = endpoint_train.loc[Z_train.index, "event"].to_numpy(int)
    times = endpoint_train.loc[Z_train.index, "time_months"].to_numpy(float)
    skf = StratifiedKFold(n_splits=min(cfg["n_folds"], max(2, events.sum())),
                          shuffle=True, random_state=RANDOM_SEED)
    cv_cindices = []

    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(Z_train, events), 1):
        fold_model = CDSIBModel(z_dim, cfg).to(device)
        optimizer = torch.optim.Adam(fold_model.parameters(), lr=cfg["lr"],
                                     weight_decay=cfg["weight_decay"])

        Z_tr = torch.tensor(Z_train.iloc[tr_idx].to_numpy(np.float32), device=device)
        t_tr = torch.tensor(times[tr_idx], dtype=torch.float32, device=device)
        e_tr = torch.tensor(events[tr_idx], dtype=torch.float32, device=device)
        w_tr = torch.tensor(_compute_ipcw_weights(times[tr_idx], events[tr_idx]),
                            dtype=torch.float32, device=device)
        Z_va = torch.tensor(Z_train.iloc[va_idx].to_numpy(np.float32), device=device)

        best_c, patience_cnt = 0.0, 0
        for epoch in range(cfg["epochs"]):
            fold_model.train()
            optimizer.zero_grad()
            log_risk, kl_loss = fold_model(Z_tr)
            loss = compute_cdsib_loss(log_risk, t_tr, e_tr, kl_loss, fold_model,
                                       w_tr, cfg, graph_L)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fold_model.parameters(), 5.0)
            optimizer.step()
            # Early stopping
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    val_risk, _ = fold_model(Z_va)
                c = harrell_cindex(times[va_idx], events[va_idx], val_risk.numpy())
                if c > best_c:
                    best_c = c
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                if patience_cnt >= cfg["patience"] // 10:
                    break
        cv_cindices.append(best_c)
        logger.info(f"[CDSIB] Fold {fold_id}: C-index={best_c:.4f}")

    cv_mean = float(np.mean(cv_cindices)) if cv_cindices else float("nan")
    cv_std = float(np.std(cv_cindices)) if cv_cindices else float("nan")
    logger.info(f"[CDSIB] CV C-index: {cv_mean:.4f} ± {cv_std:.4f}")

    # ── 训练最终模型 (全量训练集) ──
    model = CDSIBModel(z_dim, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])
    Z_all = torch.tensor(Z_train.to_numpy(np.float32), device=device)
    t_all = torch.tensor(times, dtype=torch.float32, device=device)
    e_all = torch.tensor(events, dtype=torch.float32, device=device)
    w_all = torch.tensor(_compute_ipcw_weights(times, events),
                         dtype=torch.float32, device=device)

    best_loss = float("inf")
    for epoch in range(cfg["epochs"]):
        model.train()
        optimizer.zero_grad()
        log_risk, kl_loss = model(Z_all)
        loss = compute_cdsib_loss(log_risk, t_all, e_all, kl_loss, model,
                                   w_all, cfg, graph_L)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()

    # 评估
    model.eval()
    with torch.no_grad():
        train_risk, _ = model(Z_all)
        Z_val_t = torch.tensor(Z_val.to_numpy(np.float32), device=device)
        val_risk, _ = model(Z_val_t)

    val_times = endpoint_val.loc[Z_val.index, "time_months"].to_numpy(float)
    val_events = endpoint_val.loc[Z_val.index, "event"].to_numpy(int)
    val_cindex = harrell_cindex(val_times, val_events, val_risk.numpy())

    # IPCW C-index (if sksurv available)
    ipcw_c = float("nan")
    ibs = float("nan")
    if HAS_SKSURV and Surv is not None:
        try:
            y_tr = Surv.from_arrays(events.astype(bool), times)
            y_va = Surv.from_arrays(val_events.astype(bool), val_times)
            ipcw_c = float(concordance_index_ipcw(y_tr, y_va, val_risk.numpy())[0])
        except Exception:
            pass
        try:
            eval_t = np.linspace(
                max(times.min(), val_times.min()) + 1,
                min(times.max(), val_times.max()) - 1, 20)
            s0 = _baseline_km_survival(times, events, eval_t)
            surv_matrix = np.array([
                np.power(np.clip(s0, 0.001, 1.0), np.exp(r))
                for r in val_risk.numpy()
            ])
            ibs = float(integrated_brier_score(y_tr, y_va, surv_matrix, eval_t))
        except Exception:
            pass

    logger.info(f"[CDSIB] 验证集: Harrell-C={val_cindex:.4f}, "
                f"IPCW-C={ipcw_c:.4f}, IBS={ibs:.4f}")

    return {
        "model": model,
        "cv_cindex_mean": cv_mean, "cv_cindex_std": cv_std,
        "val_cindex": val_cindex, "ipcw_cindex": ipcw_c, "ibs": ibs,
        "train_risk": train_risk.numpy(),
        "val_risk": val_risk.numpy(),
    }


# ══════════════════════════════════════════════════════════════════════
# 可解释性输出
# ══════════════════════════════════════════════════════════════════════

def compute_shap_importance(
    model: Any, Z: pd.DataFrame, config: Optional[Dict] = None,
) -> pd.DataFrame:
    """计算 Z 维度的 SHAP 特征重要性。"""
    if not HAS_SHAP or not HAS_TORCH:
        logger.warning("[CDSIB] SHAP 或 torch 不可用，跳过可解释性分析")
        return pd.DataFrame()
    model.eval()
    Z_np = Z.to_numpy(dtype=np.float32)

    # SHAP DeepExplainer 要求 forward() 返回单个 tensor，
    # 但 CDSIBModel.forward() 返回 (log_risk, kl_loss) tuple，
    # 需要包装一个仅返回 log_risk 的代理模型
    class _ShapWrapper(torch.nn.Module):
        def __init__(self, inner_model):
            super().__init__()
            self.inner = inner_model
        def forward(self, x):
            out = self.inner(x)
            out = out[0] if isinstance(out, tuple) else out
            # SHAP 要求输出至少 2D: (batch, n_outputs)
            if out.dim() == 1:
                out = out.unsqueeze(-1)
            return out

    wrapped = _ShapWrapper(model)
    wrapped.eval()

    bg_idx = np.random.choice(len(Z_np), min(50, len(Z_np)), replace=False)
    bg = torch.tensor(Z_np[bg_idx])
    explainer = shap.DeepExplainer(wrapped, bg)
    test = torch.tensor(Z_np[:min(100, len(Z_np))])
    # check_additivity=False: IBGate 的 sigmoid/GELU 算子
    # 与 DeepExplainer 的线性近似不完全兼容
    shap_values = explainer.shap_values(test, check_additivity=False)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    # SHAP 输出可能为 3D (batch, features, outputs)，squeeze 最后一维
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]
    importance = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "dimension": Z.columns,
        "mean_abs_shap": importance,
    }).sort_values("mean_abs_shap", ascending=False)
    logger.info(f"[CDSIB] SHAP 重要性 top-5: {df.head()['dimension'].tolist()}")
    return df


def compute_gene_perturbation_importance(
    encoder: Any, X_train: pd.DataFrame, gene_names: List[str],
    n_perturb: int = 200, config: Optional[Dict] = None,
) -> pd.DataFrame:
    """通过扰动 X 空间 top-方差基因观察 Z 变化，估计间接基因重要性。

    对每个基因: 将该列值随机打乱，观察编码后的 Z 变化幅度。
    变化越大的基因对因果表征影响越大。
    """
    if not HAS_TORCH:
        return pd.DataFrame()
    rng = np.random.default_rng(RANDOM_SEED)
    Z_base = transform_causal(encoder, X_train).to_numpy(float)
    gene_importance = {}
    # 只对 top-方差基因做扰动 (节省计算)
    candidate_genes = X_train.var(axis=0).sort_values(ascending=False).head(n_perturb).index
    for gene in candidate_genes:
        if gene not in X_train.columns:
            continue
        X_perturb = X_train.copy()
        X_perturb[gene] = rng.permutation(X_perturb[gene].to_numpy())
        Z_perturb = transform_causal(encoder, X_perturb).to_numpy(float)
        delta = np.mean(np.abs(Z_perturb - Z_base))
        gene_importance[gene] = float(delta)
    df = pd.DataFrame([
        {"gene": g, "perturbation_delta": v}
        for g, v in gene_importance.items()
    ]).sort_values("perturbation_delta", ascending=False)
    if not df.empty:
        logger.info(f"[CDSIB] 基因扰动重要性 top-10: {df.head(10)['gene'].tolist()}")
    return df


def plot_cdsib_results(
    val_risk: np.ndarray, endpoint_val: pd.DataFrame,
    cf_df: pd.DataFrame, out_dir: str, config: Optional[Dict] = None,
) -> None:
    """生成 KM 生存曲线 (高/低风险分组) 和 ITE 分布直方图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = config or CDSIB_CONFIG
    val_times = endpoint_val.loc[endpoint_val.index.isin(
        [str(i) for i in endpoint_val.get("PATIENT_ID", endpoint_val.index)]
    ), "time_months"].to_numpy(float)
    val_events = endpoint_val["event"].to_numpy(int)
    risk = val_risk[:len(val_times)]
    valid = np.isfinite(risk)
    if valid.sum() < 20:
        return

    # ── KM 曲线: 高/低风险分组 ──
    threshold = float(np.median(risk[valid]))
    high_mask = risk >= threshold
    low_mask = ~high_mask
    if HAS_LIFELINES and high_mask.sum() > 5 and low_mask.sum() > 5:
        fig, ax = plt.subplots(figsize=(6, 5))
        kmf = KaplanMeierFitter()
        for mask, label, color in [(low_mask, "Low risk", "#1f77b4"),
                                   (high_mask, "High risk", "#d62728")]:
            kmf.fit(val_times[mask], event_observed=val_events[mask],
                    label=f"{label} (n={int(mask.sum())})")
            kmf.plot_survival_function(ci_show=True, color=color, lw=2.0, ax=ax)
        p_text = ""
        try:
            lr = logrank_test(val_times[low_mask], val_times[high_mask],
                              event_observed_A=val_events[low_mask],
                              event_observed_B=val_events[high_mask])
            p_text = f"Log-rank p = {lr.p_value:.3g}"
        except Exception:
            pass
        if p_text:
            ax.text(0.04, 0.08, p_text, transform=ax.transAxes)
        tau = cfg.get("eval_tau_months", 36)
        ax.axvline(tau, color="#7f7f7f", lw=1.0, linestyle=":",
                   label=f"{tau} months")
        ax.set_xlabel("Time from diagnosis (months)")
        ax.set_ylabel("Overall survival probability")
        ax.set_title("CDSIB: K-M risk strata")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        fig.savefig(os.path.join(out_dir, "km_risk_strata.png"), dpi=300,
                    bbox_inches="tight")
        plt.close(fig)

    # ── ITE 分布直方图 ──
    if not cf_df.empty and "ite_tau" in cf_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(cf_df["ite_tau"], bins=30, color="#2ca02c", alpha=0.7, edgecolor="white")
        ax.axvline(0, color="red", lw=1.5, linestyle="--", label="ITE=0")
        ax.axvline(cf_df["ite_tau"].mean(), color="blue", lw=1.5, linestyle="-",
                   label=f"Mean={cf_df['ite_tau'].mean():.3f}")
        ax.set_xlabel(f"ITE(τ={cfg.get('eval_tau_months', 36)}m)")
        ax.set_ylabel("Count")
        ax.set_title("CDSIB: Individual Treatment Effect Distribution")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        fig.savefig(os.path.join(out_dir, "ite_distribution.png"), dpi=300,
                    bbox_inches="tight")
        plt.close(fig)
    logger.info(f"[CDSIB] 可视化已保存到 {out_dir}")


# ══════════════════════════════════════════════════════════════════════
# 主管线
# ══════════════════════════════════════════════════════════════════════

def run_cdsib_pipeline(timestamp: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """CDSIB 端到端管线。

    流程: 数据加载 → CausalEGM降维 → 扩散增强 → CDSIB训练 → 反事实推断 → 评估
    """
    cfg = config or CDSIB_CONFIG
    out_dir = os.path.join(RESULTS_DIR, timestamp, "031_cdsib")
    ensure_dir(out_dir)
    logger.info(f"[CDSIB] 输出目录: {out_dir}")
    t0 = time.time()

    # 1. 数据加载
    logger.info("=" * 60 + "\n[CDSIB] Phase 1: 数据加载与预处理\n" + "=" * 60)
    data = load_and_preprocess_cdsib(timestamp, cfg)
    X_train, X_val = data["X_train"], data["X_val"]
    ep_train, ep_val = data["endpoint_train"], data["endpoint_val"]

    # 2. 因果编码器降维
    logger.info("=" * 60 + "\n[CDSIB] Phase 2: 因果编码器降维\n" + "=" * 60)
    encoder, Z_train = fit_causal_encoder(X_train, data["clinical"], ep_train, cfg)
    Z_val = transform_causal(encoder, X_val)

    # 保存 Z 表征
    Z_train.to_csv(os.path.join(out_dir, "Z_train.tsv"), sep="\t")
    Z_val.to_csv(os.path.join(out_dir, "Z_val.tsv"), sep="\t")

    # 3. 扩散增强
    logger.info("=" * 60 + "\n[CDSIB] Phase 3: 条件扩散数据增强\n" + "=" * 60)
    Z_aug, ep_aug = augment_z_space(Z_train, ep_train, data["clinical"], cfg)

    # 3.5 加载 PPI 网络 (可选图 Laplacian 正则化)
    graph_L = _load_ppi_laplacian(data["gene_names"])

    # 4. 训练 CDSIB
    logger.info("=" * 60 + "\n[CDSIB] Phase 4: CDSIB 模型训练\n" + "=" * 60)
    result = train_cdsib_model(Z_aug, ep_aug, Z_val, ep_val, cfg,
                                graph_laplacian_np=graph_L)
    if not result:
        logger.error("[CDSIB] 训练失败")
        return {}

    # 5. 反事实推断
    logger.info("=" * 60 + "\n[CDSIB] Phase 5: 反事实生存推断\n" + "=" * 60)
    cf_df = estimate_counterfactual_survival(result["model"], Z_val, ep_val, config=cfg)
    if not cf_df.empty:
        cf_df.to_csv(os.path.join(out_dir, "counterfactual_ite.tsv"), sep="\t", index=False)
        logger.info(f"[CDSIB] ITE 均值: {cf_df['ite_tau'].mean():.4f}, "
                     f"中位数: {cf_df['ite_tau'].median():.4f}")

    # 6. SHAP 可解释性 + 基因扰动重要性
    logger.info("=" * 60 + "\n[CDSIB] Phase 6: 可解释性分析\n" + "=" * 60)
    shap_df = compute_shap_importance(result["model"], Z_val, cfg)
    if not shap_df.empty:
        shap_df.to_csv(os.path.join(out_dir, "shap_importance.tsv"), sep="\t", index=False)
    # 基因扰动重要性 (通过扰动 X 观察 Z 变化)
    gene_imp_df = compute_gene_perturbation_importance(
        encoder, X_train, data["gene_names"], n_perturb=200, config=cfg)
    if not gene_imp_df.empty:
        gene_imp_df.to_csv(os.path.join(out_dir, "gene_perturbation_importance.tsv"),
                           sep="\t", index=False)

    # 6.5 可视化: KM 曲线 + ITE 分布
    cf_for_plot = cf_df if not cf_df.empty else pd.DataFrame()
    plot_cdsib_results(result["val_risk"], ep_val, cf_for_plot, out_dir, cfg)

    # 7. 汇总结果
    elapsed = time.time() - t0
    summary = {
        "timestamp": timestamp,
        "elapsed_sec": round(elapsed, 1),
        "n_train": len(Z_train), "n_val": len(Z_val),
        "n_augmented": len(Z_aug),
        "latent_dim": Z_train.shape[1],
        "cv_cindex_mean": result["cv_cindex_mean"],
        "cv_cindex_std": result["cv_cindex_std"],
        "val_harrell_cindex": result["val_cindex"],
        "val_ipcw_cindex": result["ipcw_cindex"],
        "val_ibs": result["ibs"],
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(out_dir, "cdsib_summary.tsv"), sep="\t", index=False)

    # 保存风险评分
    risk_df = ep_val[["PATIENT_ID", "time_months", "event"]].copy()
    risk_df["cdsib_risk"] = result["val_risk"][:len(risk_df)]
    risk_df.to_csv(os.path.join(out_dir, "risk_scores.tsv"), sep="\t", index=False)

    # 保存模型
    model_path = os.path.join(out_dir, "cdsib_model.pt")
    if HAS_TORCH and result.get("model"):
        torch.save(result["model"].state_dict(), model_path)
    # 保存编码器元信息 (CausalEGM 内含 Keras 模型，无法直接 joblib 序列化)
    try:
        joblib.dump({"encoder": encoder, "config": cfg}, os.path.join(out_dir, "cdsib_meta.joblib"))
    except Exception as e_save:
        logger.warning(f"[CDSIB] encoder joblib 保存失败 ({e_save})，仅保存配置")
        joblib.dump({"config": cfg, "encoder_type": type(encoder).__name__},
                    os.path.join(out_dir, "cdsib_meta.joblib"))

    logger.info(f"\n{'=' * 60}")
    logger.info(f"[CDSIB] 管线完成! 耗时 {elapsed:.1f}s")
    logger.info(f"  CV C-index: {summary['cv_cindex_mean']:.4f} ± {summary['cv_cindex_std']:.4f}")
    logger.info(f"  验证集 Harrell-C: {summary['val_harrell_cindex']:.4f}")
    logger.info(f"  验证集 IPCW-C: {summary['val_ipcw_cindex']:.4f}")
    logger.info(f"  验证集 IBS: {summary['val_ibs']:.4f}")
    logger.info(f"  结果目录: {out_dir}")
    logger.info(f"  输出文件: {[f for f in os.listdir(out_dir) if not f.startswith('.')]}")
    logger.info(f"{'=' * 60}")
    return summary


# ══════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Causal-DeepSurv-IB (CDSIB) 因果深度生存预测模型")
    parser.add_argument("--timestamp", type=str, required=True,
                        help="运行时间戳 (用于定位输入/输出目录)")
    args = parser.parse_args()
    logger.info(f"[CDSIB] timestamp={args.timestamp}")
    logger.info(f"SCRIPT_DIR = {_SCRIPT_DIR}")
    logger.info(f"DATA_DIR   = {DATA_DIR}")
    logger.info(f"RESULTS_DIR= {RESULTS_DIR}")
    try:
        summary = run_cdsib_pipeline(args.timestamp)
        logger.info("[CDSIB] DONE")
    except Exception as e:
        logger.error(f"[CDSIB] 管线异常: {e}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
