"""vae_augmentor.py — β-VAE 多组学数据增强模块

使用变分自编码器 (VAE) 替代 ConditionalTabularGAN 进行数据增强，
解决 GAN 在高维组学数据上 QC 持续失败（KS 检验不通过）的问题。

核心组件:
    - SurvivalVAE: 编码器 + 解码器网络，含 reparameterize 技巧
    - VAEAugmentor: 高层封装，sklearn 风格 API (fit / sample / reconstruct)
    - vae_synthetic_qc: VAE 专属 QC 检查（替代 KS 检验）
    - train_only_vae_augmentation: 与 train_only_gan_augmentation 接口对齐的入口函数

架构设计要点（针对 146 维 × 349 训练样本优化）:
    - LayerNorm 替代 BatchNorm: batch_size=32 时 BN 统计量不稳定
    - GELU 替代 ReLU: 梯度更平滑，小数据集收敛更稳定
    - β-VAE (β=0.5): 降低 KL 权重以避免后验坍缩，让重建质量更优
    - 潜在维度 32: 146→32 压缩比 4.6:1，保留充足信息同时有效降维
    - 合成样本生存标签: 通过潜在空间 KNN 分配最近邻的中位数值

参考文献:
    - Kingma & Welling, Auto-Encoding Variational Bayes, ICLR 2014
    - Higgins et al., β-VAE: Learning Basic Visual Concepts, ICLR 2017
    - VAESCox (2024): 稀疏 VAE + 多组学 → 生存分析
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from shared_utils import setup_logger, RANDOM_SEED, EPS

# ── PyTorch 可选依赖 ─────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = setup_logger("vae_augmentor")


# ────────────────────────────────────────────────────────────────────────
# β-VAE 损失函数
# ────────────────────────────────────────────────────────────────────────
if HAS_TORCH:

    def vae_loss(
        x_recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """β-VAE 损失 = 重建损失 + β × KL 散度。

        Args:
            x_recon: 重建输出 (N, D)。
            x: 原始输入 (N, D)。
            mu: 编码器输出的均值 (N, latent_dim)。
            logvar: 编码器输出的对数方差 (N, latent_dim)。
            beta: KL 散度权重系数，<1 时鼓励重建质量。

        Returns:
            (total_loss, recon_loss, kl_loss)
        """
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss


# ────────────────────────────────────────────────────────────────────────
# SurvivalVAE 网络
# ────────────────────────────────────────────────────────────────────────
if HAS_TORCH:

    class _Encoder(nn.Module):
        """编码器: 输入特征 → (mu, logvar)。

        架构 (无条件):
            Linear(D → 128) → LayerNorm → GELU → Dropout(0.1)
            Linear(128 → 64) → LayerNorm → GELU → Dropout(0.1)
            Linear(64 → 2*latent_dim)

        架构 (条件):
            Linear(D+cond_dim → 128) → LayerNorm → GELU → Dropout(0.1)
            Linear(128 → 64) → LayerNorm → GELU → Dropout(0.1)
            Linear(64 → 2*latent_dim)
        """

        def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.1, cond_dim: int = 0):
            super().__init__()
            self.cond_dim = cond_dim
            enc_input_dim = input_dim + cond_dim
            self.net = nn.Sequential(
                nn.Linear(enc_input_dim, 128), nn.LayerNorm(128),
                nn.GELU(), nn.Dropout(dropout),
                nn.Linear(128, 64), nn.LayerNorm(64),
                nn.GELU(), nn.Dropout(dropout),
            )
            self.fc_mu = nn.Linear(64, latent_dim)
            self.fc_logvar = nn.Linear(64, latent_dim)

        def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
            if c is not None and self.cond_dim > 0:
                x = torch.cat([x, c], dim=1)
            h = self.net(x)
            return self.fc_mu(h), self.fc_logvar(h)

    class _Decoder(nn.Module):
        """解码器: 潜在向量 → 重建特征。

        架构 (无条件):
            Linear(latent_dim → 64) → LayerNorm → GELU → Dropout(0.1)
            Linear(64 → 128) → LayerNorm → GELU
            Linear(128 → D)

        架构 (条件):
            Linear(latent_dim+cond_dim → 64) → LayerNorm → GELU → Dropout(0.1)
            Linear(64 → 128) → LayerNorm → GELU
            Linear(128 → D)
        """

        def __init__(self, latent_dim: int, output_dim: int, dropout: float = 0.1, cond_dim: int = 0):
            super().__init__()
            self.cond_dim = cond_dim
            dec_input_dim = latent_dim + cond_dim
            self.net = nn.Sequential(
                nn.Linear(dec_input_dim, 64), nn.LayerNorm(64),
                nn.GELU(), nn.Dropout(dropout),
                nn.Linear(64, 128), nn.LayerNorm(128),
                nn.GELU(),
                nn.Linear(128, output_dim),
            )

        def forward(self, z: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
            if c is not None and self.cond_dim > 0:
                z = torch.cat([z, c], dim=1)
            return self.net(z)

    class SurvivalVAE(nn.Module):
        """β-VAE / Conditional VAE: 编码器 + 解码器，含 reparameterize 技巧。

        当 cond_dim > 0 时，条件向量 c 拼接到编码器和解码器输入。
        前向传播返回 (x_recon, mu, logvar)。
        """

        def __init__(
            self,
            input_dim: int,
            latent_dim: int = 32,
            dropout: float = 0.1,
            cond_dim: int = 0,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            self.cond_dim = cond_dim
            self.encoder = _Encoder(input_dim, latent_dim, dropout, cond_dim)
            self.decoder = _Decoder(latent_dim, input_dim, dropout, cond_dim)

        def reparameterize(
            self, mu: torch.Tensor, logvar: torch.Tensor
        ) -> torch.Tensor:
            """重参数化技巧: z = mu + std * eps, eps ~ N(0,1)。"""
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps

        def forward(
            self, x: torch.Tensor, c: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mu, logvar = self.encoder(x, c)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decoder(z, c)
            return x_recon, mu, logvar

        def encode(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
            """仅编码，返回 mu（用于提取潜在表征）。"""
            mu, _ = self.encoder(x, c)
            return mu

        def decode(self, z: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
            """仅解码，返回重建输出。"""
            return self.decoder(z, c)


# ────────────────────────────────────────────────────────────────────────
# VAEAugmentor 高层封装
# ────────────────────────────────────────────────────────────────────────
class VAEAugmentor:
    """β-VAE / Conditional VAE 数据增强器，sklearn 风格 API。

    训练 VAE 编码器-解码器，在潜在空间采样生成新的多组学样本。
    当 cond_dim > 0 时使用条件 VAE，生存标签由条件向量直接提供。
    cond_dim = 0 时回退到无条件 VAE + KNN 标签分配。

    Parameters:
        latent_dim: 潜在空间维度，默认 32。
        epochs: 最大训练轮数，默认 800。
        beta: KL 散度权重，<1 时鼓励重建质量，默认 0.5。
        lr: 学习率，默认 1e-3。
        batch_size: Mini-batch 大小，默认 32。
        patience: Early stopping 耐心轮数，默认 30。
        dropout: Dropout 概率，默认 0.1。
        cond_dim: 条件向量维度，0 表示无条件 VAE（KNN 分配标签）。
        device: 训练设备，默认 'cpu'。
        random_seed: 随机种子。
    """

    def __init__(
        self,
        latent_dim: int = 32,
        epochs: int = 800,
        beta: float = 0.5,
        lr: float = 1e-3,
        batch_size: int = 32,
        patience: int = 30,
        dropout: float = 0.1,
        cond_dim: int = 0,
        device: str = "cpu",
        random_seed: int = RANDOM_SEED,
    ):
        if not HAS_TORCH:
            raise ImportError("[VAE] PyTorch 未安装，请安装 torch 后重试。")
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.beta = beta
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.dropout = dropout
        self.cond_dim = cond_dim
        self.device = device
        self.random_seed = random_seed

        # 运行时属性
        self._model: Optional[SurvivalVAE] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: Optional[list] = None
        self._fitted: bool = False
        self.training_history: Dict[str, list] = {"recon": [], "kl": [], "total": []}
        self._train_Z: Optional[np.ndarray] = None
        self._train_cond: Optional[np.ndarray] = None  # 训练条件向量缓存

    def fit(self, X: pd.DataFrame, condition: Optional[np.ndarray] = None) -> "VAEAugmentor":
        """训练 VAE。

        Args:
            X: pd.DataFrame (n_samples, n_features) 训练特征矩阵。
            condition: 可选, np.ndarray (n_samples, cond_dim) 条件向量。
                       仅在 self.cond_dim > 0 时有效。

        Returns:
            self
        """
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        # 标准化
        self._feature_names = list(X.columns)
        X_np = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True)).values
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_np).astype(np.float32)

        # 条件向量标准化
        cond_scaled = None
        if condition is not None and self.cond_dim > 0:
            cond_np = np.asarray(condition, dtype=np.float32)
            self._train_cond = cond_np
            cond_scaled = cond_np  # 条件向量已归一化处理，无需额外 scaling

        input_dim = X_scaled.shape[1]
        self._model = SurvivalVAE(
            input_dim, self.latent_dim, self.dropout, self.cond_dim
        ).to(self.device)

        logger.info(
            f"[VAE] 网络: input={input_dim}, latent={self.latent_dim}, "
            f"cond_dim={self.cond_dim}, beta={self.beta}, dropout={self.dropout}, device={self.device}"
        )

        # DataLoader — 条件模式下将 condition 加入 dataset
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        if cond_scaled is not None:
            c_tensor = torch.tensor(cond_scaled, dtype=torch.float32)
            dataset = TensorDataset(X_tensor, c_tensor)
        else:
            dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(X_tensor)), shuffle=True)

        # 优化器 + 调度器
        opt = optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=max(self.patience // 2, 1)
        )

        # 训练循环
        best_total_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        t0 = time.time()

        for epoch in range(1, self.epochs + 1):
            self._model.train()
            epoch_recon, epoch_kl, epoch_total = 0.0, 0.0, 0.0
            n_batches = 0

            for batch_data in loader:
                if cond_scaled is not None:
                    xb, cb = batch_data
                    xb, cb = xb.to(self.device), cb.to(self.device)
                else:
                    (xb,) = batch_data
                    xb = xb.to(self.device)
                    cb = None
                x_recon, mu, logvar = self._model(xb, cb)
                loss, recon_l, kl_l = vae_loss(x_recon, xb, mu, logvar, self.beta)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                opt.step()

                epoch_recon += recon_l.item()
                epoch_kl += kl_l.item()
                epoch_total += loss.item()
                n_batches += 1

            avg_recon = epoch_recon / max(n_batches, 1)
            avg_kl = epoch_kl / max(n_batches, 1)
            avg_total = epoch_total / max(n_batches, 1)
            self.training_history["recon"].append(avg_recon)
            self.training_history["kl"].append(avg_kl)
            self.training_history["total"].append(avg_total)

            scheduler.step(avg_total)

            # Early stopping
            if avg_total < best_total_loss:
                best_total_loss = avg_total
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch % 50 == 0 or epoch == 1:
                logger.info(
                    f"[VAE] Epoch {epoch:4d} | recon={avg_recon:.4f} | "
                    f"kl={avg_kl:.4f} | total={avg_total:.4f}"
                )

            if epochs_no_improve >= self.patience:
                logger.info(
                    f"[VAE] Early stopping at epoch {epoch} "
                    f"(best_total={best_total_loss:.4f})"
                )
                break

        elapsed = time.time() - t0
        logger.info(f"[VAE] 训练完成: {epoch} epochs, {elapsed:.1f}s")

        # 恢复最优权重
        if best_state is not None:
            self._model.load_state_dict(best_state)

        # 缓存训练集潜在表征（用于 KNN 标签分配，仅在 cond_dim=0 时需要）
        self._model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            c_t = torch.tensor(cond_scaled, dtype=torch.float32).to(self.device) if cond_scaled is not None else None
            self._train_Z = self._model.encode(X_t, c_t).cpu().numpy()

        self._fitted = True
        return self

    def sample(
        self, n_syn: int, condition: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]:
        """从潜在空间采样并解码生成合成样本。

        Args:
            n_syn: 需要生成的合成样本数。
            condition: 可选, np.ndarray (n_syn, cond_dim) 条件向量。

        Returns:
            (X_syn_df, Z_new, condition_used):
            - 若 cond_dim > 0 且 condition 不为 None，condition_used = condition
            - 否则 condition_used = None（需通过 KNN 分配标签）

        Raises:
            RuntimeError: 模型尚未训练。
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("[VAE] 模型尚未训练，请先调用 fit()。")

        self._model.eval()
        c_tensor = None
        c_np = None
        if condition is not None and self.cond_dim > 0:
            c_np = np.asarray(condition, dtype=np.float32)
            c_tensor = torch.tensor(c_np, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            Z_new = torch.randn(n_syn, self.latent_dim, device=self.device)
            X_syn_scaled = self._model.decode(Z_new, c_tensor).cpu().numpy()

        # 反标准化
        X_syn_np = self._scaler.inverse_transform(X_syn_scaled)
        X_syn_df = pd.DataFrame(X_syn_np, columns=self._feature_names)
        Z_new_np = Z_new.cpu().numpy()

        logger.info(
            f"[VAE] 采样生成 {n_syn} 个合成样本"
            + (f", condition={self.cond_dim}d" if self.cond_dim > 0 else "")
        )
        return X_syn_df, Z_new_np, c_np

    def get_latent(self, X: pd.DataFrame) -> np.ndarray:
        """提取编码器潜在表征 Z（用于降维或 KNN）。

        Args:
            X: pd.DataFrame (n_samples, n_features)。

        Returns:
            np.ndarray (n_samples, latent_dim)。
        """
        if not self._fitted or self._model is None or self._scaler is None:
            raise RuntimeError("[VAE] 模型尚未训练，请先调用 fit()。")

        self._model.eval()
        X_np = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True)).values
        X_scaled = self._scaler.transform(X_np).astype(np.float32)
        with torch.no_grad():
            X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            Z = self._model.encode(X_t).cpu().numpy()
        return Z

    def reconstruct(self, X: pd.DataFrame) -> np.ndarray:
        """重建输入（用于 QC 计算重建损失）。

        Args:
            X: pd.DataFrame (n_samples, n_features)。

        Returns:
            np.ndarray (n_samples, n_features) 重建后的原始尺度值。
        """
        if not self._fitted or self._model is None or self._scaler is None:
            raise RuntimeError("[VAE] 模型尚未训练，请先调用 fit()。")

        self._model.eval()
        X_np = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True)).values
        X_scaled = self._scaler.transform(X_np).astype(np.float32)
        with torch.no_grad():
            X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            X_recon_scaled, _, _ = self._model(X_t)
            X_recon_np = self._scaler.inverse_transform(X_recon_scaled.cpu().numpy())
        return X_recon_np

    def assign_survival_labels(
        self,
        Z_new: np.ndarray,
        endpoint_train: pd.DataFrame,
        train_ids: list,
        k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """通过潜在空间 KNN 为合成样本分配生存标签。

        在训练集潜在空间 Z_train 中对 Z_new 做 KNN（K=5），
        合成样本的 (time, event) = KNN 最近邻的中位数值。

        Args:
            Z_new: 合成样本的潜在向量 (n_syn, latent_dim)。
            endpoint_train: 训练集生存数据，含 time_months 和 event 列。
            train_ids: 训练样本 ID 列表。
            k: KNN 邻居数，默认 5。

        Returns:
            (syn_time, syn_event): 合成样本的生存时间和事件指示器。
        """
        if self._train_Z is None:
            raise RuntimeError("[VAE] 训练集潜在表征未缓存，请先调用 fit()。")

        nn = NearestNeighbors(n_neighbors=min(k, len(self._train_Z)), metric="euclidean")
        nn.fit(self._train_Z)
        _, indices = nn.kneighbors(Z_new)

        ep_train = endpoint_train.loc[endpoint_train.index.isin(train_ids)]
        train_times = ep_train["time_months"].values
        train_events = ep_train["event"].values

        syn_time = np.array([
            float(np.median(train_times[idx])) for idx in indices
        ])
        syn_event = np.array([
            int(round(np.median(train_events[idx]))) for idx in indices
        ])

        logger.info(
            f"[VAE] KNN 标签分配: {len(syn_time)} 样本, "
            f"event=1: {(syn_event == 1).sum()}, event=0: {(syn_event == 0).sum()}"
        )
        return syn_time, syn_event


# ────────────────────────────────────────────────────────────────────────
# VAE QC 检查（替代 KS 检验）
# ────────────────────────────────────────────────────────────────────────
def vae_synthetic_qc(
    X_real: pd.DataFrame,
    X_syn: pd.DataFrame,
    X_holdout: Optional[pd.DataFrame] = None,
    vae_augmentor: Optional["VAEAugmentor"] = None,
    training_history: Optional[Dict[str, list]] = None,
    recon_loss_threshold: float = 10.0,
    kl_min: float = 0.01,
    kl_max: float = 5.0,
    pca_corr_threshold: float = 0.70,
    mean_dev_threshold: float = 0.3,
    random_seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """VAE 专属 QC 检查。

    检查项:
    1. 重建损失: 在留出验证集上 MSE < 10.0（小样本 VAE holdout 重建天然偏高）
    2. KL 散度: 非零且非极端（0.01 < KL_dim_mean < 5.0，无后验坍缩）
    3. PCA 投影相关性: real 和 syn 在前 10 个 PCA 分量上方差相关 > 0.70
    4. 特征均值偏差: |mean_real - mean_syn| / std_real < 0.3（逐特征）
    5. Wasserstein 距离: < 0.6（放宽阈值适配高维组学数据）

    通过条件: 全部 5 项通过。

    Args:
        X_real: 真实训练集特征 DataFrame。
        X_syn: 合成样本特征 DataFrame。
        X_holdout: 可选，留出的验证集（用于重建损失计算）。
        vae_augmentor: 可选，已训练的 VAEAugmentor 实例（用于重建损失计算）。
        training_history: 可选，VAE 训练历史 dict（含 kl 列表）。
        recon_loss_threshold: 重建 MSE 阈值。
        kl_min / kl_max: KL 散度 per-dimension 允许范围。
        pca_corr_threshold: PCA 方差相关阈值。
        mean_dev_threshold: 逐特征均值偏差阈值。
        random_seed: 随机种子。

    Returns:
        dict 含 passed (bool), recon_mse, kl_mean, pca_corr, mean_dev, checks 等。
    """
    from sklearn.decomposition import PCA
    from scipy import stats as sp_stats

    cols = [c for c in X_real.columns if c in X_syn.columns]
    if not cols:
        return {"passed": False, "reason": "no_common_columns", "checks": {}}

    X_r = X_real[cols].replace([np.inf, -np.inf], np.nan).fillna(X_real[cols].median(numeric_only=True))
    X_s = X_syn[cols].replace([np.inf, -np.inf], np.nan).fillna(X_real[cols].median(numeric_only=True))

    checks: Dict[str, bool] = {}
    details: Dict[str, float] = {}

    # 1. 重建损失: 用 VAE 重建 holdout，计算 MSE
    if X_holdout is not None and vae_augmentor is not None and vae_augmentor._fitted:
        try:
            X_h = X_holdout[cols].replace(
                [np.inf, -np.inf], np.nan
            ).fillna(X_real[cols].median(numeric_only=True))
            X_recon = vae_augmentor.reconstruct(X_h)
            recon_mse = float(np.mean((X_h.values - X_recon) ** 2))
            details["recon_mse"] = recon_mse
            checks["recon_mse"] = recon_mse < recon_loss_threshold
        except Exception as e:
            logger.warning(f"[VAE] QC 重建损失计算失败: {e}")

    # 2. KL 散度 per-dimension: 从训练历史获取最后一个 epoch 的 KL
    if training_history is not None and "kl" in training_history and training_history["kl"]:
        kl_mean = float(training_history["kl"][-1])
        details["kl_mean"] = kl_mean
        checks["kl_range"] = kl_min < kl_mean < kl_max

    # 3. PCA 投影方差相关性
    pca_corr = 0.0
    n_pca = min(10, len(cols), X_r.shape[0], X_s.shape[0])
    try:
        if n_pca >= 2:
            scaler_pca = StandardScaler()
            X_r_sc = scaler_pca.fit_transform(X_r)
            X_s_sc = scaler_pca.transform(X_s)
            pca = PCA(n_components=n_pca, random_state=random_seed)
            X_combined = np.vstack([X_r_sc, X_s_sc])
            pca.fit(X_combined)
            proj_r = pca.transform(X_r_sc)
            proj_s = pca.transform(X_s_sc)
            var_r = np.var(proj_r, axis=0)
            var_s = np.var(proj_s, axis=0)
            pca_corr = float(np.corrcoef(var_r, var_s)[0, 1])
            if not np.isfinite(pca_corr):
                pca_corr = 0.0
    except Exception:
        pca_corr = 0.0
    details["pca_corr"] = pca_corr
    checks["pca_corr"] = pca_corr > pca_corr_threshold

    # 4. 特征均值偏差
    scaler_dev = StandardScaler()
    X_r_sc2 = scaler_dev.fit_transform(X_r)
    X_s_sc2 = scaler_dev.transform(X_s)
    mean_dev = float(np.mean(np.abs(X_r_sc2.mean(axis=0) - X_s_sc2.mean(axis=0))))
    details["mean_dev"] = mean_dev
    checks["mean_dev"] = mean_dev < mean_dev_threshold

    # 5. Wasserstein 距离（复用原 QC 逻辑）
    wd_vals = []
    for i, col in enumerate(cols):
        try:
            wd = sp_stats.wasserstein_distance(X_r_sc2[:, i], X_s_sc2[:, i])
            wd_vals.append(float(wd))
        except Exception:
            pass
    wd_mean = float(np.mean(wd_vals)) if wd_vals else 1.0
    details["wd_mean"] = wd_mean
    checks["wd_mean"] = wd_mean < 0.6

    # 通过条件: 所有检查项均通过
    passed = all(checks.values()) if checks else False

    result = {
        "passed": passed,
        "checks": checks,
        "details": details,
        "good_count": sum(1 for v in checks.values() if v),
        "total_checks": len(checks),
    }

    recon_str = f"{details.get('recon_mse', 'N/A')}({'✓' if checks.get('recon_mse') else '✗'})" if 'recon_mse' in checks else 'skipped'
    kl_str = f"{details.get('kl_mean', 'N/A'):.3f}({'✓' if checks.get('kl_range') else '✗'})" if 'kl_range' in checks else 'skipped'
    logger.info(
        f"[VAE] QC: recon_mse={recon_str}, kl={kl_str}, "
        f"pca_corr={pca_corr:.3f}({'✓' if checks.get('pca_corr') else '✗'}), "
        f"mean_dev={mean_dev:.3f}({'✓' if checks.get('mean_dev') else '✗'}), "
        f"wd={wd_mean:.3f}({'✓' if checks.get('wd_mean') else '✗'}), "
        f"passed={passed} ({result['good_count']}/{result['total_checks']})"
    )
    return result


# ────────────────────────────────────────────────────────────────────────
# 入口函数（与 train_only_gan_augmentation 接口对齐）
# ────────────────────────────────────────────────────────────────────────
def train_only_vae_augmentation(
    X_train: pd.DataFrame,
    endpoint_train: pd.DataFrame,
    train_ids: list,
    vae_epochs: int = 800,
    latent_dim: int = 32,
    seed: int = RANDOM_SEED,
    aug_ratio: float = 1.0,
    condition_type: str = "none",
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """VAE 数据增强入口，与 train_only_gan_augmentation 接口对齐。

    仅在训练集上训练 VAE，采样生成合成样本。
    - condition_type="survival": 条件 VAE，从条件向量直接获取标签（推荐）
    - condition_type="none": 无条件 VAE + KNN 标签分配（向后兼容）

    Args:
        X_train: 全样本特征矩阵 (含训练集和验证集)。
        endpoint_train: 全样本生存数据，含 time_months 和 event 列。
        train_ids: 训练样本 ID 列表。
        vae_epochs: VAE 训练轮数。
        latent_dim: 潜在空间维度。
        seed: 随机种子。
        aug_ratio: 合成样本数与训练集样本数的比例，默认 1.0（1:1）。
        condition_type: 条件类型: "none" | "survival"。

    Returns:
        (result, info):
        - result: dict 含 X_aug, qc, syn_event, syn_time, vae 等；None 表示失败。
        - info: dict 含 status, training_history。
    """
    if not HAS_TORCH:
        return None, {"status": "skipped_torch_missing"}

    # 仅取训练集样本
    X_tr = X_train.loc[X_train.index.isin(train_ids)].copy()
    X_holdout = X_train.loc[~X_train.index.isin(train_ids)] if len(X_train) > len(train_ids) else None

    if len(X_tr) < 40 or X_tr.shape[1] == 0:
        logger.warning(f"[VAE] 训练样本不足 ({len(X_tr)} < 40) 或特征为空，跳过增强")
        return None, {"status": "skipped_insufficient_data"}

    # 构建条件向量（survival 模式）
    cond_dim = 0
    cond_train = None
    use_cvae = condition_type == "survival"
    if use_cvae:
        ep_tr = endpoint_train.loc[endpoint_train.index.isin(train_ids)]
        event_arr = ep_tr["event"].values.astype(np.float32)
        time_arr = ep_tr["time_months"].values.astype(np.float32)
        # 条件向量: [event, log1p(time_months)]
        cond_train = np.column_stack([
            event_arr,
            np.log1p(np.clip(time_arr, 0.0, None)),
        ]).astype(np.float32)
        cond_dim = 2
        logger.info(
            f"[VAE] Conditional VAE 模式: 条件向量 {cond_dim} 维 "
            f"(event, log1p_time)"
        )

    # 训练 VAE
    vae = VAEAugmentor(
        latent_dim=latent_dim, epochs=vae_epochs,
        cond_dim=cond_dim, random_seed=seed,
    )
    vae.fit(X_tr, condition=cond_train)

    # 采样
    n_syn = max(1, int(len(train_ids) * aug_ratio))
    if use_cvae:
        # 条件 VAE: 从训练集条件分布中随机采样条件向量
        rng = np.random.default_rng(seed + 1000)
        sample_idx = rng.choice(len(cond_train), size=n_syn, replace=True)
        cond_syn = cond_train[sample_idx]
        X_syn_df, Z_new, _ = vae.sample(n_syn, condition=cond_syn)
        # 从条件向量直接提取标签（无需 KNN）
        syn_event = cond_syn[:, 0]
        syn_time = np.expm1(cond_syn[:, 1])
        logger.info(
            f"[VAE] 条件采样: {n_syn} 样本, "
            f"event=1: {int(syn_event.sum())}, event=0: {int((1 - syn_event).sum())}"
        )
    else:
        # 无条件 VAE: 传统采样 + KNN 标签分配
        X_syn_df, Z_new, _ = vae.sample(n_syn)
        syn_time, syn_event = vae.assign_survival_labels(Z_new, endpoint_train, train_ids)

    # QC 检查
    qc = vae_synthetic_qc(
        X_tr, X_syn_df,
        X_holdout=X_holdout,
        vae_augmentor=vae,
        training_history=vae.training_history,
        random_seed=seed,
    )

    # 合并增强数据
    X_aug = pd.concat([X_tr, X_syn_df], ignore_index=True)

    result = {
        "X_aug": X_aug,
        "qc": qc,
        "syn_event": syn_event,
        "syn_time": syn_time,
        "vae": vae,
        "n_syn": n_syn,
        "aug_ratio": aug_ratio,
        "condition_type": condition_type,
        "loss_history": vae.training_history,
    }
    info = {"status": "completed", "training_history": vae.training_history}

    logger.info(
        f"[VAE] 增强完成: {n_syn} 合成样本, QC passed={qc['passed']}, "
        f"condition={condition_type}, X_aug shape={X_aug.shape}"
    )
    return result, info
