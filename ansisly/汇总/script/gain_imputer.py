"""gain_imputer.py — GAIN (Generative Adversarial Imputation Nets) 填补器

基于 Yoon et al. (2018) "GAIN: Missing Data Imputation using Generative Adversarial Nets"
轻量级纯 PyTorch 实现，适用于中等规模的组学数据矩阵。

设计要点:
    - 纯 PyTorch，不依赖 pytorch-lightning
    - 输入/输出均为 numpy 数组，内部自动转 tensor
    - MinMax 归一化 → 训练 → 反归一化
    - 自动检测全 NaN 列并跳过
    - Early stopping 防止过拟合
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 网络架构
# ─────────────────────────────────────────────────────────────────────────────

class _Generator(nn.Module):
    """Generator: 输入 [X_with_noise ⊙ M + Z ⊙ (1-M), M] → X_hat

    接收观测值（噪声填补）和缺失掩码，输出完整矩阵估计。
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),   # [X_input, M]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Tanh(),                         # 输出 [-1, 1]，匹配 MinMax 归一化范围
        )

    def forward(self, x_with_noise: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_with_noise, mask], dim=1))


class _Discriminator(nn.Module):
    """Discriminator: 判断每个特征值是否来自真实观测。

    接收数据矩阵和 Hint 矩阵，输出每个特征为"真实观测"的概率。
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),   # [X, H]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, hint: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, hint], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# GAIN Imputer
# ─────────────────────────────────────────────────────────────────────────────

class GAINImputer:
    """GAIN 缺失值填补器

    Parameters
    ----------
    hint_rate : float
        向判别器暴露真实缺失模式的比例 (0-1)，默认 0.9
    alpha : float
        MSE 重建损失权重系数，默认 100.0
    batch_size : int
        小批量训练大小，默认 64
    n_epochs : int
        最大训练轮数，默认 500
    lr : float
        Adam 学习率，默认 1e-3
    hidden_dim : int or None
        隐藏层维度；None 时自动设为 min(2*d, 256)
    device : str
        训练设备，默认 'cpu'
    patience : int
        Early stopping 耐心轮数，默认 50
    seed : int
        随机种子，默认 42
    """

    def __init__(
        self,
        hint_rate: float = 0.9,
        alpha: float = 100.0,
        batch_size: int = 64,
        n_epochs: int = 500,
        lr: float = 1e-3,
        hidden_dim: Optional[int] = None,
        device: str = "cpu",
        patience: int = 50,
        seed: int = 42,
    ):
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.device = device
        self.patience = patience
        self.seed = seed

    def fit_transform(
        self,
        X: np.ndarray,
        return_validation: bool = False,
        validation_mask_rate: float = 0.1,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
        """对含缺失值的矩阵执行 GAIN 填补

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            含 NaN 的浮点矩阵
        return_validation : bool
            若 True，额外返回 masking 验证指标 (RMSE/MAE/R²)
        validation_mask_rate : float
            Masking 验证时随机隐藏的已知观测值比例 (0-1)

        Returns
        -------
        np.ndarray or (np.ndarray, dict)
            填补后的完整矩阵；若 return_validation=True 则返回元组
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X = X.copy().astype(np.float32)
        n_samples, n_features = X.shape

        # ── 检测缺失模式 ──────────────────────────────────────────────────
        # mask: 1=观测值, 0=缺失值
        mask = (~np.isnan(X)).astype(np.float32)
        missing_frac = 1.0 - mask.mean()

        if missing_frac == 0.0:
            logger.info("  [GAIN] 矩阵无缺失值，直接返回")
            return X

        logger.info(
            f"  [GAIN] 矩阵: {n_samples}×{n_features}, "
            f"缺失率: {missing_frac:.2%}, 开始训练..."
        )

        # ── Masking 验证：随机隐藏部分已知观测值 ──────────────────────────
        validation_metrics = {}
        val_mask = None  # 1=被隐藏用于验证的观测值
        _val_true_normalized = None  # 保存被隐藏位置的归一化真实值
        if return_validation and 0 < validation_mask_rate < 1:
            rng_val = np.random.RandomState(self.seed + 1)
            observed_indices = np.argwhere(mask == 1.0)
            n_observed = len(observed_indices)
            n_holdout = max(1, int(n_observed * validation_mask_rate))
            holdout_idx = rng_val.choice(n_observed, size=n_holdout, replace=False)
            holdout_coords = observed_indices[holdout_idx]  # shape: (n_holdout, 2)
            # 记录坐标
            val_mask = np.zeros_like(mask)
            for idx in holdout_coords:
                val_mask[idx[0], idx[1]] = 1.0
            validation_metrics["n_holdout"] = n_holdout
            # 保存被隐藏位置的原始值（归一化前）
            _val_true_raw = X[holdout_coords[:, 0], holdout_coords[:, 1]].copy()
            # 隐藏观测值
            X[holdout_coords[:, 0], holdout_coords[:, 1]] = np.nan
            mask[holdout_coords[:, 0], holdout_coords[:, 1]] = 0.0
            logger.info(
                f"    [GAIN] Masking 验证: 隐藏 {n_holdout}/{n_observed} "
                f"({n_holdout/max(n_observed,1)*100:.1f}%) 已知观测值"
            )

        # ── 处理全 NaN 列（标记为已填充，不参与 GAIN 训练）───────────────
        col_all_nan = mask.sum(axis=0) == 0
        if col_all_nan.any():
            n_all_nan = int(col_all_nan.sum())
            logger.warning(f"  [GAIN] 发现 {n_all_nan} 列全 NaN，将以 0 填充并跳过")
            X[:, col_all_nan] = 0.0
            mask[:, col_all_nan] = 1.0   # 标记为已填充

        # ── MinMax 归一化 ──────────────────────────────────────────────────
        col_min = np.where(col_all_nan, 0.0, np.nanmin(X, axis=0))
        col_max = np.where(col_all_nan, 1.0, np.nanmax(X, axis=0))
        col_range = col_max - col_min
        col_range[col_range == 0.0] = 1.0   # 常数列：range=1 避免除零

        X_norm = (X - col_min) / col_range
        X_norm = np.where(np.isnan(X_norm), 0.0, X_norm)   # NaN → 0

        # ── 构建 PyTorch 数据集 ────────────────────────────────────────────
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)
        M_tensor = torch.tensor(mask, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, M_tensor)
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, n_samples),
            shuffle=True,
            drop_last=False,
        )

        # ── 初始化网络 ─────────────────────────────────────────────────────
        hidden_dim = self.hidden_dim or min(2 * n_features, 256)
        G = _Generator(n_features, hidden_dim).to(self.device)
        D = _Discriminator(n_features, hidden_dim).to(self.device)

        opt_G = optim.Adam(G.parameters(), lr=self.lr)
        opt_D = optim.Adam(D.parameters(), lr=self.lr)

        # ── 训练循环 ───────────────────────────────────────────────────────
        best_d_loss = float("inf")
        patience_counter = 0
        t_start = time.time()

        for epoch in range(self.n_epochs):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            n_batches = 0

            for X_batch, M_batch in loader:
                X_batch = X_batch.to(self.device)
                M_batch = M_batch.to(self.device)

                # ── 噪声输入 ──
                Z = torch.rand_like(X_batch)
                X_input = X_batch * M_batch + Z * (1 - M_batch)

                # ── 生成器前向 ──
                G_sample = G(X_input, M_batch)
                # 观测值保持不变，仅替换缺失位置
                X_hat = X_batch * M_batch + G_sample * (1 - M_batch)

                # ── Hint 矩阵 ──
                H = self._generate_hint(M_batch)

                # ── 判别器损失 ──
                D_real = D(X_batch, H)
                D_fake = D(X_hat.detach(), H)
                D_loss = -(
                    torch.mean(M_batch * torch.log(D_real + 1e-8))
                    + torch.mean((1 - M_batch) * torch.log(1 - D_fake + 1e-8))
                )

                opt_D.zero_grad()
                D_loss.backward()
                opt_D.step()

                # ── 生成器损失 ──
                Z2 = torch.rand_like(X_batch)
                X_input2 = X_batch * M_batch + Z2 * (1 - M_batch)
                G_sample2 = G(X_input2, M_batch)
                X_hat2 = X_batch * M_batch + G_sample2 * (1 - M_batch)

                D_fake2 = D(X_hat2, H)
                G_adv = -torch.mean(
                    (1 - M_batch) * torch.log(D_fake2 + 1e-8)
                )
                G_mse = (
                    torch.mean(M_batch * (X_batch - G_sample2) ** 2)
                    / (torch.mean(M_batch) + 1e-8)
                )
                G_loss = G_adv + self.alpha * G_mse

                opt_G.zero_grad()
                G_loss.backward()
                opt_G.step()

                epoch_d_loss += D_loss.item()
                epoch_g_loss += G_loss.item()
                n_batches += 1

            avg_d = epoch_d_loss / max(n_batches, 1)
            avg_g = epoch_g_loss / max(n_batches, 1)

            if (epoch + 1) % 50 == 0 or epoch == 0:
                logger.info(
                    f"    [GAIN] Epoch {epoch + 1}/{self.n_epochs} | "
                    f"D_loss={avg_d:.4f} G_loss={avg_g:.4f}"
                )

            # ── Early stopping ──
            if avg_d < best_d_loss - 1e-4:
                best_d_loss = avg_d
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                logger.info(
                    f"    [GAIN] Early stopping at epoch {epoch + 1} "
                    f"(patience={self.patience})"
                )
                break

        t_elapsed = time.time() - t_start
        logger.info(
            f"    [GAIN] 训练完成: {epoch + 1} epochs, {t_elapsed:.1f}s"
        )

        # ── 推理：用训练好的 G 填补全部缺失值 ─────────────────────────────
        G.eval()
        with torch.no_grad():
            X_all = X_tensor.to(self.device)
            M_all = M_tensor.to(self.device)
            Z_all = torch.rand_like(X_all)
            X_input_all = X_all * M_all + Z_all * (1 - M_all)
            G_imputed = G(X_input_all, M_all)
            X_filled = X_all * M_all + G_imputed * (1 - M_all)
            X_filled = X_filled.cpu().numpy()

        # ── 反归一化 ───────────────────────────────────────────────────────
        X_result = X_filled * col_range + col_min

        # ── 恢复全 NaN 列为 NaN ────────────────────────────────────────────
        if col_all_nan.any():
            X_result[:, col_all_nan] = np.nan

        # ── Masking 验证：比较预测值 vs 被隐藏的真实值 ─────────────────────
        if return_validation and val_mask is not None and val_mask.any() and _val_true_raw is not None:
            try:
                row_indices, col_indices = np.where(val_mask.astype(bool))
                # 反归一化 GAIN 预测值
                val_pred_norm = X_filled[row_indices, col_indices]
                val_pred_orig = val_pred_norm * col_range[col_indices] + col_min[col_indices]
                val_true_orig = _val_true_raw  # 原始尺度真实值
                # 计算指标
                residuals = val_true_orig - val_pred_orig
                rmse = float(np.sqrt(np.mean(residuals ** 2)))
                mae = float(np.mean(np.abs(residuals)))
                median_ae = float(np.median(np.abs(residuals)))
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((val_true_orig - np.mean(val_true_orig)) ** 2)
                r_squared = float(1.0 - ss_res / max(ss_tot, 1e-12))
                validation_metrics.update({
                    "rmse": round(rmse, 6),
                    "mae": round(mae, 6),
                    "r_squared": round(r_squared, 4),
                    "median_absolute_error": round(median_ae, 6),
                })
                logger.info(
                    f"    [GAIN] Masking 验证: RMSE={rmse:.4f}, MAE={mae:.4f}, "
                    f"R²={r_squared:.4f}, MedAE={median_ae:.4f}"
                )
            except Exception as e:
                logger.warning(f"    [GAIN] Masking 验证计算失败: {e}")

        if return_validation:
            return X_result.astype(np.float32), validation_metrics
        return X_result.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────

    def _generate_hint(self, mask: torch.Tensor) -> torch.Tensor:
        """生成判别器 Hint 矩阵

        以 hint_rate 概率暴露真实观测/缺失状态，其余位置设为 0.5（未知）。
        """
        B = torch.bernoulli(torch.full_like(mask, self.hint_rate))
        H = mask * B + 0.5 * (1 - B)
        return H
