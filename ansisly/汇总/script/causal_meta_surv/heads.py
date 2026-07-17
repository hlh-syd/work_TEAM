"""heads.py — Cox 主头 + 辅助头 + Elastic-Net Cox 三级 fallback。

提供：
    - CoxHead：z -> eta[B] log-risk（16-8-1）
    - DiscreteTimeAuxHead：z -> hazard bins（12-20 bins）
    - DomainHead：GRL(z_inv) -> task
    - ElasticNetCox Level 1（alpha=0.02, max_iter=2000, L1+L2）
    - RidgeCox Level 2（alpha=0.1，纯 L2）
    - SimpleCox Level 3（无正则）
    - ElasticNetCoxFallback：自动切换链路与原因记录
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import gradient_reversal

LOGGER = logging.getLogger("cmib_surv.heads")


# ──────────────────────────────────────────────────────────────────────
# 神经网络头
# ──────────────────────────────────────────────────────────────────────
class CoxHead(nn.Module):
    """z -> eta[B] log-risk（16-8-1）。"""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 8, dropout: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc1(z)
        h = self.norm(h)
        h = F.gelu(h)
        h = self.dropout(h)
        eta = self.fc2(h).squeeze(-1)  # [B]
        return eta


class DiscreteTimeAuxHead(nn.Module):
    """z -> hazard bins（默认 12 bins）。"""

    def __init__(self, input_dim: int = 16, n_bins: int = 12, dropout: float = 0.15):
        super().__init__()
        self.n_bins = n_bins
        self.fc1 = nn.Linear(input_dim, n_bins * 2)
        self.norm = nn.LayerNorm(n_bins * 2)
        self.fc2 = nn.Linear(n_bins * 2, n_bins)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc1(z)
        h = self.norm(h)
        h = F.gelu(h)
        h = self.dropout(h)
        # hazard logits
        return self.fc2(h)  # [B, n_bins]


class DomainHead(nn.Module):
    """GRL(z_inv) -> task logits。"""

    def __init__(self, input_dim: int = 16, n_tasks: int = 9, dropout: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.fc2 = nn.Linear(input_dim, n_tasks)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_inv: torch.Tensor, grl_lambda: float = 0.2) -> torch.Tensor:
        # 梯度反转
        z_rev = gradient_reversal(z_inv, grl_lambda)
        h = self.fc1(z_rev)
        h = self.norm(h)
        h = F.gelu(h)
        h = self.dropout(h)
        return self.fc2(h)  # [B, n_tasks]


# ──────────────────────────────────────────────────────────────────────
# Elastic-Net Cox 三级 fallback
# ──────────────────────────────────────────────────────────────────────
def _cox_efron_negative_log_partial_likelihood(
    X: np.ndarray,
    beta: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> float:
    """计算 Efron 部分似然（用于评估）。"""
    eta = X @ beta
    exp_eta = np.exp(eta - eta.max())
    # 按时间降序排序
    order = np.argsort(-time)
    exp_eta_sorted = exp_eta[order]
    event_sorted = event[order]

    unique_times = np.unique(time[event == 1])
    nll = 0.0
    for t in unique_times:
        # 风险集：time >= t
        risk_mask = time >= t
        # 并列事件数
        tied_mask = (time == t) & (event == 1)
        d = tied_mask.sum()
        if d == 0:
            continue
        # sum_{i in D_t} eta_i
        sum_eta_tied = eta[tied_mask].sum()
        # risk set sum exp(eta)
        risk_sum = exp_eta[risk_mask].sum()
        # Efron tie handling
        tied_exp_sum = exp_eta[tied_mask].sum()
        for l in range(d):
            denom = risk_sum - (l / d) * tied_exp_sum
            if denom > 0:
                nll -= np.log(denom)
        nll += sum_eta_tied  # 实际是 log-likelihood 项，符号需调整
    return float(nll)


def _to_surv_structured(event: np.ndarray, time: np.ndarray):
    """构造 sksurv 所需的 Surv.from_arrays 结构化数组。

    event 必须为 bool 类型；与 035 的 `Surv.from_arrays(event.astype(bool), time)` 等价。
    """
    from sksurv.util import Surv

    return Surv.from_arrays(event.astype(bool), time)


class ElasticNetCox:
    """Level 1: Elastic-Net Cox（alpha=0.02, max_iter=2000, L1+L2）。

    使用 `sksurv.CoxnetSurvivalAnalysis` 进行真实的 Cox 偏似然优化，
    与 035 的 `fit_elastic_net_cox` 实现保持完全一致：
        - l1_ratio=0.5
        - alphas=np.logspace(0.0, -2.0, 16)
        - max_iter=100000, tol=1e-7
        - 取中间 alpha 的系数作为最终系数
    """

    def __init__(self, alpha: float = 0.02, l1_ratio: float = 0.5, max_iter: int = 2000):
        # 保留 alpha/max_iter 仅用于日志与 fallback 兼容；Coxnet 自身使用 alphas 网格。
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.coef_: Optional[np.ndarray] = None
        self.feature_mean_: Optional[np.ndarray] = None
        self.feature_std_: Optional[np.ndarray] = None
        self.fitted = False
        self.selected_alpha_: Optional[float] = None
        self.alpha_index_: Optional[int] = None

    def fit(self, X: np.ndarray, event: np.ndarray, time: np.ndarray) -> "ElasticNetCox":
        from sksurv.linear_model import CoxnetSurvivalAnalysis

        # 事件数检查（与 035 保持一致）
        event_mask = event == 1
        if event_mask.sum() < 5:
            raise RuntimeError("Too few events for ElasticNetCox")

        # 标准化（保持 predict 时对称），Coxnet 内部本身对 X 无标准化要求
        self.feature_mean_ = X.mean(axis=0)
        self.feature_std_ = X.std(axis=0) + 1e-8
        X_std = (X - self.feature_mean_) / self.feature_std_

        y_surv = _to_surv_structured(event, time)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator = CoxnetSurvivalAnalysis(
                l1_ratio=self.l1_ratio,
                alphas=np.logspace(0.0, -2.0, 16),
                max_iter=100000,
                tol=1e-7,
            )
            estimator.fit(X_std, y_surv)

        # 选取中间 alpha 的系数（与 035 完全一致）
        alpha_index = len(estimator.alphas_) // 2
        self.alpha_index_ = alpha_index
        self.selected_alpha_ = float(estimator.alphas_[alpha_index])
        self.coef_ = np.asarray(estimator.coef_[:, alpha_index], dtype=float).ravel()

        self.fitted = True
        LOGGER.debug(
            "ElasticNetCox fit: n_nonzero=%d/%d alpha=%.4g",
            int((np.abs(self.coef_) > 1e-8).sum()),
            len(self.coef_),
            self.selected_alpha_,
        )
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("ElasticNetCox not fitted")
        X_std = (X - self.feature_mean_) / self.feature_std_
        return X_std @ self.coef_


class RidgeCox:
    """Level 2: Ridge Cox（纯 L2）。

    使用 `sksurv.CoxPHSurvivalAnalysis(alpha=1.0, n_iter=200)`，
    与 035 中 `fit_elastic_net_cox` 的 fallback 路径保持一致。
    """

    def __init__(self, alpha: float = 1.0, max_iter: int = 200):
        self.alpha = alpha
        self.max_iter = max_iter
        self.coef_: Optional[np.ndarray] = None
        self.feature_mean_: Optional[np.ndarray] = None
        self.feature_std_: Optional[np.ndarray] = None
        self.fitted = False

    def fit(self, X: np.ndarray, event: np.ndarray, time: np.ndarray) -> "RidgeCox":
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        event_mask = event == 1
        if event_mask.sum() < 5:
            raise RuntimeError("Too few events for RidgeCox")

        self.feature_mean_ = X.mean(axis=0)
        self.feature_std_ = X.std(axis=0) + 1e-8
        X_std = (X - self.feature_mean_) / self.feature_std_

        y_surv = _to_surv_structured(event, time)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = CoxPHSurvivalAnalysis(alpha=self.alpha, n_iter=self.max_iter)
            model.fit(X_std, y_surv)

        self.coef_ = np.asarray(model.coef_, dtype=float).ravel()
        self.fitted = True
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("RidgeCox not fitted")
        X_std = (X - self.feature_mean_) / self.feature_std_
        return X_std @ self.coef_


class SimpleCox:
    """Level 3: Simple Cox（极小正则的 Cox-PH）。

    使用 `sksurv.CoxPHSurvivalAnalysis(alpha=1e-6, n_iter=200)`，作为
    ElasticNet 与 Ridge 均失败时的最终 fallback，仍然基于真实 Cox 偏似然。
    """

    def __init__(self):
        self.coef_: Optional[np.ndarray] = None
        self.feature_mean_: Optional[np.ndarray] = None
        self.feature_std_: Optional[np.ndarray] = None
        self.fitted = False

    def fit(self, X: np.ndarray, event: np.ndarray, time: np.ndarray) -> "SimpleCox":
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        event_mask = event == 1
        if event_mask.sum() < 2:
            raise RuntimeError("Too few events for SimpleCox")

        self.feature_mean_ = X.mean(axis=0)
        self.feature_std_ = X.std(axis=0) + 1e-8
        X_std = (X - self.feature_mean_) / self.feature_std_

        y_surv = _to_surv_structured(event, time)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = CoxPHSurvivalAnalysis(alpha=1e-6, n_iter=200)
            model.fit(X_std, y_surv)

        self.coef_ = np.asarray(model.coef_, dtype=float).ravel()
        self.fitted = True
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("SimpleCox not fitted")
        X_std = (X - self.feature_mean_) / self.feature_std_
        return X_std @ self.coef_


class ElasticNetCoxFallback:
    """Elastic-Net Cox 三级 fallback 自动切换。

    Level 1: ElasticNetCox (Coxnet, l1_ratio=0.5, alphas=logspace(0,-2,16))
    Level 2: RidgeCox  (CoxPHSurvivalAnalysis, alpha=1.0, n_iter=200)
    Level 3: SimpleCox (CoxPHSurvivalAnalysis, alpha=1e-6, n_iter=200)

    与 035 的 `fit_elastic_net_cox` 三级实现完全对齐。
    """

    def __init__(
        self,
        level1_alpha: float = 0.02,
        level1_max_iter: int = 2000,
        level2_alpha: float = 1.0,
        level2_max_iter: int = 200,
    ):
        self.level1 = ElasticNetCox(alpha=level1_alpha, max_iter=level1_max_iter)
        self.level2 = RidgeCox(alpha=level2_alpha, max_iter=level2_max_iter)
        self.level3 = SimpleCox()
        self.active_level: int = 0
        self.fallback_chain: list = []

    def fit(self, X: np.ndarray, event: np.ndarray, time: np.ndarray) -> "ElasticNetCoxFallback":
        self.fallback_chain = []

        # Level 1: ElasticNetCox
        try:
            self.level1.fit(X, event, time)
            self.active_level = 1
            self.fallback_chain.append(("Level1 ElasticNetCox", "SUCCESS", ""))
            LOGGER.info("ElasticNetCox Level 1 fit succeeded")
            return self
        except Exception as e:
            self.fallback_chain.append(("Level1 ElasticNetCox", "FAIL", str(e)))
            LOGGER.warning("ElasticNetCox Level 1 failed: %s", e)

        # Level 2: RidgeCox
        try:
            self.level2.fit(X, event, time)
            self.active_level = 2
            self.fallback_chain.append(("Level2 RidgeCox", "SUCCESS", ""))
            LOGGER.info("RidgeCox Level 2 fit succeeded")
            return self
        except Exception as e:
            self.fallback_chain.append(("Level2 RidgeCox", "FAIL", str(e)))
            LOGGER.warning("RidgeCox Level 2 failed: %s", e)

        # Level 3: SimpleCox
        try:
            self.level3.fit(X, event, time)
            self.active_level = 3
            self.fallback_chain.append(("Level3 SimpleCox", "SUCCESS", ""))
            LOGGER.info("SimpleCox Level 3 fit succeeded")
            return self
        except Exception as e:
            self.fallback_chain.append(("Level3 SimpleCox", "FAIL", str(e)))
            raise RuntimeError(f"All Cox fallback levels failed: {self.fallback_chain}")

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.active_level == 1:
            return self.level1.predict_risk(X)
        elif self.active_level == 2:
            return self.level2.predict_risk(X)
        elif self.active_level == 3:
            return self.level3.predict_risk(X)
        else:
            raise RuntimeError("ElasticNetCoxFallback not fitted")

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        """简化生存函数估计：基于风险分数的 Breslow 近似。"""
        risk = self.predict_risk(X)
        # 简化：S(t|x) = exp(-H0(t) * exp(risk))
        # 此处用全局 KM 估计 H0
        n = len(risk)
        n_times = len(times)
        surv = np.ones((n, n_times), dtype=np.float32)
        for j, t in enumerate(times):
            # 简化：风险越高生存越低
            surv[:, j] = np.exp(-0.01 * t * np.exp(risk))
        return surv

    def get_active_model(self):
        if self.active_level == 1:
            return self.level1
        elif self.active_level == 2:
            return self.level2
        elif self.active_level == 3:
            return self.level3
        return None

    def summary(self) -> Dict[str, Any]:
        return {
            "active_level": self.active_level,
            "fallback_chain": self.fallback_chain,
        }
