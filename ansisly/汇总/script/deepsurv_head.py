"""deepsurv_head.py — DeepSurv 深度生存预测头

基于 Cox 比例风险模型的部分似然损失，使用 MLP 网络学习非线性风险函数。
原生支持右删失数据，可接收 CausalEGM 潜在表征或原始基因特征作为输入。

核心组件:
    - CoxPHLoss: Cox 偏似然损失函数
    - DeepSurvNet: MLP 风险网络
    - DeepSurvHead: 高层封装（训练/预测/评估）

参考文献:
    Katzman et al., DeepSurv, BMC Medical Research Methodology, 2018
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from shared_utils import setup_logger, RANDOM_SEED

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = setup_logger("deepsurv_head")


# ──────────────────────────────────────────────────────────────────────
# Cox 偏似然损失
# ──────────────────────────────────────────────────────────────────────
if HAS_TORCH:

    class CoxPHLoss(nn.Module):
        """Cox 比例风险模型的部分似然损失（负偏似然）。

        数学形式:
            L = -1/N_events × Σ_{i: event_i=1} [ h(x_i) - log(Σ_{j∈R(t_i)} exp(h(x_j))) ]

        其中 R(t_i) 为时间 t_i 处的风险集（所有在 t_i 时刻仍处于风险的样本）。
        实现使用 logcumsumexp 技巧高效计算。
        """

        def forward(
            self,
            predictions: torch.Tensor,
            time: torch.Tensor,
            event: torch.Tensor,
        ) -> torch.Tensor:
            """计算 Cox 偏似然损失。

            Args:
                predictions: 模型输出的对数风险值，shape (N,) 或 (N, 1)。
                time: 生存时间，shape (N,)。
                event: 事件指示器（1=事件发生，0=删失），shape (N,)。

            Returns:
                标量损失值。
            """
            # 1. 按生存时间降序排列（从最长到最短）
            sorted_idx = torch.argsort(time, descending=True)
            predictions = predictions[sorted_idx].squeeze()
            event = event[sorted_idx]

            # 数值安全：clip predictions 防止 exp 溢出
            predictions = torch.clamp(predictions, min=-20.0, max=20.0)

            # 2. 计算风险集的 log-sum-exp（累积求和 = 风险集包含当前及之后所有样本）
            log_risk = torch.logcumsumexp(predictions, dim=0)

            # 3. 仅对发生事件的样本计算似然
            uncensored_likelihood = predictions - log_risk
            loss = -torch.sum(uncensored_likelihood * event) / (event.sum() + 1e-8)

            return loss


# ──────────────────────────────────────────────────────────────────────
# DeepSurv MLP 网络
# ──────────────────────────────────────────────────────────────────────
if HAS_TORCH:

    class DeepSurvNet(nn.Module):
        """DeepSurv 网络: MLP → 单节点输出（对数风险）。

        架构: [Linear → BatchNorm1d → ReLU → Dropout] × N_layers → Linear(1)

        注意: 当 batch_size < 8 时，BatchNorm 统计量不稳定，
        自动切换为 LayerNorm。
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dims: list = None,
            dropout: float = 0.4,
            use_batch_norm: bool = True,
        ):
            """初始化 DeepSurvNet。

            Args:
                input_dim: 输入特征维度。
                hidden_dims: 隐藏层维度列表，默认 [64, 32]。
                dropout: Dropout 概率。
                use_batch_norm: 是否使用 BatchNorm1d，否则使用 LayerNorm。
            """
            super().__init__()
            if hidden_dims is None:
                hidden_dims = [64, 32]

            layers = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(h_dim))
                else:
                    layers.append(nn.LayerNorm(h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, 1))  # 输出 log h(x)
            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """前向传播，返回对数风险值。"""
            return self.network(x)


# ──────────────────────────────────────────────────────────────────────
# DeepSurvHead 高层封装
# ──────────────────────────────────────────────────────────────────────
class DeepSurvHead:
    """DeepSurv 深度生存预测模型。

    基于 Cox 部分似然损失的 MLP 网络，原生支持右删失数据。
    可接收原始基因特征或 CausalEGM 潜在表征作为输入。

    训练策略:
        - 自动划分 80/20 训练/验证集
        - Early stopping 基于验证集 loss
        - ReduceLROnPlateau 自适应学习率衰减
        - 保存 best model state_dict，训练结束后恢复最优权重
    """

    def __init__(
        self,
        hidden_dims: list = None,
        dropout: float = 0.4,
        l2_reg: float = 1e-3,
        lr: float = 5e-4,
        epochs: int = 500,
        patience: int = 50,
        batch_size: int = 32,
        random_state: int = RANDOM_SEED,
    ):
        """初始化 DeepSurvHead。

        Args:
            hidden_dims: MLP 隐藏层维度列表，默认 [64, 32]。
            dropout: Dropout 概率。
            l2_reg: L2 正则化系数（Adam weight_decay）。
            lr: 初始学习率。
            epochs: 最大训练轮数。
            patience: Early stopping 容忍轮数。
            batch_size: Mini-batch 大小。
            random_state: 随机种子。

        Raises:
            ImportError: 当 PyTorch 未安装时。
        """
        if not HAS_TORCH:
            raise ImportError(
                "[DeepSurv] PyTorch 未安装，请安装 torch 后重试。"
            )

        self.hidden_dims = hidden_dims if hidden_dims is not None else [64, 32]
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.random_state = random_state

        # 将在 fit 中初始化
        self.model_ = None       # DeepSurvNet, set in fit
        self.scaler_ = None      # StandardScaler, set in fit
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_names_ = None  # list of str, set in fit

    # ── fit ───────────────────────────────────────────────────────────
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "DeepSurvHead":
        """训练 DeepSurv 模型。

        Args:
            X: 训练特征矩阵 (N, D)。
            time: 生存时间数组 (N,)。
            event: 事件指示器数组 (N,)，1=事件发生，0=删失。
            X_val: 验证特征矩阵，若为 None 则自动划分 20% 验证集。
            time_val: 验证生存时间。
            event_val: 验证事件指示器。

        Returns:
            self，允许链式调用。
        """
        try:
            return self._fit_impl(X, time, event, X_val, time_val, event_val)
        except Exception as exc:
            logger.warning("[DeepSurv] fit 失败: %s", exc)
            return self

    def _fit_impl(self, X, time, event, X_val, time_val, event_val):
        # -- 保存特征名 --
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]

        X_np = np.asarray(X, dtype=np.float32)
        time_np = np.asarray(time, dtype=np.float32)
        event_np = np.asarray(event, dtype=np.float32)

        # -- 标准化 --
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_np)

        # -- 划分验证集 --
        if X_val is None:
            (
                X_tr, X_vl,
                time_tr, time_vl,
                event_tr, event_vl,
            ) = train_test_split(
                X_scaled, time_np, event_np,
                test_size=0.2,
                stratify=event_np,
                random_state=self.random_state,
            )
        else:
            X_tr = X_scaled
            time_tr = time_np
            event_tr = event_np
            X_vl = self.scaler_.transform(np.asarray(X_val, dtype=np.float32))
            time_vl = np.asarray(time_val, dtype=np.float32)
            event_vl = np.asarray(event_val, dtype=np.float32)

        # -- 构建网络 --
        input_dim = X_tr.shape[1]
        use_bn = self.batch_size >= 8
        self.model_ = DeepSurvNet(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            use_batch_norm=use_bn,
        ).to(self.device_)

        logger.info(
            "[DeepSurv] 网络: input=%d, hidden=%s, dropout=%.2f, norm=%s, device=%s",
            input_dim, self.hidden_dims, self.dropout,
            "BatchNorm" if use_bn else "LayerNorm", self.device_,
        )

        # -- 优化器 & 调度器 --
        criterion = CoxPHLoss()
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.l2_reg,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(self.patience // 2, 1),
        )

        # -- Tensor 转换 --
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        time_tr_t = torch.tensor(time_tr, dtype=torch.float32)
        event_tr_t = torch.tensor(event_tr, dtype=torch.float32)
        X_vl_t = torch.tensor(X_vl, dtype=torch.float32)
        time_vl_t = torch.tensor(time_vl, dtype=torch.float32)
        event_vl_t = torch.tensor(event_vl, dtype=torch.float32)

        train_ds = TensorDataset(X_tr_t, time_tr_t, event_tr_t)
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
        )

        # -- 训练循环 --
        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, self.epochs + 1):
            # --- train ---
            self.model_.train()
            epoch_loss = 0.0
            n_batches = 0
            for xb, tb, eb in train_loader:
                xb = xb.to(self.device_)
                tb = tb.to(self.device_)
                eb = eb.to(self.device_)

                preds = self.model_(xb).squeeze()
                loss = criterion(preds, tb, eb)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            train_loss = epoch_loss / max(n_batches, 1)

            # --- val ---
            self.model_.eval()
            with torch.no_grad():
                val_preds = self.model_(X_vl_t.to(self.device_)).squeeze()
                val_loss = criterion(
                    val_preds, time_vl_t.to(self.device_), event_vl_t.to(self.device_),
                ).item()

            scheduler.step(val_loss)

            # --- early stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch % 50 == 0 or epoch == 1:
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "[DeepSurv] Epoch %4d | train_loss=%.4f | val_loss=%.4f | lr=%.2e",
                    epoch, train_loss, val_loss, current_lr,
                )

            if epochs_no_improve >= self.patience:
                logger.info(
                    "[DeepSurv] Early stopping at epoch %d (best val_loss=%.4f)",
                    epoch, best_val_loss,
                )
                break

        # -- 恢复最优权重 --
        if best_state is not None:
            self.model_.load_state_dict(best_state)
            logger.info("[DeepSurv] 已恢复最优模型权重 (best val_loss=%.4f)", best_val_loss)

        return self

    # ── predict_risk ──────────────────────────────────────────────────
    def predict_risk(self, X: pd.DataFrame | np.ndarray) -> pd.Series:
        """预测风险分数。

        Args:
            X: 特征矩阵 (N, D)，需与训练时维度一致。

        Returns:
            pd.Series，index 与 X 对齐，值为对数风险分数。
        """
        try:
            if self.model_ is None or self.scaler_ is None:
                logger.warning("[DeepSurv] 模型尚未训练，请先调用 fit()")
                return pd.Series(dtype=float)

            X_np = np.asarray(X, dtype=np.float32)
            X_scaled = self.scaler_.transform(X_np)
            X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device_)

            self.model_.eval()
            with torch.no_grad():
                risk = self.model_(X_t).squeeze().cpu().numpy()

            index = X.index if isinstance(X, pd.DataFrame) else None
            return pd.Series(risk, index=index, name="risk_score")

        except Exception as exc:
            logger.warning("[DeepSurv] predict_risk 失败: %s", exc)
            return pd.Series(dtype=float)

    # ── score ─────────────────────────────────────────────────────────
    def score(
        self,
        X: pd.DataFrame | np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
    ) -> float:
        """计算 Concordance Index (C-index)。

        Args:
            X: 特征矩阵。
            time: 生存时间。
            event: 事件指示器（1/0）。

        Returns:
            C-index 值 (float)，失败时返回 0.0。
        """
        try:
            from sksurv.metrics import concordance_index_censored

            risk_scores = self.predict_risk(X)
            if risk_scores.empty:
                return 0.0

            event_bool = np.asarray(event, dtype=bool)
            time_float = np.asarray(time, dtype=float)
            risk_arr = np.asarray(risk_scores, dtype=float)

            c_index = concordance_index_censored(event_bool, time_float, risk_arr)
            return float(c_index[0])

        except Exception as exc:
            logger.warning("[DeepSurv] score 失败: %s", exc)
            return 0.0

    # ── get_feature_importance ────────────────────────────────────────
    def get_feature_importance(
        self,
        X: pd.DataFrame | np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        n_repeats: int = 10,
    ) -> pd.DataFrame:
        """基于 permutation importance 的特征重要性评估。

        对每个特征随机打乱 n_repeats 次，计算 C-index 下降量。

        Args:
            X: 特征矩阵。
            time: 生存时间。
            event: 事件指示器。
            n_repeats: 每个特征的重复打乱次数。

        Returns:
            pd.DataFrame，包含 'feature' 和 'importance' 两列，
            按 importance 降序排列。失败时返回空 DataFrame。
        """
        try:
            baseline_c = self.score(X, time, event)
            X_np = np.asarray(X, dtype=np.float32).copy()
            n_features = X_np.shape[1]
            importances = []

            rng = np.random.RandomState(self.random_state)
            for j in range(n_features):
                c_drops = []
                col_orig = X_np[:, j].copy()
                for _ in range(n_repeats):
                    X_perm = X_np.copy()
                    rng.shuffle(X_perm[:, j])
                    # 构造临时 DataFrame 以保持 index
                    if isinstance(X, pd.DataFrame):
                        X_perm_df = pd.DataFrame(X_perm, columns=X.columns, index=X.index)
                    else:
                        X_perm_df = X_perm
                    c_perm = self.score(X_perm_df, time, event)
                    c_drops.append(baseline_c - c_perm)
                importances.append(float(np.mean(c_drops)))

            names = self.feature_names_ if self.feature_names_ else [
                f"feature_{i}" for i in range(n_features)
            ]
            result = pd.DataFrame({"feature": names, "importance": importances})
            result = result.sort_values("importance", ascending=False).reset_index(drop=True)
            return result

        except Exception as exc:
            logger.warning("[DeepSurv] get_feature_importance 失败: %s", exc)
            return pd.DataFrame(columns=["feature", "importance"])
