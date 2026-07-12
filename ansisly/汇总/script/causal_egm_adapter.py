"""causal_egm_adapter.py — CausalEGM 高维降维适配器

将高维基因表达矩阵通过 CausalEGM 编码-生成模型降维到低维因果潜在表征 Z，
作为下游 LASSO-Cox / DeepSurv 的输入特征。

工作模式:
    - latent_feature (推荐): 仅用 encoder 提取 Z，不涉及生存结局
    - rmst_ate: 将 RMST 作为连续结局，直接估计 ATE

CausalEGM 原始 API 详细文档:

    1. 模型初始化:
        from causalegm import CausalEGM
        model = CausalEGM(
            latent_dim: int = 15,        # 潜在变量 Z 的维度（推荐 10-20）
            epochs: int = 200,           # 最大训练轮数
            learning_rate: float = 1e-3, # 学习率（注意参数名为 learning_rate）
            batch_size: int = 64,        # Mini-batch 大小
            flow_steps: int = 3,         # Normalizing Flow 层数
            random_seed: int = 42,       # 随机种子（注意参数名为 random_seed）
        )

    2. 模型训练（sklearn 风格）:
        model.fit(X, t=treatment, y=outcome)
        # X: np.ndarray (n_samples, p_features) — 协变量/混杂因素
        # t: np.ndarray (n_samples,) — 二值处理变量 (0/1)，关键字参数
        # y: np.ndarray (n_samples,) — 连续结局变量，关键字参数

    3. 潜在结局预测:
        model.predict(X, t=1)  # → np.ndarray (n,)  处理组潜在结局 Y(1)
        model.predict(X, t=0)  # → np.ndarray (n,)  对照组潜在结局 Y(0)
        # t 参数可为标量（所有样本相同处理）或数组（逐样本指定）
        # 支持两种调用方式:
        #   model.predict(X, t=1)         — 关键字参数（推荐）
        #   model.predict(X, t=np.ones(n)) — 数组形式（逐样本指定）

    4. 潜在表征提取（用于降维）:
        model.get_latent_representation(X)  # → np.ndarray (n, latent_dim)
        # 备选: model.encode(X)
        # 返回编码器输出的低维因果表征 Z

    5. 因果效应估计:
        ate = model.get_ate(X)  # → float  平均处理效应 E[Y(1)-Y(0)]
        # 若 get_ate() 不存在，可通过 predict 手动计算:
        #   y1 = model.predict(X, t=1)   # → (n,)
        #   y0 = model.predict(X, t=0)   # → (n,)
        #   ate = np.mean(y1 - y0)

    6. 模型持久化:
        model.save("model.pt")       # 保存模型权重
        model = CausalEGM.load("model.pt")  # 加载模型

适配器接口与 CausalEGM 原始 API 的映射:
    - adapter.fit(X_df, treatment, outcome)
      → scaler + model.fit(X_scaled, t=treatment, y=outcome)
    - adapter.transform(X_df) → Z DataFrame (n, latent_dim)
      → scaler + model.get_latent_representation(X_scaled)  # 主要方法
    - adapter.predict(X_df, t_value=1) → Y(t) ndarray (n,)
      → scaler + model.predict(X_scaled, t=t_value)  # t 可为标量或数组
    - adapter.estimate_ate(X_df) → ATE dict {ate, ate_se, ci_low, ci_high}
      → model.get_ate(X_scaled)  # 若不存在则回退到 predict 差值均值
    - adapter.save(path) / CausalEGMAdapter.load(path)
      → model.save() + joblib 序列化 scaler/元信息

参考文献:
    Liu et al., CausalEGM, Nature Computational Science, 2024
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from shared_utils import setup_logger, RANDOM_SEED, STAGE_MAP_NUMERIC, encode_ajcc_stage
from config_causal import get_causal_config

# causalegm 可选依赖：未安装时仅在运行时报错
# 注意: pip 包名为 causalegm，但安装的目录名为 CausalEGM（大小写敏感）
try:
    from CausalEGM import CausalEGM
    _HAS_CAUSALEGM = True
except ImportError:
    CausalEGM = None  # type: ignore[assignment,misc]
    _HAS_CAUSALEGM = False

try:
    from CausalEGM import Trainer as CausalEGMTrainer
    _HAS_TRAINER = True
except ImportError:
    CausalEGMTrainer = None  # type: ignore[assignment,misc]
    _HAS_TRAINER = False

logger = setup_logger("causal_egm_adapter")


# ──────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────

def _check_causalegm_available() -> bool:
    """检查 causalegm 包是否可用。

    Returns:
        True 表示 causalegm 已安装且可导入，False 表示不可用。
    """
    return _HAS_CAUSALEGM


def _build_treatment_proxy(
    clinical: pd.DataFrame,
    sample_index: pd.Index,
) -> pd.Series:
    """基于 AJCC 分期构建二值 treatment 代理变量。

    CausalEGM 需要二值处理变量。在结直肠癌场景下，
    使用 AJCC 病理分期 ≥ III 期作为 treatment=1（晚期），
    I-II 期作为 treatment=0（早期）。

    Args:
        clinical: 临床 DataFrame，需包含 PATIENT_ID 和 AJCC_PATHOLOGIC_TUMOR_STAGE 列
        sample_index: 目标样本的 PATIENT_ID 索引（用于对齐）

    Returns:
        pd.Series，index=PATIENT_ID，值为 0/1

    Raises:
        ValueError: 若分期列缺失或无法映射
    """
    # 查找分期列 —— 优先已编码列，再尝试原始列名
    stage_col_candidates = [
        "AJCC_STAGE_ENCODED",           # 预处理已编码的 0-4 数值列（优先）
        "AJCC_PATHOLOGIC_TUMOR_STAGE",  # 原始列（可能为字符串或数值码）
        "AJCC_STAGING",
        "AJCC_PATHOLOGIC_STAGE",
        "STAGE",
        "AJCC_STAGE",
    ]
    stage_col = None
    for cand in stage_col_candidates:
        if cand in clinical.columns:
            stage_col = cand
            break

    if stage_col is None:
        raise ValueError(
            f"[CausalEGM] 无法在临床数据中找到 AJCC 分期列。"
            f"已尝试: {stage_col_candidates}，"
            f"可用列: {list(clinical.columns)[:30]}"
        )

    logger.info(f"[CausalEGM] 使用分期列: '{stage_col}' 构建 treatment 代理变量")

    # 数值型 + 字符串型双轨映射
    def _map_stage(val):
        if pd.isna(val):
            return np.nan
        # 数值型: 已经是 0-4 编码，直接使用
        if isinstance(val, (int, float, np.integer, np.floating)):
            v = float(val)
            if 0 <= v <= 4:
                return v
        # 字符串型: 尝试匹配 STAGE_MAP_NUMERIC
        text = str(val).strip().upper()
        if text in STAGE_MAP_NUMERIC:
            return STAGE_MAP_NUMERIC[text]
        cleaned = text.replace("STAGE ", "").strip()
        if cleaned in STAGE_MAP_NUMERIC:
            return STAGE_MAP_NUMERIC[cleaned]
        # 尝试数值转换
        try:
            v = float(text)
            if 0 <= v <= 4:
                return v
        except (ValueError, TypeError):
            pass
        return np.nan

    # 构建 patient_id → stage_numeric 映射
    if "PATIENT_ID" in clinical.columns:
        stage_series = clinical.set_index("PATIENT_ID")[stage_col].apply(_map_stage)
    else:
        # 假设 index 就是 PATIENT_ID
        stage_series = clinical[stage_col].apply(_map_stage)
        stage_series.index = clinical.index

    # 对齐到 sample_index
    stage_aligned = stage_series.reindex(sample_index)

    # 二值化: >= 3 (Stage III/IV) → 1, 其余 → 0
    treatment = stage_aligned.apply(
        lambda v: 1 if (not pd.isna(v) and v >= 3) else (0 if not pd.isna(v) else np.nan)
    )

    # 对无法映射的样本填充中位数
    n_missing = treatment.isna().sum()
    if n_missing > 0:
        median_val = treatment.dropna().median()
        if pd.isna(median_val):
            median_val = 0  # 全部缺失时默认 0
        logger.warning(
            f"[CausalEGM] {n_missing} 个样本分期缺失，填充中位数={median_val:.1f}"
        )
        treatment = treatment.fillna(median_val)

    treatment = treatment.astype(int)

    # 验证：确保 treatment=0 和 treatment=1 各有足够样本
    n_treat = (treatment == 1).sum()
    n_ctrl = (treatment == 0).sum()
    logger.info(
        f"[CausalEGM] treatment 分布: control={n_ctrl}, treatment={n_treat}, "
        f"total={len(treatment)}"
    )
    if n_treat < 10 or n_ctrl < 10:
        raise ValueError(
            f"[CausalEGM] treatment 分组样本不足: control={n_ctrl}, treatment={n_treat}。"
            f"各组至少需要 10 个样本。"
        )

    return treatment


# ──────────────────────────────────────────────────────────────────────
# CausalEGMAdapter 类
# ──────────────────────────────────────────────────────────────────────

class CausalEGMAdapter:
    """CausalEGM 适配封装：高维基因表达 → 低维因果潜在表征。

    内部封装 causalegm.CausalEGM 的 sklearn 风格 API:
        model.fit(X, t=treatment, y=outcome)  — X: 协变量, t: 处理, y: 结局
        model.predict(X, t=1/0)               — 返回潜在结局估计

    本适配器在此基础上提供:
        - transform(X) → Z: 编码器潜在表征，作为下游 LASSO-Cox / DeepSurv 特征
        - predict(X, t_value) → 潜在结局 Y(t) 的估计
        - estimate_ate() → ATE 因果效应估计（仅 rmst_ate 模式）

    支持两种工作模式:
      - latent_feature: 仅使用 encoder 提取 Z，作为下游特征（推荐）
      - rmst_ate: 将 RMST 作为连续结局，直接估计因果效应 ATE

    Attributes:
        latent_dim: 潜在空间维度
        mode: 工作模式
        _scaler: 输入特征标准化器
        _model: CausalEGM 模型实例（sklearn 风格: fit/predict）
        _fitted: 是否已训练
    """

    def __init__(
        self,
        latent_dim: Optional[int] = None,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None,
        flow_steps: Optional[int] = None,
        mode: Optional[str] = None,
        rmst_tau: Optional[float] = None,
        random_state: int = RANDOM_SEED,
    ):
        """初始化 CausalEGMAdapter。

        Args:
            latent_dim: 潜在空间维度，推荐 10-20，小样本用 10-15
            epochs: 最大训练轮数
            lr: 学习率
            batch_size: Mini-batch 大小
            flow_steps: Normalizing Flow 层数
            mode: 工作模式，'latent_feature' 或 'rmst_ate'
            rmst_tau: RMST 截断时间（月），仅在 mode='rmst_ate' 时有效
            random_state: 随机种子

        Raises:
            ValueError: mode 不在合法值域内
        """
        # 从配置读取默认值，参数为 None 时使用配置值
        cfg = get_causal_config()
        latent_dim = latent_dim if latent_dim is not None else cfg.latent_dim
        epochs = epochs if epochs is not None else cfg.epochs
        lr = lr if lr is not None else cfg.learning_rate
        batch_size = batch_size if batch_size is not None else cfg.batch_size
        flow_steps = flow_steps if flow_steps is not None else cfg.flow_steps
        mode = mode if mode is not None else cfg.mode

        if mode not in ("latent_feature", "rmst_ate"):
            raise ValueError(
                f"[CausalEGM] mode 应为 'latent_feature' 或 'rmst_ate'，实际: '{mode}'"
            )
        if latent_dim < 1:
            raise ValueError(
                f"[CausalEGM] latent_dim 应为正整数，实际: {latent_dim}"
            )

        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.flow_steps = flow_steps
        self.mode = mode
        self.rmst_tau = rmst_tau
        self.random_state = random_state

        self._scaler: Optional[StandardScaler] = None
        self._model: Any = None
        self._fitted: bool = False
        self._ate_result: Optional[Dict[str, float]] = None
        self._feature_names: Optional[list] = None

        logger.info(
            f"[CausalEGM] Adapter 初始化: latent_dim={latent_dim}, epochs={epochs}, "
            f"lr={lr}, batch_size={batch_size}, flow_steps={flow_steps}, mode={mode}"
        )

    @classmethod
    def from_config(cls, **kwargs) -> "CausalEGMAdapter":
        """从 config_causal.CausalEGMConfig 创建 Adapter 实例。

        读取 OmegaConf 配置中的默认值，可通过 kwargs 覆盖个别参数。

        Args:
            **kwargs: 覆盖配置默认值的参数（如 latent_dim=20）

        Returns:
            CausalEGMAdapter 实例
        """
        cfg = get_causal_config()
        params = {
            "latent_dim": cfg.latent_dim,
            "epochs": cfg.epochs,
            "lr": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "flow_steps": cfg.flow_steps,
            "mode": cfg.mode,
        }
        params.update(kwargs)
        return cls(**params)

    # ── 输入验证 ────────────────────────────────────────────────────────

    @staticmethod
    def _validate_fit_inputs(
        X: pd.DataFrame,
        treatment: pd.Series,
        outcome: pd.Series,
        event: Optional[pd.Series],
    ) -> None:
        """验证 fit() 输入参数的合法性。

        Args:
            X: 基因表达矩阵
            treatment: 二值处理变量
            outcome: 结局变量
            event: 事件指示器（可选）

        Raises:
            TypeError: 输入类型不正确
            ValueError: 输入维度不匹配或值域异常
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"[CausalEGM] X 应为 pd.DataFrame，实际类型: {type(X).__name__}"
            )
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(
                f"[CausalEGM] X 不能为空矩阵，当前 shape={X.shape}"
            )
        if not isinstance(treatment, pd.Series):
            raise TypeError(
                f"[CausalEGM] treatment 应为 pd.Series，实际类型: {type(treatment).__name__}"
            )
        if not isinstance(outcome, pd.Series):
            raise TypeError(
                f"[CausalEGM] outcome 应为 pd.Series，实际类型: {type(outcome).__name__}"
            )
        # 维度一致性检查
        n = X.shape[0]
        if len(treatment) != n:
            raise ValueError(
                f"[CausalEGM] X ({n} 行) 与 treatment ({len(treatment)}) 样本数不一致"
            )
        if len(outcome) != n:
            raise ValueError(
                f"[CausalEGM] X ({n} 行) 与 outcome ({len(outcome)}) 样本数不一致"
            )
        if event is not None:
            if not isinstance(event, pd.Series):
                raise TypeError(
                    f"[CausalEGM] event 应为 pd.Series，实际类型: {type(event).__name__}"
                )
            if len(event) != n:
                raise ValueError(
                    f"[CausalEGM] X ({n} 行) 与 event ({len(event)}) 样本数不一致"
                )
        # treatment 应为二值
        t_unique = set(np.unique(treatment.dropna().astype(int)))
        if not t_unique.issubset({0, 1}):
            raise ValueError(
                f"[CausalEGM] treatment 应为二值 (0/1)，实际唯一值: {t_unique}"
            )
        # outcome 不应含负值（生存时间）
        if (outcome.dropna() < 0).any():
            logger.warning(
                "[CausalEGM] outcome 包含负值，请确认是否为合法的生存时间"
            )

    # ── fit ────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        treatment: pd.Series,
        outcome: pd.Series,
        event: Optional[pd.Series] = None,
    ) -> "CausalEGMAdapter":
        """训练 CausalEGM 模型。

        内部调用 CausalEGM 的 sklearn 风格 API:
            model.fit(X_scaled, t=treatment, y=outcome)

        Args:
            X: pd.DataFrame (n_samples, p_genes) 基因表达矩阵
            treatment: pd.Series (n_samples,) 二值处理变量 (0/1)
                       对应 CausalEGM.fit(X, t=, y=) 中的 t 参数
            outcome: pd.Series (n_samples,) 结局变量（生存时间，月）
                     对应 CausalEGM.fit(X, t=, y=) 中的 y 参数
            event: pd.Series 可选，事件指示器 (1=事件, 0=删失)

        Returns:
            self（支持链式调用）

        Raises:
            ImportError: causalegm 包未安装
            TypeError: 输入类型不正确
            ValueError: 输入维度不匹配
            RuntimeError: 训练过程中发生异常
        """
        # 1. 检查 causalegm 可用性
        if not _check_causalegm_available():
            raise ImportError(
                "[CausalEGM] causalegm 包未安装。"
                "请通过 pip install causalegm 安装。"
            )

        # 使用模块顶层已导入的 CausalEGM
        import CausalEGM as _causalegm_module

        # 输入验证
        self._validate_fit_inputs(X, treatment, outcome, event)

        logger.info(
            f"[CausalEGM] 开始训练: X={X.shape}, mode={self.mode}, "
            f"latent_dim={self.latent_dim}"
        )

        try:
            # 2. 标准化输入特征
            self._scaler = StandardScaler()
            X_np = self._scaler.fit_transform(X).astype(np.float32)
            self._feature_names = list(X.columns)

            # CausalEGM API: fit(X, t=处理, y=结局) — 关键字参数
            t_np = np.asarray(treatment, dtype=np.float32)
            y_np = np.asarray(outcome, dtype=np.float32)

            # 3. 若 mode=="rmst_ate" 且 event 非 None，计算 RMST 作为连续结局
            if self.mode == "rmst_ate" and event is not None:
                tau = self.rmst_tau
                if tau is None:
                    # 默认使用 outcome 的中位数作为 tau
                    tau = float(np.median(y_np))
                    logger.info(
                        f"[CausalEGM] rmst_tau 未指定，使用 outcome 中位数={tau:.2f}"
                    )
                y_np = self._compute_rmst(y_np, np.asarray(event, dtype=int), tau)
                logger.info(f"[CausalEGM] 使用 RMST(tau={tau:.2f}) 作为连续结局")

            # 4. 构建 CausalEGM 模型并训练（sklearn 风格 API）
            np.random.seed(self.random_state)

            model = self._build_and_train_model(_causalegm_module, X_np, y_np, t_np)
            self._model = model
            self._train_X_np = X_np  # 保存训练数据供 estimate_ate 使用

            self._fitted = True
            logger.info("[CausalEGM] 训练完成")

        except ImportError:
            raise
        except Exception as e:
            logger.warning(
                f"[CausalEGM] 训练失败: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )
            raise

        return self

    def _build_and_train_model(
        self,
        causalegm: Any,
        X_np: np.ndarray,
        y_np: np.ndarray,
        t_np: np.ndarray,
    ) -> Any:
        """构建并训练 CausalEGM 模型（v0.4.0 params dict API）。

        CausalEGM v0.4.0 使用 params dict + train(data=[X,Y,V]) 接口：
            params: dict（含 z_dims, v_dim, lr 等必需键）
            train(data=[X_treatment, Y_outcome, V_covariates], n_iter=...)

        Args:
            causalegm: 已导入的 CausalEGM 模块
            X_np: 标准化后的协变量矩阵 (n, p)，float32
            y_np: 结局数组 (n,)，float32
            t_np: 处理数组 (n,)，float32（二值 0/1）

        Returns:
            训练好的 CausalEGM 模型实例
        """
        import tempfile

        _n_samples, v_dim = X_np.shape

        # 潜在空间维度拆分: [z_shared, z_treat, z_outcome, z_noise]
        z_shared = min(self.latent_dim, max(3, v_dim // 10))
        z_treat = max(1, z_shared // 2)
        z_outcome = max(1, z_shared // 2)
        z_noise = max(1, z_shared // 3)
        z_dims = [z_shared, z_treat, z_outcome, z_noise]

        # 构建 params dict（CausalEGM v0.4.0 必需）
        tmp_dir = tempfile.mkdtemp(prefix="causalegm_")
        params = {
            "z_dims": z_dims,
            "v_dim": v_dim,
            "g_units": [64, 64],      # 生成器隐层（list 格式）
            "e_units": [64, 64],      # 编码器隐层（list 格式）
            "dz_units": [32],          # Z 判别器隐层
            "dv_units": [32],          # V 判别器隐层
            "f_units": [32],           # 结局预测网络隐层
            "h_units": [32],           # 处理预测网络隐层
            "lr": self.lr,
            "output_dir": tmp_dir,
            "dataset": "tcga_coad",
            "binary_treatment": True,
            "save_model": False,
            "save_res": False,
            "alpha": 1.0,
            "beta": 1.0,
            "gamma": 10.0,
            "use_v_gan": 1,
            "use_z_rec": 1,
            "g_d_freq": 1,
        }

        model = causalegm.CausalEGM(
            params=params,
            random_seed=self.random_state,
        )

        # 数据格式: [X=treatment, Y=outcome, V=covariates]
        X_treat = t_np.reshape(-1, 1).astype(np.float32)
        Y_outcome = y_np.reshape(-1, 1).astype(np.float32)
        V_covariates = X_np.astype(np.float32)

        # 训练（适配小样本: 限制迭代次数 + 关闭 verbose）
        n_iter = min(self.epochs * 10, 5000)
        model.train(
            data=[X_treat, Y_outcome, V_covariates],
            n_iter=n_iter,
            batch_size=self.batch_size,
            verbose=0,
            save_format="npy",
        )

        logger.info(
            f"[CausalEGM] CausalEGM(params=z_dims={z_dims}, v_dim={v_dim})."
            f"train(data=[X,Y,V], n_iter={n_iter}) 训练成功"
        )
        return model

    @staticmethod
    def _compute_rmst(
        times: np.ndarray,
        events: np.ndarray,
        tau: float,
    ) -> np.ndarray:
        """计算每个样本的 RMST（Restricted Mean Survival Time）。

        使用简单近似: RMST_i ≈ min(T_i, tau)
        这是删失数据下 RMST 的保守估计。

        Args:
            times: 生存时间数组
            events: 事件指示器 (1=事件, 0=删失)
            tau: 截断时间

        Returns:
            RMST 数组
        """
        return np.minimum(times, tau)

    # ── predict (潜在结局) ─────────────────────────────────────────────

    def predict(
        self,
        X: pd.DataFrame,
        t_value: Union[int, float] = 1,
    ) -> np.ndarray:
        """预测潜在结局 Y(t)。

        调用 CausalEGM 的 predict(X, t) 接口，返回指定处理条件下的
        潜在结局估计值。

        注意: 这与 transform() 不同。predict 返回潜在结局 Y(t)，
        而 transform 返回编码器潜在表征 Z。

        Args:
            X: pd.DataFrame (n_samples, p_genes) 基因表达矩阵
            t_value: 处理变量值，0 或 1（默认 1，即 Y(1) 潜在结局）

        Returns:
            np.ndarray (n_samples,) 潜在结局估计值

        Raises:
            RuntimeError: 模型尚未训练或 predict 接口不可用
            ValueError: t_value 不在 {0, 1} 中
        """
        if not self._fitted:
            raise RuntimeError(
                "[CausalEGM] 模型尚未训练，请先调用 fit() 方法。"
            )
        if t_value not in (0, 1, 0.0, 1.0):
            raise ValueError(
                f"[CausalEGM] t_value 应为 0 或 1，实际: {t_value}"
            )

        try:
            X_np = self._scaler.transform(X).astype(np.float32)

            model = self._model
            # CausalEGM v0.4.0: predict(data_x=treatment, data_v=covariates)
            t_arr = np.full(len(X_np), float(t_value), dtype=np.float32).reshape(-1, 1)
            y_hat = model.predict(t_arr, X_np)

            if hasattr(y_hat, "numpy"):
                y_hat = y_hat.numpy()
            result = np.asarray(y_hat, dtype=np.float32).ravel()

            logger.info(
                f"[CausalEGM] predict(t={t_value}) 完成: "
                f"n={len(result)}, mean={result.mean():.4f}, std={result.std():.4f}"
            )
            return result

        except Exception as e:
            logger.warning(
                f"[CausalEGM] predict 失败: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )
            raise

    # ── transform (编码器潜在表征) ───────────────────────────────────────

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """将基因表达矩阵编码为低维因果潜在表征 Z。

        注意: 这与 CausalEGM 的 predict() 不同。transform 提取的是
        encoder 的潜在表征 Z，用于下游 LASSO-Cox / DeepSurv 特征；
        而 predict 返回的是潜在结局 Y(t) 的估计。

        Args:
            X: pd.DataFrame (n_samples, p_genes) 基因表达矩阵

        Returns:
            pd.DataFrame (n_samples, latent_dim)，列为 Z_0, Z_1, ...

        Raises:
            RuntimeError: 模型尚未训练
        """
        if not self._fitted:
            raise RuntimeError(
                "[CausalEGM] 模型尚未训练，请先调用 fit() 方法。"
            )

        try:
            # 用保存的 scaler 标准化
            X_np = self._scaler.transform(X).astype(np.float32)

            # 调用 model 的 encode 方法获取 Z
            Z = self._encode(X_np)

            col_names = [f"Z_{i}" for i in range(Z.shape[1])]
            result = pd.DataFrame(Z, index=X.index, columns=col_names)

            logger.info(
                f"[CausalEGM] transform 完成: {X.shape} → {result.shape}"
            )
            return result

        except Exception as e:
            logger.warning(
                f"[CausalEGM] transform 失败: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )
            raise

    def _encode(self, X_np: np.ndarray) -> np.ndarray:
        """从 CausalEGM 的 e_net 编码器提取潜在表征 Z。

        CausalEGM v0.4.0: Z = e_net.predict(V), 取 z_shared 部分。

        Args:
            X_np: 标准化后的特征矩阵

        Returns:
            Z 矩阵 (n_samples, z_shared_dim)
        """
        model = self._model
        Z_full = model.e_net.predict(X_np, verbose=0)
        if hasattr(Z_full, "numpy"):
            Z_full = Z_full.numpy()
        Z_full = np.asarray(Z_full, dtype=np.float32)
        # 取 z_shared 部分（前 z_dims[0] 维）
        z_shared_dim = model.params["z_dims"][0]
        return Z_full[:, :z_shared_dim]

    # ── fit_transform ──────────────────────────────────────────────────

    def fit_transform(
        self,
        X: pd.DataFrame,
        treatment: pd.Series,
        outcome: pd.Series,
        event: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """训练模型并返回潜在表征 Z（等价于 fit + transform）。

        Args:
            X: pd.DataFrame (n_samples, p_genes) 基因表达矩阵
            treatment: pd.Series (n_samples,) 二值处理变量
            outcome: pd.Series (n_samples,) 结局变量
            event: pd.Series 可选，事件指示器

        Returns:
            pd.DataFrame (n_samples, latent_dim)，列为 Z_0, Z_1, ...
        """
        self.fit(X, treatment, outcome, event)
        return self.transform(X)

    # ── estimate_ate ───────────────────────────────────────────────────

    def estimate_ate(
        self,
        X: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """估计平均处理效应 (ATE)。

        仅在 mode='rmst_ate' 下可用。调用前需先完成 fit()。
        ATE = E[Y(1) - Y(0)]，即处理组和对照组的潜在结局均值之差。

        优先使用 CausalEGM 官方接口:
            model.get_ate(X)
        回退到手动计算:
            ATE ≈ mean(model.predict(X, t=1) - model.predict(X, t=0))

        Args:
            X: 可选，用于估计 ATE 的协变量矩阵。
               若为 None 则使用训练时的 X（需要训练数据仍在内存中）。

        Returns:
            dict 包含:
                - ate: 平均处理效应估计值
                - ate_se: ATE 的标准误
                - ci_low: 95% 置信区间下界
                - ci_high: 95% 置信区间上界

        Raises:
            RuntimeError: 模型未训练或模式不匹配
        """
        if not self._fitted:
            raise RuntimeError(
                "[CausalEGM] 模型尚未训练，请先调用 fit() 方法。"
            )
        if self.mode != "rmst_ate":
            raise RuntimeError(
                f"[CausalEGM] estimate_ate() 仅在 mode='rmst_ate' 下可用，"
                f"当前 mode='{self.mode}'。"
            )

        try:
            model = self._model

            # 准备 X_np
            if X is not None:
                X_np = self._scaler.transform(X).astype(np.float32)
            elif hasattr(self, "_train_X_np"):
                X_np = self._train_X_np
            else:
                raise RuntimeError(
                    "[CausalEGM] estimate_ate 需要提供 X 参数，"
                    "或在 fit 时保存训练数据。"
                )

            # 策略 1: CausalEGM 官方 get_ate()
            ate_result = None
            if hasattr(model, "get_ate"):
                try:
                    ate_result = model.get_ate(X_np)
                    logger.info("[CausalEGM] 使用 model.get_ate(X) 获取 ATE")
                except Exception as e:
                    logger.warning(
                        f"[CausalEGM] get_ate 失败: {e}，尝试手动计算"
                    )

            # 策略 2: 通过 predict 手动计算 ATE
            if ate_result is None and hasattr(model, "predict"):
                try:
                    y1 = model.predict(X_np, t=1)
                    y0 = model.predict(X_np, t=0)
                    if hasattr(y1, "numpy"):
                        y1 = y1.numpy()
                    if hasattr(y0, "numpy"):
                        y0 = y0.numpy()
                    y1 = np.asarray(y1, dtype=np.float64).ravel()
                    y0 = np.asarray(y0, dtype=np.float64).ravel()
                    individual_te = y1 - y0
                    ate = float(np.mean(individual_te))
                    ate_se = float(np.std(individual_te) / np.sqrt(len(individual_te)))
                    ate_result = {
                        "ate": ate,
                        "ate_se": ate_se,
                        "ci_low": ate - 1.96 * ate_se,
                        "ci_high": ate + 1.96 * ate_se,
                    }
                    logger.info(
                        "[CausalEGM] 通过 predict(t=1)-predict(t=0) 手动计算 ATE"
                    )
                except Exception as e:
                    logger.warning(f"[CausalEGM] 手动 ATE 计算失败: {e}")

            # 策略 3: 其他可能的接口
            if ate_result is None:
                if hasattr(model, "estimate_ate"):
                    ate_result = model.estimate_ate()
                elif hasattr(model, "ate"):
                    ate_result = model.ate

            if ate_result is None:
                raise RuntimeError(
                    "[CausalEGM] 模型不支持 ATE 估计。"
                    f"模型类型: {type(model).__name__}"
                )

            # 解析 ATE 结果（兼容不同返回格式）
            if isinstance(ate_result, dict) and "ate" in ate_result:
                result = ate_result  # 已经是标准格式
            else:
                result = self._parse_ate_result(ate_result)
            self._ate_result = result

            logger.info(
                f"[CausalEGM] ATE 估计: ate={result['ate']:.4f}, "
                f"se={result['ate_se']:.4f}, "
                f"95%CI=[{result['ci_low']:.4f}, {result['ci_high']:.4f}]"
            )
            return result

        except Exception as e:
            logger.warning(
                f"[CausalEGM] ATE 估计失败: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )
            raise

    @staticmethod
    def _parse_ate_result(ate_result: Any) -> Dict[str, float]:
        """解析 ATE 估计结果为标准字典格式。

        Args:
            ate_result: 模型返回的 ATE 结果，可能是 dict/tuple/namedtuple/标量

        Returns:
            标准化的 ATE 结果字典
        """
        if isinstance(ate_result, dict):
            ate = float(ate_result.get("ate", ate_result.get("ATE", 0.0)))
            se = float(ate_result.get("ate_se", ate_result.get("se", 0.0)))
            ci_low = float(ate_result.get("ci_low", ate_result.get("ci_lower", ate - 1.96 * se)))
            ci_high = float(ate_result.get("ci_high", ate_result.get("ci_upper", ate + 1.96 * se)))
        elif isinstance(ate_result, (tuple, list)) and len(ate_result) >= 2:
            ate = float(ate_result[0])
            se = float(ate_result[1])
            ci_low = float(ate_result[2]) if len(ate_result) > 2 else ate - 1.96 * se
            ci_high = float(ate_result[3]) if len(ate_result) > 3 else ate + 1.96 * se
        elif hasattr(ate_result, "ate"):
            ate = float(ate_result.ate)
            se = float(getattr(ate_result, "se", 0.0))
            ci_low = float(getattr(ate_result, "ci_low", ate - 1.96 * se))
            ci_high = float(getattr(ate_result, "ci_high", ate + 1.96 * se))
        else:
            ate = float(ate_result)
            se = 0.0
            ci_low = ate
            ci_high = ate

        return {
            "ate": ate,
            "ate_se": se,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    # ── 模型持久化 ────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """保存适配器到磁盘（含 scaler + CausalEGM 模型 + 元信息）。

        使用 CausalEGM 原生的 model.save() 保存模型权重，
        同时用 joblib 保存 scaler 和元信息。

        Args:
            path: 保存路径（目录），会创建子文件:
                  - {path}/causal_egm_model.pt  — CausalEGM 模型权重
                  - {path}/adapter_meta.joblib   — scaler + 元信息

        Raises:
            RuntimeError: 模型尚未训练
        """
        if not self._fitted:
            raise RuntimeError(
                "[CausalEGM] 模型尚未训练，无法保存。"
            )

        try:
            os.makedirs(path, exist_ok=True)

            # 保存 CausalEGM 模型（优先使用原生 save）
            model_path = os.path.join(path, "causal_egm_model.pt")
            if hasattr(self._model, "save"):
                self._model.save(model_path)
                logger.info(f"[CausalEGM] 模型权重已保存: {model_path}")
            else:
                # 回退: 使用 joblib 序列化整个模型
                import joblib
                joblib.dump(self._model, model_path)
                logger.info(f"[CausalEGM] 模型已用 joblib 保存: {model_path}")

            # 保存 scaler + 元信息
            import joblib
            meta_path = os.path.join(path, "adapter_meta.joblib")
            joblib.dump({
                "scaler": self._scaler,
                "feature_names": self._feature_names,
                "latent_dim": self.latent_dim,
                "mode": self.mode,
                "epochs": self.epochs,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "flow_steps": self.flow_steps,
                "rmst_tau": self.rmst_tau,
                "random_state": self.random_state,
            }, meta_path)
            logger.info(f"[CausalEGM] 适配器元信息已保存: {meta_path}")

        except Exception as e:
            logger.warning(
                f"[CausalEGM] 保存失败: {type(e).__name__}: {e}"
            )
            raise

    @classmethod
    def load(cls, path: str) -> "CausalEGMAdapter":
        """从磁盘加载已保存的适配器。

        Args:
            path: 保存路径（目录），需包含:
                  - {path}/causal_egm_model.pt
                  - {path}/adapter_meta.joblib

        Returns:
            恢复的 CausalEGMAdapter 实例

        Raises:
            FileNotFoundError: 必要文件缺失
            ImportError: causalegm 包未安装
        """
        import joblib

        meta_path = os.path.join(path, "adapter_meta.joblib")
        model_path = os.path.join(path, "causal_egm_model.pt")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"[CausalEGM] 元信息文件不存在: {meta_path}"
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[CausalEGM] 模型文件不存在: {model_path}"
            )

        meta = joblib.load(meta_path)

        adapter = cls(
            latent_dim=meta["latent_dim"],
            epochs=meta["epochs"],
            lr=meta["lr"],
            batch_size=meta["batch_size"],
            flow_steps=meta["flow_steps"],
            mode=meta["mode"],
            rmst_tau=meta.get("rmst_tau"),
            random_state=meta["random_state"],
        )
        adapter._scaler = meta["scaler"]
        adapter._feature_names = meta.get("feature_names")

        # 加载 CausalEGM 模型
        try:
            from CausalEGM import CausalEGM
            adapter._model = CausalEGM.load(model_path)
            logger.info(f"[CausalEGM] 模型已通过 CausalEGM.load() 加载: {model_path}")
        except (ImportError, AttributeError, Exception) as e:
            # 回退: 尝试 joblib 加载
            logger.warning(
                f"[CausalEGM] CausalEGM.load() 失败 ({e})，尝试 joblib 加载"
            )
            adapter._model = joblib.load(model_path)

        adapter._fitted = True
        logger.info(f"[CausalEGM] 适配器已从 {path} 恢复")
        return adapter

    # ── repr ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"CausalEGMAdapter(latent_dim={self.latent_dim}, mode='{self.mode}', "
            f"epochs={self.epochs}, {status})"
        )
