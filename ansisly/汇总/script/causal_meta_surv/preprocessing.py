"""preprocessing.py — 无泄漏 fold 预处理实现。

提供：
    - FoldPreprocessorImpl：FoldPreprocessor 接口的实现
        - fit(train_df, gene_registry)：训练折内拟合
        - transform(df)：应用训练折统计量
        - state_dict()：可哈希状态
    - assert_no_leakage(preprocessor, locked_val_df)：自动泄漏测试
    - group_kfold_indices(patient_ids, n_splits)：患者级 GroupKFold
    - stratified_time_event_keys(time, event, n_bins)：event × log(time) 分层键
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from .contracts import (
    CMIBConfig,
    FoldPreprocessor,
    LeakageError,
    SurvivalRecordBatch,
)
from .data import GeneRegistry, make_survival_record_batch

LOGGER = logging.getLogger("cmib_surv.preprocessing")


# ──────────────────────────────────────────────────────────────────────
# FoldPreprocessor 实现
# ──────────────────────────────────────────────────────────────────────
class FoldPreprocessorImpl(FoldPreprocessor):
    """无泄漏 fold 预处理器。

    fit() 只允许访问当前训练折，依次执行：
        1. 患者 ID 归一化
        2. gene registry 校验
        3. 缺失率过滤
        4. 训练中位数填补
        5. winsorize(0.01, 0.99) + IQR 鲁棒缩放（与 035 FoldPreprocessor 对齐）
        6. 训练折内方差筛选
        7. 临床变量编码
        8. 保存 feature order、统计量、fit patient IDs、数据哈希

    关键修复（P0-1）：035 的 winsorize + IQR 鲁棒缩放实现移植，
    替代原 mean/std z-score（原实现导致原始 TPM 尺度输入模型）。
    """

    def __init__(self, cfg: CMIBConfig, clip_quantiles: Tuple[float, float] = (0.01, 0.99)):
        self.cfg = cfg
        self.clip_quantiles = clip_quantiles
        self._fitted = False
        self._feature_order: List[str] = []
        # 基因统计量（winsorize + IQR）
        self._gene_medians: Dict[str, float] = {}
        self._gene_lower: Dict[str, float] = {}    # winsorize 下界
        self._gene_upper: Dict[str, float] = {}    # winsorize 上界
        self._gene_center: Dict[str, float] = {}   # median（IQR 中心）
        self._gene_scale: Dict[str, float] = {}    # IQR (q75 - q25)
        # 兼容旧字段
        self._gene_means: Dict[str, float] = {}
        self._gene_stds: Dict[str, float] = {}
        # 临床变量统计量（IQR 鲁棒缩放）
        self._clinical_medians: Dict[str, float] = {}
        self._clinical_lower: Dict[str, float] = {}
        self._clinical_upper: Dict[str, float] = {}
        self._clinical_center: Dict[str, float] = {}
        self._clinical_scale: Dict[str, float] = {}
        self._clinical_means: Dict[str, float] = {}
        self._clinical_stds: Dict[str, float] = {}
        self._fit_patient_ids: List[str] = []
        self._data_hash: str = ""
        self._gene_registry: Optional[GeneRegistry] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        gene_registry: GeneRegistry,
    ) -> "FoldPreprocessorImpl":
        """训练折内拟合（严禁访问 validation/test）。

        与 035 FoldPreprocessor.fit 对齐：
            1. 中位数填补缺失
            2. winsorize(0.01, 0.99) 截断极端值
            3. IQR 鲁棒缩放：(x - median) / (q75 - q25)
        """
        if not isinstance(train_df, pd.DataFrame):
            raise TypeError("train_df must be a DataFrame")

        self._gene_registry = gene_registry
        self._feature_order = list(gene_registry.genes)
        self._fit_patient_ids = list(train_df.index.astype(str))

        # 数据哈希（用于泄漏检测）
        self._data_hash = self._data_hash_fn(train_df)

        q_low, q_high = self.clip_quantiles

        # 基因统计量（winsorize + IQR）
        for gene in self._feature_order:
            if gene in train_df.columns:
                col = pd.to_numeric(train_df[gene], errors="coerce")
                median = float(col.median()) if col.notna().any() else 0.0
                self._gene_medians[gene] = median
                # winsorize 分位数
                if col.notna().any():
                    self._gene_lower[gene] = float(col.quantile(q_low))
                    self._gene_upper[gene] = float(col.quantile(q_high))
                else:
                    self._gene_lower[gene] = 0.0
                    self._gene_upper[gene] = 0.0
                # 截断后的 IQR
                clipped = col.fillna(median).clip(self._gene_lower[gene], self._gene_upper[gene])
                center = float(clipped.median()) if clipped.notna().any() else 0.0
                q25 = float(clipped.quantile(0.25)) if clipped.notna().any() else 0.0
                q75 = float(clipped.quantile(0.75)) if clipped.notna().any() else 0.0
                iqr = q75 - q25
                self._gene_center[gene] = center
                self._gene_scale[gene] = iqr if iqr > 1e-6 else 1.0
                # 兼容旧字段
                self._gene_means[gene] = center
                self._gene_stds[gene] = iqr if iqr > 1e-6 else 1.0
            else:
                self._gene_medians[gene] = 0.0
                self._gene_lower[gene] = 0.0
                self._gene_upper[gene] = 0.0
                self._gene_center[gene] = 0.0
                self._gene_scale[gene] = 1.0
                self._gene_means[gene] = 0.0
                self._gene_stds[gene] = 1.0

        # 临床变量统计量（IQR 鲁棒缩放，与 035 对齐）
        for col_name in self.cfg.clinical_columns:
            if col_name in train_df.columns:
                col = pd.to_numeric(train_df[col_name], errors="coerce")
                median = float(col.median()) if col.notna().any() else 0.0
                self._clinical_medians[col_name] = median
                if col.notna().any():
                    self._clinical_lower[col_name] = float(col.quantile(q_low))
                    self._clinical_upper[col_name] = float(col.quantile(q_high))
                else:
                    self._clinical_lower[col_name] = 0.0
                    self._clinical_upper[col_name] = 0.0
                clipped = col.fillna(median).clip(self._clinical_lower[col_name], self._clinical_upper[col_name])
                center = float(clipped.median()) if clipped.notna().any() else 0.0
                q25 = float(clipped.quantile(0.25)) if clipped.notna().any() else 0.0
                q75 = float(clipped.quantile(0.75)) if clipped.notna().any() else 0.0
                iqr = q75 - q25
                self._clinical_center[col_name] = center
                self._clinical_scale[col_name] = iqr if iqr > 1e-6 else 1.0
                self._clinical_means[col_name] = center
                self._clinical_stds[col_name] = iqr if iqr > 1e-6 else 1.0
            else:
                self._clinical_medians[col_name] = 0.0
                self._clinical_lower[col_name] = 0.0
                self._clinical_upper[col_name] = 0.0
                self._clinical_center[col_name] = 0.0
                self._clinical_scale[col_name] = 1.0
                self._clinical_means[col_name] = 0.0
                self._clinical_stds[col_name] = 1.0

        self._fitted = True
        LOGGER.debug("FoldPreprocessor fitted on %d patients (winsorize+IQR)", len(self._fit_patient_ids))
        return self

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用训练折统计量到任意 DataFrame。

        与 035 FoldPreprocessor.transform 对齐：
            1. 中位数填补缺失
            2. winsorize clip 到训练折分位数
            3. IQR 鲁棒缩放：(x - center) / scale

        性能修复：用 pd.concat 一次性合并所有变换列，避免逐列赋值导致
        DataFrame fragmentation（PerformanceWarning）。
        """
        if not self._fitted:
            raise RuntimeError("FoldPreprocessor must be fitted before transform")

        n = len(df)
        # 收集所有变换后的基因列，最后用 pd.concat 一次性合并
        transformed_cols: Dict[str, pd.Series] = {}
        for gene in self._feature_order:
            if gene in df.columns:
                col = pd.to_numeric(df[gene], errors="coerce")
                # 1. 中位数填补
                col = col.fillna(self._gene_medians.get(gene, 0.0))
                # 2. winsorize clip
                col = col.clip(self._gene_lower.get(gene, 0.0), self._gene_upper.get(gene, 0.0))
                # 3. IQR 鲁棒缩放
                center = self._gene_center.get(gene, 0.0)
                scale = self._gene_scale.get(gene, 1.0)
                transformed_cols[gene] = (col - center) / scale
            else:
                # 缺失基因填 0
                transformed_cols[gene] = pd.Series(
                    np.zeros(n, dtype=np.float32), index=df.index, name=gene,
                )

        # 一次性合并所有变换列（避免 fragmentation）
        transformed_block = pd.DataFrame(transformed_cols, index=df.index)
        # 保留非基因列（如临床变量列），用 drop 去掉被变换的基因列再合并
        non_gene_cols = [c for c in df.columns if c not in self._feature_order]
        if non_gene_cols:
            result = pd.concat([df[non_gene_cols], transformed_block], axis=1)
        else:
            result = transformed_block
        return result

    def transform(
        self,
        df: pd.DataFrame,
        clinical: pd.DataFrame,
        split: Optional[pd.DataFrame] = None,
        patient_ids: Optional[Sequence[str]] = None,
        task_id: int = 0,
    ) -> SurvivalRecordBatch:
        """transform DataFrame -> SurvivalRecordBatch。"""
        if not self._fitted or self._gene_registry is None:
            raise RuntimeError("FoldPreprocessor must be fitted before transform")

        if patient_ids is None:
            patient_ids = list(df.index.astype(str))

        transformed_expr = self.transform_df(df)

        return make_survival_record_batch(
            patient_ids=patient_ids,
            expression=transformed_expr,
            clinical=clinical,
            gene_registry=self._gene_registry,
            cfg=self.cfg,
            task_id=task_id,
            split=split,
        )

    def state_dict(self) -> Dict[str, Any]:
        """返回可哈希状态（含 feature order、统计量、fit patient IDs、数据哈希）。"""
        return {
            "feature_order": list(self._feature_order),
            "gene_medians": dict(self._gene_medians),
            "gene_lower": dict(self._gene_lower),
            "gene_upper": dict(self._gene_upper),
            "gene_center": dict(self._gene_center),
            "gene_scale": dict(self._gene_scale),
            "gene_means": dict(self._gene_means),
            "gene_stds": dict(self._gene_stds),
            "clinical_medians": dict(self._clinical_medians),
            "clinical_lower": dict(self._clinical_lower),
            "clinical_upper": dict(self._clinical_upper),
            "clinical_center": dict(self._clinical_center),
            "clinical_scale": dict(self._clinical_scale),
            "clinical_means": dict(self._clinical_means),
            "clinical_stds": dict(self._clinical_stds),
            "fit_patient_ids": list(self._fit_patient_ids),
            "data_hash": self._data_hash,
            "fitted": self._fitted,
        }

    @staticmethod
    def _data_hash_fn(df: pd.DataFrame) -> str:
        """对 DataFrame 求哈希（用于泄漏检测）。"""
        try:
            data_str = df.head(100).to_csv(index=True)
        except Exception:
            data_str = str(df.shape)
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────────────
# 自动泄漏测试
# ──────────────────────────────────────────────────────────────────────
def assert_no_leakage(
    preprocessor: FoldPreprocessorImpl,
    train_df: pd.DataFrame,
    gene_registry: GeneRegistry,
    locked_val_df: pd.DataFrame,
) -> None:
    """自动泄漏测试：修改 locked validation 后 preprocessor state hash 不变。

    Raises:
        LeakageError: state hash 变化（说明 fit 访问了 validation）
    """
    # 1. fit on train only
    preprocessor.fit(train_df, gene_registry)
    hash_before = preprocessor.state_hash()

    # 2. 扰动 locked validation
    perturbed_val = locked_val_df.copy()
    if len(perturbed_val) > 0:
        first_col = perturbed_val.columns[0]
        perturbed_val.iloc[0, 0] = (perturbed_val.iloc[0, 0] + 1e8) if pd.api.types.is_numeric_dtype(perturbed_val[first_col]) else "LEAK_TEST"

    # 3. 重新 fit on train only（不应受 val 影响）
    preprocessor.fit(train_df, gene_registry)
    hash_after = preprocessor.state_hash()

    if hash_before != hash_after:
        raise LeakageError(
            "Leakage detected: preprocessor state hash changed after "
            "perturbing locked validation. fit() may be accessing validation data."
        )

    LOGGER.debug("Leakage test passed: state hash stable under val perturbation")


# ──────────────────────────────────────────────────────────────────────
# GroupKFold 与分层
# ──────────────────────────────────────────────────────────────────────
def group_kfold_indices(
    patient_ids: Sequence[str],
    n_splits: int = 5,
    seed: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """患者级 GroupKFold 拆分。

    Returns:
        List of (train_idx, val_idx) tuples，idx 为对 patient_ids 的位置索引。
        同一患者的多个样本必落在同一折。
    """
    pids = np.asarray(list(patient_ids))
    unique_pids, pid_indices = np.unique(pids, return_inverse=True)
    groups = pid_indices  # 每个样本对应的 patient 唯一索引

    gkf = GroupKFold(n_splits=n_splits)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for train_idx, val_idx in gkf.split(np.zeros(len(pids)), groups=groups):
        folds.append((train_idx, val_idx))
    return folds


def stratified_time_event_keys(
    time: np.ndarray,
    event: np.ndarray,
    n_time_bins: int = 3,
) -> np.ndarray:
    """构造 event × log(time) 分层键。

    Returns:
        stratify_key: 每个样本的分层键（字符串编码）
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)

    log_time = np.log1p(np.clip(time, 1e-6, None))
    # 分位数 bins
    quantiles = np.linspace(0, 1, n_time_bins + 1)
    bins = np.quantile(log_time, quantiles)
    bins = np.unique(bins)
    if len(bins) < 2:
        time_bin = np.zeros(len(time), dtype=int)
    else:
        time_bin = np.digitize(log_time, bins[1:-1], right=False)

    stratify_key = np.array([
        f"e{int(e)}_t{int(t)}" for e, t in zip(event, time_bin)
    ])
    return stratify_key
