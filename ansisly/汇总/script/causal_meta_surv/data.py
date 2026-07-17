"""data.py — 数据加载、患者 ID 对齐、源癌种任务构造。

提供：
    - load_target_expression()：读取 CRC 目标表达矩阵（300 genes）
    - load_source_expression()：读取单个源癌种表达矩阵
    - load_clinical_endpoint()：读取临床终点 TSV
    - load_split_table()：读取 train/internal_validation 拆分表
    - align_patient_ids()：以 PATIENT_ID 为唯一主键对齐 X 与 survival（修复 035 标签错位 bug）
    - build_gene_registry(mode='strict'|'relaxed')：确定性面板构造（禁止 set 非确定顺序）
    - GeneFeatures02：读取 45 基因 + Lasso-Cox 系数 + 频率表
    - make_survival_record_batch()：构造 batch 并自动校验
    - SourceTask / build_source_tasks()：源癌种任务构造
    - audit_data(cfg)：Phase 0 数据审计
"""
from __future__ import annotations

import hashlib
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from shared_utils import normalize_patient_id  # noqa: E402

from .contracts import (
    CMIBConfig,
    InsufficientSourceTasksError,
    SOURCE_CANCERS,
    SurvivalRecordBatch,
    validate_batch,
)

LOGGER = logging.getLogger("cmib_surv.data")


# ──────────────────────────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────────────────────────
def _safe_read_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False)


def load_target_expression(path: str | Path) -> pd.DataFrame:
    """读取 CRC 目标表达矩阵。

    Returns:
        DataFrame，index 为 PATIENT_ID（已 normalize），列为基因名。
    """
    df = _safe_read_tsv(path)
    # 第一列通常是 PATIENT_ID 或 sample_id
    id_col = df.columns[0]
    df = df.rename(columns={id_col: "PATIENT_ID"})
    df["PATIENT_ID"] = df["PATIENT_ID"].map(normalize_patient_id)
    df = df.dropna(subset=["PATIENT_ID"]).set_index("PATIENT_ID")
    # 排除临床元数据列
    return df


def load_source_expression(cancer: str, source_dir: str | Path) -> pd.DataFrame:
    """读取单个源癌种表达矩阵。

    Args:
        cancer: 癌种代号（如 STAD）
        source_dir: 源数据目录

    Returns:
        DataFrame，index 为 PATIENT_ID（已 normalize），列为基因名。
    """
    path = Path(source_dir) / f"{cancer}_gene_expression.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Source expression not found: {path}")
    df = _safe_read_tsv(path)
    id_col = df.columns[0]
    df = df.rename(columns={id_col: "PATIENT_ID"})
    df["PATIENT_ID"] = df["PATIENT_ID"].map(normalize_patient_id)
    df = df.dropna(subset=["PATIENT_ID"]).set_index("PATIENT_ID")
    return df


def load_source_clinical(cancer: str, source_dir: str | Path) -> pd.DataFrame:
    """读取单个源癌种临床终点 TSV。

    源癌种临床文件格式：patient_id/PATIENT_ID, time_months, event。
    统一归一化为：index=PATIENT_ID（normalized），列含 OS_MONTHS, OS_EVENT。

    Args:
        cancer: 癌种代号（如 STAD）
        source_dir: 源数据目录

    Returns:
        DataFrame，index=PATIENT_ID（已 normalize），含 OS_MONTHS, OS_EVENT 列。
    """
    path = Path(source_dir) / f"{cancer}_os_clinical.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Source clinical not found: {path}")
    df = _safe_read_tsv(path)
    # 统一 patient_id 列名（部分文件用 PATIENT_ID，部分用 patient_id）
    id_col = "PATIENT_ID" if "PATIENT_ID" in df.columns else "patient_id"
    df = df.rename(columns={id_col: "PATIENT_ID"})
    # 统一生存终点列名
    if "time_months" in df.columns:
        df = df.rename(columns={"time_months": "OS_MONTHS"})
    if "event" in df.columns:
        df = df.rename(columns={"event": "OS_EVENT"})
    df["PATIENT_ID"] = df["PATIENT_ID"].map(normalize_patient_id)
    df = df.dropna(subset=["PATIENT_ID"]).set_index("PATIENT_ID")
    # 确保 OS_EVENT 为数值
    df["OS_EVENT"] = pd.to_numeric(df["OS_EVENT"], errors="coerce").fillna(0.0)
    df["OS_MONTHS"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    # 缺失时间用中位数填补
    if df["OS_MONTHS"].isna().any():
        med = df["OS_MONTHS"].median()
        df["OS_MONTHS"] = df["OS_MONTHS"].fillna(med if not pd.isna(med) else 12.0)
    # 确保 OS_MONTHS > 0
    df["OS_MONTHS"] = df["OS_MONTHS"].clip(lower=0.1)
    return df


def load_clinical_endpoint(path: str | Path) -> pd.DataFrame:
    """读取临床终点 TSV。

    Returns:
        DataFrame，含 PATIENT_ID, OS_MONTHS, OS_EVENT, AGE, SEX,
        AJCC_STAGE_ENCODED, ipcw_weight_os 等列。
    """
    df = _safe_read_tsv(path)
    df["PATIENT_ID"] = df["PATIENT_ID"].map(normalize_patient_id)
    df = df.dropna(subset=["PATIENT_ID"]).set_index("PATIENT_ID")
    return df


def load_split_table(path: str | Path) -> pd.DataFrame:
    """读取 train/internal_validation 拆分表。

    Returns:
        DataFrame，含 PATIENT_ID, split, event, time_months, stratify_key。
    """
    df = _safe_read_tsv(path)
    df["PATIENT_ID"] = df["PATIENT_ID"].map(normalize_patient_id)
    df = df.dropna(subset=["PATIENT_ID"]).set_index("PATIENT_ID")
    return df


# ──────────────────────────────────────────────────────────────────────
# 患者对齐
# ──────────────────────────────────────────────────────────────────────
def align_patient_ids(
    expression: pd.DataFrame,
    clinical: pd.DataFrame,
    split: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """以 PATIENT_ID 为唯一主键对齐 X 与 survival（修复 035 标签错位 bug）。

    Args:
        expression: 表达矩阵，index=PATIENT_ID
        clinical: 临床终点，index=PATIENT_ID
        split: 拆分表（可选），index=PATIENT_ID

    Returns:
        对齐后的 (expression, clinical, split)，三者 index 完全一致且顺序相同。
    """
    common = expression.index.intersection(clinical.index)
    if split is not None:
        common = common.intersection(split.index)
    if len(common) == 0:
        raise ValueError("No common PATIENT_ID across expression/clinical/split")

    # 关键：用 .loc 显式对齐，避免 035 中 ndarray 索引导致的错位
    expression = expression.loc[common].sort_index()
    clinical = clinical.loc[common].sort_index()
    if split is not None:
        split = split.loc[common].sort_index()

    # 校验对齐
    if not expression.index.equals(clinical.index):
        raise ValueError("Alignment failed: expression/clinical index mismatch")
    if split is not None and not expression.index.equals(split.index):
        raise ValueError("Alignment failed: expression/split index mismatch")

    return expression, clinical, split


# ──────────────────────────────────────────────────────────────────────
# 基因注册表（确定性，禁止 set 非确定顺序）
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GeneRegistry:
    """确定性基因面板。"""
    genes: Tuple[str, ...]            # 按字典序排序
    target_gene_count: int
    source_presence: Mapping[str, int]  # gene -> 出现的源癌种数
    source_cancers: Tuple[str, ...]
    mode: str = "strict"

    def to_dict(self) -> dict:
        return {
            "genes": list(self.genes),
            "n_genes": len(self.genes),
            "target_gene_count": self.target_gene_count,
            "source_presence": dict(self.source_presence),
            "source_cancers": list(self.source_cancers),
            "mode": self.mode,
        }


def build_gene_registry(
    target_expression: pd.DataFrame,
    cfg: CMIBConfig,
    source_expressions: Optional[Mapping[str, pd.DataFrame]] = None,
) -> GeneRegistry:
    """确定性构建跨癌种共享基因面板。

    Args:
        target_expression: 目标 CRC 表达矩阵
        cfg: 配置
        source_expressions: 源癌种表达矩阵字典（可选；若 None 则加载）

    Returns:
        GeneRegistry，genes 按字典序排序，禁止 set 非确定顺序。
    """
    target_genes = list(target_expression.columns)
    target_gene_set = {g: True for g in target_genes}  # dict 保持插入顺序

    if source_expressions is None:
        source_expressions = {}
        for cancer in cfg.source_cancers:
            try:
                source_expressions[cancer] = load_source_expression(cancer, cfg.source_dir)
            except FileNotFoundError:
                LOGGER.warning("Source cancer %s not found, skipped", cancer)

    if len(source_expressions) < 4:
        raise InsufficientSourceTasksError(
            f"Only {len(source_expressions)} source cancers available, "
            f"need >=4. COAD random split fallback is prohibited."
        )

    # 统计每个目标基因在源癌种中的出现数
    source_presence: Dict[str, int] = {}
    for gene in target_genes:
        count = 0
        for cancer_df in source_expressions.values():
            if gene in cancer_df.columns:
                count += 1
        source_presence[gene] = count

    # 确定性筛选（不使用 set 顺序）
    if cfg.gene_panel_mode == "strict":
        min_presence = len(source_expressions)
    elif cfg.gene_panel_mode == "relaxed":
        min_presence = max(6, cfg.min_source_presence)
    else:
        raise ValueError(f"Unknown gene_panel_mode: {cfg.gene_panel_mode}")

    # 先按 presence 降序，再按字典序，确保确定性
    selected = [
        g for g in target_genes
        if source_presence[g] >= min_presence
    ]

    # CRC 特异标记基因强制纳入面板（即使 strict 模式下未在所有源癌种中出现）
    # 这些基因是 CRC 分支 (CRCSpecificBranch) 所需的，不纳入会导致 CRC 分支失效
    crc_blocked = set(cfg.crc_branch.blocked_genes)
    for g in target_genes:
        if g in crc_blocked and g not in selected:
            selected.append(g)
            LOGGER.debug("CRC blocked gene '%s' force-included in panel (presence=%d/%d)",
                         g, source_presence.get(g, 0), len(source_expressions))

    selected_sorted = tuple(sorted(selected))  # 字典序

    return GeneRegistry(
        genes=selected_sorted,
        target_gene_count=len(target_genes),
        source_presence=source_presence,
        source_cancers=tuple(source_expressions.keys()),
        mode=cfg.gene_panel_mode,
    )


# ──────────────────────────────────────────────────────────────────────
# 02 基因特征（Lasso-Cox 系数 + 频率表）
# ──────────────────────────────────────────────────────────────────────
@dataclass
class GeneFeatures02:
    """02_gene_features 产物。"""
    gene_list: Tuple[str, ...]
    selection_frequency: Mapping[str, float]
    lasso_coef: Mapping[str, float]
    causal_priority: Optional[pd.DataFrame] = None

    @classmethod
    def load(cls, features_dir: str | Path) -> "GeneFeatures02":
        features_dir = Path(features_dir)
        gene_list_path = features_dir / "final_gene_list.pkl"
        freq_path = features_dir / "lasso_bootstrap_selection_frequency.tsv"
        coef_path = features_dir / "causal_priority_feature_table.tsv"

        if not gene_list_path.exists():
            raise FileNotFoundError(f"02 gene list not found: {gene_list_path}")

        with open(gene_list_path, "rb") as f:
            gene_list = tuple(pickle.load(f))

        freq_map: Dict[str, float] = {}
        if freq_path.exists():
            freq_df = _safe_read_tsv(freq_path)
            freq_col = "selection_frequency" if "selection_frequency" in freq_df.columns else freq_df.columns[1]
            feat_col = "feature" if "feature" in freq_df.columns else freq_df.columns[0]
            freq_map = dict(zip(freq_df[feat_col].astype(str), freq_df[freq_col].astype(float)))

        coef_map: Dict[str, float] = {}
        causal_df: Optional[pd.DataFrame] = None
        if coef_path.exists():
            causal_df = _safe_read_tsv(coef_path)
            feat_col = "feature" if "feature" in causal_df.columns else causal_df.columns[0]
            if "lasso_coef" in causal_df.columns:
                coef_map = dict(zip(causal_df[feat_col].astype(str), causal_df["lasso_coef"].astype(float)))

        return cls(
            gene_list=gene_list,
            selection_frequency=freq_map,
            lasso_coef=coef_map,
            causal_priority=causal_df,
        )

    def high_freq_genes(self, top_k: int = 2) -> Tuple[str, ...]:
        """取频率最高的 top_k 基因（用于 CRC 特异分支额外基因）。"""
        sorted_genes = sorted(
            self.gene_list,
            key=lambda g: self.selection_frequency.get(g, 0.0),
            reverse=True,
        )
        return tuple(sorted_genes[:top_k])


# ──────────────────────────────────────────────────────────────────────
# Batch 构造
# ──────────────────────────────────────────────────────────────────────
def make_survival_record_batch(
    patient_ids: Sequence[str],
    expression: pd.DataFrame,
    clinical: pd.DataFrame,
    gene_registry: GeneRegistry,
    cfg: CMIBConfig,
    task_id: int = 0,
    split: Optional[pd.DataFrame] = None,
    is_synthetic: bool = False,
    origin_patient_ids: Optional[Sequence[str]] = None,
    sample_weights: Optional[Sequence[float]] = None,
) -> SurvivalRecordBatch:
    """构造 SurvivalRecordBatch 并自动校验。

    Args:
        patient_ids: 患者 ID 列表
        expression: 表达矩阵（index=PATIENT_ID）
        clinical: 临床终点（index=PATIENT_ID）
        gene_registry: 基因面板
        cfg: 配置
        task_id: 任务 ID（0=CRC，1..N=源癌种）
        split: 拆分表（可选，提供 time/event）
        is_synthetic: 是否合成样本
        origin_patient_ids: 合成样本来源患者 ID（None 则与 patient_ids 相同）
        sample_weights: 样本权重（None 则全 1）

    Returns:
        SurvivalRecordBatch
    """
    pids = list(patient_ids)
    n = len(pids)

    # 表达矩阵对齐到 gene_registry 顺序（缺失基因填 0）
    x_gene_np = np.zeros((n, len(gene_registry.genes)), dtype=np.float32)
    for i, pid in enumerate(pids):
        if pid in expression.index:
            row = expression.loc[pid]
            for j, gene in enumerate(gene_registry.genes):
                if gene in row.index:
                    x_gene_np[i, j] = float(row[gene])

    # 临床变量
    clin_cols = list(cfg.clinical_columns)
    x_clin_np = np.zeros((n, len(clin_cols)), dtype=np.float32)
    clin_mask_np = np.zeros((n, len(clin_cols)), dtype=bool)
    for i, pid in enumerate(pids):
        if pid in clinical.index:
            row = clinical.loc[pid]
            for j, col in enumerate(clin_cols):
                if col in row.index and pd.notna(row[col]):
                    val = float(row[col])
                    x_clin_np[i, j] = val
                    clin_mask_np[i, j] = True

    # 生存终点
    if split is not None:
        time_np = np.zeros(n, dtype=np.float32)
        event_np = np.zeros(n, dtype=np.float32)
        for i, pid in enumerate(pids):
            if pid in split.index:
                time_np[i] = float(split.loc[pid, "time_months"])
                event_np[i] = float(split.loc[pid, "event"])
            else:
                # fallback 到 clinical
                if pid in clinical.index:
                    time_np[i] = float(clinical.loc[pid, "OS_MONTHS"])
                    event_np[i] = float(clinical.loc[pid, "OS_EVENT"])
                else:
                    raise ValueError(f"Patient {pid} not in split or clinical")
    else:
        time_np = clinical.loc[pids, "OS_MONTHS"].values.astype(np.float32)
        event_np = clinical.loc[pids, "OS_EVENT"].values.astype(np.float32)

    # 处理 NaN time
    time_np = np.where(np.isnan(time_np) | (time_np <= 0), 1.0, time_np)

    weights = np.ones(n, dtype=np.float32) if sample_weights is None else np.asarray(sample_weights, dtype=np.float32)
    origins = tuple(pids if origin_patient_ids is None else origin_patient_ids)

    batch = SurvivalRecordBatch(
        patient_id=tuple(pids),
        x_gene=torch.from_numpy(x_gene_np),
        x_clinical=torch.from_numpy(x_clin_np),
        clinical_mask=torch.from_numpy(clin_mask_np),
        task_id=torch.full((n,), task_id, dtype=torch.long),
        time=torch.from_numpy(time_np),
        event=torch.from_numpy(event_np),
        treatment=None,
        sample_weight=torch.from_numpy(weights),
        origin_patient_id=origins,
        is_synthetic=torch.full((n,), is_synthetic, dtype=torch.bool),
    )
    validate_batch(batch)
    return batch


# ──────────────────────────────────────────────────────────────────────
# 源癌种任务
# ──────────────────────────────────────────────────────────────────────
@dataclass
class SourceTask:
    """源癌种任务。

    P0-5 修复：新增 preprocessor 字段，Reptile 元学习内循环用预处理后的表达值。
    与 035 对齐：每个源癌种任务在构造时拟合自己的 FoldPreprocessor。
    """
    cancer: str
    task_id: int
    expression: pd.DataFrame
    clinical: pd.DataFrame
    gene_registry: GeneRegistry
    patient_ids: Tuple[str, ...]
    n_events: int
    preprocessor: Optional[Any] = None  # FoldPreprocessorImpl，延迟导入避免循环依赖

    def make_batch(self, cfg: CMIBConfig, patient_ids: Optional[Sequence[str]] = None) -> SurvivalRecordBatch:
        pids = patient_ids if patient_ids is not None else self.patient_ids
        # P0-5 修复：若 preprocessor 已拟合，用 transform（winsorize+IQR）；
        # 否则 fallback 到原始 make_survival_record_batch（兼容旧调用）
        if self.preprocessor is not None and getattr(self.preprocessor, "_fitted", False):
            # 取子集患者（若指定）
            expr_subset = self.expression.loc[list(pids)]
            return self.preprocessor.transform(
                df=expr_subset,
                clinical=self.clinical,
                split=None,
                patient_ids=list(pids),
                task_id=self.task_id,
            )
        return make_survival_record_batch(
            patient_ids=pids,
            expression=self.expression,
            clinical=self.clinical,
            gene_registry=self.gene_registry,
            cfg=cfg,
            task_id=self.task_id,
        )


def build_source_tasks(
    cfg: CMIBConfig,
    gene_registry: GeneRegistry,
    clinical: Optional[pd.DataFrame] = None,
    source_expressions: Optional[Mapping[str, pd.DataFrame]] = None,
) -> List[SourceTask]:
    """构造所有源癌种任务。

    每个源癌种使用自己的临床终点文件（{CANCER}_os_clinical.tsv），
    而非复用目标 CRC 临床终点（修复 Phase 3 患者ID对齐失败 bug）。

    Args:
        cfg: 配置
        gene_registry: 基因面板
        clinical: 目标临床终点（保留参数兼容，源癌种不再使用）
        source_expressions: 源癌种表达矩阵字典（可选；若 None 则加载）

    Returns:
        源癌种任务列表
    """
    if source_expressions is None:
        source_expressions = {
            cancer: load_source_expression(cancer, cfg.source_dir)
            for cancer in cfg.source_cancers
        }

    tasks: List[SourceTask] = []
    for idx, cancer in enumerate(cfg.source_cancers, start=1):
        if cancer not in source_expressions:
            continue
        expr_df = source_expressions[cancer]
        # 源癌种使用自己的临床终点文件（修复：不再复用目标 CRC clinical）
        try:
            source_clin = load_source_clinical(cancer, cfg.source_dir)
        except FileNotFoundError as exc:
            LOGGER.warning("Source %s clinical missing: %s, skipped", cancer, exc)
            continue

        # 以 PATIENT_ID 求交
        common_pids = expr_df.index.intersection(source_clin.index)
        if len(common_pids) < 20:
            LOGGER.warning(
                "Source %s has only %d common patients, skipped", cancer, len(common_pids)
            )
            continue

        task = SourceTask(
            cancer=cancer,
            task_id=idx,
            expression=expr_df,
            clinical=source_clin.loc[common_pids],
            gene_registry=gene_registry,
            patient_ids=tuple(common_pids.tolist()),
            n_events=int(source_clin.loc[common_pids, "OS_EVENT"].sum()),
        )
        # P0-5 修复：为每个源癌种任务拟合 FoldPreprocessor（winsorize+IQR），
        # 使 Reptile 元学习内循环用与 target 一致的预处理尺度
        try:
            from .preprocessing import FoldPreprocessorImpl
            task_preprocessor = FoldPreprocessorImpl(cfg)
            task_preprocessor.fit(expr_df.loc[common_pids], gene_registry)
            task.preprocessor = task_preprocessor
        except Exception as exc:
            LOGGER.warning(
                "Source %s preprocessor fit failed: %s, fallback to raw expression",
                cancer, exc,
            )
        tasks.append(task)
        LOGGER.info(
            "Source task %s built: %d patients, %d events (preprocessor=%s)",
            cancer, len(common_pids), task.n_events,
            "fitted" if task.preprocessor is not None else "raw",
        )

    if len(tasks) < 4:
        raise InsufficientSourceTasksError(
            f"Only {len(tasks)} source tasks built (need >=4). "
            f"COAD random split fallback is prohibited."
        )

    return tasks


# ──────────────────────────────────────────────────────────────────────
# Phase 0 数据审计
# ──────────────────────────────────────────────────────────────────────
def audit_data(cfg: CMIBConfig) -> Dict[str, Any]:
    """Phase 0 数据审计。

    Returns:
        审计报告字典，含样本数、事件数、基因面板覆盖、患者 ID 一致性。
    """
    LOGGER.info("Phase 0: data audit starting")

    target_expr = load_target_expression(cfg.target_expression_path)
    clinical = load_clinical_endpoint(cfg.target_clinical_path)
    split = load_split_table(cfg.split_file)

    target_expr, clinical, split = align_patient_ids(target_expr, clinical, split)

    # 拆分统计
    split_counts = split["split"].value_counts().to_dict()
    train_pids = split[split["split"] == "train"].index.tolist()
    val_pids = split[split["split"] == "internal_validation"].index.tolist()

    # 患者隔离校验
    train_set = set(train_pids)
    val_set = set(val_pids)
    overlap = train_set & val_set
    if overlap:
        from .contracts import LeakageError
        raise LeakageError(
            f"Train/Validation patient overlap: {len(overlap)} patients"
        )

    # 基因面板
    source_expressions = {}
    for cancer in cfg.source_cancers:
        try:
            source_expressions[cancer] = load_source_expression(cancer, cfg.source_dir)
        except FileNotFoundError:
            LOGGER.warning("Source %s missing", cancer)

    gene_registry = build_gene_registry(target_expr, cfg, source_expressions)

    # 02 基因特征
    try:
        gene_features_02 = GeneFeatures02.load(cfg.features_02_dir)
        n_genes_02 = len(gene_features_02.gene_list)
    except FileNotFoundError:
        gene_features_02 = None
        n_genes_02 = 0

    # 5 个阻断基因在目标端的覆盖
    blocked_genes = cfg.crc_branch.blocked_genes
    blocked_coverage = {
        g: (g in target_expr.columns) for g in blocked_genes
    }

    # 事件数
    n_train_events = int(split.loc[train_pids, "event"].sum()) if "event" in split.columns else int(clinical.loc[train_pids, "OS_EVENT"].sum())
    n_val_events = int(split.loc[val_pids, "event"].sum()) if "event" in split.columns else int(clinical.loc[val_pids, "OS_EVENT"].sum())

    audit = {
        "target_n_patients": int(len(target_expr)),
        "target_n_genes": int(target_expr.shape[1]),
        "split_counts": split_counts,
        "n_train": len(train_pids),
        "n_validation": len(val_pids),
        "n_train_events": n_train_events,
        "n_validation_events": n_val_events,
        "n_source_cancers": len(source_expressions),
        "gene_panel_mode": cfg.gene_panel_mode,
        "gene_panel_size": len(gene_registry.genes),
        "n_genes_02": n_genes_02,
        "blocked_genes_coverage": blocked_coverage,
        "gene_registry_first10": list(gene_registry.genes[:10]),
        "patient_overlap_train_val": 0,
        "status": "PASS",
    }

    # Go/No-Go 检查
    issues = []
    if len(source_expressions) < 4:
        issues.append(f"insufficient source cancers: {len(source_expressions)} < 4")
    if n_train_events < 30:
        issues.append(f"train events {n_train_events} < 30")
    if len(gene_registry.genes) < 64:
        issues.append(f"gene panel {len(gene_registry.genes)} < 64")

    if issues:
        audit["status"] = "FAIL"
        audit["issues"] = issues

    LOGGER.info("Phase 0 audit: %s", audit["status"])
    return audit
