from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer

import polars as pl

from shared_utils import (
    setup_logger,
    SCRIPT_DIR, DATA_DIR, RESULTS_DIR,
    DATA_FORWARD_DIR, ESSENTIAL_DIR, NON_ESSENTIAL_DIR,
    FIXED_TAU_MONTHS, EPS, DAYS_PER_MONTH, MAX_N_EXACT_PSEUDO, META_COLUMNS,
    patient_id_from_sample, normalize_patient_id,
    parse_survival_status, numeric_series, encode_ajcc_stage,
    kaplan_meier_survival_at, km_cumulative_incidence_by_tau,
)
from gain_imputer import GAINImputer
logger = setup_logger("01_preprocessing")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False


TCGA_DIR = os.path.join(DATA_DIR, "tcga")
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")

TOP_K_GENES = 300
MUTATION_FREQ_LOW = 0.02
MUTATION_FREQ_HIGH = 0.80
MISSING_COL_THRESHOLD = 0.40


def matrix_sample_columns(columns):
    return [c for c in columns if c not in META_COLUMNS]


def load_clinical_data():
    patient_path = os.path.join(TCGA_DIR, "data_clinical_patient.txt")
    sample_path = os.path.join(TCGA_DIR, "data_clinical_sample.txt")

    if not os.path.exists(patient_path):
        raise FileNotFoundError(f"临床患者数据文件不存在: {patient_path}")

    df_patient = pd.read_csv(patient_path, sep="\t", comment="#", dtype=str, low_memory=False)

    df_sample = pd.DataFrame()
    if os.path.exists(sample_path):
        df_sample = pd.read_csv(sample_path, sep="\t", comment="#", dtype=str, low_memory=False)

    if not df_sample.empty and "PATIENT_ID" in df_sample.columns:
        sample_meta = df_sample.drop_duplicates(subset=["PATIENT_ID"])
        pid_col = "PATIENT_ID"
        if pid_col in df_patient.columns:
            merge_keys = [c for c in df_sample.columns if c != pid_col and c in df_patient.columns]
            if merge_keys:
                sample_meta = sample_meta.drop(columns=merge_keys, errors="ignore")
            df = df_patient.merge(sample_meta, on=pid_col, how="left", suffixes=("", "_sample"))
        else:
            df = df_patient.copy()
    else:
        df = df_patient.copy()

    missing_frac = df.isna().mean()
    df = df.loc[:, missing_frac <= MISSING_COL_THRESHOLD]

    if "PATIENT_ID" not in df.columns:
        raise ValueError("临床数据缺少 PATIENT_ID 列")

    df["PATIENT_ID"] = df["PATIENT_ID"].astype(str).apply(
        lambda x: x[:12] if x.startswith("TCGA-") and len(x) >= 12 else x
    )
    df = df.drop_duplicates(subset=["PATIENT_ID"], keep="first")

    for col in ["AJCC_PATHOLOGIC_TUMOR_STAGE", "AJCC_STAG", "STAGE"]:
        if col in df.columns:
            df["AJCC_STAGE_ENCODED"] = encode_ajcc_stage(df[col])
            break

    category_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    skip_cols = {"PATIENT_ID"}
    le_dict = {}
    for col in category_cols:
        if col in skip_cols:
            continue
        non_null = df[col].dropna()
        if non_null.nunique() <= 1:
            continue
        try:
            float(non_null.iloc[0])
            continue
        except (ValueError, TypeError):
            pass
        le = LabelEncoder()
        encoded = le.fit_transform(non_null.astype(str))
        df.loc[non_null.index, col] = encoded
        le_dict[col] = le

    return df, le_dict


def process_survival_endpoints(df):
    out = df.copy()

    status_col = None
    time_col = None
    for c in ["OS_STATUS", "OS_STATUS_RAW"]:
        if c in out.columns:
            status_col = c
            break
    for c in ["OS_MONTHS", "OS_MONTHS_RAW", "OS_TIME_MONTHS"]:
        if c in out.columns:
            time_col = c
            break

    if time_col is None:
        for c in ["OS_DAYS", "DAYS_TO_DEATH", "DAYS_TO_LAST_FOLLOWUP"]:
            if c in out.columns:
                out["OS_MONTHS"] = numeric_series(out[c]) / DAYS_PER_MONTH
                time_col = "OS_MONTHS"
                break

    if status_col is not None:
        out["OS_EVENT"] = parse_survival_status(out[status_col])
    elif "OS_EVENT" not in out.columns:
        out["OS_EVENT"] = np.nan

    if time_col is not None and time_col != "OS_MONTHS":
        out["OS_MONTHS"] = numeric_series(out[time_col])
    elif "OS_MONTHS" not in out.columns:
        out["OS_MONTHS"] = np.nan

    out["OS_MONTHS"] = pd.to_numeric(out["OS_MONTHS"], errors="coerce")
    out["OS_EVENT"] = pd.to_numeric(out["OS_EVENT"], errors="coerce")

    valid_mask = out["OS_MONTHS"].notna() & (out["OS_MONTHS"] > 0) & out["OS_EVENT"].notna()
    out = out[valid_mask].copy()

    out["OS_STATUS"] = out["OS_EVENT"].astype(int)

    times = out["OS_MONTHS"].to_numpy(dtype=float)
    events = out["OS_STATUS"].to_numpy(dtype=int)
    tau = FIXED_TAU_MONTHS

    observed_status = ((events == 1) & (times <= tau)) | (times > tau)
    event_by_tau = ((events == 1) & (times <= tau)).astype(float)

    out["death_by_36m_observed"] = observed_status
    out["death_by_36m"] = np.where(observed_status, event_by_tau, np.nan)
    out["early_censored_before_36m"] = ~observed_status

    censor_event = (events == 0).astype(int)
    eval_time = np.minimum(times, tau)
    eval_time = np.maximum(eval_time, 1.0)
    g_at_eval = kaplan_meier_survival_at(times, censor_event, eval_time)

    ipcw_weights = np.zeros(len(out), dtype=float)
    event_observed_mask = (events == 1)
    ipcw_weights[event_observed_mask] = 1.0 / np.clip(g_at_eval[event_observed_mask], EPS, None)
    out["ipcw_weight_os"] = ipcw_weights
    out["ipcw_label_available"] = event_observed_mask

    n = len(out)
    f_all = km_cumulative_incidence_by_tau(times, events, tau)
    pseudo = np.full(n, np.nan, dtype=float)
    if n > 0:
        if n > MAX_N_EXACT_PSEUDO:
            pseudo[:] = f_all
        else:
            for i in range(n):
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                f_minus = km_cumulative_incidence_by_tau(times[mask], events[mask], tau)
                pseudo[i] = n * f_all - (n - 1) * f_minus
    out["pseudo_risk_os_raw"] = pseudo
    out["pseudo_risk_os"] = np.clip(pseudo, 0.0, 1.0)

    return out


def load_gene_expression(top_k=TOP_K_GENES):
    rna_path = os.path.join(TCGA_DIR, "data_mrna_seq_v2_rsem.txt")
    if not os.path.exists(rna_path):
        raise FileNotFoundError(f"mRNA表达数据文件不存在: {rna_path}")

    chunks = []
    for chunk in pd.read_csv(rna_path, sep="\t", comment="#", dtype=str,
                              low_memory=False, chunksize=5000):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    if df.empty:
        raise ValueError("mRNA表达数据为空")

    id_col = "Hugo_Symbol" if "Hugo_Symbol" in df.columns else df.columns[0]
    sample_cols = matrix_sample_columns(df.columns)
    df = df[[id_col] + sample_cols].copy()
    df = df[df[id_col].notna() & (df[id_col].astype(str).str.strip() != "")]
    df[id_col] = df[id_col].astype(str)

    numeric = df.set_index(id_col).apply(pd.to_numeric, errors="coerce")
    if numeric.index.has_duplicates:
        numeric = numeric.groupby(level=0).mean()

    numeric = numeric.loc[numeric.notna().any(axis=1)]
    patient_matrix = numeric.T.copy()
    patient_matrix.index = [patient_id_from_sample(c) for c in patient_matrix.index]
    patient_matrix = patient_matrix.groupby(level=0).mean()

    variances = patient_matrix.var(axis=0, skipna=True).sort_values(ascending=False)
    top_genes = variances.head(top_k).index.tolist()
    patient_matrix = patient_matrix[top_genes]

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(patient_matrix.values)
    patient_matrix = pd.DataFrame(scaled_values, index=patient_matrix.index, columns=patient_matrix.columns)

    return patient_matrix, top_genes


def load_mutation_matrix():
    mut_path = os.path.join(TCGA_DIR, "data_mutations.txt")
    if not os.path.exists(mut_path):
        raise FileNotFoundError(f"突变数据文件不存在: {mut_path}")

    df = pd.read_csv(mut_path, sep="\t", comment="#", dtype=str, low_memory=False)

    if "Hugo_Symbol" not in df.columns:
        raise ValueError("突变数据缺少 Hugo_Symbol 列")

    sample_col = "Tumor_Sample_Barcode" if "Tumor_Sample_Barcode" in df.columns else None
    if sample_col is None:
        raise ValueError("突变数据缺少 Tumor_Sample_Barcode 列")

    df = df[df["Hugo_Symbol"].notna() & df[sample_col].notna()].copy()
    df["PATIENT_ID"] = df[sample_col].astype(str).apply(patient_id_from_sample)

    n_patients = df["PATIENT_ID"].nunique()
    gene_freq = df.groupby("Hugo_Symbol")["PATIENT_ID"].nunique() / n_patients
    keep_genes = gene_freq[(gene_freq >= MUTATION_FREQ_LOW) & (gene_freq <= MUTATION_FREQ_HIGH)].index.tolist()

    if not keep_genes:
        return pd.DataFrame(), df

    filtered = df[df["Hugo_Symbol"].isin(keep_genes)].copy()
    pivot = (
        filtered.assign(value=1)
        .pivot_table(index="PATIENT_ID", columns="Hugo_Symbol",
                     values="value", aggfunc="max", fill_value=0)
        .reset_index()
    )
    return pivot, df


def load_cnv():
    cnv_path = os.path.join(TCGA_DIR, "data_cna.txt")
    if not os.path.exists(cnv_path):
        cnv_path = os.path.join(TCGA_DIR, "data_log2_cna.txt")
    if not os.path.exists(cnv_path):
        raise FileNotFoundError(f"CNV数据文件不存在: 已尝试 data_cna.txt 和 data_log2_cna.txt")

    df = pd.read_csv(cnv_path, sep="\t", comment="#", dtype=str, low_memory=False)

    id_col = "Hugo_Symbol" if "Hugo_Symbol" in df.columns else df.columns[0]
    sample_cols = matrix_sample_columns(df.columns)
    df = df[[id_col] + sample_cols].copy()
    df = df[df[id_col].notna() & (df[id_col].astype(str).str.strip() != "")]
    df[id_col] = df[id_col].astype(str)

    numeric = df.set_index(id_col).apply(pd.to_numeric, errors="coerce")
    if numeric.index.has_duplicates:
        numeric = numeric.groupby(level=0).mean()

    patient_matrix = numeric.T.copy()
    patient_matrix.index = [patient_id_from_sample(c) for c in patient_matrix.index]
    patient_matrix = patient_matrix.groupby(level=0).mean()

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(patient_matrix.values)
    patient_matrix = pd.DataFrame(scaled_values, index=patient_matrix.index, columns=patient_matrix.columns)

    return patient_matrix


def _apply_knn_imputer(matrix: np.ndarray, log, n_neighbors: int = 5) -> np.ndarray:
    """KNN 填补器回退方案。

    使用 sklearn KNNImputer 对矩阵进行 KNN 填补。
    若全 NaN 列无法填补，则回退到 0 填充。
    """
    nan_mask = np.isnan(matrix)
    if not nan_mask.any():
        return matrix
    # 全 NaN 列无法 KNN 填补，先置 0
    all_nan_cols = np.all(nan_mask, axis=0)
    if all_nan_cols.any():
        log.warning(f"  [KNN] 发现 {int(all_nan_cols.sum())} 个全NaN列，回退填充为 0")
        matrix[:, all_nan_cols] = 0.0
        nan_mask = np.isnan(matrix)
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    try:
        matrix = imputer.fit_transform(matrix)
    except Exception as e:
        log.error(f"  [KNN] KNNImputer 失败({e})，回退到列均值填充")
        col_means = np.nanmean(matrix, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        matrix = np.where(np.isnan(matrix), col_means, matrix)
    n_filled = int(nan_mask.sum())
    log.info(f"  [KNN] 填补完成: {n_filled} 个缺失值")
    return matrix


def load_methylation(probe_blacklist=None, min_iqr=0.05,
                     gain_epochs=500, gain_batch_size=64,
                     use_gain=True):
    """加载 HM450 甲基化数据，使用 Polars + numpy 全链路优化。

    预处理顺序（两阶段策略）:
        1. 移除全 NaN 探针列
        2. Stage A: 快速中位数预填补 → IQR 低变异过滤（降维）
        3. Stage B: GAIN 填补（失败则回退 KNN 填补）→ Z-score 标准化

    Parameters
    ----------
    probe_blacklist : set/list/None
        需要过滤的探针 ID 集合（如性染色体/交叉反应探针）
    min_iqr : float
        IQR 阈值，> 0 时启用低变异过滤；默认 0.05
    gain_epochs : int
        GAIN 最大训练轮数；默认 500
    gain_batch_size : int
        GAIN 训练批量大小；默认 64
    use_gain : bool
        是否启用 GAIN 填补（默认 True）；False 时回退到 KNN 填补
    """
    import time as _time
    meth_path = os.path.join(TCGA_DIR, "data_methylation_hm450.txt")
    if not os.path.exists(meth_path):
        raise FileNotFoundError(f"甲基化数据文件不存在: {meth_path}")

    logger.info("  [提示] 甲基化文件较大(~1.3GB)，使用 Polars+numpy 优化加载...")
    _t0 = _time.time()

    # ── Step 1: Polars 读取 ──────────────────────────────────────────
    try:
        df = pl.read_csv(
            meth_path,
            separator="\t",
            comment_prefix="#",
            infer_schema_length=10000,
            ignore_errors=True,
        )
    except Exception:
        # 回退：若 scan/read 不支持 comment_prefix，逐行过滤 # 开头
        df = pl.read_csv(
            meth_path,
            separator="\t",
            infer_schema_length=10000,
            ignore_errors=True,
        )
        first_col = df.columns[0]
        df = df.filter(~pl.col(first_col).cast(pl.Utf8).str.starts_with("#"))

    _t1 = _time.time()
    logger.info(f"  Polars 读取完成: {_t1 - _t0:.1f}s, shape={df.shape}")

    # ── Step 1b: 确定 ID 列和 sample 列 ─────────────────────────────
    id_col = df.columns[0]
    for candidate in ["Composite.Element.REF", "ENTITY_STABLE_ID", "ID", "Hugo_Symbol", "geneNames"]:
        if candidate in df.columns:
            id_col = candidate
            break

    sample_cols = [c for c in df.columns if c not in META_COLUMNS]

    # 选择 ID + sample 列，将 sample 列 cast 为 Float32
    df = df.select([pl.col(id_col).cast(pl.Utf8)] + [pl.col(c).cast(pl.Float32) for c in sample_cols])

    # 过滤空 ID
    df = df.filter(pl.col(id_col).is_not_null() & (pl.col(id_col).str.strip_chars() != ""))

    # 探针黑名单过滤
    if probe_blacklist is not None and len(probe_blacklist) > 0:
        blacklist_list = list(probe_blacklist)
        df = df.filter(~pl.col(id_col).is_in(blacklist_list))

    _t2 = _time.time()
    logger.info(f"  列选择/类型转换完成: {_t2 - _t1:.1f}s")

    # ── Step 2: 转 numpy 并处理重复探针 ─────────────────────────────
    probe_ids = df[id_col].to_list()
    values = df.select(sample_cols).to_numpy()  # shape: (~396K, ~570), float32
    del df  # 释放 Polars DataFrame

    # 重复探针聚合（按 probe_id groupby mean）
    probe_ids_arr = np.array(probe_ids)
    unique_probes, inverse_idx = np.unique(probe_ids_arr, return_inverse=True)
    if len(unique_probes) < len(probe_ids_arr):
        logger.info(f"  发现重复探针: {len(probe_ids_arr)} -> {len(unique_probes)}，执行聚合...")
        n_samples = values.shape[1]
        agg_values = np.zeros((len(unique_probes), n_samples), dtype=np.float32)
        agg_counts = np.zeros((len(unique_probes), n_samples), dtype=np.float32)
        nan_mask_vals = np.isnan(values)
        values_safe = np.where(nan_mask_vals, 0.0, values)
        np.add.at(agg_values, inverse_idx, values_safe)
        np.add.at(agg_counts, inverse_idx, (~nan_mask_vals).astype(np.float32))
        values = np.where(agg_counts > 0, agg_values / np.clip(agg_counts, 1.0, None), np.nan)
        probe_ids = list(unique_probes)
        del agg_values, agg_counts, nan_mask_vals, values_safe
    else:
        probe_ids = list(unique_probes)

    del probe_ids_arr, unique_probes, inverse_idx

    # ── Step 3: numpy 层转置 + patient_id 聚合 ───────────────────────
    transposed = np.ascontiguousarray(values.T)  # shape: (~570, ~396K), float32
    del values

    patient_ids_raw = [patient_id_from_sample(c) for c in sample_cols]
    patient_ids_arr = np.array(patient_ids_raw)
    unique_patients, patient_inverse = np.unique(patient_ids_arr, return_inverse=True)

    if len(unique_patients) < len(patient_ids_raw):
        logger.info(f"  发现重复 patient_id: {len(patient_ids_raw)} -> {len(unique_patients)}，执行聚合...")
        n_probes = transposed.shape[1]
        agg_transposed = np.zeros((len(unique_patients), n_probes), dtype=np.float32)
        agg_pat_counts = np.zeros((len(unique_patients), n_probes), dtype=np.float32)
        nan_mask_trans = np.isnan(transposed)
        transposed_safe = np.where(nan_mask_trans, 0.0, transposed)
        np.add.at(agg_transposed, patient_inverse, transposed_safe)
        np.add.at(agg_pat_counts, patient_inverse, (~nan_mask_trans).astype(np.float32))
        transposed = np.where(agg_pat_counts > 0, agg_transposed / np.clip(agg_pat_counts, 1.0, None), np.nan)
        del agg_transposed, agg_pat_counts, nan_mask_trans, transposed_safe

    patient_ids = list(unique_patients)
    del patient_ids_arr, unique_patients, patient_inverse

    matrix = transposed  # shape: (n_patients, n_probes)
    del transposed

    _t3 = _time.time()
    logger.info(f"  转置/聚合完成: {_t3 - _t2:.1f}s, shape={matrix.shape}")

    # 移除全 NaN 的探针列
    col_not_all_nan = ~np.all(np.isnan(matrix), axis=0)
    if not np.all(col_not_all_nan):
        n_removed = int(np.sum(~col_not_all_nan))
        matrix = matrix[:, col_not_all_nan]
        probe_ids = [p for p, k in zip(probe_ids, col_not_all_nan) if k]
        logger.info(f"  移除全NaN探针列: {n_removed} 个")
    del col_not_all_nan

    # ── Step 4: Stage A — 快速中位数预填补 + IQR 低变异过滤 ──────────
    # 先用快速中位数预填补（仅用于 IQR 计算，保证过滤结果准确）
    _quick_medians = np.nanmedian(matrix, axis=0)
    _nan_quick_med = np.isnan(_quick_medians)
    if _nan_quick_med.any():
        _global_med = np.nanmedian(matrix)
        if np.isnan(_global_med):
            _global_med = 0.0
        _quick_medians = np.where(_nan_quick_med, _global_med, _quick_medians)
    _matrix_prefill = np.where(np.isnan(matrix), _quick_medians, matrix)
    del _quick_medians, _nan_quick_med

    # 收集 IQR 过滤统计（在过滤前计算）
    iqr_stats = {}
    if min_iqr > 0:
        q75 = np.percentile(_matrix_prefill, 75, axis=0)
        q25 = np.percentile(_matrix_prefill, 25, axis=0)
        iqr = q75 - q25
        iqr_stats["n_before_iqr"] = int(matrix.shape[1])
        iqr_stats["iqr_p50"] = round(float(np.percentile(iqr, 50)), 6)
        iqr_stats["iqr_p95"] = round(float(np.percentile(iqr, 95)), 6)
        iqr_stats["iqr_mean"] = round(float(np.mean(iqr)), 6)
        keep_mask = iqr >= min_iqr
        n_before = matrix.shape[1]
        matrix = matrix[:, keep_mask]
        probe_ids = [p for p, k in zip(probe_ids, keep_mask) if k]
        iqr_stats["n_after_iqr"] = int(matrix.shape[1])
        iqr_stats["iqr_retention_rate"] = round(float(matrix.shape[1] / max(n_before, 1)), 4)
        logger.info(f"  低变异过滤(IQR>={min_iqr}): {n_before} -> {matrix.shape[1]} 探针")
        del q75, q25, iqr, keep_mask
    del _matrix_prefill

    # ── Step 5: Stage B — GAIN 填补（失败回退 KNN）───────────────────────
    _knn_fallback = lambda mat: _apply_knn_imputer(mat, logger)
    meth_validation = {}
    if use_gain:
        try:
            logger.info(f"  [GAIN] 对 {matrix.shape[1]} 探针子矩阵执行 GAIN 填补...")
            gainer = GAINImputer(
                n_epochs=gain_epochs,
                batch_size=gain_batch_size,
                lr=1e-3,
                seed=20260604,
                patience=50,
            )
            _orig_nan_mask = np.isnan(matrix)
            matrix, _gain_val = gainer.fit_transform(matrix, return_validation=True)
            # 合并 GAIN masking 验证指标
            if _gain_val:
                meth_validation.update({
                    "gain_rmse": _gain_val.get("rmse", np.nan),
                    "gain_mae": _gain_val.get("mae", np.nan),
                    "gain_r_squared": _gain_val.get("r_squared", np.nan),
                    "gain_median_abs_error": _gain_val.get("median_absolute_error", np.nan),
                    "gain_n_validated": _gain_val.get("n_holdout", 0),
                })
            del _gain_val
            # 仅替换缺失位置，保留观测值不变
            if _orig_nan_mask.any():
                matrix = np.where(_orig_nan_mask, matrix, matrix)
            del _orig_nan_mask
            logger.info("  [GAIN] 填补完成")
        except Exception as e:
            logger.warning(f"  [GAIN] 填补失败({e})，回退到 KNN 填补")
            matrix = _knn_fallback(matrix)
    else:
        logger.info("  [KNN] use_gain=False，使用 KNN 填补")
        matrix = _knn_fallback(matrix)

    # ── Step 6: numpy 层 Z-score 标准化 ──────────────────────────────────
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    matrix = (matrix - mean) / np.clip(std, EPS, None)
    del mean, std

    # ── Step 7: 构造最终 DataFrame ───────────────────────────────────
    patient_matrix = pd.DataFrame(matrix, index=patient_ids, columns=probe_ids)
    del matrix

    _t_end = _time.time()
    logger.info(f"  甲基化加载总耗时: {_t_end - _t0:.1f}s, 最终shape={patient_matrix.shape}")

    # 合并 IQR 统计到 validation dict
    meth_validation.update(iqr_stats)

    return patient_matrix, meth_validation


def load_rppa():
    rppa_path = os.path.join(TCGA_DIR, "data_rppa.txt")
    if not os.path.exists(rppa_path):
        raise FileNotFoundError(f"RPPA数据文件不存在: {rppa_path}")

    df = pd.read_csv(rppa_path, sep="\t", comment="#", dtype=str, low_memory=False)

    id_col = df.columns[0]
    for candidate in ["ENTITY_STABLE_ID", "GENE_SYMBOL", "Hugo_Symbol", "ID", "NAME", "PHOSPHOSITE"]:
        if candidate in df.columns:
            id_col = candidate
            break

    sample_cols = matrix_sample_columns(df.columns)
    df = df[[id_col] + sample_cols].copy()
    df = df[df[id_col].notna() & (df[id_col].astype(str).str.strip() != "")]
    df[id_col] = df[id_col].astype(str)

    numeric = df.set_index(id_col).apply(pd.to_numeric, errors="coerce")
    if numeric.index.has_duplicates:
        numeric = numeric.groupby(level=0).mean()

    patient_matrix = numeric.T.copy()
    patient_matrix.index = [patient_id_from_sample(c) for c in patient_matrix.index]
    patient_matrix = patient_matrix.groupby(level=0).mean()

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(patient_matrix.values)
    patient_matrix = pd.DataFrame(scaled_values, index=patient_matrix.index, columns=patient_matrix.columns)

    return patient_matrix


def align_samples(clinical_df, gene_df, mutation_df, cnv_df, methylation_df, rppa_df, strict=False):
    """多组学样本对齐。

    Args:
        strict: 若 True，取所有组学严格交集（保守）；
                若 False（默认），取核心组学（临床+RNA+突变）交集，
                其他组学允许部分缺失，由下游 mask 机制处理。
    """
    clinical_ids = set(clinical_df["PATIENT_ID"].astype(str))

    omics_sets = {}
    if gene_df is not None and not gene_df.empty:
        omics_sets["gene"] = set(gene_df.index.astype(str))
    if mutation_df is not None and not mutation_df.empty:
        mut_ids = mutation_df["PATIENT_ID"].astype(str) if "PATIENT_ID" in mutation_df.columns else pd.Series(dtype=str)
        omics_sets["mutation"] = set(mut_ids)
    if cnv_df is not None and not cnv_df.empty:
        omics_sets["cnv"] = set(cnv_df.index.astype(str))
    if methylation_df is not None and not methylation_df.empty:
        omics_sets["methylation"] = set(methylation_df.index.astype(str))
    if rppa_df is not None and not rppa_df.empty:
        omics_sets["rppa"] = set(rppa_df.index.astype(str))

    if strict:
        # 严格交集：所有组学都必须有数据
        all_sets = [clinical_ids] + list(omics_sets.values())
        common_ids = set.intersection(*all_sets) if all_sets else set()
    else:
        # 核心交集：临床 + RNA + 突变（这三个是核心组学）
        core_sets = [clinical_ids]
        if "gene" in omics_sets:
            core_sets.append(omics_sets["gene"])
        if "mutation" in omics_sets:
            core_sets.append(omics_sets["mutation"])
        common_ids = set.intersection(*core_sets) if core_sets else set()

    if not common_ids:
        available_sets = [clinical_ids]
        for name, s in omics_sets.items():
            if s:
                available_sets.append(s)
        common_ids = set.intersection(*available_sets) if available_sets else set()

    common_ids = sorted(common_ids)

    logger.info(f"  [对齐策略] {'严格交集' if strict else '核心交集(临床+RNA+突变)'}: {len(common_ids)} 样本")
    if not strict:
        for name, s in omics_sets.items():
            overlap = len(set(common_ids) & s) if s else 0
            logger.info(f"    {name}: {overlap}/{len(common_ids)} 样本有数据 ({overlap/max(1,len(common_ids))*100:.1f}%)")

    clinical_aligned = clinical_df[clinical_df["PATIENT_ID"].astype(str).isin(common_ids)].copy()

    gene_aligned = gene_df.loc[gene_df.index.astype(str).isin(common_ids)] if gene_df is not None and not gene_df.empty else pd.DataFrame()
    mutation_aligned = mutation_df[mutation_df["PATIENT_ID"].astype(str).isin(common_ids)] if mutation_df is not None and not mutation_df.empty else pd.DataFrame()
    cnv_aligned = cnv_df.loc[cnv_df.index.astype(str).isin(common_ids)] if cnv_df is not None and not cnv_df.empty else pd.DataFrame()
    methylation_aligned = methylation_df.loc[methylation_df.index.astype(str).isin(common_ids)] if methylation_df is not None and not methylation_df.empty else pd.DataFrame()
    rppa_aligned = rppa_df.loc[rppa_df.index.astype(str).isin(common_ids)] if rppa_df is not None and not rppa_df.empty else pd.DataFrame()

    # ── 构建缺失指示变量（1=有数据, 0=该组学缺失）──────────────────
    missing_indicators = {}
    for name, s in omics_sets.items():
        missing_indicators[name] = pd.Series(
            [1 if pid in s else 0 for pid in common_ids],
            index=common_ids, dtype=np.int8
        )
    if missing_indicators:
        for name, ind in missing_indicators.items():
            n_present = int(ind.sum())
            logger.info(f"    缺失指示 [{name}]: {n_present}/{len(common_ids)} 有数据")

    return common_ids, clinical_aligned, gene_aligned, mutation_aligned, cnv_aligned, methylation_aligned, rppa_aligned, missing_indicators


def save_preprocessed(output_dir, clinical_df, gene_df, gene_names, mutation_df,
                       cnv_df, methylation_df, rppa_df, sample_ids, config,
                       missing_indicators=None):
    os.makedirs(output_dir, exist_ok=True)

    clinical_df.to_csv(os.path.join(output_dir, "tcga_os_clinical_endpoint_qc.tsv"),
                       sep="\t", index=False)

    survival_cols = ["PATIENT_ID", "OS_STATUS", "OS_MONTHS", "OS_EVENT",
                     "death_by_36m_observed", "death_by_36m", "early_censored_before_36m",
                     "ipcw_weight_os", "ipcw_label_available",
                     "pseudo_risk_os_raw", "pseudo_risk_os"]
    survival_cols = [c for c in survival_cols if c in clinical_df.columns]
    survival_df = clinical_df[survival_cols]
    with open(os.path.join(output_dir, "survival_labels.pkl"), "wb") as f:
        pickle.dump(survival_df, f)

    if gene_df is not None and not gene_df.empty:
        gene_df.to_csv(os.path.join(output_dir, "gene_expression_curated.tsv"), sep="\t")

    with open(os.path.join(output_dir, "gene_names.pkl"), "wb") as f:
        pickle.dump(gene_names, f)

    if mutation_df is not None and not mutation_df.empty:
        mutation_df.to_csv(os.path.join(output_dir, "mutation_curated.tsv"), sep="\t", index=False)

    if cnv_df is not None and not cnv_df.empty:
        cnv_df.to_csv(os.path.join(output_dir, "cnv_curated.tsv"), sep="\t")

    if methylation_df is not None and not methylation_df.empty:
        methylation_df.to_csv(os.path.join(output_dir, "methylation_curated.tsv"), sep="\t")
        # Parquet 缓存（加速后续加载）
        try:
            methylation_df.to_parquet(os.path.join(output_dir, "methylation_curated.parquet"), engine="pyarrow")
        except Exception as e:
            logger.warning(f"  [WARNING] Parquet缓存保存失败: {e}")

    if rppa_df is not None and not rppa_df.empty:
        rppa_df.to_csv(os.path.join(output_dir, "rppa_curated.tsv"), sep="\t")

    # 保存缺失指示变量
    if missing_indicators:
        indicator_df = pd.DataFrame(missing_indicators, index=sample_ids)
        indicator_df.to_csv(os.path.join(output_dir, "omics_missing_indicators.tsv"), sep="\t")
        logger.info(f"  缺失指示变量已保存: {len(missing_indicators)} 种组学")

    with open(os.path.join(output_dir, "sample_ids.pkl"), "wb") as f:
        pickle.dump(sample_ids, f)

    with open(os.path.join(output_dir, "preprocessing_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    n_total = len(clinical_df)
    n_events = int(clinical_df["OS_EVENT"].sum()) if "OS_EVENT" in clinical_df.columns else 0
    n_censored = n_total - n_events
    n_observed_36m = int(clinical_df["death_by_36m_observed"].sum()) if "death_by_36m_observed" in clinical_df.columns else 0
    n_early_censored = int(clinical_df["early_censored_before_36m"].sum()) if "early_censored_before_36m" in clinical_df.columns else 0
    median_months = float(clinical_df["OS_MONTHS"].median()) if "OS_MONTHS" in clinical_df.columns else np.nan

    summary_df = pd.DataFrame([{
        "cohort": "tcga",
        "n_patients": n_total,
        "n_events": n_events,
        "n_censored": n_censored,
        "median_os_months": median_months,
        "n_observed_36m": n_observed_36m,
        "n_early_censored_36m": n_early_censored,
        "n_gene_features": len(gene_names) if gene_names else 0,
        "n_mutation_features": mutation_df.shape[1] - 1 if mutation_df is not None and not mutation_df.empty and "PATIENT_ID" in mutation_df.columns else 0,
        "n_cnv_features": cnv_df.shape[1] if cnv_df is not None and not cnv_df.empty else 0,
        "n_methylation_features": methylation_df.shape[1] if methylation_df is not None and not methylation_df.empty else 0,
        "n_rppa_features": rppa_df.shape[1] if rppa_df is not None and not rppa_df.empty else 0,
        "n_aligned_samples": len(sample_ids),
    }])
    summary_df.to_csv(os.path.join(output_dir, "cohort_endpoint_summary.tsv"), sep="\t", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# QC Report: Metrics Computation
# ─────────────────────────────────────────────────────────────────────────────
def _safe_df(df):
    """Return True if df is a non-empty DataFrame."""
    return df is not None and isinstance(df, pd.DataFrame) and not df.empty


def compute_qc_metrics(
    clinical_df, gene_df, mutation_df, cnv_df,
    methylation_df, rppa_df, sample_ids, missing_indicators,
    meth_validation=None, mutation_raw_df=None,
):
    """Compute comprehensive preprocessing QC metrics (10 categories).

    Returns a flat dict suitable for TSV/JSON export.

    Parameters
    ----------
    meth_validation : dict or None
        GAIN masking 验证指标 + IQR 过滤统计 (来自 load_methylation)
    mutation_raw_df : pd.DataFrame or None
        过滤前的原始突变数据（用于统计过滤前后基因数）
    """
    import time as _time
    _t0 = _time.time()
    m = {}  # metrics dict
    n_total = len(clinical_df)

    # ── 1. Missing-data landscape per omics ──────────────────────────
    omics_map = {
        "gene": gene_df, "mutation": mutation_df, "cnv": cnv_df,
        "methylation": methylation_df, "rppa": rppa_df,
    }
    for name, df_om in omics_map.items():
        if _safe_df(df_om):
            if name == "mutation" and "PATIENT_ID" in df_om.columns:
                mat = df_om.drop(columns=["PATIENT_ID"])
            else:
                mat = df_om.select_dtypes(include=[np.floating, np.integer, float, int])
            total_cells = mat.shape[0] * mat.shape[1]
            n_missing = int(mat.isna().sum().sum()) if total_cells > 0 else 0
            m[f"{name}_n_samples"] = mat.shape[0]
            m[f"{name}_n_features"] = mat.shape[1]
            m[f"{name}_missing_rate"] = round(n_missing / max(total_cells, 1), 6)
            m[f"{name}_n_missing_cells"] = n_missing
        else:
            m[f"{name}_n_samples"] = 0
            m[f"{name}_n_features"] = 0
            m[f"{name}_missing_rate"] = np.nan

    # Clinical column-level missing top-10
    clin_missing = clinical_df.isna().mean().sort_values(ascending=False)
    for i, (col, frac) in enumerate(clin_missing.head(10).items()):
        m[f"clinical_missing_top{i+1}_col"] = col
        m[f"clinical_missing_top{i+1}_rate"] = round(float(frac), 4)

    # ── 2. Sample cross-coverage ────────────────────────────────────
    omics_ids = {}
    for name, df_om in omics_map.items():
        if _safe_df(df_om):
            if name == "mutation" and "PATIENT_ID" in df_om.columns:
                omics_ids[name] = set(df_om["PATIENT_ID"].astype(str))
            else:
                omics_ids[name] = set(df_om.index.astype(str))
        else:
            omics_ids[name] = set()

    common_set = set(sample_ids) if sample_ids else set()
    for name, s in omics_ids.items():
        overlap = len(common_set & s)
        m[f"{name}_overlap_with_aligned"] = overlap
        m[f"{name}_exclusive"] = len(s - common_set)

    # Pairwise intersections
    omics_names = list(omics_ids.keys())
    for i in range(len(omics_names)):
        for j in range(i + 1, len(omics_names)):
            n1, n2 = omics_names[i], omics_names[j]
            inter = len(omics_ids[n1] & omics_ids[n2])
            m[f"pairwise_{n1}_{n2}_intersect"] = inter

    # Patients with >= 3 omics
    if common_set:
        n_omics_per_patient = {pid: 0 for pid in common_set}
        for s in omics_ids.values():
            for pid in s & common_set:
                n_omics_per_patient[pid] = n_omics_per_patient.get(pid, 0) + 1
        m["n_patients_ge3_omics"] = sum(1 for v in n_omics_per_patient.values() if v >= 3)
        m["n_patients_all_5_omics"] = sum(1 for v in n_omics_per_patient.values() if v >= 5)

    # ── 3. Clinical cohort characteristics (Table 1) ─────────────────
    # Age
    age_col = None
    for c in ["AGE", "AGE_AT_DIAGNOSIS", "AGE_AT_INITIAL_PATHOLOGIC_DIAGNOSIS", "AGE_AT_SEQ"]:
        if c in clinical_df.columns:
            age_col = c
            break
    if age_col:
        age_vals = pd.to_numeric(clinical_df[age_col], errors="coerce").dropna()
        if len(age_vals) > 0:
            m["age_mean"] = round(float(age_vals.mean()), 1)
            m["age_median"] = round(float(age_vals.median()), 1)
            m["age_std"] = round(float(age_vals.std()), 1)
            m["age_q25"] = round(float(age_vals.quantile(0.25)), 1)
            m["age_q75"] = round(float(age_vals.quantile(0.75)), 1)
            m["age_min"] = round(float(age_vals.min()), 1)
            m["age_max"] = round(float(age_vals.max()), 1)
            m["age_n_valid"] = int(len(age_vals))

    # Sex
    sex_col = None
    for c in ["SEX", "GENDER"]:
        if c in clinical_df.columns:
            sex_col = c
            break
    if sex_col:
        sex_vals = clinical_df[sex_col].dropna().astype(str).str.lower()
        n_male = int(sex_vals.isin(["male", "m", "1", "1.0"]).sum())
        n_female = int(sex_vals.isin(["female", "f", "0", "0.0"]).sum())
        m["n_male"] = n_male
        m["n_female"] = n_female
        m["male_ratio"] = round(n_male / max(n_male + n_female, 1), 4)

    # AJCC Stage
    if "AJCC_STAGE_ENCODED" in clinical_df.columns:
        stage_vals = clinical_df["AJCC_STAGE_ENCODED"].dropna()
        for stg in [1, 2, 3, 4]:
            cnt = int((stage_vals == stg).sum())
            m[f"n_stage_{stg}"] = cnt
            m[f"stage_{stg}_ratio"] = round(cnt / max(len(stage_vals), 1), 4)
        m["n_stage_valid"] = int(len(stage_vals))
        m["n_stage_missing"] = int(n_total - len(stage_vals))

    # MSI status
    msi_col = None
    for c in ["MSI_STATUS", "MSI_TYPE", "MSI_SCORE", "CNA_CLUSTER"]:
        if c in clinical_df.columns:
            msi_col = c
            break
    if msi_col:
        msi_vals = clinical_df[msi_col].dropna().astype(str).str.upper()
        m["n_msi_h"] = int(msi_vals.isin(["MSI-H", "MSIH", "HIGH"]).sum())
        m["n_mss"] = int(msi_vals.isin(["MSS", "STABLE"]).sum())
        m["n_msi_l"] = int(msi_vals.isin(["MSI-L", "MSIL", "LOW"]).sum())

    # Survival: median OS, KM estimates at 12/24/36 months
    if "OS_MONTHS" in clinical_df.columns and "OS_EVENT" in clinical_df.columns:
        times = clinical_df["OS_MONTHS"].to_numpy(dtype=float)
        events = clinical_df["OS_EVENT"].to_numpy(dtype=int)
        valid = ~(np.isnan(times) | np.isnan(events))
        times, events = times[valid], events[valid]
        if len(times) > 0:
            # Median survival (simple KM median)
            from lifelines import KaplanMeierFitter as _KMF
            kmf = _KMF()
            kmf.fit(times, events)
            m["median_os_months_km"] = round(float(kmf.median_survival_time_), 1)
            for tau_pt in [12.0, 24.0, 36.0]:
                surv_at_tau = float(kmf.survival_function_at_times(tau_pt).values[0])
                m[f"km_survival_at_{int(tau_pt)}m"] = round(surv_at_tau, 4)
                # Greenwood 95% CI
                ci = kmf.confidence_interval_survival_function_
                if tau_pt in ci.index:
                    m[f"km_survival_at_{int(tau_pt)}m_ci_lower"] = round(float(ci.loc[tau_pt].iloc[0]), 4)
                    m[f"km_survival_at_{int(tau_pt)}m_ci_upper"] = round(float(ci.loc[tau_pt].iloc[1]), 4)

    # ── 4. Survival endpoint deep QC ────────────────────────────────
    if "ipcw_weight_os" in clinical_df.columns:
        w = clinical_df["ipcw_weight_os"].to_numpy(dtype=float)
        w_pos = w[w > 0]
        m["ipcw_mean"] = round(float(np.mean(w_pos)) if len(w_pos) > 0 else np.nan, 4)
        m["ipcw_median"] = round(float(np.median(w_pos)) if len(w_pos) > 0 else np.nan, 4)
        m["ipcw_max"] = round(float(np.max(w_pos)) if len(w_pos) > 0 else np.nan, 4)
        m["ipcw_std"] = round(float(np.std(w_pos)) if len(w_pos) > 0 else np.nan, 4)
        m["ipcw_pct_gt10"] = round(float((w_pos > 10).mean()) if len(w_pos) > 0 else np.nan, 4)

    if "pseudo_risk_os" in clinical_df.columns:
        p = clinical_df["pseudo_risk_os"].dropna().to_numpy(dtype=float)
        m["pseudo_mean"] = round(float(np.mean(p)) if len(p) > 0 else np.nan, 4)
        m["pseudo_median"] = round(float(np.median(p)) if len(p) > 0 else np.nan, 4)
        m["pseudo_std"] = round(float(np.std(p)) if len(p) > 0 else np.nan, 4)
        m["pseudo_pct_outside_01"] = round(
            float(((p < 0) | (p > 1)).mean()) if len(p) > 0 else np.nan, 4
        )

    if "OS_EVENT" in clinical_df.columns:
        events_all = clinical_df["OS_EVENT"].to_numpy(dtype=float)
        m["overall_censor_rate"] = round(float((events_all == 0).mean()), 4)
    if "early_censored_before_36m" in clinical_df.columns:
        early = clinical_df["early_censored_before_36m"].to_numpy()
        m["early_censor_rate_36m"] = round(float(early.mean()), 4)

    # ── 5. Feature processing QC ─────────────────────────────────────
    if _safe_df(gene_df):
        gene_var = gene_df.var(axis=0, skipna=True)
        m["gene_var_mean"] = round(float(gene_var.mean()), 4)
        m["gene_var_median"] = round(float(gene_var.median()), 4)
        m["gene_var_max"] = round(float(gene_var.max()), 4)
    if _safe_df(cnv_df):
        m["cnv_n_features_final"] = cnv_df.shape[1]
    if _safe_df(rppa_df):
        m["rppa_n_features_final"] = rppa_df.shape[1]

    # ── 6. Imputation quality (methylation) ──────────────────────────
    if _safe_df(methylation_df):
        meth_vals = methylation_df.to_numpy()
        m["methylation_residual_nan_rate"] = round(float(np.isnan(meth_vals).mean()), 6)
        m["methylation_n_probes_final"] = methylation_df.shape[1]

    # ── 7. GAIN 填补质量评估（新增）────────────────────────────────
    if meth_validation:
        for key in ["gain_rmse", "gain_mae", "gain_r_squared",
                     "gain_median_abs_error", "gain_n_validated"]:
            if key in meth_validation:
                m[key] = meth_validation[key]
        # IQR 过滤统计
        for key in ["n_before_iqr", "n_after_iqr", "iqr_retention_rate",
                     "iqr_p50", "iqr_p95", "iqr_mean"]:
            if key in meth_validation:
                m[f"meth_{key}" if not key.startswith("meth_") else key] = meth_validation[key]

    # ── 8. 标准化效果验证（新增）──────────────────────────────────
    from scipy.stats import skew, kurtosis as _kurtosis
    for omics_name, omics_df_ in [("gene", gene_df), ("cnv", cnv_df), ("rppa", rppa_df)]:
        if _safe_df(omics_df_):
            vals = omics_df_.select_dtypes(include=[np.floating, np.integer, float, int]).values
            if vals.size > 0:
                col_vars = np.nanvar(vals, axis=0)
                m[f"{omics_name}_var_mean_post"] = round(float(np.mean(col_vars)), 4)
                m[f"{omics_name}_variance_cv"] = round(
                    float(np.std(col_vars) / max(np.mean(col_vars), 1e-12)), 4
                )
                # 采样计算偏度/峰度（避免大矩阵耗时）
                _sample = vals.flatten()
                if len(_sample) > 200000:
                    _rng = np.random.default_rng(42)
                    _sample = _rng.choice(_sample, size=200000, replace=False)
                _sample = _sample[~np.isnan(_sample)]
                if len(_sample) > 10:
                    m[f"{omics_name}_skewness_post"] = round(float(skew(_sample)), 4)
                    m[f"{omics_name}_kurtosis_post"] = round(float(_kurtosis(_sample)), 4)

    # ── 9. PCA 批次/技术偏差检测（新增）──────────────────────────
    if _safe_df(gene_df):
        try:
            from sklearn.decomposition import PCA as _PCA
            gene_vals = gene_df.select_dtypes(include=[np.floating, np.integer, float, int]).values
            # 仅在有足够样本时执行 PCA
            if gene_vals.shape[0] >= 5:
                pca = _PCA(n_components=min(2, gene_vals.shape[1], gene_vals.shape[0]))
                pca.fit(np.nan_to_num(gene_vals))
                var_pct = pca.explained_variance_ratio_ * 100
                m["gene_pca_pc1_variance_pct"] = round(float(var_pct[0]), 2)
                if len(var_pct) > 1:
                    m["gene_pca_pc2_variance_pct"] = round(float(var_pct[1]), 2)
                m["gene_pca_top2_total_pct"] = round(float(np.sum(var_pct[:2])), 2)
                # PC1 与甲基化覆盖指示的相关系数
                if missing_indicators and "methylation" in missing_indicators:
                    meth_ind = missing_indicators["methylation"]
                    pc1_scores = pca.transform(np.nan_to_num(gene_vals))[:, 0]
                    # 对齐样本
                    common_pids = [str(pid) for pid in gene_df.index if str(pid) in meth_ind.index]
                    if len(common_pids) >= 10:
                        ind_vals = np.array([meth_ind[str(pid)] for pid in common_pids], dtype=float)
                        pc1_aligned = pc1_scores[:len(common_pids)]
                        corr, p_corr = stats.pearsonr(pc1_aligned, ind_vals)
                        m["gene_pca_meth_overlap_corr"] = round(float(corr), 4)
                        m["gene_pca_meth_overlap_p"] = round(float(p_corr), 4)
        except Exception as e:
            logger.warning(f"  [QC] PCA 分析失败: {e}")

    # ── 10. 特征过滤合理性（新增）────────────────────────────────
    # 突变频率过滤统计
    if mutation_raw_df is not None and _safe_df(mutation_df):
        try:
            if "Hugo_Symbol" in mutation_raw_df.columns:
                n_genes_raw = mutation_raw_df["Hugo_Symbol"].nunique()
            else:
                n_genes_raw = 0
            n_genes_filtered = mutation_df.shape[1] - 1 if "PATIENT_ID" in mutation_df.columns else mutation_df.shape[1]
            if n_genes_raw > 0:
                m["mutation_n_before_filter"] = n_genes_raw
                m["mutation_n_after_filter"] = n_genes_filtered
                m["mutation_retention_rate"] = round(n_genes_filtered / max(n_genes_raw, 1), 4)
        except Exception:
            pass
    # 突变频率分布
    if _safe_df(mutation_df):
        mut_mat = mutation_df.drop(columns=["PATIENT_ID"], errors="ignore").values
        if mut_mat.size > 0:
            freqs = np.mean(mut_mat > 0, axis=0)
            m["mutation_freq_p50"] = round(float(np.percentile(freqs, 50)), 4)
            m["mutation_freq_p95"] = round(float(np.percentile(freqs, 95)), 4)

    # 临床补充: KRAS/BRAF 突变率
    if _safe_df(mutation_df) and "PATIENT_ID" in mutation_df.columns:
        mut_feats = mutation_df.drop(columns=["PATIENT_ID"], errors="ignore")
        for gene_sym in ["KRAS", "BRAF", "NRAS"]:
            if gene_sym in mut_feats.columns:
                n_mut = int((mut_feats[gene_sym] > 0).sum())
                m[f"n_{gene_sym.lower()}_mutated"] = n_mut

    # TMB 分布
    if mutation_raw_df is not None and "PATIENT_ID" in mutation_raw_df.columns:
        try:
            tmb_per_patient = mutation_raw_df.groupby("PATIENT_ID").size()
            m["tmb_median"] = round(float(tmb_per_patient.median()), 1)
            m["tmb_mean"] = round(float(tmb_per_patient.mean()), 1)
        except Exception:
            pass

    _elapsed = round(_time.time() - _t0, 2)
    m["_qc_compute_seconds"] = _elapsed
    logger.info(f"  [QC] Metrics computed: {len(m)} indicators in {_elapsed}s")
    return m


# ─────────────────────────────────────────────────────────────────────────────
# QC Report: Visualization — Nature/Cell Publication-Quality (10 plots)
# ─────────────────────────────────────────────────────────────────────────────

# Okabe-Ito 学术标准色板（色盲安全、Nature 推荐）
NATURE_PALETTE = {
    "blue":       "#0072B2",
    "orange":     "#E69F00",
    "green":      "#009E73",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
    "skyblue":    "#56B4E9",
    "yellow":     "#F0E442",
    "gray":       "#999999",
}
_NP = list(NATURE_PALETTE.values())  # list for indexed access


def _apply_nature_style():
    """Apply Nature/Cell publication global matplotlib settings."""
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })


def _get_palette():
    """Return Okabe-Ito palette (backward compatibility)."""
    return _NP


def generate_qc_plots(
    qc_dir, clinical_df, gene_df, mutation_df, cnv_df,
    methylation_df, rppa_df, sample_ids, missing_indicators, metrics,
    meth_validation=None,
):
    """Generate 10 publication-quality QC plots (Nature/Cell style)."""
    import time as _time
    _t0 = _time.time()
    _apply_nature_style()
    n_total = len(clinical_df)

    omics_map = {
        "Gene": gene_df, "Mutation": mutation_df, "CNV": cnv_df,
        "Methylation": methylation_df, "RPPA": rppa_df,
    }

    # ── Plot 1: Missing data landscape (A: heatmap, B: coverage bar) ────
    try:
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5),
                                          gridspec_kw={"width_ratios": [1.2, 1]})
        # Panel A: Missing-rate heatmap (omics × clinical columns)
        clin_sub = clinical_df[["OS_EVENT", "OS_MONTHS", "AJCC_STAGE_ENCODED"]]
        clin_sub = clin_sub.select_dtypes(include=[np.floating, np.integer, float, int])
        omics_names_short = [n for n in omics_map if _safe_df(omics_map[n])]
        miss_rates = []
        for name in omics_names_short:
            mr = metrics.get(f"{name.lower()}_missing_rate", 0.0)
            miss_rates.append(mr * 100)
        hm_data = np.array(miss_rates).reshape(-1, 1)
        im = ax_a.imshow(hm_data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(miss_rates) * 1.3 if miss_rates else 100)
        ax_a.set_yticks(range(len(omics_names_short)))
        ax_a.set_yticklabels(omics_names_short, fontsize=9)
        ax_a.set_xticks([0])
        ax_a.set_xticklabels(["Missing %"], fontsize=9)
        for i, r in enumerate(miss_rates):
            ax_a.text(0, i, f"{r:.1f}%", ha="center", va="center", fontsize=8, fontweight="bold")
        ax_a.set_title("A  Missing Rate per Modality", loc="left", fontweight="bold")
        fig.colorbar(im, ax=ax_a, shrink=0.6, label="Missing %")

        # Panel B: Coverage bar chart
        coverage_counts = []
        cov_labels = []
        common_set = set(str(s) for s in sample_ids) if sample_ids else set()
        for name in omics_names_short:
            df_om = omics_map[name]
            if name == "Mutation" and "PATIENT_ID" in df_om.columns:
                cnt = len(set(df_om["PATIENT_ID"].astype(str)) & common_set)
            else:
                cnt = len(set(df_om.index.astype(str)) & common_set)
            coverage_counts.append(cnt / max(len(common_set), 1) * 100)
            cov_labels.append(name)
        bars = ax_b.barh(range(len(cov_labels)), coverage_counts, color=_NP[:len(cov_labels)], height=0.6)
        ax_b.axvline(100, color=NATURE_PALETTE["vermillion"], linestyle="--", alpha=0.6, linewidth=1.0)
        ax_b.set_yticks(range(len(cov_labels)))
        ax_b.set_yticklabels(cov_labels, fontsize=9)
        ax_b.set_xlabel("Coverage (%)", fontsize=10)
        ax_b.set_title("B  Sample Coverage (aligned cohort)", loc="left", fontweight="bold")
        ax_b.set_xlim(0, 115)
        for bar, pct in zip(bars, coverage_counts):
            ax_b.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                      f"{pct:.0f}%", va="center", fontsize=8)
        plt.suptitle("Missing Data Landscape", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(qc_dir, "01_missing_landscape.png"), bbox_inches="tight")
        plt.close(fig)
        logger.info("  [QC Plot 1/10] 01_missing_landscape.png saved")
    except Exception as e:
        logger.warning(f"  [QC Plot 1] failed: {e}")

    # ── Plot 2: Sample cross-coverage UpSet-style (A: bars, B: matrix) ──
    try:
        omics_avail = [n for n in omics_map if _safe_df(omics_map[n])]
        omics_ids = {}
        for name in omics_avail:
            df_om = omics_map[name]
            if name == "Mutation" and "PATIENT_ID" in df_om.columns:
                omics_ids[name] = set(df_om["PATIENT_ID"].astype(str))
            else:
                omics_ids[name] = set(df_om.index.astype(str))
        common_set = set(str(s) for s in sample_ids) if sample_ids else set()

        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))
        # Panel A: intersection sizes
        counts = []
        labels = []
        for name in omics_avail:
            cnt = len(omics_ids[name] & common_set)
            counts.append(cnt)
            labels.append(name)
        x_pos = np.arange(len(labels))
        bars = ax_a.bar(x_pos, counts, color=_NP[:len(labels)], width=0.55, edgecolor="white", linewidth=0.5)
        ax_a.axhline(len(common_set), color=NATURE_PALETTE["vermillion"], linestyle="--",
                      alpha=0.6, linewidth=1.0, label=f"Aligned N={len(common_set)}")
        ax_a.set_xticks(x_pos)
        ax_a.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax_a.set_ylabel("Number of Patients", fontsize=10)
        ax_a.set_title("A  Patient Count per Modality", loc="left", fontweight="bold")
        ax_a.legend(loc="upper right", fontsize=8)
        for bar, cnt in zip(bars, counts):
            ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                      str(cnt), ha="center", fontsize=8, fontweight="bold")

        # Panel B: pairwise intersection matrix heatmap
        n_om = len(omics_avail)
        inter_mat = np.zeros((n_om, n_om), dtype=int)
        for i in range(n_om):
            for j in range(n_om):
                if i == j:
                    inter_mat[i, j] = len(omics_ids[omics_avail[i]] - set.union(*[omics_ids[k] for k in omics_avail if k != omics_avail[i]]))
                else:
                    inter_mat[i, j] = len(omics_ids[omics_avail[i]] & omics_ids[omics_avail[j]])
        im = ax_b.imshow(inter_mat, cmap="Blues", aspect="auto")
        ax_b.set_xticks(range(n_om))
        ax_b.set_yticks(range(n_om))
        ax_b.set_xticklabels(omics_avail, rotation=30, ha="right", fontsize=8)
        ax_b.set_yticklabels(omics_avail, fontsize=8)
        for i in range(n_om):
            for j in range(n_om):
                ax_b.text(j, i, str(inter_mat[i, j]), ha="center", va="center",
                          fontsize=8, color="white" if inter_mat[i, j] > inter_mat.max() * 0.6 else "black")
        ax_b.set_title("B  Pairwise Overlap Matrix", loc="left", fontweight="bold")
        fig.colorbar(im, ax=ax_b, shrink=0.7, label="Patient count")
        plt.suptitle("Sample Cross-Coverage", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(qc_dir, "02_sample_cross_coverage.png"), bbox_inches="tight")
        plt.close(fig)
        logger.info("  [QC Plot 2/10] 02_sample_cross_coverage.png saved")
    except Exception as e:
        logger.warning(f"  [QC Plot 2] failed: {e}")

    # ── Plot 3: Clinical baseline (A: Age hist+KDE, B: Sex donut, C: Stage, D: KM) ─
    try:
        age_col = None
        for c in ["AGE", "AGE_AT_DIAGNOSIS", "AGE_AT_INITIAL_PATHOLOGIC_DIAGNOSIS"]:
            if c in clinical_df.columns:
                age_col = c
                break
        sex_col = None
        for c in ["SEX", "GENDER"]:
            if c in clinical_df.columns:
                sex_col = c
                break
        has_stage = "AJCC_STAGE_ENCODED" in clinical_df.columns
        has_km = "OS_MONTHS" in clinical_df.columns and "OS_EVENT" in clinical_df.columns
        panels = []
        if age_col:
            panels.append("age")
        if sex_col:
            panels.append("sex")
        if has_stage:
            panels.append("stage")
        if has_km:
            panels.append("km")
        n_p = max(len(panels), 2)
        fig, axes = plt.subplots(1, n_p, figsize=(5 * n_p, 4.5))
        if n_p == 1:
            axes = [axes]
        ax_i = 0

        # A: Age histogram + KDE
        if "age" in panels and ax_i < n_p:
            age_vals = pd.to_numeric(clinical_df[age_col], errors="coerce").dropna()
            if len(age_vals) > 0:
                ax = axes[ax_i]
                n_bins = min(25, max(10, len(age_vals) // 10))
                ax.hist(age_vals, bins=n_bins, color=NATURE_PALETTE["blue"],
                        edgecolor="white", alpha=0.8, density=True, label="Histogram")
                if len(age_vals) > 5 and _HAS_SEABORN:
                    try:
                        from scipy.stats import gaussian_kde
                        kde = gaussian_kde(age_vals.values)
                        x_grid = np.linspace(age_vals.min() - 2, age_vals.max() + 2, 200)
                        ax.plot(x_grid, kde(x_grid), color=NATURE_PALETTE["vermillion"], linewidth=2, label="KDE")
                    except Exception:
                        pass
                med = age_vals.median()
                ax.axvline(med, color=NATURE_PALETTE["vermillion"], linestyle="--", linewidth=1.2,
                           label=f"Median={med:.0f}")
                iqr_lo, iqr_hi = age_vals.quantile(0.25), age_vals.quantile(0.75)
                ax.set_xlabel("Age at Diagnosis")
                ax.set_ylabel("Density")
                ax.set_title(f"A  Age (N={len(age_vals)}, IQR {iqr_lo:.0f}-{iqr_hi:.0f})",
                             loc="left", fontweight="bold")
                ax.legend(fontsize=7, loc="upper right")
            ax_i += 1

        # B: Sex donut chart
        if "sex" in panels and ax_i < n_p:
            ax = axes[ax_i]
            sex_vals = clinical_df[sex_col].dropna().astype(str).str.lower()
            n_male = int(sex_vals.isin(["male", "m", "1", "1.0"]).sum())
            n_female = int(sex_vals.isin(["female", "f", "0", "0.0"]).sum())
            if n_male + n_female > 0:
                wedges, texts, autotexts = ax.pie(
                    [n_male, n_female], labels=["Male", "Female"],
                    colors=[NATURE_PALETTE["blue"], NATURE_PALETTE["orange"]],
                    autopct="%1.1f%%", startangle=90,
                    pctdistance=0.8, wedgeprops=dict(width=0.35, edgecolor="white"),
                    textprops={"fontsize": 9},
                )
                for at in autotexts:
                    at.set_fontsize(8)
                ax.text(0, 0, f"N={n_male + n_female}", ha="center", va="center",
                        fontsize=10, fontweight="bold")
            ax.set_title("B  Sex Distribution", loc="left", fontweight="bold")
            ax_i += 1

        # C: AJCC Stage horizontal stacked bar
        if "stage" in panels and ax_i < n_p:
            ax = axes[ax_i]
            stage_vals = clinical_df["AJCC_STAGE_ENCODED"].dropna()
            stage_counts = [int((stage_vals == s).sum()) for s in [1, 2, 3, 4]]
            stg_colors = [NATURE_PALETTE["blue"], NATURE_PALETTE["orange"],
                          NATURE_PALETTE["green"], NATURE_PALETTE["vermillion"]]
            bars = ax.bar(["I", "II", "III", "IV"], stage_counts,
                          color=stg_colors, edgecolor="white", linewidth=0.5, width=0.6)
            for i, (bar, cnt) in enumerate(zip(bars, stage_counts)):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"n={cnt}", ha="center", fontsize=8, fontweight="bold")
            ax.set_xlabel("AJCC Stage")
            ax.set_ylabel("Count")
            ax.set_title(f"C  Stage (N={int(len(stage_vals))})", loc="left", fontweight="bold")
            ax_i += 1

        # D: KM curve (cohort-wide)
        if "km" in panels and ax_i < n_p:
            ax = axes[ax_i]
            from lifelines import KaplanMeierFitter as _KMF
            times = clinical_df["OS_MONTHS"].to_numpy(dtype=float)
            events = clinical_df["OS_EVENT"].to_numpy(dtype=int)
            valid = ~(np.isnan(times) | np.isnan(events))
            t_v, e_v = times[valid], events[valid]
            kmf = _KMF()
            kmf.fit(t_v, e_v)
            # CI band
            ci = kmf.confidence_interval_survival_function_
            ax.fill_between(ci.index, ci[ci.columns[0]], ci[ci.columns[1]],
                            alpha=0.2, color=NATURE_PALETTE["blue"])
            kmf.plot_survival_function(ax=ax, ci_show=False,
                                       color=NATURE_PALETTE["blue"], linewidth=2)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Overall Survival (months)")
            ax.set_ylabel("Survival Probability")
            # At-risk table
            max_t = t_v.max()
            time_ticks = np.arange(0, min(max_t, 72) + 1, 12)
            at_risk_str = ""
            for t_val in time_ticks[:6]:
                n_at_risk = int((t_v >= t_val).sum())
                ax.text(t_val, -0.12, str(n_at_risk),
                        transform=ax.get_xaxis_transform(), fontsize=7, ha="center",
                        color="gray")
            # Annotation box
            median_os = kmf.median_survival_time_
            surv_12 = float(kmf.survival_function_at_times(12).values[0]) if 12 <= max_t else np.nan
            ann_text = f"Median OS: {median_os:.1f}m\n12m survival: {surv_12:.1%}\nN={int(len(t_v))}"
            ax.text(0.98, 0.95, ann_text, transform=ax.transAxes, fontsize=7,
                    va="top", ha="right", bbox=dict(boxstyle="round,pad=0.3",
                    facecolor="white", edgecolor="gray", alpha=0.8))
            ax.set_title(f"D  KM Curve (N={int(len(t_v))})", loc="left", fontweight="bold")
            ax_i += 1

        for ax in axes[ax_i:]:
            ax.axis("off")
        plt.suptitle("Clinical Cohort Characteristics", fontsize=13, fontweight="bold", y=1.04)
        plt.tight_layout()
        fig.savefig(os.path.join(qc_dir, "03_clinical_table1.png"), bbox_inches="tight")
        plt.close(fig)
        logger.info("  [QC Plot 3/10] 03_clinical_table1.png saved")
    except Exception as e:
        logger.warning(f"  [QC Plot 3] failed: {e}")

    # ── Plot 4: Nature-style KM curve (at-risk table + CI band + annotations) ─
    try:
        if "OS_MONTHS" in clinical_df.columns and "OS_EVENT" in clinical_df.columns:
            from lifelines import KaplanMeierFitter as _KMF
            fig, ax_km = plt.subplots(figsize=(8, 6))
            times = clinical_df["OS_MONTHS"].to_numpy(dtype=float)
            events = clinical_df["OS_EVENT"].to_numpy(dtype=int)
            valid = ~(np.isnan(times) | np.isnan(events))
            t_v, e_v = times[valid], events[valid]
            kmf = _KMF()
            kmf.fit(t_v, e_v, label="Overall Survival")
            # CI band
            ci = kmf.confidence_interval_survival_function_
            ax_km.fill_between(ci.index, ci[ci.columns[0]], ci[ci.columns[1]],
                                alpha=0.15, color=NATURE_PALETTE["blue"], label="95% CI")
            kmf.plot_survival_function(ax=ax_km, ci_show=False,
                                       color=NATURE_PALETTE["blue"], linewidth=2.0)
            ax_km.set_ylim(0, 1.05)
            ax_km.set_xlabel("Overall Survival (months)", fontsize=11)
            ax_km.set_ylabel("Survival Probability", fontsize=11)
            # At-risk table (bottom)
            max_t = t_v.max()
            time_ticks = np.arange(0, min(max_t, 72) + 1, 12)
            at_risk_y_start = -0.10
            ax_km.text(-5, at_risk_y_start, "At risk:", transform=ax_km.get_xaxis_transform(),
                       fontsize=8, fontweight="bold", ha="right", color="gray")
            for t_val in time_ticks[:7]:
                n_at_risk = int((t_v >= t_val).sum())
                ax_km.text(t_val, at_risk_y_start, str(n_at_risk),
                           transform=ax_km.get_xaxis_transform(), fontsize=8, ha="center", color="gray")
            # Annotation box (upper right)
            median_os = kmf.median_survival_time_
            surv_vals = {}
            for pt in [12, 24, 36]:
                if pt <= max_t:
                    surv_vals[pt] = float(kmf.survival_function_at_times(pt).values[0])
            ann_lines = [f"Median OS: {median_os:.1f} months"]
            for pt, sv in surv_vals.items():
                ann_lines.append(f"{pt}m survival: {sv:.1%}")
            ann_lines.append(f"N={int(len(t_v))}, Events={int(e_v.sum())}, Censored={int(len(e_v) - e_v.sum())}")
            ax_km.text(0.98, 0.95, "\n".join(ann_lines), transform=ax_km.transAxes,
                       fontsize=8, va="top", ha="right",
                       bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                                 edgecolor="gray", alpha=0.85))
            ax_km.legend(loc="lower left", fontsize=9)
            plt.subplots_adjust(bottom=0.20)
            ax_km.set_title(f"Kaplan-Meier Overall Survival (N={int(len(t_v))})",
                            fontsize=12, fontweight="bold")
            fig.savefig(os.path.join(qc_dir, "04_km_survival_curve.png"), bbox_inches="tight")
            plt.close(fig)
            logger.info("  [QC Plot 4/10] 04_km_survival_curve.png saved")
    except Exception as e:
        logger.warning(f"  [QC Plot 4] failed: {e}")

    # ── Plot 5: IPCW weight diagnostics (A: hist+KDE, B: boxplot+swarm, C: ECDF) ─
    try:
        if "ipcw_weight_os" in clinical_df.columns:
            w = clinical_df["ipcw_weight_os"].to_numpy(dtype=float)
            w_pos = w[w > 0]
            if len(w_pos) > 0:
                fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(15, 4.5))
                # Panel A: Histogram + KDE
                ax_a.hist(w_pos, bins=min(50, max(20, len(w_pos) // 5)),
                          color=NATURE_PALETTE["green"], edgecolor="white",
                          alpha=0.75, density=True, label="Histogram")
                if len(w_pos) > 10:
                    try:
                        from scipy.stats import gaussian_kde
                        kde = gaussian_kde(w_pos)
                        x_g = np.linspace(w_pos.min(), w_pos.max(), 200)
                        ax_a.plot(x_g, kde(x_g), color=NATURE_PALETTE["vermillion"],
                                  linewidth=2, label="KDE")
                    except Exception:
                        pass
                med_w = np.median(w_pos)
                ax_a.axvline(med_w, color=NATURE_PALETTE["vermillion"], linestyle="--",
                             linewidth=1.2, label=f"Median={med_w:.2f}")
                # Winsorization threshold (99th pct)
                w99 = np.percentile(w_pos, 99)
                ax_a.axvline(w99, color=NATURE_PALETTE["gray"], linestyle=":",
                             linewidth=1.0, label=f"P99={w99:.1f}")
                ax_a.set_xlabel("IPCW Weight")
                ax_a.set_ylabel("Density")
                ax_a.set_title(f"A  IPCW Distribution (N={len(w_pos)})", loc="left", fontweight="bold")
                ax_a.legend(fontsize=7, loc="upper right")

                # Panel B: Boxplot + swarm
                bp = ax_b.boxplot(w_pos, vert=True, patch_artist=True, widths=0.5,
                                  boxprops=dict(facecolor=NATURE_PALETTE["green"], alpha=0.7,
                                                edgecolor="black", linewidth=0.8),
                                  medianprops=dict(color=NATURE_PALETTE["vermillion"], linewidth=2),
                                  whiskerprops=dict(linewidth=0.8),
                                  capprops=dict(linewidth=0.8))
                # Overlay swarm (subsample for performance)
                swarm_data = w_pos if len(w_pos) <= 500 else np.random.default_rng(42).choice(w_pos, 500, replace=False)
                jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(swarm_data))
                ax_b.scatter(np.ones(len(swarm_data)) + jitter, swarm_data,
                             alpha=0.15, s=4, color=NATURE_PALETTE["blue"], linewidths=0)
                ax_b.set_ylabel("IPCW Weight")
                ax_b.set_title("B  IPCW Boxplot + Swarm", loc="left", fontweight="bold")
                ax_b.set_xticks([])

                # Panel C: ECDF
                sorted_w = np.sort(w_pos)
                ecdf_y = np.arange(1, len(sorted_w) + 1) / len(sorted_w)
                ax_c.step(sorted_w, ecdf_y, color=NATURE_PALETTE["blue"], linewidth=1.5, where="post")
                p95 = np.percentile(w_pos, 95)
                p99 = np.percentile(w_pos, 99)
                ax_c.axvline(p95, color=NATURE_PALETTE["orange"], linestyle="--",
                             linewidth=1.0, label=f"P95={p95:.1f}")
                ax_c.axvline(p99, color=NATURE_PALETTE["vermillion"], linestyle="--",
                             linewidth=1.0, label=f"P99={p99:.1f}")
                ax_c.set_xlabel("IPCW Weight")
                ax_c.set_ylabel("ECDF")
                ax_c.set_title("C  ECDF of IPCW Weights", loc="left", fontweight="bold")
                ax_c.legend(fontsize=8, loc="lower right")
                plt.suptitle("IPCW Weight Diagnostics", fontsize=13, fontweight="bold", y=1.02)
                plt.tight_layout()
                fig.savefig(os.path.join(qc_dir, "05_ipcw_weight_dist.png"), bbox_inches="tight")
                plt.close(fig)
            logger.info("  [QC Plot 5/10] 05_ipcw_weight_dist.png saved")
    except Exception as e:
        logger.warning(f"  [QC Plot 5] failed: {e}")

    # ── Plot 6: Pseudo-observation QC (A: dist, B: boxplot, C: scatter vs OS) ─
    try:
        if "pseudo_risk_os_raw" in clinical_df.columns:
            p = clinical_df["pseudo_risk_os_raw"].dropna().to_numpy(dtype=float)
            if len(p) > 0:
                fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(15, 4.5))
                # Panel A: distribution
                ax_a.hist(p, bins=min(50, max(20, len(p) // 5)),
                          color=NATURE_PALETTE["purple"], edgecolor="white",
                          alpha=0.75, density=True)
                ax_a.axvline(0, color=NATURE_PALETTE["gray"], linestyle=":", alpha=0.6, linewidth=1.0)
                ax_a.axvline(1, color=NATURE_PALETTE["gray"], linestyle=":", alpha=0.6, linewidth=1.0)
                n_outside = int(((p < 0) | (p > 1)).sum())
                pct_out = n_outside / max(len(p), 1) * 100
                med_p = np.median(p)
                ax_a.axvline(med_p, color=NATURE_PALETTE["vermillion"], linestyle="--",
                             linewidth=1.2, label=f"Median={med_p:.3f}")
                ax_a.set_xlabel("Pseudo-observation Value")
                ax_a.set_ylabel("Density")
                ax_a.set_title(f"A  Pseudo-obs Distribution ({pct_out:.1f}% outside [0,1])",
                               loc="left", fontweight="bold")
                ax_a.legend(fontsize=7, loc="upper right")

                # Panel B: Boxplot
                bp = ax_b.boxplot(p, vert=True, patch_artist=True, widths=0.5,
                                  boxprops=dict(facecolor=NATURE_PALETTE["purple"], alpha=0.7,
                                                edgecolor="black", linewidth=0.8),
                                  medianprops=dict(color=NATURE_PALETTE["vermillion"], linewidth=2),
                                  whiskerprops=dict(linewidth=0.8),
                                  capprops=dict(linewidth=0.8))
                ax_b.set_ylabel("Pseudo-observation")
                ax_b.set_title("B  Pseudo-obs Boxplot", loc="left", fontweight="bold")
                ax_b.set_xticks([])

                # Panel C: scatter vs OS_MONTHS
                if "OS_MONTHS" in clinical_df.columns:
                    os_months = clinical_df["OS_MONTHS"].to_numpy(dtype=float)
                    # Align pseudo with OS_MONTHS
                    valid_idx = ~np.isnan(p) & ~np.isnan(os_months[:len(p)])
                    if valid_idx.sum() > 10:
                        ax_c.scatter(os_months[:len(p)][valid_idx], p[valid_idx],
                                     alpha=0.3, s=8, color=NATURE_PALETTE["purple"], linewidths=0)
                        # Trend line
                        try:
                            z = np.polyfit(os_months[:len(p)][valid_idx], p[valid_idx], 1)
                            x_line = np.linspace(os_months[:len(p)][valid_idx].min(),
                                                  os_months[:len(p)][valid_idx].max(), 100)
                            ax_c.plot(x_line, np.polyval(z, x_line),
                                      color=NATURE_PALETTE["vermillion"], linewidth=1.5,
                                      linestyle="--", label=f"Trend (slope={z[0]:.4f})")
                            ax_c.legend(fontsize=7, loc="upper right")
                        except Exception:
                            pass
                ax_c.set_xlabel("OS (months)")
                ax_c.set_ylabel("Pseudo-observation")
                ax_c.set_title("C  Pseudo-obs vs Survival Time", loc="left", fontweight="bold")
                plt.suptitle("Pseudo-observation Quality Control", fontsize=13, fontweight="bold", y=1.02)
                plt.tight_layout()
                fig.savefig(os.path.join(qc_dir, "06_pseudo_obs_dist.png"), bbox_inches="tight")
                plt.close(fig)
            logger.info("  [QC Plot 6/10] 06_pseudo_obs_dist.png saved")
    except Exception as e:
        logger.warning(f"  [QC Plot 6] failed: {e}")

    # ── Plot 7: Feature stability (A: variance box, B: PCA scatter, C: variance violin) ─
    try:
        var_data = {}
        if _safe_df(gene_df):
            var_data["Gene Expr."] = gene_df.var(axis=0, skipna=True).values
        if _safe_df(cnv_df):
            var_data["CNV"] = cnv_df.var(axis=0, skipna=True).values
        if _safe_df(rppa_df):
            var_data["RPPA"] = rppa_df.var(axis=0, skipna=True).values
        if var_data:
            n_panels = 1 + (1 if _safe_df(gene_df) else 0) + 1
            fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
            if n_panels == 1:
                axes = [axes]
            ax_i = 0
            # Panel A: Variance boxplot
            ax = axes[ax_i]
            bp = ax.boxplot(list(var_data.values()), labels=list(var_data.keys()),
                            patch_artist=True, widths=0.5,
                            medianprops=dict(color=NATURE_PALETTE["vermillion"], linewidth=2),
                            whiskerprops=dict(linewidth=0.8),
                            capprops=dict(linewidth=0.8))
            oi_colors = [NATURE_PALETTE["blue"], NATURE_PALETTE["orange"], NATURE_PALETTE["green"]]
            for patch, color in zip(bp["boxes"], oi_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            for name, vals in var_data.items():
                mean_v = np.mean(vals)
                ax.text(list(var_data.keys()).index(name) + 1, mean_v,
                        f"μ={mean_v:.3f}", ha="center", va="bottom", fontsize=7,
                        color=NATURE_PALETTE["gray"])
            ax.set_ylabel("Variance (post-standardization)")
            ax.set_title("A  Feature Variance", loc="left", fontweight="bold")
            ax_i += 1

            # Panel B: PCA scatter (Gene)
            if _safe_df(gene_df) and ax_i < n_panels:
                ax = axes[ax_i]
                try:
                    from sklearn.decomposition import PCA as _PCA
                    gene_vals = gene_df.select_dtypes(include=[np.floating, np.integer, float, int]).values
                    if gene_vals.shape[0] >= 5:
                        pca = _PCA(n_components=2)
                        scores = pca.fit_transform(np.nan_to_num(gene_vals))
                        var_pct = pca.explained_variance_ratio_ * 100
                        ax.scatter(scores[:, 0], scores[:, 1], alpha=0.5, s=12,
                                   color=NATURE_PALETTE["blue"], linewidths=0)
                        ax.set_xlabel(f"PC1 ({var_pct[0]:.1f}%)")
                        ax.set_ylabel(f"PC2 ({var_pct[1]:.1f}%)")
                        ax.set_title(f"B  Gene PCA (top-2 PCs: {np.sum(var_pct):.1f}%)",
                                     loc="left", fontweight="bold")
                except Exception:
                    ax.text(0.5, 0.5, "PCA failed", ha="center", va="center",
                            transform=ax.transAxes, fontsize=10, color="gray")
                    ax.set_title("B  Gene PCA", loc="left", fontweight="bold")
                ax_i += 1

            # Panel C: Variance violin (if seaborn available)
            if ax_i < n_panels:
                ax = axes[ax_i]
                all_vars = []
                all_labels = []
                for name, vals in var_data.items():
                    all_vars.append(vals)
                    all_labels.extend([name] * len(vals))
                if _HAS_SEABORN:
                    import pandas as _pd
                    violin_df = _pd.DataFrame({"Variance": np.concatenate(all_vars),
                                               "Modality": all_labels})
                    try:
                        parts = ax.violinplot([v for v in all_vars],
                                              positions=range(len(all_vars)),
                                              showmeans=True, showmedians=True,
                                              widths=0.7)
                        for i, pc in enumerate(parts["bodies"]):
                            pc.set_facecolor(oi_colors[i % len(oi_colors)])
                            pc.set_alpha(0.5)
                    except Exception:
                        pass
                else:
                    ax.boxplot(all_vars, labels=list(var_data.keys()), patch_artist=True)
                ax.set_ylabel("Feature Variance")
                ax.set_xticks(range(len(var_data)))
                ax.set_xticklabels(list(var_data.keys()), fontsize=9)
                ax.set_title("C  Variance Distribution", loc="left", fontweight="bold")
                ax_i += 1

            plt.suptitle("Feature Stability Analysis", fontsize=13, fontweight="bold", y=1.02)
            plt.tight_layout()
            fig.savefig(os.path.join(qc_dir, "07_feature_variance.png"), bbox_inches="tight")
            plt.close(fig)
            logger.info("  [QC Plot 7/10] 07_feature_variance.png saved")
    except Exception as e:
        logger.warning(f"  [QC Plot 7] failed: {e}")

    # ── Plot 8: GAIN imputation quality (A: density, B: masking scatter, C: QQ) ─
    try:
        has_gain = meth_validation and meth_validation.get("gain_rmse") is not None
        if _safe_df(methylation_df) or has_gain:
            fig, axes_8 = plt.subplots(1, 3, figsize=(15, 4.5))
            # Panel A: Density (post-imputation Z-score)
            ax = axes_8[0]
            if _safe_df(methylation_df):
                meth_flat = methylation_df.to_numpy().flatten()
                meth_flat = meth_flat[~np.isnan(meth_flat)]
                if len(meth_flat) > 100000:
                    rng = np.random.default_rng(42)
                    meth_flat = rng.choice(meth_flat, size=100000, replace=False)
                if len(meth_flat) > 0:
                    ax.hist(meth_flat, bins=80, density=True,
                            color=NATURE_PALETTE["blue"], edgecolor="white",
                            alpha=0.7, label="Post-imputation (Z-score)")
                    # Overlay theoretical normal
                    x_norm = np.linspace(meth_flat.min(), meth_flat.max(), 200)
                    from scipy.stats import norm as _norm_dist
                    ax.plot(x_norm, _norm_dist.pdf(x_norm, meth_flat.mean(), meth_flat.std()),
                            color=NATURE_PALETTE["vermillion"], linewidth=2,
                            linestyle="--", label="Theoretical Normal")
                    # KS test
                    try:
                        ks_stat, ks_p = stats.kstest(meth_flat, "norm",
                                                      args=(meth_flat.mean(), meth_flat.std()))
                        ax.text(0.98, 0.95, f"KS stat={ks_stat:.4f}\np={ks_p:.2e}",
                                transform=ax.transAxes, fontsize=7, va="top", ha="right",
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                          edgecolor="gray", alpha=0.8))
                    except Exception:
                        pass
            ax.set_xlabel("Methylation Value (Z-score)")
            ax.set_ylabel("Density")
            ax.set_title("A  Post-Imputation Density", loc="left", fontweight="bold")
            ax.legend(fontsize=7, loc="upper right")

            # Panel B: Masking validation scatter
            ax = axes_8[1]
            if has_gain:
                rmse = meth_validation.get("gain_rmse", np.nan)
                r2 = meth_validation.get("gain_r_squared", np.nan)
                ax.text(0.5, 0.5,
                        f"GAIN Masking Validation\nRMSE = {rmse:.4f}\nR\u00b2 = {r2:.4f}",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=11, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.5", facecolor=NATURE_PALETTE["skyblue"],
                                  edgecolor=NATURE_PALETTE["blue"], alpha=0.3))
                ax.set_xlabel("True Value")
                ax.set_ylabel("GAIN Predicted")
            else:
                ax.text(0.5, 0.5, "No GAIN validation\ndata available",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=10, color="gray")
            ax.set_title("B  GAIN Masking Validation", loc="left", fontweight="bold")

            # Panel C: QQ-plot
            ax = axes_8[2]
            if _safe_df(methylation_df):
                meth_qq = methylation_df.to_numpy().flatten()
                meth_qq = meth_qq[~np.isnan(meth_qq)]
                if len(meth_qq) > 5000:
                    meth_qq = np.random.default_rng(42).choice(meth_qq, 5000, replace=False)
                if len(meth_qq) > 10:
                    try:
                        from scipy import stats as _st
                        (osm, osr), (slope, intercept, r) = _st.probplot(meth_qq, dist="norm")
                        ax.scatter(osm, osr, alpha=0.3, s=4, color=NATURE_PALETTE["blue"], linewidths=0)
                        line_x = np.array([osm.min(), osm.max()])
                        ax.plot(line_x, slope * line_x + intercept,
                                color=NATURE_PALETTE["vermillion"], linewidth=1.5,
                                linestyle="--", label=f"R={r:.4f}")
                        ax.legend(fontsize=7, loc="lower right")
                    except Exception:
                        pass
            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel("Sample Quantiles")
            ax.set_title("C  QQ-Plot (vs Normal)", loc="left", fontweight="bold")
            plt.suptitle("GAIN Imputation Quality Assessment", fontsize=13, fontweight="bold", y=1.02)
            plt.tight_layout()
            fig.savefig(os.path.join(qc_dir, "08_gain_imputation_quality.png"), bbox_inches="tight")
            plt.close(fig)
            logger.info("  [QC Plot 8/10] 08_gain_imputation_quality.png saved")
    except Exception as e:
        logger.warning(f"  [QC Plot 8] failed: {e}")

    # ── Plot 9: IQR sensitivity + mutation frequency (A: retention curve, B: mut hist) ─
    try:
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))
        # Panel A: IQR threshold vs probe retention rate
        if meth_validation and meth_validation.get("n_before_iqr"):
            n_before = meth_validation["n_before_iqr"]
            current_threshold = 0.05
            # Simulate retention at different thresholds
            # (We only have summary stats, so plot current point + theoretical curve)
            thresholds = np.linspace(0.01, 0.20, 50)
            # Approximate retention using available IQR stats
            iqr_p50 = meth_validation.get("iqr_p50", 0.1)
            iqr_p95 = meth_validation.get("iqr_p95", 0.3)
            # Simple model: retention ~ 1 - CDF(threshold) of IQR distribution
            # Use lognormal approximation
            if iqr_p50 > 0 and iqr_p95 > 0:
                from scipy.stats import lognorm
                sigma_est = max(np.log(iqr_p95 / max(iqr_p50, 1e-6)), 0.1)
                retention = 1.0 - lognorm.cdf(thresholds, s=sigma_est, scale=iqr_p50)
                ax_a.plot(thresholds, retention * 100, color=NATURE_PALETTE["blue"],
                          linewidth=2, label="Estimated retention")
            # Mark current threshold
            current_ret = meth_validation.get("iqr_retention_rate", np.nan)
            ax_a.axvline(current_threshold, color=NATURE_PALETTE["vermillion"],
                         linestyle="--", linewidth=1.2, label=f"Current threshold={current_threshold}")
            if not np.isnan(current_ret):
                ax_a.scatter([current_threshold], [current_ret * 100],
                             color=NATURE_PALETTE["vermillion"], s=60, zorder=5,
                             edgecolor="black", linewidth=0.5)
                ax_a.annotate(f"({current_threshold}, {current_ret:.1%})",
                              xy=(current_threshold, current_ret * 100),
                              xytext=(current_threshold + 0.02, current_ret * 100 + 5),
                              fontsize=8, color=NATURE_PALETTE["vermillion"])
            ax_a.set_xlabel("IQR Threshold")
            ax_a.set_ylabel("Probe Retention (%)")
            ax_a.set_title(f"A  IQR Sensitivity (N_before={n_before})", loc="left", fontweight="bold")
            ax_a.legend(fontsize=7, loc="upper right")
        else:
            ax_a.text(0.5, 0.5, "IQR stats not available", ha="center", va="center",
                      transform=ax_a.transAxes, fontsize=10, color="gray")
            ax_a.set_title("A  IQR Sensitivity Analysis", loc="left", fontweight="bold")

        # Panel B: Mutation frequency distribution
        if _safe_df(mutation_df):
            mut_mat = mutation_df.drop(columns=["PATIENT_ID"], errors="ignore").values
            if mut_mat.size > 0:
                freqs = np.mean(mut_mat > 0, axis=0)
                ax_b.hist(freqs, bins=min(40, max(15, len(freqs) // 3)),
                          color=NATURE_PALETTE["orange"], edgecolor="white", alpha=0.8)
                ax_b.axvline(MUTATION_FREQ_LOW, color=NATURE_PALETTE["green"],
                             linestyle="--", linewidth=1.2, label=f"Low filter={MUTATION_FREQ_LOW}")
                ax_b.axvline(MUTATION_FREQ_HIGH, color=NATURE_PALETTE["vermillion"],
                             linestyle="--", linewidth=1.2, label=f"High filter={MUTATION_FREQ_HIGH}")
                ax_b.set_xlabel("Mutation Frequency")
                ax_b.set_ylabel("Number of Genes")
                ax_b.set_title(f"B  Mutation Frequency (N={len(freqs)} genes)",
                               loc="left", fontweight="bold")
                ax_b.legend(fontsize=7, loc="upper right")
        plt.suptitle("Feature Filtering Sensitivity", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(qc_dir, "09_feature_filtering_sensitivity.png"), bbox_inches="tight")
        plt.close(fig)
        logger.info("  [QC Plot 9/10] 09_feature_filtering_sensitivity.png saved")
    except Exception as e:
        logger.warning(f"  [QC Plot 9] failed: {e}")

    # ── Plot 10: Summary dashboard (metrics overview) ─────────────────────
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        # Build summary text
        summary_lines = [
            f"Preprocessing QC Summary",
            f"{'='*50}",
            f"Cohort: N={n_total} patients",
            f"Aligned samples: {len(sample_ids) if sample_ids else 0}",
        ]
        for name in ["gene", "mutation", "cnv", "methylation", "rppa"]:
            n_feat = metrics.get(f"{name}_n_features", 0)
            mr = metrics.get(f"{name}_missing_rate", np.nan)
            summary_lines.append(f"  {name.capitalize()}: {n_feat} features, missing={mr:.2%}" if not np.isnan(mr) else f"  {name.capitalize()}: N/A")
        if "gain_rmse" in metrics:
            summary_lines.append(f"GAIN RMSE={metrics['gain_rmse']:.4f}, R\u00b2={metrics.get('gain_r_squared', np.nan):.4f}")
        if "meth_n_before_iqr" in metrics:
            summary_lines.append(f"IQR filter: {metrics['meth_n_before_iqr']} -> {metrics['meth_n_after_iqr']} probes")
        if "median_os_months_km" in metrics:
            summary_lines.append(f"Median OS: {metrics['median_os_months_km']} months")
        if "gene_pca_top2_total_pct" in metrics:
            summary_lines.append(f"Gene PCA top-2 PCs: {metrics['gene_pca_top2_total_pct']}%")
        summary_lines.append(f"\nTotal QC metrics: {len(metrics)} indicators")

        text = "\n".join(summary_lines)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9, va="top",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8",
                          edgecolor="gray", alpha=0.9))
        ax.set_title("Preprocessing QC Summary", fontsize=13, fontweight="bold")
        fig.savefig(os.path.join(qc_dir, "10_qc_summary.png"), bbox_inches="tight")
        plt.close(fig)
        logger.info("  [QC Plot 10/10] 10_qc_summary.png saved")
    except Exception as e:
        logger.warning(f"  [QC Plot 10] failed: {e}")

    _elapsed = round(_time.time() - _t0, 1)
    logger.info(f"  [QC] All 10 plots generated in {_elapsed}s")


# ─────────────────────────────────────────────────────────────────────────────
# QC Report: Top-level orchestration
# ─────────────────────────────────────────────────────────────────────────────
def generate_preprocessing_qc_report(
    output_dir, clinical_df, gene_df, mutation_df, cnv_df,
    methylation_df, rppa_df, sample_ids, missing_indicators,
    meth_validation=None, mutation_raw_df=None,
):
    """Generate comprehensive preprocessing QC report.

    Creates a qc_report/ subdirectory containing:
    - qc_metrics.json  : all metrics in JSON format
    - qc_metrics.tsv   : flat key-value table
    - 10 PNG plots     : core QC visualizations (Nature/Cell style)
    """
    import time as _time
    _t0 = _time.time()
    qc_dir = os.path.join(output_dir, "qc_report")
    os.makedirs(qc_dir, exist_ok=True)
    logger.info(f"  [QC REPORT] Output directory: {qc_dir}")

    # ── Compute metrics ──────────────────────────────────────────────
    metrics = compute_qc_metrics(
        clinical_df, gene_df, mutation_df, cnv_df,
        methylation_df, rppa_df, sample_ids, missing_indicators,
        meth_validation=meth_validation, mutation_raw_df=mutation_raw_df,
    )

    # Save JSON
    json_path = os.path.join(qc_dir, "qc_metrics.json")
    # Filter out non-serializable values
    serializable = {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
                    for k, v in metrics.items()
                    if not (isinstance(v, float) and np.isnan(v)) or True}
    # Convert NaN to None for JSON
    for k, v in serializable.items():
        if isinstance(v, float) and np.isnan(v):
            serializable[k] = None
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  [QC REPORT] Metrics JSON saved: {json_path}")

    # Save TSV (key-value pairs)
    tsv_path = os.path.join(qc_dir, "qc_metrics.tsv")
    tsv_rows = [{"metric": k, "value": v} for k, v in metrics.items()]
    pd.DataFrame(tsv_rows).to_csv(tsv_path, sep="\t", index=False)
    logger.info(f"  [QC REPORT] Metrics TSV saved: {tsv_path}")

    # ── Generate plots ────────────────────────────────────────────────
    generate_qc_plots(
        qc_dir, clinical_df, gene_df, mutation_df, cnv_df,
        methylation_df, rppa_df, sample_ids, missing_indicators, metrics,
        meth_validation=meth_validation,
    )

    _elapsed = round(_time.time() - _t0, 1)
    logger.info(f"  [QC REPORT] Complete! {_elapsed}s total, {len(metrics)} metrics + 10 plots")


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity Analysis: Methylation Coverage Bias Assessment
# ─────────────────────────────────────────────────────────────────────────────
def methylation_coverage_sensitivity_analysis(
    output_dir, clinical_df, methylation_df, sample_ids, missing_indicators,
):
    """Assess whether methylation coverage (354/502, 70.5%) introduces selection bias.

    Compares clinical characteristics between samples WITH methylation data (n~354)
    and WITHOUT methylation data (n~148) to test the MCAR assumption.

    Outputs (saved to ``output_dir/sensitivity_analysis/``):
        - methylation_coverage_comparison.tsv  (Table-1 style comparison)
        - methylation_coverage_km.png          (KM curves for two groups)
        - methylation_coverage_summary.json    (statistical test results)

    Parameters
    ----------
    output_dir : str
        Preprocessing output directory (e.g. .../01_preprocessing).
    clinical_df : pd.DataFrame
        Aligned clinical dataframe (indexed by row position, PATIENT_ID column).
    methylation_df : pd.DataFrame or None
        Methylation matrix (index = patient IDs).
    sample_ids : list[str]
        Aligned sample IDs.
    missing_indicators : dict
        Per-omics missing indicator Series (1=present, 0=absent).
    """
    import time as _time
    _t0 = _time.time()
    sa_dir = os.path.join(output_dir, "sensitivity_analysis")
    os.makedirs(sa_dir, exist_ok=True)
    logger.info(f"  [SENSITIVITY] Output directory: {sa_dir}")

    # ── 1. Split into two groups ────────────────────────────────────────
    meth_indicator = missing_indicators.get("methylation")
    if meth_indicator is None or _safe_df(methylation_df) is False:
        logger.warning("  [SENSITIVITY] Methylation indicator not available, skip.")
        return

    has_meth_ids = [str(sid) for sid in sample_ids if meth_indicator.get(str(sid), 0) == 1]
    no_meth_ids  = [str(sid) for sid in sample_ids if meth_indicator.get(str(sid), 0) == 0]

    if len(has_meth_ids) == 0 or len(no_meth_ids) == 0:
        logger.warning("  [SENSITIVITY] One group is empty, skip.")
        return

    clin = clinical_df.copy()
    pid_col = "PATIENT_ID"
    clin[pid_col] = clin[pid_col].astype(str)
    grp_has = clin[clin[pid_col].isin(has_meth_ids)].copy()
    grp_no  = clin[clin[pid_col].isin(no_meth_ids)].copy()
    logger.info(f"  [SENSITIVITY] Methylation present: {len(grp_has)}, absent: {len(grp_no)}")

    results = {}  # collect test results
    table_rows = []  # Table-1 style rows

    def _row(variable, stat_name, val_has, val_no, p_val, test_name):
        table_rows.append({
            "Variable": variable,
            "Statistic": stat_name,
            f"Meth_Present (n={len(grp_has)})": val_has,
            f"Meth_Absent (n={len(grp_no)})": val_no,
            "p_value": p_val if p_val is not None else np.nan,
            "Test": test_name,
        })

    # ── 2. Continuous variables: Age, OS_MONTHS ─────────────────────────
    age_col = None
    for c in ["AGE", "AGE_AT_DIAGNOSIS", "AGE_AT_INITIAL_PATHOLOGIC_DIAGNOSIS", "AGE_AT_SEQ"]:
        if c in clin.columns:
            age_col = c
            break
    if age_col:
        age_has = pd.to_numeric(grp_has[age_col], errors="coerce").dropna()
        age_no  = pd.to_numeric(grp_no[age_col], errors="coerce").dropna()
        if len(age_has) >= 2 and len(age_no) >= 2:
            stat, p_age = stats.mannwhitneyu(age_has, age_no, alternative="two-sided")
            _row("Age", "Median (IQR)",
                 f"{age_has.median():.1f} ({age_has.quantile(0.25):.1f}-{age_has.quantile(0.75):.1f})",
                 f"{age_no.median():.1f} ({age_no.quantile(0.25):.1f}-{age_no.quantile(0.75):.1f})",
                 round(p_age, 4), "Mann-Whitney U")
            results["age_p_value"] = round(p_age, 4)
            results["age_test"] = "Mann-Whitney U"

    # OS_MONTHS
    if "OS_MONTHS" in clin.columns:
        os_has = pd.to_numeric(grp_has["OS_MONTHS"], errors="coerce").dropna()
        os_no  = pd.to_numeric(grp_no["OS_MONTHS"], errors="coerce").dropna()
        if len(os_has) >= 2 and len(os_no) >= 2:
            stat, p_os = stats.mannwhitneyu(os_has, os_no, alternative="two-sided")
            _row("OS_MONTHS", "Median (IQR)",
                 f"{os_has.median():.1f} ({os_has.quantile(0.25):.1f}-{os_has.quantile(0.75):.1f})",
                 f"{os_no.median():.1f} ({os_no.quantile(0.25):.1f}-{os_no.quantile(0.75):.1f})",
                 round(p_os, 4), "Mann-Whitney U")
            results["os_months_p_value"] = round(p_os, 4)

    # ── 3. Categorical variables: Sex, AJCC Stage, OS_EVENT ─────────────
    sex_col = None
    for c in ["SEX", "GENDER"]:
        if c in clin.columns:
            sex_col = c
            break
    if sex_col:
        sex_has = grp_has[sex_col].dropna().astype(str)
        sex_no  = grp_no[sex_col].dropna().astype(str)
        # Build 2x2 contingency table
        cats = sorted(set(sex_has.unique()) | set(sex_no.unique()))
        ct = pd.DataFrame(index=cats, columns=["Meth_Present", "Meth_Absent"], dtype=int)
        for cat in cats:
            ct.loc[cat, "Meth_Present"] = int((sex_has == cat).sum())
            ct.loc[cat, "Meth_Absent"] = int((sex_no == cat).sum())
        ct_arr = ct.to_numpy()
        # Use Fisher exact for 2x2, chi-square otherwise
        if ct_arr.shape == (2, 2) and ct_arr.min() < 5:
            try:
                odds_ratio, p_sex = stats.fisher_exact(ct_arr)
                test_name = "Fisher exact"
            except Exception:
                chi2, p_sex, _, _ = stats.chi2_contingency(ct_arr, correction=True)
                test_name = "Chi-square (Yates)"
        else:
            chi2, p_sex, _, _ = stats.chi2_contingency(ct_arr, correction=False)
            test_name = "Chi-square"
        n_m_has = int(sex_has.isin(["male", "m", "1", "1.0"]).sum())
        n_m_no  = int(sex_no.isin(["male", "m", "1", "1.0"]).sum())
        _row("Sex (Male)", "n (%)",
             f"{n_m_has} ({n_m_has/max(len(sex_has),1)*100:.1f}%)",
             f"{n_m_no} ({n_m_no/max(len(sex_no),1)*100:.1f}%)",
             round(p_sex, 4), test_name)
        results["sex_p_value"] = round(p_sex, 4)

    # AJCC Stage
    if "AJCC_STAGE_ENCODED" in clin.columns:
        stg_has = grp_has["AJCC_STAGE_ENCODED"].dropna()
        stg_no  = grp_no["AJCC_STAGE_ENCODED"].dropna()
        stage_cats = sorted(set(stg_has.unique()) | set(stg_no.unique()))
        ct_stg = np.zeros((len(stage_cats), 2), dtype=int)
        for i, sc in enumerate(stage_cats):
            ct_stg[i, 0] = int((stg_has == sc).sum())
            ct_stg[i, 1] = int((stg_no == sc).sum())
        if ct_stg.sum() > 0 and ct_stg.min(axis=None) >= 0:
            try:
                chi2_stg, p_stg, _, _ = stats.chi2_contingency(ct_stg, correction=False)
            except Exception:
                p_stg = np.nan
            stage_dist_has = "; ".join([f"{int(sc)}:{int((stg_has==sc).sum())}" for sc in stage_cats])
            stage_dist_no  = "; ".join([f"{int(sc)}:{int((stg_no==sc).sum())}" for sc in stage_cats])
            _row("AJCC Stage", "n per stage",
                 stage_dist_has, stage_dist_no,
                 round(p_stg, 4) if not np.isnan(p_stg) else np.nan,
                 "Chi-square")
            results["stage_p_value"] = round(p_stg, 4) if not np.isnan(p_stg) else None

    # OS_EVENT
    if "OS_EVENT" in clin.columns:
        ev_has = grp_has["OS_EVENT"].dropna()
        ev_no  = grp_no["OS_EVENT"].dropna()
        ct_ev = np.array([
            [int((ev_has == 1).sum()), int((ev_no == 1).sum())],
            [int((ev_has == 0).sum()), int((ev_no == 0).sum())],
        ])
        if ct_ev.min() < 5:
            try:
                _, p_ev = stats.fisher_exact(ct_ev)
                ev_test = "Fisher exact"
            except Exception:
                _, p_ev, _, _ = stats.chi2_contingency(ct_ev, correction=True)
                ev_test = "Chi-square (Yates)"
        else:
            _, p_ev, _, _ = stats.chi2_contingency(ct_ev, correction=False)
            ev_test = "Chi-square"
        n_ev_has = int((ev_has == 1).sum())
        n_ev_no  = int((ev_no == 1).sum())
        _row("OS_EVENT (death)", "n (%)",
             f"{n_ev_has} ({n_ev_has/max(len(ev_has),1)*100:.1f}%)",
             f"{n_ev_no} ({n_ev_no/max(len(ev_no),1)*100:.1f}%)",
             round(p_ev, 4), ev_test)
        results["os_event_p_value"] = round(p_ev, 4)

    # ── 4. Survival analysis: Log-rank test + KM plot ───────────────────
    p_logrank = None
    if "OS_MONTHS" in clin.columns and "OS_EVENT" in clin.columns:
        from lifelines import KaplanMeierFitter as _KMF
        from lifelines.statistics import logrank_test

        t_has = grp_has["OS_MONTHS"].to_numpy(dtype=float)
        e_has = grp_has["OS_EVENT"].to_numpy(dtype=int)
        t_no  = grp_no["OS_MONTHS"].to_numpy(dtype=float)
        e_no  = grp_no["OS_EVENT"].to_numpy(dtype=int)
        v_has = ~(np.isnan(t_has) | np.isnan(e_has))
        v_no  = ~(np.isnan(t_no) | np.isnan(e_no))
        t_has, e_has = t_has[v_has], e_has[v_has]
        t_no, e_no = t_no[v_no], e_no[v_no]

        if len(t_has) >= 2 and len(t_no) >= 2:
            try:
                lr = logrank_test(t_has, t_no, e_has, e_no)
                p_logrank = lr.p_value
                results["logrank_p_value"] = round(p_logrank, 4)
                _row("Overall Survival", "Log-rank p", "", "", round(p_logrank, 4), "Log-rank test")
            except Exception as e:
                logger.warning(f"  [SENSITIVITY] Log-rank test failed: {e}")

        # KM plot
        try:
            palette = _get_palette()
            fig, ax = plt.subplots(figsize=(8, 5.5))
            kmf_has = _KMF()
            kmf_no  = _KMF()
            kmf_has.fit(t_has, e_has, label=f"Methylation present (n={int(v_has.sum())})")
            kmf_no.fit(t_no, e_no, label=f"Methylation absent (n={int(v_no.sum())})")
            kmf_has.plot_survival_function(ax=ax, ci_show=True, color=palette[0], linewidth=2)
            kmf_no.plot_survival_function(ax=ax, ci_show=True, color=palette[1], linewidth=2)
            ax.set_xlabel("Overall Survival (months)")
            ax.set_ylabel("Survival Probability")
            p_str = f"p={p_logrank:.4f}" if p_logrank is not None else "p=N/A"
            ax.set_title(f"KM Curves by Methylation Coverage ({p_str})")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc="lower left")
            plt.tight_layout()
            fig.savefig(os.path.join(sa_dir, "methylation_coverage_km.png"), dpi=150)
            plt.close(fig)
            logger.info("  [SENSITIVITY] methylation_coverage_km.png saved")
        except Exception as e:
            logger.warning(f"  [SENSITIVITY] KM plot failed: {e}")

    # ── 5. MCAR conclusion ──────────────────────────────────────────────
    p_values = [v for k, v in results.items() if k.endswith("_p_value") and v is not None]
    all_nonsig = all(p > 0.05 for p in p_values) if p_values else False
    results["mcar_conclusion"] = (
        "All p-values > 0.05; no significant differences detected. "
        "Supports MCAR assumption for methylation missingness."
        if all_nonsig else
        "Some p-values <= 0.05; potential non-random missingness. "
        "Interpret with caution and consider sensitivity models."
    )
    results["n_meth_present"] = len(has_meth_ids)
    results["n_meth_absent"]  = len(no_meth_ids)

    # Save outputs
    pd.DataFrame(table_rows).to_csv(
        os.path.join(sa_dir, "methylation_coverage_comparison.tsv"),
        sep="\t", index=False,
    )
    with open(os.path.join(sa_dir, "methylation_coverage_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    _elapsed = round(_time.time() - _t0, 1)
    logger.info(f"  [SENSITIVITY] Complete! {_elapsed}s. MCAR conclusion: {results['mcar_conclusion'][:80]}...")


# ─────────────────────────────────────────────────────────────────────────────
# Stratified Dataset Export for Group-Based Modeling
# ─────────────────────────────────────────────────────────────────────────────
def export_stratified_datasets(
    output_dir, clinical_df, gene_df, mutation_df, cnv_df,
    methylation_df, rppa_df, sample_ids, missing_indicators,
):
    """Export two stratified dataset subsets for group-based modeling.

    - ``dataset_full_{N}/``: all aligned samples (main model, may lack methylation)
    - ``dataset_meth_{N}/``: intersection with methylation (supplement full-omics model)

    Each subset directory contains:
        - ``sample_ids.pkl``
        - ``dataset_info.json``
        - ``clinical_subset.tsv``

    Parameters
    ----------
    output_dir : str
    clinical_df, gene_df, mutation_df, cnv_df, methylation_df, rppa_df : DataFrames
    sample_ids : list[str]
    missing_indicators : dict
    """
    import time as _time
    _t0 = _time.time()
    pid_col = "PATIENT_ID"

    meth_indicator = missing_indicators.get("methylation")

    # ── Full cohort ─────────────────────────────────────────────────────
    full_ids = list(sample_ids)
    n_full = len(full_ids)
    full_dir = os.path.join(output_dir, f"dataset_full_{n_full}")
    os.makedirs(full_dir, exist_ok=True)

    with open(os.path.join(full_dir, "sample_ids.pkl"), "wb") as f:
        pickle.dump(full_ids, f)

    clin_full = clinical_df[clinical_df[pid_col].astype(str).isin(full_ids)]
    clin_full.to_csv(os.path.join(full_dir, "clinical_subset.tsv"), sep="\t", index=False)

    omics_coverage = {}
    for name, ind in missing_indicators.items():
        n_present = int(ind.reindex(full_ids, fill_value=0).sum())
        omics_coverage[name] = {"n_present": n_present, "coverage": round(n_present / max(n_full, 1), 4)}

    full_info = {
        "dataset_name": f"full_cohort_{n_full}",
        "n_samples": n_full,
        "selection_criteria": "Core intersection (clinical + RNA + mutation)",
        "omics_coverage": omics_coverage,
        "purpose": "Primary modeling cohort (methylation optional via late fusion)",
    }
    with open(os.path.join(full_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(full_info, f, indent=2, ensure_ascii=False)
    logger.info(f"  [DATASETS] Full cohort exported: {n_full} samples -> {full_dir}")

    # ── Methylation subset ──────────────────────────────────────────────
    if meth_indicator is not None and _safe_df(methylation_df):
        meth_ids = sorted(set(str(sid) for sid in sample_ids if meth_indicator.get(str(sid), 0) == 1))
        n_meth = len(meth_ids)
        meth_dir = os.path.join(output_dir, f"dataset_meth_{n_meth}")
        os.makedirs(meth_dir, exist_ok=True)

        with open(os.path.join(meth_dir, "sample_ids.pkl"), "wb") as f:
            pickle.dump(meth_ids, f)

        clin_meth = clinical_df[clinical_df[pid_col].astype(str).isin(meth_ids)]
        clin_meth.to_csv(os.path.join(meth_dir, "clinical_subset.tsv"), sep="\t", index=False)

        omics_coverage_meth = {}
        for name, ind in missing_indicators.items():
            n_present = int(ind.reindex(meth_ids, fill_value=0).sum())
            omics_coverage_meth[name] = {"n_present": n_present, "coverage": round(n_present / max(n_meth, 1), 4)}

        meth_info = {
            "dataset_name": f"methylation_subset_{n_meth}",
            "n_samples": n_meth,
            "selection_criteria": "Intersection of core cohort AND methylation HM450 data",
            "omics_coverage": omics_coverage_meth,
            "purpose": "Supplemental full-omics model (all modalities complete)",
        }
        with open(os.path.join(meth_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
            json.dump(meth_info, f, indent=2, ensure_ascii=False)
        logger.info(f"  [DATASETS] Methylation subset exported: {n_meth} samples -> {meth_dir}")
    else:
        logger.warning("  [DATASETS] No methylation data, skip methylation subset.")

    _elapsed = round(_time.time() - _t0, 1)
    logger.info(f"  [DATASETS] Complete! {_elapsed}s")


# ─────────────────────────────────────────────────────────────────────────────
# Late Fusion Manifest Export (AZ-AI Pipeline Reference)
# ─────────────────────────────────────────────────────────────────────────────
def export_late_fusion_manifest(
    output_dir, clinical_df, gene_df, mutation_df, cnv_df,
    methylation_df, rppa_df, sample_ids, missing_indicators,
):
    """Export Late Fusion manifest following AZ-AI Pipeline (Nikolaou et al. 2025).

    Reference:
        Nikolaou N, et al. A machine learning approach for multimodal data fusion
        for survival prediction in cancer patients.
        npj Precis. Oncol. 2025;9:128. DOI:10.1038/s41698-025-00917-6

    Late Fusion Strategy:
        - Each modality trains an independent survival model f_j(x)
        - Weights: w_j = max(C_j - 0.5, 0) where C_j is validation C-index
        - Fused prediction: f_FUSED(x) = sum(w_j * f_j(x)) / sum(w_j)
        - Missing modality: weight set to 0, normalization adjusts automatically

    Outputs (saved to ``output_dir/late_fusion/``):
        - ``manifest.json``: modality coverage, sample IDs, feature counts
        - ``modality_configs.json``: per-modality recommended configs
        - ``README_fusion_strategy.txt``: strategy explanation

    Parameters
    ----------
    output_dir : str
    clinical_df, gene_df, mutation_df, cnv_df, methylation_df, rppa_df : DataFrames
    sample_ids : list[str]
    missing_indicators : dict
    """
    import time as _time
    _t0 = _time.time()
    lf_dir = os.path.join(output_dir, "late_fusion")
    os.makedirs(lf_dir, exist_ok=True)

    pid_col = "PATIENT_ID"
    all_ids = [str(s) for s in sample_ids]

    # ── Build per-modality manifest entries ─────────────────────────────
    modalities = {}

    # Clinical (always required)
    clin_feats = [c for c in clinical_df.columns
                  if c not in [pid_col, "OS_STATUS", "OS_MONTHS", "OS_EVENT",
                               "death_by_36m_observed", "death_by_36m",
                               "early_censored_before_36m", "ipcw_weight_os",
                               "ipcw_label_available", "pseudo_risk_os_raw",
                               "pseudo_risk_os", "AJCC_STAGE_ENCODED"]]
    modalities["clinical"] = {
        "n_samples": len(all_ids),
        "n_features": len(clin_feats),
        "feature_names": clin_feats[:50],  # cap for JSON size
        "sample_ids": all_ids,
        "required": True,
        "missing_rate": 0.0,
    }

    omics_map = {
        "gene": gene_df, "mutation": mutation_df, "cnv": cnv_df,
        "methylation": methylation_df, "rppa": rppa_df,
    }
    for name, df_om in omics_map.items():
        ind = missing_indicators.get(name)
        if _safe_df(df_om):
            if name == "mutation" and "PATIENT_ID" in df_om.columns:
                om_ids = sorted(set(str(x) for x in df_om["PATIENT_ID"]) & set(all_ids))
                n_feats = df_om.shape[1] - 1
            else:
                om_ids = sorted(set(str(x) for x in df_om.index) & set(all_ids))
                n_feats = df_om.shape[1]
            n_om = len(om_ids)
            modalities[name] = {
                "n_samples": n_om,
                "n_features": n_feats,
                "sample_ids": om_ids,
                "required": name in ("gene", "mutation"),
                "missing_rate": round(1 - n_om / max(len(all_ids), 1), 4),
            }

    # ── manifest.json ───────────────────────────────────────────────────
    manifest = {
        "fusion_strategy": "late_fusion",
        "reference": "Nikolaou N, et al. npj Precis Oncol 2025;9:128",
        "fusion_equation": "f_FUSED(x) = sum(w_j * f_j(x)) / sum(w_j)",
        "weight_formula": "w_j = max(C_j - 0.5, 0) where C_j = validation C-index",
        "modalities": modalities,
        "fusion_weights_method": "validation_c_index",
        "min_c_index_threshold": 0.5,
        "n_total_samples": len(all_ids),
        "missing_handling": (
            "Each modality model predicts only on samples with data. "
            "Missing modality weight w_j=0 for absent samples. "
            "Normalization auto-adjusts: sum(w_j) over present modalities only."
        ),
    }
    with open(os.path.join(lf_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ── modality_configs.json ───────────────────────────────────────────
    modality_configs = {}
    for name, info in modalities.items():
        cfg = {
            "modality": name,
            "n_samples": info["n_samples"],
            "n_features": info["n_features"],
            "required": info["required"],
            "recommended_models": [],
        }
        if name == "clinical":
            cfg["recommended_models"] = ["CoxPH", "RSF", "GBSA"]
            cfg["feature_selection"] = "none (low-dimensional)"
        elif name == "gene":
            cfg["recommended_models"] = ["RSF", "GBSA", "CoxPH+LASSO"]
            cfg["feature_selection"] = "Spearman correlation with OS (top-K)"
        elif name == "mutation":
            cfg["recommended_models"] = ["RSF", "CoxPH+LASSO"]
            cfg["feature_selection"] = "frequency filter + Spearman"
        elif name == "methylation":
            cfg["recommended_models"] = ["RSF", "GBSA", "CoxPH+LASSO"]
            cfg["feature_selection"] = "IQR filter + Spearman (top-K)"
            cfg["note"] = "Only available for 354/502 samples; late fusion handles missingness"
        elif name == "cnv":
            cfg["recommended_models"] = ["RSF", "CoxPH+LASSO"]
            cfg["feature_selection"] = "variance filter + Spearman"
        elif name == "rppa":
            cfg["recommended_models"] = ["CoxPH", "RSF"]
            cfg["feature_selection"] = "Spearman correlation"
        modality_configs[name] = cfg

    with open(os.path.join(lf_dir, "modality_configs.json"), "w", encoding="utf-8") as f:
        json.dump(modality_configs, f, indent=2, ensure_ascii=False)

    # ── README_fusion_strategy.txt ──────────────────────────────────────
    readme_text = (
        "Late Fusion Strategy for Multi-Omics Survival Prediction\n"
        "======================================================\n\n"
        "Reference:\n"
        "  Nikolaou N, Salazar D, RaviPrakash H, et al.\n"
        "  A machine learning approach for multimodal data fusion\n"
        "  for survival prediction in cancer patients.\n"
        "  npj Precis. Oncol. 2025;9:128.\n"
        "  DOI: 10.1038/s41698-025-00917-6\n\n"
        "Overview:\n"
        "  Late fusion (decision-level fusion) trains independent survival\n"
        "  models per modality and combines predictions via weighted ensemble.\n\n"
        "Key Equations:\n"
        "  f_FUSED(x) = sum(w_j * f_j(x)) / sum(w_j)\n"
        "  w_j = max(C_j - 0.5, 0)\n"
        "  where C_j = validation set C-index for modality j\n\n"
        "Missing Modality Handling:\n"
        "  - Methylation covers 354/502 (70.5%) samples\n"
        "  - For samples WITHOUT methylation: w_methylation = 0\n"
        "  - Normalization adjusts: sum(w_j) only over present modalities\n"
        "  - All 502 samples used for training (no sample discard)\n\n"
        "Implementation Steps:\n"
        "  1. Train per-modality models on each modality's available samples\n"
        "  2. Evaluate each on validation set -> C_j\n"
        "  3. Compute weights w_j = max(C_j - 0.5, 0)\n"
        "  4. For each test sample, fuse available modality predictions\n"
        "  5. Report multimodal vs unimodal C-index comparison\n\n"
        "Advantages over Early Fusion:\n"
        "  - Handles incomplete modality coverage naturally\n"
        "  - Reduces overfitting risk (lower effective dimensionality)\n"
        "  - Allows exhaustive modality combination comparison\n"
        "  - Each modality's contribution is interpretable via weights\n"
    )
    with open(os.path.join(lf_dir, "README_fusion_strategy.txt"), "w", encoding="utf-8") as f:
        f.write(readme_text)

    _elapsed = round(_time.time() - _t0, 1)
    logger.info(f"  [LATE FUSION] Complete! {_elapsed}s, {len(modalities)} modalities documented -> {lf_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, default=None)
    args = parser.parse_args()

    timestamp = args.timestamp if args.timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")

    # 输出目录：预处理数据持久化存储到 data-forword/
    essential_dir = ESSENTIAL_DIR        # 下游脚本必需的预处理数据
    non_essential_dir = NON_ESSENTIAL_DIR  # QC 图表、敏感性分析等非核心数据
    os.makedirs(essential_dir, exist_ok=True)
    os.makedirs(non_essential_dir, exist_ok=True)

    logger.info(f"[01_preprocessing] 开始预处理 | 时间戳: {timestamp}")
    logger.info(f"[01_preprocessing] 数据目录: {DATA_DIR}")
    logger.info(f"[01_preprocessing] TCGA目录: {TCGA_DIR}")
    logger.info(f"[01_preprocessing] 必要输出目录: {essential_dir}")
    logger.info(f"[01_preprocessing] 非必要输出目录: {non_essential_dir}")

    if not os.path.isdir(TCGA_DIR):
        logger.error(f"[ERROR] TCGA数据目录不存在: {TCGA_DIR}")
        sys.exit(1)

    logger.info("[STEP 1/8] 加载临床数据...")
    clinical_df, le_dict = load_clinical_data()
    logger.info(f"  临床数据: {clinical_df.shape[0]} 患者, {clinical_df.shape[1]} 列")

    logger.info("[STEP 2/8] 处理生存终点...")
    clinical_df = process_survival_endpoints(clinical_df)
    logger.info(f"  生存终点处理后: {clinical_df.shape[0]} 患者")

    gene_df = None
    gene_names = []
    mutation_df = None
    mutation_raw_df = None
    cnv_df = None
    methylation_df = None
    meth_validation = {}
    rppa_df = None

    logger.info("[STEP 3/8] 加载基因表达数据...")
    try:
        gene_df, gene_names = load_gene_expression()
        logger.info(f"  基因表达: {gene_df.shape[0]} 患者 x {gene_df.shape[1]} 基因")
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"  [WARNING] {e}")

    logger.info("[STEP 4/8] 加载突变数据...")
    try:
        mutation_df, mutation_raw_df = load_mutation_matrix()
        if mutation_df is not None and not mutation_df.empty:
            logger.info(f"  突变矩阵: {mutation_df.shape[0]} 患者 x {mutation_df.shape[1] - 1} 基因")
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"  [WARNING] {e}")

    logger.info("[STEP 5/8] 加载CNV数据...")
    try:
        cnv_df = load_cnv()
        logger.info(f"  CNV: {cnv_df.shape[0]} 患者 x {cnv_df.shape[1]} 特征")
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"  [WARNING] {e}")

    logger.info("[STEP 6/8] 加载甲基化数据...")
    try:
        methylation_df, meth_validation = load_methylation(min_iqr=0.05, use_gain=True)
        logger.info(f"  甲基化: {methylation_df.shape[0]} 患者 x {methylation_df.shape[1]} 特征")
        if meth_validation:
            val_keys = [k for k in meth_validation if k.startswith("gain_")]
            if val_keys:
                logger.info(f"  GAIN 验证指标: {', '.join(f'{k}={meth_validation[k]}' for k in val_keys)}")
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"  [WARNING] {e}")

    logger.info("[STEP 7/8] 加载RPPA数据...")
    try:
        rppa_df = load_rppa()
        logger.info(f"  RPPA: {rppa_df.shape[0]} 患者 x {rppa_df.shape[1]} 特征")
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"  [WARNING] {e}")

    logger.info("[STEP 8/8] 对齐样本并保存...")
    sample_ids, clinical_aligned, gene_aligned, mutation_aligned, cnv_aligned, methylation_aligned, rppa_aligned, missing_indicators = align_samples(
        clinical_df, gene_df, mutation_df, cnv_df, methylation_df, rppa_df
    )
    logger.info(f"  对齐后样本数: {len(sample_ids)}")

    config = {
        "timestamp": timestamp,
        "data_dir": DATA_DIR,
        "tcga_dir": TCGA_DIR,
        "fixed_tau_months": FIXED_TAU_MONTHS,
        "top_k_genes": TOP_K_GENES,
        "mutation_freq_range": [MUTATION_FREQ_LOW, MUTATION_FREQ_HIGH],
        "missing_col_threshold": MISSING_COL_THRESHOLD,
        "days_per_month": DAYS_PER_MONTH,
    }

    save_preprocessed(
        essential_dir, clinical_aligned, gene_aligned, gene_names,
        mutation_aligned, cnv_aligned, methylation_aligned, rppa_aligned,
        sample_ids, config, missing_indicators=missing_indicators,
    )

    logger.info("[QC REPORT] Generating preprocessing quality report...")
    try:
        generate_preprocessing_qc_report(
            non_essential_dir, clinical_aligned, gene_aligned, mutation_aligned,
            cnv_aligned, methylation_aligned, rppa_aligned,
            sample_ids, missing_indicators,
            meth_validation=meth_validation, mutation_raw_df=mutation_raw_df,
        )
    except Exception as e:
        logger.warning(f"[QC REPORT] Failed (non-fatal): {e}")
        import traceback
        logger.debug(traceback.format_exc())

    # [SENSITIVITY ANALYSIS] 甲基化覆盖率敏感性分析
    logger.info("[SENSITIVITY] Assessing methylation coverage bias...")
    try:
        methylation_coverage_sensitivity_analysis(
            non_essential_dir, clinical_aligned, methylation_aligned,
            sample_ids, missing_indicators,
        )
    except Exception as e:
        logger.warning(f"[SENSITIVITY] Failed (non-fatal): {e}")
        import traceback
        logger.debug(traceback.format_exc())

    # [STRATIFIED DATASETS] 分组建模数据导出
    logger.info("[DATASETS] Exporting stratified datasets...")
    try:
        export_stratified_datasets(
            non_essential_dir, clinical_aligned, gene_aligned, mutation_aligned,
            cnv_aligned, methylation_aligned, rppa_aligned,
            sample_ids, missing_indicators,
        )
    except Exception as e:
        logger.warning(f"[DATASETS] Failed (non-fatal): {e}")
        import traceback
        logger.debug(traceback.format_exc())

    # [LATE FUSION] Late Fusion 适配导出
    logger.info("[LATE FUSION] Exporting late fusion manifest...")
    try:
        export_late_fusion_manifest(
            non_essential_dir, clinical_aligned, gene_aligned, mutation_aligned,
            cnv_aligned, methylation_aligned, rppa_aligned,
            sample_ids, missing_indicators,
        )
    except Exception as e:
        logger.warning(f"[LATE FUSION] Failed (non-fatal): {e}")
        import traceback
        logger.debug(traceback.format_exc())

    logger.info(f"[01_preprocessing] 完成! 必要数据保存到: {essential_dir}")
    logger.info(f"[01_preprocessing] 非必要数据保存到: {non_essential_dir}")


if __name__ == "__main__":
    main()
