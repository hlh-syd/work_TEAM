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

import polars as pl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "DATA"))
RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "results"))

TCGA_DIR = os.path.join(DATA_DIR, "tcga")
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")

FIXED_TAU_MONTHS = 36.0
EPS = 1e-8
TOP_K_GENES = 300
MUTATION_FREQ_LOW = 0.02
MUTATION_FREQ_HIGH = 0.80
MISSING_COL_THRESHOLD = 0.40
DAYS_PER_MONTH = 30.44
MAX_N_EXACT_PSEUDO = 1200

META_COLUMNS = {
    "Hugo_Symbol", "Entrez_Gene_Id", "Cytoband",
    "Composite.Element.REF", "ENTITY_STABLE_ID", "NAME",
    "DESCRIPTION", "TRANSCRIPT_ID", "ID", "GENE_SYMBOL",
    "PHOSPHOSITE", "geneNames",
}

STAGE_MAP = {
    "STAGE I": 1, "STAGE IA": 1, "STAGE IB": 1,
    "STAGE II": 2, "STAGE IIA": 2, "STAGE IIB": 2, "STAGE IIC": 2,
    "STAGE III": 3, "STAGE IIIA": 3, "STAGE IIIB": 3, "STAGE IIIC": 3,
    "STAGE IV": 4, "STAGE IVA": 4, "STAGE IVB": 4, "STAGE IVC": 4,
    "STAGE 0": 0,
    "I": 1, "IA": 1, "IB": 1,
    "II": 2, "IIA": 2, "IIB": 2, "IIC": 2,
    "III": 3, "IIIA": 3, "IIIB": 3, "IIIC": 3,
    "IV": 4, "IVA": 4, "IVB": 4, "IVC": 4,
    "0": 0,
}


def patient_id_from_sample(value):
    text = str(value)
    if text.startswith("TCGA-") and len(text) >= 12:
        return text[:12]
    if text.startswith("HTA8_"):
        parts = text.split("_")
        if len(parts) >= 2:
            return "_".join(parts[:2])
    return text


def normalize_patient_id(value):
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    if text.upper().startswith("TCGA-") and len(text) >= 12:
        return text[:12].upper()
    return re.sub(r"[\s_]+", "-", text).upper()


def parse_survival_status(series):
    values = series.fillna("").astype(str).str.strip()
    lower = values.str.lower()
    event = (
        values.str.startswith("1")
        | lower.isin({"1", "dead", "deceased", "event", "progression",
                       "recurred/progressed", "1:deceased", "yes", "true"})
    )
    censored = lower.isin({"0", "alive", "living", "0:living", "no", "false", "censored"})
    result = pd.Series(np.nan, index=series.index)
    result[event] = 1.0
    result[censored] = 0.0
    numeric = pd.to_numeric(series, errors="coerce")
    result = result.where(result.notna(), numeric)
    return result


def numeric_series(series):
    return pd.to_numeric(
        series.replace({"[Not Available]": None, "": None, "NA": None, "NaN": None}),
        errors="coerce",
    )


def kaplan_meier_survival_at(times, event_observed, query):
    order = np.argsort(times)
    times_sorted = np.asarray(times, dtype=float)[order]
    events_sorted = np.asarray(event_observed, dtype=int)[order]
    unique_event_times = np.unique(times_sorted[events_sorted == 1])
    surv = 1.0
    surv_times = []
    surv_values = []
    for t in unique_event_times:
        at_risk = np.sum(times_sorted >= t)
        d = np.sum((times_sorted == t) & (events_sorted == 1))
        if at_risk > 0:
            surv *= max(0.0, 1.0 - d / at_risk)
        surv_times.append(float(t))
        surv_values.append(float(surv))
    if not surv_times:
        return np.ones(len(query), dtype=float)
    surv_times_arr = np.asarray(surv_times)
    surv_values_arr = np.asarray(surv_values)
    idx = np.searchsorted(surv_times_arr, query, side="right") - 1
    out = np.ones(len(query), dtype=float)
    mask = idx >= 0
    out[mask] = surv_values_arr[idx[mask]]
    return np.clip(out, EPS, 1.0)


def km_cumulative_incidence_by_tau(times, events, tau):
    surv = kaplan_meier_survival_at(times, events, np.asarray([tau], dtype=float))[0]
    return float(1.0 - surv)


def matrix_sample_columns(columns):
    return [c for c in columns if c not in META_COLUMNS]


def encode_ajcc_stage(series):
    def _map(val):
        if pd.isna(val):
            return np.nan
        text = str(val).strip().upper()
        if text in STAGE_MAP:
            return STAGE_MAP[text]
        cleaned = re.sub(r"[^A-Z0-9]", "", text)
        if cleaned in STAGE_MAP:
            return STAGE_MAP[cleaned]
        match = re.search(r"(\d+|[IVX]+)", text)
        if match:
            token = match.group(1)
            if token in STAGE_MAP:
                return STAGE_MAP[token]
        return np.nan
    return series.apply(_map)


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
        return pd.DataFrame()

    filtered = df[df["Hugo_Symbol"].isin(keep_genes)].copy()
    pivot = (
        filtered.assign(value=1)
        .pivot_table(index="PATIENT_ID", columns="Hugo_Symbol",
                     values="value", aggfunc="max", fill_value=0)
        .reset_index()
    )
    return pivot


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


def load_methylation(probe_blacklist=None, min_iqr=0.0):
    """加载 HM450 甲基化数据，使用 Polars + numpy 全链路优化。

    Parameters
    ----------
    probe_blacklist : set/list/None
        需要过滤的探针 ID 集合（如性染色体/交叉反应探针）
    min_iqr : float
        IQR 阈值，> 0 时启用低变异过滤；默认 0.0 不过滤
    """
    import time as _time
    meth_path = os.path.join(TCGA_DIR, "data_methylation_hm450.txt")
    if not os.path.exists(meth_path):
        raise FileNotFoundError(f"甲基化数据文件不存在: {meth_path}")

    print("  [提示] 甲基化文件较大(~1.3GB)，使用 Polars+numpy 优化加载...")
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
    print(f"  Polars 读取完成: {_t1 - _t0:.1f}s, shape={df.shape}")

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
    print(f"  列选择/类型转换完成: {_t2 - _t1:.1f}s")

    # ── Step 2: 转 numpy 并处理重复探针 ─────────────────────────────
    probe_ids = df[id_col].to_list()
    values = df.select(sample_cols).to_numpy()  # shape: (~396K, ~570), float32
    del df  # 释放 Polars DataFrame

    # 重复探针聚合（按 probe_id groupby mean）
    probe_ids_arr = np.array(probe_ids)
    unique_probes, inverse_idx = np.unique(probe_ids_arr, return_inverse=True)
    if len(unique_probes) < len(probe_ids_arr):
        print(f"  发现重复探针: {len(probe_ids_arr)} -> {len(unique_probes)}，执行聚合...")
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
        print(f"  发现重复 patient_id: {len(patient_ids_raw)} -> {len(unique_patients)}，执行聚合...")
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
    print(f"  转置/聚合完成: {_t3 - _t2:.1f}s, shape={matrix.shape}")

    # 移除全 NaN 的探针列
    col_not_all_nan = ~np.all(np.isnan(matrix), axis=0)
    if not np.all(col_not_all_nan):
        n_removed = int(np.sum(~col_not_all_nan))
        matrix = matrix[:, col_not_all_nan]
        probe_ids = [p for p, k in zip(probe_ids, col_not_all_nan) if k]
        print(f"  移除全NaN探针列: {n_removed} 个")
    del col_not_all_nan

    # ── Step 4: 低变异过滤（可选） ───────────────────────────────────
    if min_iqr > 0:
        q75 = np.nanpercentile(matrix, 75, axis=0)
        q25 = np.nanpercentile(matrix, 25, axis=0)
        iqr = q75 - q25
        keep_mask = iqr >= min_iqr
        n_before = matrix.shape[1]
        matrix = matrix[:, keep_mask]
        probe_ids = [p for p, k in zip(probe_ids, keep_mask) if k]
        print(f"  低变异过滤(IQR>={min_iqr}): {n_before} -> {matrix.shape[1]} 探针")
        del q75, q25, iqr, keep_mask

    # ── Step 5: numpy 层缺失值填充 ───────────────────────────────────
    medians = np.nanmedian(matrix, axis=0)
    # 若某列全 NaN，nanmedian 返回 NaN，需用全局中位数回退
    nan_medians = np.isnan(medians)
    if nan_medians.any():
        global_median = np.nanmedian(matrix)
        if np.isnan(global_median):
            global_median = 0.0
        medians = np.where(nan_medians, global_median, medians)
    nan_mask = np.isnan(matrix)
    if nan_mask.any():
        matrix = np.where(nan_mask, medians, matrix)
        n_filled = int(nan_mask.sum())
        print(f"  缺失值填充: {n_filled} 个值")
    del medians, nan_mask, nan_medians

    # ── Step 6: numpy 层标准化 ───────────────────────────────────────
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    matrix = (matrix - mean) / np.clip(std, EPS, None)
    del mean, std

    # ── Step 7: 构造最终 DataFrame ───────────────────────────────────
    patient_matrix = pd.DataFrame(matrix, index=patient_ids, columns=probe_ids)
    del matrix

    _t_end = _time.time()
    print(f"  甲基化加载总耗时: {_t_end - _t0:.1f}s, 最终shape={patient_matrix.shape}")

    return patient_matrix


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


def align_samples(clinical_df, gene_df, mutation_df, cnv_df, methylation_df, rppa_df):
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

    all_sets = [clinical_ids] + list(omics_sets.values())
    common_ids = set.intersection(*all_sets) if all_sets else set()

    if not common_ids:
        available_sets = [clinical_ids]
        for name, s in omics_sets.items():
            if s:
                available_sets.append(s)
        common_ids = set.intersection(*available_sets) if available_sets else set()

    common_ids = sorted(common_ids)

    clinical_aligned = clinical_df[clinical_df["PATIENT_ID"].astype(str).isin(common_ids)].copy()

    gene_aligned = gene_df.loc[gene_df.index.astype(str).isin(common_ids)] if gene_df is not None and not gene_df.empty else pd.DataFrame()
    mutation_aligned = mutation_df[mutation_df["PATIENT_ID"].astype(str).isin(common_ids)] if mutation_df is not None and not mutation_df.empty else pd.DataFrame()
    cnv_aligned = cnv_df.loc[cnv_df.index.astype(str).isin(common_ids)] if cnv_df is not None and not cnv_df.empty else pd.DataFrame()
    methylation_aligned = methylation_df.loc[methylation_df.index.astype(str).isin(common_ids)] if methylation_df is not None and not methylation_df.empty else pd.DataFrame()
    rppa_aligned = rppa_df.loc[rppa_df.index.astype(str).isin(common_ids)] if rppa_df is not None and not rppa_df.empty else pd.DataFrame()

    return common_ids, clinical_aligned, gene_aligned, mutation_aligned, cnv_aligned, methylation_aligned, rppa_aligned


def save_preprocessed(output_dir, clinical_df, gene_df, gene_names, mutation_df,
                       cnv_df, methylation_df, rppa_df, sample_ids, config):
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
            print(f"  [WARNING] Parquet缓存保存失败: {e}")

    if rppa_df is not None and not rppa_df.empty:
        rppa_df.to_csv(os.path.join(output_dir, "rppa_curated.tsv"), sep="\t")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, default=None)
    args = parser.parse_args()

    timestamp = args.timestamp if args.timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(RESULTS_DIR, timestamp, "01_preprocessing")

    print(f"[01_preprocessing] 开始预处理 | 时间戳: {timestamp}")
    print(f"[01_preprocessing] 数据目录: {DATA_DIR}")
    print(f"[01_preprocessing] TCGA目录: {TCGA_DIR}")
    print(f"[01_preprocessing] 输出目录: {output_dir}")

    if not os.path.isdir(TCGA_DIR):
        print(f"[ERROR] TCGA数据目录不存在: {TCGA_DIR}", file=sys.stderr)
        sys.exit(1)

    print("[STEP 1/8] 加载临床数据...")
    clinical_df, le_dict = load_clinical_data()
    print(f"  临床数据: {clinical_df.shape[0]} 患者, {clinical_df.shape[1]} 列")

    print("[STEP 2/8] 处理生存终点...")
    clinical_df = process_survival_endpoints(clinical_df)
    print(f"  生存终点处理后: {clinical_df.shape[0]} 患者")

    gene_df = None
    gene_names = []
    mutation_df = None
    cnv_df = None
    methylation_df = None
    rppa_df = None

    print("[STEP 3/8] 加载基因表达数据...")
    try:
        gene_df, gene_names = load_gene_expression()
        print(f"  基因表达: {gene_df.shape[0]} 患者 x {gene_df.shape[1]} 基因")
    except (FileNotFoundError, ValueError) as e:
        print(f"  [WARNING] {e}")

    print("[STEP 4/8] 加载突变数据...")
    try:
        mutation_df = load_mutation_matrix()
        if mutation_df is not None and not mutation_df.empty:
            print(f"  突变矩阵: {mutation_df.shape[0]} 患者 x {mutation_df.shape[1] - 1} 基因")
    except (FileNotFoundError, ValueError) as e:
        print(f"  [WARNING] {e}")

    print("[STEP 5/8] 加载CNV数据...")
    try:
        cnv_df = load_cnv()
        print(f"  CNV: {cnv_df.shape[0]} 患者 x {cnv_df.shape[1]} 特征")
    except (FileNotFoundError, ValueError) as e:
        print(f"  [WARNING] {e}")

    print("[STEP 6/8] 加载甲基化数据...")
    try:
        methylation_df = load_methylation()
        print(f"  甲基化: {methylation_df.shape[0]} 患者 x {methylation_df.shape[1]} 特征")
    except (FileNotFoundError, ValueError) as e:
        print(f"  [WARNING] {e}")

    print("[STEP 7/8] 加载RPPA数据...")
    try:
        rppa_df = load_rppa()
        print(f"  RPPA: {rppa_df.shape[0]} 患者 x {rppa_df.shape[1]} 特征")
    except (FileNotFoundError, ValueError) as e:
        print(f"  [WARNING] {e}")

    print("[STEP 8/8] 对齐样本并保存...")
    sample_ids, clinical_aligned, gene_aligned, mutation_aligned, cnv_aligned, methylation_aligned, rppa_aligned = align_samples(
        clinical_df, gene_df, mutation_df, cnv_df, methylation_df, rppa_df
    )
    print(f"  对齐后样本数: {len(sample_ids)}")

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
        output_dir, clinical_aligned, gene_aligned, gene_names,
        mutation_aligned, cnv_aligned, methylation_aligned, rppa_aligned,
        sample_ids, config,
    )

    print(f"[01_preprocessing] 完成! 结果保存到: {output_dir}")


if __name__ == "__main__":
    main()
