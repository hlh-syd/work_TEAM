import os
import sys
import argparse
import json
import pickle
import warnings
import math
import gzip

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import hypergeom

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared_utils import (
    setup_logger,
    SCRIPT_DIR, DATA_DIR, RESULTS_DIR,
    ESSENTIAL_DIR,
    FIXED_TAU_MONTHS, RANDOM_SEED,
    patient_id_from_sample,
    encode_ajcc_stage as _shared_encode_ajcc_stage,
    STAGE_MAP_NUMERIC,
)
logger = setup_logger("02_gene_features")


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="lifelines")
warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning,
)

# ── 因果新增模块配置（flat 常量导入）──
from config_causal import (
    CAUSAL_EGM_ENABLED, CAUSAL_EGM_LATENT_DIM, CAUSAL_EGM_EPOCHS,
    CAUSAL_EGM_LR, CAUSAL_EGM_BATCH_SIZE, CAUSAL_EGM_FLOW_STEPS,
    CAUSAL_EGM_FALLBACK, CAUSAL_EGM_MODE,
    DEEPSURV_ENABLED, DEEPSURV_HIDDEN_DIMS, DEEPSURV_DROPOUT,
    DEEPSURV_L2_REG, DEEPSURV_LR, DEEPSURV_EPOCHS, DEEPSURV_PATIENCE,
    DEEPSURV_BATCH_SIZE, DEEPSURV_USE_CEGM_Z,
)

# --- Script-specific constants ---
PSEUDOGENE_HINTS = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9")
# AJCC 分期编码统一使用 shared_utils.STAGE_MAP_NUMERIC（粗粒度 0-4）
# 避免细粒度编码导致 One-Hot 后出现极低方差列（如 AJCC_STAGE_12.0）
_AJCC_STAGE_MAP = STAGE_MAP_NUMERIC

# ── LASSO-Cox 特征筛选参数（文献依据: BMC Cancer 2022, glmnet）──
LASSO_P_THRESHOLD = 0.20        # 预筛选p值阈值（原0.05，BMC Cancer 2022推荐放宽）
LASSO_CANDIDATE_CAP = 200       # 候选基因上限（BMC Cancer 2022: top-200 is practical）
LASSO_FALLBACK_MIN = 50         # 保底最少候选基因数
LASSO_N_ALPHAS = 300            # alpha路径点数（原100，增至300提高精度）
LASSO_ALPHA_MIN_RATIO = 1e-4    # alpha下界/alpha_max比值（glmnet nobs>nvars默认值）
BOOTSTRAP_ENABLED = True        # 是否启用bootstrap稳定性选择
BOOTSTRAP_N = 200               # bootstrap重采样次数
BOOTSTRAP_THRESHOLD = 0.6       # 选择频率阈值（Meinshausen-Bühlmann 2010 推荐）
BOOTSTRAP_SUBSAMPLE_FRAC = 0.5  # 子采样比例（无放回，Shah-Samworth 2013）


def _check_and_auto_disable():
    """根据依赖可用性自动禁用因果新模块。"""
    global CAUSAL_EGM_ENABLED, DEEPSURV_ENABLED
    try:
        import CausalEGM as causalegm  # noqa: F401
    except ImportError:
        logger.warning("[依赖] causalegm 未安装，自动禁用 CausalEGM")
        CAUSAL_EGM_ENABLED = False
    try:
        import torch  # noqa: F401
    except ImportError:
        logger.warning("[依赖] torch 未安装，自动禁用 DeepSurv")
        DEEPSURV_ENABLED = False


_check_and_auto_disable()


def is_likely_pseudogene(symbol):
    if not isinstance(symbol, str):
        return False
    s = symbol.upper()
    return s.startswith(("LOC", "LINC", "MIR", "SNORD", "RNU")) or any(s.endswith(suffix) for suffix in PSEUDOGENE_HINTS)


def bh_fdr(p_values):
    arr = np.asarray(list(p_values), dtype=float)
    out = np.full(len(arr), np.nan)
    mask = ~np.isnan(arr)
    if mask.any():
        out[mask] = multipletests(arr[mask], method="fdr_bh")[1]
    return out.tolist()


def stream_gene_matrix(path, comment="#", id_col_preference="Hugo_Symbol", chunksize=5000):
    chunks = [chunk for chunk in pd.read_csv(path, sep="\t", comment=comment, dtype=str, low_memory=False, chunksize=chunksize)]
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    id_col = id_col_preference if id_col_preference in df.columns else df.columns[0]
    meta_cols = {"Hugo_Symbol", "Entrez_Gene_Id", "Cytoband", "Composite.Element.REF",
                 "ENTITY_STABLE_ID", "NAME", "DESCRIPTION", "TRANSCRIPT_ID", "ID",
                 "GENE_SYMBOL", "PHOSPHOSITE", "geneNames"}
    sample_cols = [c for c in df.columns if c not in meta_cols]
    feature_df = df[[id_col] + sample_cols].set_index(id_col).apply(pd.to_numeric, errors="coerce")
    if feature_df.index.has_duplicates():
        feature_df = feature_df.groupby(level=0).mean()
    feature_df = feature_df.loc[feature_df.notna().any(axis=1)].T
    feature_df.index = [patient_id_from_sample(c) for c in feature_df.index]
    return feature_df.groupby(level=0).mean()


def load_gmt(path, min_size=5, max_size=500):
    gene_sets = {}
    p = os.path.abspath(path)
    if not os.path.exists(p):
        return gene_sets
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name, description = parts[0], parts[1]
            genes = {g.strip().upper() for g in parts[2:] if g.strip()}
            if min_size <= len(genes) <= max_size:
                gene_sets[name] = {"source": f"{os.path.basename(p)} -- {description}", "genes": genes}
    return gene_sets


def train_variance_screen(feature_df, train_ids, min_nonmissing=0.7, top_k=2000):
    train_mat = feature_df.reindex(train_ids)
    nonmissing = train_mat.notna().mean(axis=0)
    keep_mask = nonmissing >= min_nonmissing
    train_kept = train_mat.loc[:, keep_mask]
    variances = train_kept.var(axis=0, skipna=True)
    order = variances.sort_values(ascending=False)
    selected = order.head(top_k).index.tolist()
    diag = pd.DataFrame({
        "feature": variances.index,
        "train_variance": variances.values,
        "train_nonmissing_ratio": nonmissing.loc[variances.index].values,
    })
    diag["selected_for_univariable_cox"] = diag["feature"].isin(selected)
    return selected, diag


def univariate_cox(clinical, features, time_col, event_col, min_events=5, min_unique=3):
    # 防御性检查：如果features为空，直接返回空DataFrame
    if features.empty or len(features.columns) == 0:
        logger.warning("  [WARNING] univariate_cox 输入特征为空，跳过分析")
        return pd.DataFrame(columns=["feature", "n", "events", "coef", "hr", "p", "ph_p", "status", "fdr", "likely_pseudogene"])
    
    clin = clinical[["PATIENT_ID", time_col, event_col]].copy()
    clin["PATIENT_ID"] = clin["PATIENT_ID"].astype(str)
    clin[time_col] = pd.to_numeric(clin[time_col], errors="coerce")
    clin[event_col] = clin[event_col].fillna(False).astype(bool)
    clin = clin.dropna(subset=[time_col])
    clin = clin[clin[time_col] > 0].set_index("PATIENT_ID")
    common = clin.index.intersection(features.index)
    rows = []
    for feature in features.columns:
        x = pd.to_numeric(features.loc[common, feature], errors="coerce")
        df = pd.DataFrame({
            "time": clin.loc[common, time_col].astype(float),
            "event": clin.loc[common, event_col].astype(int),
            "feature": x,
        }).replace([np.inf, -np.inf], np.nan).dropna()
        if df["event"].sum() < min_events or df["feature"].nunique() < min_unique:
            rows.append({"feature": feature, "n": len(df), "events": int(df["event"].sum()),
                         "coef": np.nan, "hr": np.nan, "p": np.nan, "ph_p": np.nan,
                         "status": "insufficient_events_or_variance"})
            continue
        try:
            cph = CoxPHFitter(penalizer=0.0)
            cph.fit(df, duration_col="time", event_col="event")
            summary = cph.summary.loc["feature"]
            try:
                ph_p = float(proportional_hazard_test(cph, df, time_transform="rank").summary.loc["feature", "p"])
            except Exception:
                ph_p = np.nan
            rows.append({
                "feature": feature,
                "n": int(len(df)),
                "events": int(df["event"].sum()),
                "coef": float(summary["coef"]),
                "hr": float(summary["exp(coef)"]),
                "p": float(summary["p"]),
                "ph_p": ph_p,
                "status": "ok",
            })
        except Exception as exc:
            rows.append({"feature": feature, "n": len(df), "events": int(df["event"].sum()),
                         "coef": np.nan, "hr": np.nan, "p": np.nan, "ph_p": np.nan,
                         "status": f"cox_failed:{type(exc).__name__}"})
    out = pd.DataFrame(rows)
    # 防御性检查：如果out为空，返回完整结构的空DataFrame
    if out.empty:
        logger.warning("  [WARNING] univariate_cox 分析结果为空（所有特征都被过滤）")
        return pd.DataFrame(columns=["feature", "n", "events", "coef", "hr", "p", "ph_p", "status", "fdr", "likely_pseudogene"])
    ok = out["status"] == "ok"
    out["fdr"] = np.nan
    if ok.any():
        out.loc[ok, "fdr"] = bh_fdr(out.loc[ok, "p"].tolist())
    out["likely_pseudogene"] = out["feature"].apply(is_likely_pseudogene)
    return out.sort_values(["fdr", "p"], na_position="last")


def multivariable_cox(clinical, features, time_col, event_col, candidate_features,
                      confounders, penalizer=0.05):
    clin = clinical.set_index(clinical["PATIENT_ID"].astype(str))
    used_confounders = [c for c in confounders if c in clin.columns]
    if not used_confounders:
        return pd.DataFrame()
    cf = pd.get_dummies(clin[used_confounders], drop_first=True, dummy_na=False)
    cf = cf.apply(pd.to_numeric, errors="coerce")
    cf = cf.loc[:, cf.var(axis=0, skipna=True) > 0]
    common = cf.index.intersection(features.index)
    base = pd.DataFrame({
        "time": pd.to_numeric(clin.loc[common, time_col], errors="coerce").astype(float),
        "event": clin.loc[common, event_col].astype(bool).astype(int),
    }, index=common)
    cf = cf.loc[common]
    rows = []
    for feature in candidate_features:
        x = pd.to_numeric(features.loc[common, feature], errors="coerce")
        df = pd.concat([base, cf, x.rename("feature_value")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if df["event"].sum() < 10 or df["feature_value"].nunique() < 3:
            rows.append({"feature": feature, "status": "insufficient_events_or_variance_in_multivariable"})
            continue
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(df, duration_col="time", event_col="event")
            s = cph.summary.loc["feature_value"]
            try:
                ph_row = proportional_hazard_test(cph, df, time_transform="rank").summary.loc["feature_value"]
                ph_p = float(ph_row["p"])
            except Exception:
                ph_p = np.nan
            ph_violated = bool(ph_p == ph_p and ph_p < 0.05)
            rows.append({
                "feature": feature,
                "n": int(len(df)),
                "events": int(df["event"].sum()),
                "coef_adj": float(s["coef"]),
                "hr_adj": float(s["exp(coef)"]),
                "ci_low_adj": float(s["exp(coef) lower 95%"]),
                "ci_high_adj": float(s["exp(coef) upper 95%"]),
                "p_adj": float(s["p"]),
                "ph_assumption_p_adj": ph_p,
                "ph_assumption_violated": ph_violated,
                "ph_sensitivity_recommendation": "RMST_or_time_varying_sensitivity" if ph_violated else "PH_not_rejected",
                "confounders_used": "|".join(used_confounders),
                "status": "ok",
            })
        except Exception as exc:
            rows.append({"feature": feature, "status": f"cox_failed:{type(exc).__name__}"})
    out = pd.DataFrame(rows)
    ok = out.get("status", pd.Series(dtype=str)) == "ok"
    out["fdr_adj"] = np.nan
    if ok.any():
        out.loc[ok, "fdr_adj"] = bh_fdr(out.loc[ok, "p_adj"].tolist())
    return out


def standardized_mean_difference(x, a, w=None):
    """计算标准化均数差(SMD)，支持加权(ipw)和未加权版本。"""
    x = np.asarray(x, dtype=float)
    a = np.asarray(a, dtype=int)
    mask1 = a == 1
    mask0 = a == 0
    if w is not None:
        w = np.asarray(w, dtype=float)
        w1 = w[mask1]
        w0 = w[mask0]
        mean1 = np.average(x[mask1], weights=w1) if w1.sum() > 0 else np.nan
        mean0 = np.average(x[mask0], weights=w0) if w0.sum() > 0 else np.nan
        var1 = np.average((x[mask1] - mean1) ** 2, weights=w1) if w1.sum() > 0 else np.nan
        var0 = np.average((x[mask0] - mean0) ** 2, weights=w0) if w0.sum() > 0 else np.nan
    else:
        mean1 = np.nanmean(x[mask1])
        mean0 = np.nanmean(x[mask0])
        var1 = np.nanvar(x[mask1], ddof=1) if mask1.sum() > 1 else 0
        var0 = np.nanvar(x[mask0], ddof=1) if mask0.sum() > 1 else 0
    pooled_sd = np.sqrt((var1 + var0) / 2.0)
    return (mean1 - mean0) / pooled_sd if pooled_sd > 0 else 0.0


def propensity_score_diagnostics(W, A, random_seed=RANDOM_SEED):
    """倾向评分诊断：AUC、重叠性、IPW前后SMD、有效样本量。"""
    A = np.asarray(A, dtype=int)
    W_arr = np.asarray(W, dtype=float) if not isinstance(W, np.ndarray) else W.values if hasattr(W, 'values') else W

    prop = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=random_seed)
    prop.fit(W_arr, A)
    ps = prop.predict_proba(W_arr)[:, 1]

    from sklearn.metrics import roc_auc_score
    propensity_auc = float(roc_auc_score(A, ps))

    ps_clipped = np.clip(ps, 1e-6, 1 - 1e-6)
    ipw = A / ps_clipped + (1 - A) / (1 - ps_clipped)

    # IPW前后各协变量SMD
    smd_before = []
    smd_after = []
    for j in range(W_arr.shape[1]):
        smd_before.append(abs(standardized_mean_difference(W_arr[:, j], A)))
        smd_after.append(abs(standardized_mean_difference(W_arr[:, j], A, w=ipw)))

    # 有效样本量 (Kish's ESS)
    ess = float((ipw.sum()) ** 2 / (ipw ** 2).sum()) if ipw.sum() > 0 else np.nan

    return {
        "propensity_auc": propensity_auc,
        "ps_min": float(np.min(ps)),
        "ps_max": float(np.max(ps)),
        "ps_p01": float(np.percentile(ps, 1)),
        "ps_p99": float(np.percentile(ps, 99)),
        "overlap_flag": bool(np.min(ps) < 0.02 or np.max(ps) > 0.98),
        "max_smd_before": float(max(smd_before)) if smd_before else np.nan,
        "max_smd_after": float(max(smd_after)) if smd_after else np.nan,
        "effective_sample_size": ess,
    }


def causal_rmst_doubly_robust(X, exposure, time_months, event, tau=FIXED_TAU_MONTHS, random_seed=RANDOM_SEED):
    nan_result = {
        "ate_rmst": np.nan, "ate_rmst_se": np.nan, "p_value": np.nan,
        "rmst_exposed": np.nan, "rmst_unexposed": np.nan,
        "dr_rmst_ci_low": np.nan, "dr_rmst_ci_high": np.nan,
        "propensity_auc": np.nan, "ps_min": np.nan, "ps_max": np.nan,
        "ps_p01": np.nan, "ps_p99": np.nan,
        "overlap_flag": np.nan, "max_smd_before": np.nan, "max_smd_after": np.nan,
        "effective_sample_size": np.nan,
    }

    data = pd.concat([X, exposure.rename("A")], axis=1).dropna()
    if len(data) < 30 or data["A"].nunique() < 2:
        return nan_result

    A = data["A"].astype(int).to_numpy()
    W = data.drop(columns=["A"])

    all_idx = list(exposure.index)
    idx_map = {pid: i for i, pid in enumerate(all_idx)}
    T = np.array([time_months[idx_map[pid]] if pid in idx_map else np.nan for pid in data.index], dtype=float)
    E = np.array([event[idx_map[pid]] if pid in idx_map else 0 for pid in data.index], dtype=int)
    valid = np.isfinite(T)
    data = data.loc[valid]
    A = A[valid]
    T = T[valid]
    E = E[valid]
    W = W.loc[valid]

    if len(data) < 30:
        return nan_result

    T_capped = np.minimum(T, tau)

    # PS diagnostics
    ps_diag = propensity_score_diagnostics(W, A, random_seed=random_seed)

    n_folds = min(5, max(2, int(np.bincount(A).min())))
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    e_hat = np.zeros(len(data), dtype=float)
    m1_hat = np.zeros(len(data), dtype=float)
    m0_hat = np.zeros(len(data), dtype=float)

    for tr, te in folds.split(W, A):
        prop = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=random_seed)
        prop.fit(W.iloc[tr], A[tr])
        e_hat[te] = prop.predict_proba(W.iloc[te])[:, 1]

        rf0 = RandomForestRegressor(n_estimators=60, min_samples_leaf=8,
                                     random_state=random_seed, n_jobs=-1)
        rf1 = RandomForestRegressor(n_estimators=60, min_samples_leaf=8,
                                     random_state=random_seed + 1, n_jobs=-1)
        if np.sum(A[tr] == 0) >= 5:
            rf0.fit(W.iloc[tr][A[tr] == 0], T_capped[tr][A[tr] == 0])
            m0_hat[te] = rf0.predict(W.iloc[te])
        else:
            m0_hat[te] = np.mean(T_capped[tr][A[tr] == 0]) if np.any(A[tr] == 0) else np.mean(T_capped)
        if np.sum(A[tr] == 1) >= 5:
            rf1.fit(W.iloc[tr][A[tr] == 1], T_capped[tr][A[tr] == 1])
            m1_hat[te] = rf1.predict(W.iloc[te])
        else:
            m1_hat[te] = np.mean(T_capped[tr][A[tr] == 1]) if np.any(A[tr] == 1) else np.mean(T_capped)

    e_hat = np.clip(e_hat, 0.05, 0.95)

    dr = (m1_hat - m0_hat
          + A * (T_capped - m1_hat) / e_hat
          - (1 - A) * (T_capped - m0_hat) / (1 - e_hat))

    ate = float(np.mean(dr))
    se = float(np.std(dr, ddof=1) / np.sqrt(len(dr)))
    p = float(2 * stats.norm.sf(abs(ate / se))) if se > 0 else np.nan

    # Influence-function based 95% CI
    if_scores = dr - ate  # individual contributions
    ci_low = float(ate - 1.96 * se)
    ci_high = float(ate + 1.96 * se)

    return {
        "ate_rmst": ate,
        "ate_rmst_se": se,
        "p_value": p,
        "rmst_exposed": float(np.mean(T_capped[A == 1])),
        "rmst_unexposed": float(np.mean(T_capped[A == 0])),
        "dr_rmst_ci_low": ci_low,
        "dr_rmst_ci_high": ci_high,
        **ps_diag,
    }


def cis_eqtl_evidence(candidate_genes, eqtl_dir):
    """检查候选基因是否存在显著 cis-eQTL 证据（GTEx Colon 数据）。

    扫描 eqtl_dir 下所有包含 "Colon" 的 GTEx v8 independent_eqtls 文件，
    对每个候选基因汇总最显著的 cis-eQTL 证据。

    Args:
        candidate_genes: 候选基因列表（基因 symbol 或 Ensembl ID 均可）
        eqtl_dir: eQTL 数据目录路径（DATA/eqtl/）

    Returns:
        pd.DataFrame: 每个候选基因的 eQTL 证据摘要，列包括：
            feature, eqtl_tissue, eqtl_n_significant, eqtl_min_pval,
            eqtl_lead_snp, has_cis_eqtl
    """
    _COLS_REQUIRED = [
        "feature", "eqtl_tissue", "eqtl_n_significant",
        "eqtl_min_pval", "eqtl_lead_snp", "has_cis_eqtl",
    ]
    _SIGNIFICANT_THRESHOLD = 5e-8

    # ---- 空候选基因快速返回 ----
    if not candidate_genes:
        return pd.DataFrame(columns=_COLS_REQUIRED)

    candidate_set = set(str(g) for g in candidate_genes)
    # Ensembl ID 去版本号后的映射: stripped_id -> original candidate name
    ens_stripped_map = {}
    for g in candidate_set:
        if g.startswith("ENSG"):
            ens_stripped_map[g.split(".")[0]] = g

    # ---- 扫描 Colon 相关 gz 文件 ----
    if not os.path.isdir(eqtl_dir):
        logger.warning(f"[eQTL] 目录不存在: {eqtl_dir}")
        return _empty_eqtl_df(candidate_genes, _COLS_REQUIRED)

    eqtl_files = sorted(
        f for f in os.listdir(eqtl_dir)
        if "Colon" in f and f.endswith(".independent_eqtls.txt.gz")
    )
    if not eqtl_files:
        logger.warning(f"[eQTL] 未找到 Colon eQTL 文件，目录: {eqtl_dir}")
        return _empty_eqtl_df(candidate_genes, _COLS_REQUIRED)

    # ---- 逐文件解析 ----
    # gene_key -> list of (tissue, pval, variant_id) tuples
    gene_records = {}

    for fname in eqtl_files:
        tissue = fname.split(".v8.independent_eqtls")[0]  # e.g. "Colon_Transverse"
        fpath = os.path.join(eqtl_dir, fname)
        try:
            with gzip.open(fpath, "rt") as fh:
                header = fh.readline().rstrip("\n").split("\t")
                col_idx = {col: i for i, col in enumerate(header)}
                gene_id_idx = col_idx.get("gene_id")
                pval_idx = col_idx.get("pval_nominal")
                var_idx = col_idx.get("variant_id")
                if gene_id_idx is None or pval_idx is None:
                    logger.warning(f"[eQTL] 缺少关键列，跳过: {fname}")
                    continue

                for line in fh:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) <= max(gene_id_idx, pval_idx):
                        continue
                    raw_id = fields[gene_id_idx]
                    # 尝试匹配：原始 ID / 去版本号 / Ensembl stripped
                    match_key = None
                    if raw_id in candidate_set:
                        match_key = raw_id
                    elif raw_id.startswith("ENSG"):
                        stripped = raw_id.split(".")[0]
                        if stripped in ens_stripped_map:
                            match_key = ens_stripped_map[stripped]
                    if match_key is None:
                        continue

                    try:
                        pval = float(fields[pval_idx])
                    except (ValueError, IndexError):
                        continue
                    variant = fields[var_idx] if var_idx is not None and var_idx < len(fields) else "NA"
                    gene_records.setdefault(match_key, []).append((tissue, pval, variant))
        except Exception as exc:
            logger.warning(f"[eQTL] 读取失败 {fname}: {exc}")
            continue

    # ---- 汇总每个基因的证据 ----
    rows = []
    for gene in candidate_genes:
        gene_key = str(gene)
        records = gene_records.get(gene_key, [])
        if not records:
            rows.append({
                "feature": gene_key,
                "eqtl_tissue": np.nan,
                "eqtl_n_significant": np.nan,
                "eqtl_min_pval": np.nan,
                "eqtl_lead_snp": np.nan,
                "has_cis_eqtl": False,
            })
        else:
            n_sig = sum(1 for _, p, _ in records if p < _SIGNIFICANT_THRESHOLD)
            best = min(records, key=lambda r: r[1])
            rows.append({
                "feature": gene_key,
                "eqtl_tissue": best[0],
                "eqtl_n_significant": n_sig,
                "eqtl_min_pval": best[1],
                "eqtl_lead_snp": best[2],
                "has_cis_eqtl": n_sig > 0,
            })

    return pd.DataFrame(rows, columns=_COLS_REQUIRED)


def _empty_eqtl_df(candidate_genes, cols):
    """返回所有值为 NaN 的空 eQTL DataFrame（用于数据缺失时的降级返回）。"""
    rows = [{c: np.nan for c in cols} for _ in candidate_genes]
    for r, g in zip(rows, candidate_genes):
        r["feature"] = str(g)
        r["has_cis_eqtl"] = False
    return pd.DataFrame(rows, columns=cols)


def causal_forest_gene_effect(W, A, Y, seed=RANDOM_SEED):
    """因果森林估计条件处理效应(CATE)。

    优先使用 econml.dml.CausalForestDML，不可用时回退到 T-learner 交叉拟合。

    Args:
        W: 混杂矩阵 (n_samples, n_features)
        A: 二值暴露变量 (n_samples,)
        Y: 结局变量 (n_samples,)
        seed: 随机种子

    Returns:
        dict: cf_ate, cf_cate_mean, cf_cate_sd, cf_cate_q25, cf_cate_q75
    """
    nan_result = {
        "cf_ate": np.nan, "cf_cate_mean": np.nan, "cf_cate_sd": np.nan,
        "cf_cate_q25": np.nan, "cf_cate_q75": np.nan,
    }

    A = np.asarray(A, dtype=int)
    Y = np.asarray(Y, dtype=float)
    if hasattr(W, 'values'):
        W_arr = np.asarray(W, dtype=float)
    else:
        W_arr = np.asarray(W, dtype=float)

    if len(A) < 30 or len(np.unique(A)) < 2:
        return nan_result

    # 尝试使用 econml CausalForestDML
    try:
        from econml.dml import CausalForestDML
        cf = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=50, min_samples_leaf=8, random_state=seed, n_jobs=-1),
            model_t=LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed),
            n_estimators=100,
            min_samples_leaf=8,
            random_state=seed,
            n_jobs=-1,
        )
        cf.fit(Y, A, X=W_arr, W=None)
        cate = cf.effect(W_arr)
        ate = float(cf.ate_)
        return {
            "cf_ate": ate,
            "cf_cate_mean": float(np.mean(cate)),
            "cf_cate_sd": float(np.std(cate)),
            "cf_cate_q25": float(np.percentile(cate, 25)),
            "cf_cate_q75": float(np.percentile(cate, 75)),
        }
    except (ImportError, Exception):
        pass

    # T-learner 回退方案
    try:
        if np.sum(A == 0) < 5 or np.sum(A == 1) < 5:
            return nan_result
        folds = StratifiedKFold(n_splits=min(5, max(2, int(np.bincount(A).min()))),
                                 shuffle=True, random_state=seed)
        cate = np.zeros(len(A), dtype=float)
        for tr, te in folds.split(W_arr, A):
            rf0 = RandomForestRegressor(n_estimators=60, min_samples_leaf=8,
                                         random_state=seed, n_jobs=-1)
            rf1 = RandomForestRegressor(n_estimators=60, min_samples_leaf=8,
                                         random_state=seed + 1, n_jobs=-1)
            rf0.fit(W_arr[tr][A[tr] == 0], Y[tr][A[tr] == 0])
            rf1.fit(W_arr[tr][A[tr] == 1], Y[tr][A[tr] == 1])
            cate[te] = rf1.predict(W_arr[te]) - rf0.predict(W_arr[te])
        return {
            "cf_ate": float(np.mean(cate)),
            "cf_cate_mean": float(np.mean(cate)),
            "cf_cate_sd": float(np.std(cate)),
            "cf_cate_q25": float(np.percentile(cate, 25)),
            "cf_cate_q75": float(np.percentile(cate, 75)),
        }
    except Exception:
        return nan_result


def causal_cate_summary(X, exposure, outcome, random_seed):
    data = pd.concat([X, exposure.rename("A"), outcome.rename("Y")], axis=1).dropna()
    if len(data) < 40 or data["A"].nunique() < 2:
        return {"cate_sd": np.nan, "cate_iqr": np.nan}
    A = data["A"].astype(int)
    Y = data["Y"].astype(float)
    W = data.drop(columns=["A", "Y"])
    model0 = RandomForestRegressor(n_estimators=80, min_samples_leaf=8, random_state=random_seed, n_jobs=-1)
    model1 = RandomForestRegressor(n_estimators=80, min_samples_leaf=8, random_state=random_seed + 1, n_jobs=-1)
    if (A == 0).sum() < 5 or (A == 1).sum() < 5:
        return {"cate_sd": np.nan, "cate_iqr": np.nan}
    model0.fit(W.loc[A == 0], Y.loc[A == 0])
    model1.fit(W.loc[A == 1], Y.loc[A == 1])
    cate = model1.predict(W) - model0.predict(W)
    return {
        "cate_sd": float(np.std(cate)),
        "cate_iqr": float(np.quantile(cate, 0.75) - np.quantile(cate, 0.25)),
    }


def dose_response_summary(exposure, outcome, n_bins=5):
    data = pd.concat([exposure.rename("A"), outcome.rename("Y")], axis=1).dropna()
    if len(data) < 30 or data["A"].nunique() < n_bins:
        return {"dose_response_slope": np.nan, "dose_response_monotonic_spearman": np.nan}
    try:
        bins = pd.qcut(data["A"], q=n_bins, duplicates="drop")
        grouped = data.groupby(bins, observed=False).agg(A_mean=("A", "mean"), Y_mean=("Y", "mean"))
        if len(grouped) < 3:
            return {"dose_response_slope": np.nan, "dose_response_monotonic_spearman": np.nan}
        slope = stats.linregress(grouped["A_mean"], grouped["Y_mean"]).slope
        rho = stats.spearmanr(grouped["A_mean"], grouped["Y_mean"]).correlation
        return {
            "dose_response_slope": float(slope),
            "dose_response_monotonic_spearman": float(rho),
        }
    except Exception:
        return {"dose_response_slope": np.nan, "dose_response_monotonic_spearman": np.nan}


def run_causal_screening(feature_matrix, clinical_adjustment, endpoint_train,
                         random_seed, max_features, tau_months=None):
    """Run causal screening with endpoint-specific tau.

    Args:
        tau_months: endpoint-specific tau for RMST. If None, uses FIXED_TAU_MONTHS.
    """
    effective_tau = float(tau_months) if tau_months is not None else FIXED_TAU_MONTHS
    common = feature_matrix.index.intersection(endpoint_train.index)
    X_adj = clinical_adjustment.reindex(common).fillna(0)
    time_months = endpoint_train.reindex(common)["time_months"].to_numpy(dtype=float)
    events = endpoint_train.reindex(common)["event"].to_numpy(dtype=int)
    outcome_time = pd.Series(time_months, index=common)
    rows = []
    for feature in list(feature_matrix.columns)[:max_features]:
        x = feature_matrix.reindex(common)[feature]
        if x.notna().sum() < 30 or x.nunique(dropna=True) < 4:
            continue
        exposure_binary = (x > x.median()).astype(int)
        rmst_dr = causal_rmst_doubly_robust(
            X_adj, exposure_binary, time_months, events,
            tau=effective_tau, random_seed=random_seed,
        )
        cate = causal_cate_summary(X_adj, exposure_binary, outcome_time, random_seed)
        W_arr = X_adj.to_numpy(dtype=float)
        cf = causal_forest_gene_effect(W_arr, exposure_binary.to_numpy(), outcome_time.reindex(common).to_numpy(), seed=random_seed)
        dose = dose_response_summary(x, outcome_time)
        corr = stats.spearmanr(
            x.fillna(x.median()),
            outcome_time.fillna(outcome_time.median()),
        ).correlation
        rows.append({
            "feature": feature,
            "spearman_with_survival_time": float(corr) if np.isfinite(corr) else np.nan,
            **rmst_dr,
            **cate,
            **cf,
            **dose,
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["ate_abs"] = table["ate_rmst"].abs()
    table["ate_rank"] = table["ate_abs"].rank(ascending=False, method="average")
    table["p_rank"] = table["p_value"].rank(ascending=True, method="average")
    table["dose_rank"] = table["dose_response_slope"].abs().rank(ascending=False, method="average")
    table["causal_priority_score"] = (
        -table["ate_rank"].fillna(table["ate_rank"].max() + 1)
        - 0.5 * table["p_rank"].fillna(table["p_rank"].max() + 1)
        - 0.25 * table["dose_rank"].fillna(table["dose_rank"].max() + 1)
    )
    return table.sort_values("causal_priority_score", ascending=False)


def lasso_cox_screening(clinical, features, time_col, event_col, n_splits=10,
                        max_features=50, n_alphas=LASSO_N_ALPHAS,
                        alpha_min_ratio=LASSO_ALPHA_MIN_RATIO,
                        random_seed=RANDOM_SEED):
    """LASSO Cox 回归筛选（文献标准方法：univariate预筛选 → LASSO → multivariate）。

    使用 scikit-survival 或回退到 sklearn LassoCV + 伪偏残差方法。

    Args:
        clinical: 临床数据（含 PATIENT_ID, time_col, event_col）
        features: 基因表达矩阵（index=PATIENT_ID, columns=genes）
        time_col: 生存时间列名
        event_col: 事件指示列名
        n_splits: 交叉验证折数
        max_features: 最多保留特征数
        n_alphas: 正则化路径点数（默认300，原100）。BMC Cancer 2022推荐更精细路径
        alpha_min_ratio: alpha下界与alpha_max的比值（默认1e-4）。glmnet在nobs<nvars时
            默认1e-2，过大会导致"all coefficients are zero"。设为1e-4可确保充分探索低正则化区域
        random_seed: 随机种子

    Returns:
        dict: {"selected_genes": [...], "lasso_coefs": pd.DataFrame, "lambda_opt": float}
    """
    clin = clinical[["PATIENT_ID", time_col, event_col]].copy()
    clin["PATIENT_ID"] = clin["PATIENT_ID"].astype(str)
    clin[time_col] = pd.to_numeric(clin[time_col], errors="coerce")
    clin[event_col] = clin[event_col].fillna(False).astype(bool)
    clin = clin.dropna(subset=[time_col])
    clin = clin[clin[time_col] > 0]
    # 按 PATIENT_ID 去重，避免 .loc 查询时因重复索引导致行数膨胀
    clin = clin.drop_duplicates(subset=["PATIENT_ID"], keep="first")

    common = clin["PATIENT_ID"].values
    common = [pid for pid in common if pid in features.index]
    if len(common) < 30:
        return {"selected_genes": [], "lasso_coefs": pd.DataFrame(), "lambda_opt": np.nan}

    X = features.reindex(common).apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    y_time = pd.to_numeric(clin.set_index("PATIENT_ID").loc[common, time_col], errors="coerce").astype(float).values
    y_event = clin.set_index("PATIENT_ID").loc[common, event_col].astype(int).values

    # 尝试 scikit-survival (sksurv)
    try:
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        from sksurv.util import Surv
        from sklearn.model_selection import KFold
        y_surv = Surv.from_arrays(event=y_event.astype(bool), time=y_time)
        X_std = StandardScaler().fit_transform(X)
        # ── 构建精细化 alpha 路径（文献依据: glmnet lambda.min.ratio 机制）──
        # Step 1: 先用较少点快速拟合，获取 alpha_max（所有系数恰好为零的临界值）
        coxnet_init = CoxnetSurvivalAnalysis(n_alphas=10, tol=1e-5, max_iter=100000)
        coxnet_init.fit(X_std, y_surv)
        alpha_max = float(coxnet_init.alphas_[0])  # 路径起始点（最大alpha）
        alpha_min = alpha_max * alpha_min_ratio     # 路径终止点
        # Step 2: 在对数尺度上生成 n_alphas 个均匀分布的 alpha 值
        alpha_path = np.logspace(np.log10(alpha_max), np.log10(alpha_min), n_alphas)
        logger.info(f"[LASSO-Cox] Alpha path: {n_alphas} points, "
                    f"range=[{alpha_min:.6f}, {alpha_max:.4f}], ratio={alpha_min_ratio}")
        # Step 3: 手动K折CV + 全路径拟合（高效：每次fit计算整个alpha路径）
        cv_folds = min(n_splits, len(common) // 5)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
        cv_scores = np.zeros(len(alpha_path))
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_std)):
            X_tr, X_val = X_std[train_idx], X_std[val_idx]
            y_tr = y_surv[train_idx]
            # 一次 fit 计算整个正则化路径（warm-start 高效）
            coxnet_cv = CoxnetSurvivalAnalysis(alphas=alpha_path, tol=1e-5, max_iter=100000)
            coxnet_cv.fit(X_tr, y_tr)
            # 逐 alpha 评分（利用 coxnet_cv 路径上的系数）
            for j, alpha_val in enumerate(coxnet_cv.alphas_):
                coef_j = coxnet_cv.coef_[:, j] if coxnet_cv.coef_.ndim > 1 else coxnet_cv.coef_
                risk_val = X_val @ coef_j
                if np.std(risk_val) > 1e-10:
                    from lifelines.utils import concordance_index
                    time_val = y_time[val_idx]
                    event_val = y_event[val_idx].astype(bool)
                    try:
                        c = concordance_index(time_val, -risk_val, event_val)
                        cv_scores[j] += c
                    except Exception:
                        pass
            if (fold_idx + 1) % 3 == 0:
                logger.info(f"[LASSO-Cox] CV fold {fold_idx+1}/{cv_folds} done")
        cv_scores /= cv_folds
        best_idx = int(np.argmax(cv_scores))
        best_alpha = float(alpha_path[best_idx])
        logger.info(f"[LASSO-Cox] Best alpha={best_alpha:.6f}, CV C-index={cv_scores[best_idx]:.4f}")
        # Step 4: 用最优 alpha 在全数据上重新拟合
        coxnet = CoxnetSurvivalAnalysis(alphas=[best_alpha], tol=1e-5, max_iter=100000)
        coxnet.fit(X_std, y_surv)
        coefs = coxnet.coef_.flatten()
        # 取非零系数基因
        nonzero_mask = np.abs(coefs) > 1e-8
        selected = X.columns[nonzero_mask].tolist()[:max_features]
        coef_df = pd.DataFrame({
            "feature": X.columns,
            "lasso_coef": coefs,
        }).sort_values("lasso_coef", key=abs, ascending=False)
        coef_df = coef_df[coef_df["lasso_coef"].abs() > 1e-8].head(max_features)
        logger.info(f"[LASSO-Cox] scikit-survival: {len(selected)} genes selected, lambda={best_alpha:.6f}")
        return {"selected_genes": selected, "lasso_coefs": coef_df, "lambda_opt": float(best_alpha)}
    except ImportError:
        logger.info("[LASSO-Cox] scikit-survival not available, using fallback method")
    except Exception as exc:
        logger.warning(f"[LASSO-Cox] scikit-survival failed: {exc}, using fallback")

    # 回退方案：使用 LassoCV + Martingale 残差
    from lifelines import CoxPHFitter
    clin_df = pd.DataFrame({
        "time": y_time,
        "event": y_event,
    })
    # 拟合一个空模型获取 martingale 残差作为伪结局
    base_cph = CoxPHFitter(penalizer=0.01)
    base_cph.fit(clin_df, duration_col="time", event_col="event")
    # 手动计算 martingale 残差: M_i = event_i - H0(T_i) * exp(0) = event_i - H0(T_i)
    # compute_residuals 在某些 lifelines 版本中存在索引 bug，直接计算更可靠
    from scipy.interpolate import interp1d
    baseline_hazard = base_cph.baseline_hazard_
    time_points = baseline_hazard.index.values.astype(float)
    cum_hazard_values = baseline_hazard.values.flatten()
    interp_fn = interp1d(time_points, cum_hazard_values, kind="previous",
                         bounds_error=False, fill_value=(0, cum_hazard_values[-1]))
    H0_at_Ti = interp_fn(y_time.astype(float))
    martingale_residuals = y_event.astype(float) - H0_at_Ti
    logger.info(f"[LASSO-Cox] Fallback: martingale residuals computed manually, shape={martingale_residuals.shape}")

    X_std = StandardScaler().fit_transform(X)
    # 使用 martingale 残差作为连续结局进行 LASSO
    lasso_cv = LassoCV(cv=min(n_splits, len(common) // 5), random_state=random_seed,
                        max_iter=10000, tol=1e-6)
    lasso_cv.fit(X_std, martingale_residuals)
    nonzero_mask = np.abs(lasso_cv.coef_) > 1e-8
    selected = X.columns[nonzero_mask].tolist()[:max_features]
    coef_df = pd.DataFrame({
        "feature": X.columns,
        "lasso_coef": lasso_cv.coef_,
    }).sort_values("lasso_coef", key=abs, ascending=False)
    coef_df = coef_df[coef_df["lasso_coef"].abs() > 1e-8].head(max_features)
    logger.info(f"[LASSO-Cox] Fallback (martingale+LassoCV): {len(selected)} genes selected, alpha={lasso_cv.alpha_:.6f}")
    return {"selected_genes": selected, "lasso_coefs": coef_df, "lambda_opt": float(lasso_cv.alpha_)}


def bootstrap_stability_selection(clinical, features, time_col, event_col,
                                  n_bootstrap=BOOTSTRAP_N,
                                  selection_threshold=BOOTSTRAP_THRESHOLD,
                                  n_alphas=LASSO_N_ALPHAS,
                                  alpha_min_ratio=LASSO_ALPHA_MIN_RATIO,
                                  random_seed=RANDOM_SEED):
    """Bootstrap稳定性选择增强LASSO特征筛选（Lancet子刊方法）。

    对数据进行 n_bootstrap 次有放回重采样，每次运行 LASSO-Cox，
    统计每个基因被选中的频率。仅保留选择频率 >= selection_threshold 的基因。

    Args:
        clinical: 临床数据DataFrame
        features: 基因表达矩阵 (samples × genes)
        time_col: 生存时间列名
        event_col: 事件列名
        n_bootstrap: bootstrap重采样次数（推荐200-1000）
        selection_threshold: 选择频率阈值（推荐0.8-0.95）
        n_alphas: alpha路径点数
        alpha_min_ratio: alpha下界与alpha_max的比值
        random_seed: 随机种子

    Returns:
        dict: {
            "selected_genes": list,           # 通过稳定性筛选的基因
            "selection_frequency": DataFrame,  # 所有基因的选择频率
            "n_bootstrap": int,
            "selection_threshold": float,
        }
    """
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(random_seed)
    common = clinical.index.intersection(features.index)
    if len(common) == 0 and "PATIENT_ID" in clinical.columns:
        # 索引不匹配时尝试用 PATIENT_ID 对齐
        clinical = clinical.set_index("PATIENT_ID")
        common = clinical.index.astype(str).intersection(features.index.astype(str))
        clinical.index = clinical.index.astype(str)
    y_surv = np.array([(bool(clinical.loc[s, event_col]),
                        float(clinical.loc[s, time_col])) for s in common],
                      dtype=[("event", "?"), ("time", "<f8")])
    X = features.loc[common].copy()
    gene_names = X.columns.tolist()
    n_samples = len(common)
    sample_size = int(n_samples * BOOTSTRAP_SUBSAMPLE_FRAC)  # 子采样50%（无放回）

    selection_counts = {g: 0 for g in gene_names}
    n_success = 0

    for i in range(n_bootstrap):
        # 无放回子采样（Meinshausen-Bühlmann 2010, Shah-Samworth 2013）
        boot_idx = rng.choice(n_samples, size=sample_size, replace=False)
        boot_common = common[boot_idx]
        y_boot = y_surv[boot_idx]
        X_boot = X.iloc[boot_idx]

        with warnings.catch_warnings():
            from sklearn.exceptions import ConvergenceWarning
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", message=".*Optimization terminated early.*")

            try:
                X_std = StandardScaler().fit_transform(X_boot)
                # 快速拟合获取 alpha_max
                coxnet_init = CoxnetSurvivalAnalysis(n_alphas=5, tol=1e-5, max_iter=50000)
                coxnet_init.fit(X_std, y_boot)
                a_max = float(coxnet_init.alphas_[0])
                a_min = a_max * alpha_min_ratio
                # 用较少 alpha 点 + 更保守的范围加速 bootstrap 收敛
                bs_alpha_min_ratio = max(alpha_min_ratio, 1e-2)
                a_min = a_max * bs_alpha_min_ratio
                a_path = np.logspace(np.log10(a_max), np.log10(a_min), min(n_alphas, 50))
                coxnet = CoxnetSurvivalAnalysis(alphas=a_path, tol=1e-5, max_iter=50000)
                coxnet.fit(X_std, y_boot)
                coefs = coxnet.coef_.flatten()
                # 记录非零系数基因
                for j, g in enumerate(gene_names):
                    if abs(coefs[j]) > 1e-8:
                        selection_counts[g] += 1
                n_success += 1
            except Exception as exc:
                # 单次bootstrap失败不影响整体，记录首次错误用于诊断
                if n_success == 0 and i == 0:
                    logger.warning(f"[Bootstrap] 首次迭代失败（诊断）: {type(exc).__name__}: {exc}")
                continue

        # 每50次输出进度
        if (i + 1) % 50 == 0:
            logger.info(f"[Bootstrap] Progress: {i+1}/{n_bootstrap} "
                        f"({n_success} successful)")

    # 计算选择频率
    freq_df = pd.DataFrame({
        "feature": gene_names,
        "selection_frequency": [selection_counts[g] / max(n_success, 1)
                                for g in gene_names],
    }).sort_values("selection_frequency", ascending=False)

    # 筛选通过阈值的基因
    stable_genes = freq_df.loc[
        freq_df["selection_frequency"] >= selection_threshold, "feature"
    ].tolist()

    logger.info(f"[Bootstrap] {n_success}/{n_bootstrap} iterations succeeded, "
                f"{len(stable_genes)} genes passed threshold={selection_threshold}")
    return {
        "selected_genes": stable_genes,
        "selection_frequency": freq_df,
        "n_bootstrap": n_bootstrap,
        "selection_threshold": selection_threshold,
    }


def hypergeometric_enrichment(candidates, background_size, gene_sets):
    cand = {g.upper() for g in candidates}
    N = background_size
    n = len(cand)
    rows = []
    for name, info in gene_sets.items():
        K = len({g.upper() for g in info["genes"]})
        overlap = cand.intersection({g.upper() for g in info["genes"]})
        k = len(overlap)
        if K == 0 or n == 0:
            continue
        p = float(hypergeom.sf(k - 1, N, K, n)) if k > 0 else 1.0
        fold = (k / n) / (K / N) if n > 0 and N > 0 and K > 0 else np.nan
        rows.append({
            "gene_set": name,
            "source": info["source"],
            "set_size": K,
            "candidates": n,
            "background_universe": N,
            "overlap": k,
            "overlap_genes": ";".join(sorted(overlap)),
            "fold_enrichment": fold,
            "p_value": p,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["fdr_bh"] = bh_fdr(df["p_value"].tolist())
    return df.sort_values(["p_value", "fdr_bh"])


def pathway_enrichment(candidate_genes, background_genes, hallmark_path, kegg_path, go_bp_path,
                        out_dir, min_size=5, max_size=500):
    active_gene_sets = {}
    for p in [hallmark_path, kegg_path, go_bp_path]:
        gs = load_gmt(p, min_size=min_size, max_size=max_size)
        active_gene_sets.update(gs)

    if not active_gene_sets or not candidate_genes:
        return pd.DataFrame()

    background_clean = [g for g in background_genes if not is_likely_pseudogene(g)]
    enrich = hypergeometric_enrichment(candidate_genes, len(background_clean), active_gene_sets)
    if enrich.empty:
        return enrich

    enrich.to_csv(os.path.join(out_dir, "pathway_enrichment_results.tsv"), sep="\t", index=False)

    try:
        plot_df = enrich.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.30 * len(plot_df))))
        sizes = 40 + 12 * plot_df["overlap"].astype(float).clip(upper=20)
        colors = -np.log10(plot_df["p_value"].clip(lower=1e-10))
        sc = ax.scatter(plot_df["fold_enrichment"], plot_df["gene_set"], s=sizes, c=colors, cmap="viridis")
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("-log10(p)")
        ax.set_xlabel("Fold enrichment")
        ax.set_title("Pathway ORA (top 20)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "pathway_enrichment_dotplot.png"), dpi=180)
        plt.close(fig)
    except Exception:
        pass

    return enrich


def mechanism_validation_cptac(candidate_genes, cptac_protein_path, cptac_rna_path, out_dir):
    # NOTE: cptac_protein_path 文件由 01 脚本 rebuild_cptac_protein_gene_matrix() 重建
    # 格式: Hugo_Symbol(列0) × 样本ID(列1..N)，基因行为索引
    results = []
    if not os.path.exists(cptac_protein_path) or not os.path.exists(cptac_rna_path):
        logger.warning(f"[CPTAC] 文件不存在: protein={os.path.exists(cptac_protein_path)}, rna={os.path.exists(cptac_rna_path)}")
        return pd.DataFrame(results)

    try:
        prot = pd.read_csv(cptac_protein_path, sep="\t", low_memory=False)
        rna = pd.read_csv(cptac_rna_path, sep="\t", low_memory=False)
    except Exception:
        return pd.DataFrame(results)

    # 诊断：检查蛋白矩阵是否有基因表达列
    prot_id_col = prot.columns[0]
    rna_id_col = rna.columns[0]
    if prot.shape[1] <= 1:
        logger.error(f"[CPTAC] 蛋白矩阵仅含 {prot.shape[1]} 列（仅ID列，无基因表达数据）。"
                     f"文件可能损坏，需从原始数据重新生成: {cptac_protein_path}")
        return pd.DataFrame(results)
    prot = prot.set_index(prot_id_col)
    rna = rna.set_index(rna_id_col)
    prot = prot.apply(pd.to_numeric, errors="coerce")
    rna = rna.apply(pd.to_numeric, errors="coerce")

    # 诊断日志
    logger.info(f"[CPTAC] 蛋白矩阵: {prot.shape[0]} genes x {prot.shape[1]} samples")
    logger.info(f"[CPTAC] RNA矩阵: {rna.shape[0]} genes x {rna.shape[1]} samples")
    prot_in_cand = [g for g in candidate_genes if g in prot.index]
    rna_in_cand = [g for g in candidate_genes if g in rna.index]
    logger.info(f"[CPTAC] 候选基因匹配: 蛋白={len(prot_in_cand)}/{len(candidate_genes)}, RNA={len(rna_in_cand)}/{len(candidate_genes)}")

    common_samples = prot.columns.intersection(rna.columns)
    if len(common_samples) < 5:
        return pd.DataFrame(results)

    for gene in candidate_genes:
        if gene in prot.index and gene in rna.index:
            prot_vals = prot.loc[gene, common_samples]
            rna_vals = rna.loc[gene, common_samples]
            valid = prot_vals.notna() & rna_vals.notna()
            if valid.sum() >= 5:
                rho, p_val = stats.spearmanr(rna_vals[valid], prot_vals[valid])
                results.append({
                    "feature": gene,
                    "spearman_rho_rna_protein": float(rho) if np.isfinite(rho) else np.nan,
                    "spearman_p": float(p_val) if np.isfinite(p_val) else np.nan,
                    "n_samples": int(valid.sum()),
                    "sign_agreement": True,
                })
            else:
                results.append({
                    "feature": gene,
                    "spearman_rho_rna_protein": np.nan,
                    "spearman_p": np.nan,
                    "n_samples": int(valid.sum()),
                    "sign_agreement": np.nan,
                })
        else:
            results.append({
                "feature": gene,
                "spearman_rho_rna_protein": np.nan,
                "spearman_p": np.nan,
                "n_samples": 0,
                "sign_agreement": np.nan,
            })

    return pd.DataFrame(results)


# ============================================================
# 队列级机制富集检验 (Cohort-level Mechanism Enrichment Tests)
# ============================================================

def _permutation_test(observed_stat, background_genes, candidate_size, compute_stat_fn,
                      n_permutations=1000, random_seed=RANDOM_SEED):
    """通用置换检验框架：将 observed_stat 与随机基因集的零分布比较。"""
    rng = np.random.RandomState(random_seed)
    bg_list = list(background_genes)
    null_stats = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm_genes = rng.choice(bg_list, size=min(candidate_size, len(bg_list)), replace=False)
        null_stats[i] = compute_stat_fn(perm_genes)
    valid_null = null_stats[np.isfinite(null_stats)]
    if len(valid_null) < 10:
        return np.nan, np.nan, np.nan, valid_null
    p_value = float(np.mean(valid_null >= observed_stat))
    null_mean = float(np.mean(valid_null))
    null_sd = float(np.std(valid_null))
    effect_size = float((observed_stat - null_mean) / null_sd) if null_sd > 0 else np.nan
    return float(p_value), effect_size, null_mean, valid_null


def mechanism_protein_concordance(candidate_genes, background_genes,
                                  cptac_protein_path, cptac_rna_path,
                                  out_dir, n_permutations=1000):
    """队列级检验：候选基因蛋白表达协调性是否显著高于随机基因集。

    统计量 = 候选基因间蛋白水平平均|Spearman ρ|，通过置换检验获得经验p值。
    """
    tag = "protein_concordance"
    nan_result = {"test": tag, "n_genes_tested": 0, "observed_stat": np.nan,
                  "p_value": np.nan, "effect_size_cohens_d": np.nan,
                  "null_mean": np.nan, "n_permutations": n_permutations}
    if not os.path.exists(cptac_protein_path):
        return nan_result
    try:
        prot = pd.read_csv(cptac_protein_path, sep="\t", low_memory=False)
        rna = pd.read_csv(cptac_rna_path, sep="\t", low_memory=False) if os.path.exists(cptac_rna_path) else pd.DataFrame()
    except Exception:
        return nan_result
    prot = prot.set_index(prot.columns[0]).apply(pd.to_numeric, errors="coerce")
    rna = rna.set_index(rna.columns[0]).apply(pd.to_numeric, errors="coerce") if not rna.empty else pd.DataFrame()
    common_samples = prot.columns.tolist()
    if rna.empty:
        common_samples = prot.columns.tolist()
    else:
        common_samples = prot.columns.intersection(rna.columns).tolist()
    if len(common_samples) < 10:
        return nan_result

    # 提取候选基因的蛋白表达向量
    cand_vecs = {}
    for g in candidate_genes:
        if g in prot.index:
            v = prot.loc[g, common_samples].astype(float)
            if v.notna().sum() >= 10:
                cand_vecs[g] = v.fillna(v.median())
    if len(cand_vecs) < 3:
        return {**nan_result, "n_genes_tested": len(cand_vecs)}

    # 观测统计量：蛋白对间平均|Spearman|
    gene_names = list(cand_vecs.keys())
    n = len(gene_names)
    rho_sum, n_pairs = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = stats.spearmanr(cand_vecs[gene_names[i]], cand_vecs[gene_names[j]])
            if np.isfinite(r):
                rho_sum += abs(r)
                n_pairs += 1
    observed = rho_sum / n_pairs if n_pairs > 0 else np.nan
    if not np.isfinite(observed):
        return {**nan_result, "n_genes_tested": n}

    # 背景基因池
    bg_genes = [g for g in prot.index if g not in set(candidate_genes)]

    def compute_stat(perm_genes):
        vecs = []
        for g in perm_genes:
            if g in prot.index:
                v = prot.loc[g, common_samples].astype(float)
                if v.notna().sum() >= 10:
                    vecs.append(v.fillna(v.median()))
        if len(vecs) < 3:
            return np.nan
        s, c = 0.0, 0
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                r, _ = stats.spearmanr(vecs[i], vecs[j])
                if np.isfinite(r):
                    s += abs(r)
                    c += 1
        return s / c if c > 0 else np.nan

    p_val, eff, null_m, _ = _permutation_test(observed, bg_genes, n, compute_stat, n_permutations)
    logger.info(f"  [机制富集] 蛋白协调性: observed={observed:.4f}, p={p_val:.4f}, d={eff}")
    return {"test": tag, "n_genes_tested": n, "observed_stat": float(observed),
            "p_value": p_val, "effect_size_cohens_d": eff,
            "null_mean": null_m, "n_permutations": n_permutations}


def mechanism_phospho_enrichment(candidate_genes, background_genes,
                                 cptac_phospho_path, out_dir, n_permutations=1000):
    """队列级检验：候选基因的磷酸化位点数量是否显著富集。

    统计量 = 候选基因在CPTAC磷酸化蛋白矩阵中被检测到的位点总数。
    """
    tag = "phospho_site_enrichment"
    nan_result = {"test": tag, "n_genes_tested": 0, "observed_stat": np.nan,
                  "p_value": np.nan, "effect_size_cohens_d": np.nan,
                  "null_mean": np.nan, "n_permutations": n_permutations}
    if not os.path.exists(cptac_phospho_path):
        return nan_result
    try:
        phospho = pd.read_csv(cptac_phospho_path, sep="\t", low_memory=False)
    except Exception:
        return nan_result
    gene_col = "GENE_SYMBOL" if "GENE_SYMBOL" in phospho.columns else None
    if gene_col is None:
        for c in phospho.columns[:5]:
            if "gene" in c.lower():
                gene_col = c
                break
    if gene_col is None:
        return nan_result

    # 统计每个基因的磷酸化位点数
    gene_psite_counts = phospho[gene_col].value_counts().to_dict()
    cand_set = {g.upper() for g in candidate_genes}
    observed_count = sum(v for g, v in gene_psite_counts.items() if g.upper() in cand_set)
    cand_with_sites = sum(1 for g in candidate_genes if g.upper() in {k.upper() for k in gene_psite_counts})
    if observed_count == 0:
        return {**nan_result, "n_genes_tested": cand_with_sites}

    bg_genes = [g for g in gene_psite_counts if g.upper() not in cand_set]

    def compute_stat(perm_genes):
        perm_upper = {g.upper() for g in perm_genes}
        return sum(v for g, v in gene_psite_counts.items() if g.upper() in perm_upper)

    p_val, eff, null_m, _ = _permutation_test(
        float(observed_count), bg_genes, len(candidate_genes), compute_stat, n_permutations)
    logger.info(f"  [机制富集] 磷酸化富集: sites={observed_count}, p={p_val:.4f}, d={eff}")
    return {"test": tag, "n_genes_tested": cand_with_sites,
            "observed_stat": float(observed_count),
            "p_value": p_val, "effect_size_cohens_d": eff,
            "null_mean": null_m, "n_permutations": n_permutations}


def mechanism_cnv_methylation_concordance(candidate_genes, background_genes,
                                          cnv_path, methylation_path,
                                          out_dir, n_permutations=1000):
    """队列级检验：候选基因在CNV和甲基化层面是否显示协调性改变。

    CNV统计量 = |mean CNA| (候选基因整体偏向扩增或缺失)
    甲基化统计量 = mean |methylation deviation| (候选基因甲基化变异程度)

    NOTE: cnv_path 和 methylation_path 文件均由 01 脚本 REBUILD 阶段重建:
      - CNV: rebuild_cnv_gene_matrix() → Hugo_Symbol × patient_id, ~25K 基因
      - 甲基化: rebuild_methylation_gene_matrix() → Hugo_Symbol × patient_id, TSS±2kb 映射
    """
    results = {}

    # --- CNV 检验 ---
    if os.path.exists(cnv_path):
        try:
            cnv = pd.read_csv(cnv_path, sep="\t", low_memory=False)
            id_col = cnv.columns[0]
            if id_col == "Hugo_Symbol":
                cnv_mat = cnv.set_index("Hugo_Symbol").apply(pd.to_numeric, errors="coerce")
            else:
                cnv_mat = cnv.set_index(id_col).apply(pd.to_numeric, errors="coerce").T
            logger.info(f"[机制富集] CNV矩阵: {cnv_mat.shape[0]} genes x {cnv_mat.shape[1]} samples")
            cnv_matched = [g for g in candidate_genes if g in cnv_mat.index]
            logger.info(f"[机制富集] CNV候选基因匹配: {len(cnv_matched)}/{len(candidate_genes)}: {cnv_matched}")
            if len(cnv_matched) == 0:
                logger.warning(f"[机制富集] CNV矩阵仅含 {cnv_mat.shape[0]} 基因，"
                               f"候选基因均未命中。需从GISTIC2.0重新生成gene-level矩阵")
            cand_vecs = {}
            for g in candidate_genes:
                if g in cnv_mat.index:
                    v = cnv_mat.loc[g].dropna()
                    if len(v) >= 10:
                        cand_vecs[g] = v
            if len(cand_vecs) >= 3:
                observed_cnv = float(np.mean([abs(v.mean()) for v in cand_vecs.values()]))
                bg_genes = [g for g in cnv_mat.index if g not in set(candidate_genes)]
                def cnv_stat(perm_genes):
                    vals = []
                    for g in perm_genes:
                        if g in cnv_mat.index:
                            v = cnv_mat.loc[g].dropna()
                            if len(v) >= 10:
                                vals.append(abs(v.mean()))
                    return float(np.mean(vals)) if len(vals) >= 3 else np.nan
                p, eff, nm, _ = _permutation_test(observed_cnv, bg_genes, len(cand_vecs), cnv_stat, n_permutations)
                results["cnv"] = {"test": "cnv_amplitude", "n_genes_tested": len(cand_vecs),
                                  "observed_stat": observed_cnv, "p_value": p,
                                  "effect_size_cohens_d": eff, "null_mean": nm}
                logger.info(f"  [机制富集] CNV幅度: obs={observed_cnv:.4f}, p={p:.4f}, d={eff}")
        except Exception as exc:
            logger.warning(f"  [机制富集] CNV检验失败: {exc}")

    # --- 甲基化检验 ---
    if os.path.exists(methylation_path):
        try:
            meth = pd.read_csv(methylation_path, sep="\t", low_memory=False)
            id_col = meth.columns[0]
            # 诊断：检查甲基化矩阵是否有基因数据列
            if meth.shape[1] <= 1:
                logger.error(f"[机制富集] 甲基化矩阵仅含 {meth.shape[1]} 列（仅ID列，无基因甲基化数据）。"
                             f"需从HM450探针数据通过TSS proximity重新生成gene-level矩阵")
            else:
                logger.info(f"[机制富集] 甲基化矩阵: {meth.shape[0]} rows x {meth.shape[1]} cols")
            if id_col == "Hugo_Symbol":
                meth_mat = meth.set_index("Hugo_Symbol").apply(pd.to_numeric, errors="coerce")
            else:
                meth_mat = meth.set_index(id_col).apply(pd.to_numeric, errors="coerce").T
            cand_vecs = {}
            for g in candidate_genes:
                if g in meth_mat.index:
                    v = meth_mat.loc[g].dropna()
                    if len(v) >= 10:
                        cand_vecs[g] = v
            if len(cand_vecs) >= 3:
                grand_mean = float(meth_mat.stack().mean()) if meth_mat.shape[0] > 0 else 0.5
                observed_meth = float(np.mean([abs(v.mean() - grand_mean) for v in cand_vecs.values()]))
                bg_genes = [g for g in meth_mat.index if g not in set(candidate_genes)]
                def meth_stat(perm_genes):
                    vals = []
                    for g in perm_genes:
                        if g in meth_mat.index:
                            v = meth_mat.loc[g].dropna()
                            if len(v) >= 10:
                                vals.append(abs(v.mean() - grand_mean))
                    return float(np.mean(vals)) if len(vals) >= 3 else np.nan
                p, eff, nm, _ = _permutation_test(observed_meth, bg_genes, len(cand_vecs), meth_stat, n_permutations)
                results["methylation"] = {"test": "methylation_deviation", "n_genes_tested": len(cand_vecs),
                                          "observed_stat": observed_meth, "p_value": p,
                                          "effect_size_cohens_d": eff, "null_mean": nm}
                logger.info(f"  [机制富集] 甲基化偏差: obs={observed_meth:.4f}, p={p:.4f}, d={eff}")
        except Exception as exc:
            logger.warning(f"  [机制富集] 甲基化检验失败: {exc}")

    return results if results else {"cnv": {"test": "cnv_amplitude", "p_value": np.nan},
                                     "methylation": {"test": "methylation_deviation", "p_value": np.nan}}


def mechanism_multiomics_concordance(candidate_genes, multiomics_embedding_path,
                                     feature_df, out_dir, n_permutations=1000):
    """队列级检验：候选基因表达模式是否与多组学患者嵌入空间显著关联。

    将患者按候选基因中位表达分为高/低组，检验两组在多组学嵌入空间中的分离度。
    """
    tag = "multiomics_embedding_separation"
    nan_result = {"test": tag, "n_genes_tested": 0, "observed_stat": np.nan,
                  "p_value": np.nan, "effect_size_cohens_d": np.nan,
                  "null_mean": np.nan, "n_permutations": n_permutations}
    if not os.path.exists(multiomics_embedding_path) or feature_df is None or feature_df.empty:
        return nan_result
    try:
        embed = pd.read_csv(multiomics_embedding_path, sep="\t", index_col=0, low_memory=False)
        embed = embed.apply(pd.to_numeric, errors="coerce")
    except Exception:
        return nan_result
    common = feature_df.index.intersection(embed.index)
    if len(common) < 30:
        return nan_result

    # 候选基因表达聚合分数
    avail = [g for g in candidate_genes if g in feature_df.columns]
    if len(avail) < 3:
        return {**nan_result, "n_genes_tested": len(avail)}
    expr_score = feature_df.loc[common, avail].median(axis=1).dropna()
    common_valid = expr_score.index.intersection(embed.index)
    if len(common_valid) < 20:
        return {**nan_result, "n_genes_tested": len(avail)}

    # 按中位数分组
    med = expr_score.median()
    high_group = common_valid[expr_score.loc[common_valid] > med]
    low_group = common_valid[expr_score.loc[common_valid] <= med]
    if len(high_group) < 5 or len(low_group) < 5:
        return {**nan_result, "n_genes_tested": len(avail)}

    # 观测统计量：组间欧氏距离差
    embed_arr = embed.loc[common_valid].fillna(0).values
    pid_to_idx = {pid: i for i, pid in enumerate(common_valid)}
    high_idx = [pid_to_idx[p] for p in high_group if p in pid_to_idx]
    low_idx = [pid_to_idx[p] for p in low_group if p in pid_to_idx]
    if len(high_idx) < 3 or len(low_idx) < 3:
        return {**nan_result, "n_genes_tested": len(avail)}

    from scipy.spatial.distance import cdist
    cross_dists = cdist(embed_arr[high_idx], embed_arr[low_idx], metric="euclidean")
    observed = float(np.mean(cross_dists))

    # 置换检验
    all_indices = list(range(len(common_valid)))
    rng = np.random.RandomState(RANDOM_SEED)
    null_stats = []
    for _ in range(n_permutations):
        rng.shuffle(all_indices)
        perm_high = all_indices[:len(high_idx)]
        perm_low = all_indices[len(high_idx):len(high_idx) + len(low_idx)]
        cd = cdist(embed_arr[perm_high], embed_arr[perm_low], metric="euclidean")
        null_stats.append(float(np.mean(cd)))
    null_arr = np.array(null_stats)
    p_val = float(np.mean(null_arr >= observed))
    null_sd = float(np.std(null_arr))
    eff = float((observed - float(np.mean(null_arr))) / null_sd) if null_sd > 0 else np.nan
    logger.info(f"  [机制富集] 多组学分离度: obs={observed:.4f}, p={p_val:.4f}, d={eff}")
    return {"test": tag, "n_genes_tested": len(avail), "observed_stat": observed,
            "p_value": p_val, "effect_size_cohens_d": eff,
            "null_mean": float(np.mean(null_arr)), "n_permutations": n_permutations}


def mechanism_enrichment_summary(protein_result, phospho_result, cnv_meth_results,
                                 multiomics_result, out_dir):
    """汇总所有队列级机制富集检验结果，输出TSV + 柱状图。"""
    rows = []
    for r in [protein_result, phospho_result, multiomics_result]:
        if isinstance(r, dict) and "test" in r:
            rows.append(r)
    if isinstance(cnv_meth_results, dict):
        for sub_key in ["cnv", "methylation"]:
            if sub_key in cnv_meth_results:
                v = cnv_meth_results[sub_key]
                if isinstance(v, dict) and "test" in v:
                    rows.append(v)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    expected_cols = ["test", "n_genes_tested", "observed_stat", "p_value",
                     "effect_size_cohens_d", "null_mean", "n_permutations"]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[expected_cols]
    df["fdr_bh"] = bh_fdr(df["p_value"].tolist()) if df["p_value"].notna().any() else np.nan
    df["significant"] = df["p_value"].apply(lambda p: p < 0.05 if pd.notna(p) else False)
    df.to_csv(os.path.join(out_dir, "mechanism_enrichment_summary.tsv"), sep="\t", index=False)

    # 可视化
    try:
        plot_df = df[df["effect_size_cohens_d"].notna()].copy()
        if len(plot_df) > 0:
            fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(plot_df))))
            colors = ["#2ecc71" if p < 0.05 else "#95a5a6"
                      for p in plot_df["p_value"].fillna(1.0)]
            bars = ax.barh(range(len(plot_df)), plot_df["effect_size_cohens_d"], color=colors)
            ax.set_yticks(range(len(plot_df)))
            ax.set_yticklabels(plot_df["test"], fontsize=9)
            ax.set_xlabel("Effect size (Cohen's d)")
            ax.set_title("Cohort-level Mechanism Enrichment")
            ax.axvline(x=0, color="black", linewidth=0.5, linestyle="--")
            for i, (eff, p) in enumerate(zip(plot_df["effect_size_cohens_d"], plot_df["p_value"])):
                label = f"p={p:.3f}" if pd.notna(p) else "N/A"
                ax.text(eff, i, f"  {label}", va="center", fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "mechanism_enrichment_barplot.png"), dpi=180)
            plt.close(fig)
    except Exception:
        pass

    n_sig = int(df["significant"].sum())
    logger.info(f"  [机制富集] 汇总: {len(df)} tests, {n_sig} significant (p<0.05)")
    return df


def build_evidence_matrix(causal_table, univar_table, multivar_table, cptac_table,
                          candidate_genes, out_dir, cohort_mechanism_df=None):
    parts = []
    if not causal_table.empty:
        ct = causal_table[causal_table["feature"].isin(candidate_genes)].copy()
        ct["evidence_source"] = "causal_priority"
        parts.append(ct)

    if not univar_table.empty:
        ut = univar_table[univar_table["feature"].isin(candidate_genes)].copy()
        ut["evidence_source"] = "univariable_cox"
        parts.append(ut[["feature", "hr", "p", "fdr", "evidence_source"]])

    if not multivar_table.empty:
        mt = multivar_table[multivar_table["feature"].isin(candidate_genes)].copy()
        mt["evidence_source"] = "multivariable_cox"
        cols_keep = [c for c in ["feature", "hr_adj", "p_adj", "fdr_adj", "evidence_source"] if c in mt.columns]
        parts.append(mt[cols_keep])

    if not parts:
        evidence = pd.DataFrame({"feature": candidate_genes})
    else:
        evidence = pd.concat(parts, ignore_index=True, sort=False)

    if not cptac_table.empty and "spearman_rho_rna_protein" in cptac_table.columns:
        evidence = evidence.merge(
            cptac_table[["feature", "spearman_rho_rna_protein", "spearman_p", "sign_agreement"]],
            on="feature", how="left"
        )

    # 整合队列级机制富集检验结果（基因集层面证据）
    if cohort_mechanism_df is not None and not cohort_mechanism_df.empty:
        mech_sig = cohort_mechanism_df[cohort_mechanism_df.get("significant", False) == True]
        evidence["cohort_mechanism_tests_significant"] = len(mech_sig)
        evidence["cohort_mechanism_tests_total"] = len(cohort_mechanism_df)
        evidence["cohort_mechanism_max_effect"] = float(
            cohort_mechanism_df["effect_size_cohens_d"].max()
        ) if "effect_size_cohens_d" in cohort_mechanism_df.columns else np.nan
        evidence["cohort_mechanism_min_p"] = float(
            cohort_mechanism_df["p_value"].min()
        ) if "p_value" in cohort_mechanism_df.columns else np.nan

    evidence = evidence.drop_duplicates(subset=["feature"]).reset_index(drop=True)
    evidence.to_csv(os.path.join(out_dir, "core_biomarker_evidence_matrix.tsv"), sep="\t", index=False)

    try:
        indicator_cols = [c for c in ["sign_agreement"] if c in evidence.columns]
        plot_evidence = evidence.head(30)
        if indicator_cols and not plot_evidence.empty:
            heat = plot_evidence.set_index("feature")[indicator_cols].copy()
            for col in heat.columns:
                heat[col] = heat[col].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).fillna(0.5)
            heat = heat.apply(pd.to_numeric, errors="coerce").fillna(0.5).astype(float)

            num_cols_in_evidence = []
            for col in ["hr", "hr_adj", "spearman_rho_rna_protein", "causal_priority_score"]:
                if col in plot_evidence.columns:
                    num_cols_in_evidence.append(col)

            if num_cols_in_evidence:
                num_data = plot_evidence.set_index("feature")[num_cols_in_evidence].apply(
                    pd.to_numeric, errors="coerce"
                ).fillna(0)
                for col in num_data.columns:
                    col_vals = num_data[col]
                    if col_vals.max() != col_vals.min():
                        num_data[col] = (col_vals - col_vals.min()) / (col_vals.max() - col_vals.min())
                    else:
                        num_data[col] = 0.5
                heat = pd.concat([heat, num_data], axis=1)

            fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(heat.columns)), max(3.5, 0.22 * len(heat))))
            im = ax.imshow(heat.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, fraction=0.035, label="Evidence score")
            ax.set_yticks(range(len(heat.index)))
            ax.set_yticklabels(heat.index, fontsize=7)
            ax.set_xticks(range(len(heat.columns)))
            ax.set_xticklabels(heat.columns, rotation=35, ha="right", fontsize=8)
            ax.set_title("Biomarker Evidence Matrix")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "evidence_heatmap.png"), dpi=180)
            plt.close(fig)
    except Exception:
        pass

    return evidence


def encode_ajcc_stage(series):
    """使用 shared_utils 的粗粒度编码（0-4），保持与 03_model_training 一致。"""
    return _shared_encode_ajcc_stage(series)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()

    timestamp = args.timestamp
    out_dir = os.path.join(RESULTS_DIR, timestamp, "02_gene_features")
    os.makedirs(out_dir, exist_ok=True)

    prev_dir = ESSENTIAL_DIR

    clinical_path = os.path.join(prev_dir, "tcga_os_clinical_endpoint_qc.tsv")
    survival_labels_path = os.path.join(prev_dir, "survival_labels.pkl")
    gene_expr_path = os.path.join(prev_dir, "gene_expression_curated.tsv")
    sample_ids_path = os.path.join(prev_dir, "sample_ids.pkl")

    hallmark_path = os.path.join(DATA_DIR, "msigdb", "h.all.v2024.1.Hs.symbols.gmt")
    kegg_path = os.path.join(DATA_DIR, "msigdb", "c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt")
    go_bp_path = os.path.join(DATA_DIR, "msigdb", "c5.go.bp.v2024.1.Hs.symbols.gmt")
    causal_ref_path = os.path.join(DATA_DIR, "causal", "causal_priority_feature_table.tsv")
    split_path = os.path.join(DATA_DIR, "survival_models", "tcga_train_internal_validation_split.tsv")
    # ── 以下三个 gene-level 矩阵由 01_data_preprocessing.py 的 REBUILD 阶段重建 ──
    # CPTAC 蛋白矩阵: rebuild_cptac_protein_gene_matrix() 从肽段数据按 Hugo_Symbol 中位数聚合
    cptac_protein_path = os.path.join(DATA_DIR, "external", "cptac_coad_protein_gene_level_matrix.tsv")
    cptac_rna_path = os.path.join(DATA_DIR, "external", "cptac_coad_rna_gene_level_matrix.tsv")
    cptac_phospho_path = os.path.join(DATA_DIR, "external", "cptac_coad_phosphoprotein_gene_level_matrix.tsv")
    # CNV 基因矩阵: rebuild_cnv_gene_matrix() 从 data_cna.txt (~25K基因) 转换
    cnv_path = os.path.join(DATA_DIR, "preprocessed", "tcga_coadread_cnv_gene_level_matrix.tsv")
    # 甲基化基因矩阵: rebuild_methylation_gene_matrix() 从 HM450 探针 TSS±2kb 映射
    methylation_path = os.path.join(DATA_DIR, "preprocessed", "tcga_coadread_methylation_gene_level_matrix.tsv")
    multiomics_embedding_path = os.path.join(DATA_DIR, "multiomics", "tcga_multiomics_patient_embedding.tsv")

    logger.info("[02] Loading clinical endpoint...")
    clinical = pd.read_csv(clinical_path, sep="\t", low_memory=False)
    time_col = "OS_MONTHS"
    event_col = "OS_EVENT"
    clinical[time_col] = pd.to_numeric(clinical[time_col], errors="coerce")
    clinical[event_col] = clinical[event_col].fillna(False).astype(bool)
    clinical = clinical.loc[clinical["PATIENT_ID"].notna() & clinical[time_col].notna() & (clinical[time_col] > 0)].copy()
    logger.info(f"[02] Clinical rows: {len(clinical)}, events: {int(clinical[event_col].sum())}")

    logger.info("[02] Loading gene expression matrix...")
    if os.path.exists(gene_expr_path):
        feature_df = pd.read_csv(gene_expr_path, sep="\t", index_col=0, low_memory=False)
        feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    else:
        logger.info("[02] gene_expression_curated.tsv not found, trying raw RNA matrix...")
        feature_df = pd.DataFrame()

    if os.path.exists(sample_ids_path):
        with open(sample_ids_path, "rb") as f:
            sample_ids = pickle.load(f)
    else:
        sample_ids = feature_df.index.tolist()

    logger.info(f"[02] Gene expression: {feature_df.shape[0]} samples x {feature_df.shape[1]} genes")

    if os.path.exists(split_path):
        split = pd.read_csv(split_path, sep="\t")
        train_ids_from_file = split.loc[split["split"] == "train", "PATIENT_ID"].astype(str).tolist()
        # 防御性检查：只保留在feature_df中存在的train_ids
        feature_ids = set(feature_df.index.astype(str))
        train_ids = [pid for pid in train_ids_from_file if pid in feature_ids]
        logger.info(f"[02] Using predefined split: n_train_from_file={len(train_ids_from_file)}, n_train_after_intersection={len(train_ids)}")
        if len(train_ids) < 30:
            logger.warning(f"[02] WARNING: Train样本过少 ({len(train_ids)}), 回退到使用全部样本")
            train_ids = feature_df.index.astype(str).tolist()
    else:
        train_ids = clinical["PATIENT_ID"].astype(str).tolist()
        logger.info(f"[02] No split file found, using full cohort: n={len(train_ids)}")

    logger.info("[02] Step 1: Train variance screen...")
    selected, variance_diag = train_variance_screen(feature_df, train_ids, min_nonmissing=0.7, top_k=2000)
    variance_diag.to_csv(os.path.join(out_dir, "tcga_train_only_variance_prescreen.tsv"), sep="\t", index=False)
    logger.info(f"[02] Variance pre-screen: {len(selected)}/{feature_df.shape[1]} genes retained")

    logger.info("[02] Step 2: Univariable Cox screening...")
    feature_subset = feature_df[selected]
    train_clinical = clinical[clinical["PATIENT_ID"].astype(str).isin(train_ids)] if train_ids else clinical
    univ = univariate_cox(
        train_clinical,
        feature_subset.reindex(train_clinical["PATIENT_ID"].astype(str)),
        time_col, event_col,
        min_events=5, min_unique=3,
    )
    univ.to_csv(os.path.join(out_dir, "tcga_univariable_cox_feature_screening.tsv"), sep="\t", index=False)
    logger.info(f"[02] Univariable Cox: {len(univ)} features tested")

    # === 筛选策略（基于文献证据） ===
    # 标准流程: univariate Cox (p<0.2/top-200) → LASSO Cox → multivariate Cox
    # 参考: BMC Cancer 2022 (Jardillier et al.) — 16个TCGA癌症benchmark,
    #       top-200 univariate Cox p-value genes 为最优预筛选策略
    univ_ok = univ[univ["status"] == "ok"].copy()
    ok = univ_ok[univ_ok["fdr"].notna()]

    # Step 2a: FDR 门控（保留作为参考）
    fdr_sig = ok[ok["fdr"] <= 0.20].sort_values("fdr")
    fdr_sig = fdr_sig[~fdr_sig["likely_pseudogene"]].head(30)

    # Step 2b: LASSO Cox 回归筛选（文献标准方法）
    lasso_result = {"selected_genes": [], "lasso_coefs": pd.DataFrame(), "lambda_opt": np.nan}
    lasso_genes = []
    # ── LASSO 候选池构建（文献依据: BMC Cancer 2022 Jardillier et al.）──
    # 策略: 双轨制 — p<LASSO_P_THRESHOLD 放宽阈值 ∪ top-CAP 保底，确保候选池充分性
    # 参考: "Pre-screening of the top 200 genes in terms of single variable
    #        Cox model p-values is a practical way to reduce dimension"
    mask_lasso = (ok["p"] < LASSO_P_THRESHOLD) & (~ok["likely_pseudogene"])
    lasso_pool = ok[mask_lasso].sort_values("p")
    # 双轨制: p<0.2 基因 ∪ p值排序 top-N，取并集后截断到上限
    top_n_genes = ok.sort_values("p")[~ok["likely_pseudogene"]]["feature"].head(LASSO_CANDIDATE_CAP).tolist()
    p_thresh_genes = lasso_pool["feature"].tolist()
    lasso_input_genes = list(dict.fromkeys(p_thresh_genes + top_n_genes))[:LASSO_CANDIDATE_CAP]
    # 保底: 若仍不足 LASSO_FALLBACK_MIN 个，使用 p 值排序 top-50
    if len(lasso_input_genes) < LASSO_FALLBACK_MIN:
        lasso_input_genes = ok.sort_values("p")[~ok["likely_pseudogene"]]["feature"].head(LASSO_FALLBACK_MIN).tolist()
    # ═══ 新增: CausalEGM 降维（LASSO 之前）═══
    cegm_Z = None  # 标记是否成功获取潜在表征
    if CAUSAL_EGM_ENABLED and lasso_input_genes and len(train_ids) >= 30:
        try:
            from causal_egm_adapter import CausalEGMAdapter, _build_treatment_proxy
            cegm = CausalEGMAdapter(
                latent_dim=CAUSAL_EGM_LATENT_DIM, epochs=CAUSAL_EGM_EPOCHS,
                lr=CAUSAL_EGM_LR, batch_size=CAUSAL_EGM_BATCH_SIZE,
                flow_steps=CAUSAL_EGM_FLOW_STEPS, mode=CAUSAL_EGM_MODE,
                random_state=RANDOM_SEED,
            )
            cegm_features = feature_df[lasso_input_genes].reindex(train_ids).fillna(0)
            treatment_proxy = _build_treatment_proxy(train_clinical, cegm_features.index)
            outcome_time = pd.to_numeric(
                train_clinical.set_index("PATIENT_ID")
                .reindex(train_ids)[time_col], errors="coerce"
            )
            cegm_Z = cegm.fit_transform(cegm_features, treatment_proxy, outcome_time)
            logger.info(f"[CausalEGM] {cegm_Z.shape[1]}维潜在表征提取成功 "
                        f"({cegm_Z.shape[0]} samples)")
            cegm_Z.to_csv(os.path.join(out_dir, "causalegm_latent_Z.tsv"), sep="\t")
        except Exception as exc:
            logger.warning(f"[CausalEGM] 训练失败，回退到原始特征: {exc}")
            cegm_Z = None

    if lasso_input_genes and len(train_ids) >= 30:
        logger.info(f"[02] Step 2b: LASSO Cox regression on {len(lasso_input_genes)} candidate genes "
                    f"(threshold=p<{LASSO_P_THRESHOLD}, cap={LASSO_CANDIDATE_CAP})...")
        lasso_features = feature_df[lasso_input_genes].reindex(train_ids)
        lasso_result = lasso_cox_screening(
            train_clinical, lasso_features, time_col, event_col,
            n_splits=10, max_features=50,
            n_alphas=LASSO_N_ALPHAS, alpha_min_ratio=LASSO_ALPHA_MIN_RATIO,
            random_seed=RANDOM_SEED,
        )
        lasso_genes = lasso_result["selected_genes"]
        if not lasso_result["lasso_coefs"].empty:
            lasso_result["lasso_coefs"].to_csv(
                os.path.join(out_dir, "lasso_cox_coefficients.tsv"), sep="\t", index=False)
        logger.info(f"[02] LASSO Cox: {len(lasso_genes)} genes selected")

    # Step 2b-ext: Bootstrap 稳定性选择（可选，文献: Lancet子刊 bootstrap stability selection）
    selection_method = ""
    if BOOTSTRAP_ENABLED and lasso_genes and len(train_ids) >= 50:
        logger.info(f"[02] Step 2b-ext: Bootstrap stability selection "
                    f"(n={BOOTSTRAP_N}, threshold={BOOTSTRAP_THRESHOLD})...")
        # 仅对 LASSO 选出的基因做稳定性选择（而非全部 200 个候选基因）
        bootstrap_features = feature_df[lasso_genes].reindex(train_ids) if lasso_genes else lasso_features
        bs_result = bootstrap_stability_selection(
            train_clinical, bootstrap_features, time_col, event_col,
            n_bootstrap=BOOTSTRAP_N, selection_threshold=BOOTSTRAP_THRESHOLD,
            n_alphas=LASSO_N_ALPHAS, alpha_min_ratio=LASSO_ALPHA_MIN_RATIO,
            random_seed=RANDOM_SEED,
        )
        stable_genes = bs_result["selected_genes"]
        bs_result["selection_frequency"].to_csv(
            os.path.join(out_dir, "lasso_bootstrap_selection_frequency.tsv"),
            sep="\t", index=False,
        )
        if stable_genes:
            lasso_genes = stable_genes  # 用稳定性筛选结果替代原始LASSO结果
            selection_method = "lasso_cox_bootstrap_stable"
            logger.info(f"[02] Bootstrap stability: {len(stable_genes)} genes passed "
                        f"(threshold={BOOTSTRAP_THRESHOLD})")
        else:
            logger.warning(f"[02] Bootstrap stability: no genes passed "
                           f"threshold={BOOTSTRAP_THRESHOLD}, keeping original LASSO result")

    # Step 2c: 合并候选基因（FDR ∪ LASSO，去重，优先LASSO）
    extra = []
    if lasso_genes:
        # LASSO 优先，FDR补充到30个
        candidate_features = lasso_genes[:30]
        if len(candidate_features) < 30 and not fdr_sig.empty:
            extra = [g for g in fdr_sig["feature"].tolist() if g not in candidate_features]
            candidate_features.extend(extra[:30 - len(candidate_features)])
        if not selection_method.startswith("lasso_cox_bootstrap"):
            selection_method = "lasso_cox" + ("_fdr_supplement" if extra else "")
        elif extra:
            selection_method += "_fdr_supplement"
    elif not fdr_sig.empty:
        candidate_features = fdr_sig["feature"].tolist()
        selection_method = "fdr_0.20"
    else:
        # 最终回退：按 raw p-value top-30
        logger.info("[02] No FDR/LASSO genes, falling back to top-30 by p-value...")
        fallback_genes = ok[~ok["likely_pseudogene"]].sort_values("p").head(30)
        candidate_features = fallback_genes["feature"].tolist()
        selection_method = "raw_p_top30"

    logger.info(f"[02] Final candidates: {len(candidate_features)} genes (method={selection_method})")
    # ═══ 新增: DeepSurv 交叉验证 ═══
    deepsurv_results = {"c_index": np.nan, "enabled": DEEPSURV_ENABLED}
    if DEEPSURV_ENABLED and candidate_features and len(train_ids) >= 50:
        try:
            from deepsurv_head import DeepSurvHead
            if cegm_Z is not None and DEEPSURV_USE_CEGM_Z:
                ds_features = cegm_Z
                logger.info("[DeepSurv] 使用 CausalEGM 潜在表征作为输入")
            else:
                ds_features = feature_df[candidate_features].reindex(train_ids).fillna(0)
                logger.info("[DeepSurv] 使用候选基因表达矩阵作为输入")

            ds_clinical = train_clinical.set_index("PATIENT_ID").reindex(ds_features.index)
            ds_time = pd.to_numeric(ds_clinical[time_col], errors="coerce")
            ds_event = ds_clinical[event_col].astype(int)
            valid = ds_time.notna() & (ds_time > 0)

            ds = DeepSurvHead(
                hidden_dims=DEEPSURV_HIDDEN_DIMS, dropout=DEEPSURV_DROPOUT,
                l2_reg=DEEPSURV_L2_REG, lr=DEEPSURV_LR, epochs=DEEPSURV_EPOCHS,
                patience=DEEPSURV_PATIENCE, batch_size=DEEPSURV_BATCH_SIZE,
                random_state=RANDOM_SEED,
            )
            ds.fit(ds_features[valid], ds_time[valid], ds_event[valid])
            c_index = ds.score(ds_features[valid], ds_time[valid], ds_event[valid])
            deepsurv_results["c_index"] = c_index
            logger.info(f"[DeepSurv] 训练完成, C-index={c_index:.3f}")
            ds_importance = ds.get_feature_importance(ds_features[valid], ds_time[valid], ds_event[valid])
            ds_importance.to_csv(os.path.join(out_dir, "deepsurv_feature_importance.tsv"),
                                 sep="\t", index=False)
        except Exception as exc:
            logger.warning(f"[DeepSurv] 训练失败（不影响主流程）: {exc}")
            deepsurv_results["c_index"] = np.nan

    if not candidate_features:
        logger.info("[02] No candidates passed screening. Saving empty outputs.")
        pd.DataFrame().to_csv(os.path.join(out_dir, "tcga_multivariable_cox_features.tsv"), sep="\t", index=False)
        pd.DataFrame().to_csv(os.path.join(out_dir, "causal_priority_feature_table.tsv"), sep="\t", index=False)
        pd.DataFrame().to_csv(os.path.join(out_dir, "pathway_enrichment_results.tsv"), sep="\t", index=False)
        pd.DataFrame().to_csv(os.path.join(out_dir, "core_biomarker_evidence_matrix.tsv"), sep="\t", index=False)
        with open(os.path.join(out_dir, "final_gene_list.pkl"), "wb") as f:
            pickle.dump([], f)
        json.dump({"candidates": 0, "timestamp": timestamp}, open(os.path.join(out_dir, "gene_feature_config.json"), "w"), indent=2)
        return 0

    logger.info("[02] Step 3: Multivariable Cox (confounder-adjusted)...")
    clinical_adj = clinical.copy()
    if "AGE" not in clinical_adj.columns and "AGE_AT_DIAGNOSIS" in clinical_adj.columns:
        clinical_adj["AGE"] = pd.to_numeric(clinical_adj["AGE_AT_DIAGNOSIS"], errors="coerce")
    if "AJCC_PATHOLOGIC_TUMOR_STAGE" in clinical_adj.columns:
        clinical_adj["AJCC_PATHOLOGIC_TUMOR_STAGE"] = encode_ajcc_stage(clinical_adj["AJCC_PATHOLOGIC_TUMOR_STAGE"])

    multivar = multivariable_cox(
        clinical_adj,
        feature_df[candidate_features],
        time_col, event_col,
        candidate_features,
        confounders=["AGE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "SUBTYPE", "CANCER_TYPE_ACRONYM"],
        penalizer=0.05,
    )
    multivar.to_csv(os.path.join(out_dir, "tcga_multivariable_cox_features.tsv"), sep="\t", index=False)
    logger.info(f"[02] Multivariable Cox: {len(multivar)} features tested")

    logger.info("[02] Step 4: Causal RMST doubly-robust screening...")
    causal_table = pd.DataFrame()
    if train_ids and candidate_features:
        train_clinical_ids = [pid for pid in train_ids if pid in clinical_adj["PATIENT_ID"].astype(str).values]
        adj_cols = [c for c in ["AGE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE"] if c in clinical_adj.columns]
        if adj_cols and train_clinical_ids:
            # 修复：先设置索引，然后用索引进行过滤
            clin_indexed = clinical_adj.set_index("PATIENT_ID")
            train_mask = clin_indexed.index.astype(str).isin(train_clinical_ids)
            clin_adj = clin_indexed.loc[train_mask, adj_cols].copy()
            if "SEX" in clin_adj.columns:
                clin_adj["SEX"] = clin_adj["SEX"].map({"Male": 1, "Female": 0}).fillna(clin_adj["SEX"])
            clin_adj = clin_adj.astype(float, errors="ignore")

            feature_matrix_train = feature_df[candidate_features].reindex(train_clinical_ids)
            feature_matrix_train.index = pd.Index(train_clinical_ids[:len(feature_matrix_train)])

            endpoint_train = pd.DataFrame({
                "time_months": pd.to_numeric(clin_indexed.loc[train_mask, time_col], errors="coerce").to_numpy(dtype=float),
                "event": clin_indexed.loc[train_mask, event_col].astype(int).to_numpy(),
            }, index=train_clinical_ids[:len(feature_matrix_train)])

            causal_table = run_causal_screening(
                feature_matrix_train, clin_adj, endpoint_train,
                random_seed=RANDOM_SEED, max_features=len(candidate_features),
            )
            logger.info(f"[02] Causal screening: {len(causal_table)} features scored")

    base = univ[univ["feature"].isin(candidate_features)][
        ["feature", "n", "events", "coef", "hr", "p", "fdr", "ph_p", "likely_pseudogene"]
    ].copy()
    base = base.rename(columns={
        "coef": "coef_univariable", "hr": "hr_univariable",
        "p": "p_univariable", "fdr": "fdr_univariable", "ph_p": "ph_assumption_p"
    })
    if not multivar.empty:
        base = base.merge(multivar, on="feature", how="left")
    if not causal_table.empty:
        base = base.merge(
            causal_table[["feature", "ate_rmst", "p_value", "causal_priority_score",
                          "dose_response_slope", "spearman_with_survival_time"]].rename(
                columns={"ate_rmst": "ate", "p_value": "causal_p_value"}
            ),
            on="feature", how="left", suffixes=("", "_causal"),
        )

    base["causal_evidence_level"] = np.where(
        base.get("p_adj", pd.Series(1.0, index=base.index)).fillna(1) < 0.10,
        "observational_cox_adjusted",
        "univariable_prognostic_only",
    )
    base.to_csv(os.path.join(out_dir, "causal_priority_feature_table.tsv"), sep="\t", index=False)

    logger.info("[02] Step 5: Pathway enrichment...")
    background_genes = feature_df.columns.tolist()
    enrich = pathway_enrichment(candidate_features, background_genes,
                                hallmark_path, kegg_path, go_bp_path, out_dir)
    logger.info(f"[02] Pathway enrichment: {len(enrich)} gene sets tested")

    logger.info("[02] Step 6: CPTAC mechanism validation...")
    cptac_table = mechanism_validation_cptac(candidate_features, cptac_protein_path, cptac_rna_path, out_dir)
    logger.info(f"[02] CPTAC validation: {len(cptac_table)} genes checked")

    logger.info("[02] Step 6b: Cohort-level mechanism enrichment tests...")
    # 蛋白协调性检验
    protein_enrich = mechanism_protein_concordance(
        candidate_features, background_genes, cptac_protein_path, cptac_rna_path, out_dir)
    # 磷酸化位点富集
    phospho_enrich = mechanism_phospho_enrichment(
        candidate_features, background_genes, cptac_phospho_path, out_dir)
    # CNV + 甲基化协调性
    cnv_meth_enrich = mechanism_cnv_methylation_concordance(
        candidate_features, background_genes, cnv_path, methylation_path, out_dir)
    # 多组学嵌入空间分离度
    multiomics_enrich = mechanism_multiomics_concordance(
        candidate_features, multiomics_embedding_path, feature_df, out_dir)
    # 汇总
    cohort_mechanism_df = mechanism_enrichment_summary(
        protein_enrich, phospho_enrich, cnv_meth_enrich, multiomics_enrich, out_dir)
    logger.info(f"[02] Cohort-level mechanism enrichment: {len(cohort_mechanism_df)} tests performed")

    logger.info("[02] Step 7: Evidence matrix...")
    evidence = build_evidence_matrix(causal_table, univ, multivar, cptac_table,
                                      candidate_features, out_dir,
                                      cohort_mechanism_df=cohort_mechanism_df)

    logger.info("[02] Step 8: Saving final outputs...")
    with open(os.path.join(out_dir, "final_gene_list.pkl"), "wb") as f:
        pickle.dump(candidate_features, f)

    config = {
        "timestamp": timestamp,
        "top_variance_genes": 2000,
        "fdr_threshold": 0.20,
        "top_priority": 30,
        "min_events_per_feature": 5,
        "min_unique_values_per_feature": 3,
        "multivariable_penalizer": 0.05,
        "tau_months": FIXED_TAU_MONTHS,
        "confounders": ["AGE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "SUBTYPE", "CANCER_TYPE_ACRONYM"],
        "selection_method": selection_method,
        "lasso_lambda_opt": lasso_result.get("lambda_opt", np.nan),
        "lasso_n_selected": len(lasso_genes),
        "lasso_prescreen_p_threshold": LASSO_P_THRESHOLD,
        "lasso_candidate_cap": LASSO_CANDIDATE_CAP,
        "lasso_n_input_genes": len(lasso_input_genes) if lasso_input_genes else 0,
        "lasso_n_alphas": LASSO_N_ALPHAS,
        "lasso_alpha_min_ratio": LASSO_ALPHA_MIN_RATIO,
        "bootstrap_enabled": BOOTSTRAP_ENABLED,
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_threshold": BOOTSTRAP_THRESHOLD,
        "n_variance_selected": len(selected),
        "n_univariable_tested": len(univ),
        "n_fdr_significant": len(fdr_sig),
        "n_candidates": len(candidate_features),
        "n_causal_scored": len(causal_table),
        "n_pathway_tested": len(enrich),
        "n_cohort_mechanism_tests": len(cohort_mechanism_df) if cohort_mechanism_df is not None else 0,
        "n_mechanism_significant": int(cohort_mechanism_df["significant"].sum()) if cohort_mechanism_df is not None and "significant" in cohort_mechanism_df.columns else 0,
        "candidate_genes": candidate_features,
        "causal_egm_enabled": CAUSAL_EGM_ENABLED,
        "causal_egm_latent_dim": CAUSAL_EGM_LATENT_DIM,
        "causal_egm_z_extracted": cegm_Z is not None,
        "deepsurv_enabled": DEEPSURV_ENABLED,
        "deepsurv_c_index": deepsurv_results.get("c_index", np.nan),
    }
    with open(os.path.join(out_dir, "gene_feature_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"[02] Complete. {len(candidate_features)} candidate genes saved.")
    logger.info(f"[02] Outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
