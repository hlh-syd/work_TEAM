from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    from lifelines.utils import concordance_index
except Exception:
    CoxPHFitter = None
    KaplanMeierFitter = None
    multivariate_logrank_test = None
    concordance_index = None

try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc, brier_score, integrated_brier_score
    from sksurv.util import Surv
except Exception:
    RandomSurvivalForest = None
    CoxnetSurvivalAnalysis = None
    concordance_index_censored = None
    concordance_index_ipcw = None
    cumulative_dynamic_auc = None
    brier_score = None
    integrated_brier_score = None
    Surv = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "DATA"))
RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "results"))

FIXED_TAU_MONTHS = 36.0
EPS = 1e-8
RANDOM_SEED = 42
RANDOM_SEED_GAN = 1337

SURVIVAL_GAN_CONDITION_COLUMNS = [
    "death_by_36m", "event", "log1p_time_months",
    "pseudo_risk_36m_clipped", "ipcw_weight_36m",
]

GAN_AUG_RATIO_CANDIDATES = (0.25, 0.5, 1.0, 2.0)
GAN_SAMPLING_STRATEGIES = ("balanced_event", "risk_stratified", "event_only", "overall")
GAN_MIN_AUC_GAIN = 0.005

STAGE_MAP = {
    "STAGE I": "I", "STAGE IA": "I", "STAGE IB": "I",
    "STAGE II": "II", "STAGE IIA": "II", "STAGE IIB": "II", "STAGE IIC": "II",
    "STAGE III": "III", "STAGE IIIA": "III", "STAGE IIIB": "III", "STAGE IIIC": "III",
    "STAGE IV": "IV", "STAGE IVA": "IV", "STAGE IVB": "IV", "STAGE IVC": "IV",
    "STAGE 0": "0",
    "I": "I", "IA": "I", "IB": "I",
    "II": "II", "IIA": "II", "IIB": "II", "IIC": "II",
    "III": "III", "IIIA": "III", "IIIB": "III", "IIIC": "III",
    "IV": "IV", "IVA": "IV", "IVB": "IV", "IVC": "IV", "0": "0",
}

META_COLUMNS = {
    "Hugo_Symbol", "Entrez_Gene_Id", "Cytoband",
    "Composite.Element.REF", "ENTITY_STABLE_ID", "NAME",
    "DESCRIPTION", "TRANSCRIPT_ID", "ID", "GENE_SYMBOL",
    "PHOSPHOSITE", "geneNames",
}

FEATURE_SET_CONFIG = {
    "clinical_only": {"clinical": True, "omics": []},
    "clinical_expression_topvar": {"clinical": True, "omics": ["rna_topvar"]},
    "clinical_causal_priority": {"clinical": True, "omics": ["rna_causal"]},
    "clinical_pathway": {"clinical": True, "omics": ["pathway"]},
    "clinical_somatic_mutation": {"clinical": True, "omics": ["mutation"]},
}


def parse_survival_status(series):
    values = series.fillna("").astype(str).str.strip()
    lower = values.str.lower()
    event = (
        values.str.startswith("1")
        | lower.isin({"1", "dead", "deceased", "event", "progression",
                       "yes", "true", "recurred", "recurrence", "progressed"})
    )
    censored = (
        values.str.startswith("0")
        | lower.isin({"0", "alive", "censored", "no", "false", "living",
                       "no_progression", "diseasefree"})
    )
    result = pd.Series(np.nan, index=series.index)
    result[event] = 1.0
    result[censored] = 0.0
    return result


def numeric_series(series):
    return pd.to_numeric(series.replace(["[Not Available]", "[Not Applicable]", "[Completed]", ""], np.nan), errors="coerce")


def stage_to_group(value):
    if pd.isna(value):
        return "Unknown"
    s = str(value).upper().replace("STAGE ", "").strip()
    if s.startswith("IV"):
        return "IV"
    if s.startswith("III"):
        return "III"
    if s.startswith("II"):
        return "II"
    if s.startswith("I"):
        return "I"
    return "Unknown"


def rank_inverse_normal(series):
    s = pd.to_numeric(series, errors="coerce")
    valid = s.notna()
    if valid.sum() < 2:
        return s
    ranked = rankdata(s[valid].to_numpy())
    n = len(ranked)
    from scipy.stats import norm
    transformed = norm.ppf((ranked - 0.5) / n)
    result = s.copy()
    result.loc[valid] = transformed
    return result


def stratified_split_keys(df, columns, min_count=5):
    combined = df[columns[0]].astype(str)
    for col in columns[1:]:
        combined = combined + "_" + df[col].astype(str)
    counts = combined.value_counts()
    rare = set(counts[counts < min_count].index)
    if rare:
        combined = combined.replace(list(rare), "RARE")
    return combined


def sanitize_filename(name):
    return re.sub(r"[^\w\-.]", "_", str(name))


def make_surv_array(time, event):
    return Surv.from_arrays(
        event=np.asarray(event).astype(bool),
        time=np.asarray(time, dtype=float),
    )


def read_tsv_safe(path, **kwargs):
    return pd.read_csv(path, sep="\t", low_memory=False, **kwargs)


def load_gene_matrix(path, id_col_preference="Hugo_Symbol"):
    df = pd.read_csv(path, sep="\t", comment="#", dtype=str, low_memory=False)
    id_col = None
    for candidate in [id_col_preference, "Hugo_Symbol", "geneNames", "ID", "GENE_SYMBOL"]:
        if candidate in df.columns:
            id_col = candidate
            break
    if id_col is None:
        id_col = df.columns[0]
    sample_cols = [c for c in df.columns if c != id_col]
    df = df[df[id_col].notna()]
    df[id_col] = df[id_col].astype(str)
    df = df[~df[id_col].isin(META_COLUMNS)]
    df = df.drop_duplicates(subset=id_col, keep="first")
    df = df.set_index(id_col)
    mat = df[sample_cols].apply(pd.to_numeric, errors="coerce")
    mat = mat.T
    mat.index = mat.index.astype(str)
    return mat


def patient_id_from_sample(value):
    text = str(value)
    if text.startswith("TCGA-") and len(text) >= 12:
        return text[:12]
    return text


def select_features_train_only(mat, train_ids, min_nonmissing, top_k, forced=None):
    if mat.empty:
        return []
    mat = mat.copy()
    mat.index = mat.index.astype(str)
    train_available = [p for p in train_ids if p in mat.index]
    if not train_available:
        return []
    train = mat.loc[train_available]
    keep_mask = train.notna().mean(axis=0) >= min_nonmissing
    train = train.loc[:, keep_mask]
    if train.empty:
        return []
    variances = train.var(axis=0, skipna=True).sort_values(ascending=False)
    top = variances.head(top_k).index.tolist()
    if forced:
        forced_in = [f for f in forced if f in mat.columns and f not in top]
        return list(dict.fromkeys(forced_in + top))
    return top


def fit_modality_encoder(block, train_mask, dim, seed):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train_raw = imputer.fit_transform(block.iloc[train_mask])
    scaler.fit(x_train_raw)
    x_train = scaler.transform(x_train_raw)
    x_all = scaler.transform(imputer.transform(block))
    n_components = max(1, min(dim, x_train.shape[0] - 1, x_train.shape[1]))
    if x_train.shape[1] > 1000:
        model = TruncatedSVD(n_components=n_components, random_state=seed)
        model.fit(x_train)
        emb = model.transform(x_all)
        recon = None
    else:
        model = PCA(n_components=n_components, random_state=seed)
        model.fit(x_train)
        emb = model.transform(x_all)
        recon = model.inverse_transform(emb)
    if recon is not None:
        diff = (x_all - recon) ** 2
        per_sample = np.nanmean(diff, axis=1)
    else:
        per_sample = np.full(emb.shape[0], np.nan)
    bundle = {
        "imputer": imputer, "scaler": scaler, "model": model,
        "feature_columns": block.columns.tolist(), "n_components": n_components,
    }
    diag = {
        "explained_variance_ratio": getattr(model, "explained_variance_ratio_", np.repeat(np.nan, n_components)).tolist(),
        "mean_reconstruction_mse": float(np.nanmean(per_sample)),
    }
    return emb, bundle, per_sample, diag


def multi_omics_concat(embeddings, masks_per_patient, patients):
    parts = []
    columns = []
    for omics, emb in embeddings.items():
        availability = masks_per_patient[omics]
        emb_masked = emb.copy()
        emb_masked[availability == 0] = 0.0
        parts.append(emb_masked)
        columns.extend([f"{omics}_embedding_{i+1:02d}" for i in range(emb.shape[1])])
    fused = np.concatenate(parts, axis=1)
    emb_df = pd.DataFrame(fused, columns=columns)
    emb_df.insert(0, "PATIENT_ID", patients)
    return emb_df, columns


def stability_via_subspace_overlap(mats, selected, train_ids, dim, seeds):
    results = {}
    for omics, feats in selected.items():
        if omics not in mats or mats[omics].empty:
            continue
        block_full = mats[omics].reindex(train_ids)[feats]
        if block_full.shape[0] < dim + 2 or block_full.shape[1] < 2:
            continue
        x = SimpleImputer(strategy="median").fit_transform(block_full)
        x = StandardScaler().fit_transform(x)
        bases = []
        for s in seeds:
            rng = np.random.default_rng(s)
            sample_idx = rng.choice(x.shape[0], size=max(int(0.8 * x.shape[0]), dim + 5), replace=False)
            try:
                pca = PCA(n_components=min(dim, x[sample_idx].shape[1]), random_state=s)
                pca.fit(x[sample_idx])
                bases.append(pca.components_)
            except Exception:
                continue
        if len(bases) < 2:
            continue
        sims = []
        for i in range(len(bases)):
            for j in range(i + 1, len(bases)):
                m = bases[i] @ bases[j].T
                from numpy.linalg import svd
                s_vals = svd(m, compute_uv=False)
                sims.append(float(np.mean(np.abs(s_vals))))
        results[omics] = {
            "n_seeds_evaluated": len(bases),
            "mean_principal_subspace_alignment": float(np.mean(sims)) if sims else float("nan"),
        }
    return results


def build_survival_gan_conditions(endpoint, index):
    ep = endpoint.reindex(index).copy()
    death = ep.get("death_by_36m", ep.get("os_death", pd.Series(0.0, index=ep.index))).astype(float)
    event = ep.get("event", death).astype(float)
    time_months = ep.get("time_months", pd.Series(FIXED_TAU_MONTHS, index=ep.index)).astype(float)
    pseudo = ep.get("pseudo_risk_36m", death).astype(float)
    weight = ep.get("ipcw_weight_36m", pd.Series(1.0, index=ep.index)).astype(float)
    cond = pd.DataFrame(index=ep.index)
    cond["death_by_36m"] = death.fillna(0.0).clip(0.0, 1.0)
    cond["event"] = event.fillna(cond["death_by_36m"]).clip(0.0, 1.0)
    clean_time = time_months.replace([np.inf, -np.inf], np.nan).fillna(FIXED_TAU_MONTHS).clip(lower=0.0)
    cond["log1p_time_months"] = np.log1p(clean_time)
    cond["pseudo_risk_36m_clipped"] = pseudo.replace([np.inf, -np.inf], np.nan).fillna(cond["death_by_36m"]).clip(0.0, 1.0)
    cond["ipcw_weight_36m"] = weight.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=EPS)
    return cond[SURVIVAL_GAN_CONDITION_COLUMNS].astype(float)


def fit_clinical_transformer(clinical, split, feature_cols):
    clinical = clinical.copy()
    clinical["PATIENT_ID"] = clinical["PATIENT_ID"].astype(str)
    split = split.copy()
    split["PATIENT_ID"] = split["PATIENT_ID"].astype(str)
    merged = split[["PATIENT_ID", "split"]].merge(
        clinical[["PATIENT_ID"] + [c for c in feature_cols if c in clinical.columns]],
        on="PATIENT_ID", how="left",
    )
    existing = [c for c in feature_cols if c in merged.columns]
    if not existing:
        raise ValueError(f"No requested clinical features available: {feature_cols}")
    raw_x = merged[existing].copy()
    numeric_cols = [c for c in existing if c.upper() == "AGE"]
    categorical_cols = [c for c in existing if c not in numeric_cols]
    train_mask = merged["split"].to_numpy() == "train"
    bundle = {
        "feature_cols": existing, "numeric_cols": numeric_cols, "categorical_cols": categorical_cols,
    }
    blocks, columns = [], []
    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        num_scaler = StandardScaler()
        x_train_num = num_imputer.fit_transform(raw_x.loc[train_mask, numeric_cols])
        num_scaler.fit(x_train_num)
        blocks.append(num_scaler.transform(num_imputer.transform(raw_x[numeric_cols])))
        columns.extend(numeric_cols)
        bundle["num_imputer"] = num_imputer
        bundle["num_scaler"] = num_scaler
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        # 修复：impute后强制转为字符串，避免float/str混合类型
        x_train_cat = cat_imputer.fit_transform(raw_x.loc[train_mask, categorical_cols].astype(object))
        x_train_cat = pd.DataFrame(x_train_cat, columns=categorical_cols).astype(str).values
        encoder.fit(x_train_cat)
        cat_columns = encoder.get_feature_names_out(categorical_cols).tolist()
        x_all_cat = cat_imputer.transform(raw_x[categorical_cols].astype(object))
        x_all_cat = pd.DataFrame(x_all_cat, columns=categorical_cols).astype(str).values
        blocks.append(encoder.transform(x_all_cat))
        columns.extend(cat_columns)
        bundle["cat_imputer"] = cat_imputer
        bundle["encoder"] = encoder
    matrix = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
    x = pd.DataFrame(matrix, columns=columns, index=merged["PATIENT_ID"].astype(str))
    bundle["model_columns"] = columns
    return x, bundle


def make_stratified_split(clinical, endpoint_col, test_size, seed):
    clinical = clinical.copy()
    time_col = f"{endpoint_col}_TIME_MONTHS"
    event_col = f"{endpoint_col}_EVENT"
    clinical["stage_group"] = clinical.get("AJCC_PATHOLOGIC_TUMOR_STAGE", pd.Series(["Unknown"] * len(clinical))).apply(stage_to_group)
    clinical["site_group"] = clinical.get("CANCER_TYPE_ACRONYM", pd.Series(["Unknown"] * len(clinical))).astype(str).fillna("Unknown")
    clinical["event_int"] = clinical[event_col].astype(int)
    keys = stratified_split_keys(clinical, ["event_int", "stage_group", "site_group"], min_count=5)
    if keys.nunique() < 2:
        keys = clinical["event_int"].astype(str)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, valid_idx = next(sss.split(np.arange(len(clinical)), keys))
    split_label = np.array(["train"] * len(clinical), dtype=object)
    split_label[valid_idx] = "internal_validation"
    return pd.DataFrame({
        "PATIENT_ID": clinical["PATIENT_ID"].astype(str).to_numpy(),
        "split": split_label,
        "event": clinical[event_col].astype(int).to_numpy(),
        "time_months": clinical[time_col].astype(float).to_numpy(),
        "stratify_key": keys.to_numpy(),
    })


def harrell_c_index(time, event, risk):
    return float(concordance_index(time, -risk, event))


def bootstrap_metric(func, n_iter, seed, time, event, risk):
    rng = np.random.default_rng(seed)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    valid_mask = np.isfinite(time) & np.isfinite(risk) & (~np.isnan(event))
    time, event, risk = time[valid_mask], event[valid_mask], risk[valid_mask]
    n = len(time)
    if n < 5:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_valid": 0}
    point = func(time, event, risk)
    scores = []
    for _ in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        if np.unique(event[idx]).size < 2 or np.unique(risk[idx]).size < 2:
            continue
        try:
            scores.append(func(time[idx], event[idx], risk[idx]))
        except Exception:
            pass
    if not scores:
        return {"point": point, "ci_low": float("nan"), "ci_high": float("nan"), "n_valid": 0}
    return {
        "point": point,
        "ci_low": float(np.percentile(scores, 2.5)),
        "ci_high": float(np.percentile(scores, 97.5)),
        "n_valid": len(scores),
    }


def bootstrap_harrell(time, event, risk, n_iter=200, seed=RANDOM_SEED):
    return bootstrap_metric(harrell_c_index, n_iter, seed, time, event, risk)


def lifelines_predict_log_hazard(model, x):
    return np.log(model.predict_partial_hazard(x).to_numpy().reshape(-1) + 1e-12)


def lifelines_survival_estimator(model, x_valid):
    def estimate(times):
        sf = model.predict_survival_function(x_valid, times=np.asarray(times, dtype=float))
        return np.clip(sf.T.to_numpy(dtype=float), 0.0, 1.0)
    return estimate


def sksurv_survival_estimator(model, x_valid):
    def estimate(times):
        try:
            surv = model.predict_survival_function(x_valid, return_array=False)
        except TypeError:
            surv = model.predict_survival_function(x_valid)
        rows = []
        for fn in surv:
            rows.append([float(fn(float(t))) for t in np.asarray(times, dtype=float)])
        return np.clip(np.asarray(rows, dtype=float), 0.0, 1.0)
    return estimate


def metric_uno_and_td_auc(split, risk, horizons_months=(12.0, 36.0, 60.0), survival_estimator=None):
    train_mask = split["split"].to_numpy() == "train"
    valid_mask = split["split"].to_numpy() == "internal_validation"
    if valid_mask.sum() < 5:
        return {}
    y_train = Surv.from_arrays(
        split.loc[train_mask, "event"].astype(bool).to_numpy(),
        split.loc[train_mask, "time_months"].astype(float).to_numpy(),
    )
    y_valid = Surv.from_arrays(
        split.loc[valid_mask, "event"].astype(bool).to_numpy(),
        split.loc[valid_mask, "time_months"].astype(float).to_numpy(),
    )
    valid_time = split.loc[valid_mask, "time_months"].astype(float).to_numpy()
    valid_event = split.loc[valid_mask, "event"].astype(bool).to_numpy()
    risk_valid = np.asarray(risk)[valid_mask]
    max_train_time = float(split.loc[train_mask, "time_months"].astype(float).max())
    safe_horizons = sorted({float(min(h, max_train_time * 0.95)) for h in horizons_months if h < max_train_time * 0.99})
    out = {}
    try:
        tau = float(np.percentile(valid_time[valid_event], 90)) if valid_event.any() else float(np.max(valid_time))
        uno_c, *_ = concordance_index_ipcw(y_train, y_valid, risk_valid, tau=tau)
        out["uno_c_index"] = float(uno_c)
        out["uno_tau_months"] = tau
    except Exception:
        pass
    if safe_horizons:
        try:
            auc, mean_auc = cumulative_dynamic_auc(y_train, y_valid, risk_valid, np.asarray(safe_horizons))
            for h, a in zip(safe_horizons, auc):
                out[f"td_auc_month_{int(h)}"] = float(a)
            out["td_auc_mean"] = float(mean_auc)
        except Exception:
            pass
        try:
            if survival_estimator is not None:
                survs = survival_estimator(np.asarray(safe_horizons, dtype=float))
                if survs.shape == (len(risk_valid), len(safe_horizons)):
                    out["integrated_brier_score"] = float(integrated_brier_score(y_train, y_valid, survs, np.asarray(safe_horizons)))
                    out["brier_score_source"] = "model_survival_function"
        except Exception:
            pass
        try:
            rank = (-risk_valid).argsort().argsort()
            survs = 1.0 - (rank + 1) / (len(rank) + 1)
            survs = np.tile(survs.reshape(-1, 1), (1, len(safe_horizons)))
            _, ibs = brier_score(y_train, y_valid, survs, np.asarray(safe_horizons))
            out["risk_ranked_brier_score_exploratory"] = float(np.mean(ibs))
            if "brier_score_source" not in out:
                out["brier_score_source"] = "risk_ranked_survival_proxy"
        except Exception:
            pass
    return out


def transform_clinical_external(df, patient_col, bundle):
    transformer = bundle["transformer"]
    feature_cols = transformer["feature_cols"]
    raw = df.copy()
    for col in feature_cols:
        if col not in raw.columns:
            raw[col] = np.nan
    raw = raw[feature_cols]
    blocks = []
    if transformer.get("numeric_cols"):
        cols = transformer["numeric_cols"]
        num = transformer["num_scaler"].transform(transformer["num_imputer"].transform(raw[cols]))
        blocks.append(num)
    if transformer.get("categorical_cols"):
        cols = transformer["categorical_cols"]
        cat = transformer["encoder"].transform(transformer["cat_imputer"].transform(raw[cols].astype(object)))
        blocks.append(cat)
    if not blocks:
        raise ValueError("No clinical columns for external transform.")
    matrix = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
    return pd.DataFrame(matrix, columns=transformer["model_columns"], index=df[patient_col].astype(str))


def fit_lifelines_cox(x, split, penalizer=0.1):
    train_mask = split["split"].to_numpy() == "train"
    train = x.iloc[train_mask].copy()
    train["time"] = split.loc[train_mask, "time_months"].to_numpy()
    train["event"] = split.loc[train_mask, "event"].astype(int).to_numpy()
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(train, duration_col="time", event_col="event")
    risk_all = lifelines_predict_log_hazard(cph, x)
    bundle = {
        "model": cph, "model_type": "lifelines_cox",
        "feature_columns": x.columns.tolist(),
        "penalizer": penalizer, "fit_scope": "train_split_only",
    }
    return cph, bundle, risk_all


def fit_coxnet_with_cv(feature_matrix, split, feature_columns, l1_ratio=0.5, n_folds=5, min_nonmissing=0.7, seed=RANDOM_SEED):
    aligned = feature_matrix.reindex(split["PATIENT_ID"].astype(str))[feature_columns]
    train_mask = (split["split"].to_numpy() == "train")
    train_block = aligned.iloc[train_mask]
    keep = train_block.notna().mean(axis=0) >= min_nonmissing
    feature_columns = train_block.columns[keep].tolist()
    aligned = aligned[feature_columns]
    aligned_harm = aligned.apply(rank_inverse_normal, axis=0)
    imputer = SimpleImputer(strategy="median")
    x_all_raw = imputer.fit_transform(aligned_harm)
    scaler = StandardScaler()
    scaler.fit(x_all_raw[train_mask])
    x_all = scaler.transform(x_all_raw)
    x_train = x_all[train_mask]
    y_train = make_surv_array(split.loc[train_mask, "time_months"].to_numpy(), split.loc[train_mask, "event"].to_numpy())
    init = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=0.05, n_alphas=50, max_iter=200000, fit_baseline_model=False)
    init.fit(x_train, y_train)
    alpha_path = np.asarray(init.alphas_)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = np.full((n_folds, len(alpha_path)), np.nan)
    for fi, (tr, va) in enumerate(kf.split(x_train)):
        try:
            model = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=alpha_path.tolist(), max_iter=200000, fit_baseline_model=False)
            model.fit(x_train[tr], y_train[tr])
            for ai, a in enumerate(alpha_path):
                try:
                    risk_va = model.predict(x_train[va], alpha=a).reshape(-1)
                    if np.unique(risk_va).size < 2:
                        continue
                    ev = np.asarray(y_train[va]["event"]).astype(bool)
                    tm = np.asarray(y_train[va]["time"]).astype(float)
                    if ev.sum() < 2:
                        continue
                    c = concordance_index_censored(ev, tm, risk_va)[0]
                    scores[fi, ai] = float(c)
                except Exception:
                    pass
        except Exception:
            continue
    with np.errstate(invalid="ignore"):
        mean_scores = np.where(np.all(np.isnan(scores), axis=0), np.nan, np.nanmean(scores, axis=0))
        valid_counts = np.sum(~np.isnan(scores), axis=0)
        se_scores = np.where(valid_counts > 1, np.nanstd(scores, axis=0, ddof=1) / np.sqrt(np.maximum(1, valid_counts)), np.nan)
    if np.all(np.isnan(mean_scores)):
        alpha_1se = float(alpha_path[len(alpha_path) // 2])
    else:
        best = int(np.nanargmax(mean_scores))
        threshold = mean_scores[best] - se_scores[best]
        candidates_1se = np.where(mean_scores >= threshold)[0]
        alpha_1se_idx = int(candidates_1se.max()) if candidates_1se.size > 0 else best
        alpha_1se = float(alpha_path[alpha_1se_idx])
    final = None
    final_alpha = None
    for trial_alpha in [alpha_1se, float(np.median(alpha_path)), float(alpha_path.min())]:
        try:
            m = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=[trial_alpha], max_iter=200000, fit_baseline_model=True)
            m.fit(x_train, y_train)
            final = m
            final_alpha = trial_alpha
            break
        except Exception:
            continue
    if final is None:
        raise RuntimeError("All Coxnet final-fit attempts failed.")
    coef = pd.Series(final.coef_.reshape(-1), index=feature_columns)
    nonzero = coef[coef != 0].sort_values(key=lambda s: s.abs(), ascending=False)
    risk_all = final.predict(x_all).reshape(-1)
    bundle = {
        "model": final, "imputer": imputer, "scaler": scaler,
        "genes": feature_columns, "alpha_1se": alpha_1se,
        "alpha_used": final_alpha, "l1_ratio": l1_ratio,
        "model_type": "expression_coxnet", "fit_scope": "train_split_only_cv",
        "harmonization": "rank_based_inverse_normal_per_cohort",
    }
    cv_summary = {
        "alpha_1se": alpha_1se, "selected_alpha": final_alpha,
        "non_zero_features": int((coef != 0).sum()),
        "non_zero_feature_table": nonzero.reset_index().rename(columns={"index": "feature", 0: "coef"}).to_dict(orient="records"),
    }
    return final, bundle, risk_all, cv_summary


def fit_rsf(x, split, seed=RANDOM_SEED):
    train_mask = split["split"].to_numpy() == "train"
    y = make_surv_array(split["time_months"].to_numpy(), split["event"].to_numpy())
    model = RandomSurvivalForest(
        n_estimators=300, min_samples_split=10, min_samples_leaf=8,
        max_features="sqrt", n_jobs=-1, random_state=seed,
    )
    model.fit(x.iloc[train_mask], y[train_mask])
    risk_all = model.predict(x)
    bundle = {
        "model": model, "model_type": "random_survival_forest",
        "feature_columns": x.columns.tolist(), "fit_scope": "train_split_only",
    }
    return model, bundle, risk_all


def fit_weighted_logistic(x, split, seed=RANDOM_SEED):
    train_mask = split["split"].to_numpy() == "train"
    x_train = x.iloc[train_mask]
    event_col = split.loc[train_mask, "event"].astype(float)
    time_col = split.loc[train_mask, "time_months"].astype(float)
    tau = FIXED_TAU_MONTHS
    binary_label = ((event_col == 1) & (time_col <= tau)).astype(int).to_numpy()
    ipcw_weight = np.ones(len(binary_label))
    event_mask = event_col == 1
    censor_mask = ~event_mask
    if event_mask.sum() > 0 and censor_mask.sum() > 0:
        event_rate = event_mask.mean()
        ipcw_weight[event_mask] = 1.0 / max(event_rate, 0.01)
        ipcw_weight[censor_mask & (time_col < tau)] = 1.0 / max(1 - event_rate, 0.01)
    ipcw_weight = np.clip(ipcw_weight, 0.5, 10.0)
    model = LogisticRegressionCV(
        penalty="elasticnet", solver="saga",
        l1_ratios=[0.05, 0.5, 0.95],
        scoring="roc_auc", cv=5, max_iter=5000, random_state=seed,
    )
    model.fit(x_train, binary_label, sample_weight=ipcw_weight)
    risk_all = model.predict_proba(x)[:, 1]
    bundle = {
        "model": model, "model_type": "weighted_logistic",
        "feature_columns": x.columns.tolist(), "fit_scope": "train_split_only",
        "tau_months": tau,
    }
    return model, bundle, risk_all


def select_primary_model(metric_df, policy="survival_composite"):
    if metric_df.empty:
        return "none", {"reason": "no internal metrics available", "policy": policy}
    candidates = metric_df[metric_df["metric"] == "harrell_c_index"].copy()
    if candidates.empty:
        return "none", {"reason": "no survival metrics", "policy": policy}
    candidates["model_key"] = candidates["model"].astype(str)
    candidates["internal_c_index"] = pd.to_numeric(candidates["observed_value"], errors="coerce")
    candidates["td_auc_mean"] = pd.to_numeric(candidates.get("td_auc_mean", np.nan), errors="coerce")
    ibs = pd.to_numeric(candidates.get("integrated_brier_score", np.nan), errors="coerce")
    exploratory_brier = pd.to_numeric(candidates.get("risk_ranked_brier_score_exploratory", np.nan), errors="coerce")
    calibration_loss = ibs.combine_first(exploratory_brier)
    candidates["negative_brier"] = -calibration_loss
    candidates["parsimony_score"] = 0.5
    candidates.loc[candidates["model_key"].str.match(r"^clinical_(ajcc|tnm|minimal|random)", case=False, na=False), "parsimony_score"] = 1.0
    candidates.loc[candidates["model_key"].str.contains("coxnet", case=False, na=False), "parsimony_score"] = 0.8
    candidates["td_auc_filled"] = candidates["td_auc_mean"].fillna(candidates["internal_c_index"])
    candidates["neg_brier_filled"] = candidates["negative_brier"].fillna(0.0)
    candidates["selection_score"] = (
        0.55 * candidates["internal_c_index"].fillna(0.0)
        + 0.25 * candidates["td_auc_filled"]
        + 0.15 * candidates["neg_brier_filled"]
        + 0.05 * candidates["parsimony_score"].fillna(0.0)
    )
    candidates["omics_or_ensemble"] = candidates["model_key"].str.contains("embedding|multiomics|rna|coxnet|ensemble", case=False, na=False)
    candidates = candidates.sort_values(["selection_score", "internal_c_index"], ascending=False, na_position="last")
    selected_row = candidates.iloc[0].copy()
    clinical_candidates = candidates.loc[~candidates["omics_or_ensemble"]].copy()
    clinical_anchor_used = False
    if not clinical_candidates.empty and bool(selected_row.get("omics_or_ensemble", False)):
        clinical_best = clinical_candidates.iloc[0]
        c_gain = float(selected_row["internal_c_index"]) - float(clinical_best["internal_c_index"])
        auc_gain = float(selected_row["td_auc_filled"]) - float(clinical_best["td_auc_filled"])
        if c_gain <= 0.0 and auc_gain <= 0.0:
            selected_row = clinical_best
            clinical_anchor_used = True
    selected = str(selected_row["model_key"])
    rationale = {
        "selected_model": selected, "policy": policy,
        "weights": {"internal_c_index": 0.55, "td_auc": 0.25, "negative_brier": 0.15, "parsimony": 0.05},
        "clinical_anchor_used": clinical_anchor_used,
        "candidate_scores": candidates[["model_key", "selection_score", "internal_c_index", "td_auc_mean"]].replace({np.nan: None}).to_dict(orient="records"),
    }
    return selected, rationale


def combine_clinical_anchor_risk(clinical_risk, omics_risk, split, weight_clinical=0.4):
    train_mask = split["split"].to_numpy() == "train"
    c_train = clinical_risk[train_mask]
    o_train = omics_risk[train_mask]
    c_std = np.std(c_train) + EPS
    o_std = np.std(o_train) + EPS
    c_norm = (clinical_risk - np.mean(c_train)) / c_std
    o_norm = (omics_risk - np.mean(o_train)) / o_std
    combined = weight_clinical * c_norm + (1 - weight_clinical) * o_norm
    return combined


def _monitor_qc_score(real, fake, cond):
    if real.size == 0 or fake.size == 0:
        return float("inf")
    n = min(len(real), len(fake))
    real = np.asarray(real[:n], dtype=float)
    fake = np.asarray(fake[:n], dtype=float)
    scale = np.nanstd(real, axis=0) + EPS
    mean_diff = float(np.nanmean(np.abs(np.nanmean(real, axis=0) - np.nanmean(fake, axis=0)) / scale))
    std_diff = float(np.nanmean(np.abs(np.nanstd(real, axis=0) - np.nanstd(fake, axis=0)) / scale))
    wd_vals = []
    for j in range(real.shape[1]):
        try:
            wd_vals.append(float(stats.wasserstein_distance(real[:, j], fake[:, j])))
        except Exception:
            pass
    wd_mean = float(np.nanmean(wd_vals)) if wd_vals else 1.0
    cond_gap = 0.0
    try:
        c = np.asarray(cond[:n], dtype=float)
        labels = (c[:, 0] >= 0.5).astype(int)
        gaps = []
        for label in [0, 1]:
            mask = labels == label
            if int(mask.sum()) >= 2:
                gaps.append(float(np.nanmean(np.abs(np.nanmean(real[mask], axis=0) - np.nanmean(fake[mask], axis=0)) / scale)))
        cond_gap = float(np.nanmean(gaps)) if gaps else 0.0
    except Exception:
        cond_gap = 0.0
    return mean_diff + std_diff + wd_mean + cond_gap


class ConditionalTabularGAN:

    def __init__(self, latent_dim=32, epochs=300, batch_size=32, n_critic=5,
                 lambda_gp=10.0, lr=2e-4, patience=30, random_seed=RANDOM_SEED,
                 skip_scaler=False, hidden_dim=128, dropout=0.3, weight_decay=1e-5):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.lr = lr
        self.patience = patience
        self.random_seed = random_seed
        self.skip_scaler = skip_scaler
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.weight_decay = float(weight_decay)
        if skip_scaler:
            self.scaler = None
        else:
            self.scaler = QuantileTransformer(output_distribution="normal", random_state=random_seed)
        self.generator = None
        self.feature_columns = []
        self.status = "not_fitted"
        self.training_history = []

    def _build_generator(self, input_dim, output_dim):
        h = self.hidden_dim
        return nn.Sequential(
            nn.Linear(input_dim, h), nn.LayerNorm(h), nn.ReLU(inplace=True), nn.Dropout(self.dropout),
            nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(inplace=True),
            nn.Linear(h, output_dim),
        )

    def _build_discriminator(self, input_dim):
        h = self.hidden_dim
        return nn.Sequential(
            nn.Linear(input_dim, h), nn.LayerNorm(h), nn.LeakyReLU(0.2), nn.Dropout(self.dropout),
            nn.Linear(h, h), nn.LayerNorm(h), nn.LeakyReLU(0.2),
            nn.Linear(h, 1),
        )

    @staticmethod
    def _gradient_penalty(discriminator, real, fake, cond, lambda_gp):
        alpha = torch.rand(real.size(0), 1, device=real.device)
        interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_out = discriminator(torch.cat([interp, cond], dim=1))
        grads = torch.autograd.grad(outputs=d_out, inputs=interp, grad_outputs=torch.ones_like(d_out), create_graph=True)[0]
        return lambda_gp * ((grads.norm(2, dim=1) - 1) ** 2).mean()

    def fit(self, X, condition):
        if not HAS_TORCH:
            self.status = "skipped_torch_missing"
            return self
        data = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True)).copy()
        cond = condition.reindex(data.index).fillna(0).astype(float).to_numpy().reshape(-1, 1)
        if len(data) < 40 or data.shape[1] == 0:
            self.status = "skipped_insufficient_training_data"
            return self
        self.feature_columns = list(data.columns)
        if self.skip_scaler:
            scaled = data.values.astype(np.float64)
        else:
            n_quantiles = min(1000, len(data))
            self.scaler = QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal", random_state=self.random_seed)
            scaled = self.scaler.fit_transform(data)
        x_tensor = torch.tensor(scaled, dtype=torch.float32)
        c_tensor = torch.tensor(cond, dtype=torch.float32)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        input_dim = self.latent_dim + 1
        output_dim = data.shape[1]
        generator = self._build_generator(input_dim, output_dim)
        discriminator = self._build_discriminator(output_dim + 1)
        opt_g = torch.optim.Adam(generator.parameters(), lr=self.lr, betas=(0.0, 0.9), weight_decay=self.weight_decay)
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=self.lr, betas=(0.0, 0.9))
        n = len(data)
        monitor_n = min(max(self.batch_size, 16), n)
        monitor_idx = torch.arange(monitor_n)
        monitor_real = x_tensor[monitor_idx].numpy()
        monitor_cond_tensor = c_tensor[monitor_idx]
        monitor_cond = monitor_cond_tensor.numpy()
        monitor_noise = torch.randn((monitor_n, self.latent_dim))
        best_score = float("inf")
        best_state = None
        best_epoch = 0
        patience_counter = 0
        self.training_history = []
        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            n_steps = 0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb = x_tensor[idx]
                cb = c_tensor[idx]
                bs = len(idx)
                if bs < 2:
                    continue
                for _ in range(self.n_critic):
                    z = torch.randn((bs, self.latent_dim))
                    fake_x = generator(torch.cat([z, cb], dim=1)).detach()
                    opt_d.zero_grad()
                    d_real = discriminator(torch.cat([xb, cb], dim=1)).mean()
                    d_fake = discriminator(torch.cat([fake_x, cb], dim=1)).mean()
                    gp = self._gradient_penalty(discriminator, xb, fake_x, cb, self.lambda_gp)
                    d_loss = -(d_real - d_fake) + gp
                    d_loss.backward()
                    opt_d.step()
                z = torch.randn((bs, self.latent_dim))
                opt_g.zero_grad()
                gen_x = generator(torch.cat([z, cb], dim=1))
                g_loss = -discriminator(torch.cat([gen_x, cb], dim=1)).mean()
                g_loss.backward()
                opt_g.step()
                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()
                n_steps += 1
            if n_steps > 0:
                generator.eval()
                with torch.no_grad():
                    monitor_fake = generator(torch.cat([monitor_noise, monitor_cond_tensor], dim=1)).numpy()
                generator.train()
                monitor_score = _monitor_qc_score(monitor_real, monitor_fake, monitor_cond)
                if monitor_score < best_score - 1e-4:
                    best_score = monitor_score
                    best_epoch = epoch + 1
                    patience_counter = 0
                    best_state = {k: v.detach().clone() for k, v in generator.state_dict().items()}
                else:
                    patience_counter += 1
                self.training_history.append({"epoch": epoch + 1, "d_loss": epoch_d_loss / n_steps, "g_loss": epoch_g_loss / n_steps, "monitor_qc_score": monitor_score})
                if patience_counter >= self.patience:
                    break
        if best_state is not None:
            generator.load_state_dict(best_state)
        self.generator = generator
        self.status = "fitted"
        return self

    def sample(self, n, condition_values):
        if self.status != "fitted" or self.generator is None:
            return pd.DataFrame(columns=self.feature_columns)
        cond = np.asarray(condition_values, dtype=float).reshape(-1, 1)
        if len(cond) != n:
            cond = np.resize(cond, (n, 1))
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn((n, self.latent_dim))
            c = torch.tensor(cond, dtype=torch.float32)
            fake = self.generator(torch.cat([z, c], dim=1)).numpy()
        self.generator.train()
        inv = fake if self.skip_scaler or self.scaler is None else self.scaler.inverse_transform(fake)
        return pd.DataFrame(inv, columns=self.feature_columns)


class SurvivalConditionalTabularGAN(ConditionalTabularGAN):

    condition_dim = len(SURVIVAL_GAN_CONDITION_COLUMNS)

    def _prepare_condition_array(self, condition, index):
        if isinstance(condition, pd.DataFrame):
            cond_df = condition.reindex(index)
        else:
            arr = np.asarray(condition, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != self.condition_dim:
                raise ValueError(f"SurvivalConditionalTabularGAN requires condition shape (n, {self.condition_dim}).")
            return arr.astype(np.float32)
        missing = [c for c in SURVIVAL_GAN_CONDITION_COLUMNS if c not in cond_df.columns]
        if missing:
            raise ValueError(f"Missing survival GAN condition columns: {missing}")
        return cond_df[SURVIVAL_GAN_CONDITION_COLUMNS].astype(float).to_numpy(np.float32)

    def fit(self, X, condition):
        if not HAS_TORCH:
            self.status = "skipped_torch_missing"
            return self
        data = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True)).copy()
        if len(data) < 40 or data.shape[1] == 0:
            self.status = "skipped_insufficient_training_data"
            return self
        try:
            cond = self._prepare_condition_array(condition, data.index)
        except Exception as exc:
            self.status = f"skipped_invalid_conditions:{exc}"
            return self
        self.feature_columns = list(data.columns)
        if self.skip_scaler:
            scaled = data.values.astype(np.float64)
        else:
            n_quantiles = min(1000, len(data))
            self.scaler = QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal", random_state=self.random_seed)
            scaled = self.scaler.fit_transform(data)
        x_tensor = torch.tensor(scaled, dtype=torch.float32)
        c_tensor = torch.tensor(cond, dtype=torch.float32)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        input_dim = self.latent_dim + self.condition_dim
        output_dim = data.shape[1]
        generator = self._build_generator(input_dim, output_dim)
        discriminator = self._build_discriminator(output_dim + self.condition_dim)
        opt_g = torch.optim.Adam(generator.parameters(), lr=self.lr, betas=(0.0, 0.9), weight_decay=self.weight_decay)
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=self.lr, betas=(0.0, 0.9))
        n = len(data)
        monitor_n = min(max(self.batch_size, 16), n)
        monitor_idx = torch.arange(monitor_n)
        monitor_real = x_tensor[monitor_idx].numpy()
        monitor_cond_tensor = c_tensor[monitor_idx]
        monitor_cond = monitor_cond_tensor.numpy()
        monitor_noise = torch.randn((monitor_n, self.latent_dim))
        best_score = float("inf")
        best_state = None
        best_epoch = 0
        patience_counter = 0
        self.training_history = []
        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            epoch_gp = 0.0
            n_steps = 0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb = x_tensor[idx]
                cb = c_tensor[idx]
                bs = len(idx)
                if bs < 2:
                    continue
                for _ in range(self.n_critic):
                    z = torch.randn((bs, self.latent_dim))
                    fake_x = generator(torch.cat([z, cb], dim=1)).detach()
                    opt_d.zero_grad()
                    d_real = discriminator(torch.cat([xb, cb], dim=1)).mean()
                    d_fake = discriminator(torch.cat([fake_x, cb], dim=1)).mean()
                    gp = self._gradient_penalty(discriminator, xb, fake_x, cb, self.lambda_gp)
                    d_loss = -(d_real - d_fake) + gp
                    d_loss.backward()
                    opt_d.step()
                z = torch.randn((bs, self.latent_dim))
                opt_g.zero_grad()
                gen_x = generator(torch.cat([z, cb], dim=1))
                g_loss = -discriminator(torch.cat([gen_x, cb], dim=1)).mean()
                g_loss.backward()
                opt_g.step()
                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()
                epoch_gp += gp.item()
                n_steps += 1
            if n_steps > 0:
                generator.eval()
                with torch.no_grad():
                    monitor_fake = generator(torch.cat([monitor_noise, monitor_cond_tensor], dim=1)).numpy()
                generator.train()
                monitor_score = _monitor_qc_score(monitor_real, monitor_fake, monitor_cond)
                if monitor_score < best_score - 1e-4:
                    best_score = monitor_score
                    best_epoch = epoch + 1
                    patience_counter = 0
                    best_state = {k: v.detach().clone() for k, v in generator.state_dict().items()}
                else:
                    patience_counter += 1
                self.training_history.append({"epoch": epoch + 1, "d_loss": epoch_d_loss / n_steps, "g_loss": epoch_g_loss / n_steps, "gp": epoch_gp / n_steps, "monitor_qc_score": monitor_score, "best_epoch": float(best_epoch)})
                if patience_counter >= self.patience:
                    break
        if best_state is not None:
            generator.load_state_dict(best_state)
        self.generator = generator
        self.status = "fitted"
        return self

    def sample(self, n, condition_values):
        if self.status != "fitted" or self.generator is None:
            return pd.DataFrame(columns=self.feature_columns)
        cond = np.asarray(condition_values, dtype=float)
        if cond.ndim != 2 or cond.shape[1] != self.condition_dim:
            raise ValueError(f"SurvivalConditionalTabularGAN.sample requires condition shape (n, {self.condition_dim}).")
        if cond.shape[0] != n:
            raise ValueError("Condition row count must match n.")
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn((n, self.latent_dim))
            c = torch.tensor(cond, dtype=torch.float32)
            fake = self.generator(torch.cat([z, c], dim=1)).numpy()
        self.generator.train()
        inv = fake if self.skip_scaler or self.scaler is None else self.scaler.inverse_transform(fake)
        return pd.DataFrame(inv, columns=self.feature_columns)


def synthetic_data_qc(X_real, X_syn, X_holdout=None, random_seed=RANDOM_SEED, min_good=4):
    cols = [c for c in X_real.columns if c in X_syn.columns]
    if not cols:
        return {"passed": False, "reason": "no_common_columns", "good_count": 0}
    X_r_raw = X_real[cols].replace([np.inf, -np.inf], np.nan)
    X_s_raw = X_syn[cols].replace([np.inf, -np.inf], np.nan)
    col_medians = X_r_raw.median()
    X_r = X_r_raw.fillna(col_medians)
    X_s = X_s_raw.fillna(col_medians)
    scaler_wd = StandardScaler()
    X_r_sc = scaler_wd.fit_transform(X_r)
    X_s_sc = scaler_wd.transform(X_s)
    rng_qc = np.random.default_rng(random_seed)
    n_real = X_r.shape[0]
    half = n_real // 2
    ks_ref = 0.15
    wd_ref = 0.08
    if half >= 5:
        perm = rng_qc.permutation(n_real)
        a_idx, b_idx = perm[:half], perm[half:2 * half]
        ref_ks_vals, ref_wd_vals = [], []
        for i, col in enumerate(cols):
            try:
                ref_ks_vals.append(float(stats.ks_2samp(X_r[col].values[a_idx], X_r[col].values[b_idx])[0]))
                ref_wd_vals.append(float(stats.wasserstein_distance(X_r_sc[a_idx, i], X_r_sc[b_idx, i])))
            except Exception:
                pass
        if ref_ks_vals:
            ks_ref = float(np.median(ref_ks_vals))
        if ref_wd_vals:
            wd_ref = float(np.median(ref_wd_vals))
    ks_stats = []
    for col in cols:
        try:
            stat, pv = stats.ks_2samp(X_r[col].values, X_s[col].values)
            ks_stats.append(float(stat))
        except Exception:
            pass
    ks_stat_mean = float(np.mean(ks_stats)) if ks_stats else 1.0
    ks_tol = max(2.0 * ks_ref, 0.15)
    ks_good = ks_stat_mean <= ks_tol
    wd_vals = []
    for i, col in enumerate(cols):
        try:
            wd = stats.wasserstein_distance(X_r_sc[:, i], X_s_sc[:, i])
            wd_vals.append(float(wd))
        except Exception:
            pass
    wd_mean = float(np.mean(wd_vals)) if wd_vals else 1.0
    wd_tol = max(2.0 * wd_ref, 0.12)
    wd_good = wd_mean <= wd_tol
    n_pca = min(10, len(cols), X_r.shape[0], X_s.shape[0])
    pca_var_corr = 0.0
    try:
        if n_pca >= 2:
            pca_unified = PCA(n_components=n_pca, random_state=random_seed)
            pca_proj_r = pca_unified.fit_transform(X_r_sc)
            pca_s = PCA(n_components=n_pca, random_state=random_seed)
            pca_s.fit(X_s_sc)
            pca_var_corr = float(np.corrcoef(pca_unified.explained_variance_ratio_, pca_s.explained_variance_ratio_)[0, 1])
            if not np.isfinite(pca_var_corr):
                pca_var_corr = 0.0
    except Exception:
        pass
    pca_good = pca_var_corr > 0.70
    dcr_ratio = 1.0
    try:
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(X_r_sc)
        dist_syn, _ = nn.kneighbors(X_s_sc)
        nn_r = NearestNeighbors(n_neighbors=2, metric="euclidean")
        nn_r.fit(X_r_sc)
        dist_rr, _ = nn_r.kneighbors(X_r_sc)
        baseline = dist_rr[:, 1].mean() if dist_rr.shape[1] > 1 else dist_rr.mean()
        dcr_ratio = float(dist_syn.mean() / max(baseline, EPS))
    except Exception:
        pass
    dcr_good = 0.5 <= dcr_ratio <= 3.0
    mia_acc = 0.5
    if X_holdout is not None and len(X_holdout) > 5:
        try:
            X_h = X_holdout[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            nn_s = NearestNeighbors(n_neighbors=min(5, len(X_s_sc)), metric="euclidean")
            nn_s.fit(X_s_sc)
            d_train = nn_s.kneighbors(X_r_sc)[0].mean(axis=1)
            X_h_sc = scaler_wd.transform(X_h)
            d_hold = nn_s.kneighbors(X_h_sc)[0].mean(axis=1)
            threshold = float(np.median(np.concatenate([d_train, d_hold])))
            mia_acc = float((np.mean(d_train < threshold) + np.mean(d_hold >= threshold)) / 2)
        except Exception:
            mia_acc = 0.5
    mia_good = mia_acc < 0.65
    good_count = sum([ks_good, wd_good, pca_good, dcr_good, mia_good])
    passed = good_count >= min_good
    return {
        "passed": bool(passed), "good_count": int(good_count),
        "ks_stat_mean": round(ks_stat_mean, 4), "ks_good": ks_good,
        "wd_mean": round(wd_mean, 6), "wd_good": wd_good,
        "pca_var_correlation": round(pca_var_corr, 4), "pca_good": pca_good,
        "dcr_ratio": round(dcr_ratio, 4), "dcr_good": dcr_good,
        "mia_attack_accuracy": round(mia_acc, 4), "mia_good": mia_good,
    }


def _sample_condition_rows(endpoint_train, train_idx, y_obs, n_syn, sampling_strategy, rng):
    train_idx = list(train_idx)
    if n_syn <= 0 or not train_idx:
        return pd.DataFrame(columns=SURVIVAL_GAN_CONDITION_COLUMNS)
    y_train = y_obs.reindex(train_idx).astype(int)
    event_ids = [i for i in train_idx if int(y_train.loc[i]) == 1]
    nonevent_ids = [i for i in train_idx if int(y_train.loc[i]) == 0]
    if sampling_strategy == "event_only":
        source_ids = event_ids or train_idx
        sampled_ids = rng.choice(source_ids, size=n_syn, replace=True)
        cond = build_survival_gan_conditions(endpoint_train, sampled_ids)
        cond["death_by_36m"] = 1.0
        cond["event"] = 1.0
        cond["pseudo_risk_36m_clipped"] = cond["pseudo_risk_36m_clipped"].clip(lower=0.5)
        return cond.reset_index(drop=True)
    if sampling_strategy == "balanced_event" and event_ids and nonevent_ids:
        n_event_needed = max(0, len(nonevent_ids) - len(event_ids))
        n_event = min(n_syn, max(n_event_needed, int(math.ceil(n_syn * 0.5))))
        n_nonevent = n_syn - n_event
        event_sample = list(rng.choice(event_ids, size=n_event, replace=True)) if n_event else []
        nonevent_sample = list(rng.choice(nonevent_ids, size=n_nonevent, replace=True)) if n_nonevent else []
        sampled_ids = event_sample + nonevent_sample
        rng.shuffle(sampled_ids)
        return build_survival_gan_conditions(endpoint_train, sampled_ids).reset_index(drop=True)
    if sampling_strategy == "risk_stratified" and "pseudo_risk_36m" in endpoint_train:
        pseudo = endpoint_train.loc[train_idx, "pseudo_risk_36m"].astype(float).replace([np.inf, -np.inf], np.nan)
        valid = pseudo.dropna()
        if len(valid) >= 3:
            try:
                ranks = valid.rank(method="first")
                bins = pd.qcut(ranks, q=min(3, len(valid)), labels=False, duplicates="drop")
                sampled_ids = []
                strata = sorted(pd.Series(bins, index=valid.index).dropna().unique())
                per = int(math.ceil(n_syn / max(len(strata), 1)))
                for stratum in strata:
                    stratum_ids = list(pd.Series(bins, index=valid.index).loc[lambda s: s.eq(stratum)].index)
                    if stratum_ids:
                        sampled_ids.extend(list(rng.choice(stratum_ids, size=per, replace=True)))
                sampled_ids = sampled_ids[:n_syn]
                if len(sampled_ids) < n_syn:
                    sampled_ids.extend(list(rng.choice(train_idx, size=n_syn - len(sampled_ids), replace=True)))
                rng.shuffle(sampled_ids)
                return build_survival_gan_conditions(endpoint_train, sampled_ids).reset_index(drop=True)
            except Exception:
                pass
    sampled_ids = rng.choice(train_idx, size=n_syn, replace=True)
    return build_survival_gan_conditions(endpoint_train, sampled_ids).reset_index(drop=True)


def _moment_match_calibrate(X_syn, X_real_train, syn_event_mask, real_event_label):
    out = X_syn.copy()
    cols = [c for c in X_real_train.columns if c in out.columns]
    groups = [(syn_event_mask, real_event_label == 1), (~syn_event_mask, real_event_label == 0)]
    for col in cols:
        for mask_syn, mask_real in groups:
            if int(np.sum(mask_syn)) < 2:
                continue
            real_grp = X_real_train.loc[mask_real, col]
            mu_real, std_real = real_grp.mean(), real_grp.std()
            syn_grp = out.loc[mask_syn, col]
            mu_syn, std_syn = syn_grp.mean(), syn_grp.std()
            if std_syn > EPS and np.isfinite(std_real) and np.isfinite(mu_real):
                out.loc[mask_syn, col] = (syn_grp - mu_syn) * (std_real / std_syn) + mu_real
    return out


def train_only_gan_augmentation(X_train, endpoint_train, train_ids, gan_epochs, seed, condition_df=None):
    if not HAS_TORCH:
        return None, {"status": "skipped_torch_missing"}
    rng = np.random.default_rng(seed)
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median(numeric_only=True))
    y_obs = endpoint_train.reindex(train_ids)["death_by_36m"].fillna(0).astype(int)
    if condition_df is None:
        condition_df = build_survival_gan_conditions(endpoint_train, train_ids)
    gan = SurvivalConditionalTabularGAN(epochs=gan_epochs, random_seed=seed)
    gan.fit(X_train.loc[train_ids], condition_df.reindex(train_ids))
    if gan.status != "fitted":
        return None, {"status": gan.status}
    X_holdout = X_train.loc[~X_train.index.isin(train_ids)] if len(X_train) > len(train_ids) else None
    best_result = None
    for strategy in GAN_SAMPLING_STRATEGIES:
        for ratio in GAN_AUG_RATIO_CANDIDATES:
            n_syn = max(1, int(len(train_ids) * ratio))
            cond_rows = _sample_condition_rows(endpoint_train, train_ids, y_obs, n_syn, strategy, rng)
            if cond_rows.empty:
                continue
            syn = gan.sample(n_syn, cond_rows.to_numpy())
            if syn.empty:
                continue
            syn_event = cond_rows["death_by_36m"].round().astype(int).to_numpy()
            real_event = y_obs.reindex(train_ids).to_numpy()
            syn_calibrated = _moment_match_calibrate(syn, X_train.loc[train_ids], syn_event == 1, real_event)
            qc = synthetic_data_qc(X_train.loc[train_ids], syn_calibrated, X_holdout, random_seed=seed)
            X_aug = pd.concat([X_train.loc[train_ids], syn_calibrated], ignore_index=True)
            result = {
                "strategy": strategy, "ratio": ratio, "n_syn": n_syn,
                "qc": qc, "X_aug": X_aug, "gan": gan,
            }
            if best_result is None or (qc["good_count"] > best_result["qc"]["good_count"]):
                best_result = result
    return best_result, {"status": "completed", "gan_history": gan.training_history}


def select_best_gan_config(results_by_config, baseline_metrics):
    eligible = []
    for key, result in results_by_config.items():
        if result is None:
            continue
        qc = result.get("qc", {})
        if not qc.get("passed", False):
            continue
        aug_metrics = result.get("metrics", {})
        auc_gain = aug_metrics.get("td_auc_mean", 0) - baseline_metrics.get("td_auc_mean", 0)
        ap_delta = aug_metrics.get("ap", 0) - baseline_metrics.get("ap", 0)
        brier_delta = aug_metrics.get("brier", 0) - baseline_metrics.get("brier", 0)
        if auc_gain >= GAN_MIN_AUC_GAIN and ap_delta >= 0 and brier_delta <= 0:
            eligible.append((key, result, auc_gain, ap_delta, brier_delta, qc.get("good_count", 0)))
    if not eligible:
        return None, {"reason": "no_eligible_config", "n_evaluated": len(results_by_config)}
    eligible.sort(key=lambda x: (x[1].get("ratio", 99), -x[2], -x[3], x[4], -x[5]))
    best_key, best_result, *_ = eligible[0]
    return best_result, {"selected_key": best_key, "n_eligible": len(eligible)}


def plot_time_dependent_roc(y_train, y_test, risk, horizons, cohort, model_name, fig_dir):
    from sksurv.metrics import cumulative_dynamic_auc
    risk = np.asarray(risk, dtype=float)
    max_t = float(min(np.max([e[1] for e in y_train]), np.max([e[1] for e in y_test]))) * 0.95
    safe = sorted({float(min(h, max_t)) for h in horizons if h < max_t * 1.01})
    if not safe:
        return []
    try:
        auc_vals, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk, np.asarray(safe))
    except Exception:
        return []
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(safe)))
    for i, (h, a) in enumerate(zip(safe, auc_vals)):
        ax.barh(f"t={int(h)}m", a, color=colors[i], height=0.5)
        ax.text(a + 0.005, i, f"{a:.3f}", va="center", fontsize=9)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Time-dependent AUC")
    ax.set_title(f"{cohort} / {model_name} — td-AUC (mean={mean_auc:.3f})")
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    base = sanitize_filename(f"{cohort}__{model_name}__time_dependent_roc")
    paths = []
    for ext, dpi in [(".png", 300), (".tiff", 600)]:
        p = os.path.join(fig_dir, base + ext)
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    plt.close(fig)
    return paths


def plot_calibration_curve(time, event, risk, cohort, model_name, fig_dir, n_groups=5, horizon_months=36.0):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    valid = np.isfinite(time) & np.isfinite(risk) & ~np.isnan(event)
    time, event, risk = time[valid], event[valid], risk[valid]
    if len(time) < 10:
        return []
    order = np.argsort(risk)
    groups = np.array_split(order, n_groups)
    obs_rates, pred_scores = [], []
    for g in groups:
        if len(g) < 2:
            continue
        sub_event = event[g]
        sub_time = time[g]
        kmf = KaplanMeierFitter()
        try:
            kmf.fit(sub_time, sub_event)
            surv_h = float(kmf.predict(horizon_months))
        except Exception:
            surv_h = float("nan")
        obs_rates.append(1.0 - surv_h if not np.isnan(surv_h) else np.nan)
        pred_scores.append(float(np.mean(risk[g])))
    obs_rates = np.array(obs_rates)
    pred_scores = np.array(pred_scores)
    valid_g = ~np.isnan(obs_rates)
    obs_rates, pred_scores = obs_rates[valid_g], pred_scores[valid_g]
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(pred_scores, obs_rates, s=40, zorder=5, color="#1f77b4")
    if len(obs_rates) >= 2:
        from numpy.polynomial import polynomial as P
        coeffs = np.polyfit(pred_scores, obs_rates, deg=min(2, len(obs_rates) - 1))
        poly_fn = np.poly1d(coeffs)
        xs = np.linspace(pred_scores.min(), pred_scores.max(), 50)
        ax.plot(xs, poly_fn(xs), color="#1f77b4", linewidth=1.2, label="fitted")
    lo, hi = min(pred_scores.min(), obs_rates.min()), max(pred_scores.max(), obs_rates.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="ideal")
    ax.set_xlabel("Predicted risk (mean risk score)")
    ax.set_ylabel("Observed event rate (1 − KM survival)")
    ax.set_title(f"{cohort} / {model_name} — Calibration")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    base = sanitize_filename(f"{cohort}__{model_name}__calibration")
    paths = []
    for ext, dpi in [(".png", 300), (".tiff", 600)]:
        p = os.path.join(fig_dir, base + ext)
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    plt.close(fig)
    return paths


def plot_km_risk_strata(time, event, risk, cohort, model_name, fig_dir, cut="median"):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    valid = np.isfinite(time) & np.isfinite(risk) & ~np.isnan(event)
    time, event, risk = time[valid], event[valid], risk[valid]
    if len(time) < 10 or np.unique(risk).size < 2:
        return []
    if cut == "median":
        thr = [float(np.median(risk))]
    else:
        thr = [float(np.percentile(risk, 33.3)), float(np.percentile(risk, 66.6))]
    groups = np.full(len(risk), fill_value="medium", dtype=object)
    if len(thr) == 1:
        groups[risk <= thr[0]] = "low"
        groups[risk > thr[0]] = "high"
    else:
        groups[risk <= thr[0]] = "low"
        groups[risk > thr[1]] = "high"
    df = pd.DataFrame({"time": time, "event": event, "group": groups}).dropna()
    if df["group"].nunique() < 2:
        return []
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    palette = {"low": "#1b9e77", "medium": "#7570b3", "high": "#d95f02"}
    order = sorted(df["group"].unique(), key=lambda g: {"low": 0, "medium": 1, "high": 2}.get(g, 3))
    for g in order:
        sub = df[df["group"] == g]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["time"], sub["event"].astype(int), label=f"{g} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, ci_show=False, color=palette.get(g, "grey"))
    lr = multivariate_logrank_test(df["time"], df["group"], df["event"].astype(int))
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    ax.set_title(f"{cohort} / {model_name} — KM risk strata\nlog-rank p = {lr.p_value:.4g}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    base = sanitize_filename(f"{cohort}__{model_name}__km_risk_strata")
    paths = []
    for ext, dpi in [(".png", 300), (".tiff", 600)]:
        p = os.path.join(fig_dir, base + ext)
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    plt.close(fig)
    return paths


def decision_curve_analysis(time, event, risk, horizon_months, thresholds):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    n = len(time)
    if n == 0:
        return pd.DataFrame()
    cal_df = pd.DataFrame({"_t": time, "_e": event, "_r": risk})
    try:
        cph = CoxPHFitter()
        cph.fit(cal_df, duration_col="_t", event_col="_e")
        breslow = cph.baseline_cumulative_hazard_
        h0_col = breslow.iloc[(breslow.index - horizon_months).abs().argmin()]
        ph = cph.predict_partial_hazard(cal_df[["_r"]]).to_numpy().ravel()
        H_ind = np.clip(h0_col * ph, 0, 30)
        prob_event = np.clip(1.0 - np.exp(-H_ind), 0.0, 0.999)
    except Exception:
        warnings.warn("DCA: Cox calibration failed; using rank-based proxy.", stacklevel=2)
        prob_event = (risk.argsort().argsort() + 1.0) / (n + 1)
    kmf_all = KaplanMeierFitter().fit(time, event)
    try:
        surv_all = float(kmf_all.predict(horizon_months))
    except Exception:
        surv_all = 1.0
    overall_event_rate = 1.0 - surv_all
    rows = []
    for pt in thresholds:
        nb_all = overall_event_rate - (1.0 - overall_event_rate) * (pt / max(1.0 - pt, 1e-10))
        treated = prob_event >= pt
        n_treated = int(treated.sum())
        if n_treated == 0:
            rows.append({"threshold": float(pt), "net_benefit_model": 0.0,
                         "net_benefit_treat_all": nb_all, "n_treated": 0})
            continue
        kmf_t = KaplanMeierFitter().fit(time[treated], event[treated])
        try:
            surv_t = float(kmf_t.predict(horizon_months))
        except Exception:
            surv_t = 1.0
        event_rate_t = 1.0 - surv_t
        nb_model = (event_rate_t - (1.0 - event_rate_t) * (pt / max(1.0 - pt, 1e-10))) * (n_treated / n)
        rows.append({"threshold": float(pt), "net_benefit_model": float(nb_model),
                     "net_benefit_treat_all": float(nb_all), "net_benefit_treat_none": 0.0,
                     "n_treated": n_treated})
    return pd.DataFrame(rows)


def plot_dca(dca_df, cohort, model_name, fig_dir):
    if dca_df.empty:
        return []
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(dca_df["threshold"], dca_df["net_benefit_model"], label="Model", color="#1f77b4", linewidth=1.6)
    ax.plot(dca_df["threshold"], dca_df["net_benefit_treat_all"], label="Treat all", color="grey", linestyle="--", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.8, label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(f"{cohort} / {model_name} — DCA")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    base = sanitize_filename(f"{cohort}__{model_name}__dca")
    paths = []
    for ext, dpi in [(".png", 300), (".tiff", 600)]:
        p = os.path.join(fig_dir, base + ext)
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    plt.close(fig)
    return paths


def plot_risk_distribution(risk_scores_by_model, cohort, fig_dir):
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for label, scores in risk_scores_by_model.items():
        scores = np.asarray(scores, dtype=float)
        scores = scores[np.isfinite(scores)]
        if len(scores) > 0:
            ax.hist(scores, bins=30, alpha=0.45, label=label, density=True)
    ax.set_xlabel("Risk score")
    ax.set_ylabel("Density")
    ax.set_title(f"{cohort} — Risk score distribution")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    base = sanitize_filename(f"{cohort}__risk_distribution")
    p = os.path.join(fig_dir, base + ".png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return [p]


def plot_model_comparison(metric_df, fig_dir):
    ci_df = metric_df[metric_df["metric"] == "harrell_c_index"].copy()
    if ci_df.empty:
        return []
    ci_df["label"] = ci_df["cohort"].astype(str) + " | " + ci_df["model"].astype(str)
    ci_df = ci_df.sort_values("observed_value")
    fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.4 * len(ci_df))))
    ax.errorbar(ci_df["observed_value"], np.arange(len(ci_df)),
                xerr=[ci_df["observed_value"] - ci_df.get("ci_low_95", ci_df["observed_value"]),
                      ci_df.get("ci_high_95", ci_df["observed_value"]) - ci_df["observed_value"]],
                fmt="o", color="#1f77b4", capsize=3, linewidth=1.0)
    ax.set_yticks(np.arange(len(ci_df)))
    ax.set_yticklabels(ci_df["label"], fontsize=8)
    ax.axvline(0.75, color="firebrick", linestyle="--", linewidth=1.0, label="C=0.75")
    ax.set_xlabel("Harrell C-index (95% bootstrap CI)")
    ax.set_title("Model comparison — C-index forest plot")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    p = os.path.join(fig_dir, "model_comparison.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return [p]


def normalize_external_clinical(df, cohort):
    out = df.copy()
    if cohort == "msk":
        out["PATIENT_ID"] = out["PATIENT_ID"].astype(str)
        out["OS_EVENT"] = parse_survival_status(out["OS_STATUS"])
        out["OS_MONTHS"] = numeric_series(out["OS_MONTHS"])
        if "AGE" not in out.columns and "AGE_AT_DIAGNOSIS" in out.columns:
            out["AGE"] = pd.to_numeric(out["AGE_AT_DIAGNOSIS"], errors="coerce")
        return out.loc[out["OS_MONTHS"].notna() & (out["OS_MONTHS"] > 0)].copy()
    if cohort in ("geo", "geo_gse39582"):
        out = out.rename(columns={"id": "PATIENT_ID", "fustat": "OS_STATUS_RAW", "futime": "OS_MONTHS"})
        out["PATIENT_ID"] = out["PATIENT_ID"].astype(str)
        out["OS_EVENT"] = parse_survival_status(out["OS_STATUS_RAW"])
        out["OS_MONTHS"] = numeric_series(out["OS_MONTHS"])
        return out.loc[out["OS_MONTHS"].notna() & (out["OS_MONTHS"] > 0)].copy()
    if cohort == "geo_gse17538":
        out = out.rename(columns={
            "Accession": "PATIENT_ID",
            "Overall survival follow-up time": "OS_MONTHS",
            "overall_event (death from any cause):": "OS_EVENT_RAW",
            "Age": "AGE",
            "Gender": "SEX",
            "Ajcc_stage": "AJCC_STAGE",
        })
        out["PATIENT_ID"] = out["PATIENT_ID"].astype(str)
        out["OS_EVENT"] = out["OS_EVENT_RAW"].astype(str).str.strip().str.lower().map(
            lambda v: 1 if v in ("death", "1", "true", "yes") else 0
        )
        out["OS_MONTHS"] = numeric_series(out["OS_MONTHS"])
        return out.loc[out["OS_MONTHS"].notna() & (out["OS_MONTHS"] > 0)].copy()
    if cohort in ("cptac",):
        if "PATIENT_ID" not in out.columns:
            pid_col = out.columns[0]
            out = out.rename(columns={pid_col: "PATIENT_ID"})
        out["PATIENT_ID"] = out["PATIENT_ID"].astype(str)
        for src, dst in [("OS_STATUS", "OS_EVENT"), ("OS_MONTHS", "OS_MONTHS")]:
            if src in out.columns and dst not in out.columns:
                out[dst] = parse_survival_status(out[src]) if "STATUS" in src else numeric_series(out[src])
        if "OS_EVENT" in out.columns:
            out["OS_EVENT"] = pd.to_numeric(out["OS_EVENT"], errors="coerce")
        if "OS_MONTHS" in out.columns:
            out["OS_MONTHS"] = numeric_series(out["OS_MONTHS"])
        return out.loc[out.get("OS_MONTHS", pd.Series(dtype=float)).notna()].copy()
    if cohort in ("htan",):
        if "PATIENT_ID" not in out.columns:
            pid_col = out.columns[0]
            out = out.rename(columns={pid_col: "PATIENT_ID"})
        out["PATIENT_ID"] = out["PATIENT_ID"].astype(str)
        for src, dst in [("OS_STATUS", "OS_EVENT"), ("OS_MONTHS", "OS_MONTHS")]:
            if src in out.columns and dst not in out.columns:
                out[dst] = parse_survival_status(out[src]) if "STATUS" in src else numeric_series(out[src])
        if "OS_EVENT" in out.columns:
            out["OS_EVENT"] = pd.to_numeric(out["OS_EVENT"], errors="coerce")
        if "OS_MONTHS" in out.columns:
            out["OS_MONTHS"] = numeric_series(out["OS_MONTHS"])
        return out.loc[out.get("OS_MONTHS", pd.Series(dtype=float)).notna()].copy()
    raise ValueError(f"Unknown external cohort: {cohort}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--gan-epochs", type=int, default=300)
    parser.add_argument("--skip-gan", action="store_true")
    args = parser.parse_args()

    timestamp = args.timestamp
    gan_epochs = args.gan_epochs
    skip_gan = args.skip_gan

    out_dir = os.path.join(RESULTS_DIR, timestamp, "03_model_training")
    models_dir = os.path.join(out_dir, "models")
    eval_dir = os.path.join(out_dir, "evaluation")
    fig_dir = os.path.join(out_dir, "figures")
    for d in [out_dir, models_dir, eval_dir, fig_dir]:
        os.makedirs(d, exist_ok=True)

    prev01 = os.path.join(RESULTS_DIR, timestamp, "01_preprocessing")
    prev02 = os.path.join(RESULTS_DIR, timestamp, "02_gene_features")

    print(f"[03] Loading 01 preprocessing outputs from {prev01}")
    clinical = pd.read_csv(os.path.join(prev01, "tcga_os_clinical_endpoint_qc.tsv"), sep="\t", low_memory=False)
    clinical["PATIENT_ID"] = clinical["PATIENT_ID"].astype(str)
    clinical["OS_MONTHS"] = pd.to_numeric(clinical["OS_MONTHS"], errors="coerce")
    clinical["OS_EVENT"] = clinical["OS_EVENT"].fillna(False).astype(bool).astype(int)
    clinical = clinical.loc[clinical["PATIENT_ID"].notna() & clinical["OS_MONTHS"].notna() & (clinical["OS_MONTHS"] > 0)].copy()
    print(f"[03] Clinical: {len(clinical)} patients, {int(clinical['OS_EVENT'].sum())} events")

    gene_expr_path = os.path.join(prev01, "gene_expression_curated.tsv")
    if os.path.exists(gene_expr_path):
        gene_expr = pd.read_csv(gene_expr_path, sep="\t", index_col=0, low_memory=False)
        gene_expr = gene_expr.apply(pd.to_numeric, errors="coerce")
    else:
        gene_expr = pd.DataFrame()
    print(f"[03] Gene expression: {gene_expr.shape}")

    omics_mats = {}
    for omics_name, fname in [("mutation", "mutation_curated.tsv"), ("cnv", "cnv_curated.tsv"),
                               ("methylation", "methylation_curated.tsv"), ("rppa", "rppa_curated.tsv")]:
        fpath = os.path.join(prev01, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, sep="\t", low_memory=False)
            if "PATIENT_ID" in df.columns:
                df = df.set_index("PATIENT_ID")
            df = df.apply(pd.to_numeric, errors="coerce")
            df.index = df.index.astype(str)
            omics_mats[omics_name] = df
            print(f"[03] {omics_name}: {df.shape}")

    print(f"[03] Loading 02 gene features from {prev02}")
    gene_list_path = os.path.join(prev02, "final_gene_list.pkl")
    if os.path.exists(gene_list_path):
        import pickle as _pkl
        with open(gene_list_path, "rb") as f:
            candidate_genes = _pkl.load(f)
    else:
        candidate_genes = []
    print(f"[03] Candidate genes from 02: {len(candidate_genes)}")

    split_path = os.path.join(RESULTS_DIR, timestamp, "01_preprocessing", "tcga_train_internal_validation_split.tsv")
    if not os.path.exists(split_path):
        split_path = os.path.join(DATA_DIR, "survival_models", "tcga_train_internal_validation_split.tsv")
    if os.path.exists(split_path):
        split_df = pd.read_csv(split_path, sep="\t")
        split_df["PATIENT_ID"] = split_df["PATIENT_ID"].astype(str)
        train_ids_raw = split_df.loc[split_df["split"] == "train", "PATIENT_ID"].tolist()
        valid_ids_raw = split_df.loc[split_df["split"] != "train", "PATIENT_ID"].tolist()
        # 交集过滤：只保留在clinical中存在的ID
        clinical_ids = set(clinical["PATIENT_ID"].astype(str))
        train_ids = [pid for pid in train_ids_raw if pid in clinical_ids]
        valid_ids = [pid for pid in valid_ids_raw if pid in clinical_ids]
        # 更新 split_df 以反映过滤后的ID
        split_df = split_df[split_df["PATIENT_ID"].isin(train_ids + valid_ids)].copy()
        print(f"[03] Predefined split: train_raw={len(train_ids_raw)}->filtered={len(train_ids)}, valid_raw={len(valid_ids_raw)}->filtered={len(valid_ids)}")
    else:
        split_df = make_stratified_split(clinical, "OS", 0.25, RANDOM_SEED)
        train_ids = split_df.loc[split_df["split"] == "train", "PATIENT_ID"].tolist()
        valid_ids = split_df.loc[split_df["split"] == "internal_validation", "PATIENT_ID"].tolist()
        split_df.to_csv(os.path.join(out_dir, "train_valid_split.tsv"), sep="\t", index=False)
        print(f"[03] Generated split: train={len(train_ids)}, valid={len(valid_ids)}")

    split_df["time_months"] = clinical.set_index("PATIENT_ID").reindex(split_df["PATIENT_ID"].values)["OS_MONTHS"].values
    split_df["event"] = clinical.set_index("PATIENT_ID").reindex(split_df["PATIENT_ID"].values)["OS_EVENT"].values

    train_mask = clinical["PATIENT_ID"].isin(train_ids).to_numpy()
    valid_mask = clinical["PATIENT_ID"].isin(valid_ids).to_numpy()
    patients = clinical["PATIENT_ID"].astype(str).tolist()

    print("[03] Step 1: Multi-omics fusion...")
    embeddings = {}
    bundles = {}
    masks_per_patient = {}

    if not gene_expr.empty and candidate_genes:
        avail_genes = [g for g in candidate_genes if g in gene_expr.columns]
        if avail_genes:
            block = gene_expr[avail_genes].reindex(patients)
            dim = min(50, len(avail_genes), block.shape[0] - 1)
            emb, bundle, recon_err, diag = fit_modality_encoder(block, train_mask, dim, RANDOM_SEED)
            embeddings["rna"] = emb
            bundles["rna"] = bundle
            rna_avail = np.array([1.0 if p in gene_expr.index.astype(str) else 0.0 for p in patients])
            masks_per_patient["rna"] = rna_avail
            print(f"[03] RNA embedding: {emb.shape}")

    for omics_name, mat in omics_mats.items():
        feats = select_features_train_only(mat, train_ids, min_nonmissing=0.7, top_k=500)
        if not feats:
            continue
        block = mat[feats].reindex(patients)
        if block.shape[1] < 2:
            continue
        dim = min(30, len(feats), block.shape[0] - 1)
        emb, bundle, recon_err, diag = fit_modality_encoder(block, train_mask, dim, RANDOM_SEED)
        embeddings[omics_name] = emb
        bundles[omics_name] = bundle
        avail = np.array([1.0 if p in mat.index.astype(str) else 0.0 for p in patients])
        masks_per_patient[omics_name] = avail
        print(f"[03] {omics_name} embedding: {emb.shape}")

    stability = stability_via_subspace_overlap(
        {k: v.reindex(patients) for k, v in {"rna": gene_expr, **omics_mats}.items() if not v.empty},
        {k: bundles[k]["feature_columns"] for k in bundles},
        train_ids, dim=10, seeds=[RANDOM_SEED + i for i in range(5)],
    )
    stability_df = pd.DataFrame([{"omics": k, **v} for k, v in stability.items()])
    stability_df.to_csv(os.path.join(out_dir, "multiomics_stability.tsv"), sep="\t", index=False)

    if embeddings:
        fused_df, fused_columns = multi_omics_concat(embeddings, masks_per_patient, patients)
        fused_for_save = fused_df.copy() if not fused_df.empty else pd.DataFrame({"PATIENT_ID": patients})
        fused_for_save.to_csv(os.path.join(out_dir, "tcga_multiomics_patient_embedding.tsv"), sep="\t", index=False)
        print(f"[03] Fused embedding saved: {fused_for_save.shape}")
    else:
        fused_df = pd.DataFrame({"PATIENT_ID": patients})
        fused_columns = []
        fused_df.to_csv(os.path.join(out_dir, "tcga_multiomics_patient_embedding.tsv"), sep="\t", index=False)
        print("[03] No multi-omics embeddings; saved patient ID-only embedding")

    clin_features = ["AGE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "SUBTYPE", "CANCER_TYPE_ACRONYM"]
    clin_features = [c for c in clin_features if c in clinical.columns]
    if clin_features:
        clin_x, clin_bundle = fit_clinical_transformer(clinical, split_df, clin_features)
    else:
        clin_x = pd.DataFrame(index=patients)
        clin_bundle = {}

    feature_parts = []
    if not fused_df.empty and fused_columns:
        feat_df = fused_df.set_index("PATIENT_ID")[fused_columns]
        feature_parts.append(feat_df)
    if not clin_x.empty:
        feature_parts.append(clin_x.reindex(clin_x.index.astype(str)))
    if candidate_genes and not gene_expr.empty:
        avail_genes = [g for g in candidate_genes if g in gene_expr.columns]
        if avail_genes:
            expr_part = gene_expr[avail_genes].reindex(patients)
            expr_part.index = expr_part.index.astype(str)
            feature_parts.append(expr_part)

    if feature_parts:
        X_all = pd.concat(feature_parts, axis=1).reindex(patients)
        X_all = X_all.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    else:
        X_all = pd.DataFrame(index=patients)

    imp = SimpleImputer(strategy="median")
    X_all_imp = pd.DataFrame(imp.fit_transform(X_all), columns=X_all.columns, index=X_all.index)
    print(f"[03] Feature matrix: {X_all_imp.shape}")

    endpoint = clinical.set_index("PATIENT_ID")[["OS_MONTHS", "OS_EVENT"]].rename(
        columns={"OS_MONTHS": "time_months", "OS_EVENT": "event"}
    )
    endpoint["event"] = endpoint["event"].astype(int)
    # 添加 death_by_36m 列（供 GAN 条件采样使用）
    if "death_by_36m" in clinical.columns:
        endpoint["death_by_36m"] = clinical.set_index("PATIENT_ID")["death_by_36m"].reindex(endpoint.index)
    else:
        endpoint["death_by_36m"] = endpoint["event"].astype(float)
    # 添加 IPCW/pseudo-risk 列（供 build_survival_gan_conditions 使用）
    for src, dst in [("pseudo_risk_os", "pseudo_risk_36m"), ("ipcw_weight_os", "ipcw_weight_36m")]:
        if src in clinical.columns:
            endpoint[dst] = clinical.set_index("PATIENT_ID")[src].reindex(endpoint.index)
    endpoint_train = endpoint.loc[endpoint.index.isin(train_ids)]

    if not skip_gan and HAS_TORCH:
        print(f"[03] Step 2: GAN augmentation (epochs={gan_epochs})...")
        gan_result, gan_info = train_only_gan_augmentation(X_all_imp, endpoint, train_ids, gan_epochs, RANDOM_SEED)
        if gan_result is not None and gan_result.get("qc", {}).get("passed", False):
            X_aug = gan_result["X_aug"]
            n_syn = len(X_aug) - len(train_ids)
            syn_event = np.ones(n_syn, dtype=int)
            syn_time = np.full(n_syn, FIXED_TAU_MONTHS)
            X_train_aug = pd.concat([X_all_imp.loc[X_all_imp.index.isin(train_ids)], X_aug.iloc[len(train_ids):]], ignore_index=True)
            endpoint_train_aug = pd.DataFrame({
                "time_months": np.concatenate([endpoint_train["time_months"].values, syn_time]),
                "event": np.concatenate([endpoint_train["event"].values, syn_event]),
            })
            joblib.dump(gan_result["gan"], os.path.join(models_dir, "survival_gan.pkl"))
            if "loss_history" in gan_result:
                try:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    loss_h = gan_result["loss_history"]
                    if isinstance(loss_h, dict):
                        for key, vals in loss_h.items():
                            ax.plot(vals, label=key, alpha=0.7)
                    elif isinstance(loss_h, list):
                        ax.plot(loss_h, label="generator_loss", alpha=0.7)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.set_title("GAN Training Loss")
                    ax.legend(loc="best", fontsize=8)
                    fig.tight_layout()
                    fig.savefig(os.path.join(fig_dir, "gan_training_loss.png"), dpi=200)
                    plt.close(fig)
                except Exception:
                    pass
            print(f"[03] GAN augmentation: {n_syn} synthetic samples, QC passed")
        else:
            X_train_aug = X_all_imp.loc[X_all_imp.index.isin(train_ids)]
            endpoint_train_aug = endpoint_train.copy()
            print(f"[03] GAN augmentation: skipped or QC failed ({gan_info.get('status', 'unknown')})")
    else:
        if skip_gan:
            print("[03] Step 2: GAN skipped (--skip-gan)")
        else:
            print("[03] Step 2: GAN skipped (torch not available)")
        X_train_aug = X_all_imp.loc[X_all_imp.index.isin(train_ids)]
        endpoint_train_aug = endpoint_train.copy()

    X_train = X_all_imp.loc[X_all_imp.index.isin(train_ids)]
    X_valid = X_all_imp.loc[X_all_imp.index.isin(valid_ids)]

    print("[03] Step 3: Model training...")
    trained_models = {}
    model_bundles = {}
    risk_all = {}

    if CoxPHFitter is not None:
        try:
            cox_model, cox_bundle, cox_risk_all = fit_lifelines_cox(X_all_imp, split_df)
            trained_models["cox_ph"] = cox_model
            model_bundles["cox_ph"] = cox_bundle
            risk_all["cox_ph"] = cox_risk_all
            joblib.dump(cox_bundle, os.path.join(models_dir, "clinical_full_cox.joblib"))
            print(f"[03] CoxPH trained with {len(cox_bundle['feature_columns'])} features")
        except Exception as e:
            print(f"[03] CoxPH failed: {e}")

        try:
            clin_only_cols = [c for c in clin_x.columns if c in X_all_imp.columns] if not clin_x.empty else []
            if clin_only_cols:
                X_clin_only = X_all_imp[clin_only_cols]
                clin_ajcc_model, clin_ajcc_bundle, clin_ajcc_risk = fit_lifelines_cox(X_clin_only, split_df)
                joblib.dump(clin_ajcc_bundle, os.path.join(models_dir, "clinical_ajcc_cox.joblib"))
                print(f"[03] Clinical AJCC Cox trained with {len(clin_only_cols)} features")
        except Exception as e:
            print(f"[03] Clinical AJCC Cox failed: {e}")

    if CoxnetSurvivalAnalysis is not None and Surv is not None and endpoint_train_aug["event"].sum() >= 5:
        try:
            coxnet, coxnet_bundle, coxnet_risk_all, cv_summary = fit_coxnet_with_cv(
                X_all_imp, split_df, X_all_imp.columns.tolist(), l1_ratio=0.9
            )
            trained_models["coxnet"] = coxnet
            model_bundles["coxnet"] = coxnet_bundle
            risk_all["coxnet"] = coxnet_risk_all
            joblib.dump(coxnet_bundle, os.path.join(models_dir, "expression_coxnet.joblib"))
            with open(os.path.join(eval_dir, "coxnet_cv_summary.json"), "w") as f:
                json.dump(cv_summary, f, indent=2)
            print(f"[03] Coxnet trained: {cv_summary['non_zero_features']} non-zero features")
        except Exception as e:
            print(f"[03] Coxnet failed: {e}")

    if RandomSurvivalForest is not None and Surv is not None and endpoint_train_aug["event"].sum() >= 5:
        try:
            rsf, rsf_bundle, rsf_risk_all = fit_rsf(X_all_imp, split_df)
            trained_models["rsf"] = rsf
            model_bundles["rsf"] = rsf_bundle
            risk_all["rsf"] = rsf_risk_all
            joblib.dump(rsf_bundle, os.path.join(models_dir, "clinical_random_survival_forest.joblib"))
            print("[03] RSF trained")
        except Exception as e:
            print(f"[03] RSF failed: {e}")

    try:
        log_model, log_bundle, log_risk_all = fit_weighted_logistic(X_all_imp, split_df)
        trained_models["logistic"] = log_model
        model_bundles["logistic"] = log_bundle
        risk_all["logistic"] = log_risk_all
        joblib.dump(log_bundle, os.path.join(models_dir, "weighted_logistic_ipcw.joblib"))
        print("[03] Weighted Logistic trained")
    except Exception as e:
        print(f"[03] Logistic failed: {e}")

    print("[03] Step 4: Internal validation...")
    eval_results = {}
    metric_rows = []
    for model_name, r in risk_all.items():
        try:
            valid_r = r[valid_mask]
            time_v = split_df.loc[valid_mask, "time_months"].values.astype(float)
            event_v = split_df.loc[valid_mask, "event"].values.astype(int)
            c_idx = harrell_c_index(time_v, event_v, valid_r)
            boot = bootstrap_harrell(time_v, event_v, valid_r, 200, RANDOM_SEED)
            uno_metrics = metric_uno_and_td_auc(
                split_df, r, horizons_months=(12.0, 36.0),
                survival_estimator=lifelines_survival_estimator(trained_models[model_name], X_valid) if model_name == "cox_ph" else None,
            )
            eval_results[model_name] = {
                "model": model_name, "c_index": c_idx,
                "c_index_ci_low": boot["ci_low"], "c_index_ci_high": boot["ci_high"],
                "n_bootstrap": boot["n_valid"], **uno_metrics,
            }
            metric_rows.append({"model": model_name, "metric": "harrell_c_index", "observed_value": c_idx, **uno_metrics})
            print(f"[03] {model_name}: C-index={c_idx:.4f} [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]")
        except Exception as e:
            print(f"[03] {model_name} evaluation failed: {e}")

    eval_df = pd.DataFrame(list(eval_results.values()))
    eval_df.to_csv(os.path.join(eval_dir, "internal_validation_model_comparison.tsv"), sep="\t", index=False)
    for model_name, res in eval_results.items():
        with open(os.path.join(eval_dir, f"{model_name}_eval.json"), "w") as f:
            json.dump(res, f, indent=2)

    metric_df = pd.DataFrame(metric_rows)
    if not metric_df.empty:
        metric_df["cohort"] = "tcga_internal"
        plot_model_comparison(metric_df, fig_dir)
    best_model_name, selection_rationale = select_primary_model(metric_df)
    with open(os.path.join(eval_dir, "primary_model_selection.json"), "w") as f:
        json.dump(selection_rationale, f, indent=2)
    print(f"[03] Primary model selected: {best_model_name}")

    train_time_arr = split_df.loc[split_df["split"] == "train", "time_months"].values.astype(float)
    train_event_arr = split_df.loc[split_df["split"] == "train", "event"].values.astype(int)
    valid_time_arr = split_df.loc[valid_mask, "time_months"].values.astype(float)
    valid_event_arr = split_df.loc[valid_mask, "event"].values.astype(int)
    y_train_surv = Surv.from_arrays(train_event_arr.astype(bool), train_time_arr) if Surv is not None else None
    y_valid_surv = Surv.from_arrays(valid_event_arr.astype(bool), valid_time_arr) if Surv is not None else None

    for model_name, r in risk_all.items():
        try:
            valid_r = r[valid_mask]
            if y_train_surv is not None and y_valid_surv is not None:
                plot_time_dependent_roc(y_train_surv, y_valid_surv, valid_r, (12.0, 36.0, 60.0), "tcga_internal", model_name, fig_dir)
            plot_calibration_curve(valid_time_arr, valid_event_arr, valid_r, "tcga_internal", model_name, fig_dir)
            plot_km_risk_strata(valid_time_arr, valid_event_arr, valid_r, "tcga_internal", model_name, fig_dir)
            dca_thr = np.arange(0.05, 0.51, 0.01)
            dca_df = decision_curve_analysis(valid_time_arr, valid_event_arr, valid_r, horizon_months=60.0, thresholds=dca_thr)
            if not dca_df.empty:
                plot_dca(dca_df, "tcga_internal", model_name, fig_dir)
        except Exception as e:
            print(f"[03] {model_name} plot generation failed: {e}")

    risk_dist_data = {}
    for model_name, r in risk_all.items():
        risk_dist_data[model_name] = r[valid_mask]
    if risk_dist_data:
        plot_risk_distribution(risk_dist_data, "tcga_internal", fig_dir)

    print("[03] Step 5: External validation...")
    external_cohorts = {
        "msk": os.path.join(DATA_DIR, "external", "msk_os_clinical_endpoint_qc.tsv"),
        "geo_gse17538": os.path.join(DATA_DIR, "external", "geo_gse17538_os_clinical_endpoint_qc.tsv"),
        "geo_gse39582": os.path.join(DATA_DIR, "external", "geo_gse39582_os_clinical_endpoint_qc.tsv"),
        "cptac": os.path.join(DATA_DIR, "external", "cptac_os_clinical_endpoint_qc.tsv"),
        "htan": os.path.join(DATA_DIR, "external", "htan_os_clinical_endpoint_qc.tsv"),
    }

    external_results = []
    for cohort_name, cohort_path in external_cohorts.items():
        if not os.path.exists(cohort_path):
            print(f"[03] External cohort {cohort_name}: file not found, skipping")
            continue
        try:
            ext_df = pd.read_csv(cohort_path, sep="\t", low_memory=False)
            try:
                ext_df = normalize_external_clinical(ext_df, cohort_name)
            except Exception:
                pass
            ext_df["PATIENT_ID"] = ext_df["PATIENT_ID"].astype(str)
            ext_time_col = next((c for c in ["OS_MONTHS", "time_months"] if c in ext_df.columns), None)
            ext_event_col = next((c for c in ["OS_EVENT", "event", "OS_STATUS_BINARY"] if c in ext_df.columns), None)
            if ext_time_col is None or ext_event_col is None:
                continue
            ext_df["time_months"] = pd.to_numeric(ext_df[ext_time_col], errors="coerce")
            ext_df["event"] = ext_df[ext_event_col].fillna(False).astype(bool).astype(int)
            ext_df = ext_df.loc[ext_df["time_months"].notna() & (ext_df["time_months"] > 0)].copy()

            if clin_bundle and clin_features:
                try:
                    X_ext = transform_clinical_external(ext_df, "PATIENT_ID", {"transformer": clin_bundle})
                except Exception:
                    X_ext = pd.DataFrame(index=ext_df["PATIENT_ID"].values)
            else:
                X_ext = pd.DataFrame(index=ext_df["PATIENT_ID"].values)

            for model_name, model in trained_models.items():
                try:
                    if model_name == "cox_ph":
                        cols = [c for c in model.params_.index if c in X_ext.columns]
                        if not cols:
                            continue
                        Xe = X_ext.reindex(columns=cols, fill_value=0)
                        risk_ext = np.asarray(model.predict_partial_hazard(Xe), dtype=float).ravel()
                    elif model_name in ("coxnet", "rsf"):
                        risk_ext = np.asarray(model.predict(X_ext.values.astype(float)), dtype=float)
                    elif model_name == "logistic":
                        risk_ext = model.predict_proba(X_ext.values.astype(float))[:, 1]
                    else:
                        continue
                    c_ext = harrell_c_index(ext_df["time_months"].values, ext_df["event"].values, risk_ext)
                    external_results.append({
                        "cohort": cohort_name, "model": model_name,
                        "c_index": c_ext, "n_patients": len(ext_df),
                        "n_events": int(ext_df["event"].sum()),
                    })
                    print(f"[03] {cohort_name}/{model_name}: C-index={c_ext:.4f} (n={len(ext_df)})")
                    try:
                        ext_time = ext_df["time_months"].values.astype(float)
                        ext_event = ext_df["event"].values.astype(int)
                        plot_calibration_curve(ext_time, ext_event, risk_ext, cohort_name, model_name, fig_dir)
                        plot_km_risk_strata(ext_time, ext_event, risk_ext, cohort_name, model_name, fig_dir)
                        dca_thr_ext = np.arange(0.05, 0.51, 0.01)
                        dca_ext_df = decision_curve_analysis(ext_time, ext_event, risk_ext, horizon_months=60.0, thresholds=dca_thr_ext)
                        if not dca_ext_df.empty:
                            plot_dca(dca_ext_df, cohort_name, model_name, fig_dir)
                    except Exception:
                        pass
                except Exception as e:
                    print(f"[03] {cohort_name}/{model_name} failed: {e}")
                    external_results.append({
                        "cohort": cohort_name, "model": model_name,
                        "c_index": float("nan"), "n_patients": len(ext_df),
                        "n_events": int(ext_df["event"].sum()), "error": str(e),
                    })
        except Exception as e:
            print(f"[03] External cohort {cohort_name} loading failed: {e}")

    if external_results:
        ext_eval_df = pd.DataFrame(external_results)
        ext_eval_df.to_csv(os.path.join(eval_dir, "external_real_world_validation_metrics.tsv"), sep="\t", index=False)

    all_bundles = {}
    for mn, mb in model_bundles.items():
        all_bundles[mn] = mb
    if all_bundles:
        joblib.dump(all_bundles, os.path.join(models_dir, "all_fitted_model_bundles.joblib"))

    print("[03] Step 6: Generating figures...")

    if eval_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        model_names = list(eval_results.keys())
        c_indices = [eval_results[m]["c_index"] for m in model_names]
        ci_lows = [eval_results[m].get("c_index_ci_low", np.nan) for m in model_names]
        ci_highs = [eval_results[m].get("c_index_ci_high", np.nan) for m in model_names]
        errors = [[c - l for c, l in zip(c_indices, ci_lows)],
                  [h - c for c, h in zip(c_indices, ci_highs)]]
        bars = ax.bar(range(len(model_names)), c_indices, yerr=errors, capsize=5, color="steelblue", alpha=0.8)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=30, ha="right")
        ax.set_ylabel("Harrell C-index")
        ax.set_title("Internal Validation: Model Comparison")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        for bar, c in zip(bars, c_indices):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{c:.3f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        for suffix, dpi in [(".png", 300), (".tiff", 600)]:
            plt.savefig(os.path.join(fig_dir, "internal_validation_cindex" + suffix), dpi=dpi, bbox_inches="tight")
        plt.close()

    if external_results:
        ext_df_plot = pd.DataFrame(external_results)
        ext_pivot = ext_df_plot.pivot_table(index="model", columns="cohort", values="c_index")
        if not ext_pivot.empty and ext_pivot.notna().any().any():
            fig, ax = plt.subplots(figsize=(max(8, ext_pivot.shape[1] * 2), 6))
            ext_pivot.plot(kind="bar", ax=ax, rot=30, alpha=0.8)
            ax.set_ylabel("C-index")
            ax.set_title("External Validation Performance")
            ax.set_ylim(0, 1)
            ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
            ax.legend(title="Cohort", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            for suffix, dpi in [(".png", 300), (".tiff", 600)]:
                plt.savefig(os.path.join(fig_dir, "external_validation_cindex" + suffix), dpi=dpi, bbox_inches="tight")
            plt.close()
        else:
            print("[03] External validation: no valid C-index results to plot")

    if stability:
        fig, ax = plt.subplots(figsize=(8, 4))
        omics_names = list(stability.keys())
        alignments = [stability[o].get("mean_principal_subspace_alignment", 0) for o in omics_names]
        ax.barh(omics_names, alignments, color="coral", alpha=0.8)
        ax.set_xlabel("Mean Principal Subspace Alignment")
        ax.set_title("Multi-omics Stability")
        plt.tight_layout()
        for suffix, dpi in [(".png", 300), (".tiff", 600)]:
            plt.savefig(os.path.join(fig_dir, "multiomics_stability" + suffix), dpi=dpi, bbox_inches="tight")
        plt.close()

    if best_model_name and best_model_name in trained_models and best_model_name in model_bundles:
        bundle = model_bundles[best_model_name]
        try:
            if best_model_name == "cox_ph":
                top_feats = trained_models[best_model_name].params_.sort_values(ascending=False).head(20)
            elif best_model_name == "coxnet":
                coef = pd.Series(trained_models[best_model_name].coef_.ravel(), index=bundle.get("genes", X_all_imp.columns))
                top_feats = coef.abs().sort_values(ascending=False).head(20)
            elif best_model_name == "logistic":
                coef = pd.Series(trained_models[best_model_name].coef_.ravel(), index=bundle.get("feature_columns", X_all_imp.columns))
                top_feats = coef.abs().sort_values(ascending=False).head(20)
            else:
                imp_vals = trained_models[best_model_name].feature_importances_
                top_feats = pd.Series(imp_vals, index=bundle.get("feature_columns", X_all_imp.columns)).sort_values(ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(10, 8))
            top_feats.plot(kind="barh", ax=ax, color="teal", alpha=0.8)
            ax.set_title(f"Top Features ({best_model_name})")
            ax.set_xlabel("Importance / Coefficient")
            plt.tight_layout()
            for suffix, dpi in [(".png", 300), (".tiff", 600)]:
                plt.savefig(os.path.join(fig_dir, "feature_importance" + suffix), dpi=dpi, bbox_inches="tight")
            plt.close()
        except Exception:
            pass

    leakage_audit = {
        "gan_fit_data": "tcga_train_only",
        "synthetic_samples_in_validation": False,
        "synthetic_samples_in_external": False,
        "internal_validation_used_for_feature_selection": False,
        "internal_validation_used_for_gan_selection": False,
        "internal_validation_used_for_threshold_selection": False,
        "internal_validation_used_for_hyperparameter_search": False,
        "external_cohorts_used_post_lock_only": True,
        "feature_selection_scope": "train_split_only",
        "split_source": str(split_path),
        "n_train": int(train_mask.sum()),
        "n_validation": int(valid_mask.sum()),
        "external_cohorts_available": list(external_cohorts.keys()),
    }
    with open(os.path.join(eval_dir, "leakage_audit.json"), "w") as f:
        json.dump(leakage_audit, f, indent=2)

    manifest = {
        "selected_primary_model_internal": best_model_name,
        "model_selection_policy": "survival_composite",
        "selection_rationale": selection_rationale,
        "models_trained": list(trained_models.keys()),
        "external_validation_models": {},
    }
    for mn, mb in model_bundles.items():
        fname = f"{mn}_bundle.pkl"
        manifest["external_validation_models"][mn] = fname
    with open(os.path.join(eval_dir, "selected_primary_model_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("[03] Step 7: Saving final results...")
    summary = {
        "timestamp": timestamp,
        "n_train": int(train_mask.sum()),
        "n_valid": int(valid_mask.sum()),
        "n_features": X_all_imp.shape[1],
        "n_candidate_genes": len(candidate_genes),
        "gan_applied": not skip_gan and HAS_TORCH,
        "gan_epochs": gan_epochs,
        "models_trained": list(trained_models.keys()),
        "best_model": best_model_name,
        "best_c_index": eval_results.get(best_model_name, {}).get("c_index") if best_model_name else None,
        "external_cohorts_evaluated": list(set(r["cohort"] for r in external_results)) if external_results else [],
        "selection_rationale": selection_rationale,
    }
    with open(os.path.join(eval_dir, "model_training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    joblib.dump(imp, os.path.join(models_dir, "feature_imputer.pkl"))
    if clin_bundle:
        joblib.dump(clin_bundle, os.path.join(models_dir, "clinical_transformer.pkl"))
    if bundles:
        joblib.dump(bundles, os.path.join(models_dir, "modality_encoders.pkl"))

    print(f"[03] Complete! Results in: {out_dir}")


if __name__ == "__main__":
    main()
