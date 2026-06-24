


from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import math
import os
import re
import sys
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test, proportional_hazard_test
except Exception:
    CoxPHFitter = None
    KaplanMeierFitter = None
    logrank_test = None
    proportional_hazard_test = None

try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import (
        brier_score as sksurv_brier_score,
        concordance_index_ipcw,
        cumulative_dynamic_auc,
        integrated_brier_score,
    )
    from sksurv.util import Surv
except Exception:
    RandomSurvivalForest = None
    CoxnetSurvivalAnalysis = None
    sksurv_brier_score = None
    concordance_index_ipcw = None
    cumulative_dynamic_auc = None
    integrated_brier_score = None
    Surv = None

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


TAU_MONTHS = 12.0
RANDOM_SEED = 20260604
EPS = 1e-8


@dataclasses.dataclass
class PipelineConfig:
    project_root: Path
    raw_data_root: Path
    plan_path: Path
    output_root: Path
    tau_months: float = TAU_MONTHS
    random_seed: int = RANDOM_SEED
    test_size: float = 0.25
    max_expression_features: int = 300
    max_causal_features: int = 50
    causal_prescreen_features: int = 120
    gan_epochs: int = 300
    gan_batch_size: int = 32
    gan_latent_dim: int = 64
    gan_aug_ratio: float = 0.5
    gan_aug_ratio_candidates: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    gan_sampling_strategies: Tuple[str, ...] = ("balanced_event", "risk_stratified", "event_only", "overall")
    gan_n_critic: int = 5
    gan_lambda_gp: float = 10.0
    gan_lr: float = 2e-4
    gan_patience: int = 30
    gan_qc_min_good: int = 4
    gan_max_aug_ratio: float = 0.8
    gan_use_feature_space: bool = False
    gan_latent_k: int = 30
    min_external_n: int = 30
    run_gan: bool = True
    run_snp_eqtl: bool = True
    strict_dependencies: bool = False


class AuditLog:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def add(self, step: str, status: str, message: str, **extra: Any) -> None:
        rec = {"step": step, "status": status, "message": message}
        rec.update(extra)
        self.records.append(rec)
        print(f"[{status}] {step}: {message}", flush=True)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


class QuietAuditLog(AuditLog):
    def add(self, step: str, status: str, message: str, **extra: Any) -> None:
        rec = {"step": step, "status": status, "message": message}
        rec.update(extra)
        self.records.append(rec)


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_patient_id(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    if text.upper().startswith("TCGA-") and len(text) >= 12:
        return text[:12].upper()
    return re.sub(r"[\s_]+", "-", text).upper()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def read_tsv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False, **kwargs)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def detect_event_time_columns(df: pd.DataFrame) -> Tuple[str, str]:
    time_candidates = [
        "OS_TIME_MONTHS",
        "OS_MONTHS",
        "time_months",
        "TIME_MONTHS",
        "OS_time",
        "time",
    ]
    event_candidates = [
        "OS_EVENT",
        "event",
        "EVENT",
        "OS_event",
        "OS_STATUS_BINARY",
    ]
    time_col = next((c for c in time_candidates if c in df.columns), None)
    event_col = next((c for c in event_candidates if c in df.columns), None)
    if time_col is None or event_col is None:
        raise ValueError(f"Cannot detect OS time/event columns from: {list(df.columns)[:30]}")
    return time_col, event_col


def coerce_event(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    lower = series.astype(str).str.lower()
    mapped = lower.map(
        {
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
            "1:deceased": 1,
            "0:living": 0,
            "deceased": 1,
            "living": 0,
            "dead": 1,
            "alive": 0,
            "event": 1,
            "censored": 0,
        }
    )
    numeric = pd.to_numeric(series, errors="coerce")
    out = mapped.where(mapped.notna(), numeric)
    return out.fillna(0).astype(int)


def add_fixed_12m_endpoint(
    clinical: pd.DataFrame,
    tau: float = TAU_MONTHS,
    patient_col: str = "PATIENT_ID",
) -> pd.DataFrame:
    df = clinical.copy()
    time_col, event_col = detect_event_time_columns(df)
    df[patient_col] = df[patient_col].map(normalize_patient_id)
    df["time_months"] = safe_numeric(df[time_col])
    df["event"] = coerce_event(df[event_col])
    df = df[df[patient_col].ne("") & df["time_months"].notna()].copy()
    df = df[df["time_months"] >= 0].copy()

    event_by_tau = (df["event"].eq(1) & (df["time_months"] <= tau)).astype(float)
    observed_status = (df["event"].eq(1) & (df["time_months"] <= tau)) | (df["time_months"] > tau)
    df["death_by_12m_observed"] = observed_status.astype(bool)
    df["death_by_12m"] = np.where(observed_status, event_by_tau, np.nan)
    df["early_censored_before_12m"] = (~observed_status).astype(bool)
    df["survived_observed_12m"] = (df["time_months"] > tau).astype(bool)
    return df


def kaplan_meier_survival_at(times: np.ndarray, event_observed: np.ndarray, query: np.ndarray) -> np.ndarray:
    order = np.argsort(times)
    times_sorted = np.asarray(times, dtype=float)[order]
    events_sorted = np.asarray(event_observed, dtype=int)[order]
    unique_event_times = np.unique(times_sorted[events_sorted == 1])
    surv = 1.0
    surv_times: List[float] = []
    surv_values: List[float] = []
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


def compute_ipcw_weights(endpoint: pd.DataFrame, tau: float) -> pd.DataFrame:
    df = endpoint.copy()

    censor_event = (df["event"].eq(0)).astype(int).to_numpy()
    times = df["time_months"].to_numpy(dtype=float)
    label_observed = df["death_by_12m_observed"].to_numpy(dtype=bool)
    eval_time = np.minimum(times, tau)
    g_at_eval = kaplan_meier_survival_at(times, censor_event, eval_time)
    weights = np.zeros(len(df), dtype=float)
    weights[label_observed] = 1.0 / np.clip(g_at_eval[label_observed], EPS, None)
    df["ipcw_weight_12m"] = weights
    df["ipcw_label_available"] = label_observed
    return df


def km_cumulative_incidence_by_tau(times: np.ndarray, events: np.ndarray, tau: float) -> float:
    surv = kaplan_meier_survival_at(times, events, np.asarray([tau], dtype=float))[0]
    return float(1.0 - surv)


def compute_pseudo_observations(endpoint: pd.DataFrame, tau: float, max_n_exact: int = 1200) -> pd.DataFrame:
    df = endpoint.copy()
    times = df["time_months"].to_numpy(dtype=float)
    events = df["event"].to_numpy(dtype=int)
    n = len(df)
    f_all = km_cumulative_incidence_by_tau(times, events, tau)
    pseudo = np.full(n, np.nan, dtype=float)
    if n == 0:
        df["pseudo_risk_12m"] = pseudo
        return df
    if n > max_n_exact:

        pseudo[:] = f_all
    else:
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            f_minus = km_cumulative_incidence_by_tau(times[mask], events[mask], tau)
            pseudo[i] = n * f_all - (n - 1) * f_minus
    df["pseudo_risk_12m"] = pseudo
    df["pseudo_risk_12m_clipped"] = np.clip(pseudo, 0.0, 1.0)
    return df


class DenseFrameTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: Any, y: Any = None) -> "DenseFrameTransformer":
        self.fitted_ = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        if hasattr(X, "toarray"):
            return X.toarray()
        return np.asarray(X)


def select_clinical_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_candidates = ["AGE", "WEIGHT", "DAYS_LAST_FOLLOWUP"]
    categorical_candidates = [
        "SEX",
        "AJCC_PATHOLOGIC_TUMOR_STAGE",
        "PATH_T_STAGE",
        "PATH_N_STAGE",
        "PATH_M_STAGE",
        "SUBTYPE",
        "CANCER_TYPE_ACRONYM",
        "RACE",
        "GENETIC_ANCESTRY_LABEL",
    ]
    numeric = [c for c in numeric_candidates if c in df.columns]
    categorical = [c for c in categorical_candidates if c in df.columns]
    return numeric, categorical


def build_clinical_preprocessor(train_df: pd.DataFrame) -> Tuple[Pipeline, List[str]]:
    numeric, categorical = select_clinical_columns(train_df)
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("No supported clinical predictor columns were found.")
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    pipe = Pipeline([("preprocess", preprocessor), ("dense", DenseFrameTransformer())])
    pipe.fit(train_df)

    names: List[str] = []
    if numeric:
        names.extend(numeric)
    if categorical:
        ohe = pipe.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
        names.extend(list(ohe.get_feature_names_out(categorical)))
    return pipe, names


def raw_columns_required_by_column_transformer(pipe: Pipeline) -> List[str]:
    required: List[str] = []
    preprocessor = pipe.named_steps["preprocess"]
    for _, _, cols in preprocessor.transformers_:
        if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
            required.extend([str(c) for c in cols])
    return list(dict.fromkeys(required))


def numeric_columns_required_by_column_transformer(pipe: Pipeline) -> List[str]:
    preprocessor = pipe.named_steps["preprocess"]
    for name, _, cols in preprocessor.transformers_:
        if name == "num" and isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
            return [str(c) for c in cols]
    return []


def transform_clinical_frame(pipe: Pipeline, df: pd.DataFrame, output_names: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    work = df.copy()
    required = raw_columns_required_by_column_transformer(pipe)
    missing = [c for c in required if c not in work.columns]
    for col in missing:
        work[col] = np.nan
    for col in numeric_columns_required_by_column_transformer(pipe):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    transformed = pd.DataFrame(pipe.transform(work), index=df.index, columns=list(output_names))
    return transformed, missing


def matrix_patients_to_rows(path: Path, id_col_candidates: Sequence[str] = ("Hugo_Symbol", "gene", "feature")) -> pd.DataFrame:
    df = read_tsv(path)
    id_col = next((c for c in id_col_candidates if c in df.columns), df.columns[0])
    df = df.drop_duplicates(subset=[id_col]).set_index(id_col)
    mat = df.apply(pd.to_numeric, errors="coerce").T
    mat.index = [normalize_patient_id(x) for x in mat.index]
    mat = mat.groupby(mat.index).mean(numeric_only=True)
    mat = mat.apply(pd.to_numeric, errors="coerce")
    mat.index.name = "PATIENT_ID"
    return mat


def patient_feature_matrix(path: Path, prefix: str = "") -> pd.DataFrame:
    df = read_tsv(path)
    if "PATIENT_ID" not in df.columns:
        raise ValueError(f"{path.name} does not contain PATIENT_ID.")
    df["PATIENT_ID"] = df["PATIENT_ID"].map(normalize_patient_id)
    df = df[df["PATIENT_ID"].ne("")].drop_duplicates(subset=["PATIENT_ID"]).set_index("PATIENT_ID")
    mat = df.apply(pd.to_numeric, errors="coerce")
    mat = mat.loc[:, mat.notna().any(axis=0)]
    if prefix:
        mat = mat.rename(columns={c: f"{prefix}{c}" for c in mat.columns})
    mat.index.name = "PATIENT_ID"
    return mat


def select_top_variance_features(matrix: pd.DataFrame, train_ids: Sequence[str], max_features: int) -> List[str]:
    train_ids = [pid for pid in train_ids if pid in matrix.index]
    if not train_ids:
        return []
    variances = matrix.loc[train_ids].var(axis=0, skipna=True).sort_values(ascending=False)
    variances = variances[np.isfinite(variances) & (variances > 0)]
    return list(variances.head(max_features).index)


def select_train_prevalent_binary_features(
    matrix: pd.DataFrame,
    train_ids: Sequence[str],
    max_features: int,
    min_prevalence: float = 0.02,
    max_prevalence: float = 0.80,
) -> List[str]:
    train_ids = [pid for pid in train_ids if pid in matrix.index]
    if not train_ids:
        return []
    binary = (matrix.loc[train_ids].fillna(0) > 0).astype(float)
    prevalence = binary.mean(axis=0).sort_values(ascending=False)
    prevalence = prevalence[(prevalence >= min_prevalence) & (prevalence <= max_prevalence)]
    return list(prevalence.head(max_features).index)


def fit_transform_feature_matrix(
    matrix: pd.DataFrame,
    train_ids: Sequence[str],
    all_ids: Sequence[str],
    feature_names: Sequence[str],
) -> Tuple[pd.DataFrame, Pipeline]:
    subset = matrix.reindex(all_ids)[list(feature_names)].copy()
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    pipe.fit(subset.loc[list(train_ids)])
    transformed = pd.DataFrame(pipe.transform(subset), index=all_ids, columns=list(feature_names))
    return transformed, pipe


def load_gmt(path: Path) -> Dict[str, List[str]]:
    gene_sets: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                gene_sets[parts[0]] = [g for g in parts[2:] if g]
    return gene_sets


def find_msigdb_gmt(raw_root: Path) -> Optional[Path]:
    msigdb = raw_root / "MSigDB"
    if not msigdb.exists():
        return None
    candidates = sorted(msigdb.rglob("*.gmt"), key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0] if candidates else None


def compute_pathway_activity(
    expression_rows: pd.DataFrame,
    raw_root: Path,
    max_pathways: int = 100,
) -> Tuple[pd.DataFrame, Optional[Path]]:
    gmt = find_msigdb_gmt(raw_root)
    if gmt is None:
        return pd.DataFrame(index=expression_rows.index), None
    gene_sets = load_gmt(gmt)
    scores: Dict[str, pd.Series] = {}
    genes = set(expression_rows.columns)
    for name, members in gene_sets.items():
        overlap = [g for g in members if g in genes]
        if 5 <= len(overlap) <= 500:
            scores[name] = expression_rows[overlap].mean(axis=1)
        if len(scores) >= max_pathways:
            break
    if not scores:
        return pd.DataFrame(index=expression_rows.index), gmt
    return pd.DataFrame(scores, index=expression_rows.index), gmt


def make_surv_array(endpoint: pd.DataFrame) -> Any:
    if Surv is None:
        return None
    return Surv.from_arrays(endpoint["event"].astype(bool).to_numpy(), endpoint["time_months"].to_numpy(float))


def censor_safe_binary_training(endpoint: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    mask = endpoint["ipcw_label_available"].astype(bool)
    y = endpoint.loc[mask, "death_by_12m"].astype(int)
    w = endpoint.loc[mask, "ipcw_weight_12m"].astype(float)
    return y, w


def fit_weighted_logistic(
    X_train: pd.DataFrame,
    endpoint_train: pd.DataFrame,
    random_seed: int,
) -> Optional[LogisticRegressionCV]:
    y, w = censor_safe_binary_training(endpoint_train)
    if y.nunique() < 2 or len(y) < 20:
        return None
    model = LogisticRegressionCV(
        Cs=10,
        cv=min(5, max(2, int(y.value_counts().min()))),
        penalty="elasticnet",
        solver="saga",
        l1_ratios=[0.05, 0.5, 0.95],
        scoring="roc_auc",
        max_iter=5000,
        random_state=random_seed,
        n_jobs=-1,
    )
    model.fit(X_train.loc[y.index], y, sample_weight=w)
    return model


def train_only_cv_metrics_for_augmented_logistic(
    X_real: pd.DataFrame,
    endpoint_real: pd.DataFrame,
    cfg: PipelineConfig,
    ratio: float,
    sampling_strategy: str,
    random_seed: int,
) -> Dict[str, float]:

    y_real, _ = censor_safe_binary_training(endpoint_real)
    if y_real.nunique() < 2 or len(y_real) < 40:
        return {"auc_12m_observed": float("nan"), "average_precision_12m": float("nan"), "brier_12m_ipcw": float("nan")}
    min_class = int(y_real.value_counts().min())
    n_splits = min(5, max(2, min_class))
    if n_splits < 2:
        return {"auc_12m_observed": float("nan"), "average_precision_12m": float("nan"), "brier_12m_ipcw": float("nan")}
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    auc_scores: List[float] = []
    ap_scores: List[float] = []
    brier_scores: List[float] = []
    X_real_obs = X_real.loc[y_real.index]
    for fold_id, (train_pos, val_pos) in enumerate(splitter.split(X_real_obs, y_real), start=1):
        fold_train_ids = list(y_real.index[train_pos])
        fold_val_ids = list(y_real.index[val_pos])
        fold_cfg = dataclasses.replace(
            cfg,
            gan_aug_ratio=ratio,
            gan_aug_ratio_candidates=(ratio,),
            gan_sampling_strategies=(sampling_strategy,),
            random_seed=random_seed + fold_id,
        )
        quiet_audit = QuietAuditLog()
        X_fold_aug, ep_fold_aug, fold_report = train_only_gan_augmentation(
            X_real.loc[fold_train_ids],
            endpoint_real.loc[fold_train_ids],
            fold_cfg,
            quiet_audit,
            sampling_strategy=sampling_strategy,
        )
        if fold_report.get("status") != "fitted":
            X_fold_aug = X_real.loc[fold_train_ids]
            ep_fold_aug = endpoint_real.loc[fold_train_ids]
        model = fit_weighted_logistic(X_fold_aug, ep_fold_aug, random_seed + fold_id)
        if model is None:
            continue
        risk_val = risk_from_model(model, X_real.loc[fold_val_ids])
        y_val = endpoint_real.loc[fold_val_ids, "death_by_12m"].astype(int).to_numpy()
        w_val = endpoint_real.loc[fold_val_ids, "ipcw_weight_12m"].replace(0, np.nan).fillna(1.0).to_numpy(float)
        if len(np.unique(y_val)) < 2:
            continue
        try:
            auc_scores.append(float(roc_auc_score(y_val, risk_val, sample_weight=w_val)))
        except Exception:
            pass
        try:
            ap_scores.append(float(average_precision_score(y_val, risk_val, sample_weight=w_val)))
        except Exception:
            pass
        try:
            brier_scores.append(float(np.average((y_val - risk_val) ** 2, weights=w_val)))
        except Exception:
            pass
    return {
        "auc_12m_observed": float(np.nanmean(auc_scores)) if auc_scores else float("nan"),
        "average_precision_12m": float(np.nanmean(ap_scores)) if ap_scores else float("nan"),
        "brier_12m_ipcw": float(np.nanmean(brier_scores)) if brier_scores else float("nan"),
    }


def train_only_cv_auc_for_augmented_logistic(
    X_real: pd.DataFrame,
    endpoint_real: pd.DataFrame,
    cfg: PipelineConfig,
    ratio: float,
    sampling_strategy: str,
    random_seed: int,
) -> float:
    metrics = train_only_cv_metrics_for_augmented_logistic(
        X_real, endpoint_real, cfg, ratio, sampling_strategy, random_seed
    )
    return float(metrics.get("auc_12m_observed", float("nan")))


def train_only_cv_metrics_for_real_logistic(
    X_real: pd.DataFrame,
    endpoint_real: pd.DataFrame,
    random_seed: int,
) -> Dict[str, float]:

    y_real, _ = censor_safe_binary_training(endpoint_real)
    if y_real.nunique() < 2 or len(y_real) < 40:
        return {"auc_12m_observed": float("nan"), "average_precision_12m": float("nan"), "brier_12m_ipcw": float("nan")}
    min_class = int(y_real.value_counts().min())
    n_splits = min(5, max(2, min_class))
    if n_splits < 2:
        return {"auc_12m_observed": float("nan"), "average_precision_12m": float("nan"), "brier_12m_ipcw": float("nan")}
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    auc_scores: List[float] = []
    ap_scores: List[float] = []
    brier_scores: List[float] = []
    X_real_obs = X_real.loc[y_real.index]
    for fold_id, (train_pos, val_pos) in enumerate(splitter.split(X_real_obs, y_real), start=1):
        fold_train_ids = list(y_real.index[train_pos])
        fold_val_ids = list(y_real.index[val_pos])
        model = fit_weighted_logistic(
            X_real.loc[fold_train_ids],
            endpoint_real.loc[fold_train_ids],
            random_seed + fold_id,
        )
        if model is None:
            continue
        risk_val = risk_from_model(model, X_real.loc[fold_val_ids])
        y_val = endpoint_real.loc[fold_val_ids, "death_by_12m"].astype(int).to_numpy()
        w_val = endpoint_real.loc[fold_val_ids, "ipcw_weight_12m"].replace(0, np.nan).fillna(1.0).to_numpy(float)
        if len(np.unique(y_val)) < 2:
            continue
        try:
            auc_scores.append(float(roc_auc_score(y_val, risk_val, sample_weight=w_val)))
        except Exception:
            pass
        try:
            ap_scores.append(float(average_precision_score(y_val, risk_val, sample_weight=w_val)))
        except Exception:
            pass
        try:
            brier_scores.append(float(np.average((y_val - risk_val) ** 2, weights=w_val)))
        except Exception:
            pass
    return {
        "auc_12m_observed": float(np.nanmean(auc_scores)) if auc_scores else float("nan"),
        "average_precision_12m": float(np.nanmean(ap_scores)) if ap_scores else float("nan"),
        "brier_12m_ipcw": float(np.nanmean(brier_scores)) if brier_scores else float("nan"),
    }


def risk_from_model(model: Any, X: pd.DataFrame) -> np.ndarray:
    if model is None:
        return np.full(len(X), np.nan)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return np.asarray(pred, dtype=float)


def fit_lifelines_cox(
    X_train: pd.DataFrame,
    endpoint_train: pd.DataFrame,
    penalizer: float = 0.1,
) -> Optional[Any]:
    if CoxPHFitter is None:
        return None
    data = X_train.copy()
    data["time_months"] = endpoint_train["time_months"].to_numpy(float)
    data["event"] = endpoint_train["event"].to_numpy(int)
    usable = data.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    if usable["event"].sum() < 5:
        return None
    try:
        model = CoxPHFitter(penalizer=penalizer)
        model.fit(usable, duration_col="time_months", event_col="event", show_progress=False)
        return model
    except Exception:
        return None


def predict_lifelines_risk(model: Any, X: pd.DataFrame, tau: float) -> np.ndarray:
    if model is None:
        return np.full(len(X), np.nan)
    cols = [c for c in model.params_.index if c in X.columns]
    Xp = X.reindex(columns=cols, fill_value=0)
    try:
        surv = model.predict_survival_function(Xp, times=[tau]).T.iloc[:, 0]
        return 1.0 - surv.to_numpy(float)
    except Exception:
        partial = model.predict_partial_hazard(Xp)
        return np.asarray(partial, dtype=float).ravel()


def fit_sksurv_coxnet(X_train: pd.DataFrame, endpoint_train: pd.DataFrame) -> Optional[Any]:
    if CoxnetSurvivalAnalysis is None or Surv is None:
        return None
    if endpoint_train["event"].sum() < 5:
        return None
    y = make_surv_array(endpoint_train)
    try:
        model = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, n_alphas=60, max_iter=100000)
        model.fit(X_train.to_numpy(float), y)
        return model
    except Exception:
        return None


def fit_sksurv_rsf(
    X_train: pd.DataFrame,
    endpoint_train: pd.DataFrame,
    random_seed: int,
) -> Optional[Any]:
    if RandomSurvivalForest is None or Surv is None:
        return None
    if endpoint_train["event"].sum() < 5:
        return None
    y = make_surv_array(endpoint_train)
    try:
        model = RandomSurvivalForest(
            n_estimators=300,
            min_samples_split=10,
            min_samples_leaf=8,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_seed,
        )
        model.fit(X_train.to_numpy(float), y)
        return model
    except Exception:
        return None


def predict_sksurv_risk(model: Any, X: pd.DataFrame, tau: float) -> np.ndarray:
    if model is None:
        return np.full(len(X), np.nan)
    try:
        surv_fns = model.predict_survival_function(X.to_numpy(float))
        out = np.asarray([1.0 - float(fn(tau)) for fn in surv_fns], dtype=float)
        return out
    except Exception:
        try:
            return np.asarray(model.predict(X.to_numpy(float)), dtype=float)
        except Exception:
            return np.full(len(X), np.nan)


def uno_cindex_fallback(endpoint: pd.DataFrame, risk: np.ndarray) -> float:
    valid = np.isfinite(risk)
    t = endpoint.loc[valid, "time_months"].to_numpy(float)
    e = endpoint.loc[valid, "event"].to_numpy(int)
    r = risk[valid]
    permissible = 0
    concordant = 0.0
    for i in range(len(t)):
        for j in range(len(t)):
            if t[i] < t[j] and e[i] == 1:
                permissible += 1
                if r[i] > r[j]:
                    concordant += 1
                elif r[i] == r[j]:
                    concordant += 0.5
    if permissible == 0:
        return float("nan")
    return concordant / permissible


def evaluate_12m_predictions(
    train_endpoint: pd.DataFrame,
    eval_endpoint: pd.DataFrame,
    risk: np.ndarray,
    tau: float,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "n": int(len(eval_endpoint)),
        "events": int(eval_endpoint["event"].sum()),
        "early_censored_before_12m": int(eval_endpoint["early_censored_before_12m"].sum()),
    }
    valid_risk = np.isfinite(risk)
    observed = eval_endpoint["death_by_12m_observed"].to_numpy(bool) & valid_risk
    if observed.sum() >= 5 and pd.Series(eval_endpoint.loc[observed, "death_by_12m"]).nunique() == 2:
        y = eval_endpoint.loc[observed, "death_by_12m"].astype(int).to_numpy()
        scores = risk[observed]
        weights = eval_endpoint.loc[observed, "ipcw_weight_12m"].replace(0, np.nan).fillna(1.0).to_numpy(float)
        try:
            result["auc_12m_observed"] = float(roc_auc_score(y, scores, sample_weight=weights))
        except Exception:
            result["auc_12m_observed"] = float("nan")
        try:
            result["brier_12m_ipcw"] = float(np.average((y - scores) ** 2, weights=weights))
        except Exception:
            result["brier_12m_ipcw"] = float("nan")
        try:
            result["average_precision_12m"] = float(average_precision_score(y, scores, sample_weight=weights))
        except Exception:
            result["average_precision_12m"] = float("nan")
        if threshold is not None:
            pred = (scores >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            result.update({"threshold": float(threshold), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)})
    else:
        result.update(
            {
                "auc_12m_observed": float("nan"),
                "brier_12m_ipcw": float("nan"),
                "average_precision_12m": float("nan"),
            }
        )

    result["harrell_cindex"] = float(uno_cindex_fallback(eval_endpoint, risk))
    if (
        concordance_index_ipcw is not None
        and Surv is not None
        and eval_endpoint["event"].sum() >= 3
        and train_endpoint["event"].sum() >= 3
    ):
        try:
            y_train = make_surv_array(train_endpoint)
            y_eval = make_surv_array(eval_endpoint)
            c_ipcw = concordance_index_ipcw(y_train, y_eval, risk, tau=tau)[0]
            result["uno_cindex_ipcw"] = float(c_ipcw)
        except Exception:
            result["uno_cindex_ipcw"] = float("nan")
        try:
            auc_t, mean_auc = cumulative_dynamic_auc(y_train, y_eval, risk, times=np.asarray([tau]))
            result["time_dependent_auc_12m"] = float(auc_t[0])
            result["mean_auc_12m"] = float(mean_auc)
        except Exception:
            result["time_dependent_auc_12m"] = float("nan")
            result["mean_auc_12m"] = float("nan")
    else:
        result["uno_cindex_ipcw"] = float("nan")
        result["time_dependent_auc_12m"] = float("nan")
        result["mean_auc_12m"] = float("nan")
    return result


def choose_training_threshold(endpoint_train: pd.DataFrame, risk_train: np.ndarray) -> Optional[float]:
    valid = endpoint_train["death_by_12m_observed"].to_numpy(bool) & np.isfinite(risk_train)
    if valid.sum() < 10:
        return None
    y = endpoint_train.loc[valid, "death_by_12m"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return None
    scores = risk_train[valid]
    fpr, tpr, thresholds = roc_curve(y, scores)
    idx = int(np.nanargmax(tpr - fpr))
    return float(thresholds[idx])


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def save_current_figure(path_base: Path) -> List[str]:
    paths: List[str] = []
    for suffix, dpi in [(".png", 300), ((".tiff"), 600)]:
        path = path_base.with_suffix(suffix)
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(str(path))
    plt.close()
    return paths


def observed_binary_for_plot(endpoint: pd.DataFrame, risk: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = endpoint["death_by_12m_observed"].to_numpy(bool) & np.isfinite(risk)
    if valid.sum() == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=float), np.asarray([], dtype=float)
    y = endpoint.loc[valid, "death_by_12m"].astype(int).to_numpy()
    scores = np.asarray(risk[valid], dtype=float)
    weights = endpoint.loc[valid, "ipcw_weight_12m"].replace(0, np.nan).fillna(1.0).to_numpy(float)
    return y, scores, weights


def plot_time_dependent_roc(
    endpoint: pd.DataFrame,
    risk: np.ndarray,
    tau: float,
    title: str,
    path_base: Path,
) -> Optional[List[str]]:
    y, scores, weights = observed_binary_for_plot(endpoint, risk)
    if len(y) < 8 or len(np.unique(y)) < 2:
        return None
    try:
        fpr, tpr, _ = roc_curve(y, scores, sample_weight=weights)
        auc_value = roc_auc_score(y, scores, sample_weight=weights)
    except Exception:
        return None
    plt.figure(figsize=(5.2, 4.8))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2.2, label=f"AUC at {tau:.0f} mo = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.2, linestyle="--", label="Chance")
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right", frameon=False)
    return save_current_figure(path_base)


def plot_calibration_curve(
    endpoint: pd.DataFrame,
    risk: np.ndarray,
    tau: float,
    title: str,
    path_base: Path,
    n_bins: int = 5,
) -> Optional[List[str]]:
    y, scores, weights = observed_binary_for_plot(endpoint, risk)
    if len(y) < 20 or len(np.unique(y)) < 2:
        return None
    df = pd.DataFrame({"y": y, "risk": scores, "w": weights}).replace([np.inf, -np.inf], np.nan).dropna()
    if df["risk"].nunique() < 3:
        return None
    try:
        df["bin"] = pd.qcut(df["risk"], q=min(n_bins, df["risk"].nunique()), duplicates="drop")
    except Exception:
        return None
    rows = []
    for _, group in df.groupby("bin", observed=False):
        if len(group) < 3:
            continue
        w = group["w"].to_numpy(float)
        rows.append(
            {
                "predicted": float(np.average(group["risk"], weights=w)),
                "observed": float(np.average(group["y"], weights=w)),
                "n": int(len(group)),
            }
        )
    cal = pd.DataFrame(rows)
    if len(cal) < 2:
        return None
    plt.figure(figsize=(5.2, 4.8))
    sizes = np.clip(cal["n"].to_numpy(float) * 12, 35, 180)
    plt.scatter(cal["predicted"], cal["observed"], s=sizes, color="#2ca02c", alpha=0.85, edgecolor="white", linewidth=0.8)
    plt.plot(cal["predicted"], cal["observed"], color="#2ca02c", lw=1.8)
    plt.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.2, linestyle="--")
    plt.xlabel(f"Predicted {tau:.0f}-month death risk")
    plt.ylabel(f"Observed {tau:.0f}-month death risk")
    plt.title(title)
    plt.xlim(0, min(1.0, max(0.05, float(df["risk"].max()) * 1.15)))
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    return save_current_figure(path_base)


def plot_km_risk_strata(
    endpoint: pd.DataFrame,
    risk: np.ndarray,
    threshold: Optional[float],
    tau: float,
    title: str,
    path_base: Path,
) -> Optional[List[str]]:
    if KaplanMeierFitter is None:
        return None
    valid = np.isfinite(risk) & endpoint["time_months"].notna().to_numpy()
    if valid.sum() < 20:
        return None
    ep = endpoint.loc[valid].copy()
    scores = np.asarray(risk[valid], dtype=float)
    if threshold is None or not np.isfinite(threshold):
        threshold = float(np.nanmedian(scores))
    strata = np.where(scores >= threshold, "High risk", "Low risk")
    if len(np.unique(strata)) < 2 or min(pd.Series(strata).value_counts()) < 5:
        return None
    plt.figure(figsize=(5.6, 4.8))
    colors = {"Low risk": "#1f77b4", "High risk": "#d62728"}
    kmf = KaplanMeierFitter()
    for label in ["Low risk", "High risk"]:
        mask = strata == label
        kmf.fit(ep.loc[mask, "time_months"], event_observed=ep.loc[mask, "event"], label=f"{label} (n={int(mask.sum())})")
        kmf.plot_survival_function(ci_show=True, color=colors[label], lw=2.0)
    p_text = ""
    if logrank_test is not None:
        try:
            low = strata == "Low risk"
            high = strata == "High risk"
            lr = logrank_test(
                ep.loc[low, "time_months"],
                ep.loc[high, "time_months"],
                event_observed_A=ep.loc[low, "event"],
                event_observed_B=ep.loc[high, "event"],
            )
            p_text = f"Log-rank p = {lr.p_value:.3g}"
        except Exception:
            p_text = ""
    plt.axvline(tau, color="#7f7f7f", lw=1.0, linestyle=":", label=f"{tau:.0f} months")
    plt.xlabel("Time from diagnosis (months)")
    plt.ylabel("Overall survival probability")
    plt.title(title)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    if p_text:
        plt.text(0.04, 0.08, p_text, transform=plt.gca().transAxes)
    plt.legend(frameon=False)
    return save_current_figure(path_base)


def generate_prediction_figures(
    endpoint: pd.DataFrame,
    risk: np.ndarray,
    threshold: Optional[float],
    cohort_label: str,
    model_name: str,
    tau: float,
    figures_dir: Path,
) -> List[Dict[str, Any]]:
    ensure_dir(figures_dir)
    stem = f"{sanitize_filename(cohort_label)}__{sanitize_filename(model_name)}"
    figure_specs = [
        ("time_dependent_roc", plot_time_dependent_roc, f"{cohort_label}: time-dependent ROC"),
        ("calibration", plot_calibration_curve, f"{cohort_label}: calibration"),
        ("km_risk_strata", plot_km_risk_strata, f"{cohort_label}: K-M risk strata"),
    ]
    rows: List[Dict[str, Any]] = []
    for figure_type, func, title in figure_specs:
        path_base = figures_dir / f"{stem}__{figure_type}"
        if figure_type == "km_risk_strata":
            paths = func(endpoint, risk, threshold, tau, title, path_base)
        else:
            paths = func(endpoint, risk, tau, title, path_base)
        rows.append(
            {
                "cohort": cohort_label,
                "model_name": model_name,
                "figure_type": figure_type,
                "status": "created" if paths else "skipped_insufficient_data",
                "files": ";".join(paths or []),
            }
        )
    return rows


def bootstrap_ci(values: Sequence[float], alpha: float = 0.05) -> Tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))


def causal_aipw_binary_exposure(
    X: pd.DataFrame,
    exposure: pd.Series,
    outcome: pd.Series,
    random_seed: int,
) -> Dict[str, float]:
    data = pd.concat([X, exposure.rename("A"), outcome.rename("Y")], axis=1).dropna()
    if len(data) < 30 or data["A"].nunique() < 2:
        return {"ate": np.nan, "ate_se": np.nan, "p_value": np.nan}
    A = data["A"].astype(int).to_numpy()
    Y = data["Y"].astype(float).to_numpy()
    W = data.drop(columns=["A", "Y"])
    folds = StratifiedKFold(n_splits=min(5, max(2, np.bincount(A).min())), shuffle=True, random_state=random_seed)
    e_hat = np.zeros(len(data), dtype=float)
    mu0 = np.zeros(len(data), dtype=float)
    mu1 = np.zeros(len(data), dtype=float)
    for tr, te in folds.split(W, A):
        prop_model = LogisticRegression(max_iter=2000, solver="lbfgs")
        out0 = RandomForestRegressor(n_estimators=60, min_samples_leaf=8, random_state=random_seed, n_jobs=-1)
        out1 = RandomForestRegressor(n_estimators=60, min_samples_leaf=8, random_state=random_seed + 1, n_jobs=-1)
        prop_model.fit(W.iloc[tr], A[tr])
        e_hat[te] = prop_model.predict_proba(W.iloc[te])[:, 1]
        if np.sum(A[tr] == 0) >= 5:
            out0.fit(W.iloc[tr][A[tr] == 0], Y[tr][A[tr] == 0])
            mu0[te] = out0.predict(W.iloc[te])
        else:
            mu0[te] = np.mean(Y[tr][A[tr] == 0]) if np.any(A[tr] == 0) else np.mean(Y[tr])
        if np.sum(A[tr] == 1) >= 5:
            out1.fit(W.iloc[tr][A[tr] == 1], Y[tr][A[tr] == 1])
            mu1[te] = out1.predict(W.iloc[te])
        else:
            mu1[te] = np.mean(Y[tr][A[tr] == 1]) if np.any(A[tr] == 1) else np.mean(Y[tr])
    e_hat = np.clip(e_hat, 0.02, 0.98)
    aipw = mu1 - mu0 + A * (Y - mu1) / e_hat - (1 - A) * (Y - mu0) / (1 - e_hat)
    ate = float(np.mean(aipw))
    se = float(np.std(aipw, ddof=1) / math.sqrt(len(aipw)))
    p = float(2 * stats.norm.sf(abs(ate / se))) if se > 0 else np.nan
    return {"ate": ate, "ate_se": se, "p_value": p}


def causal_cate_summary(
    X: pd.DataFrame,
    exposure: pd.Series,
    outcome: pd.Series,
    random_seed: int,
) -> Dict[str, float]:
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


def dose_response_summary(exposure: pd.Series, outcome: pd.Series, n_bins: int = 5) -> Dict[str, Any]:
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


def run_causal_screening(
    feature_matrix: pd.DataFrame,
    clinical_adjustment: pd.DataFrame,
    endpoint_train: pd.DataFrame,
    random_seed: int,
    max_features: int,
) -> pd.DataFrame:
    common = feature_matrix.index.intersection(endpoint_train.index)
    X_adj = clinical_adjustment.reindex(common).fillna(0)
    outcome = endpoint_train.reindex(common)["pseudo_risk_12m_clipped"].astype(float)
    rows: List[Dict[str, Any]] = []
    for feature in list(feature_matrix.columns)[:max_features]:
        x = feature_matrix.reindex(common)[feature]
        if x.notna().sum() < 30 or x.nunique(dropna=True) < 4:
            continue
        exposure_binary = (x > x.median()).astype(int)
        aipw = causal_aipw_binary_exposure(X_adj, exposure_binary, outcome, random_seed)
        cate = causal_cate_summary(X_adj, exposure_binary, outcome, random_seed)
        dose = dose_response_summary(x, outcome)
        corr = stats.spearmanr(x.fillna(x.median()), outcome.fillna(outcome.median())).correlation
        rows.append(
            {
                "feature": feature,
                "spearman_with_pseudo_risk": float(corr) if np.isfinite(corr) else np.nan,
                **aipw,
                **cate,
                **dose,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["ate_abs"] = table["ate"].abs()
    table["ate_rank"] = table["ate_abs"].rank(ascending=False, method="average")
    table["p_rank"] = table["p_value"].rank(ascending=True, method="average")
    table["dose_rank"] = table["dose_response_slope"].abs().rank(ascending=False, method="average")
    table["causal_priority_score"] = (
        -table["ate_rank"].fillna(table["ate_rank"].max() + 1)
        -0.5 * table["p_rank"].fillna(table["p_rank"].max() + 1)
        -0.25 * table["dose_rank"].fillna(table["dose_rank"].max() + 1)
    )
    return table.sort_values("causal_priority_score", ascending=False)


def synthetic_data_qc(
    X_real: pd.DataFrame,
    X_syn: pd.DataFrame,
    X_holdout: Optional[pd.DataFrame] = None,
    random_seed: int = RANDOM_SEED,
    min_good: int = 4,
) -> Dict[str, Any]:


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


    ks_stats: List[float] = []
    ks_pvals: Dict[str, float] = {}
    for col in cols:
        try:
            stat, pv = stats.ks_2samp(X_r[col].values, X_s[col].values)
            ks_stats.append(float(stat))
            ks_pvals[col] = float(pv)
        except Exception:
            pass
    ks_stat_mean = float(np.mean(ks_stats)) if ks_stats else 1.0
    ks_similarity = float(1.0 - ks_stat_mean)
    ks_pass_rate = float(np.mean([s > 0.05 for s in ks_pvals.values()])) if ks_pvals else 0.0
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
    pca_area_ratio = 1.0
    try:
        if n_pca >= 2:
            from sklearn.decomposition import PCA

            pca_unified = PCA(n_components=n_pca, random_state=random_seed)
            pca_proj_r = pca_unified.fit_transform(X_r_sc)
            pca_proj_s = pca_unified.transform(X_s_sc)

            pca_s = PCA(n_components=n_pca, random_state=random_seed)
            pca_s.fit(X_s_sc)
            pca_var_corr = float(np.corrcoef(
                pca_unified.explained_variance_ratio_,
                pca_s.explained_variance_ratio_,
            )[0, 1])
            if not np.isfinite(pca_var_corr):
                pca_var_corr = 0.0

            coords_r = pca_proj_r[:, :2]
            coords_s = pca_proj_s[:, :2]
            from scipy.spatial import ConvexHull
            try:
                hull_r = ConvexHull(coords_r)
                hull_s = ConvexHull(coords_s)
                area_r = hull_r.volume
                area_s = hull_s.volume
                pca_area_ratio = area_s / max(area_r, EPS)
            except Exception:
                pca_area_ratio = 1.0
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
    mia_llr = 0.0
    if X_holdout is not None and len(X_holdout) > 5:
        try:
            X_h = X_holdout[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            nn_s = NearestNeighbors(n_neighbors=min(5, len(X_s_sc)), metric="euclidean")
            nn_s.fit(X_s_sc)
            d_train = nn_s.kneighbors(X_r_sc)[0].mean(axis=1)
            X_h_sc = scaler_wd.transform(X_h)
            d_hold = nn_s.kneighbors(X_h_sc)[0].mean(axis=1)
            threshold = float(np.median(np.concatenate([d_train, d_hold])))
            mia_acc = float(
                (np.mean(d_train < threshold) + np.mean(d_hold >= threshold)) / 2
            )
            mia_llr = float(np.log(max(d_hold.mean(), EPS) / max(d_train.mean(), EPS)))
        except Exception:
            mia_acc = 0.5
    mia_good = mia_acc < 0.65


    good_count = sum([ks_good, wd_good, pca_good, dcr_good, mia_good])
    passed = good_count >= min_good

    return {
        "passed": bool(passed),
        "good_count": int(good_count),
        "ks_stat_mean": round(ks_stat_mean, 4),
        "ks_similarity": round(ks_similarity, 4),
        "ks_pass_rate": round(ks_pass_rate, 4),
        "ks_ref_floor": round(ks_ref, 4),
        "ks_tol": round(ks_tol, 4),
        "ks_good": ks_good,
        "wd_mean": round(wd_mean, 6),
        "wd_ref_floor": round(wd_ref, 6),
        "wd_tol": round(wd_tol, 6),
        "wd_good": wd_good,
        "pca_var_correlation": round(pca_var_corr, 4),
        "pca_area_ratio": round(pca_area_ratio, 4),
        "pca_good": pca_good,
        "dcr_ratio": round(dcr_ratio, 4),
        "dcr_good": dcr_good,
        "mia_attack_accuracy": round(mia_acc, 4),
        "mia_log_likelihood_ratio": round(mia_llr, 4),
        "mia_good": mia_good,
    }


SURVIVAL_GAN_CONDITION_COLUMNS = [
    "death_by_12m",
    "event",
    "log1p_time_months",
    "pseudo_risk_12m_clipped",
    "ipcw_weight_12m",
]
GAN_SAMPLING_STRATEGIES = {"balanced_event", "risk_stratified", "event_only", "overall"}
GAN_MIN_AUC_GAIN = 0.005
GAN_MIN_AP_GAIN = 0.01
GAN_BRIER_TOL = 0.005


def _moment_match_calibrate(
    X_syn: pd.DataFrame,
    X_real_train: pd.DataFrame,
    syn_event_mask: np.ndarray,
    real_event_label: np.ndarray,
) -> pd.DataFrame:


    out = X_syn.copy()
    cols = [c for c in X_real_train.columns if c in out.columns]
    groups = [
        (syn_event_mask, real_event_label == 1),
        (~syn_event_mask, real_event_label == 0),
    ]
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


class SmoteLikeAugmenter:


    def __init__(self, k_neighbors: int = 5, jitter: float = 0.0, random_seed: int = RANDOM_SEED) -> None:
        self.k_neighbors = int(k_neighbors)
        self.jitter = float(jitter)
        self.random_seed = int(random_seed)
        self.feature_columns: List[str] = []
        self.status = "not_fitted"
        self._class_arrays: Dict[int, np.ndarray] = {}
        self._class_nn: Dict[int, NearestNeighbors] = {}

    def fit(self, X: pd.DataFrame, condition: Any) -> "SmoteLikeAugmenter":
        data = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True)).copy()
        if len(data) < 10 or data.shape[1] == 0:
            self.status = "skipped_insufficient_training_data"
            return self
        if isinstance(condition, pd.DataFrame):
            labels = condition.reindex(data.index)["death_by_12m"].round().clip(0, 1).astype(int).to_numpy()
        else:
            arr = np.asarray(condition, dtype=float)
            labels = (arr[:, 0] if arr.ndim == 2 else arr).round().astype(int)
        self.feature_columns = list(data.columns)
        values = data.to_numpy(float)
        for cls in (0, 1):
            mask = labels == cls
            if int(mask.sum()) >= 2:
                arr = values[mask]
                self._class_arrays[cls] = arr
                k = min(self.k_neighbors + 1, len(arr))
                nn = NearestNeighbors(n_neighbors=k)
                nn.fit(arr)
                self._class_nn[cls] = nn
        if not self._class_arrays:
            self.status = "skipped_no_class_data"
            return self
        self.status = "fitted"
        return self

    def sample(self, n: int, condition_values: np.ndarray) -> pd.DataFrame:
        if self.status != "fitted":
            return pd.DataFrame(columns=self.feature_columns)
        arr = np.asarray(condition_values, dtype=float)
        labels = (arr[:, 0] if arr.ndim == 2 else arr).round().astype(int)
        rng = np.random.default_rng(self.random_seed)
        rows = np.zeros((n, len(self.feature_columns)), dtype=float)
        for i in range(n):
            cls = int(labels[i]) if i < len(labels) else 1
            if cls not in self._class_arrays:
                cls = next(iter(self._class_arrays))
            base = self._class_arrays[cls]
            j = int(rng.integers(0, len(base)))
            nn = self._class_nn[cls]
            neigh = nn.kneighbors(base[j : j + 1], return_distance=False)[0]
            neigh = [m for m in neigh if m != j] or [j]
            m = int(rng.choice(neigh))
            lam = float(rng.random())
            new = base[j] + lam * (base[m] - base[j])
            if self.jitter > 0:
                new = new + rng.normal(0.0, self.jitter, size=new.shape)
            rows[i] = new
        return pd.DataFrame(rows, columns=self.feature_columns)


def build_survival_gan_conditions(endpoint: pd.DataFrame, index: Sequence[Any]) -> pd.DataFrame:

    ep = endpoint.reindex(index).copy()
    death = ep.get("death_by_12m", pd.Series(0.0, index=ep.index)).astype(float)
    event = ep.get("event", death).astype(float)
    time_months = ep.get("time_months", pd.Series(TAU_MONTHS, index=ep.index)).astype(float)
    pseudo = ep.get("pseudo_risk_12m_clipped", death).astype(float)
    weight = ep.get("ipcw_weight_12m", pd.Series(1.0, index=ep.index)).astype(float)
    cond = pd.DataFrame(index=ep.index)
    cond["death_by_12m"] = death.fillna(0.0).clip(0.0, 1.0)
    cond["event"] = event.fillna(cond["death_by_12m"]).clip(0.0, 1.0)
    clean_time = time_months.replace([np.inf, -np.inf], np.nan).fillna(TAU_MONTHS).clip(lower=0.0)
    cond["log1p_time_months"] = np.log1p(clean_time)
    cond["pseudo_risk_12m_clipped"] = pseudo.replace([np.inf, -np.inf], np.nan).fillna(cond["death_by_12m"]).clip(0.0, 1.0)
    cond["ipcw_weight_12m"] = weight.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=EPS)
    return cond[SURVIVAL_GAN_CONDITION_COLUMNS].astype(float)


def _sample_condition_rows(
    endpoint_train: pd.DataFrame,
    train_idx: Sequence[Any],
    y_obs: pd.Series,
    n_syn: int,
    sampling_strategy: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
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
        cond["death_by_12m"] = 1.0
        cond["event"] = 1.0
        cond["pseudo_risk_12m_clipped"] = cond["pseudo_risk_12m_clipped"].clip(lower=0.5)
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
    if sampling_strategy == "risk_stratified" and "pseudo_risk_12m_clipped" in endpoint_train:
        pseudo = endpoint_train.loc[train_idx, "pseudo_risk_12m_clipped"].astype(float).replace([np.inf, -np.inf], np.nan)
        valid = pseudo.dropna()
        if len(valid) >= 3:
            try:
                ranks = valid.rank(method="first")
                bins = pd.qcut(ranks, q=min(3, len(valid)), labels=False, duplicates="drop")
                sampled_ids: List[Any] = []
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


def _monitor_qc_score(real: np.ndarray, fake: np.ndarray, cond: np.ndarray) -> float:
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
    score = mean_diff + std_diff + wd_mean + cond_gap
    return float(score) if np.isfinite(score) else float("inf")

class ConditionalTabularGAN:


    def __init__(
        self,
        latent_dim: int = 32,
        epochs: int = 300,
        batch_size: int = 32,
        n_critic: int = 5,
        lambda_gp: float = 10.0,
        lr: float = 2e-4,
        patience: int = 30,
        random_seed: int = RANDOM_SEED,
        skip_scaler: bool = False,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        weight_decay: float = 1e-5,
    ) -> None:
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
        self.generator: Any = None
        self.feature_columns: List[str] = []
        self.status = "not_fitted"
        self.training_history: List[Dict[str, float]] = []


    def _build_generator(self, input_dim: int, output_dim: int) -> Any:
        h = self.hidden_dim


        return nn.Sequential(
            nn.Linear(input_dim, h),
            nn.LayerNorm(h),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.ReLU(inplace=True),
            nn.Linear(h, output_dim),
        )

    def _build_discriminator(self, input_dim: int) -> Any:
        h = self.hidden_dim


        return nn.Sequential(
            nn.Linear(input_dim, h),
            nn.LayerNorm(h),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.LeakyReLU(0.2),
            nn.Linear(h, 1),
        )

    @staticmethod
    def _gradient_penalty(
        discriminator: Any, real: Any, fake: Any, cond: Any, lambda_gp: float
    ) -> Any:
        alpha = torch.rand(real.size(0), 1, device=real.device)
        interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_out = discriminator(torch.cat([interp, cond], dim=1))
        grads = torch.autograd.grad(
            outputs=d_out, inputs=interp,
            grad_outputs=torch.ones_like(d_out), create_graph=True,
        )[0]
        return lambda_gp * ((grads.norm(2, dim=1) - 1) ** 2).mean()

    def fit(self, X: pd.DataFrame, condition: pd.Series) -> "ConditionalTabularGAN":
        if torch is None or nn is None:
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
            self.scaler = QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution="normal",
                random_state=self.random_seed,
            )
            scaled = self.scaler.fit_transform(data)
        x_tensor = torch.tensor(scaled, dtype=torch.float32)
        c_tensor = torch.tensor(cond, dtype=torch.float32)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        input_dim = self.latent_dim + 1
        output_dim = data.shape[1]
        generator = self._build_generator(input_dim, output_dim)
        discriminator = self._build_discriminator(output_dim + 1)
        opt_g = torch.optim.Adam(
            generator.parameters(), lr=self.lr, betas=(0.0, 0.9), weight_decay=self.weight_decay
        )
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=self.lr, betas=(0.0, 0.9))

        n = len(data)


        monitor_n = min(max(self.batch_size, 16), n)
        monitor_idx = torch.arange(monitor_n)
        monitor_real = x_tensor[monitor_idx].numpy()
        monitor_cond_tensor = c_tensor[monitor_idx]
        monitor_cond = monitor_cond_tensor.numpy()
        monitor_noise = torch.randn((monitor_n, self.latent_dim))
        best_score = float("inf")
        best_state: Optional[Dict[str, Any]] = None
        best_epoch = 0
        patience_counter = 0
        self.training_history = []

        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            n_steps = 0
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
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
                self.training_history.append({
                    "epoch": epoch + 1,
                    "d_loss": epoch_d_loss / n_steps,
                    "g_loss": epoch_g_loss / n_steps,
                    "monitor_qc_score": monitor_score,
                    "best_epoch": float(best_epoch),
                })
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            generator.load_state_dict(best_state)
        self.generator = generator
        self.status = "fitted"
        return self

    def sample(self, n: int, condition_values: np.ndarray) -> pd.DataFrame:
        if self.status != "fitted" or self.generator is None or torch is None:
            return pd.DataFrame(columns=self.feature_columns)
        cond = np.asarray(condition_values, dtype=float).reshape(-1, 1)
        if len(cond) != n:
            warnings.warn(
                f"Condition array length ({len(cond)}) != n ({n}), resizing by repeating. "
                "This may silently alter condition distribution.",
                stacklevel=2,
            )
            cond = np.resize(cond, (n, 1))
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn((n, self.latent_dim))
            c = torch.tensor(cond, dtype=torch.float32)
            fake = self.generator(torch.cat([z, c], dim=1)).numpy()
        self.generator.train()
        if self.skip_scaler or self.scaler is None:
            inv = fake
        else:
            inv = self.scaler.inverse_transform(fake)
        return pd.DataFrame(inv, columns=self.feature_columns)


class SurvivalConditionalTabularGAN(ConditionalTabularGAN):


    condition_dim = len(SURVIVAL_GAN_CONDITION_COLUMNS)

    def _prepare_condition_array(self, condition: Any, index: Sequence[Any]) -> np.ndarray:
        if isinstance(condition, pd.DataFrame):
            cond_df = condition.reindex(index)
        else:
            arr = np.asarray(condition, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != self.condition_dim:
                raise ValueError(f"SurvivalConditionalTabularGAN requires condition shape (n, {self.condition_dim}).")
            if arr.shape[0] != len(index):
                raise ValueError("Condition row count must match training/sample row count.")
            return arr.astype(np.float32)
        missing = [c for c in SURVIVAL_GAN_CONDITION_COLUMNS if c not in cond_df.columns]
        if missing:
            raise ValueError(f"Missing survival GAN condition columns: {missing}")
        return cond_df[SURVIVAL_GAN_CONDITION_COLUMNS].astype(float).to_numpy(np.float32)

    def fit(self, X: pd.DataFrame, condition: Any) -> "SurvivalConditionalTabularGAN":
        if torch is None or nn is None:
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
            self.scaler = QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution="normal",
                random_state=self.random_seed,
            )
            scaled = self.scaler.fit_transform(data)
        x_tensor = torch.tensor(scaled, dtype=torch.float32)
        c_tensor = torch.tensor(cond, dtype=torch.float32)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        input_dim = self.latent_dim + self.condition_dim
        output_dim = data.shape[1]
        generator = self._build_generator(input_dim, output_dim)
        discriminator = self._build_discriminator(output_dim + self.condition_dim)
        opt_g = torch.optim.Adam(
            generator.parameters(), lr=self.lr, betas=(0.0, 0.9), weight_decay=self.weight_decay
        )
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=self.lr, betas=(0.0, 0.9))

        n = len(data)
        monitor_n = min(max(self.batch_size, 16), n)
        monitor_idx = torch.arange(monitor_n)
        monitor_real = x_tensor[monitor_idx].numpy()
        monitor_cond_tensor = c_tensor[monitor_idx]
        monitor_cond = monitor_cond_tensor.numpy()
        monitor_noise = torch.randn((monitor_n, self.latent_dim))
        best_score = float("inf")
        best_state: Optional[Dict[str, Any]] = None
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
                idx = perm[start : start + self.batch_size]
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
                    best_state = {
                        "generator": {k: v.detach().clone() for k, v in generator.state_dict().items()},
                        "discriminator": {k: v.detach().clone() for k, v in discriminator.state_dict().items()},
                    }
                else:
                    patience_counter += 1
                self.training_history.append({
                    "epoch": epoch + 1,
                    "d_loss": epoch_d_loss / n_steps,
                    "g_loss": epoch_g_loss / n_steps,
                    "gp": epoch_gp / n_steps,
                    "monitor_qc_score": monitor_score,
                    "best_epoch": float(best_epoch),
                })
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            generator.load_state_dict(best_state["generator"])
        self.generator = generator
        self.status = "fitted"
        return self

    def sample(self, n: int, condition_values: np.ndarray) -> pd.DataFrame:
        if self.status != "fitted" or self.generator is None or torch is None:
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


class FeatureSpaceGAN:


    def __init__(
        self,
        latent_k: int = 30,
        gan_latent_dim: int = 64,
        gan_epochs: int = 300,
        gan_batch_size: int = 32,
        gan_n_critic: int = 5,
        gan_lambda_gp: float = 10.0,
        gan_lr: float = 2e-4,
        gan_patience: int = 30,
        ae_epochs: int = 500,
        ae_lr: float = 1e-3,
        ae_batch_size: int = 32,
        ae_mode: str = "plain",
        random_seed: int = RANDOM_SEED,
    ) -> None:
        self.latent_k = latent_k
        self.gan_latent_dim = gan_latent_dim
        self.gan_epochs = gan_epochs
        self.gan_batch_size = gan_batch_size
        self.gan_n_critic = gan_n_critic
        self.gan_lambda_gp = gan_lambda_gp
        self.gan_lr = gan_lr
        self.gan_patience = gan_patience
        self.ae_epochs = ae_epochs
        self.ae_lr = ae_lr
        self.ae_batch_size = ae_batch_size
        self.ae_mode = str(ae_mode).strip().lower()
        self.random_seed = random_seed
        self.scaler = QuantileTransformer(output_distribution="normal", random_state=random_seed)
        self.encoder: Any = None
        self.decoder: Any = None
        self.gan: Optional[ConditionalTabularGAN] = None
        self.feature_columns: List[str] = []
        self.status = "not_fitted"
        self.reconstruction_mse: float = float("nan")

    def fit(self, X: pd.DataFrame, condition: Any, endpoint: Optional[pd.DataFrame] = None) -> "FeatureSpaceGAN":
        if torch is None or nn is None:
            self.status = "skipped_torch_missing"
            return self
        data = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True)).copy()
        if len(data) < 40 or data.shape[1] == 0:
            self.status = "skipped_insufficient_training_data"
            return self
        self.feature_columns = list(data.columns)
        n_quantiles = min(1000, len(data))
        self.scaler = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            random_state=self.random_seed,
        )
        scaled = self.scaler.fit_transform(data)
        d = scaled.shape[1]
        k = min(self.latent_k, d // 2, len(data) // 3)
        k = max(k, 5)

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)


        encoder = nn.Sequential(
            nn.Linear(d, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, k),
        )
        decoder = nn.Sequential(
            nn.Linear(k, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, d),
        )
        risk_head = nn.Linear(k, 1) if self.ae_mode == "risk_aware" else None
        ae_params = list(encoder.parameters()) + list(decoder.parameters())
        if risk_head is not None:
            ae_params += list(risk_head.parameters())
        opt_ae = torch.optim.Adam(ae_params, lr=self.ae_lr, weight_decay=1e-5)
        x_tensor = torch.tensor(scaled, dtype=torch.float32)
        if endpoint is not None:
            y_ae = endpoint.reindex(data.index)["death_by_12m"].fillna(0).astype(float).to_numpy()
            w_ae = endpoint.reindex(data.index)["ipcw_weight_12m"].replace(0, np.nan).fillna(1.0).astype(float).to_numpy()
        else:
            cond_arr = np.asarray(condition, dtype=float)
            y_ae = cond_arr[:, 0] if cond_arr.ndim == 2 else np.asarray(condition, dtype=float)
            w_ae = np.ones(len(data), dtype=float)
        y_tensor = torch.tensor(y_ae.reshape(-1, 1), dtype=torch.float32)
        w_tensor = torch.tensor(w_ae.reshape(-1, 1), dtype=torch.float32)
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        best_mse = float("inf")
        ae_patience = 30
        ae_wait = 0
        n_ae = len(x_tensor)
        for _ in range(self.ae_epochs):
            encoder.train()
            decoder.train()
            ae_perm = torch.randperm(n_ae)
            epoch_loss = 0.0
            n_batches = 0
            for ae_start in range(0, n_ae, self.ae_batch_size):
                ae_idx = ae_perm[ae_start:ae_start + self.ae_batch_size]
                if len(ae_idx) < 2:
                    continue
                xb = x_tensor[ae_idx]
                opt_ae.zero_grad()
                z = encoder(xb)
                recon = decoder(z)
                recon_loss = nn.MSELoss()(recon, xb)
                loss = recon_loss
                if risk_head is not None:
                    logits = risk_head(z)
                    yb = y_tensor[ae_idx]
                    wb = w_tensor[ae_idx]
                    risk_loss = (bce_loss(logits, yb) * wb).sum() / torch.clamp(wb.sum(), min=EPS)
                    loss = recon_loss + 0.2 * risk_loss
                loss.backward()
                opt_ae.step()
                epoch_loss += recon_loss.item()
                n_batches += 1
            mse_val = epoch_loss / max(n_batches, 1)
            if mse_val < best_mse - 1e-6:
                best_mse = mse_val
                ae_wait = 0
            else:
                ae_wait += 1
            if ae_wait >= ae_patience:
                break
        self.reconstruction_mse = best_mse
        encoder.eval()
        decoder.eval()


        with torch.no_grad():
            latent_repr = encoder(x_tensor).numpy()
        latent_df = pd.DataFrame(latent_repr, index=data.index)


        if isinstance(condition, pd.DataFrame):
            cond = condition.reindex(data.index)
        else:
            cond = pd.DataFrame(np.asarray(condition), index=data.index, columns=SURVIVAL_GAN_CONDITION_COLUMNS)
        self.gan = SurvivalConditionalTabularGAN(
            latent_dim=self.gan_latent_dim,
            epochs=self.gan_epochs,
            batch_size=self.gan_batch_size,
            n_critic=self.gan_n_critic,
            lambda_gp=self.gan_lambda_gp,
            lr=self.gan_lr,
            patience=self.gan_patience,
            random_seed=self.random_seed,
            skip_scaler=True,
        )
        self.gan.fit(latent_df, cond)
        if self.gan.status != "fitted":
            self.status = f"gan_failed_{self.gan.status}"
            return self

        self.encoder = encoder
        self.decoder = decoder
        self.status = "fitted"
        return self

    def sample(self, n: int, condition_values: np.ndarray) -> pd.DataFrame:
        if self.status != "fitted" or self.gan is None or self.decoder is None:
            return pd.DataFrame(columns=self.feature_columns)

        X_syn_latent = self.gan.sample(n, condition_values)
        if X_syn_latent.empty:
            return pd.DataFrame(columns=self.feature_columns)

        self.decoder.eval()
        with torch.no_grad():
            latent_tensor = torch.tensor(X_syn_latent.values, dtype=torch.float32)
            recon = self.decoder(latent_tensor).numpy()
        inv = self.scaler.inverse_transform(recon)
        return pd.DataFrame(inv, columns=self.feature_columns)


def train_only_gan_augmentation(
    X_train: pd.DataFrame,
    endpoint_train: pd.DataFrame,
    cfg: PipelineConfig,
    audit: AuditLog,
    sampling_strategy: str = "overall",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:

    y_obs, _ = censor_safe_binary_training(endpoint_train)
    eligible_ids = list(y_obs.index)
    sampling_strategy = str(sampling_strategy).strip().lower()
    if sampling_strategy not in GAN_SAMPLING_STRATEGIES:
        sampling_strategy = "overall"
    if not cfg.run_gan:
        audit.add("gan_augmentation", "SKIP", "GAN disabled by command line.")
        return X_train, endpoint_train, {"status": "disabled", "n_synthetic": 0}
    if len(eligible_ids) < 40 or y_obs.nunique() < 2:
        audit.add("gan_augmentation", "SKIP", "Insufficient observed 12-month labels in TCGA_train_real.")
        return X_train, endpoint_train, {"status": "insufficient_labels", "n_synthetic": 0}

    rng_split = np.random.default_rng(cfg.random_seed)
    holdout_frac = 0.15
    n_holdout = min(len(eligible_ids) - 2, max(5, int(len(eligible_ids) * holdout_frac)))
    holdout_idx = list(rng_split.choice(eligible_ids, size=n_holdout, replace=False))
    train_idx = [i for i in eligible_ids if i not in set(holdout_idx)]
    event_train_idx = [i for i in train_idx if int(y_obs.loc[i]) == 1]
    event_holdout_idx = [i for i in holdout_idx if int(y_obs.loc[i]) == 1]
    gan_fit_idx = event_train_idx if sampling_strategy == "event_only" else train_idx
    if sampling_strategy in {"event_only", "balanced_event"} and len(event_train_idx) < 5:
        audit.add(
            "gan_augmentation",
            "SKIP",
            "Insufficient event-class training rows for event-focused GAN augmentation.",
            sampling_strategy=sampling_strategy,
            n_event_train=len(event_train_idx),
        )
        return X_train, endpoint_train, {
            "status": "insufficient_event_training_data",
            "n_synthetic": 0,
            "sampling_strategy": sampling_strategy,
        }
    if len(gan_fit_idx) < 40:
        audit.add("gan_augmentation", "SKIP", "Insufficient GAN fit rows after strategy filtering.")
        return X_train, endpoint_train, {"status": "insufficient_gan_fit_rows", "n_synthetic": 0, "sampling_strategy": sampling_strategy}

    X_train_gan = X_train.loc[gan_fit_idx]
    X_holdout = X_train.loc[holdout_idx]
    X_qc_real = X_train.loc[gan_fit_idx]
    X_qc_holdout = X_train.loc[event_holdout_idx] if sampling_strategy == "event_only" and len(event_holdout_idx) > 5 else X_holdout
    y_train_gan = y_obs.loc[gan_fit_idx]
    cond_train = build_survival_gan_conditions(endpoint_train, gan_fit_idx)

    candidate_methods: List[Tuple[str, Any]] = []
    if cfg.gan_use_feature_space and X_train_gan.shape[1] > cfg.gan_latent_k * 2:
        for ae_mode in ["plain", "risk_aware"]:
            candidate_methods.append((
                f"feature_space_{ae_mode}_survival_wgan_gp",
                FeatureSpaceGAN(
                    latent_k=cfg.gan_latent_k,
                    gan_latent_dim=cfg.gan_latent_dim,
                    gan_epochs=cfg.gan_epochs,
                    gan_batch_size=cfg.gan_batch_size,
                    gan_n_critic=cfg.gan_n_critic,
                    gan_lambda_gp=cfg.gan_lambda_gp,
                    gan_lr=cfg.gan_lr,
                    gan_patience=cfg.gan_patience,
                    ae_mode=ae_mode,
                    random_seed=cfg.random_seed,
                ),
            ))
    candidate_methods.append((
        "direct_survival_wgan_gp",
        SurvivalConditionalTabularGAN(
            latent_dim=cfg.gan_latent_dim,
            epochs=cfg.gan_epochs,
            batch_size=cfg.gan_batch_size,
            n_critic=cfg.gan_n_critic,
            lambda_gp=cfg.gan_lambda_gp,
            lr=cfg.gan_lr,
            patience=cfg.gan_patience,
            random_seed=cfg.random_seed,
        ),
    ))
    candidate_methods.append((
        "legacy_direct_wgan_gp",
        ConditionalTabularGAN(
            latent_dim=cfg.gan_latent_dim,
            epochs=cfg.gan_epochs,
            batch_size=cfg.gan_batch_size,
            n_critic=cfg.gan_n_critic,
            lambda_gp=cfg.gan_lambda_gp,
            lr=cfg.gan_lr,
            patience=cfg.gan_patience,
            random_seed=cfg.random_seed,
        ),
    ))


    candidate_methods.append((
        "smote_interpolation",
        SmoteLikeAugmenter(k_neighbors=5, random_seed=cfg.random_seed),
    ))

    n_syn_candidate = int(round(len(train_idx) * cfg.gan_aug_ratio))
    if n_syn_candidate <= 0:
        return X_train, endpoint_train, {"status": "zero_synthetic_requested", "n_synthetic": 0, "sampling_strategy": sampling_strategy}

    X_syn: Optional[pd.DataFrame] = None
    best_qc: Dict[str, Any] = {"passed": False, "good_count": 0}
    best_sampled_conds: Optional[pd.DataFrame] = None
    best_method = ""
    qc_attempts: List[Dict[str, Any]] = []

    X_real_train = X_train.loc[train_idx]
    real_event_label = y_obs.loc[train_idx].to_numpy(int)

    for method_name, gan_model in candidate_methods:
        try:
            if isinstance(gan_model, FeatureSpaceGAN):
                gan_model.fit(X_train_gan, cond_train, endpoint_train.loc[gan_fit_idx])
            elif isinstance(gan_model, (SurvivalConditionalTabularGAN, SmoteLikeAugmenter)):
                gan_model.fit(X_train_gan, cond_train)
            else:
                gan_model.fit(X_train_gan, y_train_gan)
            if gan_model.status != "fitted":
                qc_attempts.append({"method": method_name, "status": gan_model.status, "passed": False, "good_count": 0})
                continue
            for attempt in range(3):
                sampled_cond_df = _sample_condition_rows(
                    endpoint_train, train_idx, y_obs, n_syn_candidate, sampling_strategy, rng_split
                )
                if sampled_cond_df.empty:
                    continue
                if isinstance(gan_model, (FeatureSpaceGAN, SurvivalConditionalTabularGAN, SmoteLikeAugmenter)):
                    sample_condition = sampled_cond_df[SURVIVAL_GAN_CONDITION_COLUMNS].to_numpy(float)
                else:
                    sample_condition = sampled_cond_df["death_by_12m"].to_numpy(int)
                X_syn_candidate = gan_model.sample(n_syn_candidate, sample_condition)
                if X_syn_candidate.empty:
                    continue


                syn_event_mask = sampled_cond_df["death_by_12m"].to_numpy(int)[: len(X_syn_candidate)] == 1
                X_syn_candidate = _moment_match_calibrate(
                    X_syn_candidate, X_real_train, syn_event_mask, real_event_label
                )
                qc_result = synthetic_data_qc(
                    X_qc_real, X_syn_candidate, X_qc_holdout,
                    random_seed=cfg.random_seed, min_good=cfg.gan_qc_min_good,
                )
                qc_result["attempt"] = attempt + 1
                qc_result["method"] = method_name
                qc_attempts.append(qc_result)
                good_count = int(qc_result.get("good_count", 0))
                best_good = int(best_qc.get("good_count", 0))
                is_better = good_count > best_good or (
                    good_count == best_good
                    and bool(qc_result.get("passed", False))
                    and not bool(best_qc.get("passed", False))
                )
                if is_better:
                    best_qc = qc_result
                    X_syn = X_syn_candidate
                    best_sampled_conds = sampled_cond_df.copy()
                    best_method = method_name
                if qc_result.get("passed", False):
                    break
        except Exception as exc:
            qc_attempts.append({"method": method_name, "status": f"failed:{exc}", "passed": False, "good_count": 0})

    if X_syn is None or X_syn.empty or best_sampled_conds is None:
        audit.add("gan_augmentation", "SKIP", "WGAN-GP returned no synthetic rows.")
        return X_train, endpoint_train, {
            "status": "empty_sample",
            "n_synthetic": 0,
            "sampling_strategy": sampling_strategy,
            "qc_attempts": qc_attempts,
        }
    if not best_qc.get("passed", False):
        audit.add(
            "gan_augmentation",
            "WARN",
            f"Synthetic data QC failed (good={best_qc.get('good_count', 0)}/5); rejecting this GAN candidate.",
            sampling_strategy=sampling_strategy,
            qc=best_qc,
        )
        return X_train, endpoint_train, {
            "status": "qc_failed",
            "sampling_strategy": sampling_strategy,
            "n_synthetic": 0,
            "qc": best_qc,
            "qc_attempts": qc_attempts,
            "method": best_method,
        }

    n_syn = len(X_syn)
    sampled_conditions = best_sampled_conds.iloc[:n_syn].reset_index(drop=True)
    syn_ids = [f"SYN_TRAIN_ONLY_{i:05d}" for i in range(n_syn)]
    X_syn.index = syn_ids
    endpoint_syn = pd.DataFrame(index=syn_ids)
    endpoint_syn["PATIENT_ID"] = syn_ids
    endpoint_syn["time_months"] = np.expm1(sampled_conditions["log1p_time_months"].to_numpy(float))
    invalid_time = ~np.isfinite(endpoint_syn["time_months"].to_numpy(float)) | (endpoint_syn["time_months"].to_numpy(float) <= 0)
    if invalid_time.any():
        endpoint_syn.loc[invalid_time, "time_months"] = cfg.tau_months
    endpoint_syn["event"] = sampled_conditions["event"].round().clip(0, 1).astype(int).to_numpy()
    endpoint_syn["death_by_12m"] = sampled_conditions["death_by_12m"].round().clip(0, 1).astype(float).to_numpy()
    endpoint_syn["death_by_12m_observed"] = False
    endpoint_syn["early_censored_before_12m"] = False
    endpoint_syn["ipcw_label_available"] = True
    endpoint_syn["ipcw_weight_12m"] = sampled_conditions["ipcw_weight_12m"].replace(0, np.nan).fillna(1.0).astype(float).to_numpy()
    endpoint_syn["pseudo_risk_12m"] = sampled_conditions["pseudo_risk_12m_clipped"].clip(0.02, 0.98).astype(float).to_numpy()
    endpoint_syn["pseudo_risk_12m_clipped"] = endpoint_syn["pseudo_risk_12m"].clip(0.02, 0.98)


    X_aug = pd.concat([X_train, X_syn], axis=0)
    endpoint_aug = pd.concat([endpoint_train, endpoint_syn], axis=0)
    mean_shift = (X_syn.mean(numeric_only=True) - X_train.mean(numeric_only=True)).abs().mean()
    if best_method == "legacy_direct_wgan_gp":
        method_label = "Legacy direct WGAN-GP"
    elif best_method == "smote_interpolation":
        method_label = "SMOTE-style interpolation"
    elif best_method.startswith("feature_space_"):
        method_label = "Feature-space survival-aware WGAN-GP"
    else:
        method_label = "Survival-aware WGAN-GP"
    audit.add(
        "gan_augmentation",
        "OK",
        f"{method_label} produced {n_syn} synthetic rows using {sampling_strategy}; "
        f"method={best_method}; QC good_count={best_qc.get('good_count', 0)}/5.",
        sampling_strategy=sampling_strategy,
        method=best_method,
        n_synthetic=n_syn,
        mean_abs_feature_mean_shift=float(mean_shift),
        qc=best_qc,
        qc_attempts=qc_attempts,
    )
    return X_aug, endpoint_aug, {
        "status": "fitted",
        "sampling_strategy": sampling_strategy,
        "method": best_method,
        "gan_fit_n": int(len(gan_fit_idx)),
        "gan_fit_events": int(y_train_gan.sum()),
        "gan_fit_nonevents": int((1 - y_train_gan).sum()),
        "n_synthetic": n_syn,
        "mean_abs_feature_mean_shift": float(mean_shift),
        "qc": best_qc,
        "qc_attempts": qc_attempts,
    }


def select_train_only_gan_augmentation(
    X_train: pd.DataFrame,
    endpoint_train: pd.DataFrame,
    cfg: PipelineConfig,
    audit: AuditLog,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    candidate_ratios = tuple(float(r) for r in cfg.gan_aug_ratio_candidates if float(r) > 0)
    if not candidate_ratios:
        candidate_ratios = (float(cfg.gan_aug_ratio),)
    sampling_strategies = tuple(
        s.strip().lower()
        for s in cfg.gan_sampling_strategies
        if s.strip().lower() in GAN_SAMPLING_STRATEGIES
    )
    if not sampling_strategies:
        sampling_strategies = ("balanced_event", "risk_stratified", "event_only", "overall")
    baseline_metrics = train_only_cv_metrics_for_real_logistic(X_train, endpoint_train, cfg.random_seed)
    audit.add(
        "gan_ratio_selection",
        "OK",
        "Selecting GAN augmentation strategy and ratio by QC and strict train-only utility gain.",
        candidate_ratios=list(candidate_ratios),
        sampling_strategies=list(sampling_strategies),
        baseline_metrics=baseline_metrics,
    )

    candidates: List[Dict[str, Any]] = []
    for sampling_strategy in sampling_strategies:
        for ratio in candidate_ratios:
            candidate_cfg = dataclasses.replace(
                cfg,
                gan_aug_ratio=ratio,
                gan_aug_ratio_candidates=(ratio,),
                gan_sampling_strategies=(sampling_strategy,),
            )
            X_aug, endpoint_aug, report = train_only_gan_augmentation(
                X_train,
                endpoint_train,
                candidate_cfg,
                audit,
                sampling_strategy=sampling_strategy,
            )
            cv_metrics = {"auc_12m_observed": float("nan"), "average_precision_12m": float("nan"), "brier_12m_ipcw": float("nan")}
            if report.get("status") == "fitted":
                cv_metrics = train_only_cv_metrics_for_augmented_logistic(
                    X_train,
                    endpoint_train,
                    cfg,
                    ratio,
                    sampling_strategy,
                    cfg.random_seed,
                )
            qc = report.get("qc", {}) if isinstance(report.get("qc", {}), dict) else {}
            auc_gain = float(cv_metrics.get("auc_12m_observed", float("nan"))) - float(baseline_metrics.get("auc_12m_observed", float("nan")))
            ap_delta = float(cv_metrics.get("average_precision_12m", float("nan"))) - float(baseline_metrics.get("average_precision_12m", float("nan")))
            brier_delta = float(cv_metrics.get("brier_12m_ipcw", float("nan"))) - float(baseline_metrics.get("brier_12m_ipcw", float("nan")))


            meaningful_auc_gain = np.isfinite(auc_gain) and auc_gain >= GAN_MIN_AUC_GAIN
            no_ap_regression = np.isfinite(ap_delta) and ap_delta >= 0.0
            no_brier_regression = np.isfinite(brier_delta) and brier_delta <= 0.0
            utility_passed = (
                report.get("status") == "fitted"
                and bool(qc.get("passed", False))
                and meaningful_auc_gain
                and no_ap_regression
                and no_brier_regression
            )
            candidate = {
                "sampling_strategy": sampling_strategy,
                "ratio": ratio,
                "X_aug": X_aug,
                "endpoint_aug": endpoint_aug,
                "report": report,
                "cv_metrics": cv_metrics,
                "cv_auc_12m_observed": cv_metrics.get("auc_12m_observed", float("nan")),
                "cv_average_precision_12m": cv_metrics.get("average_precision_12m", float("nan")),
                "cv_brier_12m_ipcw": cv_metrics.get("brier_12m_ipcw", float("nan")),
                "auc_gain_vs_baseline": auc_gain,
                "ap_delta_vs_baseline": ap_delta,
                "brier_delta_vs_baseline": brier_delta,
                "utility_passed": bool(utility_passed),
                "qc_passed": bool(qc.get("passed", False)),
                "qc_good_count": int(qc.get("good_count", 0)),
                "n_synthetic": int(report.get("n_synthetic", 0)),
                "method": report.get("method", ""),
            }
            candidates.append(candidate)
            audit.add(
                "gan_ratio_selection",
                "OK" if candidate["utility_passed"] else "WARN",
                f"GAN {sampling_strategy} ratio {ratio:.2f} evaluated: "
                f"n_synthetic={candidate['n_synthetic']}, QC good={candidate['qc_good_count']}/5, "
                f"train-only CV AUC={candidate['cv_auc_12m_observed']:.4f}, "
                f"AP={candidate['cv_average_precision_12m']:.4f}, Brier={candidate['cv_brier_12m_ipcw']:.4f}.",
                sampling_strategy=sampling_strategy,
                ratio=ratio,
                n_synthetic=candidate["n_synthetic"],
                qc_passed=candidate["qc_passed"],
                qc_good_count=candidate["qc_good_count"],
                utility_passed=candidate["utility_passed"],
                train_only_cv_metrics=cv_metrics,
            )

    fitted = [c for c in candidates if c["report"].get("status") == "fitted"]
    if not fitted:
        return X_train, endpoint_train, {
            "status": "no_fitted_candidate",
            "n_synthetic": 0,
            "candidate_ratios": list(candidate_ratios),
            "sampling_strategies": list(sampling_strategies),
            "candidates": [
                {k: v for k, v in c.items() if k not in {"X_aug", "endpoint_aug"}}
                for c in candidates
            ],
        }

    eligible = [c for c in fitted if c["utility_passed"]]
    if not eligible:
        audit.add(
            "gan_ratio_selection",
            "WARN",
            "No GAN candidate passed strict QC + utility gate; using real training data only.",
            baseline_metrics=baseline_metrics,
        )
        return X_train, endpoint_train, {
            "status": "rejected_no_strict_utility_gain",
            "n_synthetic": 0,
            "utility_gate": f"qc_passed AND auc_gain>={GAN_MIN_AUC_GAIN:.3f} AND ap_delta>=0.000 AND brier_delta<=0.000",
            "baseline_metrics": baseline_metrics,
            "candidate_ratios": list(candidate_ratios),
            "sampling_strategies": list(sampling_strategies),
            "candidates": [
                {k: v for k, v in c.items() if k not in {"X_aug", "endpoint_aug"}}
                for c in candidates
            ],
            "selected_candidate": None,
        }

    def _selection_key(candidate: Dict[str, Any]) -> Tuple[float, float, float, float, int]:

        auc_gain = float(candidate.get("auc_gain_vs_baseline", float("nan")))
        ap_delta = float(candidate.get("ap_delta_vs_baseline", float("nan")))
        brier = float(candidate.get("cv_brier_12m_ipcw", float("nan")))
        auc_gain = auc_gain if np.isfinite(auc_gain) else -1.0
        ap_delta = ap_delta if np.isfinite(ap_delta) else -1.0
        return (
            -float(candidate["ratio"]),
            auc_gain,
            ap_delta,
            -(brier if np.isfinite(brier) else 1.0),
            int(candidate.get("qc_good_count", 0)),
        )

    selected = sorted(eligible, key=_selection_key, reverse=True)[0]
    selected_report = dict(selected["report"])
    selected_report.update(
        {
            "status": "fitted",
            "selected_sampling_strategy": selected["sampling_strategy"],
            "selected_aug_ratio": selected["ratio"],
            "selected_method": selected.get("method", ""),
            "selection_rule": (
                f"Require QC passed AND train-only CV AUC gain >= {GAN_MIN_AUC_GAIN:.3f} vs the real-only baseline, "
                "with average precision and Brier non-inferior. Among passing candidates, prefer the smallest "
                "synthetic ratio, then higher AUC gain, higher AP delta, lower Brier, and higher QC good_count."
            ),
            "utility_gate": f"qc_passed AND auc_gain>={GAN_MIN_AUC_GAIN:.3f} AND ap_delta>=0.000 AND brier_delta<=0.000",
            "baseline_metrics": baseline_metrics,
            "candidate_metrics": selected["cv_metrics"],
            "train_only_cv_auc_12m_observed": selected["cv_auc_12m_observed"],
            "train_only_cv_average_precision_12m": selected["cv_average_precision_12m"],
            "train_only_cv_brier_12m_ipcw": selected["cv_brier_12m_ipcw"],
            "candidate_ratios": list(candidate_ratios),
            "sampling_strategies": list(sampling_strategies),
            "candidates": [
                {k: v for k, v in c.items() if k not in {"X_aug", "endpoint_aug"}}
                for c in candidates
            ],
            "selected_candidate": {k: v for k, v in selected.items() if k not in {"X_aug", "endpoint_aug"}},
        }
    )
    audit.add(
        "gan_ratio_selection",
        "OK",
        f"Selected GAN {selected['sampling_strategy']} augmentation ratio {selected['ratio']:.2f}: "
        f"n_synthetic={selected['n_synthetic']}, QC good={selected['qc_good_count']}/5, "
        f"train-only CV AUC={selected['cv_auc_12m_observed']:.4f}, "
        f"AP={selected['cv_average_precision_12m']:.4f}, Brier={selected['cv_brier_12m_ipcw']:.4f}.",
        selected_sampling_strategy=selected["sampling_strategy"],
        selected_aug_ratio=selected["ratio"],
        selected_method=selected.get("method", ""),
        n_synthetic=selected["n_synthetic"],
        qc_passed=selected["qc_passed"],
        qc_good_count=selected["qc_good_count"],
        train_only_cv_auc_12m_observed=selected["cv_auc_12m_observed"],
        train_only_cv_average_precision_12m=selected["cv_average_precision_12m"],
        train_only_cv_brier_12m_ipcw=selected["cv_brier_12m_ipcw"],
    )
    return selected["X_aug"], selected["endpoint_aug"], selected_report


def load_external_cohorts(preprocessed_root: Path, tau: float, audit: AuditLog) -> Dict[str, pd.DataFrame]:
    cohorts: Dict[str, pd.DataFrame] = {}
    for path in sorted(preprocessed_root.glob("*_os_clinical_endpoint_qc.tsv")):
        if path.name.startswith("tcga_"):
            continue
        try:
            df = read_tsv(path)
            if "PATIENT_ID" not in df.columns:
                continue
            endpoint = add_fixed_12m_endpoint(df, tau=tau).set_index("PATIENT_ID", drop=False)
            endpoint = compute_ipcw_weights(endpoint, tau)
            endpoint = compute_pseudo_observations(endpoint, tau)
            name = path.name.replace("_os_clinical_endpoint_qc.tsv", "")
            cohorts[name] = endpoint
        except Exception as exc:
            audit.add("external_load", "WARN", f"Cannot load {path.name}: {exc}")
    return cohorts


def build_external_feature_matrix(
    cohort_name: str,
    endpoint: pd.DataFrame,
    raw_root: Path,
    clinical_pipe: Pipeline,
    clinical_names: List[str],
    expression_features: Sequence[str],
    expression_pipe: Optional[Pipeline],
    pathway_features: Sequence[str],
    train_pathway_pipe: Optional[Pipeline],
    mutation_features: Sequence[str],
    audit: AuditLog,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    clinical_X, missing_clinical = transform_clinical_frame(clinical_pipe, endpoint, clinical_names)
    blocks = [clinical_X]
    compatibility = {"clinical": "ok" if not missing_clinical else f"imputed_missing:{len(missing_clinical)}"}

    expr_candidates = {
        "cptac": raw_root / "preprocessed" / "cptac_coad_rna_gene_level_matrix.tsv",
        "geo_gse103479": raw_root / "preprocessed" / "geo_gse103479_expression_matrix.tsv",
        "htan": raw_root / "preprocessed" / "htan_crc_pseudobulk_rna_matrix.tsv",
    }
    expr_path = None
    for key, path in expr_candidates.items():
        if cohort_name.startswith(key) and path.exists():
            expr_path = path
            break
    expr_scaled: Optional[pd.DataFrame] = None
    if expr_path is not None and expression_features and expression_pipe is not None:
        try:
            expr = matrix_patients_to_rows(expr_path)
            present = [f for f in expression_features if f in expr.columns]
            if present:
                tmp = expr.reindex(endpoint.index).reindex(columns=list(expression_features))
                tmp = tmp.apply(pd.to_numeric, errors="coerce")
                expr_scaled = pd.DataFrame(
                    expression_pipe.transform(tmp),
                    index=endpoint.index,
                    columns=list(expression_features),
                )
                blocks.append(expr_scaled)
                compatibility["expression"] = f"ok:{len(present)}/{len(expression_features)}"
            else:
                compatibility["expression"] = "no_overlap"
        except Exception as exc:
            compatibility["expression"] = f"error:{exc}"
    else:
        compatibility["expression"] = "not_available"

    if expr_scaled is not None and pathway_features and train_pathway_pipe is not None:
        try:
            pathway_raw, _ = compute_pathway_activity(expr_scaled, raw_root, max_pathways=max(100, len(pathway_features)))
            if not pathway_raw.empty:
                tmp_pathway = pathway_raw.reindex(endpoint.index).reindex(columns=list(pathway_features))
                tmp_pathway = tmp_pathway.apply(pd.to_numeric, errors="coerce")
                pathway_scaled = pd.DataFrame(
                    train_pathway_pipe.transform(tmp_pathway),
                    index=endpoint.index,
                    columns=list(pathway_features),
                )
                blocks.append(pathway_scaled)
                present_pathways = [p for p in pathway_features if p in pathway_raw.columns]
                compatibility["pathway"] = f"ok:{len(present_pathways)}/{len(pathway_features)}"
            else:
                compatibility["pathway"] = "no_pathway_overlap"
        except Exception as exc:
            compatibility["pathway"] = f"error:{exc}"
    else:
        compatibility["pathway"] = "not_available"

    mutation_candidates = {
        "cptac": raw_root / "preprocessed" / "cptac_coad_mutation_gene_level_matrix.tsv",
        "htan": raw_root / "preprocessed" / "htan_crc_mutation_gene_level_matrix.tsv",
        "msk": raw_root / "preprocessed" / "msk_crc_2017_mutation_gene_level_matrix.tsv",
    }
    mut_path = None
    for key, path in mutation_candidates.items():
        if cohort_name.startswith(key) and path.exists():
            mut_path = path
            break
    if mut_path is not None and mutation_features:
        try:
            mut = patient_feature_matrix(mut_path, prefix="MUT_")
            present_mutations = [f for f in mutation_features if f in mut.columns]
            if present_mutations:
                mut_binary = (mut.reindex(endpoint.index).reindex(columns=list(mutation_features)).fillna(0) > 0).astype(float)
                blocks.append(mut_binary)
                compatibility["somatic_mutation"] = f"ok:{len(present_mutations)}/{len(mutation_features)}"
            else:
                compatibility["somatic_mutation"] = "no_overlap"
        except Exception as exc:
            compatibility["somatic_mutation"] = f"error:{exc}"
    else:
        compatibility["somatic_mutation"] = "not_available"

    X = pd.concat(blocks, axis=1).fillna(0.0)
    audit.add("external_features", "OK", f"{cohort_name}: {compatibility}")
    return X, compatibility


def save_json(data: Mapping[str, Any], path: Path) -> None:
    def default(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return str(obj)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=default)


def collect_dependency_report() -> pd.DataFrame:
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "joblib",
        "matplotlib",
        "lifelines",
        "sksurv",
        "torch",
        "pycox",
    ]
    rows = []
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            status = "available"
        except Exception as exc:
            version = ""
            status = f"missing:{exc.__class__.__name__}"
        rows.append({"package": pkg, "status": status, "version": version})
    rows.append({"package": "python", "status": "available", "version": sys.version.split()[0]})
    return pd.DataFrame(rows)


def write_environment_file(out_dir: Path) -> Path:
    path = out_dir / "environment_requirements.txt"
    content = """# Environment for 1YearOS.py
python==3.11.x
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
scikit-learn==1.5.1
joblib==1.4.2
matplotlib==3.9.0
lifelines==0.28.0
scikit-survival==0.23.0
torch==2.3.1
pycox==0.3.0  # optional exploratory deep survival models
"""
    path.write_text(content, encoding="utf-8")
    return path


def run_pipeline(cfg: PipelineConfig) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.random_seed)
    audit = AuditLog()
    run_dir = ensure_dir(cfg.output_root / f"1YearOS_{now_stamp()}")
    tables_dir = ensure_dir(run_dir / "tables")
    models_dir = ensure_dir(run_dir / "models")
    reports_dir = ensure_dir(run_dir / "reports")
    figures_dir = ensure_dir(run_dir / "figures")

    dep = collect_dependency_report()
    dep.to_csv(tables_dir / "dependency_report.tsv", sep="\t", index=False)
    env_path = write_environment_file(reports_dir)
    if cfg.strict_dependencies:
        missing_core = dep[dep["package"].isin(["lifelines", "sksurv"]) & dep["status"].str.startswith("missing")]
        if not missing_core.empty:
            raise RuntimeError("Strict dependency mode failed: lifelines and scikit-survival are required.")
    audit.add("environment", "OK", f"Dependency report and environment file written to {run_dir}.")

    plan_hash = sha256_file(cfg.plan_path) if cfg.plan_path.exists() else ""
    save_json(
        {
            "plan_path": cfg.plan_path,
            "plan_sha256": plan_hash,
            "project_root": cfg.project_root,
            "raw_data_root": cfg.raw_data_root,
            "tau_months": cfg.tau_months,
            "random_seed": cfg.random_seed,
        },
        reports_dir / "run_manifest.json",
    )

    preprocessed = cfg.raw_data_root / "preprocessed"
    survival_models = cfg.raw_data_root / "survival_models"
    causal_dir = cfg.raw_data_root / "causal_screening"
    tcga_clin_path = find_existing(
        [
            preprocessed / "tcga_os_clinical_endpoint_qc.tsv",
            preprocessed / "tcga_coadread_clinical_os_pfs_qc.tsv",
        ]
    )
    if tcga_clin_path is None:
        raise FileNotFoundError("TCGA clinical endpoint table was not found in rawData/preprocessed.")
    tcga = add_fixed_12m_endpoint(read_tsv(tcga_clin_path), tau=cfg.tau_months).set_index("PATIENT_ID", drop=False)
    tcga = compute_ipcw_weights(tcga, cfg.tau_months)
    tcga = compute_pseudo_observations(tcga, cfg.tau_months)
    tcga.to_csv(tables_dir / "tcga_12m_endpoint_ipcw_pseudo.tsv", sep="\t", index=False)
    audit.add(
        "endpoint",
        "OK",
        "Constructed 12-month endpoint; early censored cases retain missing fixed-time label.",
        n=int(len(tcga)),
        early_censored=int(tcga["early_censored_before_12m"].sum()),
    )

    split_path = survival_models / "tcga_train_internal_validation_split.tsv"
    if split_path.exists():
        split_df = read_tsv(split_path)
        split_df["PATIENT_ID"] = split_df["PATIENT_ID"].map(normalize_patient_id)
        train_ids = split_df.loc[split_df["split"].eq("train"), "PATIENT_ID"].tolist()
        val_ids = split_df.loc[split_df["split"].eq("internal_validation"), "PATIENT_ID"].tolist()
        train_ids = [pid for pid in train_ids if pid in tcga.index]
        val_ids = [pid for pid in val_ids if pid in tcga.index]
        split_source = str(split_path)
    else:
        observed_label = tcga["death_by_12m_observed"].astype(int).astype(str) + "|" + tcga["event"].astype(str)
        train_ids, val_ids = train_test_split(
            list(tcga.index),
            test_size=cfg.test_size,
            random_state=cfg.random_seed,
            stratify=observed_label if observed_label.value_counts().min() >= 2 else None,
        )
        split_source = "generated_by_script"
    if not train_ids or not val_ids:
        raise RuntimeError("Cannot construct TCGA_train_real and TCGA_internal_validation_real.")
    train_endpoint = tcga.loc[train_ids].copy()
    val_endpoint = tcga.loc[val_ids].copy()
    pd.DataFrame({"PATIENT_ID": train_ids + val_ids, "role": ["TCGA_train_real"] * len(train_ids) + ["TCGA_internal_validation_real"] * len(val_ids)}).to_csv(
        tables_dir / "data_roles_tcga_train_internal_validation.tsv",
        sep="\t",
        index=False,
    )
    audit.add(
        "data_split",
        "OK",
        "TCGA split loaded; internal validation is locked out from feature selection, GAN and tuning.",
        source=split_source,
        train_n=len(train_ids),
        internal_validation_n=len(val_ids),
    )

    clinical_pipe, clinical_names = build_clinical_preprocessor(train_endpoint)
    clinical_all = pd.DataFrame(clinical_pipe.transform(tcga), index=tcga.index, columns=clinical_names)
    blocks = [clinical_all]
    block_map = {"clinical": clinical_names}
    joblib.dump(clinical_pipe, models_dir / "clinical_preprocessor.joblib")

    expr_path = preprocessed / "tcga_coadread_rna_log2_tpm_like_matrix.tsv"
    expression_pipe: Optional[Pipeline] = None
    expression_features: List[str] = []
    if expr_path.exists():
        expr = matrix_patients_to_rows(expr_path)
        expression_features = select_top_variance_features(expr, train_ids, cfg.max_expression_features)
        if expression_features:
            expr_scaled, expression_pipe = fit_transform_feature_matrix(expr, train_ids, list(tcga.index), expression_features)
            blocks.append(expr_scaled)
            block_map["expression_top_variance_train_only"] = expression_features
            joblib.dump(expression_pipe, models_dir / "expression_train_only_preprocessor.joblib")
            audit.add("expression", "OK", "Loaded TCGA expression and selected train-only high variance genes.", n_features=len(expression_features))
        else:
            audit.add("expression", "WARN", "Expression matrix exists but no overlapping train features were selected.")
    else:
        audit.add("expression", "WARN", "TCGA expression matrix not found; expression models will be skipped.")

    pathway_features: List[str] = []
    pathway_pipe: Optional[Pipeline] = None
    if expression_features and expr_path.exists():
        expr_train_scaled = blocks[-1] if block_map.get("expression_top_variance_train_only") else pd.DataFrame(index=tcga.index)
        pathway_raw, gmt_path = compute_pathway_activity(expr_train_scaled, cfg.raw_data_root)
        if not pathway_raw.empty:
            pathway_features = select_top_variance_features(pathway_raw, train_ids, min(100, pathway_raw.shape[1]))
            pathway_scaled, pathway_pipe = fit_transform_feature_matrix(pathway_raw, train_ids, list(tcga.index), pathway_features)
            blocks.append(pathway_scaled)
            block_map["pathway_activity"] = pathway_features
            joblib.dump(pathway_pipe, models_dir / "pathway_train_only_preprocessor.joblib")
            audit.add("pathway", "OK", "Computed pathway activity scores from local MSigDB GMT.", gmt=str(gmt_path), n_features=len(pathway_features))
        else:
            audit.add("pathway", "WARN", "No local MSigDB GMT or no pathway-gene overlap; pathway module skipped.")

    mutation_features: List[str] = []
    mut_path = preprocessed / "tcga_coadread_mutation_gene_level_matrix.tsv"
    if mut_path.exists():
        try:
            mutation_matrix = patient_feature_matrix(mut_path, prefix="MUT_")
            mutation_features = select_train_prevalent_binary_features(
                mutation_matrix,
                train_ids,
                max_features=min(80, mutation_matrix.shape[1]),
                min_prevalence=0.02,
                max_prevalence=0.80,
            )
            if mutation_features:
                mutation_binary = (
                    mutation_matrix.reindex(tcga.index)
                    .reindex(columns=mutation_features)
                    .fillna(0)
                    .gt(0)
                    .astype(float)
                )
                blocks.append(mutation_binary)
                block_map["somatic_mutation_train_prevalent"] = mutation_features
                audit.add(
                    "somatic_mutation",
                    "OK",
                    "Loaded TCGA somatic mutation matrix as the executable alternative to unavailable patient-level SNP dosage.",
                    n_features=len(mutation_features),
                )
            else:
                audit.add("somatic_mutation", "WARN", "Mutation matrix exists but no train-prevalent mutation features passed filters.")
        except Exception as exc:
            audit.add("somatic_mutation", "WARN", f"Mutation module failed: {exc}")
    else:
        audit.add("somatic_mutation", "WARN", "TCGA somatic mutation matrix not found; mutation-compatible models will be skipped.")

    X_all = pd.concat(blocks, axis=1).fillna(0.0)
    X_train_real = X_all.loc[train_ids].copy()
    X_val_real = X_all.loc[val_ids].copy()
    X_all.to_csv(tables_dir / "tcga_model_matrix_real_samples.tsv", sep="\t")
    save_json({k: list(v) for k, v in block_map.items()}, reports_dir / "feature_blocks_manifest.json")


    causal_input_cols = expression_features[: cfg.causal_prescreen_features]
    causal_table = pd.DataFrame()
    if causal_input_cols:
        causal_table = run_causal_screening(
            X_all.loc[train_ids, causal_input_cols],
            clinical_all.loc[train_ids],
            train_endpoint,
            cfg.random_seed,
            max_features=cfg.causal_prescreen_features,
        )
        if not causal_table.empty:
            causal_table.to_csv(tables_dir / "train_only_causal_priority_features.tsv", sep="\t", index=False)
            audit.add("causal_screening", "OK", "ATE/CATE/dose-response train-only causal priority table created.", n_features=len(causal_table))
        else:
            audit.add("causal_screening", "WARN", "Causal screening produced no ranked features.")
    else:
        audit.add("causal_screening", "SKIP", "No expression features available for causal priority screening.")


    eqtl_candidates = list(cfg.raw_data_root.glob("*eQTL*")) + list((cfg.raw_data_root / "GTEx_v8_eQTL").glob("*")) if (cfg.raw_data_root / "GTEx_v8_eQTL").exists() else []
    gwas_candidates = list(cfg.raw_data_root.glob("*gwas*")) + list(cfg.raw_data_root.glob("*GWAS*"))
    snp_eqtl_status = {
        "run_requested": cfg.run_snp_eqtl,
        "status": "redesigned_no_patient_level_germline_snp_required",
        "reason": "Patient-level germline SNP dosage cannot be obtained; SNP/eQTL predicted-expression modeling is excluded from the executable primary workflow.",
        "replacement_executable_branch": "clinical + somatic_mutation_train_prevalent",
        "somatic_mutation_features": mutation_features,
        "eqtl_reference_candidates": [str(p) for p in eqtl_candidates[:20]],
        "gwas_reference_candidates": [str(p) for p in gwas_candidates[:20]],
        "allowed_use_of_eqtl_gwas": "annotation_only_not_patient_level_prediction",
    }
    save_json(snp_eqtl_status, reports_dir / "snp_eqtl_module_status.json")
    audit.add("snp_eqtl_redesign", "OK", snp_eqtl_status["status"])

    causal_features = list(causal_table.head(cfg.max_causal_features)["feature"]) if not causal_table.empty else []
    model_feature_sets: Dict[str, List[str]] = {
        "clinical_only": clinical_names,
    }
    if expression_features:
        model_feature_sets["clinical_expression_topvar"] = clinical_names + expression_features
    if causal_features:
        model_feature_sets["clinical_causal_priority"] = clinical_names + causal_features
    if pathway_features:
        model_feature_sets["clinical_pathway"] = clinical_names + pathway_features
    if mutation_features:
        model_feature_sets["clinical_somatic_mutation"] = clinical_names + mutation_features

    X_train_aug_source = X_train_real.copy()
    train_endpoint_aug_source = train_endpoint.copy()
    gan_report: Dict[str, Any] = {"status": "not_requested", "n_synthetic": 0}
    if "clinical_causal_priority" in model_feature_sets:
        gan_cols = model_feature_sets["clinical_causal_priority"]
    else:
        gan_cols = model_feature_sets.get("clinical_expression_topvar", clinical_names)
    X_train_gan_aug, endpoint_gan_aug, gan_report = select_train_only_gan_augmentation(
        X_train_real[gan_cols],
        train_endpoint,
        cfg,
        audit,
    )
    save_json(gan_report, reports_dir / "train_only_gan_augmentation_report.json")

    model_rows: List[Dict[str, Any]] = []
    risk_tables: List[pd.DataFrame] = []
    fitted_models: Dict[str, Any] = {}
    thresholds: Dict[str, Optional[float]] = {}

    for feature_set_name, cols in model_feature_sets.items():
        cols = [c for c in cols if c in X_all.columns]
        if not cols:
            continue
        Xtr = X_train_real[cols]
        Xva = X_val_real[cols]
        models_to_fit: Dict[str, Any] = {}

        logit = fit_weighted_logistic(Xtr, train_endpoint, cfg.random_seed)
        models_to_fit[f"{feature_set_name}__ipcw_logistic_elasticnet"] = ("binary", logit)

        cox = fit_lifelines_cox(Xtr, train_endpoint, penalizer=0.1)
        models_to_fit[f"{feature_set_name}__cox_ph_penalized"] = ("cox_lifelines", cox)

        if len(cols) <= 2000:
            coxnet = fit_sksurv_coxnet(Xtr, train_endpoint)
            models_to_fit[f"{feature_set_name}__coxnet"] = ("sksurv", coxnet)

        rsf = fit_sksurv_rsf(Xtr, train_endpoint, cfg.random_seed)
        models_to_fit[f"{feature_set_name}__rsf"] = ("sksurv", rsf)

        if gan_report.get("status") == "fitted" and set(cols).issubset(set(gan_cols)):
            X_aug = X_train_gan_aug[cols]
            ep_aug = endpoint_gan_aug.reindex(X_aug.index)
            gan_logit = fit_weighted_logistic(X_aug, ep_aug, cfg.random_seed)
            models_to_fit[f"{feature_set_name}__train_only_gan_ipcw_logistic"] = ("binary", gan_logit)

        for model_name, (model_kind, model) in models_to_fit.items():
            if model is None:
                audit.add("model_fit", "WARN", f"{model_name} skipped or failed.")
                continue
            if model_kind == "binary":
                risk_train = risk_from_model(model, Xtr)
                risk_val = risk_from_model(model, Xva)
            elif model_kind == "cox_lifelines":
                risk_train = predict_lifelines_risk(model, Xtr, cfg.tau_months)
                risk_val = predict_lifelines_risk(model, Xva, cfg.tau_months)
            else:
                risk_train = predict_sksurv_risk(model, Xtr, cfg.tau_months)
                risk_val = predict_sksurv_risk(model, Xva, cfg.tau_months)
            threshold = choose_training_threshold(train_endpoint, risk_train)
            thresholds[model_name] = threshold
            train_metrics = evaluate_12m_predictions(train_endpoint, train_endpoint, risk_train, cfg.tau_months, threshold)
            val_metrics = evaluate_12m_predictions(train_endpoint, val_endpoint, risk_val, cfg.tau_months, threshold)
            model_rows.append(
                {
                    "model_name": model_name,
                    "feature_set": feature_set_name,
                    "model_kind": model_kind,
                    "n_features": len(cols),
                    "n_train_real": len(Xtr),
                    "n_train_synthetic_used": int(gan_report.get("n_synthetic", 0)) if "gan" in model_name else 0,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"internal_validation_{k}": v for k, v in val_metrics.items()},
                }
            )
            fitted_models[model_name] = {"model": model, "kind": model_kind, "features": cols, "threshold": threshold}
            risk_tables.append(
                pd.DataFrame(
                    {
                        "PATIENT_ID": train_ids,
                        "cohort_role": "TCGA_train_real",
                        "model_name": model_name,
                        "risk_12m": risk_train,
                    }
                )
            )
            risk_tables.append(
                pd.DataFrame(
                    {
                        "PATIENT_ID": val_ids,
                        "cohort_role": "TCGA_internal_validation_real",
                        "model_name": model_name,
                        "risk_12m": risk_val,
                    }
                )
            )
            audit.add("model_fit", "OK", f"{model_name} trained and evaluated on real internal validation.")

    if not model_rows:
        raise RuntimeError("No model could be fitted. Check dependencies and endpoint/event availability.")

    model_metrics = pd.DataFrame(model_rows)
    model_metrics.to_csv(tables_dir / "internal_validation_model_comparison.tsv", sep="\t", index=False)
    pd.concat(risk_tables, ignore_index=True).to_csv(tables_dir / "tcga_train_internal_validation_risk_scores.tsv", sep="\t", index=False)

    sort_cols = ["internal_validation_uno_cindex_ipcw", "internal_validation_time_dependent_auc_12m", "internal_validation_brier_12m_ipcw"]
    ranked = model_metrics.copy()
    ranked["_cindex_rank"] = ranked[sort_cols[0]].rank(ascending=False, na_option="bottom")
    ranked["_auc_rank"] = ranked[sort_cols[1]].rank(ascending=False, na_option="bottom")
    ranked["_brier_rank"] = ranked[sort_cols[2]].rank(ascending=True, na_option="bottom")
    ranked["_selection_score"] = ranked["_cindex_rank"] + ranked["_auc_rank"] + ranked["_brier_rank"]
    best_row = ranked.sort_values(["_selection_score", "_cindex_rank", "_brier_rank"]).iloc[0].to_dict()
    selected_model_name = str(best_row["model_name"])
    save_json(
        {
            "selected_primary_model": selected_model_name,
            "selection_basis": "TCGA_train_real decisions plus locked TCGA_internal_validation_real estimate; external cohorts not used.",
            "metrics": best_row,
            "leakage_rule": "No augmented row is present in internal or external validation.",
        },
        reports_dir / "selected_primary_model_manifest.json",
    )
    joblib.dump(fitted_models, models_dir / "all_fitted_model_bundles.joblib")
    audit.add("model_selection", "OK", f"Selected primary model: {selected_model_name}")

    external_cohorts = load_external_cohorts(preprocessed, cfg.tau_months, audit)
    external_rows: List[Dict[str, Any]] = []
    external_risk_rows: List[pd.DataFrame] = []
    external_candidate_rows: List[Dict[str, Any]] = []
    figure_rows: List[Dict[str, Any]] = []
    selected_bundle = fitted_models[selected_model_name]
    selected_features = selected_bundle["features"]
    selected_kind = selected_bundle["kind"]
    selected_model = selected_bundle["model"]
    selected_threshold = selected_bundle["threshold"]
    if selected_kind == "binary":
        selected_train_risk = risk_from_model(selected_model, X_train_real[selected_features])
        selected_val_risk = risk_from_model(selected_model, X_val_real[selected_features])
    elif selected_kind == "cox_lifelines":
        selected_train_risk = predict_lifelines_risk(selected_model, X_train_real[selected_features], cfg.tau_months)
        selected_val_risk = predict_lifelines_risk(selected_model, X_val_real[selected_features], cfg.tau_months)
    else:
        selected_train_risk = predict_sksurv_risk(selected_model, X_train_real[selected_features], cfg.tau_months)
        selected_val_risk = predict_sksurv_risk(selected_model, X_val_real[selected_features], cfg.tau_months)
    figure_rows.extend(
        generate_prediction_figures(
            train_endpoint,
            selected_train_risk,
            selected_threshold,
            "TCGA_train_real",
            selected_model_name,
            cfg.tau_months,
            figures_dir,
        )
    )
    figure_rows.extend(
        generate_prediction_figures(
            val_endpoint,
            selected_val_risk,
            selected_threshold,
            "TCGA_internal_validation_real",
            selected_model_name,
            cfg.tau_months,
            figures_dir,
        )
    )
    for cohort_name, ext_endpoint in external_cohorts.items():
        if len(ext_endpoint) < cfg.min_external_n:
            audit.add("external_validation", "SKIP", f"{cohort_name}: n<{cfg.min_external_n}.", n=len(ext_endpoint))
            continue
        X_ext, compat = build_external_feature_matrix(
            cohort_name,
            ext_endpoint,
            cfg.raw_data_root,
            clinical_pipe,
            clinical_names,
            expression_features,
            expression_pipe,
            pathway_features,
            pathway_pipe,
            mutation_features,
            audit,
        )
        needed = selected_bundle["features"]
        missing = [c for c in needed if c not in X_ext.columns]
        if missing:

            if all(c in clinical_names for c in needed):
                pass
            else:
                audit.add("external_validation", "SKIP", f"{cohort_name}: missing selected model features.", missing_count=len(missing))
                continue
        X_eval = X_ext.reindex(columns=needed, fill_value=0.0)
        kind = selected_bundle["kind"]
        model = selected_bundle["model"]
        if kind == "binary":
            risk = risk_from_model(model, X_eval)
        elif kind == "cox_lifelines":
            risk = predict_lifelines_risk(model, X_eval, cfg.tau_months)
        else:
            risk = predict_sksurv_risk(model, X_eval, cfg.tau_months)
        metrics = evaluate_12m_predictions(train_endpoint, ext_endpoint, risk, cfg.tau_months, selected_bundle["threshold"])
        external_rows.append(
            {
                "cohort": cohort_name,
                "model_name": selected_model_name,
                "compatibility": json.dumps(compat, ensure_ascii=False),
                **metrics,
            }
        )
        external_risk_rows.append(
            pd.DataFrame(
                {
                    "PATIENT_ID": list(ext_endpoint.index),
                    "cohort": cohort_name,
                    "model_name": selected_model_name,
                    "risk_12m": risk,
                }
            )
        )
        figure_rows.extend(
            generate_prediction_figures(
                ext_endpoint,
                risk,
                selected_bundle["threshold"],
                cohort_name,
                selected_model_name,
                cfg.tau_months,
                figures_dir,
            )
        )
        audit.add("external_validation", "OK", f"{cohort_name}: real-world locked-model validation complete.")

        for candidate_name, candidate_bundle in fitted_models.items():
            candidate_features = candidate_bundle["features"]
            candidate_missing = [c for c in candidate_features if c not in X_ext.columns]
            candidate_row: Dict[str, Any] = {
                "cohort": cohort_name,
                "model_name": candidate_name,
                "is_selected_primary_model": candidate_name == selected_model_name,
                "n_required_features": len(candidate_features),
                "n_missing_features": len(candidate_missing),
                "compatible_for_external_endpoint": not candidate_missing,
                "compatibility": json.dumps(compat, ensure_ascii=False),
                "external_use_note": "diagnostic_only_not_used_for_model_selection",
            }
            if candidate_missing:
                candidate_row["skip_reason"] = "missing_locked_model_features"
                external_candidate_rows.append(candidate_row)
                continue
            candidate_kind = candidate_bundle["kind"]
            candidate_model = candidate_bundle["model"]
            X_candidate = X_ext.reindex(columns=candidate_features)
            if candidate_kind == "binary":
                candidate_risk = risk_from_model(candidate_model, X_candidate)
            elif candidate_kind == "cox_lifelines":
                candidate_risk = predict_lifelines_risk(candidate_model, X_candidate, cfg.tau_months)
            else:
                candidate_risk = predict_sksurv_risk(candidate_model, X_candidate, cfg.tau_months)
            candidate_metrics = evaluate_12m_predictions(
                train_endpoint,
                ext_endpoint,
                candidate_risk,
                cfg.tau_months,
                candidate_bundle["threshold"],
            )
            candidate_row.update(candidate_metrics)
            external_candidate_rows.append(candidate_row)

    external_metrics = pd.DataFrame(external_rows)
    external_metrics.to_csv(tables_dir / "external_real_world_validation_metrics.tsv", sep="\t", index=False)
    external_candidate_metrics = pd.DataFrame(external_candidate_rows)
    external_candidate_metrics.to_csv(tables_dir / "external_locked_candidate_model_diagnostics.tsv", sep="\t", index=False)
    pd.DataFrame(figure_rows).to_csv(tables_dir / "figure_manifest.tsv", sep="\t", index=False)
    audit.add("figures", "OK", "Wrote time-dependent ROC, calibration and K-M figure manifest.", n=len(figure_rows))
    if external_risk_rows:
        pd.concat(external_risk_rows, ignore_index=True).to_csv(tables_dir / "external_real_world_risk_scores.tsv", sep="\t", index=False)

    external_generalization = {
        "selected_primary_model": selected_model_name,
        "diagnostic_table": str(tables_dir / "external_locked_candidate_model_diagnostics.tsv"),
        "external_outcomes_not_used_for_selection": True,
        "selected_model_external_mean_uno_cindex": (
            float(external_metrics["uno_cindex_ipcw"].dropna().mean())
            if not external_metrics.empty and "uno_cindex_ipcw" in external_metrics
            else None
        ),
        "selected_model_external_mean_auc_12m": (
            float(external_metrics["time_dependent_auc_12m"].dropna().mean())
            if not external_metrics.empty and "time_dependent_auc_12m" in external_metrics
            else None
        ),
        "publishability_note": (
            "Internal validation is necessary but insufficient; external diagnostic performance must be interpreted "
            "as generalization evidence and cannot be used to tune or reselect the primary model."
        ),
    }
    save_json(external_generalization, reports_dir / "external_generalization_diagnostic_summary.json")

    leakage = {
        "tcga_train_real_n": len(train_ids),
        "tcga_internal_validation_real_n": len(val_ids),
        "intersection_train_internal": sorted(set(train_ids).intersection(val_ids)),
        "gan_status": gan_report,
        "external_cohorts_evaluated": list(external_metrics["cohort"]) if not external_metrics.empty else [],
        "external_candidate_diagnostics": str(tables_dir / "external_locked_candidate_model_diagnostics.tsv"),
        "rules": [
            "GAN fitted only on TCGA_train_real.",
            "GAN synthetic rows are not used in internal validation or external validation.",
            "TCGA_internal_validation_real is not used for feature selection, GAN choice, thresholds, or hyperparameter search.",
            "External real-world cohorts are only used after model locking.",
            "Early censoring before 12 months is not labelled as 1-year survivor.",
        ],
    }
    save_json(leakage, reports_dir / "leakage_audit.json")
    audit.add("leakage_audit", "OK", "Wrote leakage audit manifest.")

    audit.to_frame().to_csv(tables_dir / "pipeline_audit_log.tsv", sep="\t", index=False)
    return {
        "run_dir": str(run_dir),
        "selected_primary_model": selected_model_name,
        "internal_metrics_path": str(tables_dir / "internal_validation_model_comparison.tsv"),
        "external_metrics_path": str(tables_dir / "external_real_world_validation_metrics.tsv"),
        "figure_manifest_path": str(tables_dir / "figure_manifest.tsv"),
        "environment_path": str(env_path),
        "audit_path": str(tables_dir / "pipeline_audit_log.tsv"),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    default_raw = Path(r"D:\work-课题\rawData")
    parser = argparse.ArgumentParser(description="CRC 1-year OS risk prediction pipeline.")
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--raw-data-root", type=Path, default=default_raw)
    parser.add_argument("--plan-path", type=Path, default=project_root / "plan" / "headplan.txt")
    parser.add_argument("--output-root", type=Path, default=project_root / "results")
    parser.add_argument("--max-expression-features", type=int, default=300)
    parser.add_argument("--causal-prescreen-features", type=int, default=120)
    parser.add_argument("--max-causal-features", type=int, default=50)
    parser.add_argument("--gan-epochs", type=int, default=60)
    parser.add_argument("--gan-aug-ratio", type=float, default=0.5)
    parser.add_argument("--gan-aug-ratios", type=str, default="0.25,0.5,1.0,2.0")
    parser.add_argument("--gan-sampling-strategies", type=str, default="balanced_event,risk_stratified,event_only,overall")
    parser.add_argument("--no-gan", action="store_true")
    parser.add_argument("--no-snp-eqtl", action="store_true")
    parser.add_argument("--strict-dependencies", action="store_true")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    gan_aug_ratio_candidates = tuple(
        float(x.strip())
        for x in str(args.gan_aug_ratios).split(",")
        if x.strip()
    )
    gan_sampling_strategies = tuple(
        x.strip().lower()
        for x in str(args.gan_sampling_strategies).split(",")
        if x.strip()
    )
    return PipelineConfig(
        project_root=args.project_root.resolve(),
        raw_data_root=args.raw_data_root.resolve(),
        plan_path=args.plan_path.resolve(),
        output_root=args.output_root.resolve(),
        random_seed=args.seed,
        max_expression_features=args.max_expression_features,
        causal_prescreen_features=args.causal_prescreen_features,
        max_causal_features=args.max_causal_features,
        gan_epochs=args.gan_epochs,
        gan_aug_ratio=args.gan_aug_ratio,
        gan_aug_ratio_candidates=gan_aug_ratio_candidates,
        gan_sampling_strategies=gan_sampling_strategies,
        run_gan=not args.no_gan,
        run_snp_eqtl=not args.no_snp_eqtl,
        strict_dependencies=args.strict_dependencies,
    )


class TeeStream:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return bool(self.streams and getattr(self.streams[0], "isatty", lambda: False)())


def make_console_log_path(output_root: Path) -> Path:
    log_dir = ensure_dir(output_root.resolve() / "console_logs")
    return log_dir / f"{Path(__file__).stem}_{now_stamp()}.console.log"


def output_root_from_argv(argv: Optional[Sequence[str]]) -> Path:
    tokens = list(sys.argv[1:] if argv is None else argv)
    for i, token in enumerate(tokens):
        if token == "--output-root" and i + 1 < len(tokens):
            return Path(tokens[i + 1])
        prefix = "--output-root="
        if token.startswith(prefix):
            return Path(token[len(prefix):])
    return Path(__file__).resolve().parents[1] / "results"


def main(argv: Optional[Sequence[str]] = None) -> int:
    console_log_path = make_console_log_path(output_root_from_argv(argv))
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with console_log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            print(f"[OK] console_log: Mirroring console output to {console_log_path}", flush=True)
            warnings.filterwarnings("ignore", category=UserWarning)
            parser = build_arg_parser()
            args = parser.parse_args(argv)
            cfg = config_from_args(args)
            try:
                summary = run_pipeline(cfg)
            except Exception as exc:
                print(f"[FAIL] pipeline: {exc}", file=sys.stderr)
                traceback.print_exc()
                return 1
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    raise SystemExit(main())
