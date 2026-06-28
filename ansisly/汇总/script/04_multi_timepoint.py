from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test
except Exception:
    CoxPHFitter = None
    KaplanMeierFitter = None
    logrank_test = None

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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="lifelines")

from shared_utils import (
    setup_logger,
    SCRIPT_DIR, DATA_DIR, RESULTS_DIR,
    RANDOM_SEED, EPS, DEFAULT_TIMEPOINTS, FIXED_TAU_MONTHS,
    ensure_dir, safe_read_tsv, sanitize_filename, normalize_patient_id,
    kaplan_meier_survival_at, km_cumulative_incidence_by_tau,
    detect_event_time_columns, coerce_event,
    add_fixed_endpoint, compute_ipcw_weights, compute_pseudo_observations,
    rank_inverse_normal,
)
logger = setup_logger("04_multi_timepoint")

# ──────────────────────────────────────────────────────────────────────
# Task 1: 动态导入 02_gene_features 和 03_model_training 的函数
# 使每个端点能独立执行特征筛选（二分类筛选 → 因果筛选 → 特征融合）
# ──────────────────────────────────────────────────────────────────────
from importlib import import_module

_SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR_PATH not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR_PATH)

try:
    _mod02 = import_module("02_gene_features")
    train_variance_screen = _mod02.train_variance_screen
    univariate_cox = _mod02.univariate_cox
    multivariable_cox = _mod02.multivariable_cox
    run_causal_screening = _mod02.run_causal_screening
    encode_ajcc_stage = _mod02.encode_ajcc_stage
    is_likely_pseudogene = _mod02.is_likely_pseudogene
    bh_fdr = _mod02.bh_fdr
except Exception as _e02:
    logger.warning(f"[04] 02_gene_features 动态导入失败: {_e02}")
    train_variance_screen = None
    univariate_cox = None
    multivariable_cox = None
    run_causal_screening = None
    encode_ajcc_stage = None
    is_likely_pseudogene = None
    bh_fdr = None

try:
    _mod03 = import_module("03_model_training")
    SurvivalConditionalTabularGAN = _mod03.SurvivalConditionalTabularGAN
    build_survival_gan_conditions = _mod03.build_survival_gan_conditions
    HAS_TORCH = _mod03.HAS_TORCH
except Exception as _e03:
    logger.warning(f"[04] 03_model_training 动态导入失败: {_e03}")
    SurvivalConditionalTabularGAN = None
    build_survival_gan_conditions = None
    HAS_TORCH = False


# backward-compatible aliases
read_tsv = safe_read_tsv
TIMEPOINTS = DEFAULT_TIMEPOINTS


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def make_surv_array(endpoint):
    """Local wrapper: build Surv array from endpoint DataFrame."""
    if Surv is None:
        return None
    return Surv.from_arrays(
        endpoint["event"].astype(bool).to_numpy(),
        endpoint["time_months"].to_numpy(float),
    )


def add_unlimited_endpoint(df, patient_col="PATIENT_ID"):
    result = df.copy()
    time_col, event_col = detect_event_time_columns(result)
    result[patient_col] = result[patient_col].map(normalize_patient_id)
    result["time_months"] = safe_numeric(result[time_col])
    result["event"] = coerce_event(result[event_col])
    result = result[result[patient_col].ne("") & result["time_months"].notna()].copy()
    result = result[result["time_months"] >= 0].copy()
    result["os_observed"] = True
    result["os_event"] = result["event"].astype(int)
    result["os_censored"] = result["event"].eq(0)
    time_rank = result["time_months"].rank(method="average", pct=True).fillna(0.5)
    event_f = result["event"].astype(float)
    result["os_event_risk_score"] = (event_f * (1.0 - 0.5 * time_rank) + (1.0 - event_f) * 0.25 * (1.0 - time_rank)).clip(0.0, 1.0)
    return result


def save_current_figure(path_base):
    paths = []
    for suffix, dpi in [(".png", 300), (".tiff", 600)]:
        path = Path(str(path_base) + suffix)
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(str(path))
    plt.close()
    return paths


def uno_cindex_fallback(endpoint, risk):
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


def censor_safe_binary_training(endpoint, tau_int):
    mask = endpoint["ipcw_label_available"].astype(bool)
    y = endpoint.loc[mask, f"death_by_{tau_int}m"].astype(int)
    w = endpoint.loc[mask, f"ipcw_weight_{tau_int}m"].astype(float)
    return y, w


def fit_timepoint_logistic(X_train, endpoint_train, tau_int, random_seed):
    """Fits IPCW-weighted logistic regression with stratified CV hyperparameter selection."""
    if X_train.empty or len(X_train) < 10:
        return None
    y, w = censor_safe_binary_training(endpoint_train, tau_int)
    if y.nunique() < 2 or len(y) < 20:
        return None
    n_folds = min(5, max(2, int(y.value_counts().min())))
    # TODO: sklearn>=1.8 需迁移 penalty="elasticnet" 到 LogisticRegression+ElasticNet
    model = LogisticRegressionCV(
        Cs=10, cv=n_folds,
        penalty="elasticnet", solver="saga", l1_ratios=[0.05, 0.5, 0.95],
        scoring="roc_auc", max_iter=5000, random_state=random_seed, n_jobs=-1,
    )
    model.fit(X_train.loc[y.index], y, sample_weight=w)
    return model


def evaluate_with_cv(X_train, endpoint_train, tau_int, random_seed, n_folds=5):
    """Stratified CV evaluation returning out-of-fold AUC for each model type.

    Returns dict of {model_name: {"cv_auc_mean": float, "cv_auc_std": float}}.
    """
    from sklearn.model_selection import StratifiedKFold

    y, w = censor_safe_binary_training(endpoint_train, tau_int)
    if y.nunique() < 2 or len(y) < 30:
        return {}

    n_folds = min(n_folds, int(y.value_counts().min()))
    if n_folds < 2:
        return {}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    results = {}

    # Logistic regression CV scores
    log_aucs = []
    for train_idx, test_idx in skf.split(X_train.loc[y.index], y):
        fold_X_train = X_train.iloc[train_idx]
        fold_X_test = X_train.iloc[test_idx]
        fold_y = y.iloc[train_idx]
        fold_w = w[train_idx]
        fold_y_test = y.iloc[test_idx]
        if fold_y.nunique() < 2:
            continue
        # TODO: sklearn>=1.8 需迁移 penalty="elasticnet"
        fold_model = LogisticRegressionCV(
            Cs=5, cv=3, penalty="elasticnet", solver="saga",
            l1_ratios=[0.05, 0.5, 0.95], scoring="roc_auc",
            max_iter=5000, random_state=random_seed,
        )
        fold_model.fit(fold_X_train, fold_y, sample_weight=fold_w)
        risk_test = fold_model.predict_proba(fold_X_test)[:, 1]
        try:
            auc = roc_auc_score(fold_y_test, risk_test)
            log_aucs.append(auc)
        except Exception:
            pass
    if log_aucs:
        results["IPCW_Logistic"] = {
            "cv_auc_mean": float(np.mean(log_aucs)),
            "cv_auc_std": float(np.std(log_aucs)),
        }

    # RSF CV scores
    rsf_aucs = []
    for train_idx, test_idx in skf.split(X_train.loc[y.index], y):
        fold_X_train = X_train.iloc[train_idx]
        fold_X_test = X_train.iloc[test_idx]
        fold_ep = endpoint_train.iloc[train_idx]
        fold_y_test = y.iloc[test_idx]
        fold_rsf = fit_timepoint_rsf(fold_X_train, fold_ep, random_seed)
        if fold_rsf is not None:
            try:
                risk_test = fold_rsf.predict(fold_X_test.to_numpy(float))
                auc = roc_auc_score(fold_y_test, -risk_test)
                rsf_aucs.append(auc)
            except Exception:
                pass
    if rsf_aucs:
        results["RSF"] = {
            "cv_auc_mean": float(np.mean(rsf_aucs)),
            "cv_auc_std": float(np.std(rsf_aucs)),
        }

    return results


def fit_timepoint_cox(X_train, endpoint_train, penalizer=0.5):
    if CoxPHFitter is None:
        return None
    if X_train.empty or len(X_train) < 10:
        return None
    data = X_train.copy()
    data["time_months"] = endpoint_train["time_months"].to_numpy(float)
    data["event"] = endpoint_train["event"].to_numpy(int)
    usable = data.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    # 移除近零方差列，防止完全分离导致 Newton-Raphson 不收敛
    usable = usable.loc[:, usable.var() > 1e-6]
    if "event" not in usable.columns or usable["event"].sum() < 5:
        return None
    try:
        model = CoxPHFitter(penalizer=penalizer)
        model.fit(usable, duration_col="time_months", event_col="event", show_progress=False)
        return model
    except Exception:
        return None


def fit_timepoint_coxnet(X_train, endpoint_train):
    if CoxnetSurvivalAnalysis is None or Surv is None:
        return None
    if X_train.empty or endpoint_train["event"].sum() < 5:
        return None
    y = make_surv_array(endpoint_train)
    try:
        model = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, n_alphas=60, max_iter=100000)
        model.fit(X_train.to_numpy(float), y)
        return model
    except Exception:
        return None


def fit_timepoint_rsf(X_train, endpoint_train, random_seed):
    if RandomSurvivalForest is None or Surv is None:
        return None
    if X_train.empty or endpoint_train["event"].sum() < 5:
        return None
    y = make_surv_array(endpoint_train)
    try:
        model = RandomSurvivalForest(
            n_estimators=300, min_samples_split=10, min_samples_leaf=8,
            max_features="sqrt", n_jobs=-1, random_state=random_seed,
        )
        model.fit(X_train.to_numpy(float), y)
        return model
    except Exception:
        return None


def predict_logistic_risk(model, X):
    if model is None:
        return np.full(len(X), np.nan)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return np.asarray(model.predict(X), dtype=float)


def predict_cox_risk(model, X, tau):
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


def predict_cox_risk_unlimited(model, X):
    if model is None:
        return np.full(len(X), np.nan)
    try:
        cols = [c for c in model.params_.index if c in X.columns]
        Xp = X.reindex(columns=cols, fill_value=0.0)
        return np.asarray(model.predict_partial_hazard(Xp), dtype=float).ravel()
    except Exception:
        return np.full(len(X), np.nan)


def predict_sksurv_risk(model, X, tau):
    if model is None:
        return np.full(len(X), np.nan)
    try:
        surv_fns = model.predict_survival_function(X.to_numpy(float))
        return np.asarray([1.0 - float(fn(tau)) for fn in surv_fns], dtype=float)
    except Exception:
        try:
            return np.asarray(model.predict(X.to_numpy(float)), dtype=float)
        except Exception:
            return np.full(len(X), np.nan)


def predict_sksurv_risk_unlimited(model, X):
    if model is None:
        return np.full(len(X), np.nan)
    try:
        return np.asarray(model.predict(X.to_numpy(float)), dtype=float)
    except Exception:
        return np.full(len(X), np.nan)


def predict_survival_matrix(model, model_kind, x, times):
    if model is None or len(times) == 0:
        return None
    try:
        if model_kind == "cox_lifelines":
            cols = [c for c in model.params_.index if c in x.columns]
            xp = x.reindex(columns=cols, fill_value=0.0)
            sf = model.predict_survival_function(xp, times=times)
            return sf.T.to_numpy(float)
        surv_fns = model.predict_survival_function(x.to_numpy(float))
        return np.asarray([[float(fn(t)) for t in times] for fn in surv_fns], dtype=float)
    except Exception:
        return None


def choose_evaluation_times(train_endpoint, eval_endpoint):
    train_times = train_endpoint["time_months"].astype(float)
    eval_times = eval_endpoint["time_months"].astype(float)
    lower = max(float(train_times.min()), float(eval_times.min()), EPS)
    upper = min(float(train_times.max()), float(eval_times.max()))
    if upper <= lower:
        return np.asarray([], dtype=float)
    quantiles = np.linspace(0.15, 0.85, 8)
    times = np.quantile(eval_times.clip(lower=lower, upper=upper), quantiles)
    times = np.unique(times[(times > lower) & (times < upper)])
    return times.astype(float)


def evaluate_fixed_tau(train_endpoint, eval_endpoint, risk, tau, threshold=None):
    tau_int = int(tau)
    result = {
        "n": int(len(eval_endpoint)),
        "events": int(eval_endpoint["event"].sum()),
        f"early_censored_before_{tau_int}m": int(eval_endpoint[f"early_censored_before_{tau_int}m"].sum()),
    }
    valid_risk = np.isfinite(risk)
    observed = eval_endpoint[f"death_by_{tau_int}m_observed"].to_numpy(bool) & valid_risk
    if observed.sum() >= 5 and pd.Series(eval_endpoint.loc[observed, f"death_by_{tau_int}m"]).nunique() == 2:
        y = eval_endpoint.loc[observed, f"death_by_{tau_int}m"].astype(int).to_numpy()
        scores = risk[observed]
        weights = eval_endpoint.loc[observed, f"ipcw_weight_{tau_int}m"].replace(0, np.nan).fillna(1.0).to_numpy(float)
        try:
            result["ipcw_auc"] = float(roc_auc_score(y, scores, sample_weight=weights))
        except Exception:
            result["ipcw_auc"] = float("nan")
        try:
            result["ipcw_brier"] = float(np.average((y - scores) ** 2, weights=weights))
        except Exception:
            result["ipcw_brier"] = float("nan")
        try:
            result["average_precision"] = float(average_precision_score(y, scores, sample_weight=weights))
        except Exception:
            result["average_precision"] = float("nan")
        if threshold is not None:
            pred = (scores >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            result.update({"threshold": float(threshold), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)})
    else:
        result.update({"ipcw_auc": float("nan"), "ipcw_brier": float("nan"), "average_precision": float("nan")})
    result["harrell_cindex"] = float(uno_cindex_fallback(eval_endpoint, risk))
    if (concordance_index_ipcw is not None and Surv is not None
            and eval_endpoint["event"].sum() >= 3 and train_endpoint["event"].sum() >= 3):
        try:
            y_tr = make_surv_array(train_endpoint)
            y_ev = make_surv_array(eval_endpoint)
            result["uno_cindex"] = float(concordance_index_ipcw(y_tr, y_ev, risk, tau=tau)[0])
        except Exception:
            result["uno_cindex"] = float("nan")
        try:
            y_tr2 = make_surv_array(train_endpoint)
            y_ev2 = make_surv_array(eval_endpoint)
            auc_t, _ = cumulative_dynamic_auc(y_tr2, y_ev2, risk, times=np.asarray([tau]))
            result["time_dependent_auc"] = float(auc_t[0])
        except Exception:
            result["time_dependent_auc"] = float("nan")
    else:
        result["uno_cindex"] = float("nan")
        result["time_dependent_auc"] = float("nan")
    return result


def evaluate_unlimited(train_endpoint, eval_endpoint, risk, model=None, model_kind=None, x_eval=None, threshold=None):
    result = {
        "n": int(len(eval_endpoint)),
        "events": int(eval_endpoint["event"].sum()),
        "censored": int(eval_endpoint["event"].eq(0).sum()),
        "median_followup_months": float(eval_endpoint["time_months"].median()),
    }
    result["harrell_cindex"] = float(uno_cindex_fallback(eval_endpoint, risk))
    if concordance_index_ipcw is not None and Surv is not None:
        try:
            y_tr = make_surv_array(train_endpoint)
            y_ev = make_surv_array(eval_endpoint)
            result["uno_cindex"] = float(concordance_index_ipcw(y_tr, y_ev, risk)[0])
        except Exception:
            result["uno_cindex"] = float("nan")
    else:
        result["uno_cindex"] = float("nan")
    result["integrated_brier_score"] = float("nan")
    result["event_brier_score"] = float("nan")
    if (model is not None and model_kind is not None and x_eval is not None
            and integrated_brier_score is not None and Surv is not None):
        times = choose_evaluation_times(train_endpoint, eval_endpoint)
        surv = predict_survival_matrix(model, model_kind, x_eval, times)
        if surv is not None and surv.shape == (len(eval_endpoint), len(times)):
            try:
                y_tr = make_surv_array(train_endpoint)
                y_ev = make_surv_array(eval_endpoint)
                result["integrated_brier_score"] = float(integrated_brier_score(y_tr, y_ev, surv, times))
            except Exception:
                pass
    event = eval_endpoint["event"].astype(int).to_numpy()
    valid = np.isfinite(risk)
    if valid.sum() >= 5 and len(np.unique(event[valid])) == 2:
        try:
            scaled = pd.Series(risk[valid]).rank(method="average", pct=True).to_numpy(float)
            result["event_brier_score"] = float(brier_score_loss(event[valid], scaled))
        except Exception:
            pass
    if threshold is not None and valid.sum() >= 10:
        high = risk[valid] >= threshold
        ev = event[valid]
        result["threshold"] = float(threshold)
        result["high_risk_n"] = int(high.sum())
        result["low_risk_n"] = int((~high).sum())
        result["high_risk_events"] = int(ev[high].sum()) if high.any() else 0
        result["low_risk_events"] = int(ev[~high].sum()) if (~high).any() else 0
    return result


def observed_binary_for_plot(endpoint, risk, tau_int):
    valid = endpoint[f"death_by_{tau_int}m_observed"].to_numpy(bool) & np.isfinite(risk)
    if valid.sum() == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=float), np.asarray([], dtype=float)
    y = endpoint.loc[valid, f"death_by_{tau_int}m"].astype(int).to_numpy()
    scores = np.asarray(risk[valid], dtype=float)
    weights = endpoint.loc[valid, f"ipcw_weight_{tau_int}m"].replace(0, np.nan).fillna(1.0).to_numpy(float)
    return y, scores, weights


def plot_km_risk_strata(endpoint, risk, threshold, tau, title, path_base, is_unlimited=False):
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
        kmf.fit(ep.loc[mask, "time_months"], event_observed=ep.loc[mask, "event"],
                label=f"{label} (n={int(mask.sum())})")
        kmf.plot_survival_function(ci_show=True, color=colors[label], lw=2.0)
    p_text = ""
    if logrank_test is not None:
        try:
            low = strata == "Low risk"
            high = strata == "High risk"
            lr = logrank_test(ep.loc[low, "time_months"], ep.loc[high, "time_months"],
                              event_observed_A=ep.loc[low, "event"], event_observed_B=ep.loc[high, "event"])
            p_text = f"Log-rank p = {lr.p_value:.3g}"
        except Exception:
            pass
    if not is_unlimited and tau is not None:
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


def plot_time_dependent_roc(endpoint, risk, tau, title, path_base, tau_int):
    y, scores, weights = observed_binary_for_plot(endpoint, risk, tau_int)
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


def plot_calibration_curve(endpoint, risk, tau, title, path_base, tau_int, n_bins=5):
    y, scores, weights = observed_binary_for_plot(endpoint, risk, tau_int)
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
        rows.append({"predicted": float(np.average(group["risk"], weights=w)),
                     "observed": float(np.average(group["y"], weights=w)), "n": int(len(group))})
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


def plot_risk_distribution(endpoint, risk, title, path_base):
    valid = np.isfinite(risk)
    if valid.sum() < 10:
        return None
    df = pd.DataFrame({"risk": np.asarray(risk[valid], dtype=float),
                       "status": np.where(endpoint.loc[valid, "event"].astype(int).to_numpy() == 1, "Death", "Censored")})
    if df["status"].nunique() < 2:
        return None
    plt.figure(figsize=(5.6, 4.2))
    groups = [df.loc[df["status"].eq(label), "risk"].to_numpy(float) for label in ["Censored", "Death"]]
    plt.boxplot(groups, tick_labels=["Censored", "Death"], patch_artist=True)
    jitter_rng = np.random.default_rng(RANDOM_SEED)
    for i, vals in enumerate(groups, start=1):
        x = jitter_rng.normal(i, 0.035, size=len(vals))
        plt.scatter(x, vals, s=12, alpha=0.45)
    plt.ylabel("Predicted OS risk score")
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    return save_current_figure(path_base)


def plot_followup_vs_risk(endpoint, risk, title, path_base):
    valid = np.isfinite(risk) & endpoint["time_months"].notna().to_numpy()
    if valid.sum() < 10:
        return None
    ep = endpoint.loc[valid]
    scores = np.asarray(risk[valid], dtype=float)
    colors = np.where(ep["event"].astype(int).to_numpy() == 1, "#d62728", "#1f77b4")
    plt.figure(figsize=(5.8, 4.4))
    plt.scatter(scores, ep["time_months"].to_numpy(float), c=colors, s=18, alpha=0.72)
    plt.xlabel("Predicted OS risk score")
    plt.ylabel("Observed follow-up time (months)")
    plt.title(title)
    plt.grid(alpha=0.25)
    return save_current_figure(path_base)


def choose_training_threshold(endpoint_train, risk_train, tau_int):
    # Fix P0: 对齐长度——增强后 risk_train 可能比 endpoint_train 长
    if isinstance(risk_train, np.ndarray) and len(risk_train) != len(endpoint_train):
        min_len = min(len(endpoint_train), len(risk_train))
        endpoint_train = endpoint_train.iloc[:min_len]
        risk_train = risk_train[:min_len]
    elif hasattr(risk_train, 'index') and len(risk_train) != len(endpoint_train):
        common = endpoint_train.index.intersection(risk_train.index)
        endpoint_train = endpoint_train.loc[common]
        risk_train = risk_train.loc[common]
    valid = endpoint_train[f"death_by_{tau_int}m_observed"].to_numpy(bool) & np.isfinite(risk_train)
    if valid.sum() < 10:
        return None
    y = endpoint_train.loc[valid, f"death_by_{tau_int}m"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return None
    scores = risk_train[valid]
    fpr, tpr, thresholds = roc_curve(y, scores)
    idx = int(np.nanargmax(tpr - fpr))
    return float(thresholds[idx])


def choose_risk_threshold_unlimited(risk):
    valid = risk[np.isfinite(risk)]
    if len(valid) < 10:
        return None
    return float(np.nanmedian(valid))


def load_gmt(path, min_size=5, max_size=500):
    """Load MSigDB GMT file into {name: set_of_genes}."""
    gene_sets = {}
    p = os.path.abspath(path)
    if not os.path.exists(p):
        return gene_sets
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = {g.strip().upper() for g in parts[2:] if g.strip()}
            if min_size <= len(genes) <= max_size:
                gene_sets[name] = genes
    return gene_sets


def compute_pathway_activity(expr_df, gene_sets, min_overlap=3):
    """Compute pathway activity scores as mean gene expression per pathway.

    Args:
        expr_df: DataFrame (patients × genes), gene columns should be uppercase.
        gene_sets: dict of {pathway_name: set_of_genes}.
        min_overlap: minimum number of overlapping genes to keep a pathway.

    Returns:
        DataFrame (patients × pathways) with activity scores.
    """
    gene_columns = {str(c).upper(): c for c in expr_df.columns}
    scores = {}
    for name, gene_set in gene_sets.items():
        overlap = [gene_columns[g] for g in gene_set if g in gene_columns]
        if len(overlap) < min_overlap:
            continue
        scores[name] = expr_df[overlap].mean(axis=1, skipna=True)
    if not scores:
        return pd.DataFrame(index=expr_df.index)
    return pd.DataFrame(scores, index=expr_df.index)


def compute_pathway_features(expr_df):
    """Compute MSigDB pathway activity scores from gene expression matrix.

    Loads hallmark and KEGG pathway GMT files, computes mean expression
    per pathway as activity scores.

    Args:
        expr_df: DataFrame (patients x genes), gene columns should be uppercase.

    Returns:
        DataFrame (patients x pathways) with activity scores, columns prefixed with 'PW_'.
    """
    pathway_df = pd.DataFrame(index=expr_df.index)
    msigdb_dir = os.path.join(DATA_DIR, "msigdb")
    for gmt_file in ["h.all.v2024.1.Hs.symbols.gmt", "c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt"]:
        gmt_path = os.path.join(msigdb_dir, gmt_file)
        if os.path.exists(gmt_path):
            gene_sets = load_gmt(gmt_path)
            pa = compute_pathway_activity(expr_df, gene_sets)
            if not pa.empty:
                pa.columns = [f"PW_{c}" for c in pa.columns]
                pathway_df = pd.concat([pathway_df, pa], axis=1)
    return pathway_df


# Continuous variables that benefit from RINT transformation
_RINT_COLUMNS = {
    "AGE", "AGE_AT_DIAGNOSIS", "ANEUPLOIDY_SCORE",
    "TMB_NONSYNONYMOUS", "MSI_SCORE_MANTIS", "MSI_SENSOR_SCORE",
    "TBL_SCORE", "SOMATIC_STATUS",
}

# Categorical variables that should be one-hot encoded
_ONEHOT_COLUMNS = {
    "SEX", "RACE", "AJCC_PATHOLOGIC_TUMOR_STAGE",
    "PATH_T_STAGE", "PATH_N_STAGE", "PATH_M_STAGE",
    "RADIATION_THERAPY",
    "HISTORY_NEOADJUVANT_TRTYN", "PERSON_NEOPLASM_CANCER_STATUS",
    "NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT",
}


def build_clinical_features(clinical_df):
    """Extract numeric clinical features with RINT + OneHot + Imputer + Scaler.

    Pipeline:
      1. Continuous vars → RINT (rank inverse normal transform)
      2. Categorical vars → OneHot encoding
      3. Remaining numeric vars → kept as-is
      4. SimpleImputer (median) + StandardScaler

    Returns a fully numeric, scaled DataFrame indexed by PATIENT_ID.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    exclude = {
        "PATIENT_ID", "OS_MONTHS", "OS_EVENT", "OS_STATUS",
        "DSS_MONTHS", "DSS_STATUS", "PFS_MONTHS", "PFS_STATUS",
        "DAYS_LAST_FOLLOWUP", "DAYS_TO_BIRTH", "DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS",
        "death_by_36m_observed", "death_by_36m", "early_censored_before_36m",
        "ipcw_weight_os", "ipcw_label_available",
        "pseudo_risk_os_raw", "pseudo_risk_os",
        "OTHER_PATIENT_ID", "SAMPLE_ID",
    }
    work = clinical_df.copy()
    work["PATIENT_ID"] = work["PATIENT_ID"].astype(str)

    # --- Step 1: Classify columns ---
    rint_cols, onehot_cols, numeric_cols = [], [], []

    # Known continuous columns
    for col in _RINT_COLUMNS:
        if col in work.columns and col not in exclude:
            work[col] = pd.to_numeric(work[col], errors="coerce")
            if work[col].notna().sum() > len(work) * 0.3:
                rint_cols.append(col)

    # Known categorical columns
    for col in _ONEHOT_COLUMNS:
        if col in work.columns and col not in exclude:
            nunique = work[col].nunique()
            if 2 <= nunique <= 20:
                onehot_cols.append(col)

    # Remaining numeric columns
    for col in work.columns:
        if col in exclude or col in rint_cols or col in onehot_cols:
            continue
        s = pd.to_numeric(work[col], errors="coerce")
        if s.notna().sum() > len(work) * 0.5 and s.nunique() > 1:
            work[col] = s
            numeric_cols.append(col)

    if not rint_cols and not onehot_cols and not numeric_cols:
        return pd.DataFrame(index=work["PATIENT_ID"].values)

    # --- Step 2: RINT on continuous columns ---
    rint_df = pd.DataFrame(index=work.index)
    for col in rint_cols:
        rint_df[col] = rank_inverse_normal(work[col])

    # --- Step 3: OneHot on categorical columns ---
    onehot_df = pd.DataFrame(index=work.index)
    if onehot_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=float)
        cat_data = work[onehot_cols].fillna("MISSING").astype(str)
        encoded = encoder.fit_transform(cat_data)
        feature_names = encoder.get_feature_names_out(onehot_cols)
        onehot_df = pd.DataFrame(encoded, columns=feature_names, index=work.index)

    # --- Step 4: Remaining numeric columns ---
    numeric_df = work[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=work.index)

    # --- Step 5: Concatenate and scale ---
    clinical_features = pd.concat([rint_df, onehot_df, numeric_df], axis=1)
    clinical_features.index = work["PATIENT_ID"].values

    # Impute + Scale
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(imputer.fit_transform(clinical_features))
    clinical_features = pd.DataFrame(
        scaled_values,
        columns=clinical_features.columns,
        index=clinical_features.index,
    )
    return clinical_features


def select_top_variance_features(matrix: pd.DataFrame, train_ids: Sequence[str], max_features: int = 300) -> List[str]:
    """Select top-variance expression features using training samples only."""
    if matrix is None or matrix.empty:
        return []
    train_idx = [pid for pid in map(str, train_ids) if pid in matrix.index]
    if not train_idx:
        return []
    train_matrix = matrix.loc[train_idx].replace([np.inf, -np.inf], np.nan)
    variances = train_matrix.var(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan).dropna()
    variances = variances[variances > 0].sort_values(ascending=False)
    return list(variances.head(max_features).index)


# ──────────────────────────────────────────────────────────────────────
# Task 2: 端点特异性特征筛选（top 300 高方差 + 因果优先级）
# ──────────────────────────────────────────────────────────────────────
def per_endpoint_feature_screening(clinical, feature_df, train_ids, tau_months, random_seed):
    """每个端点独立的特征筛选流程：训练集 top 300 高方差表达 + 因果优先级筛选。"""
    if feature_df is None or feature_df.empty:
        logger.warning("[04] 特征筛选跳过: feature_df 为空")
        return [], [], []

    expression_genes = select_top_variance_features(feature_df, train_ids, max_features=300)
    logger.info(f"[04] 表达特征筛选 (tau={tau_months}m): train-only top variance={len(expression_genes)}")
    if not expression_genes:
        return [], [], []

    time_col, event_col = detect_event_time_columns(clinical)
    causal_genes = []
    if run_causal_screening is not None:
        logger.info(f"[04] 端点筛选 (tau={tau_months}m): 因果优先级筛选 (候选={len(expression_genes)})")
        clinical_adj = clinical.copy()
        if "AGE" not in clinical_adj.columns and "AGE_AT_DIAGNOSIS" in clinical_adj.columns:
            clinical_adj["AGE"] = pd.to_numeric(clinical_adj["AGE_AT_DIAGNOSIS"], errors="coerce")
        if "AJCC_PATHOLOGIC_TUMOR_STAGE" in clinical_adj.columns and encode_ajcc_stage is not None:
            clinical_adj["AJCC_PATHOLOGIC_TUMOR_STAGE"] = encode_ajcc_stage(clinical_adj["AJCC_PATHOLOGIC_TUMOR_STAGE"])

        adj_cols = [c for c in ["AGE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE"] if c in clinical_adj.columns]
        if adj_cols and "PATIENT_ID" in clinical_adj.columns:
            clin_indexed = clinical_adj.set_index("PATIENT_ID")
            train_clinical_ids = [pid for pid in map(str, train_ids) if pid in clin_indexed.index.astype(str).values]
            train_mask = clin_indexed.index.astype(str).isin(train_clinical_ids)
            clin_adj = clin_indexed.loc[train_mask, adj_cols].copy()
            if "SEX" in clin_adj.columns:
                clin_adj["SEX"] = clin_adj["SEX"].map({"Male": 1, "Female": 0}).fillna(clin_adj["SEX"])
            clin_adj = clin_adj.apply(pd.to_numeric, errors="coerce")

            feature_matrix_train = feature_df[expression_genes].reindex(train_clinical_ids)
            feature_matrix_train.index = pd.Index(train_clinical_ids[:len(feature_matrix_train)])
            endpoint_train = pd.DataFrame({
                "time_months": pd.to_numeric(clin_indexed.loc[train_mask, time_col], errors="coerce").to_numpy(dtype=float),
                "event": clin_indexed.loc[train_mask, event_col].astype(int).to_numpy(),
            }, index=train_clinical_ids[:len(feature_matrix_train)])
            try:
                causal_table = run_causal_screening(
                    feature_matrix_train, clin_adj, endpoint_train,
                    random_seed=random_seed, max_features=len(expression_genes),
                )
                if not causal_table.empty and "causal_priority_score" in causal_table.columns:
                    causal_genes = causal_table.sort_values("causal_priority_score", ascending=False).head(50)["feature"].tolist()
                    logger.info(f"[04] 因果优先级基因: {len(causal_genes)}")
            except Exception as exc:
                logger.warning(f"[04] 因果筛选失败: {exc}")

    fused_genes = list(dict.fromkeys(expression_genes + causal_genes))
    logger.info(f"[04] 端点筛选完成 (tau={tau_months}m): expression_topvar={len(expression_genes)}, causal={len(causal_genes)}, fused={len(fused_genes)}")
    return fused_genes, expression_genes, causal_genes


# ──────────────────────────────────────────────────────────────────────
# Task 3: DeepSurv 和 Stacking 集成模型
# ──────────────────────────────────────────────────────────────────────
def fit_deepsurv(X_train, endpoint_train, random_seed):
    """Fits a simple feedforward survival network (DeepSurv) using PyTorch.

    Uses negative partial log-likelihood loss (Cox PH) with dropout regularization.
    Returns None if torch is unavailable or insufficient data.
    """
    if not HAS_TORCH or X_train.empty or endpoint_train["event"].sum() < 5:
        return None
    import torch
    import torch.nn as nn

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    input_dim = X_train.shape[1]
    hidden_dims = [64, 32]
    dropout_rate = 0.2

    class _SurvNet(nn.Module):
        def __init__(self, input_dim, hidden_dims, dropout_rate):
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout_rate)])
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = _SurvNet(input_dim, hidden_dims, dropout_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    X_tensor = torch.tensor(X_train.to_numpy(float), dtype=torch.float32)
    times = endpoint_train["time_months"].to_numpy(float)
    events = endpoint_train["event"].to_numpy(int)

    # Sort by time descending for risk set computation
    sort_idx = np.argsort(-times)
    X_sorted = X_tensor[sort_idx]
    events_sorted = events[sort_idx]
    event_mask = torch.tensor(events_sorted, dtype=torch.float32)

    n_epochs = 200
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        h = model(X_sorted)
        # Negative partial log-likelihood (Cox PH loss)
        log_cumsum_exp = torch.logcumsumexp(h, dim=0)
        loss = ((log_cumsum_exp - h) * event_mask).sum()
        loss.backward()
        optimizer.step()

    model.eval()
    return {"model": model, "feature_columns": list(X_train.columns)}


def predict_deepsurv_risk(model_info, X):
    """Predict risk scores from DeepSurv model."""
    if model_info is None:
        return np.full(len(X), np.nan)
    import torch
    model = model_info["model"]
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X.to_numpy(float), dtype=torch.float32)
        risk = model(X_tensor).numpy()
    return risk


def fit_stacking_ensemble(risks_train, endpoint_train, tau_int, random_seed):
    """Fits a stacking ensemble using base model predictions as meta-features.

    Trains LogisticRegressionCV as meta-learner on the training predictions
    of base models.

    Args:
        risks_train: dict of {model_name: risk_array} -- training set predictions
        endpoint_train: endpoint DataFrame
        tau_int: integer tau for binary label extraction (None for unlimited)
        random_seed: random seed

    Returns:
        (meta_model, model_names) tuple, or None if insufficient models
    """
    if len(risks_train) < 2:
        return None

    # Extract binary labels
    if tau_int is not None:
        y, w = censor_safe_binary_training(endpoint_train, tau_int)
    else:
        # Unlimited case: use event as label
        mask = endpoint_train["event"].notna()
        y = endpoint_train.loc[mask, "event"].astype(int)
        w = pd.Series(1.0, index=y.index)

    if y.nunique() < 2 or len(y) < 20:
        return None

    # Build meta-features from base model predictions
    model_names = list(risks_train.keys())
    meta_features = []
    valid_names = []
    for mname in model_names:
        risk = risks_train[mname]
        if risk is None:
            continue
        risk_arr = np.asarray(risk, dtype=float)
        if len(risk_arr) != len(endpoint_train):
            continue
        # Align with y index
        risk_series = pd.Series(risk_arr, index=endpoint_train.index)
        meta_features.append(risk_series.loc[y.index].to_numpy(float))
        valid_names.append(mname)

    if len(meta_features) < 2:
        return None

    meta_X = np.column_stack(meta_features)
    n_folds = min(5, max(2, int(y.value_counts().min())))
    # TODO: sklearn>=1.8 需迁移 penalty="elasticnet"
    meta_model = LogisticRegressionCV(
        Cs=10, cv=n_folds,
        penalty="elasticnet", solver="saga", l1_ratios=[0.05, 0.5, 0.95],
        scoring="roc_auc", max_iter=5000, random_state=random_seed, n_jobs=-1,
    )
    meta_model.fit(meta_X, y, sample_weight=w)
    return (meta_model, valid_names)


def fit_stacking_ensemble_oof(X_train, endpoint_train, tau_int, random_seed):
    """Fit stacking meta-learner using out-of-fold base-model predictions."""
    if X_train.empty or len(X_train) < 20:
        return None
    if tau_int is not None:
        y, w = censor_safe_binary_training(endpoint_train, tau_int)
    else:
        mask = endpoint_train["event"].notna()
        y = endpoint_train.loc[mask, "event"].astype(int)
        w = pd.Series(1.0, index=y.index)
    if y.nunique() < 2 or len(y) < 30:
        return None
    n_folds = min(5, int(y.value_counts().min()))
    if n_folds < 2:
        return None
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    oof = pd.DataFrame(index=y.index, dtype=float)
    model_names = ["Logistic", "Cox", "Coxnet", "RSF", "DeepSurv"]
    for name in model_names:
        oof[name] = np.nan
    for fold_id, (tr, va) in enumerate(splitter.split(X_train.loc[y.index], y), start=1):
        fold_train_ids = list(y.index[tr])
        fold_val_ids = list(y.index[va])
        Xf, epf = X_train.loc[fold_train_ids], endpoint_train.loc[fold_train_ids]
        # logistic
        log_model = fit_timepoint_logistic(Xf, epf, tau_int, random_seed + fold_id)
        if log_model is not None:
            oof.loc[fold_val_ids, "Logistic"] = predict_logistic_risk(log_model, X_train.loc[fold_val_ids])
        # cox
        cox_model = fit_timepoint_cox(Xf, epf)
        if cox_model is not None:
            if tau_int is not None:
                oof.loc[fold_val_ids, "Cox"] = predict_cox_risk(cox_model, X_train.loc[fold_val_ids], tau_int)
            else:
                oof.loc[fold_val_ids, "Cox"] = predict_cox_risk_unlimited(cox_model, X_train.loc[fold_val_ids])
        # coxnet
        coxnet_model = fit_timepoint_coxnet(Xf, epf)
        if coxnet_model is not None:
            if tau_int is not None:
                oof.loc[fold_val_ids, "Coxnet"] = predict_sksurv_risk(coxnet_model, X_train.loc[fold_val_ids], tau_int)
            else:
                oof.loc[fold_val_ids, "Coxnet"] = predict_sksurv_risk_unlimited(coxnet_model, X_train.loc[fold_val_ids])
        # rsf
        rsf_model = fit_timepoint_rsf(Xf, epf, random_seed + fold_id)
        if rsf_model is not None:
            if tau_int is not None:
                oof.loc[fold_val_ids, "RSF"] = predict_sksurv_risk(rsf_model, X_train.loc[fold_val_ids], tau_int)
            else:
                oof.loc[fold_val_ids, "RSF"] = predict_sksurv_risk_unlimited(rsf_model, X_train.loc[fold_val_ids])
        # deepsurv
        deepsurv_model = fit_deepsurv(Xf, epf, random_seed + fold_id)
        if deepsurv_model is not None:
            oof.loc[fold_val_ids, "DeepSurv"] = predict_deepsurv_risk(deepsurv_model, X_train.loc[fold_val_ids])
    valid_cols = [c for c in model_names if oof[c].notna().sum() > 0]
    if len(valid_cols) < 2:
        return None
    meta_X = oof[valid_cols].dropna().to_numpy(float)
    meta_y = y.loc[oof[valid_cols].dropna().index]
    meta_w = w.loc[oof[valid_cols].dropna().index]
    if meta_y.nunique() < 2 or len(meta_y) < 20:
        return None
    n_cv = min(5, max(2, int(meta_y.value_counts().min())))
    # TODO: sklearn>=1.8 需迁移 penalty="elasticnet"
    meta_model = LogisticRegressionCV(
        Cs=10, cv=n_cv,
        penalty="elasticnet", solver="saga", l1_ratios=[0.05, 0.5, 0.95],
        scoring="roc_auc", max_iter=5000, random_state=random_seed, n_jobs=-1,
    )
    meta_model.fit(meta_X, meta_y, sample_weight=meta_w)
    return (meta_model, valid_cols)


# Stacking 内部基模型名 → risks_val/risks_train 字典键名的映射
_STACKING_NAME_MAP = {
    "Logistic": "IPCW_Logistic",
    "Cox": "Cox_PH",
    "Coxnet": "Coxnet",
    "RSF": "RSF",
    "DeepSurv": "DeepSurv",
}


def predict_stacking_risk(stacking_info, risks_val):
    """Predict risk scores using stacking ensemble.

    Args:
        stacking_info: (meta_model, model_names) tuple from fit_stacking_ensemble
        risks_val: dict of {model_name: risk_array} -- validation set predictions
    """
    if stacking_info is None:
        return None
    meta_model, model_names = stacking_info
    meta_features = []
    for mname in model_names:
        lookup = _STACKING_NAME_MAP.get(mname, mname)
        risk = risks_val.get(lookup) if lookup in risks_val else risks_val.get(mname)
        if risk is None:
            return None
        meta_features.append(np.asarray(risk, dtype=float))
    meta_X = np.column_stack(meta_features)
    return meta_model.predict_proba(meta_X)[:, 1]


def load_input_data(timestamp):
    """Task 7: 加载完整基因表达矩阵，供每个端点独立筛选。

    不再从 02_gene_features/causal_priority_feature_table.tsv 加载预计算基因列表。
    改为返回原始基因表达矩阵，由各端点函数独立执行特征筛选。
    """
    results_base = os.path.join(RESULTS_DIR, timestamp)
    preproc_dir = os.path.join(results_base, "01_preprocessing")
    clinical_path = os.path.join(preproc_dir, "tcga_os_clinical_endpoint_qc.tsv")
    split_path = os.path.join(preproc_dir, "tcga_train_internal_validation_split.tsv")
    if not os.path.exists(split_path):
        split_path = os.path.join(DATA_DIR, "survival_models", "tcga_train_internal_validation_split.tsv")
    clinical = read_tsv(clinical_path)
    split_df = read_tsv(split_path)
    # Task 7: 加载完整基因表达矩阵，供每个端点独立筛选
    expr_path = os.path.join(preproc_dir, "gene_expression_curated.tsv")
    if os.path.exists(expr_path):
        expr_df = read_tsv(expr_path, index_col=0)
        expr_df = expr_df.apply(pd.to_numeric, errors="coerce")
        logger.info(f"[04] Gene expression matrix: {expr_df.shape[0]} samples x {expr_df.shape[1]} genes")
    else:
        expr_df = None
        logger.warning("[04] WARNING: gene_expression_curated.tsv not found")
    # 修复 split 匹配：val 为任何非 "train" 的 split
    train_ids_raw = set(split_df.loc[split_df["split"] == "train", "PATIENT_ID"].astype(str))
    val_ids_raw = set(split_df.loc[split_df["split"] != "train", "PATIENT_ID"].astype(str))
    # 交集过滤：只保留在 clinical 中存在的 ID
    clinical_ids = set(clinical["PATIENT_ID"].astype(str))
    train_ids = train_ids_raw & clinical_ids
    val_ids = val_ids_raw & clinical_ids
    logger.info(f"[04] Split: train_raw={len(train_ids_raw)}->filtered={len(train_ids)}, val_raw={len(val_ids_raw)}->filtered={len(val_ids)}")
    return clinical, expr_df, train_ids, val_ids


def prepare_features(clinical_ids, feature_df, train_ids, apply_rint=True):
    """Prepare a feature matrix with train-only fit preprocessing.

    Expression-like matrices use per-column RINT followed by a median imputer and
    StandardScaler fitted on training samples only. Validation/test rows are only
    transformed by the fitted preprocessing chain.
    """
    idx = list(map(str, clinical_ids))
    if feature_df is None:
        cols = [f"f_{i}" for i in range(10)]
        rng = np.random.default_rng(RANDOM_SEED)
        return pd.DataFrame(rng.standard_normal((len(idx), len(cols))), index=idx, columns=cols)

    X = feature_df.copy()
    X.index = X.index.astype(str)
    X = X.reindex(idx)
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if X.empty or X.shape[1] == 0:
        return pd.DataFrame(index=idx)

    train_idx = [pid for pid in map(str, train_ids) if pid in X.index]
    if len(train_idx) < 3:
        logger.warning("[04] prepare_features: insufficient train rows for fit; using train/global median fallback")
        med = X.loc[train_idx].median() if train_idx else X.median()
        return X.fillna(med).fillna(0.0)

    X_work = X.apply(rank_inverse_normal, axis=0) if apply_rint else X
    fit_block = X_work.loc[train_idx]
    valid_cols = fit_block.notna().sum(axis=0) >= 1
    X_work = X_work.loc[:, valid_cols]
    if X_work.shape[1] == 0:
        return pd.DataFrame(index=idx)

    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    pipe.fit(X_work.loc[train_idx])
    transformed = pipe.transform(X_work)
    return pd.DataFrame(transformed, index=X_work.index, columns=X_work.columns)


SURVIVAL_GAN_CONDITION_COLUMNS = [
    "death_by_36m", "event", "log1p_time_months", "pseudo_risk_36m_clipped", "ipcw_weight_36m",
]
GAN_MIN_AUC_GAIN = 0.005


def build_survival_gan_conditions_for_tau(endpoint, index, tau_int):
    ep = endpoint.reindex(index).copy()
    default_tau = float(tau_int or 36)
    death_col = f"death_by_{tau_int}m" if tau_int is not None else "event"
    pseudo_col = f"pseudo_risk_{tau_int}m" if tau_int is not None else None
    ipcw_col = f"ipcw_weight_{tau_int}m" if tau_int is not None else None
    death = ep.get(death_col, ep.get("event", pd.Series(0.0, index=ep.index))).astype(float)
    event = ep.get("event", death).astype(float)
    time_months = ep.get("time_months", pd.Series(default_tau, index=ep.index)).astype(float)
    pseudo = ep[pseudo_col].astype(float) if pseudo_col and pseudo_col in ep.columns else death
    weight = ep[ipcw_col].astype(float) if ipcw_col and ipcw_col in ep.columns else pd.Series(1.0, index=ep.index)
    cond = pd.DataFrame(index=ep.index)
    cond["death_by_36m"] = death.fillna(0.0).clip(0.0, 1.0)
    cond["event"] = event.fillna(cond["death_by_36m"]).clip(0.0, 1.0)
    clean_time = time_months.replace([np.inf, -np.inf], np.nan).fillna(default_tau).clip(lower=0.0)
    cond["log1p_time_months"] = np.log1p(clean_time)
    cond["pseudo_risk_36m_clipped"] = pseudo.replace([np.inf, -np.inf], np.nan).fillna(cond["death_by_36m"]).clip(0.0, 1.0)
    cond["ipcw_weight_36m"] = weight.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=EPS)
    return cond[SURVIVAL_GAN_CONDITION_COLUMNS].astype(float)


def endpoint_from_sampled_conditions(endpoint_template, sampled_conditions, syn_index, tau_int):
    syn_ep = pd.DataFrame(index=syn_index, columns=endpoint_template.columns)
    cond = sampled_conditions.reset_index(drop=True)
    syn_ep["time_months"] = np.expm1(cond["log1p_time_months"].to_numpy(float)).clip(min=0.0)
    syn_ep["event"] = cond["event"].round().clip(0, 1).astype(int).to_numpy()
    if "PATIENT_ID" in syn_ep.columns:
        syn_ep["PATIENT_ID"] = list(syn_index)
    if tau_int is not None:
        tau_col = f"death_by_{tau_int}m"
        if tau_col in syn_ep.columns:
            syn_ep[tau_col] = cond["death_by_36m"].round().clip(0, 1).astype(float).to_numpy()
        for col in [f"death_by_{tau_int}m_observed", f"early_censored_before_{tau_int}m"]:
            if col in syn_ep.columns:
                syn_ep[col] = False
        if "ipcw_label_available" in syn_ep.columns:
            syn_ep["ipcw_label_available"] = True
        ipcw_col = f"ipcw_weight_{tau_int}m"
        if ipcw_col in syn_ep.columns:
            syn_ep[ipcw_col] = cond["ipcw_weight_36m"].replace(0, np.nan).fillna(1.0).astype(float).to_numpy()
        pseudo_col = f"pseudo_risk_{tau_int}m"
        if pseudo_col in syn_ep.columns:
            syn_ep[pseudo_col] = cond["pseudo_risk_36m_clipped"].clip(0.02, 0.98).astype(float).to_numpy()
    return syn_ep


def _moment_match_calibrate(X_syn, X_real_train, syn_event_mask, real_event_label):
    out = X_syn.copy()
    cols = [c for c in X_real_train.columns if c in out.columns]
    real_event_label = np.asarray(real_event_label, dtype=int)
    groups = [(syn_event_mask, real_event_label == 1), (~syn_event_mask, real_event_label == 0)]
    for col in cols:
        for mask_syn, mask_real in groups:
            if int(np.sum(mask_syn)) < 2 or int(np.sum(mask_real)) < 2:
                continue
            real_grp = pd.to_numeric(X_real_train.loc[mask_real, col], errors="coerce")
            syn_grp = pd.to_numeric(out.loc[mask_syn, col], errors="coerce")
            mu_real, std_real = real_grp.mean(), real_grp.std()
            mu_syn, std_syn = syn_grp.mean(), syn_grp.std()
            if std_syn > EPS and np.isfinite(std_real) and np.isfinite(mu_real):
                out.loc[mask_syn, col] = (syn_grp - mu_syn) * (std_real / max(std_syn, EPS)) + mu_real
    return out


def synthetic_data_qc(X_real, X_syn, X_holdout=None, random_seed=RANDOM_SEED, min_good=3):
    """分层 QC：仅对 top-10 主成分做 KS/Wasserstein 检验，避免高维空间距离稀释。

    当特征维度 >> 样本量时，全维度 KS 检验过于严格（review 建议）。
    改为在 PCA 投影空间做 QC，更贴合 GAN 实际学习的分布。
    """
    cols = [c for c in X_real.columns if c in X_syn.columns]
    if not cols:
        return {"passed": False, "reason": "no_common_columns", "good_count": 0}
    X_r_raw = X_real[cols].replace([np.inf, -np.inf], np.nan)
    X_s_raw = X_syn[cols].replace([np.inf, -np.inf], np.nan)
    col_medians = X_r_raw.median()
    X_r = X_r_raw.fillna(col_medians).fillna(0.0)
    X_s = X_s_raw.fillna(col_medians).fillna(0.0)
    scaler_wd = StandardScaler()
    X_r_sc = scaler_wd.fit_transform(X_r)
    X_s_sc = scaler_wd.transform(X_s)

    # ── PCA 投影: 取 top-10 主成分做 QC ──
    n_pca_qc = min(10, len(cols), X_r.shape[0] - 1, X_s.shape[0] - 1)
    if n_pca_qc >= 2:
        from sklearn.decomposition import PCA
        pca_unified = PCA(n_components=n_pca_qc, random_state=random_seed)
        pca_r_proj = pca_unified.fit_transform(X_r_sc)
        pca_s_proj = pca_unified.transform(X_s_sc)
        pca_var_explained = pca_unified.explained_variance_ratio_
    else:
        pca_r_proj = X_r_sc
        pca_s_proj = X_s_sc
        pca_var_explained = np.ones(X_r_sc.shape[1])

    rng_qc = np.random.default_rng(random_seed)
    n_real = X_r.shape[0]
    half = n_real // 2

    # ── KS 检验: 在 PCA 空间做 ──
    ks_ref = 0.20  # 放宽阈值，高维数据 PCA 空间允许更大偏差
    if half >= 5:
        perm = rng_qc.permutation(n_real)
        a_idx, b_idx = perm[:half], perm[half:2 * half]
        ref_ks_vals = []
        for i in range(pca_r_proj.shape[1]):
            try:
                ref_ks_vals.append(float(stats.ks_2samp(pca_r_proj[a_idx, i], pca_r_proj[b_idx, i])[0]))
            except Exception:
                pass
        if ref_ks_vals:
            ks_ref = max(float(np.median(ref_ks_vals)) * 2.0, 0.20)

    ks_stats, ks_pvals = [], {}
    for i in range(pca_r_proj.shape[1]):
        try:
            stat, pv = stats.ks_2samp(pca_r_proj[:, i], pca_s_proj[:, i])
            ks_stats.append(float(stat))
            ks_pvals[f"PC{i}"] = float(pv)
        except Exception:
            pass
    ks_stat_mean = float(np.mean(ks_stats)) if ks_stats else 1.0
    ks_good = ks_stat_mean <= ks_ref

    # ── Wasserstein 检验: 在 PCA 空间做 ──
    wd_ref = 0.15
    wd_vals = []
    for i in range(pca_r_proj.shape[1]):
        try:
            wd_vals.append(float(stats.wasserstein_distance(pca_r_proj[:, i], pca_s_proj[:, i])))
        except Exception:
            pass
    wd_mean = float(np.mean(wd_vals)) if wd_vals else 1.0
    wd_tol = max(wd_ref * 2.0, 0.15)
    wd_good = wd_mean <= wd_tol

    # ── PCA 方差相关性 ──
    pca_var_corr = 0.0
    try:
        if n_pca_qc >= 2:
            pca_s_fresh = PCA(n_components=n_pca_qc, random_state=random_seed).fit(X_s_sc)
            pca_var_corr = float(np.corrcoef(
                pca_unified.explained_variance_ratio_,
                pca_s_fresh.explained_variance_ratio_
            )[0, 1])
            if not np.isfinite(pca_var_corr):
                pca_var_corr = 0.0
    except Exception:
        pass
    pca_good = pca_var_corr > 0.70

    # ── DCR 比率: 在 PCA 空间做 ──
    dcr_ratio = 1.0
    try:
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(pca_r_proj)
        dist_syn, _ = nn.kneighbors(pca_s_proj)
        nn_r = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(pca_r_proj)
        dist_rr, _ = nn_r.kneighbors(pca_r_proj)
        baseline = dist_rr[:, 1].mean() if dist_rr.shape[1] > 1 else dist_rr.mean()
        dcr_ratio = float(dist_syn.mean() / max(baseline, EPS))
    except Exception:
        pass
    dcr_good = 0.5 <= dcr_ratio <= 3.0

    # ── MIA 攻击准确率 ──
    mia_acc = 0.5
    if X_holdout is not None and len(X_holdout) > 5:
        try:
            X_h = X_holdout[cols].replace([np.inf, -np.inf], np.nan).fillna(col_medians).fillna(0.0)
            X_h_proj = pca_unified.transform(scaler_wd.transform(X_h)) if n_pca_qc >= 2 else scaler_wd.transform(X_h)
            nn_s = NearestNeighbors(n_neighbors=min(5, len(pca_s_proj)), metric="euclidean").fit(pca_s_proj)
            d_train = nn_s.kneighbors(pca_r_proj)[0].mean(axis=1)
            d_hold = nn_s.kneighbors(X_h_proj)[0].mean(axis=1)
            threshold = float(np.median(np.concatenate([d_train, d_hold])))
            mia_acc = float((np.mean(d_train < threshold) + np.mean(d_hold >= threshold)) / 2)
        except Exception:
            mia_acc = 0.5
    mia_good = mia_acc < 0.65

    good_count = sum([ks_good, wd_good, pca_good, dcr_good, mia_good])
    return {
        "passed": bool(good_count >= min_good), "good_count": int(good_count),
        "ks_stat_mean": round(ks_stat_mean, 4), "ks_pass_rate": float(np.mean([p > 0.05 for p in ks_pvals.values()])) if ks_pvals else 0.0,
        "ks_tol": round(ks_ref, 4), "ks_good": bool(ks_good),
        "wd_mean": round(wd_mean, 6), "wd_tol": round(wd_tol, 6), "wd_good": bool(wd_good),
        "pca_var_correlation": round(pca_var_corr, 4), "pca_good": bool(pca_good),
        "dcr_ratio": round(dcr_ratio, 4), "dcr_good": bool(dcr_good),
        "mia_attack_accuracy": round(mia_acc, 4), "mia_good": bool(mia_good),
    }


def train_only_cv_auc_for_logistic(X_train, endpoint_train, tau_int, random_seed, augment_fn=None):
    try:
        if tau_int is not None:
            y, _ = censor_safe_binary_training(endpoint_train, tau_int)
        else:
            y = endpoint_train["event"].astype(int)
        if y.nunique() < 2 or len(y) < 30:
            return float("nan")
        n_folds = min(5, int(y.value_counts().min()))
        if n_folds < 2:
            return float("nan")
        aucs = []
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        for fold_id, (tr, va) in enumerate(splitter.split(X_train.loc[y.index], y), start=1):
            fold_train_ids = list(y.index[tr])
            fold_val_ids = list(y.index[va])
            X_fold, ep_fold = X_train.loc[fold_train_ids], endpoint_train.loc[fold_train_ids]
            if augment_fn is not None:
                X_fold, ep_fold = augment_fn(X_fold, ep_fold, random_seed + fold_id)
            model = fit_timepoint_logistic(X_fold, ep_fold, tau_int, random_seed + fold_id) if tau_int is not None else None
            if model is None:
                continue
            risk_val = predict_logistic_risk(model, X_train.loc[fold_val_ids])
            aucs.append(roc_auc_score(y.loc[fold_val_ids], risk_val))
        return float(np.mean(aucs)) if aucs else float("nan")
    except Exception:
        return float("nan")



def prepare_nonnormal_features(clinical_ids, feature_df, train_ids):
    """Prepare non-expression features: median fill only, no RINT."""
    idx = list(map(str, clinical_ids))
    if feature_df is None or feature_df.empty:
        return pd.DataFrame(index=idx)
    X = feature_df.copy()
    X.index = X.index.astype(str)
    X = X.reindex(idx).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    train_idx = [pid for pid in map(str, train_ids) if pid in X.index]
    med = X.loc[train_idx].median() if train_idx else X.median()
    return X.fillna(med).fillna(0.0)


def build_feature_set_matrices(clinical_ids, train_ids, clinical_features, feature_df,
                               expression_genes, causal_genes):
    """Build dict of feature-set matrices, each with train-only preprocessing."""
    clinical_all = prepare_nonnormal_features(clinical_ids, clinical_features, train_ids)
    matrices = {"clinical_only": clinical_all}
    metadata = {"clinical_only": list(clinical_all.columns)}
    pathway_df = compute_pathway_features(feature_df) if feature_df is not None else pd.DataFrame()
    if feature_df is not None and expression_genes:
        expr_genes = [g for g in expression_genes if g in feature_df.columns]
        if expr_genes:
            expr_all = prepare_features(clinical_ids, feature_df[expr_genes], train_ids, apply_rint=True)
            matrices["clinical_expression_topvar"] = pd.concat([clinical_all, expr_all], axis=1)
            metadata["clinical_expression_topvar"] = list(
                matrices["clinical_expression_topvar"].columns)
    if feature_df is not None and causal_genes:
        cg = [g for g in causal_genes if g in feature_df.columns]
        if cg:
            causal_all = prepare_features(clinical_ids, feature_df[cg], train_ids, apply_rint=True)
            matrices["clinical_causal_priority"] = pd.concat(
                [clinical_all, causal_all], axis=1)
            metadata["clinical_causal_priority"] = list(
                matrices["clinical_causal_priority"].columns)
    if not pathway_df.empty:
        pathway_all = prepare_nonnormal_features(clinical_ids, pathway_df, train_ids)
        matrices["clinical_pathway"] = pd.concat([clinical_all, pathway_all], axis=1)
        metadata["clinical_pathway"] = list(matrices["clinical_pathway"].columns)
    if feature_df is not None and expression_genes and not pathway_df.empty:
        expr_genes = [g for g in expression_genes if g in feature_df.columns]
        if expr_genes:
            expr_all = prepare_features(clinical_ids, feature_df[expr_genes], train_ids, apply_rint=True)
            pathway_all = prepare_nonnormal_features(clinical_ids, pathway_df, train_ids)
            matrices["clinical_expression_pathway"] = pd.concat(
                [clinical_all, expr_all, pathway_all], axis=1)
            metadata["clinical_expression_pathway"] = list(
                matrices["clinical_expression_pathway"].columns)
    return matrices, metadata


def split_feature_matrix(X_all, endpoint_full, train_ids, val_ids):
    """Split a feature matrix into train / val aligned subsets."""
    train_ep = endpoint_full[endpoint_full["PATIENT_ID"].astype(str)
                            .isin(set(map(str, train_ids)))].copy()
    val_ep = endpoint_full[endpoint_full["PATIENT_ID"].astype(str)
                          .isin(set(map(str, val_ids)))].copy()
    X_train = X_all.loc[X_all.index.isin(set(map(str, train_ids)))]
    X_val = X_all.loc[X_all.index.isin(set(map(str, val_ids)))]
    train_common = X_train.index.intersection(train_ep.index)
    val_common = X_val.index.intersection(val_ep.index)
    return (X_train.loc[train_common], train_ep.loc[train_common],
            X_val.loc[val_common], val_ep.loc[val_common])


def augment_training_data(X_train, endpoint_train, tau_int, seed=RANDOM_SEED):
    """Augment training data with GAN or SMOTE-like oversampling.

    Task 6 improvements:
    1. Prioritize SurvivalConditionalTabularGAN from 03_model_training (direct import)
    2. Condition vector: event status + risk quantiles + IPCW label
    3. Only generate synthetic samples for minority class (event group)
    4. SMOTE fallback with noise control parameters
    5. Log before/after class distribution

    Args:
        X_train: training feature matrix
        endpoint_train: endpoint DataFrame
        tau_int: integer tau for binary label (None for unlimited OS)
        seed: random seed

    Returns:
        (X_aug, ep_aug) — augmented training features and endpoint DataFrames.
    """
    rng = np.random.default_rng(seed)
    # Determine binary label column
    if tau_int is not None:
        tau_col = f"death_by_{tau_int}m"
    else:
        tau_col = "event"  # Unlimited case: use event directly
    if tau_col not in endpoint_train.columns:
        return X_train, endpoint_train
    y = endpoint_train[tau_col].fillna(0).astype(int)
    n_event = int(y.sum())
    n_nonevent = len(y) - n_event
    # Task 6.5: Log before augmentation
    logger.info(f"[04] Augment before: event={n_event}, nonevent={n_nonevent}, ratio={n_event / max(1, len(y)):.2f}")
    if n_event < 5 or n_nonevent < 5:
        return X_train, endpoint_train  # too few samples

    target_ratio = 0.5  # aim for event:total >= 0.5
    n_needed = max(0, int(n_nonevent * target_ratio / (1 - target_ratio)) - n_event)
    if n_needed == 0:
        return X_train, endpoint_train

    # --- Task 6.1: Try SurvivalGAN from 03_model_training (direct import) ---
    if HAS_TORCH and SurvivalConditionalTabularGAN is not None:
        try:
            cond = build_survival_gan_conditions_for_tau(endpoint_train, endpoint_train.index, tau_int)
            gan = SurvivalConditionalTabularGAN(epochs=150, random_seed=seed)
            gan.fit(X_train, cond)
            if gan.status == "fitted":
                event_idx = y[y == 1].index
                cond_sample = cond.loc[rng.choice(event_idx, size=min(n_needed, len(event_idx)), replace=True)]
                syn_X = gan.sample(len(cond_sample), cond_sample.to_numpy())
                syn_X.index = [f"GAN_{i}" for i in range(len(syn_X))]
                cond_sample = cond_sample.reset_index(drop=True)
                syn_event_mask = cond_sample["death_by_36m"].round().clip(0, 1).to_numpy(int) == 1
                syn_X = _moment_match_calibrate(syn_X, X_train, syn_event_mask, y.reindex(X_train.index).fillna(0).to_numpy(int))
                qc = synthetic_data_qc(X_train, syn_X, random_seed=seed)
                logger.info(f"[04] GAN QC: passed={qc.get('passed')} good_count={qc.get('good_count')} details={qc}")
                if not qc.get("passed", False):
                    logger.warning("[04] GAN rejected by QC; falling back to SMOTE")
                    raise RuntimeError("GAN QC failed")
                syn_ep = endpoint_from_sampled_conditions(endpoint_train, cond_sample, syn_X.index, tau_int)
                X_aug = pd.concat([X_train, syn_X], axis=0)
                ep_aug = pd.concat([endpoint_train, syn_ep], axis=0)
                logger.info(f"[04] GAN augmentation: +{len(syn_X)} synthetic samples (SurvivalGAN)")
                logger.info(f"[04] Augment after: event={int(ep_aug[tau_col].sum())}, nonevent={int((~ep_aug[tau_col].astype(bool)).sum())}, total={len(ep_aug)}")
                return X_aug, ep_aug
        except Exception as exc:
            logger.warning(f"[04] GAN augmentation failed, falling back to SMOTE: {exc}")

    # --- Task 6.4: Fallback: SMOTE-like oversampling with noise control ---
    event_idx = y[y == 1].index.tolist()
    syn_rows_X, syn_rows_ep = [], []
    noise_scale = 0.03  # Task 6.4: noise control parameter
    for i in range(n_needed):
        src = rng.choice(event_idx)
        row_x = X_train.loc[src].copy()
        std_val = float(row_x.std())
        noise = rng.normal(0, noise_scale * max(std_val, 0.01), size=len(row_x))
        row_x = row_x + noise
        syn_rows_X.append(row_x)
        row_ep = endpoint_train.loc[src].copy()
        syn_rows_ep.append(row_ep)
    if syn_rows_X:
        syn_X = pd.DataFrame(syn_rows_X, index=[f"SMOTE_{i}" for i in range(len(syn_rows_X))])
        syn_ep = pd.DataFrame(syn_rows_ep, index=[f"SMOTE_{i}" for i in range(len(syn_rows_ep))])
        X_aug = pd.concat([X_train, syn_X], axis=0)
        ep_aug = pd.concat([endpoint_train, syn_ep], axis=0)
        logger.info(f"[04] SMOTE augmentation: +{len(syn_X)} synthetic samples (noise_scale={noise_scale})")
        logger.info(f"[04] Augment after: event={int(ep_aug[tau_col].sum())}, nonevent={int((~ep_aug[tau_col].astype(bool)).sum())}, total={len(ep_aug)}")
        return X_aug, ep_aug
    return X_train, endpoint_train


def run_fixed_timepoint(tp_name, tau_months, eval_times, timestamp, clinical, feature_df, train_ids, val_ids):
    """Multi feature-set evaluation for fixed timepoint.

    For each feature set (clinical, expression, pathway, causal) trains models
    independently, then ranks all models together using Uno's C-index.
    """
    tau_int = int(tau_months)
    logger.info(f"\n{'='*60}\nProcessing {tp_name} (tau={tau_months}m)\n{'='*60}")
    results_base = os.path.join(RESULTS_DIR, timestamp, "04_multi_timepoint", tp_name)
    ensure_dir(results_base)

    # ── 阶段 1: 端点构建 ──
    endpoint_full = add_fixed_endpoint(clinical, tau_months)
    endpoint_full = compute_ipcw_weights(endpoint_full, tau_months)
    endpoint_full = compute_pseudo_observations(endpoint_full, tau_months)
    endpoint_full.index = endpoint_full["PATIENT_ID"].astype(str)
    endpoint_full.to_csv(os.path.join(results_base, f"tcga_{tau_int}m_endpoint_ipcw_pseudo.tsv"), sep="\t")

    # ── 阶段 2-4: 特征筛选 ──
    fused_genes, expression_genes, causal_genes = per_endpoint_feature_screening(
        clinical, feature_df, train_ids, tau_months, RANDOM_SEED)
    pd.DataFrame({"feature": expression_genes}).to_csv(
        os.path.join(results_base, "expression_topvar_genes.tsv"), sep="\t", index=False)
    pd.DataFrame({"feature": causal_genes}).to_csv(
        os.path.join(results_base, "causal_screening_genes.tsv"), sep="\t", index=False)
    pd.DataFrame({"feature": fused_genes}).to_csv(
        os.path.join(results_base, "fused_gene_list.tsv"), sep="\t", index=False)

    clinical_features = build_clinical_features(clinical)
    fs_matrices, fs_meta = build_feature_set_matrices(
        endpoint_full["PATIENT_ID"].tolist(), train_ids, clinical_features,
        feature_df, expression_genes, causal_genes)

    all_rows, all_risks_val, all_risks_train, all_models, all_kinds = [], {}, {}, {}, {}

    for fs_name, X_all in fs_matrices.items():
        logger.info(f"[04] Feature set: {fs_name} ({X_all.shape[1]} features)")
        X_train, train_ep, X_val, val_ep = split_feature_matrix(
            X_all, endpoint_full, train_ids, val_ids)
        if X_train.empty or X_val.empty:
            continue

        # ── 阶段 5: GAN 增强 ──
        X_train_aug, train_ep_aug = augment_training_data(
            X_train, train_ep, tau_int, RANDOM_SEED)

        models, risks_val, risks_train = {}, {}, {}

        log_model = fit_timepoint_logistic(X_train_aug, train_ep_aug, tau_int, RANDOM_SEED)
        if log_model is not None:
            models["IPCW_Logistic"] = log_model
            risks_val["IPCW_Logistic"] = predict_logistic_risk(log_model, X_val)
            risks_train["IPCW_Logistic"] = predict_logistic_risk(log_model, X_train_aug)

        cox_model = fit_timepoint_cox(X_train_aug, train_ep_aug)
        if cox_model is not None:
            models["Cox_PH"] = cox_model
            risks_val["Cox_PH"] = predict_cox_risk(cox_model, X_val, tau_months)
            risks_train["Cox_PH"] = predict_cox_risk(cox_model, X_train_aug, tau_months)

        coxnet_model = fit_timepoint_coxnet(X_train_aug, train_ep_aug)
        if coxnet_model is not None:
            models["Coxnet"] = coxnet_model
            risks_val["Coxnet"] = predict_sksurv_risk(coxnet_model, X_val, tau_months)
            risks_train["Coxnet"] = predict_sksurv_risk(coxnet_model, X_train_aug, tau_months)

        rsf_model = fit_timepoint_rsf(X_train_aug, train_ep_aug, RANDOM_SEED)
        if rsf_model is not None:
            models["RSF"] = rsf_model
            risks_val["RSF"] = predict_sksurv_risk(rsf_model, X_val, tau_months)
            risks_train["RSF"] = predict_sksurv_risk(rsf_model, X_train_aug, tau_months)

        deepsurv_model = fit_deepsurv(X_train_aug, train_ep_aug, RANDOM_SEED)
        if deepsurv_model is not None:
            models["DeepSurv"] = deepsurv_model
            risks_val["DeepSurv"] = predict_deepsurv_risk(deepsurv_model, X_val)
            risks_train["DeepSurv"] = predict_deepsurv_risk(deepsurv_model, X_train_aug)

        stacking_info = fit_stacking_ensemble_oof(
            X_train_aug, train_ep_aug, tau_int, RANDOM_SEED)
        if stacking_info is not None:
            stacking_risk_val = predict_stacking_risk(stacking_info, risks_val)
            if stacking_risk_val is not None:
                models["Stacking"] = stacking_info
                risks_val["Stacking"] = stacking_risk_val
                stacking_risk_train = predict_stacking_risk(stacking_info, risks_train)
                risks_train["Stacking"] = stacking_risk_train if stacking_risk_train is not None else np.full(len(X_train_aug), np.nan)

        for mname, rval in risks_val.items():
            rtrain = risks_train.get(mname)
            thresh = choose_training_threshold(train_ep_aug, rtrain, tau_int) if rtrain is not None else None
            metrics = evaluate_fixed_tau(train_ep, val_ep, rval, tau_months, threshold=thresh)
            metrics["model"] = f"{fs_name}__{mname}"
            metrics["feature_set"] = fs_name
            metrics["base_model"] = mname
            all_rows.append(metrics)
            full_name = f"{fs_name}__{mname}"
            all_risks_val[full_name] = rval
            all_risks_train[full_name] = rtrain
            all_models[full_name] = models[mname]

    if not all_rows:
        logger.warning(f"[WARN] No models fitted for {tp_name}")
        return {}

    comp_df = pd.DataFrame(all_rows)
    ranking_df = comp_df.copy()
    for col in ["uno_cindex", "ipcw_auc", "ipcw_brier"]:
        if col in ranking_df.columns:
            asc = True if col == "ipcw_brier" else False
            ranking_df[f"{col}_rank"] = ranking_df[col].rank(ascending=asc, method="average")
    rank_cols = [c for c in ranking_df.columns if c.endswith("_rank")]
    if rank_cols:
        ranking_df["selection_score"] = ranking_df[rank_cols].sum(axis=1)
        best_row = ranking_df.sort_values(
            ["selection_score", "uno_cindex_rank", "ipcw_brier_rank"]
        ).iloc[0]
        best_model_name = best_row["model"]
    else:
        best_model_name = ranking_df.iloc[0]["model"]
    logger.info(f"[04] Best model ({tp_name}): {best_model_name}, selection_basis = Uno C + AUC + Brier")

    comp_df.to_csv(os.path.join(results_base, "model_comparison.tsv"), sep="\t", index=False)
    best_risk_val = all_risks_val[best_model_name]
    best_thresh = choose_training_threshold(
        endpoint_full[endpoint_full["PATIENT_ID"].astype(str).isin(set(map(str, train_ids)))],
        all_risks_train.get(best_model_name), tau_int)

    # Write risk scores for validation set
    _, _, _, val_ep_final = split_feature_matrix(
        fs_matrices.get(comp_df[comp_df["model"] == best_model_name].iloc[0].get("feature_set", "clinical_only"),
                        fs_matrices["clinical_only"]),
        endpoint_full, train_ids, val_ids)
    risk_df = val_ep_final[["PATIENT_ID", "time_months", "event"]].copy()
    risk_df["predicted_risk"] = pd.Series(best_risk_val[:len(val_ep_final)], index=val_ep_final.index)
    risk_df.to_csv(os.path.join(results_base, "risk_scores.tsv"), sep="\t", index=False)

    cohort_label = "TCGA_internal"
    stem = sanitize_filename(cohort_label) + "__" + sanitize_filename(best_model_name)
    plot_km_risk_strata(val_ep_final, best_risk_val[:len(val_ep_final)], best_thresh, tau_months,
                        f"{cohort_label}: K-M risk strata ({tp_name})",
                        Path(results_base) / f"{stem}__km_risk_strata")
    plot_time_dependent_roc(val_ep_final, best_risk_val[:len(val_ep_final)], tau_months,
                            f"{cohort_label}: time-dependent ROC ({tp_name})",
                            Path(results_base) / f"{stem}__time_dependent_roc", tau_int)
    plot_calibration_curve(val_ep_final, best_risk_val[:len(val_ep_final)], tau_months,
                          f"{cohort_label}: calibration ({tp_name})",
                          Path(results_base) / f"{stem}__calibration", tau_int)

    best_metrics = comp_df[comp_df["model"] == best_model_name].iloc[0].to_dict() if not comp_df.empty else {}
    best_metrics["timepoint"] = tp_name
    best_metrics["best_model"] = best_model_name
    return best_metrics


def run_unlimited_timepoint(timestamp, clinical, feature_df, train_ids, val_ids):
    """Multi feature-set evaluation for unlimited OS."""
    tp_name = "Unlimited"
    logger.info(f"\n{'='*60}\nProcessing Unlimited OS\n{'='*60}")
    results_base = os.path.join(RESULTS_DIR, timestamp, "04_multi_timepoint", tp_name)
    ensure_dir(results_base)

    # ── 阶段 1: 端点构建 ──
    endpoint_full = add_unlimited_endpoint(clinical)
    endpoint_full.index = endpoint_full["PATIENT_ID"].astype(str)
    endpoint_full.to_csv(os.path.join(results_base, "tcga_unlimited_os_endpoint.tsv"), sep="\t")

    # ── 阶段 2-4: 特征筛选 ──
    fused_genes, expression_genes, causal_genes = per_endpoint_feature_screening(
        clinical, feature_df, train_ids, None, RANDOM_SEED)
    pd.DataFrame({"feature": expression_genes}).to_csv(
        os.path.join(results_base, "expression_topvar_genes.tsv"), sep="\t", index=False)
    pd.DataFrame({"feature": causal_genes}).to_csv(
        os.path.join(results_base, "causal_screening_genes.tsv"), sep="\t", index=False)
    pd.DataFrame({"feature": fused_genes}).to_csv(
        os.path.join(results_base, "fused_gene_list.tsv"), sep="\t", index=False)

    clinical_features = build_clinical_features(clinical)
    fs_matrices, fs_meta = build_feature_set_matrices(
        endpoint_full["PATIENT_ID"].tolist(), train_ids, clinical_features,
        feature_df, expression_genes, causal_genes)

    all_rows, all_risks_val, all_risks_train, all_models, all_kinds = [], {}, {}, {}, {}

    for fs_name, X_all in fs_matrices.items():
        logger.info(f"[04] Feature set: {fs_name} ({X_all.shape[1]} features)")
        X_train, train_ep, X_val, val_ep = split_feature_matrix(
            X_all, endpoint_full, train_ids, val_ids)
        if X_train.empty or X_val.empty:
            continue

        X_train_aug, train_ep_aug = augment_training_data(
            X_train, train_ep, None, RANDOM_SEED)

        models, model_kinds, risks_val, risks_train = {}, {}, {}, {}
        cox_model = fit_timepoint_cox(X_train_aug, train_ep_aug)
        if cox_model is not None:
            models["Cox_PH"] = cox_model
            model_kinds["Cox_PH"] = "cox_lifelines"
            risks_val["Cox_PH"] = predict_cox_risk_unlimited(cox_model, X_val)
            risks_train["Cox_PH"] = predict_cox_risk_unlimited(cox_model, X_train_aug)

        coxnet_model = fit_timepoint_coxnet(X_train_aug, train_ep_aug)
        if coxnet_model is not None:
            models["Coxnet"] = coxnet_model
            model_kinds["Coxnet"] = "sksurv"
            risks_val["Coxnet"] = predict_sksurv_risk_unlimited(coxnet_model, X_val)
            risks_train["Coxnet"] = predict_sksurv_risk_unlimited(coxnet_model, X_train_aug)

        rsf_model = fit_timepoint_rsf(X_train_aug, train_ep_aug, RANDOM_SEED)
        if rsf_model is not None:
            models["RSF"] = rsf_model
            model_kinds["RSF"] = "sksurv"
            risks_val["RSF"] = predict_sksurv_risk_unlimited(rsf_model, X_val)
            risks_train["RSF"] = predict_sksurv_risk_unlimited(rsf_model, X_train_aug)

        deepsurv_model = fit_deepsurv(X_train_aug, train_ep_aug, RANDOM_SEED)
        if deepsurv_model is not None:
            models["DeepSurv"] = deepsurv_model
            risks_val["DeepSurv"] = predict_deepsurv_risk(deepsurv_model, X_val)
            risks_train["DeepSurv"] = predict_deepsurv_risk(deepsurv_model, X_train_aug)

        stacking_info = fit_stacking_ensemble_oof(
            X_train_aug, train_ep_aug, None, RANDOM_SEED)
        if stacking_info is not None:
            stacking_risk_val = predict_stacking_risk(stacking_info, risks_val)
            if stacking_risk_val is not None:
                models["Stacking"] = stacking_info
                risks_val["Stacking"] = stacking_risk_val
                stacking_risk_train = predict_stacking_risk(stacking_info, risks_train)
                risks_train["Stacking"] = stacking_risk_train if stacking_risk_train is not None else np.full(len(X_train_aug), np.nan)

        for mname, rval in risks_val.items():
            rtrain = risks_train.get(mname)
            thresh = choose_risk_threshold_unlimited(rtrain) if rtrain is not None else None
            if mname in ("DeepSurv", "Stacking"):
                metrics = evaluate_unlimited(train_ep, val_ep, rval, model=None,
                                             model_kind=None, x_eval=None, threshold=thresh)
            else:
                metrics = evaluate_unlimited(train_ep, val_ep, rval, model=models[mname],
                                             model_kind=model_kinds.get(mname), x_eval=X_val, threshold=thresh)
            metrics["model"] = f"{fs_name}__{mname}"
            metrics["feature_set"] = fs_name
            metrics["base_model"] = mname
            all_rows.append(metrics)
            full_name = f"{fs_name}__{mname}"
            all_risks_val[full_name] = rval
            all_risks_train[full_name] = rtrain
            all_models[full_name] = models[mname]

    if not all_rows:
        logger.warning(f"[WARN] No models fitted for {tp_name}")
        return {}

    comp_df = pd.DataFrame(all_rows)
    ranking_df = comp_df.copy()
    for col in ["uno_cindex"]:
        if col in ranking_df.columns:
            ranking_df[f"{col}_rank"] = ranking_df[col].rank(ascending=False, method="average")
    if "integrated_brier_score" in ranking_df.columns:
        ranking_df["integrated_brier_score_rank"] = ranking_df["integrated_brier_score"].rank(
            ascending=True, method="average")
    rank_cols = [c for c in ranking_df.columns if c.endswith("_rank")]
    if rank_cols:
        ranking_df["selection_score"] = ranking_df[rank_cols].sum(axis=1)
        best_row = ranking_df.sort_values(
            ["selection_score", "uno_cindex_rank", "integrated_brier_score_rank"]
        ).iloc[0]
        best_model_name = best_row["model"]
    else:
        best_model_name = ranking_df.iloc[0]["model"]
    logger.info(f"[04] Best model (Unlimited): {best_model_name}, selection_basis = Uno C + IBS")

    comp_df.to_csv(os.path.join(results_base, "model_comparison.tsv"), sep="\t", index=False)
    best_risk_val = all_risks_val[best_model_name]
    best_thresh = choose_risk_threshold_unlimited(
        np.asarray(all_risks_train.get(best_model_name, []), dtype=float))

    best_fs = comp_df[comp_df["model"] == best_model_name].iloc[0].get("feature_set", "clinical_only")
    _, _, _, val_ep_final = split_feature_matrix(
        fs_matrices.get(best_fs, fs_matrices["clinical_only"]),
        endpoint_full, train_ids, val_ids)
    risk_df = val_ep_final[["PATIENT_ID", "time_months", "event"]].copy()
    risk_df["predicted_risk"] = pd.Series(best_risk_val[:len(val_ep_final)], index=val_ep_final.index)
    risk_df.to_csv(os.path.join(results_base, "risk_scores.tsv"), sep="\t", index=False)

    cohort_label = "TCGA_internal"
    stem = sanitize_filename(cohort_label) + "__" + sanitize_filename(best_model_name)
    plot_km_risk_strata(val_ep_final, best_risk_val[:len(val_ep_final)], best_thresh, None,
                        f"{cohort_label}: K-M risk strata (Unlimited)",
                        Path(results_base) / f"{stem}__km_risk_strata", is_unlimited=True)
    plot_risk_distribution(val_ep_final, best_risk_val[:len(val_ep_final)],
                           f"{cohort_label}: risk by event status (Unlimited)",
                           Path(results_base) / f"{stem}__risk_distribution")
    plot_followup_vs_risk(val_ep_final, best_risk_val[:len(val_ep_final)],
                          f"{cohort_label}: follow-up time vs risk (Unlimited)",
                          Path(results_base) / f"{stem}__followup_time_vs_risk")

    best_metrics = comp_df[comp_df["model"] == best_model_name].iloc[0].to_dict() if not comp_df.empty else {}
    best_metrics["timepoint"] = tp_name
    best_metrics["best_model"] = best_model_name
    return best_metrics


def generate_timepoint_comparison(all_results, timestamp):
    out_dir = os.path.join(RESULTS_DIR, timestamp, "04_multi_timepoint")
    ensure_dir(out_dir)
    rows = [res for res in all_results if res]
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, "timepoint_comparison.tsv"), sep="\t", index=False)
        logger.info(f"\nTimepoint comparison saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()
    timestamp = args.timestamp
    logger.info(f"[04_multi_timepoint] timestamp={timestamp}")
    logger.info(f"SCRIPT_DIR = {SCRIPT_DIR}")
    logger.info(f"DATA_DIR   = {DATA_DIR}")
    logger.info(f"RESULTS_DIR= {RESULTS_DIR}")
    # Task 7: load_input_data 返回原始基因表达矩阵 expr_df（而非预计算特征）
    clinical, expr_df, train_ids, val_ids = load_input_data(timestamp)
    logger.info(f"Loaded clinical: {clinical.shape}, expr_df: {expr_df.shape if expr_df is not None else 'None'}")
    logger.info(f"Train IDs: {len(train_ids)}, Val IDs: {len(val_ids)}")
    all_results = []
    for tp_name, tp_cfg in TIMEPOINTS.items():
        tau = tp_cfg["tau_months"]
        eval_times = tp_cfg["eval_times"]
        try:
            if tau is None:
                res = run_unlimited_timepoint(timestamp, clinical, expr_df, train_ids, val_ids)
            else:
                res = run_fixed_timepoint(tp_name, tau, eval_times, timestamp, clinical, expr_df, train_ids, val_ids)
            all_results.append(res)
        except Exception as exc:
            logger.error(f"[ERROR] {tp_name}: {exc}")
            traceback.print_exc()
            all_results.append({"timepoint": tp_name, "error": str(exc)})
    generate_timepoint_comparison(all_results, timestamp)
    logger.info("\n[04_multi_timepoint] DONE")


if __name__ == "__main__":
    main()
