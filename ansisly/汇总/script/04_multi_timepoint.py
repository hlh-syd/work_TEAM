from __future__ import annotations

import argparse
import math
import os
import re
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
from sklearn.model_selection import train_test_split

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

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "DATA")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
RANDOM_SEED = 20260604
EPS = 1e-8

TIMEPOINTS = {
    "1Year":     {"tau_months": 12.0, "eval_times": [6, 9, 12]},
    "2Years":    {"tau_months": 24.0, "eval_times": [12, 18, 24]},
    "3Years":    {"tau_months": 36.0, "eval_times": [12, 24, 36]},
    "Unlimited": {"tau_months": None, "eval_times": [12, 24, 36]},
}


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_tsv(path, **kwargs):
    return pd.read_csv(path, sep="\t", low_memory=False, **kwargs)


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def sanitize_filename(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def normalize_patient_id(value):
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    if text.upper().startswith("TCGA-") and len(text) >= 12:
        return text[:12].upper()
    return re.sub(r"[\s_]+", "-", text).upper()


def detect_event_time_columns(df):
    time_candidates = ["OS_TIME_MONTHS", "OS_MONTHS", "time_months", "TIME_MONTHS", "OS_time", "time"]
    event_candidates = ["OS_EVENT", "event", "EVENT", "OS_event", "OS_STATUS_BINARY"]
    time_col = next((c for c in time_candidates if c in df.columns), None)
    event_col = next((c for c in event_candidates if c in df.columns), None)
    if time_col is None or event_col is None:
        raise ValueError(f"Cannot detect OS time/event columns from: {list(df.columns)[:30]}")
    return time_col, event_col


def coerce_event(series):
    if series.dtype == bool:
        return series.astype(int)
    lower = series.astype(str).str.lower()
    mapped = lower.map({
        "true": 1, "false": 0, "1": 1, "0": 0,
        "1:deceased": 1, "0:living": 0, "deceased": 1, "living": 0,
        "dead": 1, "alive": 0, "event": 1, "censored": 0,
    })
    numeric = pd.to_numeric(series, errors="coerce")
    out = mapped.where(mapped.notna(), numeric)
    return out.fillna(0).astype(int)


def make_surv_array(endpoint):
    if Surv is None:
        return None
    return Surv.from_arrays(endpoint["event"].astype(bool).to_numpy(), endpoint["time_months"].to_numpy(float))


def kaplan_meier_survival_at(times, event_observed, query):
    order = np.argsort(times)
    times_sorted = np.asarray(times, dtype=float)[order]
    events_sorted = np.asarray(event_observed, dtype=int)[order]
    unique_event_times = np.unique(times_sorted[events_sorted == 1])
    surv = 1.0
    surv_times, surv_values = [], []
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


def add_fixed_endpoint(df, tau_months, patient_col="PATIENT_ID"):
    result = df.copy()
    time_col, event_col = detect_event_time_columns(result)
    result[patient_col] = result[patient_col].map(normalize_patient_id)
    result["time_months"] = safe_numeric(result[time_col])
    result["event"] = coerce_event(result[event_col])
    result = result[result[patient_col].ne("") & result["time_months"].notna()].copy()
    result = result[result["time_months"] >= 0].copy()
    tau = int(tau_months)
    event_by_tau = (result["event"].eq(1) & (result["time_months"] <= tau_months)).astype(float)
    observed_status = ((result["event"].eq(1) & (result["time_months"] <= tau_months)) | (result["time_months"] > tau_months))
    result[f"death_by_{tau}m_observed"] = observed_status.astype(bool)
    result[f"death_by_{tau}m"] = np.where(observed_status, event_by_tau, np.nan)
    result[f"early_censored_before_{tau}m"] = (~observed_status).astype(bool)
    return result


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


def compute_ipcw_weights(endpoint, tau):
    df = endpoint.copy()
    tau_int = int(tau)
    censor_event = (df["event"].eq(0)).astype(int).to_numpy()
    times = df["time_months"].to_numpy(dtype=float)
    label_observed = df[f"death_by_{tau_int}m_observed"].to_numpy(dtype=bool)
    eval_time = np.minimum(times, tau)
    g_at_eval = kaplan_meier_survival_at(times, censor_event, eval_time)
    weights = np.zeros(len(df), dtype=float)
    weights[label_observed] = 1.0 / np.clip(g_at_eval[label_observed], EPS, None)
    df[f"ipcw_weight_{tau_int}m"] = weights
    df["ipcw_label_available"] = label_observed
    return df


def compute_pseudo_observations(endpoint, tau, max_n_exact=1200):
    df = endpoint.copy()
    tau_int = int(tau)
    times = df["time_months"].to_numpy(dtype=float)
    events = df["event"].to_numpy(dtype=int)
    n = len(df)
    f_all = km_cumulative_incidence_by_tau(times, events, tau)
    pseudo = np.full(n, np.nan, dtype=float)
    if n == 0:
        df[f"pseudo_risk_{tau_int}m"] = pseudo
        return df
    if n > max_n_exact:
        pseudo[:] = f_all
    else:
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            f_minus = km_cumulative_incidence_by_tau(times[mask], events[mask], tau)
            pseudo[i] = n * f_all - (n - 1) * f_minus
    df[f"pseudo_risk_{tau_int}m"] = pseudo
    df[f"pseudo_risk_{tau_int}m_clipped"] = np.clip(pseudo, 0.0, 1.0)
    return df


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
    if X_train.empty or len(X_train) < 10:
        return None
    y, w = censor_safe_binary_training(endpoint_train, tau_int)
    if y.nunique() < 2 or len(y) < 20:
        return None
    model = LogisticRegressionCV(
        Cs=10, cv=min(5, max(2, int(y.value_counts().min()))),
        penalty="elasticnet", solver="saga", l1_ratios=[0.05, 0.5, 0.95],
        scoring="roc_auc", max_iter=5000, random_state=random_seed, n_jobs=-1,
    )
    model.fit(X_train.loc[y.index], y, sample_weight=w)
    return model


def fit_timepoint_cox(X_train, endpoint_train, penalizer=0.1):
    if CoxPHFitter is None:
        return None
    if X_train.empty or len(X_train) < 10:
        return None
    data = X_train.copy()
    data["time_months"] = endpoint_train["time_months"].to_numpy(float)
    data["event"] = endpoint_train["event"].to_numpy(int)
    usable = data.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
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


def build_clinical_features(clinical_df):
    """Extract numeric clinical features for model input.

    Includes demographics, staging, molecular markers — same spirit as the
    individual timepoint scripts (1YearOS.py / 3YearsOS.py etc.).
    """
    numeric_candidates = [
        "AGE", "AGE_AT_DIAGNOSIS", "SEX",
        "AJCC_PATHOLOGIC_TUMOR_STAGE", "AJCC_STAGE_ENCODED", "AJCC_STAGING_EDITION",
        "PATH_T_STAGE", "PATH_N_STAGE", "PATH_M_STAGE",
        "RACE", "RADIATION_THERAPY",
        "ANEUPLOIDY_SCORE", "TMB_NONSYNONYMOUS",
        "MSI_SCORE_MANTIS", "MSI_SENSOR_SCORE", "TBL_SCORE",
        "SOMATIC_STATUS", "NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT",
        "HISTORY_NEOADJUVANT_TRTYN", "PERSON_NEOPLASM_CANCER_STATUS",
    ]
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
    feat_cols = []
    for col in numeric_candidates:
        if col in work.columns and col not in exclude:
            work[col] = pd.to_numeric(work[col], errors="coerce")
            if work[col].notna().sum() > len(work) * 0.3:
                feat_cols.append(col)
    # Add any additional numeric columns not already captured
    for col in work.columns:
        if col not in exclude and col not in feat_cols:
            numeric_series = pd.to_numeric(work[col], errors="coerce")
            if numeric_series.notna().sum() > len(work) * 0.5 and numeric_series.nunique() > 1:
                work[col] = numeric_series
                feat_cols.append(col)
    if not feat_cols:
        return pd.DataFrame(index=work["PATIENT_ID"].values)
    clinical_features = work.set_index("PATIENT_ID")[feat_cols].copy()
    return clinical_features


def load_input_data(timestamp):
    results_base = os.path.join(RESULTS_DIR, timestamp)
    preproc_dir = os.path.join(results_base, "01_preprocessing")
    gene_dir = os.path.join(results_base, "02_gene_features")
    clinical_path = os.path.join(preproc_dir, "tcga_os_clinical_endpoint_qc.tsv")
    split_path = os.path.join(preproc_dir, "tcga_train_internal_validation_split.tsv")
    if not os.path.exists(split_path):
        split_path = os.path.join(DATA_DIR, "survival_models", "tcga_train_internal_validation_split.tsv")
    clinical = read_tsv(clinical_path)
    split_df = read_tsv(split_path)
    # 加载基因表达矩阵（患者×基因）
    expr_path = os.path.join(preproc_dir, "gene_expression_curated.tsv")
    if os.path.exists(expr_path):
        expr_df = read_tsv(expr_path, index_col=0)
        expr_df = expr_df.apply(pd.to_numeric, errors="coerce")
    else:
        expr_df = None
    # 从 02 加载候选基因列表
    causal_path = os.path.join(gene_dir, "causal_priority_feature_table.tsv")
    candidate_genes = []
    if os.path.exists(causal_path):
        causal_df = read_tsv(causal_path)
        if "feature" in causal_df.columns:
            candidate_genes = causal_df["feature"].tolist()
    # 构建特征矩阵：临床特征 + 候选基因表达子集（与旧脚本1/2/3/UnlimitedOS.py保持一致）
    clinical_features = build_clinical_features(clinical)
    print(f"[04] Clinical features: {clinical_features.shape[1]} columns: {list(clinical_features.columns)}")
    feature_df = None
    if expr_df is not None and candidate_genes:
        available_genes = [g for g in candidate_genes if g in expr_df.columns]
        if available_genes:
            expr_subset = expr_df[available_genes].copy()
            # 合并临床特征 + 基因表达特征
            combined = pd.concat([clinical_features, expr_subset], axis=1)
            feature_df = combined
            print(f"[04] Feature matrix (clinical+expression): {feature_df.shape[0]} samples x {feature_df.shape[1]} features "
                  f"({clinical_features.shape[1]} clinical + {len(available_genes)} genes)")
    if feature_df is None:
        # 仅有临床特征作为 fallback
        feature_df = clinical_features
        print(f"[04] WARNING: No gene expression features; using clinical features only: {feature_df.shape}")
    # 修复 split 匹配：val 为任何非 "train" 的 split
    train_ids_raw = set(split_df.loc[split_df["split"] == "train", "PATIENT_ID"].astype(str))
    val_ids_raw = set(split_df.loc[split_df["split"] != "train", "PATIENT_ID"].astype(str))
    # 交集过滤：只保留在 clinical 中存在的 ID
    clinical_ids = set(clinical["PATIENT_ID"].astype(str))
    train_ids = train_ids_raw & clinical_ids
    val_ids = val_ids_raw & clinical_ids
    print(f"[04] Split: train_raw={len(train_ids_raw)}->filtered={len(train_ids)}, val_raw={len(val_ids_raw)}->filtered={len(val_ids)}")
    return clinical, feature_df, train_ids, val_ids


def prepare_features(clinical_ids, feature_df, train_ids):
    if feature_df is None:
        cols = [f"f_{i}" for i in range(10)]
        rng = np.random.default_rng(RANDOM_SEED)
        idx = list(clinical_ids)
        return pd.DataFrame(rng.standard_normal((len(idx), len(cols))), index=idx, columns=cols)
    common = feature_df.index.intersection(clinical_ids)
    X = feature_df.loc[common].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(X.median())


def run_fixed_timepoint(tp_name, tau_months, eval_times, timestamp, clinical, feature_df, train_ids, val_ids):
    tau_int = int(tau_months)
    print(f"\n{'='*60}\nProcessing {tp_name} (tau={tau_months}m)\n{'='*60}", flush=True)
    results_base = os.path.join(RESULTS_DIR, timestamp, "04_multi_timepoint", tp_name)
    ensure_dir(results_base)
    endpoint_full = add_fixed_endpoint(clinical, tau_months)
    endpoint_full = compute_ipcw_weights(endpoint_full, tau_months)
    endpoint_full = compute_pseudo_observations(endpoint_full, tau_months)
    endpoint_full.index = endpoint_full["PATIENT_ID"]
    endpoint_full.to_csv(os.path.join(results_base, f"tcga_{tau_int}m_endpoint_ipcw_pseudo.tsv"), sep="\t")
    train_ep = endpoint_full[endpoint_full["PATIENT_ID"].isin(train_ids)].copy()
    val_ep = endpoint_full[endpoint_full["PATIENT_ID"].isin(val_ids)].copy()
    X_all = prepare_features(endpoint_full["PATIENT_ID"].tolist(), feature_df, train_ids)
    X_train = X_all.loc[X_all.index.isin(train_ids)]
    X_val = X_all.loc[X_all.index.isin(val_ids)]
    train_common = X_train.index.intersection(train_ep.index)
    X_train, train_ep = X_train.loc[train_common], train_ep.loc[train_common]
    val_common = X_val.index.intersection(val_ep.index)
    X_val, val_ep = X_val.loc[val_common], val_ep.loc[val_common]
    models, risks_val, risks_train = {}, {}, {}
    log_model = fit_timepoint_logistic(X_train, train_ep, tau_int, RANDOM_SEED)
    if log_model is not None:
        models["IPCW_Logistic"] = log_model
        risks_val["IPCW_Logistic"] = predict_logistic_risk(log_model, X_val)
        risks_train["IPCW_Logistic"] = predict_logistic_risk(log_model, X_train)
    cox_model = fit_timepoint_cox(X_train, train_ep)
    if cox_model is not None:
        models["Cox_PH"] = cox_model
        risks_val["Cox_PH"] = predict_cox_risk(cox_model, X_val, tau_months)
        risks_train["Cox_PH"] = predict_cox_risk(cox_model, X_train, tau_months)
    coxnet_model = fit_timepoint_coxnet(X_train, train_ep)
    if coxnet_model is not None:
        models["Coxnet"] = coxnet_model
        risks_val["Coxnet"] = predict_sksurv_risk(coxnet_model, X_val, tau_months)
        risks_train["Coxnet"] = predict_sksurv_risk(coxnet_model, X_train, tau_months)
    rsf_model = fit_timepoint_rsf(X_train, train_ep, RANDOM_SEED)
    if rsf_model is not None:
        models["RSF"] = rsf_model
        risks_val["RSF"] = predict_sksurv_risk(rsf_model, X_val, tau_months)
        risks_train["RSF"] = predict_sksurv_risk(rsf_model, X_train, tau_months)
    if not models:
        print(f"[WARN] No models fitted for {tp_name}", flush=True)
        return {}
    comparison_rows, thresholds = [], {}
    for mname, rval in risks_val.items():
        rtrain = risks_train.get(mname)
        thresh = choose_training_threshold(train_ep, rtrain, tau_int) if rtrain is not None else None
        thresholds[mname] = thresh
        metrics = evaluate_fixed_tau(train_ep, val_ep, rval, tau_months, threshold=thresh)
        metrics["model"] = mname
        comparison_rows.append(metrics)
    comp_df = pd.DataFrame(comparison_rows)
    ranking_df = comp_df.copy()
    for col in ["harrell_cindex", "ipcw_auc", "ipcw_brier"]:
        if col in ranking_df.columns:
            asc = True if col == "ipcw_brier" else False
            ranking_df[f"{col}_rank"] = ranking_df[col].rank(ascending=asc, method="average")
    rank_cols = [c for c in ranking_df.columns if c.endswith("_rank")]
    if rank_cols:
        ranking_df["selection_score"] = ranking_df[rank_cols].sum(axis=1)
        best_model_name = ranking_df.loc[ranking_df["selection_score"].idxmin(), "model"]
    else:
        best_model_name = ranking_df.iloc[0]["model"]
    comp_df.to_csv(os.path.join(results_base, "model_comparison.tsv"), sep="\t", index=False)
    best_risk_val = risks_val[best_model_name]
    best_thresh = thresholds.get(best_model_name)
    risk_df = val_ep[["PATIENT_ID", "time_months", "event"]].copy()
    risk_df["predicted_risk"] = best_risk_val
    risk_df.to_csv(os.path.join(results_base, "risk_scores.tsv"), sep="\t", index=False)
    cohort_label = "TCGA_internal"
    stem = sanitize_filename(cohort_label) + "__" + sanitize_filename(best_model_name)
    plot_km_risk_strata(val_ep, best_risk_val, best_thresh, tau_months,
                        f"{cohort_label}: K-M risk strata ({tp_name})",
                        Path(results_base) / f"{stem}__km_risk_strata")
    plot_time_dependent_roc(val_ep, best_risk_val, tau_months,
                            f"{cohort_label}: time-dependent ROC ({tp_name})",
                            Path(results_base) / f"{stem}__time_dependent_roc", tau_int)
    plot_calibration_curve(val_ep, best_risk_val, tau_months,
                          f"{cohort_label}: calibration ({tp_name})",
                          Path(results_base) / f"{stem}__calibration", tau_int)
    best_metrics = comp_df[comp_df["model"] == best_model_name].iloc[0].to_dict() if not comp_df.empty else {}
    best_metrics["timepoint"] = tp_name
    best_metrics["best_model"] = best_model_name
    return best_metrics


def run_unlimited_timepoint(timestamp, clinical, feature_df, train_ids, val_ids):
    tp_name = "Unlimited"
    print(f"\n{'='*60}\nProcessing Unlimited OS\n{'='*60}", flush=True)
    results_base = os.path.join(RESULTS_DIR, timestamp, "04_multi_timepoint", tp_name)
    ensure_dir(results_base)
    endpoint_full = add_unlimited_endpoint(clinical)
    endpoint_full.index = endpoint_full["PATIENT_ID"]
    endpoint_full.to_csv(os.path.join(results_base, "tcga_unlimited_os_endpoint.tsv"), sep="\t")
    train_ep = endpoint_full[endpoint_full["PATIENT_ID"].isin(train_ids)].copy()
    val_ep = endpoint_full[endpoint_full["PATIENT_ID"].isin(val_ids)].copy()
    X_all = prepare_features(endpoint_full["PATIENT_ID"].tolist(), feature_df, train_ids)
    X_train = X_all.loc[X_all.index.isin(train_ids)]
    X_val = X_all.loc[X_all.index.isin(val_ids)]
    train_common = X_train.index.intersection(train_ep.index)
    X_train, train_ep = X_train.loc[train_common], train_ep.loc[train_common]
    val_common = X_val.index.intersection(val_ep.index)
    X_val, val_ep = X_val.loc[val_common], val_ep.loc[val_common]
    models, model_kinds, risks_val, risks_train = {}, {}, {}, {}
    cox_model = fit_timepoint_cox(X_train, train_ep)
    if cox_model is not None:
        models["Cox_PH"] = cox_model
        model_kinds["Cox_PH"] = "cox_lifelines"
        risks_val["Cox_PH"] = predict_cox_risk_unlimited(cox_model, X_val)
        risks_train["Cox_PH"] = predict_cox_risk_unlimited(cox_model, X_train)
    coxnet_model = fit_timepoint_coxnet(X_train, train_ep)
    if coxnet_model is not None:
        models["Coxnet"] = coxnet_model
        model_kinds["Coxnet"] = "sksurv"
        risks_val["Coxnet"] = predict_sksurv_risk_unlimited(coxnet_model, X_val)
        risks_train["Coxnet"] = predict_sksurv_risk_unlimited(coxnet_model, X_train)
    rsf_model = fit_timepoint_rsf(X_train, train_ep, RANDOM_SEED)
    if rsf_model is not None:
        models["RSF"] = rsf_model
        model_kinds["RSF"] = "sksurv"
        risks_val["RSF"] = predict_sksurv_risk_unlimited(rsf_model, X_val)
        risks_train["RSF"] = predict_sksurv_risk_unlimited(rsf_model, X_train)
    if not models:
        print(f"[WARN] No models fitted for {tp_name}", flush=True)
        return {}
    comparison_rows, thresholds = [], {}
    for mname, rval in risks_val.items():
        rtrain = risks_train.get(mname)
        thresh = choose_risk_threshold_unlimited(rtrain) if rtrain is not None else None
        thresholds[mname] = thresh
        metrics = evaluate_unlimited(train_ep, val_ep, rval, model=models[mname],
                                     model_kind=model_kinds.get(mname), x_eval=X_val, threshold=thresh)
        metrics["model"] = mname
        comparison_rows.append(metrics)
    comp_df = pd.DataFrame(comparison_rows)
    ranking_df = comp_df.copy()
    for col in ["harrell_cindex", "uno_cindex"]:
        if col in ranking_df.columns:
            ranking_df[f"{col}_rank"] = ranking_df[col].rank(ascending=False, method="average")
    if "integrated_brier_score" in ranking_df.columns:
        ranking_df["integrated_brier_score_rank"] = ranking_df["integrated_brier_score"].rank(ascending=True, method="average")
    rank_cols = [c for c in ranking_df.columns if c.endswith("_rank")]
    if rank_cols:
        ranking_df["selection_score"] = ranking_df[rank_cols].sum(axis=1)
        best_model_name = ranking_df.loc[ranking_df["selection_score"].idxmin(), "model"]
    else:
        best_model_name = ranking_df.iloc[0]["model"]
    comp_df.to_csv(os.path.join(results_base, "model_comparison.tsv"), sep="\t", index=False)
    best_risk_val = risks_val[best_model_name]
    best_thresh = thresholds.get(best_model_name)
    risk_df = val_ep[["PATIENT_ID", "time_months", "event"]].copy()
    risk_df["predicted_risk"] = best_risk_val
    risk_df.to_csv(os.path.join(results_base, "risk_scores.tsv"), sep="\t", index=False)
    cohort_label = "TCGA_internal"
    stem = sanitize_filename(cohort_label) + "__" + sanitize_filename(best_model_name)
    plot_km_risk_strata(val_ep, best_risk_val, best_thresh, None,
                        f"{cohort_label}: K-M risk strata (Unlimited)",
                        Path(results_base) / f"{stem}__km_risk_strata", is_unlimited=True)
    plot_risk_distribution(val_ep, best_risk_val,
                           f"{cohort_label}: risk by event status (Unlimited)",
                           Path(results_base) / f"{stem}__risk_distribution")
    plot_followup_vs_risk(val_ep, best_risk_val,
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
        print(f"\nTimepoint comparison saved to {out_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()
    timestamp = args.timestamp
    print(f"[04_multi_timepoint] timestamp={timestamp}", flush=True)
    print(f"SCRIPT_DIR = {SCRIPT_DIR}", flush=True)
    print(f"DATA_DIR   = {DATA_DIR}", flush=True)
    print(f"RESULTS_DIR= {RESULTS_DIR}", flush=True)
    clinical, feature_df, train_ids, val_ids = load_input_data(timestamp)
    print(f"Loaded clinical: {clinical.shape}, features: {feature_df.shape if feature_df is not None else 'None'}", flush=True)
    print(f"Train IDs: {len(train_ids)}, Val IDs: {len(val_ids)}", flush=True)
    all_results = []
    for tp_name, tp_cfg in TIMEPOINTS.items():
        tau = tp_cfg["tau_months"]
        eval_times = tp_cfg["eval_times"]
        try:
            if tau is None:
                res = run_unlimited_timepoint(timestamp, clinical, feature_df, train_ids, val_ids)
            else:
                res = run_fixed_timepoint(tp_name, tau, eval_times, timestamp, clinical, feature_df, train_ids, val_ids)
            all_results.append(res)
        except Exception as exc:
            print(f"[ERROR] {tp_name}: {exc}", flush=True)
            traceback.print_exc()
            all_results.append({"timepoint": tp_name, "error": str(exc)})
    generate_timepoint_comparison(all_results, timestamp)
    print("\n[04_multi_timepoint] DONE", flush=True)


if __name__ == "__main__":
    main()
