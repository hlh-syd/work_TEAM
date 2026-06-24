from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from _pipeline_core import (
    FIXED_TAU_MONTHS, RANDOM_SEED, EPS,
    compute_ipcw_weights, compute_pseudo_observations,
    compute_ipcw_weights_tagged, compute_pseudo_observations_tagged,
    evaluate_binary_and_survival_metrics, evaluate_survival_metrics,
    kaplan_meier_survival_at, km_cumulative_incidence_by_tau,
    add_os_endpoint_full_followup, add_fixed_endpoint,
    bh_fdr, numeric_series,
    plot_volcano,
    bootstrap_metric,
    rank_inverse_normal,
    StratifiedKFold, train_test_split,
    LogisticRegression, LogisticRegressionCV,
    RandomForestRegressor, RandomForestClassifier,
)

def _legacy_causal_aipw_binary_exposure(
    X: pd.DataFrame,
    exposure: pd.Series,
    outcome: pd.Series,
    random_seed: int,
) -> Dict[str, float]:
    """[DEPRECATED] Standard AIPW for binary outcome — not appropriate for censored OS data.

    Retained for backward comparison. Use causal_rmst_doubly_robust instead.
    """
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


# Backward-compatible alias
causal_aipw_binary_exposure = _legacy_causal_aipw_binary_exposure


def compute_rmst(
    times: np.ndarray,
    events: np.ndarray,
    tau: float = FIXED_TAU_MONTHS,
) -> float:
    """Compute Restricted Mean Survival Time (RMST) via KM area-under-curve up to tau."""
    order = np.argsort(times)
    t_sorted = times[order]
    e_sorted = events[order]

    unique_times = np.unique(t_sorted)
    unique_times = unique_times[unique_times <= tau]

    surv = 1.0
    rmst = 0.0
    prev_t = 0.0

    for t in unique_times:
        rmst += surv * (t - prev_t)
        at_risk = np.sum(t_sorted >= t)
        d = np.sum((t_sorted == t) & (e_sorted == 1))
        if at_risk > 0:
            surv *= (1.0 - d / at_risk)
        prev_t = t

    # Last segment to tau
    rmst += surv * (tau - prev_t)
    return float(rmst)


def causal_rmst_doubly_robust(
    X: pd.DataFrame,
    exposure: pd.Series,
    time_months: np.ndarray,
    event: np.ndarray,
    tau: float = FIXED_TAU_MONTHS,
    random_seed: int = RANDOM_SEED,
) -> Dict[str, float]:
    """RMST-based doubly robust causal effect estimator (Tsiatis 2008, Zhao & Tian 2014).

    Estimator:
    psi_hat = n^-1 sum_i [ m_hat(X_i,1) - m_hat(X_i,0)
        + A_i/e_hat(X_i) * {T_capped_i - m_hat(X_i,1)}
        - (1-A_i)/(1-e_hat(X_i)) * {T_capped_i - m_hat(X_i,0)} ]

    where T_capped = min(T, tau).
    """
    data = pd.concat([X, exposure.rename("A")], axis=1).dropna()
    if len(data) < 30 or data["A"].nunique() < 2:
        return {"ate_rmst": np.nan, "ate_rmst_se": np.nan, "p_value": np.nan}

    A = data["A"].astype(int).to_numpy()
    W = data.drop(columns=["A"])
    common_idx = data.index

    # Align time/event arrays to the filtered data index
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
        return {"ate_rmst": np.nan, "ate_rmst_se": np.nan, "p_value": np.nan}

    # Observed restricted time: T_capped_i = min(T_i, tau)
    T_capped = np.minimum(T, tau)

    # Cross-fitted propensity score and outcome regression
    n_folds = min(5, max(2, int(np.bincount(A).min())))
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    e_hat = np.zeros(len(data), dtype=float)
    m1_hat = np.zeros(len(data), dtype=float)
    m0_hat = np.zeros(len(data), dtype=float)

    for tr, te in folds.split(W, A):
        # Propensity score
        prop = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=random_seed)
        prop.fit(W.iloc[tr], A[tr])
        e_hat[te] = prop.predict_proba(W.iloc[te])[:, 1]

        # Outcome regression m(X, a) using RF on T_capped
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

    # Doubly robust estimator
    dr = (m1_hat - m0_hat
          + A * (T_capped - m1_hat) / e_hat
          - (1 - A) * (T_capped - m0_hat) / (1 - e_hat))

    ate = float(np.mean(dr))
    se = float(np.std(dr, ddof=1) / np.sqrt(len(dr)))
    p = float(2 * stats.norm.sf(abs(ate / se))) if se > 0 else np.nan

    return {
        "ate_rmst": ate,
        "ate_rmst_se": se,
        "p_value": p,
        "rmst_exposed": float(np.mean(T_capped[A == 1])),
        "rmst_unexposed": float(np.mean(T_capped[A == 0])),
    }


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
    time_months = endpoint_train.reindex(common)["time_months"].to_numpy(dtype=float)
    events = endpoint_train.reindex(common)["event"].to_numpy(dtype=int)
    outcome_time = pd.Series(time_months, index=common)
    rows: List[Dict[str, Any]] = []
    for feature in list(feature_matrix.columns)[:max_features]:
        x = feature_matrix.reindex(common)[feature]
        if x.notna().sum() < 30 or x.nunique(dropna=True) < 4:
            continue
        exposure_binary = (x > x.median()).astype(int)
        # ★ RMST doubly robust estimator (replaces AIPW on binary outcome)
        rmst_dr = causal_rmst_doubly_robust(
            X_adj, exposure_binary, time_months, events,
            tau=FIXED_TAU_MONTHS, random_seed=random_seed,
        )
        # CATE and dose-response use survival time as outcome
        cate = causal_cate_summary(X_adj, exposure_binary, outcome_time, random_seed)
        dose = dose_response_summary(x, outcome_time)
        corr = stats.spearmanr(
            x.fillna(x.median()),
            outcome_time.fillna(outcome_time.median()),
        ).correlation
        rows.append(
            {
                "feature": feature,
                "spearman_with_survival_time": float(corr) if np.isfinite(corr) else np.nan,
                **rmst_dr,
                **cate,
                **dose,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    # ★ Rank by RMST ATE (replaces binary ATE)
    table["ate_abs"] = table["ate_rmst"].abs()
    table["ate_rank"] = table["ate_abs"].rank(ascending=False, method="average")
    table["p_rank"] = table["p_value"].rank(ascending=True, method="average")
    table["dose_rank"] = table["dose_response_slope"].abs().rank(ascending=False, method="average")
    table["causal_priority_score"] = (
        -table["ate_rank"].fillna(table["ate_rank"].max() + 1)
        -0.5 * table["p_rank"].fillna(table["p_rank"].max() + 1)
        -0.25 * table["dose_rank"].fillna(table["dose_rank"].max() + 1)
    )
    return table.sort_values("causal_priority_score", ascending=False)


def make_surv_array(endpoint: pd.DataFrame) -> Any:
    if Surv is None:
        return None
    return Surv.from_arrays(endpoint["event"].astype(bool).to_numpy(), endpoint["time_months"].to_numpy(float))


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
        l1_ratios=[0.1, 0.5, 0.9],
        scoring="roc_auc",
        max_iter=5000,
        random_state=random_seed,
        n_jobs=-1,
    )
    model.fit(X_train.loc[y.index], y, sample_weight=w)
    return model


def censor_safe_binary_training(
    endpoint: pd.DataFrame,
    tau: float = FIXED_TAU_MONTHS,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Create censoring-aware binary labels for GAN training.

    Returns:
        y_obs: Binary label series (1=event observed, 0=censored/survived), indexed by PATIENT_ID
        aux_info: Auxiliary DataFrame with censoring indicators
    """
    ep = endpoint.copy()
    if "PATIENT_ID" not in ep.columns:
        ep = ep.reset_index()
    ep["PATIENT_ID"] = ep["PATIENT_ID"].astype(str)
    ep = ep.set_index("PATIENT_ID")

    # Determine binary outcome: prefer pre-computed columns if available
    if "os_death_observed" in ep.columns:
        y_obs = ep["os_death_observed"].fillna(False).astype(int)
    elif "ipcw_label_available" in ep.columns:
        # Use IPCW-available labels
        available = ep["ipcw_label_available"].fillna(False).astype(bool)
        if "os_death" in ep.columns:
            y_obs = pd.Series(0, index=ep.index, dtype=int)
            y_obs[available] = ep.loc[available, "os_death"].fillna(0).astype(int)
        else:
            y_obs = available.astype(int)
    elif "event" in ep.columns and "time_months" in ep.columns:
        # Compute from raw survival data
        time_col = pd.to_numeric(ep["time_months"], errors="coerce")
        event_col = ep["event"].fillna(0).astype(int)
        # Event observed if: event occurred AND time >= tau (or censored before tau)
        y_obs = ((event_col == 1) | (time_col < tau)).astype(int)
    else:
        # Fallback: assume all labels observed
        y_obs = pd.Series(1, index=ep.index, dtype=int)

    aux_info = pd.DataFrame({
        "PATIENT_ID": ep.index,
        "y_obs": y_obs.values,
    }).set_index("PATIENT_ID")

    return y_obs, aux_info


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
        return {"auc_os_observed": float("nan"), "average_precision_os": float("nan"), "brier_os_ipcw": float("nan")}
    min_class = int(y_real.value_counts().min())
    n_splits = min(5, max(2, min_class))
    if n_splits < 2:
        return {"auc_os_observed": float("nan"), "average_precision_os": float("nan"), "brier_os_ipcw": float("nan")}
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
        y_val = endpoint_real.loc[fold_val_ids, "os_death"].astype(int).to_numpy()
        w_val = endpoint_real.loc[fold_val_ids, "ipcw_weight_os"].replace(0, np.nan).fillna(1.0).to_numpy(float)
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
        "auc_os_observed": float(np.nanmean(auc_scores)) if auc_scores else float("nan"),
        "average_precision_os": float(np.nanmean(ap_scores)) if ap_scores else float("nan"),
        "brier_os_ipcw": float(np.nanmean(brier_scores)) if brier_scores else float("nan"),
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
    return float(metrics.get("auc_os_observed", float("nan")))


def train_only_cv_metrics_for_real_logistic(
    X_real: pd.DataFrame,
    endpoint_real: pd.DataFrame,
    random_seed: int,
) -> Dict[str, float]:

    y_real, _ = censor_safe_binary_training(endpoint_real)
    if y_real.nunique() < 2 or len(y_real) < 40:
        return {"auc_os_observed": float("nan"), "average_precision_os": float("nan"), "brier_os_ipcw": float("nan")}
    min_class = int(y_real.value_counts().min())
    n_splits = min(5, max(2, min_class))
    if n_splits < 2:
        return {"auc_os_observed": float("nan"), "average_precision_os": float("nan"), "brier_os_ipcw": float("nan")}
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
        y_val = endpoint_real.loc[fold_val_ids, "os_death"].astype(int).to_numpy()
        w_val = endpoint_real.loc[fold_val_ids, "ipcw_weight_os"].replace(0, np.nan).fillna(1.0).to_numpy(float)
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
        "auc_os_observed": float(np.nanmean(auc_scores)) if auc_scores else float("nan"),
        "average_precision_os": float(np.nanmean(ap_scores)) if ap_scores else float("nan"),
        "brier_os_ipcw": float(np.nanmean(brier_scores)) if brier_scores else float("nan"),
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
    except Exception as exc:
        warnings.warn(f"fit_lifelines_cox failed (penalizer={penalizer}): {exc}", stacklevel=2)
        return None


def predict_lifelines_risk(model: Any, X: pd.DataFrame, tau: float = FIXED_TAU_MONTHS) -> np.ndarray:
    if model is None:
        return np.full(len(X), np.nan)
    cols = [c for c in model.params_.index if c in X.columns]
    Xp = X.reindex(columns=cols, fill_value=0)
    try:
        surv = model.predict_survival_function(Xp, times=[tau]).T.iloc[:, 0]
        return 1.0 - surv.to_numpy(float)
    except Exception as exc:
        warnings.warn(f"predict_lifelines_risk: survival_function failed, falling back to partial_hazard: {exc}", stacklevel=2)
        partial = model.predict_partial_hazard(Xp)
        return np.asarray(partial, dtype=float).ravel()


def uno_cindex_fallback(endpoint: pd.DataFrame, risk: np.ndarray) -> float:
    """Harrell C-index via lifelines (replaces O(n²) nested loop)."""
    from lifelines.utils import concordance_index as _ll_ci
    valid = np.isfinite(risk)
    t = endpoint.loc[valid, "time_months"].to_numpy(dtype=float)
    e = endpoint.loc[valid, "event"].to_numpy(dtype=int)
    r = risk[valid]
    try:
        return float(_ll_ci(t, -r, e))
    except Exception as exc:
        warnings.warn(f"uno_cindex_fallback: concordance_index failed: {exc}", stacklevel=2)
        return float("nan")


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

