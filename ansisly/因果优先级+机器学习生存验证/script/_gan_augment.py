from __future__ import annotations

import dataclasses
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from _pipeline_core import (
    RANDOM_SEED, RANDOM_SEED_GAN, EPS, FIXED_TAU_MONTHS,
    GANPipelineConfig, AuditLog, QuietAuditLog,
    kaplan_meier_survival_at,
    compute_ipcw_weights, add_os_endpoint_full_followup,
    add_fixed_endpoint, evaluate_binary_and_survival_metrics,
    evaluate_survival_metrics,
    extract_event_weighted_training,
    observed_binary_for_plot, choose_training_threshold,
    bootstrap_metric,
)

from _causal_infer import (
    fit_lifelines_cox, predict_lifelines_risk,
    risk_from_model, make_surv_array,
    save_current_figure, sanitize_filename,
    plot_km_risk_strata, uno_cindex_fallback,
    bootstrap_ci,
    fit_weighted_logistic,
    train_only_cv_metrics_for_augmented_logistic,
    train_only_cv_auc_for_augmented_logistic,
    train_only_cv_metrics_for_real_logistic,
    censor_safe_binary_training,
)


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
    "os_death",
    "event",
    "log1p_time_months",
    "pseudo_risk_os",
    "ipcw_weight_os",
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
            labels = condition.reindex(data.index)["os_death"].round().clip(0, 1).astype(int).to_numpy()
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
    death = ep.get("os_death", pd.Series(0.0, index=ep.index)).astype(float)
    event = ep.get("event", death).astype(float)
    time_months = ep.get("time_months", pd.Series(OS_RISK_TIME_MONTHS, index=ep.index)).astype(float)
    pseudo = ep.get("pseudo_risk_os", death).astype(float)
    weight = ep.get("ipcw_weight_os", pd.Series(1.0, index=ep.index)).astype(float)
    cond = pd.DataFrame(index=ep.index)
    cond["os_death"] = death.fillna(0.0).clip(0.0, 1.0)
    cond["event"] = event.fillna(cond["os_death"]).clip(0.0, 1.0)
    clean_time = time_months.replace([np.inf, -np.inf], np.nan).fillna(OS_RISK_TIME_MONTHS).clip(lower=0.0)
    cond["log1p_time_months"] = np.log1p(clean_time)
    cond["pseudo_risk_os"] = pseudo.replace([np.inf, -np.inf], np.nan).fillna(cond["os_death"]).clip(0.0, 1.0)
    cond["ipcw_weight_os"] = weight.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=EPS)
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
        cond["os_death"] = 1.0
        cond["event"] = 1.0
        cond["pseudo_risk_os"] = cond["pseudo_risk_os"].clip(lower=0.5)
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
    if sampling_strategy == "risk_stratified" and "pseudo_risk_os" in endpoint_train:
        pseudo = endpoint_train.loc[train_idx, "pseudo_risk_os"].astype(float).replace([np.inf, -np.inf], np.nan)
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
            y_ae = endpoint.reindex(data.index)["os_death"].fillna(0).astype(float).to_numpy()
            w_ae = endpoint.reindex(data.index)["ipcw_weight_os"].replace(0, np.nan).fillna(1.0).astype(float).to_numpy()
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
        audit.add("gan_augmentation", "SKIP", "Insufficient observed 36-month labels in TCGA_train_real.")
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
                    sample_condition = sampled_cond_df["os_death"].to_numpy(int)
                X_syn_candidate = gan_model.sample(n_syn_candidate, sample_condition)
                if X_syn_candidate.empty:
                    continue


                syn_event_mask = sampled_cond_df["os_death"].to_numpy(int)[: len(X_syn_candidate)] == 1
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
    endpoint_syn["os_death"] = sampled_conditions["os_death"].round().clip(0, 1).astype(float).to_numpy()
    endpoint_syn["os_death_observed"] = endpoint_syn["event"].eq(1)  # ★ only event patients are observed
    endpoint_syn["early_censored_before_os"] = False
    endpoint_syn["ipcw_label_available"] = endpoint_syn["event"].eq(1)  # ★ only event patients get IPCW weights
    endpoint_syn["ipcw_weight_os"] = sampled_conditions["ipcw_weight_os"].replace(0, np.nan).fillna(1.0).astype(float).to_numpy()
    endpoint_syn["pseudo_risk_os_raw"] = sampled_conditions["pseudo_risk_os"].clip(0.02, 0.98).astype(float).to_numpy()
    endpoint_syn["pseudo_risk_os"] = endpoint_syn["pseudo_risk_os_raw"].clip(0.02, 0.98)


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
            cv_metrics = {"auc_os_observed": float("nan"), "average_precision_os": float("nan"), "brier_os_ipcw": float("nan")}
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
            auc_gain = float(cv_metrics.get("auc_os_observed", float("nan"))) - float(baseline_metrics.get("auc_os_observed", float("nan")))
            ap_delta = float(cv_metrics.get("average_precision_os", float("nan"))) - float(baseline_metrics.get("average_precision_os", float("nan")))
            brier_delta = float(cv_metrics.get("brier_os_ipcw", float("nan"))) - float(baseline_metrics.get("brier_os_ipcw", float("nan")))


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
                "cv_auc_os_observed": cv_metrics.get("auc_os_observed", float("nan")),
                "cv_average_precision_os": cv_metrics.get("average_precision_os", float("nan")),
                "cv_brier_os_ipcw": cv_metrics.get("brier_os_ipcw", float("nan")),
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
                f"train-only CV AUC={candidate['cv_auc_os_observed']:.4f}, "
                f"AP={candidate['cv_average_precision_os']:.4f}, Brier={candidate['cv_brier_os_ipcw']:.4f}.",
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
        brier = float(candidate.get("cv_brier_os_ipcw", float("nan")))
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
            "train_only_cv_auc_os_observed": selected["cv_auc_os_observed"],
            "train_only_cv_average_precision_os": selected["cv_average_precision_os"],
            "train_only_cv_brier_os_ipcw": selected["cv_brier_os_ipcw"],
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
        f"train-only CV AUC={selected['cv_auc_os_observed']:.4f}, "
        f"AP={selected['cv_average_precision_os']:.4f}, Brier={selected['cv_brier_os_ipcw']:.4f}.",
        selected_sampling_strategy=selected["sampling_strategy"],
        selected_aug_ratio=selected["ratio"],
        selected_method=selected.get("method", ""),
        n_synthetic=selected["n_synthetic"],
        qc_passed=selected["qc_passed"],
        qc_good_count=selected["qc_good_count"],
        train_only_cv_auc_os_observed=selected["cv_auc_os_observed"],
        train_only_cv_average_precision_os=selected["cv_average_precision_os"],
        train_only_cv_brier_os_ipcw=selected["cv_brier_os_ipcw"],
    )
    return selected["X_aug"], selected["endpoint_aug"], selected_report

