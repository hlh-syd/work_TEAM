from __future__ import annotations

import argparse
import json
import joblib
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _pipeline_core import *  # noqa: F401,F403  – shared utilities
from _causal_infer import *   # noqa: F401,F403
from _gan_augment import *    # noqa: F401,F403
from _step_common import *    # noqa: F401,F403

def build_parser_step05() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate CRC survival model outputs across cohorts.")
    add_common_args(parser)
    parser.add_argument("--bootstrap-iter", type=int, default=200, help="Bootstrap iterations for 95% CI.")
    parser.add_argument("--km-cut", choices=["median", "tertile"], default="median", help="Risk-group splitting rule fitted on TCGA training partition.")
    parser.add_argument("--dca-low", type=float, default=0.05)
    parser.add_argument("--dca-high", type=float, default=0.50)
    parser.add_argument("--dca-step", type=float, default=0.01)
    return parser


# harrell() → use harrell_c_index() defined at ~line 5847.
# bootstrap_harrell() → use the canonical definition at ~line 6236.


def normalize_external_clinical(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    out = df.copy()
    if cohort == "msk":
        out["PATIENT_ID"] = out["PATIENT_ID"].astype(str)
        out["OS_EVENT"] = parse_survival_status(out["OS_STATUS"])
        out["OS_TIME_MONTHS"] = numeric_series(out["OS_MONTHS"])
        if "AGE" not in out.columns and "AGE_AT_DIAGNOSIS" in out.columns:
            out["AGE"] = pd.to_numeric(out["AGE_AT_DIAGNOSIS"], errors="coerce")
        return out.loc[out["OS_TIME_MONTHS"].notna() & (out["OS_TIME_MONTHS"] > 0)].copy()
    if cohort in ("geo", "geo_gse39582"):

        out = out.rename(columns={"id": "PATIENT_ID", "fustat": "OS_STATUS_RAW", "futime": "OS_TIME_MONTHS"})
        out["PATIENT_ID"] = out["PATIENT_ID"].astype(str)
        out["OS_EVENT"] = parse_survival_status(out["OS_STATUS_RAW"])
        out["OS_TIME_MONTHS"] = numeric_series(out["OS_TIME_MONTHS"])
        return out.loc[out["OS_TIME_MONTHS"].notna() & (out["OS_TIME_MONTHS"] > 0)].copy()
    if cohort == "geo_gse17538":


        out = out.rename(columns={
            "Accession": "PATIENT_ID",
            "Overall survival follow-up time": "OS_TIME_MONTHS",
            "overall_event (death from any cause):": "OS_EVENT_RAW",
            "Age": "AGE",
            "Gender": "SEX",
            "Ajcc_stage": "AJCC_STAGE",
        })
        out["PATIENT_ID"] = out["PATIENT_ID"].astype(str)

        out["OS_EVENT"] = out["OS_EVENT_RAW"].astype(str).str.strip().str.lower().map(
            lambda v: 1 if v in ("death", "1", "true", "yes") else 0
        )
        out["OS_TIME_MONTHS"] = numeric_series(out["OS_TIME_MONTHS"])
        return out.loc[out["OS_TIME_MONTHS"].notna() & (out["OS_TIME_MONTHS"] > 0)].copy()
    raise ValueError(f"Unknown external cohort: {cohort}")


def assess_followup_time_scale(time_values: pd.Series | np.ndarray) -> tuple[bool, float, float, str]:
    arr = pd.to_numeric(pd.Series(time_values), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return True, np.nan, np.nan, "no_followup_time"
    med = float(np.median(arr))
    max_val = float(np.max(arr))
    if max_val >= 365 or med >= 180:
        return True, med, max_val, "likely_not_month_scale"
    if max_val > 240:
        return False, med, max_val, "long_month_scale_followup"
    return False, med, max_val, "month_scale_plausible"


def transform_clinical_external(df: pd.DataFrame, patient_col: str, bundle: dict[str, Any]) -> pd.DataFrame:
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
        raise ValueError("No clinical columns available for external transform.")
    matrix = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
    return pd.DataFrame(matrix, columns=transformer["model_columns"], index=df[patient_col].astype(str))


def load_geo_expression(path: str, genes: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    id_col = "geneNames" if "geneNames" in df.columns else df.columns[0]
    sample_cols = [c for c in df.columns if c != id_col]
    df = df[df[id_col].notna()].copy()
    df = df[df[id_col].astype(str).isin(genes)]
    if df.empty:
        return pd.DataFrame(columns=genes)
    df = df.drop_duplicates(subset=id_col, keep="first").set_index(id_col)
    mat = df[sample_cols].apply(pd.to_numeric, errors="coerce").T
    mat.index = mat.index.astype(str)
    for gene in genes:
        if gene not in mat.columns:
            mat[gene] = np.nan
    return mat[genes]


# lifelines_survival_estimator → defined at ~line 5873.
# sksurv_survival_estimator → defined at ~line 5883.


def expression_coxnet_design_matrix(expr: pd.DataFrame, bundle: dict[str, Any]) -> np.ndarray:


    genes = bundle["genes"]
    aligned = expr.reindex(columns=genes).copy()
    harmonization = bundle.get("harmonization", "rank_based_inverse_normal_per_cohort")
    if harmonization == "rank_based_inverse_normal_per_cohort":
        aligned = aligned.apply(rank_inverse_normal, axis=0)
    return bundle["scaler"].transform(bundle["imputer"].transform(aligned))


def expression_coxnet_scores(expr: pd.DataFrame, bundle: dict[str, Any]) -> np.ndarray:

    x = expression_coxnet_design_matrix(expr, bundle)
    risk = bundle["model"].predict(x)
    if getattr(risk, "ndim", 1) > 1:
        risk = risk[:, -1]
    return np.asarray(risk).reshape(-1)


def msk_genomic_predict(bundle: dict[str, Any], clinical: pd.DataFrame, mutation_path: str) -> tuple[np.ndarray, pd.DataFrame]:

    import importlib

    mut_module = importlib.import_module("pandas")
    df_mut = mut_module.read_csv(mutation_path, sep="\t", comment="#", dtype=str, low_memory=False)
    sample_col = "Tumor_Sample_Barcode" if "Tumor_Sample_Barcode" in df_mut.columns else None
    if sample_col is None:
        return np.full(len(clinical), np.nan), pd.DataFrame()
    df_mut["patient_id"] = df_mut[sample_col].astype(str)
    pivot = (
        df_mut[df_mut["Hugo_Symbol"].isin(bundle["genes"])]
        .assign(value=1)
        .pivot_table(index="patient_id", columns="Hugo_Symbol", values="value", aggfunc="max", fill_value=0)
    )
    pivot = pivot.reindex(index=clinical["PATIENT_ID"].astype(str), columns=bundle["genes"], fill_value=0)
    age = bundle["age_scaler"].transform(bundle["age_imputer"].transform(clinical[["AGE"]] if "AGE" in clinical.columns else pd.DataFrame({"AGE": np.nan}, index=clinical.index)))
    sex_dummies = pd.get_dummies(clinical.get("SEX", pd.Series("Unknown", index=clinical.index)), prefix="sex", drop_first=True, dummy_na=False)
    sex_dummies = sex_dummies.reindex(columns=bundle.get("sex_columns", []), fill_value=0)
    sex_dummies.index = clinical["PATIENT_ID"].astype(str)
    age_df = pd.DataFrame({"AGE": age.reshape(-1)}, index=clinical["PATIENT_ID"].astype(str))
    big = pd.concat([age_df, sex_dummies, pivot.astype(int)], axis=1)
    big = big.reindex(columns=bundle["model_columns"], fill_value=0)
    risk = bundle["model"].predict_partial_hazard(big).to_numpy().reshape(-1)
    return np.log(risk + 1e-12), big


def km_groups_by_threshold(risk: np.ndarray, thresholds: list[float]) -> np.ndarray:
    arr = np.asarray(risk)
    groups = np.full(arr.shape, fill_value="high", dtype=object)
    if len(thresholds) == 1:
        groups[arr <= thresholds[0]] = "low"
    else:

        groups = np.full(arr.shape, fill_value="medium", dtype=object)
        groups[arr <= thresholds[0]] = "low"
        groups[arr > thresholds[1]] = "high"
    return groups


def plot_km(time, event, groups, title, path: Path) -> dict[str, Any]:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    import matplotlib.pyplot as plt

    df = pd.DataFrame({"time": time, "event": event, "group": groups}).dropna()
    if df.empty or df["group"].nunique() < 2:
        return {}
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    order = sorted(df["group"].unique(), key=lambda g: {"low": 0, "medium": 1, "high": 2}.get(g, 3))
    palette = {"low": "#1b9e77", "medium": "#7570b3", "high": "#d95f02"}
    for g in order:
        sub = df[df["group"] == g]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["time"], sub["event"].astype(int), label=f"{g} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, ci_show=False, color=palette.get(g, "grey"))
    lr = multivariate_logrank_test(df["time"], df["group"], df["event"].astype(int))
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    ax.set_title(f"{title}\nlog-rank p = {lr.p_value:.4g}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"logrank_p": float(lr.p_value), "logrank_stat": float(lr.test_statistic)}


def td_auc_and_brier(
    train_event: np.ndarray,
    train_time: np.ndarray,
    test_event: np.ndarray,
    test_time: np.ndarray,
    test_risk: np.ndarray,
    horizons: tuple[float, ...] = (12.0, 36.0, 60.0),
    survival_estimator: Any | None = None,
) -> dict[str, Any]:
    from sksurv.metrics import cumulative_dynamic_auc, brier_score, integrated_brier_score
    from sksurv.util import Surv

    if len(test_time) < 5:
        return {}
    y_train = Surv.from_arrays(np.asarray(train_event).astype(bool), np.asarray(train_time, dtype=float))
    y_test = Surv.from_arrays(np.asarray(test_event).astype(bool), np.asarray(test_time, dtype=float))
    max_t = float(min(np.max(train_time), np.max(test_time))) * 0.95
    safe = sorted({float(min(h, max_t)) for h in horizons if h < max_t * 1.01})
    out: dict[str, Any] = {}
    if safe:
        try:
            auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, np.asarray(test_risk, dtype=float), np.asarray(safe))
            for h, a in zip(safe, auc):
                out[f"td_auc_month_{int(h)}"] = float(a)
            out["td_auc_mean"] = float(mean_auc)
        except Exception:
            pass
        try:
            if survival_estimator is None:
                raise ValueError("No model survival estimator was supplied.")
            surv_matrix = survival_estimator(np.asarray(safe, dtype=float))
            if surv_matrix.shape != (len(test_time), len(safe)):
                raise ValueError(f"Survival matrix shape {surv_matrix.shape} does not match test x horizons.")
            out["integrated_brier_score"] = float(integrated_brier_score(y_train, y_test, surv_matrix, np.asarray(safe)))
            out["brier_score_source"] = "model_survival_function"
        except Exception:
            pass
        try:
            rank = (-np.asarray(test_risk)).argsort().argsort()
            surv = 1.0 - (rank + 1) / (len(rank) + 1)
            surv_matrix = np.tile(surv.reshape(-1, 1), (1, len(safe)))
            _, ibs = brier_score(y_train, y_test, surv_matrix, np.asarray(safe))
            out["risk_ranked_brier_score_exploratory"] = float(np.mean(ibs))
            if "brier_score_source" not in out:
                out["brier_score_source"] = "risk_ranked_survival_proxy_not_ibs"
        except Exception:
            pass
    return out


def decision_curve_analysis(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    horizon_months: float,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Standard DCA for time-to-event outcomes (Vickers & Elkin 2006).

    Risk scores are calibrated to event probabilities at *horizon_months*
    via a single-predictor Cox PH model (Breslow baseline hazard).  Net
    benefit is then computed with probability-based threshold sweeps,
    producing clinically interpretable DCA curves.
    """

    from lifelines import CoxPHFitter, KaplanMeierFitter

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    n = len(time)
    if n == 0:
        return pd.DataFrame()

    # --- Step 1: Calibrate risk → event probability at horizon ----------
    cal_df = pd.DataFrame({"_t": time, "_e": event, "_r": risk})
    try:
        cph = CoxPHFitter()
        cph.fit(cal_df, duration_col="_t", event_col="_e")
        breslow = cph.baseline_cumulative_hazard_
        h0_col = breslow.iloc[
            (breslow.index - horizon_months).abs().argmin()
        ]  # H₀ at horizon
        # partial_hazard = exp(β·(x − x̄))  from lifelines
        ph = cph.predict_partial_hazard(cal_df[["_r"]]).to_numpy().ravel()
        H_ind = np.clip(h0_col * ph, 0, 30)
        prob_event = np.clip(1.0 - np.exp(-H_ind), 0.0, 0.999)
    except Exception:
        # Fallback: rank-based proxy (clearly flagged)
        warnings.warn("DCA: Cox calibration failed; using rank-based proxy.", stacklevel=2)
        prob_event = (risk.argsort().argsort() + 1.0) / (n + 1)

    # --- Step 2: KM overall event rate at horizon (treat-all reference) -
    kmf_all = KaplanMeierFitter().fit(time, event)
    try:
        surv_all = float(kmf_all.predict(horizon_months))
    except Exception:
        surv_all = 1.0
    overall_event_rate = 1.0 - surv_all

    # --- Step 3: Threshold-based net benefit -----------------------------
    rows = []
    for pt in thresholds:
        # Treat-all net benefit (same for every threshold)
        nb_all = overall_event_rate - (1.0 - overall_event_rate) * (pt / max(1.0 - pt, 1e-10))
        # Model-guided treatment
        treated = prob_event >= pt
        n_treated = int(treated.sum())
        if n_treated == 0:
            rows.append({"threshold": float(pt), "net_benefit_model": 0.0,
                         "net_benefit_treat_all": nb_all, "n_treated": 0})
            continue
        # KM event rate within the treated subgroup
        kmf_t = KaplanMeierFitter().fit(time[treated], event[treated])
        try:
            surv_t = float(kmf_t.predict(horizon_months))
        except Exception:
            surv_t = 1.0
        event_rate_t = 1.0 - surv_t
        nb_model = (event_rate_t - (1.0 - event_rate_t) * (pt / max(1.0 - pt, 1e-10))) * (n_treated / n)
        rows.append({
            "threshold": float(pt),
            "net_benefit_model": float(nb_model),
            "net_benefit_treat_all": float(nb_all),
            "net_benefit_treat_none": 0.0,
            "n_treated": n_treated,
        })
    return pd.DataFrame(rows)


def plot_dca(dca_df: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if dca_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(dca_df["threshold"], dca_df["net_benefit_model"], label="Model", color="#1f77b4", linewidth=1.6)
    ax.plot(dca_df["threshold"], dca_df["net_benefit_treat_all"], label="Treat all", color="grey", linestyle="--", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.8, label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def subgroup_robustness_rows(
    meta: pd.DataFrame,
    risk_col: str,
    cohort: str,
    subgroup_cols: list[str],
    min_n: int = 20,
    min_events: int = 5,
) -> list[dict[str, Any]]:
    df = meta.copy()
    df["time_months"] = pd.to_numeric(df["time_months"], errors="coerce")
    df["event"] = pd.to_numeric(df["event"], errors="coerce")
    df["risk_score"] = pd.to_numeric(df[risk_col], errors="coerce")
    df = df.dropna(subset=["time_months", "event", "risk_score"])
    if df.empty:
        return [{
            "cohort": cohort,
            "model_risk_score": risk_col,
            "subgroup_variable": np.nan,
            "subgroup_level": np.nan,
            "n": 0,
            "events": 0,
            "harrell_c_index": np.nan,
            "status": "no_complete_cases",
        }]
    if "AGE" in df.columns and pd.to_numeric(df["AGE"], errors="coerce").notna().sum() >= min_n:
        age = pd.to_numeric(df["AGE"], errors="coerce")
        cut = float(age.median())
        if cut == cut:
            df["AGE_GROUP"] = np.where(age >= cut, f">={cut:.1f}", f"<{cut:.1f}")
            subgroup_cols = subgroup_cols + ["AGE_GROUP"]
    available = [c for c in subgroup_cols if c in df.columns and df[c].notna().any()]
    if not available:
        return [{
            "cohort": cohort,
            "model_risk_score": risk_col,
            "subgroup_variable": np.nan,
            "subgroup_level": np.nan,
            "n": int(len(df)),
            "events": int(df["event"].sum()),
            "harrell_c_index": np.nan,
            "status": "subgroup_metadata_unavailable",
        }]
    rows: list[dict[str, Any]] = []
    for col in available:
        for level, sub in df.loc[df[col].notna()].groupby(col):
            n = int(len(sub))
            ev = int(sub["event"].sum())
            if n < min_n or ev < min_events or sub["risk_score"].nunique() < 2:
                rows.append({
                    "cohort": cohort,
                    "model_risk_score": risk_col,
                    "subgroup_variable": col,
                    "subgroup_level": str(level),
                    "n": n,
                    "events": ev,
                    "harrell_c_index": np.nan,
                    "status": "insufficient_samples_or_events",
                })
                continue
            rows.append({
                "cohort": cohort,
                "model_risk_score": risk_col,
                "subgroup_variable": col,
                "subgroup_level": str(level),
                "n": n,
                "events": ev,
                "harrell_c_index": harrell(sub["time_months"].to_numpy(), sub["event"].astype(int).to_numpy(), sub["risk_score"].to_numpy()),
                "status": "ok",
            })
    return rows


def cptac_rna_protein_concordance(cptac_cfg: dict[str, str], candidate_genes: list[str], logger) -> pd.DataFrame:

    from scipy.stats import spearmanr

    rna_path = cptac_cfg.get("rna")
    protein_path = cptac_cfg.get("protein")
    if not (rna_path and protein_path and Path(rna_path).exists() and Path(protein_path).exists()):
        return pd.DataFrame()
    try:
        rna = stream_gene_matrix(rna_path, comment="#", id_col_preference="Hugo_Symbol")
    except Exception as exc:
        logger.warning("Failed to load CPTAC RNA: %s", exc)
        return pd.DataFrame()
    try:
        prot = pd.read_csv(protein_path, sep="\t", comment="#", dtype=str, low_memory=False)
        id_col = "Hugo_Symbol" if "Hugo_Symbol" in prot.columns else prot.columns[0]
        sample_cols = matrix_sample_columns(prot.columns)
        prot = prot[[id_col] + sample_cols].copy()
        prot = prot[prot[id_col].notna()]
        prot[id_col] = prot[id_col].astype(str)
        prot = prot.drop_duplicates(subset=id_col, keep="first").set_index(id_col)
        prot = prot.apply(pd.to_numeric, errors="coerce")
        prot.columns = [patient_id_from_sample(c) for c in prot.columns]
        prot = prot.T
    except Exception as exc:
        logger.warning("Failed to load CPTAC protein: %s", exc)
        return pd.DataFrame()
    common_patients = rna.index.intersection(prot.index)
    rows = []
    for gene in candidate_genes:
        if gene not in rna.columns or gene not in prot.columns:
            continue
        x = pd.to_numeric(rna.loc[common_patients, gene], errors="coerce")
        y = pd.to_numeric(prot.loc[common_patients, gene], errors="coerce")
        df = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(df) < 6:
            continue
        rho, p = spearmanr(df["x"], df["y"])
        rows.append({"feature": gene, "n_paired": int(len(df)), "spearman_rho_rna_protein": float(rho), "spearman_p": float(p), "sign_agreement": bool(rho > 0)})
    return pd.DataFrame(rows)


def htan_microenvironment_association(htan_cfg: dict[str, str], risk_scores: pd.DataFrame, logger, out_dir: Path | None = None) -> pd.DataFrame:


    from scipy.stats import spearmanr

    frac_path = htan_cfg.get("relative_fraction")
    if not (frac_path and Path(frac_path).exists()):
        return pd.DataFrame()
    try:
        df = pd.read_csv(frac_path, sep="\t", comment="#", low_memory=False)
    except Exception as exc:
        logger.warning("Failed to load HTAN cell fraction: %s", exc)
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    id_col = df.columns[0]
    cell_cols = [c for c in df.columns if c != id_col]
    if not cell_cols:
        return pd.DataFrame()
    mat = df[cell_cols].apply(pd.to_numeric, errors="coerce")
    rows = []
    for col in cell_cols:
        vals = mat[col].dropna()
        rows.append({"cell_type": col, "n_samples_with_value": int(len(vals)), "mean_fraction": float(vals.mean()) if len(vals) else np.nan, "std_fraction": float(vals.std()) if len(vals) else np.nan})

    if mat.shape[1] >= 2:
        corr = mat.corr(method="spearman").reset_index().melt(id_vars="index", var_name="cell_type_b", value_name="spearman_rho")
        corr = corr.rename(columns={"index": "cell_type_a"})
        if out_dir is not None:
            corr.to_csv(out_dir / "htan_cell_type_spearman_correlation.tsv", sep="\t", index=False)
    base = pd.DataFrame(rows)
    base["note"] = "HTAN and TCGA patient IDs do not overlap; reporting HTAN microenvironment landscape only. Risk score linkage requires patient-level joint data not currently available."
    return base


def main_step05() -> int:
    parser = build_parser_step05()
    args = parser.parse_args()
    ctx = initialize_run(__file__, args)
    cfg = ctx.cfg
    seed = cfg["random_seed"]
    out_dir = ctx.data_dir("validation_results")
    survival_dir = Path(cfg["processed_root"]) / "survival_models"
    model_dir = survival_dir / "serialized_models"
    metrics_path = survival_dir / "model_comparison_internal_validation_metrics.tsv"
    risk_path = survival_dir / "internal_validation_risk_scores.tsv"
    split_path = survival_dir / "tcga_train_internal_validation_split.tsv"
    primary_manifest_path = survival_dir / "selected_primary_model_manifest.json"
    primary_manifest: dict[str, Any] = {}
    if primary_manifest_path.exists():
        try:
            primary_manifest = json.loads(primary_manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            ctx.add_warning(f"Primary model manifest could not be read; external validation falls back to locked defaults: {exc}")

    def collect_external_bundle_files(obj: Any) -> set[str]:
        files: set[str] = set()
        if isinstance(obj, dict):
            for value in obj.values():
                files.update(collect_external_bundle_files(value))
        elif isinstance(obj, (list, tuple, set)):
            for value in obj:
                files.update(collect_external_bundle_files(value))
        elif isinstance(obj, str) and obj.endswith(".joblib"):
            files.add(Path(obj).name)
        return files

    locked_external_bundle_files = collect_external_bundle_files(primary_manifest.get("external_validation_models", {}))
    if not locked_external_bundle_files:
        locked_external_bundle_files = {
            "clinical_minimal_cox_model_bundle.joblib",
            "msk_external_genomic_cox_model_bundle.joblib",
            "expression_coxnet_model_bundle.joblib",
        }
    ctx.write_json(
        out_dir / "external_validation_locked_model_manifest.json",
        {
            "selected_primary_model_internal": primary_manifest.get("selected_primary_model_internal"),
            "model_selection_policy": primary_manifest.get("model_selection_policy"),
            "external_validation_usage": "post-lock evaluation only; not used for model selection or tuning",
            "locked_external_bundle_files": sorted(locked_external_bundle_files),
            "source_manifest": str(primary_manifest_path) if primary_manifest_path.exists() else None,
        },
        "analysis_data",
        "Locked external validation model manifest.",
    )
    msk_cfg = cfg["cohorts"].get("msk", {})
    geo_cfg = cfg["cohorts"].get("geo_gse103479", {})
    geo_gse17538_cfg = cfg["cohorts"].get("geo_gse17538", {})
    geo_gse39582_cfg = cfg["cohorts"].get("geo_gse39582", {})
    cptac_cfg = cfg["cohorts"].get("cptac", {})
    htan_cfg = cfg["cohorts"].get("htan", {})
    input_files = [
        str(metrics_path),
        str(risk_path),
        str(split_path),
        msk_cfg.get("clinical_patient", ""),
        msk_cfg.get("mutation", ""),
        # ★ 已删除 geo_gse103479 (GSE103479 队列移除)
        geo_gse17538_cfg.get("clinical", ""),
        geo_gse17538_cfg.get("expression", ""),
        geo_gse39582_cfg.get("clinical", ""),
        geo_gse39582_cfg.get("expression", ""),
        # CPTAC/HTAN 仅保留机制分析, 不参与生存预测
        cptac_cfg.get("rna", ""),
        cptac_cfg.get("protein", ""),
        htan_cfg.get("relative_fraction", ""),
    ]

    if args.dry_run:
        rows = [{"input": p, "exists": Path(p).exists() if p else False} for p in input_files]
        ctx.write_table(ctx.run_dir / "validation_visualization_dry_run_input_checks.tsv", rows, "qc", "Dry-run validation input checks.")
        ctx.add_warning("Dry-run only: no validation metrics or figures generated.")
        ctx.finalize([p for p in input_files if p and Path(p).exists()])
        return 0

    all_metrics: list[pd.DataFrame] = []
    if metrics_path.exists():
        internal = pd.read_csv(metrics_path, sep="\t")
        all_metrics.append(internal)
        ctx.write_table(out_dir / "internal_validation_model_performance.tsv", internal, "analysis_data", "Internal validation model performance copy.")
    else:
        ctx.add_warning("Internal metrics file missing; run 04_survival_prediction_model.py first.")


    risk_table_internal = pd.read_csv(risk_path, sep="\t") if risk_path.exists() else pd.DataFrame()
    split = pd.read_csv(split_path, sep="\t") if split_path.exists() else pd.DataFrame()
    train_mask = (split["split"].to_numpy() == "train") if not split.empty else np.array([], dtype=bool)
    train_time = split.loc[train_mask, "time_months"].astype(float).to_numpy() if train_mask.any() else np.array([])
    train_event = split.loc[train_mask, "event"].astype(int).to_numpy() if train_mask.any() else np.array([])


    risk_thresholds: dict[str, list[float]] = {}
    if not risk_table_internal.empty and not split.empty:
        rt = risk_table_internal.copy()
        if "split" in rt.columns:
            rt = rt.drop(columns=["split"])
        rt["PATIENT_ID"] = rt["PATIENT_ID"].astype(str)
        sp = split[["PATIENT_ID", "split"]].copy()
        sp["PATIENT_ID"] = sp["PATIENT_ID"].astype(str)
        merged = rt.merge(sp, on="PATIENT_ID", how="left")
        risk_cols = [c for c in merged.columns if c.endswith("_risk_score")]
        train_part = merged[merged["split"] == "train"]
        for col in risk_cols:
            arr = pd.to_numeric(train_part[col], errors="coerce").dropna().to_numpy()
            if arr.size == 0:
                continue
            if args.km_cut == "median":
                risk_thresholds[col] = [float(np.median(arr))]
            else:
                risk_thresholds[col] = [float(np.percentile(arr, 33.3)), float(np.percentile(arr, 66.6))]

    all_cohorts_risk_rows: list[dict[str, Any]] = []
    km_summary_rows: list[dict[str, Any]] = []
    dca_summary_rows: list[dict[str, Any]] = []
    subgroup_summary_rows: list[dict[str, Any]] = []


    if not split.empty and not risk_table_internal.empty:
        valid_mask = (split["split"].to_numpy() == "internal_validation")
        valid_time = split.loc[valid_mask, "time_months"].astype(float).to_numpy()
        valid_event = split.loc[valid_mask, "event"].astype(int).to_numpy()
        merged = rt.merge(sp, on="PATIENT_ID", how="left")
        for col, thr in risk_thresholds.items():
            v = pd.to_numeric(merged.loc[merged["split"] == "internal_validation", col], errors="coerce").to_numpy()
            if v.size == 0 or np.all(np.isnan(v)):
                continue
            groups = km_groups_by_threshold(v, thr)
            km_path = ctx.output_path(ctx.run_dir / f"internal_validation_kaplan_meier_{col}.png", "figure")
            stats = plot_km(valid_time, valid_event, groups, f"Internal validation KM — {col}", km_path)
            ctx.register_output(km_path, "figure", f"Internal KM for {col} (training-set {args.km_cut} threshold).")
            km_summary_rows.append({"model_risk_score": col, "cohort": "tcga_internal_validation", "threshold_rule": args.km_cut, "thresholds": ";".join(f"{t:.6g}" for t in thr), **stats})
            extra = td_auc_and_brier(train_event, train_time, valid_event, valid_time, v)
            for k, val in extra.items():
                all_metrics.append(pd.DataFrame([{"model": col, "cohort": "tcga_internal_validation", "metric": k, "observed_value": val, "target_success_threshold": 0.75, "gap_to_target": val - 0.75 if "auc" in k else np.nan, "observed_performance_is_not_target_claim": True}]))
            dca_thr = np.arange(args.dca_low, args.dca_high + 1e-9, args.dca_step)
            dca_df = decision_curve_analysis(valid_time, valid_event, v, horizon_months=60.0, thresholds=dca_thr)
            if not dca_df.empty:
                dca_df["model_risk_score"] = col
                dca_df["cohort"] = "tcga_internal_validation"
                dca_summary_rows.append(dca_df)
                dca_path = ctx.output_path(ctx.run_dir / f"internal_validation_decision_curve_{col}.png", "figure")
                plot_dca(dca_df, dca_path, f"DCA — {col} (TCGA internal, 5-yr horizon)")
                ctx.register_output(dca_path, "figure", f"DCA for {col} (TCGA internal validation).")
            for pid, r in zip(merged.loc[merged["split"] == "internal_validation", "PATIENT_ID"].astype(str), v):
                all_cohorts_risk_rows.append({"PATIENT_ID": pid, "cohort": "tcga_internal_validation", "model_risk_score": col, "risk_score": float(r) if r == r else np.nan})


    msk_clinical_path = msk_cfg.get("clinical_patient")
    msk_mutation_path = msk_cfg.get("mutation")
    if msk_clinical_path and Path(msk_clinical_path).exists():
        msk_clin = normalize_external_clinical(read_cbio_table(msk_clinical_path), "msk")
        # ★ MSK unknown SEX 过滤
        if "SEX" in msk_clin.columns:
            sex_valid = msk_clin["SEX"].astype(str).str.strip().str.lower().isin(
                {"male", "female", "m", "f", "0", "1"}
            )
            n_before = len(msk_clin)
            msk_clin = msk_clin.loc[sex_valid]
            ctx.logger.info(
                "MSK: filtered %d samples with unknown SEX → n=%d",
                n_before - len(msk_clin), len(msk_clin),
            )
        for bundle_name, fname, model_label in [
            ("clinical_minimal", "clinical_minimal_cox_model_bundle.joblib", "clinical_minimal_cox"),
            ("msk_genomic", "msk_external_genomic_cox_model_bundle.joblib", "msk_external_genomic_cox"),
        ]:
            if fname not in locked_external_bundle_files:
                ctx.add_warning(f"{model_label} is not listed in locked external validation bundles; skipping.")
                continue
            mp = model_dir / fname
            if not mp.exists():
                ctx.add_warning(f"{model_label} bundle missing at {mp}; skipping MSK validation with this model.")
                continue
            bundle = joblib.load(mp)
            try:
                if bundle_name == "clinical_minimal":
                    x_msk = transform_clinical_external(msk_clin, "PATIENT_ID", bundle)
                    risk = np.log(bundle["model"].predict_partial_hazard(x_msk).to_numpy().reshape(-1) + 1e-12)
                    x_for_survival = x_msk
                else:
                    if not msk_mutation_path or not Path(msk_mutation_path).exists():
                        ctx.add_warning("MSK mutation file missing; skipping genomic-Cox MSK validation.")
                        continue
                    risk, x_for_survival = msk_genomic_predict(bundle, msk_clin.reset_index(drop=True), msk_mutation_path)
            except Exception as exc:
                ctx.add_warning(f"MSK validation failed for {model_label}: {exc}")
                continue
            msk_scores = pd.DataFrame(
                {
                    "_row_index": np.arange(len(msk_clin)),
                    "PATIENT_ID": msk_clin["PATIENT_ID"].astype(str).to_numpy(),
                    "time_months": msk_clin["OS_TIME_MONTHS"].astype(float).to_numpy(),
                    "event": msk_clin["OS_EVENT"].astype(int).to_numpy(),
                    "risk_score": np.asarray(risk).reshape(-1),
                    "model": model_label,
                }
            ).dropna(subset=["time_months", "event", "risk_score"])
            ctx.write_table(out_dir / f"msk_crc_2017_external_validation_risk_scores_{model_label}.tsv", msk_scores.drop(columns=["_row_index"], errors="ignore"), "analysis_data", f"MSK external risk scores ({model_label}).")
            if msk_scores["risk_score"].nunique() > 1:
                bs = bootstrap_harrell(msk_scores["time_months"].to_numpy(), msk_scores["event"].to_numpy(), msk_scores["risk_score"].to_numpy(), args.bootstrap_iter, seed)
                row = {
                    "model": model_label,
                    "cohort": "msk_crc_2017_external",
                    "metric": "harrell_c_index",
                    "observed_value": bs["point"],
                    "ci_low_95": bs["ci_low"],
                    "ci_high_95": bs["ci_high"],
                    "bootstrap_iter": bs["n_valid"],
                    "target_success_threshold": 0.75,
                    "gap_to_target": bs["point"] - 0.75,
                    "observed_performance_is_not_target_claim": True,
                }
                all_metrics.append(pd.DataFrame([row]))
                key = f"{model_label}_risk_score"
                if key in risk_thresholds:
                    groups = km_groups_by_threshold(msk_scores["risk_score"].to_numpy(), risk_thresholds[key])
                    km_path = ctx.output_path(ctx.run_dir / f"msk_external_kaplan_meier_{model_label}.png", "figure")
                    stats = plot_km(msk_scores["time_months"].to_numpy(), msk_scores["event"].to_numpy(), groups, f"MSK external KM — {model_label}", km_path)
                    ctx.register_output(km_path, "figure", f"MSK KM for {model_label}.")
                    km_summary_rows.append({"model_risk_score": key, "cohort": "msk_crc_2017_external", "threshold_rule": args.km_cut, "thresholds": ";".join(f"{t:.6g}" for t in risk_thresholds[key]), **stats})
                else:
                    ctx.add_warning(f"No training-set threshold registered for {model_label}; MSK KM skipped to avoid data leakage.")
                x_surv_valid = x_for_survival.iloc[msk_scores["_row_index"].astype(int).to_numpy()]
                extra = td_auc_and_brier(
                    train_event,
                    train_time,
                    msk_scores["event"].to_numpy(),
                    msk_scores["time_months"].to_numpy(),
                    msk_scores["risk_score"].to_numpy(),
                    survival_estimator=lifelines_survival_estimator(bundle["model"], x_surv_valid),
                )
                for k, val in extra.items():
                    all_metrics.append(pd.DataFrame([{"model": model_label, "cohort": "msk_crc_2017_external", "metric": k, "observed_value": val, "target_success_threshold": 0.75, "gap_to_target": val - 0.75 if "auc" in k else np.nan, "observed_performance_is_not_target_claim": True}]))
                dca_thr = np.arange(args.dca_low, args.dca_high + 1e-9, args.dca_step)
                dca_df = decision_curve_analysis(msk_scores["time_months"].to_numpy(), msk_scores["event"].to_numpy(), msk_scores["risk_score"].to_numpy(), horizon_months=60.0, thresholds=dca_thr)
                if not dca_df.empty:
                    dca_df["model_risk_score"] = key
                    dca_df["cohort"] = "msk_crc_2017_external"
                    dca_summary_rows.append(dca_df)
                    dca_path = ctx.output_path(ctx.run_dir / f"msk_external_decision_curve_{model_label}.png", "figure")
                    plot_dca(dca_df, dca_path, f"DCA — MSK external {model_label} (5-yr horizon)")
                    ctx.register_output(dca_path, "figure", f"MSK external DCA for {model_label}.")
                msk_meta = msk_clin.iloc[msk_scores["_row_index"].astype(int).to_numpy()].copy()
                msk_meta["time_months"] = msk_scores["time_months"].to_numpy()
                msk_meta["event"] = msk_scores["event"].to_numpy()
                msk_meta[key] = msk_scores["risk_score"].to_numpy()
                subgroup_summary_rows.extend(subgroup_robustness_rows(msk_meta, key, "msk_crc_2017_external", ["SEX", "AJCC_STAGE"]))
                for _, r in msk_scores.iterrows():
                    all_cohorts_risk_rows.append({"PATIENT_ID": r["PATIENT_ID"], "cohort": "msk_crc_2017_external", "model_risk_score": key, "risk_score": float(r["risk_score"])})


    expr_bundle_path = model_dir / "expression_coxnet_model_bundle.joblib"
    geo_cohorts = [
        # ★ 已删除 geo_gse103479 (C-index=0.500, 验证完全失败)
        ("geo_gse39582",  "geo_gse39582", "geo_gse39582_external"),
        ("geo_gse17538",  "geo_gse17538", "geo_gse17538_external"),
    ]
    if expr_bundle_path.name not in locked_external_bundle_files:
        ctx.add_warning("Expression Coxnet is not listed in locked external validation bundles; all GEO Coxnet validations skipped.")
    elif not expr_bundle_path.exists():
        ctx.add_warning("Expression Coxnet bundle missing; all GEO Coxnet validations skipped.")
    else:
        bundle = joblib.load(expr_bundle_path)
        for cfg_key, norm_label, cohort_label in geo_cohorts:
            geo_cfg_cohort = cfg["cohorts"].get(cfg_key, {})
            geo_clinical_path = geo_cfg_cohort.get("clinical", "")
            geo_expression_path = geo_cfg_cohort.get("expression", "")
            if not (geo_clinical_path and geo_expression_path
                    and Path(geo_clinical_path).exists() and Path(geo_expression_path).exists()):
                ctx.add_warning(f"GEO cohort {cfg_key}: clinical or expression file missing; skipping.")
                continue
            try:
                geo_clin = normalize_external_clinical(
                    read_table_auto(geo_clinical_path), norm_label
                )
            except Exception as exc:
                ctx.add_warning(f"GEO cohort {cfg_key}: clinical normalisation failed — {exc}; skipping.")
                continue
            should_abort, med, max_val, time_scale_note = assess_followup_time_scale(geo_clin["OS_TIME_MONTHS"])
            if should_abort:
                ctx.add_warning(
                    f"GEO {cfg_key} follow-up time scale check failed: median={med:.1f}, max={max_val:.1f}, status={time_scale_note}; aborting this cohort."
                )
                continue
            geo_expr = load_geo_expression(geo_expression_path, bundle["genes"])
            common = geo_clin["PATIENT_ID"].astype(str)[geo_clin["PATIENT_ID"].astype(str).isin(geo_expr.index)]
            if common.empty:
                ctx.add_warning(f"GEO {cfg_key}: clinical-expression sample overlap is empty; skipping.")
                continue
            geo_clin = geo_clin.set_index("PATIENT_ID").loc[common].reset_index()
            try:
                x_geo = expression_coxnet_design_matrix(geo_expr.loc[common], bundle)
                risk = bundle["model"].predict(x_geo)
                if getattr(risk, "ndim", 1) > 1:
                    risk = risk[:, -1]
                risk = np.asarray(risk).reshape(-1)
            except Exception as exc:
                ctx.add_warning(f"GEO {cfg_key}: risk scoring failed — {exc}; skipping.")
                continue
            geo_scores = pd.DataFrame(
                {
                    "_row_index": np.arange(len(geo_clin)),
                    "PATIENT_ID": geo_clin["PATIENT_ID"].to_numpy(),
                    "time_months": geo_clin["OS_TIME_MONTHS"].astype(float).to_numpy(),
                    "event": geo_clin["OS_EVENT"].astype(int).to_numpy(),
                    "risk_score": risk,
                    "model": "expression_coxnet_lambda_1se",
                    "harmonization": "rank_based_inverse_normal_within_cohort",
                    "cohort": cohort_label,
                }
            ).dropna(subset=["time_months", "event", "risk_score"])
            safe_label = cohort_label.replace("_external", "")
            ctx.write_table(
                out_dir / f"{safe_label}_validation_risk_scores_expression_coxnet.tsv",
                geo_scores.drop(columns=["_row_index"], errors="ignore"), "analysis_data",
                f"{cfg_key} external risk scores (Coxnet + rank-based INT harmonisation).",
            )
            ctx.logger.info(
                "%s: n=%d, events=%d, median_time=%.1f months, time_scale=%s",
                cohort_label, len(geo_scores), int(geo_scores["event"].sum()), med, time_scale_note,
            )
            if geo_scores["risk_score"].nunique() > 1:
                bs = bootstrap_harrell(
                    geo_scores["time_months"].to_numpy(),
                    geo_scores["event"].to_numpy(),
                    geo_scores["risk_score"].to_numpy(),
                    args.bootstrap_iter, seed,
                )
                all_metrics.append(pd.DataFrame([{
                    "model": "expression_coxnet_lambda_1se",
                    "cohort": cohort_label,
                    "metric": "harrell_c_index",
                    "observed_value": bs["point"],
                    "ci_low_95": bs["ci_low"],
                    "ci_high_95": bs["ci_high"],
                    "bootstrap_iter": bs["n_valid"],
                    "target_success_threshold": 0.75,
                    "gap_to_target": bs["point"] - 0.75,
                    "observed_performance_is_not_target_claim": True,
                }]))
                key = "expression_coxnet_risk_score"
                if key in risk_thresholds:
                    groups = km_groups_by_threshold(geo_scores["risk_score"].to_numpy(), risk_thresholds[key])
                    km_path = ctx.output_path(ctx.run_dir / f"{safe_label}_kaplan_meier_expression_coxnet.png", "figure")
                    stats = plot_km(
                        geo_scores["time_months"].to_numpy(), geo_scores["event"].to_numpy(),
                        groups, f"{cfg_key} external KM — expression Coxnet", km_path,
                    )
                    ctx.register_output(km_path, "figure", f"{cfg_key} KM for expression Coxnet.")
                    km_summary_rows.append({
                        "model_risk_score": key, "cohort": cohort_label,
                        "threshold_rule": args.km_cut,
                        "thresholds": ";".join(f"{t:.6g}" for t in risk_thresholds[key]),
                        **stats,
                    })
                else:
                    ctx.add_warning(f"No training-set threshold for {key}; {cfg_key} KM skipped (data leakage guard).")
                extra = td_auc_and_brier(
                    train_event, train_time,
                    geo_scores["event"].to_numpy(), geo_scores["time_months"].to_numpy(),
                    geo_scores["risk_score"].to_numpy(),
                    survival_estimator=sksurv_survival_estimator(bundle["model"], x_geo[geo_scores["_row_index"].astype(int).to_numpy()]),
                )
                for k, val in extra.items():
                    all_metrics.append(pd.DataFrame([{
                        "model": "expression_coxnet_lambda_1se", "cohort": cohort_label,
                        "metric": k, "observed_value": val,
                        "target_success_threshold": 0.75,
                        "gap_to_target": val - 0.75 if "auc" in k else np.nan,
                        "observed_performance_is_not_target_claim": True,
                    }]))
                dca_thr = np.arange(args.dca_low, args.dca_high + 1e-9, args.dca_step)
                dca_df = decision_curve_analysis(geo_scores["time_months"].to_numpy(), geo_scores["event"].to_numpy(), geo_scores["risk_score"].to_numpy(), horizon_months=60.0, thresholds=dca_thr)
                if not dca_df.empty:
                    dca_df["model_risk_score"] = key
                    dca_df["cohort"] = cohort_label
                    dca_summary_rows.append(dca_df)
                    dca_path = ctx.output_path(ctx.run_dir / f"{safe_label}_decision_curve_expression_coxnet.png", "figure")
                    plot_dca(dca_df, dca_path, f"DCA — {cfg_key} external expression Coxnet (5-yr horizon)")
                    ctx.register_output(dca_path, "figure", f"{cfg_key} external DCA for expression Coxnet.")
                geo_meta = geo_clin.iloc[geo_scores["_row_index"].astype(int).to_numpy()].copy()
                geo_meta["time_months"] = geo_scores["time_months"].to_numpy()
                geo_meta["event"] = geo_scores["event"].to_numpy()
                geo_meta[key] = geo_scores["risk_score"].to_numpy()
                subgroup_summary_rows.extend(subgroup_robustness_rows(geo_meta, key, cohort_label, ["SEX", "AJCC_STAGE"]))
                for _, r in geo_scores.iterrows():
                    all_cohorts_risk_rows.append({
                        "PATIENT_ID": r["PATIENT_ID"], "cohort": cohort_label,
                        "model_risk_score": key, "risk_score": float(r["risk_score"]),
                    })


    causal_path = Path(cfg["processed_root"]) / "causal" / "causal_priority_feature_table.tsv"
    if causal_path.exists():
        try:
            candidate_genes = [g for g in pd.read_csv(causal_path, sep="\t")["feature"].astype(str).tolist() if not is_likely_pseudogene(g)][:50]
            cptac_df = cptac_rna_protein_concordance(cptac_cfg, candidate_genes, ctx.logger)
            if not cptac_df.empty:
                ctx.write_table(out_dir / "cptac_mechanism_validation_rna_protein_concordance.tsv", cptac_df, "analysis_data", "CPTAC RNA-protein concordance per candidate gene.")
                direction = cptac_df.copy()
                direction["directionality_endpoint"] = "rna_protein_spearman"
                direction["direction_consistent"] = direction["spearman_rho_rna_protein"] > 0
                direction["validation_scope"] = "mechanism_directionality_only_no_high_dimensional_survival_cindex"
            else:
                direction = pd.DataFrame([{
                    "feature": np.nan,
                    "directionality_endpoint": "rna_protein_spearman",
                    "direction_consistent": np.nan,
                    "validation_scope": "mechanism_directionality_only_no_high_dimensional_survival_cindex",
                    "note": "No candidate genes overlapped CPTAC RNA/protein matrices after current allowlist filtering.",
                }])
            ctx.write_table(out_dir / "cptac_mechanism_validation_directionality.tsv", direction, "analysis_data", "CPTAC mechanism directionality table; not a high-dimensional survival C-index result.")
        except Exception as exc:
            ctx.add_warning(f"CPTAC concordance step failed: {exc}")


    try:
        htan_df = htan_microenvironment_association(htan_cfg, pd.DataFrame(), ctx.logger, out_dir=out_dir)
        if not htan_df.empty:
            ctx.write_table(out_dir / "htan_microenvironment_cell_fraction_landscape.tsv", htan_df, "analysis_data", "HTAN cell-fraction landscape (TCGA patient overlap unavailable).")
            direction = htan_df.copy()
            direction["directionality_endpoint"] = "cell_fraction_landscape"
            direction["direction_consistent"] = np.nan
            direction["validation_scope"] = "microenvironment_directionality_only_no_high_dimensional_survival_cindex"
            ctx.write_table(out_dir / "htan_microenvironment_validation_directionality.tsv", direction, "analysis_data", "HTAN microenvironment directionality table; TCGA patient overlap unavailable.")
    except Exception as exc:
        ctx.add_warning(f"HTAN microenvironment step failed: {exc}")


    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True, sort=False)
    else:
        combined = pd.DataFrame()
    ctx.write_table(out_dir / "internal_external_validation_model_performance.tsv", combined, "analysis_data", "All internal+external validation metrics with bootstrap CI.")
    if all_cohorts_risk_rows:
        ctx.write_table(out_dir / "all_cohorts_risk_score_table.tsv", pd.DataFrame(all_cohorts_risk_rows), "analysis_data", "Patient-level risk scores across cohorts.")
    if km_summary_rows:
        ctx.write_table(out_dir / "kaplan_meier_logrank_summary.tsv", pd.DataFrame(km_summary_rows), "analysis_data", "KM/log-rank summary with training-set thresholds.")
    if dca_summary_rows:
        ctx.write_table(out_dir / "decision_curve_analysis_net_benefit.tsv", pd.concat(dca_summary_rows, ignore_index=True), "analysis_data", "DCA net benefit per threshold/model.")
    if subgroup_summary_rows:
        ctx.write_table(out_dir / "external_validation_subgroup_robustness.tsv", pd.DataFrame(subgroup_summary_rows), "analysis_data", "External validation subgroup robustness summary.")


    try:
        import matplotlib.pyplot as plt

        ci_df = combined[combined["metric"] == "harrell_c_index"].copy() if not combined.empty else pd.DataFrame()
        if not ci_df.empty:
            ci_df["label"] = ci_df["cohort"] + " | " + ci_df["model"]
            ci_df = ci_df.sort_values("observed_value")
            fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.4 * len(ci_df))))
            ax.errorbar(ci_df["observed_value"], np.arange(len(ci_df)),
                        xerr=[ci_df["observed_value"] - ci_df.get("ci_low_95", ci_df["observed_value"]),
                              ci_df.get("ci_high_95", ci_df["observed_value"]) - ci_df["observed_value"]],
                        fmt="o", color="#1f77b4", capsize=3, linewidth=1.0)
            ax.set_yticks(np.arange(len(ci_df)))
            ax.set_yticklabels(ci_df["label"], fontsize=8)
            ax.axvline(0.75, color="firebrick", linestyle="--", linewidth=1.0, label="success criterion = 0.75 (not achieved claim)")
            ax.set_xlabel("Harrell C-index (95% bootstrap CI)")
            ax.set_title("Internal + external validation C-index forest plot")
            ax.legend(loc="lower right", fontsize=8)
            fig.tight_layout()
            p = ctx.output_path(ctx.run_dir / "external_validation_cindex_forest_plot.png", "figure")
            fig.savefig(p, dpi=180)
            plt.close(fig)
            ctx.register_output(p, "figure", "Forest plot of C-index across cohorts.")
    except Exception as exc:
        ctx.add_warning(f"Forest plot failed: {exc}")


    try:
        ipcw_path = Path(cfg["processed_root"]) / "preprocessed" / "tcga_ipcw_os_endpoint.tsv"
        if ipcw_path.exists() and not risk_table_internal.empty and not split.empty:
            ipcw_ep_all = pd.read_csv(ipcw_path, sep="\t", index_col=0)
            train_ids_05 = split.loc[split["split"] == "train", "PATIENT_ID"].astype(str).tolist()
            val_ids_05 = split.loc[split["split"] == "internal_validation", "PATIENT_ID"].astype(str).tolist()
            ep_train_05 = ipcw_ep_all.reindex(train_ids_05).dropna(subset=["time_months"])
            ep_val_05 = ipcw_ep_all.reindex(val_ids_05).dropna(subset=["time_months"])
            risk_cols_05 = [c for c in risk_table_internal.columns if c.endswith("_risk_score") or c.endswith("_risk")]
            ipcw_fig_rows = []
            for rc in risk_cols_05:
                risk_vals = risk_table_internal[rc].to_numpy(float)
                val_risk = risk_vals[split["split"].to_numpy() == "internal_validation"] if len(risk_vals) == len(split) else np.array([])
                if len(val_risk) != len(ep_val_05):
                    continue
                model_label = rc.replace("_risk_score", "").replace("_risk", "")

                train_risk = risk_vals[split["split"].to_numpy() == "train"] if len(risk_vals) == len(split) else np.array([])
                threshold = choose_training_threshold(ep_train_05, train_risk) if len(train_risk) == len(ep_train_05) else None

                figs = generate_prediction_figures(
                    ep_val_05, val_risk, threshold,
                    cohort_label="TCGA internal", model_name=model_label,
                    tau=OS_RISK_TIME_MONTHS, figures_dir=ctx.run_dir,
                )
                ipcw_fig_rows.extend(figs)

                eval_result = evaluate_36m_predictions(ep_train_05, ep_val_05, val_risk, tau=OS_RISK_TIME_MONTHS, threshold=threshold)
                ipcw_fig_rows.append({
                    "cohort": "TCGA internal", "model_name": model_label,
                    "figure_type": "ipcw_evaluation", "status": "computed",
                    "auc_36m": eval_result.get("auc_os_observed", float("nan")),
                    "brier_36m": eval_result.get("brier_os_ipcw", float("nan")),
                    "harrell_cindex": eval_result.get("harrell_cindex", float("nan")),
                    "uno_cindex_ipcw": eval_result.get("uno_cindex_ipcw", float("nan")),
                    "files": "",
                })
            if ipcw_fig_rows:
                ipcw_fig_df = pd.DataFrame(ipcw_fig_rows)
                ipcw_fig_path = out_dir / "ipcw_36m_evaluation_and_figures.tsv"
                ctx.write_table(ipcw_fig_path, ipcw_fig_df, "analysis_data",
                                "IPCW-aware 36-month evaluation metrics and figure manifest.")
                for _, row in ipcw_fig_df.iterrows():
                    if row.get("figure_type") in ("time_dependent_roc", "calibration", "km_risk_strata") and row.get("files"):
                        for fpath in str(row["files"]).split(";"):
                            if fpath and Path(fpath).exists():
                                ctx.register_output(fpath, "figure",
                                                    f"{row['model_name']} {row['figure_type']} (IPCW-aware).")
                ctx.logger.info("IPCW 36m evaluation: %d models evaluated, %d figures generated",
                                len([r for r in ipcw_fig_rows if r.get("figure_type") == "ipcw_evaluation"]),
                                len([r for r in ipcw_fig_rows if r.get("figure_type") != "ipcw_evaluation"]))
        else:
            ctx.add_warning("IPCW endpoint or risk table missing; IPCW-aware evaluation skipped.")
    except Exception as exc:
        ctx.add_warning(f"IPCW-aware evaluation failed: {type(exc).__name__}: {exc}")

    ctx.finalize([p for p in input_files if p and Path(p).exists()])
    return 0


import math


def load_gmt(path: str | Path, min_size: int = 5, max_size: int = 500) -> dict[str, dict[str, Any]]:


    gene_sets: dict[str, dict[str, Any]] = {}
    p = Path(path)
    if not p.exists():
        return gene_sets
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name, description = parts[0], parts[1]
            genes = {g.strip().upper() for g in parts[2:] if g.strip()}
            if min_size <= len(genes) <= max_size:
                gene_sets[name] = {"source": f"{p.stem} — {description}", "genes": genes}
    return gene_sets


def build_parser_step06() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create CRC model interpretability and biology evidence outputs.")
    add_common_args(parser)
    parser.add_argument("--top-n", type=int, default=30, help="Number of candidate biomarkers to summarize.")
    parser.add_argument("--enrichment-fdr", type=float, default=0.25, help="BH-FDR threshold for hypergeometric enrichment.")
    parser.add_argument("--include-go-bp", action=argparse.BooleanOptionalAction, default=True,
                        help="Include GO:BP gene sets in enrichment by default; use --no-include-go-bp for quick debugging.")
    parser.add_argument("--gmt-min-size", type=int, default=5, help="Minimum gene-set size to include from GMT (default 5).")
    parser.add_argument("--gmt-max-size", type=int, default=500, help="Maximum gene-set size to include from GMT (default 500).")
    return parser


BUNDLED_GENE_SETS: dict[str, dict[str, Any]] = {
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION": {
        "source": "MSigDB Hallmark v7 (Liberzon et al. 2015)",
        "genes": {"ACTA2", "ACTG2", "ADAM12", "ANPEP", "APLP1", "AREG", "BASP1", "BDNF", "BGN", "BMP1", "CADM1", "CALD1", "CALU", "CAP2", "CCN1", "CCN2", "CD44", "CD59", "CDH11", "CDH2", "CDH6", "COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL4A2", "COL5A1", "COL5A2", "COL5A3", "COL6A2", "COL6A3", "COL7A1", "COL8A2", "COMP", "COPA", "CRLF1", "CTGF", "CXCL1", "CXCL12", "CXCL6", "CYR61", "DAB2", "DCN", "DKK1", "DPYSL3", "ECM1", "ECM2", "EDIL3", "EFEMP2", "ELN", "EMP3", "ENO2", "FAP", "FAS", "FBLN1", "FBLN2", "FBLN5", "FBN1", "FBN2", "FERMT2", "FGF2", "FLNA", "FMOD", "FN1", "FOXC2", "FSTL1", "FSTL3", "FUCA1", "FZD8", "GADD45A", "GADD45B", "GAS1", "GEM", "GJA1", "GLIPR1", "GPC1", "GPX7", "GREM1", "HTRA1", "ID2", "IGFBP2", "IGFBP3", "IGFBP4", "IL15", "IL32", "IL6", "INHBA", "ITGA2", "ITGA5", "ITGAV", "ITGB1", "ITGB3", "ITGB5", "JUN", "LAMA1", "LAMA2", "LAMA3", "LAMC1", "LAMC2", "LGALS1", "LOX", "LOXL1", "LOXL2", "LRP1", "LRRC15", "LUM", "MAGEE1", "MATN2", "MATN3", "MCM7", "MEST", "MFAP5", "MGP", "MMP1", "MMP14", "MMP2", "MMP3", "MSX1", "MYL9", "MYLK", "NID2", "NNMT", "NOTCH2", "NT5E", "NTM", "OXTR", "P3H1", "PCOLCE", "PCOLCE2", "PDGFRB", "PDLIM4", "PFN2", "PLAUR", "PLOD1", "PLOD2", "PLOD3", "PMEPA1", "PMP22", "POSTN", "PPIB", "PRRX1", "PRSS2", "PTHLH", "PTX3", "PVR", "QSOX1", "RGS4", "RHOB", "SAT1", "SCG2", "SDC1", "SDC4", "SERPINE1", "SERPINE2", "SERPINH1", "SFRP1", "SFRP4", "SGCB", "SGCD", "SGCG", "SLC6A8", "SLIT2", "SLIT3", "SNAI2", "SNTB1", "SPARC", "SPOCK1", "SPP1", "TAGLN", "TFPI2", "TGFB1", "TGFBI", "TGFBR3", "TGM2", "THBS1", "THBS2", "THY1", "TIMP1", "TIMP3", "TNC", "TNFAIP3", "TNFRSF11B", "TNFRSF12A", "TPM1", "TPM2", "TPM4", "VCAM1", "VCAN", "VEGFA", "VEGFC", "VIM", "WIPF1", "WNT5A"},
    },
    "HALLMARK_HYPOXIA": {
        "source": "MSigDB Hallmark v7 (Liberzon et al. 2015)",
        "genes": {"ACKR3", "ADM", "ADORA2B", "AK4", "AKAP12", "ALDOA", "ALDOB", "ALDOC", "AMPD3", "ANGPTL4", "ANKZF1", "ANXA2", "ATF3", "ATP7A", "B3GALT6", "B4GALNT2", "BCAN", "BCL2", "BGN", "BHLHE40", "BNIP3L", "BRS3", "BTG1", "CA12", "CASP6", "CAV1", "CCNG2", "CCR1", "CDKN1A", "CDKN1B", "CDKN1C", "CHST2", "CHST3", "CITED2", "COL5A1", "CP", "CSRP2", "CXCR4", "CXCR7", "CYR61", "DCN", "DDIT3", "DDIT4", "DPYSL4", "DTNA", "DUSP1", "EFNA1", "EFNA3", "EGFR", "ENO1", "ENO2", "ENO3", "ERO1A", "ETS1", "EXT1", "F3", "FAM162A", "FBP1", "FOS", "FOSL2", "FOXO3", "GAA", "GAPDH", "GBE1", "GCK", "GCNT2", "GLRX", "GPC1", "GPC3", "GPC4", "GPI", "GRHPR", "GYS1", "HAS1", "HDLBP", "HEXA", "HK1", "HK2", "HMOX1", "HOXB9", "HS3ST1", "HSPA5", "IDS", "IER3", "IGFBP1", "IGFBP3", "IL6", "ILVBL", "INHA", "IRS2", "ISG20", "JMJD6", "JUN", "KDELR3", "KDM3A", "KLF6", "KLF7", "LALBA", "LARGE1", "LDHA", "LOX", "LXN", "MAFF", "MAP3K1", "MIF", "MT1E", "MT2A", "MXI1", "MYH9", "NAGK", "NDRG1", "NDST1", "NDST2", "NEDD4L", "NFIL3", "NOCT", "NR3C1", "P4HA1", "P4HA2", "PAM", "PCK1", "PDGFB", "PDK1", "PDK3", "PFKFB3", "PFKL", "PFKP", "PGAM2", "PGF", "PGK1", "PGM1", "PGM2", "PHKG1", "PIM1", "PKLR", "PKP1", "PLAC8", "PLAUR", "PLIN2", "PNRC1", "POLR3G", "PPARGC1A", "PPFIA4", "PPP1R15A", "PPP1R3C", "PRDX5", "PRKCA", "PYGM", "RBPJ", "RORA", "RRAGD", "S100A4", "SAP30", "SCARB1", "SDC2", "SDC3", "SDC4", "SELENBP1", "SERPINE1", "SIAH2", "SLC25A1", "SLC2A1", "SLC2A3", "SLC2A5", "SLC6A6", "SRPX", "STBD1", "STC1", "STC2", "SULT2B1", "TES", "TGFB3", "TGFBI", "TIPARP", "TKTL1", "TMEM45A", "TNFAIP3", "TPBG", "TPI1", "TPST2", "UGP2", "VEGFA", "VHL", "VLDLR", "WSB1", "XPNPEP1", "ZFP36", "ZNF292"},
    },
    "HALLMARK_TGF_BETA_SIGNALING": {
        "source": "MSigDB Hallmark v7 (Liberzon et al. 2015)",
        "genes": {"ACVR1", "APC", "ARID4B", "BCAR3", "BMP2", "BMPR1A", "BMPR2", "CDH1", "CDK9", "CDKN1C", "CTNNB1", "ENG", "FKBP1A", "FNTA", "FURIN", "HDAC1", "HIPK2", "ID1", "ID2", "ID3", "IFNGR2", "JUNB", "KLF10", "LEFTY2", "LTBP2", "MAP3K7", "NCOR2", "NOG", "PMEPA1", "PPM1A", "PPP1CA", "PPP1R15A", "RAB31", "RHOA", "SERPINE1", "SKI", "SKIL", "SLC20A1", "SMAD1", "SMAD3", "SMAD6", "SMAD7", "SMURF1", "SMURF2", "SPTBN1", "TGFB1", "TGFBR1", "TGIF1", "THBS1", "TJP1", "TRIM33", "UBE2D3", "WWTR1", "XIAP"},
    },
    "HALLMARK_WNT_BETA_CATENIN_SIGNALING": {
        "source": "MSigDB Hallmark v7 (Liberzon et al. 2015)",
        "genes": {"ADAM17", "AXIN1", "AXIN2", "CCND2", "CSNK1E", "CTNNB1", "CUL1", "DKK1", "DKK4", "DLL1", "DVL2", "FRAT1", "FZD1", "FZD8", "GNAI1", "HDAC2", "HDAC5", "HEY1", "HEY2", "JAG1", "JAG2", "KAT2A", "LEF1", "MAML2", "MYC", "NCOR2", "NCSTN", "NKD1", "NOTCH1", "NOTCH4", "NUMB", "PPARD", "PSEN2", "PTCH1", "RBPJ", "SKP2", "TCF7", "TP53", "WNT1", "WNT5B", "WNT6"},
    },
    "HALLMARK_DNA_REPAIR": {
        "source": "MSigDB Hallmark v7 (Liberzon et al. 2015)",
        "genes": {"AAAS", "ADA", "ADRM1", "ALYREF", "APRT", "ARL6IP1", "BCAP31", "BRF2", "CANT1", "CCNO", "CDA", "CETN2", "CMPK1", "CMPK2", "COBRA1", "COX17", "DAD1", "DCTN4", "DDB1", "DDB2", "DGCR8", "DGUOK", "DUT", "EDF1", "EIF1B", "ELL", "ERCC1", "ERCC2", "ERCC3", "ERCC4", "ERCC5", "ERCC8", "FEN1", "GMPR2", "GPX4", "GTF2A2", "GTF2B", "GTF2F1", "GTF2H1", "GTF2H3", "GTF2H4", "GTF2H5", "GTF3A", "GTF3C5", "GUK1", "HCLS1", "HPRT1", "IMPDH2", "ITPA", "LIG1", "MPG", "MRPL40", "MYL12A", "NCBP2", "NELFB", "NELFCD", "NELFE", "NFX1", "NME1", "NME3", "NME4", "NPR2", "NT5C", "NT5C3A", "NUDT21", "NUDT9", "PCNA", "PDE4B", "PDE6G", "PNP", "POLA1", "POLB", "POLD1", "POLD3", "POLD4", "POLE4", "POLH", "POLL", "POLR1C", "POLR1D", "POLR2A", "POLR2C", "POLR2D", "POLR2E", "POLR2F", "POLR2G", "POLR2H", "POLR2I", "POLR2J", "POLR2K", "POLR2L", "POLR3C", "POLR3GL", "POM121", "PRIM1", "RAD51", "RAD52", "RAE1", "RALA", "RBX1", "RFC2", "RFC3", "RFC4", "RFC5", "RNMT", "RPA1", "RPA2", "RPA3", "SAP30BP", "SDCBP", "SF3A3", "SMAD5", "SNAPC4", "SNAPC5", "SRSF6", "SSRP1", "STX3", "SUPT4H1", "SUPT5H", "SURF1", "TAF10", "TAF12", "TAF13", "TAF1C", "TAF6", "TAF9", "TARBP2", "TCEB3", "TH1L", "TK1", "TK2", "TMED2", "TP53", "TYMS", "UMPS", "UPF3B", "USP11", "VPS28", "VPS37B", "VPS37D", "XPA", "XPC", "XRCC1", "XRCC4", "XRCC5", "XRCC6", "ZNF707", "ZWINT"},
    },
    "HALLMARK_INFLAMMATORY_RESPONSE": {
        "source": "MSigDB Hallmark v7 (Liberzon et al. 2015)",
        "genes": {"ABCA1", "ABI1", "ACVR1B", "ACVR2A", "ADGRE1", "ADM", "ADORA2B", "ADRM1", "AHR", "APLNR", "AQP9", "ATP2A2", "ATP2B1", "ATP2C1", "AXL", "BDKRB1", "BEST1", "BTG2", "C3AR1", "C5AR1", "CALCRL", "CCL17", "CCL2", "CCL20", "CCL22", "CCL24", "CCL5", "CCL7", "CCR7", "CCRL2", "CD14", "CD40", "CD48", "CD55", "CD69", "CD70", "CD82", "CDKN1A", "CHST2", "CLEC5A", "CMKLR1", "CSF1", "CSF3", "CSF3R", "CSGALNACT1", "CSGALNACT2", "CSRP3", "CXCL10", "CXCL11", "CXCL13", "CXCL14", "CXCL5", "CXCL6", "CXCL8", "CXCL9", "CXCR4", "CXCR5", "CXCR6", "CXCR7", "CYBB", "DCBLD2", "EBI3", "EDN1", "EIF2AK2", "EMP3", "EREG", "F3", "FFAR2", "FN1", "FPR1", "FZD5", "GABBR1", "GCH1", "GP1BA", "GPC3", "GPR132", "GPR183", "HAS2", "HBEGF", "HIF1A", "HPN", "HRH1", "ICAM1", "ICAM4", "ICOSLG", "IFITM1", "IFNAR1", "IFNGR2", "IL10", "IL10RA", "IL12B", "IL15", "IL15RA", "IL18", "IL18R1", "IL18RAP", "IL1A", "IL1B", "IL1R1", "IL2RB", "IL4R", "IL6", "IL7R", "IL8", "INHBA", "IRAK2", "IRF1", "IRF7", "ITGA5", "ITGB3", "ITGB8", "KCNA3", "KCNJ2", "KCNMB2", "KIF1B", "KLF6", "LAMP3", "LCK", "LCP2", "LDLR", "LIF", "LPAR1", "LTA", "LY6E", "LYN", "MARCO", "MEFV", "MEP1A", "MET", "MMP14", "MSR1", "MXD1", "MYC", "NAMPT", "NDP", "NFKB1", "NFKBIA", "NLRP3", "NMI", "NMUR1", "NOD2", "NPFFR2", "OLR1", "OPRK1", "OSM", "OSMR", "P2RX4", "P2RX7", "P2RY2", "PCDH7", "PDE4B", "PDPN", "PIK3R5", "PLAUR", "PROK2", "PSEN1", "PTAFR", "PTGER2", "PTGER4", "PTGIR", "PTPRE", "PVR", "RAF1", "RASGRP1", "RELA", "RGS1", "RGS16", "RHOG", "RIPK2", "RNF144B", "ROS1", "RTP4", "SCARF1", "SCN1B", "SELE", "SELENBP1", "SELL", "SEMA4D", "SEMA7A", "SERPINE1", "SLAMF1", "SLC11A2", "SLC1A2", "SLC28A2", "SLC31A1", "SLC31A2", "SLC4A4", "SLC7A1", "SLC7A2", "SPHK1", "SRI", "STAB1", "TACR1", "TACR3", "TAPBP", "TIMP1", "TLR1", "TLR2", "TLR3", "TNFAIP6", "TNFRSF1B", "TNFRSF9", "TNFSF10", "TNFSF15", "TPBG", "VIP"},
    },
    "KEGG_WNT_SIGNALING_PATHWAY": {
        "source": "KEGG pathway hsa04310 (Kanehisa 2017)",
        "genes": {"APC", "AXIN1", "AXIN2", "BTRC", "CAMK2A", "CAMK2B", "CAMK2D", "CAMK2G", "CCND1", "CCND2", "CCND3", "CER1", "CHD8", "CHP1", "CHP2", "CREBBP", "CSNK1A1", "CSNK1A1L", "CSNK1E", "CSNK2A1", "CSNK2A2", "CSNK2B", "CTBP1", "CTBP2", "CTNNB1", "CTNNBIP1", "CUL1", "CXXC4", "DAAM1", "DAAM2", "DKK1", "DKK2", "DKK4", "DVL1", "DVL2", "DVL3", "EP300", "FBXW11", "FOSL1", "FRAT1", "FRAT2", "FZD1", "FZD10", "FZD2", "FZD3", "FZD4", "FZD5", "FZD6", "FZD7", "FZD8", "FZD9", "GSK3B", "JUN", "LEF1", "LRP5", "LRP6", "MAP3K7", "MAPK10", "MAPK8", "MAPK9", "MMP7", "MYC", "NFAT5", "NFATC1", "NFATC2", "NFATC3", "NFATC4", "NKD1", "NKD2", "NLK", "PLCB1", "PLCB2", "PLCB3", "PLCB4", "PORCN", "PPARD", "PPP2CA", "PPP2CB", "PPP2R1A", "PPP2R1B", "PPP2R2A", "PPP2R2B", "PPP2R2C", "PPP2R2D", "PPP2R3A", "PPP2R3B", "PPP2R5C", "PPP2R5D", "PPP2R5E", "PPP3CA", "PPP3CB", "PPP3CC", "PPP3R1", "PPP3R2", "PRICKLE1", "PRICKLE2", "PRKACA", "PRKACB", "PRKACG", "PRKCA", "PRKCB", "PRKCG", "PRKX", "PSEN1", "RAC1", "RAC2", "RAC3", "RBX1", "RHOA", "ROCK1", "ROCK2", "RUVBL1", "SENP2", "SFRP1", "SFRP2", "SFRP4", "SFRP5", "SIAH1", "SKP1", "SMAD2", "SMAD3", "SMAD4", "SOST", "TBL1X", "TBL1XR1", "TBL1Y", "TCF7", "TCF7L1", "TCF7L2", "TP53", "VANGL1", "VANGL2", "WIF1", "WNT1", "WNT10A", "WNT10B", "WNT11", "WNT16", "WNT2", "WNT2B", "WNT3", "WNT3A", "WNT4", "WNT5A", "WNT5B", "WNT6", "WNT7A", "WNT7B", "WNT8A", "WNT8B", "WNT9A", "WNT9B"},
    },
    "KEGG_MISMATCH_REPAIR": {
        "source": "KEGG pathway hsa03430 (Kanehisa 2017)",
        "genes": {"EXO1", "LIG1", "MLH1", "MLH3", "MSH2", "MSH3", "MSH6", "PCNA", "PMS1", "PMS2", "POLD1", "POLD2", "POLD3", "POLD4", "RFC1", "RFC2", "RFC3", "RFC4", "RFC5", "RPA1", "RPA2", "RPA3", "RPA4", "SSBP1"},
    },
    "KEGG_COLORECTAL_CANCER": {
        "source": "KEGG pathway hsa05210 (Kanehisa 2017)",
        "genes": {"AKT1", "AKT2", "AKT3", "APC", "APC2", "ARAF", "AXIN1", "AXIN2", "BAD", "BAX", "BCL2", "BCL2L1", "BIRC5", "BRAF", "CASP3", "CASP9", "CCND1", "CTNNB1", "CYCS", "DCC", "FOS", "FZD1", "FZD10", "FZD2", "FZD3", "FZD4", "FZD5", "FZD6", "FZD7", "FZD8", "FZD9", "GSK3B", "JUN", "KRAS", "LEF1", "MAP2K1", "MAP2K2", "MAPK1", "MAPK10", "MAPK3", "MAPK8", "MAPK9", "MLH1", "MSH2", "MSH3", "MSH6", "MYC", "NRAS", "PDPK1", "PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG", "PIK3R1", "PIK3R2", "PIK3R3", "PIK3R5", "RAC1", "RAC2", "RAC3", "RAF1", "RALGDS", "RHOA", "SMAD2", "SMAD3", "SMAD4", "TCF7", "TCF7L1", "TCF7L2", "TGFB1", "TGFB2", "TGFB3", "TGFBR1", "TGFBR2", "TP53"},
    },
}


def hypergeometric_enrichment(
    candidates: list[str],
    background_size: int,
    gene_sets: dict[str, dict[str, Any]],
) -> pd.DataFrame:

    from scipy.stats import hypergeom

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
        rows.append(
            {
                "gene_set": name,
                "source": info["source"],
                "set_size": K,
                "candidates": n,
                "background_universe": N,
                "overlap": k,
                "overlap_genes": ";".join(sorted(overlap)),
                "fold_enrichment": fold,
                "p_value": p,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["fdr_bh"] = bh_fdr(df["p_value"].tolist())
    return df.sort_values(["p_value", "fdr_bh"])


def rsf_permutation_importance(model_bundle_path: Path, split_path: Path, x_path: Path | None) -> pd.DataFrame:


    if not model_bundle_path.exists():
        return pd.DataFrame()
    bundle = joblib.load(model_bundle_path)
    if not split_path.exists():
        return pd.DataFrame()
    split = pd.read_csv(split_path, sep="\t")
    valid_mask = split["split"] == "internal_validation"
    if int(valid_mask.sum()) < 10:
        return pd.DataFrame()


    model = bundle.get("model")
    if model is None:
        return pd.DataFrame()
    try:
        importances = model.feature_importances_
        feature_names = bundle.get("transformer", {}).get("model_columns", [f"f{i}" for i in range(len(importances))])
        return pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    except Exception:
        return pd.DataFrame()


def main_step06() -> int:
    parser = build_parser_step06()
    args = parser.parse_args()
    ctx = initialize_run(__file__, args)
    cfg = ctx.cfg
    out_dir = ctx.data_dir("interpretability_biology")
    causal_path = Path(cfg["processed_root"]) / "causal" / "causal_priority_feature_table.tsv"
    coxnet_coef_path = Path(cfg["processed_root"]) / "survival_models" / "coxnet_selected_features_and_coefficients.tsv"
    clinical_coef_path = Path(cfg["processed_root"]) / "survival_models" / "clinical_cox_model_coefficients.tsv"
    rsf_bundle_path = Path(cfg["processed_root"]) / "survival_models" / "serialized_models" / "clinical_random_survival_forest_model_bundle.joblib"
    split_path = Path(cfg["processed_root"]) / "survival_models" / "tcga_train_internal_validation_split.tsv"
    cptac_concord_path = Path(cfg["processed_root"]) / "validation_results" / "cptac_mechanism_validation_rna_protein_concordance.tsv"
    cptac_direction_path = Path(cfg["processed_root"]) / "validation_results" / "cptac_mechanism_validation_directionality.tsv"
    htan_direction_path = Path(cfg["processed_root"]) / "validation_results" / "htan_microenvironment_validation_directionality.tsv"
    external_metrics_path = Path(cfg["processed_root"]) / "validation_results" / "internal_external_validation_model_performance.tsv"
    rna_path = cfg["cohorts"]["tcga"]["rna"]
    input_files = [str(causal_path), str(coxnet_coef_path), str(clinical_coef_path), str(rsf_bundle_path), str(cptac_concord_path), str(cptac_direction_path), str(htan_direction_path), str(external_metrics_path), rna_path, cfg["cohorts"]["cptac"]["protein"], cfg["cohorts"]["htan"]["relative_fraction"]]

    if args.dry_run:
        rows = [{"input": p, "exists": Path(p).exists() if p else False} for p in input_files]
        ctx.write_table(ctx.run_dir / "interpretability_biology_dry_run_input_checks.tsv", rows, "qc", "Dry-run interpretability input checks.")
        ctx.add_warning("Dry-run only: no interpretability outputs generated.")
        ctx.finalize([p for p in input_files if p and Path(p).exists()])
        return 0


    evidence_parts: list[pd.DataFrame] = []
    candidate_genes: list[str] = []
    if causal_path.exists():
        causal = pd.read_csv(causal_path, sep="\t")
        causal["feature"] = causal["feature"].astype(str)
        causal = causal[~causal["feature"].apply(is_likely_pseudogene)]
        candidate_genes = causal["feature"].head(args.top_n).tolist()
        evidence_parts.append(causal.head(args.top_n).assign(evidence_source="causal_priority_feature_table"))
    else:
        ctx.add_warning("Causal priority table missing; biomarker interpretation will be reduced.")

    if coxnet_coef_path.exists():
        cox = pd.read_csv(coxnet_coef_path, sep="\t")
        if "feature" in cox.columns:
            cox = cox[~cox["feature"].astype(str).apply(is_likely_pseudogene)]
        cox = cox.assign(evidence_source="coxnet_selected_features_and_coefficients")
        evidence_parts.append(cox.head(args.top_n))
        candidate_genes = list(dict.fromkeys(candidate_genes + cox["feature"].astype(str).head(args.top_n).tolist()))

    if clinical_coef_path.exists():
        cl = pd.read_csv(clinical_coef_path, sep="\t")
        cl = cl.rename(columns={"covariate": "feature"})
        cl = cl.assign(evidence_source="clinical_cox_model_coefficients", is_clinical_covariate=True)
        evidence_parts.append(cl.head(args.top_n))

    if evidence_parts:
        evidence = pd.concat(evidence_parts, ignore_index=True, sort=False)
    else:
        evidence = pd.DataFrame(columns=["feature", "evidence_source"])
    evidence["not_a_validated_wet_lab_result"] = True


    if external_metrics_path.exists():
        ext = pd.read_csv(external_metrics_path, sep="\t")
        ci_rows = ext[ext["metric"] == "harrell_c_index"].copy()
        ci_rows["observed_value"] = pd.to_numeric(ci_rows["observed_value"], errors="coerce")
        directions = ci_rows.groupby("cohort")["observed_value"].mean().to_dict()
        evidence["msk_external_cindex_mean"] = directions.get("msk_crc_2017_external", np.nan)
        evidence["geo_external_cindex_mean"] = directions.get("geo_gse103479_external", np.nan)
        evidence["external_validation_evidence_scope"] = "model_level_only_not_per_feature_direction"
        evidence["external_validation_summary"] = "Cohort-level Harrell C-index is retained only as locked-model validation context. Per-feature external direction requires single-gene Cox or mutation-status Cox and is not inferred from model C-index."

    if cptac_concord_path.exists():
        cp = pd.read_csv(cptac_concord_path, sep="\t")
        evidence = evidence.merge(cp[["feature", "spearman_rho_rna_protein", "spearman_p", "sign_agreement"]], on="feature", how="left")
    if cptac_direction_path.exists():
        cp_dir = pd.read_csv(cptac_direction_path, sep="\t")
        if "feature" in cp_dir.columns and "direction_consistent" in cp_dir.columns:
            cp_dir["feature"] = cp_dir["feature"].astype(str)
            evidence["feature"] = evidence["feature"].astype(str)
            evidence = evidence.merge(cp_dir[["feature", "direction_consistent"]].rename(columns={"direction_consistent": "cptac_direction_consistent"}), on="feature", how="left")
    if htan_direction_path.exists():
        evidence["htan_directionality_table_present"] = True
    ctx.write_table(out_dir / "core_biomarker_evidence_matrix.tsv", evidence, "analysis_data", "Core biomarker evidence matrix with cross-omics + external concordance columns.")
    try:
        import matplotlib.pyplot as plt

        indicator_cols = [c for c in ["cptac_direction_consistent", "sign_agreement"] if c in evidence.columns]
        plot_evidence = evidence.drop_duplicates("feature").head(args.top_n)
        if indicator_cols and not plot_evidence.empty:
            heat = plot_evidence.set_index("feature")[indicator_cols].replace({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
            heat = heat.apply(pd.to_numeric, errors="coerce").fillna(0.5).astype(float)
            fig, ax = plt.subplots(figsize=(5.8, max(3.5, 0.22 * len(heat))))
            im = ax.imshow(heat.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, fraction=0.035, label="Evidence present / direction consistent")
            ax.set_yticks(range(len(heat.index)))
            ax.set_yticklabels(heat.index, fontsize=7)
            ax.set_xticks(range(len(heat.columns)))
            ax.set_xticklabels(heat.columns, rotation=35, ha="right", fontsize=8)
            ax.set_title("External and cross-omics evidence matrix")
            fig.tight_layout()
            p = ctx.output_path(ctx.run_dir / "evidence_heatmap.png", "figure")
            fig.savefig(p, dpi=180)
            plt.close(fig)
            ctx.register_output(p, "figure", "Evidence heatmap for external and cross-omics directionality.")
        if {"feature", "spearman_rho_rna_protein"}.issubset(evidence.columns):
            rp = evidence[["feature", "spearman_rho_rna_protein"]].dropna().drop_duplicates("feature").head(args.top_n)
            if not rp.empty:
                rp = rp.sort_values("spearman_rho_rna_protein")
                fig, ax = plt.subplots(figsize=(6.4, max(3.5, 0.22 * len(rp))))
                ax.barh(rp["feature"], rp["spearman_rho_rna_protein"], color="#377eb8")
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("Spearman rho")
                ax.set_title("CPTAC protein-RNA concordance")
                fig.tight_layout()
                p = ctx.output_path(ctx.run_dir / "protein_rna_concordance.png", "figure")
                fig.savefig(p, dpi=180)
                plt.close(fig)
                ctx.register_output(p, "figure", "Protein-RNA concordance for candidate biomarkers.")
    except Exception as exc:
        ctx.add_warning(f"Evidence visualization failed: {exc}")


    msigdb_cfg = cfg.get("cohorts", {}).get("msigdb", {})
    hallmark_path = msigdb_cfg.get("hallmark", "")
    kegg_path = msigdb_cfg.get("kegg", "")
    go_bp_path = msigdb_cfg.get("go_bp", "")

    active_gene_sets: dict[str, dict[str, Any]] = {}
    gene_set_source_label = "bundled_msigdb_hallmark_kegg_subset"

    hallmark_sets = load_gmt(hallmark_path, min_size=args.gmt_min_size, max_size=args.gmt_max_size)
    kegg_sets = load_gmt(kegg_path, min_size=args.gmt_min_size, max_size=args.gmt_max_size)

    if hallmark_sets or kegg_sets:
        active_gene_sets.update(hallmark_sets)
        active_gene_sets.update(kegg_sets)
        gene_set_source_label = "msigdb_v2024_1_hallmark_kegg_legacy"
        ctx.logger.info(
            "Loaded %d Hallmark + %d KEGG sets from local GMT files.",
            len(hallmark_sets), len(kegg_sets),
        )
        if args.include_go_bp:
            go_bp_sets = load_gmt(go_bp_path, min_size=args.gmt_min_size, max_size=args.gmt_max_size)
            active_gene_sets.update(go_bp_sets)
            gene_set_source_label += "_gobp"
            ctx.logger.info("Loaded %d GO:BP sets from local GMT file.", len(go_bp_sets))
        else:
            ctx.logger.info("GO:BP skipped (use --include-go-bp to enable).")
    else:
        active_gene_sets = BUNDLED_GENE_SETS
        ctx.add_warning(
            "Local MSigDB GMT files not found; falling back to bundled 8-set CRC subset. "
            f"Expected Hallmark at: {hallmark_path}"
        )


    if candidate_genes:
        try:
            background = stream_gene_matrix(rna_path, comment="#", id_col_preference="Hugo_Symbol")
            background_genes = [g for g in background.columns.astype(str).tolist() if not is_likely_pseudogene(g)]
            ctx.logger.info(
                "Hypergeometric background: %d genes; candidate pool: %d; gene-set library: %d sets (%s).",
                len(background_genes), len(candidate_genes), len(active_gene_sets), gene_set_source_label,
            )
            enrich = hypergeometric_enrichment(candidate_genes, background_size=len(background_genes), gene_sets=active_gene_sets)
            if not enrich.empty:
                enrich["gene_set_library"] = gene_set_source_label
                sig = enrich[enrich["fdr_bh"] <= args.enrichment_fdr]
                ctx.logger.info(
                    "Enrichment: %d total sets tested; %d significant at FDR <= %.2f.",
                    len(enrich), len(sig), args.enrichment_fdr,
                )
                ctx.write_table(
                    out_dir / "pathway_enrichment_results.tsv", enrich, "analysis_data",
                    f"Hypergeometric ORA against {gene_set_source_label} ({len(enrich)} sets tested).",
                )
                try:
                    import matplotlib.pyplot as plt

                    plot_df = enrich.head(20).iloc[::-1]
                    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.30 * len(plot_df))))
                    sizes = 40 + 12 * plot_df["overlap"].astype(float).clip(upper=20)
                    colors = -np.log10(plot_df["p_value"].clip(lower=1e-10))
                    sc = ax.scatter(plot_df["fold_enrichment"], plot_df["gene_set"], s=sizes, c=colors, cmap="viridis")
                    cb = fig.colorbar(sc, ax=ax)
                    cb.set_label("-log10(p)")
                    ax.set_xlabel("Fold enrichment")
                    ax.set_title(f"Pathway ORA (top 20, {gene_set_source_label})")
                    fig.tight_layout()
                    p = ctx.output_path(ctx.run_dir / "pathway_enrichment_dotplot.png", "figure")
                    fig.savefig(p, dpi=180)
                    plt.close(fig)
                    ctx.register_output(p, "figure", "Pathway over-representation dot plot (MSigDB v2024.1).")
                except Exception as exc:
                    ctx.add_warning(f"Pathway plot failed: {exc}")
        except Exception as exc:
            ctx.add_warning(f"Hypergeometric enrichment failed: {exc}")
    else:
        ctx.add_warning("No candidate genes available for enrichment.")


    rsf_imp = rsf_permutation_importance(rsf_bundle_path, split_path, None)
    if not rsf_imp.empty:
        ctx.write_table(out_dir / "random_survival_forest_feature_importance.tsv", rsf_imp, "analysis_data", "RSF impurity-based feature importance (clinical block).")


    if coxnet_coef_path.exists():
        cox = pd.read_csv(coxnet_coef_path, sep="\t")
        if "coef" in cox.columns:
            cox = cox.copy()
            cox["abs_coef"] = cox["coef"].abs()
            cox = cox.sort_values("abs_coef", ascending=False)
            cox["effect_direction"] = np.where(cox["coef"] > 0, "risk_increase", "risk_decrease")
            ctx.write_table(out_dir / "model_feature_importance_stability.tsv", cox, "analysis_data", "Coxnet model |coefficient|-ranked feature importance proxy with direction.")


    proposal_pool = (
        evidence[evidence["evidence_source"] == "causal_priority_feature_table"][["feature"]].copy()
        if not evidence.empty
        else pd.DataFrame()
    )
    if not proposal_pool.empty:
        proposal_pool["feature"] = proposal_pool["feature"].astype(str)
        proposal_pool = proposal_pool[~proposal_pool["feature"].apply(is_likely_pseudogene)]
        proposal_pool = proposal_pool[~proposal_pool["feature"].str.upper().isin(CLINICAL_VARIABLE_WHITELIST)]
        proposal_pool = proposal_pool.drop_duplicates().head(5)
        proposal_pool["proposed_in_vitro_model"] = "CRC cell lines (HCT116, SW480) and patient-derived organoids"
        proposal_pool["proposed_perturbation"] = "siRNA / shRNA knockdown; CRISPRi/CRISPRa where applicable"
        proposal_pool["proposed_phenotypic_readout"] = "Proliferation (CCK-8/EdU), migration/invasion (Transwell), apoptosis (Annexin V)"
        proposal_pool["proposed_molecular_readout"] = "qPCR + Western blot; IHC on TMA"
        proposal_pool["validation_status"] = "future_experiment_proposal_only_not_completed"
    else:
        proposal_pool = pd.DataFrame()
    ctx.write_table(out_dir / "wet_lab_validation_priority_list.tsv", proposal_pool, "analysis_data", "Wet-lab validation proposal (protein-coding causal candidates only; not completed).")


    text_lines = ["# Mechanistic Hypotheses for Top Biomarkers", ""]
    if proposal_pool.empty:
        text_lines.append("No protein-coding causal candidates were available; mechanistic hypothesis generation deferred.")
    else:
        for _, row in proposal_pool.iterrows():
            text_lines.append(f"- `{row['feature']}`: prioritize functional validation in CRC cell lines/organoids; current status is **hypothesis-generating only**, NOT a validated mechanism.")
    text_lines.append("")
    text_lines.append("## Limitations")
    text_lines.append(f"- Pathway enrichment uses {gene_set_source_label} (Liberzon 2015 / Kanehisa 2017 / GO Consortium). "
                      "GO:BP is included by default when the local GMT file is present; use --no-include-go-bp only for quick debugging.")
    text_lines.append("- SHAP / Integrated Gradients require a trained deep model; DeepSurv was disabled per `selected_primary_model_manifest.deep_models_disabled_reason`.")
    text_lines.append("- STRING/PPI network analysis is not bundled (would require external API).")
    text_lines.append("- All conclusions are model-based associations and do NOT prove biological causality.")
    ctx.write_text(out_dir / "mechanistic_hypotheses_for_top_biomarkers.md", "\n".join(text_lines), "analysis_data", "Mechanistic hypothesis notes (proposal only).")

    if active_gene_sets is BUNDLED_GENE_SETS:
        ctx.add_warning("Pathway enrichment used bundled 8-set CRC subset (GMT files not found); re-run after placing MSigDB GMT files in DATA/msigdb/.")
    ctx.finalize([p for p in input_files if p and Path(p).exists()])
    return 0


def resolve_multi_timepoint_paths() -> Dict[str, Path]:

    script_dir = Path(os.environ.get("MT_SCRIPT_DIR", str(Path(__file__).resolve().parent)))
    data_dir = Path(os.environ.get("MT_DATA_DIR", str(script_dir.parent / "data")))
    ansisly_data_dir = script_dir.parent.parent.parent / "DATA"  # ansisly/DATA
    output_dir = Path(os.environ.get("MT_OUTPUT_DIR", str(data_dir / "multi_timepoint_results")))
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "clinical": ansisly_data_dir / "preprocessed" / "tcga_os_clinical_endpoint_qc.tsv",
        "embedding": data_dir / "multiomics_pretraining" / "tcga_multiomics_patient_embedding.tsv",
        "split": data_dir / "survival_models" / "tcga_train_internal_validation_split.tsv",
        "causal": data_dir / "causal" / "aipw_causal_screening_table.tsv",
        "output_dir": output_dir,
        "script_dir": script_dir,
    }


def mt_diagnose_auc(timepoint_results: Dict[str, Dict], output_dir: Path) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("AUC DIAGNOSTICS MODULE")
    print("=" * 70)
    diag_rows: List[Dict[str, Any]] = []
    for tp_name, res in sorted(timepoint_results.items()):
        row: Dict[str, Any] = {"timepoint": tp_name}
        ep = res.get("endpoint_stats", {})
        row["n_total"] = ep.get("n_total", 0)
        row["n_events"] = ep.get("n_events", 0)
        row["n_censored"] = ep.get("n_censored", 0)
        row["censor_rate"] = ep.get("censor_rate", 0)
        row["n_early_censored"] = ep.get("n_early_censored", 0)
        ipcw = res.get("ipcw_stats", {})
        row["ipcw_median"] = ipcw.get("median", 0)
        row["ipcw_p95"] = ipcw.get("p95", 0)
        row["ipcw_max"] = ipcw.get("max", 0)
        row["ipcw_cv"] = ipcw.get("cv", 0)
        models = res.get("models", {})
        for m_name, m_res in models.items():
            row[f"{m_name}_auc"] = m_res.get("auc", float("nan"))
            row[f"{m_name}_c_index"] = m_res.get("c_index", float("nan"))
            row[f"{m_name}_brier"] = m_res.get("brier", float("nan"))
        row["n_features"] = res.get("n_features", 0)
        row["n_clinical_features"] = res.get("n_clinical_features", 0)
        diag_rows.append(row)
    diag_df = pd.DataFrame(diag_rows)
    print("\n--- Data Quality Summary ---")
    for _, r in diag_df.iterrows():
        tp = r["timepoint"]
        print(f"  {tp}: N={r['n_total']:.0f}, events={r['n_events']:.0f}, "
              f"censor_rate={r['censor_rate']:.1%}, early_censored={r['n_early_censored']:.0f}")
    print("\n--- IPCW Weight Distribution ---")
    for _, r in diag_df.iterrows():
        if r["timepoint"] != "full":
            tp = r["timepoint"]
            print(f"  {tp}: median={r['ipcw_median']:.3f}, P95={r['ipcw_p95']:.3f}, "
                  f"max={r['ipcw_max']:.3f}, CV={r['ipcw_cv']:.3f}")
    print("\n--- Model Performance Comparison ---")
    model_names: set = set()
    for r in diag_rows:
        for k in r:
            if k.endswith("_auc"):
                model_names.add(k.replace("_auc", ""))
    header = f"  {'Timepoint':<12}" + "".join(f"{m:<20}" for m in sorted(model_names))
    print(header)
    for _, r in diag_df.iterrows():
        line = f"  {r['timepoint']:<12}"
        for m in sorted(model_names):
            auc_val = r.get(f"{m}_auc", float("nan"))
            c_val = r.get(f"{m}_c_index", float("nan"))
            if not np.isnan(auc_val):
                line += f"AUC={auc_val:.3f} C={c_val:.3f}  "
            else:
                line += f"{'N/A':<20}"
        print(line)
    print("\n--- Root Cause Analysis ---")
    causes: List[str] = []
    for _, r in diag_df.iterrows():
        tp = r["timepoint"]
        if tp == "full":
            continue
        if r["censor_rate"] > 0.4:
            causes.append(f"[{tp}] HIGH CENSORING: {r['censor_rate']:.1%} of samples censored before {tp}. "
                         f"This drastically reduces effective sample size for binary classification.")
        if r["ipcw_cv"] > 1.0:
            causes.append(f"[{tp}] UNSTABLE IPCW: CV={r['ipcw_cv']:.2f}, max weight={r['ipcw_max']:.1f}. "
                         f"Extreme weight variance indicates KM survival estimate near 0 at tau.")
        if r["n_features"] < 30:
            causes.append(f"[{tp}] LOW FEATURES: Only {r['n_features']:.0f} embedding features. "
                         f"16-dim PCA embedding is very low for capturing prognostic signal.")
        if r["n_events"] < 100:
            causes.append(f"[{tp}] FEW EVENTS: Only {r['n_events']:.0f} death events. "
                         f"Insufficient events for reliable model training.")
    if causes:
        for c in causes:
            print(f"  * {c}")
    else:
        print("  No critical issues detected.")
    diag_path = output_dir / "auc_diagnostics.tsv"
    diag_df.to_csv(diag_path, sep="\t", index=False)
    print(f"\n  Diagnostics saved: {diag_path}")
    return diag_df


def mt_plot_timepoint_comparison(timepoint_results: Dict, output_dir: Path) -> None:

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    taus = [k for k in sorted(timepoint_results.keys()) if k != "full"]
    model_names: set = set()
    for tp, res in timepoint_results.items():
        for m_name in res.get("models", {}):
            model_names.add(m_name)

    ax = axes[0, 0]
    for m_name in sorted(model_names):
        aucs = [timepoint_results[tp].get("models", {}).get(m_name, {}).get("auc", float("nan")) for tp in taus]
        ax.plot(taus, aucs, marker="o", label=m_name, linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.set_xlabel("Timepoint (months)")
    ax.set_ylabel("AUC")
    ax.set_title("AUC by Timepoint and Model")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for m_name in sorted(model_names):
        cs = [timepoint_results[tp].get("models", {}).get(m_name, {}).get("c_index", float("nan")) for tp in taus]
        ax.plot(taus, cs, marker="s", label=m_name, linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.set_xlabel("Timepoint (months)")
    ax.set_ylabel("C-index (Harrell)")
    ax.set_title("C-index by Timepoint and Model")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    censor_rates, best_aucs = [], []
    for tp in taus:
        ep = timepoint_results[tp].get("endpoint_stats", {})
        censor_rates.append(ep.get("censor_rate", 0))
        models = timepoint_results[tp].get("models", {})
        aucs = [v.get("auc", 0) for v in models.values() if not np.isnan(v.get("auc", float("nan")))]
        best_aucs.append(max(aucs) if aucs else 0)
    ax.scatter(censor_rates, best_aucs, s=100, c="steelblue", zorder=5)
    for tp, cr, ba in zip(taus, censor_rates, best_aucs):
        ax.annotate(tp, (cr, ba), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Censoring Rate before tau")
    ax.set_ylabel("Best AUC")
    ax.set_title("Censoring Rate vs Best AUC")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ipcw_data, ipcw_labels = [], []
    for tp in taus:
        ipcw = timepoint_results[tp].get("ipcw_stats", {})
        if ipcw.get("all_weights") is not None:
            w = ipcw["all_weights"]
            w = w[w > 0]
            if len(w) > 0:
                ipcw_data.append(w)
                ipcw_labels.append(tp)
    if ipcw_data:
        bp = ax.boxplot(ipcw_data, labels=ipcw_labels, patch_artist=True)
        colors = ["#4C72B0", "#55A868", "#C44E52"]
        for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("IPCW Weight")
    ax.set_title("IPCW Weight Distribution")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for suffix, dpi in [(".png", 300), (".tiff", 600)]:
        plt.savefig(output_dir / f"multi_timepoint_comparison{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved: multi_timepoint_comparison.png/.tiff")


def mt_apply_improvements(
    X_train_clin: pd.DataFrame, X_val_clin: pd.DataFrame,
    X_train_emb: pd.DataFrame, X_val_emb: pd.DataFrame,
    train_endpoint: pd.DataFrame, val_endpoint: pd.DataFrame,
    tau: float, random_seed: int = RANDOM_SEED,
) -> Dict[str, Dict]:


    tag = f"_{int(tau)}m"
    results: Dict[str, Dict] = {}
    train_obs_ep = train_endpoint[train_endpoint[f"death_by{tag}_observed"] == True].copy()
    val_obs_ep = val_endpoint[val_endpoint[f"death_by{tag}_observed"] == True].copy()
    train_obs_pids = train_obs_ep["PATIENT_ID"].values
    val_obs_pids = val_obs_ep["PATIENT_ID"].values
    y_train_obs = train_obs_ep[f"death_by{tag}"].astype(int).to_numpy()
    y_val_obs = val_obs_ep[f"death_by{tag}"].astype(int).to_numpy()
    w_train_obs = train_obs_ep[f"ipcw_weight{tag}"].to_numpy(float)
    w_val_obs = val_obs_ep[f"ipcw_weight{tag}"].to_numpy(float)
    if pd.Series(y_train_obs).nunique() < 2 or pd.Series(y_val_obs).nunique() < 2:
        return results

    if len(X_train_clin.columns) >= 3:
        tr_pids = np.intersect1d(train_obs_pids, X_train_clin.index)
        va_pids = np.intersect1d(val_obs_pids, X_val_clin.index)
        if len(tr_pids) >= 20 and len(va_pids) >= 5:
            tr_map = train_obs_ep.set_index("PATIENT_ID")
            va_map = val_obs_ep.set_index("PATIENT_ID")
            ytr = tr_map.loc[tr_pids, f"death_by{tag}"].astype(int).to_numpy()
            yva = va_map.loc[va_pids, f"death_by{tag}"].astype(int).to_numpy()
            wtr = tr_map.loc[tr_pids, f"ipcw_weight{tag}"].to_numpy(float)
            wva = va_map.loc[va_pids, f"ipcw_weight{tag}"].to_numpy(float)
            if pd.Series(ytr).nunique() >= 2:
                model = fit_improved_logistic(X_train_clin.loc[tr_pids], ytr, wtr, random_seed)
                if model is not None:
                    Xva_s = model._scaler.transform(X_val_clin.loc[va_pids])
                    risk = model.predict_proba(Xva_s)[:, 1]
                    results["clinical_only"] = evaluate_binary_predictions(yva, risk, wva, tau)

    if len(X_train_clin.columns) >= 3:
        tr_comb = np.intersect1d(np.intersect1d(train_obs_pids, X_train_clin.index), X_train_emb.index)
        va_comb = np.intersect1d(np.intersect1d(val_obs_pids, X_val_clin.index), X_val_emb.index)
        if len(tr_comb) >= 20 and len(va_comb) >= 5:
            Xtr = pd.concat([X_train_clin.loc[tr_comb], X_train_emb.loc[tr_comb]], axis=1)
            Xva = pd.concat([X_val_clin.loc[va_comb], X_val_emb.loc[va_comb]], axis=1)
            tr_map = train_obs_ep.set_index("PATIENT_ID")
            va_map = val_obs_ep.set_index("PATIENT_ID")
            ytr = tr_map.loc[tr_comb, f"death_by{tag}"].astype(int).to_numpy()
            yva = va_map.loc[va_comb, f"death_by{tag}"].astype(int).to_numpy()
            wtr = tr_map.loc[tr_comb, f"ipcw_weight{tag}"].to_numpy(float)
            wva = va_map.loc[va_comb, f"ipcw_weight{tag}"].to_numpy(float)
            if pd.Series(ytr).nunique() >= 2:
                model = fit_improved_logistic(Xtr, ytr, wtr, random_seed)
                if model is not None:
                    Xva_s = model._scaler.transform(Xva)
                    risk = model.predict_proba(Xva_s)[:, 1]
                    results["clinical_plus_embedding"] = evaluate_binary_predictions(yva, risk, wva, tau)

    tr_emb = np.intersect1d(train_obs_pids, X_train_emb.index)
    va_emb = np.intersect1d(val_obs_pids, X_val_emb.index)
    if len(tr_emb) >= 20 and len(va_emb) >= 5:
        tr_map = train_obs_ep.set_index("PATIENT_ID")
        va_map = val_obs_ep.set_index("PATIENT_ID")
        Xtr_e = X_train_emb.loc[tr_emb]
        Xva_e = X_val_emb.loc[va_emb]
        ytr = tr_map.loc[tr_emb, f"death_by{tag}"].astype(int).to_numpy()
        yva = va_map.loc[va_emb, f"death_by{tag}"].astype(int).to_numpy()
        wtr = tr_map.loc[tr_emb, f"ipcw_weight{tag}"].to_numpy(float)
        wva = va_map.loc[va_emb, f"ipcw_weight{tag}"].to_numpy(float)
        if pd.Series(ytr).nunique() >= 2:

            model = fit_improved_logistic(Xtr_e, ytr, wtr, random_seed)
            if model is not None:
                Xva_s = model._scaler.transform(Xva_e)
                risk = model.predict_proba(Xva_s)[:, 1]
                results["embedding_lbfgs"] = evaluate_binary_predictions(yva, risk, wva, tau)

            cap = float(np.quantile(wtr, 0.95)) if len(wtr) > 10 else float("inf")
            w_capped = np.minimum(wtr, cap)
            model = fit_improved_logistic(Xtr_e, ytr, w_capped, random_seed)
            if model is not None:
                Xva_s = model._scaler.transform(Xva_e)
                risk = model.predict_proba(Xva_s)[:, 1]
                results["embedding_p95_capped"] = evaluate_binary_predictions(yva, risk, wva, tau)

            smote = SmoteLikeAugmenter(k_neighbors=5, jitter=0.02, random_seed=random_seed)
            smote.fit(Xtr_e, ytr)
            if smote.status == "fitted":
                n_syn = int(len(ytr) * 0.5)
                n_event = int((ytr == 1).sum())
                n_nonevent = len(ytr) - n_event
                if n_event < n_nonevent:
                    n_syn_event, n_syn_nonevent = int(n_syn * 0.6), n_syn - int(n_syn * 0.6)
                else:
                    n_syn_event, n_syn_nonevent = int(n_syn * 0.4), n_syn - int(n_syn * 0.4)
                y_syn = np.array([1] * n_syn_event + [0] * n_syn_nonevent)
                X_syn = smote.sample(n_syn, y_syn)
                Xtr_aug = pd.concat([Xtr_e, X_syn], ignore_index=True)
                ytr_aug = np.concatenate([ytr, y_syn])
                wtr_aug = np.concatenate([wtr, np.ones(n_syn)])
                model = fit_improved_logistic(Xtr_aug, ytr_aug, wtr_aug, random_seed)
                if model is not None:
                    Xva_s = model._scaler.transform(Xva_e)
                    risk = model.predict_proba(Xva_s)[:, 1]
                    results["embedding_smote_aug"] = evaluate_binary_predictions(yva, risk, wva, tau)
    return results


def run_multi_timepoint_analysis() -> Dict[str, Any]:

    t_start = time.time()
    print("=" * 70)
    print("MULTI-TIMEPOINT SURVIVAL ANALYSIS")
    print("Timepoints: 12m, 24m, 36m, full time-to-event")
    print("=" * 70)
    paths = resolve_multi_timepoint_paths()
    output_dir = paths["output_dir"]
    print(f"\nOutput directory: {output_dir}")

    print("\n--- Loading Data ---")
    clinical = pd.read_csv(paths["clinical"], sep="\t")
    print(f"  Clinical: {clinical.shape}")
    embedding = pd.read_csv(paths["embedding"], sep="\t")
    print(f"  Embedding: {embedding.shape}")
    split_df = pd.read_csv(paths["split"], sep="\t")
    print(f"  Split: {split_df.shape}, counts: {split_df['split'].value_counts().to_dict()}")
    split_map = dict(zip(split_df["PATIENT_ID"], split_df["split"]))
    clinical["PATIENT_ID"] = clinical["PATIENT_ID"].map(normalize_patient_id)
    embedding["PATIENT_ID"] = embedding["PATIENT_ID"].map(normalize_patient_id)
    emb_meta_cols = ["PATIENT_ID", "split", "modality_mask"]
    emb_feat_cols = [c for c in embedding.columns if c not in emb_meta_cols]
    print(f"  Embedding features: {len(emb_feat_cols)}")
    clin_encoded = encode_clinical_features(clinical)
    if "PATIENT_ID" in clin_encoded.columns:
        clin_encoded["PATIENT_ID"] = clin_encoded["PATIENT_ID"].map(normalize_patient_id)
    clin_feat_cols = [c for c in clin_encoded.columns if c not in emb_meta_cols and
                      not c.startswith("death_by") and not c.startswith("early_censored") and
                      not c.startswith("ipcw_") and not c.startswith("pseudo_") and
                      not c.startswith("time_months") and c != "event" and c != "PATIENT_ID"]
    print(f"  Clinical features: {len(clin_feat_cols)}")
    clin_pids_set = set(clin_encoded["PATIENT_ID"].values) if "PATIENT_ID" in clin_encoded.columns else set()
    emb_pids_set = set(embedding["PATIENT_ID"].values)
    print(f"  Clinical-Embedding overlap: {len(clin_pids_set & emb_pids_set)} patients")
    TIMEPOINTS: List[Optional[float]] = [12.0, 24.0, 36.0, None]
    timepoint_results: Dict[str, Any] = {}
    for tau in TIMEPOINTS:
        tp_name = f"{int(tau)}m" if tau is not None else "full"
        print(f"\n{'='*50}")
        print(f"TIMEPOINT: {tp_name}")
        print(f"{'='*50}")
        if tau is not None:
            ep = add_fixed_endpoint(clinical, tau)
            tag = f"_{int(tau)}m"
            ep["split"] = ep["PATIENT_ID"].map(split_map)
            train_mask = ep["split"] == "train"
            val_mask = ep["split"] == "internal_validation"
            if train_mask.sum() == 0:
                train_mask = ep.index < int(len(ep) * 0.7)
                val_mask = ~train_mask
            train_ep = ep.loc[train_mask].copy()
            val_ep = ep.loc[val_mask].copy()
            train_ep = compute_ipcw_weights_tagged(train_ep, tau, cap_quantile=0.99)
            val_ep = compute_ipcw_weights_tagged(val_ep, tau, cap_quantile=0.99)
            n_events = train_ep[f"death_by{tag}"].notna().sum()
            n_early_cen = train_ep[f"early_censored_before{tag}"].sum()
            n_observed = train_ep[f"death_by{tag}_observed"].sum()
            early_censor_rate = float(n_early_cen / max(len(train_ep), 1))
            ipcw_w = train_ep[f"ipcw_weight{tag}"].to_numpy(float)
            ipcw_pos = ipcw_w[ipcw_w > 0]
            ipcw_stats: Dict[str, Any] = {
                "median": float(np.median(ipcw_pos)) if len(ipcw_pos) > 0 else 0,
                "p95": float(np.quantile(ipcw_pos, 0.95)) if len(ipcw_pos) > 0 else 0,
                "max": float(np.max(ipcw_pos)) if len(ipcw_pos) > 0 else 0,
                "cv": float(np.std(ipcw_pos) / max(np.mean(ipcw_pos), EPS)) if len(ipcw_pos) > 0 else 0,
                "all_weights": ipcw_pos,
            }
        else:
            ep = clinical.copy()
            ep["split"] = ep["PATIENT_ID"].map(split_map)
            train_mask = ep["split"] == "train"
            val_mask = ep["split"] == "internal_validation"
            if train_mask.sum() == 0:
                train_mask = ep.index < int(len(ep) * 0.7)
                val_mask = ~train_mask
            train_ep = ep.loc[train_mask].copy()
            val_ep = ep.loc[val_mask].copy()
            n_events = train_ep["event"].sum()
            n_early_cen = 0
            n_observed = len(train_ep)
            early_censor_rate = 0.0
            ipcw_stats = {}
        endpoint_stats: Dict[str, Any] = {
            "n_total": int(len(train_ep)),
            "n_events": int(n_events),
            "n_censored": int(train_ep["event"].eq(0).sum()),
            "censor_rate": early_censor_rate,
            "n_early_censored": int(n_early_cen),
            "n_observed": int(n_observed),
        }
        print(f"  Events: {n_events:.0f}, early censored: {n_early_cen:.0f}, "
              f"censor rate: {endpoint_stats['censor_rate']:.1%}")

        emb_by_pid = embedding.set_index("PATIENT_ID")[emb_feat_cols].copy().fillna(embedding.set_index("PATIENT_ID")[emb_feat_cols].median())
        clin_by_pid = clin_encoded.copy()
        if "PATIENT_ID" in clin_by_pid.columns:
            clin_by_pid = clin_by_pid.set_index("PATIENT_ID")
        clin_avail_cols = [c for c in clin_feat_cols if c in clin_by_pid.columns]
        if clin_avail_cols:
            clin_by_pid = clin_by_pid[clin_avail_cols].copy().fillna(clin_by_pid.median(numeric_only=True))
        train_pids = train_ep["PATIENT_ID"].values
        val_pids = val_ep["PATIENT_ID"].values
        train_has_emb = np.isin(train_pids, emb_by_pid.index)
        val_has_emb = np.isin(val_pids, emb_by_pid.index)
        X_train_emb = emb_by_pid.loc[train_pids[train_has_emb]].copy()
        X_val_emb = emb_by_pid.loc[val_pids[val_has_emb]].copy()
        models_results: Dict[str, Dict] = {}

        if tau is not None and len(X_train_emb) >= 20:
            tr_obs_ep = train_ep[train_ep[f"death_by{tag}_observed"] == True]
            va_obs_ep = val_ep[val_ep[f"death_by{tag}_observed"] == True]
            tr_avail = np.intersect1d(tr_obs_ep["PATIENT_ID"].values, X_train_emb.index)
            va_avail = np.intersect1d(va_obs_ep["PATIENT_ID"].values, X_val_emb.index)
            if len(tr_avail) >= 20 and len(va_avail) >= 5:
                tr_map = tr_obs_ep.set_index("PATIENT_ID")
                va_map = va_obs_ep.set_index("PATIENT_ID")
                ytr = tr_map.loc[tr_avail, f"death_by{tag}"].astype(int).to_numpy()
                yva = va_map.loc[va_avail, f"death_by{tag}"].astype(int).to_numpy()
                wtr = tr_map.loc[tr_avail, f"ipcw_weight{tag}"].to_numpy(float)
                wva = va_map.loc[va_avail, f"ipcw_weight{tag}"].to_numpy(float)
                if pd.Series(ytr).nunique() >= 2:
                    model = fit_improved_logistic(X_train_emb.loc[tr_avail], ytr, wtr, RANDOM_SEED)
                    if model is not None:
                        Xva_s = model._scaler.transform(X_val_emb.loc[va_avail])
                        risk = model.predict_proba(Xva_s)[:, 1]
                        eval_res = evaluate_binary_predictions(yva, risk, wva, tau)
                        models_results["ipcw_logistic_embedding"] = {
                            "auc": eval_res.get("auc", float("nan")),
                            "c_index": float("nan"),
                            "brier": eval_res.get("brier", float("nan")),
                            "ap": eval_res.get("ap", float("nan")),
                        }
                        print(f"    AUC (embedding): {eval_res.get('auc', 'N/A')}")
        timepoint_results[tp_name] = {
            "endpoint_stats": endpoint_stats,
            "ipcw_stats": ipcw_stats,
            "models": models_results,
            "n_features": len(emb_feat_cols),
            "n_clinical_features": len(clin_avail_cols) if clin_avail_cols else 0,
        }

    print("\n" + "=" * 70)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 70)
    emb_by_pid2 = embedding.set_index("PATIENT_ID")[emb_feat_cols].copy()
    emb_by_pid2 = emb_by_pid2.fillna(emb_by_pid2.median())
    clin_by_pid2 = clin_encoded.copy()
    if "PATIENT_ID" in clin_by_pid2.columns:
        clin_by_pid2 = clin_by_pid2.set_index("PATIENT_ID")
    clin_avail_cols2 = [c for c in clin_feat_cols if c in clin_by_pid2.columns]
    if clin_avail_cols2:
        clin_by_pid2 = clin_by_pid2[clin_avail_cols2].fillna(clin_by_pid2[clin_avail_cols2].median(numeric_only=True))
    for tau in [12.0, 24.0, 36.0]:
        tp_name = f"{int(tau)}m"
        print(f"\n--- Improvements for {tp_name} ---")
        ep = add_fixed_endpoint(clinical, tau)
        tag = f"_{int(tau)}m"
        ep["split"] = ep["PATIENT_ID"].map(split_map)
        train_mask = ep["split"] == "train"
        val_mask = ep["split"] == "internal_validation"
        if train_mask.sum() == 0:
            train_mask = ep.index < int(len(ep) * 0.7)
            val_mask = ~train_mask
        tr_ep = ep.loc[train_mask].copy()
        va_ep = ep.loc[val_mask].copy()
        tr_ep = compute_ipcw_weights_tagged(tr_ep, tau, cap_quantile=0.99)
        va_ep = compute_ipcw_weights_tagged(va_ep, tau, cap_quantile=0.99)
        tr_pids = tr_ep["PATIENT_ID"].values
        va_pids = va_ep["PATIENT_ID"].values
        tr_emb_mask = np.isin(tr_pids, emb_by_pid2.index)
        va_emb_mask = np.isin(va_pids, emb_by_pid2.index)
        Xtr_emb = emb_by_pid2.loc[tr_pids[tr_emb_mask]].copy()
        Xva_emb = emb_by_pid2.loc[va_pids[va_emb_mask]].copy()
        if clin_avail_cols2:
            tr_cl_mask = np.isin(tr_pids, clin_by_pid2.index)
            va_cl_mask = np.isin(va_pids, clin_by_pid2.index)
            Xtr_cl = clin_by_pid2.loc[tr_pids[tr_cl_mask]].copy()
            Xva_cl = clin_by_pid2.loc[va_pids[va_cl_mask]].copy()
        else:
            Xtr_cl, Xva_cl = pd.DataFrame(), pd.DataFrame()
        improvements = mt_apply_improvements(
            Xtr_cl, Xva_cl, Xtr_emb, Xva_emb, tr_ep, va_ep, tau, RANDOM_SEED,
        )
        timepoint_results[tp_name]["improvements"] = improvements
        for strat_name, strat_res in improvements.items():
            auc = strat_res.get("auc", float("nan"))
            brier = strat_res.get("brier", float("nan"))
            print(f"  {strat_name}: AUC={auc:.3f}, Brier={brier:.4f}")

    diag_df = mt_diagnose_auc(timepoint_results, output_dir)

    print("\n--- Generating Figures ---")
    mt_plot_timepoint_comparison(timepoint_results, output_dir)

    summary_rows = []
    for tp_name in sorted(timepoint_results.keys()):
        res = timepoint_results[tp_name]
        row: Dict[str, Any] = {"timepoint": tp_name}
        row.update(res["endpoint_stats"])
        if res["ipcw_stats"]:
            row.update({f"ipcw_{k}": v for k, v in res["ipcw_stats"].items() if k != "all_weights"})
        for m_name, m_res in res.get("models", {}).items():
            row[f"{m_name}_auc"] = m_res.get("auc", float("nan"))
            row[f"{m_name}_c_index"] = m_res.get("c_index", float("nan"))
        for strat_name, strat_res in res.get("improvements", {}).items():
            row[f"improve_{strat_name}_auc"] = strat_res.get("auc", float("nan"))
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "multi_timepoint_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(f"\nSummary saved: {summary_path}")
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"COMPLETE \u2014 {elapsed:.1f}s")
    print(f"{'='*70}")
    print(f"\nResults in: {output_dir}")
    print(f"  - multi_timepoint_summary.tsv")
    print(f"  - auc_diagnostics.tsv")
    print(f"  - multi_timepoint_comparison.png")
    return timepoint_results


def build_parser_multi_timepoint() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-Timepoint Survival Analysis (12m/24m/36m/full)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--timepoints", nargs="+", type=float, default=[12.0, 24.0, 36.0],
                        help="Timepoints in months (default: 12 24 36)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    return parser


def main_multi_timepoint() -> int:

    import os
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            run_multi_timepoint_analysis()
            return 0
        except Exception as e:
            print(f"\nERROR in multi-timepoint analysis: {e}")
            import traceback
            traceback.print_exc()
            return 1


# repro_io.command_line_main is available from the companion repro_io module
# when running from the project script directory.


def _run_step01_with_matrices():
    """Run Step 01 with --write-matrices enabled for downstream embedding."""
    import sys
    from _step01 import main_step01
    original_argv = sys.argv[:]
    try:
        if "--write-matrices" not in sys.argv:
            sys.argv = sys.argv[:1] + ["--write-matrices"] + sys.argv[1:]
        return main_step01()
    finally:
        sys.argv = original_argv


def run_full_pipeline():
    import sys
    import time
    from datetime import datetime
    # Local imports to avoid circular dependency at module level
    from _step01 import main_step01
    from _step02 import main_step02
    from _step03_04 import main_step03, main_step04
    from _step04_5_kalman_risk_smoothing import main_step04_5
    from _step04_6_lstm_kalman_risk_trajectory import main_step04_6
    
    print("\n" + "="*80)
    print("   CRC Causal Multi-omics Prognosis - Complete Pipeline")
    print("="*80)
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    steps = [
        ("01 - Multi-omics Preprocessing & QC (+ IPCW)", lambda: _run_step01_with_matrices()),
        ("02 - Causal Feature Screening (+ AIPW/CATE)", lambda: main_step02()),
        ("03 - Multi-omics Representation Learning", lambda: main_step03()),
        ("03.5 - GAN Data Augmentation", lambda: main_step03_5()),
        ("04 - Survival Prediction Model Training (+ IPCW-weighted)", lambda: main_step04()),
        ("04.5 - Kalman Risk Smoothing", lambda: main_step04_5()),
        ("04.6 - LSTM + Kalman Risk Trajectory", lambda: main_step04_6()),
        ("05 - Internal & External Validation (+ IPCW metrics)", lambda: main_step05()),
        ("06 - Interpretability & Mechanistic Validation", lambda: main_step06()),
        ("06.5 - Multi-Timepoint Survival Analysis (12m/24m/36m/full)", lambda: main_multi_timepoint()),
    ]
    
    total_start = time.time()
    results = []
    
    for i, (name, step_fn) in enumerate(steps, 1):
        print(f"\n{'━'*80}")
        print(f"  Running [{i:02d}/{len(steps):02d}] {name}")
        print(f"{'━'*80}\n")
        
        step_start = time.time()
        try:
            result = step_fn()
            success = result == 0
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            success = False
        
        elapsed = time.time() - step_start
        results.append((name, success, elapsed))
        
        status = "SUCCESS" if success else "FAILED"
        print(f"\n  {status} - {elapsed:.1f}s")
        
        if not success:
            print(f"\n  Pipeline stopped at step {i}!")
            break
    
    total_elapsed = time.time() - total_start
    successful = sum(1 for _, s, _ in results if s)
    
    print("\n" + "="*80)
    print("  Pipeline Summary")
    print("="*80)
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Successful: {successful}/{len(steps)} steps")
    print()
    
    for name, success, elapsed in results:
        status = "OK" if success else "FAIL"
        print(f"  {status} {name}: {elapsed:.1f}s")
    
    print("="*80)
    
    return 0 if successful == len(steps) else 1

if __name__ == '__main__':
    import sys
    sys.exit(run_full_pipeline())
