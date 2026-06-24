from __future__ import annotations

import argparse
import json
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

def build_parser_step02() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Screen prognostic and causal-priority CRC features.")
    add_common_args(parser)
    parser.add_argument(
        "--top-variance-genes",
        type=int,
        default=2000,
        help="Top-variance gene cap for univariable screening (train-only variance).",
    )
    parser.add_argument(
        "--top-priority",
        type=int,
        default=30,
        help="Number of FDR-significant features to retain into causal-priority table.",
    )
    parser.add_argument(
        "--fdr-threshold",
        type=float,
        default=0.20,
        help="Univariable Cox BH-FDR threshold for candidate retention.",
    )
    return parser


def load_tcga_endpoint(cfg: dict[str, Any], endpoint: str) -> pd.DataFrame:
    preprocessed = Path(cfg["processed_root"]) / "preprocessed" / f"tcga_{endpoint.lower()}_clinical_endpoint_qc.tsv"
    if preprocessed.exists():
        df = pd.read_csv(preprocessed, sep="\t", low_memory=False)
    else:
        df = read_cbio_table(cfg["cohorts"]["tcga"]["clinical_patient"])
        status_col, time_col = f"{endpoint}_STATUS", f"{endpoint}_MONTHS"
        if status_col not in df.columns or time_col not in df.columns:
            if endpoint != "OS":
                raise ValueError(f"Endpoint {endpoint} not available in TCGA clinical file.")
            status_col, time_col = "OS_STATUS", "OS_MONTHS"
        df[f"{endpoint}_EVENT"] = parse_survival_status(df[status_col])
        df[f"{endpoint}_TIME_MONTHS"] = numeric_series(df[time_col])
    time_col = f"{endpoint}_TIME_MONTHS"
    event_col = f"{endpoint}_EVENT"
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[event_col] = df[event_col].fillna(False).astype(bool)
    return df.loc[df["PATIENT_ID"].notna() & df[time_col].notna() & (df[time_col] > 0)].copy()


def attach_clinical_confounders(clinical: pd.DataFrame, raw_clinical_path: str) -> pd.DataFrame:

    needed = ["AJCC_PATHOLOGIC_TUMOR_STAGE", "SUBTYPE", "AGE", "AGE_AT_DIAGNOSIS", "SEX", "CANCER_TYPE_ACRONYM"]
    missing = [c for c in needed if c not in clinical.columns]
    if not missing:
        return clinical
    raw = read_cbio_table(raw_clinical_path)
    merge_cols = ["PATIENT_ID"] + [c for c in needed if c in raw.columns]
    if "PATIENT_ID" not in raw.columns:
        return clinical
    out = clinical.merge(raw[merge_cols], on="PATIENT_ID", how="left", suffixes=("", "_raw"))
    if "AGE" not in out.columns and "AGE_AT_DIAGNOSIS" in out.columns:
        out["AGE"] = pd.to_numeric(out["AGE_AT_DIAGNOSIS"], errors="coerce")
    return out


def train_variance_screen(
    feature_df: pd.DataFrame,
    train_ids: list[str],
    min_nonmissing: float,
    top_k: int,
) -> tuple[list[str], pd.DataFrame]:

    train_mat = feature_df.reindex(train_ids)
    nonmissing = train_mat.notna().mean(axis=0)
    keep_mask = nonmissing >= min_nonmissing
    train_kept = train_mat.loc[:, keep_mask]
    variances = train_kept.var(axis=0, skipna=True)
    order = variances.sort_values(ascending=False)
    selected = order.head(top_k).index.tolist()
    diag = pd.DataFrame(
        {
            "feature": variances.index,
            "train_variance": variances.values,
            "train_nonmissing_ratio": nonmissing.loc[variances.index].values,
        }
    )
    diag["selected_for_univariable_cox"] = diag["feature"].isin(selected)
    return selected, diag


def univariable_cox(
    clinical: pd.DataFrame,
    features: pd.DataFrame,
    endpoint: str,
    min_events: int,
    min_unique: int,
) -> pd.DataFrame:

    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    time_col = f"{endpoint}_TIME_MONTHS"
    event_col = f"{endpoint}_EVENT"
    clin = clinical[["PATIENT_ID", time_col, event_col]].copy()
    clin["PATIENT_ID"] = clin["PATIENT_ID"].astype(str)
    clin[time_col] = pd.to_numeric(clin[time_col], errors="coerce")
    clin[event_col] = clin[event_col].fillna(False).astype(bool)
    clin = clin.dropna(subset=[time_col])
    clin = clin[clin[time_col] > 0].set_index("PATIENT_ID")
    common = clin.index.intersection(features.index)
    rows: list[dict[str, Any]] = []
    for feature in features.columns:
        x = pd.to_numeric(features.loc[common, feature], errors="coerce")
        df = pd.DataFrame(
            {
                "time": clin.loc[common, time_col].astype(float),
                "event": clin.loc[common, event_col].astype(int),
                "feature": x,
            }
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if df["event"].sum() < min_events or df["feature"].nunique() < min_unique:
            rows.append({"feature": feature, "n": len(df), "events": int(df["event"].sum()), "coef": np.nan, "hr": np.nan, "p": np.nan, "ph_p": np.nan, "status": "insufficient_events_or_variance"})
            continue
        try:
            cph = CoxPHFitter(penalizer=0.0)
            cph.fit(df, duration_col="time", event_col="event")
            summary = cph.summary.loc["feature"]
            try:
                ph_p = float(proportional_hazard_test(cph, df, time_transform="rank").summary.loc["feature", "p"])
            except Exception:
                ph_p = np.nan
            rows.append(
                {
                    "feature": feature,
                    "n": int(len(df)),
                    "events": int(df["event"].sum()),
                    "coef": float(summary["coef"]),
                    "hr": float(summary["exp(coef)"]),
                    "p": float(summary["p"]),
                    "ph_p": ph_p,
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append({"feature": feature, "n": len(df), "events": int(df["event"].sum()), "coef": np.nan, "hr": np.nan, "p": np.nan, "ph_p": np.nan, "status": f"cox_failed:{type(exc).__name__}"})
    out = pd.DataFrame(rows)
    ok = out["status"] == "ok"
    out["fdr"] = np.nan
    if ok.any():
        out.loc[ok, "fdr"] = bh_fdr(out.loc[ok, "p"].tolist())
    out["likely_pseudogene"] = out["feature"].apply(is_likely_pseudogene)
    return out.sort_values(["fdr", "p"], na_position="last")


def multivariable_cox(
    clinical: pd.DataFrame,
    features: pd.DataFrame,
    endpoint: str,
    candidate_features: list[str],
    confounders: list[str],
    penalizer: float = 0.05,
) -> pd.DataFrame:

    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    time_col = f"{endpoint}_TIME_MONTHS"
    event_col = f"{endpoint}_EVENT"
    clin = clinical.set_index(clinical["PATIENT_ID"].astype(str))
    used_confounders = [c for c in confounders if c in clin.columns]
    if not used_confounders:
        return pd.DataFrame()
    cf = pd.get_dummies(clin[used_confounders], drop_first=True, dummy_na=False)
    cf = cf.apply(pd.to_numeric, errors="coerce")
    cf = cf.loc[:, cf.var(axis=0, skipna=True) > 0]
    common = cf.index.intersection(features.index)
    base = pd.DataFrame(
        {
            "time": pd.to_numeric(clin.loc[common, time_col], errors="coerce").astype(float),
            "event": clin.loc[common, event_col].astype(bool).astype(int),
        },
        index=common,
    )
    cf = cf.loc[common]
    rows: list[dict[str, Any]] = []
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
            rows.append(
                {
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
                }
            )
        except Exception as exc:
            rows.append({"feature": feature, "status": f"cox_failed:{type(exc).__name__}"})
    out = pd.DataFrame(rows)
    ok = out.get("status", pd.Series(dtype=str)) == "ok"
    out["fdr_adj"] = np.nan
    if ok.any():
        out.loc[ok, "fdr_adj"] = bh_fdr(out.loc[ok, "p_adj"].tolist())
    return out


def main_step02() -> int:
    parser = build_parser_step02()
    args = parser.parse_args()
    ctx = initialize_run(__file__, args)
    cfg = ctx.cfg
    endpoint = args.endpoint
    filters = cfg.get("filters", {"min_nonmissing_ratio_train": 0.7, "min_events_per_feature": 5, "min_unique_values_per_feature": 3})
    min_nonmissing_ratio = float(filters.get("min_nonmissing_ratio_train", filters.get("min_nonmissing_ratio", 0.7)))
    causal_dir = ctx.data_dir("causal_screening")
    rna_path = cfg["cohorts"]["tcga"]["rna"]
    clinical_path = cfg["cohorts"]["tcga"]["clinical_patient"]
    split_path = Path(cfg["processed_root"]) / "survival_models" / "tcga_train_internal_validation_split.tsv"
    input_files = [clinical_path, rna_path, str(split_path)]

    clinical = load_tcga_endpoint(cfg, endpoint)
    clinical = attach_clinical_confounders(clinical, clinical_path)
    ctx.logger.info("TCGA clinical rows after QC: %d, events: %d", len(clinical), int(clinical[f"{endpoint}_EVENT"].sum()))

    if args.dry_run:
        rows = [{"check": "tcga_clinical_rows", "value": int(len(clinical))}, {"check": "tcga_rna_path_exists", "value": int(Path(rna_path).exists())}]
        ctx.write_table(ctx.run_dir / "causal_screening_dry_run_input_checks.tsv", rows, "qc", "Dry-run input checks.")
        ctx.add_warning("Dry-run only: no Cox, multivariable adjustment, or QTL lookup executed.")
        ctx.finalize([p for p in input_files if Path(p).exists()])
        return 0

    ctx.logger.info("Loading full TCGA RNA matrix (this can take a minute)...")
    feature_df = stream_gene_matrix(rna_path, comment="#", id_col_preference="Hugo_Symbol")
    ctx.logger.info("Loaded RNA: patients=%d, genes=%d", feature_df.shape[0], feature_df.shape[1])

    if split_path.exists():
        split = pd.read_csv(split_path, sep="\t")
        train_ids = split.loc[split["split"] == "train", "PATIENT_ID"].astype(str).tolist()
        ctx.logger.info("Using existing train/internal split with n_train=%d", len(train_ids))
    else:
        train_ids = clinical["PATIENT_ID"].astype(str).tolist()
        ctx.add_warning("No train/internal split file found; using full TCGA cohort for screening. Re-run after 04 creates the split for strict leakage control.")

    selected, variance_diag = train_variance_screen(
        feature_df,
        train_ids,
        min_nonmissing=min_nonmissing_ratio,
        top_k=int(args.top_variance_genes),
    )
    ctx.write_table(causal_dir / "tcga_train_only_variance_prescreen_diagnostics.tsv", variance_diag, "analysis_data", "Train-only variance pre-screen diagnostics (per-gene).")
    ctx.logger.info("Variance pre-screen kept %d/%d genes (top-k=%d).", len(selected), feature_df.shape[1], args.top_variance_genes)

    feature_subset = feature_df[selected]
    train_clinical = clinical[clinical["PATIENT_ID"].astype(str).isin(train_ids)] if train_ids else clinical
    univ = univariable_cox(
        train_clinical,
        feature_subset.reindex(train_clinical["PATIENT_ID"].astype(str)),
        endpoint,
        min_events=int(filters.get("min_events_per_feature", 5)),
        min_unique=int(filters.get("min_unique_values_per_feature", 3)),
    )
    ctx.write_table(causal_dir / "tcga_univariable_cox_prognostic_candidates.tsv", univ, "analysis_data", "Train-only univariable Cox prognostic candidates with PH-test p and pseudogene flag.")

    univ_ok = univ[univ["status"] == "ok"].copy()
    ok = univ_ok[univ_ok["fdr"].notna()]
    fdr_sig = ok[ok["fdr"] <= args.fdr_threshold].sort_values("fdr")
    fdr_sig = fdr_sig[~fdr_sig["likely_pseudogene"]].head(args.top_priority)
    if fdr_sig.empty:
        ctx.add_warning(f"No protein-coding genes pass FDR ≤ {args.fdr_threshold}; falling back to top-{args.top_priority} by p value (excluding pseudogenes) for downstream pipeline continuity.")
        fdr_sig = ok[~ok["likely_pseudogene"]].sort_values("p").head(args.top_priority)
    candidate_features = fdr_sig["feature"].tolist()
    ctx.logger.info("Univariable FDR-passed protein-coding candidates: %d", len(candidate_features))

    multivariable = multivariable_cox(
        clinical,
        feature_df[candidate_features],
        endpoint,
        candidate_features,
        confounders=["AGE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "SUBTYPE", "CANCER_TYPE_ACRONYM"],
        penalizer=0.05,
    )
    ctx.write_table(causal_dir / "tcga_multivariable_cox_confounder_adjusted_features.tsv", multivariable, "analysis_data", "Multivariable Cox adjusted for clinical confounders.")

    eqtl_table = cis_eqtl_evidence(candidate_features, cfg["cohorts"].get("qtl", {}), ctx.logger)
    if not eqtl_table.empty:
        ctx.write_table(causal_dir / "gtex_colon_cis_eqtl_instrument_availability.tsv", eqtl_table, "analysis_data", "GTEx colon cis-eQTL instrument availability for candidates.")

    base = univ[univ["feature"].isin(candidate_features)][["feature", "n", "events", "coef", "hr", "p", "fdr", "ph_p", "likely_pseudogene"]].copy()
    base = base.rename(columns={"coef": "coef_univariable", "hr": "hr_univariable", "p": "p_univariable", "fdr": "fdr_univariable", "ph_p": "ph_assumption_p"})
    if not multivariable.empty:
        base = base.merge(multivariable, on="feature", how="left")
    if not eqtl_table.empty:
        base = base.merge(eqtl_table, on="feature", how="left")
    
    _has_eqtl = base["has_cis_eqtl_instrument"] if "has_cis_eqtl_instrument" in base.columns else pd.Series(False, index=base.index)
    _p_adj = base["p_adj"] if "p_adj" in base.columns else pd.Series(1.0, index=base.index)
    base["causal_evidence_level"] = np.where(
        _has_eqtl.fillna(False) & (_p_adj.fillna(1) < 0.10),
        "observational_cox_adjusted_with_cis_eqtl_instrument",
        np.where(
            _p_adj.fillna(1) < 0.10,
            "observational_cox_adjusted_only",
            "univariable_prognostic_only",
        ),
    )
    base["interpretation"] = "Prognostic association candidate from Cox + AIPW screening."
    ctx.write_table(causal_dir / "causal_priority_feature_table.tsv", base, "analysis_data", "Causal-priority feature table with multivariable Cox + cis-eQTL evidence and explicit limitations.")

    plot_volcano(univ, ctx.output_path(ctx.run_dir / "tcga_univariable_cox_volcano.png", "figure"), args.fdr_threshold)
    ctx.register_output(ctx.run_dir / "tcga_univariable_cox_volcano.png", "figure", "Volcano plot of univariable Cox screening with FDR threshold.")


    try:
        ipcw_path = Path(cfg["processed_root"]) / "preprocessed" / "tcga_ipcw_os_endpoint.tsv"
        if ipcw_path.exists() and candidate_features:
            ipcw_ep = pd.read_csv(ipcw_path, sep="\t", index_col=0)
            train_clinical_ids = train_clinical["PATIENT_ID"].astype(str).tolist()
            adj_cols = [c for c in ["AGE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE"] if c in train_clinical.columns]
            if adj_cols:
                clinical_adj = train_clinical.set_index("PATIENT_ID")[adj_cols].copy()
                if "SEX" in clinical_adj.columns:
                    clinical_adj["SEX"] = clinical_adj["SEX"].map({"Male": 1, "Female": 0}).fillna(clinical_adj["SEX"])
                if "AJCC_PATHOLOGIC_TUMOR_STAGE" in clinical_adj.columns:
                    _stage_map = {"STAGE I": 1, "STAGE IA": 2, "STAGE IB": 3,
                                  "STAGE II": 4, "STAGE IIA": 5, "STAGE IIB": 6, "STAGE IIC": 7,
                                  "STAGE III": 8, "STAGE IIIA": 9, "STAGE IIIB": 10, "STAGE IIIC": 11,
                                  "STAGE IV": 12, "STAGE IVA": 13, "STAGE IVB": 14, "STAGE IVC": 15}
                    clinical_adj["AJCC_PATHOLOGIC_TUMOR_STAGE"] = (
                        clinical_adj["AJCC_PATHOLOGIC_TUMOR_STAGE"].astype(str).str.strip().str.upper()
                        .map(_stage_map)
                    )
                clinical_adj = clinical_adj.astype(float, errors="ignore")
                feature_matrix_train = feature_df[candidate_features].reindex(train_clinical_ids).set_index(pd.Index(train_clinical_ids))
                aipw_table = run_causal_screening(
                    feature_matrix_train, clinical_adj, ipcw_ep.reindex(train_clinical_ids),
                    random_seed=cfg["random_seed"], max_features=len(candidate_features),
                )
                if not aipw_table.empty:
                    ctx.write_table(causal_dir / "aipw_causal_screening_table.tsv", aipw_table,
                                    "analysis_data", "AIPW doubly-robust causal screening with CATE + dose-response.")

                    base = base.merge(
                        aipw_table[["feature", "ate", "p_value", "causal_priority_score",
                                     "dose_response_slope", "spearman_with_pseudo_risk"]],
                        on="feature", how="left", suffixes=("", "_aipw"),
                    )
                    ctx.write_table(causal_dir / "causal_priority_feature_table.tsv", base,
                                    "analysis_data", "Updated causal-priority table with AIPW evidence.")
                    ctx.logger.info("AIPW screening: %d features scored", len(aipw_table))
            else:
                ctx.add_warning("No clinical adjustment columns available for AIPW screening.")
        else:
            ctx.add_warning("IPCW endpoint or candidate features missing; AIPW screening skipped.")
    except Exception as exc:
        ctx.add_warning(f"AIPW causal screening failed: {exc}")

    
    ctx.finalize([p for p in input_files if Path(p).exists()])
    return 0


import joblib
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

