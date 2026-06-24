import os
import sys
import argparse
import json
import pickle
import warnings
import math

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import hypergeom

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

FIXED_TAU_MONTHS = 36.0
RANDOM_SEED = 42
EPS = 1e-8
PSEUDOGENE_HINTS = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9")
STAGE_MAP = {
    "STAGE I": 1, "STAGE IA": 2, "STAGE IB": 3,
    "STAGE II": 4, "STAGE IIA": 5, "STAGE IIB": 6, "STAGE IIC": 7,
    "STAGE III": 8, "STAGE IIIA": 9, "STAGE IIIB": 10, "STAGE IIIC": 11,
    "STAGE IV": 12, "STAGE IVA": 13, "STAGE IVB": 14, "STAGE IVC": 15,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "DATA")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")


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


def patient_id_from_sample(value):
    text = str(value)
    if text.startswith("TCGA-") and len(text) >= 12:
        return text[:12]
    return text.split("-")[0] if "-" in text else text


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


def causal_rmst_doubly_robust(X, exposure, time_months, event, tau=FIXED_TAU_MONTHS, random_seed=RANDOM_SEED):
    data = pd.concat([X, exposure.rename("A")], axis=1).dropna()
    if len(data) < 30 or data["A"].nunique() < 2:
        return {"ate_rmst": np.nan, "ate_rmst_se": np.nan, "p_value": np.nan}

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
        return {"ate_rmst": np.nan, "ate_rmst_se": np.nan, "p_value": np.nan}

    T_capped = np.minimum(T, tau)

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

    return {
        "ate_rmst": ate,
        "ate_rmst_se": se,
        "p_value": p,
        "rmst_exposed": float(np.mean(T_capped[A == 1])),
        "rmst_unexposed": float(np.mean(T_capped[A == 0])),
    }


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
                         random_seed, max_features):
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
            tau=FIXED_TAU_MONTHS, random_seed=random_seed,
        )
        cate = causal_cate_summary(X_adj, exposure_binary, outcome_time, random_seed)
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
    results = []
    if not os.path.exists(cptac_protein_path) or not os.path.exists(cptac_rna_path):
        return pd.DataFrame(results)

    try:
        prot = pd.read_csv(cptac_protein_path, sep="\t", low_memory=False)
        rna = pd.read_csv(cptac_rna_path, sep="\t", low_memory=False)
    except Exception:
        return pd.DataFrame(results)

    prot_id_col = prot.columns[0]
    rna_id_col = rna.columns[0]
    prot = prot.set_index(prot_id_col)
    rna = rna.set_index(rna_id_col)
    prot = prot.apply(pd.to_numeric, errors="coerce")
    rna = rna.apply(pd.to_numeric, errors="coerce")

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


def build_evidence_matrix(causal_table, univar_table, multivar_table, cptac_table,
                          candidate_genes, out_dir):
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

    evidence = evidence.drop_duplicates(subset=["feature"]).reset_index(drop=True)
    evidence.to_csv(os.path.join(out_dir, "core_biomarker_evidence_matrix.tsv"), sep="\t", index=False)

    try:
        indicator_cols = [c for c in ["sign_agreement"] if c in evidence.columns]
        plot_evidence = evidence.head(30)
        if indicator_cols and not plot_evidence.empty:
            heat = plot_evidence.set_index("feature")[indicator_cols].replace(
                {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}
            )
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
    return series.astype(str).str.strip().str.upper().map(STAGE_MAP)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()

    timestamp = args.timestamp
    out_dir = os.path.join(RESULTS_DIR, timestamp, "02_gene_features")
    os.makedirs(out_dir, exist_ok=True)

    prev_dir = os.path.join(RESULTS_DIR, timestamp, "01_preprocessing")

    clinical_path = os.path.join(prev_dir, "tcga_os_clinical_endpoint_qc.tsv")
    survival_labels_path = os.path.join(prev_dir, "survival_labels.pkl")
    gene_expr_path = os.path.join(prev_dir, "gene_expression_curated.tsv")
    sample_ids_path = os.path.join(prev_dir, "sample_ids.pkl")

    hallmark_path = os.path.join(DATA_DIR, "msigdb", "h.all.v2024.1.Hs.symbols.gmt")
    kegg_path = os.path.join(DATA_DIR, "msigdb", "c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt")
    go_bp_path = os.path.join(DATA_DIR, "msigdb", "c5.go.bp.v2024.1.Hs.symbols.gmt")
    causal_ref_path = os.path.join(DATA_DIR, "causal", "causal_priority_feature_table.tsv")
    split_path = os.path.join(DATA_DIR, "survival_models", "tcga_train_internal_validation_split.tsv")
    cptac_protein_path = os.path.join(DATA_DIR, "external", "cptac_coad_protein_gene_level_matrix.tsv")
    cptac_rna_path = os.path.join(DATA_DIR, "external", "cptac_coad_rna_gene_level_matrix.tsv")

    print("[02] Loading clinical endpoint...")
    clinical = pd.read_csv(clinical_path, sep="\t", low_memory=False)
    time_col = "OS_MONTHS"
    event_col = "OS_EVENT"
    clinical[time_col] = pd.to_numeric(clinical[time_col], errors="coerce")
    clinical[event_col] = clinical[event_col].fillna(False).astype(bool)
    clinical = clinical.loc[clinical["PATIENT_ID"].notna() & clinical[time_col].notna() & (clinical[time_col] > 0)].copy()
    print(f"[02] Clinical rows: {len(clinical)}, events: {int(clinical[event_col].sum())}")

    print("[02] Loading gene expression matrix...")
    if os.path.exists(gene_expr_path):
        feature_df = pd.read_csv(gene_expr_path, sep="\t", index_col=0, low_memory=False)
        feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    else:
        print("[02] gene_expression_curated.tsv not found, trying raw RNA matrix...")
        feature_df = pd.DataFrame()

    if os.path.exists(sample_ids_path):
        with open(sample_ids_path, "rb") as f:
            sample_ids = pickle.load(f)
    else:
        sample_ids = feature_df.index.tolist()

    print(f"[02] Gene expression: {feature_df.shape[0]} samples x {feature_df.shape[1]} genes")

    if os.path.exists(split_path):
        split = pd.read_csv(split_path, sep="\t")
        train_ids = split.loc[split["split"] == "train", "PATIENT_ID"].astype(str).tolist()
        print(f"[02] Using predefined split: n_train={len(train_ids)}")
    else:
        train_ids = clinical["PATIENT_ID"].astype(str).tolist()
        print(f"[02] No split file found, using full cohort: n={len(train_ids)}")

    print("[02] Step 1: Train variance screen...")
    selected, variance_diag = train_variance_screen(feature_df, train_ids, min_nonmissing=0.7, top_k=2000)
    variance_diag.to_csv(os.path.join(out_dir, "tcga_train_only_variance_prescreen.tsv"), sep="\t", index=False)
    print(f"[02] Variance pre-screen: {len(selected)}/{feature_df.shape[1]} genes retained")

    print("[02] Step 2: Univariable Cox screening...")
    feature_subset = feature_df[selected]
    train_clinical = clinical[clinical["PATIENT_ID"].astype(str).isin(train_ids)] if train_ids else clinical
    univ = univariate_cox(
        train_clinical,
        feature_subset.reindex(train_clinical["PATIENT_ID"].astype(str)),
        time_col, event_col,
        min_events=5, min_unique=3,
    )
    univ.to_csv(os.path.join(out_dir, "tcga_univariable_cox_feature_screening.tsv"), sep="\t", index=False)
    print(f"[02] Univariable Cox: {len(univ)} features tested")

    univ_ok = univ[univ["status"] == "ok"].copy()
    ok = univ_ok[univ_ok["fdr"].notna()]
    fdr_sig = ok[ok["fdr"] <= 0.20].sort_values("fdr")
    fdr_sig = fdr_sig[~fdr_sig["likely_pseudogene"]].head(30)
    if fdr_sig.empty:
        print("[02] No FDR-significant genes, falling back to top-30 by p-value...")
        fdr_sig = ok[~ok["likely_pseudogene"]].sort_values("p").head(30)
    candidate_features = fdr_sig["feature"].tolist()
    print(f"[02] FDR-passed candidates: {len(candidate_features)}")

    if not candidate_features:
        print("[02] No candidates passed screening. Saving empty outputs.")
        pd.DataFrame().to_csv(os.path.join(out_dir, "tcga_multivariable_cox_features.tsv"), sep="\t", index=False)
        pd.DataFrame().to_csv(os.path.join(out_dir, "causal_priority_feature_table.tsv"), sep="\t", index=False)
        pd.DataFrame().to_csv(os.path.join(out_dir, "pathway_enrichment_results.tsv"), sep="\t", index=False)
        pd.DataFrame().to_csv(os.path.join(out_dir, "core_biomarker_evidence_matrix.tsv"), sep="\t", index=False)
        with open(os.path.join(out_dir, "final_gene_list.pkl"), "wb") as f:
            pickle.dump([], f)
        json.dump({"candidates": 0, "timestamp": timestamp}, open(os.path.join(out_dir, "gene_feature_config.json"), "w"), indent=2)
        return 0

    print("[02] Step 3: Multivariable Cox (confounder-adjusted)...")
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
    print(f"[02] Multivariable Cox: {len(multivar)} features tested")

    print("[02] Step 4: Causal RMST doubly-robust screening...")
    causal_table = pd.DataFrame()
    if train_ids and candidate_features:
        train_clinical_ids = [pid for pid in train_ids if pid in clinical_adj["PATIENT_ID"].astype(str).values]
        adj_cols = [c for c in ["AGE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE"] if c in clinical_adj.columns]
        if adj_cols and train_clinical_ids:
            clin_adj = clinical_adj.set_index("PATIENT_ID").loc[
                clinical_adj["PATIENT_ID"].astype(str).isin(train_clinical_ids), adj_cols
            ].copy()
            if "SEX" in clin_adj.columns:
                clin_adj["SEX"] = clin_adj["SEX"].map({"Male": 1, "Female": 0}).fillna(clin_adj["SEX"])
            clin_adj = clin_adj.astype(float, errors="ignore")

            feature_matrix_train = feature_df[candidate_features].reindex(train_clinical_ids)
            feature_matrix_train.index = pd.Index(train_clinical_ids[:len(feature_matrix_train)])

            endpoint_train = pd.DataFrame({
                "time_months": pd.to_numeric(clinical_adj.set_index("PATIENT_ID").loc[
                    clinical_adj["PATIENT_ID"].astype(str).isin(train_clinical_ids), time_col
                ], errors="coerce").to_numpy(dtype=float),
                "event": clinical_adj.set_index("PATIENT_ID").loc[
                    clinical_adj["PATIENT_ID"].astype(str).isin(train_clinical_ids), event_col
                ].astype(int).to_numpy(),
            }, index=train_clinical_ids[:len(feature_matrix_train)])

            causal_table = run_causal_screening(
                feature_matrix_train, clin_adj, endpoint_train,
                random_seed=RANDOM_SEED, max_features=len(candidate_features),
            )
            print(f"[02] Causal screening: {len(causal_table)} features scored")

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

    print("[02] Step 5: Pathway enrichment...")
    background_genes = feature_df.columns.tolist()
    enrich = pathway_enrichment(candidate_features, background_genes,
                                hallmark_path, kegg_path, go_bp_path, out_dir)
    print(f"[02] Pathway enrichment: {len(enrich)} gene sets tested")

    print("[02] Step 6: CPTAC mechanism validation...")
    cptac_table = mechanism_validation_cptac(candidate_features, cptac_protein_path, cptac_rna_path, out_dir)
    print(f"[02] CPTAC validation: {len(cptac_table)} genes checked")

    print("[02] Step 7: Evidence matrix...")
    evidence = build_evidence_matrix(causal_table, univ, multivar, cptac_table,
                                      candidate_features, out_dir)

    print("[02] Step 8: Saving final outputs...")
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
        "n_variance_selected": len(selected),
        "n_univariable_tested": len(univ),
        "n_fdr_significant": len(fdr_sig),
        "n_candidates": len(candidate_features),
        "n_causal_scored": len(causal_table),
        "n_pathway_tested": len(enrich),
        "candidate_genes": candidate_features,
    }
    with open(os.path.join(out_dir, "gene_feature_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"[02] Complete. {len(candidate_features)} candidate genes saved.")
    print(f"[02] Outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
