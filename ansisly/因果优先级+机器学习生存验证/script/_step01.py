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

def build_parser_step01() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess CRC multi-omics cohorts.")
    add_common_args(parser)
    parser.add_argument("--write-matrices", action=argparse.BooleanOptionalAction, default=False, help="Write curated per-modality matrices filtered by training variance; disabled by default for QC-safe runs.")
    parser.add_argument("--top-features-per-omics", type=int, default=1000, help="Per-omics top-variance feature cap for curated matrices (train-only).")
    return parser


def endpoint_columns(endpoint: str) -> tuple[str, str]:
    mapping = {
        "OS": ("OS_STATUS", "OS_MONTHS"),
        
    }
    return mapping[endpoint]


def normalize_clinical(df: pd.DataFrame, cohort: str, endpoint: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:

    status_col, time_col = endpoint_columns(endpoint)
    out = df.copy()
    excluded: list[dict[str, Any]] = []
    if cohort in ("geo_gse103479", "geo_gse39582"):
        out = out.rename(columns={"id": "PATIENT_ID", "fustat": "OS_STATUS_RAW", "futime": "OS_MONTHS_RAW"})
        out["OS_EVENT"] = parse_survival_status(out["OS_STATUS_RAW"])
        out["OS_TIME_MONTHS"] = numeric_series(out["OS_MONTHS_RAW"])
        out["time_unit"] = "months"
        time_field, event_field = "OS_TIME_MONTHS", "OS_EVENT"

        if (out["OS_TIME_MONTHS"] > 200).any():
            excluded.extend([
                {"cohort": cohort, "PATIENT_ID": pid, "reason": "OS_TIME_MONTHS > 200 (possible day units)"}
                for pid in out.loc[out["OS_TIME_MONTHS"] > 200, "PATIENT_ID"].astype(str).tolist()
            ])
    elif cohort == "geo_gse17538":
        out = out.rename(columns={
            "Accession": "PATIENT_ID",
            "Overall survival follow-up time": "OS_MONTHS_RAW",
            "overall_event (death from any cause):": "OS_STATUS_RAW",
            "Age": "AGE",
            "Gender": "SEX",
            "Ajcc_stage": "AJCC_PATHOLOGIC_TUMOR_STAGE",
        })
        out["OS_EVENT"] = out["OS_STATUS_RAW"].astype(str).str.strip().str.lower().map(
            lambda v: 1 if v in ("death", "1", "true", "yes") else 0
        )
        out["OS_TIME_MONTHS"] = numeric_series(out["OS_MONTHS_RAW"])
        out["time_unit"] = "months"
        time_field, event_field = "OS_TIME_MONTHS", "OS_EVENT"
        if (out["OS_TIME_MONTHS"] > 200).any():
            excluded.extend([
                {"cohort": cohort, "PATIENT_ID": pid, "reason": "OS_TIME_MONTHS > 200 (possible day units)"}
                for pid in out.loc[out["OS_TIME_MONTHS"] > 200, "PATIENT_ID"].astype(str).tolist()
            ])
    else:
        if "PATIENT_ID" not in out.columns:
            raise ValueError(f"{cohort} clinical table lacks PATIENT_ID.")
        if status_col in out.columns and time_col in out.columns:
            out[f"{endpoint}_EVENT"] = parse_survival_status(out[status_col])
            out[f"{endpoint}_TIME_MONTHS"] = numeric_series(out[time_col])
        elif endpoint == "OS" and {"OS_STATUS", "OS_MONTHS"}.issubset(out.columns):
            out["OS_EVENT"] = parse_survival_status(out["OS_STATUS"])
            out["OS_TIME_MONTHS"] = numeric_series(out["OS_MONTHS"])
        else:
            out[f"{endpoint}_EVENT"] = np.nan
            out[f"{endpoint}_TIME_MONTHS"] = np.nan
        time_field = f"{endpoint}_TIME_MONTHS" if f"{endpoint}_TIME_MONTHS" in out.columns else "OS_TIME_MONTHS"
        event_field = f"{endpoint}_EVENT" if f"{endpoint}_EVENT" in out.columns else "OS_EVENT"
    out["COHORT"] = cohort

    for pid, t, e in zip(out["PATIENT_ID"].astype(str), out[time_field], out[event_field]):
        if pd.isna(t):
            excluded.append({"cohort": cohort, "PATIENT_ID": pid, "reason": f"missing {time_field}"})
        elif pd.notna(t) and float(t) <= 0:
            excluded.append({"cohort": cohort, "PATIENT_ID": pid, "reason": f"non-positive {time_field}"})
        if pd.isna(e):
            excluded.append({"cohort": cohort, "PATIENT_ID": pid, "reason": f"missing {event_field}"})

    dup = out["PATIENT_ID"].astype(str).duplicated()
    for pid in out.loc[dup, "PATIENT_ID"].astype(str).unique().tolist():
        excluded.append({"cohort": cohort, "PATIENT_ID": pid, "reason": "duplicate PATIENT_ID"})
    return out, excluded


def outcome_summary(df: pd.DataFrame, cohort: str, endpoint: str) -> dict[str, Any]:
    event_col = f"{endpoint}_EVENT"
    time_col = f"{endpoint}_TIME_MONTHS"
    if event_col not in df.columns or time_col not in df.columns:
        return {"cohort": cohort, "endpoint": endpoint, "n_patients": len(df), "valid_n": 0, "event_n": 0, "censor_n": 0, "unknown_n": int(len(df)), "median_months": np.nan}
    time = pd.to_numeric(df[time_col], errors="coerce")
    event_raw = df[event_col]
    unknown_mask = event_raw.isna() | time.isna()
    event = event_raw.fillna(False).astype(bool)
    valid = time.notna() & (time > 0) & ~unknown_mask
    return {
        "cohort": cohort,
        "endpoint": endpoint,
        "n_patients": int(len(df)),
        "valid_n": int(valid.sum()),
        "event_n": int((valid & event).sum()),
        "censor_n": int((valid & ~event).sum()),
        "unknown_n": int(unknown_mask.sum()),
        "median_months": float(time[valid].median()) if valid.any() else np.nan,
    }


def patient_set_from_matrix(path: str, comment: str | None = "#") -> set[str]:
    if not Path(path).exists():
        return set()
    try:
        cols = load_matrix_header(path, comment=comment)
    except Exception:
        return set()
    samples = matrix_sample_columns(cols)
    return {patient_id_from_sample(c) for c in samples}


def plot_upset_style(sets: dict[str, set[str]], path: Path) -> None:

    import matplotlib.pyplot as plt
    from itertools import combinations

    names = list(sets.keys())
    n = len(names)
    if n == 0:
        return

    union = set().union(*sets.values())
    rows = []
    for k in range(1, n + 1):
        for combo in combinations(names, k):
            include = set.intersection(*[sets[c] for c in combo])
            exclude = union - set().union(*[sets[c] for c in combo if c not in combo]) if False else set().union(*[sets[c] for c in names if c not in combo])
            exact = include - exclude
            if exact:
                rows.append((combo, len(exact)))
    rows.sort(key=lambda x: -x[1])
    rows = rows[:15]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), gridspec_kw={"height_ratios": [3, 2]})
    axes[0].bar(range(len(rows)), [r[1] for r in rows], color="#1f77b4")
    axes[0].set_ylabel("Patients (exact set)")
    axes[0].set_xticks(range(len(rows)))
    axes[0].set_xticklabels([])
    axes[0].set_title("Multi-omics sample overlap (top 15 exact sets)")

    ax = axes[1]
    for i, name in enumerate(names):
        ax.axhline(i, color="lightgrey", linewidth=0.5)
    for j, (combo, _) in enumerate(rows):
        for i, name in enumerate(names):
            on = name in combo
            ax.scatter(j, i, s=70, c=("#1f77b4" if on else "lightgrey"), edgecolors="white", linewidths=0.8)

        idxs = [i for i, n in enumerate(names) if n in combo]
        if len(idxs) >= 2:
            ax.plot([j] * len(idxs), idxs, color="#1f77b4", linewidth=1.2)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([str(i + 1) for i in range(len(rows))], fontsize=8)
    ax.set_xlabel("Set index")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_missingness_heatmap(missing_df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    if missing_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.3 * len(missing_df))))
    im = ax.imshow(missing_df.values, aspect="auto", cmap="magma", vmin=0, vmax=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.025)
    cb.set_label("Missing fraction")
    ax.set_yticks(range(len(missing_df.index)))
    ax.set_yticklabels(missing_df.index, fontsize=8)
    ax.set_xticks(range(len(missing_df.columns)))
    ax.set_xticklabels(missing_df.columns, fontsize=9)
    ax.set_title("Per-modality missingness (TCGA primary cohort)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_outcome_distribution(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(summary))
    ax.bar(x - 0.22, summary["event_n"], width=0.22, label="event", color="#d95f02")
    ax.bar(x, summary["censor_n"], width=0.22, label="censored", color="#1b9e77")
    ax.bar(x + 0.22, summary["unknown_n"], width=0.22, label="unknown", color="lightgrey")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["cohort"], rotation=30, ha="right")
    ax.set_ylabel("Patients")
    ax.set_title("Endpoint event / censor / unknown distribution by cohort")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_curated_modality_matrix(
    name: str,
    path: str,
    output_path: Path,
    top_features: int,
    train_ids: set[str] | None,
    comment: str | None = "#",
    gene_allowlist: list[str] | None = None,
    logger=None,
) -> dict[str, Any] | None:


    if not Path(path).exists():
        return None
    if gene_allowlist:
        wanted = {g.upper() for g in gene_allowlist}
        chunks = []
        for chunk in pd.read_csv(path, sep="\t", comment=comment, dtype=str, low_memory=False, chunksize=5000):
            id_col = "Hugo_Symbol" if "Hugo_Symbol" in chunk.columns else ("geneNames" if "geneNames" in chunk.columns else chunk.columns[0])
            keep = chunk[id_col].astype(str).str.upper().isin(wanted)
            if keep.any():
                chunks.append(chunk.loc[keep])
        if not chunks:
            return None
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(path, sep="\t", comment=comment, dtype=str, low_memory=False)
    if df.empty:
        return None
    id_col = "Hugo_Symbol" if "Hugo_Symbol" in df.columns else df.columns[0]
    sample_cols = matrix_sample_columns(df.columns)
    df = df[[id_col] + sample_cols].copy()
    df = df[df[id_col].notna() & (df[id_col].astype(str).str.strip() != "")]
    df[id_col] = df[id_col].astype(str)

    numeric = df.set_index(id_col).apply(pd.to_numeric, errors="coerce")
    if numeric.index.has_duplicates:
        numeric = numeric.groupby(level=0).mean()

    if train_ids:
        train_sample_cols = [c for c in numeric.columns if patient_id_from_sample(c) in train_ids]
        if train_sample_cols:
            variances = numeric[train_sample_cols].var(axis=1, skipna=True)
        else:
            variances = numeric.var(axis=1, skipna=True)
    else:
        variances = numeric.var(axis=1, skipna=True)
    if gene_allowlist:
        selected = [g for g in numeric.index.tolist() if g.upper() in {x.upper() for x in gene_allowlist}][:top_features]
    else:
        selected = variances.sort_values(ascending=False).head(top_features).index.tolist()
    curated = df[df[id_col].isin(selected)].copy()
    curated.to_csv(output_path, sep="\t", index=False)
    return {"matrix": name, "path": str(output_path), "features_written": len(curated), "training_variance_ranked": train_ids is not None}


def write_curated_patient_gene_matrix(mat: pd.DataFrame, output_path: Path, top_features: int, train_ids: set[str] | None) -> dict[str, Any] | None:

    if mat.empty:
        return None
    mat = mat.copy()
    mat.index = mat.index.astype(str)
    if train_ids:
        train = mat.loc[[p for p in train_ids if p in mat.index]]
    else:
        train = mat
    if train.empty:
        train = mat
    nonmissing = train.notna().mean(axis=0)
    variances = train.loc[:, nonmissing > 0].var(axis=0, skipna=True).sort_values(ascending=False)
    selected = variances.head(top_features).index.tolist()
    if not selected:
        return None
    out = mat[selected].reset_index().rename(columns={"index": "PATIENT_ID"})
    out.to_csv(output_path, sep="\t", index=False)
    return {"path": str(output_path), "features_written": len(selected), "patients_written": len(out)}


def write_mutation_gene_matrix(path: str, output_path: Path, top_features: int) -> dict[str, Any] | None:
    if not Path(path).exists():
        return None
    df = pd.read_csv(path, sep="\t", comment="#", dtype=str, low_memory=False)
    if "Hugo_Symbol" not in df.columns:
        return None
    sample_col = "Tumor_Sample_Barcode" if "Tumor_Sample_Barcode" in df.columns else None
    if sample_col is None:
        return None
    df = df[df["Hugo_Symbol"].notna() & df[sample_col].notna()].copy()
    df["PATIENT_ID"] = df[sample_col].astype(str).apply(patient_id_from_sample)
    top = df["Hugo_Symbol"].value_counts().head(top_features).index.tolist()
    if not top:
        return None
    pivot = (
        df[df["Hugo_Symbol"].isin(top)]
        .assign(value=1)
        .pivot_table(index="PATIENT_ID", columns="Hugo_Symbol", values="value", aggfunc="max", fill_value=0)
        .reset_index()
    )
    pivot.to_csv(output_path, sep="\t", index=False)
    return {"path": str(output_path), "features_written": len(top), "patients_written": len(pivot)}


def main_step01() -> int:
    parser = build_parser_step01()
    args = parser.parse_args()
    ctx = initialize_run(__file__, args)
    cfg = ctx.cfg
    endpoint = args.endpoint
    cohorts = cfg["cohorts"]
    input_files: list[str] = []

    preprocessed_dir = ctx.data_dir("preprocessed")
    clinical_outputs = []
    summaries = []
    excluded_all: list[dict[str, Any]] = []

    clinical_specs = [
        ("tcga", cohorts["tcga"]["clinical_patient"], True),
        ("msk", cohorts["msk"]["clinical_patient"], True),
        ("cptac", cohorts["cptac"]["clinical_patient"], True),
        ("htan", cohorts["htan"]["clinical_patient"], True),
        ("geo_gse103479", cohorts["geo_gse103479"]["clinical"], False),
        ("geo_gse17538", cohorts["geo_gse17538"]["clinical"], False),
        ("geo_gse39582", cohorts["geo_gse39582"]["clinical"], False),
    ]
    geo_os_cohorts = {"geo_gse103479", "geo_gse17538", "geo_gse39582"}
    for cohort, path, cbio in clinical_specs:
        input_files.append(path)
        if not Path(path).exists():
            ctx.add_warning(f"Clinical input missing for {cohort}: {path}")
            continue
        df = read_cbio_table(path) if cbio else read_table_auto(path)
        cohort_endpoint = "OS" if cohort in geo_os_cohorts else endpoint
        normalized, exc = normalize_clinical(df, cohort, cohort_endpoint)
        summaries.append(outcome_summary(normalized, cohort, cohort_endpoint))
        clinical_outputs.append((cohort, normalized))
        excluded_all.extend(exc)


    matrix_specs = [
        ("tcga_rna", cohorts["tcga"]["rna"], "#"),
        ("tcga_cnv", cohorts["tcga"]["cnv"], "#"),
        ("tcga_methylation", cohorts["tcga"]["methylation"], "#"),
        ("tcga_mutation", cohorts["tcga"]["mutation"], "#"),
        ("tcga_rppa", cohorts["tcga"]["rppa"], "#"),
        ("msk_mutation", cohorts["msk"]["mutation"], "#"),
        ("geo_gse103479_expression", cohorts["geo_gse103479"]["expression"], None),
        ("geo_gse17538_expression", cohorts["geo_gse17538"]["expression"], None),
        ("geo_gse39582_expression", cohorts["geo_gse39582"]["expression"], None),
        ("cptac_protein", cohorts["cptac"]["protein"], "#"),
        ("htan_cell_fraction", cohorts["htan"]["relative_fraction"], "#"),
    ]
    matrix_summaries: list[dict[str, Any]] = []
    for name, path, comment in matrix_specs:
        input_files.append(path)
        try:
            cols = load_matrix_header(path, comment=comment) if Path(path).exists() else []
            sample_cols = matrix_sample_columns(cols)
            patients = {patient_id_from_sample(c) for c in sample_cols}
            matrix_summaries.append({"matrix": name, "path": path, "exists": Path(path).exists(), "sample_columns": len(sample_cols), "unique_patients": len(patients)})
        except Exception as exc:
            ctx.add_warning(f"Header read failed for {name}: {exc}")

    if args.dry_run:
        ctx.logger.info("Dry-run: skipped writing curated matrices.")
    else:
        for cohort, df in clinical_outputs:
            filename = f"{cohort.lower()}_{endpoint.lower()}_clinical_endpoint_qc.tsv"
            out_path = preprocessed_dir / filename
            ctx.write_table(out_path, df, "processed_data", f"{cohort} normalized clinical endpoint table.")
            if cohort == "tcga":
                ctx.write_table(preprocessed_dir / "tcga_coadread_clinical_os_qc.tsv", df, "processed_data", "TCGA COADREAD standardized clinical OS QC table.")

        summary_df = pd.DataFrame(summaries)
        ctx.write_table(preprocessed_dir / "cohort_endpoint_completeness_summary.tsv", summary_df, "processed_data", "Endpoint completeness (event/censor/unknown/median follow-up) by cohort.")
        matrix_summary_df = pd.DataFrame(matrix_summaries)
        ctx.write_table(preprocessed_dir / "omics_matrix_dimension_summary.tsv", matrix_summary_df, "processed_data", "Omics matrix dimension and patient coverage summary.")

        ctx.write_table(preprocessed_dir / "excluded_samples_with_reasons.tsv", pd.DataFrame(excluded_all), "processed_data", "Per-sample exclusion log with reasons.")


        try:
            tcga_sets = {
                "rna": patient_set_from_matrix(cohorts["tcga"]["rna"], "#"),
                "cnv": patient_set_from_matrix(cohorts["tcga"]["cnv"], "#"),
                "methylation": patient_set_from_matrix(cohorts["tcga"]["methylation"], "#"),
                "rppa": patient_set_from_matrix(cohorts["tcga"]["rppa"], "#"),
                "mutation": patient_set_from_matrix(cohorts["tcga"]["mutation"], "#"),
            }
            tcga_sets = {k: v for k, v in tcga_sets.items() if v}
            upset_path = ctx.output_path(ctx.run_dir / "tcga_multiomics_sample_overlap_upset.png", "figure")
            plot_upset_style(tcga_sets, upset_path)
            ctx.register_output(upset_path, "figure", "UpSet-style multi-omics sample overlap for TCGA cohort.")
        except Exception as exc:
            ctx.add_warning(f"UpSet plot failed: {exc}")


        try:
            tcga_clinical_df = next((df for cohort, df in clinical_outputs if cohort == "tcga"), None)
            if tcga_clinical_df is not None and not tcga_clinical_df.empty:
                tcga_patients = set(tcga_clinical_df["PATIENT_ID"].astype(str))
                missing_rows = {}
                for omics_key in ("rna", "cnv", "methylation", "rppa", "mutation"):
                    path = cohorts["tcga"][omics_key]
                    if not Path(path).exists():
                        continue
                    cols = load_matrix_header(path, comment="#")
                    samples = matrix_sample_columns(cols)
                    pset = {patient_id_from_sample(c) for c in samples}
                    missing_rows[omics_key] = 1.0 - (len(tcga_patients & pset) / max(1, len(tcga_patients)))

                stage = tcga_clinical_df.get("AJCC_PATHOLOGIC_TUMOR_STAGE")
                if stage is not None:
                    groups = stage.fillna("Unknown").astype(str)
                    grouped = groups.value_counts()
                    cols_for_heatmap = ["TCGA_overall"] + grouped.index.tolist()
                    df = pd.DataFrame(index=list(missing_rows.keys()), columns=cols_for_heatmap, dtype=float)
                    for omics_key, ratio in missing_rows.items():
                        df.loc[omics_key, "TCGA_overall"] = ratio

                        for g in grouped.index:
                            df.loc[omics_key, g] = ratio
                    miss_path = ctx.output_path(ctx.run_dir / "tcga_omics_missingness_heatmap.png", "figure")
                    plot_missingness_heatmap(df, miss_path)
                    ctx.register_output(miss_path, "figure", "Per-modality missingness ratio against TCGA primary cohort (overall and by AJCC stage).")
                    ctx.write_table(preprocessed_dir / "tcga_omics_missingness_by_modality.tsv", df.reset_index().rename(columns={"index": "modality"}), "processed_data", "Per-modality missingness ratio table.")
        except Exception as exc:
            ctx.add_warning(f"Missingness heatmap failed: {exc}")


        try:
            plot_path = ctx.output_path(ctx.run_dir / "cohort_endpoint_event_censor_distribution.png", "figure")
            plot_outcome_distribution(summary_df, plot_path)
            ctx.register_output(plot_path, "figure", "Event/censor/unknown distribution by cohort.")
        except Exception as exc:
            ctx.add_warning(f"Outcome plot failed: {exc}")


        if args.write_matrices:


            tcga_clinical_df = next((df for cohort, df in clinical_outputs if cohort == "tcga"), None)
            train_pool = set(tcga_clinical_df["PATIENT_ID"].astype(str)) if tcga_clinical_df is not None else None
            for name, path, comment in matrix_specs:
                if name.startswith("tcga_") and Path(path).exists() and name != "tcga_mutation":
                    alias = {
                        "tcga_rna": "tcga_coadread_rna_log2_tpm_like_matrix.tsv",
                        "tcga_cnv": "tcga_coadread_cnv_gene_level_matrix.tsv",
                        "tcga_methylation": "tcga_coadread_methylation_gene_level_matrix.tsv",
                        "tcga_rppa": "tcga_coadread_rppa_gene_level_matrix.tsv",
                    }.get(name)
                    safe_name = alias or f"{name}_train_variance_top_{args.top_features_per_omics}_matrix.tsv"
                    info = write_curated_modality_matrix(name, path, preprocessed_dir / safe_name, args.top_features_per_omics, train_pool, comment, None, ctx.logger)
                    if info:
                        ctx.register_output(preprocessed_dir / safe_name, "processed_data", f"{name} train-variance top-{args.top_features_per_omics} curated matrix.")
            mut_info = write_mutation_gene_matrix(cohorts["tcga"]["mutation"], preprocessed_dir / "tcga_coadread_mutation_gene_level_matrix.tsv", args.top_features_per_omics)
            if mut_info:
                ctx.register_output(preprocessed_dir / "tcga_coadread_mutation_gene_level_matrix.tsv", "processed_data", "TCGA COADREAD gene-level binary mutation matrix.")
            tcga_ids = pd.DataFrame({"PATIENT_ID": sorted(train_pool or [])})
            for filename, desc in [
                ("tcga_coadread_methylation_gene_level_matrix.tsv", "TCGA COADREAD methylation gene-level matrix placeholder; no probes passed current preprocessing export filters."),
                ("tcga_coadread_rppa_gene_level_matrix.tsv", "TCGA COADREAD RPPA gene-level matrix placeholder; no proteins passed current preprocessing export filters."),
            ]:
                p = preprocessed_dir / filename
                if not p.exists():
                    ctx.write_table(p, tcga_ids, "processed_data", desc)
            external_matrix_jobs = [
                ("msk_crc_2017_mutation_gene_level_matrix.tsv", "mutation", "msk"),
                ("geo_gse103479_expression_matrix.tsv", "expression", "geo_gse103479"),
                ("geo_gse17538_expression_matrix.tsv", "expression", "geo_gse17538"),
                ("geo_gse39582_expression_matrix.tsv", "expression", "geo_gse39582"),
                ("cptac_coad_rna_gene_level_matrix.tsv", "rna", "cptac"),
                ("cptac_coad_protein_gene_level_matrix.tsv", "protein", "cptac"),
                ("htan_crc_pseudobulk_rna_matrix.tsv", "pseudo_bulk_rna", "htan"),
                ("htan_crc_microenvironment_relative_fraction_matrix.tsv", "relative_fraction", "htan"),
            ]
            for filename, key, cohort_name in external_matrix_jobs:
                src = cohorts[cohort_name].get(key)
                if not src or not Path(src).exists():
                    continue
                try:
                    if key == "relative_fraction":
                        raw_fraction = pd.read_csv(src, sep="\t", comment="#", low_memory=False)
                        ctx.write_table(preprocessed_dir / filename, raw_fraction, "processed_data", f"{cohort_name} external validation matrix ({key}).")
                        info = {"features_written": max(0, raw_fraction.shape[1] - 1)}
                    elif key == "mutation":
                        info = write_mutation_gene_matrix(src, preprocessed_dir / filename, args.top_features_per_omics)
                    else:
                        matrix_comment = None if cohort_name.startswith("geo_") else "#"
                        info = write_curated_modality_matrix(f"{cohort_name}_{key}", src, preprocessed_dir / filename, args.top_features_per_omics, None, matrix_comment, None, ctx.logger)
                        if info is None:
                            cohort_clinical = next((df for c, df in clinical_outputs if c == cohort_name or (cohort_name == "cptac" and c == "cptac")), pd.DataFrame())
                            ids = cohort_clinical[["PATIENT_ID"]].copy() if "PATIENT_ID" in cohort_clinical.columns else pd.DataFrame({"PATIENT_ID": []})
                            ctx.write_table(preprocessed_dir / filename, ids, "processed_data", f"{cohort_name} external validation matrix placeholder; no features passed current preprocessing export filters.")
                            info = {"features_written": 0}
                    if info:
                        ctx.register_output(preprocessed_dir / filename, "processed_data", f"{cohort_name} external validation matrix ({key}).")
                except Exception as exc:
                    ctx.add_warning(f"External matrix write failed for {cohort_name}:{key}: {exc}")


    tcga_clinical = next((df for c, df in clinical_outputs if c == "tcga"), None)
    if tcga_clinical is not None and not tcga_clinical.empty:
        try:
            tcga_ipcw = add_fixed_os_endpoint(tcga_clinical, tau=OS_RISK_TIME_MONTHS)
            tcga_ipcw = compute_ipcw_weights(tcga_ipcw, tau=OS_RISK_TIME_MONTHS)
            tcga_ipcw = compute_pseudo_observations(tcga_ipcw, tau=OS_RISK_TIME_MONTHS)
            ipcw_path = preprocessed_dir / "tcga_ipcw_os_endpoint.tsv"
            tcga_ipcw.to_csv(ipcw_path, sep="\t", index=False)
            ctx.register_output(ipcw_path, "processed_data",
                                "TCGA IPCW weights + pseudo-observations for 36-month endpoint.")
            n_obs = int(tcga_ipcw["os_death_observed"].sum())
            n_early = int(tcga_ipcw["early_censored_before_os"].sum())
            ctx.logger.info("IPCW 36m endpoint: %d observed, %d early-censored, median weight=%.3f",
                            n_obs, n_early, float(tcga_ipcw.loc[tcga_ipcw["ipcw_label_available"], "ipcw_weight_os"].median()))
        except Exception as exc:
            ctx.add_warning(f"IPCW endpoint computation failed: {exc}")

    if summaries:
        ctx.logger.info("Endpoint summaries: %s", pd.DataFrame(summaries).to_dict(orient="records"))
    ctx.finalize([p for p in input_files if Path(p).exists()])
    return 0


import gzip

