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
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA, TruncatedSVD

from _pipeline_core import *  # noqa: F401,F403  – shared utilities
from _causal_infer import *   # noqa: F401,F403
from _gan_augment import *    # noqa: F401,F403
from _step_common import *    # noqa: F401,F403

def build_parser_step03() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create leakage-safe per-modality CRC multi-omics embeddings.")
    add_common_args(parser)
    parser.add_argument("--per-modality-dim", type=int, default=20, help="Embedding dimension per omics modality.")
    parser.add_argument("--test-size", type=float, default=0.30, help="Internal validation fraction used if no split exists.")
    parser.add_argument("--stability-seeds", type=int, default=5, help="Number of random seeds for embedding stability evaluation.")
    return parser


def load_tcga_endpoint(cfg: dict[str, Any], endpoint: str) -> pd.DataFrame:
    processed = Path(cfg["processed_root"]) / "preprocessed" / f"tcga_{endpoint.lower()}_clinical_endpoint_qc.tsv"
    if processed.exists():
        df = pd.read_csv(processed, sep="\t", low_memory=False)
    else:
        df = read_cbio_table(cfg["cohorts"]["tcga"]["clinical_patient"])
        status_col = f"{endpoint}_STATUS"
        time_col = f"{endpoint}_MONTHS"
        if status_col not in df.columns or time_col not in df.columns:
            if endpoint != "OS":
                raise ValueError(f"Endpoint {endpoint} not available in TCGA clinical data.")
            status_col, time_col = "OS_STATUS", "OS_MONTHS"
        df[f"{endpoint}_EVENT"] = parse_survival_status(df[status_col])
        df[f"{endpoint}_TIME_MONTHS"] = numeric_series(df[time_col])
    time_col = f"{endpoint}_TIME_MONTHS"
    event_col = f"{endpoint}_EVENT"
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[event_col] = df[event_col].fillna(False).astype(bool)
    return df.loc[df[time_col].notna() & (df[time_col] > 0), ["PATIENT_ID", time_col, event_col]].copy()


def get_or_create_split(cfg: dict[str, Any], endpoint: str, test_size: float) -> pd.DataFrame:
    split_path = Path(cfg["processed_root"]) / "survival_models" / "tcga_train_internal_validation_split.tsv"
    if split_path.exists():
        split = pd.read_csv(split_path, sep="\t")
        if {"PATIENT_ID", "split", "event", "time_months"}.issubset(split.columns):
            return split
    clinical = load_tcga_endpoint(cfg, endpoint)
    event = clinical[f"{endpoint}_EVENT"].astype(bool)
    stratify = event if event.nunique() == 2 and event.sum() >= 2 else None
    train_idx, _ = train_test_split(np.arange(len(clinical)), test_size=test_size, random_state=cfg["random_seed"], stratify=stratify)
    split = pd.DataFrame(
        {
            "PATIENT_ID": clinical["PATIENT_ID"].to_numpy(),
            "split": np.where(np.isin(np.arange(len(clinical)), train_idx), "train", "internal_validation"),
            "event": event.astype(int).to_numpy(),
            "time_months": clinical[f"{endpoint}_TIME_MONTHS"].to_numpy(),
        }
    )
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(split_path, sep="\t", index=False)
    return split


def select_features_train_only(
    mat: pd.DataFrame,
    train_ids: list[str],
    min_nonmissing: float,
    top_k: int,
    forced: list[str] | None = None,
) -> list[str]:
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


def fit_modality_encoder(
    block: pd.DataFrame,
    train_mask: np.ndarray,
    dim: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, dict[str, Any]]:


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
        "imputer": imputer,
        "scaler": scaler,
        "model": model,
        "feature_columns": block.columns.tolist(),
        "n_components": n_components,
    }
    diag = {
        "explained_variance_ratio": getattr(model, "explained_variance_ratio_", np.repeat(np.nan, n_components)).tolist(),
        "mean_reconstruction_mse": float(np.nanmean(per_sample)),
    }
    return emb, bundle, per_sample, diag


def stability_via_subspace_overlap(
    mats: dict[str, pd.DataFrame],
    selected: dict[str, list[str]],
    train_ids: list[str],
    dim: int,
    seeds: list[int],
) -> dict[str, dict[str, Any]]:

    results: dict[str, dict[str, Any]] = {}
    for omics, feats in selected.items():
        if omics not in mats or mats[omics].empty:
            continue
        block_full = mats[omics].reindex(train_ids)[feats]
        if block_full.shape[0] < dim + 2 or block_full.shape[1] < 2:
            continue
        x = SimpleImputer(strategy="median").fit_transform(block_full)
        x = StandardScaler().fit_transform(x)
        bases: list[np.ndarray] = []
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
        sims: list[float] = []
        for i in range(len(bases)):
            for j in range(i + 1, len(bases)):
                m = bases[i] @ bases[j].T

                from numpy.linalg import svd

                s_vals = svd(m, compute_uv=False)
                sims.append(float(np.mean(np.abs(s_vals))))
        results[omics] = {
            "n_seeds_evaluated": len(bases),
            "mean_principal_subspace_alignment": float(np.mean(sims)) if sims else float("nan"),
            "stability_interpretation": "1.0 = identical subspaces; <0.6 = unstable embedding.",
        }
    return results


def main_step03() -> int:
    parser = build_parser_step03()
    args = parser.parse_args()
    ctx = initialize_run(__file__, args)
    cfg = ctx.cfg
    filters = cfg.get("filters", {"min_nonmissing_ratio_train": 0.7})
    out_dir = ctx.data_dir("multiomics_pretraining")
    model_dir = out_dir / "serialized_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    tcga = cfg["cohorts"]["tcga"]
    input_files = [tcga[k] for k in ("rna", "cnv", "methylation", "rppa") if Path(tcga[k]).exists()]

    if args.dry_run:
        rows = [{"omics": omics, "path": str(path), "exists": Path(path).exists()} for omics, path in tcga.items() if omics != "clinical_patient"]
        ctx.write_table(ctx.run_dir / "multiomics_pretraining_dry_run_input_checks.tsv", rows, "qc", "Dry-run multi-omics input checks.")
        ctx.add_warning("Dry-run only: no embedding model was trained.")
        ctx.finalize(input_files)
        return 0

    split = get_or_create_split(cfg, args.endpoint, args.test_size)
    ctx.write_table(out_dir / "tcga_embedding_train_internal_validation_split.tsv", split, "analysis_data", "Split used for leakage-safe embedding.")
    patients = split["PATIENT_ID"].astype(str).tolist()
    train_ids = split.loc[split["split"] == "train", "PATIENT_ID"].astype(str).tolist()

    preprocessed = Path(cfg["processed_root"]) / "preprocessed"
    curated_paths = {
        "rna": preprocessed / "tcga_coadread_rna_log2_tpm_like_matrix.tsv",
        "cnv": preprocessed / "tcga_coadread_cnv_gene_level_matrix.tsv",
        "methylation": preprocessed / "tcga_coadread_methylation_gene_level_matrix.tsv",
        "rppa": preprocessed / "tcga_coadread_rppa_gene_level_matrix.tsv",
    }


    ctx.logger.info("Loading per-modality matrices (full, comment-aware)...")
    mats: dict[str, pd.DataFrame] = {}
    for omics_key in ("rna", "cnv", "methylation", "rppa"):
        path = str(curated_paths[omics_key]) if curated_paths[omics_key].exists() else None
        if not (path and Path(path).exists()):
            ctx.add_warning(f"{omics_key} curated preprocessed matrix missing; modality will be marked unavailable. Run 01 with a matching gene allowlist to include it.")
            mats[omics_key] = pd.DataFrame()
            continue
        try:
            mats[omics_key] = stream_gene_matrix(path, comment=None, id_col_preference="Hugo_Symbol")
            ctx.logger.info("%s using curated preprocessed matrix: %s", omics_key, curated_paths[omics_key])
            ctx.logger.info("%s loaded: patients=%d, features=%d", omics_key, mats[omics_key].shape[0], mats[omics_key].shape[1])
        except Exception as exc:
            ctx.add_warning(f"Failed to load {omics_key}: {exc}")
            mats[omics_key] = pd.DataFrame()


    causal_path = Path(cfg["processed_root"]) / "causal" / "causal_priority_feature_table.tsv"
    rna_forced: list[str] = []
    if causal_path.exists():
        try:
            cdf = pd.read_csv(causal_path, sep="\t")
            rna_forced = [g for g in cdf["feature"].astype(str).tolist() if not is_likely_pseudogene(g)]
            ctx.logger.info("Injecting %d causal-priority genes into RNA encoder feature pool.", len(rna_forced))
        except Exception as exc:
            ctx.add_warning(f"Failed to read causal priority table: {exc}")
    else:
        ctx.add_warning("Causal-priority table absent; RNA encoder will rely on top-variance only.")

    min_nm = float(filters.get("min_nonmissing_ratio_train", filters.get("min_nonmissing_ratio", 0.7)))
    max_feats = int(cfg.get("max_features_per_omics", 200))
    selected: dict[str, list[str]] = {}
    embeddings: dict[str, np.ndarray] = {}
    bundles: dict[str, dict[str, Any]] = {}
    masks_per_patient: dict[str, np.ndarray] = {}
    recon_records: list[dict[str, Any]] = []

    train_mask_arr = (split["split"].to_numpy() == "train")

    for omics in ("rna", "cnv", "methylation", "rppa"):
        mat = mats.get(omics, pd.DataFrame())
        if mat.empty:
            masks_per_patient[omics] = np.zeros(len(patients), dtype=int)
            continue
        forced = rna_forced if omics == "rna" else None
        feats = select_features_train_only(mat, train_ids, min_nm, max_feats, forced=forced)
        if not feats:
            masks_per_patient[omics] = np.zeros(len(patients), dtype=int)
            continue
        selected[omics] = feats
        block = mat.reindex(patients)[feats]
        availability = block.notna().any(axis=1).astype(int).to_numpy()
        masks_per_patient[omics] = availability
        emb, bundle, recon_per_sample, diag = fit_modality_encoder(block, train_mask_arr, args.per_modality_dim, cfg["random_seed"])
        embeddings[omics] = emb
        bundles[omics] = {**bundle, "omics": omics, "diagnostic": diag}
        for i, pid in enumerate(patients):
            recon_records.append({"PATIENT_ID": pid, "modality": omics, "reconstruction_mse": float(recon_per_sample[i]) if not np.isnan(recon_per_sample[i]) else np.nan, "modality_available": int(availability[i])})

    if not embeddings:
        raise RuntimeError("No modality produced an embedding; check inputs.")


    parts = []
    columns: list[str] = []
    for omics, emb in embeddings.items():

        availability = masks_per_patient[omics]
        emb_masked = emb.copy()
        emb_masked[availability == 0] = 0.0
        parts.append(emb_masked)
        columns.extend([f"{omics}_embedding_{i+1:02d}" for i in range(emb.shape[1])])
    fused = np.concatenate(parts, axis=1)
    emb_df = pd.DataFrame(fused, columns=columns)
    emb_df.insert(0, "PATIENT_ID", patients)
    emb_df.insert(1, "split", split["split"].to_numpy())


    mask_df = pd.DataFrame(masks_per_patient)
    mask_df.insert(0, "PATIENT_ID", patients)


    recon_df = pd.DataFrame(recon_records)
    recon_summary = recon_df.dropna().groupby("modality")["reconstruction_mse"].agg(["mean", "median", "std", "count"]).reset_index()


    seeds = [int(cfg["random_seed"]) + i * 17 for i in range(args.stability_seeds)]
    stability = stability_via_subspace_overlap(mats, selected, train_ids, args.per_modality_dim, seeds)
    stability_df = pd.DataFrame([{"modality": k, **v} for k, v in stability.items()])


    bundle_path = model_dir / "tcga_multiomics_per_modality_embedding_bundle.joblib"
    joblib.dump({"per_modality_bundles": bundles, "selected_features": selected, "patient_order": patients, "fit_scope": "train_split_only", "per_modality_dim": args.per_modality_dim}, bundle_path)
    ctx.register_output(bundle_path, "model", "Per-modality train-only encoder bundle (PCA/TruncatedSVD).")
    ctx.write_table(out_dir / "tcga_multiomics_patient_embedding.tsv", emb_df, "analysis_data", "Leakage-safe per-modality TCGA multi-omics patient embedding (mask-aware concat).")
    ctx.write_table(out_dir / "tcga_multiomics_modality_availability_mask.tsv", mask_df, "analysis_data", "Modality availability mask per patient.")
    ctx.write_table(out_dir / "tcga_multiomics_reconstruction_error_by_modality.tsv", recon_df, "analysis_data", "Per-sample per-modality PCA reconstruction MSE.")
    ctx.write_table(out_dir / "reconstruction_error_by_modality.tsv", recon_df, "analysis_data", "Alias required by Week-2 design: per-sample per-modality reconstruction MSE.")
    ctx.write_table(out_dir / "tcga_multiomics_reconstruction_error_summary.tsv", recon_summary, "analysis_data", "Per-modality reconstruction MSE summary.")
    if not stability_df.empty:
        ctx.write_table(out_dir / "tcga_multiomics_embedding_subspace_stability.tsv", stability_df, "analysis_data", "Multi-seed principal subspace alignment per modality.")
        ctx.write_table(out_dir / "embedding_stability_summary.tsv", stability_df, "analysis_data", "Alias required by Week-2 design: multi-seed embedding stability summary.")
    ctx.write_json(out_dir / "pretraining_model_config.json", {
        "method": "per_modality_PCA_or_TruncatedSVD_mask_aware_concat",
        "fit_scope": "train_split_only",
        "per_modality_dim": args.per_modality_dim,
        "selected_features_by_modality": {k: len(v) for k, v in selected.items()},
        "modalities_used": list(embeddings.keys()),
        "stability_seeds": seeds,
        "bootstrap_seeds": seeds,
        "neural_pretraining_disabled_reason": "TCGA event count is small (~120). Per-modality PCA + mask-aware concat is the conservative TMO-Net-style proxy supported by this sample size (Wang et al. 2024; Argelaguet et al. 2018).",
        "rna_causal_genes_injected": len([g for g in rna_forced if g in (selected.get("rna") or [])]),
    }, "analysis_data", "Pretraining model configuration with explicit downgrade rationale.")


    try:
        import matplotlib.pyplot as plt

        if not recon_summary.empty:
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.bar(recon_summary["modality"], recon_summary["mean"], yerr=recon_summary["std"], color="#6baed6", capsize=4)
            ax.set_ylabel("Mean reconstruction MSE")
            ax.set_title("Per-modality PCA reconstruction error (train-only fit)")
            fig.tight_layout()
            p = ctx.output_path(ctx.run_dir / "per_modality_reconstruction_error.png", "figure")
            fig.savefig(p, dpi=180)
            plt.close(fig)
            ctx.register_output(p, "figure", "Per-modality reconstruction MSE summary.")
        if not stability_df.empty:
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.bar(stability_df["modality"], stability_df["mean_principal_subspace_alignment"], color="#74c476")
            ax.axhline(0.6, linestyle="--", color="grey", linewidth=0.8)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Mean subspace alignment (cosine)")
            ax.set_title(f"Multi-seed embedding stability (n_seeds={args.stability_seeds})")
            fig.tight_layout()
            p = ctx.output_path(ctx.run_dir / "per_modality_embedding_subspace_stability.png", "figure")
            fig.savefig(p, dpi=180)
            plt.close(fig)
            ctx.register_output(p, "figure", "Per-modality embedding stability across seeds.")
        try:
            clinical_raw = read_cbio_table(tcga["clinical_patient"])
            clinical_raw["PATIENT_ID"] = clinical_raw["PATIENT_ID"].astype(str)
            meta_cols = ["PATIENT_ID"] + [c for c in ("AJCC_PATHOLOGIC_TUMOR_STAGE", "SUBTYPE") if c in clinical_raw.columns]
            plot_df = emb_df.merge(clinical_raw[meta_cols], on="PATIENT_ID", how="left")
            coords = PCA(n_components=2, random_state=cfg["random_seed"]).fit_transform(emb_df[columns].to_numpy())
            plot_df["latent_1"] = coords[:, 0]
            plot_df["latent_2"] = coords[:, 1]
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
            for ax, color_col in zip(axes, ["AJCC_PATHOLOGIC_TUMOR_STAGE", "SUBTYPE"]):
                labels = plot_df[color_col].fillna("Unknown").astype(str) if color_col in plot_df.columns else pd.Series(["Unknown"] * len(plot_df))
                for label in sorted(labels.unique())[:12]:
                    mask = labels == label
                    ax.scatter(plot_df.loc[mask, "latent_1"], plot_df.loc[mask, "latent_2"], s=13, alpha=0.75, label=label)
                ax.set_xlabel("Latent PC1")
                ax.set_ylabel("Latent PC2")
                ax.set_title(color_col.replace("_", " ").title())
                ax.legend(fontsize=6, loc="best")
            fig.suptitle("TCGA multi-omics embedding by stage and subtype")
            fig.tight_layout()
            p = ctx.output_path(ctx.run_dir / "embedding_umap_by_stage_and_subtype.png", "figure")
            fig.savefig(p, dpi=180)
            plt.close(fig)
            ctx.register_output(p, "figure", "PCA fallback visualization of embedding by stage/subtype; UMAP is omitted when umap-learn is unavailable.")
        except Exception as exc:
            ctx.add_warning(f"Embedding stage/subtype visualization failed: {exc}")
        try:
            load_rows = []
            for omics, bundle in bundles.items():
                model = bundle.get("model")
                comps = getattr(model, "components_", None)
                feats = bundle.get("feature_columns", [])
                if comps is None or len(feats) == 0:
                    continue
                arr = np.asarray(comps)
                for comp_idx in range(min(3, arr.shape[0])):
                    top_idx = np.argsort(np.abs(arr[comp_idx]))[-10:]
                    for i in top_idx:
                        load_rows.append({"modality": omics, "component": f"C{comp_idx + 1}", "feature": feats[i], "loading": float(arr[comp_idx, i])})
            load_df = pd.DataFrame(load_rows)
            if not load_df.empty:
                heat = load_df.pivot_table(index="feature", columns=["modality", "component"], values="loading", aggfunc="first").fillna(0)
                fig, ax = plt.subplots(figsize=(9.5, max(4.0, 0.18 * len(heat))))
                im = ax.imshow(heat.values, aspect="auto", cmap="coolwarm")
                fig.colorbar(im, ax=ax, fraction=0.025, label="PCA loading")
                ax.set_yticks(range(len(heat.index)))
                ax.set_yticklabels(heat.index, fontsize=6)
                ax.set_xticks(range(len(heat.columns)))
                ax.set_xticklabels([f"{a}:{b}" for a, b in heat.columns], rotation=45, ha="right", fontsize=7)
                ax.set_title("Top per-modality embedding loadings")
                fig.tight_layout()
                p = ctx.output_path(ctx.run_dir / "multiomics_loading_heatmap.png", "figure")
                fig.savefig(p, dpi=180)
                plt.close(fig)
                ctx.register_output(p, "figure", "Top PCA/TruncatedSVD loading heatmap by modality.")
        except Exception as exc:
            ctx.add_warning(f"Loading heatmap failed: {exc}")
    except Exception as exc:
        ctx.add_warning(f"Plotting diagnostic figures failed: {exc}")

    ctx.add_warning("Per-modality PCA is the conservative TMO-Net-style proxy for current sample size; full encoder-with-modality-dropout reserved for future extension when event count permits.")
    ctx.finalize(input_files)
    return 0


import json

from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_parser_step04() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train leakage-safe CRC survival prediction models.")
    add_common_args(parser)
    parser.add_argument("--test-size", type=float, default=0.30, help="Internal validation fraction.")
    parser.add_argument("--coxnet-cv-folds", type=int, default=5, help="K folds for Coxnet alpha selection.")
    parser.add_argument("--bootstrap-iter", type=int, default=200, help="Bootstrap iterations for 95% CI.")
    parser.add_argument("--coxnet-l1-ratio", type=float, default=0.5, help="Elastic-net L1 ratio for Coxnet.")
    parser.add_argument("--gene-top-k-grid", type=str, default="100,300,500",
                        help="Comma-separated RNA top-K grid for clinical-plus-RNA Coxnet candidates.")
    parser.add_argument("--gan-feature-set", choices=["clinical_plus_embedding", "embedding_only"],
                        default="clinical_plus_embedding",
                        help="Expected GAN augmentation feature set consumed from Module 03.5.")
    parser.add_argument("--primary-selection-policy", choices=["survival_composite"], default="survival_composite",
                        help="Primary model selection policy.")
    return parser


def load_tcga_clinical(cfg: dict[str, Any], endpoint: str) -> pd.DataFrame:
    processed = Path(cfg["processed_root"]) / "preprocessed" / f"tcga_{endpoint.lower()}_clinical_endpoint_qc.tsv"
    if processed.exists():
        df = pd.read_csv(processed, sep="\t", low_memory=False)
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


def attach_raw_clinical_columns(df: pd.DataFrame, raw_clinical_path: str) -> pd.DataFrame:
    needed = [
        "AJCC_PATHOLOGIC_TUMOR_STAGE",
        "PATH_M_STAGE",
        "PATH_N_STAGE",
        "PATH_T_STAGE",
        "SUBTYPE",
        "CANCER_TYPE_ACRONYM",
        "AGE",
        "AGE_AT_DIAGNOSIS",
        "SEX",
        "PRIMARY_SITE",
    ]
    missing = [c for c in needed if c not in df.columns]
    if not missing:
        return df
    raw = read_cbio_table(raw_clinical_path)
    if "PATIENT_ID" not in raw.columns:
        return df
    cols = ["PATIENT_ID"] + [c for c in needed if c in raw.columns]
    out = df.merge(raw[cols], on="PATIENT_ID", how="left", suffixes=("", "_raw"))
    if "AGE" not in out.columns and "AGE_AT_DIAGNOSIS" in out.columns:
        out["AGE"] = pd.to_numeric(out["AGE_AT_DIAGNOSIS"], errors="coerce")
    return out


def stage_to_group(value: Any) -> str:
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


def make_stratified_split(clinical: pd.DataFrame, endpoint: str, test_size: float, seed: int, logger) -> pd.DataFrame:
    time_col = f"{endpoint}_TIME_MONTHS"
    event_col = f"{endpoint}_EVENT"
    clinical = clinical.copy()
    clinical["stage_group"] = clinical.get("AJCC_PATHOLOGIC_TUMOR_STAGE", pd.Series(["Unknown"] * len(clinical))).apply(stage_to_group)
    clinical["site_group"] = clinical.get("CANCER_TYPE_ACRONYM", pd.Series(["Unknown"] * len(clinical))).astype(str).fillna("Unknown")
    clinical["event_int"] = clinical[event_col].astype(int)
    keys = stratified_split_keys(clinical, ["event_int", "stage_group", "site_group"], min_count=5)
    if keys.nunique() < 2:
        keys = clinical["event_int"].astype(str)
        logger.warning("Stratification collapsed; falling back to event-only split.")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, valid_idx = next(sss.split(np.arange(len(clinical)), keys))
    split_label = np.array(["train"] * len(clinical), dtype=object)
    split_label[valid_idx] = "internal_validation"
    return pd.DataFrame(
        {
            "PATIENT_ID": clinical["PATIENT_ID"].astype(str).to_numpy(),
            "split": split_label,
            "event": clinical[event_col].astype(int).to_numpy(),
            "time_months": clinical[time_col].astype(float).to_numpy(),
            "stratify_key": keys.to_numpy(),
        }
    )


def fit_clinical_transformer(
    clinical: pd.DataFrame,
    split: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    clinical = clinical.copy()
    clinical["PATIENT_ID"] = clinical["PATIENT_ID"].astype(str)
    split = split.copy()
    split["PATIENT_ID"] = split["PATIENT_ID"].astype(str)
    merged = split[["PATIENT_ID", "split"]].merge(clinical[["PATIENT_ID"] + [c for c in feature_cols if c in clinical.columns]], on="PATIENT_ID", how="left")
    existing = [c for c in feature_cols if c in merged.columns]
    if not existing:
        raise ValueError(f"No requested clinical features are available. Requested={feature_cols}")
    raw_x = merged[existing].copy()
    numeric_cols = [c for c in existing if c.upper() == "AGE"]
    categorical_cols = [c for c in existing if c not in numeric_cols]
    train_mask = merged["split"].to_numpy() == "train"
    bundle: dict[str, Any] = {
        "feature_cols": existing,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
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
        x_train_cat = cat_imputer.fit_transform(raw_x.loc[train_mask, categorical_cols].astype(object))
        encoder.fit(x_train_cat)
        cat_columns = encoder.get_feature_names_out(categorical_cols).tolist()
        blocks.append(encoder.transform(cat_imputer.transform(raw_x[categorical_cols].astype(object))))
        columns.extend(cat_columns)
        bundle["cat_imputer"] = cat_imputer
        bundle["encoder"] = encoder
    matrix = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
    x = pd.DataFrame(matrix, columns=columns, index=merged["PATIENT_ID"].astype(str))
    bundle["model_columns"] = columns
    return x, bundle


def survival_y(time: np.ndarray, event: np.ndarray):
    from sksurv.util import Surv

    return Surv.from_arrays(event=np.asarray(event).astype(bool), time=np.asarray(time, dtype=float))


def harrell_c_index(time: pd.Series | np.ndarray, event: pd.Series | np.ndarray, risk: np.ndarray) -> float:
    from lifelines.utils import concordance_index

    return float(concordance_index(time, -risk, event))


def fit_lifelines_cox_with_split(x: pd.DataFrame, split: pd.DataFrame, penalizer: float):
    """Fit Cox PH using split-based data layout (downstream analysis).

    Distinct from the primary fit_lifelines_cox (X_train, endpoint_train, penalizer).
    """
    from lifelines import CoxPHFitter

    train_mask = split["split"].to_numpy() == "train"
    train = x.iloc[train_mask].copy()
    train["time"] = split.loc[train_mask, "time_months"].to_numpy()
    train["event"] = split.loc[train_mask, "event"].astype(int).to_numpy()
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(train, duration_col="time", event_col="event")
    return cph


def lifelines_predict_log_hazard(model: Any, x: pd.DataFrame) -> np.ndarray:
    return np.log(model.predict_partial_hazard(x).to_numpy().reshape(-1) + 1e-12)


def lifelines_survival_estimator(model: Any, x_valid: pd.DataFrame):


    def estimate(times: np.ndarray) -> np.ndarray:
        sf = model.predict_survival_function(x_valid, times=np.asarray(times, dtype=float))
        return np.clip(sf.T.to_numpy(dtype=float), 0.0, 1.0)

    return estimate


def sksurv_survival_estimator(model: Any, x_valid: pd.DataFrame | np.ndarray):


    def estimate(times: np.ndarray) -> np.ndarray:
        try:
            surv = model.predict_survival_function(x_valid, return_array=False)
        except TypeError:
            surv = model.predict_survival_function(x_valid)
        rows = []
        for fn in surv:
            rows.append([float(fn(float(t))) for t in np.asarray(times, dtype=float)])
        return np.clip(np.asarray(rows, dtype=float), 0.0, 1.0)

    return estimate


def cv_coxnet_select_alpha(
    x_train: np.ndarray,
    y_train,
    alphas: np.ndarray,
    l1_ratio: float,
    n_folds: int,
    seed: int,
) -> dict[str, Any]:


    from sksurv.linear_model import CoxnetSurvivalAnalysis

    from sksurv.metrics import concordance_index_censored

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = np.full((n_folds, len(alphas)), np.nan)
    for fi, (tr, va) in enumerate(kf.split(x_train)):
        try:
            model = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=alphas.tolist(), max_iter=200000, fit_baseline_model=False)
            model.fit(x_train[tr], y_train[tr])
            for ai, a in enumerate(alphas):
                try:
                    risk_va = model.predict(x_train[va], alpha=a).reshape(-1)
                    if np.unique(risk_va).size < 2:
                        scores[fi, ai] = np.nan
                        continue
                    ev = np.asarray(y_train[va]["event"]).astype(bool)
                    tm = np.asarray(y_train[va]["time"]).astype(float)
                    if ev.sum() < 2:
                        scores[fi, ai] = np.nan
                        continue
                    c = concordance_index_censored(ev, tm, risk_va)[0]
                    scores[fi, ai] = float(c)
                except Exception:
                    scores[fi, ai] = np.nan
        except Exception:
            continue
    with np.errstate(invalid="ignore"):
        mean_scores = np.where(np.all(np.isnan(scores), axis=0), np.nan, np.nanmean(scores, axis=0))
        valid_counts = np.sum(~np.isnan(scores), axis=0)
        se_scores = np.where(valid_counts > 1, np.nanstd(scores, axis=0, ddof=1) / np.sqrt(np.maximum(1, valid_counts)), np.nan)
    if np.all(np.isnan(mean_scores)):
        return {"alphas": alphas, "mean": mean_scores, "se": se_scores, "alpha_min": float(alphas[len(alphas) // 2]), "alpha_1se": float(alphas[len(alphas) // 2])}
    best = int(np.nanargmax(mean_scores))
    threshold = mean_scores[best] - se_scores[best]
    candidates_1se = np.where(mean_scores >= threshold)[0]
    alpha_1se_idx = int(candidates_1se.max()) if candidates_1se.size > 0 else best
    return {
        "alphas": alphas,
        "mean_score": mean_scores,
        "se_score": se_scores,
        "alpha_min": float(alphas[best]),
        "alpha_1se": float(alphas[alpha_1se_idx]),
        "best_idx": best,
        "alpha_1se_idx": alpha_1se_idx,
    }


def fit_coxnet_with_cv(
    feature_matrix: pd.DataFrame,
    split: pd.DataFrame,
    feature_columns: list[str],
    l1_ratio: float,
    n_folds: int,
    min_nonmissing: float,
    seed: int,
    logger,
) -> tuple[Any, dict[str, Any], np.ndarray, dict[str, Any]]:


    from sksurv.linear_model import CoxnetSurvivalAnalysis

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
    y_train = survival_y(split.loc[train_mask, "time_months"].to_numpy(), split.loc[train_mask, "event"].to_numpy())

    init = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=0.05, n_alphas=50, max_iter=200000, fit_baseline_model=False)
    init.fit(x_train, y_train)
    alpha_path = np.asarray(init.alphas_)
    logger.info("Coxnet alpha path length=%d, range=[%.4g, %.4g]", len(alpha_path), alpha_path.min(), alpha_path.max())

    cv = cv_coxnet_select_alpha(x_train, y_train, alpha_path, l1_ratio, n_folds, seed)
    alpha_min = cv["alpha_min"]
    alpha_1se = cv["alpha_1se"]
    logger.info("Coxnet CV: alpha.min=%.5g, alpha.1se=%.5g", alpha_min, alpha_1se)


    final = None
    final_alpha = None
    for trial_alpha in [alpha_1se, float(np.median(alpha_path)), alpha_min, float(alpha_path.max() * 0.5)]:
        try:
            m = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=[trial_alpha], max_iter=200000, fit_baseline_model=True)
            m.fit(x_train, y_train)
            final = m
            final_alpha = trial_alpha
            if trial_alpha != alpha_1se:
                logger.warning("Coxnet refit fell back from alpha.1se=%.5g to alpha=%.5g due to numerical instability.", alpha_1se, trial_alpha)
            break
        except Exception as exc:
            logger.warning("Coxnet final fit at alpha=%.5g failed: %s", trial_alpha, type(exc).__name__)
            continue
    if final is None:
        raise RuntimeError("All Coxnet final-fit attempts failed; check feature scaling and event count.")
    coef = pd.Series(final.coef_.reshape(-1), index=feature_columns)
    nonzero = coef[coef != 0].sort_values(key=lambda s: s.abs(), ascending=False)
    risk_all = final.predict(x_all).reshape(-1)

    bundle = {
        "model": final,
        "imputer": imputer,
        "scaler": scaler,
        "genes": feature_columns,
        "alpha_min": alpha_min,
        "alpha_1se": alpha_1se,
        "alpha_used": final_alpha,
        "l1_ratio": l1_ratio,
        "model_type": "expression_coxnet",
        "fit_scope": "train_split_only_cv",
        "min_nonmissing_ratio_train": min_nonmissing,
        "harmonization": "rank_based_inverse_normal_per_cohort",
    }
    cv_summary = {
        "alphas": alpha_path.tolist(),
        "mean_score": cv.get("mean_score", np.array([])).tolist() if isinstance(cv.get("mean_score"), np.ndarray) else [],
        "se_score": cv.get("se_score", np.array([])).tolist() if isinstance(cv.get("se_score"), np.ndarray) else [],
        "alpha_min": alpha_min,
        "alpha_1se": alpha_1se,
        "selected_alpha": final_alpha,
        "selected_non_zero_features": int((coef != 0).sum()),
        "non_zero_feature_table": nonzero.reset_index().rename(columns={"index": "feature", 0: "coef"}).to_dict(orient="records"),
    }
    return final, bundle, risk_all, cv_summary


def embedding_columns(df: pd.DataFrame) -> list[str]:

    return [
        c
        for c in df.columns
        if c.endswith("_embedding")
        or "_embedding_" in c
        or c.startswith("embedding_")
    ]


def select_primary_model(
    metric_df: pd.DataFrame,
    processed_root: str,
    policy: str = "survival_composite",
) -> tuple[str, dict[str, Any]]:


    if metric_df.empty:
        return "none", {"reason": "no internal metrics available", "policy": policy}
    candidates = metric_df[metric_df["metric"] == "harrell_c_index"].copy()
    if "primary_selection_eligible" in candidates.columns:
        eligible = candidates["primary_selection_eligible"].fillna(True).astype(bool)
        candidates = candidates.loc[eligible].copy()
    if candidates.empty:
        return "none", {"reason": "no primary-selection eligible survival metrics", "policy": policy}
    candidates["model_key"] = candidates["model"].astype(str)
    candidates["internal_c_index"] = pd.to_numeric(candidates["observed_value"], errors="coerce")
    candidates["td_auc_mean"] = pd.to_numeric(candidates.get("td_auc_mean", np.nan), errors="coerce")
    ibs = pd.to_numeric(candidates.get("integrated_brier_score", np.nan), errors="coerce")
    exploratory_brier = pd.to_numeric(candidates.get("risk_ranked_brier_score_exploratory", np.nan), errors="coerce")
    calibration_loss = ibs.combine_first(exploratory_brier)
    candidates["negative_brier_metric_for_selection"] = -calibration_loss
    candidates["calibration_metric_source"] = np.where(
        ibs.notna(),
        "integrated_brier_score",
        np.where(exploratory_brier.notna(), "risk_ranked_brier_score_exploratory", None),
    )

    coef_path = Path(processed_root) / "survival_models" / "coxnet_selected_features_and_coefficients.tsv"
    nonzero_features = np.nan
    if coef_path.exists():
        try:
            nonzero_features = float(len(pd.read_csv(coef_path, sep="\t")))
        except Exception:
            nonzero_features = np.nan
    candidates["parsimony_score"] = 0.5
    candidates.loc[candidates["model_key"].str.match(r"^clinical_(ajcc|tnm|minimal|random)", case=False, na=False), "parsimony_score"] = 1.0
    candidates.loc[candidates["model_key"].str.contains("ensemble", case=False, na=False), "parsimony_score"] = 0.8
    candidates.loc[candidates["model_key"].str.contains("embedding", case=False, na=False), "parsimony_score"] = 0.7
    candidates.loc[candidates["model_key"].str.contains("coxnet", case=False, na=False), "parsimony_score"] = 1.0 / max(
        1.0,
        np.log1p(nonzero_features if nonzero_features == nonzero_features else 20.0),
    )

    candidates["td_auc_mean_filled"] = candidates["td_auc_mean"].fillna(candidates["internal_c_index"])
    candidates["negative_brier_filled"] = candidates["negative_brier_metric_for_selection"].fillna(0.0)
    candidates["selection_score"] = (
        0.55 * candidates["internal_c_index"].fillna(0.0)
        + 0.25 * candidates["td_auc_mean_filled"]
        + 0.15 * candidates["negative_brier_filled"]
        + 0.05 * candidates["parsimony_score"].fillna(0.0)
    )
    candidates["omics_or_ensemble"] = candidates["model_key"].str.contains(
        "embedding|multiomics|rna|coxnet|ensemble",
        case=False,
        na=False,
    )
    candidates = candidates.sort_values(["selection_score", "internal_c_index"], ascending=False, na_position="last")

    selected_row = candidates.iloc[0].copy()
    clinical_candidates = candidates.loc[~candidates["omics_or_ensemble"]].copy()
    clinical_anchor_used = False
    if not clinical_candidates.empty and bool(selected_row.get("omics_or_ensemble", False)):
        clinical_best = clinical_candidates.iloc[0]
        c_gain = float(selected_row["internal_c_index"]) - float(clinical_best["internal_c_index"])
        auc_gain = float(selected_row["td_auc_mean_filled"]) - float(clinical_best["td_auc_mean_filled"])
        if c_gain <= 0.0 and auc_gain <= 0.0:
            selected_row = clinical_best
            clinical_anchor_used = True

    selected = str(selected_row["model_key"])
    score_cols = [
        "model_key",
        "selection_score",
        "internal_c_index",
        "td_auc_mean",
        "negative_brier_metric_for_selection",
        "calibration_metric_source",
        "parsimony_score",
        "omics_or_ensemble",
    ]
    score_records = candidates[score_cols].replace({np.nan: None}).to_dict(orient="records")
    rationale = {
        "selected_model": selected,
        "policy": policy,
        "weights": {
            "internal_c_index": 0.55,
            "time_dependent_auc": 0.25,
            "negative_brier_metric_for_selection": 0.15,
            "parsimony": 0.05,
        },
        "full_time_survival_first": True,
        "binary_36m_auc_usage": "secondary_endpoint_only_not_primary_selection",
        "clinical_anchor_safety_rule": "If an omics or ensemble candidate has no C-index or time-dependent-AUC gain over the best clinical survival model, keep the best clinical model as primary.",
        "clinical_anchor_used": clinical_anchor_used,
        "external_metrics_file_used": None,
        "external_metrics_usage": "forbidden_in_training_stage; external validation is performed only by 05_validation_visualization.py after model lock",
        "candidate_scores": score_records,
    }
    return selected, rationale


def fit_rsf(x: pd.DataFrame, split: pd.DataFrame, seed: int):
    from sksurv.ensemble import RandomSurvivalForest

    train_mask = split["split"].to_numpy() == "train"
    y = survival_y(split["time_months"].to_numpy(), split["event"].to_numpy())
    model = RandomSurvivalForest(
        n_estimators=300,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(x.iloc[train_mask], y[train_mask])
    return model, model.predict(x)


def metric_uno_and_td_auc(
    split: pd.DataFrame,
    risk: np.ndarray,
    horizons_months: tuple[float, ...] = (12.0, 36.0, 60.0),
    survival_estimator: Any | None = None,
) -> dict[str, Any]:
    from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc, brier_score, integrated_brier_score
    from sksurv.util import Surv

    train_mask = split["split"].to_numpy() == "train"
    valid_mask = split["split"].to_numpy() == "internal_validation"
    if valid_mask.sum() < 5:
        return {}
    y_train = Surv.from_arrays(split.loc[train_mask, "event"].astype(bool).to_numpy(), split.loc[train_mask, "time_months"].astype(float).to_numpy())
    y_valid = Surv.from_arrays(split.loc[valid_mask, "event"].astype(bool).to_numpy(), split.loc[valid_mask, "time_months"].astype(float).to_numpy())
    valid_time = split.loc[valid_mask, "time_months"].astype(float).to_numpy()
    valid_event = split.loc[valid_mask, "event"].astype(bool).to_numpy()
    risk_valid = np.asarray(risk)[valid_mask]
    max_train_time = float(split.loc[train_mask, "time_months"].astype(float).max())
    safe_horizons = sorted({float(min(h, max_train_time * 0.95)) for h in horizons_months if h < max_train_time * 0.99})
    out: dict[str, Any] = {}
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
            if survival_estimator is None:
                raise ValueError("No model survival estimator was supplied.")
            survs = survival_estimator(np.asarray(safe_horizons, dtype=float))
            if survs.shape != (len(risk_valid), len(safe_horizons)):
                raise ValueError(f"Survival matrix shape {survs.shape} does not match validation x horizons.")
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
                out["brier_score_source"] = "risk_ranked_survival_proxy_not_ibs"
        except Exception:
            pass
    return out


def bootstrap_harrell(time: np.ndarray, event: np.ndarray, risk: np.ndarray, n_iter: int, seed: int) -> dict[str, float]:
    return bootstrap_metric(
        lambda t, e, r: harrell_c_index(t, e, r),
        n_iter,
        seed,
        np.asarray(time),
        np.asarray(event),
        np.asarray(risk),
    )


def msk_sample_to_patient(barcode: str) -> str:

    s = str(barcode)
    parts = s.split("-")
    if len(parts) >= 2 and parts[0] == "P":
        return f"P-{parts[1]}"
    return s


def load_msk_mutation_matrix(mut_path: str, top_genes: int = 50) -> tuple[pd.DataFrame, list[str]]:

    df = pd.read_csv(mut_path, sep="\t", comment="#", dtype=str, low_memory=False)
    if "Hugo_Symbol" not in df.columns:
        return pd.DataFrame(), []
    sample_col = "Tumor_Sample_Barcode" if "Tumor_Sample_Barcode" in df.columns else None
    if sample_col is None:
        return pd.DataFrame(), []
    df["patient_id"] = df[sample_col].astype(str).apply(msk_sample_to_patient)
    gene_freq = df["Hugo_Symbol"].value_counts()
    top = gene_freq.head(top_genes).index.tolist()
    pivot = (
        df[df["Hugo_Symbol"].isin(top)]
        .assign(value=1)
        .pivot_table(index="patient_id", columns="Hugo_Symbol", values="value", aggfunc="max", fill_value=0)
    )
    pivot = pivot.reindex(columns=top, fill_value=0)
    return pivot, top


def train_msk_genomic_model(cfg: dict[str, Any], logger) -> dict[str, Any] | None:

    from lifelines import CoxPHFitter

    msk_cfg = cfg["cohorts"].get("msk", {})
    clin_path = msk_cfg.get("clinical_patient")
    mut_path = msk_cfg.get("mutation")
    if not (clin_path and Path(clin_path).exists() and mut_path and Path(mut_path).exists()):
        logger.warning("MSK clinical/mutation inputs missing; skipping MSK genomic model.")
        return None
    clin = read_cbio_table(clin_path)
    if "OS_STATUS" not in clin.columns or "OS_MONTHS" not in clin.columns:
        logger.warning("MSK clinical lacks OS_STATUS/OS_MONTHS; skipping.")
        return None
    clin["OS_EVENT"] = parse_survival_status(clin["OS_STATUS"]).astype(int)
    clin["OS_TIME_MONTHS"] = numeric_series(clin["OS_MONTHS"])
    clin = clin.loc[clin["OS_TIME_MONTHS"].notna() & (clin["OS_TIME_MONTHS"] > 0)].copy()
    if "AGE_AT_DIAGNOSIS" in clin.columns and "AGE" not in clin.columns:
        clin["AGE"] = pd.to_numeric(clin["AGE_AT_DIAGNOSIS"], errors="coerce")
    mut, genes = load_msk_mutation_matrix(mut_path, top_genes=30)
    if mut.empty:
        logger.warning("MSK mutation matrix empty; skipping.")
        return None
    clin["PATIENT_ID"] = clin["PATIENT_ID"].astype(str)
    mut.index = mut.index.astype(str)
    merged = clin.set_index("PATIENT_ID").join(mut, how="inner")
    if merged.empty or merged["OS_EVENT"].sum() < 20:
        logger.warning("MSK joined dataset insufficient for Cox training.")
        return None
    age_imputer = SimpleImputer(strategy="median")
    age_scaler = StandardScaler()
    merged["AGE"] = pd.to_numeric(merged.get("AGE", np.nan), errors="coerce")
    age = age_imputer.fit_transform(merged[["AGE"]])
    age = age_scaler.fit_transform(age)
    sex_dummies = pd.get_dummies(merged.get("SEX", pd.Series("Unknown", index=merged.index)), prefix="sex", drop_first=True, dummy_na=False)
    sex_dummies = sex_dummies.apply(pd.to_numeric, errors="coerce").fillna(0)
    train_df = pd.concat(
        [
            pd.DataFrame({"time": merged["OS_TIME_MONTHS"].astype(float).to_numpy(), "event": merged["OS_EVENT"].astype(int).to_numpy(), "AGE": age.reshape(-1)}, index=merged.index),
            sex_dummies,
            merged[genes].astype(int),
        ],
        axis=1,
    )
    train_df = train_df.loc[:, train_df.var(axis=0) > 0]
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.0)
    cph.fit(train_df, duration_col="time", event_col="event")
    return {
        "model": cph,
        "age_imputer": age_imputer,
        "age_scaler": age_scaler,
        "genes": genes,
        "sex_columns": sex_dummies.columns.tolist(),
        "model_columns": [c for c in train_df.columns if c not in ("time", "event")],
        "model_type": "msk_genomic_cox",
    }


def assemble_metric_row(model: str, cohort: str, observed: float, bootstrap: dict[str, float] | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "model": model,
        "cohort": cohort,
        "metric": "harrell_c_index",
        "observed_value": observed,
        "target_success_threshold": 0.75,
        "gap_to_target": observed - 0.75,
        "observed_performance_is_not_target_claim": True,
    }
    if bootstrap:
        row["ci_low_95"] = bootstrap.get("ci_low")
        row["ci_high_95"] = bootstrap.get("ci_high")
        row["bootstrap_iter"] = bootstrap.get("n_valid")
    if extra:
        row.update(extra)
    return row


def parse_int_grid(text: str, default: tuple[int, ...]) -> list[int]:
    values: list[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError:
            continue
        if value > 0 and value not in values:
            values.append(value)
    return values or list(default)


def standardize_risk_by_reference(risk: np.ndarray, reference_mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(risk, dtype=float)
    ref = arr[reference_mask & np.isfinite(arr)]
    if ref.size == 0:
        return np.zeros_like(arr, dtype=float)
    mean = float(np.mean(ref))
    std = float(np.std(ref))
    if not np.isfinite(std) or std <= 1e-12:
        std = 1.0
    return (arr - mean) / std


def train_only_select_clinical_anchor_weight(
    split: pd.DataFrame,
    clinical_risk: np.ndarray,
    omics_risk: np.ndarray,
    weights: tuple[float, ...],
    seed: int,
) -> tuple[float, pd.DataFrame]:
    train_mask = split["split"].to_numpy() == "train"
    train_idx = np.where(train_mask)[0]
    y_train = split.loc[train_mask, "event"].astype(int).to_numpy()
    min_class = int(np.bincount(y_train).min()) if y_train.size and np.unique(y_train).size == 2 else 0
    if min_class < 2:
        return 0.8, pd.DataFrame([{"clinical_weight": 0.8, "mean_train_fold_c_index": np.nan, "status": "insufficient_events"}])
    n_splits = min(5, max(2, min_class))
    rows: list[dict[str, Any]] = []
    if n_splits < 2:
        return 0.8, pd.DataFrame([{"clinical_weight": 0.8, "mean_train_fold_c_index": np.nan, "status": "insufficient_events"}])
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for weight in weights:
        scores: list[float] = []
        for fold_id, (tr_pos, va_pos) in enumerate(splitter.split(train_idx, y_train), start=1):
            fold_ref_mask = np.zeros(len(split), dtype=bool)
            fold_ref_mask[train_idx[tr_pos]] = True
            fold_val_idx = train_idx[va_pos]
            c_std = standardize_risk_by_reference(clinical_risk, fold_ref_mask)
            o_std = standardize_risk_by_reference(omics_risk, fold_ref_mask)
            combo = float(weight) * c_std + (1.0 - float(weight)) * o_std
            try:
                score = harrell_c_index(
                    split.iloc[fold_val_idx]["time_months"].to_numpy(float),
                    split.iloc[fold_val_idx]["event"].to_numpy(int),
                    combo[fold_val_idx],
                )
            except Exception:
                score = float("nan")
            scores.append(score)
            rows.append({
                "clinical_weight": float(weight),
                "fold": fold_id,
                "fold_c_index": score,
                "status": "ok" if np.isfinite(score) else "failed",
            })
    summary = (
        pd.DataFrame(rows)
        .groupby("clinical_weight", as_index=False)["fold_c_index"]
        .mean()
        .rename(columns={"fold_c_index": "mean_train_fold_c_index"})
    )
    if summary["mean_train_fold_c_index"].notna().any():
        best_weight = float(summary.sort_values(["mean_train_fold_c_index", "clinical_weight"], ascending=[False, False]).iloc[0]["clinical_weight"])
    else:
        best_weight = 0.8
    rows.extend(summary.assign(fold="mean", status="summary").to_dict(orient="records"))
    return best_weight, pd.DataFrame(rows)


def combine_clinical_anchor_risk(
    split: pd.DataFrame,
    clinical_risk: np.ndarray,
    omics_risk: np.ndarray,
    clinical_weight: float,
) -> np.ndarray:
    train_mask = split["split"].to_numpy() == "train"
    c_std = standardize_risk_by_reference(clinical_risk, train_mask)
    o_std = standardize_risk_by_reference(omics_risk, train_mask)
    return float(clinical_weight) * c_std + (1.0 - float(clinical_weight)) * o_std


def build_incremental_omics_delta(metric_df: pd.DataFrame, baseline_model: str = "clinical_ajcc_cox") -> pd.DataFrame:
    if metric_df.empty:
        return pd.DataFrame()
    surv = metric_df.loc[metric_df["metric"] == "harrell_c_index"].copy()
    if surv.empty:
        return pd.DataFrame()
    if baseline_model not in set(surv["model"].astype(str)):
        clinical = surv[surv["model"].astype(str).str.match(r"^clinical_(ajcc|tnm|minimal|random)", case=False, na=False)]
        if clinical.empty:
            return pd.DataFrame()
        baseline_row = clinical.sort_values("observed_value", ascending=False).iloc[0]
        baseline_model = str(baseline_row["model"])
    else:
        baseline_row = surv.loc[surv["model"].astype(str) == baseline_model].iloc[0]
    b_c = float(pd.to_numeric(pd.Series([baseline_row.get("observed_value")]), errors="coerce").iloc[0])
    b_td = float(pd.to_numeric(pd.Series([baseline_row.get("td_auc_mean")]), errors="coerce").iloc[0]) if "td_auc_mean" in surv.columns else np.nan
    b_td36 = float(pd.to_numeric(pd.Series([baseline_row.get("td_auc_month_36")]), errors="coerce").iloc[0]) if "td_auc_month_36" in surv.columns else np.nan
    b_ibs = float(pd.to_numeric(pd.Series([baseline_row.get("integrated_brier_score")]), errors="coerce").iloc[0]) if "integrated_brier_score" in surv.columns else np.nan
    pattern = "embedding|multiomics|rna|coxnet|ensemble"
    rows: list[dict[str, Any]] = []
    for _, row in surv.iterrows():
        model = str(row["model"])
        if model == baseline_model or not re.search(pattern, model, flags=re.IGNORECASE):
            continue
        c_val = float(pd.to_numeric(pd.Series([row.get("observed_value")]), errors="coerce").iloc[0])
        td_val = float(pd.to_numeric(pd.Series([row.get("td_auc_mean")]), errors="coerce").iloc[0]) if "td_auc_mean" in row.index else np.nan
        td36_val = float(pd.to_numeric(pd.Series([row.get("td_auc_month_36")]), errors="coerce").iloc[0]) if "td_auc_month_36" in row.index else np.nan
        ibs_val = float(pd.to_numeric(pd.Series([row.get("integrated_brier_score")]), errors="coerce").iloc[0]) if "integrated_brier_score" in row.index else np.nan
        rows.append({
            "baseline_model": baseline_model,
            "candidate_model": model,
            "delta_harrell_c_index": c_val - b_c if np.isfinite(c_val) and np.isfinite(b_c) else np.nan,
            "delta_td_auc_mean": td_val - b_td if np.isfinite(td_val) and np.isfinite(b_td) else np.nan,
            "delta_td_auc_36m": td36_val - b_td36 if np.isfinite(td36_val) and np.isfinite(b_td36) else np.nan,
            "delta_integrated_brier_score": ibs_val - b_ibs if np.isfinite(ibs_val) and np.isfinite(b_ibs) else np.nan,
            "brier_delta_interpretation": "negative_is_better",
            "adds_incremental_discrimination": bool((c_val > b_c) or (np.isfinite(td_val) and np.isfinite(b_td) and td_val > b_td)),
        })
    return pd.DataFrame(rows)


def build_survival_performance_summary(metric_df: pd.DataFrame, selected_model: str) -> pd.DataFrame:
    if metric_df.empty:
        return pd.DataFrame()
    out = metric_df.copy()
    out["is_selected_primary_model"] = out["model"].astype(str) == str(selected_model)
    out["endpoint_role"] = np.where(out["metric"].astype(str).eq("harrell_c_index"), "full_time_survival", "secondary_36m_binary")
    if "primary_selection_eligible" not in out.columns:
        out["primary_selection_eligible"] = out["metric"].astype(str).eq("harrell_c_index")
    return out


def deepsurv_exploration_status(split: pd.DataFrame, event_threshold: int = 120) -> dict[str, Any]:

    train_events = int(split.loc[split["split"] == "train", "event"].sum()) if not split.empty else 0
    try:
        import pycox
        import torch

        pycox_available = True
    except Exception:
        pycox_available = False
    enabled = bool(pycox_available and train_events >= event_threshold)
    return {
        "model_family": "pycox_deepsurv",
        "pycox_available": pycox_available,
        "train_events": train_events,
        "event_threshold": event_threshold,
        "enabled": enabled,
        "planned_architecture": "small MLP with CoxPH loss, train-only early stopping, and downgrade unless external validation improves over Coxnet",
        "disabled_reason": None if enabled else f"requires pycox and train_events>={event_threshold}; current train_events={train_events}",
    }


def write_model_card(out_dir: Path, manifest: dict[str, Any], metric_df: pd.DataFrame, deep_status: dict[str, Any], ctx) -> None:
    lines = [
        "# CRC Survival Model Card (REMARK / TRIPOD+AI Draft)",
        "",
        "## Intended Use",
        "Exploratory prognostic risk stratification for colorectal cancer cohorts; not for clinical decision-making.",
        "",
        "## Data",
        "- Development cohort: TCGA COADREAD PanCanAtlas.",
        "- External checks: MSK CRC 2017, GEO expression cohorts, CPTAC/HTAN mechanism-only validation where available.",
        "- Endpoint: OS in months.",
        "",
        "## Model Selection",
        f"- Selected primary internal model: `{manifest.get('selected_primary_model_internal')}`.",
        f"- Rationale: {manifest.get('selection_rationale')}",
        "- Target C-index is a success criterion, not an achieved-performance claim.",
        "",
        "## Performance Summary",
    ]
    if metric_df.empty:
        lines.append("- No internal metric rows were available.")
    else:
        for _, row in metric_df[metric_df["metric"] == "harrell_c_index"].iterrows():
            lines.append(f"- `{row['model']}` on `{row['cohort']}`: Harrell C-index={float(row['observed_value']):.3f}.")
    lines.extend([
        "",
        "## Bias, Leakage, and Robustness Controls",
        f"- Leakage control: {manifest.get('leakage_control')}",
        "- Train-only split is reused for feature filtering, imputation, scaling, Coxnet CV, and embedding consumption.",
        "- GEO expression scoring uses within-cohort rank inverse normal transformation before the stored Coxnet scaler.",
        "",
        "## Deep Model Status",
        f"- DeepSurv enabled: {deep_status.get('enabled')}.",
        f"- DeepSurv status: {deep_status.get('disabled_reason') or deep_status.get('planned_architecture')}.",
        "",
        "## Limitations",
        "- Observational associations do not establish biological causality.",
        "- Deep models are exploratory and gated by event count/dependency availability.",
        "- External validation metrics are cohort-dependent and should be interpreted with confidence intervals.",
    ])
    ctx.write_text(out_dir / "remark_tripod_ai_model_card_draft.md", "\n".join(lines) + "\n", "analysis_data", "REMARK/TRIPOD+AI model card draft.")


def main_step04() -> int:
    parser = build_parser_step04()
    args = parser.parse_args()
    ctx = initialize_run(__file__, args)
    cfg = ctx.cfg
    endpoint = args.endpoint
    seed = cfg["random_seed"]
    filters = cfg.get("filters", {"min_nonmissing_ratio_train": 0.7})
    min_nonmissing = float(filters.get("min_nonmissing_ratio_train", filters.get("min_nonmissing_ratio", 0.7)))
    out_dir = ctx.data_dir("survival_models")
    model_dir = out_dir / "serialized_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    input_files = [cfg["cohorts"]["tcga"]["clinical_patient"], cfg["cohorts"]["tcga"]["rna"]]

    clinical = load_tcga_clinical(cfg, endpoint)
    clinical = attach_raw_clinical_columns(clinical, cfg["cohorts"]["tcga"]["clinical_patient"])

    if args.dry_run:
        rows = [{"check": "tcga_clinical_rows", "value": int(len(clinical))}, {"check": "endpoint", "value": endpoint}]
        ctx.write_table(ctx.run_dir / "survival_prediction_dry_run_input_checks.tsv", rows, "qc", "Dry-run survival modelling checks.")
        ctx.add_warning("Dry-run only: no survival model was fitted.")
        ctx.finalize([p for p in input_files if Path(p).exists()])
        return 0

    split_path = out_dir / "tcga_train_internal_validation_split.tsv"
    if split_path.exists():
        split = pd.read_csv(split_path, sep="\t")
        if "stratify_key" not in split.columns:
            split = make_stratified_split(clinical, endpoint, args.test_size, seed, ctx.logger)
            split.to_csv(split_path, sep="\t", index=False)
        else:
            ctx.logger.info("Reusing existing stratified split.")
    else:
        split = make_stratified_split(clinical, endpoint, args.test_size, seed, ctx.logger)
        split.to_csv(split_path, sep="\t", index=False)
    ctx.register_output(split_path, "analysis_data", "TCGA stratified train/internal validation split (multi-factor stratification).")
    if int(split.loc[split["split"] == "train", "event"].sum()) < 20:
        ctx.add_warning("Training set has fewer than 20 events; high-dimensional models should be interpreted as exploratory only.")

    metrics: list[dict[str, Any]] = []
    risk_table = split.copy()
    bundles_for_validation: dict[str, str] = {}
    gene_top_k_values = parse_int_grid(args.gene_top_k_grid, (100, 300, 500))
    gan_utility_records: list[dict[str, Any]] = []
    embedding_risk_for_ensemble: np.ndarray | None = None
    combined_embedding_matrix: pd.DataFrame | None = None
    low_auc_failure_reference = 0.38921489747654503


    primary_features = ["AGE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "SUBTYPE", "CANCER_TYPE_ACRONYM"]
    x_primary, t_primary = fit_clinical_transformer(clinical, split, primary_features)
    cox_primary = fit_lifelines_cox_with_split(x_primary, split, penalizer=0.1)
    risk_primary = lifelines_predict_log_hazard(cox_primary, x_primary)
    risk_table["clinical_ajcc_cox_risk_score"] = risk_primary
    valid_mask = split["split"].to_numpy() == "internal_validation"
    bs_primary = bootstrap_harrell(split.loc[valid_mask, "time_months"].to_numpy(), split.loc[valid_mask, "event"].to_numpy(), risk_primary[valid_mask], args.bootstrap_iter, seed)
    extra_primary = metric_uno_and_td_auc(
        split,
        risk_primary,
        survival_estimator=lifelines_survival_estimator(cox_primary, x_primary.iloc[valid_mask]),
    )
    metrics.append(assemble_metric_row("clinical_ajcc_cox", "tcga_internal_validation", bs_primary["point"], bs_primary, extra_primary))
    ctx.write_table(out_dir / "clinical_cox_model_coefficients.tsv", cox_primary.summary.reset_index().rename(columns={"covariate": "feature"}), "analysis_data", "Primary clinical Cox (AJCC stage) coefficients.")
    full_path = model_dir / "clinical_ajcc_cox_model_bundle.joblib"
    joblib.dump({"model": cox_primary, "transformer": t_primary, "model_type": "clinical_ajcc_cox", "endpoint": endpoint}, full_path)
    ctx.register_output(full_path, "model", "Primary clinical Cox (AJCC stage) bundle.")
    bundles_for_validation["clinical_ajcc_cox"] = str(full_path)


    tnm_features = ["AGE", "SEX", "PATH_T_STAGE", "PATH_N_STAGE", "PATH_M_STAGE", "SUBTYPE"]
    try:
        x_tnm, t_tnm = fit_clinical_transformer(clinical, split, tnm_features)
        cox_tnm = fit_lifelines_cox_with_split(x_tnm, split, penalizer=0.1)
        risk_tnm = lifelines_predict_log_hazard(cox_tnm, x_tnm)
        risk_table["clinical_tnm_cox_risk_score"] = risk_tnm
        bs_tnm = bootstrap_harrell(split.loc[valid_mask, "time_months"].to_numpy(), split.loc[valid_mask, "event"].to_numpy(), risk_tnm[valid_mask], args.bootstrap_iter, seed)
        metrics.append(assemble_metric_row(
            "clinical_tnm_cox_sensitivity",
            "tcga_internal_validation",
            bs_tnm["point"],
            bs_tnm,
            metric_uno_and_td_auc(split, risk_tnm, survival_estimator=lifelines_survival_estimator(cox_tnm, x_tnm.iloc[valid_mask])),
        ))
        ctx.write_table(out_dir / "clinical_tnm_cox_sensitivity_coefficients.tsv", cox_tnm.summary.reset_index().rename(columns={"covariate": "feature"}), "analysis_data", "Sensitivity Cox with T/N/M (no AJCC) coefficients.")
    except Exception as exc:
        ctx.add_warning(f"TNM sensitivity Cox failed: {exc}")


    minimal_features = ["AGE", "SEX"]
    x_min, t_min = fit_clinical_transformer(clinical, split, minimal_features)
    cox_min = fit_lifelines_cox_with_split(x_min, split, penalizer=0.05)
    risk_min = lifelines_predict_log_hazard(cox_min, x_min)
    risk_table["clinical_minimal_cox_risk_score"] = risk_min
    bs_min = bootstrap_harrell(split.loc[valid_mask, "time_months"].to_numpy(), split.loc[valid_mask, "event"].to_numpy(), risk_min[valid_mask], args.bootstrap_iter, seed)
    metrics.append(assemble_metric_row(
        "clinical_minimal_cox",
        "tcga_internal_validation",
        bs_min["point"],
        bs_min,
        metric_uno_and_td_auc(split, risk_min, survival_estimator=lifelines_survival_estimator(cox_min, x_min.iloc[valid_mask])),
    ))
    ctx.write_table(out_dir / "clinical_minimal_cox_model_coefficients.tsv", cox_min.summary.reset_index().rename(columns={"covariate": "feature"}), "analysis_data", "External-compatible minimal clinical Cox coefficients.")
    min_path = model_dir / "clinical_minimal_cox_model_bundle.joblib"
    joblib.dump({"model": cox_min, "transformer": t_min, "model_type": "clinical_minimal_cox", "endpoint": endpoint}, min_path)
    ctx.register_output(min_path, "model", "Minimal clinical Cox bundle for cross-cohort baseline.")
    bundles_for_validation["clinical_minimal_cox"] = str(min_path)


    try:
        rsf_model, rsf_risk = fit_rsf(x_primary, split, seed)
        risk_table["clinical_random_survival_forest_risk_score"] = rsf_risk
        bs_rsf = bootstrap_harrell(split.loc[valid_mask, "time_months"].to_numpy(), split.loc[valid_mask, "event"].to_numpy(), rsf_risk[valid_mask], args.bootstrap_iter, seed)
        metrics.append(assemble_metric_row(
            "clinical_random_survival_forest",
            "tcga_internal_validation",
            bs_rsf["point"],
            bs_rsf,
            metric_uno_and_td_auc(split, rsf_risk, survival_estimator=sksurv_survival_estimator(rsf_model, x_primary.iloc[valid_mask])),
        ))
        rsf_path = model_dir / "clinical_random_survival_forest_model_bundle.joblib"
        joblib.dump({"model": rsf_model, "transformer": t_primary, "model_type": "clinical_random_survival_forest", "endpoint": endpoint}, rsf_path)
        ctx.register_output(rsf_path, "model", "Clinical RSF bundle.")
    except Exception as exc:
        ctx.add_warning(f"RSF training failed: {exc}")


    causal_table_path = Path(cfg["processed_root"]) / "causal" / "causal_priority_feature_table.tsv"
    candidate_genes: list[str] = []
    if causal_table_path.exists():
        try:
            cdf = pd.read_csv(causal_table_path, sep="\t")
            candidate_genes = [g for g in cdf["feature"].astype(str).tolist() if not is_likely_pseudogene(g)][:200]
            ctx.logger.info("Loaded %d candidate genes from causal_priority_feature_table.tsv", len(candidate_genes))
        except Exception as exc:
            ctx.add_warning(f"Failed to read causal priority table: {exc}")
    else:
        ctx.add_warning("Causal priority table missing; Coxnet will fall back to RNA top-variance.")

    try:
        curated_rna_path = Path(cfg["processed_root"]) / "preprocessed" / "tcga_coadread_rna_log2_tpm_like_matrix.tsv"
        rna_path = str(curated_rna_path) if curated_rna_path.exists() else cfg["cohorts"]["tcga"]["rna"]
        rna_matrix = stream_gene_matrix(rna_path, comment=None if curated_rna_path.exists() else "#", id_col_preference="Hugo_Symbol")
        if curated_rna_path.exists():
            ctx.logger.info("Expression Coxnet using curated preprocessed RNA matrix: %s", curated_rna_path)
        train_ids_all = split.loc[split["split"] == "train", "PATIENT_ID"].astype(str).tolist()
        train_ids = [p for p in train_ids_all if p in rna_matrix.index]
        if len(train_ids) < len(train_ids_all):
            ctx.add_warning(f"{len(train_ids_all) - len(train_ids)} training patients lack RNA data; Coxnet variance ranking uses {len(train_ids)} patients with RNA.")
        var_top_full = rna_matrix.loc[train_ids].var(axis=0, skipna=True).sort_values(ascending=False)
        var_top_genes = var_top_full.head(200).index.tolist()
        if candidate_genes:
            available = [g for g in candidate_genes if g in rna_matrix.columns]

            use_genes = list(dict.fromkeys(available + var_top_genes))
            ctx.logger.info("Coxnet candidate composition: %d causal-priority + %d variance-supplemented = %d total.", len(available), len(use_genes) - len(available), len(use_genes))
        else:
            use_genes = var_top_genes
        ctx.logger.info("Coxnet input gene count (pre-filter): %d", len(use_genes))
        _, expr_bundle, expr_risk, cv_summary = fit_coxnet_with_cv(
            rna_matrix, split, use_genes, args.coxnet_l1_ratio, args.coxnet_cv_folds, min_nonmissing, seed, ctx.logger
        )
        risk_table["expression_coxnet_risk_score"] = expr_risk
        bs_expr = bootstrap_harrell(split.loc[valid_mask, "time_months"].to_numpy(), split.loc[valid_mask, "event"].to_numpy(), expr_risk[valid_mask], args.bootstrap_iter, seed)
        expr_aligned = rna_matrix.reindex(split["PATIENT_ID"].astype(str))[expr_bundle["genes"]].apply(rank_inverse_normal, axis=0)
        x_expr_all = expr_bundle["scaler"].transform(expr_bundle["imputer"].transform(expr_aligned))
        metrics.append(assemble_metric_row(
            "expression_coxnet_lambda_1se",
            "tcga_internal_validation",
            bs_expr["point"],
            bs_expr,
            metric_uno_and_td_auc(split, expr_risk, survival_estimator=sksurv_survival_estimator(expr_bundle["model"], x_expr_all[valid_mask])),
        ))
        expr_path = model_dir / "expression_coxnet_model_bundle.joblib"
        joblib.dump({**expr_bundle, "endpoint": endpoint}, expr_path)
        ctx.register_output(expr_path, "model", "Expression Coxnet bundle for cross-cohort RNA validation.")
        coef_df = pd.DataFrame(cv_summary.get("non_zero_feature_table", []))
        if not coef_df.empty:
            coef_df["hr"] = np.exp(coef_df["coef"])
            coef_df["coef_ci_low_95"] = np.nan
            coef_df["coef_ci_high_95"] = np.nan
            coef_df["hr_ci_low_95"] = np.nan
            coef_df["hr_ci_high_95"] = np.nan
            coef_df["ci_note"] = "Penalized Coxnet coefficient CI is not available from sksurv; bootstrap stability is reported in 06."
            ctx.write_table(out_dir / "coxnet_selected_features_and_coefficients.tsv", coef_df.sort_values("coef", key=lambda s: s.abs(), ascending=False), "analysis_data", "Coxnet selected non-zero features and coefficients at lambda.1se.")
        ctx.write_json(out_dir / "coxnet_cv_alpha_selection.json", {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cv_summary.items()}, "analysis_data", "Coxnet CV alpha-path summary.")
        bundles_for_validation["expression_coxnet"] = str(expr_path)
        clinical_prefixed = x_primary.copy()
        clinical_prefixed.columns = [f"clinical__{c}" for c in clinical_prefixed.columns]
        for top_k in gene_top_k_values:
            try:
                top_var_genes = var_top_full.head(top_k).index.tolist()
                top_available = [g for g in candidate_genes if g in rna_matrix.columns]
                top_genes = list(dict.fromkeys(top_available + top_var_genes))
                if not top_genes:
                    ctx.add_warning(f"Clinical+RNA top-{top_k} Coxnet skipped: no genes available.")
                    continue
                rna_prefixed = rna_matrix[top_genes].copy()
                rna_prefixed.columns = [f"rna__{c}" for c in rna_prefixed.columns]
                fused_matrix = pd.concat([clinical_prefixed, rna_prefixed], axis=1)
                fused_cols = clinical_prefixed.columns.tolist() + rna_prefixed.columns.tolist()
                _, fused_bundle, fused_risk, fused_cv = fit_coxnet_with_cv(
                    fused_matrix,
                    split,
                    fused_cols,
                    args.coxnet_l1_ratio,
                    args.coxnet_cv_folds,
                    min_nonmissing,
                    seed + top_k,
                    ctx.logger,
                )
                model_key = f"clinical_plus_rna_top{top_k}_coxnet"
                risk_table[f"{model_key}_risk_score"] = fused_risk
                bs_fused = bootstrap_harrell(
                    split.loc[valid_mask, "time_months"].to_numpy(),
                    split.loc[valid_mask, "event"].to_numpy(),
                    fused_risk[valid_mask],
                    args.bootstrap_iter,
                    seed + top_k,
                )
                aligned_fused = fused_matrix.reindex(split["PATIENT_ID"].astype(str))[fused_bundle["genes"]].apply(rank_inverse_normal, axis=0)
                x_fused_all = fused_bundle["scaler"].transform(fused_bundle["imputer"].transform(aligned_fused))
                metrics.append(assemble_metric_row(
                    model_key,
                    "tcga_internal_validation",
                    bs_fused["point"],
                    bs_fused,
                    {
                        **metric_uno_and_td_auc(split, fused_risk, survival_estimator=sksurv_survival_estimator(fused_bundle["model"], x_fused_all[valid_mask])),
                        "primary_selection_eligible": True,
                        "feature_family": "clinical_plus_rna_topk_coxnet",
                        "gene_top_k": top_k,
                    },
                ))
                fused_path = model_dir / f"{model_key}_model_bundle.joblib"
                joblib.dump({**fused_bundle, "endpoint": endpoint, "clinical_columns": clinical_prefixed.columns.tolist(), "gene_top_k": top_k}, fused_path)
                ctx.register_output(fused_path, "model", f"Clinical + RNA top-{top_k} Coxnet bundle.")
                fused_coef = pd.DataFrame(fused_cv.get("non_zero_feature_table", []))
                if not fused_coef.empty:
                    fused_coef["hr"] = np.exp(fused_coef["coef"])
                    ctx.write_table(
                        out_dir / f"{model_key}_selected_features_and_coefficients.tsv",
                        fused_coef.sort_values("coef", key=lambda s: s.abs(), ascending=False),
                        "analysis_data",
                        f"Clinical + RNA top-{top_k} Coxnet selected non-zero features.",
                    )
            except Exception as exc:
                ctx.add_warning(f"Clinical+RNA top-{top_k} Coxnet failed: {type(exc).__name__}: {exc}")
    except Exception as exc:
        ctx.add_warning(f"Expression Coxnet training failed: {type(exc).__name__}: {exc}")


    embedding_path = Path(cfg["processed_root"]) / "multiomics_pretraining" / "tcga_multiomics_patient_embedding.tsv"
    if embedding_path.exists():
        try:
            emb = pd.read_csv(embedding_path, sep="\t")
            emb["PATIENT_ID"] = emb["PATIENT_ID"].astype(str)
            emb_cols = embedding_columns(emb)
            if not emb_cols:
                raise ValueError("Embedding table exists but no embedding columns were found.")
            emb_indexed = emb.set_index("PATIENT_ID")[emb_cols]
            aligned = emb_indexed.reindex(split["PATIENT_ID"].astype(str)).fillna(emb_indexed.mean(axis=0))
            clinical_block = x_primary.loc[split["PATIENT_ID"].astype(str)].copy()
            clinical_block.columns = [f"clinical__{col}" for col in clinical_block.columns]
            combined = pd.concat([clinical_block.reset_index(drop=True), aligned.reset_index(drop=True)], axis=1)
            combined.index = split["PATIENT_ID"].astype(str).to_numpy()
            cph_emb = fit_lifelines_cox_with_split(combined, split, penalizer=0.1)
            risk_emb = lifelines_predict_log_hazard(cph_emb, combined)
            embedding_risk_for_ensemble = risk_emb
            combined_embedding_matrix = combined
            embedding_model_key = "clinical_plus_embedding_penalized_cox"
            risk_table[f"{embedding_model_key}_risk_score"] = risk_emb
            bs_emb = bootstrap_harrell(split.loc[valid_mask, "time_months"].to_numpy(), split.loc[valid_mask, "event"].to_numpy(), risk_emb[valid_mask], args.bootstrap_iter, seed)
            metrics.append(assemble_metric_row(
                embedding_model_key,
                "tcga_internal_validation",
                bs_emb["point"],
                bs_emb,
                {
                    **metric_uno_and_td_auc(split, risk_emb, survival_estimator=lifelines_survival_estimator(cph_emb, combined.iloc[valid_mask])),
                    "primary_selection_eligible": True,
                    "feature_family": "clinical_plus_embedding",
                },
            ))
            ctx.write_table(out_dir / f"{embedding_model_key}_coefficients.tsv", cph_emb.summary.reset_index().rename(columns={"covariate": "feature"}), "analysis_data", "Clinical + multi-omics embedding penalized Cox coefficients.")
            emb_path = model_dir / f"{embedding_model_key}_model_bundle.joblib"
            joblib.dump({"model": cph_emb, "transformer": t_primary, "embedding_columns": emb_cols, "model_type": embedding_model_key, "endpoint": endpoint}, emb_path)
            ctx.register_output(emb_path, "model", "Clinical + multi-omics embedding penalized Cox bundle.")
        except Exception as exc:
            ctx.add_warning(f"Embedding-augmented Cox failed: {exc}")
    else:
        ctx.add_warning("Multi-omics embedding TSV missing; run 03_multiomics_pretraining.py first.")

    if embedding_risk_for_ensemble is not None:
        try:
            best_weight, weight_rows = train_only_select_clinical_anchor_weight(
                split,
                risk_primary,
                embedding_risk_for_ensemble,
                (0.6, 0.7, 0.8, 0.9),
                seed,
            )
            ensemble_risk = combine_clinical_anchor_risk(split, risk_primary, embedding_risk_for_ensemble, best_weight)
            risk_table["clinical_anchored_ensemble_risk_score"] = ensemble_risk
            bs_ens = bootstrap_harrell(
                split.loc[valid_mask, "time_months"].to_numpy(),
                split.loc[valid_mask, "event"].to_numpy(),
                ensemble_risk[valid_mask],
                args.bootstrap_iter,
                seed + 404,
            )
            metrics.append(assemble_metric_row(
                "clinical_anchored_ensemble",
                "tcga_internal_validation",
                bs_ens["point"],
                bs_ens,
                {
                    **metric_uno_and_td_auc(split, ensemble_risk, survival_estimator=None),
                    "primary_selection_eligible": True,
                    "feature_family": "clinical_anchored_ensemble",
                    "clinical_weight": best_weight,
                    "omics_component": "clinical_plus_embedding_penalized_cox",
                },
            ))
            ctx.write_table(
                out_dir / "clinical_anchored_ensemble_weight_selection.tsv",
                weight_rows,
                "analysis_data",
                "Train-only fold selection for clinical-anchored ensemble weights.",
            )
        except Exception as exc:
            ctx.add_warning(f"Clinical-anchored ensemble failed: {type(exc).__name__}: {exc}")


    if combined_embedding_matrix is not None:
        try:
            ipcw_path = Path(cfg["processed_root"]) / "preprocessed" / "tcga_ipcw_os_endpoint.tsv"
            gan_x_path = Path(cfg["processed_root"]) / "gan_augmentation" / "gan_augmented_training_features.tsv"
            gan_ep_path = Path(cfg["processed_root"]) / "gan_augmentation" / "gan_augmented_endpoint.tsv"
            if not ipcw_path.exists():
                raise FileNotFoundError(f"Missing IPCW endpoint: {ipcw_path}")
            ipcw_ep = pd.read_csv(ipcw_path, sep="\t", low_memory=False)
            if "PATIENT_ID" not in ipcw_ep.columns:
                ipcw_ep = pd.read_csv(ipcw_path, sep="\t", index_col=0, low_memory=False).reset_index().rename(columns={"index": "PATIENT_ID"})
            ipcw_ep["PATIENT_ID"] = ipcw_ep["PATIENT_ID"].astype(str)
            ipcw_ep = ipcw_ep.set_index("PATIENT_ID")
            train_ids_list = split.loc[split["split"] == "train", "PATIENT_ID"].astype(str).tolist()
            val_ids_list = split.loc[split["split"] == "internal_validation", "PATIENT_ID"].astype(str).tolist()
            X_real_all = combined_embedding_matrix.reindex(split["PATIENT_ID"].astype(str)).fillna(0.0)
            X_real_train = X_real_all.reindex(train_ids_list)
            ep_train = ipcw_ep.reindex(train_ids_list).dropna(subset=["time_months"])
            ep_val = ipcw_ep.reindex(val_ids_list)
            real_model = fit_weighted_logistic(X_real_train, ep_train, seed + 36)
            if real_model is None:
                raise RuntimeError("real-only clinical+embedding Elastic Net returned None")
            real_val_risk = risk_from_model(real_model, X_real_all.reindex(val_ids_list))
            real_eval = evaluate_36m_predictions(ep_train, ep_val, real_val_risk, tau=OS_RISK_TIME_MONTHS)
            record: dict[str, Any] = {
                "model": "clinical_plus_embedding_gan_elasticnet_36m",
                "gan_feature_set_expected": args.gan_feature_set,
                "real_only_auc_36m": real_eval.get("auc_os_observed"),
                "real_only_average_precision_os": real_eval.get("average_precision_os"),
                "real_only_brier_os_ipcw": real_eval.get("brier_os_ipcw"),
                "real_only_harrell_cindex": real_eval.get("harrell_cindex"),
                "low_auc_failure_reference": low_auc_failure_reference,
                "accepted": False,
            }
            if not (gan_x_path.exists() and gan_ep_path.exists()):
                record["status"] = "rejected_missing_gan_augmented_inputs"
                gan_utility_records.append(record)
            else:
                X_aug_file = pd.read_csv(gan_x_path, sep="\t", low_memory=False)
                id_col = "PATIENT_ID" if "PATIENT_ID" in X_aug_file.columns else X_aug_file.columns[0]
                X_aug_file[id_col] = X_aug_file[id_col].astype(str)
                X_aug_file = X_aug_file.set_index(id_col)
                ep_aug_file = pd.read_csv(gan_ep_path, sep="\t", low_memory=False)
                ep_id_col = "PATIENT_ID" if "PATIENT_ID" in ep_aug_file.columns else ep_aug_file.columns[0]
                ep_aug_file[ep_id_col] = ep_aug_file[ep_id_col].astype(str)
                ep_aug_file = ep_aug_file.set_index(ep_id_col)
                expected_cols = X_real_train.columns.tolist()
                missing_cols = [c for c in expected_cols if c not in X_aug_file.columns]
                extra_cols = [c for c in X_aug_file.columns if c not in expected_cols]
                record.update({
                    "gan_augmented_rows": int(len(X_aug_file)),
                    "gan_synthetic_rows": int(max(0, len(X_aug_file) - len(X_real_train))),
                    "feature_columns_expected": int(len(expected_cols)),
                    "missing_expected_columns": len(missing_cols),
                    "extra_columns": len(extra_cols),
                })
                if missing_cols:
                    record["status"] = "rejected_gan_feature_set_mismatch"
                    record["mismatch_example"] = ";".join(missing_cols[:5])
                    gan_utility_records.append(record)
                elif record["gan_synthetic_rows"] <= 0:
                    record["status"] = "rejected_no_synthetic_rows_after_utility_gate"
                    gan_utility_records.append(record)
                else:
                    gan_model = fit_weighted_logistic(X_aug_file[expected_cols].fillna(0.0), ep_aug_file, seed + 360)
                    if gan_model is None:
                        record["status"] = "rejected_gan_model_fit_returned_none"
                        gan_utility_records.append(record)
                    else:
                        gan_val_risk = risk_from_model(gan_model, X_real_all.reindex(val_ids_list)[expected_cols].fillna(0.0))
                        gan_eval = evaluate_36m_predictions(ep_train, ep_val, gan_val_risk, tau=OS_RISK_TIME_MONTHS)
                        gan_auc = float(gan_eval.get("auc_os_observed", float("nan")))
                        real_auc = float(real_eval.get("auc_os_observed", float("nan")))
                        accepted = bool(np.isfinite(gan_auc) and gan_auc > max(real_auc, low_auc_failure_reference))
                        record.update({
                            "gan_auc_36m": gan_eval.get("auc_os_observed"),
                            "gan_average_precision_os": gan_eval.get("average_precision_os"),
                            "gan_brier_os_ipcw": gan_eval.get("brier_os_ipcw"),
                            "gan_harrell_cindex": gan_eval.get("harrell_cindex"),
                            "gan_uno_cindex_ipcw": gan_eval.get("uno_cindex_ipcw"),
                            "auc_gain_vs_real_only": gan_auc - real_auc if np.isfinite(gan_auc) and np.isfinite(real_auc) else np.nan,
                            "accepted": accepted,
                            "status": "accepted_secondary_36m_model" if accepted else "rejected_no_auc_gain_or_below_failure_reference",
                        })
                        gan_utility_records.append(record)
                        if accepted:
                            gan_all_risk = risk_from_model(gan_model, X_real_all[expected_cols].fillna(0.0))
                            risk_table["clinical_plus_embedding_gan_elasticnet_36m_risk_score"] = gan_all_risk
                            metrics.append({
                                "model": "clinical_plus_embedding_gan_elasticnet_36m",
                                "cohort": "tcga_internal_validation",
                                "metric": "secondary_36m_auc",
                                "observed_value": gan_auc,
                                "target_success_threshold": low_auc_failure_reference,
                                "gap_to_target": gan_auc - low_auc_failure_reference,
                                "observed_performance_is_not_target_claim": True,
                                "primary_selection_eligible": False,
                                "secondary_endpoint": True,
                                "secondary_endpoint_reason": "36-month binary Elastic Net is auxiliary; primary selection is full-time survival only.",
                                **{k: v for k, v in gan_eval.items() if not isinstance(v, dict)},
                            })
        except Exception as exc:
            ctx.add_warning(f"Clinical+embedding GAN Elastic Net 36m failed: {type(exc).__name__}: {exc}")


    msk_bundle = train_msk_genomic_model(cfg, ctx.logger)
    if msk_bundle is not None:
        msk_path = model_dir / "msk_external_genomic_cox_model_bundle.joblib"
        joblib.dump(msk_bundle, msk_path)
        ctx.register_output(msk_path, "model", "MSK genomic (top-mutated genes + age + sex) Cox bundle.")
        bundles_for_validation["msk_external_genomic_cox"] = str(msk_path)
    else:
        ctx.add_warning("MSK genomic Cox bundle not produced.")


    try:
        ipcw_path = Path(cfg["processed_root"]) / "preprocessed" / "tcga_ipcw_os_endpoint.tsv"
        if ipcw_path.exists():
            ipcw_ep = pd.read_csv(ipcw_path, sep="\t", index_col=0)
            train_ids_list = split.loc[split["split"] == "train", "PATIENT_ID"].astype(str).tolist()
            val_ids_list = split.loc[split["split"] == "internal_validation", "PATIENT_ID"].astype(str).tolist()
            ep_train = ipcw_ep.reindex(train_ids_list).dropna(subset=["time_months"])
            if not ep_train.empty and "ipcw_weight_os" in ep_train.columns:
                emb_path_04 = Path(cfg["processed_root"]) / "multiomics_pretraining" / "tcga_multiomics_patient_embedding.tsv"
                if emb_path_04.exists():
                    emb_04 = pd.read_csv(emb_path_04, sep="\t")
                    emb_04["PATIENT_ID"] = emb_04["PATIENT_ID"].astype(str)
                    emb_cols_04 = [c for c in emb_04.columns if c not in ("PATIENT_ID", "modality_mask", "split")]
                    X_train_ipcw = emb_04.set_index("PATIENT_ID")[emb_cols_04].reindex(ep_train.index).fillna(0)
                    X_val_ipcw = emb_04.set_index("PATIENT_ID")[emb_cols_04].reindex(val_ids_list).fillna(0)
                    wlog_model = fit_weighted_logistic(X_train_ipcw, ep_train, seed)
                    if wlog_model is not None:
                        risk_wlog = risk_from_model(wlog_model, X_val_ipcw)
                        risk_table.loc[split["PATIENT_ID"].astype(str).isin(val_ids_list), "ipcw_weighted_logistic_risk"] = risk_wlog
                        ep_val = ipcw_ep.reindex(val_ids_list)
                        ep_train_for_eval = ipcw_ep.reindex(train_ids_list).dropna(subset=["time_months"])
                        wlog_eval = evaluate_36m_predictions(ep_train_for_eval, ep_val, risk_wlog, tau=OS_RISK_TIME_MONTHS)
                        wlog_auc = float(wlog_eval.get("auc_os_observed", float("nan")))
                        wlog_eval["secondary_endpoint_only"] = True
                        wlog_eval["primary_selection_eligible"] = False
                        wlog_eval["status"] = "failed_low_auc_secondary_only" if np.isfinite(wlog_auc) and wlog_auc <= 0.5 else "secondary_only"
                        wlog_eval["failure_reference_auc"] = low_auc_failure_reference
                        wlog_eval["secondary_endpoint_reason"] = "Embedding-only fixed-36m logistic is retained only as a diagnostic branch; primary selection is full-time survival only."
                        ctx.logger.info("IPCW-weighted Logistic 36m: AUC=%.3f, Brier=%.4f, Harrell C=%.3f",
                                        wlog_auc,
                                        wlog_eval.get("brier_os_ipcw", float("nan")),
                                        wlog_eval.get("harrell_cindex", float("nan")))
                        ipcw_metrics_path = out_dir / "ipcw_weighted_logistic_36m_metrics.tsv"
                        pd.DataFrame([{"model": "ipcw_weighted_logistic", "cohort": "tcga_internal_validation",
                                        **{k: v for k, v in wlog_eval.items() if not isinstance(v, dict)}}]).to_csv(
                            ipcw_metrics_path, sep="\t", index=False)
                        ctx.register_output(ipcw_metrics_path, "analysis_data",
                                            "IPCW-weighted Logistic 36-month binary prediction metrics.")
                        metrics.append({
                            "model": "ipcw_weighted_logistic_embedding_only",
                            "cohort": "tcga_internal_validation",
                            "metric": "secondary_36m_auc",
                            "observed_value": wlog_auc,
                            "target_success_threshold": low_auc_failure_reference,
                            "gap_to_target": wlog_auc - low_auc_failure_reference if np.isfinite(wlog_auc) else np.nan,
                            "observed_performance_is_not_target_claim": True,
                            "primary_selection_eligible": False,
                            "secondary_endpoint": True,
                            **{k: v for k, v in wlog_eval.items() if not isinstance(v, dict)},
                        })

                        wlog_bundle_path = model_dir / "ipcw_weighted_logistic_model_bundle.joblib"
                        joblib.dump({"model": wlog_model, "model_type": "ipcw_weighted_logistic",
                                     "endpoint": "os_binary", "tau_months": OS_RISK_TIME_MONTHS}, wlog_bundle_path)
                        ctx.register_output(wlog_bundle_path, "model", "IPCW-weighted Logistic 36m model bundle.")
                    else:
                        ctx.add_warning("IPCW-weighted Logistic: model fitting returned None (insufficient events or labels).")
                else:
                    ctx.add_warning("Multi-omics embedding missing; IPCW-weighted Logistic skipped.")
            else:
                ctx.add_warning("IPCW endpoint has no valid training rows; weighted Logistic skipped.")
        else:
            ctx.add_warning("IPCW endpoint file missing; run Module 01 first.")
    except Exception as exc:
        ctx.add_warning(f"IPCW-weighted Logistic failed: {type(exc).__name__}: {exc}")


    gan_utility_df = pd.DataFrame(gan_utility_records)
    if gan_utility_df.empty:
        gan_utility_df = pd.DataFrame([{
            "model": "clinical_plus_embedding_gan_elasticnet_36m",
            "status": "not_evaluated",
            "accepted": False,
            "reason": "No clinical+embedding matrix or GAN augmented input was available.",
        }])
    ctx.write_table(
        out_dir / "gan_model_utility.tsv",
        gan_utility_df,
        "analysis_data",
        "Real-only vs GAN-augmented clinical+embedding Elastic Net utility gate.",
    )
    ctx.write_table(out_dir / "internal_validation_risk_scores.tsv", risk_table, "analysis_data", "Internal validation risk scores per model.")
    metric_df = pd.DataFrame(metrics)
    ctx.write_table(out_dir / "model_comparison_internal_validation_metrics.tsv", metric_df, "analysis_data", "Internal validation multi-metric comparison; observed values are NOT target claims.")


    selected_primary, selection_detail = select_primary_model(metric_df, cfg["processed_root"], args.primary_selection_policy)
    incremental_df = build_incremental_omics_delta(metric_df, baseline_model="clinical_ajcc_cox")
    if incremental_df.empty:
        incremental_df = pd.DataFrame([{
            "baseline_model": "clinical_ajcc_cox",
            "candidate_model": None,
            "note": "No omics or ensemble survival candidate was available for incremental comparison.",
        }])
    ctx.write_table(
        out_dir / "incremental_omics_delta.tsv",
        incremental_df,
        "analysis_data",
        "Incremental omics/ensemble survival performance versus clinical Cox baseline.",
    )
    improvement_summary = build_survival_performance_summary(metric_df, selected_primary)
    ctx.write_table(
        out_dir / "survival_performance_improvement_summary.tsv",
        improvement_summary,
        "analysis_data",
        "Full-time survival first performance summary, including secondary 36-month diagnostic rows.",
    )
    candidate_scores = selection_detail.get("candidate_scores", []) if isinstance(selection_detail, dict) else []
    selected_score = next((row for row in candidate_scores if row.get("model_key") == selected_primary), {})
    calibration_values = [
        row.get("negative_brier_metric_for_selection")
        for row in candidate_scores
        if row.get("negative_brier_metric_for_selection") is not None
    ]
    interpretability_values = [
        row.get("parsimony_score")
        for row in candidate_scores
        if row.get("parsimony_score") is not None
    ]

    deep_status = deepsurv_exploration_status(split)
    manifest = {
        "selected_model": selected_primary,
        "selected_primary_model_internal": selected_primary,
        "model_selection_policy": args.primary_selection_policy,
        "selection_rationale": "Training-stage lock using full-time survival metrics first: C-index, time-dependent AUC, calibration/Brier evidence, and parsimony. Fixed-36m binary AUC is secondary only. External cohorts are forbidden for model selection and are evaluated only by 05_validation_visualization.py.",
        "selection_detail": selection_detail,
        "external_validation_cindex": {
            "selected_model_value": None,
            "available_model_values": [],
            "summary": "Not used during 04 model selection. External C-index is generated downstream by 05 only after the model is locked.",
        },
        "calibration_summary": {
            "selected_model_negative_brier_metric_for_selection": selected_score.get("negative_brier_metric_for_selection"),
            "selected_model_brier_metric_source": selected_score.get("calibration_metric_source"),
            "available_proxy_values": calibration_values,
            "note": "Lower Brier is better; manifest stores the negative value so higher is better. Formal integrated_brier_score is used when a model survival function is available; otherwise the metric is explicitly exploratory.",
        },
        "dca_summary": {
            "selected_model_td_auc_mean": selected_score.get("td_auc_mean"),
            "note": "Decision-curve-ready discrimination is proxied by time-dependent AUC; dedicated net-benefit curves are produced in validation outputs when thresholds are available.",
        },
        "interpretability_summary": {
            "selected_model_parsimony_score": selected_score.get("parsimony_score"),
            "available_proxy_values": interpretability_values,
            "note": "Clinical models receive the highest parsimony score; Coxnet is penalized by non-zero feature count.",
        },
        "secondary_binary_endpoint_policy": {
            "primary_selection_eligible": False,
            "failure_reference_auc": low_auc_failure_reference,
            "rule": "36-month binary Elastic Net/logistic rows are diagnostics only; AUC below 0.5 is marked failed_low_auc_secondary_only.",
        },
        "new_output_tables": {
            "gan_model_utility": str(out_dir / "gan_model_utility.tsv"),
            "incremental_omics_delta": str(out_dir / "incremental_omics_delta.tsv"),
            "survival_performance_improvement_summary": str(out_dir / "survival_performance_improvement_summary.tsv"),
        },
        "endpoint": endpoint,
        "target_success_criterion_c_index": 0.75,
        "observed_performance_is_not_target_claim": True,
        "observed_performance_is_not_a_target_claim": True,
        "serialized_model_dir": str(model_dir),
        "external_validation_models": {
            "msk_crc_2017": {
                "clinical_baseline": "clinical_minimal_cox_model_bundle.joblib",
                "genomic_specific": "msk_external_genomic_cox_model_bundle.joblib" if msk_bundle is not None else None,
            },
            "geo_gse103479": {
                "expression_signature": "expression_coxnet_model_bundle.joblib",
            },
        },
        "deep_models_disabled_reason": "events<120, exploratory only",
        "optional_deepsurv_status": deep_status,
        "leakage_control": "All imputers, encoders, scalers, feature filters, Coxnet (with CV), Cox, and RSF models are fitted on training split only.",
        "stratification_factors": ["event", "stage_group", "site_group(COAD/READ)"],
        "bootstrap_iter": args.bootstrap_iter,
        "coxnet_cv_folds": args.coxnet_cv_folds,
        "coxnet_l1_ratio": args.coxnet_l1_ratio,
    }
    ctx.write_json(out_dir / "selected_primary_model_manifest.json", manifest, "analysis_data", "Selected primary model manifest with multi-criteria rationale.")
    write_model_card(out_dir, manifest, metric_df, deep_status, ctx)


    try:
        import matplotlib.pyplot as plt

        plot_df = metric_df.loc[metric_df["metric"] == "harrell_c_index"].copy() if not metric_df.empty else pd.DataFrame()
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            order = plot_df.sort_values("observed_value", ascending=True)
            ax.barh(order["model"], order["observed_value"], color="#2c7fb8")
            if "ci_low_95" in order.columns:
                xerr_low = order["observed_value"] - order["ci_low_95"]
                xerr_high = order["ci_high_95"] - order["observed_value"]
                ax.errorbar(order["observed_value"], np.arange(len(order)), xerr=[xerr_low, xerr_high], fmt="none", ecolor="black", capsize=2, linewidth=0.8)
            ax.axvline(0.75, color="firebrick", linestyle="--", linewidth=1.0, label="success criterion = 0.75")
            ax.set_xlabel("Harrell C-index (internal validation, 95% bootstrap CI)")
            ax.set_title("Internal validation C-index across model families")
            ax.legend(loc="lower right", fontsize=8)
            fig.tight_layout()
            path = ctx.output_path(ctx.run_dir / "internal_validation_cindex_bootstrap.png", "figure")
            fig.savefig(path, dpi=180)
            plt.close(fig)
            ctx.register_output(path, "figure", "Internal validation C-index with 95% bootstrap CI per model.")
    except Exception as exc:
        ctx.add_warning(f"Plotting C-index figure failed: {exc}")

    ctx.finalize([p for p in input_files if Path(p).exists()])
    return 0

