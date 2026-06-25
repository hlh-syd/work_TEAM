from __future__ import annotations

import argparse
import json
import joblib
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from _pipeline_core import *  # noqa: F401,F403
from _causal_infer import *  # noqa: F401,F403
from _gan_augment import *   # noqa: F401,F403

CLINICAL_PROGNOSTIC_COLS = [
    "AGE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "PATH_T_STAGE",
    "PATH_N_STAGE", "PATH_M_STAGE", "SUBTYPE", "WEIGHT",
]

EXCLUDED_EXTERNAL_COHORTS = {"htan", "cptac", "geo_gse103479"}


def normalize_external_clinical(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """外部队列临床列名映射与清洗 (MSK→AGE映射, unknown SEX过滤)."""
    out = df.copy()
    if cohort == "msk":
        out["PATIENT_ID"] = out["PATIENT_ID"].astype(str)
        if "AGE" not in out.columns and "AGE_AT_DIAGNOSIS" in out.columns:
            out["AGE"] = pd.to_numeric(out["AGE_AT_DIAGNOSIS"], errors="coerce")
        out = out.loc[out["OS_TIME_MONTHS"].notna() & (out["OS_TIME_MONTHS"] > 0)]
        if "SEX" in out.columns:
            sex_valid = out["SEX"].astype(str).str.strip().str.lower().isin(
                {"male", "female", "m", "f", "0", "1"}
            )
            out = out.loc[sex_valid]
        return out.copy()
    if cohort in ("geo_gse39582",):
        out["PATIENT_ID"] = out["PATIENT_ID"].astype(str)
        return out.loc[out["OS_TIME_MONTHS"].notna() & (out["OS_TIME_MONTHS"] > 0)].copy()
    if cohort == "geo_gse17538":
        out["PATIENT_ID"] = out["PATIENT_ID"].astype(str)
        return out.loc[out["OS_TIME_MONTHS"].notna() & (out["OS_TIME_MONTHS"] > 0)].copy()
    return out


def encode_clinical_features(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()
    if "SEX" in out.columns:
        sex_map = {"Male": 1, "Female": 0, "MALE": 1, "FEMALE": 0, "male": 1, "female": 0}
        out["SEX"] = out["SEX"].map(sex_map).fillna(pd.to_numeric(out["SEX"], errors="coerce"))
    if "AJCC_PATHOLOGIC_TUMOR_STAGE" in out.columns:
        # Ordinal integer encoding: preserves stage ordering without
        # implying equal-interval distances (avoids fractional stage values
        # that have no clinical meaning, e.g. STAGE IA=1.1 → 2).
        stage_map = {
            "STAGE I": 1, "STAGE IA": 2, "STAGE IB": 3,
            "STAGE II": 4, "STAGE IIA": 5, "STAGE IIB": 6, "STAGE IIC": 7,
            "STAGE III": 8, "STAGE IIIA": 9, "STAGE IIIB": 10, "STAGE IIIC": 11,
            "STAGE IV": 12, "STAGE IVA": 13, "STAGE IVB": 14, "STAGE IVC": 15,
        }
        out["AJCC_PATHOLOGIC_TUMOR_STAGE"] = (
            out["AJCC_PATHOLOGIC_TUMOR_STAGE"].astype(str).str.strip().str.upper().map(stage_map)
        )
    for col in ["PATH_T_STAGE", "PATH_N_STAGE", "PATH_M_STAGE"]:
        if col in out.columns:
            out[col] = pd.to_numeric(
                out[col].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce"
            )
    if "SUBTYPE" in out.columns:
        subtype_dummies = pd.get_dummies(out["SUBTYPE"], prefix="SUBTYPE", drop_first=True, dtype=float)
        out = pd.concat([out.drop(columns=["SUBTYPE"]), subtype_dummies], axis=1)
    saved_pids = out["PATIENT_ID"].copy() if "PATIENT_ID" in out.columns else None
    skip_cols = {"PATIENT_ID", "COHORT", "OS_STATUS", "OS_EVENT", "OS_TIME_MONTHS",
                 "OS_MONTHS", "split", "time_months", "event"}
    numeric_cols = []
    for c in out.columns:
        if c in skip_cols:
            continue
        try:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            numeric_cols.append(c)
        except Exception:
            continue
    keep_cols = [c for c in numeric_cols if out[c].notna().sum() > len(out) * 0.3]
    result = out[keep_cols].copy()
    if saved_pids is not None:
        result.insert(0, "PATIENT_ID", saved_pids)
    return result


def fit_clinical_cox(X_clin: pd.DataFrame, endpoint: pd.DataFrame) -> Optional[Any]:

    if CoxPHFitter is None:
        return None
    df = X_clin.copy()
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    var_cols = [c for c in df.columns if df[c].std() > 0]
    df = df[var_cols].copy()
    df["time_months"] = endpoint["time_months"].values
    df["event"] = endpoint["event"].values
    df = df.loc[:, ~df.columns.duplicated()]
    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(df, duration_col="time_months", event_col="event")
        return cph
    except Exception:
        try:
            cph = CoxPHFitter(penalizer=1.0)
            cph.fit(df, duration_col="time_months", event_col="event")
            return cph
        except Exception as e2:
            print(f"  [WARN] Clinical Cox failed: {e2}")
            return None


def fit_coxnet_mt(X: pd.DataFrame, endpoint: pd.DataFrame, random_seed: int = RANDOM_SEED) -> Optional[Any]:

    try:
        from sksurv.linear_model import CoxnetSurvivalAnalysis as _Coxnet
        y = make_surv_array(endpoint)
        if y is None:
            return None
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        model = _Coxnet(n_alphas=50, alpha_min_ratio=0.01, l1_ratio=0.5, max_iter=10000)
        model.fit(X_scaled, y)
        model._scaler = scaler
        return model
    except Exception as e:
        print(f"  [WARN] Coxnet failed: {e}")
        return None


def fit_rsf_mt(X: pd.DataFrame, endpoint: pd.DataFrame, random_seed: int = RANDOM_SEED) -> Optional[Any]:

    try:
        from sksurv.ensemble import RandomSurvivalForest as _RSF
        y = make_surv_array(endpoint)
        if y is None:
            return None
        model = _RSF(n_estimators=200, min_samples_split=10, min_samples_leaf=5,
                     max_features="sqrt", random_state=random_seed, n_jobs=-1)
        model.fit(X, y)
        return model
    except Exception as e:
        print(f"  [WARN] RSF failed: {e}")
        return None


def fit_improved_logistic(
    X: pd.DataFrame, y: np.ndarray, w: np.ndarray,
    random_seed: int = RANDOM_SEED,
) -> Optional[LogisticRegression]:

    y_series = pd.Series(y)
    if y_series.nunique() < 2 or len(y_series) < 20:
        return None
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    model = LogisticRegression(
        C=1.0, penalty="l2", solver="lbfgs",
        max_iter=10000, random_state=random_seed,
    )
    model.fit(X_scaled, y, sample_weight=w)
    model._scaler = scaler
    return model


def evaluate_binary_predictions(
    y_true: np.ndarray, risk: np.ndarray, weights: np.ndarray, tau: float,
) -> Dict[str, float]:

    valid = np.isfinite(risk) & np.isfinite(weights) & (weights > 0)
    if valid.sum() < 5 or pd.Series(y_true[valid]).nunique() < 2:
        return {"auc": float("nan"), "brier": float("nan"), "ap": float("nan")}
    y_v = y_true[valid].astype(int)
    r_v = risk[valid]
    w_v = weights[valid]
    result: Dict[str, float] = {}
    try:
        result["auc"] = float(roc_auc_score(y_v, r_v, sample_weight=w_v))
    except Exception:
        result["auc"] = float("nan")
    try:
        result["brier"] = float(np.average((y_v - r_v) ** 2, weights=w_v))
    except Exception:
        result["brier"] = float("nan")
    try:
        result["ap"] = float(average_precision_score(y_v, r_v, sample_weight=w_v))
    except Exception:
        result["ap"] = float("nan")
    return result


def evaluate_survival_model(
    train_endpoint: pd.DataFrame,
    eval_endpoint: pd.DataFrame,
    risk_train: np.ndarray,
    risk_eval: np.ndarray,
    tau: Optional[float] = None,
) -> Dict[str, float]:

    result: Dict[str, float] = {
        "n_eval": len(eval_endpoint),
        "events_eval": int(eval_endpoint["event"].sum()),
    }
    try:
        from lifelines.utils import concordance_index
        times = eval_endpoint["time_months"].to_numpy(float)
        events = eval_endpoint["event"].to_numpy(int)
        result["harrell_c"] = float(concordance_index(times, -risk_eval, events))
    except Exception:
        result["harrell_c"] = float("nan")
    if tau is not None and concordance_index_ipcw is not None and Surv is not None:
        try:
            y_train = make_surv_array(train_endpoint)
            y_eval = make_surv_array(eval_endpoint)
            c_ipcw = concordance_index_ipcw(y_train, y_eval, risk_eval, tau=tau)[0]
            result["cindex_ipcw"] = float(c_ipcw)
        except Exception:
            result["cindex_ipcw"] = float("nan")
        try:
            y_train = make_surv_array(train_endpoint)
            y_eval = make_surv_array(eval_endpoint)
            auc_t, mean_auc = cumulative_dynamic_auc(y_train, y_eval, risk_eval, times=np.asarray([tau]))
            result["time_dep_auc"] = float(auc_t[0])
        except Exception:
            result["time_dep_auc"] = float("nan")
    else:
        result["cindex_ipcw"] = result.get("harrell_c", float("nan")) if tau is None else float("nan")
        result["time_dep_auc"] = float("nan")
    return result


def load_external_cohorts(preprocessed_root: Path, tau: float, audit: AuditLog) -> Dict[str, pd.DataFrame]:
    cohorts: Dict[str, pd.DataFrame] = {}
    for path in sorted(preprocessed_root.glob("*_os_clinical_endpoint_qc.tsv")):
        if path.name.startswith("tcga_"):
            continue
        name = path.name.replace("_os_clinical_endpoint_qc.tsv", "")
        if name in EXCLUDED_EXTERNAL_COHORTS:
            audit.add("external_load", "SKIP", f"Excluded cohort: {name}")
            continue
        try:
            df = read_tsv(path)
            if "PATIENT_ID" not in df.columns:
                continue
            # 临床列映射 + MSK unknown SEX 过滤
            df = normalize_external_clinical(df, name)
            endpoint = add_fixed_os_endpoint(df, tau=tau).set_index("PATIENT_ID", drop=False)
            endpoint = compute_ipcw_weights(endpoint, tau)
            endpoint = compute_pseudo_observations(endpoint, tau)
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
    train_feature_medians: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    # 先做临床列映射
    endpoint = normalize_external_clinical(endpoint, cohort_name)
    clinical_X, missing_clinical = transform_clinical_frame(clinical_pipe, endpoint, clinical_names)
    blocks = [clinical_X]
    compatibility = {"clinical": "ok" if not missing_clinical else f"imputed_missing:{len(missing_clinical)}"}

    # 仅保留 GEO 芯片表达数据 (已删除 HTAN/CPTAC/GSE103479)
    expr_candidates = {
        "geo_gse17538": raw_root / "external" / "geo_gse17538_geneMatrix.txt",
        "geo_gse39582": raw_root / "external" / "geo_gse39582_geneMatrix.txt",
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
                # ★ RINT per-cohort: 芯片数据先做 rank-based INT 再 StandardScaler transform
                tmp = tmp.apply(rank_inverse_normal, axis=0)
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

    # 仅保留 MSK 突变数据 (已删除 HTAN/CPTAC)
    mutation_candidates = {
        "msk": raw_root / "external" / "msk_crc_2017_mutation_gene_level_matrix.tsv",
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

    # ★ 缺失特征用训练中位数填充而非 0
    X = pd.concat(blocks, axis=1)
    if train_feature_medians is not None:
        med = train_feature_medians.reindex(X.columns)
        X = X.fillna(med)
    X = X.fillna(0.0)
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
    content = """# Environment for integrated_pipeline.py
python==3.12.4
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
    run_dir = ensure_dir(cfg.output_root / f"pipeline_{run_dir_timestamp()}")
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
    causal_dir = cfg.raw_data_root / "causal"
    tcga_clin_path = find_existing(
        [
            preprocessed / "tcga_os_clinical_endpoint_qc.tsv",
            preprocessed / "tcga_coadread_clinical_os_qc.tsv",
        ]
    )
    if tcga_clin_path is None:
        raise FileNotFoundError("TCGA clinical endpoint table was not found in rawData/preprocessed.")
    tcga = add_fixed_os_endpoint(read_tsv(tcga_clin_path), tau=cfg.tau_months).set_index("PATIENT_ID", drop=False)
    tcga = compute_ipcw_weights(tcga, cfg.tau_months)
    tcga = compute_pseudo_observations(tcga, cfg.tau_months)
    tcga.to_csv(tables_dir / "tcga_os_endpoint_ipcw_pseudo.tsv", sep="\t", index=False)
    audit.add(
        "endpoint",
        "OK",
        "Constructed 36-month endpoint; early censored cases retain missing fixed-time label.",
        n=int(len(tcga)),
        early_censored=int(tcga["early_censored_before_os"].sum()),
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
        observed_label = tcga["os_death_observed"].astype(int).astype(str) + "|" + tcga["event"].astype(str)
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

        # ★ PRIMARY MODELS: native survival analysis (Cox PH / RSF / Coxnet)
        cox = fit_lifelines_cox(Xtr, train_endpoint, penalizer=0.1)
        models_to_fit[f"{feature_set_name}__cox_ph_penalized"] = ("cox_lifelines", cox)

        if len(cols) <= 2000:
            coxnet = fit_sksurv_coxnet(Xtr, train_endpoint)
            models_to_fit[f"{feature_set_name}__coxnet"] = ("sksurv", coxnet)

        rsf = fit_sksurv_rsf(Xtr, train_endpoint, cfg.random_seed)
        models_to_fit[f"{feature_set_name}__rsf"] = ("sksurv", rsf)

        # GAN-augmented Cox PH (survival model, not logistic)
        if gan_report.get("status") == "fitted" and set(cols).issubset(set(gan_cols)):
            X_aug = X_train_gan_aug[cols]
            ep_aug = endpoint_gan_aug.reindex(X_aug.index)
            gan_cox = fit_lifelines_cox(X_aug, ep_aug, penalizer=0.1)
            models_to_fit[f"{feature_set_name}__train_only_gan_cox_ph"] = ("cox_lifelines", gan_cox)

        for model_name, (model_kind, model) in models_to_fit.items():
            if model is None:
                audit.add("model_fit", "WARN", f"{model_name} skipped or failed.")
                continue
            if model_kind == "cox_lifelines":
                risk_train = predict_lifelines_risk(model, Xtr, cfg.tau_months)
                risk_val = predict_lifelines_risk(model, Xva, cfg.tau_months)
            else:  # sksurv
                risk_train = predict_sksurv_risk(model, Xtr, cfg.tau_months)
                risk_val = predict_sksurv_risk(model, Xva, cfg.tau_months)
            threshold = choose_training_threshold(train_endpoint, risk_train)
            thresholds[model_name] = threshold
            # ★ PRIMARY evaluation: native survival metrics
            train_metrics = evaluate_survival_metrics(train_endpoint, train_endpoint, risk_train, cfg.tau_months)
            val_metrics = evaluate_survival_metrics(train_endpoint, val_endpoint, risk_val, cfg.tau_months)
            # Also compute supplementary binary metrics for backward comparison
            binary_val = evaluate_36m_predictions(train_endpoint, val_endpoint, risk_val, cfg.tau_months, threshold)
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
                    **{f"supplementary_binary_{k}": v for k, v in binary_val.items()},
                }
            )
            fitted_models[model_name] = {"model": model, "kind": model_kind, "features": cols, "threshold": threshold}
            risk_tables.append(
                pd.DataFrame(
                    {
                        "PATIENT_ID": train_ids,
                        "cohort_role": "TCGA_train_real",
                        "model_name": model_name,
                        "risk_36m": risk_train,
                    }
                )
            )
            risk_tables.append(
                pd.DataFrame(
                    {
                        "PATIENT_ID": val_ids,
                        "cohort_role": "TCGA_internal_validation_real",
                        "model_name": model_name,
                        "risk_36m": risk_val,
                    }
                )
            )
            audit.add("model_fit", "OK", f"{model_name} trained and evaluated on real internal validation.")

    if not model_rows:
        raise RuntimeError("No model could be fitted. Check dependencies and endpoint/event availability.")

    model_metrics = pd.DataFrame(model_rows)
    model_metrics.to_csv(tables_dir / "internal_validation_model_comparison.tsv", sep="\t", index=False)
    pd.concat(risk_tables, ignore_index=True).to_csv(tables_dir / "tcga_train_internal_validation_risk_scores.tsv", sep="\t", index=False)

    # ★ Model selection based on native survival metrics
    sort_cols = ["internal_validation_uno_cindex_ipcw", "internal_validation_time_dependent_auc", "internal_validation_harrell_cindex"]
    ranked = model_metrics.copy()
    # Fill missing columns with NaN (some models may not have all metrics)
    for c in sort_cols:
        if c not in ranked.columns:
            ranked[c] = float("nan")
    ranked["_cindex_rank"] = ranked[sort_cols[0]].rank(ascending=False, na_option="bottom")
    ranked["_auc_rank"] = ranked[sort_cols[1]].rank(ascending=False, na_option="bottom")
    ranked["_harrell_rank"] = ranked[sort_cols[2]].rank(ascending=False, na_option="bottom")
    ranked["_selection_score"] = ranked["_cindex_rank"] + ranked["_auc_rank"] + ranked["_harrell_rank"]
    best_row = ranked.sort_values(["_selection_score", "_cindex_rank", "_harrell_rank"]).iloc[0].to_dict()
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
    train_feature_medians = X_train_real.median()  # ★ 训练中位数用于缺失填充
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
            train_feature_medians=train_feature_medians,
        )
        needed = selected_bundle["features"]
        missing = [c for c in needed if c not in X_ext.columns]
        if missing:

            if all(c in clinical_names for c in needed):
                pass
            else:
                audit.add("external_validation", "SKIP", f"{cohort_name}: missing selected model features.", missing_count=len(missing))
                continue
        # ★ 缺失特征优先用训练中位数填充
        X_eval = X_ext.reindex(columns=needed)
        med = train_feature_medians.reindex(needed)
        X_eval = X_eval.fillna(med).fillna(0.0)
        kind = selected_bundle["kind"]
        model = selected_bundle["model"]
        if kind == "binary":
            risk = risk_from_model(model, X_eval)
        elif kind == "cox_lifelines":
            risk = predict_lifelines_risk(model, X_eval, cfg.tau_months)
        else:
            risk = predict_sksurv_risk(model, X_eval, cfg.tau_months)
        metrics = evaluate_36m_predictions(train_endpoint, ext_endpoint, risk, cfg.tau_months, selected_bundle["threshold"])
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
                    "risk_36m": risk,
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
            candidate_metrics = evaluate_36m_predictions(
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
        "selected_model_external_mean_auc_os": (
            float(external_metrics["time_dependent_auc_os"].dropna().mean())
            if not external_metrics.empty and "time_dependent_auc_os" in external_metrics
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
            "Early censoring before event window is not labelled as OS event.",
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
    default_raw = script_path.parents[2] / "DATA"
    parser = argparse.ArgumentParser(description="CRC OS risk prediction pipeline.")
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--raw-data-root", type=Path, default=default_raw)
    parser.add_argument("--plan-path", type=Path, default=project_root / "plan" / "headplan.txt")
    parser.add_argument("--output-root", type=Path, default=project_root / "results")
    parser.add_argument("--max-expression-features", type=int, default=300)
    parser.add_argument("--causal-prescreen-features", type=int, default=120)
    parser.add_argument("--max-causal-features", type=int, default=50)
    parser.add_argument("--gan-epochs", type=int, default=60)
    parser.add_argument("--gan-aug-ratio", type=float, default=0.5)
    parser.add_argument("--gan-aug-ratios", type=str, default="0.25,0.5,1.0")
    parser.add_argument("--gan-sampling-strategies", type=str, default="balanced_event,risk_stratified,event_only,overall")
    parser.add_argument("--no-gan", action="store_true")
    
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

        strict_dependencies=args.strict_dependencies,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
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




def build_parser_step03_5() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GAN data augmentation for CRC survival embeddings.")
    add_common_args(parser)
    parser.add_argument("--run-gan", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable GAN augmentation (default: True).")
    parser.add_argument("--gan-epochs", type=int, default=300, help="GAN training epochs.")
    parser.add_argument("--gan-batch-size", type=int, default=32, help="GAN batch size.")
    parser.add_argument("--gan-aug-ratio", type=float, default=0.5,
                        help="Ratio of synthetic to real training samples.")
    parser.add_argument("--gan-aug-ratios", type=str, default="0.25,0.5,1.0",
                        help="Candidate synthetic/real ratios for train-only GAN selection.")
    parser.add_argument("--gan-sampling-strategies", type=str, default="balanced_event,risk_stratified,event_only,overall",
                        help="Candidate sampling strategies for train-only GAN selection.")
    parser.add_argument("--gan-feature-set", choices=["clinical_plus_embedding", "embedding_only"],
                        default="clinical_plus_embedding",
                        help="Feature set used by GAN augmentation.")
    parser.add_argument("--gan-latent-dim", type=int, default=64, help="GAN latent dimension.")
    parser.add_argument("--gan-qc-min-good", type=int, default=4,
                        help="Minimum QC dimensions to pass (out of 5).")
    parser.add_argument("--gan-use-feature-space", action=argparse.BooleanOptionalAction, default=False,
                        help="Use autoencoder latent space for GAN (default: False).")
    return parser


def load_tcga_clinical(cfg: dict, endpoint: str) -> pd.DataFrame:
    """Load TCGA clinical table for the given endpoint."""
    clinical_path = cfg["cohorts"]["tcga"]["clinical_patient"]
    return read_cbio_table(str(clinical_path))


def attach_raw_clinical_columns(clinical: pd.DataFrame, clinical_path: str) -> pd.DataFrame:
    """Attach raw clinical columns (AGE, SEX, stage, subtype) to clinical table."""
    raw = read_cbio_table(str(clinical_path))
    if "PATIENT_ID" not in raw.columns:
        return clinical
    keep_cols = ["PATIENT_ID"]
    for col in ["AGE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "SUBTYPE", "CANCER_TYPE_ACRONYM"]:
        if col in raw.columns:
            keep_cols.append(col)
    raw_subset = raw[keep_cols].drop_duplicates("PATIENT_ID")
    return clinical.merge(raw_subset, on="PATIENT_ID", how="left")


def get_or_create_split(cfg: dict, endpoint: str, test_size: float) -> pd.DataFrame:
    """Load existing train/validation split or create a new one."""
    split_path = Path(cfg["processed_root"]) / "survival_models" / "tcga_train_internal_validation_split.tsv"
    if split_path.exists():
        return pd.read_csv(split_path, sep="\t")
    # Create new split if not exists
    clinical_path = cfg["cohorts"]["tcga"]["clinical_patient"]
    clinical = read_cbio_table(str(clinical_path))
    if "PATIENT_ID" not in clinical.columns:
        raise ValueError("clinical table missing PATIENT_ID")
    patient_ids = clinical["PATIENT_ID"].astype(str).unique()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(patient_ids)
    n_val = int(len(patient_ids) * test_size)
    val_ids = set(patient_ids[:n_val])
    split_df = pd.DataFrame({
        "PATIENT_ID": patient_ids,
        "split": ["internal_validation" if p in val_ids else "train" for p in patient_ids],
    })
    return split_df


def fit_clinical_transformer(
    clinical: pd.DataFrame,
    split: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, Any]:
    """Encode clinical features with median imputation and standard scaling, fit on train split."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    clinical_work = clinical.copy()
    clinical_work["PATIENT_ID"] = clinical_work["PATIENT_ID"].astype(str)
    split_work = split.copy()
    split_work["PATIENT_ID"] = split_work["PATIENT_ID"].astype(str)

    train_ids = set(split_work.loc[split_work["split"] == "train", "PATIENT_ID"])
    train_mask = clinical_work["PATIENT_ID"].isin(train_ids)

    available_features = [f for f in features if f in clinical_work.columns]
    if not available_features:
        # Return empty DataFrame with patient index
        result = pd.DataFrame(index=clinical_work["PATIENT_ID"].values)
        result.index.name = "PATIENT_ID"
        return result, None

    X = clinical_work[available_features].copy()
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype).startswith("category"):
            X[col] = X[col].map({"Male": 1, "Female": 0, "M": 1, "F": 0}).fillna(0)
        X[col] = pd.to_numeric(X[col], errors="coerce")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = X.loc[train_mask].values
    X_all = X.values

    X_train_imp = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imp)

    X_all_imp = imputer.transform(X_all)
    X_all_scaled = scaler.transform(X_all_imp)

    result = pd.DataFrame(X_all_scaled, columns=available_features, index=clinical_work["PATIENT_ID"].values)
    result.index.name = "PATIENT_ID"
    return result, (imputer, scaler)


def main_step03_5() -> int:


    parser = build_parser_step03_5()
    args = parser.parse_args()
    ctx = initialize_run(__file__, args)
    cfg = ctx.cfg

    if not args.run_gan:
        ctx.logger.info("GAN augmentation disabled by --no-run-gan.")
        ctx.add_warning("GAN augmentation disabled; downstream models will use real data only.")
        ctx.finalize([])
        return 0

    gan_cfg = GANPipelineConfig(
        tau_months=OS_RISK_TIME_MONTHS,
        random_seed=cfg.get("random_seed", RANDOM_SEED_GAN),
        gan_epochs=args.gan_epochs,
        gan_batch_size=args.gan_batch_size,
        gan_aug_ratio=args.gan_aug_ratio,
        gan_aug_ratio_candidates=tuple(float(x.strip()) for x in str(args.gan_aug_ratios).split(",") if x.strip()),
        gan_sampling_strategies=tuple(x.strip().lower() for x in str(args.gan_sampling_strategies).split(",") if x.strip()),
        gan_latent_dim=args.gan_latent_dim,
        gan_qc_min_good=args.gan_qc_min_good,
        gan_use_feature_space=args.gan_use_feature_space,
    )


    embedding_path = Path(cfg["processed_root"]) / "multiomics_pretraining" / "tcga_multiomics_patient_embedding.tsv"
    if not embedding_path.exists():
        ctx.add_warning(f"Embedding not found: {embedding_path}. Run Module 03 first.")
        ctx.finalize([])
        return 1

    emb_df = pd.read_csv(embedding_path, sep="\t", low_memory=False)
    emb_cols = [c for c in emb_df.columns if "_embedding_" in c or c.startswith("embedding_")]
    train_ids = emb_df.loc[emb_df["split"] == "train", "PATIENT_ID"].astype(str).tolist()


    endpoint_path = Path(cfg["processed_root"]) / "preprocessed" / "tcga_os_clinical_endpoint_qc.tsv"
    if not endpoint_path.exists():
        ctx.add_warning(f"Endpoint not found: {endpoint_path}. Run Module 01 first.")
        ctx.finalize([embedding_path])
        return 1

    endpoint_df = pd.read_csv(endpoint_path, sep="\t", low_memory=False)
    endpoint_df["PATIENT_ID"] = endpoint_df["PATIENT_ID"].astype(str)
    if "time_months" not in endpoint_df.columns and "OS_TIME_MONTHS" in endpoint_df.columns:
        endpoint_df["time_months"] = pd.to_numeric(endpoint_df["OS_TIME_MONTHS"], errors="coerce")
    if "event" not in endpoint_df.columns and "OS_EVENT" in endpoint_df.columns:
        endpoint_df["event"] = endpoint_df["OS_EVENT"].fillna(False).astype(int)


    if "ipcw_weight_os" not in endpoint_df.columns:

        ep_work = endpoint_df.copy()
        if "OS_TIME_MONTHS" in ep_work.columns and "OS_EVENT" in ep_work.columns:
            ep_work["time_months"] = pd.to_numeric(ep_work["OS_TIME_MONTHS"], errors="coerce")
            ep_work["event"] = ep_work["OS_EVENT"].fillna(False).astype(int)
            ep_work = add_fixed_os_endpoint(ep_work, tau=gan_cfg.tau_months)
            ep_work = compute_ipcw_weights(ep_work, tau=gan_cfg.tau_months)
            ep_work = compute_pseudo_observations(ep_work, tau=gan_cfg.tau_months)

            for col in ["os_death_observed", "os_death", "early_censored_before_os",
                        "ipcw_weight_os", "ipcw_label_available",
                        "pseudo_risk_os_raw", "pseudo_risk_os"]:
                if col in ep_work.columns:
                    endpoint_df = endpoint_df.merge(
                        ep_work[["PATIENT_ID", col]].drop_duplicates("PATIENT_ID"),
                        on="PATIENT_ID", how="left"
                    )


    emb_indexed = emb_df.set_index(emb_df["PATIENT_ID"].astype(str))[emb_cols]
    if args.gan_feature_set == "clinical_plus_embedding":
        clinical = load_tcga_clinical(cfg, args.endpoint)
        clinical = attach_raw_clinical_columns(clinical, cfg["cohorts"]["tcga"]["clinical_patient"])
        split = get_or_create_split(cfg, args.endpoint, 0.30)
        primary_features = ["AGE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "SUBTYPE", "CANCER_TYPE_ACRONYM"]
        clinical_x, _ = fit_clinical_transformer(clinical, split, primary_features)
        clinical_x = clinical_x.add_prefix("clinical__")
        X_train = pd.concat(
            [
                clinical_x.reindex(train_ids).fillna(0.0),
                emb_indexed.reindex(train_ids).fillna(0.0),
            ],
            axis=1,
        )
    else:
        X_train = emb_indexed.reindex(train_ids).fillna(0.0)
    X_train.index = X_train.index.astype(str)
    X_train.index.name = "PATIENT_ID"
    ep_train = endpoint_df.set_index("PATIENT_ID").reindex(train_ids)
    ep_train.index = ep_train.index.astype(str)
    ep_train.index.name = "PATIENT_ID"


    if "ipcw_weight_os" not in ep_train.columns:
        ep_train["ipcw_weight_os"] = 1.0
        ep_train["ipcw_label_available"] = True
        ep_train["os_death_observed"] = True
        if "OS_EVENT" in ep_train.columns:
            ep_train["os_death"] = ep_train["OS_EVENT"].fillna(False).astype(float)
        else:
            ep_train["os_death"] = ep_train.get("event", pd.Series(0, index=ep_train.index)).astype(float)
    if "pseudo_risk_os" not in ep_train.columns:
        ep_train["pseudo_risk_os"] = ep_train["os_death"].fillna(0).clip(0, 1)

    ctx.logger.info(
        "GAN augmentation input: %d patients, %d features, feature_set=%s",
        len(X_train), X_train.shape[1], args.gan_feature_set,
    )


    audit = AuditLog()
    try:
        X_aug, ep_aug, report = select_train_only_gan_augmentation(X_train, ep_train, gan_cfg, audit)
        report["gan_feature_set"] = args.gan_feature_set
    except Exception as exc:
        ctx.add_warning(f"GAN augmentation failed: {exc}")
        ctx.finalize([embedding_path, endpoint_path])
        return 1

    out_dir = ctx.data_dir("gan_augmentation") if hasattr(ctx, 'data_dir') else ctx.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)


    aug_x_path = out_dir / "gan_augmented_training_features.tsv"
    X_aug.reset_index().to_csv(aug_x_path, sep="\t", index=False)
    ctx.register_output(aug_x_path, "processed_data",
                        f"GAN-augmented training features (n_real={len(X_train)}, n_total={len(X_aug)}).")

    aug_ep_path = out_dir / "gan_augmented_endpoint.tsv"
    ep_aug.reset_index().to_csv(aug_ep_path, sep="\t", index=False)
    ctx.register_output(aug_ep_path, "processed_data",
                        "GAN-augmented endpoint table with IPCW weights.")


    qc_records = report.get("qc_attempts", [])
    if qc_records:
        qc_path = out_dir / "gan_qc_report.tsv"
        pd.DataFrame(qc_records).to_csv(qc_path, sep="\t", index=False)
        ctx.register_output(qc_path, "qc", "GAN synthetic data QC report (all candidate attempts).")
    candidate_records = report.get("candidates", [])
    if candidate_records:
        utility_path = out_dir / "gan_model_utility.tsv"
        pd.DataFrame(candidate_records).to_csv(utility_path, sep="\t", index=False)
        ctx.register_output(utility_path, "analysis_data",
                            "Train-only GAN utility gate by candidate ratio and sampling strategy.")
    report_path = out_dir / "train_only_gan_augmentation_report.json"
    save_json(report, report_path)
    ctx.register_output(report_path, "analysis_data", "Selected train-only GAN augmentation report.")


    audit_path = out_dir / "gan_audit_log.tsv"
    audit.to_frame().to_csv(audit_path, sep="\t", index=False)

    status = report.get("status", "unknown")
    n_syn = report.get("n_synthetic", 0)
    method = report.get("method", "none")
    ctx.logger.info("GAN augmentation: status=%s, n_synthetic=%d, method=%s", status, n_syn, method)

    ctx.finalize([embedding_path, endpoint_path])
    return 0


