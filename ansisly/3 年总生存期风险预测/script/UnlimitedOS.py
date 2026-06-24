


from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split


def load_three_years_base() -> Any:
    base_path = Path(__file__).resolve().with_name("3YearsOS.py")
    spec = importlib.util.spec_from_file_location("three_years_os_base", base_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base pipeline from {base_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_three_years_base()

RANDOM_SEED = base.RANDOM_SEED
EPS = base.EPS


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_environment_file(out_dir: Path) -> Path:
    path = out_dir / "environment_requirements.txt"
    content = """# Environment for UnlimitedOS.py
python==3.11.x
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
scikit-learn==1.5.1
joblib==1.4.2
matplotlib==3.9.0
lifelines==0.28.0
scikit-survival==0.23.0
torch==2.3.1  # optional train-only augmentation backend
pycox==0.3.0  # optional exploratory deep survival models
"""
    path.write_text(content, encoding="utf-8")
    return path


def add_unlimited_os_endpoint(
    clinical: pd.DataFrame,
    patient_col: str = "PATIENT_ID",
) -> pd.DataFrame:
    df = clinical.copy()
    time_col, event_col = base.detect_event_time_columns(df)
    df[patient_col] = df[patient_col].map(base.normalize_patient_id)
    df["time_months"] = base.safe_numeric(df[time_col])
    df["event"] = base.coerce_event(df[event_col])
    df = df[df[patient_col].ne("") & df["time_months"].notna()].copy()
    df = df[df["time_months"] >= 0].copy()
    df["os_observed"] = True
    df["os_event"] = df["event"].astype(int)
    df["os_censored"] = df["event"].eq(0)
    df["log1p_time_months"] = np.log1p(df["time_months"].clip(lower=0))
    df["os_event_risk_score"] = make_os_event_risk_proxy(df)
    df["os_case_weight"] = 1.0
    return df


def make_os_event_risk_proxy(endpoint: pd.DataFrame) -> pd.Series:
    time_rank = endpoint["time_months"].rank(method="average", pct=True).fillna(0.5)
    event = endpoint["event"].astype(float)
    censored_penalty = 0.25 * (1.0 - time_rank)
    score = event * (1.0 - 0.5 * time_rank) + (1.0 - event) * censored_penalty
    return score.clip(0.0, 1.0)


def load_external_cohorts_unlimited(
    preprocessed_root: Path,
    audit: Any,
) -> Dict[str, pd.DataFrame]:
    cohorts: Dict[str, pd.DataFrame] = {}
    for path in sorted(preprocessed_root.glob("*_os_clinical_endpoint_qc.tsv")):
        if path.name.startswith("tcga_"):
            continue
        try:
            df = base.read_tsv(path)
            if "PATIENT_ID" not in df.columns:
                continue
            endpoint = add_unlimited_os_endpoint(df).set_index("PATIENT_ID", drop=False)
            name = path.name.replace("_os_clinical_endpoint_qc.tsv", "")
            cohorts[name] = endpoint
        except Exception as exc:
            audit.add("external_load", "WARN", f"Cannot load {path.name}: {exc}")
    return cohorts


def run_causal_screening_os(
    feature_matrix: pd.DataFrame,
    clinical_adjustment: pd.DataFrame,
    endpoint_train: pd.DataFrame,
    random_seed: int,
    max_features: int,
) -> pd.DataFrame:
    common = feature_matrix.index.intersection(endpoint_train.index)
    x_adj = clinical_adjustment.reindex(common).fillna(0)
    outcome = endpoint_train.reindex(common)["os_event_risk_score"].astype(float)
    rows: List[Dict[str, Any]] = []
    for feature in list(feature_matrix.columns)[:max_features]:
        x = feature_matrix.reindex(common)[feature]
        if x.notna().sum() < 30 or x.nunique(dropna=True) < 4:
            continue
        exposure_binary = (x > x.median()).astype(int)
        aipw = base.causal_aipw_binary_exposure(x_adj, exposure_binary, outcome, random_seed)
        cate = base.causal_cate_summary(x_adj, exposure_binary, outcome, random_seed)
        dose = base.dose_response_summary(x, outcome)
        corr = stats.spearmanr(x.fillna(x.median()), outcome.fillna(outcome.median())).correlation
        rows.append(
            {
                "feature": feature,
                "spearman_with_os_event_risk_score": float(corr) if np.isfinite(corr) else np.nan,
                **aipw,
                **cate,
                **dose,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["ate_abs"] = table["ate"].abs()
    table["ate_rank"] = table["ate_abs"].rank(ascending=False, method="average")
    table["p_rank"] = table["p_value"].rank(ascending=True, method="average")
    table["dose_rank"] = table["dose_response_slope"].abs().rank(ascending=False, method="average")
    table["causal_priority_score"] = (
        -table["ate_rank"].fillna(table["ate_rank"].max() + 1)
        -0.5 * table["p_rank"].fillna(table["p_rank"].max() + 1)
        -0.25 * table["dose_rank"].fillna(table["dose_rank"].max() + 1)
    )
    return table.sort_values("causal_priority_score", ascending=False)


def adapt_endpoint_for_train_only_augmentation(endpoint: pd.DataFrame) -> pd.DataFrame:
    adapted = endpoint.copy()
    event_col, raw_event_col, log_time_col, pseudo_clip_col, weight_col = base.SURVIVAL_GAN_CONDITION_COLUMNS
    observed_col = f"{event_col}_observed"
    early_col = "early_censored_before_survival_augmentation"
    pseudo_col = pseudo_clip_col.removesuffix("_clipped")
    adapted[event_col] = adapted["event"].astype(float)
    adapted[raw_event_col] = adapted["event"].astype(float)
    adapted[log_time_col] = np.log1p(adapted["time_months"].astype(float).clip(lower=0.0))
    adapted[observed_col] = True
    adapted[early_col] = False
    adapted[pseudo_col] = adapted["os_event_risk_score"].astype(float)
    adapted[pseudo_clip_col] = adapted["os_event_risk_score"].astype(float).clip(0.0, 1.0)
    adapted[weight_col] = adapted["os_case_weight"].astype(float)
    adapted["ipcw_label_available"] = True
    return adapted


def restore_endpoint_after_train_only_augmentation(endpoint_aug: pd.DataFrame) -> pd.DataFrame:
    restored = endpoint_aug.copy()
    if "os_event_risk_score" not in restored.columns:
        pseudo_clip_col = base.SURVIVAL_GAN_CONDITION_COLUMNS[3]
        restored["os_event_risk_score"] = restored.get(
            pseudo_clip_col,
            pd.Series(restored["event"].astype(float), index=restored.index),
        ).astype(float)
    restored["os_observed"] = True
    restored["os_event"] = restored["event"].astype(int)
    restored["os_censored"] = restored["event"].eq(0)
    restored["log1p_time_months"] = np.log1p(restored["time_months"].clip(lower=0))
    restored["os_case_weight"] = 1.0
    return restored


def select_train_only_survival_augmentation(
    x_train: pd.DataFrame,
    endpoint_train: pd.DataFrame,
    cfg: Any,
    audit: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if not cfg.run_gan:
        return x_train, endpoint_train, {"status": "not_requested", "n_synthetic": 0}
    adapted = adapt_endpoint_for_train_only_augmentation(endpoint_train)
    x_aug, endpoint_aug, report = base.select_train_only_gan_augmentation(
        x_train,
        adapted,
        cfg,
        audit,
    )
    report = dict(report)
    report["endpoint_adapter"] = "full_os_time_event_for_train_only_augmentation"
    return x_aug, restore_endpoint_after_train_only_augmentation(endpoint_aug), report


def predict_os_risk_score(model: Any, model_kind: str, x: pd.DataFrame) -> np.ndarray:
    if model is None:
        return np.full(len(x), np.nan)
    if model_kind == "cox_lifelines":
        try:
            cols = [c for c in model.params_.index if c in x.columns]
            xp = x.reindex(columns=cols, fill_value=0.0)
            return np.asarray(model.predict_partial_hazard(xp), dtype=float).ravel()
        except Exception:
            return np.full(len(x), np.nan)
    try:
        return np.asarray(model.predict(x.to_numpy(float)), dtype=float).ravel()
    except Exception:
        return np.full(len(x), np.nan)


def predict_survival_matrix(
    model: Any,
    model_kind: str,
    x: pd.DataFrame,
    times: np.ndarray,
) -> Optional[np.ndarray]:
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


def choose_evaluation_times(train_endpoint: pd.DataFrame, eval_endpoint: pd.DataFrame) -> np.ndarray:
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


def choose_risk_threshold_os(risk: np.ndarray) -> Optional[float]:
    valid = risk[np.isfinite(risk)]
    if len(valid) < 10:
        return None
    return float(np.nanmedian(valid))


def threshold_counts_os(
    endpoint: pd.DataFrame,
    risk: np.ndarray,
    threshold: Optional[float],
) -> Dict[str, Any]:
    if threshold is None:
        return {}
    valid = np.isfinite(risk)
    if valid.sum() < 10:
        return {"threshold": float(threshold)}
    high = risk[valid] >= threshold
    event = endpoint.loc[valid, "event"].astype(int).to_numpy()
    return {
        "threshold": float(threshold),
        "high_risk_n": int(high.sum()),
        "low_risk_n": int((~high).sum()),
        "high_risk_events": int(event[high].sum()) if high.any() else 0,
        "low_risk_events": int(event[~high].sum()) if (~high).any() else 0,
    }


def evaluate_os_predictions(
    train_endpoint: pd.DataFrame,
    eval_endpoint: pd.DataFrame,
    risk: np.ndarray,
    threshold: Optional[float] = None,
    model: Any = None,
    model_kind: Optional[str] = None,
    x_eval: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "n": int(len(eval_endpoint)),
        "events": int(eval_endpoint["event"].sum()),
        "censored": int(eval_endpoint["event"].eq(0).sum()),
        "median_followup_months": float(eval_endpoint["time_months"].median()),
    }
    result.update(threshold_counts_os(eval_endpoint, risk, threshold))
    result["harrell_cindex"] = float(base.uno_cindex_fallback(eval_endpoint, risk))
    if base.concordance_index_ipcw is not None and base.Surv is not None:
        try:
            y_train = base.make_surv_array(train_endpoint)
            y_eval = base.make_surv_array(eval_endpoint)
            result["uno_cindex_ipcw"] = float(base.concordance_index_ipcw(y_train, y_eval, risk)[0])
        except Exception:
            result["uno_cindex_ipcw"] = float("nan")
    else:
        result["uno_cindex_ipcw"] = float("nan")

    result["integrated_brier_score"] = float("nan")
    result["event_brier_score"] = float("nan")
    if (
        model is not None
        and model_kind is not None
        and x_eval is not None
        and base.integrated_brier_score is not None
        and base.Surv is not None
    ):
        times = choose_evaluation_times(train_endpoint, eval_endpoint)
        surv = predict_survival_matrix(model, model_kind, x_eval, times)
        if surv is not None and surv.shape == (len(eval_endpoint), len(times)):
            try:
                y_train = base.make_surv_array(train_endpoint)
                y_eval = base.make_surv_array(eval_endpoint)
                result["integrated_brier_score"] = float(base.integrated_brier_score(y_train, y_eval, surv, times))
            except Exception:
                result["integrated_brier_score"] = float("nan")
    event = eval_endpoint["event"].astype(int).to_numpy()
    valid = np.isfinite(risk)
    if valid.sum() >= 5 and len(np.unique(event[valid])) == 2:
        try:
            scaled = pd.Series(risk[valid]).rank(method="average", pct=True).to_numpy(float)
            result["event_brier_score"] = float(brier_score_loss(event[valid], scaled))
        except Exception:
            result["event_brier_score"] = float("nan")
    return result


def save_current_figure(path_base: Path) -> List[str]:
    paths: List[str] = []
    for suffix, dpi in [(".png", 300), (".tiff", 600)]:
        path = path_base.with_suffix(suffix)
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(str(path))
    plt.close()
    return paths


def plot_km_risk_strata_os(
    endpoint: pd.DataFrame,
    risk: np.ndarray,
    threshold: Optional[float],
    title: str,
    path_base: Path,
) -> Optional[List[str]]:
    if base.KaplanMeierFitter is None:
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
    kmf = base.KaplanMeierFitter()
    for label in ["Low risk", "High risk"]:
        mask = strata == label
        kmf.fit(ep.loc[mask, "time_months"], event_observed=ep.loc[mask, "event"], label=f"{label} (n={int(mask.sum())})")
        kmf.plot_survival_function(ci_show=True, color=colors[label], lw=2.0)
    p_text = ""
    if base.logrank_test is not None:
        try:
            low = strata == "Low risk"
            high = strata == "High risk"
            lr = base.logrank_test(
                ep.loc[low, "time_months"],
                ep.loc[high, "time_months"],
                event_observed_A=ep.loc[low, "event"],
                event_observed_B=ep.loc[high, "event"],
            )
            p_text = f"Log-rank p = {lr.p_value:.3g}"
        except Exception:
            p_text = ""
    plt.xlabel("Time from diagnosis (months)")
    plt.ylabel("Overall survival probability")
    plt.title(title)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    if p_text:
        plt.text(0.04, 0.08, p_text, transform=plt.gca().transAxes)
    plt.legend(frameon=False)
    return save_current_figure(path_base)


def plot_risk_distribution_by_event(
    endpoint: pd.DataFrame,
    risk: np.ndarray,
    title: str,
    path_base: Path,
) -> Optional[List[str]]:
    valid = np.isfinite(risk)
    if valid.sum() < 10:
        return None
    df = pd.DataFrame(
        {
            "risk": np.asarray(risk[valid], dtype=float),
            "status": np.where(endpoint.loc[valid, "event"].astype(int).to_numpy() == 1, "Death", "Censored"),
        }
    )
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


def plot_followup_time_vs_risk(
    endpoint: pd.DataFrame,
    risk: np.ndarray,
    title: str,
    path_base: Path,
) -> Optional[List[str]]:
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


def generate_os_prediction_figures(
    endpoint: pd.DataFrame,
    risk: np.ndarray,
    threshold: Optional[float],
    cohort_label: str,
    model_name: str,
    figures_dir: Path,
) -> List[Dict[str, Any]]:
    ensure_dir(figures_dir)
    stem = f"{base.sanitize_filename(cohort_label)}__{base.sanitize_filename(model_name)}"
    specs = [
        ("km_risk_strata", plot_km_risk_strata_os, f"{cohort_label}: K-M risk strata"),
        ("risk_distribution_by_event", plot_risk_distribution_by_event, f"{cohort_label}: risk by event status"),
        ("followup_time_vs_risk", plot_followup_time_vs_risk, f"{cohort_label}: follow-up time vs risk"),
    ]
    rows: List[Dict[str, Any]] = []
    for figure_type, func, title in specs:
        path_base = figures_dir / f"{stem}__{figure_type}"
        if figure_type == "km_risk_strata":
            paths = func(endpoint, risk, threshold, title, path_base)
        else:
            paths = func(endpoint, risk, title, path_base)
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


def save_json(data: Mapping[str, Any], path: Path) -> None:
    base.save_json(data, path)


def build_arg_parser() -> argparse.ArgumentParser:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    default_raw = Path(r"D:\work-课题\rawData")
    parser = argparse.ArgumentParser(description="CRC unlimited follow-up OS risk prediction pipeline.")
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--raw-data-root", type=Path, default=default_raw)
    parser.add_argument("--plan-path", type=Path, default=project_root / "plan" / "headplan.txt")
    parser.add_argument("--output-root", type=Path, default=project_root / "results")
    parser.add_argument("--max-expression-features", type=int, default=300)
    parser.add_argument("--causal-prescreen-features", type=int, default=120)
    parser.add_argument("--max-causal-features", type=int, default=50)
    parser.add_argument("--gan-epochs", type=int, default=60)
    parser.add_argument("--gan-aug-ratio", type=float, default=0.5)
    parser.add_argument("--gan-aug-ratios", type=str, default="0.25,0.5,1.0,2.0")
    parser.add_argument("--gan-sampling-strategies", type=str, default="balanced_event,risk_stratified,event_only,overall")
    parser.add_argument("--no-gan", action="store_true")
    parser.add_argument("--no-snp-eqtl", action="store_true")
    parser.add_argument("--strict-dependencies", action="store_true")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser


def config_from_args(args: argparse.Namespace) -> Any:
    gan_aug_ratio_candidates = tuple(float(x.strip()) for x in str(args.gan_aug_ratios).split(",") if x.strip())
    gan_sampling_strategies = tuple(x.strip().lower() for x in str(args.gan_sampling_strategies).split(",") if x.strip())
    return base.PipelineConfig(
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
        run_snp_eqtl=not args.no_snp_eqtl,
        strict_dependencies=args.strict_dependencies,
    )


class TeeStream:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return bool(self.streams and getattr(self.streams[0], "isatty", lambda: False)())


def make_console_log_path(output_root: Path) -> Path:
    log_dir = ensure_dir(output_root.resolve() / "console_logs")
    return log_dir / f"{Path(__file__).stem}_{now_stamp()}.console.log"


def output_root_from_argv(argv: Optional[Sequence[str]]) -> Path:
    tokens = list(sys.argv[1:] if argv is None else argv)
    for i, token in enumerate(tokens):
        if token == "--output-root" and i + 1 < len(tokens):
            return Path(tokens[i + 1])
        prefix = "--output-root="
        if token.startswith(prefix):
            return Path(token[len(prefix):])
    return Path(__file__).resolve().parents[1] / "results"


def run_pipeline(cfg: Any) -> Dict[str, Any]:
    audit = base.AuditLog()
    run_dir = ensure_dir(cfg.output_root / f"UnlimitedOS_{now_stamp()}")
    tables_dir = ensure_dir(run_dir / "tables")
    models_dir = ensure_dir(run_dir / "models")
    reports_dir = ensure_dir(run_dir / "reports")
    figures_dir = ensure_dir(run_dir / "figures")

    dep = base.collect_dependency_report()
    dep.to_csv(tables_dir / "dependency_report.tsv", sep="\t", index=False)
    env_path = write_environment_file(reports_dir)
    if cfg.strict_dependencies:
        missing_core = dep[dep["package"].isin(["lifelines", "sksurv"]) & dep["status"].str.startswith("missing")]
        if not missing_core.empty:
            raise RuntimeError("Strict dependency mode failed: lifelines and scikit-survival are required.")
    audit.add("environment", "OK", f"Dependency report and environment file written to {run_dir}.")

    plan_hash = base.sha256_file(cfg.plan_path) if cfg.plan_path.exists() else ""
    save_json(
        {
            "plan_path": cfg.plan_path,
            "plan_sha256": plan_hash,
            "project_root": cfg.project_root,
            "raw_data_root": cfg.raw_data_root,
            "endpoint_definition": "unlimited_followup_os_time_event",
            "random_seed": cfg.random_seed,
        },
        reports_dir / "run_manifest.json",
    )

    preprocessed = cfg.raw_data_root / "preprocessed"
    survival_models = cfg.raw_data_root / "survival_models"
    tcga_clin_path = base.find_existing(
        [
            preprocessed / "tcga_os_clinical_endpoint_qc.tsv",
            preprocessed / "tcga_coadread_clinical_os_pfs_qc.tsv",
        ]
    )
    if tcga_clin_path is None:
        raise FileNotFoundError("TCGA clinical endpoint table was not found in rawData/preprocessed.")
    tcga = add_unlimited_os_endpoint(base.read_tsv(tcga_clin_path)).set_index("PATIENT_ID", drop=False)
    tcga.to_csv(tables_dir / "tcga_unlimited_os_endpoint.tsv", sep="\t", index=False)
    audit.add(
        "endpoint",
        "OK",
        "Constructed unlimited follow-up OS time/event endpoint.",
        n=int(len(tcga)),
        events=int(tcga["event"].sum()),
        censored=int(tcga["event"].eq(0).sum()),
    )

    split_path = survival_models / "tcga_train_internal_validation_split.tsv"
    if split_path.exists():
        split_df = base.read_tsv(split_path)
        split_df["PATIENT_ID"] = split_df["PATIENT_ID"].map(base.normalize_patient_id)
        train_ids = split_df.loc[split_df["split"].eq("train"), "PATIENT_ID"].tolist()
        val_ids = split_df.loc[split_df["split"].eq("internal_validation"), "PATIENT_ID"].tolist()
        train_ids = [pid for pid in train_ids if pid in tcga.index]
        val_ids = [pid for pid in val_ids if pid in tcga.index]
        split_source = str(split_path)
    else:
        stratify = tcga["event"].astype(str)
        train_ids, val_ids = train_test_split(
            list(tcga.index),
            test_size=cfg.test_size,
            random_state=cfg.random_seed,
            stratify=stratify if stratify.value_counts().min() >= 2 else None,
        )
        split_source = "generated_by_script"
    if not train_ids or not val_ids:
        raise RuntimeError("Cannot construct TCGA_train_real and TCGA_internal_validation_real.")
    train_endpoint = tcga.loc[train_ids].copy()
    val_endpoint = tcga.loc[val_ids].copy()
    pd.DataFrame(
        {
            "PATIENT_ID": train_ids + val_ids,
            "role": ["TCGA_train_real"] * len(train_ids) + ["TCGA_internal_validation_real"] * len(val_ids),
        }
    ).to_csv(tables_dir / "data_roles_tcga_train_internal_validation.tsv", sep="\t", index=False)
    audit.add(
        "data_split",
        "OK",
        "TCGA split loaded; internal validation is locked out from feature selection, augmentation and tuning.",
        source=split_source,
        train_n=len(train_ids),
        internal_validation_n=len(val_ids),
    )

    clinical_pipe, clinical_names = base.build_clinical_preprocessor(train_endpoint)
    clinical_all = pd.DataFrame(clinical_pipe.transform(tcga), index=tcga.index, columns=clinical_names)
    blocks = [clinical_all]
    block_map = {"clinical": clinical_names}
    joblib.dump(clinical_pipe, models_dir / "clinical_preprocessor.joblib")

    expr_path = preprocessed / "tcga_coadread_rna_log2_tpm_like_matrix.tsv"
    expression_pipe: Optional[Any] = None
    expression_features: List[str] = []
    if expr_path.exists():
        expr = base.matrix_patients_to_rows(expr_path)
        expression_features = base.select_top_variance_features(expr, train_ids, cfg.max_expression_features)
        if expression_features:
            expr_scaled, expression_pipe = base.fit_transform_feature_matrix(expr, train_ids, list(tcga.index), expression_features)
            blocks.append(expr_scaled)
            block_map["expression_top_variance_train_only"] = expression_features
            joblib.dump(expression_pipe, models_dir / "expression_train_only_preprocessor.joblib")
            audit.add("expression", "OK", "Loaded TCGA expression and selected train-only high variance genes.", n_features=len(expression_features))
        else:
            audit.add("expression", "WARN", "Expression matrix exists but no overlapping train features were selected.")
    else:
        audit.add("expression", "WARN", "TCGA expression matrix not found; expression models will be skipped.")

    pathway_features: List[str] = []
    pathway_pipe: Optional[Any] = None
    if expression_features and expr_path.exists():
        expr_train_scaled = blocks[-1] if block_map.get("expression_top_variance_train_only") else pd.DataFrame(index=tcga.index)
        pathway_raw, gmt_path = base.compute_pathway_activity(expr_train_scaled, cfg.raw_data_root)
        if not pathway_raw.empty:
            pathway_features = base.select_top_variance_features(pathway_raw, train_ids, min(100, pathway_raw.shape[1]))
            pathway_scaled, pathway_pipe = base.fit_transform_feature_matrix(pathway_raw, train_ids, list(tcga.index), pathway_features)
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
            mutation_matrix = base.patient_feature_matrix(mut_path, prefix="MUT_")
            mutation_features = base.select_train_prevalent_binary_features(
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
                audit.add("somatic_mutation", "OK", "Loaded TCGA somatic mutation matrix.", n_features=len(mutation_features))
            else:
                audit.add("somatic_mutation", "WARN", "Mutation matrix exists but no train-prevalent mutation features passed filters.")
        except Exception as exc:
            audit.add("somatic_mutation", "WARN", f"Mutation module failed: {exc}")
    else:
        audit.add("somatic_mutation", "WARN", "TCGA somatic mutation matrix not found; mutation-compatible models will be skipped.")

    x_all = pd.concat(blocks, axis=1).fillna(0.0)
    x_train_real = x_all.loc[train_ids].copy()
    x_val_real = x_all.loc[val_ids].copy()
    x_all.to_csv(tables_dir / "tcga_model_matrix_real_samples.tsv", sep="\t")
    save_json({k: list(v) for k, v in block_map.items()}, reports_dir / "feature_blocks_manifest.json")

    causal_input_cols = expression_features[: cfg.causal_prescreen_features]
    causal_table = pd.DataFrame()
    if causal_input_cols:
        causal_table = run_causal_screening_os(
            x_all.loc[train_ids, causal_input_cols],
            clinical_all.loc[train_ids],
            train_endpoint,
            cfg.random_seed,
            max_features=cfg.causal_prescreen_features,
        )
        if not causal_table.empty:
            causal_table.to_csv(tables_dir / "train_only_causal_priority_features.tsv", sep="\t", index=False)
            audit.add("causal_screening", "OK", "Train-only OS causal priority table created.", n_features=len(causal_table))
        else:
            audit.add("causal_screening", "WARN", "Causal screening produced no ranked features.")
    else:
        audit.add("causal_screening", "SKIP", "No expression features available for causal priority screening.")

    eqtl_candidates = list(cfg.raw_data_root.glob("*eQTL*")) + list((cfg.raw_data_root / "GTEx_v8_eQTL").glob("*")) if (cfg.raw_data_root / "GTEx_v8_eQTL").exists() else []
    gwas_candidates = list(cfg.raw_data_root.glob("*gwas*")) + list(cfg.raw_data_root.glob("*GWAS*"))
    snp_eqtl_status = {
        "run_requested": cfg.run_snp_eqtl,
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
    model_feature_sets: Dict[str, List[str]] = {"clinical_only": clinical_names}
    if expression_features:
        model_feature_sets["clinical_expression_topvar"] = clinical_names + expression_features
    if causal_features:
        model_feature_sets["clinical_causal_priority"] = clinical_names + causal_features
    if pathway_features:
        model_feature_sets["clinical_pathway"] = clinical_names + pathway_features
    if mutation_features:
        model_feature_sets["clinical_somatic_mutation"] = clinical_names + mutation_features

    if "clinical_causal_priority" in model_feature_sets:
        gan_cols = model_feature_sets["clinical_causal_priority"]
    else:
        gan_cols = model_feature_sets.get("clinical_expression_topvar", clinical_names)
    x_train_aug, endpoint_aug, gan_report = select_train_only_survival_augmentation(
        x_train_real[gan_cols],
        train_endpoint,
        cfg,
        audit,
    )
    save_json(gan_report, reports_dir / "train_only_survival_augmentation_report.json")

    model_rows: List[Dict[str, Any]] = []
    risk_tables: List[pd.DataFrame] = []
    fitted_models: Dict[str, Any] = {}

    for feature_set_name, cols in model_feature_sets.items():
        cols = [c for c in cols if c in x_all.columns]
        if not cols:
            continue
        xtr = x_train_real[cols]
        xva = x_val_real[cols]
        models_to_fit: Dict[str, Tuple[str, Any]] = {}

        cox = base.fit_lifelines_cox(xtr, train_endpoint, penalizer=0.1)
        models_to_fit[f"{feature_set_name}__cox_ph_penalized"] = ("cox_lifelines", cox)

        if len(cols) <= 2000:
            coxnet = base.fit_sksurv_coxnet(xtr, train_endpoint)
            models_to_fit[f"{feature_set_name}__coxnet"] = ("sksurv", coxnet)

        rsf = base.fit_sksurv_rsf(xtr, train_endpoint, cfg.random_seed)
        models_to_fit[f"{feature_set_name}__rsf"] = ("sksurv", rsf)

        if gan_report.get("status") == "fitted" and set(cols).issubset(set(gan_cols)):
            x_aug = x_train_aug[cols]
            ep_aug = endpoint_aug.reindex(x_aug.index)
            gan_cox = base.fit_lifelines_cox(x_aug, ep_aug, penalizer=0.1)
            models_to_fit[f"{feature_set_name}__train_only_augmented_cox_ph"] = ("cox_lifelines", gan_cox)
            if len(cols) <= 2000:
                gan_coxnet = base.fit_sksurv_coxnet(x_aug, ep_aug)
                models_to_fit[f"{feature_set_name}__train_only_augmented_coxnet"] = ("sksurv", gan_coxnet)
            gan_rsf = base.fit_sksurv_rsf(x_aug, ep_aug, cfg.random_seed)
            models_to_fit[f"{feature_set_name}__train_only_augmented_rsf"] = ("sksurv", gan_rsf)

        for model_name, (model_kind, model) in models_to_fit.items():
            if model is None:
                audit.add("model_fit", "WARN", f"{model_name} skipped or failed.")
                continue
            risk_train = predict_os_risk_score(model, model_kind, xtr)
            risk_val = predict_os_risk_score(model, model_kind, xva)
            threshold = choose_risk_threshold_os(risk_train)
            train_metrics = evaluate_os_predictions(train_endpoint, train_endpoint, risk_train, threshold, model, model_kind, xtr)
            val_metrics = evaluate_os_predictions(train_endpoint, val_endpoint, risk_val, threshold, model, model_kind, xva)
            model_rows.append(
                {
                    "model_name": model_name,
                    "feature_set": feature_set_name,
                    "model_kind": model_kind,
                    "n_features": len(cols),
                    "n_train_real": len(xtr),
                    "n_train_synthetic_used": int(gan_report.get("n_synthetic", 0)) if "augmented" in model_name else 0,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"internal_validation_{k}": v for k, v in val_metrics.items()},
                }
            )
            fitted_models[model_name] = {"model": model, "kind": model_kind, "features": cols, "threshold": threshold}
            risk_tables.append(
                pd.DataFrame(
                    {
                        "PATIENT_ID": train_ids,
                        "cohort_role": "TCGA_train_real",
                        "model_name": model_name,
                        "risk_score": risk_train,
                    }
                )
            )
            risk_tables.append(
                pd.DataFrame(
                    {
                        "PATIENT_ID": val_ids,
                        "cohort_role": "TCGA_internal_validation_real",
                        "model_name": model_name,
                        "risk_score": risk_val,
                    }
                )
            )
            audit.add("model_fit", "OK", f"{model_name} trained and evaluated on real internal validation.")

    if not model_rows:
        raise RuntimeError("No survival model could be fitted. Check dependencies and endpoint/event availability.")

    model_metrics = pd.DataFrame(model_rows)
    model_metrics.to_csv(tables_dir / "internal_validation_model_comparison.tsv", sep="\t", index=False)
    pd.concat(risk_tables, ignore_index=True).to_csv(tables_dir / "tcga_train_internal_validation_risk_scores.tsv", sep="\t", index=False)

    ranked = model_metrics.copy()
    ranked["_uno_rank"] = ranked["internal_validation_uno_cindex_ipcw"].rank(ascending=False, na_option="bottom")
    ranked["_harrell_rank"] = ranked["internal_validation_harrell_cindex"].rank(ascending=False, na_option="bottom")
    ranked["_ibs_rank"] = ranked["internal_validation_integrated_brier_score"].rank(ascending=True, na_option="bottom")
    ranked["_selection_score"] = ranked["_uno_rank"] + ranked["_harrell_rank"] + 0.5 * ranked["_ibs_rank"]
    best_row = ranked.sort_values(["_selection_score", "_uno_rank", "_harrell_rank", "_ibs_rank"]).iloc[0].to_dict()
    selected_model_name = str(best_row["model_name"])
    save_json(
        {
            "selected_primary_model": selected_model_name,
            "selection_basis": "Locked internal validation survival metrics; external cohorts not used.",
            "metrics": best_row,
            "leakage_rule": "No augmented row is present in internal or external validation.",
        },
        reports_dir / "selected_primary_model_manifest.json",
    )
    joblib.dump(fitted_models, models_dir / "all_fitted_model_bundles.joblib")
    audit.add("model_selection", "OK", f"Selected primary model: {selected_model_name}")

    external_cohorts = load_external_cohorts_unlimited(preprocessed, audit)
    external_rows: List[Dict[str, Any]] = []
    external_risk_rows: List[pd.DataFrame] = []
    external_candidate_rows: List[Dict[str, Any]] = []
    figure_rows: List[Dict[str, Any]] = []
    selected_bundle = fitted_models[selected_model_name]
    selected_features = selected_bundle["features"]
    selected_kind = selected_bundle["kind"]
    selected_model = selected_bundle["model"]
    selected_threshold = selected_bundle["threshold"]

    selected_train_risk = predict_os_risk_score(selected_model, selected_kind, x_train_real[selected_features])
    selected_val_risk = predict_os_risk_score(selected_model, selected_kind, x_val_real[selected_features])
    figure_rows.extend(generate_os_prediction_figures(train_endpoint, selected_train_risk, selected_threshold, "TCGA_train_real", selected_model_name, figures_dir))
    figure_rows.extend(generate_os_prediction_figures(val_endpoint, selected_val_risk, selected_threshold, "TCGA_internal_validation_real", selected_model_name, figures_dir))

    for cohort_name, ext_endpoint in external_cohorts.items():
        if len(ext_endpoint) < cfg.min_external_n:
            audit.add("external_validation", "SKIP", f"{cohort_name}: n<{cfg.min_external_n}.", n=len(ext_endpoint))
            continue
        x_ext, compat = base.build_external_feature_matrix(
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
        )
        needed = selected_bundle["features"]
        missing = [c for c in needed if c not in x_ext.columns]
        if missing:
            if all(c in clinical_names for c in needed):
                pass
            else:
                audit.add("external_validation", "SKIP", f"{cohort_name}: missing selected model features.", missing_count=len(missing))
                continue
        x_eval = x_ext.reindex(columns=needed, fill_value=0.0)
        risk = predict_os_risk_score(selected_model, selected_kind, x_eval)
        metrics = evaluate_os_predictions(train_endpoint, ext_endpoint, risk, selected_bundle["threshold"], selected_model, selected_kind, x_eval)
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
                    "risk_score": risk,
                }
            )
        )
        figure_rows.extend(generate_os_prediction_figures(ext_endpoint, risk, selected_bundle["threshold"], cohort_name, selected_model_name, figures_dir))
        audit.add("external_validation", "OK", f"{cohort_name}: real-world locked-model validation complete.")

        for candidate_name, candidate_bundle in fitted_models.items():
            candidate_features = candidate_bundle["features"]
            candidate_missing = [c for c in candidate_features if c not in x_ext.columns]
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
            x_candidate = x_ext.reindex(columns=candidate_features)
            candidate_risk = predict_os_risk_score(candidate_model, candidate_kind, x_candidate)
            candidate_metrics = evaluate_os_predictions(
                train_endpoint,
                ext_endpoint,
                candidate_risk,
                candidate_bundle["threshold"],
                candidate_model,
                candidate_kind,
                x_candidate,
            )
            candidate_row.update(candidate_metrics)
            external_candidate_rows.append(candidate_row)

    external_metrics = pd.DataFrame(external_rows)
    external_metrics.to_csv(tables_dir / "external_real_world_validation_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(external_candidate_rows).to_csv(tables_dir / "external_locked_candidate_model_diagnostics.tsv", sep="\t", index=False)
    pd.DataFrame(figure_rows).to_csv(tables_dir / "figure_manifest.tsv", sep="\t", index=False)
    audit.add("figures", "OK", "Wrote full follow-up OS figure manifest.", n=len(figure_rows))
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
        "selected_model_external_mean_harrell_cindex": (
            float(external_metrics["harrell_cindex"].dropna().mean())
            if not external_metrics.empty and "harrell_cindex" in external_metrics
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
        "augmentation_status": gan_report,
        "external_cohorts_evaluated": list(external_metrics["cohort"]) if not external_metrics.empty else [],
        "external_candidate_diagnostics": str(tables_dir / "external_locked_candidate_model_diagnostics.tsv"),
        "rules": [
            "Train-only augmentation is fitted only on TCGA_train_real.",
            "Synthetic rows are not used in internal validation or external validation.",
            "TCGA_internal_validation_real is not used for feature selection, augmentation choice, thresholds, or hyperparameter search.",
            "External real-world cohorts are only used after model locking.",
            "The primary unlimited OS endpoint is the observed time/event pair across full follow-up.",
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    console_log_path = make_console_log_path(output_root_from_argv(argv))
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with console_log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            print(f"[OK] console_log: Mirroring console output to {console_log_path}", flush=True)
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
            return 0
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    raise SystemExit(main())
