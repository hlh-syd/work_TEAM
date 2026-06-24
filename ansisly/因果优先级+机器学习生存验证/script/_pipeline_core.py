#& 'C:\Users\he\AppData\Local\Programs\Python\Python312\python.exe' 'D:\work-课题\ansisly\“因果优先级 + 机器学习生存验 证\script\integrated_pipeline.py'
#!/usr/bin/env python

from __future__ import annotations


import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata as importlib_metadata
import json
import logging
import os
import random
import shutil
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REQUIRED_PYTHON_VERSION = (3, 12, 4)
REQUIRED_PYTHON_EXECUTABLE = Path(
    r"C:\Users\he\AppData\Local\Programs\Python\Python312\python.exe"
)
REQUIREMENTS_LOCK = PROJECT_ROOT / "requirements.txt"


def _normalized_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _read_requirements_lock(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text()


def _fail_environment_check(message: str) -> None:
    raise RuntimeError(
        "\n".join(
            [
                "Project Python environment check failed.",
                message,
                "",
                "This project uses one Python environment only:",
                f"  {REQUIRED_PYTHON_EXECUTABLE}",
                f"  Python {'.'.join(map(str, REQUIRED_PYTHON_VERSION))}",
                "",
                "After any pip install, upgrade, or uninstall, update the lock file with:",
                f"  & '{REQUIRED_PYTHON_EXECUTABLE}' -m pip freeze > '{REQUIREMENTS_LOCK}'",
            ]
        )
    )


def enforce_project_environment() -> None:

    current_version = sys.version_info[:3]
    if current_version != REQUIRED_PYTHON_VERSION:
        _fail_environment_check(
            "Current interpreter version is "
            f"{current_version[0]}.{current_version[1]}.{current_version[2]}, "
            f"expected {'.'.join(map(str, REQUIRED_PYTHON_VERSION))}."
        )

    current_executable = Path(sys.executable).resolve()
    expected_executable = REQUIRED_PYTHON_EXECUTABLE.resolve()
    if str(current_executable).casefold() != str(expected_executable).casefold():
        _fail_environment_check(
            f"Current interpreter is {current_executable}; expected {expected_executable}. "
            "Do not run this pipeline from Anaconda base, another Conda environment, "
            "a virtualenv, or a different system Python."
        )

    if not REQUIREMENTS_LOCK.exists():
        _fail_environment_check(f"Missing dependency lock file: {REQUIREMENTS_LOCK}")

    requirements_lines = _normalized_lines(_read_requirements_lock(REQUIREMENTS_LOCK))
    unpinned = [
        line
        for line in requirements_lines
        if not line.startswith("#") and not line.startswith("--") and "==" not in line and " @ " not in line
    ]
    if unpinned:
        _fail_environment_check(
            "requirements.txt must contain exact package pins only. "
            f"Found unpinned entries: {', '.join(unpinned[:5])}"
        )

    freeze = subprocess.run(
        [str(REQUIRED_PYTHON_EXECUTABLE), "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    frozen_lines = _normalized_lines(freeze.stdout)
    if requirements_lines != frozen_lines:
        _fail_environment_check(
            "requirements.txt is not synchronized with the active Python 3.12 environment."
        )


enforce_project_environment()

import numpy as np
import pandas as pd


import dataclasses
import re
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
from collections import defaultdict
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

try:
    from lifelines import CoxPHFitter as LL_CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test, proportional_hazard_test as ll_ph_test
except Exception:
    LL_CoxPHFitter = None
    KaplanMeierFitter = None
    logrank_test = None
    ll_ph_test = None

CoxPHFitter = LL_CoxPHFitter

try:
    from sksurv.metrics import (
        concordance_index_ipcw as gan_concordance_index_ipcw,
        cumulative_dynamic_auc as gan_cumulative_dynamic_auc,
        integrated_brier_score as gan_integrated_brier_score,
    )
    from sksurv.util import Surv as GanSurv
except Exception:
    gan_concordance_index_ipcw = None
    gan_cumulative_dynamic_auc = None
    gan_integrated_brier_score = None
    GanSurv = None

Surv = GanSurv
concordance_index_ipcw = gan_concordance_index_ipcw
cumulative_dynamic_auc = gan_cumulative_dynamic_auc
integrated_brier_score = gan_integrated_brier_score

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BANNED_OUTPUT_STEMS = {
    "figure", "figure1", "fig", "fig1", "plot", "image", "output", "result",
    "results", "data", "table", "tmp", "temp", "untitled", "new", "copy",
    "test", "default",
}

DATASET_PROVENANCE = {
    "tcga_coadread": {"study_id": "coadread_tcga_pan_can_atlas_2018", "source": "cBioPortal"},
    "msk_crc_2017": {"study_id": "crc_msk_2017", "source": "cBioPortal"},
    "geo_gse103479": {"study_id": "GSE103479", "source": "NCBI GEO"},
    "geo_gse17538": {"study_id": "GSE17538", "source": "NCBI GEO"},
    "geo_gse39582": {"study_id": "GSE39582", "source": "NCBI GEO"},
    "msigdb": {"source": "MSigDB"},
    "cptac_coad_2019": {"study_id": "coad_cptac_2019", "source": "cBioPortal / CPTAC"},
    "htan_crc_2024": {"study_id": "crc_hta8_htan_2024", "source": "cBioPortal / HTAN"},
}


def script_root() -> Path:
    return Path(__file__).resolve().parent


def project_root() -> Path:
    return script_root().parent.parent.parent


def default_config() -> dict[str, Any]:
    root = project_root()
    module_data_root = script_root().parent / "data"
    raw_root = root / "rawData"
    return {
        "project_root": str(root),
        "script_root": str(script_root()),
        "module_data_root": str(module_data_root),
        "source_data_pool": str(root / "rawData"),
        "raw_root": str(raw_root),
        "processed_root": str(module_data_root),
        "result_root": str(script_root().parent / "results"),
        "random_seed": 20260525,
        "endpoint": "OS",
        "max_features_per_omics": 300,
        "filters": {
            "min_nonmissing_ratio": 0.7,
            "min_nonmissing_ratio_train": 0.7,
            "low_variance_quantile": 0.05,
            "univariable_fdr_threshold": 0.20,
            "min_events_per_feature": 5,
            "min_unique_values_per_feature": 3,
        },
        "cohorts": {
            "tcga": {
                "clinical_patient": str(raw_root / "coadread_tcga_pan_can_atlas_2018" / "coadread_tcga_pan_can_atlas_2018" / "data_clinical_patient.txt"),
                "rna": str(raw_root / "coadread_tcga_pan_can_atlas_2018" / "coadread_tcga_pan_can_atlas_2018" / "data_mrna_seq_v2_rsem.txt"),
                "cnv": str(raw_root / "coadread_tcga_pan_can_atlas_2018" / "coadread_tcga_pan_can_atlas_2018" / "data_log2_cna.txt"),
                "methylation": str(raw_root / "coadread_tcga_pan_can_atlas_2018" / "coadread_tcga_pan_can_atlas_2018" / "data_methylation_hm450.txt"),
                "mutation": str(raw_root / "coadread_tcga_pan_can_atlas_2018" / "coadread_tcga_pan_can_atlas_2018" / "data_mutations.txt"),
                "rppa": str(raw_root / "coadread_tcga_pan_can_atlas_2018" / "coadread_tcga_pan_can_atlas_2018" / "data_rppa.txt"),
            },
            "msk": {
                "clinical_patient": str(raw_root / "crc_msk_2017" / "crc_msk_2017" / "data_clinical_patient.txt"),
                "mutation": str(raw_root / "crc_msk_2017" / "crc_msk_2017" / "data_mutations.txt"),
                "cnv": str(raw_root / "crc_msk_2017" / "crc_msk_2017" / "data_cna.txt"),
            },
            "geo_gse103479": {"clinical": str(raw_root / "GEO_COAD" / "GSE103479" / "clinical.txt"), "expression": str(raw_root / "GEO_COAD" / "GSE103479" / "geneMatrix.txt")},
            "geo_gse17538": {"clinical": str(raw_root / "GEO_COAD" / "GSE17538" / "clinical.txt"), "expression": str(raw_root / "GEO_COAD" / "GSE17538" / "geneMatrix.txt")},
            "geo_gse39582": {"clinical": str(raw_root / "GEO_COAD" / "GSE39582" / "clinical.txt"), "expression": str(raw_root / "GEO_COAD" / "GSE39582" / "geneMatrix.txt")},
            "msigdb": {
                "hallmark": str(raw_root / "MSigDB" / "h.all.v2024.1.Hs.symbols.gmt"),
                "kegg": str(raw_root / "MSigDB" / "c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt"),
                "go_bp": str(raw_root / "MSigDB" / "c5.go.bp.v2024.1.Hs.symbols.gmt"),
            },
            "cptac": {
                "clinical_patient": str(raw_root / "coad_cptac_2019" / "coad_cptac_2019" / "data_clinical_patient.txt"),
                "rna": str(raw_root / "coad_cptac_2019" / "coad_cptac_2019" / "data_mrna_seq_v2_rsem.txt"),
                "protein": str(raw_root / "coad_cptac_2019" / "coad_cptac_2019" / "data_protein_quantification.txt"),
                "phosphoprotein": str(raw_root / "coad_cptac_2019" / "coad_cptac_2019" / "data_phosphoprotein_quantification.txt"),
                "cnv": str(raw_root / "coad_cptac_2019" / "coad_cptac_2019" / "data_log2_cna.txt"),
                "mutation": str(raw_root / "coad_cptac_2019" / "coad_cptac_2019" / "data_mutations.txt"),
            },
            "htan": {
                "clinical_patient": str(raw_root / "crc_hta8_htan_2024" / "crc_hta8_htan_2024" / "data_clinical_patient.txt"),
                "pseudo_bulk_rna": str(raw_root / "crc_hta8_htan_2024" / "crc_hta8_htan_2024" / "data_pseudo_bulk_rna_seq_cpm.txt"),
                "relative_fraction": str(raw_root / "crc_hta8_htan_2024" / "crc_hta8_htan_2024" / "data_relative_fraction.txt"),
                "cnv": str(raw_root / "crc_hta8_htan_2024" / "crc_hta8_htan_2024" / "data_cna.txt"),
                "mutation": str(raw_root / "crc_hta8_htan_2024" / "crc_hta8_htan_2024" / "data_mutations.txt"),
            },
        },
    }


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | None = None) -> dict[str, Any]:
    cfg = default_config()
    path = Path(config_path).resolve() if config_path else script_root() / "project_config.yaml"
    if path.exists():
        try:
            import yaml
            deep_update(cfg, yaml.safe_load(path.read_text(encoding="utf-8")) or {})
        except Exception as exc:
            raise RuntimeError(f"Cannot read config {path}: {exc}") from exc
    module_data_root = (script_root().parent / "data").resolve()
    cfg.update({
        "project_root": str(project_root().resolve()),
        "script_root": str(script_root().resolve()),
        "module_data_root": str(module_data_root),
        "source_data_pool": str((project_root() / "rawData").resolve()),
        "raw_root": str((project_root() / "rawData").resolve()),
        "processed_root": str(module_data_root),
        "result_root": str((script_root().parent / "results").resolve()),
        "endpoint": "OS",
    })
    return cfg


def resolve_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def assert_safe_paths(cfg: dict[str, Any]) -> None:
    source_data_pool = resolve_path(cfg["source_data_pool"])
    raw_root = resolve_path(cfg["raw_root"])
    module_data_root = resolve_path(cfg["module_data_root"])
    if source_data_pool != (project_root() / "rawData").resolve():
        raise RuntimeError(f"source_data_pool must be the approved project data pool: {project_root() / 'rawData'}")
    if raw_root != (project_root() / "rawData").resolve():
        raise RuntimeError(f"raw_root must point to project rawData pool: {project_root() / 'rawData'}")


def assert_safe_write_path(path_like: str | Path, cfg: dict[str, Any]) -> Path:
    path = resolve_path(path_like)
    source_data_pool = resolve_path(cfg["source_data_pool"])
    raw_root = resolve_path(cfg["raw_root"])
    allowed = [resolve_path(cfg["processed_root"]), resolve_path(cfg["result_root"]), resolve_path(cfg["script_root"])]
    if is_relative_to(path, source_data_pool) or is_relative_to(path, raw_root):
        raise PermissionError(f"Refusing unsafe write path: {path}")
    if not any(is_relative_to(path, root) for root in allowed):
        raise PermissionError(f"Write path is outside approved project locations: {path}")
    return path


def validate_semantic_filename(path_like: str | Path) -> None:
    stem = Path(path_like).stem.lower()
    if stem in BANNED_OUTPUT_STEMS or re.fullmatch(r"(figure|fig|plot|image|table|output|result|data)[_-]?\d*", stem):
        raise ValueError(f"Non-semantic output filename is not allowed: {Path(path_like).name}")


def ensure_project_dir(path_like: str | Path, cfg: dict[str, Any]) -> Path:
    path = assert_safe_write_path(path_like, cfg)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir_timestamp() -> str:
    return dt.datetime.now().strftime("%m%d%H%M%S")


def create_run_dir(script_file: str | Path, cfg: dict[str, Any]) -> Path:
    result_root = ensure_project_dir(cfg["result_root"], cfg)
    run_dir = result_root / run_dir_timestamp()
    counter = 1
    while run_dir.exists():
        run_dir = result_root / f"{run_dir_timestamp()}_{counter:02d}"
        counter += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def setup_logger(run_dir: Path, script_stem: str) -> logging.Logger:
    logger = logging.getLogger(script_stem)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")
    for handler in (logging.FileHandler(run_dir / "analysis.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def file_fingerprint(path_like: str | Path) -> dict[str, Any]:
    path = resolve_path(path_like)
    row = {"path": str(path), "exists": path.exists(), "size_bytes": None, "mtime": None, "sha256": None}
    if not path.exists() or not path.is_file():
        return row
    stat = path.stat()
    row.update({"size_bytes": stat.st_size, "mtime": dt.datetime.fromtimestamp(stat.st_mtime).isoformat()})
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    row["sha256"] = hasher.hexdigest()
    return row


class RunContext:
    def __init__(self, script_file: str | Path, args: argparse.Namespace, cfg: dict[str, Any]):
        self.script_file = resolve_path(script_file)
        self.script_stem = self.script_file.stem
        self.args = args
        self.cfg = cfg
        self.run_dir = create_run_dir(self.script_file, cfg)
        self.logger = setup_logger(self.run_dir, self.script_stem)
        self.outputs: list[dict[str, str]] = []
        self.warnings: list[str] = []

    @property
    def processed_root(self) -> Path:
        return resolve_path(self.cfg["processed_root"])

    @property
    def raw_root(self) -> Path:
        return resolve_path(self.cfg["raw_root"])

    def data_dir(self, name: str) -> Path:
        return ensure_project_dir(self.processed_root / name, self.cfg)

    def output_path(self, path_like: str | Path, kind: str = "data", semantic: bool = True) -> Path:
        path = assert_safe_write_path(path_like, self.cfg)
        if semantic:
            validate_semantic_filename(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def register_output(self, path_like: str | Path, kind: str, description: str) -> None:
        self.outputs.append({"path": str(resolve_path(path_like)), "kind": kind, "description": description})

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)
        self.logger.warning(message)

    def write_text(self, path_like: str | Path, text: str, kind: str, description: str, semantic: bool = True) -> Path:
        path = self.output_path(path_like, kind=kind, semantic=semantic)
        path.write_text(text, encoding="utf-8")
        self.register_output(path, kind, description)
        return path

    def write_json(self, path_like: str | Path, obj: Any, kind: str, description: str, semantic: bool = True) -> Path:
        return self.write_text(path_like, json.dumps(obj, indent=2, ensure_ascii=False), kind, description, semantic)

    def write_table(self, path_like: str | Path, rows_or_df: Any, kind: str, description: str, semantic: bool = True) -> Path:
        path = self.output_path(path_like, kind=kind, semantic=semantic)
        (rows_or_df if hasattr(rows_or_df, "to_csv") else pd.DataFrame(rows_or_df)).to_csv(path, sep="\t", index=False)
        self.register_output(path, kind, description)
        return path

    def finalize(self, input_files: Iterable[str | Path] | None = None) -> None:
        pd.DataFrame([file_fingerprint(p) for p in (input_files or [])]).to_csv(self.run_dir / "input_file_hashes.tsv", sep="\t", index=False)
        pd.DataFrame(self.outputs).to_csv(self.run_dir / "output_file_manifest.tsv", sep="\t", index=False)
        metadata = {"script": str(self.script_file), "run_dir": str(self.run_dir), "timestamp": dt.datetime.now().isoformat(), "args": vars(self.args), "config": self.cfg, "dataset_provenance": DATASET_PROVENANCE, "outputs": self.outputs, "warning_count": len(self.warnings)}
        (self.run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.run_dir / "warnings_and_limitations.md").write_text("\n".join(f"- {w}" for w in self.warnings) or "- No limitations were raised during this run.\n", encoding="utf-8")
        self.logger.info("Finalized run metadata and manifest.")


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=None, help="Optional YAML config file.")
    parser.add_argument("--endpoint", default="OS", choices=["OS"], help="Survival endpoint (OS only).")
    parser.add_argument("--seed", default=20260525, type=int, help="Random seed.")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and environment without writing processed data.")
    parser.add_argument("--max-features-per-omics", default=300, type=int, help="Feature cap for smoke-safe modelling.")
    return parser


def stage_required_data_from_pool(cfg: dict[str, Any]) -> Path:
    raw_root = resolve_path(cfg["raw_root"])
    raw_root.mkdir(parents=True, exist_ok=True)
    manifest_path = resolve_path(cfg["module_data_root"]) / "staged_data_manifest.tsv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"status": "not_copied_by_integrated_pipeline", "note": "Use existing module-local data/raw_inputs."}]).to_csv(manifest_path, sep="\t", index=False)
    return manifest_path


def initialize_run(script_file: str | Path, args: argparse.Namespace) -> RunContext:
    cfg = load_config(getattr(args, "config", None))
    cfg["endpoint"] = "OS"
    cfg["random_seed"] = int(getattr(args, "seed", cfg.get("random_seed", 20260525)))
    cfg["max_features_per_omics"] = int(getattr(args, "max_features_per_omics", cfg.get("max_features_per_omics", 300)))
    assert_safe_paths(cfg)
    stage_manifest = stage_required_data_from_pool(cfg)
    set_global_seed(cfg["random_seed"])
    ctx = RunContext(script_file, args, cfg)
    ctx.logger.info("Initialized run directory: %s", ctx.run_dir)
    ctx.logger.info("Staged data manifest: %s", stage_manifest)
    return ctx


def assert_data_input_path(path_like: str | Path, cfg: dict[str, Any] | None = None) -> Path:
    path = resolve_path(path_like)
    cfg = cfg or load_config(None)
    allowed_roots = [
        resolve_path(cfg["module_data_root"]),
        resolve_path(cfg["raw_root"]),
        resolve_path(cfg["source_data_pool"]),
    ]
    if not any(is_relative_to(path, root) for root in allowed_roots):
        raise PermissionError(f"Input data path is outside approved directories: {path}")
    return path


def read_cbio_table(path_like: str | Path, nrows: int | None = None) -> pd.DataFrame:
    assert_data_input_path(path_like)
    return pd.read_csv(path_like, sep="\t", comment="#", dtype=str, low_memory=False, nrows=nrows)


def read_table_auto(path_like: str | Path, nrows: int | None = None) -> pd.DataFrame:
    assert_data_input_path(path_like)
    return pd.read_csv(path_like, sep="\t", dtype=str, low_memory=False, nrows=nrows)


def patient_id_from_sample(value: Any) -> str:
    text = str(value)
    if text.startswith("TCGA-") and len(text) >= 12:
        return text[:12]
    if text.startswith("HTA8_"):
        parts = text.split("_")
        if len(parts) >= 2:
            return "_".join(parts[:2])
    return text


def parse_survival_status(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.strip()
    return values.str.startswith("1") | values.str.lower().isin({"1", "dead", "deceased", "event", "progression", "recurred/progressed"})


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace({"[Not Available]": None, "": None, "NA": None, "NaN": None}), errors="coerce")


def bh_fdr(p_values: Iterable[float]) -> list[float]:
    from statsmodels.stats.multitest import multipletests
    arr = np.asarray(list(p_values), dtype=float)
    out = np.full(len(arr), np.nan)
    mask = ~np.isnan(arr)
    if mask.any():
        out[mask] = multipletests(arr[mask], method="fdr_bh")[1]
    return out.tolist()


def load_matrix_header(path_like: str | Path, comment: str | None = "#") -> list[str]:
    kwargs = {"sep": "\t", "nrows": 0, "dtype": str}
    if comment is not None:
        kwargs["comment"] = comment
    return list(pd.read_csv(path_like, **kwargs).columns)


def matrix_sample_columns(columns: Iterable[str]) -> list[str]:
    meta_cols = {"Hugo_Symbol", "Entrez_Gene_Id", "Cytoband", "Composite.Element.REF", "ENTITY_STABLE_ID", "NAME", "DESCRIPTION", "TRANSCRIPT_ID", "ID", "GENE_SYMBOL", "PHOSPHOSITE", "geneNames"}
    return [c for c in columns if c not in meta_cols]


def stream_gene_matrix(path_like: str | Path, comment: str | None = "#", id_col_preference: str | None = "Hugo_Symbol", chunksize: int = 5000) -> pd.DataFrame:
    chunks = [chunk for chunk in pd.read_csv(path_like, sep="\t", comment=comment, dtype=str, low_memory=False, chunksize=chunksize)]
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    id_col = id_col_preference if id_col_preference in df.columns else df.columns[0]
    sample_cols = matrix_sample_columns(df.columns)
    feature_df = df[[id_col] + sample_cols].set_index(id_col).apply(pd.to_numeric, errors="coerce")
    if feature_df.index.has_duplicates:
        feature_df = feature_df.groupby(level=0).mean()
    feature_df = feature_df.loc[feature_df.notna().any(axis=1)].T
    feature_df.index = [patient_id_from_sample(c) for c in feature_df.index]
    return feature_df.groupby(level=0).mean()


def rank_inverse_normal(series: pd.Series) -> pd.Series:
    from scipy.stats import norm
    s = pd.to_numeric(series, errors="coerce")
    n = s.notna().sum()
    if n < 3:
        return s * np.nan
    return pd.Series(norm.ppf((s.rank(method="average") - 0.5) / n), index=s.index)


def stratified_split_keys(df: pd.DataFrame, cols: list[str], min_count: int = 5) -> pd.Series:
    keys = df[cols].astype(str).fillna("Unknown").agg("|".join, axis=1)
    counts = keys.value_counts()
    return keys.where(~keys.isin(counts.index[counts < min_count]), other="_rare_pooled")


def bootstrap_metric(func, n_iter: int, seed: int, *arrays) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    point = float(func(*arrays))
    samples = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        try:
            samples.append(float(func(*(np.asarray(a)[idx] for a in arrays))))
        except Exception:
            continue
    arr = np.asarray(samples, dtype=float)
    arr = arr[~np.isnan(arr)]
    return {"point": point, "ci_low": float(np.percentile(arr, 2.5)) if arr.size else np.nan, "ci_high": float(np.percentile(arr, 97.5)) if arr.size else np.nan, "n_valid": int(arr.size)}


def harrell(time_months: np.ndarray, event: np.ndarray, risk_score: np.ndarray) -> float:
    """Compute Harrell's C-index (concordance index)."""
    from lifelines.utils import concordance_index as _ci
    try:
        return float(_ci(time_months, risk_score, event))
    except Exception:
        return float("nan")


def bootstrap_harrell(time_months: np.ndarray, event: np.ndarray, risk_score: np.ndarray,
                      n_iter: int = 1000, seed: int = 42) -> dict:
    """Bootstrap Harrell's C-index with confidence interval."""
    from lifelines.utils import concordance_index as _ci
    def _harrell(t, e, r):
        try:
            return float(_ci(t, r, e))
        except Exception:
            return float("nan")
    return bootstrap_metric(_harrell, int(n_iter), int(seed), time_months, event, risk_score)


CLINICAL_VARIABLE_WHITELIST = {"AGE", "AGE_AT_DIAGNOSIS", "SEX", "RACE", "ETHNICITY", "AJCC_PATHOLOGIC_TUMOR_STAGE", "PATH_M_STAGE", "PATH_N_STAGE", "PATH_T_STAGE", "SUBTYPE", "CANCER_TYPE_ACRONYM", "GENETIC_ANCESTRY_LABEL", "PRIMARY_SITE"}
PSEUDOGENE_HINTS = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9")


def is_likely_pseudogene(symbol: str) -> bool:
    if not isinstance(symbol, str):
        return False
    s = symbol.upper()
    return s.startswith(("LOC", "LINC", "MIR", "SNORD", "RNU")) or any(s.endswith(suffix) for suffix in PSEUDOGENE_HINTS)


OS_RISK_TIME_MONTHS = None
RANDOM_SEED = 42
RANDOM_SEED_GAN = 20260604
EPS = 1e-8
FIXED_TAU_MONTHS = 36.0  # 固定的临床评估时间窗（月），确保跨队列可比性


@dataclasses.dataclass
class GANPipelineConfig:

    tau_months: float = FIXED_TAU_MONTHS
    random_seed: int = RANDOM_SEED_GAN
    gan_epochs: int = 300
    gan_batch_size: int = 32
    gan_latent_dim: int = 64
    gan_aug_ratio: float = 0.5
    gan_aug_ratio_candidates: tuple = (0.25, 0.5, 1.0, 2.0)
    gan_sampling_strategies: tuple = ("balanced_event", "risk_stratified", "event_only", "overall")
    gan_n_critic: int = 5
    gan_lambda_gp: float = 10.0
    gan_lr: float = 2e-4
    gan_patience: int = 30
    gan_qc_min_good: int = 4
    gan_max_aug_ratio: float = 0.8
    gan_use_feature_space: bool = False
    gan_latent_k: int = 30
    run_gan: bool = True


PipelineConfig = GANPipelineConfig


class AuditLog:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def add(self, step: str, status: str, message: str, **extra: Any) -> None:
        rec = {"step": step, "status": status, "message": message}
        rec.update(extra)
        self.records.append(rec)
        print(f"[{status}] {step}: {message}", flush=True)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


class QuietAuditLog(AuditLog):
    def add(self, step: str, status: str, message: str, **extra: Any) -> None:
        rec = {"step": step, "status": status, "message": message}
        rec.update(extra)
        self.records.append(rec)


def normalize_patient_id(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    if text.upper().startswith("TCGA-") and len(text) >= 12:
        return text[:12].upper()
    return re.sub(r"[\s_]+", "-", text).upper()


def ensure_dir_simple(path: Path) -> Path:
    """Create directory (no safety checks). Used by run_pipeline internally.

    For path-safe directory creation with cfg checks, use ensure_project_dir.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


# Backward-compatible alias
ensure_dir = ensure_dir_simple


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def detect_event_time_columns(df: pd.DataFrame) -> Tuple[str, str]:
    time_candidates = [
        "OS_TIME_MONTHS",
        "OS_MONTHS",
        "time_months",
        "TIME_MONTHS",
        "OS_time",
        "time",
    ]
    event_candidates = [
        "OS_EVENT",
        "event",
        "EVENT",
        "OS_event",
        "OS_STATUS_BINARY",
    ]
    time_col = next((c for c in time_candidates if c in df.columns), None)
    event_col = next((c for c in event_candidates if c in df.columns), None)
    if time_col is None or event_col is None:
        raise ValueError(f"Cannot detect OS time/event columns from: {list(df.columns)[:30]}")
    return time_col, event_col


def coerce_event(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    lower = series.astype(str).str.lower()
    mapped = lower.map(
        {
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
            "1:deceased": 1,
            "0:living": 0,
            "deceased": 1,
            "living": 0,
            "dead": 1,
            "alive": 0,
            "event": 1,
            "censored": 0,
        }
    )
    numeric = pd.to_numeric(series, errors="coerce")
    out = mapped.where(mapped.notna(), numeric)
    n_missing = int(out.isna().sum())
    if n_missing > 0:
        warnings.warn(f"coerce_event: {n_missing} unrecognized values filled as 0", stacklevel=2)
    return out.fillna(0).astype(int)


def kaplan_meier_survival_at(times: np.ndarray, event_observed: np.ndarray, query: np.ndarray) -> np.ndarray:
    order = np.argsort(times)
    times_sorted = np.asarray(times, dtype=float)[order]
    events_sorted = np.asarray(event_observed, dtype=int)[order]
    unique_event_times = np.unique(times_sorted[events_sorted == 1])
    surv = 1.0
    surv_times: List[float] = []
    surv_values: List[float] = []
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


def km_cumulative_incidence_by_tau(times: np.ndarray, events: np.ndarray, tau: float) -> float:
    surv = kaplan_meier_survival_at(times, events, np.asarray([tau], dtype=float))[0]
    return float(1.0 - surv)




def add_os_endpoint_full_followup(
    clinical: pd.DataFrame,
    tau: float | None = None,
    patient_col: str = "PATIENT_ID",
) -> pd.DataFrame:
    """Add OS endpoint columns (os_death, os_death_observed) using full follow-up.

    Note: 'full_followup' clarifies this uses all available time, not a fixed tau.
    """
    df = clinical.copy()
    time_col, event_col = detect_event_time_columns(df)
    df[patient_col] = df[patient_col].map(normalize_patient_id)
    df["time_months"] = safe_numeric(df[time_col])
    df["event"] = coerce_event(df[event_col])
    df = df[df[patient_col].ne("") & df["time_months"].notna()].copy()
    df = df[df["time_months"] >= 0].copy()
    df["os_death_observed"] = pd.Series(True, index=df.index)
    df["os_death"] = df["event"].astype(float)
    return df


# Backward-compatible alias
add_fixed_os_endpoint = add_os_endpoint_full_followup


def compute_ipcw_weights(endpoint: pd.DataFrame, tau: float = FIXED_TAU_MONTHS) -> pd.DataFrame:
    """Compute IPCW weights for OS censoring — only event patients receive weights.

    Fixes (2025): tau truncation + weights restricted to event==1 patients.
    """
    df = endpoint.copy()
    censor_event = (df["event"].eq(0)).astype(int).to_numpy()
    times = df["time_months"].to_numpy(dtype=float)
    fixed_tau = float(tau) if tau is not None else FIXED_TAU_MONTHS
    # ★ 截断至tau，确保时间窗一致
    eval_time = np.minimum(times, fixed_tau)
    eval_time = np.maximum(eval_time, 1.0)
    g_at_eval = kaplan_meier_survival_at(times, censor_event, eval_time)
    weights = np.zeros(len(df), dtype=float)
    # ★ 仅事件患者（event==1，非删失）获得IPCW权重
    event_observed = (df["event"].eq(1)).to_numpy(dtype=bool)
    weights[event_observed] = 1.0 / np.clip(g_at_eval[event_observed], EPS, None)
    df["ipcw_weight_os"] = weights
    df["ipcw_label_available"] = event_observed
    return df


def compute_pseudo_observations(endpoint: pd.DataFrame, tau: float = FIXED_TAU_MONTHS, max_n_exact: int = 1200) -> pd.DataFrame:
    """Compute pseudo-observations at fixed tau for OS (ensures cross-cohort comparability)."""
    df = endpoint.copy()
    times = df["time_months"].to_numpy(dtype=float)
    events = df["event"].to_numpy(dtype=int)
    n = len(df)
    # ★ 固定tau，确保跨队列可比性（替代动态95%分位数）
    fixed_tau = float(tau) if tau is not None else FIXED_TAU_MONTHS
    f_all = km_cumulative_incidence_by_tau(times, events, fixed_tau)
    pseudo = np.full(n, np.nan, dtype=float)
    if n == 0:
        df["pseudo_risk_os_raw"] = pseudo
        df["pseudo_risk_os"] = pseudo
        return df
    if n > max_n_exact:
        pseudo[:] = f_all
    else:
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            f_minus = km_cumulative_incidence_by_tau(times[mask], events[mask], fixed_tau)
            pseudo[i] = n * f_all - (n - 1) * f_minus
    df["pseudo_risk_os_raw"] = pseudo
    df["pseudo_risk_os"] = np.clip(pseudo, 0.0, 1.0)
    return df


def evaluate_binary_and_survival_metrics(
    train_endpoint: pd.DataFrame,
    eval_endpoint: pd.DataFrame,
    risk: np.ndarray,
    tau: float | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Evaluate OS predictions using binary (AUC/Brier) and survival (C-index/AUC) metrics.

    Note: evaluates across full follow-up, not just 36-month timepoint.
    """
    result: dict[str, Any] = {"n": int(len(eval_endpoint)), "events": int(eval_endpoint["event"].sum())}
    valid_risk = np.isfinite(risk)
    observed = eval_endpoint["os_death_observed"].to_numpy(bool) & valid_risk
    if observed.sum() >= 5 and pd.Series(eval_endpoint.loc[observed, "os_death"]).nunique() == 2:
        y = eval_endpoint.loc[observed, "os_death"].astype(int).to_numpy()
        scores = risk[observed]
        weights = eval_endpoint.loc[observed, "ipcw_weight_os"].replace(0, np.nan).fillna(1.0).to_numpy(float)
        for metric, func in [("auc_os_observed", roc_auc_score), ("average_precision_os", average_precision_score)]:
            try:
                result[metric] = float(func(y, scores, sample_weight=weights))
            except Exception as exc:
                warnings.warn(f"evaluate_36m_predictions: {metric} failed: {exc}", stacklevel=2)
                result[metric] = float("nan")
        try:
            result["brier_os_ipcw"] = float(np.average((y - scores) ** 2, weights=weights))
        except Exception as exc:
            warnings.warn(f"evaluate_36m_predictions: brier_os_ipcw failed: {exc}", stacklevel=2)
            result["brier_os_ipcw"] = float("nan")
    else:
        result.update({"auc_os_observed": float("nan"), "brier_os_ipcw": float("nan"), "average_precision_os": float("nan")})
    result["harrell_cindex"] = float(uno_cindex_fallback(eval_endpoint, risk))
    if concordance_index_ipcw is not None and Surv is not None:
        try:
            y_train = make_surv_array(train_endpoint)
            y_eval = make_surv_array(eval_endpoint)
            result["uno_cindex_ipcw"] = float(concordance_index_ipcw(y_train, y_eval, risk, tau=tau)[0])
        except Exception as exc:
            warnings.warn(f"evaluate_36m_predictions: uno_cindex_ipcw failed: {exc}", stacklevel=2)
            result["uno_cindex_ipcw"] = float("nan")
        try:
            t_auc = np.asarray([tau]) if tau is not None else np.asarray([float(np.percentile(eval_endpoint["time_months"], 75))])
            auc_t, mean_auc = cumulative_dynamic_auc(y_train, y_eval, risk, times=t_auc)
            result["time_dependent_auc_os"] = float(auc_t[0])
            result["mean_auc_os"] = float(mean_auc)
        except Exception as exc:
            warnings.warn(f"evaluate_36m_predictions: time_dependent_auc failed: {exc}", stacklevel=2)
            result["time_dependent_auc_os"] = float("nan")
            result["mean_auc_os"] = float("nan")
    else:
        result.update({"uno_cindex_ipcw": float("nan"), "time_dependent_auc_os": float("nan"), "mean_auc_os": float("nan")})
    return result


# Backward-compatible alias
evaluate_36m_predictions = evaluate_binary_and_survival_metrics


def evaluate_survival_metrics(
    train_endpoint: pd.DataFrame,
    eval_endpoint: pd.DataFrame,
    risk: np.ndarray,
    tau: float = FIXED_TAU_MONTHS,
) -> Dict[str, float]:
    """Native survival analysis evaluation: Harrell C, Uno C-index, time-dependent AUC, IBS.

    This is the PRIMARY evaluation function for survival models (Cox PH, RSF, Coxnet).
    Replaces binary-metric-centric evaluate_36m_predictions for survival model selection.
    """
    result: Dict[str, float] = {
        "n": int(len(eval_endpoint)),
        "events": int(eval_endpoint["event"].sum()),
    }
    valid_risk = np.isfinite(risk)
    if valid_risk.sum() < 5:
        return {**result, "harrell_cindex": float("nan"), "uno_cindex_ipcw": float("nan"),
                "time_dependent_auc": float("nan"), "integrated_brier_score": float("nan")}

    ep = eval_endpoint.loc[valid_risk]
    r = risk[valid_risk]
    t = ep["time_months"].to_numpy(dtype=float)
    e = ep["event"].to_numpy(dtype=int)

    # Harrell C-index
    try:
        from lifelines.utils import concordance_index as _lifelines_ci
        result["harrell_cindex"] = float(_lifelines_ci(t, -r, e))
    except Exception as exc:
        warnings.warn(f"evaluate_survival_metrics: lifelines concordance_index failed: {exc}", stacklevel=2)
        result["harrell_cindex"] = float(uno_cindex_fallback(ep, r))

    # Uno C-index (IPCW-based)
    if concordance_index_ipcw is not None and Surv is not None:
        try:
            y_train = make_surv_array(train_endpoint)
            y_eval = make_surv_array(ep)
            result["uno_cindex_ipcw"] = float(concordance_index_ipcw(y_train, y_eval, r, tau=tau)[0])
        except Exception as exc:
            warnings.warn(f"evaluate_survival_metrics: concordance_index_ipcw failed: {exc}", stacklevel=2)
            result["uno_cindex_ipcw"] = float("nan")
    else:
        result["uno_cindex_ipcw"] = float("nan")

    # Time-dependent AUC at tau
    if cumulative_dynamic_auc is not None and Surv is not None:
        try:
            y_train = make_surv_array(train_endpoint)
            y_eval = make_surv_array(ep)
            auc_t, mean_auc = cumulative_dynamic_auc(y_train, y_eval, r, times=np.asarray([tau]))
            result["time_dependent_auc"] = float(auc_t[0])
            result["mean_auc"] = float(mean_auc)
        except Exception as exc:
            warnings.warn(f"evaluate_survival_metrics: cumulative_dynamic_auc failed: {exc}", stacklevel=2)
            result["time_dependent_auc"] = float("nan")
            result["mean_auc"] = float("nan")
    else:
        result["time_dependent_auc"] = float("nan")
        result["mean_auc"] = float("nan")

    # Integrated Brier Score
    if integrated_brier_score is not None and Surv is not None:
        try:
            y_train = make_surv_array(train_endpoint)
            y_eval = make_surv_array(ep)
            result["integrated_brier_score"] = float(
                integrated_brier_score(y_train, y_eval, r, times=np.linspace(0, tau, 50))
            )
        except Exception as exc:
            warnings.warn(f"evaluate_survival_metrics: integrated_brier_score failed: {exc}", stacklevel=2)
            result["integrated_brier_score"] = float("nan")
    else:
        result["integrated_brier_score"] = float("nan")

    return result


def plot_volcano(univ: pd.DataFrame, out_path: Path, fdr_threshold: float) -> None:
    pass


def cis_eqtl_evidence(genes: list[str], qtl_cfg: dict[str, str], logger) -> pd.DataFrame:
    return pd.DataFrame()


def add_fixed_endpoint(clinical: pd.DataFrame, tau: float, patient_col: str = "PATIENT_ID") -> pd.DataFrame:
    """Add fixed-time endpoint columns (multi-timepoint analysis)."""
    df = clinical.copy()
    time_col, event_col = detect_event_time_columns(df)
    df[patient_col] = df[patient_col].map(normalize_patient_id)
    df["time_months"] = safe_numeric(df[time_col])
    df["event"] = coerce_event(df[event_col])
    df = df[df[patient_col].ne("") & df["time_months"].notna()].copy()
    df = df[df["time_months"] >= 0].copy()
    tag = f"_{int(tau)}m"
    event_by_tau = (df["event"].eq(1) & (df["time_months"] <= tau)).astype(float)
    observed_status = (df["event"].eq(1) & (df["time_months"] <= tau)) | (df["time_months"] > tau)
    df[f"death_by{tag}_observed"] = observed_status.astype(bool)
    df[f"death_by{tag}"] = np.where(observed_status, event_by_tau, np.nan)
    df[f"early_censored_before{tag}"] = (~observed_status).astype(bool)
    df[f"survived_observed{tag}"] = (df["time_months"] > tau).astype(bool)
    return df


def compute_ipcw_weights_tagged(endpoint: pd.DataFrame, tau: float, cap_quantile: float = 1.0) -> pd.DataFrame:
    """Compute IPCW weights for tagged endpoint (multi-timepoint)."""
    df = endpoint.copy()
    tag = f"_{int(tau)}m"
    censor_event = (df["event"].eq(0)).astype(int).to_numpy()
    times = df["time_months"].to_numpy(dtype=float)
    label_observed = df[f"death_by{tag}_observed"].to_numpy(dtype=bool)
    eval_time = np.minimum(times, tau)
    g_at_eval = kaplan_meier_survival_at(times, censor_event, eval_time)
    weights = np.zeros(len(df), dtype=float)
    weights[label_observed] = 1.0 / np.clip(g_at_eval[label_observed], EPS, None)
    if cap_quantile < 1.0 and weights[label_observed].size > 0:
        cap_val = float(np.quantile(weights[label_observed], cap_quantile))
        weights = np.minimum(weights, cap_val)
    df[f"ipcw_weight{tag}"] = weights
    df[f"ipcw_label_available{tag}"] = label_observed
    return df


def compute_pseudo_observations_tagged(endpoint: pd.DataFrame, tau: float, max_n_exact: int = 1200) -> pd.DataFrame:
    """Compute pseudo-observations for tagged endpoint (multi-timepoint)."""
    df = endpoint.copy()
    tag = f"_{int(tau)}m"
    times = df["time_months"].to_numpy(dtype=float)
    events = df["event"].to_numpy(dtype=int)
    n = len(df)
    f_all = km_cumulative_incidence_by_tau(times, events, tau)
    pseudo = np.full(n, np.nan, dtype=float)
    if n == 0:
        df[f"pseudo_risk{tag}"] = pseudo
        return df
    if n > max_n_exact:
        pseudo[:] = f_all
    else:
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            f_minus = km_cumulative_incidence_by_tau(times[mask], events[mask], tau)
            pseudo[i] = n * f_all - (n - 1) * f_minus
    df[f"pseudo_risk{tag}"] = pseudo
    df[f"pseudo_risk{tag}_clipped"] = np.clip(pseudo, 0.0, 1.0)
    return df


def choose_training_threshold(endpoint_train: pd.DataFrame, risk_train: np.ndarray) -> float | None:
    """Choose threshold via Youden index on OS outcome."""
    valid = endpoint_train["os_death_observed"].to_numpy(bool) & np.isfinite(risk_train)
    if valid.sum() < 10:
        return None
    y = endpoint_train.loc[valid, "os_death"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return None
    scores = risk_train[valid]
    fpr, tpr, thresholds = roc_curve(y, scores)
    idx = int(np.nanargmax(tpr - fpr))
    return float(thresholds[idx])


def extract_event_weighted_training(endpoint: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Extract event-only OS outcome and IPCW weights for binary model training.

    Only returns patients with ipcw_label_available==True (event patients).
    """
    mask = endpoint["ipcw_label_available"].astype(bool)
    y = endpoint.loc[mask, "os_death"].astype(int)
    w = endpoint.loc[mask, "ipcw_weight_os"].astype(float)
    return y, w


# Backward-compatible alias
censor_safe_binary_training = extract_event_weighted_training


def observed_binary_for_plot(endpoint: pd.DataFrame, risk: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract OS outcome, risk, and weights for plotting."""
    valid = endpoint["os_death_observed"].to_numpy(bool) & np.isfinite(risk)
    if valid.sum() == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=float), np.asarray([], dtype=float)
    y = endpoint.loc[valid, "os_death"].astype(int).to_numpy()
    scores = np.asarray(risk[valid], dtype=float)
    weights = endpoint.loc[valid, "ipcw_weight_os"].replace(0, np.nan).fillna(1.0).to_numpy(float)
    return y, scores, weights
