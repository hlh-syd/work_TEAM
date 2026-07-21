"""CausalMeta-IB-DeepSurv for leakage-safe CRC survival modelling.

This module fuses the useful ideas from ``031_Casual_model.py`` and
``034_CausalModel.py`` while deliberately not importing their training code.
The old scripts contain incompatible data alignment and validation behaviour;
all patient alignment, fold preprocessing, Reptile updates, and model selection
are implemented here with explicit contracts.

The default command operates on the development cohort only.  Access to the
locked internal validation cohort requires both ``--scope locked`` and
``--unlock-internal-validation``.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import itertools
import json
import logging
import math
import os
import warnings
import random
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

try:
    from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
    from sksurv.metrics import (
        concordance_index_censored,
        concordance_index_ipcw,
        cumulative_dynamic_auc,
        integrated_brier_score,
    )
    from sksurv.util import Surv

    HAS_SKSURV = True
except ImportError:  # pragma: no cover - project environment includes sksurv
    HAS_SKSURV = False

# ── 高级评价指标与出版级绘图 ──
try:
    from survival_metrics import SurvivalEvaluator
    HAS_SURVIVAL_METRICS = True
except ImportError:
    HAS_SURVIVAL_METRICS = False

try:
    from publication_plots import (
        setup_publication_style, save_figure,
        plot_km_survival, plot_time_dependent_auc,
        plot_calibration_curve as pub_plot_calibration,
        plot_decision_curve as pub_plot_dca,
        plot_composite_panel,
    )
    HAS_PUB_PLOTS = True
except ImportError:
    HAS_PUB_PLOTS = False

try:
    from shared_utils import (
        ESSENTIAL_DIR,
        RANDOM_SEED,
        RESULTS_DIR,
        SCRIPT_DIR,
        normalize_patient_id,
    )
except ImportError:  # pragma: no cover - supports importlib loading in tests
    _HERE = Path(__file__).resolve().parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    from shared_utils import (  # type: ignore[no-redef]
        ESSENTIAL_DIR,
        RANDOM_SEED,
        RESULTS_DIR,
        SCRIPT_DIR,
        normalize_patient_id,
    )


LOGGER = logging.getLogger("cmib_surv")
SOURCE_CANCERS = ("STAD", "ESCA", "PAAD", "BRCA", "LUAD", "KIRC", "LIHC", "UCEC")
_GATE_PRIOR_ALPHA: float = 0.02
_GATE_PRIOR_MAX_ITER: int = 2000
DEFAULT_CLINICAL_COLUMNS = ("AGE", "SEX", "AJCC_STAGE_ENCODED")
DEFAULT_TARGET_EXPRESSION = Path(ESSENTIAL_DIR) / "gene_expression_curated.tsv"
DEFAULT_TARGET_CLINICAL = Path(ESSENTIAL_DIR) / "tcga_os_clinical_endpoint_qc.tsv"
DEFAULT_SPLIT_FILE = (
    Path(SCRIPT_DIR).resolve().parents[1]
    / "DATA"
    / "survival_models"
    / "tcga_train_internal_validation_split.tsv"
)
DEFAULT_SOURCE_DIR = Path(SCRIPT_DIR).resolve().parents[1] / "rawData" / "multicancer_preprocessed"
EPS = 1e-8


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch without mutating process-wide threads."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_ids(ids: Sequence[str]) -> str:
    return sha256_bytes("\n".join(sorted(map(str, ids))).encode("utf-8"))


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if dataclasses.is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


@dataclass
class CMIBConfig:
    """Serializable runtime configuration.

    Standard mode implements the requested 5x3 nested CV.  Smoke mode keeps
    identical data boundaries but reduces candidates and epochs for CI.
    """

    profile: str = "standard"
    scope: str = "development"
    causal_mode: str = "prognostic_invariance"
    seed: int = RANDOM_SEED
    device: str = "cpu"
    source_cancers: tuple[str, ...] = SOURCE_CANCERS
    clinical_columns: tuple[str, ...] = DEFAULT_CLINICAL_COLUMNS
    gene_panel_mode: str = "strict"
    min_source_presence: int = 8
    max_missing_gene_fraction: float = 0.02
    projection_dim: int = 64
    meta_hidden_dim: int = 48
    meta_dim: int = 24
    invariant_hidden_dim: int = 32
    invariant_dim: int = 16
    clinical_dim: int = 8
    adapter_rank: int = 4
    fusion_dim: int = 32
    ib_dim: int = 16
    active_gene_target: int = 64
    dropout: float = 0.15
    gate_temperature: float = 2.0 / 3.0
    gate_gamma: float = -0.1
    gate_zeta: float = 1.1
    min_branch_weight: float = 0.05
    cox_weight: float = 1.0
    rank_weight: float = 0.10
    kl_beta: float = 1e-4
    kl_warmup_epochs: int = 15
    kl_free_bits: float = 0.01
    gate_l0_weight: float = 2e-4
    gate_count_weight: float = 2e-3
    gate_bootstraps: int = 30
    source_gate_bootstraps: int = 15
    elastic_l1: float = 2e-5
    elastic_l2: float = 2e-5
    domain_weight: float = 0.02
    coral_weight: float = 0.01
    orthogonality_weight: float = 0.01
    consistency_weight: float = 0.05
    anchor_weight: float = 1e-4
    fusion_entropy_weight: float = 0.02
    treatment_mmd_weight: float = 0.02
    causal_distill_weight: float = 0.05
    causal_teacher_epochs: int = 50
    gradient_reversal_weight: float = 0.2
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    max_epochs_stage1: int = 35
    max_epochs_stage2: int = 25
    patience: int = 8
    min_delta: float = 1e-3
    unlock_projection_min_gain: float = 0.005
    gene_mask_probability: float = 0.08
    gene_noise_std: float = 0.03
    outer_folds: int = 5
    inner_folds: int = 3
    final_seeds: tuple[int, ...] = (RANDOM_SEED, RANDOM_SEED + 17, RANDOM_SEED + 43)
    meta_iterations: int = 60
    meta_batch_size: int = 4
    meta_inner_steps: int = 3
    meta_inner_lr: float = 5e-3
    meta_lr: float = 0.10
    episode_support_size: int = 64
    episode_query_size: int = 48
    episode_min_support_events: int = 8
    episode_min_query_events: int = 5
    ensemble_step: float = 0.05
    eval_tau_months: float = 36.0
    bootstrap_repeats: int = 500
    enable_diffusion_qc: bool = False
    diffusion_epochs: int = 100
    diffusion_steps: int = 20
    treatment_file: Optional[str] = None
    output_dir: Optional[str] = None
    unlock_internal_validation: bool = False

    @classmethod
    def for_profile(cls, profile: str, **overrides: Any) -> "CMIBConfig":
        config = cls(profile=profile)
        if profile == "smoke":
            config = replace(
                config,
                outer_folds=2,
                inner_folds=2,
                final_seeds=(RANDOM_SEED,),
                meta_iterations=2,
                meta_batch_size=2,
                meta_inner_steps=1,
                episode_support_size=24,
                episode_query_size=16,
                episode_min_support_events=2,
                episode_min_query_events=1,
                max_epochs_stage1=3,
                max_epochs_stage2=2,
                patience=2,
                bootstrap_repeats=20,
                gate_bootstraps=3,
                diffusion_epochs=3,
                diffusion_steps=4,
                causal_teacher_epochs=3,
            )
        elif profile != "standard":
            raise ValueError(f"Unknown profile: {profile!r}")
        for key, value in overrides.items():
            if not hasattr(config, key):
                raise KeyError(f"Unknown configuration key: {key}")
            setattr(config, key, value)
        config.validate()
        return config

    def validate(self) -> None:
        if self.scope not in {"development", "locked"}:
            raise ValueError("scope must be 'development' or 'locked'")
        if self.causal_mode not in {"prognostic_invariance", "treatment_causal"}:
            raise ValueError("Unsupported causal mode")
        if self.scope == "locked" and not self.unlock_internal_validation:
            raise PermissionError(
                "Locked validation access requires --unlock-internal-validation"
            )
        if self.causal_mode == "treatment_causal" and not self.treatment_file:
            raise ValueError("treatment_causal mode requires --treatment-file")
        if not 0.0 <= self.min_branch_weight < 1.0 / 3.0:
            raise ValueError("min_branch_weight must be in [0, 1/3)")
        if not 0.0 < self.meta_lr <= 1.0 or self.meta_inner_lr <= 0:
            raise ValueError("Invalid Reptile learning rates")

    def candidates(self) -> list[dict[str, Any]]:
        if self.profile == "smoke":
            return [
                {
                    "active_gene_target": self.active_gene_target,
                    "ib_dim": self.ib_dim,
                    "kl_beta": self.kl_beta,
                }
            ]
        return [
            {"active_gene_target": k, "ib_dim": d, "kl_beta": beta}
            for k, (d, beta) in itertools.product(
                (32, 64, 96),
                ((8, 1e-4), (16, 1e-4), (16, 5e-4), (24, 5e-4)),
            )
        ]


@dataclass(frozen=True)
class GeneRegistry:
    genes: tuple[str, ...]
    target_gene_count: int
    source_presence: Mapping[str, int]
    source_cancers: tuple[str, ...]
    mode: str = "strict"

    def to_dict(self) -> dict[str, Any]:
        return {
            "genes": list(self.genes),
            "n_genes": len(self.genes),
            "target_gene_count": self.target_gene_count,
            "source_presence": dict(self.source_presence),
            "source_cancers": list(self.source_cancers),
            "mode": self.mode,
            "gene_order_sha256": sha256_bytes("\n".join(self.genes).encode("utf-8")),
        }


@dataclass
class SurvivalCohort:
    patient_id: list[str]
    genes: pd.DataFrame
    clinical: pd.DataFrame
    clinical_mask: pd.DataFrame
    time: np.ndarray
    event: np.ndarray
    task_id: np.ndarray
    task_name: str
    treatment: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        n = len(self.patient_id)
        arrays = (self.time, self.event, self.task_id)
        if any(len(item) != n for item in arrays):
            raise ValueError("Cohort arrays have inconsistent lengths")
        if len(self.genes) != n or len(self.clinical) != n or len(self.clinical_mask) != n:
            raise ValueError("Cohort frames have inconsistent lengths")
        normalized = [normalize_patient_id(value) for value in self.patient_id]
        if any(not value for value in normalized):
            raise ValueError("Empty patient ID after normalization")
        if len(set(normalized)) != n:
            raise ValueError("Duplicate patient IDs are not permitted")
        if not np.all(np.isfinite(self.time)) or np.any(self.time <= 0):
            raise ValueError("Survival times must be finite and positive")
        if not set(np.unique(self.event)).issubset({0, 1, False, True}):
            raise ValueError("Event must be binary")
        self.patient_id = normalized
        self.genes = self.genes.copy()
        self.clinical = self.clinical.copy()
        self.clinical_mask = self.clinical_mask.copy()
        self.time = np.ascontiguousarray(self.time, dtype=float)
        self.event = np.ascontiguousarray(self.event, dtype=int)
        self.task_id = np.ascontiguousarray(self.task_id, dtype=int)
        if self.treatment is not None:
            self.treatment = np.ascontiguousarray(self.treatment, dtype=float)
        self.genes.index = normalized
        self.clinical.index = normalized
        self.clinical_mask.index = normalized

    def subset(self, indices: Sequence[int] | np.ndarray) -> "SurvivalCohort":
        idx = np.asarray(indices, dtype=int)
        treatment = None if self.treatment is None else np.asarray(self.treatment)[idx]
        return SurvivalCohort(
            patient_id=[self.patient_id[i] for i in idx],
            genes=self.genes.iloc[idx].copy(),
            clinical=self.clinical.iloc[idx].copy(),
            clinical_mask=self.clinical_mask.iloc[idx].copy(),
            time=np.asarray(self.time, dtype=float)[idx],
            event=np.asarray(self.event, dtype=int)[idx],
            task_id=np.asarray(self.task_id, dtype=int)[idx],
            task_name=self.task_name,
            treatment=treatment,
        )


@dataclass(frozen=True)
class SurvivalRecordBatch:
    patient_id: list[str]
    x_gene: torch.FloatTensor
    x_clinical: torch.FloatTensor
    clinical_mask: torch.BoolTensor
    task_id: torch.LongTensor
    time: torch.FloatTensor
    event: torch.BoolTensor
    treatment: Optional[torch.Tensor] = None
    sample_weight: Optional[torch.FloatTensor] = None
    origin_patient_id: Optional[list[str]] = None
    is_synthetic: Optional[torch.BoolTensor] = None

    def __post_init__(self) -> None:
        n = len(self.patient_id)
        tensors = (
            self.x_gene,
            self.x_clinical,
            self.clinical_mask,
            self.task_id,
            self.time,
            self.event,
        )
        if any(tensor.shape[0] != n for tensor in tensors):
            raise ValueError("Batch fields have inconsistent first dimensions")
        if torch.any(self.time <= 0) or not torch.isfinite(self.time).all():
            raise ValueError("Batch contains invalid survival times")
        if len(set(self.patient_id)) != n:
            raise ValueError("Batch patient IDs must be unique")

    def to(self, device: str | torch.device) -> "SurvivalRecordBatch":
        return SurvivalRecordBatch(
            patient_id=list(self.patient_id),
            x_gene=self.x_gene.to(device),
            x_clinical=self.x_clinical.to(device),
            clinical_mask=self.clinical_mask.to(device),
            task_id=self.task_id.to(device),
            time=self.time.to(device),
            event=self.event.to(device),
            treatment=None if self.treatment is None else self.treatment.to(device),
            sample_weight=None if self.sample_weight is None else self.sample_weight.to(device),
            origin_patient_id=(
                list(self.origin_patient_id) if self.origin_patient_id is not None else None
            ),
            is_synthetic=(
                None if self.is_synthetic is None else self.is_synthetic.to(device)
            ),
        )


def _read_header(path: Path) -> list[str]:
    frame = pd.read_csv(path, sep="\t", nrows=0)
    if frame.shape[1] < 2:
        raise ValueError(f"Expression file has no gene columns: {path}")
    return [str(column).strip() for column in frame.columns[1:]]


def build_gene_registry(
    target_expression_path: str | Path = DEFAULT_TARGET_EXPRESSION,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    source_cancers: Sequence[str] = SOURCE_CANCERS,
    mode: str = "strict",
    min_source_presence: Optional[int] = None,
) -> GeneRegistry:
    """Build a deterministic target-measured cross-cancer gene registry."""

    target_path = Path(target_expression_path)
    source_path = Path(source_dir)
    if not target_path.exists():
        raise FileNotFoundError(target_path)
    target_genes = _read_header(target_path)
    if len(target_genes) != len(set(target_genes)):
        raise ValueError("Target expression contains duplicate gene columns")
    required = len(source_cancers) if mode == "strict" else (min_source_presence or 6)
    if required < 1 or required > len(source_cancers):
        raise ValueError("Invalid source-presence threshold")
    presence = {gene: 0 for gene in target_genes}
    for cancer in source_cancers:
        path = source_path / f"{cancer}_gene_expression.tsv"
        if not path.exists():
            raise FileNotFoundError(f"Missing real source task: {path}")
        source_genes = set(_read_header(path))
        for gene in target_genes:
            presence[gene] += int(gene in source_genes)
    genes = tuple(sorted(gene for gene in target_genes if presence[gene] >= required))
    if not genes:
        raise ValueError("Cross-cancer gene registry is empty")
    return GeneRegistry(
        genes=genes,
        target_gene_count=len(target_genes),
        source_presence={gene: presence[gene] for gene in genes},
        source_cancers=tuple(source_cancers),
        mode=mode,
    )


def _read_expression(path: Path, genes: Sequence[str]) -> pd.DataFrame:
    header = pd.read_csv(path, sep="\t", nrows=0)
    id_column = str(header.columns[0])
    available = set(map(str, header.columns[1:]))
    missing = sorted(set(genes) - available)
    if missing:
        raise ValueError(f"{path.name} lacks {len(missing)} registered genes")
    frame = pd.read_csv(path, sep="\t", usecols=[id_column, *genes], low_memory=False)
    ids = frame[id_column].map(normalize_patient_id)
    frame = frame.drop(columns=[id_column]).apply(pd.to_numeric, errors="coerce")
    frame.index = ids
    if frame.index.duplicated().any():
        raise ValueError(f"Duplicate patient IDs in {path}")
    return frame.loc[:, list(genes)]


def _coerce_clinical(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = pd.DataFrame(index=frame.index)
    mask = pd.DataFrame(index=frame.index)
    for column in columns:
        if column not in frame:
            values = pd.Series(np.nan, index=frame.index, dtype=float)
        else:
            values = pd.to_numeric(frame[column], errors="coerce")
        output[column] = values.astype(float)
        mask[column] = np.isfinite(values.to_numpy(dtype=float))
    return output, mask.astype(bool)


def _load_target_ids(split_path: Path) -> tuple[list[str], list[str]]:
    split = pd.read_csv(split_path, sep="\t", usecols=["PATIENT_ID", "split"])
    split["PATIENT_ID"] = split["PATIENT_ID"].map(normalize_patient_id)
    if split["PATIENT_ID"].duplicated().any():
        raise ValueError("Duplicate patients in target split file")
    development = split.loc[split["split"].eq("train"), "PATIENT_ID"].tolist()
    locked = split.loc[split["split"].eq("internal_validation"), "PATIENT_ID"].tolist()
    if set(development) & set(locked):
        raise AssertionError("Development and locked validation cohorts overlap")
    if not development or not locked:
        raise ValueError("Target split file does not contain both required cohorts")
    return development, locked


def assert_disjoint_partitions(partitions: Mapping[str, Sequence[str]]) -> None:
    """Raise when any normalized patient occurs in two logical partitions."""

    normalized = {
        name: {normalize_patient_id(value) for value in values}
        for name, values in partitions.items()
    }
    for left, right in itertools.combinations(normalized, 2):
        overlap = normalized[left] & normalized[right]
        if overlap:
            examples = ", ".join(sorted(overlap)[:3])
            raise ValueError(
                f"Patient overlap between {left} and {right}: {len(overlap)} ({examples})"
            )


def validate_treatment_file(
    path: str | Path,
    cohort_ids: Sequence[str],
    min_known_fraction: float = 0.80,
    min_group_size: int = 30,
) -> pd.Series:
    """Validate an external, real baseline treatment assignment.

    Stage, event status, and any outcome-derived field are explicitly rejected.
    """

    frame = pd.read_csv(path, sep=None, engine="python")
    required = {"PATIENT_ID", "treatment"}
    if not required.issubset(frame.columns):
        raise ValueError("Treatment file must contain PATIENT_ID and treatment")
    forbidden = {"stage", "event", "status", "death", "outcome"}
    suspicious_columns = [
        str(column)
        for column in frame.columns
        if column not in required
        and any(token in str(column).lower() for token in forbidden)
    ]
    if any(token in str(path).lower() for token in forbidden) or suspicious_columns:
        raise ValueError("Treatment input may not be an outcome or stage proxy")
    frame["PATIENT_ID"] = frame["PATIENT_ID"].map(normalize_patient_id)
    if frame["PATIENT_ID"].duplicated().any():
        raise ValueError("Treatment file contains duplicate patients")
    values = pd.to_numeric(frame["treatment"], errors="coerce")
    observed = set(values.dropna().unique())
    if not observed.issubset({0, 1}) or len(observed) != 2:
        raise ValueError("Treatment must be binary with both groups observed")
    series = pd.Series(values.to_numpy(), index=frame["PATIENT_ID"]).reindex(cohort_ids)
    if series.notna().mean() < min_known_fraction:
        raise ValueError("Known treatment fraction is below the required 80%")
    counts = series.dropna().value_counts()
    if counts.min() < min_group_size:
        raise ValueError("Each treatment group must contain at least 30 patients")
    return series.astype(float)


def load_target_development(
    registry: GeneRegistry,
    expression_path: str | Path = DEFAULT_TARGET_EXPRESSION,
    clinical_path: str | Path = DEFAULT_TARGET_CLINICAL,
    split_path: str | Path = DEFAULT_SPLIT_FILE,
    clinical_columns: Sequence[str] = DEFAULT_CLINICAL_COLUMNS,
    scope: str = "development",
    unlock_internal_validation: bool = False,
    treatment_file: Optional[str] = None,
) -> tuple[SurvivalCohort, dict[str, Any]]:
    """Load exactly one target partition and align every field by patient ID."""

    development_ids, locked_ids = _load_target_ids(Path(split_path))
    if scope == "development":
        requested_ids = development_ids
    elif scope == "locked" and unlock_internal_validation:
        requested_ids = locked_ids
    elif scope == "locked":
        raise PermissionError("Locked validation remains sealed")
    else:
        raise ValueError(f"Unknown scope: {scope}")

    expression = _read_expression(Path(expression_path), registry.genes)
    clinical_usecols = [
        "PATIENT_ID",
        "OS_MONTHS",
        "OS_EVENT",
        *clinical_columns,
    ]
    clinical_raw = pd.read_csv(
        clinical_path,
        sep="\t",
        usecols=lambda column: column in set(clinical_usecols),
        low_memory=False,
    )
    clinical_raw["PATIENT_ID"] = clinical_raw["PATIENT_ID"].map(normalize_patient_id)
    if clinical_raw["PATIENT_ID"].duplicated().any():
        raise ValueError("Duplicate patient IDs in target clinical data")
    clinical_raw = clinical_raw.set_index("PATIENT_ID")
    available = set(expression.index) & set(clinical_raw.index)
    aligned_ids = [patient for patient in requested_ids if patient in available]
    excluded_ids = [patient for patient in requested_ids if patient not in available]
    locked_aligned_ids = [patient for patient in locked_ids if patient in available]
    locked_excluded_ids = [patient for patient in locked_ids if patient not in available]
    if not aligned_ids:
        raise ValueError("No target patients align across split, expression, and clinical tables")
    expression = expression.reindex(aligned_ids)
    clinical_raw = clinical_raw.reindex(aligned_ids)
    time_values = pd.to_numeric(clinical_raw["OS_MONTHS"], errors="coerce").to_numpy(float)
    event_values = pd.to_numeric(clinical_raw["OS_EVENT"], errors="coerce").to_numpy(float)
    valid = np.isfinite(time_values) & (time_values > 0) & np.isin(event_values, [0, 1])
    if not valid.all():
        raise ValueError(f"Target partition contains {(~valid).sum()} invalid endpoints")
    clinical, clinical_mask = _coerce_clinical(clinical_raw, clinical_columns)
    treatment = None
    if treatment_file:
        treatment = validate_treatment_file(treatment_file, aligned_ids).to_numpy(float)
    cohort = SurvivalCohort(
        patient_id=list(aligned_ids),
        genes=expression,
        clinical=clinical,
        clinical_mask=clinical_mask,
        time=time_values,
        event=event_values.astype(int),
        task_id=np.full(len(aligned_ids), len(registry.source_cancers), dtype=int),
        task_name="CRC",
        treatment=treatment,
    )
    manifest = {
        "scope": scope,
        "n_samples": len(cohort.patient_id),
        "split_declared_n": len(requested_ids),
        "excluded_missing_target_n": len(excluded_ids),
        "excluded_missing_target_id_sha256": hash_ids(excluded_ids),
        "n_events": int(cohort.event.sum()),
        "patient_id_sha256": hash_ids(cohort.patient_id),
        "locked_patient_id_sha256": hash_ids(locked_ids),
        "locked_n": len(locked_aligned_ids),
        "locked_declared_n": len(locked_ids),
        "locked_excluded_missing_target_n": len(locked_excluded_ids),
        "locked_excluded_missing_target_id_sha256": hash_ids(locked_excluded_ids),
        "locked_validation_accessed": scope == "locked",
        "expression_file_sha256": hash_file(Path(expression_path)),
        "clinical_file_sha256": hash_file(Path(clinical_path)),
        "split_file_sha256": hash_file(Path(split_path)),
    }
    return cohort, manifest


def load_source_tasks(
    registry: GeneRegistry,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    clinical_columns: Sequence[str] = DEFAULT_CLINICAL_COLUMNS,
) -> list[SurvivalCohort]:
    """Load real source cancer tasks; there is intentionally no synthetic fallback."""

    tasks: list[SurvivalCohort] = []
    source_path = Path(source_dir)
    for task_index, cancer in enumerate(registry.source_cancers):
        expression_path = source_path / f"{cancer}_gene_expression.tsv"
        endpoint_path = source_path / f"{cancer}_os_clinical.tsv"
        if not expression_path.exists() or not endpoint_path.exists():
            raise FileNotFoundError(f"Missing real source files for {cancer}")
        expression = _read_expression(expression_path, registry.genes)
        endpoint = pd.read_csv(endpoint_path, sep="\t", low_memory=False)
        id_column = "patient_id" if "patient_id" in endpoint else "PATIENT_ID"
        time_column = "time_months" if "time_months" in endpoint else "OS_MONTHS"
        event_column = "event" if "event" in endpoint else "OS_EVENT"
        endpoint[id_column] = endpoint[id_column].map(normalize_patient_id)
        if endpoint[id_column].duplicated().any():
            raise ValueError(f"Duplicate patient IDs in {endpoint_path}")
        endpoint = endpoint.set_index(id_column)
        common = sorted(set(expression.index) & set(endpoint.index))
        if len(common) < 40:
            raise ValueError(f"Source task {cancer} has only {len(common)} aligned samples")
        endpoint = endpoint.reindex(common)
        expression = expression.reindex(common)
        time_values = pd.to_numeric(endpoint[time_column], errors="coerce").to_numpy(float)
        event_values = pd.to_numeric(endpoint[event_column], errors="coerce").to_numpy(float)
        valid = np.isfinite(time_values) & (time_values > 0) & np.isin(event_values, [0, 1])
        common = [patient for patient, keep in zip(common, valid) if keep]
        expression = expression.loc[common]
        endpoint = endpoint.loc[common]
        time_values = time_values[valid]
        event_values = event_values[valid].astype(int)
        clinical, clinical_mask = _coerce_clinical(endpoint, clinical_columns)
        tasks.append(
            SurvivalCohort(
                patient_id=common,
                genes=expression,
                clinical=clinical,
                clinical_mask=clinical_mask,
                time=time_values,
                event=event_values,
                task_id=np.full(len(common), task_index, dtype=int),
                task_name=cancer,
            )
        )
    if len(tasks) != len(registry.source_cancers):
        raise RuntimeError("Not all real source tasks were loaded")
    return tasks


class FoldPreprocessor:
    """Fold-local winsorisation, median imputation, and robust scaling."""

    def __init__(self, clip_quantiles: tuple[float, float] = (0.01, 0.99)) -> None:
        self.clip_quantiles = clip_quantiles
        self.fitted = False
        self.gene_names: list[str] = []
        self.clinical_names: list[str] = []
        self.gene_median: Optional[np.ndarray] = None
        self.gene_lower: Optional[np.ndarray] = None
        self.gene_upper: Optional[np.ndarray] = None
        self.gene_center: Optional[np.ndarray] = None
        self.gene_scale: Optional[np.ndarray] = None
        self.clinical_median: Optional[np.ndarray] = None
        self.clinical_center: Optional[np.ndarray] = None
        self.clinical_scale: Optional[np.ndarray] = None
        self.fit_patient_ids: list[str] = []

    @staticmethod
    def _finite_median(values: np.ndarray) -> np.ndarray:
        result = np.zeros(values.shape[1], dtype=float)
        for index in range(values.shape[1]):
            finite = values[np.isfinite(values[:, index]), index]
            if finite.size:
                result[index] = float(np.median(finite))
        return result

    @staticmethod
    def _finite_quantile(values: np.ndarray, q: float) -> np.ndarray:
        with np.errstate(all="ignore"):
            result = np.nanquantile(values, q, axis=0)
        return np.where(np.isfinite(result), result, 0.0)

    def fit(
        self,
        train_data: SurvivalCohort | pd.DataFrame,
        gene_registry: Optional[GeneRegistry | Sequence[str]] = None,
    ) -> "FoldPreprocessor":
        if isinstance(train_data, SurvivalCohort):
            genes = train_data.genes
            clinical = train_data.clinical
            patient_ids = train_data.patient_id
        elif isinstance(train_data, pd.DataFrame):
            if gene_registry is None:
                raise ValueError("gene_registry is required for a plain DataFrame")
            names = list(gene_registry.genes if isinstance(gene_registry, GeneRegistry) else gene_registry)
            missing = set(names) - set(train_data.columns)
            if missing:
                raise ValueError(f"Training frame lacks {len(missing)} genes")
            genes = train_data.loc[:, names]
            clinical = pd.DataFrame(index=train_data.index)
            patient_ids = list(map(str, train_data.index))
        else:
            raise TypeError("FoldPreprocessor.fit expects SurvivalCohort or DataFrame")
        if len(patient_ids) < 2:
            raise ValueError("At least two training patients are required")
        self.gene_names = list(genes.columns)
        self.clinical_names = list(clinical.columns)
        x_gene = genes.to_numpy(dtype=float)
        self.gene_median = self._finite_median(x_gene)
        imputed = np.where(np.isfinite(x_gene), x_gene, self.gene_median)
        q_low, q_high = self.clip_quantiles
        self.gene_lower = self._finite_quantile(imputed, q_low)
        self.gene_upper = self._finite_quantile(imputed, q_high)
        clipped = np.clip(imputed, self.gene_lower, self.gene_upper)
        self.gene_center = np.median(clipped, axis=0)
        q25 = np.quantile(clipped, 0.25, axis=0)
        q75 = np.quantile(clipped, 0.75, axis=0)
        self.gene_scale = np.where(q75 - q25 > 1e-6, q75 - q25, 1.0)

        if self.clinical_names:
            x_clinical = clinical.to_numpy(dtype=float)
            self.clinical_median = self._finite_median(x_clinical)
            clinical_imputed = np.where(
                np.isfinite(x_clinical), x_clinical, self.clinical_median
            )
            self.clinical_center = np.median(clinical_imputed, axis=0)
            c25 = np.quantile(clinical_imputed, 0.25, axis=0)
            c75 = np.quantile(clinical_imputed, 0.75, axis=0)
            self.clinical_scale = np.where(c75 - c25 > 1e-6, c75 - c25, 1.0)
        else:
            self.clinical_median = np.zeros(0, dtype=float)
            self.clinical_center = np.zeros(0, dtype=float)
            self.clinical_scale = np.ones(0, dtype=float)
        self.fit_patient_ids = [normalize_patient_id(value) for value in patient_ids]
        self.fitted = True
        return self

    def transform(self, data: SurvivalCohort) -> SurvivalRecordBatch:
        if not self.fitted:
            raise RuntimeError("FoldPreprocessor must be fitted before transform")
        if list(data.genes.columns) != self.gene_names:
            raise ValueError("Gene order differs from fitted preprocessor")
        if list(data.clinical.columns) != self.clinical_names:
            raise ValueError("Clinical order differs from fitted preprocessor")
        assert self.gene_median is not None
        assert self.gene_lower is not None
        assert self.gene_upper is not None
        assert self.gene_center is not None
        assert self.gene_scale is not None
        x_gene = data.genes.to_numpy(dtype=float)
        x_gene = np.where(np.isfinite(x_gene), x_gene, self.gene_median)
        x_gene = np.clip(x_gene, self.gene_lower, self.gene_upper)
        x_gene = (x_gene - self.gene_center) / self.gene_scale
        assert self.clinical_median is not None
        assert self.clinical_center is not None
        assert self.clinical_scale is not None
        x_clinical = data.clinical.to_numpy(dtype=float)
        x_clinical = np.where(
            np.isfinite(x_clinical), x_clinical, self.clinical_median
        )
        x_clinical = (x_clinical - self.clinical_center) / self.clinical_scale
        n = len(data.patient_id)
        return SurvivalRecordBatch(
            patient_id=list(data.patient_id),
            x_gene=torch.as_tensor(x_gene, dtype=torch.float32),
            x_clinical=torch.as_tensor(x_clinical, dtype=torch.float32),
            clinical_mask=torch.as_tensor(
                data.clinical_mask.to_numpy(dtype=bool), dtype=torch.bool
            ),
            task_id=torch.as_tensor(data.task_id, dtype=torch.long),
            time=torch.as_tensor(data.time, dtype=torch.float32),
            event=torch.as_tensor(data.event.astype(bool), dtype=torch.bool),
            treatment=(
                None
                if data.treatment is None
                else torch.as_tensor(data.treatment, dtype=torch.float32)
            ),
            sample_weight=torch.ones(n, dtype=torch.float32),
            origin_patient_id=list(data.patient_id),
            is_synthetic=torch.zeros(n, dtype=torch.bool),
        )

    def state_dict(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("Cannot serialize an unfitted preprocessor")
        payload = {
            "clip_quantiles": list(self.clip_quantiles),
            "gene_names": self.gene_names,
            "clinical_names": self.clinical_names,
            "gene_median": self.gene_median.tolist(),  # type: ignore[union-attr]
            "gene_lower": self.gene_lower.tolist(),  # type: ignore[union-attr]
            "gene_upper": self.gene_upper.tolist(),  # type: ignore[union-attr]
            "gene_center": self.gene_center.tolist(),  # type: ignore[union-attr]
            "gene_scale": self.gene_scale.tolist(),  # type: ignore[union-attr]
            "clinical_median": self.clinical_median.tolist(),  # type: ignore[union-attr]
            "clinical_center": self.clinical_center.tolist(),  # type: ignore[union-attr]
            "clinical_scale": self.clinical_scale.tolist(),  # type: ignore[union-attr]
            "fit_patient_ids": self.fit_patient_ids,
            "fit_patient_id_sha256": hash_ids(self.fit_patient_ids),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["state_sha256"] = sha256_bytes(canonical.encode("utf-8"))
        return payload

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "FoldPreprocessor":
        instance = cls(tuple(state["clip_quantiles"]))
        instance.gene_names = list(state["gene_names"])
        instance.clinical_names = list(state["clinical_names"])
        for name in (
            "gene_median",
            "gene_lower",
            "gene_upper",
            "gene_center",
            "gene_scale",
            "clinical_median",
            "clinical_center",
            "clinical_scale",
        ):
            setattr(instance, name, np.asarray(state[name], dtype=float))
        instance.fit_patient_ids = list(state["fit_patient_ids"])
        instance.fitted = True
        return instance


class HardConcreteSparseGate(nn.Module):
    """Stochastic hard-concrete gates with an analytic expected L0 penalty."""

    def __init__(
        self,
        n_features: int,
        temperature: float = 2.0 / 3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
        target_active: Optional[int] = None,
    ) -> None:
        super().__init__()
        if n_features < 1:
            raise ValueError("n_features must be positive")
        if temperature <= 0 or gamma >= 0 or zeta <= 1:
            raise ValueError("Invalid hard-concrete constants")
        if target_active is not None and not 1 <= target_active <= n_features:
            raise ValueError("target_active must lie within the feature count")
        self.n_features = n_features
        self.temperature = float(temperature)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        initial_probability = (
            min(max(target_active / n_features, 1e-6), 1.0 - 1e-6)
            if target_active is not None
            else 0.5
        )
        offset = self.temperature * math.log(-self.gamma / self.zeta)
        initial_logit = math.log(initial_probability / (1.0 - initial_probability)) + offset
        self.log_alpha = nn.Parameter(torch.full((n_features,), initial_logit))

    def expected_probability(self) -> torch.Tensor:
        offset = self.temperature * math.log(-self.gamma / self.zeta)
        return torch.sigmoid(self.log_alpha - offset)

    def expected_active(self) -> torch.Tensor:
        return self.expected_probability().sum()

    def _sample_gate(self) -> torch.Tensor:
        uniform = torch.rand_like(self.log_alpha).clamp_(1e-6, 1.0 - 1e-6)
        logistic = torch.log(uniform) - torch.log1p(-uniform)
        concrete = torch.sigmoid((logistic + self.log_alpha) / self.temperature)
        stretched = concrete * (self.zeta - self.gamma) + self.gamma
        return stretched.clamp(0.0, 1.0)

    def deterministic_gate(self) -> torch.Tensor:
        concrete = torch.sigmoid(self.log_alpha)
        return (concrete * (self.zeta - self.gamma) + self.gamma).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self._sample_gate() if self.training else self.deterministic_gate()
        return x * gate


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, weight: float) -> torch.Tensor:
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.weight * gradient, None


def gradient_reverse(x: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, weight)


class MetaEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TaskAdapter(nn.Module):
    def __init__(self, input_dim: int, rank: int, n_tasks: int) -> None:
        super().__init__()
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.task_scale = nn.Embedding(n_tasks, rank)
        self.up = nn.Linear(rank, input_dim, bias=False)
        nn.init.zeros_(self.up.weight)
        nn.init.ones_(self.task_scale.weight)

    def forward(self, x: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        safe_id = task_id.clamp(0, self.task_scale.num_embeddings - 1)
        residual = self.up(self.down(x) * self.task_scale(safe_id))
        return x + residual


class ReliabilityFusion(nn.Module):
    def __init__(
        self,
        meta_dim: int,
        invariant_dim: int,
        clinical_dim: int,
        output_dim: int,
        min_weight: float,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Linear(meta_dim, output_dim),
                nn.Linear(invariant_dim, output_dim),
                nn.Linear(clinical_dim, output_dim),
            ]
        )
        self.logits = nn.Parameter(torch.zeros(3))
        self.min_weight = min_weight
        self.norm = nn.LayerNorm(output_dim)

    def weights(self) -> torch.Tensor:
        raw = torch.softmax(self.logits, dim=0)
        return self.min_weight + (1.0 - 3.0 * self.min_weight) * raw

    def forward(
        self,
        meta: torch.Tensor,
        invariant: torch.Tensor,
        clinical: torch.Tensor,
        branch_mode: str = "full",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = [layer(value) for layer, value in zip(self.projections, (meta, invariant, clinical))]
        if branch_mode == "meta_only":
            weights = torch.tensor([1.0, 0.0, 0.0], device=meta.device, dtype=meta.dtype)
        elif branch_mode == "invariant_only":
            weights = torch.tensor([0.0, 1.0, 0.0], device=meta.device, dtype=meta.dtype)
        elif branch_mode == "clinical_only":
            weights = torch.tensor([0.0, 0.0, 1.0], device=meta.device, dtype=meta.dtype)
        elif branch_mode == "full":
            weights = self.weights()
        else:
            raise ValueError(f"Unknown branch mode: {branch_mode}")
        fused = sum(weight * value for weight, value in zip(weights, projected))
        return self.norm(F.gelu(fused)), weights


class VariationalInformationBottleneck(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.mu = nn.Linear(input_dim, latent_dim)
        self.logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.mu(x)
        logvar = self.logvar(x).clamp(-8.0, 6.0)
        if self.training:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            z = mu
        return z, mu, logvar


@dataclass
class CMIBOutput:
    log_risk: torch.Tensor
    z: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z_meta: torch.Tensor
    z_invariant: torch.Tensor
    z_clinical: torch.Tensor
    fused: torch.Tensor
    fusion_weights: torch.Tensor
    domain_logits: torch.Tensor
    gate_probability: torch.Tensor


class CausalMetaIBSurv(nn.Module):
    """Low-capacity causal-invariant and meta-learned survival network."""

    def __init__(
        self,
        n_genes: int,
        n_clinical: int,
        n_tasks: int,
        config: CMIBConfig,
        branch_mode: str = "full",
    ) -> None:
        super().__init__()
        self.config = copy.deepcopy(config)
        self.branch_mode = branch_mode
        self.gene_gate = HardConcreteSparseGate(
            n_genes,
            temperature=config.gate_temperature,
            gamma=config.gate_gamma,
            zeta=config.gate_zeta,
            target_active=config.active_gene_target,
        )
        self.gene_projection = nn.Sequential(
            nn.Linear(n_genes, config.projection_dim),
            nn.LayerNorm(config.projection_dim),
            nn.GELU(),
        )
        clinical_input_dim = n_clinical * 2
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_input_dim, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, config.clinical_dim),
            nn.GELU(),
        )
        self.meta_encoder = MetaEncoder(
            config.projection_dim,
            config.meta_hidden_dim,
            config.meta_dim,
            config.dropout,
        )
        self.task_adapter = TaskAdapter(config.meta_dim, config.adapter_rank, n_tasks)
        self.invariant_encoder = nn.Sequential(
            nn.Linear(config.projection_dim + config.clinical_dim, config.invariant_hidden_dim),
            nn.LayerNorm(config.invariant_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.invariant_hidden_dim, config.invariant_dim),
            nn.LayerNorm(config.invariant_dim),
            nn.GELU(),
        )
        self.fusion = ReliabilityFusion(
            config.meta_dim,
            config.invariant_dim,
            config.clinical_dim,
            config.fusion_dim,
            config.min_branch_weight,
        )
        self.vib = VariationalInformationBottleneck(config.fusion_dim, config.ib_dim)
        self.cox_head = nn.Sequential(
            nn.Linear(config.ib_dim, 8),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(8, 1),
        )
        self.domain_head = nn.Sequential(
            nn.Linear(config.invariant_dim, 16),
            nn.GELU(),
            nn.Linear(16, n_tasks),
        )
        self.n_tasks = n_tasks

    def forward(
        self,
        x_gene: torch.Tensor,
        x_clinical: torch.Tensor,
        clinical_mask: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
        branch_mode: Optional[str] = None,
    ) -> CMIBOutput:
        if task_id is None:
            task_id = torch.zeros(x_gene.shape[0], dtype=torch.long, device=x_gene.device)
        gated = self.gene_gate(x_gene)
        projected = self.gene_projection(gated)
        clinical_input = torch.cat(
            [x_clinical, clinical_mask.to(dtype=x_clinical.dtype)], dim=1
        )
        z_clinical = self.clinical_encoder(clinical_input)
        z_meta = self.task_adapter(self.meta_encoder(projected), task_id)
        z_invariant = self.invariant_encoder(torch.cat([projected, z_clinical], dim=1))
        fused, fusion_weights = self.fusion(
            z_meta,
            z_invariant,
            z_clinical,
            branch_mode=branch_mode or self.branch_mode,
        )
        z, mu, logvar = self.vib(fused)
        log_risk = self.cox_head(z).squeeze(-1)
        domain_logits = self.domain_head(
            gradient_reverse(z_invariant, self.config.gradient_reversal_weight)
        )
        return CMIBOutput(
            log_risk=log_risk,
            z=z,
            mu=mu,
            logvar=logvar,
            z_meta=z_meta,
            z_invariant=z_invariant,
            z_clinical=z_clinical,
            fused=fused,
            fusion_weights=fusion_weights,
            domain_logits=domain_logits,
            gate_probability=self.gene_gate.expected_probability(),
        )

    @torch.no_grad()
    def predict_risk(self, batch: SurvivalRecordBatch) -> np.ndarray:
        was_training = self.training
        self.eval()
        output = self(
            batch.x_gene,
            batch.x_clinical,
            batch.clinical_mask,
            batch.task_id,
        )
        if was_training:
            self.train()
        return output.log_risk.detach().cpu().numpy()

    @torch.no_grad()
    def predict_survival(
        self,
        batch: SurvivalRecordBatch,
        baseline_times: np.ndarray,
        baseline_cumulative_hazard: np.ndarray,
        evaluation_times: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        risk = self.predict_risk(batch)
        times = baseline_times if evaluation_times is None else np.asarray(evaluation_times, float)
        cumulative = np.interp(
            times,
            baseline_times,
            baseline_cumulative_hazard,
            left=0.0,
            right=baseline_cumulative_hazard[-1],
        )
        return np.exp(-np.exp(risk)[:, None] * cumulative[None, :])

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def efron_cox_loss(
    log_risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Negative Cox partial log likelihood with exact Efron tie handling."""

    eta = log_risk.reshape(-1)
    observed_time = time.reshape(-1)
    observed_event = event.reshape(-1)
    if not (eta.numel() == observed_time.numel() == observed_event.numel()):
        raise ValueError("Cox inputs have different lengths")
    if eta.numel() == 0:
        raise ValueError("Cox loss received an empty batch")
    if not torch.isfinite(eta).all() or not torch.isfinite(observed_time).all():
        raise ValueError("Cox inputs must be finite")
    if torch.any(observed_time <= 0):
        raise ValueError("Survival times must be positive")
    if not torch.all((observed_event == 0) | (observed_event == 1)):
        raise ValueError("Event must be binary")
    observed_event = observed_event.bool()
    n_events = int(observed_event.sum().item())
    if n_events == 0:
        raise ValueError("Cox loss requires at least one event")
    order = torch.argsort(observed_time, descending=True, stable=True)
    sorted_time = observed_time[order]
    sorted_event = observed_event[order]
    sorted_eta = eta[order]
    log_prefix_risk = torch.logcumsumexp(sorted_eta, dim=0)
    _, counts = torch.unique_consecutive(sorted_time, return_counts=True)
    partial_log_likelihood = sorted_eta.sum() * 0.0
    start = 0
    one_minus_epsilon = 1.0 - torch.finfo(sorted_eta.dtype).eps
    for count in counts.tolist():
        stop = start + count
        event_eta = sorted_eta[start:stop][sorted_event[start:stop]]
        tied_events = event_eta.numel()
        if tied_events:
            log_risk_set = log_prefix_risk[stop - 1]
            log_tied_risk = torch.logsumexp(event_eta, dim=0)
            fractions = (
                torch.arange(tied_events, device=eta.device, dtype=eta.dtype)
                / float(tied_events)
            )
            ratio = (
                fractions * torch.exp(log_tied_risk - log_risk_set)
            ).clamp(max=one_minus_epsilon)
            denominator = float(tied_events) * log_risk_set + torch.log1p(-ratio).sum()
            partial_log_likelihood = partial_log_likelihood + event_eta.sum() - denominator
        start = stop
    negative = -partial_log_likelihood
    if reduction == "mean":
        return negative / float(n_events)
    if reduction == "sum":
        return negative
    raise ValueError("reduction must be 'mean' or 'sum'")


def comparable_rank_loss(
    log_risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    temperature: float = 1.0,
    tau: Optional[float] = None,
) -> torch.Tensor:
    """Pairwise ranking loss restricted to valid right-censored comparisons."""

    if tau is not None:
        temperature = tau
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    eta = log_risk.reshape(-1)
    observed_time = time.reshape(-1)
    observed_event_raw = event.reshape(-1)
    if not (eta.numel() == observed_time.numel() == observed_event_raw.numel()):
        raise ValueError("Rank-loss inputs have different lengths")
    if not torch.isfinite(eta).all() or not torch.isfinite(observed_time).all():
        raise ValueError("Rank-loss inputs must be finite")
    if torch.any(observed_time <= 0):
        raise ValueError("Survival times must be positive")
    if not torch.all((observed_event_raw == 0) | (observed_event_raw == 1)):
        raise ValueError("Event must be binary")
    observed_event = observed_event_raw.bool()
    comparable = observed_event[:, None] & (observed_time[:, None] < observed_time[None, :])
    if not bool(comparable.any()):
        return eta.sum() * 0.0
    margin = (eta[:, None] - eta[None, :])[comparable] / temperature
    return F.softplus(-margin).mean()


def _squared_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (
        x.square().sum(dim=1, keepdim=True)
        + y.square().sum(dim=1).unsqueeze(0)
        - 2.0 * x @ y.T
    ).clamp_min(0.0)


def weighted_rbf_mmd2(
    x: torch.Tensor,
    y: torch.Tensor,
    weight_x: Optional[torch.Tensor] = None,
    weight_y: Optional[torch.Tensor] = None,
    bandwidths: Optional[Sequence[float] | torch.Tensor] = None,
) -> torch.Tensor:
    """Biased weighted multi-bandwidth RBF MMD, stable for small groups."""

    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or len(x) == 0 or len(y) == 0:
        raise ValueError("MMD requires two non-empty matrices with equal widths")

    def normalize(weight: Optional[torch.Tensor], n: int, reference: torch.Tensor) -> torch.Tensor:
        if weight is None:
            value = torch.ones(n, device=reference.device, dtype=reference.dtype)
        else:
            value = weight.to(reference).reshape(-1)
        if len(value) != n or not torch.isfinite(value).all() or torch.any(value < 0):
            raise ValueError("Invalid MMD weights")
        if float(value.sum().item()) <= 0:
            raise ValueError("MMD weights sum to zero")
        return value / value.sum()

    normalized_x = normalize(weight_x, len(x), x)
    normalized_y = normalize(weight_y, len(y), y)
    if bandwidths is None:
        with torch.no_grad():
            pooled = torch.cat([x, y], dim=0)
            distances = _squared_distance(pooled, pooled)
            positive = distances[distances > 0]
            sigma = (
                torch.sqrt(positive.median()).clamp_min(torch.finfo(x.dtype).eps)
                if positive.numel()
                else x.new_tensor(1.0)
            )
            bandwidth_tensor = torch.stack([0.5 * sigma, sigma, 2.0 * sigma])
    else:
        bandwidth_tensor = torch.as_tensor(
            bandwidths, device=x.device, dtype=x.dtype
        ).clamp_min(torch.finfo(x.dtype).eps)

    def kernel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        exponent = -_squared_distance(left, right).unsqueeze(-1) / (
            2.0 * bandwidth_tensor.square()
        )
        return torch.exp(exponent).mean(dim=-1)

    xx = normalized_x @ kernel(x, x) @ normalized_x
    yy = normalized_y @ kernel(y, y) @ normalized_y
    xy = normalized_x @ kernel(x, y) @ normalized_y
    return xx + yy - 2.0 * xy


def coral_loss(representation: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
    """Mean pairwise covariance discrepancy across observed tasks."""

    groups = torch.unique(task_id)
    covariances: list[torch.Tensor] = []
    for group in groups:
        values = representation[task_id == group]
        if len(values) >= 2:
            centered = values - values.mean(dim=0, keepdim=True)
            covariances.append(centered.T @ centered / float(len(values) - 1))
    if len(covariances) < 2:
        return representation.sum() * 0.0
    differences = [
        (left - right).square().mean()
        for left, right in itertools.combinations(covariances, 2)
    ]
    return torch.stack(differences).mean()


def representation_orthogonality(meta: torch.Tensor, invariant: torch.Tensor) -> torch.Tensor:
    if len(meta) < 2:
        return meta.sum() * 0.0
    meta_centered = F.normalize(meta - meta.mean(dim=0, keepdim=True), dim=0)
    invariant_centered = F.normalize(
        invariant - invariant.mean(dim=0, keepdim=True), dim=0
    )
    cross_covariance = meta_centered.T @ invariant_centered
    return cross_covariance.square().mean()


def variational_kl(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits: float = 0.0,
) -> torch.Tensor:
    per_dimension = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar)
    mean_dimension = per_dimension.mean(dim=0)
    if free_bits > 0:
        mean_dimension = mean_dimension.clamp_min(free_bits)
    return mean_dimension.sum()


def elastic_network_penalty(model: CausalMetaIBSurv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean-normalised L1/L2 penalties for high-signal weight matrices.

    Per the plan we regularise the gene projection, the low-rank task adapter
    and the Cox head kernels only.  Biases and normalisation parameters remain
    unregularised so that they do not compete with AdamW weight decay, and the
    penalty is averaged over parameters to keep the coefficient scale stable
    when the architecture grows.
    """

    selected: list[torch.Tensor] = [
        model.gene_projection[0].weight,
        model.task_adapter.down.weight,
        model.task_adapter.up.weight,
    ]
    for layer in model.cox_head:
        if isinstance(layer, nn.Linear):
            selected.append(layer.weight)
    count = max(len(selected), 1)
    l1 = sum(parameter.abs().mean() for parameter in selected) / count
    l2 = sum(parameter.square().mean() for parameter in selected) / count
    return l1, l2


def fusion_entropy_penalty(
    fusion_weights: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Non-negative deviation from a maximum-entropy fusion weight vector.

    Encourages the meta/causal/clinical gate to stay reasonably balanced
    early in training (plan section 3.5), returning zero when weights are
    uniform and growing towards ``log(K)`` under branch monopoly.
    """

    weights = fusion_weights.reshape(-1)
    if weights.numel() < 2:
        return reference.sum() * 0.0
    normalised = weights.clamp_min(EPS)
    entropy = -(normalised * normalised.log()).sum()
    ceiling = math.log(float(weights.numel()))
    return torch.clamp(ceiling - entropy, min=0.0)


def compute_cmib_loss(
    model: CausalMetaIBSurv,
    output: CMIBOutput,
    batch: SurvivalRecordBatch,
    config: CMIBConfig,
    epoch: int = 0,
    augmented_output: Optional[CMIBOutput] = None,
    anchor_state: Optional[Mapping[str, torch.Tensor]] = None,
    causal_teacher: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Compute the auditable multi-objective CMIB loss."""

    cox = efron_cox_loss(output.log_risk, batch.time, batch.event)
    ranking = comparable_rank_loss(output.log_risk, batch.time, batch.event)
    kl = variational_kl(output.mu, output.logvar, config.kl_free_bits)
    warmup = min(1.0, (epoch + 1) / max(config.kl_warmup_epochs, 1))
    expected_active = model.gene_gate.expected_active()
    n_genes = float(model.gene_gate.n_features)
    gate_l0 = expected_active / n_genes
    gate_count = ((expected_active - config.active_gene_target) / n_genes).square()
    l1, l2 = elastic_network_penalty(model)
    unique_tasks = torch.unique(batch.task_id)
    if len(unique_tasks) >= 2:
        domain = F.cross_entropy(output.domain_logits, batch.task_id)
        coral = coral_loss(output.z_invariant, batch.task_id)
    else:
        domain = output.domain_logits.sum() * 0.0
        coral = output.z_invariant.sum() * 0.0
    orthogonality = representation_orthogonality(output.z_meta, output.z_invariant)
    fusion_entropy = fusion_entropy_penalty(output.fusion_weights, output.log_risk)
    consistency = (
        F.mse_loss(output.log_risk, augmented_output.log_risk)
        if augmented_output is not None
        else output.log_risk.sum() * 0.0
    )
    anchor = output.log_risk.sum() * 0.0
    if anchor_state:
        terms = []
        for name, parameter in model.named_parameters():
            if name in anchor_state and (
                name.startswith("gene_projection") or name.startswith("meta_encoder")
            ):
                terms.append((parameter - anchor_state[name].to(parameter)).square().mean())
        if terms:
            anchor = torch.stack(terms).mean()
    treatment_mmd = output.z_invariant.sum() * 0.0
    causal_distill = output.z_invariant.sum() * 0.0
    if config.causal_mode == "treatment_causal":
        if batch.treatment is None:
            raise ValueError("Strict causal mode requires validated treatment assignments")
        known = torch.isfinite(batch.treatment)
        treated = known & batch.treatment.eq(1)
        control = known & batch.treatment.eq(0)
        if int(treated.sum()) < 2 or int(control.sum()) < 2:
            raise ValueError("Each treatment group needs at least two samples in a fold")
        treatment_mmd = weighted_rbf_mmd2(
            output.z_invariant[treated],
            output.z_invariant[control],
            None if batch.sample_weight is None else batch.sample_weight[treated],
            None if batch.sample_weight is None else batch.sample_weight[control],
        )
        if causal_teacher is None:
            raise ValueError("Strict causal mode requires a fold-local CausalEGM teacher")
        if causal_teacher.shape != output.z_invariant.shape:
            raise ValueError("Causal teacher representation has the wrong shape")
        causal_distill = F.mse_loss(
            F.normalize(output.z_invariant, dim=1),
            F.normalize(causal_teacher.detach(), dim=1),
        )
    total = (
        config.cox_weight * cox
        + config.rank_weight * ranking
        + config.kl_beta * warmup * kl
        + config.gate_l0_weight * gate_l0
        + config.gate_count_weight * gate_count
        + config.elastic_l1 * l1
        + config.elastic_l2 * l2
        + config.domain_weight * domain
        + config.coral_weight * coral
        + config.orthogonality_weight * orthogonality
        + config.consistency_weight * consistency
        + config.anchor_weight * anchor
        + config.fusion_entropy_weight * fusion_entropy
        + config.treatment_mmd_weight * treatment_mmd
        + config.causal_distill_weight * causal_distill
    )
    return {
        "total": total,
        "cox": cox,
        "rank": ranking,
        "kl": kl,
        "kl_warmup": output.log_risk.new_tensor(warmup),
        "gate_l0": gate_l0,
        "gate_count": gate_count,
        "elastic_l1": l1,
        "elastic_l2": l2,
        "domain": domain,
        "coral": coral,
        "orthogonality": orthogonality,
        "consistency": consistency,
        "anchor": anchor,
        "fusion_entropy": fusion_entropy,
        "treatment_mmd": treatment_mmd,
        "causal_distill": causal_distill,
    }


def _harrell_fallback(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    concordant = 0.0
    comparable = 0.0
    for i in range(len(time)):
        if not event[i]:
            continue
        mask = time[i] < time
        comparable += float(mask.sum())
        concordant += float((risk[i] > risk[mask]).sum())
        concordant += 0.5 * float((risk[i] == risk[mask]).sum())
    return float(concordant / comparable) if comparable else float("nan")


def harrell_c_index(time: Sequence[float], event: Sequence[int], risk: Sequence[float]) -> float:
    time_array = np.asarray(time, dtype=float)
    event_array = np.asarray(event, dtype=bool)
    risk_array = np.asarray(risk, dtype=float)
    if HAS_SKSURV:
        try:
            return float(concordance_index_censored(event_array, time_array, risk_array)[0])
        except Exception:
            pass
    return _harrell_fallback(time_array, event_array, risk_array)


def breslow_baseline_hazard(
    time: Sequence[float], event: Sequence[int], log_risk: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate cumulative baseline hazard for deployment survival curves."""

    time_array = np.asarray(time, dtype=float)
    event_array = np.asarray(event, dtype=bool)
    risk = np.exp(np.clip(np.asarray(log_risk, dtype=float), -30.0, 30.0))
    event_times = np.unique(time_array[event_array])
    increments = []
    for value in event_times:
        deaths = int(np.sum(event_array & np.isclose(time_array, value)))
        denominator = risk[time_array >= value].sum()
        increments.append(deaths / max(denominator, EPS))
    return event_times, np.cumsum(np.asarray(increments, dtype=float))


def kaplan_meier_survival_probability(
    time: Sequence[float], event: Sequence[int], query_time: float
) -> float:
    time_array = np.asarray(time, dtype=float)
    event_array = np.asarray(event, dtype=bool)
    survival = 1.0
    for value in np.unique(time_array[(time_array <= query_time) & event_array]):
        at_risk = np.sum(time_array >= value)
        deaths = np.sum((time_array == value) & event_array)
        if at_risk:
            survival *= 1.0 - deaths / at_risk
    return float(survival)


def evaluate_survival_predictions(
    time: Sequence[float],
    event: Sequence[int],
    risk: Sequence[float],
    train_time: Optional[Sequence[float]] = None,
    train_event: Optional[Sequence[int]] = None,
    survival_probability: Optional[np.ndarray] = None,
    survival_times: Optional[np.ndarray] = None,
    tau_months: float = 36.0,
    time_auc_months: Sequence[float] = (12.0, 36.0, 60.0),
) -> dict[str, float]:
    """Evaluate discrimination and, when available, IPCW/IBS calibration.

    ``time_auc_months`` lists the pre-registered follow-up horizons for
    time-dependent AUC (plan section 9.1); values that fall outside the
    censoring-safe window silently record ``nan`` rather than raising.
    """

    time_array = np.asarray(time, dtype=float)
    event_array = np.asarray(event, dtype=bool)
    risk_array = np.asarray(risk, dtype=float)
    metrics: dict[str, float] = {
        "harrell_c": harrell_c_index(time_array, event_array, risk_array),
        "n": float(len(time_array)),
        "events": float(event_array.sum()),
    }
    metrics["uno_c"] = float("nan")
    metrics["ibs"] = float("nan")
    metrics["calibration_slope"] = float("nan")
    metrics["calibration_in_large_36"] = float("nan")
    for horizon in time_auc_months:
        metrics[f"time_auc_{int(round(float(horizon)))}m"] = float("nan")
    if HAS_SKSURV and train_time is not None and train_event is not None:
        train_surv = Surv.from_arrays(
            np.asarray(train_event, dtype=bool), np.asarray(train_time, dtype=float)
        )
        test_surv = Surv.from_arrays(event_array, time_array)
        if np.std(risk_array) > 1e-8:
            try:
                calibration_model = CoxPHSurvivalAnalysis(alpha=1e-6, n_iter=200)
                calibration_model.fit(risk_array.reshape(-1, 1), test_surv)
                metrics["calibration_slope"] = float(calibration_model.coef_[0])
            except Exception:
                pass
        upper = min(
            tau_months,
            float(np.max(time_array)) - 1e-6,
            float(np.max(np.asarray(train_time, dtype=float))) - 1e-6,
        )
        if upper > float(np.min(time_array)):
            try:
                metrics["uno_c"] = float(
                    concordance_index_ipcw(train_surv, test_surv, risk_array, tau=upper)[0]
                )
            except Exception:
                pass
        auc_floor = max(float(np.min(time_array)), float(np.min(np.asarray(train_time, dtype=float))))
        auc_ceiling = min(
            float(np.max(time_array)) - 1e-6,
            float(np.max(np.asarray(train_time, dtype=float))) - 1e-6,
        )
        for horizon in time_auc_months:
            horizon_value = float(horizon)
            if not (auc_floor < horizon_value < auc_ceiling):
                continue
            try:
                auc_values, _ = cumulative_dynamic_auc(
                    train_surv, test_surv, risk_array, times=[horizon_value]
                )
                metrics[f"time_auc_{int(round(horizon_value))}m"] = float(auc_values[0])
            except Exception:
                pass
        if survival_probability is not None and survival_times is not None:
            valid_times = np.asarray(survival_times, dtype=float)
            valid = (
                (valid_times > max(float(np.min(time_array)), float(np.min(train_time))))
                & (valid_times < upper)
            )
            if valid.sum() >= 2:
                try:
                    metrics["ibs"] = float(
                        integrated_brier_score(
                            train_surv,
                            test_surv,
                            np.asarray(survival_probability)[:, valid],
                            valid_times[valid],
                        )
                    )
                except Exception:
                    pass
            if valid_times.size:
                nearest = int(np.argmin(np.abs(valid_times - tau_months)))
                if abs(float(valid_times[nearest]) - tau_months) <= 1.0:
                    predicted_event = 1.0 - float(
                        np.mean(np.asarray(survival_probability)[:, nearest])
                    )
                    observed_event = 1.0 - kaplan_meier_survival_probability(
                        time_array, event_array, tau_months
                    )
                    metrics["calibration_in_large_36"] = predicted_event - observed_event
    return metrics


def bootstrap_c_index(
    time: Sequence[float],
    event: Sequence[int],
    risk: Sequence[float],
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    time_array = np.asarray(time)
    event_array = np.asarray(event)
    risk_array = np.asarray(risk)
    values: list[float] = []
    for _ in range(repeats):
        indices = rng.integers(0, len(time_array), len(time_array))
        value = harrell_c_index(time_array[indices], event_array[indices], risk_array[indices])
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())  # type: ignore[return-value]


def _forward_batch(
    model: CausalMetaIBSurv,
    batch: SurvivalRecordBatch,
    branch_mode: Optional[str] = None,
) -> CMIBOutput:
    return model(
        batch.x_gene,
        batch.x_clinical,
        batch.clinical_mask,
        batch.task_id,
        branch_mode=branch_mode,
    )


def _augment_batch(batch: SurvivalRecordBatch, config: CMIBConfig) -> SurvivalRecordBatch:
    mask = torch.rand_like(batch.x_gene) < config.gene_mask_probability
    noise = torch.randn_like(batch.x_gene) * config.gene_noise_std
    augmented_gene = torch.where(mask, torch.zeros_like(batch.x_gene), batch.x_gene + noise)
    return SurvivalRecordBatch(
        patient_id=list(batch.patient_id),
        x_gene=augmented_gene,
        x_clinical=batch.x_clinical,
        clinical_mask=batch.clinical_mask,
        task_id=batch.task_id,
        time=batch.time,
        event=batch.event,
        treatment=batch.treatment,
        sample_weight=batch.sample_weight,
        origin_patient_id=batch.origin_patient_id,
        is_synthetic=batch.is_synthetic,
    )


def _attach_propensity_weights(batch: SurvivalRecordBatch) -> SurvivalRecordBatch:
    """Fit a fold-only clinical propensity model and attach stabilized weights."""

    if batch.treatment is None:
        raise ValueError("Treatment assignments are required for propensity weighting")
    treatment = batch.treatment.detach().cpu().numpy()
    known = np.isfinite(treatment)
    if known.sum() < 10 or len(np.unique(treatment[known])) != 2:
        raise ValueError("Fold lacks enough known assignments for propensity estimation")
    features = torch.cat(
        [batch.x_clinical, batch.clinical_mask.to(batch.x_clinical.dtype)], dim=1
    ).detach().cpu().numpy()
    estimator = LogisticRegression(C=0.5, max_iter=2000, random_state=0)
    estimator.fit(features[known], treatment[known].astype(int))
    propensity = estimator.predict_proba(features)[:, 1]
    overlap = known & (propensity >= 0.05) & (propensity <= 0.95)
    if overlap.sum() / known.sum() < 0.80:
        raise ValueError("Insufficient treatment propensity overlap in this fold")
    propensity = np.clip(propensity, 0.05, 0.95)
    prevalence = float(np.mean(treatment[known]))
    weights = np.ones(len(treatment), dtype=np.float32)
    weights[known & (treatment == 1)] = prevalence / propensity[known & (treatment == 1)]
    weights[known & (treatment == 0)] = (1.0 - prevalence) / (
        1.0 - propensity[known & (treatment == 0)]
    )
    weights = np.minimum(weights, np.quantile(weights[known], 0.99)).astype(np.float32)
    return SurvivalRecordBatch(
        patient_id=list(batch.patient_id),
        x_gene=batch.x_gene,
        x_clinical=batch.x_clinical,
        clinical_mask=batch.clinical_mask,
        task_id=batch.task_id,
        time=batch.time,
        event=batch.event,
        treatment=batch.treatment,
        sample_weight=torch.as_tensor(weights, device=batch.x_gene.device),
        origin_patient_id=batch.origin_patient_id,
        is_synthetic=batch.is_synthetic,
    )


def _kaplan_meier_rmst(time_values: np.ndarray, event_values: np.ndarray, tau: float) -> float:
    """Area under the Kaplan-Meier curve on [0, tau]."""

    time_array = np.asarray(time_values, dtype=float)
    event_array = np.asarray(event_values, dtype=bool)
    order = np.argsort(time_array, kind="stable")
    time_array = time_array[order]
    event_array = event_array[order]
    survival = 1.0
    area = 0.0
    previous = 0.0
    for value in np.unique(time_array[time_array <= tau]):
        area += survival * max(0.0, float(value) - previous)
        at_risk = int(np.sum(time_array >= value))
        deaths = int(np.sum((time_array == value) & event_array))
        if at_risk > 0:
            survival *= 1.0 - deaths / at_risk
        previous = float(value)
    area += survival * max(0.0, tau - previous)
    return float(area)


def rmst_pseudo_observations(
    time_values: Sequence[float],
    event_values: Sequence[int],
    tau: float = 36.0,
) -> np.ndarray:
    """Jackknife RMST pseudo-observations with right-censoring correction."""

    time_array = np.asarray(time_values, dtype=float)
    event_array = np.asarray(event_values, dtype=int)
    if len(time_array) != len(event_array) or len(time_array) < 3:
        raise ValueError("RMST pseudo-observations require at least three aligned records")
    if tau <= 0 or np.any(time_array <= 0) or not np.isin(event_array, [0, 1]).all():
        raise ValueError("Invalid RMST inputs")
    full = _kaplan_meier_rmst(time_array, event_array, tau)
    n = len(time_array)
    pseudo = np.empty(n, dtype=float)
    for index in range(n):
        keep = np.arange(n) != index
        leave_one_out = _kaplan_meier_rmst(time_array[keep], event_array[keep], tau)
        pseudo[index] = n * full - (n - 1) * leave_one_out
    return pseudo


def fit_fold_causal_teacher(
    batch: SurvivalRecordBatch,
    config: CMIBConfig,
) -> torch.Tensor:
    """Fit CausalEGM on this training fold using corrected RMST pseudo-outcomes."""

    if config.causal_mode != "treatment_causal" or batch.treatment is None:
        raise ValueError("CausalEGM teacher is available only with validated treatment")
    known = torch.isfinite(batch.treatment).detach().cpu().numpy()
    if known.sum() < 20:
        raise ValueError("Too few known treatments for a fold-local causal teacher")
    try:
        from causal_egm_adapter import CausalEGMAdapter
    except ImportError as error:  # pragma: no cover - optional strict mode dependency
        raise ImportError("CausalEGM is required in treatment_causal mode") from error
    feature_values = _elastic_features(batch)
    feature_names = [f"V{index}" for index in range(feature_values.shape[1])]
    frame = pd.DataFrame(feature_values, index=batch.patient_id, columns=feature_names)
    treatment = pd.Series(
        batch.treatment.detach().cpu().numpy(), index=batch.patient_id, name="treatment"
    )
    pseudo_values = rmst_pseudo_observations(
            batch.time.detach().cpu().numpy(),
            batch.event.detach().cpu().numpy(),
            tau=config.eval_tau_months,
        )
    lower, upper = np.quantile(pseudo_values, [0.01, 0.99])
    pseudo = pd.Series(
        np.clip(pseudo_values, lower, upper),
        index=batch.patient_id,
        name="rmst_pseudo",
    )
    adapter = CausalEGMAdapter(
        latent_dim=config.invariant_dim,
        epochs=config.causal_teacher_epochs,
        lr=1e-3,
        batch_size=min(64, int(known.sum())),
        mode="latent_feature",
        random_state=config.seed,
    )
    adapter.fit(frame.loc[known], treatment.loc[known], pseudo.loc[known])
    teacher = adapter.transform(frame).to_numpy(dtype=np.float32)
    if teacher.shape != (len(batch.patient_id), config.invariant_dim):
        raise RuntimeError(f"Unexpected CausalEGM teacher shape: {teacher.shape}")
    if not np.isfinite(teacher).all():
        raise FloatingPointError("CausalEGM teacher contains non-finite values")
    return torch.as_tensor(teacher, device=batch.x_gene.device, dtype=batch.x_gene.dtype)


def _stratification_labels(
    time_values: Sequence[float],
    event_values: Sequence[int],
    n_splits: int,
) -> np.ndarray:
    """Event x log-time labels, falling back to event when cells are sparse."""

    time_array = np.asarray(time_values, dtype=float)
    event_array = np.asarray(event_values, dtype=int)
    try:
        bins = pd.qcut(np.log1p(time_array), q=3, labels=False, duplicates="drop")
        labels = np.asarray([f"{event}|{int(bin_id)}" for event, bin_id in zip(event_array, bins)])
        counts = pd.Series(labels).value_counts()
        if len(counts) >= 2 and int(counts.min()) >= n_splits:
            return labels
    except (ValueError, TypeError):
        pass
    counts = pd.Series(event_array).value_counts()
    if len(counts) < 2 or int(counts.min()) < n_splits:
        raise ValueError(f"Cannot form {n_splits} survival-stratified folds")
    return event_array.astype(str)


def survival_stratified_splits(
    time_values: Sequence[float],
    event_values: Sequence[int],
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = _stratification_labels(time_values, event_values, n_splits)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    placeholder = np.zeros(len(labels))
    return [(train, validation) for train, validation in splitter.split(placeholder, labels)]


def _sample_episode(
    task: SurvivalCohort,
    support_size: int,
    query_size: int,
    min_support_events: int,
    min_query_events: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    events = np.flatnonzero(task.event == 1)
    censored = np.flatnonzero(task.event == 0)
    if len(events) < min_support_events + min_query_events:
        raise ValueError(f"Task {task.task_name} has too few events for an episode")
    rng.shuffle(events)
    rng.shuffle(censored)
    support_event_count = min(
        len(events) - min_query_events,
        max(min_support_events, int(round(support_size * len(events) / len(task.event)))),
    )
    query_event_count = min(
        len(events) - support_event_count,
        max(min_query_events, int(round(query_size * len(events) / len(task.event)))),
    )
    support_censored_count = min(len(censored), max(0, support_size - support_event_count))
    query_censored_count = min(
        len(censored) - support_censored_count,
        max(0, query_size - query_event_count),
    )
    support = np.concatenate(
        [events[:support_event_count], censored[:support_censored_count]]
    )
    query = np.concatenate(
        [
            events[support_event_count : support_event_count + query_event_count],
            censored[
                support_censored_count : support_censored_count + query_censored_count
            ],
        ]
    )
    if len(support) < 2 or len(query) < 2 or set(support) & set(query):
        raise RuntimeError("Could not construct disjoint support/query episode")
    rng.shuffle(support)
    rng.shuffle(query)
    return support, query


def concatenate_batches(batches: Sequence[SurvivalRecordBatch]) -> SurvivalRecordBatch:
    if not batches:
        raise ValueError("Cannot concatenate an empty batch list")
    treatment = None
    if all(batch.treatment is not None for batch in batches):
        treatment = torch.cat([batch.treatment for batch in batches if batch.treatment is not None])
    return SurvivalRecordBatch(
        patient_id=[patient for batch in batches for patient in batch.patient_id],
        x_gene=torch.cat([batch.x_gene for batch in batches], dim=0),
        x_clinical=torch.cat([batch.x_clinical for batch in batches], dim=0),
        clinical_mask=torch.cat([batch.clinical_mask for batch in batches], dim=0),
        task_id=torch.cat([batch.task_id for batch in batches], dim=0),
        time=torch.cat([batch.time for batch in batches], dim=0),
        event=torch.cat([batch.event for batch in batches], dim=0),
        treatment=treatment,
        sample_weight=torch.cat(
            [
                batch.sample_weight
                if batch.sample_weight is not None
                else torch.ones(len(batch.patient_id), device=batch.x_gene.device)
                for batch in batches
            ],
            dim=0,
        ),
        origin_patient_id=[
            patient
            for batch in batches
            for patient in (batch.origin_patient_id or batch.patient_id)
        ],
        is_synthetic=torch.cat(
            [
                batch.is_synthetic
                if batch.is_synthetic is not None
                else torch.zeros(
                    len(batch.patient_id), dtype=torch.bool, device=batch.x_gene.device
                )
                for batch in batches
            ],
            dim=0,
        ),
    )


def reptile_parameter_update(
    initial_state: Mapping[str, torch.Tensor],
    adapted_states: Sequence[Mapping[str, torch.Tensor]],
    meta_lr: float,
    parameter_names: Optional[Sequence[str]] = None,
) -> dict[str, torch.Tensor]:
    """Pure Reptile update used by the trainer and closed-form tests."""

    if not adapted_states:
        raise ValueError("Reptile requires at least one adapted task state")
    allowed = None if parameter_names is None else set(parameter_names)
    updated: dict[str, torch.Tensor] = {}
    for name, initial in initial_state.items():
        if allowed is not None and name not in allowed:
            updated[name] = initial.detach().clone()
            continue
        if not torch.is_floating_point(initial):
            updated[name] = initial.detach().clone()
            continue
        deltas = []
        for state in adapted_states:
            if name not in state:
                raise KeyError(f"Adapted Reptile state lacks {name}")
            deltas.append(state[name].to(initial) - initial)
        updated[name] = initial + float(meta_lr) * torch.stack(deltas).mean(dim=0)
    return updated


def reptile_update_(
    model: nn.Module,
    adapted_states: Sequence[Mapping[str, torch.Tensor]],
    meta_lr: float,
    parameter_names: Optional[Sequence[str]] = None,
) -> None:
    """Apply a correct in-place meta-batch Reptile interpolation."""

    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    updated = reptile_parameter_update(initial, adapted_states, meta_lr, parameter_names)
    model.load_state_dict(updated, strict=True)


def _meta_parameter_names(model: CausalMetaIBSurv) -> list[str]:
    prefixes = ("gene_projection", "meta_encoder", "vib", "cox_head")
    return [name for name, _ in model.named_parameters() if name.startswith(prefixes)]


def _task_sampling_probabilities(
    source_tasks: Sequence[SurvivalCohort], quantile_cap: float = 0.90
) -> np.ndarray:
    """Return sqrt-event weighted, quantile-capped task sampling probabilities.

    Plan section 3.3 requires balancing across cancers by the square root of
    their event counts while capping the largest tasks so that BRCA/LUAD do
    not dominate low-event UCEC.  Tasks without events fall back to a small
    positive floor so that the pipeline still exposes them.
    """

    events = np.asarray(
        [max(int(task.event.sum()), 1) for task in source_tasks], dtype=float
    )
    weights = np.sqrt(events)
    if quantile_cap < 1.0 and weights.size > 1:
        cap = float(np.quantile(weights, quantile_cap))
        weights = np.minimum(weights, cap)
    total = float(weights.sum())
    if total <= 0:
        return np.full(len(source_tasks), 1.0 / len(source_tasks), dtype=float)
    return weights / total


def reptile_meta_train(
    model: CausalMetaIBSurv,
    source_tasks: Sequence[SurvivalCohort],
    config: CMIBConfig,
) -> tuple[CausalMetaIBSurv, pd.DataFrame]:
    """Meta-train from a common theta0 using disjoint support/query episodes."""

    if len(source_tasks) < 3:
        raise ValueError("At least three real source tasks are required")
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)
    model = model.to(device)
    parameter_names = _meta_parameter_names(model)
    task_probabilities = _task_sampling_probabilities(source_tasks)
    history: list[dict[str, Any]] = []
    for iteration in range(config.meta_iterations):
        selected = rng.choice(
            len(source_tasks),
            size=min(config.meta_batch_size, len(source_tasks)),
            replace=False,
            p=task_probabilities,
        )
        adapted_states: list[Mapping[str, torch.Tensor]] = []
        query_scores: list[float] = []
        task_names: list[str] = []
        invariant_support_batches: list[SurvivalRecordBatch] = []
        for task_index in selected:
            task = source_tasks[int(task_index)]
            support_indices, query_indices = _sample_episode(
                task,
                config.episode_support_size,
                config.episode_query_size,
                config.episode_min_support_events,
                config.episode_min_query_events,
                rng,
            )
            support_cohort = task.subset(support_indices)
            query_cohort = task.subset(query_indices)
            preprocessor = FoldPreprocessor().fit(support_cohort)
            support = preprocessor.transform(support_cohort).to(device)
            query = preprocessor.transform(query_cohort).to(device)
            invariant_support_batches.append(support)
            fast_model = copy.deepcopy(model).to(device)
            for name, parameter in fast_model.named_parameters():
                parameter.requires_grad_(name in parameter_names)
            optimizer = torch.optim.SGD(
                [parameter for parameter in fast_model.parameters() if parameter.requires_grad],
                lr=config.meta_inner_lr,
            )
            fast_model.train()
            for inner_step in range(config.meta_inner_steps):
                optimizer.zero_grad(set_to_none=True)
                output = _forward_batch(fast_model, support, branch_mode="meta_only")
                loss = efron_cox_loss(output.log_risk, support.time, support.event)
                loss = loss + config.rank_weight * comparable_rank_loss(
                    output.log_risk, support.time, support.event
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite Reptile inner loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fast_model.parameters(), max_norm=5.0)
                optimizer.step()
            adapted_states.append(
                {name: value.detach().cpu().clone() for name, value in fast_model.state_dict().items()}
            )
            fast_model.eval()
            query_risk = fast_model.predict_risk(query)
            query_scores.append(
                harrell_c_index(
                    query.time.detach().cpu().numpy(),
                    query.event.detach().cpu().numpy(),
                    query_risk,
                )
            )
            task_names.append(task.task_name)
        reptile_update_(model, adapted_states, config.meta_lr, parameter_names)
        invariant_batch = concatenate_batches(invariant_support_batches)
        invariant_prefixes = ("gene_projection", "invariant_encoder", "domain_head")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(invariant_prefixes))
        invariant_optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=config.meta_inner_lr,
        )
        model.train()
        invariant_optimizer.zero_grad(set_to_none=True)
        invariant_output = _forward_batch(model, invariant_batch, branch_mode="invariant_only")
        domain_objective = F.cross_entropy(
            invariant_output.domain_logits, invariant_batch.task_id
        )
        coral_objective = coral_loss(
            invariant_output.z_invariant, invariant_batch.task_id
        )
        invariance_loss = (
            config.domain_weight * domain_objective
            + config.coral_weight * coral_objective
        )
        invariance_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        invariant_optimizer.step()
        history.append(
            {
                "phase": "meta",
                "iteration": iteration + 1,
                "tasks": ",".join(task_names),
                "query_c_index": float(np.nanmean(query_scores)),
                "domain_loss": float(domain_objective.detach().cpu()),
                "coral_loss": float(coral_objective.detach().cpu()),
            }
        )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model, pd.DataFrame(history)


@dataclass
class TargetFitResult:
    model: CausalMetaIBSurv
    preprocessor: FoldPreprocessor
    validation_risk: np.ndarray
    validation_score: float
    training_history: pd.DataFrame
    baseline_times: np.ndarray
    baseline_cumulative_hazard: np.ndarray
    selected_stage: str


_GATE_PRIOR_CACHE_LIMIT = 64
_GATE_PRIOR_CACHE: dict[str, np.ndarray] = {}


def _cache_gate_prior(cache_key: str, frequencies: np.ndarray) -> None:
    """Insert a fold-local frequency vector with FIFO eviction.

    The cache is bounded so that long nested-CV sweeps do not grow the
    module-level dictionary unboundedly.  Python 3.7+ preserves insertion
    order, so ``next(iter(...))`` yields the oldest key.
    """

    while len(_GATE_PRIOR_CACHE) >= _GATE_PRIOR_CACHE_LIMIT:
        _GATE_PRIOR_CACHE.pop(next(iter(_GATE_PRIOR_CACHE)))
    _GATE_PRIOR_CACHE[cache_key] = frequencies.copy()


def _bootstrap_gate_prior(
    batch: SurvivalRecordBatch,
    target_active: int,
    bootstraps: int,
    seed: int,
    source_frequencies: Optional[np.ndarray] = None,
    target_prior_weight: float = 0.7,
) -> np.ndarray:
    """Training-fold elastic-net Cox stability frequencies for gate initialisation.

    When ``source_frequencies`` are supplied they are blended into the target
    stability vector with ``target_prior_weight`` (plan section 3.2 item 4);
    the target signal keeps the higher weight because it directly reflects the
    CRC endpoint being modelled.
    """

    cache_key = sha256_bytes(
        (
            hash_ids(batch.patient_id)
            + f"|{target_active}|{bootstraps}|{batch.x_gene.shape[1]}|{seed}"
            + (
                "|" + sha256_bytes(np.ascontiguousarray(source_frequencies).tobytes())
                + f"|w={target_prior_weight:.3f}"
                if source_frequencies is not None
                else ""
            )
        ).encode("utf-8")
    )
    if cache_key in _GATE_PRIOR_CACHE:
        return _GATE_PRIOR_CACHE[cache_key].copy()
    x = batch.x_gene.detach().cpu().numpy()
    time_array = batch.time.detach().cpu().numpy()
    event_array = batch.event.detach().cpu().numpy().astype(bool)
    rng = np.random.default_rng(seed)
    selected = np.zeros(x.shape[1], dtype=float)
    successful = 0
    for _ in range(max(1, bootstraps)):
        indices = rng.integers(0, len(x), len(x))
        if event_array[indices].sum() < 2:
            continue
        coefficient: Optional[np.ndarray] = None
        if HAS_SKSURV:
            try:
                estimator = CoxnetSurvivalAnalysis(
                    l1_ratio=0.5,
                    alphas=[_GATE_PRIOR_ALPHA],
                    max_iter=_GATE_PRIOR_MAX_ITER,
                    tol=1e-3,
                )
                outcome = Surv.from_arrays(event_array[indices], time_array[indices])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    estimator.fit(x[indices], outcome)
                coefficient = np.abs(np.asarray(estimator.coef_[:, 0], dtype=float))
            except Exception:
                coefficient = None
        if coefficient is None or not np.isfinite(coefficient).all() or coefficient.max() == 0:
            # 优先 ridge Cox 兜底（保留删失结构），失败再退回单变量相关
            try:
                ridge = CoxPHSurvivalAnalysis(alpha=1.0, n_iter=200)
                ridge_outcome = Surv.from_arrays(event_array[indices], time_array[indices])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ridge.fit(x[indices], ridge_outcome)
                ridge_coef = np.asarray(ridge.coef_, dtype=float)
                if np.isfinite(ridge_coef).all() and ridge_coef.max() > 0:
                    coefficient = np.abs(ridge_coef)
                else:
                    coefficient = None
            except Exception:
                coefficient = None
        if coefficient is None or not np.isfinite(coefficient).all() or coefficient.max() == 0:
            pseudo_risk = event_array[indices].astype(float) / np.maximum(time_array[indices], 1.0)
            centered_x = x[indices] - x[indices].mean(axis=0, keepdims=True)
            centered_y = pseudo_risk - pseudo_risk.mean()
            coefficient = np.abs(centered_x.T @ centered_y)
        top = np.argsort(-coefficient, kind="stable")[: min(target_active, x.shape[1])]
        selected[top] += 1.0
        successful += 1
    if successful == 0:
        frequencies = np.full(x.shape[1], target_active / x.shape[1], dtype=float)
    else:
        frequencies = (selected + 0.5) / (successful + 1.0)
    if source_frequencies is not None:
        if source_frequencies.shape != frequencies.shape:
            raise ValueError(
                "source_frequencies must align with the target gene panel"
            )
        if not np.isfinite(source_frequencies).all():
            raise ValueError("source_frequencies contain non-finite entries")
        weight = float(np.clip(target_prior_weight, 0.0, 1.0))
        frequencies = weight * frequencies + (1.0 - weight) * np.clip(
            source_frequencies, 0.0, 1.0
        )
    _cache_gate_prior(cache_key, frequencies)
    return frequencies.copy()


_SOURCE_GATE_PRIOR: dict[int, np.ndarray] = {}


def register_source_gate_priors(
    priors_by_active: Mapping[int, np.ndarray],
) -> None:
    """Register per-``target_active`` source-cancer gate frequencies.

    Fold-level initialisation blends the target-train bootstrap frequencies
    with these vectors so that the shared gene panel benefits from pan-cancer
    stability evidence (plan section 3.2 item 4).  Callers own the lifetime
    of the registry and must clear it between ablations that must not share
    source information.
    """

    _SOURCE_GATE_PRIOR.clear()
    for active, prior in priors_by_active.items():
        array = np.asarray(prior, dtype=float)
        if array.ndim != 1:
            raise ValueError("Source gate prior must be one-dimensional")
        if not np.isfinite(array).all():
            raise ValueError("Source gate prior contains non-finite entries")
        _SOURCE_GATE_PRIOR[int(active)] = array.copy()


def clear_source_gate_priors() -> None:
    _SOURCE_GATE_PRIOR.clear()


def compute_source_gate_priors(
    source_tasks: Sequence[SurvivalCohort],
    target_actives: Iterable[int],
    bootstraps: int,
    seed: int,
) -> dict[int, np.ndarray]:
    """Aggregate per-gene stability frequencies across source cancers.

    Runs the fold-local elastic-net Cox stability protocol on each source
    task independently for every requested ``target_active`` value, then
    averages the resulting selection-frequency vectors.  The returned dict
    is ready to feed into :func:`register_source_gate_priors`.
    """

    if not source_tasks:
        raise ValueError("At least one source task is required")
    unique_actives = sorted({int(value) for value in target_actives})
    if not unique_actives:
        raise ValueError("target_actives must contain at least one value")
    priors: dict[int, np.ndarray] = {}
    for target_active in unique_actives:
        stack: list[np.ndarray] = []
        for offset, task in enumerate(source_tasks):
            preprocessor = FoldPreprocessor().fit(task)
            batch = preprocessor.transform(task)
            stack.append(
                _bootstrap_gate_prior(batch, target_active, bootstraps, seed + offset)
            )
        priors[target_active] = np.mean(np.stack(stack, axis=0), axis=0)
    return priors


def _fold_gate_prior(
    batch: SurvivalRecordBatch,
    target_active: int,
    bootstraps: int,
    seed: int,
    target_prior_weight: float = 0.7,
) -> np.ndarray:
    """Fold-local gate prior blended with the registered source frequencies.

    Falls back to the pure target-train stability vector when no matching
    source prior has been registered for the requested ``target_active``.
    """

    source_prior = _SOURCE_GATE_PRIOR.get(int(target_active))
    return _bootstrap_gate_prior(
        batch,
        target_active,
        bootstraps,
        seed,
        source_frequencies=source_prior,
        target_prior_weight=target_prior_weight,
    )


def _initialize_gate_from_prior(
    gate: HardConcreteSparseGate,
    frequencies: np.ndarray,
    target_active: int,
) -> None:
    probabilities = np.clip(np.asarray(frequencies, dtype=float), 1e-4, 1.0 - 1e-4)
    base_logits = np.log(probabilities / (1.0 - probabilities))
    low, high = -30.0, 30.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        expected = np.sum(1.0 / (1.0 + np.exp(-(base_logits + midpoint))))
        if expected < target_active:
            low = midpoint
        else:
            high = midpoint
    calibrated = base_logits + (low + high) / 2.0
    offset = gate.temperature * math.log(-gate.gamma / gate.zeta)
    with torch.no_grad():
        gate.log_alpha.copy_(
            torch.as_tensor(calibrated + offset, device=gate.log_alpha.device, dtype=gate.log_alpha.dtype)
        )


def _set_trainable_stage(model: CausalMetaIBSurv, stage: str) -> None:
    """Match the plan's staged unfreezing schedule for target adaptation.

    Stage 1 keeps the shared encoder frozen; only the target-side clinical
    residual, task adapter, fusion, IB and Cox head are trainable.  Stage 2
    additionally unfreezes ``meta_encoder`` and ``invariant_encoder``, while
    stage 3 unlocks the gene gate and projection when the inner CV motivates
    it.  The ``domain_head`` follows the invariant encoder so that the GRL
    branch is trained coherently.
    """

    stage1 = (
        "clinical_encoder",
        "task_adapter",
        "fusion",
        "vib",
        "cox_head",
    )
    stage2 = stage1 + ("meta_encoder", "invariant_encoder", "domain_head")
    stage3 = stage2 + ("gene_gate", "gene_projection")
    prefixes = {"stage1": stage1, "stage2": stage2, "stage3": stage3}[stage]
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))


def _fit_stage(
    model: CausalMetaIBSurv,
    train_batch: SurvivalRecordBatch,
    validation_batch: SurvivalRecordBatch,
    config: CMIBConfig,
    stage: str,
    max_epochs: int,
    anchor_state: Mapping[str, torch.Tensor],
    causal_teacher: Optional[torch.Tensor] = None,
) -> tuple[dict[str, torch.Tensor], float, list[dict[str, Any]]]:
    _set_trainable_stage(model, stage)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(f"No trainable parameters in {stage}")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_score = -np.inf
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = _forward_batch(model, train_batch)
        augmented_output = _forward_batch(model, _augment_batch(train_batch, config))
        losses = compute_cmib_loss(
            model,
            output,
            train_batch,
            config,
            epoch=epoch,
            augmented_output=augmented_output,
            anchor_state=anchor_state,
            causal_teacher=causal_teacher,
        )
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"Non-finite target loss in {stage}")
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
        optimizer.step()
        model.eval()
        validation_risk = model.predict_risk(validation_batch)
        score = harrell_c_index(
            validation_batch.time.detach().cpu().numpy(),
            validation_batch.event.detach().cpu().numpy(),
            validation_risk,
        )
        row: dict[str, Any] = {
            "phase": "target",
            "stage": stage,
            "epoch": epoch + 1,
            "validation_c_index": score,
            "expected_active_genes": float(model.gene_gate.expected_active().detach().cpu()),
        }
        row.update({f"loss_{name}": float(value.detach().cpu()) for name, value in losses.items()})
        history.append(row)
        if np.isfinite(score) and score > best_score + config.min_delta:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break
    model.load_state_dict(best_state)
    return best_state, float(best_score), history


def fit_target_fold(
    train_cohort: SurvivalCohort,
    validation_cohort: SurvivalCohort,
    config: CMIBConfig,
    meta_state: Optional[Mapping[str, torch.Tensor]] = None,
    branch_mode: str = "full",
) -> TargetFitResult:
    """Fit one target fold with staged unfreezing and fold-only preprocessing."""

    assert_disjoint_partitions(
        {"target_train": train_cohort.patient_id, "target_validation": validation_cohort.patient_id}
    )
    seed_everything(config.seed)
    device = torch.device(config.device)
    preprocessor = FoldPreprocessor().fit(train_cohort)
    train_batch = preprocessor.transform(train_cohort).to(device)
    validation_batch = preprocessor.transform(validation_cohort).to(device)
    if config.causal_mode == "treatment_causal":
        train_batch = _attach_propensity_weights(train_batch)
        causal_teacher = fit_fold_causal_teacher(train_batch, config)
    else:
        causal_teacher = None
    n_tasks = int(max(train_cohort.task_id.max(), validation_cohort.task_id.max())) + 1
    model = CausalMetaIBSurv(
        n_genes=train_batch.x_gene.shape[1],
        n_clinical=train_batch.x_clinical.shape[1],
        n_tasks=n_tasks,
        config=config,
        branch_mode=branch_mode,
    ).to(device)
    if meta_state is not None:
        compatible = {
            name: value
            for name, value in meta_state.items()
            if name in model.state_dict() and model.state_dict()[name].shape == value.shape
        }
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        LOGGER.debug("Meta transfer missing=%d unexpected=%d", len(missing), len(unexpected))
    gate_prior = _fold_gate_prior(
        train_batch,
        config.active_gene_target,
        config.gate_bootstraps,
        config.seed,
    )
    _initialize_gate_from_prior(model.gene_gate, gate_prior, config.active_gene_target)
    anchor_state = {
        name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
    }
    all_history: list[dict[str, Any]] = []
    state1, score1, history1 = _fit_stage(
        model,
        train_batch,
        validation_batch,
        config,
        "stage1",
        config.max_epochs_stage1,
        anchor_state,
        causal_teacher,
    )
    all_history.extend(history1)
    selected_state = state1
    selected_score = score1
    selected_stage = "stage1"
    state2, score2, history2 = _fit_stage(
        model,
        train_batch,
        validation_batch,
        config,
        "stage2",
        config.max_epochs_stage2,
        anchor_state,
        causal_teacher,
    )
    all_history.extend(history2)
    if score2 > selected_score:
        selected_state, selected_score, selected_stage = state2, score2, "stage2"
    else:
        model.load_state_dict(selected_state)
    if score2 - score1 >= config.unlock_projection_min_gain:
        state3, score3, history3 = _fit_stage(
            model,
            train_batch,
            validation_batch,
            config,
            "stage3",
            max(2, config.max_epochs_stage2 // 2),
            anchor_state,
            causal_teacher,
        )
        all_history.extend(history3)
        if score3 > selected_score:
            selected_state, selected_score, selected_stage = state3, score3, "stage3"
    model.load_state_dict(selected_state)
    model.eval()
    validation_risk = model.predict_risk(validation_batch)
    train_risk = model.predict_risk(train_batch)
    baseline_times, baseline_cumulative = breslow_baseline_hazard(
        train_cohort.time, train_cohort.event, train_risk
    )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return TargetFitResult(
        model=model,
        preprocessor=preprocessor,
        validation_risk=validation_risk,
        validation_score=harrell_c_index(
            validation_cohort.time, validation_cohort.event, validation_risk
        ),
        training_history=pd.DataFrame(all_history),
        baseline_times=baseline_times,
        baseline_cumulative_hazard=baseline_cumulative,
        selected_stage=selected_stage,
    )


def _elastic_features(batch: SurvivalRecordBatch) -> np.ndarray:
    return torch.cat(
        [
            batch.x_gene,
            batch.x_clinical,
            batch.clinical_mask.to(batch.x_gene.dtype),
        ],
        dim=1,
    ).detach().cpu().numpy()


def fit_elastic_net_cox(
    train_cohort: SurvivalCohort,
    validation_cohort: SurvivalCohort,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit the parsimonious fold-local elastic-net Cox guardrail model."""

    preprocessor = FoldPreprocessor().fit(train_cohort)
    train_batch = preprocessor.transform(train_cohort)
    validation_batch = preprocessor.transform(validation_cohort)
    x_train = _elastic_features(train_batch)
    x_validation = _elastic_features(validation_batch)
    if not HAS_SKSURV:
        raise ImportError("scikit-survival is required for the Cox guardrail")
    outcome = Surv.from_arrays(train_cohort.event.astype(bool), train_cohort.time)
    artifact: dict[str, Any]
    try:
        estimator = CoxnetSurvivalAnalysis(
            l1_ratio=0.5,
            alphas=np.logspace(0.0, -2.0, 16),
            max_iter=100000,
            tol=1e-7,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(x_train, outcome)
        alpha_index = len(estimator.alphas_) // 2
        alpha = float(estimator.alphas_[alpha_index])
        coefficient = np.asarray(estimator.coef_[:, alpha_index], dtype=float)
        kind = "coxnet"
    except Exception as error:
        LOGGER.warning("Coxnet failed (%s); using ridge Cox guardrail", error)
        estimator = CoxPHSurvivalAnalysis(alpha=1.0, n_iter=200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(x_train, outcome)
        coefficient = np.asarray(estimator.coef_, dtype=float)
        alpha = 1.0
        kind = "ridge_cox"
    validation_risk = x_validation @ coefficient
    artifact = {
        "kind": kind,
        "alpha": alpha,
        "coefficient": coefficient.astype(np.float32),
        "preprocessor": preprocessor.state_dict(),
        "feature_count": int(len(coefficient)),
        "seed": seed,
    }
    return validation_risk, artifact


@dataclass
class NestedCVResult:
    oof_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    ablation_metrics: pd.DataFrame
    training_history: pd.DataFrame
    gate_probabilities: pd.DataFrame
    selected_configs: list[dict[str, Any]]
    ensemble_weights_by_outer_fold: dict[str, dict[str, float]] = field(default_factory=dict)


def _candidate_config(config: CMIBConfig, candidate: Mapping[str, Any], seed: int) -> CMIBConfig:
    values = asdict(config)
    values.update(candidate)
    values["seed"] = seed
    return CMIBConfig(**values)


def _fit_fold_risk(
    train_cohort: SurvivalCohort,
    validation_cohort: SurvivalCohort,
    config: CMIBConfig,
    meta_state: Optional[Mapping[str, torch.Tensor]],
    branch_mode: str,
) -> TargetFitResult:
    return fit_target_fold(
        train_cohort,
        validation_cohort,
        config,
        meta_state=meta_state,
        branch_mode=branch_mode,
    )


def _inner_epoch_choice(history: pd.DataFrame, stage: str, fallback: int) -> int:
    subset = history.loc[history["stage"].eq(stage)]
    if subset.empty or not np.isfinite(subset["validation_c_index"]).any():
        return fallback
    best_index = subset["validation_c_index"].astype(float).idxmax()
    return max(1, int(subset.loc[best_index, "epoch"]))


def run_nested_cv(
    cohort: SurvivalCohort,
    config: CMIBConfig,
    meta_state: Optional[Mapping[str, torch.Tensor]] = None,
) -> NestedCVResult:
    """Run leakage-safe 5x3 (or smoke 2x2) target nested cross-validation."""

    seed_everything(config.seed)
    outer_splits = survival_stratified_splits(
        cohort.time, cohort.event, config.outer_folds, config.seed
    )
    risk_names = (
        "full",
        "meta_only",
        "invariant_only",
        "clinical_only",
        "random_init",
        "elastic_net",
    )
    oof = {name: np.full(len(cohort.patient_id), np.nan, dtype=float) for name in risk_names}
    fold_assignment = np.full(len(cohort.patient_id), -1, dtype=int)
    metrics_rows: list[dict[str, Any]] = []
    history_frames: list[pd.DataFrame] = []
    gate_rows: list[dict[str, Any]] = []
    selected_configs: list[dict[str, Any]] = []

    for outer_index, (outer_train_index, outer_validation_index) in enumerate(outer_splits):
        outer_train = cohort.subset(outer_train_index)
        outer_validation = cohort.subset(outer_validation_index)
        inner_splits = survival_stratified_splits(
            outer_train.time,
            outer_train.event,
            config.inner_folds,
            config.seed + outer_index,
        )
        candidate_scores: list[tuple[float, dict[str, Any]]] = []
        candidate_budgets: dict[str, dict[str, Any]] = {}
        for candidate_index, candidate in enumerate(config.candidates()):
            scores: list[float] = []
            stage1_choices: list[int] = []
            stage2_choices: list[int] = []
            selected_stages: list[str] = []
            for inner_train_index, inner_validation_index in inner_splits:
                inner_config = _candidate_config(config, candidate, config.seed)
                result = _fit_fold_risk(
                    outer_train.subset(inner_train_index),
                    outer_train.subset(inner_validation_index),
                    inner_config,
                    meta_state,
                    "full",
                )
                scores.append(result.validation_score)
                stage1_choices.append(
                    _inner_epoch_choice(
                        result.training_history, "stage1", inner_config.max_epochs_stage1
                    )
                )
                stage2_choices.append(
                    _inner_epoch_choice(
                        result.training_history, "stage2", inner_config.max_epochs_stage2
                    )
                )
                selected_stages.append(result.selected_stage)
            mean_score = float(np.nanmean(scores))
            candidate_scores.append((mean_score, dict(candidate)))
            candidate_key = json.dumps(candidate, sort_keys=True)
            candidate_budgets[candidate_key] = {
                "stage1_epochs": max(1, int(round(float(np.median(stage1_choices))))),
                "stage2_epochs": max(1, int(round(float(np.median(stage2_choices))))),
                "run_stage2": float(
                    np.mean([stage in {"stage2", "stage3"} for stage in selected_stages])
                )
                >= 0.5,
                "unlock_projection": float(
                    np.mean([stage == "stage3" for stage in selected_stages])
                )
                >= 0.5,
            }
            metrics_rows.append(
                {
                    "level": "inner_hpo",
                    "outer_fold": outer_index + 1,
                    "candidate": candidate_index + 1,
                    "model": "full",
                    "harrell_c": mean_score,
                    "config": json.dumps(candidate, sort_keys=True),
                }
            )
        candidate_scores.sort(
            key=lambda item: (
                -np.nan_to_num(item[0], nan=-np.inf),
                item[1]["active_gene_target"],
                item[1]["ib_dim"],
            )
        )
        best_candidate = candidate_scores[0][1]
        best_budget = candidate_budgets[json.dumps(best_candidate, sort_keys=True)]
        selected_configs.append(
            {
                "outer_fold": outer_index + 1,
                **best_candidate,
                **best_budget,
                "inner_c": candidate_scores[0][0],
            }
        )

        seed_risks: list[np.ndarray] = []
        seed_gate_probabilities: list[np.ndarray] = []
        for seed in config.final_seeds:
            fold_config = _candidate_config(config, best_candidate, seed + outer_index)
            result = fit_final_development(
                outer_train,
                fold_config,
                meta_state,
                branch_mode="full",
                stage1_epochs=best_budget["stage1_epochs"],
                stage2_epochs=best_budget["stage2_epochs"],
                run_stage2=best_budget["run_stage2"],
                unlock_projection=best_budget["unlock_projection"],
            )
            outer_batch = result.preprocessor.transform(outer_validation).to(fold_config.device)
            seed_risks.append(result.model.predict_risk(outer_batch))
            seed_gate_probabilities.append(
                result.model.gene_gate.expected_probability().detach().cpu().numpy()
            )
            history = result.training_history.copy()
            history["outer_fold"] = outer_index + 1
            history["seed"] = seed
            history["model"] = "full"
            history_frames.append(history)
        oof["full"][outer_validation_index] = np.mean(seed_risks, axis=0)
        mean_gate = np.mean(seed_gate_probabilities, axis=0)
        for gene, probability in zip(cohort.genes.columns, mean_gate):
            gate_rows.append(
                {
                    "outer_fold": outer_index + 1,
                    "gene": gene,
                    "gate_probability": float(probability),
                    "selected": bool(probability >= 0.5),
                }
            )

        ablation_config = _candidate_config(config, best_candidate, config.seed + outer_index)
        for model_name, branch_mode, state in (
            ("meta_only", "meta_only", meta_state),
            ("invariant_only", "invariant_only", meta_state),
            ("clinical_only", "clinical_only", meta_state),
            ("random_init", "full", None),
        ):
            result = fit_final_development(
                outer_train,
                ablation_config,
                state,
                branch_mode=branch_mode,
                stage1_epochs=best_budget["stage1_epochs"],
                stage2_epochs=best_budget["stage2_epochs"],
                run_stage2=best_budget["run_stage2"],
                unlock_projection=best_budget["unlock_projection"],
            )
            outer_batch = result.preprocessor.transform(outer_validation).to(
                ablation_config.device
            )
            oof[model_name][outer_validation_index] = result.model.predict_risk(outer_batch)
            history = result.training_history.copy()
            history["outer_fold"] = outer_index + 1
            history["seed"] = ablation_config.seed
            history["model"] = model_name
            history_frames.append(history)

        elastic_risk, _ = fit_elastic_net_cox(
            outer_train, outer_validation, config.seed + outer_index
        )
        oof["elastic_net"][outer_validation_index] = elastic_risk
        fold_assignment[outer_validation_index] = outer_index + 1
        for model_name in risk_names:
            fold_risk = oof[model_name][outer_validation_index]
            fold_metrics = evaluate_survival_predictions(
                outer_validation.time,
                outer_validation.event,
                fold_risk,
                train_time=outer_train.time,
                train_event=outer_train.event,
                tau_months=config.eval_tau_months,
            )
            metrics_rows.append(
                {
                    "level": "outer_fold",
                    "outer_fold": outer_index + 1,
                    "model": model_name,
                    **fold_metrics,
                    "config": json.dumps(best_candidate, sort_keys=True),
                }
            )

    if np.any(fold_assignment < 0) or any(np.isnan(values).any() for values in oof.values()):
        raise RuntimeError("Nested CV did not produce exactly one prediction per patient")
    prediction_frame = pd.DataFrame(
        {
            "PATIENT_ID": cohort.patient_id,
            "outer_fold": fold_assignment,
            "time_months": cohort.time,
            "event": cohort.event,
            **{f"risk_{name}": values for name, values in oof.items()},
        }
    )
    ablation_rows = []
    for model_name in risk_names:
        metrics = evaluate_survival_predictions(
            cohort.time, cohort.event, oof[model_name], tau_months=config.eval_tau_months
        )
        lower, upper = bootstrap_c_index(
            cohort.time,
            cohort.event,
            oof[model_name],
            config.bootstrap_repeats,
            config.seed,
        )
        ablation_rows.append(
            {
                "model": model_name,
                **metrics,
                "harrell_c_ci_low": lower,
                "harrell_c_ci_high": upper,
            }
        )
    training_history = (
        pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    )
    return NestedCVResult(
        oof_predictions=prediction_frame,
        fold_metrics=pd.DataFrame(metrics_rows),
        ablation_metrics=pd.DataFrame(ablation_rows),
        training_history=training_history,
        gate_probabilities=pd.DataFrame(gate_rows),
        selected_configs=selected_configs,
    )


def _rank_normalize(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return (rankdata(array, method="average") - 0.5) / len(array)


def fit_oof_ensemble(
    predictions: pd.DataFrame,
    time_values: Optional[Sequence[float]] = None,
    event_values: Optional[Sequence[int]] = None,
    columns: Sequence[str] = ("risk_full", "risk_meta_only", "risk_elastic_net"),
    step: float = 0.05,
) -> tuple[dict[str, float], np.ndarray]:
    """Fit non-negative simplex weights on rank-normalized OOF risks."""

    if not set(columns).issubset(predictions.columns):
        raise ValueError("OOF prediction frame lacks requested ensemble columns")
    time_array = np.asarray(
        predictions["time_months"] if time_values is None else time_values, dtype=float
    )
    event_array = np.asarray(
        predictions["event"] if event_values is None else event_values, dtype=int
    )
    matrix = np.column_stack([_rank_normalize(predictions[column]) for column in columns])
    units = int(round(1.0 / step))
    if units < 1 or not np.isclose(units * step, 1.0):
        raise ValueError("Ensemble step must evenly divide one")
    best_score = -np.inf
    best_weight = np.zeros(len(columns), dtype=float)
    for composition in itertools.product(range(units + 1), repeat=len(columns)):
        if sum(composition) != units:
            continue
        weight = np.asarray(composition, dtype=float) / units
        risk = matrix @ weight
        score = harrell_c_index(time_array, event_array, risk)
        if score > best_score + 1e-12:
            best_score = score
            best_weight = weight
    blended = matrix @ best_weight
    weights = {column: float(value) for column, value in zip(columns, best_weight)}
    weights["oof_harrell_c"] = float(best_score)
    return weights, blended


def crossfit_oof_ensemble(
    predictions: pd.DataFrame,
    columns: Sequence[str] = ("risk_full", "risk_meta_only", "risk_elastic_net"),
    step: float = 0.05,
) -> tuple[dict[str, dict[str, float]], np.ndarray]:
    """Evaluate ensemble weights on patients excluded from their weight fit."""

    if "outer_fold" not in predictions:
        raise ValueError("Cross-fitted ensemble requires outer_fold labels")
    risk = np.full(len(predictions), np.nan, dtype=float)
    weights_by_fold: dict[str, dict[str, float]] = {}
    fold_values = np.asarray(predictions["outer_fold"])
    for fold in sorted(pd.unique(fold_values)):
        train_mask = fold_values != fold
        validation_mask = fold_values == fold
        training_frame = predictions.loc[train_mask].reset_index(drop=True)
        weights, _ = fit_oof_ensemble(
            training_frame, columns=columns, step=step
        )
        validation_matrix = np.column_stack(
            [
                _empirical_percentile(
                    training_frame[column], predictions.loc[validation_mask, column]
                )
                for column in columns
            ]
        )
        weight_vector = np.asarray([weights[column] for column in columns], dtype=float)
        risk[validation_mask] = validation_matrix @ weight_vector
        weights_by_fold[str(int(fold))] = weights
    if not np.isfinite(risk).all():
        raise RuntimeError("Cross-fitted ensemble left non-finite predictions")
    return weights_by_fold, risk


@dataclass
class FinalDevelopmentResult:
    model: CausalMetaIBSurv
    preprocessor: FoldPreprocessor
    baseline_times: np.ndarray
    baseline_cumulative_hazard: np.ndarray
    training_history: pd.DataFrame
    train_risk: np.ndarray


def _fit_fixed_stage(
    model: CausalMetaIBSurv,
    batch: SurvivalRecordBatch,
    config: CMIBConfig,
    stage: str,
    epochs: int,
    anchor_state: Mapping[str, torch.Tensor],
    causal_teacher: Optional[torch.Tensor] = None,
) -> list[dict[str, Any]]:
    _set_trainable_stage(model, stage)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: list[dict[str, Any]] = []
    for epoch in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = _forward_batch(model, batch)
        augmented = _forward_batch(model, _augment_batch(batch, config))
        losses = compute_cmib_loss(
            model,
            output,
            batch,
            config,
            epoch=epoch,
            augmented_output=augmented,
            anchor_state=anchor_state,
            causal_teacher=causal_teacher,
        )
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError("Non-finite final development loss")
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        optimizer.step()
        row: dict[str, Any] = {
            "phase": "final_development",
            "stage": stage,
            "epoch": epoch + 1,
            "expected_active_genes": float(model.gene_gate.expected_active().detach().cpu()),
        }
        row.update({f"loss_{key}": float(value.detach().cpu()) for key, value in losses.items()})
        history.append(row)
    return history


def fit_final_development(
    cohort: SurvivalCohort,
    config: CMIBConfig,
    meta_state: Optional[Mapping[str, torch.Tensor]],
    branch_mode: str = "full",
    stage1_epochs: Optional[int] = None,
    stage2_epochs: Optional[int] = None,
    run_stage2: bool = True,
    unlock_projection: bool = False,
) -> FinalDevelopmentResult:
    """Retrain on all development patients with CV-derived fixed epoch counts."""

    seed_everything(config.seed)
    device = torch.device(config.device)
    preprocessor = FoldPreprocessor().fit(cohort)
    batch = preprocessor.transform(cohort).to(device)
    if config.causal_mode == "treatment_causal":
        batch = _attach_propensity_weights(batch)
        causal_teacher = fit_fold_causal_teacher(batch, config)
    else:
        causal_teacher = None
    model = CausalMetaIBSurv(
        batch.x_gene.shape[1],
        batch.x_clinical.shape[1],
        int(cohort.task_id.max()) + 1,
        config,
        branch_mode=branch_mode,
    ).to(device)
    if meta_state is not None:
        compatible = {
            name: value
            for name, value in meta_state.items()
            if name in model.state_dict() and model.state_dict()[name].shape == value.shape
        }
        model.load_state_dict(compatible, strict=False)
    prior = _fold_gate_prior(
        batch, config.active_gene_target, config.gate_bootstraps, config.seed
    )
    _initialize_gate_from_prior(model.gene_gate, prior, config.active_gene_target)
    anchor_state = {
        name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
    }
    history = _fit_fixed_stage(
        model,
        batch,
        config,
        "stage1",
        stage1_epochs or config.max_epochs_stage1,
        anchor_state,
        causal_teacher,
    )
    if run_stage2:
        history.extend(
            _fit_fixed_stage(
                model,
                batch,
                config,
                "stage2",
                stage2_epochs or config.max_epochs_stage2,
                anchor_state,
                causal_teacher,
            )
        )
    if run_stage2 and unlock_projection:
        history.extend(
            _fit_fixed_stage(
                model,
                batch,
                config,
                "stage3",
                max(2, (stage2_epochs or config.max_epochs_stage2) // 2),
                anchor_state,
                causal_teacher,
            )
        )
    model.eval()
    train_risk = model.predict_risk(batch)
    baseline_times, baseline_cumulative = breslow_baseline_hazard(
        cohort.time, cohort.event, train_risk
    )
    return FinalDevelopmentResult(
        model=model,
        preprocessor=preprocessor,
        baseline_times=baseline_times,
        baseline_cumulative_hazard=baseline_cumulative,
        training_history=pd.DataFrame(history),
        train_risk=train_risk,
    )


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_development_bundle(
    path: str | Path,
    registry: GeneRegistry,
    config: CMIBConfig,
    full_results: Sequence[FinalDevelopmentResult],
    meta_result: FinalDevelopmentResult,
    elastic_artifact: Mapping[str, Any],
    ensemble_weights: Mapping[str, float],
    ensemble_reference: Mapping[str, Sequence[float]],
    development_time: Sequence[float],
    development_event: Sequence[int],
    data_manifest: Mapping[str, Any],
) -> Path:
    """Persist only tensors and explicit metadata, never pickled model objects."""

    if not full_results:
        raise ValueError("At least one final full model is required")
    coefficient = elastic_artifact["coefficient"]
    payload = {
        "format_version": 1,
        "model_name": "CausalMetaIBDeepSurv",
        "config": asdict(config),
        "registry": registry.to_dict(),
        "model_spec": {
            "n_genes": len(registry.genes),
            "n_clinical": len(config.clinical_columns),
            "n_tasks": len(registry.source_cancers) + 1,
        },
        "full_model_states": [_cpu_state_dict(result.model) for result in full_results],
        "meta_model_state": _cpu_state_dict(meta_result.model),
        "preprocessor": full_results[0].preprocessor.state_dict(),
        "meta_preprocessor": meta_result.preprocessor.state_dict(),
        "baseline_times": torch.as_tensor(full_results[0].baseline_times),
        "baseline_cumulative_hazard": torch.as_tensor(
            full_results[0].baseline_cumulative_hazard
        ),
        "elastic_artifact": {
            **{key: value for key, value in elastic_artifact.items() if key != "coefficient"},
            "coefficient": torch.as_tensor(coefficient, dtype=torch.float32),
        },
        "ensemble_weights": dict(ensemble_weights),
        "ensemble_reference": {
            key: torch.as_tensor(np.sort(np.asarray(value, dtype=float)), dtype=torch.float32)
            for key, value in ensemble_reference.items()
        },
        "development_time": torch.as_tensor(development_time, dtype=torch.float32),
        "development_event": torch.as_tensor(development_event, dtype=torch.bool),
        "data_manifest": dict(data_manifest),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    return output


def load_development_bundle(path: str | Path) -> dict[str, Any]:
    """Load a trusted local CMIB bundle."""

    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _config_from_payload(payload: Mapping[str, Any]) -> CMIBConfig:
    values = dict(payload)
    for name in ("source_cancers", "clinical_columns", "final_seeds"):
        if name in values:
            values[name] = tuple(values[name])
    return CMIBConfig(**values)


def restore_model_from_bundle(
    bundle: Mapping[str, Any],
    member: int = 0,
    component: str = "full",
) -> CausalMetaIBSurv:
    config = _config_from_payload(bundle["config"])
    spec = bundle["model_spec"]
    branch_mode = "meta_only" if component == "meta" else "full"
    model = CausalMetaIBSurv(
        int(spec["n_genes"]),
        int(spec["n_clinical"]),
        int(spec["n_tasks"]),
        config,
        branch_mode=branch_mode,
    )
    state = (
        bundle["meta_model_state"]
        if component == "meta"
        else bundle["full_model_states"][member]
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _empirical_percentile(reference: Sequence[float], values: Sequence[float]) -> np.ndarray:
    sorted_reference = np.sort(np.asarray(reference, dtype=float))
    values_array = np.asarray(values, dtype=float)
    return (np.searchsorted(sorted_reference, values_array, side="right") - 0.5) / max(
        len(sorted_reference), 1
    )


def predict_from_bundle(
    bundle: Mapping[str, Any],
    cohort: SurvivalCohort,
) -> dict[str, np.ndarray]:
    """Apply frozen preprocessing, neural members, and OOF-calibrated ensemble."""

    preprocessor = FoldPreprocessor.from_state_dict(bundle["preprocessor"])
    batch = preprocessor.transform(cohort)
    full_risks = [
        restore_model_from_bundle(bundle, member=index).predict_risk(batch)
        for index in range(len(bundle["full_model_states"]))
    ]
    full_risk = np.mean(full_risks, axis=0)
    meta_preprocessor = FoldPreprocessor.from_state_dict(bundle["meta_preprocessor"])
    meta_batch = meta_preprocessor.transform(cohort)
    meta_risk = restore_model_from_bundle(bundle, component="meta").predict_risk(meta_batch)
    elastic = bundle["elastic_artifact"]
    elastic_preprocessor = FoldPreprocessor.from_state_dict(elastic["preprocessor"])
    elastic_batch = elastic_preprocessor.transform(cohort)
    coefficient = np.asarray(elastic["coefficient"], dtype=float)
    elastic_risk = _elastic_features(elastic_batch) @ coefficient
    components = {
        "risk_full": full_risk,
        "risk_meta_only": meta_risk,
        "risk_elastic_net": elastic_risk,
    }
    ensemble = np.zeros(len(cohort.patient_id), dtype=float)
    for name, values in components.items():
        reference = np.asarray(bundle["ensemble_reference"][name], dtype=float)
        ensemble += float(bundle["ensemble_weights"].get(name, 0.0)) * _empirical_percentile(
            reference, values
        )
    components["risk_ensemble"] = ensemble
    return components


class LatentDiffusionAugmentor(nn.Module):
    """Small DDPM used only as a fold-local representation quality experiment."""

    def __init__(self, latent_dim: int, steps: int = 20) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.steps = steps
        self.denoiser = nn.Sequential(
            nn.Linear(latent_dim + 1, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, latent_dim),
        )
        beta = torch.linspace(1e-4, 0.02, steps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def _time_feature(self, time_index: torch.Tensor) -> torch.Tensor:
        return time_index.float().unsqueeze(1) / max(self.steps - 1, 1)

    def forward(self, noisy: torch.Tensor, time_index: torch.Tensor) -> torch.Tensor:
        return self.denoiser(torch.cat([noisy, self._time_feature(time_index)], dim=1))

    def fit_latent(
        self,
        latent: torch.Tensor,
        epochs: int,
        seed: int,
        learning_rate: float = 1e-3,
    ) -> list[float]:
        torch.manual_seed(seed)
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        history = []
        self.train()
        for _ in range(epochs):
            time_index = torch.randint(0, self.steps, (len(latent),), device=latent.device)
            noise = torch.randn_like(latent)
            cumulative = self.alpha_bar[time_index].unsqueeze(1)
            noisy = cumulative.sqrt() * latent + (1.0 - cumulative).sqrt() * noise
            predicted = self(noisy, time_index)
            loss = F.mse_loss(predicted, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach().cpu()))
        return history

    @torch.no_grad()
    def sample(self, n: int, seed: int, device: torch.device) -> torch.Tensor:
        generator = torch.Generator(device=device).manual_seed(seed)
        value = torch.randn((n, self.latent_dim), generator=generator, device=device)
        self.eval()
        for index in reversed(range(self.steps)):
            time_index = torch.full((n,), index, dtype=torch.long, device=device)
            predicted_noise = self(value, time_index)
            alpha = self.alpha[index]
            alpha_bar = self.alpha_bar[index]
            mean = (
                value
                - (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar) * predicted_noise
            ) / torch.sqrt(alpha)
            if index > 0:
                noise = torch.randn(value.shape, generator=generator, device=device)
                value = mean + torch.sqrt(self.beta[index]) * noise
            else:
                value = mean
        return value


def _cross_validated_discriminator_auc(real: np.ndarray, synthetic: np.ndarray, seed: int) -> float:
    x = np.vstack([real, synthetic])
    y = np.concatenate([np.zeros(len(real), dtype=int), np.ones(len(synthetic), dtype=int)])
    folds = min(3, int(np.bincount(y).min()))
    if folds < 2:
        return float("nan")
    scores = np.zeros(len(y), dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_index, validation_index in splitter.split(x, y):
        estimator = LogisticRegression(C=0.5, max_iter=2000, random_state=seed)
        estimator.fit(x[train_index], y[train_index])
        scores[validation_index] = estimator.predict_proba(x[validation_index])[:, 1]
    auc = float(roc_auc_score(y, scores))
    return max(auc, 1.0 - auc)


def run_diffusion_quality_gate(
    result: FinalDevelopmentResult,
    cohort: SurvivalCohort,
    config: CMIBConfig,
) -> pd.DataFrame:
    """Train latent diffusion on development only; never assign survival labels."""

    device = torch.device(config.device)
    batch = result.preprocessor.transform(cohort).to(device)
    result.model.eval()
    with torch.no_grad():
        latent = _forward_batch(result.model, batch).mu.detach()
    augmentor = LatentDiffusionAugmentor(latent.shape[1], config.diffusion_steps).to(device)
    losses = augmentor.fit_latent(latent, config.diffusion_epochs, config.seed)
    synthetic_tensor = augmentor.sample(len(latent), config.seed + 991, device)
    real = latent.cpu().numpy()
    synthetic = synthetic_tensor.cpu().numpy()
    mmd = float(weighted_rbf_mmd2(latent, synthetic_tensor).detach().cpu())
    discriminator_auc = _cross_validated_discriminator_auc(real, synthetic, config.seed)
    real_distance = pairwise_distances(real)
    np.fill_diagonal(real_distance, np.inf)
    real_nn = float(np.median(real_distance.min(axis=1)))
    synthetic_nn = float(np.median(pairwise_distances(synthetic, real).min(axis=1)))
    nn_ratio = synthetic_nn / max(real_nn, EPS)
    near_duplicate_fraction = float(
        np.mean(pairwise_distances(synthetic, real).min(axis=1) < max(real_nn * 0.05, 1e-6))
    )
    passed = bool(
        np.isfinite(discriminator_auc)
        and discriminator_auc <= 0.80
        and 0.25 <= nn_ratio <= 4.0
        and near_duplicate_fraction <= 0.01
    )
    return pd.DataFrame(
        [
            {
                "status": "pass" if passed else "fail",
                "use_in_survival_risk_set": False,
                "n_real": len(real),
                "n_synthetic": len(synthetic),
                "latent_dim": real.shape[1],
                "mmd2": mmd,
                "discriminator_auc": discriminator_auc,
                "nearest_neighbor_ratio": nn_ratio,
                "near_duplicate_fraction": near_duplicate_fraction,
                "final_denoising_loss": losses[-1],
            }
        ]
    )


def _select_final_candidate(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(selected)
    keys = ["active_gene_target", "ib_dim", "kl_beta"]
    grouped = (
        frame.groupby(keys, as_index=False)["inner_c"]
        .mean()
        .sort_values(
            ["inner_c", "active_gene_target", "ib_dim"],
            ascending=[False, True, True],
            kind="stable",
        )
    )
    row = grouped.iloc[0]
    return {
        "active_gene_target": int(row["active_gene_target"]),
        "ib_dim": int(row["ib_dim"]),
        "kl_beta": float(row["kl_beta"]),
    }


def _cv_epoch_budget(history: pd.DataFrame, stage: str, fallback: int) -> int:
    if history.empty:
        return fallback
    subset = history[(history["model"] == "full") & (history["stage"] == stage)]
    if subset.empty:
        return fallback
    maxima = subset.groupby(["outer_fold", "seed"])["epoch"].max()
    return max(1, int(round(float(maxima.median()))))


def _source_manifest(tasks: Sequence[SurvivalCohort]) -> list[dict[str, Any]]:
    return [
        {
            "task": task.task_name,
            "n": len(task.patient_id),
            "events": int(task.event.sum()),
            "patient_id_sha256": hash_ids(task.patient_id),
        }
        for task in tasks
    ]


def _registry_from_bundle(payload: Mapping[str, Any]) -> GeneRegistry:
    return GeneRegistry(
        genes=tuple(payload["genes"]),
        target_gene_count=int(payload["target_gene_count"]),
        source_presence={str(key): int(value) for key, value in payload["source_presence"].items()},
        source_cancers=tuple(payload["source_cancers"]),
        mode=str(payload.get("mode", "strict")),
    )


def run_development_pipeline(
    config: CMIBConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute source meta-training, nested target CV, and frozen retraining."""

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    registry = build_gene_registry(
        source_cancers=config.source_cancers,
        mode=config.gene_panel_mode,
        min_source_presence=config.min_source_presence,
    )
    target, target_manifest = load_target_development(
        registry,
        clinical_columns=config.clinical_columns,
        scope="development",
        treatment_file=config.treatment_file,
    )
    source_tasks = load_source_tasks(
        registry, clinical_columns=config.clinical_columns
    )
    _, locked_ids = _load_target_ids(DEFAULT_SPLIT_FILE)
    partitions: dict[str, Sequence[str]] = {
        "target_development": target.patient_id,
        "target_locked_declared": locked_ids,
    }
    partitions.update({f"source_{task.task_name}": task.patient_id for task in source_tasks})
    assert_disjoint_partitions(partitions)
    data_manifest = {
        **target_manifest,
        "source_tasks": _source_manifest(source_tasks),
        "gene_count": len(registry.genes),
        "causal_mode": config.causal_mode,
    }
    atomic_json_dump(asdict(config), output_dir / "config.json")
    atomic_json_dump(data_manifest, output_dir / "data_manifest.json")
    atomic_json_dump(registry.to_dict(), output_dir / "gene_registry.json")

    meta_model = CausalMetaIBSurv(
        len(registry.genes),
        len(config.clinical_columns),
        len(source_tasks) + 1,
        config,
    )
    if meta_model.trainable_parameter_count() > 100_000:
        raise RuntimeError("Target model exceeds the 100k trainable-parameter budget")
    LOGGER.info(
        "Meta-training %d real tasks with %d genes and %d parameters",
        len(source_tasks),
        len(registry.genes),
        meta_model.trainable_parameter_count(),
    )
    meta_model, meta_history = reptile_meta_train(meta_model, source_tasks, config)
    meta_state = _cpu_state_dict(meta_model)
    torch.save(
        {
            "state_dict": meta_state,
            "config": asdict(config),
            "registry": registry.to_dict(),
        },
        output_dir / "meta_checkpoint.pt",
    )

    candidate_actives = {int(item["active_gene_target"]) for item in config.candidates()}
    candidate_actives.add(int(config.active_gene_target))
    LOGGER.info(
        "Computing source-cancer gate priors for target_active values %s",
        sorted(candidate_actives),
    )
    source_priors = compute_source_gate_priors(
        source_tasks,
        target_actives=candidate_actives,
        bootstraps=config.source_gate_bootstraps,
        seed=config.seed,
    )
    register_source_gate_priors(source_priors)
    try:
        nested = run_nested_cv(target, config, meta_state)
    finally:
        clear_source_gate_priors()
    ensemble_weights, tuned_ensemble_risk = fit_oof_ensemble(
        nested.oof_predictions, step=config.ensemble_step
    )
    crossfit_weights, ensemble_risk = crossfit_oof_ensemble(
        nested.oof_predictions, step=config.ensemble_step
    )
    nested.oof_predictions["risk_ensemble"] = ensemble_risk
    nested.oof_predictions["risk_ensemble_deployment_fit"] = tuned_ensemble_risk
    ensemble_metrics = evaluate_survival_predictions(
        target.time, target.event, ensemble_risk, tau_months=config.eval_tau_months
    )
    ensemble_ci = bootstrap_c_index(
        target.time,
        target.event,
        ensemble_risk,
        config.bootstrap_repeats,
        config.seed,
    )
    nested.ablation_metrics = pd.concat(
        [
            nested.ablation_metrics,
            pd.DataFrame(
                [
                    {
                        "model": "oof_ensemble_crossfit",
                        **ensemble_metrics,
                        "harrell_c_ci_low": ensemble_ci[0],
                        "harrell_c_ci_high": ensemble_ci[1],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    atomic_json_dump(
        {
            "deployment_weights": ensemble_weights,
            "crossfit_weights_by_outer_fold": crossfit_weights,
        },
        output_dir / "ensemble_weights.json",
    )
    atomic_json_dump(nested.selected_configs, output_dir / "selected_configs.json")
    nested.oof_predictions.to_csv(output_dir / "oof_predictions.tsv", sep="\t", index=False)
    nested.fold_metrics.to_csv(output_dir / "nested_cv_metrics.tsv", sep="\t", index=False)
    nested.ablation_metrics.to_csv(output_dir / "ablation_metrics.tsv", sep="\t", index=False)
    nested.gate_probabilities.to_csv(
        output_dir / "gate_probabilities.tsv", sep="\t", index=False
    )
    feature_stability = (
        nested.gate_probabilities.groupby("gene", as_index=False)
        .agg(
            mean_gate_probability=("gate_probability", "mean"),
            sd_gate_probability=("gate_probability", "std"),
            selection_frequency=("selected", "mean"),
        )
        .sort_values("mean_gate_probability", ascending=False, kind="stable")
    )
    feature_stability.to_csv(output_dir / "feature_stability.tsv", sep="\t", index=False)

    candidate = _select_final_candidate(nested.selected_configs)
    stage1_epochs = _cv_epoch_budget(
        nested.training_history, "stage1", config.max_epochs_stage1
    )
    stage2_epochs = _cv_epoch_budget(
        nested.training_history, "stage2", config.max_epochs_stage2
    )
    final_run_stage2 = bool(
        np.mean([bool(item.get("run_stage2", True)) for item in nested.selected_configs])
        >= 0.5
    )
    final_unlock_projection = bool(
        np.mean(
            [bool(item.get("unlock_projection", False)) for item in nested.selected_configs]
        )
        >= 0.5
    )
    full_results: list[FinalDevelopmentResult] = []
    final_history_frames: list[pd.DataFrame] = []
    register_source_gate_priors(source_priors)
    try:
        for seed in config.final_seeds:
            final_config = _candidate_config(config, candidate, seed)
            result = fit_final_development(
                target,
                final_config,
                meta_state,
                branch_mode="full",
                stage1_epochs=stage1_epochs,
                stage2_epochs=stage2_epochs,
                run_stage2=final_run_stage2,
                unlock_projection=final_unlock_projection,
            )
            full_results.append(result)
            history = result.training_history.copy()
            history["model"] = "final_full"
            history["seed"] = seed
            final_history_frames.append(history)
        bundle_config = _candidate_config(config, candidate, config.final_seeds[0])
        meta_result = fit_final_development(
            target,
            bundle_config,
            meta_state,
            branch_mode="meta_only",
            stage1_epochs=stage1_epochs,
            stage2_epochs=stage2_epochs,
            run_stage2=final_run_stage2,
            unlock_projection=final_unlock_projection,
        )
    finally:
        clear_source_gate_priors()
    meta_final_history = meta_result.training_history.copy()
    meta_final_history["model"] = "final_meta_only"
    meta_final_history["seed"] = bundle_config.seed
    final_history_frames.append(meta_final_history)
    _, elastic_artifact = fit_elastic_net_cox(target, target, config.seed)
    combined_history = pd.concat(
        [meta_history, nested.training_history, *final_history_frames],
        ignore_index=True,
        sort=False,
    )
    combined_history.to_csv(output_dir / "training_history.tsv", sep="\t", index=False)

    bundle_path = save_development_bundle(
        output_dir / "final_development_bundle.pt",
        registry,
        bundle_config,
        full_results,
        meta_result,
        elastic_artifact,
        ensemble_weights,
        {
            "risk_full": nested.oof_predictions["risk_full"],
            "risk_meta_only": nested.oof_predictions["risk_meta_only"],
            "risk_elastic_net": nested.oof_predictions["risk_elastic_net"],
        },
        target.time,
        target.event,
        data_manifest,
    )
    if config.enable_diffusion_qc:
        diffusion_qc = run_diffusion_quality_gate(full_results[0], target, bundle_config)
    else:
        diffusion_qc = pd.DataFrame(
            [{"status": "disabled", "use_in_survival_risk_set": False}]
        )
    diffusion_qc.to_csv(output_dir / "diffusion_qc.tsv", sep="\t", index=False)
    summary = pd.DataFrame(
        [
            {
                "scope": "development",
                "locked_validation_accessed": False,
                "n": len(target.patient_id),
                "events": int(target.event.sum()),
                "gene_count": len(registry.genes),
                "trainable_parameters": meta_model.trainable_parameter_count(),
                "oof_ensemble_harrell_c": ensemble_metrics["harrell_c"],
                "oof_ensemble_ci_low": ensemble_ci[0],
                "oof_ensemble_ci_high": ensemble_ci[1],
                "target_harrell_c": 0.72,
                "target_met_on_development_oof": bool(ensemble_metrics["harrell_c"] > 0.72),
                "final_candidate": json.dumps(candidate, sort_keys=True),
                "bundle": str(bundle_path),
            }
        ]
    )
    summary.to_csv(output_dir / "development_summary.tsv", sep="\t", index=False)

    # ── 出版级图表生成 (Nature/Cell 风格) ──
    if HAS_SURVIVAL_METRICS and HAS_PUB_PLOTS:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            setup_publication_style()
            pub_dir = output_dir / "publication_quality"
            pub_dir.mkdir(parents=True, exist_ok=True)
            # 使用 OOF ensemble 风险评分
            oof_risk = np.asarray(nested.oof_predictions["risk_ensemble"], dtype=float)
            oof_time = np.asarray(target.time, dtype=float)
            oof_event = np.asarray(target.event, dtype=int)
            evaluator = SurvivalEvaluator(
                oof_time, oof_event, oof_time, oof_event, oof_risk)
            report = evaluator.full_report(
                horizons=(12.0, 36.0, 60.0),
                dca_horizon=config.eval_tau_months,
                calibration_horizon=config.eval_tau_months)
            # 保存完整指标
            atomic_json_dump(
                {k: v for k, v in report.items() if k not in ("calibration_data", "dca_data")},
                pub_dir / "cmib_full_metrics.json")
            # KM 曲线
            fig_km = plot_km_survival(oof_time, oof_event, oof_risk,
                                      title="CMIB-Surv OOF Risk Stratification")
            save_figure(fig_km, str(pub_dir / "cmib_km"))
            plt.close(fig_km)
            # 校准曲线
            cal_df = pd.DataFrame(report.get("calibration_data", []))
            if not cal_df.empty:
                fig_cal = pub_plot_calibration(cal_df, title="CMIB-Surv",
                                               horizon=config.eval_tau_months)
                save_figure(fig_cal, str(pub_dir / "cmib_calibration"))
                plt.close(fig_cal)
            # DCA
            dca_df = pd.DataFrame(report.get("dca_data", []))
            if not dca_df.empty:
                fig_dca = pub_plot_dca(dca_df, title="CMIB-Surv")
                save_figure(fig_dca, str(pub_dir / "cmib_dca"))
                plt.close(fig_dca)
            # 组合面板
            fig_comp = plot_composite_panel(
                oof_time, oof_event, oof_risk,
                train_time=oof_time, train_event=oof_event,
                cal_data=cal_df, dca_data=dca_df,
                title="CMIB-Surv — Comprehensive Evaluation")
            save_figure(fig_comp, str(pub_dir / "cmib_composite"))
            plt.close(fig_comp)
            LOGGER.info("Publication-quality figures saved to %s", pub_dir)
        except Exception as e:
            LOGGER.warning("Publication figure generation failed: %s", e)

    return {
        "output_dir": str(output_dir),
        "bundle": str(bundle_path),
        "oof_harrell_c": ensemble_metrics["harrell_c"],
        "locked_validation_accessed": False,
    }


def run_locked_evaluation(
    config: CMIBConfig,
    output_dir: Path,
    bundle_path: str | Path,
) -> dict[str, Any]:
    """Run one frozen evaluation; no fitting or model selection is permitted."""

    if not config.unlock_internal_validation:
        raise PermissionError("Locked validation has not been explicitly unlocked")
    bundle = load_development_bundle(bundle_path)
    registry = _registry_from_bundle(bundle["registry"])
    model_config = _config_from_payload(bundle["config"])
    locked, manifest = load_target_development(
        registry,
        clinical_columns=model_config.clinical_columns,
        scope="locked",
        unlock_internal_validation=True,
        treatment_file=config.treatment_file,
    )
    predictions = predict_from_bundle(bundle, locked)
    metrics = evaluate_survival_predictions(
        locked.time,
        locked.event,
        predictions["risk_ensemble"],
        train_time=np.asarray(bundle["development_time"], dtype=float),
        train_event=np.asarray(bundle["development_event"], dtype=bool),
        tau_months=model_config.eval_tau_months,
    )
    lower, upper = bootstrap_c_index(
        locked.time,
        locked.event,
        predictions["risk_ensemble"],
        model_config.bootstrap_repeats,
        model_config.seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.DataFrame(
        {
            "PATIENT_ID": locked.patient_id,
            "time_months": locked.time,
            "event": locked.event,
            **predictions,
        }
    )
    prediction_frame.to_csv(
        output_dir / "locked_validation_predictions.tsv", sep="\t", index=False
    )
    metric_frame = pd.DataFrame(
        [{**metrics, "harrell_c_ci_low": lower, "harrell_c_ci_high": upper}]
    )
    metric_frame.to_csv(
        output_dir / "locked_validation_metrics.tsv", sep="\t", index=False
    )
    atomic_json_dump(manifest, output_dir / "data_manifest.json")
    return {
        "output_dir": str(output_dir),
        "locked_harrell_c": metrics["harrell_c"],
        "locked_validation_accessed": True,
    }


def _load_config_file(path: Optional[str]) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Configuration file must contain a JSON object")
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--profile", choices=("smoke", "standard"), default=None)
    parser.add_argument("--scope", choices=("development", "locked"), default=None)
    parser.add_argument(
        "--causal-mode",
        choices=("prognostic_invariance", "treatment_causal"),
        default=None,
    )
    parser.add_argument("--treatment-file")
    parser.add_argument("--enable-diffusion-qc", action="store_true", default=None)
    parser.add_argument("--config")
    parser.add_argument("--unlock-internal-validation", action="store_true", default=None)
    parser.add_argument("--bundle", help="Frozen development bundle required for locked scope")
    parser.add_argument("--output-dir")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--gate-alpha", type=float, default=None)
    parser.add_argument("--gate-max-iter", type=int, default=None)
    parser.add_argument("--source-bootstraps", type=int, default=None)
    parser.add_argument("--gate-l0-weight", type=float, default=None)
    parser.add_argument("--gate-count-weight", type=float, default=None)
    parser.add_argument("--elastic-l1", type=float, default=None)
    parser.add_argument("--elastic-l2", type=float, default=None)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> CMIBConfig:
    file_values = _load_config_file(args.config)
    profile = args.profile or file_values.pop("profile", "standard")
    for name in ("source_cancers", "clinical_columns", "final_seeds"):
        if name in file_values:
            file_values[name] = tuple(file_values[name])
    config = CMIBConfig.for_profile(profile, **file_values)
    cli_values = {
        "scope": args.scope,
        "causal_mode": args.causal_mode,
        "treatment_file": args.treatment_file,
        "enable_diffusion_qc": args.enable_diffusion_qc,
        "unlock_internal_validation": args.unlock_internal_validation,
        "output_dir": args.output_dir,
        "source_gate_bootstraps": args.source_bootstraps,
        "gate_l0_weight": args.gate_l0_weight,
        "gate_count_weight": args.gate_count_weight,
        "elastic_l1": args.elastic_l1,
        "elastic_l2": args.elastic_l2,
    }
    for name, value in cli_values.items():
        if value is not None:
            setattr(config, name, value)
    config.validate()
    return config


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    global _GATE_PRIOR_ALPHA, _GATE_PRIOR_MAX_ITER
    if args.gate_alpha is not None:
        _GATE_PRIOR_ALPHA = args.gate_alpha
    if args.gate_max_iter is not None:
        _GATE_PRIOR_MAX_ITER = args.gate_max_iter
    output_dir = (
        Path(config.output_dir)
        if config.output_dir
        else Path(RESULTS_DIR) / args.timestamp / "035_causal_meta_ib"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "pipeline.log", encoding="utf-8"),
        ],
        force=True,
    )
    LOGGER.info(
        "Starting CMIB-Surv profile=%s scope=%s causal_mode=%s",
        config.profile,
        config.scope,
        config.causal_mode,
    )
    LOGGER.info(
        "Gate prior params: alpha=%s max_iter=%s source_bootstraps=%s",
        _GATE_PRIOR_ALPHA,
        _GATE_PRIOR_MAX_ITER,
        config.source_gate_bootstraps,
    )
    LOGGER.info(
        "Regularizer weights: gate_l0=%s gate_count=%s elastic_l1=%s elastic_l2=%s",
        config.gate_l0_weight,
        config.gate_count_weight,
        config.elastic_l1,
        config.elastic_l2,
    )
    if config.scope == "development":
        result = run_development_pipeline(config, output_dir)
    else:
        if not args.bundle:
            raise ValueError("--bundle is required for locked validation evaluation")
        result = run_locked_evaluation(config, output_dir, args.bundle)
    LOGGER.info("Completed: %s", json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
