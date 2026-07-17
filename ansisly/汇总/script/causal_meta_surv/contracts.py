"""contracts.py — CMIB-Surv 数据契约与配置定义。

定义所有模块共享的不可变 dataclass：
    - SurvivalRecordBatch：样本级数据接口（11 字段，frozen=True）
    - CMIBConfig：运行时配置（含 CRC 特异分支配置）
    - CRCSpecificConfig：CRC 特异分支子配置
    - CMIBOutput：模型前向输出
    - CMIBLoss：统一损失（11 项 + total）
    - FoldPreprocessor：无泄漏预处理抽象接口
    - 异常类：LeakageError / InsufficientSourceTasksError

校验函数：
    - validate_batch(batch)：硬断言 patient_id 对齐、time>0、event∈{0,1}
    - assert_no_patient_overlap(...)：跨折患者隔离校验
    - make_smoke_config()：构造冒烟测试配置
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

# 复用 shared_utils 的常量（避免重复定义）
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_utils import RANDOM_SEED, ESSENTIAL_DIR, SCRIPT_DIR  # noqa: E402

SOURCE_CANCERS: Tuple[str, ...] = (
    "STAD", "ESCA", "PAAD", "BRCA", "LUAD", "KIRC", "LIHC", "UCEC",
)
DEFAULT_CLINICAL_COLUMNS: Tuple[str, ...] = ("AGE", "SEX", "AJCC_STAGE_ENCODED")
DEFAULT_TARGET_EXPRESSION = Path(ESSENTIAL_DIR) / "gene_expression_curated.tsv"
DEFAULT_TARGET_CLINICAL = Path(ESSENTIAL_DIR) / "tcga_os_clinical_endpoint_qc.tsv"
DEFAULT_SPLIT_FILE = (
    Path(SCRIPT_DIR).resolve().parents[1]
    / "DATA" / "survival_models"
    / "tcga_train_internal_validation_split.tsv"
)
DEFAULT_SOURCE_DIR = Path(SCRIPT_DIR).resolve().parents[1] / "rawData" / "multicancer_preprocessed"
DEFAULT_02_FEATURES_DIR = (
    Path(SCRIPT_DIR).resolve().parents[0]
    / "results" / "30260705_205046" / "02_gene_features"
)
EPS: float = 1e-8


# ──────────────────────────────────────────────────────────────────────
# 异常
# ──────────────────────────────────────────────────────────────────────
class LeakageError(RuntimeError):
    """预处理或训练流程检测到数据泄漏时抛出。"""


class InsufficientSourceTasksError(RuntimeError):
    """真实源癌种任务不足时抛出（禁止 COAD 随机切块伪造）。"""


# ──────────────────────────────────────────────────────────────────────
# 数据契约
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SurvivalRecordBatch:
    """不可变样本级数据接口。

    所有模块只接受此对象作为输入，直到进入单个训练 step 才转换 tensor。
    字段说明参见 spec.md SurvivalRecordBatch。
    """
    patient_id: Tuple[str, ...]
    x_gene: torch.Tensor              # [B, P]
    x_clinical: torch.Tensor          # [B, Q]
    clinical_mask: torch.BoolTensor   # [B, Q]
    task_id: torch.Tensor             # [B]
    time: torch.Tensor                # [B], 月，> 0
    event: torch.Tensor               # [B] ∈ {0,1}
    treatment: Optional[torch.Tensor] # [B] 或 None
    sample_weight: torch.Tensor       # [B]
    origin_patient_id: Tuple[str, ...]
    is_synthetic: torch.Tensor        # [B] bool

    @property
    def batch_size(self) -> int:
        return int(self.x_gene.shape[0])


# ──────────────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────────────
@dataclass
class CRCSpecificConfig:
    """CRC 特异分支配置（v2.0 Path D 双分支 Branch B）。"""
    enabled: bool = True
    blocked_genes: Tuple[str, ...] = (
        "DEFA5", "BMS1P20", "TPT1", "REG3A", "SLC26A3",
    )
    extra_high_freq_genes: Tuple[str, ...] = ()  # 由 02 高频基因动态填充
    hidden_dim: int = 16
    dropout: float = 0.15
    l2: float = 1e-4
    use_lasso_init: bool = True
    lasso_init_scale: float = 1.0
    freeze_first_layer_epochs: int = 0
    orthogonality_weight: float = 0.01

    def to_dict(self) -> dict:
        result = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, tuple):
                result[f.name] = list(val)
            else:
                result[f.name] = val
        return result


@dataclass
class CMIBConfig:
    """运行时配置（参考 035 CMIBConfig 结构，新增 CRC 分支与 7 阶段调度）。

    Standard profile 实现 5×3 嵌套 CV；smoke profile 仅保留数据边界与极小训练量。
    """
    profile: str = "standard"
    scope: str = "development"
    causal_mode: str = "prognostic_invariance"
    seed: int = RANDOM_SEED
    device: str = "cpu"
    source_cancers: Tuple[str, ...] = SOURCE_CANCERS
    clinical_columns: Tuple[str, ...] = DEFAULT_CLINICAL_COLUMNS
    gene_panel_mode: str = "strict"            # strict | relaxed
    min_source_presence: int = 8
    max_missing_gene_fraction: float = 0.02

    # 模型尺寸
    projection_dim: int = 64
    meta_hidden_dim: int = 48
    meta_dim: int = 24
    invariant_hidden_dim: int = 32
    invariant_dim: int = 16
    clinical_dim: int = 8
    adapter_rank: int = 4
    fusion_dim: int = 32
    ib_dim: int = 16
    active_gene_target: int = 64  # 迭代方案D: 回退到 P3 原值
    dropout: float = 0.25  # 迭代方案D: 回退到 P3 值

    # Hard-Concrete 门控
    gate_temperature: float = 2.0 / 3.0
    gate_gamma: float = -0.1
    gate_zeta: float = 1.1
    gate_l0_weight: float = 5e-3  # P1-1: 2e-4 → 5e-3
    gate_count_weight: float = 2e-3
    gate_group_weight: float = 1e-4
    gate_bootstraps: int = 30
    source_gate_bootstraps: int = 15
    gate_prior_alignment_weight: float = 0.001  # v4: 0.01 → 0.001（降低 P1-1 约束）

    # 损失权重
    cox_weight: float = 1.0
    rank_weight: float = 0.10
    kl_beta: float = 1e-4
    kl_warmup_epochs: int = 15
    kl_free_bits: float = 0.01
    elastic_l1: float = 5e-5  # P3-3.3: 2e-5 → 5e-5
    elastic_l2: float = 5e-5  # P3-3.3: 2e-5 → 5e-5
    domain_weight: float = 0.02
    coral_weight: float = 0.01
    consistency_weight: float = 0.10  # 迭代方案D: 回退到 P3 值
    anchor_weight: float = 1e-4  # 迭代方案D: 回退到 P3 值
    anchor_warmup_epochs: int = 10  # P2-2: anchor warmup
    fusion_entropy_weight: float = 0.02
    treatment_mmd_weight: float = 0.02
    causal_distill_weight: float = 0.05
    gradient_reversal_weight: float = 0.2
    min_branch_weight: float = 0.10  # P1-2: 0.05 → 0.10

    # 训练
    learning_rate: float = 2e-3
    weight_decay: float = 5e-4  # 迭代方案D: 回退到 P3 值
    max_epochs_stage1: int = 50  # 迭代方案E: 回退到 P3 原值
    max_epochs_stage2: int = 25
    patience: int = 8
    min_delta: float = 5e-4  # 迭代方案D: 回退到 P3 值
    gene_mask_probability: float = 0.08
    gene_noise_std: float = 0.03

    # 判别式学习率（target_trainer 用）
    lr_head: float = 3e-3  # P2-1: 2e-3 → 3e-3
    lr_adapter: float = 1.5e-3  # P2-1: 1e-3 → 1.5e-3
    lr_shared: float = 1e-3  # P2-1: 5e-4 → 1e-3
    lr_warmup_epochs: int = 5  # P2-1: 线性 warmup
    grad_clip: float = 1.0
    inner_patience: int = 12  # P3-3.2: 8 → 12

    # VIB / beta schedule（target_trainer 用）
    use_vib: bool = True
    beta_warmup_frac: float = 0.15
    beta_max: float = 1e-4

    # Stage3 解锁投影层所需的最小 inner CV 增益（与 035 对齐）
    unlock_projection_min_gain: float = 0.005

    # 因果教师训练轮数（treatment_causal 模式用，与 035 对齐）
    causal_teacher_epochs: int = 50

    # 因果 gate 先验注入的 boost 缩放因子（036 新增，由 CLI 传入）
    # _inject_causal_gate_prior 用：boost_j = final_weight_j * causal_boost_scale
    causal_boost_scale: float = 2.0

    # 融合 / 集成开关
    use_ensemble: bool = True
    fusion_min_weight: float = 0.05

    # 嵌套 CV
    outer_folds: int = 5
    inner_folds: int = 3
    final_seeds: Tuple[int, ...] = (RANDOM_SEED, RANDOM_SEED + 17, RANDOM_SEED + 43,
                                    RANDOM_SEED + 71, RANDOM_SEED + 103)  # P3-4.3: 3 → 5 seeds

    # Reptile
    meta_iterations: int = 60
    meta_batch_size: int = 4
    meta_inner_steps: int = 3
    meta_inner_lr: float = 5e-3
    meta_lr: float = 0.10
    meta_lr_min: float = 0.005
    episode_support_size: int = 64
    episode_query_size: int = 48
    episode_min_support_events: int = 8
    episode_min_query_events: int = 5

    # 评估
    ensemble_step: float = 0.05  # 迭代方案E: 回退到 P3 原值
    eval_tau_months: float = 36.0
    bootstrap_repeats: int = 500
    bootstrap_n: int = 500  # alias for bootstrap_repeats（ensemble/evaluation 用）

    # 增强简化（噪声 + 质量门）
    augmentation_enabled: bool = True
    aug_synthetic_weight: float = 0.3

    # CRC 特异分支
    crc_branch: CRCSpecificConfig = field(default_factory=CRCSpecificConfig)

    # 路径
    target_expression_path: str = str(DEFAULT_TARGET_EXPRESSION)
    target_clinical_path: str = str(DEFAULT_TARGET_CLINICAL)
    split_file: str = str(DEFAULT_SPLIT_FILE)
    source_dir: str = str(DEFAULT_SOURCE_DIR)
    features_02_dir: str = str(DEFAULT_02_FEATURES_DIR)
    treatment_file: Optional[str] = None
    output_dir: Optional[str] = None

    # 锁定验证
    unlock_internal_validation: bool = False

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

    @property
    def max_epochs(self) -> int:
        """统一入口：默认用 stage1 的 max_epochs。"""
        return self.max_epochs_stage1

    @property
    def meta_epochs(self) -> int:
        """alias for meta_iterations（meta_trainer 用）。"""
        return self.meta_iterations

    @property
    def inner_steps(self) -> int:
        """alias for meta_inner_steps。"""
        return self.meta_inner_steps

    @property
    def min_support_events(self) -> int:
        """alias for episode_min_support_events。"""
        return self.episode_min_support_events

    @property
    def min_query_events(self) -> int:
        """alias for episode_min_query_events。"""
        return self.episode_min_query_events

    @property
    def aug_noise_std(self) -> float:
        """alias for gene_noise_std（augmentation 用）。"""
        return self.gene_noise_std

    @property
    def aug_gene_mask_frac(self) -> float:
        """alias for gene_mask_probability（augmentation 用）。"""
        return self.gene_mask_probability

    def to_dict(self) -> dict:
        """将配置序列化为字典（用于 checkpoint 持久化）。"""
        result = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, tuple):
                result[f.name] = list(val)
            elif isinstance(val, CRCSpecificConfig):
                result[f.name] = val.to_dict()
            else:
                result[f.name] = val
        return result

    def candidates(self) -> list:
        """返回 HPO 候选配置列表（用于 nested CV inner fold 搜索）。

        smoke 只返回 1 个候选（当前配置），standard 返回 active_gene_target x ib_dim x kl_beta 网格。
        """
        import itertools
        if self.profile == "smoke":
            return [{
                "active_gene_target": self.active_gene_target,
                "ib_dim": self.ib_dim,
                "kl_beta": self.kl_beta,
            }]
        return [
            {"active_gene_target": k, "ib_dim": d, "kl_beta": beta}
            for k, (d, beta) in itertools.product(
                (32, 64, 96),
                ((8, 1e-4), (16, 1e-4), (16, 5e-4), (24, 5e-4)),
            )
        ]

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
            )
        elif profile != "standard":
            raise ValueError(f"Unknown profile: {profile!r}")
        for key, value in overrides.items():
            if not hasattr(config, key):
                raise KeyError(f"Unknown configuration key: {key}")
            setattr(config, key, value)
        config.validate()
        return config


def make_smoke_config(**overrides: Any) -> CMIBConfig:
    """构造冒烟测试配置（与 for_profile('smoke') 等价）。"""
    return CMIBConfig.for_profile("smoke", **overrides)


# ──────────────────────────────────────────────────────────────────────
# 模型输出与损失
# ──────────────────────────────────────────────────────────────────────
@dataclass
class CMIBOutput:
    """模型前向输出。"""
    log_risk: torch.Tensor         # [B] 主 Cox log-risk
    mu: torch.Tensor               # [B, ib_dim]
    logvar: torch.Tensor           # [B, ib_dim]
    z: torch.Tensor                # [B, ib_dim] 重参数化采样
    z_meta: torch.Tensor           # [B, meta_dim]
    z_invariant: torch.Tensor      # [B, invariant_dim]
    z_task: Optional[torch.Tensor] # [B, adapter_rank] 或 None
    z_clin: torch.Tensor           # [B, clinical_dim]
    z_crc: Optional[torch.Tensor]  # [B, crc hidden] CRC 特异分支输出
    fusion_weights: torch.Tensor   # [B, 3] 三分支权重
    gate_prob: torch.Tensor        # [P] 基因被选概率
    domain_logits: Optional[torch.Tensor] = None  # [B, n_tasks]
    aux_hazard: Optional[torch.Tensor] = None     # [B, n_bins] 离散时间辅助头


@dataclass
class CMIBLoss:
    """统一损失（11 项 + total）。

    字段存储 torch.Tensor（可微，用于 backward）或 float（不可微项）。
    as_dict() 将所有项转为 float 用于日志/序列化。
    """
    cox: Any = 0.0
    rank: Any = 0.0
    ib: Any = 0.0
    gate: Any = 0.0
    elastic: Any = 0.0
    distill: Any = 0.0
    bal: Any = 0.0
    domain: Any = 0.0
    orth: Any = 0.0
    cons: Any = 0.0
    anchor: Any = 0.0
    total: Any = 0.0

    def as_dict(self) -> dict:
        """转为 float 字典（用于日志/序列化）。"""
        def _to_float(v: Any) -> float:
            if isinstance(v, torch.Tensor):
                return float(v.item())
            return float(v)
        return {k: _to_float(v) for k, v in asdict(self).items()}


# ──────────────────────────────────────────────────────────────────────
# FoldPreprocessor 抽象接口
# ──────────────────────────────────────────────────────────────────────
class FoldPreprocessor:
    """无泄漏预处理抽象接口。

    fit() 只允许访问当前训练折；transform() 应用训练折统计量；
    state_dict() 返回可哈希状态用于泄漏检测。
    """

    def fit(self, train_df, gene_registry) -> "FoldPreprocessor":
        raise NotImplementedError

    def transform(self, df) -> "SurvivalRecordBatch":
        raise NotImplementedError

    def state_dict(self) -> dict:
        raise NotImplementedError

    def state_hash(self) -> str:
        """对 state_dict 序列化后求 SHA256，用于泄漏检测。"""
        payload = json.dumps(self.state_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────────────
# 校验函数
# ──────────────────────────────────────────────────────────────────────
def validate_batch(batch: SurvivalRecordBatch) -> None:
    """校验 SurvivalRecordBatch 硬约束。

    Raises:
        ValueError: 患者ID错位 / time<=0 / event∉{0,1} / 维度不匹配
    """
    n = batch.batch_size
    if len(batch.patient_id) != n:
        raise ValueError(
            f"patient_id length {len(batch.patient_id)} != batch_size {n}"
        )
    if len(batch.origin_patient_id) != n:
        raise ValueError(
            f"origin_patient_id length {len(batch.origin_patient_id)} != batch_size {n}"
        )
    for tensor_name, tensor in (
        ("x_clinical", batch.x_clinical),
        ("clinical_mask", batch.clinical_mask),
        ("task_id", batch.task_id),
        ("time", batch.time),
        ("event", batch.event),
        ("sample_weight", batch.sample_weight),
        ("is_synthetic", batch.is_synthetic),
    ):
        if tensor.shape[0] != n:
            raise ValueError(f"{tensor_name} batch dim {tensor.shape[0]} != {n}")
    if batch.treatment is not None and batch.treatment.shape[0] != n:
        raise ValueError("treatment batch dim mismatch")

    # time > 0
    if torch.any(batch.time <= 0):
        bad_idx = torch.nonzero(batch.time <= 0).flatten().tolist()
        raise ValueError(f"time <= 0 at indices {bad_idx}")

    # event ∈ {0,1}
    unique_events = torch.unique(batch.event)
    if not torch.all((unique_events == 0) | (unique_events == 1)):
        raise ValueError(f"event must be binary, got {unique_events.tolist()}")

    # patient_id 唯一（同一 batch 内不应重复）
    if len(set(batch.patient_id)) != n:
        from collections import Counter
        counts = Counter(batch.patient_id)
        dup = [pid for pid, c in counts.items() if c > 1]
        raise ValueError(f"Duplicate patient_id in batch: {dup[:5]}")


def assert_no_patient_overlap(*patient_id_lists: Sequence[str]) -> None:
    """校验多组患者 ID 无交集（跨折隔离）。

    Raises:
        LeakageError: 任两组之间存在交集
    """
    for i, lst_i in enumerate(patient_id_lists):
        set_i = set(lst_i)
        for j, lst_j in enumerate(patient_id_lists):
            if j <= i:
                continue
            overlap = set_i & set(lst_j)
            if overlap:
                raise LeakageError(
                    f"Patient overlap between group {i} and {j}: "
                    f"{len(overlap)} patients, e.g. {list(overlap)[:3]}"
                )


def hash_state_dict(state: Mapping[str, Any]) -> str:
    """对 state_dict 求哈希（用于持久化校验）。"""
    payload = json.dumps(state, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
