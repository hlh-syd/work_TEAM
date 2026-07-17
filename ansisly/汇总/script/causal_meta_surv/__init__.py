"""causal_meta_surv — CMIB-Surv (Causal Meta-learning Information-Bottleneck Survival) 模块包。

模块化重构自 035_CasualModel.py，落地 Fusion_Plan.md 提出的 11 模块架构，
采用 v2.0 Path D 双分支设计保留 CRC 组织特异性信号。

子模块：
    contracts      — 数据契约（SurvivalRecordBatch / CMIBConfig / CMIBOutput / CMIBLoss）
    data           — 数据加载、患者 ID 对齐、源癌种任务构造
    preprocessing  — 无泄漏 fold 预处理 + 泄漏测试
    sparse_gate    — Hard-Concrete L0 稀疏门控
    encoders       — Meta/Causal/Clinical/TaskAdapter 编码器（LayerNorm）
    fusion         — ReliabilityFusion 三分支 + CRCSpecificBranch + VariationalIB + 主模型
    heads          — CoxHead / DiscreteTimeAuxHead / DomainHead + ElasticNetCox 三级 fallback
    losses         — compute_cmib_loss 统一损失（11 项）
    meta_trainer   — 修正版 Reptile 元学习
    target_trainer — CRC 适配训练器（inner CV 早停）
    augmentation   — 简化噪声增强 + 质量门
    ensemble       — OOF 融合 + ElasticNet Cox 保护栏
    evaluation     — Harrell/Uno/IPCW C-index / IBS / calibration / bootstrap
    serialization  — 模型/预处理持久化 + hash 校验
"""
from __future__ import annotations

# 版本
__version__ = "1.0.0"

# 导出公共接口（延迟导入避免循环依赖）
from .contracts import (
    CMIBConfig,
    CMIBLoss,
    CMIBOutput,
    CRCSpecificConfig,
    FoldPreprocessor,
    InsufficientSourceTasksError,
    LeakageError,
    SurvivalRecordBatch,
    assert_no_patient_overlap,
    hash_state_dict,
    make_smoke_config,
    validate_batch,
)
from .data import (
    GeneFeatures02,
    GeneRegistry,
    SourceTask,
    align_patient_ids,
    audit_data,
    build_gene_registry,
    build_source_tasks,
    load_clinical_endpoint,
    load_split_table,
    load_source_expression,
    load_target_expression,
    make_survival_record_batch,
)
from .preprocessing import (
    FoldPreprocessorImpl,
    assert_no_leakage,
    group_kfold_indices,
    stratified_time_event_keys,
)
from .sparse_gate import (
    HardConcreteSparseGate,
    compute_gate_loss,
    fold_gate_prior,
    jaccard_stability,
)
from .encoders import (
    CausalInvariantEncoder,
    ClinicalEncoder,
    GeneProjection,
    GradientReversalFunction,
    MetaEncoder,
    TaskAdapter,
    gradient_reversal,
)
from .fusion import (
    CRCSpecificBranch,
    CausalMetaIBSurv,
    ReliabilityFusion,
    VariationalIB,
    orth_loss,
)
from .heads import (
    CoxHead,
    DiscreteTimeAuxHead,
    DomainHead,
    ElasticNetCox,
    ElasticNetCoxFallback,
    RidgeCox,
    SimpleCox,
)
from .losses import (
    anchor_loss,
    bal_loss,
    compute_cmib_loss,
    cons_loss,
    cox_efron_loss,
    distill_loss,
    domain_loss,
    elastic_loss,
    gate_loss,
    ib_loss,
    rank_loss,
)
from .meta_trainer import (
    ReptileMetaTrainer,
    clone_parameters,
    compute_delta,
    apply_delta,
    mean_deltas,
    restore_parameters,
    stratified_survival_episode,
    leave_one_cancer_out_eval,
)
from .target_trainer import (
    TargetTrainer,
    run_inner_cv,
)
from .augmentation import (
    Augmentor,
    augment_for_representation,
)
from .ensemble import (
    EnsembleWithFallback,
    OOFFoldEnsemble,
    convex_blend,
    detect_complementary_errors,
    locked_validation_eval,
)
from .evaluation import (
    bootstrap_ci,
    calibration_slope_intercept,
    evaluate_full,
    harrell_c_index,
    integrated_brier_score,
    ipcw_c_index,
    paired_bootstrap_delta_c,
    uno_c_index,
)
from .serialization import (
    load_json,
    load_model_state,
    load_preprocessor,
    save_audit_report,
    save_json,
    save_metrics,
    save_model_state,
    save_preprocessor,
)

__all__ = [
    # contracts
    "CMIBConfig", "CMIBLoss", "CMIBOutput", "CRCSpecificConfig",
    "FoldPreprocessor", "InsufficientSourceTasksError", "LeakageError",
    "SurvivalRecordBatch", "assert_no_patient_overlap",
    "hash_state_dict", "make_smoke_config", "validate_batch",
    # data
    "GeneFeatures02", "GeneRegistry", "SourceTask",
    "align_patient_ids", "audit_data", "build_gene_registry",
    "build_source_tasks", "load_clinical_endpoint", "load_split_table",
    "load_source_expression", "load_target_expression",
    "make_survival_record_batch",
    # preprocessing
    "FoldPreprocessorImpl", "assert_no_leakage",
    "group_kfold_indices", "stratified_time_event_keys",
    # sparse_gate
    "HardConcreteSparseGate", "compute_gate_loss",
    "fold_gate_prior", "jaccard_stability",
    # encoders
    "CausalInvariantEncoder", "ClinicalEncoder", "GeneProjection",
    "GradientReversalFunction", "MetaEncoder", "TaskAdapter",
    "gradient_reversal",
    # fusion
    "CRCSpecificBranch", "CausalMetaIBSurv", "ReliabilityFusion",
    "VariationalIB", "orth_loss",
    # heads
    "CoxHead", "DiscreteTimeAuxHead", "DomainHead",
    "ElasticNetCox", "ElasticNetCoxFallback", "RidgeCox", "SimpleCox",
    # losses
    "anchor_loss", "bal_loss", "compute_cmib_loss", "cons_loss",
    "cox_efron_loss", "distill_loss", "domain_loss", "elastic_loss",
    "gate_loss", "ib_loss", "rank_loss",
    # meta_trainer
    "ReptileMetaTrainer", "clone_parameters", "compute_delta",
    "apply_delta", "mean_deltas", "restore_parameters",
    "stratified_survival_episode", "leave_one_cancer_out_eval",
    # target_trainer
    "TargetTrainer", "run_inner_cv",
    # augmentation
    "Augmentor", "augment_for_representation",
    # ensemble
    "EnsembleWithFallback", "OOFFoldEnsemble",
    "convex_blend", "detect_complementary_errors",
    "locked_validation_eval",
    # evaluation
    "bootstrap_ci", "calibration_slope_intercept", "evaluate_full",
    "harrell_c_index", "integrated_brier_score", "ipcw_c_index",
    "paired_bootstrap_delta_c", "uno_c_index",
    # serialization
    "load_json", "load_model_state", "load_preprocessor",
    "save_audit_report", "save_json", "save_metrics",
    "save_model_state", "save_preprocessor",
]
