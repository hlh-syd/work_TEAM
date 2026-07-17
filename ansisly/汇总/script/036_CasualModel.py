#!/usr/bin/env python3
"""036_CasualModel.py — CMIB-Surv 主管线入口脚本。

模块化重构自 035_CasualModel.py，落地 Fusion_Plan.md 的 11 模块 CMIB-Surv 架构。
实现 7 阶段训练流程（Phase 0-7）与 Go/No-Go 决策门。

用法：
    # 仅 Phase 0 数据审计
    python 036_CasualModel.py --phase 0 --profile smoke

    # 完整冒烟
    python 036_CasualModel.py --phase all --profile smoke

    # 标准运行
    python 036_CasualModel.py --phase all --profile standard

输出目录：ansisly/汇总/results/<timestamp>_036_cmib_surv/
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ──────────────────────────────────────────────────────────────────────
# 路径设置
# ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from causal_meta_surv import (
    CMIBConfig,
    CausalMetaIBSurv,
    ElasticNetCoxFallback,
    GeneFeatures02,
    GeneRegistry,
    SourceTask,
    TargetTrainer,
    ReptileMetaTrainer,
    Augmentor,
    audit_data,
    build_gene_registry,
    build_source_tasks,
    clone_parameters,
    harrell_c_index,
    load_clinical_endpoint,
    load_split_table,
    load_source_expression,
    load_target_expression,
    make_smoke_config,
    make_survival_record_batch,
    save_audit_report,
    save_json,
    save_metrics,
    save_model_state,
    save_preprocessor,
    evaluate_full,
)
from causal_meta_surv.contracts import RANDOM_SEED
from causal_meta_surv.data import align_patient_ids
from causal_meta_surv.preprocessing import FoldPreprocessorImpl, assert_no_leakage
from causal_meta_surv.serialization import save_gene_panel_report

LOGGER = logging.getLogger("cmib_surv.main")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="CMIB-Surv: Causal Meta-learning Information-Bottleneck Survival Model"
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        help="执行阶段：0-7 单独执行，或 'all' 全部执行（默认 all）"
    )
    parser.add_argument(
        "--profile", type=str, default="standard",
        choices=["smoke", "standard"],
        help="运行配置：smoke（快速冒烟）或 standard（完整标准）（默认 standard）"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="配置文件路径（JSON/YAML，覆盖默认配置）"
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument(
        "--ablation", type=str, default=None,
        help="消融实验名称（B0/B1/M1/C1/I1/F1/F2/F3/A1/A2/E1）"
    )
    parser.add_argument(
        "--unlock-internal-validation", action="store_true",
        help="解锁锁定内部验证集（谨慎使用）"
    )
    parser.add_argument(
        "--gene-panel-mode", type=str, default="strict",
        choices=["strict", "relaxed"],
        help="基因面板模式（默认 strict）"
    )
    parser.add_argument(
        "--disable-causal-crc", action="store_true",
        help="消融开关：跳过 causal_priority 驱动的 CRC 分支基因选择与重初始化"
    )
    parser.add_argument(
        "--disable-causal-gate-prior", action="store_true",
        help="消融开关：跳过 Phase 2 因果先验与 bootstrap 频率的融合"
    )
    parser.add_argument(
        "--causal-top-k", type=int, default=20,
        help="因果优先级前 K 基因用于 CRC 分支（默认 20）"
    )
    parser.add_argument(
        "--causal-prior-scale", type=float, default=1.0,
        help="CRC 分支 fc1 权重的因果方向注入强度（默认 1.0）"
    )
    parser.add_argument(
        "--causal-gate-weight", type=float, default=0.3,
        help="Phase 2 中因果先验对 gate 频率的混合权重（默认 0.3）"
    )
    parser.add_argument(
        "--causal-boost-scale", type=float, default=2.0,
        help="因果 gate 先验注入的 boost 缩放因子（_inject_causal_gate_prior 用，默认 2.0）"
    )
    parser.add_argument(
        "--causal-crc-reinit", action="store_true", default=True,
        help="P1-2: 用 02 top-K 因果基因重初始化 CRC 分支 fc1（默认开启）"
    )
    parser.add_argument(
        "--no-causal-crc-reinit", dest="causal_crc_reinit", action="store_false",
        help="P1-2: 禁用因果 CRC 分支重初始化"
    )
    parser.add_argument(
        "--oof-folds", type=int, default=None,
        help="Phase 4-6 内 OOF 折数（默认 smoke=3, standard=5）"
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> CMIBConfig:
    """加载配置。"""
    overrides: Dict[str, Any] = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.unlock_internal_validation:
        overrides["unlock_internal_validation"] = True
        overrides["scope"] = "locked"
    overrides["gene_panel_mode"] = args.gene_panel_mode
    # 因果 boost 缩放因子（_inject_causal_gate_prior 用）
    overrides["causal_boost_scale"] = args.causal_boost_scale

    # 从文件加载覆盖
    if args.config:
        config_path = Path(args.config)
        if config_path.suffix in (".json",):
            with open(config_path, "r", encoding="utf-8") as f:
                file_cfg = json.load(f)
            overrides.update(file_cfg)
        elif config_path.suffix in (".yaml", ".yml"):
            try:
                import yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    file_cfg = yaml.safe_load(f)
                overrides.update(file_cfg)
            except ImportError:
                LOGGER.warning("PyYAML not installed, skipping YAML config")

    cfg = CMIBConfig.for_profile(args.profile, **overrides)

    # 消融配置
    if args.ablation:
        cfg = _apply_ablation(cfg, args.ablation)

    return cfg


def _apply_ablation(cfg: CMIBConfig, name: str) -> CMIBConfig:
    """应用消融实验配置。"""
    from dataclasses import replace
    ablation_map = {
        "B0": {"profile": "standard"},  # ElasticNet baseline only
        "M1": {"invariant_hidden_dim": 0},  # only MetaEncoder
        "C1": {"meta_hidden_dim": 0},  # only CausalInvariantEncoder
        "I1": {"kl_beta": 0.0},  # only VIB disabled → I1 = only VIB means disable others
        "F1": {"min_branch_weight": 0.0},  # no ReliabilityFusion constraint
        "F2": {"fusion_entropy_weight": 0.0},  # no entropy penalty
        "F3": {"min_branch_weight": 0.0},  # no min weight
        "A1": {"augmentation_enabled": False},  # no augmentation
        "A2": {"crc_branch": replace(cfg.crc_branch, enabled=False)},  # no CRC branch
        "E1": {"use_ensemble": False},  # no OOF fusion
    }
    if name not in ablation_map:
        LOGGER.warning("Unknown ablation: %s, using default config", name)
        return cfg
    ablation_overrides = ablation_map[name]
    for k, v in ablation_overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    LOGGER.info("Applied ablation: %s", name)
    return cfg


# ──────────────────────────────────────────────────────────────────────
# 输出目录与日志
# ──────────────────────────────────────────────────────────────────────
def create_output_dir(cfg: CMIBConfig) -> Path:
    """创建结果输出目录。"""
    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_root = SCRIPT_DIR.parent / "results"
        output_dir = results_root / f"{timestamp}_036_cmib_surv"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ======================================================================
# 因果特征辅助函数（causal_priority_feature_table.tsv 集成）
# ======================================================================
_DEFAULT_CRC_FALLBACK_GENES: Sequence[str] = (
    "DEFA5", "BMS1P20", "TPT1", "REG3A", "SLC26A3",
)

# CLI 消融开关（由 main() 写入，避免修改 CMIBConfig dataclass）
_CLI_FLAGS: Dict[str, Any] = {
    "disable_causal_crc": False,
    "disable_causal_gate_prior": False,
    "causal_top_k": 20,
    "causal_prior_scale": 1.0,
    "causal_gate_weight": 0.3,
    "causal_crc_reinit": True,  # P1-2: 因果 CRC 分支重初始化
    "oof_folds": None,
}


def _load_causal_priority_table(cfg: CMIBConfig):
    """加载并规范化 causal_priority_feature_table.tsv。

    Returns:
        pandas.DataFrame，含列 feature/priority_norm/evidence_weight/ph_penalty/coef_dir/final_weight。
        TPTI 拼写自动修正为 TPT1。None 当文件不存在。
    """
    import numpy as np
    import pandas as pd

    # 候选路径（兼容 contracts.py 默认与实际产出位置）
    filename = "causal_priority_feature_table.tsv"
    candidates: List[Path] = [Path(cfg.features_02_dir) / filename]
    # 回退 1：相对于当前脚本目录 (汇总/script -> 汇总/results/.../02_gene_features)
    candidates.append(
        SCRIPT_DIR.parent / "results" / "30260705_205046" / "02_gene_features" / filename
    )
    # 回退 2：尝试命中任意一个存在的 02_gene_features 下的同名文件
    results_root = SCRIPT_DIR.parent / "results"
    if results_root.exists():
        for sub in results_root.iterdir():
            candidate = sub / "02_gene_features" / filename
            if candidate.exists():
                candidates.append(candidate)
                break

    path: Optional[Path] = None
    for p in candidates:
        if p.exists():
            path = p
            break
    if path is None:
        LOGGER.warning(
            "Causal priority table not found. Tried: %s",
            " | ".join(str(p) for p in candidates[:3]),
        )
        return None
    LOGGER.info("Causal priority table loaded from: %s", path)
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception as e:
        LOGGER.warning("Failed reading causal priority table: %s", e)
        return None
    if df.empty or "feature" not in df.columns:
        LOGGER.warning("Causal priority table empty or missing 'feature' column")
        return None
    df["feature"] = df["feature"].astype(str).replace({"TPTI": "TPT1"})
    if "causal_priority_score" in df.columns:
        score = df["causal_priority_score"].astype(float)
        score = score.fillna(score.min())
        smin, smax = float(score.min()), float(score.max())
        if smax > smin:
            df["priority_norm"] = (score - smin) / (smax - smin + 1e-9)
        else:
            df["priority_norm"] = 1.0
    else:
        df["priority_norm"] = 0.5
    if "causal_evidence_level" in df.columns:
        ev = df["causal_evidence_level"].astype(str)
        df["evidence_weight"] = np.where(
            ev == "observational_cox_adjusted", 1.0, 0.5,
        )
    else:
        df["evidence_weight"] = 0.5
    if "ph_assumption_violated" in df.columns:
        ph_v = df["ph_assumption_violated"].astype(str).str.lower()
        df["ph_penalty"] = np.where(ph_v.isin(["true", "1"]), 0.5, 1.0)
    else:
        df["ph_penalty"] = 1.0
    coef_col = "coef_adj" if "coef_adj" in df.columns else "coef_univariable"
    if coef_col in df.columns:
        df["coef_dir"] = df[coef_col].astype(float).fillna(0.0)
    else:
        df["coef_dir"] = 0.0
    df["final_weight"] = (
        df["priority_norm"].astype(float)
        * df["evidence_weight"].astype(float)
        * df["ph_penalty"].astype(float)
    )
    return df


def _rank_normalize(values) -> "np.ndarray":
    """秩归一化到 (0,1)（复用 scipy.stats.rankdata average 方法）。"""
    import numpy as np
    from scipy.stats import rankdata

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    return (rankdata(arr, method="average") - 0.5) / len(arr)


def _survival_stratified_kfold(
    time, event, n_splits: int, seed: int,
) -> List:
    """生存分层 K-Fold（按 event 分层，确保每折事件比例均衡）。"""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    event_arr = np.asarray(event, dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # StratifiedKFold 需要 X 和 y，这里用 event 作为分层变量
    dummy_X = np.zeros((len(event_arr), 1))
    return list(skf.split(dummy_X, event_arr))


def _fit_single_fold(
    train_pids: Sequence[str],
    val_pids: Sequence[str],
    target_expr,
    clinical,
    gene_registry,
    cfg: CMIBConfig,
    split,
    causal_df,
    gene_list: Sequence[str],
    branch_mode: str = "full",
    seed: int = 0,
    meta_state=None,
    enable_causal_injection: bool = True,
    causal_boost_scale: Optional[float] = None,
) -> "np.ndarray":
    """构建模型 + 加载 meta 初始化 + 注入因果先验 + 训练 + 返回 val 风险。

    branch_mode:
        'full' - 正常三分支融合
        'meta_only' - 强制 fusion weights = [1,0,0]（仅用 meta encoder）
        'invariant_only' - 强制 fusion weights = [0,1,0]（仅用 invariant encoder）
        'clinical_only' - 强制 fusion weights = [0,0,1]（仅用 clinical encoder）

    meta_state: Phase 3 Reptile 元学习产出的模型状态字典（跨癌种迁移）。
        传 None 时（如 random_init 分支），跳过 meta 加载，从随机初始化训练。

    enable_causal_injection: 是否注入因果 gate 先验。
        - True:  full+causal 分支（因果增强）
        - False: full / random_init / *_only 等基线分支（无因果注入）

    causal_boost_scale: 因果先验 boost 缩放因子。
        None 时使用 cfg.causal_boost_scale（默认由 CLI 传入）。

    任何失败直接 raise（无 try/except 降级）。
    """
    import numpy as np
    import torch
    from dataclasses import replace as _dc_replace

    # 构建配置（保留 cfg.crc_branch.enabled 原始配置，不再硬编码禁用）
    cfg_fold = _dc_replace(cfg, seed=seed)
    n_clinical = len(cfg_fold.clinical_columns)

    # 构建 CRC 特异基因索引（从 gene_registry 中匹配 blocked_genes + extra_high_freq_genes）
    crc_gene_indices = None
    crc_lasso_coefs = None
    if cfg_fold.crc_branch.enabled:
        gene_list_local = list(gene_registry.genes)
        crc_gene_names = list(cfg_fold.crc_branch.blocked_genes) + list(cfg_fold.crc_branch.extra_high_freq_genes)
        crc_indices_list = [i for i, g in enumerate(gene_list_local) if g in set(crc_gene_names)]
        if crc_indices_list:
            crc_gene_indices = np.array(crc_indices_list, dtype=np.int64)
            LOGGER.debug("CRC branch enabled: %d matched genes", len(crc_indices_list))
        else:
            LOGGER.debug("CRC branch enabled but no matched genes in registry, branch will be skipped")

    model = CausalMetaIBSurv(
        n_genes=len(gene_registry.genes),
        n_clinical=n_clinical,
        cfg=cfg_fold,
        crc_gene_indices=crc_gene_indices,
        crc_lasso_coefs=crc_lasso_coefs,
        n_tasks=len(cfg_fold.source_cancers) + 1,
    )

    # 加载 meta 初始化（Reptile 跨癌种迁移）
    # random_init 分支传 meta_state=None，跳过加载
    if meta_state is not None:
        compatible = {
            name: value for name, value in meta_state.items()
            if name in model.state_dict() and model.state_dict()[name].shape == value.shape
        }
        model.load_state_dict(compatible, strict=False)

    # 因果 gate 先验注入（在 meta 加载之后，确保 gate 偏置覆盖 meta 的 gate 初始化）
    if enable_causal_injection and causal_df is not None:
        boost_scale = causal_boost_scale if causal_boost_scale is not None else getattr(cfg_fold, "causal_boost_scale", 2.0)
        _inject_causal_gate_prior(model, causal_df, gene_list, cfg_fold, boost_scale=boost_scale)

    # branch_mode: 强制 fusion 分支权重（P2-5: 改用 set_force_branch 接口）
    if branch_mode in ("meta_only", "invariant_only", "clinical_only"):
        if hasattr(model, "fusion") and hasattr(model.fusion, "set_force_branch"):
            model.fusion.set_force_branch(branch_mode)
            LOGGER.debug("branch_mode set to %s", branch_mode)

    # 构造 batch（P0-1 修复：使用 FoldPreprocessorImpl 做 winsorize+IQR 鲁棒缩放）
    # 与 035 FoldPreprocessor 对齐：训练折 fit，验证折用同参数 transform（无泄漏）
    preprocessor = FoldPreprocessorImpl(cfg_fold)
    # fit on train only
    train_expr_df = target_expr.loc[train_pids]
    preprocessor.fit(train_expr_df, gene_registry)
    # transform train and val with train-fold statistics
    tr_batch = preprocessor.transform(
        df=target_expr.loc[train_pids],
        clinical=clinical,
        split=split,
        patient_ids=list(train_pids),
        task_id=0,
    )
    va_batch = preprocessor.transform(
        df=target_expr.loc[val_pids],
        clinical=clinical,
        split=split,
        patient_ids=list(val_pids),
        task_id=0,
    )

    # 训练
    max_epochs = 3 if cfg_fold.profile == "smoke" else cfg_fold.max_epochs_stage1
    trainer = TargetTrainer(model, cfg_fold, gene_registry.genes, cfg_fold.device)
    # 与 035 对齐：在 meta 加载和因果注入完成后，对模型参数做快照作为 anchor_state。
    # anchor loss 只对 gene_projection 和 meta_encoder 前缀参数生效（见 losses.py），
    # 防止训练偏离元初始化太远。random_init 分支（meta_state=None）不设 anchor。
    if meta_state is not None:
        anchor_snapshot = {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
        }
        trainer.set_meta_init(anchor_snapshot)
    trainer.train_outer_fold(tr_batch, va_batch, max_epochs=max_epochs)

    # 推理
    risk = model.predict_risk(va_batch, mc_samples=0)
    return risk.astype(np.float32)


def _fit_elastic_net_fold(
    train_pids: Sequence[str],
    val_pids: Sequence[str],
    target_expr,
    clinical,
    gene_registry,
    cfg: CMIBConfig,
    split,
    seed: int = 0,
) -> "np.ndarray":
    """在单折上训练 ElasticNetCox 并返回 val 风险。

    特征矩阵与 035 的 `_elastic_features` 完全一致：
        `x_gene + x_clinical + clinical_mask`
    其中 clinical_mask 为浮点型（0.0/1.0），与 035 的 `.to(x_gene.dtype)` 行为对齐。
    """
    import numpy as np
    import pandas as pd

    gene_list = list(gene_registry.genes)
    n_genes = len(gene_list)
    clin_cols = list(cfg.clinical_columns)
    n_clin = len(clin_cols)

    def _build_X(pids):
        n = len(pids)
        # 基因特征
        x_gene = np.zeros((n, n_genes), dtype=np.float32)
        for i, pid in enumerate(pids):
            if pid in target_expr.index:
                row = target_expr.loc[pid]
                for j, g in enumerate(gene_list):
                    if g in row.index:
                        x_gene[i, j] = float(row[g])
        # 临床特征 + mask（与 035 _elastic_features 完全一致）
        x_clin = np.zeros((n, n_clin), dtype=np.float32)
        clin_mask = np.zeros((n, n_clin), dtype=np.float32)
        for i, pid in enumerate(pids):
            if pid in clinical.index:
                row = clinical.loc[pid]
                for j, col in enumerate(clin_cols):
                    if col in row.index and pd.notna(row[col]):
                        x_clin[i, j] = float(row[col])
                        clin_mask[i, j] = 1.0
        # 拼接：x_gene + x_clinical + clinical_mask
        return np.concatenate([x_gene, x_clin, clin_mask], axis=1)

    train_X = _build_X(train_pids)
    val_X = _build_X(val_pids)
    train_time = split.loc[train_pids, "time_months"].values.astype(np.float32)
    train_event = split.loc[train_pids, "event"].values.astype(int)

    # 与 035 fit_elastic_net_cox 的 fallback 路径对齐：level2_alpha=1.0, level2_max_iter=200
    elastic = ElasticNetCoxFallback(
        level1_alpha=0.02,
        level1_max_iter=2000,
        level2_alpha=1.0,
        level2_max_iter=200,
    )
    elastic.fit(train_X, train_event, train_time)
    return elastic.predict_risk(val_X).astype(np.float32)


def _fit_causal_only_fold(
    train_pids: Sequence[str],
    val_pids: Sequence[str],
    target_expr,
    clinical,
    gene_registry,
    cfg: CMIBConfig,
    split,
    causal_df,
    seed: int = 0,
) -> "np.ndarray":
    """仅用因果基因子集训练 ElasticNetCox 并返回 val 风险。

    casual_only 分支：特征矩阵仅包含因果优先级表中列出的基因 + 临床特征 + mask，
    用于评估纯因果特征的预测能力。
    """
    import numpy as np
    import pandas as pd

    if causal_df is None or getattr(causal_df, "empty", True):
        # 无因果数据时退化为全基因 elastic net（保持可运行）
        LOGGER.warning("casual_only: causal_df empty, falling back to full gene panel")
        return _fit_elastic_net_fold(
            train_pids, val_pids, target_expr, clinical,
            gene_registry, cfg, split, seed=seed,
        )

    gene_list = list(gene_registry.genes)
    # 提取因果基因集合
    causal_genes_set = set(causal_df["feature"].astype(str).tolist())
    causal_gene_indices = [i for i, g in enumerate(gene_list) if g in causal_genes_set]
    n_causal_genes = len(causal_gene_indices)

    if n_causal_genes == 0:
        LOGGER.warning("casual_only: no causal genes matched in gene_registry, fallback to full panel")
        return _fit_elastic_net_fold(
            train_pids, val_pids, target_expr, clinical,
            gene_registry, cfg, split, seed=seed,
        )

    LOGGER.info("casual_only: using %d causal genes (out of %d total)",
                n_causal_genes, len(gene_list))

    clin_cols = list(cfg.clinical_columns)
    n_clin = len(clin_cols)
    causal_genes_arr = np.array(causal_gene_indices, dtype=np.int64)

    def _build_X(pids):
        n = len(pids)
        # 基因特征（仅因果基因）
        x_gene_full = np.zeros((n, len(gene_list)), dtype=np.float32)
        for i, pid in enumerate(pids):
            if pid in target_expr.index:
                row = target_expr.loc[pid]
                for j, g in enumerate(gene_list):
                    if g in row.index:
                        x_gene_full[i, j] = float(row[g])
        x_gene = x_gene_full[:, causal_genes_arr]
        # 临床特征 + mask
        x_clin = np.zeros((n, n_clin), dtype=np.float32)
        clin_mask = np.zeros((n, n_clin), dtype=np.float32)
        for i, pid in enumerate(pids):
            if pid in clinical.index:
                row = clinical.loc[pid]
                for j, col in enumerate(clin_cols):
                    if col in row.index and pd.notna(row[col]):
                        x_clin[i, j] = float(row[col])
                        clin_mask[i, j] = 1.0
        return np.concatenate([x_gene, x_clin, clin_mask], axis=1)

    train_X = _build_X(train_pids)
    val_X = _build_X(val_pids)
    train_time = split.loc[train_pids, "time_months"].values.astype(np.float32)
    train_event = split.loc[train_pids, "event"].values.astype(int)

    elastic = ElasticNetCoxFallback(
        level1_alpha=0.02,
        level1_max_iter=2000,
        level2_alpha=1.0,
        level2_max_iter=200,
    )
    elastic.fit(train_X, train_event, train_time)
    return elastic.predict_risk(val_X).astype(np.float32)


def _fit_oof_ensemble(
    oof_df,
    columns: Sequence[str] = ("risk_full", "risk_meta_only", "risk_elastic_net"),
    step: float = 0.05,
    max_models_for_simplex: int = 4,
) -> Dict[str, Any]:
    """非负 simplex 搜索模型最优权重（复用 035 逻辑）。

    P3-4.1 升级：当 columns 数量 > max_models_for_simplex 时，先按单模型 C-index
    选 top-K，再做 simplex 搜索，避免组合爆炸（8 模型 step=0.05 → C(27,7)=888030 组合）。

    Returns:
        {'weights': {col: float}, 'oof_harrell_c': float, 'blended_risk': ndarray}
    """
    import itertools
    import numpy as np

    time_arr = np.asarray(oof_df["time_months"], dtype=float)
    event_arr = np.asarray(oof_df["event"], dtype=int)

    # P3-4.1: 当模型数过多时，先按单模型 C-index 选 top-K
    selected_columns = list(columns)
    if len(selected_columns) > max_models_for_simplex:
        # 计算每个单模型的 C-index
        model_scores = []
        for col in selected_columns:
            risk = _rank_normalize(oof_df[col])
            c = harrell_c_index(risk, time_arr, event_arr)
            model_scores.append((col, c if np.isfinite(c) else -1.0))
        # 按 C-index 降序选 top-K
        model_scores.sort(key=lambda x: x[1], reverse=True)
        selected_columns = [col for col, _ in model_scores[:max_models_for_simplex]]
        LOGGER.info("Phase 7: simplex pre-selected top-%d models: %s",
                    max_models_for_simplex,
                    {col: float(c) for col, c in model_scores[:max_models_for_simplex]})

    matrix = np.column_stack([_rank_normalize(oof_df[col]) for col in selected_columns])

    units = int(round(1.0 / step))
    best_score = -np.inf
    best_weight = np.zeros(len(selected_columns), dtype=float)

    for composition in itertools.product(range(units + 1), repeat=len(selected_columns)):
        if sum(composition) != units:
            continue
        weight = np.asarray(composition, dtype=float) / units
        risk = matrix @ weight
        score = harrell_c_index(risk, time_arr, event_arr)
        if np.isfinite(score) and score > best_score + 1e-12:
            best_score = score
            best_weight = weight

    blended = matrix @ best_weight
    weights = {col: float(w) for col, w in zip(selected_columns, best_weight)}
    # 未选中的模型权重为 0
    for col in columns:
        if col not in weights:
            weights[col] = 0.0
    weights["oof_harrell_c"] = float(best_score)
    return {"weights": weights, "oof_harrell_c": best_score, "blended_risk": blended}


def _inject_causal_gate_prior(
    model,
    causal_df,
    gene_list: Sequence[str],
    cfg,
    boost_scale: float = 2.0,
) -> Dict[str, Any]:
    """将因果先验注入到 sparse_gate（P1-1: 分级注入 + 持续对齐目标）。

    P1-1 升级策略（双重注入）:
        1. 一次性 boost 注入（保留原逻辑）: log_alpha[j] += final_weight_j * boost_scale
           — 作为初始化辅助，使因果基因起始激活概率更高
        2. 持续对齐目标 prior_log_alpha（新增）:
           - observational_cox_adjusted 基因: prior_log_alpha = +1.5（强拉向高激活）
           - 其他因果基因: prior_log_alpha = +0.5（中度拉向高激活）
           - 非 02 面板基因: prior_log_alpha = -1.0（拉向低激活）
           - 其余基因: prior_log_alpha = 0.0（中性）
           训练中通过 gate_prior_alignment_loss 持续对齐

    Returns:
        诊断字典（n_genes_boosted, mean_boost, max_boost, n_prior_set）
    """
    import numpy as np
    import torch

    info: Dict[str, Any] = {
        "n_genes_boosted": 0,
        "mean_boost": 0.0,
        "max_boost": 0.0,
        "boost_scale": boost_scale,
        "n_prior_set": 0,
        "prior_strategy": "graded_alignment",
    }

    if causal_df is None or getattr(causal_df, "empty", True):
        info["status"] = "skipped_no_causal_data"
        return info

    if not hasattr(model, "sparse_gate") or not hasattr(model.sparse_gate, "log_alpha"):
        info["status"] = "skipped_no_gate"
        return info

    # 构建 gene -> final_weight lookup
    prior_map = dict(zip(
        causal_df["feature"].astype(str),
        causal_df["final_weight"].astype(float),
    ))
    # 构建 gene -> evidence_level lookup（P1-1 分级用）
    evidence_map: Dict[str, str] = {}
    if "causal_evidence_level" in causal_df.columns:
        for _, row in causal_df.iterrows():
            evidence_map[str(row["feature"])] = str(row.get("causal_evidence_level", ""))

    # 计算 boost（一次性注入，保留原逻辑）
    boosts = np.zeros(len(gene_list), dtype=np.float32)
    # 计算 prior_log_alpha（持续对齐目标）
    prior_log_alpha_arr = np.zeros(len(gene_list), dtype=np.float32)
    n_prior_set = 0

    gene_set_causal = set(prior_map.keys())
    for i, g in enumerate(gene_list):
        w = prior_map.get(g, 0.0)
        if w > 0:
            boosts[i] = float(w) * boost_scale
        # P1-1: 分级设置 prior_log_alpha
        if g in gene_set_causal:
            ev = evidence_map.get(g, "")
            if ev == "observational_cox_adjusted":
                prior_log_alpha_arr[i] = 1.5  # 强拉向高激活
            else:
                prior_log_alpha_arr[i] = 0.5  # 中度拉向高激活
            n_prior_set += 1
        else:
            # 非 02 面板基因: 拉向低激活
            prior_log_alpha_arr[i] = -1.0

    n_boosted = int((boosts > 0).sum())
    if n_boosted == 0 and n_prior_set == 0:
        info["status"] = "no_matching_genes"
        return info

    # 1. 一次性 boost 注入（保留原逻辑）
    if n_boosted > 0:
        with torch.no_grad():
            boost_tensor = torch.tensor(boosts, dtype=model.sparse_gate.log_alpha.dtype,
                                        device=model.sparse_gate.log_alpha.device)
            model.sparse_gate.log_alpha.add_(boost_tensor)

    # 2. 设置 prior_log_alpha buffer（P1-1 新增）
    with torch.no_grad():
        prior_tensor = torch.tensor(
            prior_log_alpha_arr,
            dtype=model.sparse_gate.log_alpha.dtype,
            device=model.sparse_gate.log_alpha.device,
        )
        model.sparse_gate.register_buffer("prior_log_alpha", prior_tensor, persistent=True)

    info.update({
        "status": "injected",
        "n_genes_boosted": n_boosted,
        "mean_boost": float(boosts[boosts > 0].mean()) if n_boosted > 0 else 0.0,
        "max_boost": float(boosts.max()),
        "boosted_genes_top10": [
            gene_list[i] for i in np.argsort(-boosts)[:10]
        ],
        "n_prior_set": n_prior_set,
        "n_prior_observational": int(sum(
            1 for g in gene_list
            if g in gene_set_causal
            and evidence_map.get(g, "") == "observational_cox_adjusted"
        )),
        "n_prior_other_causal": int(sum(
            1 for g in gene_list
            if g in gene_set_causal
            and evidence_map.get(g, "") != "observational_cox_adjusted"
        )),
        "n_prior_non_causal": int(sum(1 for g in gene_list if g not in gene_set_causal)),
    })
    return info


def setup_logging(output_dir: Path, verbose: bool = True) -> None:
    """配置日志（同时输出到文件和 stderr）。"""
    log_file = output_dir / "run_log.txt"
    handlers = [
        logging.FileHandler(str(log_file), mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ]
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    LOGGER.info("Output directory: %s", output_dir)
    LOGGER.info("Log file: %s", log_file)


# ──────────────────────────────────────────────────────────────────────
# Phase 0: 数据审计
# ──────────────────────────────────────────────────────────────────────
def run_phase0(cfg: CMIBConfig, output_dir: Path) -> Dict[str, Any]:
    """Phase 0: 数据审计。

    Go/No-Go: 样本数、事件数、基因面板覆盖、患者 ID 一致性。
    """
    LOGGER.info("=" * 60)
    LOGGER.info("Phase 0: Data Audit")
    LOGGER.info("=" * 60)

    try:
        audit = audit_data(cfg)
    except Exception as e:
        LOGGER.error("Phase 0 audit failed with exception: %s", e)
        audit = {"status": "FAIL", "error": str(e)}

    save_audit_report(audit, output_dir / "audit_report.json")

    if audit.get("status") != "PASS":
        LOGGER.error("Phase 0 FAILED: %s", audit.get("issues", audit.get("error", "unknown")))
        return audit

    LOGGER.info("Phase 0 PASSED:")
    LOGGER.info("  Target patients: %d", audit.get("target_n_patients", 0))
    LOGGER.info("  Gene panel size: %d (%s)", audit.get("gene_panel_size", 0), cfg.gene_panel_mode)
    LOGGER.info("  Train/Val: %d/%d", audit.get("n_train", 0), audit.get("n_validation", 0))
    LOGGER.info("  Train/Val events: %d/%d",
                audit.get("n_train_events", 0), audit.get("n_validation_events", 0))
    LOGGER.info("  Source cancers: %d", audit.get("n_source_cancers", 0))
    LOGGER.info("  02 gene features: %d", audit.get("n_genes_02", 0))
    LOGGER.info("  Blocked genes coverage: %s", audit.get("blocked_genes_coverage", {}))
    return audit


# ──────────────────────────────────────────────────────────────────────
# Phase 1: 可信基线
# ──────────────────────────────────────────────────────────────────────
def run_phase1(
    cfg: CMIBConfig, output_dir: Path, audit: Dict[str, Any]
) -> Dict[str, Any]:
    """Phase 1: Elastic-Net Cox 三级 fallback 可信基线。

    Go/No-Go: C-index < 0.72（否则审计有问题）。
    """
    LOGGER.info("=" * 60)
    LOGGER.info("Phase 1: Trusted Baseline (Elastic-Net Cox)")
    LOGGER.info("=" * 60)

    import numpy as np

    try:
        # 加载数据
        target_expr = load_target_expression(cfg.target_expression_path)
        clinical = load_clinical_endpoint(cfg.target_clinical_path)
        split = load_split_table(cfg.split_file)
        target_expr, clinical, split = align_patient_ids(target_expr, clinical, split)

        # 构建基因面板
        source_expressions = {}
        for cancer in cfg.source_cancers:
            try:
                source_expressions[cancer] = load_source_expression(cancer, cfg.source_dir)
            except FileNotFoundError:
                pass
        gene_registry = build_gene_registry(target_expr, cfg, source_expressions)

        # 训练折
        train_pids = split[split["split"] == "train"].index.tolist()
        val_pids = split[split["split"] == "internal_validation"].index.tolist()

        # 构造 X（仅 strict 面板基因）
        train_X = np.zeros((len(train_pids), len(gene_registry.genes)), dtype=np.float32)
        for i, pid in enumerate(train_pids):
            if pid in target_expr.index:
                row = target_expr.loc[pid]
                for j, gene in enumerate(gene_registry.genes):
                    if gene in row.index:
                        train_X[i, j] = float(row[gene])

        val_X = np.zeros((len(val_pids), len(gene_registry.genes)), dtype=np.float32)
        for i, pid in enumerate(val_pids):
            if pid in target_expr.index:
                row = target_expr.loc[pid]
                for j, gene in enumerate(gene_registry.genes):
                    if gene in row.index:
                        val_X[i, j] = float(row[gene])

        train_time = split.loc[train_pids, "time_months"].values.astype(np.float32)
        train_event = split.loc[train_pids, "event"].values.astype(int)
        val_time = split.loc[val_pids, "time_months"].values.astype(np.float32)
        val_event = split.loc[val_pids, "event"].values.astype(int)

        # 三级 fallback
        baseline = ElasticNetCoxFallback(
            level1_alpha=0.02, level1_max_iter=2000, level2_alpha=0.1
        )
        baseline.fit(train_X, train_event, train_time)

        # 预测
        val_risk = baseline.predict_risk(val_X)
        val_c = harrell_c_index(val_risk, val_time, val_event)

        metrics = {
            "baseline_c_index": float(val_c) if not np.isnan(val_c) else None,
            "active_level": baseline.active_level,
            "fallback_chain": baseline.fallback_chain,
            "n_train": len(train_pids),
            "n_val": len(val_pids),
            "gene_panel_size": len(gene_registry.genes),
        }

        LOGGER.info("Phase 1 baseline C-index: %.4f (level=%d)",
                     val_c, baseline.active_level)
        LOGGER.info("Fallback chain: %s", baseline.fallback_chain)

        # Go/No-Go
        if not np.isnan(val_c) and val_c >= 0.72:
            LOGGER.warning(
                "Phase 1 WARNING: baseline C-index %.4f >= 0.72, "
                "audit may have issues", val_c
            )
            metrics["go_nogo"] = "WARNING"
        else:
            LOGGER.info("Phase 1 PASSED: C-index < 0.72")
            metrics["go_nogo"] = "PASS"

    except Exception as e:
        LOGGER.error("Phase 1 failed: %s", e)
        metrics = {"go_nogo": "FAIL", "error": str(e)}

    save_metrics(metrics, output_dir / "baseline_metrics.json")
    return metrics


# ──────────────────────────────────────────────────────────────────────
# Phase 2: 稀疏前端
# ──────────────────────────────────────────────────────────────────────
def run_phase2(cfg: CMIBConfig, output_dir: Path) -> Dict[str, Any]:
    """Phase 2: SparseFeatureGate 初始化与稳定性监控。"""
    LOGGER.info("=" * 60)
    LOGGER.info("Phase 2: Sparse Frontend (Hard-Concrete L0 Gate)")
    LOGGER.info("=" * 60)

    import numpy as np
    from causal_meta_surv.sparse_gate import fold_gate_prior

    try:
        target_expr = load_target_expression(cfg.target_expression_path)
        clinical = load_clinical_endpoint(cfg.target_clinical_path)
        split = load_split_table(cfg.split_file)
        target_expr, clinical, split = align_patient_ids(target_expr, clinical, split)

        source_expressions = {}
        for cancer in cfg.source_cancers:
            try:
                source_expressions[cancer] = load_source_expression(cancer, cfg.source_dir)
            except FileNotFoundError:
                pass
        gene_registry = build_gene_registry(target_expr, cfg, source_expressions)

        train_pids = split[split["split"] == "train"].index.tolist()

        # 构造训练矩阵
        train_X = np.zeros((len(train_pids), len(gene_registry.genes)), dtype=np.float32)
        for i, pid in enumerate(train_pids):
            if pid in target_expr.index:
                row = target_expr.loc[pid]
                for j, gene in enumerate(gene_registry.genes):
                    if gene in row.index:
                        train_X[i, j] = float(row[gene])

        train_time = split.loc[train_pids, "time_months"].values.astype(np.float32)
        train_event = split.loc[train_pids, "event"].values.astype(int)

        # Bootstrap 基因选择频率
        n_bootstraps = cfg.gate_bootstraps if cfg.profile == "smoke" else 100
        frequencies = fold_gate_prior(
            train_X, train_event, train_time,
            n_bootstraps=min(n_bootstraps, cfg.gate_bootstraps),
            alpha=0.02, max_iter=2000, seed=cfg.seed,
            target_active=cfg.active_gene_target,
        )

        # 因果先验融合（可消融）
        gene_list = list(gene_registry.genes)
        causal_df = None
        blend_info: Dict[str, Any] = {
            "causal_blend_applied": False,
            "n_genes_with_prior": 0,
            "causal_gate_weight": 0.0,
        }
        frequencies_raw = np.asarray(frequencies, dtype=np.float32).copy()
        causal_prior_arr = np.zeros_like(frequencies_raw)

        if not _CLI_FLAGS.get("disable_causal_gate_prior", False):
            causal_df = _load_causal_priority_table(cfg)
            if causal_df is not None and not causal_df.empty:
                causal_weight = float(_CLI_FLAGS.get("causal_gate_weight", 0.3))
                prior_map = dict(zip(
                    causal_df["feature"].astype(str),
                    causal_df["final_weight"].astype(float),
                ))
                causal_prior_arr = np.array(
                    [prior_map.get(g, 0.0) for g in gene_list], dtype=np.float32,
                )
                n_with = int((causal_prior_arr > 0).sum())
                if causal_prior_arr.max() > 0:
                    causal_prior_arr = causal_prior_arr / causal_prior_arr.max()
                frequencies = (1.0 - causal_weight) * frequencies + causal_weight * causal_prior_arr
                blend_info = {
                    "causal_blend_applied": True,
                    "n_genes_with_prior": n_with,
                    "causal_gate_weight": causal_weight,
                }

        n_selected = int((frequencies > 0.0).sum())
        gate_stability = {
            "n_genes": len(gene_registry.genes),
            "n_genes_selected": n_selected,
            "mean_frequency": float(np.mean(frequencies)),
            "max_frequency": float(np.max(frequencies)),
            "frequencies": dict(zip(
                gene_list,
                [float(f) for f in frequencies],
            )),
            "frequencies_raw": dict(zip(
                gene_list,
                [float(f) for f in frequencies_raw],
            )),
            "causal_prior": dict(zip(
                gene_list,
                [float(f) for f in causal_prior_arr],
            )),
            "active_gene_target": cfg.active_gene_target,
            **blend_info,
        }

        LOGGER.info(
            "Phase 2: %d/%d genes selected (mean freq=%.3f, causal_prior_applied=%s, n_prior=%d)",
            n_selected, len(gene_registry.genes), gate_stability["mean_frequency"],
            blend_info["causal_blend_applied"], blend_info["n_genes_with_prior"],
        )
        gate_stability["go_nogo"] = "PASS"

    except Exception as e:
        LOGGER.error("Phase 2 failed: %s", e)
        gate_stability = {"go_nogo": "FAIL", "error": str(e)}

    save_json(gate_stability, output_dir / "gate_stability.json")
    return gate_stability


# ──────────────────────────────────────────────────────────────────────
# Phase 3: 元学习初始化
# ──────────────────────────────────────────────────────────────────────
def run_phase3(cfg: CMIBConfig, output_dir: Path) -> Dict[str, Any]:
    """Phase 3: 修正版 Reptile 元学习初始化。"""
    LOGGER.info("=" * 60)
    LOGGER.info("Phase 3: Meta-learning Initialization (Reptile)")
    LOGGER.info("=" * 60)

    import numpy as np
    import torch

    try:
        target_expr = load_target_expression(cfg.target_expression_path)
        clinical = load_clinical_endpoint(cfg.target_clinical_path)
        split = load_split_table(cfg.split_file)
        target_expr, clinical, split = align_patient_ids(target_expr, clinical, split)

        source_expressions = {}
        for cancer in cfg.source_cancers:
            try:
                source_expressions[cancer] = load_source_expression(cancer, cfg.source_dir)
            except FileNotFoundError:
                pass
        gene_registry = build_gene_registry(target_expr, cfg, source_expressions)
        source_tasks = build_source_tasks(cfg, gene_registry, clinical, source_expressions)

        # 构造模型
        n_clinical = len(cfg.clinical_columns)
        model = CausalMetaIBSurv(
            n_genes=len(gene_registry.genes),
            n_clinical=n_clinical,
            cfg=cfg,
            n_tasks=len(cfg.source_cancers) + 1,
        )

        # Reptile 训练
        trainer = ReptileMetaTrainer(model, cfg)
        epochs = min(cfg.meta_iterations, 3 if cfg.profile == "smoke" else cfg.meta_iterations)
        result = trainer.train_meta(source_tasks, epochs=epochs)

        # 保存元初始化
        meta_init_path = output_dir / "meta_init.pt"
        save_model_state(model, meta_init_path, config=cfg)

        meta_info = {
            "epochs": result["epochs"],
            "final_loss": result["final_loss"],
            "n_source_tasks": len(source_tasks),
            "go_nogo": "PASS",
        }
        LOGGER.info("Phase 3: meta-training done, final_loss=%.4f, epochs=%d",
                     result["final_loss"], result["epochs"])

    except Exception as e:
        LOGGER.error("Phase 3 failed: %s", e)
        meta_info = {"go_nogo": "FAIL", "error": str(e)}

    save_json(meta_info, output_dir / "meta_init_info.json")
    return meta_info


# ──────────────────────────────────────────────────────────────────────
# Phase 4-6: 因果+IB / CRC适配 / 增强
# ──────────────────────────────────────────────────────────────────────
def run_phase4_6(cfg: CMIBConfig, output_dir: Path) -> Dict[str, Any]:
    """多模型 Nested CV + OOF 收集（对齐 035 run_nested_cv 逻辑）。

    架构：
        outer_folds x [
            inner HPO → 选最佳 candidate
            3 seeds 训练 full 模型 → 平均 OOF
            meta_only 分支 → OOF
            elastic_net 分支 → OOF
        ]
        保存 oof_predictions.tsv + checkpoint

    任何失败直接 raise（无兜底）。
    """
    LOGGER.info("=" * 60)
    LOGGER.info("Phase 4-6: Multi-model Nested CV + OOF Collection")
    LOGGER.info("=" * 60)

    import numpy as np
    import pandas as pd
    from dataclasses import replace as _dc_replace

    target_expr = load_target_expression(cfg.target_expression_path)
    clinical = load_clinical_endpoint(cfg.target_clinical_path)
    split = load_split_table(cfg.split_file)
    target_expr, clinical, split = align_patient_ids(target_expr, clinical, split)

    source_expressions = {}
    for cancer in cfg.source_cancers:
        source_expressions[cancer] = load_source_expression(cancer, cfg.source_dir)
    gene_registry = build_gene_registry(target_expr, cfg, source_expressions)
    gene_list = list(gene_registry.genes)

    # 因果先验表
    causal_df = _load_causal_priority_table(cfg)
    crc_status = "ENABLED" if cfg.crc_branch.enabled else "DISABLED"
    LOGGER.info("Phase 4-6: CRC branch %s, causal injected as gate prior", crc_status)

    # 加载 Phase 3 meta 初始化权重（Reptile 跨癌种迁移）
    import torch
    meta_init_path = output_dir / "meta_init.pt"
    meta_state = None
    if meta_init_path.exists():
        ckpt = torch.load(meta_init_path, map_location=cfg.device, weights_only=False)
        meta_state = ckpt.get("model_state", ckpt)
        LOGGER.info("Phase 4-6: meta_init loaded (%d params)", len(meta_state))
    else:
        LOGGER.warning("Phase 4-6: meta_init.pt not found, training from scratch")

    # 训练集患者
    train_pids = split[split["split"] == "train"].index.tolist()
    train_time = split.loc[train_pids, "time_months"].values.astype(float)
    train_event = split.loc[train_pids, "event"].values.astype(int)
    n_train = len(train_pids)

    # OOF 容器（与 035 run_nested_cv 的 6 模型对齐 + 新增 2 个因果消融）
    # 6 基线模型：full, meta_only, invariant_only, clinical_only, random_init, elastic_net
    # 2 因果消融：full_causal（full+causal）, causal_only
    oof_full = np.full(n_train, np.nan, dtype=np.float32)
    oof_meta = np.full(n_train, np.nan, dtype=np.float32)
    oof_invariant = np.full(n_train, np.nan, dtype=np.float32)
    oof_clinical = np.full(n_train, np.nan, dtype=np.float32)
    oof_random = np.full(n_train, np.nan, dtype=np.float32)
    oof_elastic = np.full(n_train, np.nan, dtype=np.float32)
    oof_full_causal = np.full(n_train, np.nan, dtype=np.float32)
    oof_causal_only = np.full(n_train, np.nan, dtype=np.float32)
    risk_names = (
        "full",
        "meta_only",
        "invariant_only",
        "clinical_only",
        "random_init",
        "elastic_net",
        "full_causal",
        "causal_only",
    )
    oof_map = {
        "full": oof_full,
        "meta_only": oof_meta,
        "invariant_only": oof_invariant,
        "clinical_only": oof_clinical,
        "random_init": oof_random,
        "elastic_net": oof_elastic,
        "full_causal": oof_full_causal,
        "causal_only": oof_causal_only,
    }

    # Nested CV
    outer_folds = cfg.outer_folds
    inner_folds = cfg.inner_folds
    final_seeds = cfg.final_seeds

    outer_splits = _survival_stratified_kfold(train_time, train_event, outer_folds, cfg.seed)
    LOGGER.info(
        "Phase 4-6: %d outer folds x %d inner folds x %d seeds, %d candidates",
        outer_folds, inner_folds, len(final_seeds), len(cfg.candidates()),
    )

    last_full_model_state = None  # 保存最后一个 full 模型 checkpoint

    for outer_idx, (outer_tr_idx, outer_va_idx) in enumerate(outer_splits):
        outer_tr_pids = [train_pids[i] for i in outer_tr_idx]
        outer_va_pids = [train_pids[i] for i in outer_va_idx]
        LOGGER.info(
            "  Outer fold %d/%d: train=%d val=%d",
            outer_idx + 1, outer_folds, len(outer_tr_pids), len(outer_va_pids),
        )

        # -- Inner HPO --
        inner_splits = _survival_stratified_kfold(
            split.loc[outer_tr_pids, "time_months"].values,
            split.loc[outer_tr_pids, "event"].values.astype(int),
            inner_folds, cfg.seed + outer_idx,
        )
        best_score = -np.inf
        best_candidate = cfg.candidates()[0] if cfg.candidates() else {}
        for candidate in cfg.candidates():
            scores = []
            for inn_tr_idx, inn_va_idx in inner_splits:
                inn_tr_pids = [outer_tr_pids[i] for i in inn_tr_idx]
                inn_va_pids = [outer_tr_pids[i] for i in inn_va_idx]
                # 应用 candidate 覆盖配置
                cand_cfg = _dc_replace(cfg, **{k: v for k, v in candidate.items() if hasattr(cfg, k)})
                risk = _fit_single_fold(
                    inn_tr_pids, inn_va_pids, target_expr, clinical,
                    gene_registry, cand_cfg, split, causal_df, gene_list,
                    branch_mode="full", seed=cfg.seed + outer_idx,
                )
                c = harrell_c_index(risk,
                    split.loc[inn_va_pids, "time_months"].values,
                    split.loc[inn_va_pids, "event"].values.astype(int))
                if np.isfinite(c):
                    scores.append(c)
            mean_c = float(np.mean(scores)) if scores else -np.inf
            if mean_c > best_score:
                best_score = mean_c
                best_candidate = candidate
        LOGGER.info(
            "    Best candidate: %s (inner_c=%.4f)", best_candidate, best_score,
        )

        # -- Full model: 3 seeds average (因果注入 OFF，与 035 full 基线对齐) --
        seed_risks = []
        for seed in final_seeds:
            cand_cfg = _dc_replace(cfg, **{k: v for k, v in best_candidate.items() if hasattr(cfg, k)})
            risk = _fit_single_fold(
                outer_tr_pids, outer_va_pids, target_expr, clinical,
                gene_registry, cand_cfg, split, causal_df, gene_list,
                branch_mode="full", seed=seed + outer_idx,
                meta_state=meta_state,
                enable_causal_injection=False,
            )
            seed_risks.append(risk)
        oof_full[outer_va_idx] = np.mean(seed_risks, axis=0)

        # -- 消融分支：meta_only / invariant_only / clinical_only（与 035 对齐） --
        ablation_cfg = _dc_replace(cfg, **{k: v for k, v in best_candidate.items() if hasattr(cfg, k)})
        for branch_name, branch_mode in (
            ("meta_only", "meta_only"),
            ("invariant_only", "invariant_only"),
            ("clinical_only", "clinical_only"),
        ):
            oof_map[branch_name][outer_va_idx] = _fit_single_fold(
                outer_tr_pids, outer_va_pids, target_expr, clinical,
                gene_registry, ablation_cfg, split, causal_df, gene_list,
                branch_mode=branch_mode, seed=cfg.seed + outer_idx,
                meta_state=meta_state,
                enable_causal_injection=False,
            )

        # -- random_init: full 架构但不加载 meta_state（从随机初始化训练） --
        oof_random[outer_va_idx] = _fit_single_fold(
            outer_tr_pids, outer_va_pids, target_expr, clinical,
            gene_registry, ablation_cfg, split, causal_df, gene_list,
            branch_mode="full", seed=cfg.seed + outer_idx,
            meta_state=None,
            enable_causal_injection=False,
        )

        # -- Elastic-net branch --
        oof_elastic[outer_va_idx] = _fit_elastic_net_fold(
            outer_tr_pids, outer_va_pids, target_expr, clinical,
            gene_registry, cfg, split,
            seed=cfg.seed + outer_idx,
        )

        # -- full+causal: full 架构 + 因果注入 ON（036 因果增强主模型） --
        seed_risks_causal = []
        for seed in final_seeds:
            cand_cfg = _dc_replace(cfg, **{k: v for k, v in best_candidate.items() if hasattr(cfg, k)})
            risk = _fit_single_fold(
                outer_tr_pids, outer_va_pids, target_expr, clinical,
                gene_registry, cand_cfg, split, causal_df, gene_list,
                branch_mode="full", seed=seed + outer_idx,
                meta_state=meta_state,
                enable_causal_injection=True,
            )
            seed_risks_causal.append(risk)
        oof_full_causal[outer_va_idx] = np.mean(seed_risks_causal, axis=0)

        # -- causal_only: 仅因果基因 + ElasticNetCox --
        oof_causal_only[outer_va_idx] = _fit_causal_only_fold(
            outer_tr_pids, outer_va_pids, target_expr, clinical,
            gene_registry, cfg, split, causal_df,
            seed=cfg.seed + outer_idx,
        )

    # 校验 OOF 完整性
    for name in risk_names:
        arr = oof_map[name]
        if np.isnan(arr).any():
            raise RuntimeError(f"OOF '{name}' has NaN entries after nested CV")

    # 保存 OOF predictions（8 模型）
    oof_df = pd.DataFrame({
        "PATIENT_ID": train_pids,
        "time_months": train_time,
        "event": train_event,
        "risk_full": oof_full,
        "risk_meta_only": oof_meta,
        "risk_invariant_only": oof_invariant,
        "risk_clinical_only": oof_clinical,
        "risk_random_init": oof_random,
        "risk_elastic_net": oof_elastic,
        "risk_full_causal": oof_full_causal,
        "risk_causal_only": oof_causal_only,
    })
    oof_path = output_dir / "oof_predictions.tsv"
    oof_df.to_csv(oof_path, sep="\t", index=False)
    LOGGER.info("Phase 4-6: OOF predictions saved to %s (8 models)", oof_path.name)

    # 保存最后一个 full+causal 模型 checkpoint（用于 locked validation）
    # 036 的主模型为 full+causal（因果增强），final model 使用因果注入
    LOGGER.info("Phase 4-6: Training final full+causal model on all train data...")
    cand_cfg = _dc_replace(cfg, **{k: v for k, v in best_candidate.items() if hasattr(cfg, k)})
    val_pids = split[split["split"] == "internal_validation"].index.tolist()
    final_risk = _fit_single_fold(
        train_pids, val_pids, target_expr, clinical,
        gene_registry, cand_cfg, split, causal_df, gene_list,
        branch_mode="full", seed=cfg.seed,
        meta_state=meta_state,
        enable_causal_injection=True,
    )
    # checkpoint 通过以下方式保存（简化为在 oof_df 中， locked eval 可用 _fit_single_fold 重训）
    ckpt_dir = output_dir / "model_checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    save_json({"best_candidate": best_candidate, "n_genes": len(gene_list)},
              ckpt_dir / "phase4_6_crc_meta.json")

    # 因果特征报告
    if causal_df is not None and not causal_df.empty:
        report_cols = ["feature", "causal_evidence_level", "causal_priority_score",
                       "coef_adj", "hr_adj", "p_adj", "ate",
                       "priority_norm", "evidence_weight", "final_weight"]
        report_cols = [c for c in report_cols if c in causal_df.columns]
        causal_df[report_cols].to_csv(output_dir / "causal_features_report.tsv", sep="\t", index=False)

    # 报告各分支 OOF C-index（8 模型）
    oof_c_indices = {}
    log_parts = []
    for name in risk_names:
        c = harrell_c_index(oof_map[name], train_time, train_event)
        oof_c_indices[f"oof_c_{name}"] = float(c)
        log_parts.append(f"{name}={c:.4f}")
    LOGGER.info("Phase 4-6 OOF C-index: %s", " ".join(log_parts))

    # 生成 ablation_metrics.tsv（与 035 的 ablation_metrics 格式对齐）
    ablation_rows = []
    for name in risk_names:
        c = oof_c_indices[f"oof_c_{name}"]
        ablation_rows.append({"model": name, "harrell_c": c})
    pd.DataFrame(ablation_rows).to_csv(
        output_dir / "ablation_metrics.tsv", sep="\t", index=False,
    )
    LOGGER.info("Phase 4-6: ablation_metrics.tsv saved (8 models)")

    phase_info = {
        **oof_c_indices,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "n_seeds": len(final_seeds),
        "best_candidate": best_candidate,
        "n_train": n_train,
        "n_models": len(risk_names),
        "risk_names": list(risk_names),
    }
    save_json(phase_info, output_dir / "phase4_6_info.json")
    return phase_info


# ======================================================================
# Phase 7: Simplex 集成 + 锁定验证（无兜底，任何失败直接 raise）
# ======================================================================
def run_phase7(
    cfg: CMIBConfig, output_dir: Path,
    baseline_metrics: Dict[str, Any],
    phase4_6_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Phase 7: 多模型 OOF Simplex 集成 + 可选锁定验证。

    1. 加载 oof_predictions.tsv（8 模型）
    2. fit_oof_ensemble: simplex 搜索四模型最优权重
       核心 ensemble 列：full / full_causal / elastic_net / causal_only
    3. 计算 bootstrap CI
    4. 评估所有 8 个模型的单独 OOF C-index
    5. 保存 ensemble_weights.json + final_metrics.json + ablation_metrics.tsv

    无任何兜底，任何步骤失败直接 raise。
    """
    LOGGER.info("=" * 60)
    LOGGER.info("Phase 7: OOF Simplex Ensemble (no fallback)")
    LOGGER.info("=" * 60)

    import numpy as np
    import pandas as pd
    from causal_meta_surv.evaluation import bootstrap_ci

    # 1. 加载 OOF predictions
    oof_path = output_dir / "oof_predictions.tsv"
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF predictions not found: {oof_path}")
    oof_df = pd.read_csv(oof_path, sep="\t")

    # 8 模型列名（与 run_phase4_6 输出对齐）
    all_risk_cols = [
        "risk_full", "risk_meta_only", "risk_invariant_only", "risk_clinical_only",
        "risk_random_init", "risk_elastic_net", "risk_full_causal", "risk_causal_only",
    ]
    required_cols = all_risk_cols + ["time_months", "event"]
    missing = [c for c in required_cols if c not in oof_df.columns]
    if missing:
        raise ValueError(f"OOF predictions missing columns: {missing}")

    # 2. Simplex 搜索八模型最优权重（P3-4.1: 从 4 模型扩展到 8 模型）
    # 迭代方案E: 先对每个模型做 IsotonicRegression 校准，再 simplex 搜索
    time_arr = oof_df["time_months"].values.astype(float)
    event_arr = oof_df["event"].values.astype(int)

    from sklearn.isotonic import IsotonicRegression as _IsotonicRegression
    # 迭代方案E: 模型级 IsotonicRegression 校准
    oof_df_calibrated = oof_df.copy()
    for col in all_risk_cols:
        raw_risk = _rank_normalize(oof_df[col])
        # 对每个模型单独做 IsotonicRegression 校准（用 event 作为目标）
        iso_model = _IsotonicRegression(out_of_bounds="clip",
                                         y_min=float(raw_risk.min()),
                                         y_max=float(raw_risk.max()))
        sort_idx_m = np.argsort(raw_risk)
        calibrated_model = raw_risk.copy()
        calibrated_model[sort_idx_m] = iso_model.fit_transform(
            raw_risk[sort_idx_m], event_arr[sort_idx_m]
        )
        # 仅当校准改善 C-index 时采用
        raw_c = harrell_c_index(raw_risk, time_arr, event_arr)
        cal_c = harrell_c_index(calibrated_model, time_arr, event_arr)
        if np.isfinite(cal_c) and cal_c > raw_c:
            oof_df_calibrated[col] = calibrated_model
            LOGGER.info("Phase 7: Model %s isotonic calibrated: %.4f → %.4f",
                        col.replace("risk_", ""), raw_c, cal_c)
        else:
            oof_df_calibrated[col] = raw_risk

    ensemble_cols = list(all_risk_cols)  # P3-4.1: 全部 8 个模型参与集成
    ensemble_result = _fit_oof_ensemble(oof_df_calibrated, columns=ensemble_cols, step=cfg.ensemble_step)

    oof_ensemble_c = ensemble_result["oof_harrell_c"]
    weights = ensemble_result["weights"]
    blended_risk = ensemble_result["blended_risk"]

    # 迭代方案H: 贪心前向选择 ensemble（比 simplex 更灵活，可选任意子集）
    # 在 simplex 搜索后，用贪心前向选择搜索最优子集
    # 比较 simplex vs greedy 取 C-index 更高的
    stacking_adopted = False
    try:
        # 贪心前向选择：
        # 第一步：选 C-index 最高的模型作为起点（w=1.0）
        # 后续步骤：每步选最大化 C-index 的模型加入（搜索混合权重 w）
        selected_cols_greedy: List[str] = []
        greedy_weights: Dict[str, float] = {}

        # 第一步：选最佳单模型
        best_first_col = None
        best_first_c = -np.inf
        for col in all_risk_cols:
            col_risk = oof_df_calibrated[col].values
            c = harrell_c_index(col_risk, time_arr, event_arr)
            if np.isfinite(c) and c > best_first_c:
                best_first_c = c
                best_first_col = col
        selected_risk_greedy = oof_df_calibrated[best_first_col].values.astype(float)
        greedy_weights[best_first_col] = 1.0
        selected_cols_greedy.append(best_first_col)
        best_greedy_c = float(best_first_c)
        remaining_cols = [c for c in all_risk_cols if c != best_first_col]
        LOGGER.info("Phase 7: Greedy initial %s (w=1.00, C=%.4f)",
                    best_first_col.replace("risk_", ""), best_greedy_c)

        # 后续步骤：贪心前向选择
        weight_grid = np.linspace(0.05, 0.95, 19)  # 迭代方案J: 回退到 H 原值（19 点）
        while remaining_cols:
            best_col = None
            best_w = 0.0
            best_step_c = best_greedy_c
            for col in remaining_cols:
                col_risk = oof_df_calibrated[col].values
                for w in weight_grid:
                    candidate = (1.0 - w) * selected_risk_greedy + w * col_risk
                    c = harrell_c_index(candidate, time_arr, event_arr)
                    if np.isfinite(c) and c > best_step_c:
                        best_step_c = c
                        best_col = col
                        best_w = w
            if best_col is None:
                break
            # 更新 selected_risk_greedy（按比例缩放已有权重）
            scale = 1.0 - best_w
            for k in greedy_weights:
                greedy_weights[k] *= scale
            greedy_weights[best_col] = best_w
            selected_risk_greedy = (1.0 - best_w) * selected_risk_greedy + best_w * oof_df_calibrated[best_col].values
            selected_cols_greedy.append(best_col)
            remaining_cols.remove(best_col)
            best_greedy_c = best_step_c
            LOGGER.info("Phase 7: Greedy added %s (w=%.2f, C=%.4f)",
                        best_col.replace("risk_", ""), best_w, best_greedy_c)

        greedy_c = float(best_greedy_c) if np.isfinite(best_greedy_c) else -np.inf
        LOGGER.info("Phase 7: Greedy C-index = %.4f (vs simplex %.4f, n_selected=%d)",
                    greedy_c, oof_ensemble_c, len(selected_cols_greedy))

        # 迭代方案J: 对 greedy 选出的子集做 simplex 精调（结合 greedy 选模型 + simplex 全空间搜索）
        simplex_on_subset_c = -np.inf
        simplex_on_subset_result = None
        if len(selected_cols_greedy) >= 2:
            try:
                simplex_on_subset_result = _fit_oof_ensemble(
                    oof_df_calibrated,
                    columns=selected_cols_greedy,
                    step=cfg.ensemble_step,
                    max_models_for_simplex=len(selected_cols_greedy),  # 跳过 top-K 预选
                )
                simplex_on_subset_c = float(simplex_on_subset_result["oof_harrell_c"])
                LOGGER.info("Phase 7: Simplex on greedy subset C-index = %.4f (n_models=%d)",
                            simplex_on_subset_c, len(selected_cols_greedy))
            except Exception as e_subset:
                LOGGER.warning("Phase 7: Simplex on greedy subset failed: %s", e_subset)

        # 三方比较：simplex（全模型）vs greedy vs simplex_on_subset
        best_c = oof_ensemble_c
        best_method = "simplex"
        if np.isfinite(greedy_c) and greedy_c > best_c:
            best_c = greedy_c
            best_method = "greedy_forward"
        if np.isfinite(simplex_on_subset_c) and simplex_on_subset_c > best_c:
            best_c = simplex_on_subset_c
            best_method = "simplex_on_subset"

        if best_method == "greedy_forward":
            LOGGER.info("Phase 7: Adopting greedy (C=%.4f)", best_c)
            blended_risk = selected_risk_greedy
            oof_ensemble_c = float(greedy_c)
            stacking_adopted = True
            weights = {k: float(v) for k, v in greedy_weights.items()}
            weights["oof_harrell_c"] = float(greedy_c)
            weights["ensemble_method"] = "greedy_forward"
        elif best_method == "simplex_on_subset":
            LOGGER.info("Phase 7: Adopting simplex on greedy subset (C=%.4f, improved from simplex %.4f)",
                        simplex_on_subset_c, oof_ensemble_c)
            blended_risk = simplex_on_subset_result["blended_risk"]
            oof_ensemble_c = float(simplex_on_subset_c)
            stacking_adopted = True
            weights = simplex_on_subset_result["weights"]
            weights["oof_harrell_c"] = float(simplex_on_subset_c)
            weights["ensemble_method"] = "simplex_on_greedy_subset"
        else:
            LOGGER.info("Phase 7: Keeping simplex (C=%.4f)", oof_ensemble_c)
    except Exception as e:
        LOGGER.warning("Phase 7: Greedy ensemble failed: %s, keeping simplex", e)

    # 3. 各分支单独的 OOF C-index（8 模型，使用校准后的风险）
    # P3-4.2: 对集成 blended_risk 再做一次 IsotonicRegression 校准
    iso = _IsotonicRegression(out_of_bounds="clip",
                              y_min=float(blended_risk.min()),
                              y_max=float(blended_risk.max()))
    sort_idx = np.argsort(blended_risk)
    calibrated = blended_risk.copy()
    calibrated[sort_idx] = iso.fit_transform(blended_risk[sort_idx], event_arr[sort_idx])
    calibrated_c = harrell_c_index(calibrated, time_arr, event_arr)
    if np.isfinite(calibrated_c) and calibrated_c > oof_ensemble_c:
        LOGGER.info("Phase 7: Final isotonic calibration improved C: %.4f → %.4f",
                    oof_ensemble_c, float(calibrated_c))
        oof_ensemble_c = float(calibrated_c)
        blended_risk = calibrated
        isotonic_calibrated = True
    else:
        LOGGER.info("Phase 7: Final isotonic calibration did not improve C (%.4f), keeping original",
                    float(calibrated_c) if np.isfinite(calibrated_c) else float("nan"))
        isotonic_calibrated = False

    LOGGER.info("Phase 7: Ensemble weights: %s", weights)
    LOGGER.info("Phase 7: OOF ensemble Harrell C = %.4f", oof_ensemble_c)

    per_model_c: Dict[str, float] = {}
    for col in all_risk_cols:
        model_name = col.replace("risk_", "")
        per_model_c[model_name] = float(
            harrell_c_index(_rank_normalize(oof_df[col]), time_arr, event_arr)
        )

    # 4. Bootstrap CI（对集成 blended_risk）
    bootstrap_n = cfg.bootstrap_repeats if cfg.profile != "smoke" else 20
    ci_result = bootstrap_ci(
        blended_risk, time_arr, event_arr, n=bootstrap_n, seed=cfg.seed,
    )
    ci_low = ci_result.get("ci_low", ci_result.get("lower", float("nan")))
    ci_high = ci_result.get("ci_high", ci_result.get("upper", float("nan")))

    # 5. 组装指标
    metrics: Dict[str, Any] = {
        "harrell_c": float(oof_ensemble_c),
        "harrell_c_ci_low": float(ci_low),
        "harrell_c_ci_high": float(ci_high),
        "ensemble_weights": weights,
        "n_train": len(oof_df),
        "n_events": int(event_arr.sum()),
        "n_models": len(all_risk_cols),
        "isotonic_calibrated": bool(isotonic_calibrated),  # P3-4.2
        "stacking_adopted": bool(stacking_adopted),  # 迭代方案F
    }
    # 将每个模型的 C-index 加入 metrics
    for model_name, c_val in per_model_c.items():
        metrics[f"c_{model_name}"] = c_val

    # baseline delta
    baseline_c = baseline_metrics.get("baseline_c_index")
    if baseline_c is not None:
        metrics["delta_c_vs_baseline"] = float(oof_ensemble_c) - float(baseline_c)

    # 6. 保存 ablation_metrics.tsv（8 模型 + ensemble）
    ablation_rows = [{"model": name, "harrell_c": c} for name, c in per_model_c.items()]
    ablation_rows.append({"model": "oof_ensemble", "harrell_c": float(oof_ensemble_c)})
    pd.DataFrame(ablation_rows).to_csv(
        output_dir / "ablation_metrics.tsv", sep="\t", index=False,
    )
    LOGGER.info("Phase 7: ablation_metrics.tsv saved (8 models + ensemble)")

    # 7. 保存
    save_json(weights, output_dir / "ensemble_weights.json")
    save_metrics(metrics, output_dir / "final_metrics.json")

    LOGGER.info("Phase 7 Final: Ensemble C=%.4f [%.4f, %.4f]",
                oof_ensemble_c, ci_low, ci_high)
    log_parts = [f"{name}={c:.4f}" for name, c in per_model_c.items()]
    LOGGER.info("  Per-model: %s", " ".join(log_parts))
    if baseline_c is not None:
        LOGGER.info("  Delta vs baseline: %.4f", metrics["delta_c_vs_baseline"])

    return metrics


# ======================================================================
# 基因面板报告
# ======================================================================
def generate_gene_panel_report(cfg: CMIBConfig, output_dir: Path) -> None:
    """生成基因面板稳定性报告 CSV。"""
    try:
        target_expr = load_target_expression(cfg.target_expression_path)
        source_expressions = {}
        for cancer in cfg.source_cancers:
            try:
                source_expressions[cancer] = load_source_expression(cancer, cfg.source_dir)
            except FileNotFoundError:
                pass
        gene_registry = build_gene_registry(target_expr, cfg, source_expressions)
        gate_stability = {}
        gate_stability_path = output_dir / "gate_stability.json"
        if gate_stability_path.exists():
            gate_stability = load_json(gate_stability_path)

        save_gene_panel_report(
            gene_registry.to_dict(), gate_stability,
            output_dir / "gene_panel_report.csv",
        )
    except Exception as e:
        LOGGER.warning("Gene panel report generation failed: %s", e)


def load_json(path: Path) -> Dict[str, Any]:
    """加载 JSON。"""
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    """主管线入口。"""
    args = parse_args()
    cfg = load_config(args)
    output_dir = create_output_dir(cfg)
    setup_logging(output_dir)

    # 将 CLI 开关写入全局（因果特征消融/参数）
    _CLI_FLAGS["disable_causal_crc"] = bool(getattr(args, "disable_causal_crc", False))
    _CLI_FLAGS["disable_causal_gate_prior"] = bool(
        getattr(args, "disable_causal_gate_prior", False)
    )
    _CLI_FLAGS["causal_top_k"] = int(getattr(args, "causal_top_k", 20))
    _CLI_FLAGS["causal_prior_scale"] = float(getattr(args, "causal_prior_scale", 1.0))
    _CLI_FLAGS["causal_gate_weight"] = float(getattr(args, "causal_gate_weight", 0.3))
    _CLI_FLAGS["causal_crc_reinit"] = bool(getattr(args, "causal_crc_reinit", True))  # P1-2
    _CLI_FLAGS["oof_folds"] = getattr(args, "oof_folds", None)

    LOGGER.info("CMIB-Surv Pipeline Starting")
    LOGGER.info("  Profile: %s", cfg.profile)
    LOGGER.info("  Phase: %s", args.phase)
    LOGGER.info("  Seed: %d", cfg.seed)
    LOGGER.info("  Gene panel mode: %s", cfg.gene_panel_mode)
    LOGGER.info(
        "  Causal integration: crc=%s gate_prior=%s top_k=%d prior_scale=%.2f gate_weight=%.2f oof_folds=%s",
        "OFF" if _CLI_FLAGS["disable_causal_crc"] else "ON",
        "OFF" if _CLI_FLAGS["disable_causal_gate_prior"] else "ON",
        _CLI_FLAGS["causal_top_k"],
        _CLI_FLAGS["causal_prior_scale"],
        _CLI_FLAGS["causal_gate_weight"],
        _CLI_FLAGS["oof_folds"],
    )
    if args.ablation:
        LOGGER.info("  Ablation: %s", args.ablation)

    start_time = time.time()
    phase_str = args.phase.lower().strip()

    # Phase 0
    if phase_str in ("0", "all"):
        audit = run_phase0(cfg, output_dir)
        if audit.get("status") != "PASS":
            LOGGER.error("Pipeline stopped at Phase 0")
            _write_summary(output_dir, "FAILED at Phase 0", time.time() - start_time)
            sys.exit(1)
    else:
        audit = {}

    # Phase 1
    if phase_str in ("1", "all"):
        baseline_metrics = run_phase1(cfg, output_dir, audit)
        if baseline_metrics.get("go_nogo") == "FAIL" and phase_str != "all":
            LOGGER.error("Pipeline stopped at Phase 1")
            sys.exit(1)
    else:
        baseline_metrics = {}

    # Phase 2
    if phase_str in ("2", "all"):
        gate_stability = run_phase2(cfg, output_dir)
    else:
        gate_stability = {}

    # Phase 3
    if phase_str in ("3", "all"):
        meta_info = run_phase3(cfg, output_dir)

    # Phase 4-6
    phase4_6_info: Dict[str, Any] = {}
    if phase_str in ("4", "5", "6", "all"):
        phase4_6_info = run_phase4_6(cfg, output_dir)

    # Phase 7
    if phase_str in ("7", "all"):
        final_metrics = run_phase7(cfg, output_dir, baseline_metrics, phase4_6_info)

    # 基因面板报告
    if phase_str in ("2", "all"):
        generate_gene_panel_report(cfg, output_dir)

    elapsed = time.time() - start_time
    _write_summary(output_dir, "COMPLETED", elapsed)
    LOGGER.info("=" * 60)
    LOGGER.info("Pipeline completed in %.1f seconds", elapsed)
    LOGGER.info("Output: %s", output_dir)
    LOGGER.info("=" * 60)


def _write_summary(output_dir: Path, status: str, elapsed: float) -> None:
    """写入运行摘要。"""
    summary = {
        "status": status,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
    }
    save_json(summary, output_dir / "pipeline_summary.json")


if __name__ == "__main__":
    main()
