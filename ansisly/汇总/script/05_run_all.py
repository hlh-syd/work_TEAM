"""05_run_all.py — CRC 3 年总生存期预测流水线编排脚本

功能:
  - 数据文件复制（rawData → DATA）
  - 顺序/选择性执行 01~04 分析步骤
  - 执行审计日志（audit.json）
  - 汇总图表与 README 生成

CLI 示例:
  python 05_run_all.py                          # 完整运行
  python 05_run_all.py --only 01 03             # 仅运行步骤 01、03
  python 05_run_all.py --skip 04                # 跳过步骤 04
  python 05_run_all.py --resume 20250101_120000 # 从指定时间戳恢复
  python 05_run_all.py --config my_config.json  # 使用自定义配置
  python 05_run_all.py --copy-only              # 仅复制数据
  python 05_run_all.py --skip-copy              # 跳过数据复制
"""
import os
import sys
import shutil
import subprocess
import argparse
import datetime
import json
import time as _time

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shared_utils import (
    SCRIPT_DIR, DATA_DIR, RESULTS_DIR,
    ESSENTIAL_DIR,
    ensure_dir, setup_logger, load_pipeline_config,
)

RAW_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "..", "rawData"))

logger = setup_logger("05_run_all")

# ──────────────────────────────────────────────────────────────────────
# 数据复制清单
# ──────────────────────────────────────────────────────────────────────
COPY_MANIFEST = [
    ("tcga", "coadread_tcga_pan_can_atlas_2018/coadread_tcga_pan_can_atlas_2018/data_clinical_patient.txt"),
    ("tcga", "coadread_tcga_pan_can_atlas_2018/coadread_tcga_pan_can_atlas_2018/data_clinical_sample.txt"),
    ("tcga", "coadread_tcga_pan_can_atlas_2018/coadread_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt"),
    ("tcga", "coadread_tcga_pan_can_atlas_2018/coadread_tcga_pan_can_atlas_2018/data_methylation_hm450.txt"),
    ("tcga", "coadread_tcga_pan_can_atlas_2018/coadread_tcga_pan_can_atlas_2018/data_cna.txt"),
    ("tcga", "coadread_tcga_pan_can_atlas_2018/coadread_tcga_pan_can_atlas_2018/data_rppa.txt"),
    ("tcga", "coadread_tcga_pan_can_atlas_2018/coadread_tcga_pan_can_atlas_2018/data_mutations.txt"),
    ("external", "preprocessed/msk_os_clinical_endpoint_qc.tsv"),
    ("external", "preprocessed/geo_gse17538_os_clinical_endpoint_qc.tsv"),
    ("external", "preprocessed/geo_gse39582_os_clinical_endpoint_qc.tsv"),
    ("external", "preprocessed/cptac_os_clinical_endpoint_qc.tsv"),
    ("external", "preprocessed/htan_os_clinical_endpoint_qc.tsv"),
    ("external", "preprocessed/geo_gse103479_os_clinical_endpoint_qc.tsv"),
    ("external", "preprocessed/geo_gse103479_expression_matrix.tsv"),
    ("external", "preprocessed/msk_crc_2017_mutation_gene_level_matrix.tsv"),
    ("external", "preprocessed/cptac_coad_protein_gene_level_matrix.tsv"),
    ("external", "preprocessed/cptac_coad_rna_gene_level_matrix.tsv"),
    ("external", "preprocessed/htan_crc_pseudobulk_rna_matrix.tsv"),
    ("external", "preprocessed/htan_crc_microenvironment_relative_fraction_matrix.tsv"),
    ("external", "GEO_COAD/GSE17538/geneMatrix.txt"),
    ("external", "GEO_COAD/GSE17538/clinical.txt"),
    ("external", "GEO_COAD/GSE39582/geneMatrix.txt"),
    ("external", "GEO_COAD/GSE39582/clinical.txt"),
    ("msigdb", "MSigDB/h.all.v2024.1.Hs.symbols.gmt"),
    ("msigdb", "MSigDB/c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt"),
    ("msigdb", "MSigDB/c5.go.bp.v2024.1.Hs.symbols.gmt"),
    ("eqtl", "GTEx_Analysis_v8_eQTL_independent/GTEx_Analysis_v8_eQTL_independent/Colon_Sigmoid.v8.independent_eqtls.txt.gz"),
    ("eqtl", "GTEx_Analysis_v8_eQTL_independent/GTEx_Analysis_v8_eQTL_independent/Colon_Transverse.v8.independent_eqtls.txt.gz"),
    ("eqtl", "eQTLGen_blood_eQTL/2018-09-04-cis-eQTLsFDR0.05-Zscore4-probeLevel.txt.gz"),
    ("causal", "causal_screening/causal_priority_feature_table.tsv"),
    ("causal", "causal_screening/tcga_univariable_cox_feature_screening.tsv"),
    ("survival_models", "survival_models/tcga_train_internal_validation_split.tsv"),
    ("multiomics", "multiomics_pretraining/tcga_multiomics_patient_embedding.tsv"),
    ("preprocessed", "preprocessed/tcga_os_clinical_endpoint_qc.tsv"),
    ("preprocessed", "preprocessed/tcga_coadread_rna_log2_tpm_like_matrix.tsv"),
    ("preprocessed", "preprocessed/tcga_coadread_mutation_gene_level_matrix.tsv"),
    ("preprocessed", "preprocessed/tcga_coadread_cnv_gene_level_matrix.tsv"),
    ("preprocessed", "preprocessed/tcga_coadread_methylation_gene_level_matrix.tsv"),
    ("preprocessed", "preprocessed/tcga_coadread_rppa_gene_level_matrix.tsv"),
]

# ──────────────────────────────────────────────────────────────────────
# Pipeline 步骤定义
# ──────────────────────────────────────────────────────────────────────
PIPELINE_STEPS = [
    ("01_data_preprocessing", "01_data_preprocessing.py"),
    ("02_gene_features",      "02_gene_features.py"),
    ("03_model_training",     "03_model_training.py"),
    ("04_multi_timepoint",    "04_multi_timepoint.py"),
]

# 每步预期输出文件（数据契约）—— 用于 --resume 时校验
STEP_OUTPUTS = {
    "01_data_preprocessing": [
        "cohort_endpoint_summary.tsv",
        "survival_labels.pkl",
        "gene_expression_curated.tsv",
        "preprocessing_config.json",
        "tcga_os_clinical_endpoint_qc.tsv",
        "gene_names.pkl",
        "sample_ids.pkl",
    ],
    "02_gene_features": [
        "final_gene_list.pkl",
        "causal_priority_feature_table.tsv",
        "gene_feature_config.json",
    ],
    "03_model_training": [
        "evaluation/model_training_summary.json",
        "models/all_fitted_model_bundles.joblib",
    ],
    "04_multi_timepoint": [],  # 输出文件名含时间点，不做静态校验
}


# ──────────────────────────────────────────────────────────────────────
# 数据复制
# ──────────────────────────────────────────────────────────────────────
def copy_data(raw_dir: str, data_dir: str) -> None:
    """将 rawData 中的源文件复制到 DATA 目录。"""
    total = len(COPY_MANIFEST)
    copied = skipped = missing = 0

    for idx, (sub_dir, rel_src) in enumerate(COPY_MANIFEST, 1):
        src_path = os.path.join(raw_dir, rel_src)
        dst_dir  = os.path.join(data_dir, sub_dir)
        dst_path = os.path.join(dst_dir, os.path.basename(rel_src))

        os.makedirs(dst_dir, exist_ok=True)

        if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
            logger.debug("[%d/%d] 跳过: %s (已存在)", idx, total, os.path.basename(rel_src))
            skipped += 1
            continue

        if not os.path.exists(src_path):
            logger.warning("[%d/%d] 源文件不存在: %s", idx, total, rel_src)
            missing += 1
            continue

        shutil.copy2(src_path, dst_path)
        copied += 1
        logger.info("[%d/%d] 已复制: %s", idx, total, os.path.basename(rel_src))

    logger.info("复制完成: 已复制=%d, 已跳过=%d, 缺失=%d, 总计=%d", copied, skipped, missing, total)


# ──────────────────────────────────────────────────────────────────────
# 数据契约校验
# ──────────────────────────────────────────────────────────────────────
def validate_step_outputs(results_dir: str, timestamp: str, step_name: str) -> bool:
    """验证某步骤输出文件是否完整。返回 True 表示完整可跳过。

    特殊处理: 01_data_preprocessing 的输出存储在 ESSENTIAL_DIR（跨 timestamp 持久化目录），
    而非 results_dir/timestamp/ 下。
    """
    expected = STEP_OUTPUTS.get(step_name, [])
    if not expected:
        return False  # 无静态清单，不做校验

    # 01_data_preprocessing 输出存储在持久化的 ESSENTIAL_DIR，而非时间戳目录
    if step_name == "01_data_preprocessing":
        step_dir = ESSENTIAL_DIR
    else:
        step_dir = os.path.join(results_dir, timestamp, step_name)

    if not os.path.isdir(step_dir):
        return False

    for fname in expected:
        fpath = os.path.join(step_dir, fname)
        if not (os.path.exists(fpath) and os.path.getsize(fpath) > 0):
            logger.warning("契约校验失败: 缺少 %s/%s", step_name, fname)
            return False

    logger.info("契约校验通过: %s (%d 个输出文件)", step_name, len(expected))
    return True


# ──────────────────────────────────────────────────────────────────────
# 审计日志
# ──────────────────────────────────────────────────────────────────────
def _audit_path(results_dir: str, timestamp: str) -> str:
    return os.path.join(results_dir, timestamp, "audit.json")


def load_audit(results_dir: str, timestamp: str) -> dict:
    """加载已有审计日志，若不存在则返回空结构。"""
    path = _audit_path(results_dir, timestamp)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"timestamp": timestamp, "steps": {}, "started_at": None, "finished_at": None}


def save_audit(results_dir: str, timestamp: str, audit: dict) -> None:
    path = _audit_path(results_dir, timestamp)
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────────────
# Pipeline 执行
# ──────────────────────────────────────────────────────────────────────
def run_pipeline(
    timestamp: str,
    results_dir: str,
    only_steps: list | None = None,
    skip_steps: list | None = None,
    resume: bool = False,
) -> dict:
    """
    执行 pipeline 步骤，支持选择性执行、跳过和断点恢复。

    Parameters
    ----------
    timestamp    : 运行时间戳
    results_dir  : 结果根目录
    only_steps   : 仅执行这些步骤前缀（如 ["01", "03"]）
    skip_steps   : 跳过这些步骤前缀（如 ["04"]）
    resume       : 若为 True，已通过契约校验的步骤将被跳过

    Returns
    -------
    audit dict
    """
    only_steps  = only_steps  or []
    skip_steps  = skip_steps  or []

    audit = load_audit(results_dir, timestamp)
    if audit.get("started_at") is None:
        audit["started_at"] = datetime.datetime.now().isoformat(timespec="seconds")

    for step_name, script_name in PIPELINE_STEPS:
        # 前缀匹配：支持 "01" 匹配 "01_data_preprocessing"
        step_prefix = step_name.split("_")[0]

        if only_steps and not any(step_prefix.startswith(s) or step_name.startswith(s) for s in only_steps):
            logger.info("跳过步骤 %s（不在 --only 列表中）", step_name)
            audit["steps"][step_name] = {"status": "skipped", "reason": "--only filter"}
            continue

        if any(step_prefix.startswith(s) or step_name.startswith(s) for s in skip_steps):
            logger.info("跳过步骤 %s（在 --skip 列表中）", step_name)
            audit["steps"][step_name] = {"status": "skipped", "reason": "--skip filter"}
            continue

        if resume and validate_step_outputs(results_dir, timestamp, step_name):
            logger.info("跳过步骤 %s（--resume 且契约校验通过）", step_name)
            audit["steps"][step_name] = {"status": "skipped", "reason": "resume: outputs valid"}
            continue

        script_path = os.path.join(SCRIPT_DIR, script_name)
        if not os.path.exists(script_path):
            logger.error("脚本不存在: %s", script_path)
            audit["steps"][step_name] = {"status": "error", "reason": "script not found"}
            save_audit(results_dir, timestamp, audit)
            sys.exit(1)

        logger.info("=" * 60)
        logger.info("执行步骤: %s (%s)", step_name, script_name)
        logger.info("=" * 60)

        step_record = {"status": "running", "started_at": datetime.datetime.now().isoformat(timespec="seconds")}
        audit["steps"][step_name] = step_record
        save_audit(results_dir, timestamp, audit)

        t0 = _time.time()
        cmd = [sys.executable, script_path, "--timestamp", timestamp]
        result = subprocess.run(cmd, cwd=SCRIPT_DIR)
        elapsed = round(_time.time() - t0, 1)

        step_record["elapsed_seconds"] = elapsed
        step_record["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")

        if result.returncode != 0:
            step_record["status"] = "failed"
            step_record["returncode"] = result.returncode
            logger.error("步骤 %s 执行失败 (返回码: %d, 耗时: %.1fs)", step_name, result.returncode, elapsed)
            save_audit(results_dir, timestamp, audit)
            sys.exit(1)

        step_record["status"] = "success"
        logger.info("步骤 %s 执行成功 (耗时: %.1fs)", step_name, elapsed)
        save_audit(results_dir, timestamp, audit)

    audit["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    save_audit(results_dir, timestamp, audit)
    logger.info("所有 pipeline 步骤执行完成")
    return audit


# ──────────────────────────────────────────────────────────────────────
# 汇总图表
# ──────────────────────────────────────────────────────────────────────
def generate_summary_figures(results_dir: str, timestamp: str) -> None:
    model_eval_dir = os.path.join(results_dir, timestamp, "03_model_training", "evaluation")
    multi_tp_dir   = os.path.join(results_dir, timestamp, "04_multi_timepoint")
    summary_dir    = os.path.join(results_dir, timestamp, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    auc_records = []

    if os.path.isdir(model_eval_dir):
        for fname in os.listdir(model_eval_dir):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(model_eval_dir, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                model_name = data.get("model", fname.replace(".json", ""))
                auc_val = data.get("auc", data.get("c_index", None))
                if auc_val is not None:
                    auc_records.append({"model": model_name, "timepoint": "primary", "auc": float(auc_val)})
            except Exception:
                pass

    if os.path.isdir(multi_tp_dir):
        for fname in os.listdir(multi_tp_dir):
            if not fname.endswith(".tsv"):
                continue
            try:
                df = pd.read_csv(os.path.join(multi_tp_dir, fname), sep="\t")
                for _, row in df.iterrows():
                    model_name = row.get("model", row.get("Model", "unknown"))
                    tp         = row.get("timepoint", row.get("time", "unknown"))
                    auc_val    = row.get("auc", row.get("AUC", row.get("c_index", None)))
                    if auc_val is not None:
                        auc_records.append({"model": str(model_name), "timepoint": str(tp), "auc": float(auc_val)})
            except Exception:
                pass

    if not auc_records:
        logger.warning("未找到 AUC 结果数据，跳过汇总图表生成")
        return

    auc_df = pd.DataFrame(auc_records)
    pivot  = auc_df.pivot_table(index="model", columns="timepoint", values="auc", aggfunc="mean")
    if len(pivot.columns) > 0:
        pivot = pivot.sort_values(by=pivot.columns[0], ascending=False)

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 3), 8))
    x     = np.arange(len(pivot.index))
    width = 0.8 / max(len(pivot.columns), 1)

    for i, col in enumerate(pivot.columns):
        vals  = pivot[col].values.astype(float)
        bars  = ax.bar(x + i * width - (len(pivot.columns) - 1) * width / 2, vals, width, label=str(col))
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=45,
                )

    ax.set_ylabel("AUC / C-index")
    ax.set_title("Model Performance Comparison Across Timepoints")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend(title="Timepoint", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    fig_path = os.path.join(summary_dir, "auc_comparison_barplot.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("已保存 AUC 比较条形图: %s", fig_path)

    summary_tsv = os.path.join(summary_dir, "model_performance_summary.tsv")
    auc_df.sort_values(["timepoint", "auc"], ascending=[True, False]).to_csv(summary_tsv, sep="\t", index=False)
    logger.info("已保存模型性能汇总表: %s", summary_tsv)


# ──────────────────────────────────────────────────────────────────────
# README 生成
# ──────────────────────────────────────────────────────────────────────
def generate_readme(results_dir: str, timestamp: str) -> None:
    summary_dir = os.path.join(results_dir, timestamp, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    readme_path = os.path.join(summary_dir, "README.md")

    content = f"""# CRC 3 年总生存期预测分析流水线

## 项目简介

本流水线针对结直肠癌（CRC）3 年总生存期（OS）预测，整合 TCGA-COAD/READ 多组学数据与多个外部验证队列，
执行从数据预处理、基因特征筛选、多模型训练到多时间点分析的完整流程。

运行时间戳: `{timestamp}`

## 环境要求

- Python 3.12+
- 操作系统: Windows / Linux

## 安装依赖

```bash
pip install -r requirements.txt
```

## 目录结构

```
汇总/
├── script/
│   ├── shared_utils.py            公共工具模块
│   ├── 01_data_preprocessing.py   数据预处理与 QC
│   ├── 02_gene_features.py        基因特征筛选与多组学整合
│   ├── 03_model_training.py       多模型训练与评估
│   ├── 04_multi_timepoint.py      多时间点生存分析
│   └── 05_run_all.py              一键运行编排脚本
├── results/
│   └── {timestamp}/
│       ├── 01_data_preprocessing/
│       ├── 02_gene_features/
│       ├── 03_model_training/
│       ├── 04_multi_timepoint/
│       ├── summary/
│       └── audit.json
└── DATA/ (由 05_run_all.py 自动从 rawData 复制)
```

## 运行方式

### 一键运行（含数据复制）

```bash
python 05_run_all.py
```

### 选择性执行

```bash
python 05_run_all.py --only 01 03    # 仅执行步骤 01、03
python 05_run_all.py --skip 04       # 跳过步骤 04
```

### 断点恢复

```bash
python 05_run_all.py --resume {timestamp}
```

### 使用自定义配置

```bash
python 05_run_all.py --config pipeline_config.json
```

### 仅复制数据

```bash
python 05_run_all.py --copy-only
```

### 跳过数据复制

```bash
python 05_run_all.py --skip-copy
```

## 结果目录结构

```
results/{timestamp}/
├── 01_data_preprocessing/    预处理后的临床与组学矩阵
├── 02_gene_features/         特征筛选结果与多组学特征集
├── 03_model_training/
│   ├── models/               训练好的模型文件
│   ├── evaluation/           模型评估指标 (JSON)
│   └── figures/              训练过程可视化
├── 04_multi_timepoint/       多时间点 AUC/C-index 结果
├── summary/
│   ├── auc_comparison_barplot.png
│   ├── model_performance_summary.tsv
│   └── README.md
└── audit.json                执行审计日志
```

## 各脚本功能说明

| 脚本 | 功能 |
|---|---|
| `shared_utils.py` | 公共工具：路径常量、I/O、ID 转换、生存分析核心、端点工程 |
| `01_data_preprocessing.py` | 读取 TCGA 与外部队列原始数据，执行 QC、标准化、端点定义 |
| `02_gene_features.py` | 差异表达分析、通路富集、多组学特征整合、因果特征筛选 |
| `03_model_training.py` | Cox-PH、RSF、GBM、DeepSurv、Stacking 等多模型训练与内部验证 |
| `04_multi_timepoint.py` | 1年/2年/3年/无限制多时间点 AUC、DCA、Kaplan-Meier 分析 |
| `05_run_all.py` | 数据复制 + 选择性执行 01-04 + 审计日志 + 汇总图表与文档 |
"""

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("已生成 README: %s", readme_path)


# ──────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="CRC 生存分析流水线一键运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python 05_run_all.py                          完整运行
  python 05_run_all.py --only 01 03             仅运行步骤 01、03
  python 05_run_all.py --skip 04                跳过步骤 04
  python 05_run_all.py --resume 20250101_120000 从指定时间戳恢复
  python 05_run_all.py --config config.json     使用自定义配置
""",
    )
    parser.add_argument("--copy-only", action="store_true",
                        help="仅复制数据文件，不运行 pipeline")
    parser.add_argument("--skip-copy", action="store_true",
                        help="跳过数据复制，直接运行 pipeline")
    parser.add_argument("--timestamp", type=str, default=None,
                        help="指定运行时间戳 (格式: YYYYMMDD_HHMMSS)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="仅执行指定步骤前缀 (如 01 03)")
    parser.add_argument("--skip", nargs="+", default=None,
                        help="跳过指定步骤前缀 (如 04)")
    parser.add_argument("--resume", action="store_true",
                        help="从 --timestamp 指定的时间戳恢复执行（已通过校验的步骤跳过）")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON 配置文件路径，覆盖默认参数")
    args = parser.parse_args()

    # 加载配置（目前仅做验证，后续 Task 6 会全面使用）
    if args.config:
        cfg = load_pipeline_config(args.config)
        logger.info("已加载配置: %s (%d 个参数)", args.config, len(cfg))

    # 时间戳
    if args.resume and not args.timestamp:
        parser.error("--resume 需要同时指定 --timestamp")

    timestamp = args.timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("运行时间戳  : %s", timestamp)
    logger.info("脚本目录    : %s", SCRIPT_DIR)
    logger.info("数据目录    : %s", DATA_DIR)
    logger.info("原始数据目录: %s", RAW_DIR)
    logger.info("结果目录    : %s", RESULTS_DIR)

    # 仅复制数据
    if args.copy_only:
        logger.info("模式: 仅复制数据")
        copy_data(RAW_DIR, DATA_DIR)
        return

    # 数据复制
    if not args.skip_copy:
        logger.info("步骤 1/4: 复制数据文件")
        copy_data(RAW_DIR, DATA_DIR)
    else:
        logger.info("已跳过数据复制步骤")

    # 运行 pipeline
    logger.info("步骤 2/4: 运行分析 pipeline")
    run_pipeline(
        timestamp=timestamp,
        results_dir=RESULTS_DIR,
        only_steps=args.only,
        skip_steps=args.skip,
        resume=args.resume,
    )

    # 汇总图表
    logger.info("步骤 3/4: 生成汇总图表")
    generate_summary_figures(RESULTS_DIR, timestamp)

    # README
    logger.info("步骤 4/4: 生成 README 文档")
    generate_readme(RESULTS_DIR, timestamp)

    logger.info("全部流程完成! 结果保存在: %s", os.path.join(RESULTS_DIR, timestamp))


if __name__ == "__main__":
    main()
