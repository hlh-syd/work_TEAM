import os
import sys
import shutil
import subprocess
import argparse
import datetime
import json

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "DATA"))
RAW_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "..", "rawData"))
RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "results"))

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

PIPELINE_STEPS = [
    ("01_data_preprocessing", "01_data_preprocessing.py"),
    ("02_gene_features", "02_gene_features.py"),
    ("03_model_training", "03_model_training.py"),
    ("04_multi_timepoint", "04_multi_timepoint.py"),
]


def copy_data(raw_dir, data_dir):
    total = len(COPY_MANIFEST)
    copied = 0
    skipped = 0
    missing = 0

    for idx, (sub_dir, rel_src) in enumerate(COPY_MANIFEST, 1):
        src_path = os.path.join(raw_dir, rel_src)
        dst_dir = os.path.join(data_dir, sub_dir)
        dst_path = os.path.join(dst_dir, os.path.basename(rel_src))

        os.makedirs(dst_dir, exist_ok=True)

        if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
            print(f"[{idx}/{total}] 跳过: {os.path.basename(rel_src)} (已存在)")
            skipped += 1
            continue

        if not os.path.exists(src_path):
            print(f"[{idx}/{total}] 警告: 源文件不存在: {rel_src}")
            missing += 1
            continue

        shutil.copy2(src_path, dst_path)
        copied += 1
        print(f"[{idx}/{total}] 已复制: {os.path.basename(rel_src)}")

    print(f"\n复制完成: 已复制={copied}, 已跳过={skipped}, 缺失={missing}, 总计={total}")


def run_pipeline(timestamp):
    for step_name, script_name in PIPELINE_STEPS:
        script_path = os.path.join(SCRIPT_DIR, script_name)
        print(f"\n{'='*60}")
        print(f"执行步骤: {step_name} ({script_name})")
        print(f"{'='*60}")

        if not os.path.exists(script_path):
            print(f"错误: 脚本不存在: {script_path}")
            sys.exit(1)

        cmd = [sys.executable, script_path, "--timestamp", timestamp]
        result = subprocess.run(cmd, cwd=SCRIPT_DIR)

        if result.returncode != 0:
            print(f"\n错误: {step_name} 执行失败 (返回码: {result.returncode})")
            sys.exit(1)

        print(f"步骤 {step_name} 执行成功")

    print("\n所有pipeline步骤执行完成")


def generate_summary_figures(results_dir, timestamp):
    model_eval_dir = os.path.join(results_dir, timestamp, "03_model_training", "evaluation")
    multi_tp_dir = os.path.join(results_dir, timestamp, "04_multi_timepoint")
    summary_dir = os.path.join(results_dir, timestamp, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    auc_records = []

    if os.path.isdir(model_eval_dir):
        for fname in os.listdir(model_eval_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(model_eval_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    model_name = data.get("model", fname.replace(".json", ""))
                    auc_val = data.get("auc", data.get("c_index", None))
                    if auc_val is not None:
                        auc_records.append({"model": model_name, "timepoint": "primary", "auc": float(auc_val)})
                except Exception:
                    pass

    if os.path.isdir(multi_tp_dir):
        for fname in os.listdir(multi_tp_dir):
            if fname.endswith(".tsv"):
                fpath = os.path.join(multi_tp_dir, fname)
                try:
                    df = pd.read_csv(fpath, sep="\t")
                    for _, row in df.iterrows():
                        model_name = row.get("model", row.get("Model", "unknown"))
                        tp = row.get("timepoint", row.get("time", "unknown"))
                        auc_val = row.get("auc", row.get("AUC", row.get("c_index", None)))
                        if auc_val is not None:
                            auc_records.append({"model": str(model_name), "timepoint": str(tp), "auc": float(auc_val)})
                except Exception:
                    pass

    if not auc_records:
        print("未找到AUC结果数据，跳过汇总图表生成")
        return

    auc_df = pd.DataFrame(auc_records)

    pivot = auc_df.pivot_table(index="model", columns="timepoint", values="auc", aggfunc="mean")
    pivot = pivot.sort_values(by=pivot.columns[0], ascending=False) if len(pivot.columns) > 0 else pivot

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 3), 8))
    x = np.arange(len(pivot.index))
    width = 0.8 / max(len(pivot.columns), 1)

    for i, col in enumerate(pivot.columns):
        vals = pivot[col].values.astype(float)
        bars = ax.bar(x + i * width - (len(pivot.columns) - 1) * width / 2, vals, width, label=str(col))
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

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
    print(f"已保存AUC比较条形图: {fig_path}")

    summary_tsv = os.path.join(summary_dir, "model_performance_summary.tsv")
    auc_df_sorted = auc_df.sort_values(["timepoint", "auc"], ascending=[True, False])
    auc_df_sorted.to_csv(summary_tsv, sep="\t", index=False)
    print(f"已保存模型性能汇总表: {summary_tsv}")


def generate_readme(results_dir, timestamp):
    summary_dir = os.path.join(results_dir, timestamp, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    readme_path = os.path.join(summary_dir, "README.md")

    content = f"""# CRC 3年总生存期预测分析流水线

## 项目简介

本流水线针对结直肠癌（CRC）3年总生存期（OS）预测，整合TCGA-COAD/READ多组学数据与多个外部验证队列，
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
│   ├── 01_data_preprocessing.py   数据预处理与QC
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
│       └── summary/
└── DATA/ (由05_run_all.py自动从rawData复制)
```

## 运行方式

### 一键运行（含数据复制）

```bash
python 05_run_all.py
```

### 仅复制数据

```bash
python 05_run_all.py --copy-only
```

### 跳过数据复制直接运行

```bash
python 05_run_all.py --skip-copy
```

### 指定时间戳

```bash
python 05_run_all.py --timestamp {timestamp}
```

### 分步运行

```bash
python 01_data_preprocessing.py --timestamp {timestamp}
python 02_gene_features.py --timestamp {timestamp}
python 03_model_training.py --timestamp {timestamp}
python 04_multi_timepoint.py --timestamp {timestamp}
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
├── 04_multi_timepoint/       多时间点AUC/C-index结果
└── summary/
    ├── auc_comparison_barplot.png   跨时间点AUC比较条形图
    ├── model_performance_summary.tsv 模型性能汇总表
    └── README.md                    本文件
```

## 各脚本功能说明

| 脚本 | 功能 |
|---|---|
| `01_data_preprocessing.py` | 读取TCGA与外部队列原始数据，执行QC、标准化、端点定义 |
| `02_gene_features.py` | 差异表达分析、通路富集、多组学特征整合、因果特征筛选 |
| `03_model_training.py` | Cox-PH、RSF、GBM、DeepSurv、Stacking等多模型训练与内部验证 |
| `04_multi_timepoint.py` | 1年/2年/3年/5年多时间点AUC、DCA、Kaplan-Meier分析 |
| `05_run_all.py` | 数据复制 + 顺序执行01-04 + 生成汇总图表与文档 |
"""

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"已生成README: {readme_path}")


def main():
    parser = argparse.ArgumentParser(description="CRC生存分析流水线一键运行脚本")
    parser.add_argument("--copy-only", action="store_true", help="仅复制数据文件，不运行pipeline")
    parser.add_argument("--skip-copy", action="store_true", help="跳过数据复制，直接运行pipeline")
    parser.add_argument("--timestamp", type=str, default=None, help="指定运行时间戳 (格式: YYYYMMDD_HHMMSS)")
    args = parser.parse_args()

    timestamp = args.timestamp if args.timestamp else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"运行时间戳: {timestamp}")
    print(f"脚本目录: {SCRIPT_DIR}")
    print(f"数据目录: {DATA_DIR}")
    print(f"原始数据目录: {RAW_DIR}")
    print(f"结果目录: {RESULTS_DIR}")

    if args.copy_only:
        print("\n模式: 仅复制数据")
        copy_data(RAW_DIR, DATA_DIR)
        return

    if not args.skip_copy:
        print("\n步骤1: 复制数据文件")
        copy_data(RAW_DIR, DATA_DIR)
    else:
        print("\n已跳过数据复制步骤")

    print("\n步骤2: 运行分析pipeline")
    run_pipeline(timestamp)

    print("\n步骤3: 生成汇总图表")
    generate_summary_figures(RESULTS_DIR, timestamp)

    print("\n步骤4: 生成README文档")
    generate_readme(RESULTS_DIR, timestamp)

    print(f"\n全部流程完成! 结果保存在: {os.path.join(RESULTS_DIR, timestamp)}")


if __name__ == "__main__":
    main()
