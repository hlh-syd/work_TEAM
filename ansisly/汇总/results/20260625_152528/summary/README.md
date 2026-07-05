# CRC 3年总生存期预测分析流水线

## 项目简介

本流水线针对结直肠癌（CRC）3年总生存期（OS）预测，整合TCGA-COAD/READ多组学数据与多个外部验证队列，
执行从数据预处理、基因特征筛选、多模型训练到多时间点分析的完整流程。

运行时间戳: `20260625_152528`

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
│   └── 20260625_152528/
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
python 05_run_all.py --timestamp 20260625_152528
```

### 分步运行

```bash
python 01_data_preprocessing.py --timestamp 20260625_152528
python 02_gene_features.py --timestamp 20260625_152528
python 03_model_training.py --timestamp 20260625_152528
python 04_multi_timepoint.py --timestamp 20260625_152528
```

## 结果目录结构

```
results/20260625_152528/
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
