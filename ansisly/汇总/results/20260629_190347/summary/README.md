# CRC 3 年总生存期预测分析流水线

## 项目简介

本流水线针对结直肠癌（CRC）3 年总生存期（OS）预测，整合 TCGA-COAD/READ 多组学数据与多个外部验证队列，
执行从数据预处理、基因特征筛选、多模型训练到多时间点分析的完整流程。

运行时间戳: `20260629_190347`

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
│   └── 20260629_190347/
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
python 05_run_all.py --resume 20260629_190347
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
results/20260629_190347/
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
