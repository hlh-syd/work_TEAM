# CRC 3年总生存期风险预测 — 汇总分析流水线

## 项目简介

本模块为结直肠癌（CRC）3年总生存期（OS）预测的**汇总分析流水线**，整合 TCGA-COAD/READ 多组学数据与多个外部验证队列，执行从数据预处理、基因特征筛选、多模型训练到多时间点评估的完整流程。

## 环境要求

- Python 3.12+
- 依赖: `pip install -r requirements.txt`（项目根目录）

核心依赖:
| 包 | 用途 |
|---|---|
| `scikit-learn` | 机器学习模型与评估 |
| `lifelines` | Kaplan-Meier、Cox-PH 生存分析 |
| `scikit-survival` | RSF、Coxnet、IPCW 评估指标 |
| `statsmodels` | 多重检验校正（BH-FDR） |
| `scipy` / `numpy` / `pandas` | 数值计算 |
| `matplotlib` | 可视化 |
| `torch`（可选） | DeepSurv 深度学习模型 |

## 目录结构

```
汇总/
├── script/
│   ├── 01_data_preprocessing.py   # 数据预处理与QC
│   ├── 02_gene_features.py        # 基因特征筛选与因果优先级
│   ├── 03_model_training.py       # 多模型训练与评估
│   ├── 04_multi_timepoint.py      # 多时间点生存分析
│   └── 05_run_all.py              # 一键运行编排脚本
├── results/                       # 分析结果（按时间戳组织）
└── DATA/                          # 由 05_run_all.py 自动从 rawData 复制
```

## 运行方式

### 一键运行（推荐）

```bash
# 完整流程：数据复制 + 4步分析 + 汇总图表
python 05_run_all.py

# 仅复制数据
python 05_run_all.py --copy-only

# 跳过数据复制
python 05_run_all.py --skip-copy

# 指定时间戳
python 05_run_all.py --timestamp 20260624_120000
```

### 分步运行

```bash
python 01_data_preprocessing.py --timestamp <时间戳>
python 02_gene_features.py      --timestamp <时间戳>
python 03_model_training.py     --timestamp <时间戳>
python 04_multi_timepoint.py    --timestamp <时间戳>
```

## 各脚本功能说明

### 01 — 数据预处理与QC

| 步骤 | 内容 |
|---|---|
| 临床数据加载 | TCGA 患者临床数据，AJCC 分期数值化编码，分类变量 LabelEncoder |
| 生存终点定义 | OS 事件/时间，36个月固定终点（τ=36m），IPCW 权重，伪观测值（pseudo-observations） |
| 多组学加载 | mRNA 表达（top-300 方差基因）、突变（频率 2%–80% 过滤）、CNV、甲基化、RPPA |
| 样本对齐 | 多组学取共有患者 ID，StandardScaler 标准化 |
| 输出 | 临床终点 TSV、各组学矩阵、生存标签 pickle、队列摘要统计 |

### 02 — 基因特征筛选与因果优先级

| 步骤 | 方法 |
|---|---|
| 方差预筛 | 训练集 top-2000 方差基因（nonmissing ≥ 70%） |
| 单变量 Cox | 逐基因 Cox-PH + BH-FDR 校正，PH 假设检验 |
| 多变量 Cox | 调整年龄/性别/分期/分型等混杂，惩罚 Cox 回归 |
| 因果 RMST 双重稳健估计 | 倾向得分 + 随机森林结局模型，计算 ATE-RMST |
| 因果优先级评分 | 综合 ATE 排名 + P 值排名 + 剂量反应排名 |
| 通路富集 | MSigDB Hallmark / KEGG / GO-BP 超几何检验 |
| CPTAC 蛋白验证 | RNA-蛋白 Spearman 一致性 |
| 证据矩阵 | 汇总 Cox HR、因果 ATE、蛋白验证等多维证据 |

### 03 — 多模型训练与评估

**模型池:**

| 模型 | 类型 |
|---|---|
| Cox-PH | 比例风险回归（lifelines） |
| RSF | 随机生存森林（scikit-survival） |
| Coxnet | Lasso/Elastic-Net Cox（scikit-survival） |
| IPCW-Logistic | IPCW 加权逻辑回归（sklearn） |
| DeepSurv | 深度生存模型（PyTorch，可选） |
| Stacking | 多模型集成 |

**特征集配置:**
- `clinical_only` — 仅临床特征
- `clinical_expression_topvar` — 临床 + RNA top 方差基因
- `clinical_expression_causal` — 临床 + 因果优先级基因
- `multiomics` — 多组学整合特征

**GAN 增强:**
- 条件 GAN 生成合成样本，增强小样本事件数
- 自动搜索最优增强比例（0.25x / 0.5x / 1.0x / 2.0x）
- 多种采样策略: balanced_event / risk_stratified / event_only / overall

**评估指标:**
- Harrell's C-index、Uno's C-index（IPCW）
- 时间依赖 AUC（cumulative/dynamic）
- IPCW Brier Score、Integrated Brier Score

### 04 — 多时间点生存分析

| 时间点 | τ（月） | 评估时间 |
|---|---|---|
| 1Year | 12 | 6, 9, 12 月 |
| 2Years | 24 | 12, 18, 24 月 |
| 3Years | 36 | 12, 24, 36 月 |
| Unlimited | 无限制 | 12, 24, 36 月 |

每个时间点独立训练 IPCW-Logistic、Cox-PH、Coxnet、RSF 四模型，自动选择最优模型，输出:
- K-M 风险分层曲线（PNG 300 DPI + TIFF 600 DPI）
- 时间依赖 ROC 曲线
- 校准曲线
- 风险分布图、随访时间 vs 风险散点图

### 05 — 一键编排

1. **数据复制** — 从 `rawData/` 按 manifest 复制到 `DATA/`（54 个文件）
2. **顺序执行** 01 → 02 → 03 → 04
3. **汇总图表** — 跨时间点 AUC 比较条形图、模型性能汇总表
4. **文档生成** — 自动生成本次运行 README

## 结果目录结构

```
results/<timestamp>/
├── 01_preprocessing/
│   ├── tcga_os_clinical_endpoint_qc.tsv
│   ├── gene_expression_curated.tsv
│   ├── mutation/cnv/methylation/rppa_curated.tsv
│   ├── survival_labels.pkl
│   ├── cohort_endpoint_summary.tsv
│   └── preprocessing_config.json
├── 02_gene_features/
│   ├── tcga_univariable_cox_feature_screening.tsv
│   ├── tcga_multivariable_cox_features.tsv
│   ├── causal_priority_feature_table.tsv
│   ├── pathway_enrichment_results.tsv
│   ├── core_biomarker_evidence_matrix.tsv
│   ├── evidence_heatmap.png
│   └── gene_feature_config.json
├── 03_model_training/
│   ├── models/              # 训练好的模型（joblib）
│   ├── evaluation/          # 模型评估 JSON
│   └── figures/             # 训练可视化
├── 04_multi_timepoint/
│   ├── 1Year/  2Year/  3Year/  Unlimited/
│   │   ├── model_comparison.tsv
│   │   ├── risk_scores.tsv
│   │   └── *.png / *.tiff
│   └── timepoint_comparison.tsv
└── summary/
    ├── auc_comparison_barplot.png
    ├── model_performance_summary.tsv
    └── README.md
```

## 数据来源

| 数据类别 | 来源 | 说明 |
|---|---|---|
| TCGA 多组学 | cBioPortal Pan-Cancer Atlas | RNA-seq、突变、CNV、甲基化、RPPA、临床 |
| 外部验证 | GEO / CPTAC / HTAN / MSK | GSE17538、GSE39582、GSE103479、CPTAC-COAD、HTAN-CRC、MSK-CRC |
| 通路基因集 | MSigDB v2024.1 | Hallmark、KEGG、GO-BP |
| eQTL 参考 | GTEx v8 / eQTLGen | 结肠/血液顺式 eQTL |

## 关键参数

| 参数 | 值 | 说明 |
|---|---|---|
| `FIXED_TAU_MONTHS` | 36.0 | 主要终点: 3年 OS |
| `TOP_K_GENES` | 300 | 初始方差筛选基因数 |
| `MUTATION_FREQ` | 0.02–0.80 | 突变频率过滤范围 |
| `FDR_THRESHOLD` | 0.20 | 特征筛选 FDR 阈值 |
| `GAN_AUG_RATIO` | 0.25–2.0x | GAN 增强比例搜索空间 |
| `RANDOM_SEED` | 42 | 全局随机种子 |
