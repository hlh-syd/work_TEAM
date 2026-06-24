# CRC因果多组学预后风险分层 — Code Wiki

> 项目全称：Colorectal Cancer Causal Multi-Omics Prognostic Risk Stratification  
> 文档版本：v1.0 | 生成日期：2026-05-26

---

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [目录结构](#3-目录结构)
4. [核心配置](#4-核心配置)
5. [共享基础设施模块 repro_io.py](#5-共享基础设施模块-repro_iopy)
6. [流水线各步骤详解](#6-流水线各步骤详解)
7. [数据流与依赖关系](#7-数据流与依赖关系)
8. [Python包依赖](#8-python包依赖)
9. [项目运行方式](#9-项目运行方式)
10. [关键设计决策](#10-关键设计决策)
11. [数据集来源与许可](#11-数据集来源与许可)

---

## 1. 项目概述

本项目旨在构建一个**因果优先的多组学预后风险分层系统**，用于结直肠癌（CRC）患者的生存预测。核心假设为：

> **因果优先筛选的特征（经孟德尔随机化/SMR证据支持）比单纯预后关联特征具有更强的跨队列泛化能力。**

项目采用 7 步顺序流水线架构，覆盖从数据预处理到模型验证再到生物学解释的完整研究闭环，并在每一步实施**leakage-safe（防泄露）**策略，确保训练集信息不污染测试集。

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    project_config.yaml (全局配置)                    │
│                    repro_io.py (共享基础设施)                        │
└─────────────────────────────────────────────────────────────────────┘
         │                │                │                │
         ▼                ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ 01 多组学    │ │ 02 因果优先  │ │ 03 多组学    │ │ 04 生存预测  │
│ 数据预处理   │→│ 特征筛选     │→│ 预训练嵌入   │→│ 模型训练     │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
                                                      │
                                                      ▼
                              ┌──────────────┐ ┌──────────────┐
                              │ 05 验证与    │ │ 06 可解释性  │
                              │ 可视化       │→│ 与生物学     │
                              └──────────────┘ └──────────────┘
                                                      │
                                                      ▼
                              ┌──────────────┐
                              │ 07 可重复性  │
                              │ IO与日志     │
                              └──────────────┘
```

**架构特点：**

- **顺序依赖**：每一步的输出是下一步的输入，不可跳步执行
- **防泄露设计**：特征选择、模型拟合、阈值确定均严格限定在训练集
- **配置驱动**：所有路径、参数、阈值由 `project_config.yaml` 统一管理
- **可重复性框架**：`RunContext` 自动记录文件指纹、包版本、语义化文件名

---

## 3. 目录结构

```
d:\work-课题\
│
├── ansisly\因果-多组学-预后\          # 核心分析代码
│   ├── project_config.yaml            # 全局配置文件
│   ├── repro_io.py                    # 共享基础设施模块 (862行)
│   ├── 01_multiomics_preprocessing.py # 步骤1: 数据预处理 (537行)
│   ├── 02_causal_feature_screening.py # 步骤2: 因果特征筛选 (428行)
│   ├── 03_multiomics_pretraining.py   # 步骤3: 多组学嵌入 (456行)
│   ├── 04_survival_prediction_model.py# 步骤4: 生存模型 (986行)
│   ├── 05_validation_visualization.py # 步骤5: 验证可视化 (796行)
│   ├── 06_interpretability_biology.py # 步骤6: 可解释性 (443行)
│   ├── 07_reproducibility_io_logging.py# 步骤7: 可重复性 (10行)
│   └── plan\                          # 研究设计文档
│       ├── plan.md                    # 总体研究计划
│       └── ...                        # 其他设计文档
│
├── rawData\                           # 原始数据 (只读,不可写入)
│   ├── GEO_COAD\                      # GEO表达数据
│   ├── GTEx_v8_eQTL\                  # GTEx eQTL数据
│   ├── GoDMC_mQTL\                    # mQTL数据
│   ├── MSigDB\                        # 基因集数据库
│   ├── coadread_tcga_pan_can_atlas_2018\ # TCGA泛癌图谱
│   ├── coad_cptac_2019\               # CPTAC蛋白质组
│   ├── coad_cptac_gdc\                # CPTAC GDC数据
│   ├── crc_hta8_htan_2024\            # HTAN微环境数据
│   ├── crc_msk_2017\                  # MSK基因组数据
│   └── eQTLGen_blood_eQTL\            # eQTLGen血液eQTL
│
├── data\                              # 中间处理数据
│   ├── preprocessed\                  # 步骤1输出
│   ├── causal_screening\              # 步骤2输出
│   ├── multiomics_pretraining\        # 步骤3输出
│   ├── survival_models\               # 步骤4输出
│   ├── validation_results\            # 步骤5输出
│   └── interpretability_biology\      # 步骤6输出
│
├── result\                            # 带时间戳的运行结果目录
│   └── 20YYMMDD_HHMMSS_stepXX\       # 每次运行的完整输出
│
└── gwas\                              # GWAS目录数据与补充图表
```

---

## 4. 核心配置

### project_config.yaml 关键参数

| 参数类别 | 参数名 | 默认值 | 说明 |
|---------|--------|--------|------|
| **路径** | `project_root` | `d:\work-课题` | 项目根目录 |
| | `script_root` | `ansisly\因果-多组学-预后` | 脚本目录 |
| | `raw_root` | `rawData` | 原始数据目录(只读) |
| | `processed_root` | `data` | 中间数据目录 |
| | `result_root` | `result` | 结果输出目录 |
| **研究** | `endpoint` | `OS` | 终点类型(总生存) |
| | `random_seed` | `20260525` | 全局随机种子 |
| | `max_features_per_omics` | `300` | 每组学最大特征数 |
| **分割策略** | `split_policy.method` | `stratified_shuffle_split` | 分层随机分割 |
| | `split_policy.test_size` | `0.30` | 测试集比例 |
| | `split_policy.strata_columns` | `event, ajcc_stage, site` | 分层变量 |
| **筛选阈值** | `filters.min_nonmissing_ratio` | `0.7` | 最小非缺失比例 |
| | `filters.univariable_fdr_threshold` | `0.20` | 单变量FDR阈值 |
| | `filters.min_events_per_feature` | `5` | 每特征最小事件数 |
| **软件后端** | `software_backends` | `lifelines, sksurv, statsmodels` | 统计软件包 |

---

## 5. 共享基础设施模块 repro_io.py

> 文件路径：`ansisly\因果-多组学-预后\repro_io.py` | 862行  
> 被所有 01-06 脚本导入，是整个流水线的基石

### 5.1 核心类

#### `RunContext`

流水线每次运行的核心管理对象，负责：

- 创建带时间戳的运行目录 (`result/YYYYMMDD_HHMMSS_stepXX/`)
- 初始化日志系统 (INFO级别，同时输出到终端和文件)
- 注册输出文件并计算SHA256指纹
- 管理警告列表 (`warnings`)
- 提供结构化写入方法：`write_table()`, `write_json()`, `write_text()`
- `finalize()` 生成元数据 (`run_metadata.json`) 和清单 (`pipeline_manifest.tsv`)

```python
ctx = RunContext(step_name="01_multiomics_preprocessing", config=config)
ctx.write_table(df, "curated_clinical_tcga.tsv")
ctx.finalize()
```

### 5.2 关键函数

| 函数 | 功能 | 说明 |
|------|------|------|
| `default_config()` | 返回完整默认配置字典 | 包含所有5个队列的路径定义 |
| `load_config(yaml_path)` | 加载YAML并深度更新默认配置 | 优先级：YAML覆盖 > 默认值 |
| `assert_safe_paths(config)` | 验证路径安全性 | 禁止写入rawData目录 |
| `assert_safe_write_path(path, raw_root)` | 单路径写入安全检查 | 抛出PathSafetyError |
| `validate_semantic_filename(name)` | 语义化文件名验证 | 禁止figure1/plot/tmp等非描述性名称 |
| `set_global_seed(seed)` | 设置全局随机种子 | 覆盖random/numpy/torch，启用CUDA确定性 |
| `file_fingerprint(filepath)` | 计算文件SHA256哈希 | 用于可重复性追踪 |
| `package_versions()` | 记录所有依赖包版本 | 返回 `{包名: 版本号}` 字典 |
| `initialize_run(argv)` | 运行初始化编排 | 加载配置→路径安全→种子→创建RunContext |
| `add_common_args(parser)` | 添加通用CLI参数 | --config, --endpoint, --seed, --dry-run, --max-features-per-omics |
| `read_cbio_table(path)` | 读取cBioPortal TSV表格 | 处理[Not Available]标记 |
| `read_table_auto(path)` | 自动检测格式读取表格 | 支持.tsv/.csv/.txt |
| `patient_id_from_sample(sample_id)` | TCGA条码→12字符患者ID | 也支持HTAN ID解析 |
| `parse_survival_status(status_str)` | 状态字符串→布尔事件映射 | 支持Dead/Living/Deceased/Alive等 |
| `numeric_series(series)` | 强制数值转换 | 处理[Not Available]为NaN |
| `bh_fdr(pvalues)` | Benjamini-Hochberg FDR控制 | 通过statsmodels实现，NaN安全 |
| `stream_gene_matrix(path, ...)` | 加载cBioPortal基因×样本矩阵 | 转换为患者×基因DataFrame |
| `rank_inverse_normal(series)` | Blom分数逆正态变换 | 用于跨平台数据调和 |
| `stratified_split_keys(df, ...)` | 多列分层分割 | 含稀有层合并逻辑 |
| `safe_concordance_index_ipcw(...)` | Uno's IPCW C-index | 通过sksurv实现 |
| `bootstrap_metric(metric_func, ...)` | 通用Bootstrap置信区间 | 95%百分位CI |
| `is_likely_pseudogene(gene)` | 假基因过滤 | 匹配LOC/LINC/MIR/SNORD/RNU/P1-P8 |
| `matrix_sample_columns(df)` | 分离元数据列与样本列 | 基于TCGA条码模式识别 |

### 5.3 重要常量

| 常量 | 类型 | 说明 |
|------|------|------|
| `DATASET_PROVENANCE` | `dict` | 所有队列来源、许可、引用的文档化记录 |
| `CLINICAL_VARIABLE_WHITELIST` | `list` | 允许的临床协变量名称白名单 |

### 5.4 CLI入口

`command_line_main()` 函数提供步骤07的命令行入口，用于独立运行可重复性检查和日志汇总。

---

## 6. 流水线各步骤详解

### 6.1 步骤01 — 多组学数据预处理

> 文件：`01_multiomics_preprocessing.py` | 537行  
> 输入：rawData下5个队列的原始数据  
> 输出：`data/preprocessed/` 下标准化后的临床与组学矩阵

#### 主要函数

| 函数 | 功能 |
|------|------|
| `normalize_clinical(raw_df, cohort_key, config)` | 标准化各队列临床数据：统一列名、AJCC分期映射、生存状态解析 |
| `outcome_summary(clinical_df, endpoint)` | 生成终点统计摘要（事件数/中位随访/缺失率） |
| `patient_set_from_matrix(matrix_df)` | 从组学矩阵提取有效患者集合 |
| `plot_upset_style(patient_sets_dict, ctx)` | 绘制UpSet图展示多组学交集 |
| `plot_missingness_heatmap(modality_dict, ctx)` | 绘制缺失率热图 |
| `plot_outcome_distribution(clinical_df, endpoint, ctx)` | 绘制终点分布图 |
| `write_curated_modality_matrix(df, modality, cohort, ctx)` | 写入标准化组学矩阵（含训练集方差排序） |
| `write_curated_patient_gene_matrix(df, cohort, ctx)` | 写入患者×基因表达矩阵 |
| `write_mutation_gene_matrix(df, cohort, ctx)` | 写入突变基因矩阵 |

#### 处理队列

| 队列 | 数据类型 | 来源 |
|------|----------|------|
| TCGA | 临床+表达+突变+拷贝数+甲基化 | coadread_tcga_pan_can_atlas_2018 |
| MSK | 临床+基因组突变 | crc_msk_2017 |
| CPTAC | 临床+蛋白质组+RNA | coad_cptac_2019 |
| HTAN | 临床+微环境细胞分数 | crc_hta8_htan_2024 |
| GEO GSE103479 | 表达 | GEO_COAD |

#### 关键设计

- **训练集方差排序**：特征按训练集方差降序排列，避免测试集信息泄露
- **假基因过滤**：`is_likely_pseudogene()` 移除LOC/LINC/MIR等假基因
- **缺失率阈值**：低于 `min_nonmissing_ratio=0.7` 的特征被剔除

---

### 6.2 步骤02 — 因果优先特征筛选

> 文件：`02_causal_feature_screening.py` | 428行  
> 输入：步骤01的标准化矩阵 + GTEx eQTL数据  
> 输出：`data/causal_screening/` 下筛选结果与因果优先特征表

#### 主要函数

| 函数 | 功能 |
|------|------|
| `load_tcga_endpoint(config)` | 加载TCGA终点数据（OS时间+事件） |
| `attach_clinical_confounders(expr_df, clinical_df)` | 附加临床混杂因子（AJCC分期/年龄/性别/部位） |
| `train_variance_screen(df, train_keys, max_features)` | 训练集方差预筛选 |
| `univariable_cox(df, time, event, train_keys)` | 单变量Cox回归 + PH假设检验（Grambsch-Therneau） |
| `multivariable_cox(df, time, event, train_keys, confounders)` | 多变量Cox回归（强制纳入临床混杂因子） |
| `cis_eqtl_evidence(gene, gtex_eqtl_df)` | GTEx结肠eQTL工具变量查找 |
| `plot_volcano(results_df, ctx)` | 绘制火山图 |

#### 筛选流程

```
全部特征 → 训练集方差预筛选(max_features_per_omics)
         → 单变量Cox回归(FDR < 0.20)
         → 多变量Cox回归(调整临床混杂因子)
         → cis-eQTL工具变量可用性评估
         → causal_priority_feature_table.tsv
```

#### 关键设计

- **PH假设检验**：对每个单变量Cox模型执行Grambsch-Therneau检验，标记违反比例风险假设的特征
- **因果优先分级**：特征按因果证据强度分级（有cis-eQTL > 仅预后关联）
- **MR/SMR限制文档化**：明确记录当前无法执行完整MR/SMR分析（缺少CRC OS GWAS汇总统计），并标注为未来工作

---

### 6.3 步骤03 — 多组学预训练嵌入

> 文件：`03_multiomics_pretraining.py` | 456行  
> 输入：步骤01的标准化矩阵 + 步骤02的因果优先特征表  
> 输出：`data/multiomics_pretraining/` 下患者嵌入、模态掩码、模型序列化

#### 主要函数

| 函数 | 功能 |
|------|------|
| `get_or_create_split(clinical_df, config, ctx)` | 获取或创建训练/测试分割（分层随机） |
| `select_features_train_only(df, train_keys, causal_genes, max_features)` | 训练集专用特征选择（因果优先基因注入RNA通道） |
| `fit_modality_encoder(df, train_keys, n_components, method)` | 训练集拟合PCA/TruncatedSVD编码器（含Imputer+Scaler） |
| `stability_via_subspace_overlap(train_df, seeds, n_components)` | 多种子子空间重叠稳定性评估（cosine相似度） |

#### 嵌入策略

```
RNA表达 → 因果优先基因注入 + PCA嵌入
突变    → TruncatedSVD嵌入 (稀疏矩阵)
拷贝数  → PCA嵌入
甲基化  → PCA嵌入
蛋白质  → PCA嵌入
→ 模态可用性掩码 + 零填充缺失模态 → 拼接为统一嵌入向量
```

#### 关键设计

- **因果优先注入**：步骤02筛选的因果优先基因被强制注入RNA通道，不参与方差竞争
- **模态可用性掩码**：为每个患者生成二进制掩码向量，标记哪些组学模态可用
- **训练集专用拟合**：PCA/SVD仅在训练集上拟合，测试集通过transform投影
- **稳定性评估**：多种子子空间重叠分析验证嵌入的鲁棒性

---

### 6.4 步骤04 — 生存预测模型训练

> 文件：`04_survival_prediction_model.py` | 986行  
> 输入：步骤03的嵌入 + 步骤01的临床数据  
> 输出：`data/survival_models/` 下风险分数、模型指标、模型清单、模型卡片

#### 主要函数

| 函数 | 功能 |
|------|------|
| `make_stratified_split(clinical_df, config)` | 创建分层训练/测试分割 |
| `fit_clinical_transformer(clinical_df, train_keys, test_keys)` | 临床变量标准化转换器 |
| `fit_lifelines_cox(df, time, event, train_keys, covariates)` | lifelines CoxPHFitter拟合 |
| `cv_coxnet_select_alpha(X, y, train_keys)` | K折交叉验证选择Coxnet最优alpha（lambda.min/lambda.1se规则） |
| `fit_coxnet_with_cv(X_train, y_train, alphas)` | Coxnet拟合（含rank逆正态调和） |
| `fit_rsf(X_train, y_train)` | Random Survival Forest拟合 |
| `metric_uno_and_td_auc(model, X_test, y_test, y_train)` | Uno's C-index + 时间依赖AUC |
| `select_primary_model(metrics_dict)` | 多准则加权评分选择主模型 |
| `train_msk_genomic_model(msk_df, config)` | MSK外部基因组Cox模型训练 |
| `deepsurv_exploration_status(config)` | DeepSurv可行性评估（事件数≥120门槛） |
| `write_model_card(metrics, model_info, ctx)` | 生成REMARK/TRIPOD+AI模型卡片 |

#### 训练模型清单

| 模型ID | 类型 | 输入特征 | 说明 |
|--------|------|----------|------|
| `clinical_ajcc_cox` | Cox PH | AJCC分期 | 基准临床模型 |
| `clinical_tnm_cox_sensitivity` | Cox PH | T/N/M分期 | TNM敏感性分析 |
| `clinical_minimal_cox` | Cox PH | 最小临床变量 | 简化临床模型 |
| `clinical_random_survival_forest` | RSF | 临床变量 | 非线性临床模型 |
| `expression_coxnet` | Coxnet | 表达嵌入 | 弹性网惩罚Cox |
| `clinical_plus_multiomics_embedding_cox` | Cox PH | 临床+多组学嵌入 | 主模型候选 |
| `msk_external_genomic_cox` | Cox PH | MSK基因组 | 外部验证模型 |

#### 关键设计

- **多准则模型选择**：综合C-index、td-AUC、IBS、DCA净收益的加权评分
- **Coxnet CV**：采用lambda.min（最优预测）和lambda.1se（更简约）双规则
- **DeepSurv门槛**：事件数≥120才启用DeepSurv，否则降级为Coxnet
- **rank逆正态调和**：跨队列表达数据通过rank-based inverse normal transform统一尺度
- **模型卡片**：自动生成符合REMARK/TRIPOD+AI标准的模型文档

---

### 6.5 步骤05 — 验证与可视化

> 文件：`05_validation_visualization.py` | 796行  
> 输入：步骤04的模型 + 各外部队列数据  
> 输出：`data/validation_results/` 下森林图、KM曲线、DCA曲线、综合指标表

#### 主要函数

| 函数 | 功能 |
|------|------|
| `normalize_external_clinical(df, cohort_key)` | 标准化外部队列临床数据 |
| `transform_clinical_external(df, transformer)` | 应用训练集拟合的临床转换器到外部数据 |
| `load_geo_expression(config, geo_id)` | 加载GEO表达数据 |
| `expression_coxnet_scores(df, model_bundle, train_ref)` | Coxnet风险分数预测（含within-cohort rank INT调和） |
| `msk_genomic_predict(df, model_bundle)` | MSK基因组模型预测 |
| `km_groups_by_threshold(scores, time, event, threshold_source)` | 训练集冻结阈值分组 |
| `plot_km(groups, time, event, ctx)` | Kaplan-Meier曲线 + log-rank检验 |
| `td_auc_and_brier(model, X, y, y_train, times)` | 时间依赖AUC + Integrated Brier Score |
| `decision_curve_analysis(risk_scores, time, event, thresholds)` | 决策曲线分析（Vickers & Elkin 2006） |
| `plot_dca(dca_results, ctx)` | DCA曲线绘制 |
| `cptac_rna_protein_concordance(rna_df, protein_df)` | CPTAC RNA↔蛋白质Spearman一致性 |
| `htan_microenvironment_association(risk_scores, cell_fractions)` | HTAN微环境细胞分数关联分析 |

#### 验证队列与策略

| 队列 | 验证模型 | 调和策略 |
|------|----------|----------|
| MSK | minimal Cox + genomic Cox | 直接应用 |
| GEO GSE103479 | expression Coxnet | within-cohort rank INT |
| GEO GSE39582 | expression Coxnet | within-cohort rank INT |
| GEO GSE17538 | expression Coxnet | within-cohort rank INT |
| CPTAC | RNA↔protein Spearman | 配对相关性 |
| HTAN | 微环境细胞分数 | Spearman关联 |

#### 关键设计

- **训练集冻结阈值**：KM分组的阈值来自训练集，不在测试/外部集上重新确定
- **within-cohort rank INT**：外部表达数据先在自身队列内做rank逆正态变换，再投影到Coxnet
- **DCA净收益**：评估模型在不同阈值概率下的临床净收益，超越单纯C-index

---

### 6.6 步骤06 — 可解释性与生物学

> 文件：`06_interpretability_biology.py` | 443行  
> 输入：步骤02-05的全部结果 + MSigDB GMT文件  
> 输出：`data/interpretability_biology/` 下证据矩阵、通路富集、湿实验优先级

#### 主要函数

| 函数 | 功能 |
|------|------|
| `load_gmt(gmt_path)` | 加载MSigDB GMT基因集文件 |
| `hypergeometric_enrichment(gene_list, gene_sets, background)` | 超几何分布过表达分析 |
| `rsf_permutation_importance(rsf_model, X, y)` | RSF置换重要性 |

#### 内嵌基因集 (`BUNDLED_GENE_SETS`)

作为MSigDB GMT不可用时的回退，包含8个CRC相关Hallmark + KEGG集合：

- HALLMARK_WNT_BETA_CATENIN_SIGNALING
- HALLMARK_APOPTOSIS
- HALLMARK_DNA_REPAIR
- HALLMARK_E2F_TARGETS
- HALLMARK_MITOTIC_SPINDLE
- HALLMARK_GLYCOLYSIS
- KEGG_COLORECTAL_CANCER
- KEGG_CELL_CYCLE

#### 输出产物

| 产物 | 说明 |
|------|------|
| `core_biomarker_evidence_matrix` | 因果优先+Coxnet系数+临床系数+外部方向性综合证据 |
| `pathway_enrichment_results` | MSigDB通路ORA结果（Hallmark/KEGG/可选GO:BP） |
| `feature_importance_stability` | 特征重要性稳定性评估 |
| `wet_lab_validation_priority_list` | 湿实验验证优先级列表 |
| `mechanistic_hypotheses.md` | 机制假说文档 |

---

### 6.7 步骤07 — 可重复性IO与日志

> 文件：`07_reproducibility_io_logging.py` | 10行  
> 功能：兼容性入口点，导入并调用 `repro_io.command_line_main()`

此步骤不执行独立分析，而是提供命令行接口用于：
- 汇总所有步骤的运行日志
- 验证文件指纹一致性
- 生成完整的流水线清单

---

## 7. 数据流与依赖关系

### 7.1 步骤间数据流

```
rawData/ (只读)
    │
    ▼
01_multiomics_preprocessing.py
    │ 输出: data/preprocessed/
    │   ├── curated_clinical_*.tsv
    │   ├── curated_expression_*.tsv
    │   ├── curated_mutation_*.tsv
    │   ├── curated_cna_*.tsv
    │   ├── curated_methylation_*.tsv
    │   ├── curated_protein_*.tsv
    │   └── outcome_summary_*.tsv
    │
    ▼
02_causal_feature_screening.py
    │ 输入: data/preprocessed/ + rawData/GTEx_v8_eQTL/
    │ 输出: data/causal_screening/
    │   ├── univariable_cox_results.tsv
    │   ├── multivariable_cox_results.tsv
    │   ├── cis_eqtl_evidence.tsv
    │   └── causal_priority_feature_table.tsv
    │
    ▼
03_multiomics_pretraining.py
    │ 输入: data/preprocessed/ + data/causal_screening/
    │ 输出: data/multiomics_pretraining/
    │   ├── patient_embedding.tsv
    │   ├── modality_availability_mask.tsv
    │   ├── reconstruction_error.tsv
    │   ├── stability_summary.tsv
    │   └── model_bundle.joblib
    │
    ▼
04_survival_prediction_model.py
    │ 输入: data/preprocessed/ + data/multiomics_pretraining/
    │ 输出: data/survival_models/
    │   ├── risk_scores_*.tsv
    │   ├── model_comparison_metrics.tsv
    │   ├── model_manifest.tsv
    │   ├── model_card.md
    │   └── model_bundle_*.joblib
    │
    ▼
05_validation_visualization.py
    │ 输入: data/survival_models/ + rawData/外部队列
    │ 输出: data/validation_results/
    │   ├── forest_plot.pdf
    │   ├── km_*.pdf
    │   ├── dca_*.pdf
    │   └── combined_metrics.tsv
    │
    ▼
06_interpretability_biology.py
    │ 输入: data/causal_screening/ + data/survival_models/ + rawData/MSigDB/
    │ 输出: data/interpretability_biology/
    │   ├── core_biomarker_evidence_matrix.tsv
    │   ├── pathway_enrichment_results.tsv
    │   ├── wet_lab_validation_priority_list.tsv
    │   └── mechanistic_hypotheses.md
```

### 7.2 脚本间导入依赖

```
01_multiomics_preprocessing.py ──→ repro_io.py
02_causal_feature_screening.py ──→ repro_io.py
03_multiomics_pretraining.py  ──→ repro_io.py
04_survival_prediction_model.py ──→ repro_io.py
05_validation_visualization.py ──→ repro_io.py
06_interpretability_biology.py ──→ repro_io.py
07_reproducibility_io_logging.py ──→ repro_io.py (直接调用command_line_main)
```

所有脚本仅共享 `repro_io.py`，无脚本间直接导入。步骤间依赖完全通过文件I/O传递。

---

## 8. Python包依赖

### 8.1 核心依赖

| 包名 | 用途 | 使用脚本 |
|------|------|----------|
| `numpy` | 数值计算 | 全部 |
| `pandas` | 数据表格操作 | 全部 |
| `scipy` | 统计检验、超几何分布 | 02, 06 |
| `lifelines` | Cox PH回归、PH假设检验 | 02, 04, 05 |
| `scikit-survival (sksurv)` | Coxnet、RSF、Uno's C-index、td-AUC、IBS | 03, 04, 05 |
| `statsmodels` | BH FDR、统计模型 | repro_io, 02 |
| `scikit-learn` | PCA、TruncatedSVD、Imputer、Scaler、StratifiedShuffleSplit | 01, 03, 04 |
| `matplotlib` | 绘图 | 01, 05 |
| `seaborn` | 统计可视化 | 01, 05 |
| `joblib` | 模型序列化 | 03, 04 |
| `yaml` (PyYAML) | 配置文件解析 | repro_io |
| `hashlib` (标准库) | SHA256文件指纹 | repro_io |

### 8.2 可选依赖

| 包名 | 用途 | 条件 |
|------|------|------|
| `torch` | DeepSurv神经网络 | 事件数≥120时启用 |
| `pycox` | DeepSurv实现 | 事件数≥120时启用 |
| `torchtuples` | pycox依赖 | 同pycox |

### 8.3 版本记录机制

`repro_io.package_versions()` 在每次运行时自动记录所有已安装包的版本号，写入 `run_metadata.json`。

---

## 9. 项目运行方式

### 9.1 环境准备

```bash
# 安装核心依赖
pip install numpy pandas scipy lifelines scikit-survival statsmodels scikit-learn matplotlib seaborn joblib pyyaml

# 可选：安装DeepSurv依赖
pip install torch pycox torchtuples
```

### 9.2 配置文件

编辑 `ansisly/因果-多组学-预后/project_config.yaml`，确认以下路径正确：

- `project_root`: 项目根目录绝对路径
- 各队列在 `rawData/` 下的子目录名称
- `result_root`: 结果输出目录

### 9.3 顺序执行流水线

```bash
# 步骤1: 数据预处理
python ansisly/因果-多组学-预后/01_multiomics_preprocessing.py --config ansisly/因果-多组学-预后/project_config.yaml

# 步骤2: 因果特征筛选
python ansisly/因果-多组学-预后/02_causal_feature_screening.py --config ansisly/因果-多组学-预后/project_config.yaml

# 步骤3: 多组学嵌入
python ansisly/因果-多组学-预后/03_multiomics_pretraining.py --config ansisly/因果-多组学-预后/project_config.yaml

# 步骤4: 生存模型训练
python ansisly/因果-多组学-预后/04_survival_prediction_model.py --config ansisly/因果-多组学-预后/project_config.yaml

# 步骤5: 验证与可视化
python ansisly/因果-多组学-预后/05_validation_visualization.py --config ansisly/因果-多组学-预后/project_config.yaml

# 步骤6: 可解释性与生物学
python ansisly/因果-多组学-预后/06_interpretability_biology.py --config ansisly/因果-多组学-预后/project_config.yaml

# 步骤7: 可重复性汇总（可选）
python ansisly/因果-多组学-预后/07_reproducibility_io_logging.py --config ansisly/因果-多组学-预后/project_config.yaml
```

### 9.4 通用CLI参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | YAML配置文件路径 | `project_config.yaml` |
| `--endpoint` | 终点类型 (OS/PFS/DFS) | 配置文件中的值 |
| `--seed` | 覆盖全局随机种子 | 配置文件中的值 |
| `--dry-run` | 干运行模式（不写入文件） | False |
| `--max-features-per-omics` | 覆盖每组学最大特征数 | 配置文件中的值 |

### 9.5 输出组织

每次运行在 `result/` 下创建带时间戳的目录：

```
result/20260525_143000_step01/
    ├── run_metadata.json          # 运行元数据（种子、包版本、配置快照）
    ├── pipeline_manifest.tsv      # 所有输出文件清单+SHA256指纹
    ├── run.log                    # 运行日志
    ├── warnings.json              # 警告列表
    └── *.tsv / *.pdf / *.joblib   # 步骤输出文件
```

---

## 10. 关键设计决策

### 10.1 Leakage-Safe（防泄露）策略

| 环节 | 防泄露措施 |
|------|-----------|
| 特征选择 | 仅在训练集上计算方差、执行Cox回归 |
| 模型拟合 | PCA/SVD/Cox/RSF仅在训练集上fit |
| 阈值确定 | KM分组阈值来自训练集，冻结后应用到测试/外部集 |
| 嵌入投影 | 测试集通过transform投影，不参与编码器拟合 |

### 10.2 跨平台数据调和

- **Rank-based Inverse Normal Transform (Blom Score)**：将不同平台/队列的表达数据通过rank变换映射到标准正态分布，消除平台效应
- **Within-cohort Rank INT**：外部验证时先在自身队列内做rank INT，再投影到训练集拟合的模型

### 10.3 缺失组学处理

- **模态可用性掩码**：二进制向量标记每个患者的可用组学模态
- **零填充**：缺失模态的嵌入维度用零填充
- 模型可感知哪些维度是真实观测、哪些是填充

### 10.4 因果优先分级

特征按因果证据强度分级：
1. **有cis-eQTL工具变量** → 最高优先级（可进行MR/SMR因果推断）
2. **仅预后关联** → 次优先级（需谨慎解释跨队列泛化性）
3. **当前MR/SMR限制** → 明确文档化（缺少CRC OS GWAS汇总统计）

### 10.5 模型选择策略

采用多准则加权评分而非单一C-index：
- Uno's IPCW C-index (权重最高)
- 时间依赖AUC
- Integrated Brier Score
- DCA净收益

### 10.6 DeepSurv降级规则

当训练集事件数 < 120 时，自动降级为Coxnet（弹性网惩罚Cox），避免神经网络在小样本上过拟合。

---

## 11. 数据集来源与许可

| 队列 | 来源 | 数据类型 | 许可 |
|------|------|----------|------|
| TCGA COAD/READ | cBioPortal (Pan-Can Atlas 2018) | 临床+多组学 | CC BY 4.0 |
| MSKCC | cBioPortal (crc_msk_2017) | 临床+基因组 | CC BY 4.0 |
| CPTAC | cBioPortal (coad_cptac_2019) | 临床+蛋白质组 | CC BY 4.0 |
| HTAN | cBioPortal (crc_hta8_htan_2024) | 临床+微环境 | CC BY 4.0 |
| GEO GSE103479 | NCBI GEO | 表达 | 公共数据 |
| GTEx v8 eQTL | GTEx Portal | eQTL | CC0 |
| eQTLGen | eQTLGen Consortium | 血液eQTL | 学术使用 |
| MSigDB | Broad Institute | 基因集 | 学术使用 |

> 详细来源、许可和引用信息见 `repro_io.DATASET_PROVENANCE`

---

*本文档基于项目源代码自动分析生成，涵盖项目整体架构、7步流水线各模块职责与关键函数、数据流依赖关系、Python包依赖、运行方式及核心设计决策。*