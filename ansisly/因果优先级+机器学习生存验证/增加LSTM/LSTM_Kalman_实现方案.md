# LSTM + 卡尔曼滤波实现方案

## 1. 执行摘要

当前课题的主线仍然是 **CRC 总生存（OS）风险预测**。现有流水线的核心数据形态以 **基线多组学 + 临床特征 + OS 生存结局** 为主，而不是患者层面的真实重复纵向测量序列。因此，**不建议直接把 LSTM 作为主生存模型替代 Cox / Coxnet / RSF**。

更合理的引入顺序是：

1. **先引入 Kalman 风险平滑/融合模块**
2. **再扩展为多时间点风险轨迹分析**
3. **最后再探索 LSTM + Kalman 的补充性时序建模**

也就是说，LSTM + 卡尔曼滤波更适合被定位为：

- 风险分数的平滑器
- 多模型观测的融合器
- 多时间窗 risk trajectory 的探索性模块

而不是直接替代当前的主生存模型。

---

## 2. 现有流水线结构与可复用接口

当前项目已经有比较完整的模块化设计，主要入口如下：

- `script/integrated_pipeline.py`
- `script/_step01.py`
- `script/_step02.py`
- `script/_step03_04.py`
- `script/_step05_rest.py`
- `script/_pipeline_core.py`
- `script/_causal_infer.py`
- `script/_gan_augment.py`
- `script/_step_common.py`

### 2.1 Step 01：数据预处理

Step 01 负责：

- 临床结局标准化
- OS 时间/事件统一
- IPCW 权重构建
- pseudo-observation 构建
- 多队列 QC

这意味着后续如果要做多时间点风险轨迹，已有的结局标准化和 IPCW 逻辑可以直接复用。

### 2.2 Step 02：因果优先级筛选

Step 02 负责：

- train-only 方差筛选
- 单变量 Cox
- 多变量 Cox（混杂调整）
- RMST doubly robust 因果筛选

这说明项目已经有很强的“生存 + 因果筛选”基础，LSTM 不应破坏这一主线，而应作为补充模块加入。

### 2.3 Step 03 / 04：生存模型训练

当前主模型包括：

- clinical Cox
- TNM Cox
- minimal Cox
- Random Survival Forest
- expression Coxnet
- clinical + RNA Coxnet
- clinical + embedding Cox
- clinical-anchored ensemble
- GAN 增强辅助模型

模型选择仍以 **完整生存指标** 为主，而非单纯 36 个月二分类。

### 2.4 Step 05：验证与可视化

Step 05 主要负责：

- 锁定模型 manifest
- 外部验证
- KM 分层
- DCA / ROC / calibration
- 输出清单与最终报告

这意味着若新增 LSTM + Kalman 模块，最合适的做法是：

- 作为 Step 04 的候选扩展
- 或作为 Step 05 的后处理验证模块
- 不直接嵌入主模型选择逻辑

### 2.5 multi-timepoint 分析

项目里已经存在 multi-timepoint 的分析框架，这为“多时间窗风险轨迹 + Kalman 平滑”提供了天然接口。

---

## 3. 为什么 LSTM 不适合作为当前主模型

### 3.1 数据形态不匹配

LSTM 需要的是**真实序列输入**，例如同一患者多时间点的：

- 组学检测
- ctDNA
- 生化指标
- 影像特征
- 治疗响应轨迹

但当前数据更多是单次基线特征 + 生存结局，因此直接把 LSTM 作为主模型会显得“序列来源不真实”。

### 3.2 生存删失问题不能直接忽略

普通 LSTM 分类/回归并不能自然处理：

- 右删失
- 不同随访长度
- 固定时间窗风险定义
- IPCW / pseudo-observation 结构

如果硬上 LSTM，必须有生存专用 loss 设计，否则方法学上不稳。

### 3.3 容易过拟合

当前流程已经明确对深度模型保持谨慎。若样本量和事件数不足，LSTM 很容易学到噪声而非真实信号。

### 3.4 审稿/答辩风险

如果没有真实 longitudinal data，直接宣称“LSTM 做动态生存预测”容易被质疑为：

- 伪时间序列
- 伪纵向建模
- 过度包装深度学习创新

所以 LSTM 更适合做补充实验，而不是主结论。

---

## 4. 第一阶段：Kalman 风险平滑模块

### 4.1 目标

对现有模型输出的风险分数进行平滑和融合，减少噪声波动，提高稳定性和 calibration。

### 4.2 适用输入

可将以下结果作为 Kalman 观测输入：

- clinical Cox risk
- Coxnet risk
- RSF risk
- embedding Cox risk
- ensemble risk
- 多时间点 tau 风险分数

### 4.3 推荐方法

将风险看成一个潜在状态过程：

- 状态方程：真实风险随时间缓慢演化
- 观测方程：不同模型输出是带噪观测

可采用最简的一维状态空间模型：

- `x_t = x_{t-1} + w_t`
- `z_t = x_t + v_t`

其中：

- `x_t`：潜在真实风险
- `z_t`：某模型或某时间点观测风险
- `w_t`：状态噪声
- `v_t`：观测噪声

### 4.4 输出文件建议

建议新增：

- `kalman_smoothed_risk_scores.tsv`
- `kalman_smoothing_utility.tsv`
- `kalman_smoothing_manifest.json`

### 4.5 评价指标

对比平滑前后：

- Harrell C-index
- Uno C-index IPCW
- time-dependent AUC
- Brier score
- calibration curve
- KM risk strata

### 4.6 定位

Kalman 风险平滑应定位为：

- post-hoc calibration
- robustness analysis
- auxiliary fusion module

不要参与 primary model selection。

---

## 5. 第二阶段：多时间点 Kalman 风险轨迹

### 5.1 目标

把不同时间窗下的风险预测看成一个 trajectory：

- 12m
- 24m
- 36m
- 48m
- 60m

然后用 Kalman filter / smoother 生成更稳定的时间轨迹。

### 5.2 已有可复用能力

现有代码已经支持：

- 固定时间窗结局
- tagged IPCW 权重
- tagged pseudo-observation

这些函数可作为多时间点扩展的基础。

### 5.3 推荐输出

可输出：

- `multi_timepoint_risk_trajectory_raw.tsv`
- `multi_timepoint_risk_trajectory_kalman.tsv`
- `multi_timepoint_kalman_summary.tsv`

### 5.4 适合展示的结果

- 个体患者的风险轨迹图
- 高风险/低风险组平均轨迹图
- 不同 tau 的 AUC 曲线
- 平滑前后 calibration 对比

### 5.5 定位

这一步仍然不需要 LSTM，可先把时间窗风险做成序列化状态估计问题。

---

## 6. 第三阶段：探索性 LSTM + Kalman 风险轨迹模型

### 6.1 适合什么输入

如果要引入 LSTM，建议输入不是“原始静态表格硬序列化”，而是以下更合理的形式：

1. **clinical + omics embedding**
2. **causal-priority genes**
3. **多时间点 pseudo-observation risk**
4. **不同模型的风险向量序列**

### 6.2 模型思路

LSTM 负责学习多时间窗风险变化模式：

- 输入：患者特征 + 时间编码
- 输出：每个 tau 的风险预测

然后再用 Kalman 对输出进行平滑：

- LSTM 给出 raw trajectory
- Kalman 输出 smoothed trajectory

### 6.3 损失函数建议

如果要训练，建议使用 survival-aware 的回归目标，而不是普通分类：

- IPCW 加权 MSE
- pseudo-observation 回归损失
- 结合排序损失的辅助目标

例如：

- `Loss = IPCW_MSE + lambda * ranking_loss`

### 6.4 应明确的限制

这个方案应明确标注为：

- exploratory
- supplementary
- not primary selection eligible

### 6.5 不建议的说法

不要直接写成：

- longitudinal omics LSTM
- patient-level temporal omics sequence model

除非后续真的有多时间点重复测量。

更稳妥的名字是：

- multi-horizon risk trajectory learner
- pseudo-longitudinal risk model
- exploratory LSTM-Kalman trajectory module

---

## 7. 建议的代码模块划分

如果未来要真正实现，建议保持和当前流水线一致的模块化方式。

### 7.1 推荐新模块

- `_step04_5_kalman_risk_smoothing.py`
- `_step04_6_lstm_kalman_risk_trajectory.py`

### 7.2 可能的集成位置

- `integrated_pipeline.py` 增加新的导出入口
- Step 04 后加入 Kalman / LSTM-Kalman
- Step 05 读取并验证新增输出

### 7.3 输入输出职责

#### `_step04_5_kalman_risk_smoothing.py`

输入：

- Step 04 风险表
- 多时间点 risk

输出：

- 平滑风险表
- 平滑性能报告
- manifest

#### `_step04_6_lstm_kalman_risk_trajectory.py`

输入：

- training features
- multi-horizon pseudo-risk labels
- IPCW 权重

输出：

- LSTM raw trajectory
- Kalman-smoothed trajectory
- 训练/验证指标
- 可视化结果

---

## 8. 评价指标与比较策略

建议比较以下指标：

- Harrell C-index
- Uno C-index IPCW
- time-dependent AUC
- integrated Brier score
- calibration curve
- KM stratification

### 8.1 选择原则

- 主模型选择：仍以完整生存指标为主
- Kalman / LSTM-Kalman：作为辅助比较，不直接参与主模型锁定
- 外部队列：只用于锁定后泛化评估

---

## 9. 泄漏控制与审稿风险控制

必须严格遵守以下规则：

1. **所有 scaler / imputer / feature selection / LSTM / Kalman 参数估计仅在训练集进行**
2. **internal validation 不得参与调参**
3. **external cohorts 不得用于模型选择**
4. **Kalman / LSTM 结果不得回流到特征筛选阶段**
5. **没有真实纵向重复测量时，不得把 LSTM 说成真实 longitudinal model**

如果违反这些规则，整个方案的可信度会下降。

---

## 10. 推荐实施路线

### P1：先做 Kalman 多模型风险平滑

这是最稳妥、成本最低、最符合当前数据结构的方案。

### P2：再做 multi-timepoint Kalman 风险轨迹

利用已有固定 tau / pseudo-observation 框架。

### P3：最后做探索性 LSTM + Kalman

仅用于补充实验、方法创新展示和敏感性分析。

---

## 11. 可直接写入论文/方案的表述

### 中文表述

鉴于现有队列主要提供基线多组学特征而非重复纵向测量，本研究不将 LSTM 作为主要生存预测模型，而是优先引入基于卡尔曼滤波的潜在风险平滑模块，用于融合多模型或多时间窗的带噪风险估计。在此基础上，可进一步探索基于 pseudo-observation 构建的多时间点风险轨迹，并使用 LSTM-Kalman 进行补充性时序建模，但该部分仅作为探索性分析，不参与主模型选择。

### English draft

Given that the available cohorts primarily provide baseline multi-omics profiles rather than repeated longitudinal measurements, we do not use LSTM as the primary survival learner. Instead, we first introduce a Kalman filter-based latent risk smoothing module to integrate noisy risk estimates generated by heterogeneous survival models or multiple time horizons. On top of that, we may further explore pseudo-observation-derived multi-timepoint risk trajectories with an exploratory LSTM-Kalman module. This component should be treated as supplementary analysis rather than a primary model selection criterion.

---

## 12. 最终结论

### 推荐策略

**当前课题最适合的路线是：**

- 主模型仍然保持 Cox / Coxnet / RSF / ensemble
- 先加 Kalman 风险平滑
- 再扩展多时间点风险轨迹
- LSTM + Kalman 仅作为探索性补充

### 不推荐策略

- 直接把 LSTM 作为主模型
- 没有真实时间序列却强行做“纵向 LSTM”
- 让深度模型参与主模型选择与锁定

这份方案可作为后续实现、论文方法设计和答辩说明的基础文档。