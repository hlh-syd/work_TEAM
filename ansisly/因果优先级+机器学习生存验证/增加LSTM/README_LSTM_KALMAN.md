# LSTM + Kalman Filter Implementation

## Overview

This directory implements the LSTM + Kalman filter risk smoothing and trajectory analysis as described in `LSTM_Kalman_实现方案.md`.

**IMPORTANT POSITIONING:**
- LSTM + Kalman modules are **NOT primary survival models**
- They are **NOT eligible for primary model selection**
- They are positioned as **exploratory/supplementary analysis only**
- Primary models remain: Cox, Coxnet, RSF, ensemble

## Implementation Structure

### Stage 1 + Stage 2: `_step04_5_kalman_risk_smoothing.py`

**Stage 1: Multi-model Risk Smoothing**
- Applies Kalman filter to fuse risk scores from multiple models
- Treats each model's output as a noisy observation of latent true risk
- Reduces noise and improves calibration

**Stage 2: Multi-timepoint Risk Trajectory**
- Generates risk trajectories across multiple time horizons (12/24/36/48/60 months)
- Uses `add_fixed_endpoint`, `compute_ipcw_weights_tagged`, `compute_pseudo_observations_tagged`
- Applies Kalman smoothing across time to create stable trajectories

**Usage:**
```bash
python _step04_5_kalman_risk_smoothing.py \
    --config project_config.yaml \
    --endpoint OS \
    --tau-grid "12,24,36,48,60"
```

**Outputs:**
- `survival_models/kalman_smoothed_risk_scores.tsv` - Smoothed risk scores (Stage 1)
- `survival_models/kalman_smoothing_utility.tsv` - Utility metrics
- `survival_models/multi_timepoint_risk_trajectory_raw.tsv` - Raw trajectories (Stage 2)
- `survival_models/multi_timepoint_risk_trajectory_kalman.tsv` - Smoothed trajectories
- `survival_models/multi_timepoint_kalman_summary.tsv` - Summary statistics
- `survival_models/kalman_smoothing_manifest.json` - Metadata

### Stage 3: `_step04_6_lstm_kalman_risk_trajectory.py`

**LSTM + Kalman Risk Trajectory (Exploratory)**
- Trains LSTM over an explicit `tau_list` sequence
- Uses survival-aware loss: IPCW-weighted MSE on pseudo-observations
- Applies Kalman smoothing to LSTM predictions
- Uses `clinical_plus_embedding` as the default input source
- **EXPLORATORY ONLY** - not a primary survival model

**Usage:**
```bash
python _step04_6_lstm_kalman_risk_trajectory.py \
    --config project_config.yaml \
    --endpoint OS \
    --tau-grid "12,24,36,48,60" \
    --lstm-hidden-dim 64 \
    --lstm-num-layers 1 \
    --lstm-dropout 0.3 \
    --lstm-epochs 100 \
    --lstm-batch-size 32 \
    --feature-source clinical_plus_embedding
```

**Parameters:**
- `--lstm-hidden-dim`: LSTM hidden layer size (default: 64)
- `--lstm-num-layers`: Number of LSTM layers (default: 1)
- `--lstm-dropout`: Dropout rate (default: 0.3)
- `--lstm-epochs`: Maximum training epochs (default: 100)
- `--lstm-batch-size`: Batch size (default: 32)
- `--feature-source`: Feature set (`clinical`, `embedding`, `clinical_plus_embedding`)

**Outputs:**
- `survival_models/lstm_risk_trajectory_raw.tsv` - LSTM predictions
- `survival_models/lstm_risk_trajectory_kalman_smoothed.tsv` - Kalman-smoothed LSTM
- `survival_models/lstm_kalman_trajectory_manifest.json` - Metadata

**Requirements:**
- PyTorch must be installed for LSTM training
- If PyTorch is not available, module will skip gracefully

## Integration with Pipeline

The new modules are integrated into `integrated_pipeline.py` and can be called as:

```python
from integrated_pipeline import main_step04_5, main_step04_6

# After Step 04 has generated risk scores
main_step04_5()  # Kalman smoothing (Stage 1 + 2)
main_step04_6()  # LSTM + Kalman (Stage 3, optional)
```

The default full pipeline also includes these steps:
- Step 04.5: Kalman risk smoothing
- Step 04.6: LSTM + Kalman risk trajectory

## Prerequisites

1. **Step 04 must be completed first** to generate:
   - `survival_models/internal_validation_risk_scores.tsv`
   - `survival_models/tcga_train_internal_validation_split.tsv`

2. **For Stage 3 (LSTM):**
   - PyTorch installation recommended
   - Clinical endpoint data
   - Feature data (clinical, embedding, or combined)

## Leakage Control

All modules strictly follow leakage-safe practices:

✅ **Correct:**
- Kalman parameters fitted on TRAIN split only
- LSTM trained on TRAIN split, validated on internal validation
- Feature scalers fitted on TRAIN split only
- No external cohorts used for training/tuning

❌ **Prevented:**
- Internal validation never used for parameter tuning
- External cohorts never used for model selection
- No information leakage across splits

## Evaluation Metrics

The modules compute:
- **Harrell C-index** (concordance)
- **Uno C-index** (IPCW-based, if sksurv available)
- **Time-dependent AUC** (if sksurv available)
- **Integrated Brier Score** (if sksurv available)
- **Correlation with pseudo-observations** (for trajectory analysis)

## Positioning for Publications

### Recommended Language

**English:**
> Given that the available cohorts primarily provide baseline multi-omics profiles rather than repeated longitudinal measurements, we do not use LSTM as the primary survival learner. Instead, we first introduce a Kalman filter-based latent risk smoothing module to integrate noisy risk estimates generated by heterogeneous survival models or multiple time horizons. On top of that, we may further explore pseudo-observation-derived multi-timepoint risk trajectories with an exploratory LSTM-Kalman module. This component should be treated as supplementary analysis rather than a primary model selection criterion.

**中文:**
> 鉴于现有队列主要提供基线多组学特征而非重复纵向测量，本研究不将 LSTM 作为主要生存预测模型，而是优先引入基于卡尔曼滤波的潜在风险平滑模块，用于融合多模型或多时间窗的带噪风险估计。在此基础上，可进一步探索基于 pseudo-observation 构建的多时间点风险轨迹，并使用 LSTM-Kalman 进行补充性时序建模，但该部分仅作为探索性分析，不参与主模型选择。

### What NOT to Say

❌ "LSTM as primary survival model"
❌ "Longitudinal omics LSTM" (without real time-series data)
❌ "Patient-level temporal omics sequence model" (misleading)
❌ "Deep learning outperforms Cox models" (positioning issue)

### What TO Say

✅ "Multi-horizon risk trajectory learner"
✅ "Pseudo-longitudinal risk model"
✅ "Exploratory LSTM-Kalman trajectory module"
✅ "Post-hoc risk calibration and smoothing"
✅ "Supplementary sensitivity analysis"

## File Structure

```
增加LSTM/
├── LSTM_Kalman_实现方案.md              # Implementation plan (approved)
├── README_LSTM_KALMAN.md                # This file
├── _step04_5_kalman_risk_smoothing.py   # Stage 1 + 2 implementation
├── _step04_6_lstm_kalman_risk_trajectory.py  # Stage 3 implementation
├── integrated_pipeline.py               # Updated orchestrator
└── [other existing modules...]          # Mirror of script/
```

## Troubleshooting

### Module Import Errors
The modules use `from _pipeline_core import *` which triggers `enforce_project_environment()` at import time. This checks:
- Python version (3.12.4 required)
- Virtual environment path
- requirements.txt sync

If you see environment errors, ensure you're using the correct Python interpreter.

### LSTM Training Fails
If PyTorch is not available:
- The module will log a warning and skip LSTM training gracefully
- Stage 1 and Stage 2 (Kalman only) will still work
- Install PyTorch: `pip install torch`

### Missing Input Files
Ensure Step 04 has been run successfully before calling these modules. Required files:
- `survival_models/internal_validation_risk_scores.tsv`
- `survival_models/tcga_train_internal_validation_split.tsv`
- `preprocessed/tcga_os_clinical_endpoint_qc.tsv` (or raw clinical file)

## Contact & Maintenance

For questions about this implementation, refer to:
1. `LSTM_Kalman_实现方案.md` - The original specification
2. Module docstrings in the Python files
3. The plan document's positioning guidelines (Section 10-12)

---

**Last Updated:** 2026-06-22
**Implementation Version:** 1.1
**Status:** Complete (Stage 1, 2, 3)
