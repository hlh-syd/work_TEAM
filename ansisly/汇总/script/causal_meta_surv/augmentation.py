"""augmentation.py — 简化版数据增强（噪声增强 + 质量门）。

实现（spec.md / Fusion_Plan §6）：
- Level 1: 标签保持的轻扰动（Gaussian noise + gene masking）
- 质量门：origin_patient_id + is_synthetic 标志
- 禁止复制最近邻生存标签（伪重复防护）
- 增强样本低权重进入辅助头

注意：完整扩散模型实现复杂度高，本模块实现 Level 1 轻扰动 + 质量门骨架，
docstring 中注明简化范围。完整 Level 2/3 可后续扩展。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .contracts import CMIBConfig, SurvivalRecordBatch
from .meta_trainer import _index_batch


class Augmentor:
    """简化版数据增强器（Level 1 轻扰动）。

    简化范围：
    - 仅实现 Level 1：Gaussian noise + gene masking
    - 不实现 Level 2 latent mixup 与 Level 3 扩散模型
    - 不创建新的 Cox 风险集成员（synthetic 仅进辅助头）
    - 带质量门：is_synthetic=True, sample_weight=aug_synthetic_weight
    """

    def __init__(self, config: CMIBConfig, gene_std: Optional[np.ndarray] = None):
        self.config = config
        self.gene_std = gene_std  # 训练折基因标准差（用于缩放噪声）
        self.rng = np.random.RandomState(config.seed)

    def augment_batch(
        self,
        batch: SurvivalRecordBatch,
        noise_std: Optional[float] = None,
        mask_frac: Optional[float] = None,
    ) -> SurvivalRecordBatch:
        """对 batch 应用 Level 1 增强。

        Args:
            batch: 原始真实样本 batch
            noise_std: 噪声标准差（None 用 config.aug_noise_std）
            mask_frac: 基因 mask 比例（None 用 config.aug_gene_mask_frac）

        Returns:
            新的 SurvivalRecordBatch，is_synthetic=True，
            sample_weight=config.aug_synthetic_weight
        """
        noise_std = noise_std if noise_std is not None else self.config.aug_noise_std
        mask_frac = mask_frac if mask_frac is not None else self.config.aug_gene_mask_frac

        x = batch.x_gene.clone()
        # Gaussian noise（用训练折统计缩放）
        if self.gene_std is not None:
            std_tensor = torch.tensor(
                self.gene_std, dtype=x.dtype, device=x.device
            )
            std_tensor = torch.where(std_tensor > 0, std_tensor, torch.ones_like(std_tensor))
            noise = torch.randn_like(x) * (noise_std * std_tensor)
        else:
            noise = torch.randn_like(x) * noise_std
        x = x + noise

        # gene masking
        if mask_frac > 0:
            mask = (torch.rand_like(x) > mask_frac).float()
            x = x * mask

        # 填补 NaN
        x = torch.nan_to_num(x, nan=0.0)

        # 标签保持：time/event/treatment 不变
        # origin_patient_id = 原始 patient_id
        origin = list(batch.patient_id)  # 来源 = 自身（增强视图）
        weights = torch.full_like(batch.sample_weight, self.config.aug_synthetic_weight)

        return SurvivalRecordBatch(
            patient_id=list(batch.patient_id),
            x_gene=x,
            x_clinical=batch.x_clinical.clone(),
            clinical_mask=batch.clinical_mask.clone(),
            task_id=batch.task_id.clone(),
            time=batch.time.clone(),
            event=batch.event.clone(),
            treatment=batch.treatment.clone() if batch.treatment is not None else None,
            sample_weight=weights,
            origin_patient_id=origin,
            is_synthetic=torch.ones_like(batch.is_synthetic, dtype=torch.bool),
        )

    def quality_gate(
        self,
        real_batch: SurvivalRecordBatch,
        aug_batch: SurvivalRecordBatch,
    ) -> bool:
        """质量门：检查增强样本是否通过。

        检查项（简化）：
        1. 无复制：is_synthetic=True
        2. 标签保持：time/event 与 real 一致
        3. 分布一致：增强后均值漂移 < 0.5 * std

        Returns:
            True 通过，False 不通过
        """
        # 1. is_synthetic 标志
        if not bool(aug_batch.is_synthetic.all().item()):
            return False
        # 2. 标签保持
        if not torch.equal(aug_batch.time, real_batch.time):
            return False
        if not torch.equal(aug_batch.event, real_batch.event):
            return False
        # 3. 分布漂移
        real_mean = real_batch.x_gene.nanmean().item()
        aug_mean = aug_batch.x_gene.nanmean().item()
        real_std = real_batch.x_gene.nanstd().item()
        if abs(aug_mean - real_mean) > 0.5 * max(real_std, 1e-6):
            return False
        return True


def augment_for_representation(
    batch: SurvivalRecordBatch,
    config: CMIBConfig,
    gene_std: Optional[np.ndarray] = None,
) -> Tuple[SurvivalRecordBatch, SurvivalRecordBatch]:
    """生成表示学习的增强视图。

    用于 consistency loss：返回 (real_view, aug_view)，
    其中 real_view 是原 batch，aug_view 是增强视图。

    简化：仅用于 cons_loss，不进入 Cox 主风险集。
    """
    aug = Augmentor(config, gene_std)
    aug_batch = aug.augment_batch(batch)
    return batch, aug_batch
