"""target_trainer.py — CRC 适配训练器。

实现（spec.md / Fusion_Plan §7 Phase 5）：
1. train_outer_fold: 含 inner CV 早停
2. 早停只用 target-train inner fold（严禁 locked validation 选 epoch）
3. beta schedule 与 free-bits
4. gradient clipping 与 AdamW
5. save_checkpoint / load_checkpoint
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .contracts import CMIBConfig, CMIBOutput, SurvivalRecordBatch
from .evaluation import harrell_c_index
from .losses import compute_cmib_loss
from .meta_trainer import clone_parameters, restore_parameters
from .preprocessing import FoldPreprocessorImpl, group_kfold_indices

LOGGER = logging.getLogger("cmib_surv.target_trainer")


def _augment_batch(batch: SurvivalRecordBatch, config: CMIBConfig) -> SurvivalRecordBatch:
    """基因 mask + 高斯噪声增强（与 035 _augment_batch 对齐）。

    Args:
        batch: 原始训练 batch
        config: CMIBConfig（用 gene_mask_probability 和 gene_noise_std）

    Returns:
        新 SurvivalRecordBatch（仅 x_gene 被增强，其余字段保持不变）
    """
    mask = torch.rand_like(batch.x_gene) < config.gene_mask_probability
    noise = torch.randn_like(batch.x_gene) * config.gene_noise_std
    augmented_gene = torch.where(mask, torch.zeros_like(batch.x_gene), batch.x_gene + noise)
    return SurvivalRecordBatch(
        patient_id=list(batch.patient_id),
        x_gene=augmented_gene,
        x_clinical=batch.x_clinical,
        clinical_mask=batch.clinical_mask,
        task_id=batch.task_id,
        time=batch.time,
        event=batch.event,
        treatment=batch.treatment,
        sample_weight=batch.sample_weight,
        origin_patient_id=batch.origin_patient_id,
        is_synthetic=batch.is_synthetic,
    )


def _set_trainable_stage(model: nn.Module, stage: str) -> None:
    """分阶段解冻调度（参考 035 _set_trainable_stage）。

    stage1: 只训 head/adapter/fusion/vib/crc_branch
    stage2: + meta_encoder/causal_encoder/clinical_encoder/domain_head
    stage3: + sparse_gate/gene_proj
    """
    stage1_prefixes = (
        "cox_head", "aux_head", "domain_head",
        "task_adapter", "fusion", "vib", "crc_branch",
    )
    stage2_prefixes = stage1_prefixes + (
        "meta_encoder", "causal_encoder", "clinical_encoder",
    )
    stage3_prefixes = stage2_prefixes + (
        "sparse_gate", "gene_proj",
    )
    prefixes = {
        "stage1": stage1_prefixes,
        "stage2": stage2_prefixes,
        "stage3": stage3_prefixes,
    }[stage]
    for name, param in model.named_parameters():
        param.requires_grad_(any(name.startswith(p) for p in prefixes))


class TargetTrainer:
    """CRC 适配训练器（含 inner CV 早停）。"""

    def __init__(
        self,
        model: nn.Module,
        config: CMIBConfig,
        gene_registry: Sequence[str],
        device: str = "cpu",
    ):
        self.model = model
        self.config = config
        self.gene_registry = list(gene_registry)
        self.device = device
        self.theta0_state: Optional[Dict[str, torch.Tensor]] = None  # 元初始化锚点
        self.history: List[Dict[str, Any]] = []

    def set_meta_init(self, theta0_state: Dict[str, torch.Tensor]) -> None:
        """设置元初始化参数（用于 anchor loss）。"""
        self.theta0_state = theta0_state

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """判别式学习率：head > adapter/fusion > shared。"""
        head_params, adapter_params, shared_params = [], [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in name for k in ["cox_head", "aux_head", "domain_head"]):
                head_params.append(p)
            elif any(k in name for k in ["task_adapter", "fusion", "vib", "crc_branch"]):
                adapter_params.append(p)
            else:
                shared_params.append(p)
        groups = []
        if head_params:
            groups.append({"params": head_params, "lr": self.config.lr_head,
                           "base_lr": self.config.lr_head})  # P2-1: 记录 base_lr
        if adapter_params:
            groups.append({"params": adapter_params, "lr": self.config.lr_adapter,
                           "base_lr": self.config.lr_adapter})  # P2-1
        if shared_params:
            groups.append({"params": shared_params, "lr": self.config.lr_shared,
                           "base_lr": self.config.lr_shared})  # P2-1
        return torch.optim.AdamW(groups, weight_decay=self.config.weight_decay)

    def _beta_schedule(self, epoch: int, max_epochs: int) -> float:
        """beta warm-up：前 15% epoch 从 0 线性增到 beta_max。"""
        warmup = max(1, int(self.config.beta_warmup_frac * max_epochs))
        if epoch < warmup:
            return self.config.beta_max * (epoch + 1) / warmup
        return self.config.beta_max

    def train_outer_fold(
        self,
        train_batch: SurvivalRecordBatch,
        val_batch: Optional[SurvivalRecordBatch] = None,
        max_epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """训练一个 outer fold（分阶段解冻）。

        Stage 1: 只训 cox_head/aux_head/domain_head/task_adapter/fusion/vib/crc_branch
        Stage 2: 解冻 meta_encoder/causal_encoder/clinical_encoder/domain_head
        Stage 3: 解冻 sparse_gate/gene_proj（仅当 stage2 改善 > min_delta 时）

        每阶段独立早停和 best_state 追踪。

        Returns:
            {'best_epoch': int, 'best_val_c': float, 'history': list}
        """
        max_epochs = max_epochs or self.config.max_epochs
        stage1_epochs = max_epochs
        stage2_epochs = getattr(self.config, 'max_epochs_stage2', max(2, max_epochs * 2 // 3))

        # Stage 1: head/adapter/fusion/vib only
        _set_trainable_stage(self.model, "stage1")
        s1_state, s1_score, s1_history = self._fit_stage(
            train_batch, val_batch, stage1_epochs, stage_name="stage1"
        )

        # Stage 2: + encoders
        _set_trainable_stage(self.model, "stage2")
        s2_state, s2_score, s2_history = self._fit_stage(
            train_batch, val_batch, stage2_epochs, stage_name="stage2"
        )

        # Pick better of stage1/stage2
        if s2_score > s1_score:
            best_state, best_score, selected_stage = s2_state, s2_score, "stage2"
        else:
            best_state, best_score, selected_stage = s1_state, s1_score, "stage1"
            restore_parameters(self.model, s1_state)

        # Stage 3 (optional): + gate/projection (only if stage2 gained)
        s3_history: List[Dict[str, Any]] = []
        gain = s2_score - s1_score
        unlock_gate_gain = getattr(self.config, 'unlock_projection_min_gain', 0.005)
        if gain >= unlock_gate_gain:
            _set_trainable_stage(self.model, "stage3")
            s3_state, s3_score, s3_history = self._fit_stage(
                train_batch, val_batch, max(2, stage2_epochs // 2), stage_name="stage3"
            )
            if s3_score > best_score:
                best_state, best_score, selected_stage = s3_state, s3_score, "stage3"

        # Restore best overall state and unfreeze all
        restore_parameters(self.model, best_state)
        for p in self.model.parameters():
            p.requires_grad_(True)

        all_history = s1_history + s2_history + s3_history
        total_epochs = len(all_history)
        return {
            "best_epoch": total_epochs,
            "best_val_c": best_score,
            "history": all_history,
            "selected_stage": selected_stage,
        }

    def _fit_stage(
        self,
        train_batch: SurvivalRecordBatch,
        val_batch: Optional[SurvivalRecordBatch],
        max_epochs: int,
        stage_name: str = "stage",
    ) -> Tuple[Dict[str, torch.Tensor], float, List[Dict[str, Any]]]:
        """单阶段训练循环：独立 optimizer + 早停 + best state。"""
        optimizer = self._build_optimizer()
        best_val_c = -np.inf
        best_state = clone_parameters(self.model)
        epochs_no_improve = 0
        history: List[Dict[str, Any]] = []

        # P2-1: LR warmup 配置
        warmup_epochs = max(1, int(getattr(self.config, "lr_warmup_epochs", 5)))

        for epoch in range(max_epochs):
            # P2-1: 线性 LR warmup + P3-3.1: Cosine annealing
            if epoch < warmup_epochs:
                warmup_scale = float(epoch + 1) / float(warmup_epochs)
                for pg in optimizer.param_groups:
                    pg["lr"] = pg["base_lr"] * warmup_scale
            else:
                # P3-3.1: Cosine annealing from base_lr to eta_min
                eta_min = 1e-5
                progress = float(epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
                cosine_scale = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
                for pg in optimizer.param_groups:
                    pg["lr"] = eta_min + (pg["base_lr"] - eta_min) * cosine_scale

            self.model.train()
            optimizer.zero_grad()
            output = self.model(train_batch, sample_latent=True)
            # 增强前向：得到 z_meta_aug 用于 consistency loss（与 035 对齐）
            z_meta_aug: Optional[torch.Tensor] = None
            if self.config.augmentation_enabled:
                aug_batch = _augment_batch(train_batch, self.config)
                try:
                    aug_output = self.model(aug_batch, sample_latent=True)
                    z_meta_aug = aug_output.z_meta
                except Exception:
                    z_meta_aug = None
            beta = self._beta_schedule(epoch, max_epochs) if self.config.use_vib else 0.0
            loss = compute_cmib_loss(
                output, train_batch, self.config, self.model,
                theta0_state=self.theta0_state, beta=beta, stage="full",
                z_meta_aug=z_meta_aug,
                current_epoch=epoch,  # Bug A 修复：传 current_epoch 让 anchor warmup 生效
            )
            if not torch.isfinite(loss.total):
                break
            loss.total.backward()
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.grad_clip,
                )
            optimizer.step()

            # Early stopping on val
            val_c = float("nan")
            if val_batch is not None:
                self.model.eval()
                with torch.no_grad():
                    val_output = self.model(val_batch, sample_latent=False)
                val_risk = val_output.log_risk.cpu().numpy()
                val_time = val_batch.time.cpu().numpy()
                val_event = val_batch.event.cpu().numpy().astype(int)
                val_c = harrell_c_index(val_risk, val_time, val_event)
                if np.isfinite(val_c) and val_c > best_val_c + self.config.min_delta:
                    best_val_c = val_c
                    best_state = clone_parameters(self.model)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            else:
                epochs_no_improve = 0  # no val → train full epochs

            history.append({
                "stage": stage_name,
                "epoch": epoch,
                "train_loss": float(loss.total.detach().item()),
                "val_c": val_c,
                "beta": beta,
                "consistency_loss": float(loss.cons.detach().item()),  # P2-3: 记录 cons
                "anchor_loss": float(loss.anchor.detach().item()),  # 可观测性
            })

            # P2-3: 每 5 epoch 输出 consistency loss
            if (epoch + 1) % 5 == 0:
                LOGGER.info(
                    "[%s epoch %d] train_loss=%.4f val_c=%.4f cons=%.4f anchor=%.4f lr=%.2e",
                    stage_name, epoch, float(loss.total.detach().item()),
                    val_c, float(loss.cons.detach().item()),
                    float(loss.anchor.detach().item()),
                    optimizer.param_groups[0]["lr"],
                )

            if epochs_no_improve >= self.config.inner_patience:
                break

        restore_parameters(self.model, best_state)
        return best_state, best_val_c, history

    def save_checkpoint(self, path: str) -> None:
        """保存 checkpoint。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config.to_dict(),
            "gene_registry": self.gene_registry,
            "theta0_state": self.theta0_state,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """加载 checkpoint。"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        if ckpt.get("theta0_state"):
            self.theta0_state = ckpt["theta0_state"]


def run_inner_cv(
    model_factory: Any,
    train_batch: SurvivalRecordBatch,
    config: CMIBConfig,
    n_folds: int = 3,
) -> Dict[str, Any]:
    """inner CV 用于超参选择（不使用 locked validation）。

    Args:
        model_factory: callable 返回新模型（含已加载的元初始化）
        train_batch: target train batch
        config: 配置
        n_folds: inner folds

    Returns:
        {'mean_c': float, 'std_c': float, 'fold_cs': list}
    """
    n = train_batch.batch_size
    folds = group_kfold_indices(train_batch.patient_id, n_splits=n_folds,
                                 seed=config.seed)
    fold_cs: List[float] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        # 子 batch
        from .meta_trainer import _index_batch
        tr_batch = _index_batch(train_batch, tr_idx)
        va_batch = _index_batch(train_batch, va_idx)
        if tr_batch.batch_size == 0 or va_batch.batch_size == 0:
            continue
        model = model_factory()
        trainer = TargetTrainer(model, config, [], config.device)
        trainer.train_outer_fold(tr_batch, va_batch, max_epochs=config.max_epochs // 2)
        # 评估
        model.eval()
        with torch.no_grad():
            out = model(va_batch, sample_latent=False)
        risk = out.log_risk.cpu().numpy()
        time = va_batch.time.cpu().numpy()
        event = va_batch.event.cpu().numpy().astype(int)
        c = harrell_c_index(risk, time, event)
        if not np.isnan(c):
            fold_cs.append(c)

    if not fold_cs:
        return {"mean_c": float("nan"), "std_c": float("nan"), "fold_cs": []}
    return {
        "mean_c": float(np.mean(fold_cs)),
        "std_c": float(np.std(fold_cs)),
        "fold_cs": fold_cs,
    }
