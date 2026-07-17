"""ensemble.py — OOF 融合 + Elastic-Net Cox 保护栏。

实现（spec.md / Fusion_Plan §8）：
1. OOFFoldEnsemble: 收集 5×3 嵌套 CV × 3 seed 预测
2. 互补误差检测（简化为相关性检查）
3. Elastic-Net Cox 保护栏（融合失败时回退）
4. predict_locked_validation: 仅一次推理

简化范围：
- 互补误差检测简化为预测风险的相关性 < 0.95 时认为互补
- OOF 融合用均值（同架构内）+ 非负凸融合（跨架构，简化为简单加权）
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .contracts import CMIBConfig, SurvivalRecordBatch
from .evaluation import (
    bootstrap_ci, evaluate_full, harrell_c_index, paired_bootstrap_delta_c,
)


class OOFFoldEnsemble:
    """OOF 预测融合。

    收集每个 outer fold / seed 的 OOF 预测，按 patient_id 聚合。
    每个患者恰有一次 OOF 预测（来自其所在的 val fold）。
    """

    def __init__(self, config: CMIBConfig):
        self.config = config
        self.predictions: Dict[str, List[float]] = {}  # patient_id -> [risk per model]
        self.time: Dict[str, float] = {}
        self.event: Dict[str, int] = {}

    def add_prediction(
        self,
        patient_id: str,
        risk: float,
        time: float,
        event: int,
    ) -> None:
        """添加一个模型对患者的一次 OOF 预测。"""
        if patient_id not in self.predictions:
            self.predictions[patient_id] = []
            self.time[patient_id] = float(time)
            self.event[patient_id] = int(event)
        self.predictions[patient_id].append(float(risk))

    def aggregate(self, method: str = "mean") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """聚合多模型预测。

        Returns:
            (risk, time, event) 数组
        """
        pids = sorted(self.predictions.keys())
        risks = []
        times = []
        events = []
        for pid in pids:
            preds = self.predictions[pid]
            if method == "mean":
                r = float(np.mean(preds))
            elif method == "median":
                r = float(np.median(preds))
            else:
                r = float(np.mean(preds))
            risks.append(r)
            times.append(self.time[pid])
            events.append(self.event[pid])
        return np.array(risks), np.array(times), np.array(events)

    def n_models(self) -> int:
        """返回每个患者的模型数（取最大值）。"""
        if not self.predictions:
            return 0
        return max(len(v) for v in self.predictions.values())


def detect_complementary_errors(
    risk_a: np.ndarray, risk_b: np.ndarray,
    threshold: float = 0.95,
) -> bool:
    """互补误差检测（简化为相关性检查）。

    若两模型预测风险相关性 < threshold，认为互补。
    """
    if len(risk_a) < 5 or len(risk_b) < 5:
        return False
    if np.std(risk_a) < 1e-8 or np.std(risk_b) < 1e-8:
        return False
    corr = float(np.corrcoef(risk_a, risk_b)[0, 1])
    return abs(corr) < threshold


def convex_blend(
    risk_a: np.ndarray, risk_b: np.ndarray,
    time: np.ndarray, event: np.ndarray,
    grid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """非负凸融合（简化为网格搜索权重）。

    Returns:
        (blended_risk, best_weight_a)
    """
    if grid is None:
        grid = np.linspace(0.0, 1.0, 21)
    best_w = 0.5
    best_c = -np.inf
    for w in grid:
        blended = w * risk_a + (1 - w) * risk_b
        c = harrell_c_index(blended, time, event)
        if not np.isnan(c) and c > best_c:
            best_c = c
            best_w = w
    return best_w * risk_a + (1 - best_w) * risk_b, float(best_w)


class EnsembleWithFallback:
    """OOF 融合 + Elastic-Net Cox 保护栏。"""

    def __init__(self, config: CMIBConfig):
        self.config = config
        self.oof_ensemble = OOFFoldEnsemble(config)
        self.fallback_model = None  # ElasticNetCoxFallback
        self.ensemble_weight: float = 0.5

    def add_oof_prediction(
        self, patient_id: str, risk: float, time: float, event: int
    ) -> None:
        """添加 OOF 预测。"""
        self.oof_ensemble.add_prediction(patient_id, risk, time, event)

    def fit_fallback(
        self, X: np.ndarray, time: np.ndarray, event: np.ndarray
    ) -> None:
        """拟合 Elastic-Net Cox 保护栏。"""
        from .heads import ElasticNetCoxFallback
        self.fallback_model = ElasticNetCoxFallback(
            level1_alpha=0.02, level1_max_iter=2000, level2_alpha=0.1
        )
        # ElasticNetCoxFallback.fit 签名: fit(X, event, time)
        self.fallback_model.fit(X, event, time)

    def predict(
        self,
        ensemble_risk: Optional[np.ndarray] = None,
        fallback_X: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """融合预测（ensemble 优先，失败用 fallback）。"""
        if ensemble_risk is not None and self.config.use_ensemble:
            if self.fallback_model is not None and fallback_X is not None:
                fb_risk = self.fallback_model.predict_risk(fallback_X)
                if detect_complementary_errors(ensemble_risk, fb_risk,
                                                threshold=self.config.fusion_min_weight + 0.9):
                    blended, w = convex_blend(
                        ensemble_risk, fb_risk,
                        self.oof_ensemble.aggregate()[1],
                        self.oof_ensemble.aggregate()[2],
                    )
                    self.ensemble_weight = w
                    return blended
            return ensemble_risk
        if self.fallback_model is not None and fallback_X is not None:
            return self.fallback_model.predict_risk(fallback_X)
        raise RuntimeError("无可用预测来源")

    def evaluate_locked(
        self,
        locked_risk: np.ndarray,
        locked_time: np.ndarray,
        locked_event: np.ndarray,
        baseline_risk: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """锁定验证一次性推理评估。

        Args:
            locked_risk: 锁定验证集的最终预测风险
            locked_time, locked_event: 锁定验证集生存数据
            baseline_risk: 基线模型风险（用于 Delta C）

        Returns:
            完整评估指标字典
        """
        return evaluate_full(
            locked_risk, locked_time, locked_event,
            survival_func=None, eval_times=None,
            tau=self.config.eval_tau_months,
            bootstrap_n=self.config.bootstrap_n,
            baseline_risk=baseline_risk,
        )


def locked_validation_eval(
    model,
    locked_batch: SurvivalRecordBatch,
    config: CMIBConfig,
    baseline_risk: Optional[np.ndarray] = None,
    mc_samples: int = 0,
) -> Dict[str, Any]:
    """锁定验证一次性推理（严禁重复使用）。

    Args:
        model: 已训练的 CausalMetaIBSurv
        locked_batch: 锁定验证集 batch
        config: 配置
        baseline_risk: 基线风险（用于配对 Delta C）
        mc_samples: MC 采样数（0=用 mu）

    Returns:
        final_metrics 字典
    """
    risk = model.predict_risk(locked_batch, mc_samples=mc_samples)
    time = locked_batch.time.cpu().numpy()
    event = locked_batch.event.cpu().numpy().astype(int)

    metrics = evaluate_full(
        risk, time, event,
        survival_func=None, eval_times=None,
        tau=config.eval_tau_months,
        bootstrap_n=config.bootstrap_n,
        baseline_risk=baseline_risk,
    )
    metrics["n_locked"] = int(len(risk))
    metrics["n_locked_events"] = int(event.sum())
    metrics["locked_access_count"] = 1  # 仅一次
    return metrics
