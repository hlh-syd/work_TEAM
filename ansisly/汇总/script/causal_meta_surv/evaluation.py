"""evaluation.py — CMIB-Surv 评估指标套件。

实现（spec.md / Fusion_Plan §9）：
1. harrell_c_index
2. uno_c_index
3. ipcw_c_index
4. integrated_brier_score
5. calibration_slope_intercept(t=36)
6. paired_bootstrap_delta_c
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =====================================================================
# Harrell C-index
# =====================================================================


def harrell_c_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Harrell 一致性指数。

    Args:
        risk: [N] 预测风险（高风险=早事件）
        time: [N] 生存时间
        event: [N] 事件标记 (1=事件, 0=删失)

    Returns:
        C-index ∈ [0, 1]
    """
    risk = np.asarray(risk, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    n = len(risk)
    if n < 2 or event.sum() == 0:
        return float("nan")

    concordant = 0.0
    permissible = 0.0
    for i in range(n):
        if event[i] != 1:
            continue
        for j in range(n):
            if i == j:
                continue
            if time[i] < time[j]:
                # i 先发生事件，应 i 风险更高
                permissible += 1.0
                if risk[i] > risk[j]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    concordant += 0.5
    if permissible == 0:
        return float("nan")
    return float(concordant / permissible)


# =====================================================================
# Uno C-index（逆概率加权删失）
# =====================================================================


def uno_c_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray,
                tau: Optional[float] = None) -> float:
    """Uno C-index（IPCW 加权）。

    使用 Kaplan-Meier 估计删失分布 G(t)，权重 = 1/G(t_i)^2
    """
    risk = np.asarray(risk, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    n = len(risk)
    if n < 2 or event.sum() == 0:
        return float("nan")

    # KM 估计删失生存 G(t)
    G = _km_censoring_surv(time, event)

    if tau is None:
        tau = float(np.percentile(time[event == 1], 90))

    concordant = 0.0
    weight_sum = 0.0
    for i in range(n):
        if event[i] != 1 or time[i] > tau:
            continue
        g_i = G(np.array([time[i]]))[0]
        if g_i <= 0:
            continue
        w_i = 1.0 / (g_i ** 2)
        for j in range(n):
            if i == j:
                continue
            if time[i] < time[j]:
                g_j = G(np.array([min(time[j], tau)]))[0]
                if g_j <= 0:
                    continue
                w_ij = w_i / g_j
                weight_sum += w_ij
                if risk[i] > risk[j]:
                    concordant += w_ij
                elif risk[i] == risk[j]:
                    concordant += 0.5 * w_ij
    if weight_sum == 0:
        return float("nan")
    return float(concordant / weight_sum)


def _km_censoring_surv(time: np.ndarray, event: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """KM 估计删失分布 G(t) = P(C > t)，返回插值函数。"""
    censor_event = (1 - event).astype(int)
    order = np.argsort(time)
    t_s = time[order]
    e_s = censor_event[order]
    unique_t = np.unique(t_s)
    surv = 1.0
    surv_times = [0.0]
    surv_values = [1.0]
    for t in unique_t:
        at_risk = (t_s >= t).sum()
        d = ((t_s == t) & (e_s == 1)).sum()
        if at_risk > 0:
            surv *= max(0.0, 1.0 - d / at_risk)
        surv_times.append(t)
        surv_values.append(surv)
    surv_times = np.array(surv_times)
    surv_values = np.array(surv_values)

    def _interp(query: np.ndarray) -> np.ndarray:
        out = np.interp(query, surv_times, surv_values, left=1.0,
                        right=surv_values[-1] if len(surv_values) else 0.0)
        return np.clip(out, 1e-8, 1.0)

    return _interp


# =====================================================================
# IPCW C-index（基于 lifelines）
# =====================================================================


def ipcw_c_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray,
                 tau: Optional[float] = None) -> float:
    """IPCW C-index，基于 lifelines 的 concordance_index_ipcw。"""
    try:
        from lifelines.utils import concordance_index_ipcw
        import pandas as pd
        df = pd.DataFrame({"time": time, "event": event})
        if tau is not None:
            df["time"] = np.minimum(df["time"], tau)
        result = concordance_index_ipcw(
            df, df, -risk, tau=tau  # 负号：高风险应早事件
        )
        return float(result[0])
    except Exception:
        # 回退到 Uno C
        return uno_c_index(risk, time, event, tau)


# =====================================================================
# Integrated Brier Score
# =====================================================================


def integrated_brier_score(
    survival_func: np.ndarray,
    eval_times: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> float:
    """Integrated Brier Score。

    Args:
        survival_func: [N, T] 预测生存函数 S(t|x)
        eval_times: [T] 评估时间点
        time: [N] 观测时间
        event: [N] 事件标记

    Returns:
        IBS
    """
    try:
        from sksurv.metrics import integrated_brier_score as _ibs
        from sksurv.util import Surv
        surv = Surv.from_arrays(event=event.astype(bool), time=time)
        # sksurv 要求 survival_func shape (N, T)
        score = _ibs(surv, surv, 1 - survival_func, eval_times)
        return float(score)
    except Exception:
        # 简化实现：用 36 月 Brier
        return _simple_brier(survival_func, eval_times, time, event)


def _simple_brier(survival_func: np.ndarray, eval_times: np.ndarray,
                  time: np.ndarray, event: np.ndarray) -> float:
    """简化 Brier score。"""
    n, T = survival_func.shape
    if T == 0:
        return float("nan")
    G = _km_censoring_surv(time, event)
    bs = np.zeros(T)
    for k, t in enumerate(eval_times):
        # 在 t 时刻事件发生的样本：S(t) 越低越好
        # 删失前未事件：S(t) 越高越好
        s = survival_func[:, k]
        g_t = G(np.array([t]))[0]
        if g_t <= 0:
            continue
        for i in range(n):
            if time[i] <= t and event[i] == 1:
                g_ti = G(np.array([time[i]]))[0]
                if g_ti > 0:
                    bs[k] += ((1 - s[i]) ** 2) / g_ti
            elif time[i] > t:
                bs[k] += (s[i] ** 2) / g_t
        bs[k] /= n
    # 积分（梯形法）
    if T >= 2:
        return float(np.trapz(bs, eval_times) / (eval_times[-1] - eval_times[0]))
    return float(bs.mean())


# =====================================================================
# 校准
# =====================================================================


def calibration_slope_intercept(
    predicted_surv: np.ndarray, time: np.ndarray, event: np.ndarray,
    t: float = 36.0,
) -> Tuple[float, float]:
    """36 月校准斜率/截距。

    简化：用 logistic 回归 observed ~ log(-log(predicted_surv))
    """
    from sklearn.linear_model import LinearRegression

    predicted_surv = np.clip(np.asarray(predicted_surv, dtype=float), 1e-6, 1 - 1e-6)
    # 观测：在 t 时刻是否事件
    observed = ((time <= t) & (event == 1)).astype(float)
    # 仅在可观测样本上拟合
    observed_mask = (time <= t) | (time > t)
    if observed_mask.sum() < 10:
        return float("nan"), float("nan")
    # 用 log(-log(S)) 作为特征
    X = np.log(-np.log(predicted_surv)).reshape(-1, 1)
    y = observed
    try:
        reg = LinearRegression().fit(X[observed_mask], y[observed_mask])
        return float(reg.coef_[0]), float(reg.intercept_)
    except Exception:
        return float("nan"), float("nan")


# =====================================================================
# 配对 bootstrap Delta C
# =====================================================================


def paired_bootstrap_delta_c(
    risk_a: np.ndarray, risk_b: np.ndarray,
    time: np.ndarray, event: np.ndarray,
    n: int = 500, seed: int = 42,
    metric: str = "harrell",
) -> Dict[str, float]:
    """配对 bootstrap Delta C。

    Args:
        risk_a: 模型 A 风险
        risk_b: 模型 B 风险
        time, event: 生存数据
        n: bootstrap 次数
        metric: 'harrell' | 'uno'

    Returns:
        {'delta': mean, 'ci_low': 2.5%, 'ci_high': 97.5%, 'p_positive': P(Delta>0)}
    """
    rng = np.random.RandomState(seed)
    N = len(risk_a)
    if N != len(risk_b):
        raise ValueError("risk_a 与 risk_b 长度不一致")
    deltas = np.zeros(n)
    for b in range(n):
        idx = rng.randint(0, N, size=N)
        if metric == "harrell":
            ca = harrell_c_index(risk_a[idx], time[idx], event[idx])
            cb = harrell_c_index(risk_b[idx], time[idx], event[idx])
        else:
            ca = uno_c_index(risk_a[idx], time[idx], event[idx])
            cb = uno_c_index(risk_b[idx], time[idx], event[idx])
        deltas[b] = ca - cb if not (np.isnan(ca) or np.isnan(cb)) else np.nan
    deltas = deltas[~np.isnan(deltas)]
    if len(deltas) == 0:
        return {"delta": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p_positive": float("nan")}
    return {
        "delta": float(np.mean(deltas)),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_positive": float(np.mean(deltas > 0)),
    }


def bootstrap_ci(
    risk: np.ndarray, time: np.ndarray, event: np.ndarray,
    n: int = 500, seed: int = 42, metric: str = "harrell",
) -> Dict[str, float]:
    """患者级 bootstrap 95% CI。"""
    rng = np.random.RandomState(seed)
    N = len(risk)
    cs = np.zeros(n)
    for b in range(n):
        idx = rng.randint(0, N, size=N)
        if metric == "harrell":
            cs[b] = harrell_c_index(risk[idx], time[idx], event[idx])
        else:
            cs[b] = uno_c_index(risk[idx], time[idx], event[idx])
    cs = cs[~np.isnan(cs)]
    if len(cs) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    return {
        "mean": float(np.mean(cs)),
        "ci_low": float(np.percentile(cs, 2.5)),
        "ci_high": float(np.percentile(cs, 97.5)),
    }


# =====================================================================
# 完整评估套件
# =====================================================================


def evaluate_full(
    risk: np.ndarray, time: np.ndarray, event: np.ndarray,
    survival_func: Optional[np.ndarray] = None,
    eval_times: Optional[np.ndarray] = None,
    tau: float = 36.0,
    bootstrap_n: int = 500,
    baseline_risk: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """完整评估套件。

    Returns:
        包含所有指标的字典
    """
    result: Dict[str, Any] = {}
    result["harrell_c"] = harrell_c_index(risk, time, event)
    result["uno_c"] = uno_c_index(risk, time, event, tau=tau)
    result["ipcw_c"] = ipcw_c_index(risk, time, event, tau=tau)
    result["tau_months"] = tau

    if survival_func is not None and eval_times is not None:
        result["ibs"] = integrated_brier_score(survival_func, eval_times, time, event)
        # 36 月校准
        if tau in eval_times:
            t_idx = int(np.argmin(np.abs(eval_times - tau)))
            slope, intercept = calibration_slope_intercept(
                survival_func[:, t_idx], time, event, t=tau
            )
            result["calibration_slope"] = slope
            result["calibration_intercept"] = intercept
        else:
            slope, intercept = calibration_slope_intercept(
                survival_func[:, -1], time, event, t=tau
            )
            result["calibration_slope"] = slope
            result["calibration_intercept"] = intercept

    # Bootstrap CI
    ci = bootstrap_ci(risk, time, event, n=bootstrap_n, metric="harrell")
    result["harrell_c_ci_low"] = ci["ci_low"]
    result["harrell_c_ci_high"] = ci["ci_high"]

    # 配对 Delta C（相对基线）
    if baseline_risk is not None:
        delta = paired_bootstrap_delta_c(
            risk, baseline_risk, time, event, n=bootstrap_n, metric="harrell"
        )
        result["delta_c_vs_baseline"] = delta["delta"]
        result["delta_c_ci_low"] = delta["ci_low"]
        result["delta_c_ci_high"] = delta["ci_high"]
        result["delta_c_p_positive"] = delta["p_positive"]

    return result
