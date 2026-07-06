"""shared_utils.py — CRC 3 年总生存期预测流水线公共工具模块

消除跨脚本函数重复，提供统一的事实来源。
模块分区:
    1. PATH CONSTANTS
    2. I/O HELPERS
    3. ID HELPERS
    4. CLINICAL CODECS
    5. SURVIVAL CORE
    6. ENDPOINT ENGINEERING
    7. FEATURE ENGINEERING
    8. LOGGING
    9. CONFIGURATION
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata

try:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    HAS_ITERATIVE_IMPUTER = True
except ImportError:
    HAS_ITERATIVE_IMPUTER = False

# ──────────────────────────────────────────────────────────────────────
# 1. PATH CONSTANTS
# ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "DATA"))
RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "results"))

# 预处理数据前向目录（跨 timestamp 持久化存储）
DATA_FORWARD_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data-forword"))
ESSENTIAL_DIR = os.path.join(DATA_FORWARD_DIR, "必要")       # 下游脚本必需的预处理数据
NON_ESSENTIAL_DIR = os.path.join(DATA_FORWARD_DIR, "非必要")  # QC 图表、敏感性分析等非核心数据

FIXED_TAU_MONTHS = 36.0
RANDOM_SEED = 20260604
RANDOM_SEED_GAN = 20260604
EPS = 1e-8
DAYS_PER_MONTH = 30.44
MAX_N_EXACT_PSEUDO = 1200

META_COLUMNS = frozenset({
    "Hugo_Symbol", "Entrez_Gene_Id", "Cytoband",
    "Composite.Element.REF", "ENTITY_STABLE_ID", "NAME",
    "DESCRIPTION", "TRANSCRIPT_ID", "ID", "GENE_SYMBOL",
    "PHOSPHOSITE", "geneNames",
})

# 数值型 AJCC 分期编码（01_data_preprocessing 使用）
STAGE_MAP_NUMERIC: Dict[str, int] = {
    "STAGE I": 1, "STAGE IA": 1, "STAGE IB": 1,
    "STAGE II": 2, "STAGE IIA": 2, "STAGE IIB": 2, "STAGE IIC": 2,
    "STAGE III": 3, "STAGE IIIA": 3, "STAGE IIIB": 3, "STAGE IIIC": 3,
    "STAGE IV": 4, "STAGE IVA": 4, "STAGE IVB": 4, "STAGE IVC": 4,
    "STAGE 0": 0,
    "I": 1, "IA": 1, "IB": 1,
    "II": 2, "IIA": 2, "IIB": 2, "IIC": 2,
    "III": 3, "IIIA": 3, "IIIB": 3, "IIIC": 3,
    "IV": 4, "IVA": 4, "IVB": 4, "IVC": 4, "0": 0,
}

# 分组型 AJCC 分期编码（03_model_training 使用）
STAGE_MAP_GROUP: Dict[str, str] = {
    "STAGE I": "I", "STAGE IA": "I", "STAGE IB": "I",
    "STAGE II": "II", "STAGE IIA": "II", "STAGE IIB": "II", "STAGE IIC": "II",
    "STAGE III": "III", "STAGE IIIA": "III", "STAGE IIIB": "III", "STAGE IIIC": "III",
    "STAGE IV": "IV", "STAGE IVA": "IV", "STAGE IVB": "IV", "STAGE IVC": "IV",
    "STAGE 0": "0",
    "I": "I", "IA": "I", "IB": "I",
    "II": "II", "IIA": "II", "IIB": "II", "IIC": "II",
    "III": "III", "IIIA": "III", "IIIB": "III", "IIIC": "III",
    "IV": "IV", "IVA": "IV", "IVB": "IV", "IVC": "IV", "0": "0",
}

# 多时间点配置（默认值，可被 pipeline_config.json 覆盖）
DEFAULT_TIMEPOINTS: Dict[str, Dict[str, Any]] = {
    "1Year":     {"tau_months": 12.0, "eval_times": [6, 9, 12]},
    "2Years":    {"tau_months": 24.0, "eval_times": [12, 18, 24]},
    "3Years":    {"tau_months": 36.0, "eval_times": [12, 24, 36]},
}


# ──────────────────────────────────────────────────────────────────────
# 2. I/O HELPERS
# ──────────────────────────────────────────────────────────────────────
def ensure_dir(path: Union[str, Path]) -> Path:
    """创建目录（含递归），返回 Path 对象。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_read_tsv(path: str, **kwargs) -> pd.DataFrame:
    """安全读取 TSV 文件，默认 low_memory=False。"""
    return pd.read_csv(path, sep="\t", low_memory=False, **kwargs)


def sanitize_filename(name: str) -> str:
    """将字符串转为文件名安全形式。"""
    return re.sub(r"[^\w\-.]", "_", str(name))


# ──────────────────────────────────────────────────────────────────────
# 3. ID HELPERS
# ──────────────────────────────────────────────────────────────────────
def patient_id_from_sample(value: Any) -> str:
    """从 TCGA sample barcode 或 HTA8 标识中提取患者 ID。

    TCGA-XX-XXXX-01 → TCGA-XX-XXXX (前12字符)
    HTA8_P001_T1    → HTA8_P001
    """
    text = str(value)
    if text.startswith("TCGA-") and len(text) >= 12:
        return text[:12]
    if text.startswith("HTA8_"):
        parts = text.split("_")
        if len(parts) >= 2:
            return "_".join(parts[:2])
    return text


def normalize_patient_id(value: Any) -> str:
    """标准化患者 ID：去除空白，统一大写，TCGA 截断。"""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    if text.upper().startswith("TCGA-") and len(text) >= 12:
        return text[:12].upper()
    return re.sub(r"[\s_]+", "-", text).upper()


# ──────────────────────────────────────────────────────────────────────
# 4. CLINICAL CODECS
# ──────────────────────────────────────────────────────────────────────
def parse_survival_status(series: pd.Series) -> pd.Series:
    """解析 OS 状态字段为 0/1 浮点 Series。

    识别: dead/deceased/event/progression → 1.0
          alive/living/censored            → 0.0
    支持数值型输入。
    """
    values = series.fillna("").astype(str).str.strip()
    lower = values.str.lower()
    event = (
        values.str.startswith("1")
        | lower.isin({
            "1", "dead", "deceased", "event", "progression",
            "recurred/progressed", "1:deceased", "yes", "true",
            "recurred", "recurrence", "progressed",
        })
    )
    censored = (
        values.str.startswith("0")
        | lower.isin({
            "0", "alive", "living", "0:living", "no", "false",
            "censored", "no_progression", "diseasefree",
        })
    )
    result = pd.Series(np.nan, index=series.index)
    result[event] = 1.0
    result[censored] = 0.0
    # fallback: 尝试数值转换
    numeric = pd.to_numeric(series, errors="coerce")
    result = result.where(result.notna(), numeric)
    return result


def numeric_series(series: pd.Series) -> pd.Series:
    """强制转换为浮点 Series，处理 TCGA 特殊占位符。"""
    return pd.to_numeric(
        series.replace(
            {"[Not Available]": None, "[Not Applicable]": None,
             "[Completed]": None, "": None, "NA": None, "NaN": None},
        ),
        errors="coerce",
    )


def stage_to_group(value: Any) -> str:
    """将 AJCC 分期值映射到主分组 (I/II/III/IV/Unknown)。"""
    if pd.isna(value):
        return "Unknown"
    s = str(value).upper().replace("STAGE ", "").strip()
    if s.startswith("IV"):
        return "IV"
    if s.startswith("III"):
        return "III"
    if s.startswith("II"):
        return "II"
    if s.startswith("I"):
        return "I"
    return "Unknown"


def encode_ajcc_stage(series: pd.Series) -> pd.Series:
    """将 AJCC 分期 Series 编码为数值（0-4），使用 STAGE_MAP_NUMERIC。"""
    def _map(val):
        if pd.isna(val):
            return np.nan
        text = str(val).strip().upper()
        if text in STAGE_MAP_NUMERIC:
            return STAGE_MAP_NUMERIC[text]
        cleaned = re.sub(r"[^A-Z0-9]", "", text)
        if cleaned in STAGE_MAP_NUMERIC:
            return STAGE_MAP_NUMERIC[cleaned]
        match = re.search(r"(\d+|[IVX]+)", text)
        if match:
            token = match.group(1)
            if token in STAGE_MAP_NUMERIC:
                return STAGE_MAP_NUMERIC[token]
        return np.nan
    return series.apply(_map)


# ──────────────────────────────────────────────────────────────────────
# 5. SURVIVAL CORE
# ──────────────────────────────────────────────────────────────────────
def make_surv_array(time, event):
    """构造 sksurv Surv 结构化数组。

    Args:
        time: 生存时间数组或 Series
        event: 事件标记数组或 Series (bool/int)

    Returns:
        sksurv Surv.from_arrays 结果

    Raises:
        ImportError: 如果 sksurv 不可用
    """
    try:
        from sksurv.util import Surv
    except ImportError:
        raise ImportError("scikit-survival (sksurv) is required for make_surv_array")
    return Surv.from_arrays(
        event=np.asarray(event).astype(bool),
        time=np.asarray(time, dtype=float),
    )


def kaplan_meier_survival_at(
    times: np.ndarray,
    event_observed: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    """Kaplan-Meier 生存率估计在给定的查询时间点。

    Args:
        times: 观测时间数组
        event_observed: 事件标记数组 (1=事件, 0=删失)
        query: 需要估计生存率的时间点数组

    Returns:
        与 query 等长的生存率数组，clip 到 [EPS, 1.0]
    """
    order = np.argsort(times)
    times_sorted = np.asarray(times, dtype=float)[order]
    events_sorted = np.asarray(event_observed, dtype=int)[order]
    unique_event_times = np.unique(times_sorted[events_sorted == 1])

    surv = 1.0
    surv_times: List[float] = []
    surv_values: List[float] = []
    for t in unique_event_times:
        at_risk = np.sum(times_sorted >= t)
        d = np.sum((times_sorted == t) & (events_sorted == 1))
        if at_risk > 0:
            surv *= max(0.0, 1.0 - d / at_risk)
        surv_times.append(float(t))
        surv_values.append(float(surv))

    if not surv_times:
        return np.ones(len(query), dtype=float)

    surv_times_arr = np.asarray(surv_times)
    surv_values_arr = np.asarray(surv_values)
    idx = np.searchsorted(surv_times_arr, query, side="right") - 1
    out = np.ones(len(query), dtype=float)
    mask = idx >= 0
    out[mask] = surv_values_arr[idx[mask]]
    return np.clip(out, EPS, 1.0)


def km_cumulative_incidence_by_tau(
    times: np.ndarray,
    events: np.ndarray,
    tau: float,
) -> float:
    """tau 时刻的 Kaplan-Meier 累积发病率 (1 - S(tau))。"""
    surv = kaplan_meier_survival_at(times, events, np.asarray([tau], dtype=float))[0]
    return float(1.0 - surv)


# ──────────────────────────────────────────────────────────────────────
# 6. ENDPOINT ENGINEERING
# ──────────────────────────────────────────────────────────────────────
def detect_event_time_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """自动检测 DataFrame 中的生存时间和事件列。"""
    time_candidates = [
        "OS_TIME_MONTHS", "OS_MONTHS", "time_months",
        "TIME_MONTHS", "OS_time", "time",
    ]
    event_candidates = [
        "OS_EVENT", "event", "EVENT", "OS_event", "OS_STATUS_BINARY",
    ]
    time_col = next((c for c in time_candidates if c in df.columns), None)
    event_col = next((c for c in event_candidates if c in df.columns), None)
    if time_col is None or event_col is None:
        raise ValueError(f"Cannot detect OS time/event columns from: {list(df.columns)[:30]}")
    return time_col, event_col


def coerce_event(series: pd.Series) -> pd.Series:
    """将事件列强制转换为 int (0/1)。"""
    if series.dtype == bool:
        return series.astype(int)
    lower = series.astype(str).str.lower()
    mapped = lower.map({
        "true": 1, "false": 0, "1": 1, "0": 0,
        "1:deceased": 1, "0:living": 0, "deceased": 1, "living": 0,
        "dead": 1, "alive": 0, "event": 1, "censored": 0,
    })
    numeric = pd.to_numeric(series, errors="coerce")
    out = mapped.where(mapped.notna(), numeric)
    return out.fillna(0).astype(int)


def add_fixed_endpoint(
    df: pd.DataFrame,
    tau_months: float,
    patient_col: str = "PATIENT_ID",
) -> pd.DataFrame:
    """构造固定时间点的二分类端点。

    新增列:
        death_by_{tau}m_observed: 是否可观测到 tau 时点状态
        death_by_{tau}m: 二分类标签 (1=死亡, 0=存活, NaN=不可观测)
        early_censored_before_{tau}m: 是否在 tau 前被删失

    Args:
        df: 含生存时间和事件的 DataFrame
        tau_months: 固定时间点（月）
        patient_col: 患者 ID 列名

    Returns:
        添加了端点列的新 DataFrame
    """
    result = df.copy()
    time_col, event_col = detect_event_time_columns(result)
    result[patient_col] = result[patient_col].map(normalize_patient_id)
    result["time_months"] = numeric_series(result[time_col])
    result["event"] = coerce_event(result[event_col])
    result = result[result[patient_col].ne("") & result["time_months"].notna()].copy()
    result = result[result["time_months"] >= 0].copy()
    tau = int(tau_months)

    event_by_tau = (result["event"].eq(1) & (result["time_months"] <= tau_months)).astype(float)
    observed_status = (
        (result["event"].eq(1) & (result["time_months"] <= tau_months))
        | (result["time_months"] > tau_months)
    )
    result[f"death_by_{tau}m_observed"] = observed_status.astype(bool)
    result[f"death_by_{tau}m"] = np.where(observed_status, event_by_tau, np.nan)
    result[f"early_censored_before_{tau}m"] = (~observed_status).astype(bool)
    return result


def compute_ipcw_weights(
    endpoint: pd.DataFrame,
    tau: float,
) -> pd.DataFrame:
    """基于 Kaplan-Meier 删失分布的逆概率加权 (IPCW)。

    新增列:
        ipcw_weight_{tau}m: IPCW 权重
        ipcw_label_available: 标签是否可用

    Args:
        endpoint: 含 time_months, event, death_by_{tau}m_observed 的 DataFrame
        tau: 固定时间点（月）

    Returns:
        添加了 IPCW 权重列的新 DataFrame
    """
    df = endpoint.copy()
    tau_int = int(tau)
    censor_event = (df["event"].eq(0)).astype(int).to_numpy()
    times = df["time_months"].to_numpy(dtype=float)
    label_observed = df[f"death_by_{tau_int}m_observed"].to_numpy(dtype=bool)
    eval_time = np.minimum(times, tau)
    g_at_eval = kaplan_meier_survival_at(times, censor_event, eval_time)
    weights = np.zeros(len(df), dtype=float)
    weights[label_observed] = 1.0 / np.clip(g_at_eval[label_observed], EPS, None)
    # Winsorization: 1%/99% 分位数截断，防止极端权重导致下游模型不稳定
    positive_w = weights[weights > 0]
    if len(positive_w) >= 10:
        w_lo, w_hi = np.percentile(positive_w, [1, 99])
        weights = np.clip(weights, w_lo, w_hi)
    # 硬上限安全阀：无论 Winsorization 是否执行，都限制最大权重
    weights = np.clip(weights, None, 50.0)
    df[f"ipcw_weight_{tau_int}m"] = weights
    df["ipcw_label_available"] = label_observed
    return df


def compute_pseudo_observations(
    endpoint: pd.DataFrame,
    tau: float,
    max_n_exact: int = MAX_N_EXACT_PSEUDO,
) -> pd.DataFrame:
    """计算 tau 时点的伪观测值 (leave-one-out Jackknife)。

    新增列:
        pseudo_risk_{tau}m: 原始伪观测值
        pseudo_risk_{tau}m_clipped: clip 到 [0,1] 的伪观测值

    Args:
        endpoint: 含 time_months, event 的 DataFrame
        tau: 固定时间点（月）
        max_n_exact: 精确计算的最大样本量，超过则用全样本估计

    Returns:
        添加了伪观测值列的新 DataFrame
    """
    df = endpoint.copy()
    tau_int = int(tau)
    times = df["time_months"].to_numpy(dtype=float)
    events = df["event"].to_numpy(dtype=int)
    n = len(df)
    f_all = km_cumulative_incidence_by_tau(times, events, tau)
    pseudo = np.full(n, np.nan, dtype=float)

    if n == 0:
        df[f"pseudo_risk_{tau_int}m"] = pseudo
        return df

    if n > max_n_exact:
        pseudo[:] = f_all
    else:
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            f_minus = km_cumulative_incidence_by_tau(times[mask], events[mask], tau)
            pseudo[i] = n * f_all - (n - 1) * f_minus

    df[f"pseudo_risk_{tau_int}m"] = pseudo
    df[f"pseudo_risk_{tau_int}m_clipped"] = np.clip(pseudo, 0.0, 1.0)
    return df


# ──────────────────────────────────────────────────────────────────────
# 7. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────
def rank_inverse_normal(series: pd.Series) -> pd.Series:
    """秩次逆正态变换 (Rank Inverse Normal Transformation, RINT)。

    将连续变量变换为近似标准正态分布，减少离群值影响。

    Args:
        series: 待变换的数值 Series

    Returns:
        变换后的 Series，缺失值保持为 NaN
    """
    s = pd.to_numeric(series, errors="coerce")
    valid = s.notna()
    if valid.sum() < 2:
        return s
    ranked = rankdata(s[valid].to_numpy())
    n = len(ranked)
    transformed = stats.norm.ppf((ranked - 0.5) / n)
    result = s.copy()
    result.loc[valid] = transformed
    return result


def stratified_split_keys(
    df: pd.DataFrame,
    columns: List[str],
    min_count: int = 5,
) -> pd.Series:
    """构造分层分割键，用于保证 train/test 分割的均衡性。

    将多列拼接为单列键，稀有组合归入 "RARE" 组。

    Args:
        df: 输入 DataFrame
        columns: 用于拼接的列名列表
        min_count: 最少出现次数阈值

    Returns:
        分层键 Series
    """
    combined = df[columns[0]].astype(str)
    for col in columns[1:]:
        combined = combined + "_" + df[col].astype(str)
    counts = combined.value_counts()
    rare = set(counts[counts < min_count].index)
    if rare:
        combined = combined.replace(list(rare), "RARE")
    return combined


# ──────────────────────────────────────────────────────────────────────
# 10. IMPUTATION UTILITIES
# ──────────────────────────────────────────────────────────────────────

# 临床变量前缀：用于 TieredImputer 自动识别低维临床列
CLINICAL_FEATURE_PREFIXES = (
    "AGE", "SEX", "GENDER", "AJCC", "STAGE", "TUMOR", "T_",
    "LYMPH", "N_", "METAST", "M_", "GRADE", "HISTOLOGY",
    "RACE", "ETHNICITY", "BMI", "MSI", "CIMP", "KRAS", "BRAF", "NRAS",
    "TMB", "TUMOR_MUTATIONAL", "SURVIVAL_GAN", "GAN_SYNTHETIC",
    "rna_embedding", "mutation_embedding", "cnv_embedding",
    "methylation_embedding", "rppa_embedding",
    "clinical_", "CNV_",
)


def is_clinical_feature(col: str) -> bool:
    """判断列名是否为低维临床/非基因组特征。"""
    return any(col.startswith(p) for p in CLINICAL_FEATURE_PREFIXES)


class TieredImputer:
    """分层插补器：临床变量用 missForest/MICE，基因表达用中位数。

    设计原则：
    - fit() 仅在训练集上调用，杜绝数据泄漏
    - 低维临床变量（<50列）使用 IterativeImputer (missForest/MICE)
    - 高维基因组学特征使用中位数插补（计算高效、p>>n 安全）
    - 兼容 sklearn Pipeline 接口 (fit/transform/fit_transform)
    - 支持 joblib 序列化

    Parameters
    ----------
    clinical_cols : list[str] or None
        低维临床变量列名。若为 None，则通过 is_clinical_feature() 自动识别。
    method : str
        临床变量插补方法: 'missforest' | 'mice' | 'median'
    n_estimators : int
        missForest 随机森林的树数量（默认 50）
    max_iter : int
        IterativeImputer 最大迭代轮数（默认 10）
    random_state : int
        随机种子
    """

    def __init__(
        self,
        clinical_cols: Optional[List[str]] = None,
        method: str = "missforest",
        n_estimators: int = 50,
        max_iter: int = 10,
        random_state: int = 42,
    ):
        self.clinical_cols = clinical_cols
        self.method = method
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.random_state = random_state
        # 内部状态（fit 后填充）
        self._clinical_imputer = None
        self._genomic_imputer = None
        self._clin_cols_fit: List[str] = []
        self._genomic_cols_fit: List[str] = []

    def _resolve_clinical_cols(self, columns) -> List[str]:
        """解析临床列：优先使用显式指定，否则自动识别。"""
        if self.clinical_cols is not None:
            return [c for c in self.clinical_cols if c in columns]
        return [c for c in columns if is_clinical_feature(c)]

    def _build_clinical_imputer(self):
        """根据 method 参数构建临床变量插补器。"""
        if not HAS_ITERATIVE_IMPUTER or self.method == "median":
            from sklearn.impute import SimpleImputer
            return SimpleImputer(strategy="median")

        if self.method == "missforest":
            from sklearn.ensemble import RandomForestRegressor
            return IterativeImputer(
                estimator=RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_features="sqrt",
                    random_state=self.random_state,
                    n_jobs=-1,
                ),
                max_iter=self.max_iter,
                random_state=self.random_state,
                sample_posterior=False,
            )
        elif self.method == "mice":
            return IterativeImputer(
                max_iter=self.max_iter,
                random_state=self.random_state,
                sample_posterior=True,
            )
        else:
            from sklearn.impute import SimpleImputer
            return SimpleImputer(strategy="median")

    def fit(self, X_train: pd.DataFrame) -> "TieredImputer":
        """仅在训练集上拟合插补器（防数据泄漏的关键）。"""
        clin = self._resolve_clinical_cols(X_train.columns)
        genomic = [c for c in X_train.columns if c not in clin]

        if clin:
            self._clinical_imputer = self._build_clinical_imputer()
            self._clinical_imputer.fit(X_train[clin])
        self._clin_cols_fit = clin

        from sklearn.impute import SimpleImputer
        self._genomic_imputer = SimpleImputer(strategy="median")
        if genomic:
            self._genomic_imputer.fit(X_train[genomic])
        self._genomic_cols_fit = genomic

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """应用已拟合的插补器（不修改拟合参数）。"""
        result = X.copy()

        if self._clinical_imputer is not None and self._clin_cols_fit:
            cols = [c for c in self._clin_cols_fit if c in X.columns]
            if cols:
                result[cols] = self._clinical_imputer.transform(X[cols])

        if self._genomic_imputer is not None and self._genomic_cols_fit:
            cols = [c for c in self._genomic_cols_fit if c in X.columns]
            if cols:
                result[cols] = self._genomic_imputer.transform(X[cols])

        return result

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """拟合并变换（仅用于训练集）。"""
        return self.fit(X_train).transform(X_train)

    def get_params(self, deep=True):
        """sklearn 兼容接口。"""
        return {
            "clinical_cols": self.clinical_cols,
            "method": self.method,
            "n_estimators": self.n_estimators,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        """sklearn 兼容接口。"""
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ──────────────────────────────────────────────────────────────────────
# 8. LOGGING
# ──────────────────────────────────────────────────────────────────────
def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.DEBUG,
) -> logging.Logger:
    """配置并返回 Logger 实例。

    - stdout handler: INFO 级别
    - file handler: DEBUG 级别（仅当 log_dir 非空时）

    Args:
        name: Logger 名称（通常用 __name__）
        log_dir: 日志文件目录，None 则仅输出到 stdout
        level: Logger 根级别

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 避免重复添加 handler
    logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # file handler
    if log_dir is not None:
        ensure_dir(log_dir)
        fh = logging.FileHandler(
            os.path.join(log_dir, f"{name}.log"), encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# ──────────────────────────────────────────────────────────────────────
# 9. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
_DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "pipeline_config.json")


def load_pipeline_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载 pipeline 配置 JSON 文件。

    如果文件不存在，返回包含默认值的字典。
    配置可覆盖 shared_utils 中的默认常量。

    Args:
        config_path: JSON 配置文件路径，默认查找 SCRIPT_DIR/pipeline_config.json

    Returns:
        配置字典
    """
    path = config_path or _DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        return _default_config()
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # 合并默认值（配置中未指定的键使用默认值）
    defaults = _default_config()
    for key, default_val in defaults.items():
        if key not in cfg:
            cfg[key] = default_val
    return cfg


def _default_config() -> Dict[str, Any]:
    """返回默认 pipeline 配置。"""
    return {
        "random_seed": RANDOM_SEED,
        "fixed_tau_months": FIXED_TAU_MONTHS,
        "timepoints": DEFAULT_TIMEPOINTS,
        "gan": {
            "enabled": True,
            "latent_dim": 32,
            "hidden_dim": 256,
            "dropout": 0.1,
            "epochs": 300,
            "batch_size": 32,
            "n_critic": 5,
            "lambda_gp": 10.0,
            "lr": 2e-4,
            "patience": 30,
            "weight_decay": 1e-5,
            "augmentation_ratios": [0.25, 0.5, 1.0, 2.0],
            "sampling_strategies": [
                "balanced_event", "risk_stratified", "event_only", "overall",
            ],
        },
        "feature_engineering": {
            "rint_enabled": True,
            "pathway_activity": True,
            "msigdb_collections": ["h.all", "c2.cp.kegg_legacy", "c5.go.bp"],
        },
        "model_selection": {
            "cv_folds": 5,
            "primary_metric": "uno_c_index",
            "clinical_anchor_weight": 0.4,
        },
        "imputation": {
            "method": "missforest",
            "clinical_n_estimators": 50,
            "clinical_max_iter": 10,
            "genomic_strategy": "median",
            "min_nonmissing": 0.7,
        },
    }


def get_config_value(cfg: Dict, key: str, default: Any = None) -> Any:
    """安全获取嵌套配置值，支持点分隔路径。

    Args:
        cfg: 配置字典
        key: 点分隔的键路径，如 "gan.epochs"
        default: 键不存在时的默认值

    Returns:
        配置值或默认值
    """
    parts = key.split(".")
    current = cfg
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current
