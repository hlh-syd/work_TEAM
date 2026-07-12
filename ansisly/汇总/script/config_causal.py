"""因果模块配置 — CausalEGM + DeepSurv（OmegaConf 结构化配置）

本模块集中管理 CausalEGM 降维模块和 DeepSurv 生存预测头的配置参数。
通过修改 enabled 开关即可控制模块启用/禁用，无需改动主流程代码。

使用方式::

    from config_causal import get_causal_config, get_deepsurv_config
    cegm_cfg = get_causal_config()
    dsurv_cfg = get_deepsurv_config()

参考文献:
    - CausalEGM: Liu et al., Nature Computational Science, 2024
    - DeepSurv: Katzman et al., BMC Medical Research Methodology, 2018
"""

from dataclasses import dataclass, field
from typing import List

from omegaconf import OmegaConf, DictConfig


# ── 数据类定义 ──────────────────────────────────────────────────────────

@dataclass
class CausalEGMConfig:
    """CausalEGM 编码-生成模型配置"""
    enabled: bool = True               # 总开关：False 则完全跳过 CausalEGM 降维
    latent_dim: int = 15               # 潜在空间维度（推荐 10-20，小样本用 10-15）
    epochs: int = 200                  # 最大训练轮数
    learning_rate: float = 1e-3        # 学习率
    batch_size: int = 64              # 批大小（小样本建议 32-64）
    flow_steps: int = 3               # Normalizing Flow 层数（越多表达力越强但越慢）
    fallback: bool = True             # 训练失败时是否回退到原始特征
    mode: str = "latent_feature"      # 工作模式: "latent_feature"(推荐) | "rmst_ate"


@dataclass
class DeepSurvConfig:
    """DeepSurv 生存预测头配置"""
    enabled: bool = True               # 总开关：False 则完全跳过 DeepSurv
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])  # MLP 隐藏层维度
    dropout: float = 0.4              # Dropout 率（小样本需强正则化）
    l2_reg: float = 1e-3              # L2 权重衰减系数
    learning_rate: float = 1e-4       # 学习率（较小值更稳定，Katzman 2018 推荐 1e-5~1e-4）
    epochs: int = 500                 # 最大训练轮数
    patience: int = 50               # Early stopping patience（连续 N epoch 无改善则停止）
    batch_size: int = 32             # Mini-batch 大小
    use_cegm_z: bool = True          # 是否优先使用 CausalEGM 的 Z 作为输入


@dataclass
class CausalModuleConfig:
    """因果模块顶层配置"""
    causal_egm: CausalEGMConfig = field(default_factory=CausalEGMConfig)
    deepsurv: DeepSurvConfig = field(default_factory=DeepSurvConfig)


# ── 工厂函数 ────────────────────────────────────────────────────────────

def get_causal_config() -> DictConfig:
    """获取 CausalEGM 配置（DictConfig）"""
    return OmegaConf.structured(CausalEGMConfig())


def get_deepsurv_config() -> DictConfig:
    """获取 DeepSurv 配置（DictConfig）"""
    return OmegaConf.structured(DeepSurvConfig())


def get_full_causal_config() -> DictConfig:
    """获取完整因果模块配置（DictConfig）"""
    return OmegaConf.structured(CausalModuleConfig())


# ════════════════════════════════════════════════════════════════════════
# Flat 常量（供后续模块直接 import 使用）
# ════════════════════════════════════════════════════════════════════════

# ── CausalEGM 配置 ─────────────────────────────────────────────────────
CAUSAL_EGM_ENABLED = True          # 总开关：False 则完全跳过 CausalEGM 降维
CAUSAL_EGM_LATENT_DIM = 15         # 潜在空间维度（推荐 10-20，小样本用 10-15）
CAUSAL_EGM_EPOCHS = 200            # 最大训练轮数
CAUSAL_EGM_LR = 1e-3               # 学习率
CAUSAL_EGM_BATCH_SIZE = 64         # 批大小（小样本建议 32-64）
CAUSAL_EGM_FLOW_STEPS = 3          # Normalizing Flow 层数（越多表达力越强但越慢）
CAUSAL_EGM_FALLBACK = True         # 训练失败时是否回退到原始特征
CAUSAL_EGM_MODE = "latent_feature" # 工作模式: "latent_feature"(推荐) | "rmst_ate"

# ── DeepSurv 配置 ──────────────────────────────────────────────────────
DEEPSURV_ENABLED = True            # 总开关：False 则完全跳过 DeepSurv
DEEPSURV_HIDDEN_DIMS = [64, 32]    # MLP 隐藏层维度
DEEPSURV_DROPOUT = 0.4             # Dropout 率（小样本需强正则化）
DEEPSURV_L2_REG = 1e-3             # L2 权重衰减系数
DEEPSURV_LR = 1e-4                 # 学习率（较小值更稳定，Katzman 2018 推荐 1e-5~1e-4）
DEEPSURV_EPOCHS = 500              # 最大训练轮数
DEEPSURV_PATIENCE = 50             # Early stopping patience（连续 N epoch 无改善则停止）
DEEPSURV_BATCH_SIZE = 32           # Mini-batch 大小
DEEPSURV_USE_CEGM_Z = True         # 是否优先使用 CausalEGM 的 Z 作为 DeepSurv 输入
