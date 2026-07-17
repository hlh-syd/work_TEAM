"""serialization.py — 模型/预处理持久化 + hash 校验。

提供：
    - save_model_state(model, path, preprocessor_hash)：保存模型 state_dict + preprocessor hash
    - load_model_state(path, expected_preprocessor_hash)：加载并校验 hash
    - save_preprocessor(preprocessor, path)：保存预处理器
    - load_preprocessor(path)：加载预处理器
    - save_audit_report(audit, path)：保存审计报告 JSON
    - load_json(path) / save_json(obj, path)：通用 JSON I/O
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .contracts import CMIBConfig, hash_state_dict

LOGGER = logging.getLogger("cmib_surv.serialization")


# ──────────────────────────────────────────────────────────────────────
# 通用 JSON I/O
# ──────────────────────────────────────────────────────────────────────
def save_json(obj: Any, path: str | Path) -> None:
    """保存对象为 JSON 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    LOGGER.debug("Saved JSON: %s", path)


def load_json(path: str | Path) -> Any:
    """加载 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────
# 模型持久化
# ──────────────────────────────────────────────────────────────────────
def save_model_state(
    model: torch.nn.Module,
    path: str | Path,
    preprocessor_hash: Optional[str] = None,
    config: Optional[CMIBConfig] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """保存模型 state_dict + preprocessor hash + 配置。

    Args:
        model: PyTorch 模型
        path: 保存路径（.pt 或 .pth）
        preprocessor_hash: 预处理器的 state_hash（用于加载时校验无泄漏）
        config: 运行时配置
        extra: 额外元数据
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
        "preprocessor_hash": preprocessor_hash,
        "config": config.to_dict() if config is not None else None,
        "extra": extra or {},
    }
    torch.save(payload, str(path))
    LOGGER.info("Saved model state: %s (preprocessor_hash=%s)", path, preprocessor_hash)


def load_model_state(
    path: str | Path,
    expected_preprocessor_hash: Optional[str] = None,
    map_location: str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], Optional[str], Dict[str, Any]]:
    """加载模型 state_dict 并校验 preprocessor hash。

    Args:
        path: 模型文件路径
        expected_preprocessor_hash: 期望的 preprocessor hash（None 跳过校验）
        map_location: 设备映射

    Returns:
        (state_dict, preprocessor_hash, extra)

    Raises:
        ValueError: preprocessor hash 不匹配（可能存在泄漏或版本不一致）
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model state not found: {path}")

    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    state_dict = payload["model_state_dict"]
    stored_hash = payload.get("preprocessor_hash")
    extra = payload.get("extra", {})

    if expected_preprocessor_hash is not None and stored_hash != expected_preprocessor_hash:
        raise ValueError(
            f"Preprocessor hash mismatch: expected {expected_preprocessor_hash}, "
            f"got {stored_hash}. Model may be incompatible with current preprocessor "
            f"(potential leakage or version mismatch)."
        )

    LOGGER.info("Loaded model state: %s (preprocessor_hash=%s)", path, stored_hash)
    return state_dict, stored_hash, extra


# ──────────────────────────────────────────────────────────────────────
# 预处理器持久化
# ──────────────────────────────────────────────────────────────────────
def save_preprocessor(preprocessor, path: str | Path) -> str:
    """保存预处理器（含 state_dict + hash）。

    Args:
        preprocessor: FoldPreprocessorImpl 实例
        path: 保存路径（.pkl）

    Returns:
        state_hash: 预处理器的 state_hash
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = preprocessor.state_dict()
    state_hash = hash_state_dict(state)

    payload = {
        "state": state,
        "state_hash": state_hash,
        "class": preprocessor.__class__.__name__,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    LOGGER.info("Saved preprocessor: %s (hash=%s)", path, state_hash)
    return state_hash


def load_preprocessor(path: str | Path) -> Tuple[Dict[str, Any], str]:
    """加载预处理器 state_dict + hash。

    Args:
        path: 预处理器文件路径

    Returns:
        (state_dict, state_hash)

    Raises:
        ValueError: 存储的 hash 与 state_dict 重新计算的 hash 不一致
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {path}")

    with open(path, "rb") as f:
        payload = pickle.load(f)

    state = payload["state"]
    stored_hash = payload.get("state_hash", "")
    recomputed_hash = hash_state_dict(state)

    if stored_hash and stored_hash != recomputed_hash:
        raise ValueError(
            f"Preprocessor hash corruption: stored {stored_hash} != "
            f"recomputed {recomputed_hash}"
        )

    LOGGER.info("Loaded preprocessor: %s (hash=%s)", path, recomputed_hash)
    return state, recomputed_hash


# ──────────────────────────────────────────────────────────────────────
# 审计报告
# ──────────────────────────────────────────────────────────────────────
def save_audit_report(audit: Dict[str, Any], path: str | Path) -> None:
    """保存 Phase 0 审计报告。"""
    save_json(audit, path)
    LOGGER.info("Saved audit report: %s (status=%s)", path, audit.get("status", "UNKNOWN"))


def save_metrics(metrics: Dict[str, Any], path: str | Path) -> None:
    """保存评估指标 JSON。"""
    save_json(metrics, path)
    LOGGER.info("Saved metrics: %s", path)


def save_gene_panel_report(
    gene_registry_data: Dict[str, Any],
    gate_stability: Dict[str, Any],
    path: str | Path,
) -> None:
    """保存基因面板稳定性报告 CSV。

    Args:
        gene_registry_data: GeneRegistry.to_dict() 输出
        gate_stability: 跨 fold Jaccard 稳定性
        path: 输出 CSV 路径
    """
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    genes = gene_registry_data.get("genes", [])
    source_presence = gene_registry_data.get("source_presence", {})

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gene", "source_presence_count", "panel_mode"])
        for gene in genes:
            writer.writerow([
                gene,
                source_presence.get(gene, 0),
                gene_registry_data.get("mode", "strict"),
            ])

    LOGGER.info("Saved gene panel report: %s (%d genes)", path, len(genes))
