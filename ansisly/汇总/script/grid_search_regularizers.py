"""Single-dimension 3-point validation for gate_l0_weight.

Based on literature review (Louizatos 2018, Alemi 2017, Projection-Head-as-IB 2025):
  - L0 regularization is robust across 1e-4 ~ 1e-2 (plateau effect)
  - gate_count_weight, elastic_l1, elastic_l2 are fixed at defaults (secondary hyperparams)

Grid: 3 values for gate_l0_weight only → 3 runs (~30 min total)
  Values: 1e-4, 2e-4 (default), 5e-4

Usage:
  python grid_search_regularizers.py
  python grid_search_regularizers.py --timestamp 30260705_203434
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
MAIN_SCRIPT = SCRIPT_DIR / "035_CasualModel.py"
RESULTS_CSV = SCRIPT_DIR / "grid_search_regularizers_results.csv"
REPORT_MD = SCRIPT_DIR / "grid_search_regularizers_report.md"
TIMESTAMP = "30260705_203434"

# Single-dimension grid: only gate_l0_weight
GRID_VALUES = [1e-4, 2e-4, 5e-4]
DEFAULT_VALUE = 2e-4

# Fixed defaults for other regularizer params (not searched)
FIXED_DEFAULTS = {
    "gate_count_weight": 2e-3,
    "elastic_l1": 2e-5,
    "elastic_l2": 2e-5,
}

C_INDEX_RE = re.compile(r'"oof_harrell_c":\s*([\d.]+)')


def run_single(gate_l0_weight: float) -> dict:
    """Run a single configuration and return metrics."""
    label = f"l0_{gate_l0_weight:.0e}".replace(".", "").replace("+", "")
    output_dir = SCRIPT_DIR.parent / "results" / "grid_search_reg" / label
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--timestamp", TIMESTAMP,
        "--output-dir", str(output_dir),
        "--gate-l0-weight", str(gate_l0_weight),
    ]

    # Skip if already completed (resume support)
    bundle_file = output_dir / "final_development_bundle.pt"
    log_file = output_dir / "pipeline.log"
    if bundle_file.exists() and log_file.exists():
        log_text = log_file.read_text(encoding="utf-8")
        match = C_INDEX_RE.search(log_text)
        if match:
            c_index = float(match.group(1))
            print(f"  SKIP (already done) | C-Index={c_index:.4f}", flush=True)
            return {
                "gate_l0_weight": gate_l0_weight,
                "c_index": c_index,
                "elapsed_min": 0.0,
                "status": "skipped",
                "output_dir": str(output_dir),
            }

    print(f"  Running: gate_l0_weight={gate_l0_weight:.1e}", flush=True)
    print(f"  Start:   {time.strftime('%H:%M:%S')}", flush=True)
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    elapsed = time.time() - start

    c_index = None
    combined = (result.stdout or "") + (result.stderr or "")
    match = C_INDEX_RE.search(combined)
    if match:
        c_index = float(match.group(1))

    status = "ok" if c_index is not None else "failed"
    print(f"  Done in {elapsed/60:.1f} min | C-Index={c_index} | {status}", flush=True)

    return {
        "gate_l0_weight": gate_l0_weight,
        "c_index": c_index,
        "elapsed_min": round(elapsed / 60, 1),
        "status": status,
        "output_dir": str(output_dir),
    }


def write_csv(results: list[dict], path: Path) -> None:
    """Write results to CSV."""
    if not results:
        return
    fieldnames = ["gate_l0_weight", "c_index", "elapsed_min", "status", "output_dir"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def write_report(results: list[dict], output_path: Path) -> None:
    """Write Markdown report."""
    valid = [r for r in results if r.get("c_index") is not None]
    valid.sort(key=lambda r: r["c_index"], reverse=True)

    baseline = next(
        (r for r in valid if abs(r["gate_l0_weight"] - DEFAULT_VALUE) < 1e-15),
        None,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# gate_l0_weight 单维度 3 点验证报告\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 搜索策略\n\n")
        f.write("基于文献调研（Louizatos 2018, Alemi 2017, Projection-Head-as-IB 2025）：\n")
        f.write('- L0 正则化在 1e-4 ~ 1e-2 区间具有"平台效应"，对 λ 不敏感\n')
        f.write("- `gate_count_weight`、`elastic_l1`、`elastic_l2` 固定为经验默认值\n\n")

        f.write("## 固定参数\n\n")
        f.write("| 参数 | 值 | 依据 |\n")
        f.write("|---|---|---|\n")
        for name, val in FIXED_DEFAULTS.items():
            f.write(f"| `{name}` | {val:.1e} | 文献经验默认值 |\n")

        f.write("\n## 搜索网格\n\n")
        f.write(f"| gate_l0_weight | 候选值 |\n")
        f.write(f"|---|---|\n")
        f.write(f"| 搜索范围 | {', '.join(f'{v:.1e}' for v in GRID_VALUES)} |\n")
        f.write(f"| 默认值 | {DEFAULT_VALUE:.1e} |\n")

        f.write("\n## 结果\n\n")
        f.write("| Rank | gate_l0_weight | C-Index | Δ vs Default | Time(min) |\n")
        f.write("|---|---|---|---|---|\n")
        baseline_c = baseline["c_index"] if baseline else None
        for i, r in enumerate(valid, 1):
            delta = ""
            if baseline_c is not None and abs(r["gate_l0_weight"] - DEFAULT_VALUE) > 1e-15:
                d = r["c_index"] - baseline_c
                delta = f"{d:+.4f}"
            elif abs(r["gate_l0_weight"] - DEFAULT_VALUE) < 1e-15:
                delta = "(baseline)"
            f.write(f"| {i} | {r['gate_l0_weight']:.1e} | "
                    f"**{r['c_index']:.4f}** | {delta} | {r['elapsed_min']} |\n")

        if len(valid) < len(results):
            f.write(f"\n> {len(results) - len(valid)} run(s) failed\n")

        f.write("\n## 结论\n\n")
        if valid and baseline_c is not None:
            best = valid[0]
            delta = best["c_index"] - baseline_c
            if abs(delta) < 0.005:
                f.write(f"**C-Index 变化 < 0.005**，gate_l0_weight 在 [{GRID_VALUES[0]:.1e}, "
                        f"{GRID_VALUES[-1]:.1e}] 范围内稳健。\n")
                f.write(f"建议保持默认值 {DEFAULT_VALUE:.1e}，无需额外调参。\n")
            elif delta > 0:
                f.write(f"最佳值: {best['gate_l0_weight']:.1e} (C-Index={best['c_index']:.4f}, "
                        f"Δ={delta:+.4f})\n")
            else:
                f.write(f"默认值 {DEFAULT_VALUE:.1e} 已是最优，无需调整。\n")


def main() -> int:
    global TIMESTAMP
    parser = argparse.ArgumentParser(description="gate_l0_weight 3-point validation")
    parser.add_argument("--timestamp", default=TIMESTAMP)
    args = parser.parse_args()
    TIMESTAMP = args.timestamp

    print("=" * 60)
    print("gate_l0_weight 单维度 3 点验证")
    print(f"候选值: {GRID_VALUES}")
    print(f"固定参数: gate_count_weight={FIXED_DEFAULTS['gate_count_weight']:.1e}, "
          f"elastic_l1={FIXED_DEFAULTS['elastic_l1']:.1e}, "
          f"elastic_l2={FIXED_DEFAULTS['elastic_l2']:.1e}")
    print(f"预计耗时: ~{len(GRID_VALUES) * 10} min")
    print("=" * 60)

    results = []
    for i, value in enumerate(GRID_VALUES, 1):
        print(f"\n[{i}/{len(GRID_VALUES)}]")
        result = run_single(value)
        results.append(result)
        write_csv(results, RESULTS_CSV)

    # Summary
    print(f"\n{'='*60}")
    print("验证完成 - 结果汇总")
    print(f"{'='*60}")
    print(f"{'gate_l0_weight':<16} {'C-Index':<12} {'Time(min)':<12} {'Status'}")
    print("-" * 50)
    valid = [r for r in results if r.get("c_index") is not None]
    valid.sort(key=lambda r: r["c_index"], reverse=True)
    for r in valid:
        print(f"{r['gate_l0_weight']:<16.1e} {r['c_index']:<12.4f} "
              f"{r['elapsed_min']:<12} {r['status']}")

    if valid:
        best = valid[0]
        print(f"\n最佳: gate_l0_weight={best['gate_l0_weight']:.1e} "
              f"C-Index={best['c_index']:.4f}")

    write_report(results, REPORT_MD)
    print(f"\nResults CSV: {RESULTS_CSV}")
    print(f"Report MD:   {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
