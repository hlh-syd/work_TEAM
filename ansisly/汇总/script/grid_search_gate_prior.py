"""Grid search for gate prior hyperparameters.

Runs the CMIB-Surv pipeline with different combinations of:
  - alpha: Coxnet regularization strength in _bootstrap_gate_prior
  - max_iter: Coxnet iteration limit in _bootstrap_gate_prior
  - source_bootstraps: number of bootstrap samples for source-cancer gate priors

Grid: 3 alpha x 2 max_iter x 2 source_bootstraps = 12 runs (~2 hours total)
Each run takes ~10 minutes.
"""

from __future__ import annotations

import csv
import itertools
import json
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
MAIN_SCRIPT = SCRIPT_DIR / "035_CasualModel.py"
RESULTS_CSV = SCRIPT_DIR / "grid_search_results.csv"
TIMESTAMP = "30260705_203434"

GRID = {
    "alpha": [0.01, 0.02, 0.05],
    "max_iter": [500, 2000],
    "source_bootstraps": [15, 30],
}

C_INDEX_RE = re.compile(r'"oof_harrell_c":\s*([\d.]+)')


def run_single(alpha: float, max_iter: int, source_bootstraps: int) -> dict:
    label = f"a{alpha}_mi{max_iter}_sb{source_bootstraps}"
    output_dir = SCRIPT_DIR.parent / "results" / "grid_search" / label
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--timestamp", TIMESTAMP,
        "--output-dir", str(output_dir),
        "--gate-alpha", str(alpha),
        "--gate-max-iter", str(max_iter),
        "--source-bootstraps", str(source_bootstraps),
    ]

    print(f"\n{'='*60}")
    print(f"Running: alpha={alpha} max_iter={max_iter} source_bootstraps={source_bootstraps}")
    print(f"Output:  {output_dir}")
    print(f"Start:   {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}", flush=True)

    # Skip if already completed
    bundle_file = output_dir / "final_development_bundle.pt"
    log_file = output_dir / "pipeline.log"
    if bundle_file.exists() and log_file.exists():
        log_text = log_file.read_text(encoding="utf-8")
        match = C_INDEX_RE.search(log_text)
        if match:
            c_index = float(match.group(1))
            print(f"SKIP (already done) | C-Index={c_index}", flush=True)
            return {
                "alpha": alpha, "max_iter": max_iter, "source_bootstraps": source_bootstraps,
                "c_index": c_index, "elapsed_min": 0.0, "output_dir": str(output_dir),
                "stdout_tail": "", "stderr_tail": "",
            }

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    elapsed = time.time() - start

    c_index = None
    combined_output = (result.stdout or "") + (result.stderr or "")
    match = C_INDEX_RE.search(combined_output)
    if match:
        c_index = float(match.group(1))

    print(f"Done in {elapsed/60:.1f} min | C-Index={c_index}", flush=True)

    return {
        "alpha": alpha,
        "max_iter": max_iter,
        "source_bootstraps": source_bootstraps,
        "c_index": c_index,
        "elapsed_min": round(elapsed / 60, 1),
        "output_dir": str(output_dir),
        "stdout_tail": result.stdout[-500:] if result.stdout else "",
        "stderr_tail": result.stderr[-500:] if result.stderr else "",
    }


def main() -> int:
    combinations = list(itertools.product(
        GRID["alpha"], GRID["max_iter"], GRID["source_bootstraps"]
    ))
    print(f"Grid search: {len(combinations)} combinations")
    print(f"Estimated total time: ~{len(combinations) * 10} minutes")

    results = []
    for i, (alpha, max_iter, source_bootstraps) in enumerate(combinations, 1):
        print(f"\n[{i}/{len(combinations)}]", flush=True)
        result = run_single(alpha, max_iter, source_bootstraps)
        results.append(result)

        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "alpha", "max_iter", "source_bootstraps", "c_index", "elapsed_min"
            ])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "alpha": r["alpha"],
                    "max_iter": r["max_iter"],
                    "source_bootstraps": r["source_bootstraps"],
                    "c_index": r["c_index"],
                    "elapsed_min": r["elapsed_min"],
                })

    print(f"\n{'='*60}")
    print("GRID SEARCH COMPLETE - Results Summary")
    print(f"{'='*60}")
    print(f"{'alpha':<8} {'max_iter':<10} {'bootstraps':<12} {'C-Index':<10} {'min'}")
    print("-" * 55)
    valid = [r for r in results if r["c_index"] is not None]
    valid.sort(key=lambda r: r["c_index"], reverse=True)
    for r in valid:
        print(f"{r['alpha']:<8} {r['max_iter']:<10} {r['source_bootstraps']:<12} {r['c_index']:<10.4f} {r['elapsed_min']}")
    if len(valid) < len(results):
        print(f"\n{len(results) - len(valid)} run(s) failed (no C-Index)")
    if valid:
        best = valid[0]
        print(f"\nBest: alpha={best['alpha']} max_iter={best['max_iter']} "
              f"bootstraps={best['source_bootstraps']} C-Index={best['c_index']:.4f}")
    print(f"\nResults saved to: {RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
