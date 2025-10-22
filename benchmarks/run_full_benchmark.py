#!/usr/bin/env python3
"""One-shot wrapper to run the PMD benchmark and generate plots/models.

This orchestrates:
- run_characterize_pmd.py (produces results.csv/null.csv/config.json)
- plot_characterize_pmd.py (produces characterize_pmd.png and comparators figure)
- fit_lambda_model.py (produces lambda_model.csv)

Example:
  python benchmarks/run_full_benchmark.py \
    --K 10 --N1 2000 --N2 2000 --smax 10 --iters 5 --num-boot 200 \
    --executor sequential --store-null true --expand-union true --progress true \
    --tag rstyle --pdf true
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from typing import List
from datetime import datetime


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # Core simulation knobs (forwarded to run_characterize_pmd.py)
    ap.add_argument("--K", type=int, default=10)
    # Match R's characterize_pmd() defaults
    ap.add_argument("--N1", type=int, default=1000)
    ap.add_argument("--N2", type=int, default=1000)
    ap.add_argument("--smax", type=int, default=10)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--num-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pseudocount", choices=["none", "one"], default="none")
    ap.add_argument("--composition", choices=["uniform", "dirichlet"], default="uniform")
    ap.add_argument("--dirichlet-alpha", type=float, default=1.0)
    ap.add_argument("--effect", choices=["overlap_shift", "fold_change"], default="overlap_shift")
    ap.add_argument("--effect-indices", type=str, default="")
    ap.add_argument("--effect-fold", type=float, default=3.0)
    ap.add_argument("--sampling", choices=["multinomial", "poisson_multinomial", "poisson_independent"], default="multinomial")
    ap.add_argument("--null", choices=["permutation", "parametric"], default="permutation")
    ap.add_argument("--executor", choices=["sequential", "multiprocessing"], default="sequential")
    ap.add_argument("--store-null", type=str, default="true")
    ap.add_argument("--expand-union", type=str, default="true")
    ap.add_argument("--progress", type=str, default="true")
    ap.add_argument("--tag", type=str, default="full")
    # Outputs
    ap.add_argument("--pdf", type=str, default="true", help="true/false to also emit a multi-page PDF")
    ap.add_argument("--outdir", type=str, default="", help="optional explicit output directory for the characterize run")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def script_path(rel: str) -> str:
        return os.path.join(repo_root, rel)

    # Determine output dir
    # Determine run directory: if --outdir provided, write directly there; otherwise create a new one
    run_dir = args.outdir or os.path.join("benchmarks", "out", datetime.now().strftime("%Y%m%d_%H%M%S") + "_full")
    os.makedirs(run_dir, exist_ok=True)

    # 1) Run the characterization sweep
    run_cmd: List[str] = [
        sys.executable,
        script_path("benchmarks/run_characterize_pmd.py"),
        "--K", str(args.K),
        "--N1", str(args.N1),
        "--N2", str(args.N2),
        "--smax", str(args.smax),
        "--iters", str(args.iters),
        "--num-boot", str(args.num_boot),
        "--seed", str(args.seed),
        "--pseudocount", args.pseudocount,
        "--composition", args.composition,
        "--dirichlet-alpha", str(args.dirichlet_alpha),
        "--effect", args.effect,
        "--effect-indices", args.effect_indices,
        "--effect-fold", str(args.effect_fold),
        "--sampling", args.sampling,
        "--null", args.null,
        "--executor", args.executor,
        "--store-null", args.store_null,
        "--expand-union", args.expand_union,
        "--progress", args.progress,
        "--tag", args.tag,
        "--outdir", run_dir,
    ]
    subprocess.run(run_cmd, check=True)

    # run_dir already defined

    # 3) Plot the figures
    pdf_flag = (str(args.pdf).lower() in {"1", "true", "yes"})
    plot_cmd: List[str] = [
        sys.executable,
        script_path("benchmarks/plot_characterize_pmd.py"),
        "--dir", run_dir,
        "--super-grid", "true",
    ]
    if pdf_flag:
        plot_cmd += ["--pdf", os.path.join(run_dir, "characterize_pmd.pdf")]
    subprocess.run(plot_cmd, check=True)

    # 4) Fit lambda smoother
    fit_cmd: List[str] = [
        sys.executable,
        script_path("benchmarks/fit_lambda_model.py"),
        "--dir", run_dir,
    ]
    subprocess.run(fit_cmd, check=True)

    # 5) Print summary of outputs
    print("\nBenchmark complete. Outputs in:")
    print(f"  {run_dir}")
    print(f"  - results.csv")
    print(f"  - null.csv")
    print(f"  - config.json")
    print(f"  - characterize_pmd.png")
    print(f"  - characterize_comparators.png")
    if pdf_flag:
        print(f"  - characterize_pmd.pdf")
    print(f"  - lambda_model.csv")


if __name__ == "__main__":
    main()
