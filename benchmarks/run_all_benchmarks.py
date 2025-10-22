#!/usr/bin/env python3
"""Run all PMD analyses in one command: characterize, overlays, invariance, plots, lambda fit, and optional parity.

Example (reasonable defaults):
  python benchmarks/run_all_benchmarks.py \
    --K 10 --N1 2000 --N2 2000 --smax 10 --iters 5 --num-boot 200 \
    --N1-list 250,250 --N2-list 500,1000 \
    --num-clust-range 4,8,12,16 --percent-difference 0,0.25,0.5,0.75,1.0 \
    --expand-union true --store-null true --r-compat true --pdf true --super-grid true --tag all

If you have an R CSV to compare against for the single-characterize run:
  add: --r-compare /path/to/R_out_df.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
from datetime import datetime


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # Single characterize
    ap.add_argument("--K", type=int, default=10)
    # Single-characterize defaults match R's characterize_pmd()
    ap.add_argument("--N1", type=int, default=1000)
    ap.add_argument("--N2", type=int, default=1000)
    ap.add_argument("--smax", type=int, default=10)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--num-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--expand-union", type=str, default="true")
    ap.add_argument("--store-null", type=str, default="true")
    ap.add_argument("--r-compat", type=str, default="true")
    ap.add_argument("--r-num-sim", type=int, default=100)
    ap.add_argument("--pdf", type=str, default="true")
    ap.add_argument("--super-grid", type=str, default="true")
    ap.add_argument("--merged-grid", type=str, default="false")
    ap.add_argument("--lambda-violin-by", choices=["", "K", "s"], default="")
    ap.add_argument("--lambda-ridgeline-by", choices=["", "K", "s"], default="")
    # Multi-config overlays
    ap.add_argument("--N1-list", type=str, default="10000,10000,1000,250,250,100")
    ap.add_argument("--N2-list", type=str, default="10000,1000,1000,2000,250,100")
    # Invariance
    ap.add_argument("--num-clust-range", type=str, default="4,8,12,16")
    ap.add_argument("--percent-difference", type=str, default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--b1-sizes", type=str, default="10000,10000,1000,250,250,100")
    ap.add_argument("--b2-sizes", type=str, default="10000,1000,1000,2000,250,100")
    ap.add_argument("--expand-b2-only", type=str, default="false")
    # Optional parity compare (R CSV)
    ap.add_argument("--r-compare", type=str, default="")
    # Tag for output dirs
    ap.add_argument("--tag", type=str, default="all")
    return ap.parse_args()


def latest_for_tag(tag: str) -> str:
    candidates = sorted(glob.glob(os.path.join("benchmarks", "out", f"*_{tag}")))
    if not candidates:
        return ""
    return candidates[-1]


def main() -> None:
    args = parse_args()
    # Create a single base output directory
    base_dir = os.path.join("benchmarks", "out", datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.tag}")
    os.makedirs(base_dir, exist_ok=True)
    char_dir = os.path.join(base_dir, "characterize")
    char_rc_dir = os.path.join(base_dir, "characterize_r")
    multi_dir = os.path.join(base_dir, "overlays")
    inv_dir = os.path.join(base_dir, "invariance")
    os.makedirs(char_dir, exist_ok=True)
    os.makedirs(char_rc_dir, exist_ok=True)
    os.makedirs(multi_dir, exist_ok=True)
    os.makedirs(inv_dir, exist_ok=True)
    inv_sym_dir = os.path.join(inv_dir, "symmetric"); os.makedirs(inv_sym_dir, exist_ok=True)
    inv_b2_dir = os.path.join(inv_dir, "expand_b2_only"); os.makedirs(inv_b2_dir, exist_ok=True)
    inv_priv_dir = os.path.join(inv_dir, "fixed_union_private"); os.makedirs(inv_priv_dir, exist_ok=True)

    # 1) Single-characterize full run (with plots + lambda model)
    cmd_char = [
        "python", os.path.join("benchmarks", "run_full_benchmark.py"),
        "--K", str(args.K),
        "--N1", str(args.N1),
        "--N2", str(args.N2),
        "--smax", str(args.smax),
        "--iters", str(args.iters),
        "--num-boot", str(args.num_boot),
        "--seed", str(args.seed),
        "--executor", "sequential",
        "--store-null", args.store_null,
        "--expand-union", args.expand_union,
        "--progress", "true",
        "--tag", f"{args.tag}_char",
        "--outdir", char_dir,
        "--pdf", args.pdf,
    ]
    subprocess.run(cmd_char, check=True)
    # char_dir already defined

    # Write R-compat for single characterize
    cmd_char_rc = [
        "python", os.path.join("benchmarks", "run_characterize_pmd.py"),
        "--K", str(args.K),
        "--N1", str(args.N1),
        "--N2", str(args.N2),
        "--smax", str(args.smax),
        "--iters", str(args.iters),
        "--num-boot", str(args.num_boot),
        "--seed", str(args.seed),
        "--executor", "sequential",
        "--store-null", args.store_null,
        "--expand-union", args.expand_union,
        "--r-compat", args.r_compat,
        "--r-num-sim", str(args.r_num_sim),
        "--tag", f"{args.tag}_char_rc",
        "--outdir", char_rc_dir,
    ]
    subprocess.run(cmd_char_rc, check=True)
    # char_rc_dir already defined

    # Enhance plotting (super-grid/merged-grid) on characterize output
    cmd_plot = [
        "python", os.path.join("benchmarks", "plot_characterize_pmd.py"),
        "--dir", char_dir,
        "--super-grid", args.super_grid,
        "--merged-grid", args.merged_grid,
        "--null-density-out", os.path.join(char_dir, "null_pmd_density.png"),
    ]
    subprocess.run(cmd_plot, check=True)

    # Standalone focus figures (PMD + lambda)
    subprocess.run(["python", os.path.join("benchmarks", "plot_pmd_focus.py"), "--dir", char_dir], check=True)
    cmd_lambda = ["python", os.path.join("benchmarks", "plot_lambda.py"), "--dir", char_dir]
    if args.lambda_violin_by:
        cmd_lambda += ["--violin-by", args.lambda_violin_by]
    if args.lambda_ridgeline_by:
        cmd_lambda += ["--ridgeline-by", args.lambda_ridgeline_by]
    subprocess.run(cmd_lambda, check=True)

    # Parity compare if R CSV provided
    if args.r_compare:
        ours_csv = os.path.join(char_rc_dir, "results_r.csv")
        cmd_cmp = [
            "python", os.path.join("benchmarks", "compare_with_r_csv.py"),
            "--ours", ours_csv,
            "--r", args.r_compare,
        ]
        subprocess.run(cmd_cmp, check=True)

    # 2) Multi-config overlays (also R-compat)
    cmd_multi = [
        "python", os.path.join("benchmarks", "run_multi_config.py"),
        "--K", str(args.K),
        "--N1-list", args.N1_list,
        "--N2-list", args.N2_list,
        "--smax", str(args.smax),
        "--iters", str(args.iters),
        "--num-boot", str(args.num_boot),
        "--seed", str(args.seed),
        "--expand-union", args.expand_union,
        "--r-compat", args.r_compat,
        "--r-num-sim", str(args.r_num_sim),
        "--tag", f"{args.tag}_multi",
        "--outdir", multi_dir,
    ]
    subprocess.run(cmd_multi, check=True)
    # Overlay lambda figures across dataset sizes
    try:
        cmd_lambda_multi = ["python", os.path.join("benchmarks", "plot_lambda.py"), "--dir", multi_dir]
        if args.lambda_violin_by:
            cmd_lambda_multi += ["--violin-by", args.lambda_violin_by]
        if args.lambda_ridgeline_by:
            cmd_lambda_multi += ["--ridgeline-by", args.lambda_ridgeline_by]
        subprocess.run(cmd_lambda_multi, check=True)
    except Exception:
        print("warning: overlay lambda plots failed; continuing")

    # 3a) Invariance property — symmetric (also R-compat)
    cmd_inv = [
        "python", os.path.join("benchmarks", "run_invariance_property.py"),
        "--num-clust-range", args.num_clust_range,
        "--percent-difference", args.percent_difference,
        "--b1-sizes", args.b1_sizes,
        "--b2-sizes", args.b2_sizes,
        "--iters", str(args.iters),
        "--mode", "symmetric",
        "--num-boot", str(args.num_boot),
        "--seed", str(args.seed),
        "--r-compat", args.r_compat,
        "--tag", f"{args.tag}_invariance_sym",
        "--outdir", inv_sym_dir,
    ]
    subprocess.run(cmd_inv, check=True)

    # 3b) Invariance property — expand_b2_only
    cmd_inv_b2 = [
        "python", os.path.join("benchmarks", "run_invariance_property.py"),
        "--num-clust-range", args.num_clust_range,
        "--percent-difference", args.percent_difference,
        "--b1-sizes", args.b1_sizes,
        "--b2-sizes", args.b2_sizes,
        "--iters", str(args.iters),
        "--expand-b2-only", "true",
        "--num-boot", str(args.num_boot),
        "--seed", str(args.seed),
        "--r-compat", args.r_compat,
        "--tag", f"{args.tag}_invariance_b2",
        "--outdir", inv_b2_dir,
    ]
    subprocess.run(cmd_inv_b2, check=True)

    # 3c) Invariance property — fixed_union_private
    cmd_inv_priv = [
        "python", os.path.join("benchmarks", "run_invariance_property.py"),
        "--num-clust-range", args.num_clust_range,
        "--percent-difference", args.percent_difference,
        "--b1-sizes", args.b1_sizes,
        "--b2-sizes", args.b2_sizes,
        "--iters", str(args.iters),
        "--mode", "fixed_union_private",
        "--num-boot", str(args.num_boot),
        "--seed", str(args.seed),
        "--r-compat", args.r_compat,
        "--tag", f"{args.tag}_invariance_priv",
        "--outdir", inv_priv_dir,
    ]
    subprocess.run(cmd_inv_priv, check=True)

    # Summary
    # Invariance super-grids per mode
    for sub in (inv_sym_dir, inv_b2_dir, inv_priv_dir):
        inv_csv = os.path.join(sub, "invariance_results.csv")
        if os.path.exists(inv_csv):
            subprocess.run(["python", os.path.join("benchmarks", "plot_invariance.py"), "--file", inv_csv, "--include-comparators", "true"], check=True)

    # Generate a simple programmatic report
    subprocess.run(["python", os.path.join("benchmarks", "generate_report.py"), "--base", base_dir], check=True)

    print("\nAll analyses complete. Outputs in:")
    print(f"  {base_dir}")


if __name__ == "__main__":
    main()
