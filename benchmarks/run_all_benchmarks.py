#!/usr/bin/env python3
"""Run all PMD analyses in one command: characterize, overlays, invariance, plots, and optional parity.

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
import json
import warnings
import pandas as pd
from benchmarks.config import (
    DEFAULT_DATASET_PAIR_SET,
    RENAME_MAP_PER_SCENARIO,
    dataset_pair_sets,
    default_dataset_pairs_str,
    default_invariance,
    parallel_env,
    resolve_max_workers,
)


def run_with_log(cmd, log_path):
    env = os.environ.copy()
    env.update(parallel_env())
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as fh:
        fh.write("$ " + " ".join(cmd) + "\n\n")
        fh.flush()
        subprocess.run(cmd, stdout=fh, stderr=fh, check=True, env=env)


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
    ap.add_argument("--validate-invariance", type=str, default="false")
    # Multi-config overlays from SSoT
    default_set = DEFAULT_DATASET_PAIR_SET
    ap.add_argument("--dataset-pair-set", choices=dataset_pair_sets(), default=default_set,
                    help=f"choose dataset pair regime (default: {default_set})")
    ap.add_argument("--N1-list", type=str, default=None)
    ap.add_argument("--N2-list", type=str, default=None)
    # Invariance from SSoT
    inv = default_invariance()
    ap.add_argument("--num-clust-range", type=str, default=inv["k_list_str"]) 
    ap.add_argument("--percent-difference", type=str, default=inv["pdiffs_str"]) 
    ap.add_argument("--b1-sizes", type=str, default=None)
    ap.add_argument("--b2-sizes", type=str, default=None)
    ap.add_argument("--expand-b2-only", type=str, default="false")
    # Optional parity compare (R CSV)
    ap.add_argument("--r-compare", type=str, default="")
    # Parallel + Tag for output dirs
    ap.add_argument("--parallel", type=str, default="false", help=argparse.SUPPRESS)
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument("--tag", type=str, default="all")
    return ap.parse_args()


def latest_for_tag(tag: str) -> str:
    candidates = sorted(glob.glob(os.path.join("benchmarks", "out", f"*_{tag}")))
    if not candidates:
        return ""
    return candidates[-1]


def main() -> None:
    args = parse_args()
    n1_default, n2_default = default_dataset_pairs_str(args.dataset_pair_set)
    if args.N1_list is None:
        args.N1_list = n1_default
    if args.N2_list is None:
        args.N2_list = n2_default
    if args.b1_sizes is None:
        args.b1_sizes = n1_default
    if args.b2_sizes is None:
        args.b2_sizes = n2_default
    max_workers = resolve_max_workers(args.max_workers)
    if str(args.parallel).lower() not in {"false", "0", "no"}:
        warnings.warn("--parallel is deprecated and ignored; use --max-workers instead", DeprecationWarning)
    os.environ.update(parallel_env())

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

    skip_plots = str(os.environ.get("PMD_SKIP_PLOTS", "")).lower() in {"1", "true", "yes"}

    if skip_plots:
        cmd_char = [
            "python", os.path.join("benchmarks", "run_characterize_pmd.py"),
            "--K", str(args.K),
            "--N1", str(args.N1),
            "--N2", str(args.N2),
            "--smax", str(args.smax),
            "--iters", str(args.iters),
            "--num-boot", str(args.num_boot),
            "--seed", str(args.seed),
            "--max-workers", str(max_workers),
            "--store-null", args.store_null,
            "--expand-union", args.expand_union,
            "--r-compat", args.r_compat,
            "--r-num-sim", str(args.r_num_sim),
            "--tag", f"{args.tag}_char",
            "--outdir", char_dir,
        ]
        subprocess.run(cmd_char, check=True)
    else:
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
            "--max-workers", str(max_workers),
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
            "--max-workers", str(max_workers),
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
        "--max-workers", str(max_workers),
        "--dataset-pair-set", args.dataset_pair_set,
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

    # 3) Invariance scenarios (build commands; optionally run in parallel)
    cmd_inv = [
        "python", os.path.join("benchmarks", "run_invariance_property.py"),
        "--dataset-pair-set", args.dataset_pair_set,
        "--max-workers", str(max_workers),
        "--num-clust-range", args.num_clust_range,
        "--percent-difference", args.percent_difference,
        "--b1-sizes", args.b1_sizes,
        "--b2-sizes", args.b2_sizes,
        "--iters", str(args.iters),
        "--mode", "symmetric",
        "--num-boot", str(args.num_boot),
        "--seed", str(args.seed + 0),
        "--r-compat", args.r_compat,
        "--tag", f"{args.tag}_invariance_sym",
        "--outdir", inv_sym_dir,
    ]
    cmd_inv_b2 = [
        "python", os.path.join("benchmarks", "run_invariance_property.py"),
        "--dataset-pair-set", args.dataset_pair_set,
        "--max-workers", str(max_workers),
        "--num-clust-range", args.num_clust_range,
        "--percent-difference", args.percent_difference,
        "--b1-sizes", args.b1_sizes,
        "--b2-sizes", args.b2_sizes,
        "--iters", str(args.iters),
        "--expand-b2-only", "true",
        "--num-boot", str(args.num_boot),
        "--seed", str(args.seed + 1),
        "--r-compat", args.r_compat,
        "--tag", f"{args.tag}_invariance_b2",
        "--outdir", inv_b2_dir,
    ]
    cmd_inv_priv = [
        "python", os.path.join("benchmarks", "run_invariance_property.py"),
        "--dataset-pair-set", args.dataset_pair_set,
        "--max-workers", str(max_workers),
        "--num-clust-range", args.num_clust_range,
        "--percent-difference", args.percent_difference,
        "--b1-sizes", args.b1_sizes,
        "--b2-sizes", args.b2_sizes,
        "--iters", str(args.iters),
        "--mode", "fixed_union_private",
        "--num-boot", str(args.num_boot),
        "--seed", str(args.seed + 2),
        "--r-compat", args.r_compat,
        "--tag", f"{args.tag}_invariance_priv",
        "--outdir", inv_priv_dir,
    ]

    # Decide parallel vs sequential for invariance
    invariance_jobs = [
        ("invariance_symmetric", cmd_inv, os.path.join(inv_dir, "symmetric", "run.log")),
        ("invariance_expand_b2_only", cmd_inv_b2, os.path.join(inv_dir, "expand_b2_only", "run.log")),
        ("invariance_fixed_union_private", cmd_inv_priv, os.path.join(inv_dir, "fixed_union_private", "run.log")),
    ]
    for name, cmd, log in invariance_jobs:
        run_with_log(cmd, log)

    # Summary
    # Invariance super-grids per mode
    invariance_dirs = [
        (inv_sym_dir, "symmetric"),
        (inv_b2_dir, "expand_b2_only"),
        (inv_priv_dir, "fixed_union_private"),
    ]
    for sub, mode_name in invariance_dirs:
        inv_csv = os.path.join(sub, "invariance_results.csv")
        if os.path.exists(inv_csv):
            subprocess.run(["python", os.path.join("benchmarks", "plot_invariance.py"), "--file", inv_csv, "--include-comparators", "true"], check=True)
            if str(args.validate_invariance).lower() in {"1", "true", "yes"}:
                subprocess.run([
                    "python", os.path.join("benchmarks", "validate_invariance_linear.py"),
                    "--file", inv_csv,
                    "--mode", mode_name,
                    "--out-dir", sub,
                ], check=True)
                subprocess.run([
                    "python", os.path.join("benchmarks", "validate_invariance_onehot.py"),
                    "--file", inv_csv,
                    "--mode", mode_name,
                    "--out-dir", sub,
                ], check=True)

    # Generate a simple programmatic report
    subprocess.run(["python", os.path.join("benchmarks", "generate_report.py"), "--base", base_dir], check=True)

    # Collate unified results (best-effort)
    try:
        unified_rows = []
        unified_null_rows = []

        def _standardize(df: pd.DataFrame, scenario: str, mode: str | None = None) -> pd.DataFrame:
            rename_map = RENAME_MAP_PER_SCENARIO.get(scenario, {})
            if rename_map:
                for src, dest in rename_map.items():
                    if src in df.columns and (dest not in df.columns or dest == src):
                        df = df.rename(columns={src: dest}, errors="ignore")
            if "dataset_sizes" not in df.columns:
                if {"b1_size", "b2_size"}.issubset(df.columns):
                    ds = df["b1_size"].astype(int).astype(str) + "_vs_" + df["b2_size"].astype(int).astype(str)
                elif {"N1", "N2"}.issubset(df.columns):
                    ds = df["N1"].astype(int).astype(str) + "_vs_" + df["N2"].astype(int).astype(str)
                else:
                    label = mode or scenario
                    ds = pd.Series([label] * len(df))
                df["dataset_sizes"] = ds
            if "b1_size" not in df.columns and "N1" in df.columns:
                df["b1_size"] = df["N1"]
            if "b2_size" not in df.columns and "N2" in df.columns:
                df["b2_size"] = df["N2"]
            if "K" not in df.columns:
                if "NumberOfClusters" in df.columns:
                    df["K"] = df["NumberOfClusters"]
                elif "total_number_of_clusters" in df.columns:
                    df["K"] = df["total_number_of_clusters"]
            if "dataset_pair_set" not in df.columns:
                df["dataset_pair_set"] = args.dataset_pair_set
            if "n_workers" not in df.columns:
                df["n_workers"] = max_workers
            return df

        # characterize results/nulls
        p = os.path.join(char_dir, "results.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            df = _standardize(df, "characterize")
            df["scenario"] = "characterize"
            unified_rows.append(df)
        p = os.path.join(char_dir, "null.csv")
        if os.path.exists(p):
            nd = pd.read_csv(p)
            nd = _standardize(nd, "characterize")
            nd["scenario"] = "characterize"
            unified_null_rows.append(nd)

        # overlays results/nulls
        p = os.path.join(multi_dir, "results.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            df = _standardize(df, "overlays")
            df["scenario"] = "overlays"
            unified_rows.append(df)
        p = os.path.join(multi_dir, "null.csv")
        if os.path.exists(p):
            nd = pd.read_csv(p)
            nd = _standardize(nd, "overlays")
            nd["scenario"] = "overlays"
            unified_null_rows.append(nd)

        # invariance results/nulls
        for sub, mode_name in [(inv_sym_dir, "symmetric"), (inv_b2_dir, "expand_b2_only"), (inv_priv_dir, "fixed_union_private")]:
            p = os.path.join(sub, "invariance_results.csv")
            if os.path.exists(p):
                df = pd.read_csv(p)
                df = _standardize(df, "invariance", mode_name)
                df["scenario"] = "invariance"
                df["mode"] = mode_name
                unified_rows.append(df)
            p = os.path.join(sub, "null.csv")
            if os.path.exists(p):
                nd = pd.read_csv(p)
                nd = _standardize(nd, "invariance", mode_name)
                nd["scenario"] = "invariance"
                nd["mode"] = mode_name
                unified_null_rows.append(nd)

        if unified_rows:
            unified = pd.concat(unified_rows, ignore_index=True, sort=False)
            unified.to_csv(os.path.join(base_dir, "unified_results.csv"), index=False)
            manifest = {
                "base_dir": base_dir,
                "scenarios": [j[0] for j in invariance_jobs] + ["characterize", "overlays"],
                "seed": args.seed,
                "dataset_pair_set": args.dataset_pair_set,
                "max_workers": max_workers,
            }
            with open(os.path.join(base_dir, "manifest.json"), "w") as fh:
                json.dump(manifest, fh, indent=2)
        if unified_null_rows:
            unified_nulls = pd.concat(unified_null_rows, ignore_index=True, sort=False)
            unified_nulls.to_csv(os.path.join(base_dir, "unified_nulls.csv"), index=False)
    except Exception as e:
        print(f"warning: collation failed: {e}")

    print("\nAll analyses complete. Outputs in:")
    print(f"  {base_dir}")


if __name__ == "__main__":
    main()
