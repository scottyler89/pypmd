#!/usr/bin/env python3
"""Run PMD characterization for multiple (N1, N2) pairs and aggregate overlays.

Example:
  python benchmarks/run_multi_config.py \
    --K 10 --N1-list 250,250 --N2-list 500,1000 --smax 10 --iters 5 \
    --num-boot 200 --expand-union true --tag multi
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
from datetime import datetime
from typing import List

import pandas as pd
import numpy as np

from percent_max_diff.benchmarking import SimulationConfig, run_simulation_grid


def parse_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    # Defaults match R's do_full_pmd_characterization pairs
    ap.add_argument("--N1-list", type=str, default="10000,10000,1000,250,250,100", help="comma-separated list, e.g., 10000,10000,1000,250,250,100")
    ap.add_argument("--N2-list", type=str, default="10000,1000,1000,2000,250,100", help="comma-separated list, e.g., 10000,1000,1000,2000,250,100")
    ap.add_argument("--smax", type=int, default=10)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--num-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pseudocount", choices=["none", "one"], default="none")
    ap.add_argument("--composition", choices=["uniform", "dirichlet"], default="uniform")
    ap.add_argument("--dirichlet-alpha", type=float, default=1.0)
    ap.add_argument("--effect", choices=["overlap_shift", "fold_change"], default="overlap_shift")
    ap.add_argument("--sampling", choices=["multinomial", "poisson_multinomial", "poisson_independent"], default="multinomial")
    ap.add_argument("--null", choices=["permutation", "parametric"], default="permutation")
    ap.add_argument("--executor", choices=["sequential", "multiprocessing", "ray"], default="sequential")
    ap.add_argument("--store-null", type=str, default="true")
    ap.add_argument("--expand-union", type=str, default="true")
    ap.add_argument("--progress", type=str, default="true")
    ap.add_argument("--warn-eps", type=float, default=1e-9)
    ap.add_argument("--tag", type=str, default="multi")
    ap.add_argument("--outdir", type=str, default="", help="optional explicit output directory")
    ap.add_argument("--r-compat", type=str, default="false")
    ap.add_argument("--r-num-sim", type=int, default=100)
    ap.add_argument("--entropy-method", choices=["hypergeo","bootstrap"], default="hypergeo")
    ap.add_argument("--lambda-mode", choices=["bootstrap","approx"], default="bootstrap")
    ap.add_argument("--lambda-model", type=str, default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    N1s = parse_list(args.N1_list)
    N2s = parse_list(args.N2_list)
    s_values = list(range(0, max(0, args.smax) + 1))

    if args.outdir:
        out_dir = args.outdir
        os.makedirs(out_dir, exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("benchmarks", "out", f"{ts}_{args.tag}")
        os.makedirs(out_dir, exist_ok=True)

    results_rows = []
    null_rows = []

    for N1, N2 in zip(N1s, N2s):
        cfg = SimulationConfig(
            K=args.K,
            N1=N1,
            N2=N2,
            s_values=s_values,
            iters=args.iters,
            num_boot=args.num_boot,
            random_state=args.seed,
            pseudocount_mode=args.pseudocount,
            composition=args.composition,
            dirichlet_alpha=(args.dirichlet_alpha if args.composition == "dirichlet" else None),
            effect=args.effect,
            sampling_model=args.sampling,
            null_model=args.null,
            store_null_draws=True,
            executor=args.executor,
            show_progress=(str(args.progress).lower() in {"1", "true", "yes"}),
            expand_union=(str(args.expand_union).lower() in {"1", "true", "yes"}),
            warn_eps=float(args.warn_eps),
            entropy_method=args.entropy_method,
            lambda_mode=args.lambda_mode,
            lambda_model_path=(args.lambda_model or None),
        )
        res = run_simulation_grid(cfg)
        if isinstance(res, tuple):
            df, nd = res
        else:
            df, nd = res, None
        df = df.copy()
        df["dataset_sizes"] = f"{N1}_vs_{N2}"
        results_rows.append(df)
        if nd is not None and not nd.empty:
            nd = nd.copy()
            nd["dataset_sizes"] = f"{N1}_vs_{N2}"
            null_rows.append(nd)

    results_df = pd.concat(results_rows, ignore_index=True)
    results_path = os.path.join(out_dir, "results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"wrote {results_path}")

    if null_rows:
        null_df = pd.concat(null_rows, ignore_index=True)
        null_path = os.path.join(out_dir, "null.csv")
        null_df.to_csv(null_path, index=False)
        print(f"wrote {null_path}")

    # Write a small config JSON for traceability
    import json

    cfg_path = os.path.join(out_dir, "config.json")
    with open(cfg_path, "w") as fh:
        json.dump({
            "K": args.K,
            "N1_list": N1s,
            "N2_list": N2s,
            "smax": args.smax,
            "iters": args.iters,
            "num_boot": args.num_boot,
            "seed": args.seed,
            "pseudocount": args.pseudocount,
            "composition": args.composition,
            "dirichlet_alpha": args.dirichlet_alpha,
            "effect": args.effect,
            "sampling": args.sampling,
            "null": args.null,
            "executor": args.executor,
            "expand_union": args.expand_union,
            "progress": args.progress,
            "warn_eps": args.warn_eps,
        }, fh, indent=2)
    print(f"wrote {cfg_path}")

    # Optional: R-compatible CSVs per aggregate
    if str(args.r_compat).lower() in {"1", "true", "yes"}:
        dfr = results_df.copy()
        # Prefer robust computation from chi2 and observed rows (dof = (r-1)(c-1), c=2)
        if ("chi2" in dfr.columns) and ("observed_clusters" in dfr.columns or "total_clusters" in dfr.columns) and ("chi_neg_log_p" not in dfr.columns):
            try:
                from scipy import stats as _sps
                rows_obs = dfr.get("observed_clusters", dfr.get("total_clusters"))
                dof = np.maximum(rows_obs.astype(float) - 1.0, 1.0)
                dfr["chi_neg_log_p"] = -_sps.chi2.logsf(dfr["chi2"].to_numpy(dtype=float), dof.to_numpy(dtype=float)) / np.log(10.0)
            except Exception:
                if "chi2_p" in dfr.columns:
                    dfr["chi_neg_log_p"] = -np.log10(dfr["chi2_p"])  # no clipping fallback
        elif "chi2_p" in dfr.columns and "chi_neg_log_p" not in dfr.columns:
            dfr["chi_neg_log_p"] = -np.log10(dfr["chi2_p"])  # no clipping fallback
        dfr["total_cells"] = dfr["N1"] + dfr["N2"]
        dfr["total_number_of_clusters"] = dfr.get("observed_clusters", dfr.get("total_clusters", np.nan))
        dfr["batch_size_difference"] = (dfr["N1"] - dfr["N2"]).abs()
        dfr["min_of_expected_mat"] = dfr.get("min_E", np.nan)
        dfr["null_distribution_lambda_estimate"] = dfr.get("pmd_lambda", np.nan)
        dfr["num_non_shared_clusters"] = dfr.get("s", np.nan)
        dfr["raw_pmd"] = dfr.get("pmd_raw", np.nan)
        # Prefer bias-corrected V
        if "cramers_v_bc" in dfr.columns:
            dfr["cramers_v"] = dfr["cramers_v_bc"]
        dfr["inverse_simp"] = dfr.get("inv_simpson", np.nan)
        dfr["shannon_entropy"] = dfr.get("entropy", np.nan)
        cols_r = [
            "dataset_sizes","iter","total_cells","total_number_of_clusters","N1","N2",
            "batch_size_difference","min_of_expected_mat","null_distribution_lambda_estimate",
            "num_non_shared_clusters","raw_pmd","pmd","chi2","chi_neg_log_p","cramers_v",
            "inverse_simp","shannon_entropy",
        ]
        # Rename N1/N2 to b1_size/b2_size to match R
        dfr = dfr.rename(columns={"N1":"b1_size","N2":"b2_size","chi2":"chi_sq"})
        cols_r = [
            "dataset_sizes","iter","total_cells","total_number_of_clusters","b1_size","b2_size",
            "batch_size_difference","min_of_expected_mat","null_distribution_lambda_estimate",
            "num_non_shared_clusters","raw_pmd","pmd","chi_sq","chi_neg_log_p","cramers_v",
            "inverse_simp","shannon_entropy",
        ]
        dfr = dfr.reindex(columns=cols_r)
        dfr = dfr.sort_values(by=["dataset_sizes","iter","num_non_shared_clusters"]).reset_index(drop=True)
        out_r = os.path.join(out_dir, "results_r.csv")
        dfr.to_csv(out_r, index=False)
        print(f"wrote {out_r}")

        if null_rows:
            nd = pd.concat(null_rows, ignore_index=True)
            out_rows = []
            for (ds, it), sub in nd.groupby(["dataset_sizes","iter"]):
                sub0 = sub[sub.get("s", 0) == 0]
                vals = sub0["pmd_null"].to_numpy()
                if vals.size == 0:
                    continue
                if vals.size >= args.r_num_sim:
                    take = vals[: args.r_num_sim]
                else:
                    idx = np.random.default_rng(int(args.seed)).choice(np.arange(vals.size), size=args.r_num_sim, replace=True)
                    take = vals[idx]
                out_rows.extend([{"dataset_sizes": ds, "pmd_null": float(v)} for v in take])
            null_r = pd.DataFrame(out_rows, columns=["dataset_sizes","pmd_null"]) if out_rows else pd.DataFrame(columns=["dataset_sizes","pmd_null"])
            null_r_path = os.path.join(out_dir, "null_r.csv")
            null_r.to_csv(null_r_path, index=False)
            print(f"wrote {null_r_path}")

    # Plot using the directory-aware plotter
    plot_cmd = [
        shutil.which("python") or "python",
        os.path.join("benchmarks", "plot_characterize_pmd.py"),
        "--dir", out_dir,
        "--super-grid", "true",
    ]
    try:
        import subprocess
        subprocess.run(plot_cmd, check=True)
    except Exception:
        print("plotting failed; results and null CSV are available")


if __name__ == "__main__":
    main()
