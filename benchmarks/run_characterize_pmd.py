#!/usr/bin/env python3
"""Run a small PMD characterization sweep and write CSV outputs.

Usage (defaults are small and quick):
  python benchmarks/run_characterize_pmd.py --K 10 --N1 2000 --N2 2000 \
      --smax 10 --iters 5 --num-boot 200 --pseudocount none \
      --composition uniform --effect overlap_shift --sampling multinomial \
      --null permutation --executor sequential --store-null false --tag quick --progress true

Outputs are written to benchmarks/out/YYYYMMDD_HHMMSS_tag/{results.csv,null.csv,config.json}.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import pandas as pd
import numpy as np

from percent_max_diff.benchmarking import SimulationConfig, run_simulation_grid


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    # Match R's characterize_pmd() defaults
    ap.add_argument("--N1", type=int, default=1000)
    ap.add_argument("--N2", type=int, default=1000)
    ap.add_argument("--smax", type=int, default=10, help="max s (inclusive) for overlap shift sweep")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--num-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pseudocount", choices=["none", "one"], default="none")
    ap.add_argument("--composition", choices=["uniform", "dirichlet"], default="uniform")
    ap.add_argument("--dirichlet-alpha", type=float, default=1.0)
    ap.add_argument("--effect", choices=["overlap_shift", "fold_change"], default="overlap_shift")
    ap.add_argument("--effect-indices", type=str, default="", help="comma-separated 0-based indices for fold_change")
    ap.add_argument("--effect-fold", type=float, default=3.0)
    ap.add_argument("--sampling", choices=["multinomial", "poisson_multinomial", "poisson_independent"], default="multinomial")
    ap.add_argument("--null", choices=["permutation", "parametric"], default="permutation")
    ap.add_argument("--executor", choices=["sequential", "multiprocessing"], default="sequential")
    ap.add_argument("--store-null", type=str, default="false", help="true/false")
    ap.add_argument("--tag", type=str, default="quick")
    ap.add_argument("--outdir", type=str, default="", help="optional explicit output directory")
    ap.add_argument("--progress", type=str, default="false")
    ap.add_argument("--expand-union", type=str, default="false", help="true/false; non-wrapping shift to match R")
    ap.add_argument("--warn-eps", type=float, default=1e-9, help="threshold for tiny expected counts E_ij warnings")
    ap.add_argument("--cramers-v-bias-correct", type=str, default="true", help="true/false: bias-corrected Cramer's V column (cramers_v_bc)")
    ap.add_argument("--r-compat", type=str, default="false", help="true/false: also write R-compatible CSVs (results_r.csv, null_r.csv)")
    ap.add_argument("--r-num-sim", type=int, default=100, help="num null draws per iter for R-compatible null_r.csv (s==0)")
    ap.add_argument("--entropy-method", choices=["hypergeo","bootstrap"], default="hypergeo")
    ap.add_argument("--lambda-mode", choices=["bootstrap","approx"], default="bootstrap")
    ap.add_argument("--lambda-model", type=str, default="", help="path to lambda_model.csv for --lambda-mode approx")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    s_values = list(range(0, max(0, args.smax) + 1))
    store_null = str(args.store_null).lower() in {"1", "true", "yes"}
    effect_indices = (
        [int(x) for x in args.effect_indices.split(",") if x.strip()]
        if args.effect_indices
        else None
    )

    cfg = SimulationConfig(
        K=args.K,
        N1=args.N1,
        N2=args.N2,
        s_values=s_values,
        iters=args.iters,
        num_boot=args.num_boot,
        random_state=args.seed,
        pseudocount_mode=args.pseudocount,
        composition=args.composition,
        dirichlet_alpha=(args.dirichlet_alpha if args.composition == "dirichlet" else None),
        effect=args.effect,
        effect_indices=effect_indices,
        effect_fold=args.effect_fold,
        sampling_model=args.sampling,
        null_model=args.null,
        store_null_draws=store_null,
        executor=args.executor,
        show_progress=(str(args.progress).lower() in {"1", "true", "yes"}),
        expand_union=(str(args.expand_union).lower() in {"1", "true", "yes"}),
        warn_eps=float(args.warn_eps),
        entropy_method=args.entropy_method,
        lambda_mode=args.lambda_mode,
        lambda_model_path=(args.lambda_model or None),
    )

    if args.outdir:
        out_dir = args.outdir
        os.makedirs(out_dir, exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("benchmarks", "out", f"{ts}_{args.tag}")
        os.makedirs(out_dir, exist_ok=True)

    res = run_simulation_grid(cfg)
    if isinstance(res, tuple):
        results_df, null_df = res
    else:
        results_df, null_df = res, None

    results_path = os.path.join(out_dir, "results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"wrote {results_path}")

    if null_df is not None:
        null_path = os.path.join(out_dir, "null.csv")
        null_df.to_csv(null_path, index=False)
        print(f"wrote {null_path}")

    cfg_path = os.path.join(out_dir, "config.json")
    with open(cfg_path, "w") as fh:
        json.dump({
            "K": cfg.K,
            "N1": cfg.N1,
            "N2": cfg.N2,
            "s_values": list(cfg.s_values),
            "iters": cfg.iters,
            "num_boot": cfg.num_boot,
            "random_state": cfg.random_state,
            "pseudocount_mode": cfg.pseudocount_mode,
            "composition": cfg.composition,
            "dirichlet_alpha": cfg.dirichlet_alpha,
            "effect": cfg.effect,
            "effect_indices": cfg.effect_indices,
            "effect_fold": cfg.effect_fold,
            "sampling_model": cfg.sampling_model,
            "null_model": cfg.null_model,
            "store_null_draws": cfg.store_null_draws,
            "executor": cfg.executor,
        }, fh, indent=2)
    print(f"wrote {cfg_path}")

    # Optional: write R-compatible CSVs
    if str(args.r_compat).lower() in {"1", "true", "yes"}:
        ds_label = f"{args.N1}_vs_{args.N2}"
        # Build results_r.csv with exact R column names/order
        df_r = results_df.copy()
        if "chi_sq" not in df_r.columns and "chi2" in results_df.columns:
            df_r["chi_sq"] = results_df["chi2"].values
        if ("chi_sq" in df_r.columns) and ("chi_neg_log_p" not in df_r.columns):
            # Prefer robust log-survival when we can infer dof
            rows_obs = results_df.get("observed_clusters", results_df.get("total_clusters", np.nan))
            if rows_obs is not None and rows_obs.notna().any():
                try:
                    from scipy import stats as _sps
                    dof = np.maximum(rows_obs.astype(float) - 1.0, 1.0)
                    df_r["chi_neg_log_p"] = -_sps.chi2.logsf(df_r["chi_sq"].to_numpy(dtype=float), dof.to_numpy(dtype=float)) / np.log(10.0)
                except Exception:
                    if "chi2_p" in results_df.columns:
                        df_r["chi_neg_log_p"] = -np.log10(results_df["chi2_p"])  # no clipping
            elif "chi2_p" in results_df.columns:
                df_r["chi_neg_log_p"] = -np.log10(results_df["chi2_p"])  # no clipping
        df_r["dataset_sizes"] = ds_label
        df_r["iter"] = df_r.get("iter", 0)
        df_r["total_cells"] = args.N1 + args.N2
        df_r["total_number_of_clusters"] = df_r.get("observed_clusters", df_r.get("total_clusters", np.nan))
        df_r["b1_size"], df_r["b2_size"] = args.N1, args.N2
        df_r["batch_size_difference"] = abs(args.N1 - args.N2)
        df_r["min_of_expected_mat"] = df_r.get("min_E", np.nan)
        df_r["null_distribution_lambda_estimate"] = df_r.get("pmd_lambda", np.nan)
        df_r["num_non_shared_clusters"] = df_r.get("s", np.nan)
        df_r["raw_pmd"] = df_r.get("pmd_raw", np.nan)
        # Use bias-corrected V under the R name
        if "cramers_v_bc" in df_r.columns:
            df_r["cramers_v"] = df_r["cramers_v_bc"]
        # Map inverse simpson and entropy
        df_r["inverse_simp"] = df_r.get("inv_simpson", np.nan)
        df_r["shannon_entropy"] = df_r.get("entropy", np.nan)
        # Select and order columns
        cols_r = [
            "dataset_sizes","iter","total_cells","total_number_of_clusters","b1_size","b2_size",
            "batch_size_difference","min_of_expected_mat","null_distribution_lambda_estimate",
            "num_non_shared_clusters","raw_pmd","pmd","chi_sq","chi_neg_log_p","cramers_v",
            "inverse_simp","shannon_entropy",
        ]
        df_r = df_r.reindex(columns=cols_r)
        # Sort by iter then s to resemble R appends
        if "num_non_shared_clusters" in df_r.columns:
            df_r = df_r.sort_values(by=["iter","num_non_shared_clusters"]).reset_index(drop=True)
        out_r_path = os.path.join(out_dir, "results_r.csv")
        df_r.to_csv(out_r_path, index=False)
        print(f"wrote {out_r_path}")

        # Null: keep only s==0 and exactly r_num_sim per iter
        if null_df is not None and not null_df.empty:
            s0 = null_df[null_df.get("s", 0) == 0].copy()
            # attach dataset_sizes
            s0["dataset_sizes"] = ds_label
            # ensure r_num_sim per iter by sampling with replacement if needed
            out_null_rows = []
            for it, sub in s0.groupby("iter"):
                vals = sub["pmd_null"].to_numpy()
                if vals.size == 0:
                    continue
                if vals.size >= args.r_num_sim:
                    take = vals[: args.r_num_sim]
                else:
                    idx = np.random.default_rng(int(args.seed)).choice(np.arange(vals.size), size=args.r_num_sim, replace=True)
                    take = vals[idx]
                out_null_rows.extend([{"dataset_sizes": ds_label, "pmd_null": float(v)} for v in take])
            null_r = pd.DataFrame(out_null_rows, columns=["dataset_sizes","pmd_null"]) if out_null_rows else pd.DataFrame(columns=["dataset_sizes","pmd_null"])
            null_r_path = os.path.join(out_dir, "null_r.csv")
            null_r.to_csv(null_r_path, index=False)
            print(f"wrote {null_r_path}")


if __name__ == "__main__":
    main()
