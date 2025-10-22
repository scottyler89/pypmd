#!/usr/bin/env python3
"""Replicate the 'characterize_invariance_property' R experiment.

Sweeps:
- Number of clusters per batch (K range)
- Percent difference in clusters (0..1)
- Two modes: expand_b2_only (batch1 concentrated in a single cluster) or symmetric

Outputs: invariance_results.csv and invariance.png in an output directory.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import List

import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from percent_max_diff.benchmarking import (
    _downsample_column_hypergeo,
    compute_pmd_metrics,
    cramers_v_stat,
    inverse_simpson_per_column,
    shannon_entropy_across_columns,
    js_distance,
    total_variation,
    hellinger as hellinger_dist,
    bray_curtis,
    canberra as canberra_dist,
    cosine_distance,
)


def parse_list_float(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def parse_list_int(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def simulate_invariance_counts(
    K: int,
    pcnt_diff: float,
    mode: str,
    N1: int,
    N2: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate invariance modes.

    Modes:
      - expand_b2_only: batch1 all mass in overlap cluster; batch2 splits remainder across K private clusters
      - symmetric: uniform K for batch1, non-wrapping shift for batch2 (union grows)
      - fixed_union_private: keep union fixed at K; batch1 all mass in one cluster; batch2 allocates (1-p) to that overlap cluster and p across remaining K-1 clusters
    """
    if mode == "expand_b2_only":
        # P1 has a single overlap cluster (index 0)
        p1 = np.array([1.0])
        # P2 splits (1-p) on overlap, remainder equally across K clusters
        p2 = np.concatenate([[1.0 - pcnt_diff], np.full(K, pcnt_diff / max(1, K))])
        # Build union counts: rows = 1 + K
        x1 = np.zeros(1 + K, dtype=int)
        x1[0] = N1
        x2 = np.random.multinomial(int(N2), p2).astype(int)
        X = np.stack([x1, x2], axis=1)
        return X
    elif mode == "symmetric":
        # symmetric case: uniform K vs shifted uniform (non-wrapping)
        base = np.full(K, 1.0 / K)
        shift = int(round(K * pcnt_diff))
        newK = K + max(0, shift)
        p1_pad = np.zeros(newK, dtype=float)
        p2_pad = np.zeros(newK, dtype=float)
        p1_pad[0:K] = base
        p2_pad[shift : shift + K] = base
        x1 = np.random.multinomial(int(N1), p1_pad).astype(int)
        x2 = np.random.multinomial(int(N2), p2_pad).astype(int)
        X = np.stack([x1, x2], axis=1)
        return X
    elif mode == "fixed_union_private":
        # Keep union fixed at K; batch1 uses only cluster 0; batch2 shares overlap with cluster 0 and spreads remainder over others
        p1 = np.zeros(K, dtype=float); p1[0] = 1.0
        p2 = np.zeros(K, dtype=float)
        p2[0] = max(0.0, 1.0 - pcnt_diff)
        if K > 1:
            p2[1:] = pcnt_diff / (K - 1)
        x1 = np.random.multinomial(int(N1), p1).astype(int)
        x2 = np.random.multinomial(int(N2), p2).astype(int)
        X = np.stack([x1, x2], axis=1)
        return X
    else:
        raise ValueError("unknown invariance mode")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clust-range", type=str, default="4,8,12,16")
    ap.add_argument("--percent-difference", type=str, default="0,0.25,0.5,0.75,1.0")
    # Default dataset sizes to match original R script
    ap.add_argument("--b1-sizes", type=str, default="10000,10000,1000,250,250,100")
    ap.add_argument("--b2-sizes", type=str, default="10000,1000,1000,2000,250,100")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--expand-b2-only", type=str, default="false")
    ap.add_argument("--mode", choices=["expand_b2_only","symmetric","fixed_union_private"], default="symmetric")
    ap.add_argument("--num-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="invariance")
    ap.add_argument("--r-compat", type=str, default="false")
    ap.add_argument("--entropy-method", choices=["hypergeo","bootstrap"], default="hypergeo")
    ap.add_argument("--outdir", type=str, default="", help="optional explicit output directory")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    K_list = parse_list_int(args.num_clust_range)
    pdiffs = parse_list_float(args.percent_difference)
    N1s = parse_list_int(args.b1_sizes)
    N2s = parse_list_int(args.b2_sizes)
    expand_b2_only = str(args.expand_b2_only).lower() in {"1", "true", "yes"}
    mode = ("expand_b2_only" if expand_b2_only else args.mode)

    if args.outdir:
        out_dir = args.outdir
        os.makedirs(out_dir, exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("benchmarks", "out", f"{ts}_{args.tag}")
        os.makedirs(out_dir, exist_ok=True)

    rows = []
    for N1, N2 in zip(N1s, N2s):
        ds_label = f"{N1}_vs_{N2}"
        for K in K_list:
            for pcnt in pdiffs:
                for it in range(args.iters):
                    X = simulate_invariance_counts(K, pcnt, mode, N1, N2, rng)
                    raw, pmd, lam, _ = compute_pmd_metrics(X, num_boot=args.num_boot, null_model="permutation", rng=rng)
                    # For chi-square & Cramer's V, drop all-zero rows to avoid expected zero errors
                    mask = (X.sum(axis=1) > 0)
                    X_chi = X[mask, :]
                    if X_chi.shape[0] >= 2:
                        chi2, pval, v = cramers_v_stat(X_chi, bias_correction=False)
                        _, _, vbc = cramers_v_stat(X_chi, bias_correction=True)
                    else:
                        chi2, pval, v, vbc = (float('nan'), float('nan'), float('nan'), float('nan'))
                    inv_simp = inverse_simpson_per_column(X)
                    ent = shannon_entropy_across_columns(X, downsample=True, downsample_method=args.entropy_method, rng=rng)

                    rows.append({
                        "dataset_sizes": ds_label,
                        "iter": it,
                        "NumberOfClusters": K,
                        "percent_different_clusters_numeric": pcnt,
                        "mode": mode,
                        "raw_pmd": raw,
                        "pmd": pmd,
                        "chi_sq": chi2,
                        # Robust -log10(p) via log survival; dof = (r-1)(c-1) with c=2
                        "chi_neg_log_p": (
                            (-stats.chi2.logsf(chi2, max(int(X_chi.shape[0]) - 1, 1)) / np.log(10.0)) if np.isfinite(chi2) else float('nan')
                        ),
                        "cramers_v": v,
                        "cramers_v_bc": vbc,
                        "inverse_simp": inv_simp,
                        "shannon_entropy": ent,
                        # Comparator distances (probability-based)
                        "jsd": js_distance(X),
                        "tv": total_variation(X),
                        "hellinger": hellinger_dist(X),
                        "braycurtis": bray_curtis(X),
                        "canberra": canberra_dist(X),
                        "cosine": cosine_distance(X),
                        "b1_size": N1,
                        "b2_size": N2,
                        "total_number_of_clusters": int((X.sum(axis=1) > 0).sum()),
                    })

    df = pd.DataFrame.from_records(rows)
    out_csv = os.path.join(out_dir, "invariance_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    # Optional R-compatible invariance CSV
    if str(args.r_compat).lower() in {"1", "true", "yes"}:
        dfr = df.copy()
        dfr["total_cells"] = dfr["b1_size"] + dfr["b2_size"]
        # Factors in R appear as separate columns; we include both numeric/factor-like
        dfr["PrcntDiffClusters"] = dfr["percent_different_clusters_numeric"].astype(str)
        dfr["NumberClusters"] = dfr["NumberOfClusters"].astype(str)
        # Prefer bias-corrected V
        if "cramers_v_bc" in dfr.columns:
            dfr["cramers_v"] = dfr["cramers_v_bc"]
        cols_r = [
            "dataset_sizes","PrcntDiffClusters","percent_different_clusters_numeric",
            "NumberClusters","NumberOfClusters","iter","total_cells","total_number_of_clusters",
            "b1_size","b2_size","raw_pmd","pmd","chi_sq","chi_neg_log_p","cramers_v",
            "inverse_simp","shannon_entropy",
        ]
        # Ensure chi_sq naming
        if "chi_sq" not in dfr.columns and "chi_sq" in dfr.rename(columns={"chi2":"chi_sq"}).columns:
            dfr = dfr.rename(columns={"chi2":"chi_sq"})
        dfr = dfr.reindex(columns=cols_r)
        out_r = os.path.join(out_dir, "invariance_results_r.csv")
        dfr.to_csv(out_r, index=False)
        print(f"wrote {out_r}")

    # Plot a grid similar to the R layout (subset of panels)
    sns.set_context("talk")
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=False)
    ax = axes.ravel()
    # 1) Raw PMD vs NumberOfClusters (colored by percent diff, linetype by dataset)
    sns.lineplot(data=df, x="NumberOfClusters", y="raw_pmd", hue="percent_different_clusters_numeric", style="dataset_sizes", markers=False, dashes=True, ax=ax[0])
    ax[0].set_title("Raw PMD vs #Clusters")
    # 2) PMD* vs NumberOfClusters
    sns.lineplot(data=df, x="NumberOfClusters", y="pmd", hue="percent_different_clusters_numeric", style="dataset_sizes", markers=False, dashes=True, ax=ax[1], legend=False)
    ax[1].set_title("PMD* vs #Clusters")
    # 3) Chi-square vs NumberOfClusters
    sns.lineplot(data=df, x="NumberOfClusters", y="chi_sq", hue="percent_different_clusters_numeric", style="dataset_sizes", markers=False, dashes=True, ax=ax[2], legend=False)
    ax[2].set_title("Chi-square vs #Clusters")
    # 4) -log10 p vs NumberOfClusters
    sns.lineplot(data=df, x="NumberOfClusters", y="chi_neg_log_p", hue="percent_different_clusters_numeric", style="dataset_sizes", markers=False, dashes=True, ax=ax[3], legend=False)
    ax[3].set_title("-log10 p vs #Clusters")
    # 5) Cramer's V (bias-corrected) vs NumberOfClusters — no fallback
    if "cramers_v_bc" in df.columns:
        sns.lineplot(data=df, x="NumberOfClusters", y="cramers_v_bc", hue="percent_different_clusters_numeric", style="dataset_sizes", markers=False, dashes=True, ax=ax[4], legend=False)
        ax[4].set_title("Cramer's V (BC) vs #Clusters")
    else:
        ax[4].axis("off"); ax[4].text(0.5, 0.5, "cramers_v_bc missing", ha="center", va="center")
    # 6) Inverse Simpson vs NumberOfClusters
    sns.lineplot(data=df, x="NumberOfClusters", y="inverse_simp", hue="percent_different_clusters_numeric", style="dataset_sizes", markers=False, dashes=True, ax=ax[5], legend=False)
    ax[5].set_title("Inv Simpson vs #Clusters")
    # 7) Shannon entropy vs NumberOfClusters
    sns.lineplot(data=df, x="NumberOfClusters", y="shannon_entropy", hue="percent_different_clusters_numeric", style="dataset_sizes", markers=False, dashes=True, ax=ax[6], legend=False)
    ax[6].set_title("Shannon Entropy vs #Clusters")
    # 8) PMD* vs percent difference (faceted by NumberOfClusters)
    # For simplicity, show one combined axes summarizing by NumberOfClusters
    sns.lineplot(data=df, x="percent_different_clusters_numeric", y="pmd", hue="NumberOfClusters", style="dataset_sizes", markers=False, dashes=True, ax=ax[7])
    ax[7].set_title("PMD* vs % Different Clusters")

    plt.tight_layout()
    out_png = os.path.join(out_dir, "invariance.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}")
    # Also build the 2x6 super-grid using the dedicated plotter
    try:
        import subprocess, sys
        plot_cmd = [
            sys.executable,
            os.path.join("benchmarks", "plot_invariance.py"),
            "--file", out_csv,
            "--full-grid", "true",
        ]
        subprocess.run(plot_cmd, check=True)
    except Exception:
        print("warning: failed to render invariance super grid; results CSV is available")


if __name__ == "__main__":
    main()
