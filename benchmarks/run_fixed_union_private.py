#!/usr/bin/env python3
"""Sweep percent overlap with fixed union size using the 'private cluster' modality.

Batch1: all mass in cluster 0 (overlap).
Batch2: (1 - p) mass in cluster 0, and p spread evenly over the other K-1 clusters.

Outputs: private_overlap_results.csv and private_overlap.png.
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

from percent_max_diff.benchmarking import compute_pmd_metrics, cramers_v_stat, inverse_simpson_per_column, shannon_entropy_across_columns


def parse_list_float(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--N1", type=int, default=2000)
    ap.add_argument("--N2", type=int, default=2000)
    ap.add_argument("--percents", type=str, default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--num-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--entropy-method", choices=["hypergeo","bootstrap"], default="hypergeo")
    ap.add_argument("--tag", type=str, default="private_overlap")
    return ap.parse_args()


def simulate_fixed_union_private(K: int, p: float, N1: int, N2: int, rng: np.random.Generator) -> np.ndarray:
    p1 = np.zeros(K, dtype=float); p1[0] = 1.0
    p2 = np.zeros(K, dtype=float)
    p2[0] = max(0.0, 1.0 - p)
    if K > 1:
        p2[1:] = p / (K - 1)
    x1 = rng.multinomial(int(N1), p1).astype(int)
    x2 = rng.multinomial(int(N2), p2).astype(int)
    return np.stack([x1, x2], axis=1)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    pcts = parse_list_float(args.percents)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("benchmarks", "out", f"{ts}_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for p in pcts:
        for it in range(args.iters):
            X = simulate_fixed_union_private(args.K, p, args.N1, args.N2, rng)
            raw, pmdd, lam, _ = compute_pmd_metrics(X, num_boot=args.num_boot, null_model="permutation", rng=rng)
            chi2, pval, v = cramers_v_stat(X, bias_correction=False)
            _, _, vbc = cramers_v_stat(X, bias_correction=True)
            invs = inverse_simpson_per_column(X)
            ent = shannon_entropy_across_columns(X, downsample=True, downsample_method=args.entropy_method, rng=rng)
            rows.append({
                "percent_overlap_private": 1.0 - p,
                "percent_difference": p,
                "iter": it,
                "raw_pmd": raw,
                "pmd": pmdd,
                "chi_sq": chi2,
                # Robust -log10(p) via log survival; dof = (r-1)(c-1) with c=2 (clamp inf to float max)
                "chi_neg_log_p": (lambda v: (np.finfo(float).max if not np.isfinite(v) else v))(-stats.chi2.logsf(chi2, max(int(X.shape[0]) - 1, 1)) / np.log(10.0)),
                "cramers_v": v,
                "cramers_v_bc": vbc,
                "inverse_simp": invs,
                "shannon_entropy": ent,
                "K": args.K,
                "N1": args.N1,
                "N2": args.N2,
                "total_number_of_clusters": int((X.sum(axis=1) > 0).sum()),
            })

    df = pd.DataFrame.from_records(rows)
    out_csv = os.path.join(out_dir, "private_overlap_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    sns.set_context("talk")
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    ax = axes.ravel()
    sns.lineplot(data=df, x="percent_difference", y="raw_pmd", estimator="mean", errorbar="sd", ax=ax[0])
    ax[0].set_title("Raw PMD vs % difference (fixed union)")
    sns.lineplot(data=df, x="percent_difference", y="pmd", estimator="mean", errorbar="sd", ax=ax[1])
    ax[1].set_title("PMD* vs % difference")
    sns.lineplot(data=df, x="percent_difference", y="chi_sq", estimator="mean", errorbar="sd", ax=ax[2])
    ax[2].set_title("Chi-square vs % difference")
    # Bias-corrected Cramer's V only; no fallback
    if "cramers_v_bc" in df.columns:
        sns.lineplot(data=df, x="percent_difference", y="cramers_v_bc", estimator="mean", errorbar="sd", ax=ax[3])
        ax[3].set_title("Cramer's V (BC) vs % difference")
    else:
        ax[3].axis("off"); ax[3].text(0.5, 0.5, "cramers_v_bc missing", ha="center", va="center")
    sns.lineplot(data=df, x="percent_difference", y="inverse_simp", estimator="mean", errorbar="sd", ax=ax[4])
    ax[4].set_title("Inv Simpson vs % difference")
    sns.lineplot(data=df, x="percent_difference", y="shannon_entropy", estimator="mean", errorbar="sd", ax=ax[5])
    ax[5].set_title("Shannon entropy vs % difference")
    plt.tight_layout()
    out_png = os.path.join(out_dir, "private_overlap.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
