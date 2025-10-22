#!/usr/bin/env python3
"""Compare our benchmark CSVs with R-generated CSVs and make parity plots.

Inputs:
- --ours: path to our results.csv (required)
- --r: path to R out_df CSV (required), e.g., columns like:
  dataset_sizes, iter, total_cells, total_number_of_clusters, b1_size, b2_size,
  min_of_expected_mat, null_distribution_lambda_estimate, num_non_shared_clusters,
  raw_pmd, pmd, chi_sq, chi_neg_log_p, cramers_v, inverse_simp, shannon_entropy

Outputs:
- CSV with per-metric RMSE/MAE per dataset_sizes
- Overlay plots for each metric vs s (saved next to our file under compare/)
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, help="path to our results.csv")
    ap.add_argument("--r", required=True, help="path to R out_df CSV")
    ap.add_argument("--outdir", default="", help="output directory (default next to ours)")
    return ap.parse_args()


def align_and_compare(ours: pd.DataFrame, rdf: pd.DataFrame) -> pd.DataFrame:
    # Normalize column names
    ours = ours.copy()
    ours = ours.rename(columns={
        "s": "num_non_shared_clusters",
        "min_E": "min_of_expected_mat",
        "pmd_lambda": "null_distribution_lambda_estimate",
        "total_clusters": "total_number_of_clusters",
    })
    # Required keys
    key_cols = ["dataset_sizes", "num_non_shared_clusters"]
    # Metrics to compare
    metrics = [
        ("raw_pmd", "raw_pmd"),
        ("pmd", "pmd"),
        ("chi2", "chi_sq"),
        ("chi_neg_log_p", "chi_neg_log_p"),
        ("cramers_v_bc" if "cramers_v_bc" in ours.columns else "cramers_v", "cramers_v"),
        ("inverse_simp", "inverse_simp"),
        ("entropy", "shannon_entropy"),
        ("null_distribution_lambda_estimate", "null_distribution_lambda_estimate"),
        ("min_of_expected_mat", "min_of_expected_mat"),
    ]
    ours_sub = ours[key_cols + [m[0] for m in metrics] if metrics else key_cols]
    r_sub = rdf[key_cols + [m[1] for m in metrics] if metrics else key_cols]
    merged = pd.merge(ours_sub, r_sub, on=key_cols, how="inner", suffixes=("_ours", "_r"))
    rows = []
    for ds, group in merged.groupby("dataset_sizes"):
        rec: Dict[str, float] = {"dataset_sizes": ds}
        for m_ours, m_r in metrics:
            if (m_ours + "_ours") not in group.columns or (m_r + "_r") not in group.columns:
                continue
            a = group[m_ours + "_ours"].to_numpy(dtype=float)
            b = group[m_r + "_r"].to_numpy(dtype=float)
            diffs = a - b
            rec[f"RMSE_{m_ours}"] = float(np.sqrt(np.nanmean(diffs**2)))
            rec[f"MAE_{m_ours}"] = float(np.nanmean(np.abs(diffs)))
        rows.append(rec)
    return pd.DataFrame.from_records(rows)


def overlay_plots(ours: pd.DataFrame, rdf: pd.DataFrame, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    # Map columns
    ours = ours.rename(columns={
        "s": "num_non_shared_clusters",
        "min_E": "min_of_expected_mat",
        "pmd_lambda": "null_distribution_lambda_estimate",
    })
    vcol = "cramers_v_bc" if "cramers_v_bc" in ours.columns else "cramers_v"
    metrics = [
        ("raw_pmd", "raw_pmd", "Raw PMD"),
        ("pmd", "pmd", "PMD*"),
        ("chi2", "chi_sq", "Chi-square"),
        ("chi_neg_log_p", "chi_neg_log_p", "-log10 p"),
        (vcol, "cramers_v", "Cramer's V"),
        ("inverse_simp", "inverse_simp", "Inverse Simpson"),
        ("entropy", "shannon_entropy", "Shannon entropy"),
    ]
    for m_ours, m_r, title in metrics:
        if m_ours not in ours.columns or m_r not in rdf.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(data=rdf, x="num_non_shared_clusters", y=m_r, hue="dataset_sizes", style="dataset_sizes", ax=ax, dashes=True)
        sns.lineplot(data=ours, x="num_non_shared_clusters", y=m_ours, hue="dataset_sizes", style="dataset_sizes", ax=ax, dashes=False, legend=False)
        ax.set_title(f"Parity: {title}")
        out_path = os.path.join(outdir, f"parity_{m_ours}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    ours = pd.read_csv(args.ours)
    rdf = pd.read_csv(args.r)
    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.ours)), "compare")

    summary = align_and_compare(ours, rdf)
    summary_path = os.path.join(outdir, "summary_diffs.csv")
    os.makedirs(outdir, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")

    overlay_plots(ours, rdf, outdir)
    print(f"wrote overlay plots to {outdir}")


if __name__ == "__main__":
    main()

