#!/usr/bin/env python3
"""Small-n parity spot checks vs R outputs.

Loads an R-compatible CSV produced by our runner (results_r.csv) and an R out_df
CSV, aligns by dataset_sizes and num_non_shared_clusters, and asserts tolerances
for key metrics on small datasets.
"""

from __future__ import annotations

import argparse
import sys
import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, help="path to our results_r.csv")
    ap.add_argument("--r", required=True, help="path to R out_df CSV")
    ap.add_argument("--rmse-threshold", type=float, default=0.05)
    ap.add_argument("--mae-threshold", type=float, default=0.03)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ours = pd.read_csv(args.ours)
    rdf = pd.read_csv(args.r)
    # Ensure required columns exist
    for col in ["dataset_sizes","num_non_shared_clusters"]:
        if col not in ours.columns or col not in rdf.columns:
            print(f"missing required column {col}")
            sys.exit(2)

    metrics = [
        ("raw_pmd","raw_pmd"),
        ("pmd","pmd"),
        ("chi_sq","chi_sq"),
        ("chi_neg_log_p","chi_neg_log_p"),
        ("cramers_v","cramers_v"),
        ("inverse_simp","inverse_simp"),
        ("shannon_entropy","shannon_entropy"),
    ]
    merged = pd.merge(ours, rdf, on=["dataset_sizes","num_non_shared_clusters"], suffixes=("_ours","_r"))
    fails = []
    for m_ours, m_r in metrics:
        if f"{m_ours}_ours" not in merged.columns or f"{m_r}_r" not in merged.columns:
            continue
        a = merged[f"{m_ours}_ours"].astype(float).to_numpy()
        b = merged[f"{m_r}_r"].astype(float).to_numpy()
        rmse = float(np.sqrt(np.nanmean((a-b)**2)))
        mae = float(np.nanmean(np.abs(a-b)))
        ok = (rmse <= args.rmse_threshold) and (mae <= args.mae_threshold)
        print(f"{m_ours}: RMSE={rmse:.4f}, MAE={mae:.4f} -> {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append((m_ours, rmse, mae))
    if fails:
        sys.exit(1)
    print("parity checks passed")


if __name__ == "__main__":
    main()

