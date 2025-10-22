#!/usr/bin/env python3
"""Summarize a results.csv from PMD benchmarking.

Usage:
  python benchmarks/summarize_results.py path/to/results.csv
"""

from __future__ import annotations

import argparse
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="path to results.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    group_cols = [c for c in ["K", "N1", "N2", "s"] if c in df.columns]
    metric_cols = [
        c
        for c in [
            "pmd_raw",
            "pmd",
            "chi2",
            "chi2_p",
            "cramers_v",
            "inv_simpson",
            "entropy",
            "jsd",
            "tv",
            "hellinger",
            "braycurtis",
            "canberra",
            "cosine",
        ]
        if c in df.columns
    ]
    agg = (
        df.groupby(group_cols)[metric_cols]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    print(agg.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

