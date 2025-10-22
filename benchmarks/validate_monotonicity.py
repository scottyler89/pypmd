#!/usr/bin/env python3
"""Check monotonicity trends vs s for key metrics.

We compute per-s means and verify that metrics generally increase with s
(on average), tolerating small violations.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


def is_monotone_non_decreasing(arr: np.ndarray, tol: float = 1e-9, allow_dips: int = 1) -> bool:
    dips = 0
    for i in range(1, len(arr)):
        if arr[i] + tol < arr[i - 1]:
            dips += 1
            if dips > allow_dips:
                return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="path to results.csv")
    ap.add_argument("--metrics", type=str, default="pmd,jsd,tv,hellinger,braycurtis,cosine")
    ap.add_argument("--allow-dips", type=int, default=1)
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    ok = True
    if "s" not in df.columns:
        print("results.csv lacks 's' column; cannot assess monotonicity")
        return

    means = df.groupby("s")[metrics].mean(numeric_only=True).sort_index()
    for m in metrics:
        if m not in means.columns:
            print(f"skip {m} (not present)")
            continue
        arr = means[m].to_numpy()
        passed = is_monotone_non_decreasing(arr, allow_dips=args.allow_dips)
        print(f"{m}: {'OK' if passed else 'FAIL'} — per-s means: {arr}")
        ok = ok and passed

    if ok:
        print("SUCCESS: monotonicity checks passed (within tolerance)")
    else:
        print("WARN: some monotonicity checks failed; review trends")


if __name__ == "__main__":
    main()

