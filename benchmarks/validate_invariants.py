#!/usr/bin/env python3
"""Validate basic metric invariants for a results CSV produced by benchmarking.

Checks bounds/symmetry-like properties (where applicable) aggregated per s.
"""

from __future__ import annotations

import argparse
import math
import pandas as pd


def _within(x, lo, hi):
    try:
        return (x >= lo) and (x <= hi)
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="path to results.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    ok = True

    def check_col(col, lo, hi):
        nonlocal ok
        if col not in df.columns:
            return
        vals = df[col].dropna()
        bad = vals[~vals.apply(lambda v: _within(v, lo, hi))]
        if len(bad) > 0:
            ok = False
            print(f"FAIL: {col} has {len(bad)} out-of-bound values [{lo},{hi}] e.g., {bad.iloc[0]}")
        else:
            print(f"OK: {col} within [{lo},{hi}] over {len(vals)} values")

    # Bounds checks
    check_col("pmd", 0.0, 1.0)
    check_col("pmd_raw", 0.0, 1.0)
    check_col("cramers_v", 0.0, 1.0)
    check_col("chi2_p", 0.0, 1.0)
    check_col("jsd", 0.0, 1.0)
    check_col("tv", 0.0, 1.0)
    check_col("hellinger", 0.0, 1.0)
    check_col("braycurtis", 0.0, 1.0)
    check_col("cosine", 0.0, 1.0)
    # Canberra is non-negative; no strict upper bound
    if "canberra" in df.columns:
        bad = df["canberra"].dropna()
        neg = bad[bad < 0]
        if len(neg) > 0:
            ok = False
            print(f"FAIL: canberra has {len(neg)} negative values")
        else:
            print(f"OK: canberra non-negative over {len(bad)} values")

    # Entropy (base-2, 2 columns) bounded in [0,1]
    check_col("entropy", 0.0, 1.0)

    # Inverse Simpson should be >= 1
    if "inv_simpson" in df.columns:
        vals = df["inv_simpson"].dropna()
        neg = vals[vals < 1.0]
        if len(neg) > 0:
            ok = False
            print(f"WARN: inv_simpson has {len(neg)} values < 1.0 (could indicate degenerate columns)")
        else:
            print(f"OK: inv_simpson >= 1 over {len(vals)} values")

    if ok:
        print("SUCCESS: invariants passed")
    else:
        print("FAIL: one or more invariant checks failed")


if __name__ == "__main__":
    main()

