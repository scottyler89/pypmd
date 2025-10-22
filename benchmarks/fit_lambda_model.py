#!/usr/bin/env python3
"""Fit a lambda ~ f(log2(min(E))) smoother and save to CSV.

Usage:
  python benchmarks/fit_lambda_model.py --dir benchmarks/out/<run_dir>
  python benchmarks/fit_lambda_model.py results.csv

Outputs lambda_model.csv (columns: log2_minE, lambda_smooth) next to results.
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    _HAS_LOWESS = True
except Exception:
    _HAS_LOWESS = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="", help="path to results.csv or omit with --dir")
    ap.add_argument("--dir", default="", help="run directory containing results.csv")
    ap.add_argument("--points", type=int, default=200, help="number of grid points for the smoother curve")
    args = ap.parse_args()

    if args.dir:
        res_path = os.path.join(args.dir, "results.csv")
    else:
        res_path = args.results
    if not os.path.exists(res_path):
        raise SystemExit(f"results.csv not found: {res_path}")

    df = pd.read_csv(res_path)
    if "min_E" not in df.columns or "pmd_lambda" not in df.columns:
        raise SystemExit("results.csv must contain 'min_E' and 'pmd_lambda'")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["min_E", "pmd_lambda"])
    df = df[df["min_E"] > 0]
    df["log2_minE"] = np.log2(df["min_E"])

    x = df["log2_minE"].to_numpy()
    y = df["pmd_lambda"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 5:
        raise SystemExit("Not enough points to fit lambda model")

    x_grid = np.linspace(x.min(), x.max(), args.points)
    if _HAS_LOWESS:
        fitted = lowess(y, x, frac=0.6, return_sorted=True)
        # Interpolate onto grid
        lam_grid = np.interp(x_grid, fitted[:, 0], fitted[:, 1])
    else:
        # Fallback: rolling mean in bins
        bins = np.linspace(x.min(), x.max(), max(10, min(50, args.points // 4)))
        idx = np.digitize(x, bins)
        lam_bin = {b: y[idx == b].mean() for b in np.unique(idx)}
        lam_grid = np.interp(x_grid, bins, [lam_bin.get(i, np.nan) for i in range(len(bins))])

    out = pd.DataFrame({"log2_minE": x_grid, "lambda_smooth": lam_grid})
    out_path = os.path.join(os.path.dirname(os.path.abspath(res_path)), "lambda_model.csv")
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

