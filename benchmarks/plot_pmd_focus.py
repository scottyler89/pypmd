#!/usr/bin/env python3
"""Standalone PMD-focused figures: raw PMD and debiased PMD vs s.

Usage:
  python benchmarks/plot_pmd_focus.py --dir <run_dir>
  or: python benchmarks/plot_pmd_focus.py results.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    _HAS_LOWESS = True
except Exception:
    _HAS_LOWESS = False


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="", help="path to results.csv or omit when using --dir")
    ap.add_argument("--dir", default="", help="run directory containing results.csv")
    ap.add_argument("--out", default="", help="output path (default: <dir>/pmd_focus.png)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = ""
    res_path = args.results
    if args.dir:
        base_dir = args.dir
        res_path = os.path.join(base_dir, "results.csv")
    elif os.path.isdir(res_path):
        base_dir = res_path
        res_path = os.path.join(base_dir, "results.csv")
    if not os.path.exists(res_path):
        raise SystemExit(f"results.csv not found: {res_path}")

    df = pd.read_csv(res_path).copy()
    if "s" not in df.columns:
        raise SystemExit("results.csv missing 's' column for PMD focus plots")
    df["dataset_label"] = df.apply(lambda r: f"K={int(r['K'])}, N1={int(r['N1'])}, N2={int(r['N2'])}", axis=1)
    s_max = int(np.nanmax(df["s"]))

    sns.set_context("talk")
    # Stable palette by dataset
    labels = sorted(df["dataset_label"].dropna().unique())
    palette = sns.color_palette("tab10", n_colors=len(labels))
    color_map = {lab: palette[i] for i, lab in enumerate(labels)}
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    ax_raw, ax_pmd = axes

    def loess(axh, metric, title, bounds=(0, 1)):
        sns.scatterplot(data=df, x="s", y=metric, hue="dataset_label", palette=color_map, alpha=0.25, s=20, ax=axh, legend=False)
        for label, sub in df.groupby("dataset_label"):
            xs = sub["s"].to_numpy(dtype=float)
            ys = sub[metric].to_numpy(dtype=float)
            ok = np.isfinite(xs) & np.isfinite(ys)
            xs, ys = xs[ok], ys[ok]
            if xs.size < 2:
                continue
            if _HAS_LOWESS:
                fitted = sm_lowess(ys, xs, frac=0.6, return_sorted=True)
                axh.plot(fitted[:, 0], fitted[:, 1], lw=2, color=color_map.get(label))
            else:
                m = sub.groupby("s")[metric].mean(numeric_only=True)
                axh.plot(m.index.values, m.values, lw=2, color=color_map.get(label))
        axh.set_title(title)
        axh.set_xlim(0, s_max)
        if bounds is not None:
            axh.set_ylim(*bounds)

    if "pmd_raw" in df.columns:
        loess(ax_raw, "pmd_raw", "Raw PMD vs non-shared clusters", bounds=(0, 1))
        ax_raw.set_ylabel("raw PMD")
    if "pmd" in df.columns:
        loess(ax_pmd, "pmd", "Debiased PMD vs non-shared clusters", bounds=(0, 1))
        ax_pmd.set_ylabel("PMD*")
        ax_pmd.set_xlabel("non-shared clusters (s)")

    out = args.out or os.path.join(base_dir or os.path.dirname(os.path.abspath(res_path)), "pmd_focus.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
