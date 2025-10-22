#!/usr/bin/env python3
"""Standalone lambda-focused figures.

Outputs in the run directory by default:
  - lambda_density.png           (KDE of pmd_lambda, colored by dataset)
  - lambda_by_predictors.png     (2 panels: lambda vs total_clusters; lambda vs log2(min(E)))
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
    ap.add_argument("--density-out", default="", help="optional output for lambda_density.png")
    ap.add_argument("--predictors-out", default="", help="optional output for lambda_by_predictors.png")
    ap.add_argument("--violin-by", choices=["", "K", "s"], default="", help="optional: emit lambda_violin_by_<col>.png grouped by this column")
    ap.add_argument("--ridgeline-by", choices=["", "K", "s"], default="", help="optional: emit lambda_ridgeline_by_<col>.png faceted KDE rows by this column")
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
    if "pmd_lambda" not in df.columns:
        raise SystemExit("results.csv missing 'pmd_lambda' for lambda plots")
    # Determine a robust hue label for grouping across datasets
    if "dataset_label" in df.columns:
        hue_col = "dataset_label"
    elif "dataset_sizes" in df.columns:
        hue_col = "dataset_sizes"
    else:
        hue_col = "dataset_label"
        df[hue_col] = df.apply(lambda r: f"K={int(r['K'])}, N1={int(r['N1'])}, N2={int(r['N2'])}", axis=1)
    if "min_E" in df.columns:
        df["log2_minE"] = np.log2(df["min_E"].replace(0, np.nan))

    sns.set_context("talk")
    # Stable palette by dataset
    labels = sorted(df[hue_col].dropna().unique())
    palette = sns.color_palette("tab10", n_colors=len(labels))
    color_map = {lab: palette[i] for i, lab in enumerate(labels)}
    # Lambda density
    fig_d, ax_d = plt.subplots(1, 1, figsize=(8, 4.5))
    sns.kdeplot(data=df, x="pmd_lambda", hue=hue_col, palette=color_map, bw_adjust=1.5, fill=True, alpha=0.3, ax=ax_d)
    ax_d.set_xlim(0, 1)
    ax_d.set_title("Lambda density")
    ax_d.set_ylabel("density")
    out_d = args.density_out or os.path.join(base_dir or os.path.dirname(os.path.abspath(res_path)), "lambda_density.png")
    fig_d.savefig(out_d, dpi=150, bbox_inches="tight"); plt.close(fig_d)
    print(f"wrote {out_d}")

    # Lambda vs predictors
    fig_p, axes_p = plt.subplots(1, 2, figsize=(14, 5))
    ax1, ax2 = axes_p
    if {"pmd_lambda", "total_clusters"}.issubset(df.columns):
        sns.scatterplot(data=df, x="total_clusters", y="pmd_lambda", hue=hue_col, palette=color_map, alpha=0.3, s=20, ax=ax1, legend=False)
        for label, sub in df.groupby(hue_col):
            xs = sub["total_clusters"].to_numpy(); ys = sub["pmd_lambda"].to_numpy()
            ok = np.isfinite(xs) & np.isfinite(ys)
            xs, ys = xs[ok], ys[ok]
            if xs.size >= 3 and _HAS_LOWESS:
                fitted = sm_lowess(ys, xs, frac=0.6, return_sorted=True)
                ax1.plot(fitted[:, 0], fitted[:, 1], lw=2, color=color_map.get(label))
        # lock x bounds
        try:
            xmin = float(np.nanmin(df["total_clusters"])); xmax = float(np.nanmax(df["total_clusters"]))
            ax1.set_xlim(xmin, xmax)
        except Exception:
            pass
        ax1.set_title("lambda vs total clusters"); ax1.set_ylabel("lambda (mean null PMD)")
    if {"pmd_lambda", "log2_minE"}.issubset(df.columns):
        sns.scatterplot(data=df, x="log2_minE", y="pmd_lambda", hue=hue_col, palette=color_map, alpha=0.3, s=20, ax=ax2, legend=False)
        for label, sub in df.groupby(hue_col):
            xs = sub["log2_minE"].to_numpy(); ys = sub["pmd_lambda"].to_numpy()
            ok = np.isfinite(xs) & np.isfinite(ys)
            xs, ys = xs[ok], ys[ok]
            if xs.size >= 3 and _HAS_LOWESS:
                fitted = sm_lowess(ys, xs, frac=0.8, return_sorted=True)
                ax2.plot(fitted[:, 0], fitted[:, 1], lw=2, color=color_map.get(label))
        try:
            xmin = float(np.nanmin(df["log2_minE"])); xmax = float(np.nanmax(df["log2_minE"]))
            ax2.set_xlim(xmin, xmax)
        except Exception:
            pass
        ax2.set_title("lambda vs log2(min(E))"); ax2.set_ylabel("lambda (mean null PMD)")
    out_p = args.predictors_out or os.path.join(base_dir or os.path.dirname(os.path.abspath(res_path)), "lambda_by_predictors.png")
    fig_p.tight_layout(); fig_p.savefig(out_p, dpi=150, bbox_inches="tight"); plt.close(fig_p)
    print(f"wrote {out_p}")

    # Optional violin by K or s
    if args.violin_by:
        col = args.violin_by
        if col in df.columns:
            fig_v, ax_v = plt.subplots(1, 1, figsize=(10, 5))
            sns.violinplot(data=df, x=col, y="pmd_lambda", hue=hue_col, palette=color_map, ax=ax_v, cut=0, density_norm="area")
            ax_v.set_ylim(0, 1)
            ax_v.set_title(f"Lambda by {col}")
            out_v = os.path.join(base_dir or os.path.dirname(os.path.abspath(res_path)), f"lambda_violin_by_{col}.png")
            fig_v.savefig(out_v, dpi=150, bbox_inches="tight"); plt.close(fig_v)
            print(f"wrote {out_v}")

    # Optional ridgeline by K or s (FacetGrid rows)
    if args.ridgeline_by:
        col = args.ridgeline_by
        if col in df.columns:
            try:
                g = sns.FacetGrid(df, row=col, hue=hue_col, aspect=6, height=1.0, sharex=True, sharey=False, palette=color_map)
                g.map(sns.kdeplot, "pmd_lambda", fill=True, alpha=0.4, bw_adjust=1.5)
                g.set(xlim=(0,1))
                g.set_titles(row_template=f"{col} = {{row_name}}")
                for ax in g.axes.flat:
                    ax.set_ylabel("")
                out_r = os.path.join(base_dir or os.path.dirname(os.path.abspath(res_path)), f"lambda_ridgeline_by_{col}.png")
                g.fig.savefig(out_r, dpi=150, bbox_inches="tight")
                print(f"wrote {out_r}")
            except Exception:
                pass


if __name__ == "__main__":
    main()
