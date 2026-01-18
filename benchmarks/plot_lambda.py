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
    ap.add_argument(
        "--linestyle-by",
        choices=["", "K", "s", "total_clusters"],
        default="",
        help="use line styles for this variable (auto: K if present else s; also supports total_clusters)",
    )
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
    if "K" not in df.columns and "NumberOfClusters" in df.columns:
        df["K"] = df["NumberOfClusters"]
    if "K" in df.columns:
        df["K"] = pd.to_numeric(df["K"], errors="coerce")
    # Normalize total cluster count naming across scenarios
    if "total_clusters" not in df.columns and "total_number_of_clusters" in df.columns:
        df["total_clusters"] = df["total_number_of_clusters"]

    sns.set_context("talk")
    # Stable palette by dataset
    labels = sorted(df[hue_col].dropna().unique())
    palette = sns.color_palette("tab10", n_colors=len(labels))
    color_map = {lab: palette[i] for i, lab in enumerate(labels)}
    # Lambda density with dataset colors and optional linestyles
    k_values = []
    if "K" in df.columns:
        k_values = sorted([k for k in df["K"].dropna().unique()])
    multi_k = len(k_values) > 1
    # Choose style column
    requested = args.linestyle_by
    # If the user requests total_clusters but only total_number_of_clusters exists, remap happened above
    if requested == "total_clusters" and "total_clusters" not in df.columns:
        requested = ""
    style_col = requested or ("K" if multi_k else ("s" if ("s" in df.columns and df["s"].nunique() > 1) else ""))
    style_vals = []
    if style_col:
        style_vals = sorted([v for v in df[style_col].dropna().unique()])
    ls_cycle = ["solid", "dashed", "dashdot", "dotted", (0, (3, 1, 1, 1))]
    style_map = {style_vals[i]: ls_cycle[i % len(ls_cycle)] for i in range(len(style_vals))}
    mode_groups = sorted(df["mode"].dropna().unique()) if "mode" in df.columns else []
    multi_mode = len(mode_groups) > 1
    if not mode_groups:
        mode_groups = [None]

    # Write a small legend summary for tests/inspection
    try:
        import json as _json
        legend_summary = {
            "hue_col": hue_col,
            "colors": [str(l) for l in labels],
            "k_values": [int(k) if float(k).is_integer() else float(k) for k in k_values] if k_values else [],
            "modes": [str(m) for m in mode_groups if m is not None],
        }
        out_json = os.path.join(base_dir or os.path.dirname(os.path.abspath(res_path)), "lambda_legend.json")
        with open(out_json, "w") as _fh:
            _json.dump(legend_summary, _fh, indent=2)
    except Exception:
        pass
    from matplotlib.gridspec import GridSpec
    fig_d = plt.figure(figsize=(12, 4.5))
    gs_d = GridSpec(1, 2, width_ratios=[4, 1], figure=fig_d)
    ax_main = fig_d.add_subplot(gs_d[0, 0])
    ax_leg = fig_d.add_subplot(gs_d[0, 1]); ax_leg.set_axis_off()
    for md in mode_groups:
        sub_mode = df if md is None else df[df["mode"] == md]
        for lbl in labels:
            sub_ds = sub_mode[sub_mode[hue_col] == lbl]
            if sub_ds.empty:
                continue
            if style_col:
                for val in style_vals:
                    subset = sub_ds[sub_ds[style_col] == val]
                    if subset.empty:
                        continue
                    sns.kdeplot(
                        data=subset,
                        x="pmd_lambda",
                        color=color_map[lbl],
                        bw_adjust=1.5,
                        fill=False,
                        ax=ax_main,
                        linewidth=1.8,
                        alpha=0.9,
                        linestyle=style_map.get(val, "solid"),
                        legend=False,
                    )
            else:
                sns.kdeplot(
                    data=sub_ds,
                    x="pmd_lambda",
                    color=color_map[lbl],
                    bw_adjust=1.5,
                    fill=True,
                    alpha=0.3,
                    ax=ax_main,
                    linewidth=1.8,
                    legend=False,
                )
    ax_main.set_xlim(0, 1)
    ax_main.set_title("Lambda density")
    ax_main.set_ylabel("density")
    from matplotlib.lines import Line2D
    color_handles = [Line2D([], [], color=color_map[lbl], lw=3, label=str(lbl)) for lbl in labels]
    leg_elems = list(color_handles)
    leg_title = f"colors={hue_col}"
    if style_col:
        ls_handles = [
            Line2D([], [], color="black", lw=2, linestyle=style_map[v], label=f"{style_col}={int(v) if (isinstance(v,(int,float)) and float(v).is_integer()) else v}")
            for v in style_vals
        ]
        leg_elems.extend(ls_handles)
        leg_title += f"; lines={style_col}"
    if multi_mode:
        mode_handles = [Line2D([], [], color="black", lw=0, label=f"mode={m}") for m in mode_groups if m is not None]
        leg_elems.extend(mode_handles)
    ax_leg.legend(handles=leg_elems, title=leg_title, frameon=False, loc="center left")
    out_d = args.density_out or os.path.join(base_dir or os.path.dirname(os.path.abspath(res_path)), "lambda_density.png")
    fig_d.savefig(out_d, dpi=150, bbox_inches="tight"); plt.close(fig_d)
    print(f"wrote {out_d}")

    # Lambda vs predictors
    # Predictors figure with a legend panel on the right
    from matplotlib.gridspec import GridSpec
    fig_p = plt.figure(figsize=(16, 5))
    gsp = GridSpec(1, 3, width_ratios=[1, 1, 0.6], figure=fig_p)
    ax1 = fig_p.add_subplot(gsp[0, 0])
    ax2 = fig_p.add_subplot(gsp[0, 1])
    axL = fig_p.add_subplot(gsp[0, 2]); axL.set_axis_off()
    if {"pmd_lambda", "total_clusters"}.issubset(df.columns):
        for md in mode_groups:
            sub_mode = df if md is None else df[df["mode"] == md]
            for lbl in labels:
                sub_ds = sub_mode[sub_mode[hue_col] == lbl]
                if sub_ds.empty:
                    continue
                sns.scatterplot(
                    data=sub_ds,
                    x="total_clusters",
                    y="pmd_lambda",
                    color=color_map[lbl],
                    alpha=0.25,
                    s=18,
                    ax=ax1,
                    legend=False,
                )
                if multi_k:
                    for k in k_values:
                        subset = sub_ds[sub_ds["K"] == k]
                        xs = subset["total_clusters"].to_numpy()
                        ys = subset["pmd_lambda"].to_numpy()
                        ok = np.isfinite(xs) & np.isfinite(ys)
                        xs, ys = xs[ok], ys[ok]
                        if xs.size >= 3 and _HAS_LOWESS:
                            fitted = sm_lowess(ys, xs, frac=0.6, return_sorted=True)
                            ax1.plot(
                                fitted[:, 0],
                                fitted[:, 1],
                                lw=2,
                                color=color_map.get(lbl),
                                linestyle=k_style_map.get(k, "solid"),
                            )
                else:
                    xs = sub_ds["total_clusters"].to_numpy()
                    ys = sub_ds["pmd_lambda"].to_numpy()
                    ok = np.isfinite(xs) & np.isfinite(ys)
                    xs, ys = xs[ok], ys[ok]
                    if xs.size >= 3 and _HAS_LOWESS:
                        fitted = sm_lowess(ys, xs, frac=0.6, return_sorted=True)
                        ax1.plot(fitted[:, 0], fitted[:, 1], lw=2, color=color_map.get(lbl))
        # lock x bounds
        try:
            xmin = float(np.nanmin(df["total_clusters"])); xmax = float(np.nanmax(df["total_clusters"]))
            ax1.set_xlim(xmin, xmax)
        except Exception:
            pass
        ax1.set_title("lambda vs total clusters"); ax1.set_ylabel("lambda (mean null PMD)")
    if {"pmd_lambda", "log2_minE"}.issubset(df.columns):
        for md in mode_groups:
            sub_mode = df if md is None else df[df["mode"] == md]
            for lbl in labels:
                sub_ds = sub_mode[sub_mode[hue_col] == lbl]
                if sub_ds.empty:
                    continue
                sns.scatterplot(
                    data=sub_ds,
                    x="log2_minE",
                    y="pmd_lambda",
                    color=color_map[lbl],
                    alpha=0.25,
                    s=18,
                    ax=ax2,
                    legend=False,
                )
                if multi_k:
                    for k in k_values:
                        subset = sub_ds[sub_ds["K"] == k]
                        xs = subset["log2_minE"].to_numpy()
                        ys = subset["pmd_lambda"].to_numpy()
                        ok = np.isfinite(xs) & np.isfinite(ys)
                        xs, ys = xs[ok], ys[ok]
                        if xs.size >= 3 and _HAS_LOWESS:
                            fitted = sm_lowess(ys, xs, frac=0.8, return_sorted=True)
                            ax2.plot(
                                fitted[:, 0],
                                fitted[:, 1],
                                lw=2,
                                color=color_map.get(lbl),
                                linestyle=k_style_map.get(k, "solid"),
                            )
                else:
                    xs = sub_ds["log2_minE"].to_numpy()
                    ys = sub_ds["pmd_lambda"].to_numpy()
                    ok = np.isfinite(xs) & np.isfinite(ys)
                    xs, ys = xs[ok], ys[ok]
                    if xs.size >= 3 and _HAS_LOWESS:
                        fitted = sm_lowess(ys, xs, frac=0.8, return_sorted=True)
                        ax2.plot(fitted[:, 0], fitted[:, 1], lw=2, color=color_map.get(lbl))
        try:
            xmin = float(np.nanmin(df["log2_minE"])); xmax = float(np.nanmax(df["log2_minE"]))
            ax2.set_xlim(xmin, xmax)
        except Exception:
            pass
        ax2.set_title("lambda vs log2(min(E))"); ax2.set_ylabel("lambda (mean null PMD)")
    # Build legend content in axL
    from matplotlib.lines import Line2D
    color_handles = [Line2D([], [], color=color_map[lbl], lw=3, label=str(lbl)) for lbl in labels]
    leg_elems = color_handles
    leg_title = f"colors={hue_col}"
    if multi_mode:
        ls_cycle = ["solid", "dashed", "dashdot", "dotted"]
        style_map = {modes[i]: ls_cycle[i % len(ls_cycle)] for i in range(len(modes))}
        ls_handles = [Line2D([], [], color="black", lw=2, linestyle=style_map[m], label=str(m)) for m in modes]
        leg_elems = color_handles + ls_handles
        leg_title += "; lines=mode"
    axL.legend(handles=leg_elems, title=leg_title, frameon=False, loc="center left")

    out_p = args.predictors_out or os.path.join(base_dir or os.path.dirname(os.path.abspath(res_path)), "lambda_by_predictors.png")
    fig_p.savefig(out_p, dpi=150, bbox_inches="tight"); plt.close(fig_p)
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
