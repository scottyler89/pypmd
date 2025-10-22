#!/usr/bin/env python3
"""Render invariance figures, including a 2x6 super-grid matching the R layout.

Usage:
  python benchmarks/plot_invariance.py \
    --file benchmarks/out/<ts>_invariance/invariance_results.csv \
    --out invariance_super_grid.png \
    --full-grid true --include-raw-percent false
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
from matplotlib import gridspec
from matplotlib.lines import Line2D


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="path to invariance_results.csv")
    ap.add_argument("--out", default="", help="output image path for the super grid; default next to file")
    ap.add_argument("--full-grid", type=str, default="true", help="true/false: emit 2x6 super grid")
    ap.add_argument("--include-raw-percent", type=str, default="false", help="true/false: also render a raw-PMD vs percent-different panel")
    ap.add_argument("--include-comparators", type=str, default="true", help="true/false: also render comparator invariance super-grid")
    ap.add_argument("--separate-panels", type=str, default="false", help="true/false: also save each panel as a separate PNG next to the super grid")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.file)
    sns.set_context("talk")

    # Quick helper for polynomial smoothing by group (bottom row panels)
    def poly_smooth_by_group(axh, data, x, y, color_group_col, style_col=None):
        # scatter with color by group and optional linetype by style_col (dataset_sizes)
        groups = sorted(data[color_group_col].dropna().unique())
        palette = sns.color_palette("tab10", n_colors=len(groups))
        color_map = {g: palette[i] for i, g in enumerate(groups)}
        if style_col and style_col in data.columns:
            sns.scatterplot(data=data, x=x, y=y, hue=color_group_col, palette=color_map, style=style_col, alpha=0.25, s=15, ax=axh, legend=False)
        else:
            sns.scatterplot(data=data, x=x, y=y, hue=color_group_col, palette=color_map, alpha=0.25, s=15, ax=axh, legend=False)
        # Linestyle mapping for style levels
        style_vals = []
        style_map = {}
        if style_col and style_col in data.columns:
            style_vals = sorted(data[style_col].dropna().unique())
            ls_cycle = ["solid", "dashed", "dashdot", "dotted"]
            for i, sv in enumerate(style_vals):
                style_map[sv] = ls_cycle[i % len(ls_cycle)]
        for g in groups:
            sub_g = data[data[color_group_col] == g]
            if style_map:
                for sv in style_vals:
                    sub = sub_g[sub_g[style_col] == sv]
                    xs = sub[x].to_numpy(dtype=float)
                    ys = sub[y].to_numpy(dtype=float)
                    ok = np.isfinite(xs) & np.isfinite(ys)
                    xs, ys = xs[ok], ys[ok]
                    if xs.size < 2:
                        continue
                    deg = 1 if np.unique(xs).size <= 2 else 2
                    try:
                        coef = np.polyfit(xs, ys, deg)
                        xx = np.linspace(xs.min(), xs.max(), 200)
                        yy = np.polyval(coef, xx)
                        axh.plot(xx, yy, color=color_map[g], lw=2, linestyle=style_map.get(sv, "solid"))
                    except Exception:
                        m = sub.groupby(x)[y].mean(numeric_only=True)
                        axh.plot(m.index.values, m.values, color=color_map[g], lw=2, linestyle=style_map.get(sv, "solid"))
            else:
                sub = sub_g
                xs = sub[x].to_numpy(dtype=float)
                ys = sub[y].to_numpy(dtype=float)
                ok = np.isfinite(xs) & np.isfinite(ys)
                xs, ys = xs[ok], ys[ok]
                if xs.size < 2:
                    continue
                deg = 1 if np.unique(xs).size <= 2 else 2
                try:
                    coef = np.polyfit(xs, ys, deg)
                    xx = np.linspace(xs.min(), xs.max(), 200)
                    yy = np.polyval(coef, xx)
                    axh.plot(xx, yy, color=color_map[g], lw=2)
                except Exception:
                    m = sub.groupby(x)[y].mean(numeric_only=True)
                    axh.plot(m.index.values, m.values, color=color_map[g], lw=2)

    # Only produce the larger 2x6 super-grid when requested
    if str(args.full_grid).lower() in {"1", "true", "yes"}:
        # Build a 2x7 grid (6 panels + 1 legend column per row)
        fig = plt.figure(figsize=(26, 11))
        gs = gridspec.GridSpec(2, 7, width_ratios=[1,1,1,1,1,1,0.5], wspace=0.28, hspace=0.38)
        ax = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(6)]
        ax_legend_top = fig.add_subplot(gs[0, 6])
        ax_legend_bot = fig.add_subplot(gs[1, 6])
        for la in (ax_legend_top, ax_legend_bot):
            la.set_axis_off()

        hue = "percent_different_clusters_numeric" if "percent_different_clusters_numeric" in df.columns else None
        style = "dataset_sizes" if "dataset_sizes" in df.columns else None

        # Row 1: vs NumberOfClusters (6 panels)
        ln0 = sns.lineplot(data=df, x="NumberOfClusters", y="pmd", hue=hue, style=style, ax=ax[0], legend=True)
        ax[0].set_title("PMD* vs #Clusters")
        sns.lineplot(data=df, x="NumberOfClusters", y="chi_sq", hue=hue, style=style, ax=ax[1], legend=False)
        ax[1].set_title("Chi-square vs #Clusters")
        if "chi_neg_log_p" in df.columns:
            sns.lineplot(data=df, x="NumberOfClusters", y="chi_neg_log_p", hue=hue, style=style, ax=ax[2], legend=False)
            ax[2].set_title("-log10 p vs #Clusters")
        if "cramers_v_bc" in df.columns:
            sns.lineplot(data=df, x="NumberOfClusters", y="cramers_v_bc", hue=hue, style=style, ax=ax[3], legend=False)
            ax[3].set_title("Cramer's V (BC) vs #Clusters")
        else:
            ax[3].axis("off"); ax[3].text(0.5, 0.5, "cramers_v_bc missing", ha="center", va="center")
        sns.lineplot(data=df, x="NumberOfClusters", y="inverse_simp", hue=hue, style=style, ax=ax[4], legend=False)
        ax[4].set_title("Inv Simpson vs #Clusters")
        sns.lineplot(data=df, x="NumberOfClusters", y="shannon_entropy", hue=hue, style=style, ax=ax[5], legend=False)
        ax[5].set_title("Shannon entropy vs #Clusters")

        # Row 2: vs percent_different_clusters_numeric (6 panels) with per-NumberOfClusters smoothing
        if "percent_different_clusters_numeric" in df.columns:
            # PMD*
            poly_smooth_by_group(ax[6], df, "percent_different_clusters_numeric", "pmd", "NumberOfClusters", style_col="dataset_sizes")
            ax[6].set_title("PMD* vs % different")
            # Chi-square
            poly_smooth_by_group(ax[7], df, "percent_different_clusters_numeric", "chi_sq", "NumberOfClusters", style_col="dataset_sizes")
            ax[7].set_title("Chi-square vs % different")
            # -log10 p
            if "chi_neg_log_p" in df.columns:
                poly_smooth_by_group(ax[8], df, "percent_different_clusters_numeric", "chi_neg_log_p", "NumberOfClusters", style_col="dataset_sizes")
                ax[8].set_title("-log10 p vs % different")
            # Cramer's V (BC)
            if "cramers_v_bc" in df.columns:
                poly_smooth_by_group(ax[9], df, "percent_different_clusters_numeric", "cramers_v_bc", "NumberOfClusters", style_col="dataset_sizes")
                ax[9].set_title("Cramer's V (BC) vs % different")
            else:
                ax[9].axis("off"); ax[9].text(0.5, 0.5, "cramers_v_bc missing", ha="center", va="center")
            # Inv Simpson
            poly_smooth_by_group(ax[10], df, "percent_different_clusters_numeric", "inverse_simp", "NumberOfClusters", style_col="dataset_sizes")
            ax[10].set_title("Inv Simpson vs % different")
            # Shannon entropy
            poly_smooth_by_group(ax[11], df, "percent_different_clusters_numeric", "shannon_entropy", "NumberOfClusters", style_col="dataset_sizes")
            ax[11].set_title("Shannon entropy vs % different")

        # Lock X-axis bounds consistently across panels (top row K, bottom row %different)
        try:
            if "NumberOfClusters" in df.columns and df["NumberOfClusters"].notna().any():
                xmin = float(np.nanmin(df["NumberOfClusters"]))
                xmax = float(np.nanmax(df["NumberOfClusters"]))
                for i in range(0, 6):
                    ax[i].set_xlim(xmin, xmax)
            if "percent_different_clusters_numeric" in df.columns and df["percent_different_clusters_numeric"].notna().any():
                xmin = float(np.nanmin(df["percent_different_clusters_numeric"]))
                xmax = float(np.nanmax(df["percent_different_clusters_numeric"]))
                for i in range(6, 12):
                    ax[i].set_xlim(xmin, xmax)
        except Exception:
            pass

        # Build separate legends
        try:
            handles, labels = ax[0].get_legend_handles_labels()
            if handles and labels:
                ax[0].legend_.remove()
                ax_legend_top.legend(handles, labels, title="Top: colors=%Different, lines=dataset_sizes", frameon=False, loc="center left")
        except Exception:
            pass

        # Bottom legend (colors = NumberOfClusters; linestyles = dataset_sizes)
        try:
            # Color legend proxies for NumberOfClusters
            k_groups = sorted(df["NumberOfClusters"].dropna().unique()) if "NumberOfClusters" in df.columns else []
            pal_k = sns.color_palette("tab10", n_colors=len(k_groups))
            color_handles = [Line2D([], [], color=pal_k[i], lw=2, label=str(k_groups[i])) for i in range(len(k_groups))]
            # Linestyle proxies for dataset_sizes
            style_vals = sorted(df["dataset_sizes"].dropna().unique()) if "dataset_sizes" in df.columns else []
            ls_cycle = ["solid", "dashed", "dashdot", "dotted"]
            style_handles = [Line2D([], [], color="black", lw=2, linestyle=ls_cycle[i % len(ls_cycle)], label=str(style_vals[i])) for i in range(len(style_vals))]
            handles_bot = color_handles + style_handles
            labels_bot = [h.get_label() for h in handles_bot]
            ax_legend_bot.legend(handles_bot, labels_bot, title="Bottom: colors=#Clusters, lines=dataset_sizes", frameon=False, loc="center left")
        except Exception:
            pass

        title = "Invariance Super Grid (2x6)"
        if "mode" in df.columns and df["mode"].nunique() == 1:
            title += f" — mode: {df['mode'].iloc[0]}"
        plt.suptitle(title, y=1.02, fontsize=18)
        plt.tight_layout()
        out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.file)), "invariance_super_grid.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")
        # Optional separate panels saved alongside the super grid
        if str(args.separate_panels).lower() in {"1", "true", "yes"}:
            base = os.path.dirname(out)
            # Re-render 14 R-style pages explicitly
            def save_lineplot(x, y, title, fname):
                f, a = plt.subplots(1,1, figsize=(8,4.5))
                sns.lineplot(data=df, x=x, y=y, hue="percent_different_clusters_numeric", style="dataset_sizes", ax=a)
                a.set_title(title)
                # lock x bounds
                try:
                    xmin = float(np.nanmin(df[x])); xmax = float(np.nanmax(df[x])); a.set_xlim(xmin, xmax)
                except Exception:
                    pass
                f.savefig(os.path.join(base, fname), dpi=150, bbox_inches="tight"); plt.close(f)
            # 1: Raw PMD vs K
            if "raw_pmd" in df.columns:
                save_lineplot("NumberOfClusters", "raw_pmd", "Raw PMD vs #Clusters", "pmd_characterization1.png")
            # 2: PMD* vs K
            save_lineplot("NumberOfClusters", "pmd", "PMD* vs #Clusters", "pmd_characterization2.png")
            # 3: Chi-square vs K
            save_lineplot("NumberOfClusters", "chi_sq", "Chi-square vs #Clusters", "pmd_characterization3.png")
            # 4: -log10 p vs K
            if "chi_neg_log_p" in df.columns:
                save_lineplot("NumberOfClusters", "chi_neg_log_p", "-log10 p vs #Clusters", "pmd_characterization4.png")
            # 5: Cramer's V (BC) vs K
            if "cramers_v_bc" in df.columns:
                save_lineplot("NumberOfClusters", "cramers_v_bc", "Cramer's V (BC) vs #Clusters", "pmd_characterization5.png")
            # 6: Inv Simpson vs K
            save_lineplot("NumberOfClusters", "inverse_simp", "Inverse Simpson vs #Clusters", "pmd_characterization6.png")
            # 7: Shannon vs K
            save_lineplot("NumberOfClusters", "shannon_entropy", "Shannon entropy vs #Clusters", "pmd_characterization7.png")
            # Bottom row with polynomial smoothing per K and style by dataset
            def save_poly(x, y, title, fname):
                f, a = plt.subplots(1,1, figsize=(8,4.5))
                poly_smooth_by_group(a, df, x, y, "NumberOfClusters", style_col="dataset_sizes")
                a.set_title(title)
                try:
                    xmin = float(np.nanmin(df[x])); xmax = float(np.nanmax(df[x])); a.set_xlim(xmin, xmax)
                except Exception:
                    pass
                f.savefig(os.path.join(base, fname), dpi=150, bbox_inches="tight"); plt.close(f)
            # 8: Raw PMD vs %
            if "raw_pmd" in df.columns and "percent_different_clusters_numeric" in df.columns:
                save_poly("percent_different_clusters_numeric", "raw_pmd", "Raw PMD vs % different", "pmd_characterization8.png")
            # 9: PMD* vs %
            save_poly("percent_different_clusters_numeric", "pmd", "PMD* vs % different", "pmd_characterization9.png")
            # 10: Chi-square vs %
            save_poly("percent_different_clusters_numeric", "chi_sq", "Chi-square vs % different", "pmd_characterization10.png")
            # 11: -log10 p vs %
            if "chi_neg_log_p" in df.columns:
                save_poly("percent_different_clusters_numeric", "chi_neg_log_p", "-log10 p vs % different", "pmd_characterization11.png")
            # 12: Cramer's V (BC) vs %
            if "cramers_v_bc" in df.columns:
                save_poly("percent_different_clusters_numeric", "cramers_v_bc", "Cramer's V (BC) vs % different", "pmd_characterization12.png")
            # 13: Inv Simpson vs %
            save_poly("percent_different_clusters_numeric", "inverse_simp", "Inverse Simpson vs % different", "pmd_characterization13.png")
            # 14: Shannon vs %
            save_poly("percent_different_clusters_numeric", "shannon_entropy", "Shannon entropy vs % different", "pmd_characterization14.png")

        # Optional raw PMD vs percent different panel
        if str(args.include_raw_percent).lower() in {"1", "true", "yes"} and ("raw_pmd" in df.columns) and ("percent_different_clusters_numeric" in df.columns):
            fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4))
            poly_smooth_by_group(ax2, df, "percent_different_clusters_numeric", "raw_pmd", "NumberOfClusters")
            ax2.set_title("Raw PMD vs % different (smoothed by #Clusters)")
            out2 = os.path.join(os.path.dirname(out), "invariance_extra_raw_percent.png")
            fig2.savefig(out2, dpi=150, bbox_inches="tight")
            print(f"wrote {out2}")
            plt.close(fig2)
        plt.close(fig)

        # Comparator invariance super-grid
        if str(args.include_comparators).lower() in {"1", "true", "yes"}:
            comp_metrics = [
                ("jsd", "JSD"), ("tv", "TV"), ("hellinger", "Hellinger"),
                ("braycurtis", "Bray–Curtis"), ("cosine", "Cosine"), ("canberra", "Canberra")
            ]
            # Only proceed if at least one column exists
            if any(m[0] in df.columns for m in comp_metrics):
                figC = plt.figure(figsize=(26, 11))
                gsC = gridspec.GridSpec(2, 7, width_ratios=[1,1,1,1,1,1,0.5], wspace=0.28, hspace=0.38)
                axC = [figC.add_subplot(gsC[r, c]) for r in range(2) for c in range(6)]
                axLTop = figC.add_subplot(gsC[0, 6]); axLBot = figC.add_subplot(gsC[1, 6])
                for la in (axLTop, axLBot): la.set_axis_off()
                # Top row vs K
                for i, (col, title_m) in enumerate(comp_metrics[:6]):
                    if col in df.columns:
                        ln = sns.lineplot(data=df, x="NumberOfClusters", y=col, hue=hue, style=style, ax=axC[i], legend=(i==0))
                        axC[i].set_title(f"{title_m} vs #Clusters");
                # Bottom row vs %Different with smoothing per K and style by dataset
                for i, (col, title_m) in enumerate(comp_metrics[:6]):
                    idx = 6 + i
                    if col in df.columns:
                        poly_smooth_by_group(axC[idx], df, "percent_different_clusters_numeric", col, "NumberOfClusters", style_col="dataset_sizes")
                        axC[idx].set_title(f"{title_m} vs % different")
                # X locks
                try:
                    if "NumberOfClusters" in df.columns:
                        xmin = float(np.nanmin(df["NumberOfClusters"]))
                        xmax = float(np.nanmax(df["NumberOfClusters"]))
                        for i in range(0, 6): axC[i].set_xlim(xmin, xmax)
                    if "percent_different_clusters_numeric" in df.columns:
                        xmin = float(np.nanmin(df["percent_different_clusters_numeric"]))
                        xmax = float(np.nanmax(df["percent_different_clusters_numeric"]))
                        for i in range(6, 12): axC[i].set_xlim(xmin, xmax)
                except Exception:
                    pass
                # Legends (top row legend)
                try:
                    hC, lC = axC[0].get_legend_handles_labels()
                    if hC and lC:
                        axC[0].legend_.remove()
                        axLTop.legend(hC, lC, title="Top: colors=%Different, lines=dataset_sizes", frameon=False, loc="center left")
                except Exception:
                    pass
                # Bottom legend (colors = NumberOfClusters; lines = dataset_sizes)
                try:
                    k_groups = sorted(df["NumberOfClusters"].dropna().unique()) if "NumberOfClusters" in df.columns else []
                    pal_k = sns.color_palette("tab10", n_colors=len(k_groups))
                    color_handles = [Line2D([], [], color=pal_k[i], lw=2, label=str(k_groups[i])) for i in range(len(k_groups))]
                    style_vals = sorted(df["dataset_sizes"].dropna().unique()) if "dataset_sizes" in df.columns else []
                    ls_cycle = ["solid", "dashed", "dashdot", "dotted"]
                    style_handles = [Line2D([], [], color="black", lw=2, linestyle=ls_cycle[i % len(ls_cycle)], label=str(style_vals[i])) for i in range(len(style_vals))]
                    handles_bot = color_handles + style_handles
                    labels_bot = [h.get_label() for h in handles_bot]
                    axLBot.legend(handles_bot, labels_bot, title="Bottom: colors=#Clusters, lines=dataset_sizes", frameon=False, loc="center left")
                except Exception:
                    pass
                titleC = "Comparator Invariance Super Grid (2x6)"
                if "mode" in df.columns and df["mode"].nunique() == 1:
                    titleC += f" — mode: {df['mode'].iloc[0]}"
                plt.suptitle(titleC, y=1.02, fontsize=18)
                outC = args.out or os.path.join(os.path.dirname(os.path.abspath(args.file)), "invariance_comparators_super_grid.png")
                figC.savefig(outC, dpi=150, bbox_inches="tight")
                print(f"wrote {outC}")
                plt.close(figC)
    else:
        # Retain the compact 2x4 figure (legacy)
        fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharex=False)
        ax = axes.ravel()
        hue = "percent_different_clusters_numeric" if "percent_different_clusters_numeric" in df.columns else None
        style = "dataset_sizes" if "dataset_sizes" in df.columns else None
        sns.lineplot(data=df, x="NumberOfClusters", y="raw_pmd", hue=hue, style=style, ax=ax[0])
        ax[0].set_title("Raw PMD vs #Clusters")
        sns.lineplot(data=df, x="NumberOfClusters", y="pmd", hue=hue, style=style, ax=ax[1], legend=False)
        ax[1].set_title("PMD* vs #Clusters")
        sns.lineplot(data=df, x="NumberOfClusters", y="chi_sq", hue=hue, style=style, ax=ax[2], legend=False)
        ax[2].set_title("Chi-square vs #Clusters")
        if "cramers_v_bc" in df.columns:
            sns.lineplot(data=df, x="NumberOfClusters", y="cramers_v_bc", hue=hue, style=style, ax=ax[3], legend=False)
            ax[3].set_title("Cramer's V (BC) vs #Clusters")
        else:
            ax[3].axis("off"); ax[3].text(0.5, 0.5, "cramers_v_bc missing", ha="center", va="center")
        sns.lineplot(data=df, x="NumberOfClusters", y="inverse_simp", hue=hue, style=style, ax=ax[4], legend=False)
        ax[4].set_title("Inverse Simpson vs #Clusters")
        sns.lineplot(data=df, x="NumberOfClusters", y="shannon_entropy", hue=hue, style=style, ax=ax[5], legend=False)
        ax[5].set_title("Shannon entropy vs #Clusters")
        if "percent_different_clusters_numeric" in df.columns:
            sns.lineplot(data=df, x="percent_different_clusters_numeric", y="pmd", hue="NumberOfClusters", style=style, ax=ax[6])
            ax[6].set_title("PMD* vs % Different Clusters")
        if "chi_neg_log_p" in df.columns:
            sns.lineplot(data=df, x="NumberOfClusters", y="chi_neg_log_p", hue=hue, style=style, ax=ax[7], legend=False)
            ax[7].set_title("-log10 p vs #Clusters")
        title = "Invariance (compact)"
        if "mode" in df.columns and df["mode"].nunique() == 1:
            title += f" — mode: {df['mode'].iloc[0]}"
        plt.suptitle(title, y=1.02, fontsize=18)
        plt.tight_layout()
        out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.file)), "invariance.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
