#!/usr/bin/env python3
"""Reproduce the R-style characterization plots for PMD benchmarking.

Inputs:
- Either pass a `results.csv` path, OR pass `--dir DIR` pointing to a run
  directory containing `results.csv` (and optional `null.csv`).
- You can also pass `--latest-tag TAG` to auto-pick the newest directory
  matching `benchmarks/out/*_{TAG}`.

Outputs: a multi-panel PNG saved next to results.csv by default.
"""

from __future__ import annotations

import argparse
import os
import math

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    _HAS_LOWESS = True
except Exception:
    _HAS_LOWESS = False


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="", help="path to results.csv or omit when using --dir/--latest-tag")
    ap.add_argument("--dir", default="", help="run directory containing results.csv and optional null.csv")
    ap.add_argument("--latest-tag", default="", help="auto-pick latest benchmarks/out/*_{TAG} directory")
    ap.add_argument("--null", default="", help="path to null.csv (optional)")
    ap.add_argument("--out", default="", help="output image path; default next to results")
    ap.add_argument("--pdf", type=str, default="", help="optional multi-page PDF output path; per-dataset pages")
    ap.add_argument("--merged-grid", type=str, default="false", help="true/false: save a merged grid image combining main and comparators figures")
    ap.add_argument("--super-grid", type=str, default="false", help="true/false: render all panels (main + comparators) into a single large figure")
    ap.add_argument("--null-density-out", type=str, default="", help="optional dedicated null-PMD density image path")
    ap.add_argument("--separate-panels", type=str, default="false", help="true/false: also save each main panel as a separate PNG next to results")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = ""
    res_path = args.results
    # Resolve directory modes
    if args.dir:
        base_dir = args.dir
        res_path = os.path.join(base_dir, "results.csv")
    elif args.latest_tag:
        import glob

        candidates = sorted(glob.glob(os.path.join("benchmarks", "out", f"*_{args.latest_tag}")))
        if not candidates:
            raise SystemExit(f"No runs found for tag '{args.latest_tag}' under benchmarks/out/")
        base_dir = candidates[-1]
        res_path = os.path.join(base_dir, "results.csv")
    elif os.path.isdir(res_path):
        base_dir = res_path
        res_path = os.path.join(base_dir, "results.csv")

    if not os.path.exists(res_path):
        raise SystemExit(f"results.csv not found: {res_path}")

    df = pd.read_csv(res_path)
    # Derive helpers for grouping and axes
    df["dataset_label"] = df.apply(lambda r: f"K={int(r['K'])}, N1={int(r['N1'])}, N2={int(r['N2'])}", axis=1)
    s_max = int(np.nanmax(df["s"])) if "s" in df.columns else None
    # Resolve null path
    null_path = args.null
    if not null_path and base_dir:
        npth = os.path.join(base_dir, "null.csv")
        null_path = npth if os.path.exists(npth) else ""
    null_df = pd.read_csv(null_path) if null_path else None
    merged_null = None
    if null_df is not None and not null_df.empty:
        merged_null = null_df.copy()
        if "dataset_label" not in merged_null.columns and {"K","N1","N2"}.issubset(merged_null.columns):
            label_map = df.drop_duplicates(subset=["K","N1","N2"])[["K","N1","N2","dataset_label"]]
            merged_null = merged_null.merge(label_map, on=["K","N1","N2"], how="left")

    df = df.copy()
    # Prefer robust -log10(p) via log-survival when possible
    df["neglog10_p"] = np.nan
    if ("chi2" in df.columns) and ("observed_clusters" in df.columns or "total_clusters" in df.columns):
        rows = df["observed_clusters"].fillna(df.get("total_clusters", np.nan)) if "observed_clusters" in df.columns else df["total_clusters"]
        dof = np.maximum(rows.astype(float) - 1.0, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            df["neglog10_p"] = -sps.chi2.logsf(df["chi2"].to_numpy(dtype=float), dof.to_numpy(dtype=float)) / np.log(10.0)
    elif "chi2_p" in df.columns:
        # Fallback: direct transform (no clipping)
        df["neglog10_p"] = -np.log10(df["chi2_p"])
    if "min_E" in df.columns:
        df["log2_minE"] = np.log2(df["min_E"].replace(0, np.nan))
    if "pmd_lambda" not in df.columns and "pmd" in df.columns and "pmd_raw" in df.columns:
        # Best-effort: back-compute lambda if needed (not exact if missing)
        df["pmd_lambda"] = np.nan

    # Figure layout (2x5 grid)
    sns.set_context("talk")
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharex=True)
    plt.subplots_adjust(wspace=0.25, hspace=0.35)
    ax = axes.ravel()

    # Stable dataset palette
    labels = sorted(df["dataset_label"].dropna().unique())
    pal = sns.color_palette("tab10", n_colors=len(labels))
    color_map_ds = {lab: pal[i] for i, lab in enumerate(labels)}

    # Helper: scatter + per-group loess lines for metric vs s
    def loess(axh, metric, title, ylabel, bounds=None):
        if "s" not in df.columns or metric not in df.columns:
            axh.axis("off")
            axh.text(0.5, 0.5, f"{metric} N/A", ha="center", va="center")
            return
        # scatter
        sns.scatterplot(
            data=df,
            x="s",
            y=metric,
            hue="dataset_label",
            palette=color_map_ds,
            alpha=0.25,
            s=15,
            ax=axh,
            legend=False,
        )
        # loess per group
        for label, sub in df.groupby("dataset_label"):
            xs = sub["s"].to_numpy()
            ys = sub[metric].to_numpy()
            ok = np.isfinite(xs) & np.isfinite(ys)
            xs, ys = xs[ok], ys[ok]
            if xs.size < 3:
                continue
            if _HAS_LOWESS:
                fitted = sm_lowess(ys, xs, frac=0.6, return_sorted=True)
                axh.plot(fitted[:, 0], fitted[:, 1], label=label, lw=2, color=color_map_ds.get(label))
            else:
                # fallback: mean per s
                m = sub.groupby("s")[metric].mean(numeric_only=True)
                axh.plot(m.index.values, m.values, label=label, lw=2, color=color_map_ds.get(label))
        axh.set_title(title)
        axh.set_ylabel(ylabel)
        if s_max is not None:
            axh.set_xlim(0, s_max)
        if bounds is not None:
            axh.set_ylim(*bounds)

    # 1) raw PMD vs s
    if "pmd_raw" in df.columns:
        loess(ax[0], "pmd_raw", "Raw PMD vs non-shared clusters", "raw PMD", bounds=(0, 1))
    # 2) chi2 vs s
    if "chi2" in df.columns:
        loess(ax[1], "chi2", "Chi-square vs non-shared clusters", "chi2")
    # 3) -log10(p) vs s
    if "neglog10_p" in df.columns:
        loess(ax[2], "neglog10_p", "-log10(p) vs non-shared clusters", "-log10 p")
    # 4) Cramer's V vs s (bias-corrected only; no fallback)
    vcol = "cramers_v_bc" if "cramers_v_bc" in df.columns else None
    if vcol is not None:
        loess(ax[3], vcol, "Cramer's V (bias-corrected) vs non-shared clusters", "Cramer's V (BC)", bounds=(0, 1))
    else:
        ax[3].axis("off"); ax[3].text(0.5, 0.5, "cramers_v_bc missing", ha="center", va="center")
    # 5) Inverse Simpson vs s
    if "inv_simpson" in df.columns:
        loess(ax[4], "inv_simpson", "Inverse Simpson vs non-shared clusters", "inv Simpson")
    # 6) Shannon entropy vs s
    if "entropy" in df.columns:
        loess(ax[5], "entropy", "Entropy vs non-shared clusters", "entropy (bits)", bounds=(0, 1))
    # 7) Null density of PMD (s=0) — color by dataset when available
    if merged_null is not None and "pmd_null" in merged_null.columns:
        s0 = merged_null[merged_null.get("s", 0) == 0]
        hue = "dataset_label" if ("dataset_label" in s0.columns and s0["dataset_label"].nunique() > 1) else None
        sns.kdeplot(data=s0, x="pmd_null", hue=hue, bw_adjust=1.5, fill=True, alpha=0.3, ax=ax[6])
        ax[6].set_xlim(0, 1)
        ax[6].set_title("Null PMD density (s=0)")
        ax[6].set_ylabel("density")
    # 8) lambda vs total_number_of_clusters
    if {"pmd_lambda", "total_clusters"}.issubset(df.columns):
        sns.scatterplot(data=df, x="total_clusters", y="pmd_lambda", hue="dataset_label", palette=color_map_ds, alpha=0.3, s=20, ax=ax[7], legend=False)
        # lowess per dataset
        for label, sub in df.groupby("dataset_label"):
            xs = sub["total_clusters"].to_numpy(); ys = sub["pmd_lambda"].to_numpy()
            ok = np.isfinite(xs) & np.isfinite(ys)
            xs, ys = xs[ok], ys[ok]
            if xs.size < 3:
                continue
            if _HAS_LOWESS:
                fitted = sm_lowess(ys, xs, frac=0.6, return_sorted=True)
                ax[7].plot(fitted[:, 0], fitted[:, 1], lw=2, color=color_map_ds.get(label))
        ax[7].set_title("lambda vs total clusters")
        ax[7].set_ylabel("lambda (mean null PMD)")
    # 9) lambda vs log2(min(E))
    if {"pmd_lambda", "log2_minE"}.issubset(df.columns):
        sns.scatterplot(data=df, x="log2_minE", y="pmd_lambda", hue="dataset_label", palette=color_map_ds, alpha=0.3, s=20, ax=ax[8], legend=False)
        for label, sub in df.groupby("dataset_label"):
            xs = sub["log2_minE"].to_numpy(); ys = sub["pmd_lambda"].to_numpy()
            ok = np.isfinite(xs) & np.isfinite(ys)
            xs, ys = xs[ok], ys[ok]
            if xs.size < 3:
                continue
            if _HAS_LOWESS:
                fitted = sm_lowess(ys, xs, frac=0.8, return_sorted=True)
                ax[8].plot(fitted[:, 0], fitted[:, 1], lw=2, color=color_map_ds.get(label))
        ax[8].set_title("lambda vs log2(min(E))")
        ax[8].set_ylabel("lambda (mean null PMD)")
    # 10) debiased PMD vs s
    if "pmd" in df.columns:
        loess(ax[9], "pmd", "Debiased PMD vs non-shared clusters", "PMD*", bounds=(0, 1))

    # Save
    out = args.out
    if not out:
        base = base_dir or os.path.dirname(os.path.abspath(res_path))
        out = os.path.join(base, "characterize_pmd.png")
    fig.suptitle("PMD Characterization", y=1.02, fontsize=16)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    # Optionally export a comparators figure as well
    comp_out = os.path.join(os.path.dirname(out), "characterize_comparators.png")
    fig2, axes2 = plt.subplots(1, 5, figsize=(22, 4.5), sharex=True)
    comps = [
        ("jsd", "Jensen–Shannon distance", (0, 1)),
        ("tv", "Total Variation", (0, 1)),
        ("hellinger", "Hellinger", (0, 1)),
        ("braycurtis", "Bray–Curtis", (0, 1)),
        ("cosine", "Cosine distance", (0, 1)),
    ]
    for axh, (metric, title, bnds) in zip(axes2, comps):
        loess(axh, metric, f"{title} vs non-shared clusters", title, bounds=bnds)
    fig2.suptitle("Comparator Distances", y=1.02, fontsize=16)
    fig2.savefig(comp_out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"wrote {comp_out}")

    # Optional separate single-panel images with R-style numbering: pmd_characterization%01d.png
    if str(args.separate_panels).lower() in {"1", "true", "yes"}:
        base = os.path.dirname(out)
        def _new_ax(w=6, h=4.5):
            f, a = plt.subplots(1, 1, figsize=(w, h))
            return f, a
        idx = 1
        # 1 raw PMD
        if "pmd_raw" in df.columns:
            f,a=_new_ax(); loess(a, "pmd_raw", "Raw PMD vs non-shared clusters", "raw PMD", bounds=(0,1)); f.savefig(os.path.join(base, f"pmd_characterization{idx}.png"), dpi=150, bbox_inches="tight"); plt.close(f); idx+=1
        # 2 chi2
        if "chi2" in df.columns:
            f,a=_new_ax(); loess(a, "chi2", "Chi-square vs non-shared clusters", "chi2"); f.savefig(os.path.join(base, f"pmd_characterization{idx}.png"), dpi=150, bbox_inches="tight"); plt.close(f); idx+=1
        # 3 -log10 p
        if "neglog10_p" in df.columns:
            f,a=_new_ax(); loess(a, "neglog10_p", "-log10(p) vs non-shared clusters", "-log10 p"); f.savefig(os.path.join(base, f"pmd_characterization{idx}.png"), dpi=150, bbox_inches="tight"); plt.close(f); idx+=1
        # 4 Cramer's V (BC)
        if vcol is not None:
            f,a=_new_ax(); loess(a, vcol, "Cramer's V (BC) vs non-shared clusters", "Cramer's V (BC)", bounds=(0,1)); f.savefig(os.path.join(base, f"pmd_characterization{idx}.png"), dpi=150, bbox_inches="tight"); plt.close(f); idx+=1
        # 5 inverse simpson
        if "inv_simpson" in df.columns:
            f,a=_new_ax(); loess(a, "inv_simpson", "Inverse Simpson vs non-shared clusters", "inv Simpson"); f.savefig(os.path.join(base, f"pmd_characterization{idx}.png"), dpi=150, bbox_inches="tight"); plt.close(f); idx+=1
        # 6 entropy
        if "entropy" in df.columns:
            f,a=_new_ax(); loess(a, "entropy", "Entropy vs non-shared clusters", "entropy (bits)", bounds=(0,1)); f.savefig(os.path.join(base, f"pmd_characterization{idx}.png"), dpi=150, bbox_inches="tight"); plt.close(f); idx+=1
        # 7 null density
        if merged_null is not None and "pmd_null" in merged_null.columns:
            s0 = merged_null[merged_null.get("s", 0) == 0]
            hue = "dataset_label" if ("dataset_label" in s0.columns and s0["dataset_label"].nunique() > 1) else None
            f,a=_new_ax(); sns.kdeplot(data=s0, x="pmd_null", hue=hue, bw_adjust=1.5, fill=True, alpha=0.3, ax=a); a.set_xlim(0,1); a.set_title("Null PMD density (s=0)"); a.set_ylabel("density"); f.savefig(os.path.join(base, f"pmd_characterization{idx}.png"), dpi=150, bbox_inches="tight"); plt.close(f); idx+=1
        # 8 lambda vs total clusters
        if {"pmd_lambda","total_clusters"}.issubset(df.columns):
            f,a=_new_ax(); sns.scatterplot(data=df, x="total_clusters", y="pmd_lambda", hue="dataset_label", alpha=0.3, s=20, ax=a, legend=False); a.set_title("lambda vs total clusters"); a.set_ylabel("lambda (mean null PMD)"); f.savefig(os.path.join(base, f"pmd_characterization{idx}.png"), dpi=150, bbox_inches="tight"); plt.close(f); idx+=1
        # 9 lambda vs log2 minE
        if {"pmd_lambda","log2_minE"}.issubset(df.columns):
            f,a=_new_ax(); sns.scatterplot(data=df, x="log2_minE", y="pmd_lambda", hue="dataset_label", alpha=0.3, s=20, ax=a, legend=False); a.set_title("lambda vs log2(min(E))"); a.set_ylabel("lambda (mean null PMD)"); f.savefig(os.path.join(base, f"pmd_characterization{idx}.png"), dpi=150, bbox_inches="tight"); plt.close(f); idx+=1
        # 10 debiased PMD
        if "pmd" in df.columns:
            f,a=_new_ax(); loess(a, "pmd", "Debiased PMD vs non-shared clusters", "PMD*", bounds=(0,1)); f.savefig(os.path.join(base, f"pmd_characterization{idx}.png"), dpi=150, bbox_inches="tight"); plt.close(f); idx+=1

    # Optional dedicated null-PMD density figure
    if args.null_density_out and merged_null is not None and "pmd_null" in merged_null.columns:
        s0 = merged_null[merged_null.get("s", 0) == 0]
        hue = "dataset_label" if ("dataset_label" in s0.columns and s0["dataset_label"].nunique() >= 1) else None
        figN, axN = plt.subplots(1, 1, figsize=(8, 4.5))
        sns.kdeplot(data=s0, x="pmd_null", hue=hue, bw_adjust=1.5, fill=True, alpha=0.3, ax=axN)
        axN.set_xlim(0, 1)
        axN.set_title("Null PMD density (s=0)")
        axN.set_ylabel("density")
        figN.savefig(args.null_density_out, dpi=150, bbox_inches="tight")
        print(f"wrote {args.null_density_out}")

    # Optional super-grid: render all panels into a single figure (3x5)
    if str(args.super_grid).lower() in {"1", "true", "yes"}:
        fig3, axes3 = plt.subplots(3, 5, figsize=(26, 13), sharex=False)
        plt.subplots_adjust(wspace=0.28, hspace=0.38)
        axg = axes3.ravel()
        # Row 1: five main panels
        if "pmd_raw" in df.columns:
            loess(axg[0], "pmd_raw", "Raw PMD vs non-shared clusters", "raw PMD", bounds=(0, 1))
        if "chi2" in df.columns:
            loess(axg[1], "chi2", "Chi-square vs non-shared clusters", "chi2")
        if "neglog10_p" in df.columns:
            loess(axg[2], "neglog10_p", "-log10(p) vs non-shared clusters", "-log10 p")
        if vcol is not None:
            loess(axg[3], vcol, "Cramer's V (bias-corrected) vs non-shared clusters", "Cramer's V (BC)", bounds=(0, 1))
        else:
            axg[3].axis("off"); axg[3].text(0.5, 0.5, "cramers_v_bc missing", ha="center", va="center")
        if "inv_simpson" in df.columns:
            loess(axg[4], "inv_simpson", "Inverse Simpson vs non-shared clusters", "inv Simpson")
        # Row 2: entropy, null density, lambda vs clusters, lambda vs log2(minE), PMD*
        if "entropy" in df.columns:
            loess(axg[5], "entropy", "Entropy vs non-shared clusters", "entropy (bits)", bounds=(0, 1))
        if merged_null is not None and "pmd_null" in merged_null.columns:
            s0 = merged_null[merged_null.get("s", 0) == 0]
            hue = "dataset_label" if ("dataset_label" in s0.columns and s0["dataset_label"].nunique() > 1) else None
            sns.kdeplot(data=s0, x="pmd_null", hue=hue, bw_adjust=1.5, fill=True, alpha=0.3, ax=axg[6]); axg[6].set_xlim(0, 1); axg[6].set_title("Null PMD density (s=0)"); axg[6].set_ylabel("density")
        if {"pmd_lambda", "total_clusters"}.issubset(df.columns):
            sns.scatterplot(data=df, x="total_clusters", y="pmd_lambda", hue="dataset_label", alpha=0.3, s=20, ax=axg[7], legend=False)
            for label, sub in df.groupby("dataset_label"):
                xs = sub["total_clusters"].to_numpy(); ys = sub["pmd_lambda"].to_numpy(); ok = np.isfinite(xs) & np.isfinite(ys)
                if ok.sum() >= 3 and _HAS_LOWESS:
                    fitted = sm_lowess(ys[ok], xs[ok], frac=0.6, return_sorted=True); axg[7].plot(fitted[:,0], fitted[:,1], lw=2)
            axg[7].set_title("lambda vs total clusters"); axg[7].set_ylabel("lambda (mean null PMD)")
        if {"pmd_lambda", "log2_minE"}.issubset(df.columns):
            sns.scatterplot(data=df, x="log2_minE", y="pmd_lambda", hue="dataset_label", alpha=0.3, s=20, ax=axg[8], legend=False)
            for label, sub in df.groupby("dataset_label"):
                xs = sub["log2_minE"].to_numpy(); ys = sub["pmd_lambda"].to_numpy(); ok = np.isfinite(xs) & np.isfinite(ys)
                if ok.sum() >= 3 and _HAS_LOWESS:
                    fitted = sm_lowess(ys[ok], xs[ok], frac=0.8, return_sorted=True); axg[8].plot(fitted[:,0], fitted[:,1], lw=2)
            axg[8].set_title("lambda vs log2(min(E))"); axg[8].set_ylabel("lambda (mean null PMD)")
        if "pmd" in df.columns:
            loess(axg[9], "pmd", "Debiased PMD vs non-shared clusters", "PMD*", bounds=(0, 1))
        # Row 3: comparators
        comps = [("jsd","JSD"),("tv","TV"),("hellinger","Hellinger"),("braycurtis","Bray–Curtis"),("cosine","Cosine")]
        for ai, (metric, title) in enumerate(comps, start=10):
            loess(axg[ai], metric, f"{title} vs non-shared clusters", title, bounds=(0, 1))
        super_out = os.path.join(base, "characterize_super_grid.png")
        fig3.suptitle("PMD Characterization — Super Grid", y=1.01, fontsize=18)
        fig3.savefig(super_out, dpi=150, bbox_inches="tight")
        print(f"wrote {super_out}")

    # Optional merged grid by vertically concatenating the two images
    if str(args.merged_grid).lower() in {"1", "true", "yes"}:
        try:
            from PIL import Image

            im1 = Image.open(out)
            im2 = Image.open(comp_out)
            # pad widths to max
            w = max(im1.width, im2.width)
            def pad(im):
                if im.width == w:
                    return im
                bg = Image.new("RGB", (w, im.height), (255, 255, 255))
                bg.paste(im, (0, 0))
                return bg
            im1p = pad(im1)
            im2p = pad(im2)
            merged = Image.new("RGB", (w, im1p.height + im2p.height), (255, 255, 255))
            merged.paste(im1p, (0, 0))
            merged.paste(im2p, (0, im1p.height))
            merged_path = os.path.join(base, "characterize_merged.png")
            merged.save(merged_path)
            print(f"wrote {merged_path}")
        except Exception as e:
            print(f"merged-grid failed ({e}); install Pillow to enable image concatenation")

    # Optional multi-page PDF per dataset_label with both figures
    if args.pdf:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_path = args.pdf
        with PdfPages(pdf_path) as pdf:
            for label, sub in df.groupby("dataset_label"):
                # results-only figure
                _df = sub.copy()
                _fig, _axes = plt.subplots(2, 5, figsize=(22, 9), sharex=True)
                plt.subplots_adjust(wspace=0.25, hspace=0.35)
                _ax = _axes.ravel()
                # reuse loess helper against _df
                def _lo(axh, metric, title, ylabel, bounds=None):
                    nonlocal _df
                    if "s" not in _df.columns or metric not in _df.columns:
                        axh.axis("off"); axh.text(0.5, 0.5, f"{metric} N/A", ha="center", va="center"); return
                    sns.scatterplot(data=_df, x="s", y=metric, alpha=0.25, s=15, ax=axh, legend=False)
                    xs = _df["s"].to_numpy(); ys = _df[metric].to_numpy()
                    ok = np.isfinite(xs) & np.isfinite(ys); xs, ys = xs[ok], ys[ok]
                    if xs.size >= 3 and _HAS_LOWESS:
                        fitted = sm_lowess(ys, xs, frac=0.6, return_sorted=True)
                        axh.plot(fitted[:,0], fitted[:,1], lw=2, color="crimson")
                    axh.set_title(title)
                    axh.set_ylabel(ylabel)
                    if s_max is not None: axh.set_xlim(0, s_max)
                    if bounds is not None: axh.set_ylim(*bounds)
                if "chi2_p" in _df.columns:
                    # Do not clip p-values; allow -log10(0) -> inf if underflow occurs
                    _df["neglog10_p"] = -np.log10(_df["chi2_p"]) 
                if "min_E" in _df.columns:
                    _df["log2_minE"] = np.log2(_df["min_E"].replace(0, np.nan))
                # panels
                if "pmd_raw" in _df.columns: _lo(_ax[0], "pmd_raw", f"Raw PMD — {label}", "raw PMD", bounds=(0,1))
                if "chi2" in _df.columns: _lo(_ax[1], "chi2", f"Chi-square — {label}", "chi2")
                if "neglog10_p" in _df.columns: _lo(_ax[2], "neglog10_p", f"-log10(p) — {label}", "-log10 p")
                if "cramers_v_bc" in _df.columns:
                    _lo(_ax[3], "cramers_v_bc", f"Cramer's V (BC) — {label}", "Cramer's V (BC)", bounds=(0,1))
                else:
                    _ax[3].axis("off"); _ax[3].text(0.5, 0.5, "cramers_v_bc missing", ha="center", va="center")
                if "inv_simpson" in _df.columns: _lo(_ax[4], "inv_simpson", f"Inv Simpson — {label}", "inv Simpson")
                if "entropy" in _df.columns: _lo(_ax[5], "entropy", f"Entropy — {label}", "entropy (bits)", bounds=(0,1))
                if merged_null is not None and "pmd_null" in merged_null.columns:
                    # filter null by this dataset's K/N1/N2
                    k, n1, n2 = int(_df["K"].iloc[0]), int(_df["N1"].iloc[0]), int(_df["N2"].iloc[0])
                    s0 = merged_null[(merged_null.get("s", 0) == 0) & (merged_null.get("K",-1)==k) & (merged_null.get("N1",-1)==n1) & (merged_null.get("N2",-1)==n2)]
                    sns.kdeplot(data=s0, x="pmd_null", bw_adjust=1.5, fill=True, alpha=0.3, ax=_ax[6])
                    _ax[6].set_xlim(0,1); _ax[6].set_title("Null PMD density (s=0)"); _ax[6].set_ylabel("density")
                if {"pmd_lambda","total_clusters"}.issubset(_df.columns):
                    sns.scatterplot(data=_df, x="total_clusters", y="pmd_lambda", alpha=0.3, s=20, ax=_ax[7], legend=False)
                    xs = _df["total_clusters"].to_numpy(); ys = _df["pmd_lambda"].to_numpy()
                    ok = np.isfinite(xs) & np.isfinite(ys); xs, ys = xs[ok], ys[ok]
                    if xs.size>=3 and _HAS_LOWESS:
                        fitted = sm_lowess(ys, xs, frac=0.6, return_sorted=True); _ax[7].plot(fitted[:,0], fitted[:,1], lw=2)
                    _ax[7].set_title("lambda vs total clusters"); _ax[7].set_ylabel("lambda (mean null PMD)")
                if {"pmd_lambda","log2_minE"}.issubset(_df.columns):
                    sns.scatterplot(data=_df, x="log2_minE", y="pmd_lambda", alpha=0.3, s=20, ax=_ax[8], legend=False)
                    xs = _df["log2_minE"].to_numpy(); ys = _df["pmd_lambda"].to_numpy()
                    ok = np.isfinite(xs) & np.isfinite(ys); xs, ys = xs[ok], ys[ok]
                    if xs.size>=3 and _HAS_LOWESS:
                        fitted = sm_lowess(ys, xs, frac=0.8, return_sorted=True); _ax[8].plot(fitted[:,0], fitted[:,1], lw=2)
                    _ax[8].set_title("lambda vs log2(min(E))"); _ax[8].set_ylabel("lambda (mean null PMD)")
                if "pmd" in _df.columns: _lo(_ax[9], "pmd", f"PMD* — {label}", "PMD*", bounds=(0,1))
                _fig.suptitle(f"PMD Characterization — {label}", y=1.02, fontsize=16)
                pdf.savefig(_fig, bbox_inches="tight"); plt.close(_fig)
                # comparator page
                _fig2, _axes2 = plt.subplots(1,5, figsize=(22,4.5), sharex=True)
                comps = [("jsd","JSD"),("tv","TV"),("hellinger","Hellinger"),("braycurtis","Bray–Curtis"),("cosine","Cosine")]
                for axh,(metric,title) in zip(_axes2, comps):
                    _lo(axh, metric, f"{title} vs s — {label}", title, bounds=(0,1))
                _fig2.suptitle(f"Comparator Distances — {label}", y=1.02, fontsize=16)
                pdf.savefig(_fig2, bbox_inches="tight"); plt.close(_fig2)
        print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
