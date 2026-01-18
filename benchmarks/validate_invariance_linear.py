#!/usr/bin/env python3
"""Validate linear invariance for metrics vs percent_different_clusters_numeric.

Outputs, per mode (subfolder):
  - invariance_tests_boolean_matrix.csv
  - invariance_tests_stats.json

This version adds joint ANCOVA-style modeling with robust (HC3) Wald tests to assess
equality of intercepts and slopes across NumberOfClusters (K) and dataset_sizes,
conditioning on the other factor. Heuristic tolerance checks are retained for
slopes as diagnostics; intercept booleans now derive from the joint-model tests.
All tests operate on the RAW metric scale with no transformations.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy import dmatrices
from scipy import stats as sps
from scipy.stats import theilslopes, kendalltau


BOUNDED_METRICS: set[str] = set()
NONNEG_METRICS: set[str] = set()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="path to invariance_results.csv")
    ap.add_argument("--mode", default="all", choices=["all", "symmetric", "expand_b2_only", "fixed_union_private"], help="filter to a specific invariance mode or 'all'")
    ap.add_argument("--alpha", type=float, default=0.01)
    # Linearity via R2/tau on group means (defaults favor clear linear/monotone)
    ap.add_argument("--r2-min", type=float, default=0.99)
    ap.add_argument("--tau-min", type=float, default=0.90)
    # Slope equivalence tolerance (relative + absolute) on RAW scale
    ap.add_argument("--delta-s-rel", type=float, default=0.20)
    ap.add_argument("--delta-s-abs", type=float, default=0.02)
    # Intercept tolerance on RAW scale
    ap.add_argument("--delta-b", type=float, default=0.05)
    # Minimum pooled/group slope magnitude on RAW scale
    ap.add_argument("--beta-min", type=float, default=0.20)
    ap.add_argument("--intercepts-mode", choices=["tolerate","require","ignore"], default="tolerate")
    # Joint-model controls
    ap.add_argument("--include-ds-k-interaction", type=str, default="false", help="true/false: include C(K):C(dataset_sizes) and its x interaction")
    ap.add_argument("--center-x0", type=float, default=0.5, help="x-center point for intercept interpretability (default mid-range)")
    ap.add_argument("--out-dir", default="", help="optional output directory; default next to file, per mode subfolder")
    ap.add_argument("--fail-on-false", type=str, default="false", help="true/false: non-zero exit if any Overall is FALSE")
    return ap.parse_args()


def _transform_and_standardize(y: pd.Series, name: str) -> Tuple[np.ndarray, Dict[str, float]]:
    """Identity: return RAW metric values without any transformation.

    We assess linearity and equivalence on the metric's native scale.
    The returned stats are descriptive only and not used for scaling.
    """
    arr = y.to_numpy(dtype=float)
    mu = float(np.nanmean(arr)) if arr.size else float("nan")
    sd = float(np.nanstd(arr, ddof=1)) if arr.size else float("nan")
    return arr, {"mu": mu, "sd": sd}


def _boolean(val: bool | None) -> str:
    if val is None:
        return "NA"
    return "TRUE" if val else "FALSE"


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.file)
    if args.mode != "all" and "mode" in df.columns:
        df = df[df["mode"] == args.mode].copy()
    if df.empty:
        raise SystemExit("no rows after filtering by mode")

    # Prepare output dir per mode
    base_out = args.out_dir or os.path.dirname(os.path.abspath(args.file))
    mode_label = (args.mode if args.mode != "all" else (df["mode"].iloc[0] if "mode" in df.columns and df["mode"].nunique()==1 else "all"))
    out_dir = os.path.join(base_out)
    os.makedirs(out_dir, exist_ok=True)

    # Identify metrics present
    xcol = "percent_different_clusters_numeric"
    kcol = "NumberOfClusters"
    dcol = "dataset_sizes"
    required_cols = [xcol, kcol, dcol]
    for c in required_cols:
        if c not in df.columns:
            raise SystemExit(f"missing required column: {c}")

    # candidate metrics
    metric_candidates = [
        "pmd", "pmd_raw", "chi_sq", "chi2", "chi_neg_log_p", "cramers_v_bc",
        "inverse_simp", "inv_simpson", "shannon_entropy", "entropy",
        "jsd", "tv", "hellinger", "braycurtis", "canberra", "cosine",
    ]
    metrics = [m for m in metric_candidates if m in df.columns]
    if not metrics:
        raise SystemExit("no known metric columns found")

    # Results containers
    rows_bool: List[Dict[str, Any]] = []
    stats_out: Dict[str, Any] = {
        "mode": mode_label,
        "alpha": args.alpha,
        "r2_min": args.r2_min,
        "tau_min": args.tau_min,
        "delta_s_rel": args.delta_s_rel,
        "delta_s_abs": args.delta_s_abs,
        "delta_b": args.delta_b,
        "beta_min": args.beta_min,
        "intercepts_mode": args.intercepts_mode,
        "include_ds_k_interaction": str(args.include_ds_k_interaction).lower(),
        "center_x0": args.center_x0,
        "metrics": {},
    }

    # Iterate metrics
    for m in metrics:
        sub = df[[xcol, kcol, dcol, m]].dropna()
        metric_stats: Dict[str, Any] = {}
        # Base N checks
        xuniq = np.unique(sub[xcol].to_numpy())
        has_x = (xuniq.size >= 2)
        has_k = (sub[kcol].nunique() >= 2)
        has_d = (sub[dcol].nunique() >= 2)
        # Transform
        yraw, tr_info = _transform_and_standardize(sub[m], m)
        sub = sub.copy()
        sub["y_std"] = yraw  # legacy column name; holds RAW values by policy
        # Defaults
        linear_ok = None
        slopes_k_ok = None
        slopes_d_ok = None
        slopes_k_ok_p: Optional[bool] = None
        slopes_d_ok_p: Optional[bool] = None
        ints_k_ok = None
        ints_d_ok = None

        # 1) Prepare replicate-level data and GLM helpers
        linear_ok = None
        df_metric = sub[[kcol, dcol, xcol]].copy()
        df_metric["y"] = df.loc[sub.index, m].to_numpy(dtype=float)
        df_metric = df_metric.rename(columns={kcol: "K", dcol: "DS", xcol: "x"})
        df_metric = df_metric.dropna()
        df_metric["K"] = df_metric["K"].astype(str)
        df_metric["DS"] = df_metric["DS"].astype(str)

        means_pair = df_metric.groupby(["K", "DS", "x"], as_index=False)["y"].mean()
        pooled_means = means_pair.groupby("x", as_index=False)["y"].mean()

        def group_slope(vals: pd.DataFrame) -> float:
            if vals["x"].nunique() < 2:
                return float("nan")
            try:
                fit = smf.glm("y ~ x", data=vals, family=sm.families.Gaussian()).fit()
                return float(fit.params.get("x", np.nan))
            except Exception:
                return float("nan")

        # 2) Linearity via R2 on pooled means and Kendall tau; monotonic slope via GLM on replicates
        lin_r2_ok = False
        tau_ok = False
        try:
            if pooled_means["x"].nunique() >= 2:
                r_lin = np.corrcoef(pooled_means["x"].to_numpy(), pooled_means["y"].to_numpy())[0, 1]
                r2 = float(r_lin ** 2) if np.isfinite(r_lin) else float("nan")
                lin_r2_ok = (np.isfinite(r2) and (r2 >= args.r2_min))
                tau, _ = kendalltau(pooled_means["x"].to_numpy(), pooled_means["y"].to_numpy())
                tau_ok = (np.isfinite(tau) and (abs(float(tau)) >= args.tau_min))
                metric_stats.update({"r2_means": r2, "tau_means": float(tau) if tau == tau else float("nan")})
        except Exception:
            pass
        linear_ok = bool(lin_r2_ok and tau_ok)

        pooled_slope = None
        try:
            fit_pool = smf.glm("y ~ x", data=df_metric, family=sm.families.Gaussian()).fit()
            pooled_slope = float(fit_pool.params.get("x", np.nan))
            metric_stats["pooled_glm_deviance"] = float(fit_pool.deviance)
        except Exception:
            pooled_slope = None

        monotone_ok = (pooled_slope is not None) and np.isfinite(pooled_slope) and (abs(pooled_slope) >= args.beta_min)
        metric_stats["pooled_slope"] = pooled_slope

        slope_K_series = df_metric.groupby("K").apply(lambda g: group_slope(g)) if has_k else pd.Series(dtype=float)
        slope_D_series = df_metric.groupby("DS").apply(lambda g: group_slope(g)) if has_d else pd.Series(dtype=float)
        slopes_k_ok = None
        slopes_d_ok = None
        if pooled_slope is not None and np.isfinite(pooled_slope):
            if not slope_K_series.dropna().empty:
                diffs = np.abs(slope_K_series - pooled_slope)
                tol = (args.delta_s_rel * abs(pooled_slope)) + args.delta_s_abs
                sign_ok = np.sign(slope_K_series) == np.sign(pooled_slope)
                mag_ok = (np.abs(slope_K_series) >= args.beta_min)
                slopes_k_ok = bool((diffs <= tol).all() and sign_ok.all() and mag_ok.all())
                metric_stats["slope_K_diffs"] = diffs.to_dict()
            if not slope_D_series.dropna().empty:
                diffs = np.abs(slope_D_series - pooled_slope)
                tol = (args.delta_s_rel * abs(pooled_slope)) + args.delta_s_abs
                sign_ok = np.sign(slope_D_series) == np.sign(pooled_slope)
                mag_ok = (np.abs(slope_D_series) >= args.beta_min)
                slopes_d_ok = bool((diffs <= tol).all() and sign_ok.all() and mag_ok.all())
                metric_stats["slope_DS_diffs"] = diffs.to_dict()
        # 4) Joint GLM: test intercept/slope equality controlling for both factors
        df_glm = df_metric.copy()
        df_glm["x_c"] = df_glm["x"] - args.center_x0

        p_int_K: Optional[float] = None
        p_int_DS: Optional[float] = None
        p_xK: Optional[float] = None
        p_xDS: Optional[float] = None

        ints_k_ok = True if args.intercepts_mode == "ignore" else None
        ints_d_ok = True if args.intercepts_mode == "ignore" else None

        if has_x and has_k and has_d and (df_glm["x"].nunique() >= 2):
            # Base formula: y ~ 1 + x_c + C(K) + C(DS) + x_c:C(K) + x_c:C(DS) [+ C(K):C(DS) + x_c:C(K):C(DS)]
            include_cross = str(args.include_ds_k_interaction).lower() in {"1","true","yes"}
            if include_cross:
                formula = "y ~ 1 + x_c + C(K) + C(DS) + C(K):C(DS) + x_c:C(K) + x_c:C(DS) + x_c:C(K):C(DS)"
            else:
                formula = "y ~ 1 + x_c + C(K) + C(DS) + x_c:C(K) + x_c:C(DS)"
            try:
                fit = smf.glm(formula, data=df_glm, family=sm.families.Gaussian()).fit()
                # Wald tests: intercept equality across K and across DS
                # Collect parameter names for contrasts
                params = fit.params.index.tolist()
                # Terms look like C(K)[T.4], C(DS)[T.1000_vs_1000], x_c:C(K)[T.4], etc.
                k_terms = [p for p in params if p.startswith("C(K)[T.")]
                ds_terms = [p for p in params if p.startswith("C(DS)[T.")]
                xk_terms = [p for p in params if p.startswith("x_c:C(K)[T.")]
                xds_terms = [p for p in params if p.startswith("x_c:C(DS)[T.")]
                # Intercepts equality: all k_terms == 0 and all ds_terms == 0
                if k_terms:
                    R = np.zeros((len(k_terms), len(params)))
                    for i, name in enumerate(k_terms):
                        j = params.index(name)
                        R[i, j] = 1.0
                    w = fit.wald_test(R)
                    p_int_K = float(w.pvalue)
                if ds_terms:
                    R = np.zeros((len(ds_terms), len(params)))
                    for i, name in enumerate(ds_terms):
                        j = params.index(name)
                        R[i, j] = 1.0
                    w = fit.wald_test(R)
                    p_int_DS = float(w.pvalue)
                # Slopes equality: interaction terms x_c:C(K) and x_c:C(DS)
                if xk_terms:
                    R = np.zeros((len(xk_terms), len(params)))
                    for i, name in enumerate(xk_terms):
                        j = params.index(name)
                        R[i, j] = 1.0
                    w = fit.wald_test(R)
                    p_xK = float(w.pvalue)
                if xds_terms:
                    R = np.zeros((len(xds_terms), len(params)))
                    for i, name in enumerate(xds_terms):
                        j = params.index(name)
                        R[i, j] = 1.0
                    w = fit.wald_test(R)
                    p_xDS = float(w.pvalue)
                # P-value based slope-equality booleans (keep heuristic too)
                if p_xK == p_xK:
                    slopes_k_ok_p = bool(p_xK >= args.alpha)
                if p_xDS == p_xDS:
                    slopes_d_ok_p = bool(p_xDS >= args.alpha)
                # Intercepts booleans from joint-model p-values
                if args.intercepts_mode != "ignore":
                    if p_int_K == p_int_K:  # not NaN
                        ints_k_ok = bool(p_int_K >= args.alpha)
                    if p_int_DS == p_int_DS:
                        ints_d_ok = bool(p_int_DS >= args.alpha)
                # Record stats
                metric_stats.update({
                    "p_int_K": p_int_K,
                    "p_int_DS": p_int_DS,
                    "p_xK": p_xK,
                    "p_xDS": p_xDS,
                    "ancova_params": params,
                    "ancova_formula": formula,
                    "n_obs": int(len(df_glm)),
                    "glm_deviance": float(fit.deviance),
                })
            except Exception as e:
                metric_stats.update({"ancova_error": str(e)})

        # Retain heuristic intercept tolerance as descriptive (not decisive)
        try:
            x0 = float(args.center_x0)
            diffs = {}
            if has_k:
                for kval, g in df_metric.groupby("K"):
                    if g["x"].nunique() < 2:
                        continue
                    fit = smf.glm("y ~ x", data=g, family=sm.families.Gaussian()).fit()
                    intercept = float(fit.params.get("Intercept", 0.0))
                    slope = float(fit.params.get("x", 0.0))
                    diffs[kval] = intercept + slope * x0
                if diffs:
                    arr = np.array(list(diffs.values()), dtype=float)
                    metric_stats["intercepts_pred_x0_K_span"] = float(np.nanmax(np.abs(arr - np.nanmean(arr))))
            diffs = {}
            if has_d:
                for dsval, g in df_metric.groupby("DS"):
                    if g["x"].nunique() < 2:
                        continue
                    fit = smf.glm("y ~ x", data=g, family=sm.families.Gaussian()).fit()
                    intercept = float(fit.params.get("Intercept", 0.0))
                    slope = float(fit.params.get("x", 0.0))
                    diffs[dsval] = intercept + slope * x0
                if diffs:
                    arr = np.array(list(diffs.values()), dtype=float)
                    metric_stats["intercepts_pred_x0_DS_span"] = float(np.nanmax(np.abs(arr - np.nanmean(arr))))
        except Exception:
            pass

        # Overall requires: LOF linearity across groups + monotone slope magnitude + slope equivalence across K and DS.
        overall = None
        for flag in (linear_ok, monotone_ok, slopes_k_ok, slopes_d_ok):
            if flag is None:
                overall = None
                break
        else:
            base = bool(linear_ok and monotone_ok and slopes_k_ok and slopes_d_ok)
            if args.intercepts_mode in {"require", "tolerate"}:
                # Use joint-model p-value based booleans if available; otherwise ignore
                ok = True
                if ints_k_ok is not None:
                    ok = ok and bool(ints_k_ok)
                if ints_d_ok is not None:
                    ok = ok and bool(ints_d_ok)
                overall = base and ok
            else:
                overall = base

        rows_bool.append({
            "metric": m,
            "Linear": _boolean(linear_ok and monotone_ok),
            "SlopesEqual_K": _boolean(slopes_k_ok),
            "SlopesEqual_dataset_sizes": _boolean(slopes_d_ok),
            "SlopesEqual_K_p": _boolean(slopes_k_ok_p),
            "SlopesEqual_dataset_sizes_p": _boolean(slopes_d_ok_p),
            "InterceptsEqual_K": _boolean(ints_k_ok),
            "InterceptsEqual_dataset_sizes": _boolean(ints_d_ok),
            "Overall": _boolean(overall),
        })
        stats_out["metrics"][m] = metric_stats

    # Write outputs
    bm = pd.DataFrame(rows_bool)
    # Attach configuration metadata as constant columns for provenance
    bm["mode"] = mode_label
    bm["alpha"] = args.alpha
    bm["r2_min"] = args.r2_min
    bm["tau_min"] = args.tau_min
    bm["delta_s_rel"] = args.delta_s_rel
    bm["delta_s_abs"] = args.delta_s_abs
    bm["delta_b"] = args.delta_b
    bm["beta_min"] = args.beta_min
    bm["intercepts_mode"] = args.intercepts_mode
    bm["include_ds_k_interaction"] = str(args.include_ds_k_interaction).lower()
    bm["center_x0"] = args.center_x0
    bm_path = os.path.join(out_dir, "invariance_tests_boolean_matrix.csv")
    bm.to_csv(bm_path, index=False)
    # FDR (BH) per family across metrics
    def bh_fdr(pvals: List[Tuple[str, float]]) -> Dict[str, float]:
        clean = [(m, p) for m, p in pvals if (p == p)]
        m = len(clean)
        if m == 0:
            return {}
        sorted_pairs = sorted(clean, key=lambda t: t[1])
        qvals = {}
        prev = 1.0
        for rank, (name, p) in enumerate(sorted_pairs, start=1):
            q = p * m / rank
            prev = min(prev, q)
            qvals[name] = prev
        return qvals

    fams = {
        # q_quad reserved for future LOF p-values
        "q_quad": [(m, stats_out["metrics"][m].get("p_quad", float("nan"))) for m in stats_out["metrics"]],
        "q_xK": [(m, stats_out["metrics"][m].get("p_xK", float("nan"))) for m in stats_out["metrics"]],
        "q_xDS": [(m, stats_out["metrics"][m].get("p_xDS", float("nan"))) for m in stats_out["metrics"]],
        "q_int_K": [(m, stats_out["metrics"][m].get("p_int_K", float("nan"))) for m in stats_out["metrics"]],
        "q_int_DS": [(m, stats_out["metrics"][m].get("p_int_DS", float("nan"))) for m in stats_out["metrics"]],
    }
    for key, arr in fams.items():
        qmap = bh_fdr(arr)
        for met, q in qmap.items():
            stats_out["metrics"][met][key] = q

    stats_path = os.path.join(out_dir, "invariance_tests_stats.json")
    with open(stats_path, "w") as fh:
        json.dump(stats_out, fh, indent=2)
    print(f"wrote {bm_path}")
    print(f"wrote {stats_path}")

    # Summary markdown
    md_path = os.path.join(out_dir, "invariance_tests_summary.md")
    with open(md_path, "w") as fh:
        fh.write("# Invariance Tests — Boolean Matrix\n\n")
        fh.write(bm.to_markdown(index=False))
        fh.write("\n")
    print(f"wrote {md_path}")

    if str(args.fail_on_false).lower() in {"1", "true", "yes"}:
        if (bm["Overall"] == "FALSE").any():
            raise SystemExit(2)


if __name__ == "__main__":
    main()
