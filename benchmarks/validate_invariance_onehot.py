#!/usr/bin/env python3
"""Validate invariance via numeric vs one-hot modeling of percent_different_clusters.

This complements the ANCOVA-based validator by explicitly comparing:
  • M_num:  y ~ 1 + x + C(K) + C(DS)           (linear in x)
  • M_cat:  y ~ 1 + C(x) + C(K) + C(DS)        (one-hot x levels)

Optional interactions (`--include-interactions true`) add x:C(K) + x:C(DS) to both models.
All analyses operate on RAW metric values (no transformations).

Outputs per mode:
  - invariance_onehot_boolean_matrix.csv
  - invariance_onehot_stats.json
  - invariance_onehot_summary.md

The boolean matrix exposes both linearity diagnostics (numeric vs one-hot) and checks that
no additional categorical predictors (K or dataset_sizes) explain the means beyond x.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.stats import kendalltau
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy import dmatrices


@dataclass
class CVResult:
    rmse_num: Optional[float]
    rmse_cat: Optional[float]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to invariance_results.csv")
    ap.add_argument("--mode", default="all", choices=["all", "symmetric", "expand_b2_only", "fixed_union_private"], help="Filter by invariance mode or use all rows")
    ap.add_argument("--alpha", type=float, default=0.01, help="Significance level for Wald tests")
    ap.add_argument("--bic-tolerance", type=float, default=30.0, help="Tolerance when comparing BIC (numeric must be within tol of categorical to be considered linear)")
    ap.add_argument("--cv-folds", type=int, default=0, help="Optional k-fold CV (k > 1) for RMSE comparison")
    ap.add_argument("--cv-epsilon", type=float, default=0.0, help="Tolerance on CV RMSE (numeric <= categorical + epsilon)")
    ap.add_argument("--delta-b", type=float, default=0.05, help="Tolerance for intercept equivalence based on coefficient magnitude")
    ap.add_argument("--include-interactions", type=str, default="false", help="true/false: include x:C(K) and x:C(DS) terms")
    ap.add_argument("--dev-improvement-min", type=float, default=0.05, help="Minimum relative deviance reduction (categorical vs numeric) required to call the relationship non-linear")
    ap.add_argument("--out-dir", default="", help="Optional output directory; defaults next to input file")
    ap.add_argument("--fail-on-false", type=str, default="false", help="true/false: non-zero exit if any Overall is FALSE")
    return ap.parse_args()


def _boolean(val: Optional[bool]) -> str:
    if val is None:
        return "NA"
    return "TRUE" if val else "FALSE"


def _clean_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "all":
        return df.copy()
    if "mode" not in df.columns:
        raise SystemExit("mode column absent; cannot filter")
    sub = df[df["mode"] == mode].copy()
    if sub.empty:
        raise SystemExit(f"no rows remain after filtering mode={mode}")
    return sub


def _fit_model(formula: str, data: pd.DataFrame):
    try:
        model = smf.glm(formula, data=data, family=sm.families.Gaussian())
        res = model.fit(cov_type="HC3")
        return res
    except Exception:
        return None


def _prepare_cv_folds(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    if k <= 1 or k > n:
        raise ValueError("cv-folds must be between 2 and number of observations")
    indices = np.arange(n)
    rng.shuffle(indices)
    fold_ids = np.zeros(n, dtype=int)
    for i, idx in enumerate(indices):
        fold_ids[idx] = i % k
    return fold_ids


def _cv_rmse(data: pd.DataFrame, formula_num: str, formula_cat: str, folds: int, rng: np.random.Generator) -> CVResult:
    try:
        fold_ids = _prepare_cv_folds(len(data), folds, rng)
    except ValueError:
        return CVResult(None, None)

    se_num = 0.0
    se_cat = 0.0
    n_num = 0
    n_cat = 0
    for fold in range(folds):
        train = data[fold_ids != fold]
        test = data[fold_ids == fold]
        if train["x"].nunique() < 2 or test.empty:
            continue
        fit_num = _fit_model(formula_num, train)
        fit_cat = _fit_model(formula_cat, train)
        if fit_num is not None:
            try:
                preds = fit_num.predict(test)
                se_num += float(np.sum((test["y"] - preds) ** 2))
                n_num += len(test)
            except Exception:
                pass
        if fit_cat is not None:
            try:
                preds = fit_cat.predict(test)
                se_cat += float(np.sum((test["y"] - preds) ** 2))
                n_cat += len(test)
            except Exception:
                pass

    rmse_num = np.sqrt(se_num / n_num) if n_num > 0 else None
    if n_cat > 0:
        rmse_cat = np.sqrt(se_cat / n_cat)
    elif n_num > 0:
        rmse_cat = float("inf")
    else:
        rmse_cat = None
    return CVResult(rmse_num, rmse_cat)


def _loxo_rmse(data: pd.DataFrame, formula_num: str, formula_cat: str) -> CVResult:
    unique_x = data["x"].unique()
    if unique_x.size < 2:
        return CVResult(None, None)
    se_num = 0.0
    se_cat = 0.0
    n_num = 0
    n_cat = 0
    for xv in unique_x:
        train = data[data["x"] != xv]
        test = data[data["x"] == xv]
        if train.empty or test.empty:
            continue
        if train["x"].nunique() < 2:
            continue
        fit_num = _fit_model(formula_num, train)
        fit_cat = _fit_model(formula_cat, train)
        if fit_num is not None:
            try:
                preds = fit_num.predict(test)
                se_num += float(np.sum((test["y"] - preds) ** 2))
                n_num += len(test)
            except Exception:
                pass
        if fit_cat is not None:
            try:
                preds = fit_cat.predict(test)
                se_cat += float(np.sum((test["y"] - preds) ** 2))
                n_cat += len(test)
            except Exception:
                pass
    rmse_num = np.sqrt(se_num / n_num) if n_num > 0 else None
    if n_cat > 0:
        rmse_cat = np.sqrt(se_cat / n_cat)
    elif n_num > 0:
        rmse_cat = float("inf")
    else:
        rmse_cat = None
    return CVResult(rmse_num, rmse_cat)


def _wald_pvalue(res, term_prefix: str) -> Optional[float]:
    if res is None:
        return None
    params = res.params.index.tolist()
    idxs = [i for i, name in enumerate(params) if name.startswith(term_prefix)]
    if not idxs:
        return None
    R = np.zeros((len(idxs), len(params)))
    for row, col in enumerate(idxs):
        R[row, col] = 1.0
    try:
        w = res.wald_test(R)
        return float(w.pvalue)
    except Exception:
        return None


def _partial_r2(res_full, res_reduced) -> Optional[float]:
    if res_full is None or res_reduced is None:
        return None
    try:
        dev_full = float(res_full.deviance)
        dev_red = float(res_reduced.deviance)
        if dev_red <= 0:
            return None
        return max(0.0, 1.0 - dev_full / dev_red)
    except Exception:
        return None


def _fit_reduced(formula: str, data: pd.DataFrame) -> Optional[Any]:
    try:
        return smf.glm(formula, data=data, family=sm.families.Gaussian()).fit(cov_type="HC3")
    except Exception:
        return None


def _bh_fdr(pairs: List[Tuple[str, Optional[float]]]) -> Dict[str, float]:
    clean = [(name, p) for name, p in pairs if p is not None and p == p]
    m = len(clean)
    if m == 0:
        return {}
    sorted_pairs = sorted(clean, key=lambda t: t[1])
    qvals: Dict[str, float] = {}
    prev = 1.0
    for rank, (name, p) in enumerate(sorted_pairs, start=1):
        q = p * m / rank
        prev = min(prev, q)
        qvals[name] = prev
    return qvals


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(42)

    df = pd.read_csv(args.file)
    df = _clean_mode(df, args.mode)

    xcol = "percent_different_clusters_numeric"
    kcol = "NumberOfClusters"
    dcol = "dataset_sizes"

    for col in (xcol, kcol, dcol):
        if col not in df.columns:
            raise SystemExit(f"missing required column: {col}")

    metrics = [c for c in df.columns if c not in {xcol, kcol, dcol, "mode", "iter", "b1_size", "b2_size", "total_number_of_clusters", "total_cells"}]
    numeric_metrics = []
    for m in metrics:
        if pd.api.types.is_numeric_dtype(df[m]):
            numeric_metrics.append(m)
    if not numeric_metrics:
        raise SystemExit("no numeric metric columns found")

    base_dir = args.out_dir or os.path.dirname(os.path.abspath(args.file))
    mode_label = args.mode if args.mode != "all" else (df["mode"].iloc[0] if "mode" in df.columns and df["mode"].nunique() == 1 else "all")
    out_dir = os.path.join(base_dir, f"{mode_label}")
    os.makedirs(out_dir, exist_ok=True)

    include_interactions = str(args.include_interactions).lower() in {"1", "true", "yes"}
    matrices_dir = os.path.join(out_dir, "model_matrices")
    os.makedirs(matrices_dir, exist_ok=True)

    bool_rows: List[Dict[str, str]] = []
    stats: Dict[str, Any] = {
        "mode": mode_label,
        "alpha": args.alpha,
        "bic_tolerance": args.bic_tolerance,
        "cv_folds": args.cv_folds,
        "cv_epsilon": args.cv_epsilon,
        "delta_b": args.delta_b,
        "include_interactions": include_interactions,
        "dev_improvement_min": args.dev_improvement_min,
        "metrics": {},
    }

    for metric in sorted(numeric_metrics):
        detail: Dict[str, Any] = {}
        df_metric = df[[xcol, kcol, dcol, metric]].dropna().copy()
        df_metric = df_metric.rename(columns={xcol: "x", kcol: "K", dcol: "DS", metric: "y"})
        df_metric["K"] = df_metric["K"].astype(str)
        df_metric["DS"] = df_metric["DS"].astype(str)
        if df_metric.empty:
            bool_rows.append({
                "metric": metric,
                "Linear_onehot_vs_numeric": "NA",
                "Slope_nonzero": "NA",
                "InterceptsEqual_K_zero": "NA",
                "InterceptsEqual_DS_zero": "NA",
                "SlopesEqual_K_zero": "NA",
                "SlopesEqual_DS_zero": "NA",
                "Overall": "NA",
            })
            stats["metrics"][metric] = detail
            continue

        has_x = df_metric["x"].nunique() >= 2
        has_k = df_metric["K"].nunique() >= 2
        has_ds = df_metric["DS"].nunique() >= 2

        # Baseline defaults
        linear_bool = None
        slope_bool = None
        intercept_k_bool = None
        intercept_ds_bool = None
        slope_k_bool = None
        slope_ds_bool = None
        linear_bic = None
        linear_cv = None
        linear_loxo = None
        linear_lr = None
        linear_dev = None

        if not has_x:
            detail["warning"] = "insufficient unique x"
        
        # Prepare formulas
        if has_x:
            terms_num = ["1", "x"]
            terms_cat = ["1", "C(x)"]
            if has_k:
                terms_num.append("C(K)")
                terms_cat.append("C(K)")
            if has_ds:
                terms_num.append("C(DS)")
                terms_cat.append("C(DS)")
            interaction_terms = []
            if include_interactions:
                if has_k:
                    interaction_terms.append("x:C(K)")
                if has_ds:
                    interaction_terms.append("x:C(DS)")
            formula_num = "y ~ " + " + ".join(dict.fromkeys(terms_num + interaction_terms))
            formula_cat = "y ~ " + " + ".join(dict.fromkeys(terms_cat + interaction_terms))
            if include_interactions:
                detail["interactions"] = interaction_terms

            fit_num = _fit_model(formula_num, df_metric)
            fit_cat = _fit_model(formula_cat, df_metric)

            safe_metric = metric.replace(os.sep, "_")
            try:
                y_num, X_num = dmatrices(formula_num, df_metric, return_type="dataframe")
                X_num.insert(0, "y", y_num.iloc[:, 0])
                num_path = os.path.join(matrices_dir, f"{safe_metric}_numeric_design.csv")
                X_num.to_csv(num_path, index=False)
                detail["design_numeric_csv"] = num_path
            except Exception as exc:
                detail["design_numeric_error"] = str(exc)
            try:
                y_cat, X_cat = dmatrices(formula_cat, df_metric, return_type="dataframe")
                X_cat.insert(0, "y", y_cat.iloc[:, 0])
                cat_path = os.path.join(matrices_dir, f"{safe_metric}_categorical_design.csv")
                X_cat.to_csv(cat_path, index=False)
                detail["design_categorical_csv"] = cat_path
            except Exception as exc:
                detail["design_categorical_error"] = str(exc)

            if fit_num is not None:
                detail.update({
                    "aic_num": float(fit_num.aic),
                    "bic_num": float(fit_num.bic),
                    "df_resid_num": float(fit_num.df_resid),
                })
            if fit_cat is not None:
                detail.update({
                    "aic_cat": float(fit_cat.aic),
                    "bic_cat": float(fit_cat.bic),
                    "df_resid_cat": float(fit_cat.df_resid),
                })

            linear_dev = None
            if fit_num is not None and fit_cat is not None:
                delta_bic = float(fit_cat.bic - fit_num.bic)
                detail["delta_bic"] = delta_bic
                linear_bic = bool(delta_bic >= -args.bic_tolerance)
                detail["linear_bic"] = linear_bic
                try:
                    df_diff = max(int(round(fit_cat.df_model - fit_num.df_model)), 1)
                    dev_diff = float(fit_num.deviance - fit_cat.deviance)
                    if dev_diff < 0:
                        dev_diff = 0.0
                    p_lr = float(sps.chi2.sf(dev_diff, df_diff)) if df_diff > 0 else float("nan")
                    detail["p_linear_lr"] = p_lr
                    linear_lr = bool(not np.isfinite(p_lr) or p_lr >= args.alpha)
                except Exception:
                    linear_lr = None
                try:
                    dev_num = float(fit_num.deviance)
                    dev_cat = float(fit_cat.deviance)
                    if np.isfinite(dev_num) and dev_num > 1e-9:
                        dev_improve = max(dev_num - dev_cat, 0.0)
                        dev_ratio = dev_improve / dev_num
                        detail["dev_improvement_ratio"] = dev_ratio
                        linear_dev = bool(dev_ratio <= args.dev_improvement_min)
                except Exception:
                    linear_dev = None

            if args.cv_folds and args.cv_folds > 1 and has_x:
                cv = _cv_rmse(df_metric, formula_num, formula_cat, args.cv_folds, rng)
                detail.update({
                    "cv_rmse_num": cv.rmse_num,
                    "cv_rmse_cat": cv.rmse_cat,
                })
                if cv.rmse_num is not None and cv.rmse_cat is not None:
                    linear_cv = bool(cv.rmse_num <= cv.rmse_cat + args.cv_epsilon)
                    detail["linear_cv"] = linear_cv

            loxo = _loxo_rmse(df_metric, formula_num, formula_cat)
            detail.update({
                "loxo_rmse_num": loxo.rmse_num,
                "loxo_rmse_cat": loxo.rmse_cat,
            })
            if loxo.rmse_num is not None and loxo.rmse_cat is not None:
                linear_loxo = bool(loxo.rmse_num <= loxo.rmse_cat + args.cv_epsilon)
                detail["linear_loxo"] = linear_loxo

            if fit_num is not None:
                try:
                    beta_x = float(fit_num.params.get("x", np.nan))
                    p_x = float(fit_num.pvalues.get("x", np.nan))
                    detail.update({"beta_x": beta_x, "p_x": p_x})
                    slope_bool = bool(np.isfinite(p_x) and p_x < args.alpha and np.isfinite(beta_x) and beta_x != 0.0)
                except Exception:
                    pass

            # Kendall tau on pooled means
            try:
                pooled = df_metric.groupby("x", as_index=False)["y"].mean()
                if pooled["x"].nunique() >= 2:
                    tau, _ = kendalltau(pooled["x"].to_numpy(), pooled["y"].to_numpy())
                    detail["kendall_tau"] = float(tau)
            except Exception:
                pass

            if fit_num is not None and has_k:
                p_int_k = _wald_pvalue(fit_num, "C(K)[T.")
                detail["p_int_K"] = p_int_k
                intercept_k_bool = None if p_int_k is None else bool(p_int_k >= args.alpha)
            if fit_num is not None and has_ds:
                p_int_ds = _wald_pvalue(fit_num, "C(DS)[T.")
                detail["p_int_DS"] = p_int_ds
                intercept_ds_bool = None if p_int_ds is None else bool(p_int_ds >= args.alpha)

            if fit_num is not None and has_k:
                k_effects = [abs(float(fit_num.params.get(name, 0.0))) for name in fit_num.params.index if name.startswith("C(K)[T.")]
                if k_effects:
                    max_k = max(k_effects)
                    detail["max_abs_CK_coef"] = max_k
                    within_tol = bool(max_k <= args.delta_b)
                    if intercept_k_bool is None:
                        intercept_k_bool = within_tol
                    else:
                        intercept_k_bool = bool(intercept_k_bool or within_tol)
            if fit_num is not None and has_ds:
                ds_effects = [abs(float(fit_num.params.get(name, 0.0))) for name in fit_num.params.index if name.startswith("C(DS)[T.")]
                if ds_effects:
                    max_ds = max(ds_effects)
                    detail["max_abs_CDS_coef"] = max_ds
                    within_tol = bool(max_ds <= args.delta_b)
                    if intercept_ds_bool is None:
                        intercept_ds_bool = within_tol
                    else:
                        intercept_ds_bool = bool(intercept_ds_bool or within_tol)

            # Partial R^2 for K and DS blocks (main effects)
            if fit_num is not None:
                if has_k:
                    terms_no_k = ["1", "x"]
                    if has_ds:
                        terms_no_k.append("C(DS)")
                    if include_interactions and has_ds:
                        terms_no_k.append("x:C(DS)")
                    formula_no_k = "y ~ " + " + ".join(dict.fromkeys(terms_no_k))
                    fit_no_k = _fit_reduced(formula_no_k, df_metric)
                    detail["partial_r2_K"] = _partial_r2(fit_num, fit_no_k)
                if has_ds:
                    terms_no_ds = ["1", "x"]
                    if has_k:
                        terms_no_ds.append("C(K)")
                    if include_interactions and has_k:
                        terms_no_ds.append("x:C(K)")
                    formula_no_ds = "y ~ " + " + ".join(dict.fromkeys(terms_no_ds))
                    fit_no_ds = _fit_reduced(formula_no_ds, df_metric)
                    detail["partial_r2_DS"] = _partial_r2(fit_num, fit_no_ds)

            if include_interactions and fit_num is not None:
                if has_k:
                    p_slope_k = _wald_pvalue(fit_num, "x:C(K)[T.")
                    detail["p_slope_K"] = p_slope_k
                    slope_k_bool = None if p_slope_k is None else bool(p_slope_k >= args.alpha)
                    terms_no_xk = ["1", "x"]
                    if has_k:
                        terms_no_xk.append("C(K)")
                    if has_ds:
                        terms_no_xk.append("C(DS)")
                    if include_interactions and has_ds:
                        terms_no_xk.append("x:C(DS)")
                    formula_no_xk = "y ~ " + " + ".join(dict.fromkeys(terms_no_xk))
                    fit_no_xk = _fit_reduced(formula_no_xk, df_metric)
                    detail["partial_r2_xK"] = _partial_r2(fit_num, fit_no_xk)
                if has_ds:
                    p_slope_ds = _wald_pvalue(fit_num, "x:C(DS)[T.")
                    detail["p_slope_DS"] = p_slope_ds
                    slope_ds_bool = None if p_slope_ds is None else bool(p_slope_ds >= args.alpha)
                    terms_no_xds = ["1", "x"]
                    if has_k:
                        terms_no_xds.append("C(K)")
                    if has_ds:
                        terms_no_xds.append("C(DS)")
                    if include_interactions and has_k:
                        terms_no_xds.append("x:C(K)")
                    formula_no_xds = "y ~ " + " + ".join(dict.fromkeys(terms_no_xds))
                    fit_no_xds = _fit_reduced(formula_no_xds, df_metric)
                    detail["partial_r2_xDS"] = _partial_r2(fit_num, fit_no_xds)

        # Resolve linear bool precedence: LOXO > CV > BIC
        if linear_loxo is not None:
            if linear_bic is None:
                linear_bool = linear_loxo
            else:
                linear_bool = bool(linear_loxo and linear_bic)
        elif linear_cv is not None:
            if linear_bic is None:
                linear_bool = linear_cv
            else:
                linear_bool = bool(linear_cv and linear_bic)
        elif linear_bic is not None:
            linear_bool = linear_bic

        if linear_lr is not None:
            detail["linear_lr"] = linear_lr
            if linear_bool is None:
                linear_bool = linear_lr
            else:
                linear_bool = bool(linear_bool and linear_lr)

        if linear_dev is not None:
            detail["linear_dev"] = linear_dev
            if linear_bool is None:
                linear_bool = linear_dev
            else:
                linear_bool = bool(linear_bool and linear_dev)
                dev_num = float(fit_num.deviance)
                dev_cat = float(fit_cat.deviance)
                if np.isfinite(dev_num) and dev_num > 1e-9:
                    dev_improve = max(dev_num - dev_cat, 0.0)
                    dev_ratio = dev_improve / dev_num
                    detail["dev_improvement_ratio"] = dev_ratio
                    linear_dev = bool(dev_ratio <= args.dev_improvement_min)

        # Overall decision combines available booleans
        overall = None
        requirements = [linear_bool, slope_bool, intercept_k_bool, intercept_ds_bool]
        if include_interactions:
            requirements.extend([slope_k_bool, slope_ds_bool])
        if all(r is None for r in requirements):
            overall = None
        else:
            ok = True
            for r in requirements:
                if r is not None:
                    ok = ok and bool(r)
            overall = ok

        bool_row = {
            "metric": metric,
            "Linear_onehot_vs_numeric": _boolean(linear_bool),
            "Slope_nonzero": _boolean(slope_bool),
            "InterceptsEqual_K_zero": _boolean(intercept_k_bool),
            "InterceptsEqual_DS_zero": _boolean(intercept_ds_bool),
        }
        if include_interactions:
            bool_row["SlopesEqual_K_zero"] = _boolean(slope_k_bool)
            bool_row["SlopesEqual_DS_zero"] = _boolean(slope_ds_bool)
        else:
            bool_row["SlopesEqual_K_zero"] = "NA"
            bool_row["SlopesEqual_DS_zero"] = "NA"
        bool_row["Overall"] = _boolean(overall)
        bool_rows.append(bool_row)
        stats["metrics"][metric] = detail

    bm = pd.DataFrame(bool_rows)
    bm_path = os.path.join(out_dir, "invariance_onehot_boolean_matrix.csv")
    bm.to_csv(bm_path, index=False)

    # FDR adjustments
    fam_linearity = [(m, stats["metrics"].get(m, {}).get("p_x")) for m in stats["metrics"]]
    fam_int_k = [(m, stats["metrics"].get(m, {}).get("p_int_K")) for m in stats["metrics"]]
    fam_int_ds = [(m, stats["metrics"].get(m, {}).get("p_int_DS")) for m in stats["metrics"]]
    fam_slope_k = [(m, stats["metrics"].get(m, {}).get("p_slope_K")) for m in stats["metrics"]]
    fam_slope_ds = [(m, stats["metrics"].get(m, {}).get("p_slope_DS")) for m in stats["metrics"]]

    stats.setdefault("q_values", {})
    stats["q_values"]["q_x"] = _bh_fdr(fam_linearity)
    stats["q_values"]["q_int_K"] = _bh_fdr(fam_int_k)
    stats["q_values"]["q_int_DS"] = _bh_fdr(fam_int_ds)
    stats["q_values"]["q_slope_K"] = _bh_fdr(fam_slope_k)
    stats["q_values"]["q_slope_DS"] = _bh_fdr(fam_slope_ds)

    stats_path = os.path.join(out_dir, "invariance_onehot_stats.json")
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)

    summary_path = os.path.join(out_dir, "invariance_onehot_summary.md")
    with open(summary_path, "w") as fh:
        fh.write("# Invariance One-Hot Validator — Boolean Matrix\n\n")
        fh.write(bm.to_markdown(index=False))
        fh.write("\n")

    print(f"wrote {bm_path}")
    print(f"wrote {stats_path}")
    print(f"wrote {summary_path}")

    if str(args.fail_on_false).lower() in {"1", "true", "yes"}:
        if (bm["Overall"] == "FALSE").any():
            raise SystemExit(2)


if __name__ == "__main__":
    main()
