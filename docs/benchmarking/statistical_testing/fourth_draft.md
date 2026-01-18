# One-Hot vs Numeric Invariance Validator — Fourth Draft Plan

Owner: PMD maintainers  
Date: 2025-10-25

All items begin pending `[ ]`; mark `[x]` when finished. Work in phases so we can compare against the previous ANCOVA-based validator while it remains available.

## Phase 1 — CLI & Data Prep
- [x] Create `benchmarks/validate_invariance_onehot.py` with CLI options:
  - [x] `--file`, `--mode`, `--alpha`, `--bic-tolerance` (default 30.0), `--cv-folds`, `--cv-epsilon`, `--include-interactions`, `--out-dir`, `--fail-on-false`.
- [x] Load `invariance_results.csv`, filter by mode, keep columns `{percent_different_clusters_numeric, NumberOfClusters, dataset_sizes, metric}`.
- [x] Aggregate to group means per `(NumberOfClusters, dataset_sizes, x)` for each metric.
- [x] Guardrails: if <2 unique `x` or <2 levels in factors, record NA and continue.

## Phase 2 — Model Specifications (Raw Scale)
- [x] Encode `NumberOfClusters` and `dataset_sizes` as categorical dummies (drop-one).
- [x] Fit numeric-x model `M_num: y ~ 1 + x + C(K) + C(DS)` with HC3 covariance.
- [x] Fit categorical-x model `M_cat: y ~ 1 + C(x) + C(K) + C(DS)` with HC3 covariance.
- [x] Optional interactions (`--include-interactions`): add `+ x:C(K) + x:C(DS)` to both models (use numeric x for interaction).
- [x] Persist fit statistics (params, HC3 p-values, RSS, df_resid, AIC, BIC).

## Phase 3 — Linearity & Monotonicity Decisions
- [x] Compute ΔBIC = `BIC_cat - BIC_num`; declare `Linear_onehot_vs_numeric = (ΔBIC >= -bic_tolerance)`.
- [x] Optional CV (`--cv-folds > 0`): leave-one-x-level-out RMSE for both models; declare linear if `RMSE_num <= RMSE_cat + cv_epsilon`.
- [x] Test slope non-zero using `M_num` coefficient for `x` (HC3 p-value) and record sign.
- [x] Compute Kendall’s τ on pooled means as a diagnostic.

## Phase 4 — “Only x matters” Invariance Tests
- [x] Wald test that all `C(K)` coefficients = 0 → `InterceptsEqual_K_zero`.
- [x] Wald test that all `C(DS)` coefficients = 0 → `InterceptsEqual_DS_zero`.
- [x] (If interactions enabled) Wald test all `x:C(K)` = 0 → `SlopesEqual_K_zero`.
- [x] (If interactions enabled) Wald test all `x:C(DS)` = 0 → `SlopesEqual_DS_zero`.
- [x] Record partial R² or variance explained per factor block.

- [x] Write `<mode>/invariance_onehot_boolean_matrix.csv` with columns:
  - `Linear_onehot_vs_numeric`, `Slope_nonzero`, `InterceptsEqual_K_zero`, `InterceptsEqual_DS_zero`, optional slope columns, and `Overall` variants.
- [x] Write `<mode>/invariance_onehot_stats.json` capturing fit metrics, p-values, ΔBIC, CV stats, Kendall’s τ, sample sizes.
- [x] Apply BH FDR per family (linearity, intercepts, slopes) and store q-values.
- [x] Emit markdown summary `<mode>/invariance_onehot_summary.md` embedding the boolean matrix and key deltas.
- [x] Persist model design matrices (`model_matrices/<metric>_{numeric|categorical}_design.csv`).

## Phase 6 — Integration & Tests
- [x] Update `benchmarks/run_all_benchmarks.py` to invoke the new validator when `--validate-invariance true`.
- [x] Add smoke tests with synthetic CSVs covering: perfect linear invariance, nonlinear x, intercept shift by DS, slope shift by K.
- [ ] Document defaults in `docs/benchmarking/invariance_tests.md` after validator stabilizes.
