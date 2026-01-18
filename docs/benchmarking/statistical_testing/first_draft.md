# Invariance Statistical Testing — First Draft Plan

Owner: PMD maintainers
Date: 2025‑10‑22

## Objective
For each metric function independently (e.g., PMD*, chi‑square, JSD, …), test whether the relationship between percent different clusters (`x = percent_different_clusters_numeric`) and the metric (`y`) in the invariance experiments satisfies all of the following on the bottom rows of the super‑grids:

1) Linearity: the relationship is linear in `x` (no curvature).
2) Slope equality across NumberOfClusters (`K`): no slope differences by `K`.
3) Slope equality across dataset sizes (`dataset_sizes`): no slope differences by dataset sizes.
4) Intercept equality across `K`: no intercept (level) differences by `K`.
5) Intercept equality across `dataset_sizes`: no intercept differences by dataset sizes.

We evaluate each metric independently and populate a Boolean matrix of results (TRUE/FALSE per test per metric), per invariance mode (symmetric, expand_b2_only, fixed_union_private).

## Inputs
- File: `invariance_results.csv` per invariance run (already produced by our benchmark), containing at least:
  - `percent_different_clusters_numeric` (x)
  - `NumberOfClusters` (K)
  - `dataset_sizes` (e.g., "250_vs_500")
  - Per‑metric columns: `pmd`, `chi_sq`, `chi_neg_log_p`, `cramers_v_bc`, `inverse_simp`, `shannon_entropy`, `jsd`, `tv`, `hellinger`, `braycurtis`, `cosine`, `canberra` (subset may be missing depending on config)
  - Optional: `mode` (invariance modality label)

## Metrics Covered (each tested independently)
- Core: `pmd`, `chi_sq`, `chi_neg_log_p`, `cramers_v_bc`, `inverse_simp`, `shannon_entropy`
- Comparators: `jsd`, `tv`, `hellinger`, `braycurtis`, `cosine`, `canberra`

## Data preparation
- Keep rows with finite values for `(x, y, K, dataset_sizes)`.
- Require at least two distinct `x` values and at least two groups for both `K` and `dataset_sizes` to run the homogeneity tests; otherwise the corresponding group tests are marked NA (excluded from decision aggregation but reported).

## Transformations (variance stabilization; applied metric‑wise)
- Bounded metrics in [0, 1]: `pmd`, `jsd`, `tv`, `hellinger`, `braycurtis`, `cosine`, `shannon_entropy`, `cramers_v_bc`
  - Transform: `y* = logit(clip(y, ε, 1−ε))`, with ε = 1e‑6.
- Non‑negative unbounded: `chi_sq`, `chi_neg_log_p`, `canberra`, `inverse_simp`
  - Transform: `y* = log1p(y)`.
- Standardize: `y_std = (y* − mean(y*)) / sd(y*)` for comparability of effect‑size thresholds across metrics.

## Models and Tests (metric‑agnostic; identical across metrics)
Let `x = percent_different_clusters_numeric`, `G1 = C(NumberOfClusters)`, `G2 = C(dataset_sizes)`.
All regression fits use OLS with HC3 robust covariance (heteroskedasticity‑robust). Where needed, we may add a cluster‑robust option on `(G1, G2)` pairs.

1) Linearity (pooled lack‑of‑fit)
- Fit L1: `y_std ~ 1 + x`
- Fit L2: `y_std ~ 1 + x + x^2`
- Wald test for `β2 = 0` with HC3 SE; record p‑value `p_quad`.
- Compute `ΔR^2 = R^2(L2) − R^2(L1)`.
- Decision: Linear = TRUE if `p_quad > α` AND `ΔR^2 < ε_R2`.
  - Defaults: `α = 0.01`, `ε_R2 = 0.02`.

2) Homogeneity of slopes across `K` and `dataset_sizes` (ANCOVA interaction tests)
- Fit: `y_std ~ 1 + x + G1 + G2 + x:G1 + x:G2`.
- Joint robust Wald tests:
  - `H0_Kslope`: all `x:G1` coefficients = 0 ⇒ p‑value `p_xK`.
  - `H0_Dslope`: all `x:G2` coefficients = 0 ⇒ p‑value `p_xDS`.
- Guardrail effect‑size: compute per‑group slopes from groupwise `y_std ~ x` fits; require
  - `max_group_slope_diff ≤ δ_s`, with `δ_s = 0.10` (in standardized units) or ≤10% of pooled `|β1|` when `β1 ≠ 0`.
- Decision: SlopesEqual = TRUE if `p_xK > α` AND `p_xDS > α` AND guardrail satisfied.

3) Homogeneity of intercepts across `K` and `dataset_sizes`
- Fit reduced (no interactions): `y_std ~ 1 + x + G1 + G2`.
- Joint robust Wald tests:
  - `H0_Kint`: all `G1` levels equal ⇒ p‑value `p_K`.
  - `H0_Dint`: all `G2` levels equal ⇒ p‑value `p_DS`.
- Guardrail: compute group intercepts (fitted means at `x = 0`); require
  - `max_group_intercept_diff ≤ δ_b`, `δ_b = 0.10`.
- Decision: InterceptsEqual = TRUE if `p_K > α` AND `p_DS > α` AND guardrail satisfied.

4) Overall metric decision (per mode)
- TRUE if (1) Linearity AND (2) SlopesEqual AND (3) InterceptsEqual all pass.
- Record each component decision separately (no masking).

## Boolean Matrix Output
- For each invariance mode independently (e.g., symmetric, expand_b2_only, fixed_union_private):
  - Rows: metric functions (e.g., `pmd`, `chi_sq`, ..., `canberra`).
  - Columns (booleans):
    - `Linear`
    - `SlopesEqual_K`
    - `SlopesEqual_dataset_sizes`
    - `InterceptsEqual_K`
    - `InterceptsEqual_dataset_sizes`
    - `Overall` (AND of the above three families: Linear & SlopesEqual (both) & InterceptsEqual (both))
  - NA allowed when a test cannot be run (e.g., only one `K` present). `Overall` becomes NA if any required component is NA.
- Store as CSV and JSON for machine and human use:
  - `invariance_tests_boolean_matrix_<mode>.csv`
  - `invariance_tests_stats_<mode>.json` (contains p‑values, ΔR^2, max diff magnitudes, sample sizes).

## Edge Cases & Rules
- If `x` has < 2 unique values, mark Linear=NA and skip other tests.
- If a group factor has < 2 levels, mark corresponding slope/intercept tests NA.
- If a metric is entirely missing or constant after transform, mark all tests NA.
- Robustness: use HC3; optionally enable clustered SE by `(G1,G2)` if repeated measures inflates Type I error.

## Implementation Outline (Phase A)
- Script: `benchmarks/validate_invariance_linear.py`
  - CLI: `--file`, `--mode {all|symmetric|expand_b2_only|fixed_union_private}`, `--alpha`, `--delta-s`, `--delta-b`, `--epsilon-r2`, `--out-dir`.
  - Steps per mode:
    1. Load, filter by mode (if provided), complete cases for (x, y, G1, G2).
    2. Transform and standardize `y` per metric.
    3. Fit L1/L2; compute `p_quad`, `ΔR^2` → Linear boolean.
    4. Fit interaction model; joint Wald tests; compute slope guardrail → `SlopesEqual_K`, `SlopesEqual_dataset_sizes` booleans.
    5. Fit reduced model; joint Wald tests; compute intercept guardrail → `InterceptsEqual_K`, `InterceptsEqual_dataset_sizes` booleans.
    6. Assemble Boolean matrix row for the metric; write per‑mode CSV + JSON with stats.
  - Exit code non‑zero if any metric Overall=FALSE (optional gate for CI).

## Integration (Phase A.1)
- `benchmarks/run_all_benchmarks.py` (optional flag `--validate-invariance true`) to invoke the validator after each invariance mode and collect outputs.

## Documentation (Phase C)
- Add `docs/benchmarking/invariance_tests.md` (later) with:
  - Model equations, transformation table, thresholds, interpretation.
  - Illustrative PASS/FAIL examples and rationale for guardrails.

## Defaults (tunable)
- α = 0.01 (strict; robust to multiple metrics)
- ε_R2 = 0.02 (quadratic incremental fit threshold)
- δ_s = 0.10 (max slope difference on standardized scale)
- δ_b = 0.10 (max intercept difference on standardized scale)

## Notes on Independence Between Functions
- Each metric function is tested independently of the others: data preparation, transformations, model fits, and decisions are performed metric‑wise with no pooling or borrowing of strength across metrics.
- The Boolean matrix aggregates independent decisions; no cross‑metric dependence is assumed.

