# Invariance Statistical Testing — Second Draft (Synthesis)

Owner: PMD maintainers
Date: 2025‑10‑22

This second draft integrates the peer‑review feedback to strengthen validity, align more closely with “practical invariance,” and keep outputs simple (a Boolean matrix per metric, per mode).

## Goals
- Per‑metric, per‑mode decisions with a Boolean matrix (Linear, SlopesEqual_K, SlopesEqual_DS, InterceptsEqual_K, InterceptsEqual_DS, Overall).
- Treat each metric independently and uniformly across core and comparator panels.
- Prefer “practical invariance” via equivalence margins in addition to NHST.

## Phase 1 — Baseline (Implementable Now)
- [ ] Script `benchmarks/validate_invariance_linear.py` implementing:
  - [ ] Data prep: complete cases for (x, y, K, dataset_sizes); minimum two x values; at least two levels for K and dataset_sizes or mark tests NA.
  - [ ] Transformations (per metric): logit for [0,1] metrics (ε=1e‑6); log1p for non‑negative metrics; then standardize `y_std`.
  - [ ] Linearity test: OLS‑HC3 compare L1 (y~1+x) vs L2 (y~1+x+x²); record p_quad, ΔR²; Linear = (p_quad>α & ΔR²<ε_R2).
  - [ ] Slope homogeneity (NHST): OLS‑HC3 with interactions y~1+x+G1+G2+x:G1+x:G2; robust Wald for all x:G1 (p_xK) and x:G2 (p_xDS).
  - [ ] Intercept homogeneity (NHST): y~1+x+G1+G2; robust Wald for G1 (p_K) and G2 (p_DS).
  - [ ] Guardrails: max slope diff ≤ δ_s; max intercept diff ≤ δ_b (in standardized units) computed from groupwise fits.
  - [ ] Decision matrix per metric: booleans for Linear, SlopesEqual_K, SlopesEqual_DS, InterceptsEqual_K, InterceptsEqual_DS, Overall.
  - [ ] Outputs: `<mode>/invariance_tests_boolean_matrix.csv` and `<mode>/invariance_tests_stats.json` with p‑values, ΔR², max diffs, sample sizes.
- [ ] run_all integration (flag `--validate-invariance true`) to call the validator per mode.

Defaults: α=0.01, ε_R2=0.02, δ_s=0.10, δ_b=0.10.

## Phase 2 — Practical Equivalence (Preferred)
- [ ] Add TOST‑equivalence tests for slopes and intercepts:
  - [ ] SlopesEqual_* (equivalence): test |β1_g−β1| ≤ δ_s for all groups using TOST with robust SE; SlopesEqual_* = TRUE only if both NHST (no diff) OR equivalence passes.
  - [ ] InterceptsEqual_* (equivalence): test |b0_g−b0| ≤ δ_b at a reference x₀; InterceptsEqual_* = TRUE only if both NHST (no diff) OR equivalence passes.
- [ ] Reference point for intercepts: use x₀ = 0.5 (interior) rather than 0 to avoid boundary artifacts; compute model‑based predictions at x₀.
- [ ] Report both NHST p‑values and equivalence p‑values; keep the same Boolean columns but base them on “practical invariance” logic.

## Phase 3 — Robust Inference Options
- [ ] Cluster‑robust covariance (by (K,dataset_sizes)) and wild‑cluster bootstrap for interaction terms to stabilize inference under few clusters and dependence across iters.
- [ ] Option `--cov-type {HC3,CR,BOOT}` to select inference type; default HC3, allow CR/BOOT when requested.

## Phase 4 — Flexible Linearity Check
- [ ] GAM option: y_std ~ s(x) with penalized spline; test edf≈1 (linear) vs edf>1.
- [ ] Cross‑validated linear vs spline comparison: 10‑fold CV RMSE; Linear = TRUE if linear within ε_RMSE of spline; maintain ΔR² check as quick screen.
- [ ] Option `--linearity {quad,gam,cv}` to select method (default `quad`).

## Phase 5 — Monotonicity (Diagnostic)
- [ ] Optional diagnostic: one‑sided test for β1 ≥ 0 on pooled linear model or isotonic regression goodness‑of‑fit; reported but not used in Overall.

## Phase 6 — Multiple Testing and Reporting
- [ ] FDR control within each mode over all tests (or per family: Linear, Slopes, Intercepts); publish q‑values alongside p‑values in stats JSON; final Boolean decisions remain threshold‑based with equivalence.
- [ ] Summary report: compact markdown table per mode with the Boolean matrix and key stats.

## Phase 7 — Documentation and Examples
- [ ] Add `docs/benchmarking/invariance_tests.md` with:
  - [ ] Equations, transformation map, defaults (α, ε_R2, δ_s, δ_b).
  - [ ] Interpretation of NHST vs TOST results; examples of PASS/FAIL.
  - [ ] Guidance on when to enable CR/BOOT and GAM/CV options.

## Notes and Constraints
- Each metric is analyzed independently, producing its row in the Boolean matrix; there is no sharing across metrics.
- NA handling: if a factor has fewer than 2 levels or x grid is inadequate, only the applicable tests are attempted; Overall is NA when required components are NA.
- Performance: OLS‑HC3 path is lightweight; GAM/BOOT paths are more expensive—exposed as options.

