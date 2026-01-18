# Invariance Statistical Testing — Implementation Plan (Phased TODOs)

Owner: PMD maintainers
Date: 2025‑10‑22

This plan tracks code we still need to write to realize the statistical testing framework (Boolean matrix per metric, per mode). All items below are pending unless explicitly checked.

Policy update (2025‑10‑24): All tests operate on RAW metric scales with no transformations. Any references to transformed scales in older phases are superseded and retained only for historical context.

## Phase 1 — Baseline Validator (OLS + HC3)
- [x] Add script `benchmarks/validate_invariance_linear.py` with CLI:
  - [x] `--file <invariance_results.csv>`
  - [x] `--mode {all|symmetric|expand_b2_only|fixed_union_private}`
  - [x] `--alpha 0.01 --delta-s 0.10 --delta-b 0.10 --epsilon-r2 0.02`
  - [x] `--out-dir <path>`
- [x] Implement data prep:
  - [x] Filter by mode (if provided); drop rows with non‑finite (x, y, K, dataset_sizes).
  - [x] Verify x has ≥ 2 unique values; verify K and dataset_sizes each have ≥ 2 levels; otherwise mark relevant tests NA.
  - [x] Iterate metrics present in columns (core + comparator set), treating each metric independently.
- [x] Implement per‑metric transforms: (superseded by policy update)
  - [x] Identity on RAW scale (no transform). Prior transform plan removed.
- [x] Implement tests (per metric):
  - [x] Linearity: fit L1 (y~1+x), L2 (y~1+x+x²) with HC3; compute `p_quad`, ΔR²; set `Linear` boolean.
  - [x] SlopesEqual_K/DS (heuristic and joint tests):
    - [x] Heuristic: equivalence to pooled Theil–Sen slope with adaptive tolerance on RAW scale (diagnostic only).
    - [x] Joint ANCOVA: y ~ 1 + x_c + C(K) + C(DS) + x_c:C(K) + x_c:C(DS) [HC3]; robust Wald for x_c:C(K) (p_xK) and x_c:C(DS) (p_xDS).
    - [x] Add parallel p-value booleans to output matrix: `SlopesEqual_K_p`, `SlopesEqual_dataset_sizes_p` (keep heuristic booleans).
  - [x] InterceptsEqual_K/DS (joint tests):
    - [x] Joint ANCOVA above; robust Wald for C(K) (p_int_K) and C(DS) (p_int_DS) at centered x_c (x0=0.5 by default).
  - [x] Overall = Linear & SlopesEqual_K & SlopesEqual_DS, with intercepts incorporated per `intercepts_mode` using joint-model p‑values.
- [x] Outputs per mode:
  - [x] `<mode>/invariance_tests_boolean_matrix.csv` (Boolean matrix)
  - [x] `<mode>/invariance_tests_stats.json` (p‑values, ΔR², max diffs, sample sizes)
- [x] Logging and non‑zero exit on any Overall=FALSE (optional flag `--fail-on-false`).

## Phase 2 — Practical Equivalence (TOST)
- [ ] Add TOST for slopes (per group) with robust SE; accept if |β1_g − β1| ≤ δ_s for all groups.
- [ ] Add TOST for intercepts at reference x₀ (default 0.5); accept if |b0_g(x₀) − b0(x₀)| ≤ δ_b for all groups.
- [ ] Expose `--equivalence true|false` and `--x0 0.5` options; when `true`, booleans reflect TOST (NHST reported as supplemental stats).

## Phase 3 — Robust Inference Options
- [ ] Add covariance options via `--cov-type {HC3,CR,BOOT}`:
  - [ ] HC3 (default): existing path.
  - [ ] CR: cluster‑robust by (K, dataset_sizes).
  - [ ] BOOT: wild cluster bootstrap for interaction terms; compute p‑values for x:G1, x:G2 and main‑effects.

## Phase 4 — Flexible Linearity Methods
- [ ] Add `--linearity {quad,gam,cv}`:
  - [ ] `quad`: current quadratic term (default).
  - [ ] `gam`: fit `y_std ~ s(x)`; flag non‑linearity if edf>1 at α.
  - [ ] `cv`: 10‑fold CV RMSE comparison linear vs spline; `Linear` if linear within ε_RMSE of spline; add `--epsilon-rmse`.

## Phase 5 — Monotonicity Diagnostic (Optional)
- [ ] Add `--diagnostics-monotone true|false` to run one‑sided β1 ≥ 0 test or isotonic GOF; record in stats JSON (not used for Overall).

## Phase 6 — Multiple Testing and Reporting
- [ ] Compute FDR (BH) q‑values per family (Linearity, Slopes, Intercepts) within each mode; include in stats JSON.
- [ ] Add summary markdown per mode with the Boolean matrix, key p/q values, ΔR², max diffs.

## Phase 7 — Integration Hooks
- [x] Update `benchmarks/run_all_benchmarks.py`:
  - [x] Add flag `--validate-invariance true|false`.
  - [x] If `true`, call the validator for each invariance mode and collect outputs under the mode directories.

## Phase 8 — Documentation and Examples
- [ ] Add `docs/benchmarking/invariance_tests.md` with:
  - [ ] Equations and rationale (link to third_draft.md and methods_math.md).
  - [ ] Default thresholds and guidance for δ_s, δ_b, ε_R².
  - [ ] Example PASS/FAIL outputs with small CSVs.

## Phase 9 — Calibration and CI
- [ ] Simulate calibration sets to tune δ_s, δ_b, ε_R² on natural scales; document results.
- [ ] Add CI job to run the validator on a tiny invariance CSV and assert all outputs exist (matrix + stats), and that Overall is boolean/NA.
## Phase 1.5 — Options and Reporting (new)
- [x] Expose `--center-x0` (default 0.5) and record in stats JSON.
- [x] Expose `--include-ds-k-interaction` (default false) for optional C(K):C(DS) and x_c:C(K):C(DS) terms.
- [x] Record ANCOVA formula, parameter names, and n of group means.
