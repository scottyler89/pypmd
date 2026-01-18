# PMD Benchmarking and Simulation — First Implementation Strategy

Owner: PMD maintainers  
Context: Plan for a flexible, reproducible simulation and benchmarking suite inspired by the R script (`characterize_PMD.R`).  
Note: Use the checkboxes to track progress. `[ ]` = pending, `[x]` = done.

## Update — R Parity Policy (Oct 19, 2025)

- [x] Always use bias‑corrected Cramer's V (Bergsma 2013) everywhere the R script uses Cramer's V. We do not maintain a fallback to the uncorrected statistic in any plotting or reporting surface.
- [x] Do not clip p‑values (or their transformations) that Python can compute. Specifically, we no longer floor p at machine eps for `-log10(p)` displays; if numerical underflow would yield `p=0`, we allow `-log10(p)=inf` to surface rather than silently clipping. Future work may switch to using distribution log‑survival functions to avoid underflow without clipping.

Open follow‑ups for this update:
- [x] Ensure all plotting code requires `cramers_v_bc` and clearly errors (or annotates the panel) if absent.
- [x] Replace any residual `clip(lower=...)` usage on p‑values or `-log10(p)` in plotting, CSV writers, and report helpers.
- [x] Add a tiny test CSV exercising extremely small p‑values to confirm no clipping and that `-log10(p)` can be `inf` without breaking plotting.
- [x] Compute `neglog10_p` from `logsf` to avoid underflow while preserving exactness (no clipping).

## R Parity — Additional Figures (New)

Goal: include the standalone figures that the original R script emitted in addition to the super‑grids, without duplicating panels already present in super‑grids.

- [x] Spec canonical figure manifest and filenames (no panel clones). Document and freeze the contract.
- [x] Invariance: hold X‑axis constant in each row (top: `NumberOfClusters`, bottom: `% different`) even when some datasets have NaNs at endpoints.
- [x] Invariance legends: add two distinct legends per super‑grid (top row: colors=%Different, lines=dataset_sizes; comparator figure mirrors this behavior as well).
- [x] Characterize: ensure X‑axis bounds 0..s_max across all s‑based panels (main + comparators + standalones); verify overlays.
- [x] Add PMD‑only focus figure: raw PMD vs s and PMD* vs s (per‑dataset LOWESS), saved as `pmd_focus.png`.
- [x] Add lambda figures:
  - [x] `lambda_density.png`: kernel density of `pmd_lambda`, colored by dataset.
  - [x] `lambda_by_predictors.png`: two panels — λ vs total clusters, λ vs log2(min(E)), with per‑dataset LOWESS.
- [x] Overlays: also emit lambda_density.png and lambda_by_predictors.png in overlays directory (colored by dataset_sizes) to compare multiple dataset sizes in one place.
- [x] Comparator invariance: add `invariance_comparators_super_grid.png` (2×6: JSD, TV, Hellinger, Bray–Curtis, Cosine, Canberra) with the same axis-locking and legend layout.
- [x] Optional violin of λ by K or s for richer inspection (CLI flag `--violin-by` in `plot_lambda.py`; wireable via `run_all_benchmarks.py --lambda-violin-by`).
- [x] Optional ridgeline of λ by K or s (CLI flag `--ridgeline-by` in `plot_lambda.py`; wireable via `run_all_benchmarks.py --lambda-ridgeline-by`).
- [x] Wire PMD & lambda standalones into `run_all_benchmarks.py`; remove default per‑panel clones.
- [x] Add small smoke tests for the new scripts and axis‑lock assertions.
- [x] Style pass to unify palettes/linetypes across all figures; doc the scheme (tab10 palette per dataset; consistent linestyle cycle per dataset_sizes; consistent color mapping for fitted lines).

## Principles — Math Parity With R

- [x] Default math matches the original R implementation:
  - [x] Null debiasing via permutation of counts within columns (preserve column totals); λ estimated as the empirical mean of null PMD.
  - [x] Entropy fairness via downsampling without replacement (multivariate hypergeometric), with the single‑row duplication special case.
- [x] Cramer’s V uses the bias‑correction (Bergsma 2013), identical to rcompanion::cramerV(bias.correct=TRUE). This is the only variant exposed in plots/reports.
  - [x] Non‑shared clusters simulated via non‑wrapping label shift (union grows to K + s) for parity; additional modalities are offered but not default.

## Phase 1 — Design & Scaffolding

- [x] Spec: Define a minimal, stable simulation API (config schema, inputs → outputs).
- [x] Config object (dataclass): `K`, `N1`, `N2`, `s_grid`, `iters`, `random_state`, `composition_model`, `sampling_model`, `null_model`, `metrics`, `executor`.
- [x] Composition models:
  - [x] Uniform over `K` clusters for each batch.
  - [x] Dirichlet(α) with scalar/vector α.
  - [x] Effect operators: (a) overlap shift with `s` non‑shared clusters; (b) targeted fold‑change on selected rows; (c) multi‑row effect vector.
- [x] Sampling models:
  - [x] Multinomial with fixed depth per column.
  - [x] Poisson depth, then Multinomial (or independent Poisson) for overdispersion.
  - [x] Reproducible RNG plumbing via `random_state`.
- [x] Null models (common interface):
  - [x] Permutation null (shuffle row labels within columns; preserve column totals).
  - [x] Parametric null (simulate from independence model using `E = r c^T / N`).
  - [ ] Optional parametric fit to null PMD (e.g., Poisson) to estimate λ; default to empirical mean.
- [x] Metrics (orthogonal calculators):
  - [x] PMD (raw) and PMD* (debiased with λ).
  - [x] χ² statistic and p‑value (test of independence).
  - [x] Cramer’s V.
  - [x] Inverse Simpson index per column; aggregate across columns.
  - [x] Shannon entropy of row distribution per batch with optional downsampling fairness.
  - [x] `min(E)` and other summary covariates (used in R analysis).
  - [x] Jensen–Shannon divergence/distance (JSD) between column compositions using `scipy.spatial.distance.jensenshannon` (set `base=2` for bits; SciPy returns the distance = √JSD). Pairwise for B>2, scalar when B=2.  
        References: Wikipedia JSD (https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence), SciPy implementation (https://raw.githubusercontent.com/scipy/scipy/refs/tags/v1.16.2/scipy/spatial/distance.py).
  - [x] Total Variation (TV) distance on probability vectors (0.5 · L1).
  - [x] Hellinger distance.
  - [x] Bray–Curtis dissimilarity (equals TV on probability‑normalized inputs; still include for ecology familiarity).
  - [x] Canberra distance (sensitive to rare categories; careful zero handling).
  - [x] Cosine distance (shape‑based; scale‑invariant after normalization).

- [x] Numerical stability policy (pseudocounts and NA):
  - [x] Default: do not add pseudocounts; operate directly in count or probability space.
  - [x] Allow optional global +1 pseudocount mode (log1p‑style) as an explicit user choice for metrics requiring strictly positive support (e.g., KL if enabled); document when it’s theoretically justified in count space.
  - [x] Where a metric is undefined under the given regime (e.g., KL with zeros and pseudocounts disabled), report NA instead of failing or silently altering data.
  - [x] Normalize to probabilities with safe handling of zero‑sum vectors; if a column has zero total, set all metrics that require probabilities to NA and continue.
- [ ] Runner:
  - [x] Grid iterator over `(K, N1, N2, s, iter)` with seeded RNG per run.
  - [x] Executor abstraction: sequential, multiprocessing, and Ray with graceful fallback.
- [x] Results schema:
  - [x] `results_df`: one row per run with config + metrics + λ.
  - [x] `null_df`: long format of null PMD draws with linkage to run id.
  - [x] Include audit trail: seed, wall time, package versions (optional).

## Phase 2 — Minimal Viable Implementation (MVP)

- [x] Implement composition generators (uniform, Dirichlet) and effect operators (overlap `s`, fold‑change on chosen rows).
- [x] Implement sampling models (Multinomial; Poisson‑Multinomial; independent Poisson).
- [x] Implement null models (permutation; parametric); return λ and optional null draws.
- [x] Implement metric calculators (PMD raw/debiased; χ²; V; diversity; entropy; JSD/TV/Hellinger/Bray‑Curtis/Canberra/Cosine).
- [x] Implement the runner with a tiny smoke battery (few runs) and CSV output (benchmarks/run_characterize_pmd.py).
- [x] Example: Reproduce the “non‑shared clusters” sweep (`s = 0..K`) for a small grid of `(K, N1, N2)` via CLI options.
- [x] Add expand_union option to match R’s non‑wrapping shift (union size = K + s) and record `total_clusters` and `observed_clusters` per run.
 - [x] Entropy fairness: implement downsampling without replacement (multivariate hypergeometric) and the single‑row duplication special case to mirror `get_avg_entropy`; keep current with‑replacement mode as a fast fallback.
 - [x] Cramer's V (bias‑corrected): add an option to compute bias‑corrected V (to match `rcompanion::cramerV(bias.correct=TRUE)`), document default vs corrected behavior.
 - [x] Multi‑config orchestration: accept multiple `(N1, N2)` pairs in a single run and emit aggregated overlays with `dataset_sizes` labels (e.g., `250_vs_500`), mirroring `do_full_pmd_characterization`.
 - [x] Invariance property driver: add simulator to sweep `NumberClusters` and `percent_difference` under two modes (`expand_b2_only` and symmetric), emitting R‑matching outputs (raw PMD, PMD*, χ², −log10 p, Cramer's V, inverse Simpson, Shannon entropy).
 - [x] Fixed‑union private modality: add dedicated runner `benchmarks/run_fixed_union_private.py` to vary overlap while keeping union fixed (batch1 in one private/overlap cluster; batch2 mixes overlap and private mass).
 - [x] Entropy method selection: CLIs accept `--entropy-method` (hypergeo/bootstrap) to control downsampling.

## Phase 3 — Benchmark Packaging & Extras

- [x] Introduce `[benchmark]` extra in `pyproject.toml`:
  - [x] Candidates: `tqdm` (progress), `seaborn` (plots), `scikit-bio` (optional), `polars` (optional).
  - [x] Keep core install unchanged; extras are opt‑in.
  - [x] Note: JSD uses SciPy (`scipy.spatial.distance.jensenshannon`), already a core dependency.
- [x] Add `benchmarks/` scripts (no library coupling):
  - [x] `run_characterize_pmd.py` to replicate R‑style sweeps and emit CSVs.
  - [x] `summarize_results.py` for quick aggregations.
- [x] Output layout: `benchmarks/out/YYYYMMDD_tag/{results.csv,null.csv,config.json}`.
- [x] Plotting script `benchmarks/plot_characterize_pmd.py` to recreate the 10‑panel characterization figure (loess lines, null density at s=0, λ vs total clusters, λ vs log2(min(E))).
- [x] One-shot wrapper: `benchmarks/run_full_benchmark.py` orchestrates run + plot + λ fit in a single command.
- [x] Multi‑config runner CLI: `benchmarks/run_multi_config.py` accepts `--N1-list` / `--N2-list`, aggregates overlays.
- [x] Invariance property CLI: `benchmarks/run_invariance_property.py` replicates the invariance experiment and outputs invariance_results.csv and invariance.png.
- [x] All‑in‑one orchestrator: `benchmarks/run_all_benchmarks.py` runs characterize (+R‑compat, plots, λ), multi‑config overlays (+R‑compat), and invariance (+R‑compat) in one command; optional parity comparison if an R CSV is provided.
 - [ ] Example notebooks (Jupyter) that orchestrate characterize, overlays, invariance, parity visuals.

## Phase 4 — Validation & Parity Checks

- [ ] Invariants and properties:
  - [x] 0 ≤ PMD ≤ 1; PMD* ≈ 0 under null (s=0).
  - [x] PMD* increases monotonically with `s` (on average).
  - [x] χ² and Cramer’s V track with effect magnitude (sanity checks + plots).
  - [x] JSD distance ∈ [0,1], symmetric; increases with `s` (on average) when columns are normalized to probability vectors.
  - [x] TV, Hellinger, Bray–Curtis, Canberra, Cosine: verify bounds/symmetry and sensible monotone trends with `s` (on average); confirm Bray–Curtis == TV on normalized inputs in tests.
 - [x] Pseudocount policy: ensure metrics that are undefined without pseudocounts return NA (not silently adjusted), and that enabling global +1 resolves only those metrics that mathematically require it.  
       Added tests in `tests/test_pseudocount_policy.py`.
 - [x] Small‑n parity spot checks vs. R outputs (qualitative trends; identical seeds where feasible).  
      Implemented `benchmarks/check_small_n_parity.py` with RMSE/MAE thresholds.
 - [x] Document any expected differences (e.g., RNG, smoothing, λ fit) in `docs/benchmarking/diffs_vs_R.md`.
- [x] Parity plots: scripted comparison (our CSV vs. R CSV) for key curves (raw PMD, χ², V, PMD*, λ vs log2(min(E))) with tolerance bands.  
      Implemented `benchmarks/compare_with_r_csv.py` (outputs summary_diffs.csv + overlay plots).
- [x] Verify expand_union increases union size and that `total_clusters` tracks K + s (subject to sampling);
    ensure null density panel uses s=0 null draws.

## Phase 5 — Performance & Stability

- [x] Chunked iteration to bound memory; progress bars (`tqdm`).
- [x] Optional Ray executor for large grids; graceful shutdown/fallback.
- [x] Controls for `num_boot` vs. runtime; warn on tiny `E_{ij}`.  
      Added `warn_eps` handling and outputs (`tiny_E_count`, `tiny_E_frac`) in results rows; CLI `--warn-eps`.
- [x] JSD computation: vectorized helper `pairwise_js_distance` for (K,B) probabilities; O(B²K) via broadcasting.

## Phase 6 — Reporting

- [x] Minimal plotting utility: `plot_characterize_pmd.py` reproduces R plot layout.
- [x] Plotting UX improvements: accepts a run directory via `--dir` or auto-detects newest run with `--latest-tag TAG`.
- [x] Comparator figure: additional panel grid for JSD/TV/Hellinger/Bray–Curtis/Cosine vs `s` saved as `characterize_comparators.png`.
- [x] Optional per-dataset multi-page PDF export via `--pdf`.
 - [x] Programmatic report script `benchmarks/generate_report.py` creates report.md summarizing outputs (no notebooks required).
  - [x] `benchmarks/summarize_results.py` prints aggregates from results.csv.
 - [x] R‑style merged grid page: `plot_characterize_pmd.py --super-grid true` builds a single large figure combining key panels (main + comparators).
 - [x] Plot polish for parity: remove any floors/clips on `−log10(p)`; allow `inf` when applicable. Use bias‑corrected Cramer's V only (no fallback).
- [x] Add a super‑grid for invariance property panels analogous to characterization super‑grid (2×6: top row vs #Clusters, bottom row vs %Different with per‑K polynomial smoothing). Implemented in `benchmarks/plot_invariance.py`; auto‑invoked by `benchmarks/run_invariance_property.py`.
- [x] Overlays: 2×6 characterization super‑grid covering all datasets in a multi‑config run via `benchmarks/run_multi_config.py` (passes `--super-grid true` to `plot_characterize_pmd.py`).

## Phase 7 — CI & Quality

- [x] Unit tests for calculators (basic bounds and monotonicity trends) under `tests/`.
- [x] Determinism tests with fixed seeds.
- [ ] Lint (ruff) and formatting (black) pre‑commit hooks (optional for repo).
- [x] CI: small smoke matrix on PR via GitHub Actions; runs tests on Python 3.9 and 3.11.

### Validation scripts (pre‑tests stopgap)

- [x] `benchmarks/validate_invariants.py` — bounds and sanity checks.
- [x] `benchmarks/validate_monotonicity.py` — trend checks vs `s`.

---

## Mathematical Notes (Reference)

Let `X ∈ ℕ^{K×B}` (rows = clusters/features, columns = batches). Totals: `N = Σ_{i,j} X_{ij}`, row sums `r_i = Σ_j X_{ij}`, column sums `c_j = Σ_i X_{ij}`.

Expected counts under independence: `E_{ij} = (r_i / N) · c_j`.

Raw PMD: `Δ_obs = Σ_{i,j} |X_{ij} − E_{ij}|`. Construct `O_max` by placing each column total on its diagonal cell (zeros elsewhere) to represent maximal separation; compute `E_max` from `O_max` and `Δ_max = Σ_{i,j} |O_max,ij − E_max,ij|`. Raw `PMD = Δ_obs / Δ_max ∈ [0,1]`.

Debiasing: Generate null PMD values `{PMD^{(b)}}` by (a) permutation within columns (preserve `c_j`), or (b) parametric draws from independence; set `λ = mean_b PMD^{(b)}` and `PMD* = (PMD − λ) / (1 − λ)` (centered near 0 under null, preserves upper bound).

χ² test of independence: `χ² = Σ (X_{ij} − E_{ij})² / E_{ij}` with df `(K−1)(B−1)`; p‑value from χ² distribution.

Cramer’s V: `V = sqrt( χ² / (N · min(K−1, B−1)) )`.

Inverse Simpson (per column j): `D_j^{-1} = 1 / Σ_i p_{ij}²`, where `p_{ij} = X_{ij} / c_j`; aggregate via mean across `j`.

Shannon entropy fairness: Optionally downsample each column to common depth `d = min_j c_j` prior to per‑row entropy of `(X_{i•} / Σ X_{i•})` across batches.

Overlap operator (R‑style): For `s ∈ {0,…,K}`, shift the label set in batch 2 by `s` (mod `K`) to create `K − s` overlapping clusters; sample with uniform `P` (or Dirichlet) to create controlled non‑shared mass.

Jensen–Shannon divergence (JSD): For discrete distributions `P, Q ∈ Δ^K`, let `M = (P+Q)/2`.  
JSD (divergence) = `½ KL(P || M) + ½ KL(Q || M)` (finite and symmetric).  
Jensen–Shannon distance = `√JSD`. With `log` base 2, distance ∈ [0, 1]. In practice we normalize columns to probability vectors `p_j`, then compute JSD between `p_1` and `p_2` (or all pairs for B>2). Use SciPy’s `scipy.spatial.distance.jensenshannon(p, q, base=2)` which returns the distance.  
References: Wikipedia (https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence), SciPy source (https://raw.githubusercontent.com/scipy/scipy/refs/tags/v1.16.2/scipy/spatial/distance.py).

---

## Dependencies & Extras (Planned)

- Core (already in repo): `numpy`, `pandas`, `numba`, `scipy`, `statsmodels`, `scikit-learn`, `h5py`, `ray`, `matplotlib`.
- `[benchmark]` extra (to be added here later):
  - `tqdm` (progress bars)
  - `seaborn` (visualization)
  - `scikit-bio` (optional diversity implementations)
  - `polars` (optional fast CSV/aggregation)

---

## Acceptance Criteria

- [x] MVP runner produces tidy `results.csv` and `null.csv` for a small grid within minutes on a laptop.
- [x] PMD* ≈ 0 at `s=0`, increases with `s` on average; χ² and V correlate with effect.
- [ ] JSD distance computed and reported; behaves sensibly with `s`.
- [x] All components seeded and reproducible; parameters logged per run.
- [x] Benchmarks are opt‑in via `[benchmark]` extra; core library unaffected.
