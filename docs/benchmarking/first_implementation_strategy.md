# PMD Benchmarking and Simulation — First Implementation Strategy

Owner: PMD maintainers  
Context: Plan for a flexible, reproducible simulation and benchmarking suite inspired by the R script (`characterize_PMD.R`).  
Note: Use the checkboxes to track progress. `[ ]` = pending, `[x]` = done.

## Phase 1 — Design & Scaffolding

- [ ] Spec: Define a minimal, stable simulation API (config schema, inputs → outputs).
- [ ] Config object (dataclass): `K`, `N1`, `N2`, `s_grid`, `iters`, `random_state`, `composition_model`, `sampling_model`, `null_model`, `metrics`, `executor`.
- [ ] Composition models:
  - [ ] Uniform over `K` clusters for each batch.
  - [ ] Dirichlet(α) with scalar/vector α.
  - [ ] Effect operators: (a) overlap shift with `s` non‑shared clusters; (b) targeted fold‑change on selected rows; (c) multi‑row effect vector.
- [ ] Sampling models:
  - [ ] Multinomial with fixed depth per column.
  - [ ] Poisson depth, then Multinomial (or independent Poisson) for overdispersion.
  - [ ] Reproducible RNG plumbing via `random_state`.
- [ ] Null models (common interface):
  - [ ] Permutation null (shuffle row labels within columns; preserve column totals).
  - [ ] Parametric null (simulate from independence model using `E = r c^T / N`).
  - [ ] Optional parametric fit to null PMD (e.g., Poisson) to estimate λ; default to empirical mean.
- [ ] Metrics (orthogonal calculators):
  - [ ] PMD (raw) and PMD* (debiased with λ).
  - [ ] χ² statistic and p‑value (test of independence).
  - [ ] Cramer’s V.
  - [ ] Inverse Simpson index per column; aggregate across columns.
  - [ ] Shannon entropy of row distribution per batch with optional downsampling fairness.
  - [ ] `min(E)` and other summary covariates (used in R analysis).
  - [ ] Jensen–Shannon divergence/distance (JSD) between column compositions using `scipy.spatial.distance.jensenshannon` (set `base=2` for bits; SciPy returns the distance = √JSD). Pairwise for B>2, scalar when B=2.  
        References: Wikipedia JSD (https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence), SciPy implementation (https://raw.githubusercontent.com/scipy/scipy/refs/tags/v1.16.2/scipy/spatial/distance.py).
  - [ ] Total Variation (TV) distance on probability vectors (0.5 · L1).
  - [ ] Hellinger distance.
  - [ ] Bray–Curtis dissimilarity (equals TV on probability‑normalized inputs; still include for ecology familiarity).
  - [ ] Canberra distance (sensitive to rare categories; careful zero handling).
  - [ ] Cosine distance (shape‑based; scale‑invariant after normalization).

- [ ] Numerical stability policy (pseudocounts and NA):
  - [ ] Default: do not add pseudocounts; operate directly in count or probability space.
  - [ ] Allow optional global +1 pseudocount mode (log1p‑style) as an explicit user choice for metrics requiring strictly positive support (e.g., KL if enabled); document when it’s theoretically justified in count space.
  - [ ] Where a metric is undefined under the given regime (e.g., KL with zeros and pseudocounts disabled), report NA instead of failing or silently altering data.
  - [ ] Normalize to probabilities with safe handling of zero‑sum vectors; if a column has zero total, set all metrics that require probabilities to NA and continue.
- [ ] Runner:
  - [ ] Grid iterator over `(K, N1, N2, s, iter)` with seeded RNG per run.
  - [ ] Executor abstraction: sequential, multiprocessing, Ray (optional).
- [ ] Results schema:
  - [ ] `results_df`: one row per run with config + metrics + λ.
  - [ ] `null_df`: long format of null PMD draws with linkage to run id.
  - [ ] Include audit trail: seed, wall time, package versions (optional).

## Phase 2 — Minimal Viable Implementation (MVP)

- [ ] Implement composition generators (uniform, Dirichlet) and effect operators (overlap `s`, fold‑change on chosen rows).
- [ ] Implement sampling models (Multinomial; Poisson‑Multinomial).
- [ ] Implement null models (permutation; parametric); return λ and optional SD.
- [ ] Implement metric calculators (PMD raw/debiased; χ²; V; diversity; entropy).
- [ ] Implement the runner with a tiny smoke battery (few runs) and CSV output.
- [ ] Example: Reproduce the “non‑shared clusters” sweep (`s = 0..K`) for a small grid of `(K, N1, N2)`.

## Phase 3 — Benchmark Packaging & Extras

- [ ] Introduce `[benchmark]` extra in `pyproject.toml` (planned here only):
  - [ ] Candidates: `tqdm` (progress), `seaborn` (plots), `scikit-bio` (diversity metrics, if used), `polars` (optional speed for post‑processing).
  - [ ] Keep core install unchanged; extras are opt‑in.
  - [ ] Note: JSD uses SciPy (`scipy.spatial.distance.jensenshannon`), already a core dependency.
- [ ] Add `benchmarks/` scripts (no library coupling):
  - [ ] `run_characterize_pmd.py` to replicate R‑style sweeps and emit CSVs.
  - [ ] `summarize_results.py` for quick aggregations and sanity plots (optional).
- [ ] Output layout: `benchmarks/out/YYYYMMDD_tag/{results.csv,null.csv,config.json}`.

## Phase 4 — Validation & Parity Checks

- [ ] Invariants and properties:
  - [ ] 0 ≤ PMD ≤ 1; PMD* ≈ 0 under null (s=0).
  - [ ] PMD* increases monotonically with `s` (on average).
  - [ ] χ² and Cramer’s V track with effect magnitude.
  - [ ] JSD distance ∈ [0,1], symmetric; increases with `s` (on average) when columns are normalized to probability vectors.
  - [ ] TV, Hellinger, Bray–Curtis, Canberra, Cosine: verify bounds/symmetry and sensible monotone trends with `s` (on average); confirm Bray–Curtis == TV on normalized inputs in tests.
  - [ ] Pseudocount policy: ensure metrics that are undefined without pseudocounts return NA (not silently adjusted), and that enabling global +1 resolves only those metrics that mathematically require it.
- [ ] Small‑n parity spot checks vs. R outputs (qualitative trends; identical seeds where feasible).
- [ ] Document any expected differences (e.g., RNG, parametric null choice).

## Phase 5 — Performance & Stability

- [ ] Chunked iteration to bound memory; progress bars (`tqdm`).
- [ ] Optional Ray executor for large grids; graceful shutdown.
- [ ] Controls for `num_boot` vs. runtime; warn on tiny `E_{ij}`.
- [ ] Optional λ approximation as a function of `min(E)` to reduce bootstraps (fit after collecting data; cache table).
 - [ ] JSD computation: vectorize probability normalization and pairwise distances; expect O(B²K) per run when B>2.

## Phase 6 — Reporting

- [ ] Minimal plotting utilities (kept outside library): PMD vs. `s`, χ² vs. `s`, effect of `K` and depth imbalance.
- [ ] Example notebooks or scripts (optional) that read CSVs and generate plots.

## Phase 7 — CI & Quality

- [ ] Unit tests for calculators (PMD bounds, debiasing under null, χ²/V, diversity, entropy fairness).
- [ ] Determinism tests with fixed seeds.
- [ ] Lint (ruff) and formatting (black) pre‑commit hooks (optional for repo).
- [ ] CI: small smoke matrix on PR; scheduled heavier benchmark (nightly) optional.

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

- [ ] MVP runner produces tidy `results.csv` and `null.csv` for a small grid within minutes on a laptop.
- [ ] PMD* ≈ 0 at `s=0`, increases with `s` on average; χ² and V correlate with effect.
- [ ] JSD distance computed and reported; behaves sensibly with `s`.
- [ ] All components seeded and reproducible; parameters logged per run.
- [ ] Benchmarks are opt‑in via `[benchmark]` extra; core library unaffected.
