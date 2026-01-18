# Second Implementation Strategy — SSoT + Parallel Unified Runner

Owner: PMD maintainers  
Date: 2025-10-25

This plan replaces scattered defaults with a Single Source of Truth (SSoT) and adds a parallel, unified benchmark runner that collates all scenario results into one table. Items start pending [ ], and will be checked [x] as they are completed.

## Phase 1 — Single Source Of Truth (SSoT)
- [x] Add `benchmarks/config.py` with canonical defaults:
  - [x] `DATASET_PAIR_SETS = {"signal": [...], "low_signal": [...], "all": [...]}` splitting high-signal and low-signal regimes
  - [x] Characterize: `CHAR_S_VALUES`/`CHAR_SMAX`, `CHAR_ITERS`, `CHAR_NUM_BOOT`
  - [x] Invariance: `INV_K_LIST`, `INV_PDIFFS`, `INV_ITERS`, `INV_NUM_BOOT`
  - [x] Accessors: `default_dataset_pairs()`, `default_dataset_pairs_str()`, `default_invariance()`
- [x] Refactor consumers to import SSoT defaults:
  - [x] `benchmarks/run_multi_config.py`
  - [x] `benchmarks/run_characterize_pmd.py`
  - [x] `benchmarks/run_invariance_property.py`
  - [x] `benchmarks/run_all_benchmarks.py`
  - [x] Validators/tests that encode defaults in fixtures
- [x] Keep CLI override flags; set default values from SSoT helpers
- [x] Add `tests/test_config_ssot.py` asserting CLI-exposed defaults match `benchmarks.config`

## Phase 2 — Parallel Orchestration (no code bloat)
- [x] Extend `benchmarks/run_all_benchmarks.py` with:
  - [x] `--parallel true|false` (default false) and `--max-workers <int>`
  - [x] Scenario list derived from SSoT (initial focus on invariance runs)
  - [x] Submit scenarios via `concurrent.futures.ProcessPoolExecutor`
  - [x] Deterministic per-scenario seeds (base_seed + offset)
  - [x] Worker env guards: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=GNU`
  - [x] Per-scenario subdirs under `benchmarks/out/<ts>_all/…`
  - [x] Write `manifest.json` listing scenarios/args/seeds (in collation step)

## Phase 3 — Unified Results Collation
- [x] After futures complete, collate per-scenario outputs into:
  - [x] `<ts>_all/unified_results.csv>`
  - [x] `<ts>_all/unified_nulls.csv` (if nulls exist)
- [x] Normalize schema with standard columns:
  - [x] Add `scenario`, `mode` (for invariance)
  - [x] Prefer `cramers_v_bc` as `cramers_v`
  - [x] Map invariance `percent_different_clusters_numeric -> x_numeric`, `NumberOfClusters -> K`
- [x] Keep a small rename map in `benchmarks/config.py` to avoid stringly-typed code

## Phase 4 — Tests & Reproducibility
- [x] Add `tests/test_run_all_parallel.py` (tiny run: `num_boot=10`, `iters=1`, 1–2 dataset pairs):
  - [x] Assert scenario subdirs exist and contain `results.csv`
  - [x] Assert `unified_results.csv` exists and includes multiple `scenario` tags
- [x] Repro check: fixed seed yields identical `unified_results.csv` shape and summary stats

## Phase 5 — Documentation & Developer Hygiene
- [x] Document SSoT policy and defaults location in `docs/benchmarking/`
- [x] Document `--parallel` usage and unified output schema
- [x] Provide `python -m benchmarks.config --print` (optional) to list current defaults
- [x] Update examples/README to reference SSoT (no literals)

## Non-Goals / Constraints
- [x] Do not introduce new top-level runners; extend `run_all_benchmarks.py`
- [x] Avoid class hierarchies for scenarios; keep a simple list of dicts
- [x] Preserve sequential path; `--parallel` is opt-in
