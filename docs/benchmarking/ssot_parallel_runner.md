# Benchmark Defaults & Parallel Runner Guide

Last updated: 2025-10-25

This note summarizes how benchmark defaults are managed from a single source of truth (SSoT), how to discover them, and how to use the parallel unified runner outputs that now accompany every run.

## Single Source of Truth (SSoT)
- `benchmarks/config.py` defines all canonical benchmark defaults (dataset pairs, `smax`, iteration counts, bootstraps, etc.). Dataset sizes are partitioned into named sets (`signal`, `low_signal`, and `all`) so you can run only regimes where signal is expected or deliberately stress the low-count edge cases. Runners, plotters, and validators import from this module instead of hard-coding literals.
- CLI defaults in `benchmarks/run_all_benchmarks.py`, `run_invariance_property.py`, and `run_multi_config.py` are automatically pulled from `benchmarks.config`, so overriding on the command line still works while keeping a single default location.
- Inspect the current defaults at any time:
  ```
  python -m benchmarks.config --print
  ```
  which emits a JSON payload containing dataset-pair tuples alongside characterize/invariance parameters.

## Parallel Unified Runner
- The orchestrator `benchmarks/run_all_benchmarks.py` exposes a single switch: `--max-workers <int>`. Passing `0`/`1` (or omitting the flag) runs everything sequentially; any value `>1` enables a process pool that is shared across all simulations. Invariance scenarios are still launched one after another, but each scenario consumes the full worker pool internally, avoiding nested pools while saturating the requested cores. Use `--dataset-pair-set {signal,low_signal,all}` to swap between the high-signal and indistinguishable-noise regimes without redefining `--N1-list`/`--b1-sizes` by hand (overrides still work when explicitly provided).
- Each run creates a timestamped directory under `benchmarks/out/`. Within it you will find:
  - `characterize/`, `overlays/`, and one sub-directory per invariance mode (`symmetric/`, `expand_b2_only/`, `fixed_union_private/`) containing the raw CSV outputs.
  - `unified_results.csv` and `unified_nulls.csv`, which collate all scenarios into a consistent schema. Standardised columns include:
    - `scenario`: `"characterize"`, `"overlays"`, or `"invariance"`.
    - `mode`: invariance sub-mode when applicable.
    - `dataset_sizes`: stringified `b1_size_vs_b2_size`.
    - `K`, `x_numeric`, and harmonised metric names (e.g. `cramers_v` for the bias-corrected estimate).
  - `manifest.json` recording the command-line arguments and per-scenario seeds used for the run.
- To limit GUI requirements on CI or headless machines, set `PMD_SKIP_PLOTS=1` before invoking the runner; all analyses still execute, only plot generation is skipped.

## Workflow Tips
- Use the SSoT helpers when writing new tests or utilities:
  ```python
  from benchmarks.config import default_dataset_pairs, default_invariance
  ```
  This keeps fixtures aligned with production defaults, avoiding drift.
- When adding new scenarios, extend the structures in `benchmarks.config` and reuse the `_standardize` helper in `run_all_benchmarks.py` so that unified outputs remain schema compatible.
- The runner applies a shared BLAS/OpenMP guard (`parallel_env()` in `benchmarks.config`) so each worker stays single-threaded, preventing oversubscription. Deterministic seed allocation (base seed + per-scenario offsets and per-work-item seeds) means repeated runs with identical arguments yield identical CSVs regardless of worker count—a property covered by `tests/test_run_all_parallel.py` and `tests/test_run_simulation_grid_workers.py`.
