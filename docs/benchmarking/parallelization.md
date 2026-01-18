# Parallelization Plan — Single Switch, SSoT, No Nested Pools

Owner: PMD maintainers
Date: 2025-10-26

This phased plan consolidates all concurrency behavior behind a single user‑facing flag, centralizes defaults in SSoT, avoids nested pools, and keeps the CLI surface small.

## Phase A — SSoT Parallel Policy
- [x] Add `DEFAULT_MAX_WORKERS = 0` to `benchmarks/config.py` (0/1 → sequential, >1 → pooled).
- [x] Add `resolve_max_workers(n)` helper to clamp to available CPUs and normalize 0/None/−1.
- [x] Add `parallel_env()` helper that returns BLAS/MKL/OpenBLAS single‑thread caps.
- [x] Document the parallel policy in `benchmarks/config.py` docstring.

## Phase B — Core Engine Switch (percent_max_diff/benchmarking.py)
- [x] Extend `SimulationConfig` with `n_workers: int = 1` (kept picklable).
- [x] In `run_simulation_grid`, if `n_workers > 1` use `ProcessPoolExecutor(max_workers=n_workers)`, else run sequentially.
- [x] Treat legacy string `executor` as deprecated; map to `n_workers` internally (keep parsing for back‑compat).
- [x] Preserve per‑work‑item RNG seeding so results are reproducible across worker counts.
- [x] Add a tiny unit test that the `n_workers=1` vs `n_workers=2` outputs match for a small grid.

## Phase 2 — Annotation Audit
- [x] Catalogue current CSV producers/consumers and capture gaps in `docs/benchmarking/annotation_audit.md`.
- [ ] Address outstanding annotation gaps listed in the audit (tracked in Phase 3 plan).

## Phase C — Runners Accept One Flag
- [x] Add `--max-workers` to:
  - [x] `benchmarks/run_characterize_pmd.py`
  - [x] `benchmarks/run_multi_config.py`
  - [x] `benchmarks/run_invariance_property.py`
  - [x] `benchmarks/run_full_benchmark.py`
- [x] Forward `n_workers=resolve_max_workers(--max-workers)` into `SimulationConfig`.
- [x] Deprecate `--executor`: still accepted, emit a warning if provided and ignore it.
- [x] Apply `parallel_env()` when a runner is invoked directly (so direct calls remain fast and consistent).

## Phase D — Orchestrator Simplification (`run_all_benchmarks.py`)
- [x] Add `--max-workers` (single concurrency switch for the whole suite).
- [x] Keep `--parallel` for back‑compat but warn that it is ignored; remove from docs.
- [x] Pass `--max-workers` through to all subcommands.
- [x] Run invariance scenarios sequentially; each scenario uses the full worker pool internally (no nested pools).
- [x] Record `max_workers` in `manifest.json`.

## Phase E — Tests & Docs
- [x] Update `tests/test_run_all_parallel.py` to use `--max-workers 2` and drop `--parallel`.
- [x] Add a unit test for `run_simulation_grid` with `n_workers=2` (small grid) verifying equality to `n_workers=1`.
- [x] Update `docs/benchmarking/ssot_parallel_runner.md` and `README.md` to document the single switch and show example commands.
- [x] Mention environment honors (`parallel_env()`) and determinism guarantees.

## Phase F — Optional (Later)
- [ ] Add hidden `--distribute-scenarios` flag to split `N` across invariance modes (default off), ensuring no oversubscription.
- [ ] Capture simple throughput benchmarks vs. `N` and add a short “Performance notes” section.

## Acceptance Criteria
- [ ] A single user flag (`--max-workers`) controls parallelism across all entry points.
- [ ] No nested process pools; CPU utilization scales to `N` cores during simulations.
- [ ] Outputs are bit‑identical for `N=1` vs `N>1` on the same seed (for fixed BLAS settings).
- [ ] Back‑compat flags accepted with deprecation warnings; docs and examples reference only the new flag.

## Non‑Goals
- [ ] Do not introduce thread‑pool execution inside numerical kernels.
- [ ] Do not change RNG or null‑generation algorithms beyond ensuring determinism across worker counts.
- [ ] Do not expand the CLI surface beyond the single switch and minimal deprecation handling.
