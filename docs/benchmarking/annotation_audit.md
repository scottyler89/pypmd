# Annotation Audit — Benchmark Outputs

Date: 2025-10-26
Owner: PMD maintainers

## Scope
- Characterize runs (`run_characterize_pmd.py`, `run_full_benchmark.py` produced `results.csv`, `null.csv`, `config.json`).
- Multi-config overlays (`run_multi_config.py`).
- Invariance runs (`run_invariance_property.py`).
- Unified collation (`run_all_benchmarks.py`).
- Validator outputs (`validate_invariance_*`).
- Plotting utilities that interpret the CSVs (e.g., `plot_lambda.py`, `plot_invariance.py`).

## Producers & Required Columns

| Producer | Outputs | Expected annotations |
|----------|---------|-----------------------|
| run_characterize_pmd.py | results.csv, null.csv, config.json | K, N1, N2, s, iter, num_boot, composition, effect, sampling_model, null_model, dataset_sizes, n_workers |
| run_multi_config.py | results.csv, null.csv, config.json | K, N1, N2, s, iter, num_boot, sampling_model, null_model, dataset_sizes, dataset_pair_set, n_workers |
| run_invariance_property.py | invariance_results.csv | b1_size, b2_size, NumberOfClusters/K, percent_different_clusters_numeric, iter, mode, num_boot, entropy_method, n_workers |
| validate_invariance_linear.py | invariance_tests_stats.json, invariance_tests_boolean_matrix.csv | mode, metric name, thresholds used |
| validate_invariance_onehot.py | invariance_tests_stats.json, model matrices | mode, dataset_sizes, K, percent_different_clusters_numeric |
| run_all_benchmarks.py | unified_results.csv, unified_nulls.csv | scenario, mode, dataset_sizes, K, x_numeric, n_workers, dataset_pair_set |

## Gaps Identified

1. **run_characterize_pmd.py**
   - ✅ Dataset annotation added (Phase 1) — no outstanding gaps.

2. **run_multi_config.py**
   - ❌ Results `results.csv` lacks explicit `dataset_pair_set` column (only implicit via dataset_sizes). Needed for downstream comparisons.
   - ⚠ Null CSV includes dataset_sizes but not simulation parameters (`s`, K) — acceptable? ( Already present via columns )

3. **run_invariance_property.py**
   - ✅ Added `K` column. Still missing `num_boot` column in results.
   - Results lack explicit `seed` per row (not critical, but config captures base seed).

4. **Validators**
   - `validate_invariance_linear.py` boolean matrix lacks metadata about thresholds (alpha, tau_min, etc.) — only stats JSON holds it. Consider embedding summary columns?
   - Model matrices written by `validate_invariance_onehot.py` include dataset_sizes and K but no percent_different_clusters_numeric? (Check: currently included as column — yes). No action.

5. **Unified collation**
   - `unified_results.csv` includes scenario and mode. After per-scenario additions, it now inherits dataset_pair_set via manifest? No column—should we include `dataset_pair_set` (from manifest args) in the unified CSV? Currently not.
   - `unified_nulls.csv` same issue.

6. **Plotting utilities**
   - `plot_invariance.py` expects NumberOfClusters, percent_different_clusters_numeric, dataset_sizes, mode — all present.
   - `plot_lambda.py` now requires `K` (which we added) but overlays results may still lack `mode` (they don't have mode — acceptable). Need to guard for missing `mode` gracefully (currently handled by default to None).

## Action Items

1. Add `dataset_pair_set` column to outputs from `run_multi_config.py` and propagate to unified results.
2. Include `num_boot` (and optionally `entropy_method`, `n_workers`) columns in `invariance_results.csv` so plots can faceting.
3. Update unified collation to append `dataset_pair_set` and `n_workers` columns when present.
4. Consider adding threshold metadata to validator boolean matrices (optional; depends on usage).

## Remediation Plan

1. **Multi-config outputs** – Extend `run_multi_config.py` writers so both `results.csv` and `null.csv` contain `dataset_pair_set` and `n_workers`; adjust `config.json` accordingly. Update `run_all_benchmarks.py` collation to carry these columns forward. Add a regression check in `tests/test_run_all_parallel.py` for the unified CSV.
2. **Invariance annotations** – In `run_invariance_property.py`, append `num_boot`, `entropy_method`, and `n_workers` columns to `invariance_results.csv`. Add a small fixture-based test to confirm their presence.
3. **Unified collation** – After (1) and (2), ensure `_standardize` preserves `dataset_pair_set`/`n_workers` when concatenating scenario outputs. Expand the existing smoke test to assert these fields exist in `unified_results.csv` and `unified_nulls.csv`.
4. **Validator metadata (optional)** – If richer provenance proves useful, update `validate_invariance_linear.py` to write alpha/threshold parameters alongside the boolean matrix (e.g., extra columns or companion JSON); document the schema prior to implementation.
