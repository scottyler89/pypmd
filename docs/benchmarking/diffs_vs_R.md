# Differences vs R Characterization Scripts

This note summarizes minor differences between the original R characterization code and the current Python benchmarking implementation.

- Randomness and null estimation
  - R fits a Poisson to the null PMD to estimate λ, which equals the empirical mean for PMD on [0,1]. Our default λ is the empirical mean; both approaches agree numerically.
  - Bootstrapping and sampling use NumPy/Numba in Python vs base R random generators; numeric trajectories will differ slightly run-to-run even under equal seeds.

- Entropy fairness
  - R downsamples batches to equal depth without replacement then averages per-row entropy across batches; special case duplicates a single row.
  - Python mirrors this exactly with a multivariate hypergeometric downsampler and the single-row duplication rule (default). A bootstrap (with-replacement) mode remains available as a fast fallback.

- Cramer’s V
  - R uses rcompanion::cramerV(bias.correct=TRUE). Python exposes both standard V and the bias-corrected version (default used in plots). Slight differences may appear if sample sizes are extremely small.

- Non-shared clusters and union size
  - R expands the union size to K + s with non-wrapping shifts; Python supports this via `expand_union=True` (default in characterization CLIs).

- Plot smoothing and layout
  - R uses loess; Python uses statsmodels LOWESS where available (frac tuned per panel) and seaborn for line/point aesthetics. Curves should match qualitatively; minor smoothing differences are expected.
  - The Python plotter provides a 10-panel figure, a comparator figure, an optional merged/super-grid, and per-dataset PDF pages; titles/legends differ slightly from the ggplot originals.

- Pseudocounts
  - By default we do not add pseudocounts; metrics undefined under zeros return NaN. A global +1 mode (log1p-style) is optionally available where strictly positive support is required.

- File structure and CLIs
  - The Python version separates simulation, plotting, parity checks, and invariance into small CLI tools; R combines workflows into single functions. Outputs (CSV + PNG/PDF) are aligned in content.

