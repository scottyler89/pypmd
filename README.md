# Percent Maximum Difference (PMD)

### What is this repository for? ###

* Percent Maximum Difference (PMD) is a relative distance metric for count data that can linearly quantify how similar/different your observations are based on the composition of their features. It has an upper bound of 1 when completely different, and is centered (not bounded) around 0. You can also subtract it from 1 to get reverse PMD (rPMD), which can be more intuitive in some cases because it behaves like a similarity.

### How do I get set up? ###

This repository is pip installable:
`python3 -m pip install percent_max_diff`

You can also clone the repository and install using the setup.py script in the distribution like so:
`python3 setup.py install`
or
`python3 -m pip install .`


### How do I use it? ###

```
import numpy as np
from percent_max_diff import pmd  # or: from pypmd import pmd

pmd_res = pmd(np.array([[100,0,0],[0,100,0],[0,0,100]]), num_boot=200)
print(pmd_res.pmd)
```

### Tutorial: Column comparisons and standardized residuals

This section shows how to:
- compute overall PMD for a count matrix,
- get pairwise PMD between columns (samples/conditions), and
- interpret cell‑wise standardized residuals (`z_scores`).

Setup (NumPy or pandas works; labels are preserved for DataFrames):

```
import numpy as np
import pandas as pd

# Toy counts: rows = features, cols = samples
X = pd.DataFrame(
    [[30, 5,  5],
     [10, 25, 5],
     [ 0, 10, 35]],
    index=["featA", "featB", "featC"],
    columns=["sample1", "sample2", "sample3"],
)

from percent_max_diff import pmd  # or: from pypmd import pmd
pmd_res = pmd(X, num_boot=500)  # increase num_boot for more stable null
print("Overall PMD:", pmd_res.pmd)
```

Pairwise PMD between columns

```
# Post-hoc pairwise PMD (computed by default unless skip_posthoc=True)
pairwise = pmd_res.post_hoc  # pandas.DataFrame indexed by columns
print(pairwise)

# Access a specific pair
print("PMD(sample1, sample2) =", pairwise.loc["sample1", "sample2"])
```

- Values are in [0, 1] after debiasing; 0 ≈ similar composition, 1 ≈ maximally different given marginals.
- For large matrices, set `skip_posthoc=True` during construction and compute pairs later with advanced helpers.

Standardized residuals (z‑scores)

`pmd` also returns cell‑wise residual diagnostics that show which features drive differences.

```
Z = pmd_res.z_scores            # standardized residuals (DataFrame)
P = pmd_res.p_vals              # per-cell p-values
Q = pmd_res.corrected_p_vals    # FDR-corrected p-values

# Example: top features (by magnitude) for each sample
topk = (
    Z.abs()
      .stack()
      .groupby(level=1)
      .nlargest(3)
      .reset_index(level=0, drop=True)
)
print(topk)

# Example: features significantly enriched/depleted in each sample
signif = (Q < 0.05)
print({col: list(signif.index[signif[col]]) for col in signif.columns})
```

Notes
- Bootstrapping introduces randomness; results can vary slightly run‑to‑run.
- For quick exploration use `num_boot=200–500`; for publication‑grade, increase as needed.
- Import either alias: `from percent_max_diff import pmd` or `from pypmd import pmd`.

### Benchmarking defaults & runner

Benchmark configuration and orchestration now follow a single source of truth. Inspect the canonical defaults any time with:

```
python -m benchmarks.config --print
```

To execute the end-to-end benchmark suite (characterize, overlays, invariance) use `benchmarks/run_all_benchmarks.py`. Control concurrency with a single flag: `--max-workers <N>` (0/1 or unset keeps it sequential; >1 enables a shared process pool with BLAS guards). Choose between the high-signal and low-signal regimes with `--dataset-pair-set {signal,low_signal,all}`; explicit `--N1-list/--b1-sizes` overrides remain honoured. The pipeline deterministically reseeds every scenario, so identical arguments yield identical outputs regardless of worker count. Full details live in `docs/benchmarking/ssot_parallel_runner.md`, including the unified CSV schema and directory layout under `benchmarks/out/`.

### Example: Differential abundance via GLM on z‑scores

You can test for differential abundance of features across groups using the cell‑wise standardized residuals (`pmd_res.z_scores`). Below we simulate 10 features across 6 samples (3 untreated, 3 treated) where one feature is enriched in the treated group, then fit a simple GLM with Gaussian family per feature.

```
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection
from pypmd import pmd   # or: from percent_max_diff import pmd

# --- Simulate counts: 10 features x 6 samples (3 control, 3 treated)
rng = np.random.default_rng(42)
n_feat = 10
n_per_group = 3
depths = rng.poisson(2000, size=2*n_per_group)  # library sizes
base = rng.dirichlet(np.ones(n_feat))           # baseline composition

diff_feat = 6   # 0-based index; the 7th feature is truly enriched in treated
treated_shift = 3.0
p_ctrl = base.copy()
p_trt = base.copy(); p_trt[diff_feat] *= treated_shift; p_trt /= p_trt.sum()

counts = []
cols = []
for i in range(n_per_group):
    cols.append(f"ctrl_{i+1}")
    counts.append(rng.multinomial(depths[i], p_ctrl))
for i in range(n_per_group):
    cols.append(f"trt_{i+1}")
    counts.append(rng.multinomial(depths[n_per_group+i], p_trt))

X = pd.DataFrame(
    np.vstack(counts).T,
    index=[f"feat{i+1}" for i in range(n_feat)],
    columns=cols,
)

# --- Compute PMD and extract z-scores (skip pairwise to speed up)
res = pmd(X, num_boot=300, skip_posthoc=True)
Z = res.z_scores  # rows=features, cols=samples

# --- Build design matrix: intercept + group indicator (0=control, 1=treated)
group = pd.Series([0]*n_per_group + [1]*n_per_group, index=Z.columns, name="treated")
Xdesign = sm.add_constant(group.values)

# --- Fit per-feature GLM (Gaussian) on z-scores
rows = []
for feat in Z.index:
    y = Z.loc[feat].values
    fit = sm.GLM(y, Xdesign, family=sm.families.Gaussian()).fit()
    rows.append((feat, float(fit.params[1]), float(fit.pvalues[1])))

glm_df = pd.DataFrame(rows, columns=["feature", "coef_treated", "pvalue"]).set_index("feature")
glm_df["qvalue"] = fdrcorrection(glm_df["pvalue"].values)[1]

print(glm_df.sort_values("pvalue"))
print("Ground-truth enriched feature:", f"feat{diff_feat+1}")
```

Interpretation
- Positive `coef_treated` indicates higher z‑scores (enrichment) in treated; negative indicates depletion.
- Use `qvalue` (FDR) to account for testing many features.
- This simple GLM treats per‑feature z‑scores as approximately Gaussian; for complex designs consider robust regressions or hierarchical models.

### License ###
This project is licensed under the GNU Affero General Public License v3.0 (AGPLv3). Commercial licensing inquiries: scottyler89@gmail.com

### Who do I talk to? ###

* Repo owner/admin: scottyler89+bitbucket@gmail.com
