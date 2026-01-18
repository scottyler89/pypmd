# Methods — Mathematical Formulation

This section describes the mathematical definitions, simulation designs, and statistical testing procedures used in our benchmarking and invariance analyses. Unless otherwise noted, all symbols refer to a two–column contingency table \(X\in\mathbb{N}^{K\times 2}\) with rows (clusters) indexed by \(i\in\{1,\dots,K\}\) and columns (batches) indexed by \(j\in\{1,2\}\).

## 1. Notation and basic quantities
Let
- \(X_{ij}\ge 0\) denote observed counts, 
- row sums \(r_i=\sum_j X_{ij}\),
- column sums \(c_j=\sum_i X_{ij}\),
- total \(N=\sum_{i,j} X_{ij}=\sum_i r_i=\sum_j c_j\).

The independence model has expected counts
\[
E_{ij} = \frac{r_i\,c_j}{N}.\tag{1}
\]

Where probabilities are needed, we use column–wise compositions
\[
 p_{ij}=\frac{X_{ij}}{c_j}, \quad j\in\{1,2\},\tag{2}
\]
with the convention that a column with \(c_j=0\) yields undefined (NA) metric values unless otherwise stated. For probability–based metrics we optionally apply a global +1 pseudo–count when requested (not used by default).

## 2. Percent Maximum Difference (PMD)
Let \(\Delta(\cdot,\cdot)=\sum_{i,j}\lvert A_{ij}-B_{ij}\rvert\) denote elementwise \(\ell_1\) distance over matrices. Define
- observed discrepancy \(\Delta_{\text{obs}}=\Delta(X,E)\),
- a maximally separated table \(O_{\max}\) obtained by placing each column total on a distinct row (diagonal) and zeros elsewhere, then its independence expectation \(E_{\max}\) by (1), and
- \(\Delta_{\max}=\Delta(O_{\max},E_{\max})\).

The raw PMD is
\[
\mathrm{PMD}_{\text{raw}}\;=\; \frac{\Delta_{\text{obs}}}{\Delta_{\max}}\;\in[0,1].\tag{3}
\]

### 2.1 Debiasing via a null distribution
We estimate a small-sample bias parameter \(\lambda\in[0,1]\) as the mean of PMD draws under a null that preserves the column totals. We consider two nulls:
- Permutation null: randomly permute row labels within each column (preserves \(c_j\)).
- Parametric independence null: simulate columns as multinomials with probabilities proportional to row sums \(r_i/N\) and totals \(c_j\).

With \(\{\mathrm{PMD}^{(b)}\}_{b=1}^B\) null draws and \(\hat\lambda=\frac1B\sum_b \mathrm{PMD}^{(b)}\), the debiased PMD is
\[
\mathrm{PMD}^*\;=\;\frac{\mathrm{PMD}_{\text{raw}}-\hat\lambda}{1-\hat\lambda}.\tag{4}
\]

## 3. Classical statistics and association
- Chi–square statistic: \(\chi^2=\sum_{i,j}\frac{(X_{ij}-E_{ij})^2}{E_{ij}}\) with \(\text{df}=(K-1)(2-1)=K-1\). P–values use the \(\chi^2\) distribution; for numerical stability we compute \(-\log_{10}p=-\frac{\log \Pr(\chi^2_{K-1}\ge\chi^2)}{\log 10}\) via the log–survival function.
- Cramer’s \(V\): \(\phi^2=\chi^2/N\); bias–corrected \(V\) (Bergsma 2013):
\[
\phi^2_{\text{corr}} = \max\Big\{0,\;\phi^2-\frac{(r-1)(k-1)}{N-1}\Big\},\quad r=K,\;k=2,\tag{5}
\]
\[
 r_{\text{corr}}=r-\frac{(r-1)^2}{N-1},\quad k_{\text{corr}}=k-\frac{(k-1)^2}{N-1},\quad
 V=\sqrt{\frac{\phi^2_{\text{corr}}}{\min(r_{\text{corr}}-1,\,k_{\text{corr}}-1)}}.\tag{6}
\]

## 4. Diversity and entropy
- Inverse Simpson (per column \(j\)): \(D_j^{-1}=1/\sum_i p_{ij}^2\); we report the mean across columns.
- Shannon entropy fairness: to avoid depth bias, we optionally downsample each column without replacement to a common depth \(d=\min_j c_j\) (multivariate hypergeometric), then compute Shannon entropy across columns for each row and average.

## 5. Probability and compositional distances
All distances are computed on the probability columns \(p_{\cdot 1}, p_{\cdot 2}\).
- Jensen–Shannon distance (base 2): \(\mathrm{JSD}(p,q)=\sqrt{\tfrac12\mathrm{KL}(p\,\|\,m)+\tfrac12\mathrm{KL}(q\,\|\,m)}\) with \(m=\tfrac12(p+q)\), bounded in \([0,1]\).
- Total variation: \(\mathrm{TV}(p,q)=\tfrac12\sum_i |p_i-q_i|\).
- Hellinger: \(H(p,q)=\frac1{\sqrt{2}}\lVert\sqrt{p}-\sqrt{q}\rVert_2\).
- Bray–Curtis: \(\mathrm{BC}(p,q)=\sum_i |p_i-q_i|/\sum_i (p_i+q_i)\) (equals TV for probability vectors).
- Cosine distance: \(1-\frac{p\cdot q}{\lVert p\rVert_2\lVert q\rVert_2}\).
- Canberra: \(\sum_i \frac{|p_i-q_i|}{p_i+q_i}\) (terms with zero denominators omitted).

## 6. Simulation designs for invariance experiments
We generate paired columns under controlled differences in cluster overlap. Let \(x\in[0,1]\) denote the fraction of clusters that are different between batches (plotted as `percent_different_clusters_numeric`). We study three mechanisms:

1) Symmetric overlap shift (non‑wrapping): base composition \(u=\mathrm{Unif}(K)\). Column 1 uses \(u\); column 2 shifts \(u\) by \(s=\lfloor Kx\rfloor\) to new indices in an expanded union, preserving mass.
2) Expand\_b2\_only: column 1 allocates all mass to a single “overlap” cluster; column 2 allocates \((1-x)\) mass to that cluster and spreads \(x\) uniformly across \(K\) private clusters.
3) Fixed union, private (variant): column 1 uses only the overlap cluster; column 2 allocates \((1-x)\) to overlap and \(x\) uniformly over the remaining \(K-1\) clusters.

For each design we sample counts using Multinomial draws with user–specified column totals \((N_1,N_2)\). We benchmark across dataset sizes \((N_1,N_2)\in\{(10{,}000,10{,}000), (10{,}000,1{,}000), (1{,}000,1{,}000), (250,2{,}000), (250,250), (100,100)\}\) to mirror the original R experiments.

## 7. Plot construction and invariance super‑grids
We assemble two 2×6 super‑grids per mode:
- Core metrics: top row vs \(K\): PMD*, \(\chi^2\), \(-\log_{10}p\), Cramer’s \(V\) (BC), inverse Simpson, Shannon; bottom row vs \(x\) with per‑\(K\) polynomial smoothing and dataset‑dependent line styles.
- Comparator metrics: top row vs \(K\): JSD, TV, Hellinger, Bray–Curtis, Cosine, Canberra; bottom row vs \(x\) with the same smoothing and styles.

Within each row, x‑axis limits are locked to the global range to avoid visual compression when some groups have undefined endpoints.

## 8. Statistical testing of invariance (bottom rows)
We test, independently for each metric, whether the relationship \(y\) vs \(x\) is (i) linear, (ii) has equal slopes across \(K\) and dataset sizes, and (iii) has equal intercepts across \(K\) and dataset sizes.

### 8.1 Transformations
To stabilize variance and unify thresholds across metrics, we work on a transformed and standardized response \(y_{\text{std}}\):
- If \(y\in[0,1]\): \(y^*=\mathrm{logit}(\min(\max(y,\varepsilon),1-\varepsilon))\) with \(\varepsilon=10^{-6}\).
- If \(y\ge 0\): \(y^*=\log(1+y)\).
- Standardize: \(y_{\text{std}}=(y^*-\bar y^*)/s(y^*)\).

### 8.2 Linearity
Fit two pooled models with heteroskedasticity–robust (HC3) covariance:
\[
\text{L1: } y_{\text{std}}\sim 1 + x,\qquad
\text{L2: } y_{\text{std}}\sim 1 + x + x^2.\tag{7}
\]
Declare linearity if the quadratic term is non‑significant (robust Wald \(p>\alpha\)) and the incremental fit is negligible \(\Delta R^2=R^2(\text{L2})-R^2(\text{L1})<\varepsilon_{R^2}\).

### 8.3 Equality of slopes (ANCOVA)
Let \(G_1=\mathrm{C}(K)\) and \(G_2=\mathrm{C}(\text{dataset\_sizes})\). Fit
\[
 y_{\text{std}}\sim 1 + x + G_1 + G_2 + x:G_1 + x:G_2.\tag{8}
\]
Test jointly (robust Wald) that all \(x:G_1\) and all \(x:G_2\) coefficients are zero. As a practical guardrail, compute groupwise slopes \(\beta_{1g}\) from \(y_{\text{std}}\sim x\) fits and require \(\max_g |\beta_{1g}-\bar\beta_1|\le \delta_s\) (standardized units).

### 8.4 Equality of intercepts
Fit the reduced model (no interactions):
\[
 y_{\text{std}}\sim 1 + x + G_1 + G_2.\tag{9}
\]
Test jointly that all \(G_1\) terms and all \(G_2\) terms are zero. As a guardrail, compare group intercepts at \(x=0\) (or an interior \(x_0\), see below) and require \(\max_g |b_{0g}-\bar b_0|\le \delta_b\).

### 8.5 Practical equivalence (optional but recommended)
In addition to NHST, we may apply two one‑sided tests (TOST) for equivalence:
- Slopes: test \(|\beta_{1g}-\bar\beta_1|\le \delta_s\) for all groups with robust SE.
- Intercepts: test \(|b_{0g}-\bar b_0|\le \delta_b\) at a reference \(x_0\) (we recommend \(x_0=0.5\) to avoid boundary artifacts).

### 8.6 Decisions and reporting
For each metric and mode we record booleans for Linearity, SlopesEqual\_K, SlopesEqual\_dataset\_sizes, InterceptsEqual\_K, InterceptsEqual\_dataset\_sizes, and Overall (logical AND). We also store supporting statistics (p‑values, \(\Delta R^2\), max differences, sample sizes) and optionally FDR–adjusted q‑values.

### 8.7 Defaults
Unless otherwise specified, we use \(\alpha=0.01\), \(\varepsilon_{R^2}=0.02\), \(\delta_s=0.10\), and \(\delta_b=0.10\) on the standardized scale. Covariance is HC3–robust by default; cluster–robust or bootstrap options can be enabled when group dependence is a concern.

## 9. Reproducibility and implementation
All simulations and metrics are seeded for reproducibility. PMD and null debiasing are computed via our package implementation; distances and statistics use standard scientific Python libraries. Plots are rendered headlessly (Agg backend) with consistent palettes and line styles across figures.

