# Invariance Statistical Testing — Third Draft (Math‑Forward)

Owner: PMD maintainers
Date: 2025‑10‑22

This draft formalizes the hypotheses and test statistics to guide the final implementation. It complements the second draft by making explicit the modeling assumptions, test equations, and decision rules.

## 1. Data model and invariance objects
We study metric–wise relationships between
\[
 x\;=\;\mathrm{percent\_different\_clusters\_numeric}\in[0,1],\qquad
 y\;=\;m\bigl(X\bigr),
\]
where \(m(\cdot)\) is any scalar metric (PMD*, \(\chi^2\), \(-\log_{10}p\), Cramer’s \(V\) (BC), inverse Simpson, Shannon, JSD, TV, Hellinger, Bray–Curtis, Cosine, Canberra). Observations are indexed by tuples \((g_1,g_2,t)\) with
- \(g_1\in\mathcal{K}\) = NumberOfClusters levels, 
- \(g_2\in\mathcal{D}\) = dataset\_sizes levels (e.g., "250\_vs\_500"),
- \(t\) = replicate index (iter),
- \(x\in\mathcal{X}\subset[0,1]\) the design grid.

The bottom row of the invariance super‑grids visualizes \(y\) vs \(x\) with color keyed to \(g_1\) and line style to \(g_2\). Invariance requires a single linear curve across all groups.

## 2. Transformations and working scale
To stabilize variance and unify thresholds across heterogeneous metrics we define
\[
 y^*\;=\;T(y)\;=\;\begin{cases}
 \operatorname{logit}(\min(\max(y,\varepsilon),1-\varepsilon)) & y\in[0,1],\\
 \log(1+y) & y\ge 0,\\
 \text{identity} & \text{otherwise},
\end{cases}
\quad y_{\text{std}}\;=\;\frac{y^*-\bar y^*}{s(y^*)},\tag{1}
\]
with \(\varepsilon=10^{-6}\). Decisions are taken on the standardized scale; when needed we back‑map effect‑size guardrails to the natural scale via the inverse link and delta method.

## 3. Hypotheses
Let \(G_1=\mathrm{C}(g_1)\) and \(G_2=\mathrm{C}(g_2)\) be dummy‑coded factors. Our target is a single global line
\[
\mathcal{H}_{\text{inv}}:\quad y_{\text{std}}\;=\;\beta_0+\beta_1 x\quad \text{(common to all \(g_1,g_2\)).}\tag{2}
\]
This induces the following nulls (tested separately):
- Linearity: pooled relationship is linear in \(x\) (no curvature on the working scale).
- Slope invariance by \(g_1\): all slopes equal across NumberOfClusters.
- Slope invariance by \(g_2\): all slopes equal across dataset\_sizes.
- Intercept invariance by \(g_1\) and \(g_2\): all group offsets equal.

## 4. Models and tests
### 4.1 Linearity (pooled lack‑of‑fit)
Fit
\[
\text{L1:}\; y_{\text{std}}=\beta_0+\beta_1 x+\varepsilon,\qquad
\text{L2:}\; y_{\text{std}}=\beta_0+\beta_1 x+\beta_2 x^2+\varepsilon.\tag{3}
\]
Use a robust Wald test (HC3) for \(\beta_2=0\) with p‑value \(p_{\mathrm{quad}}\). Also compute \(\Delta R^2=R^2(\text{L2})-R^2(\text{L1})\). Declare linear if
\[
 p_{\mathrm{quad}} > \alpha \quad \text{and}\quad \Delta R^2 < \varepsilon_{R^2}.\tag{4}
\]
Optionally (final version), replace/augment by a GAM smooth \(y_{\text{std}}\sim s(x)\) and test \(\mathrm{edf}\approx 1\) vs \(>1\), or a 10‑fold CV comparison of linear vs cubic spline.

### 4.2 Homogeneity of slopes (ANCOVA)
Fit the interaction model
\[
 y_{\text{std}}=\beta_0+\beta_1 x+\gamma_1^\top G_1+\gamma_2^\top G_2+\delta_1^\top (x\cdot G_1)+\delta_2^\top (x\cdot G_2)+\varepsilon.\tag{5}
\]
Robust joint Wald tests (HC3) for
\[
 H_{0,\text{slope},K}:\;\delta_1=0,\qquad H_{0,\text{slope},DS}:\;\delta_2=0.\tag{6}
\]
As a practical equivalence margin, compute groupwise slopes \(\hat\beta_{1g}\) from separate \(y_{\text{std}}\sim x\) fits and require
\[
\max_g |\hat\beta_{1g}-\hat\beta_1|\le \delta_s.\tag{7}
\]
Final decision for slope invariance per factor can follow either
- NHST rule: \(p>\alpha\) and (7) holds; or
- Equivalence rule (preferred): two one‑sided tests (TOST) for each group difference \(\hat\beta_{1g}-\hat\beta_1\) within \([ -\delta_s,\,\delta_s ]\) using robust SE; accept if all pass.

### 4.3 Homogeneity of intercepts (ANCOVA without interactions)
Fit
\[
 y_{\text{std}}=\beta_0+\beta_1 x+\gamma_1^\top G_1+\gamma_2^\top G_2+\varepsilon.\tag{8}
\]
Robust joint Wald tests for \(\gamma_1=0\) and \(\gamma_2=0\). For the practical margin, compare group offsets at a reference \(x_0\) (we recommend \(x_0=0.5\)) using the fitted line:
\[
 b_{0g}(x_0)=\widehat{\mathbb{E}}[y_{\text{std}}|x=x_0, g],\quad \max_g |b_{0g}(x_0)-\bar b_0(x_0)|\le \delta_b.\tag{9}
\]
Again, prefer a TOST‑equivalence decision at \(x_0\) in the final version.

### 4.4 Robust covariance and bootstrap
All regressions use HC3 by default. The finalized implementation will add options for
- Cluster‑robust covariance clustered by \((g_1,g_2)\), and
- Wild cluster bootstrap for interaction terms, providing p‑values resilient to few clusters and within‑group dependence.

## 5. Multiple testing and reporting
Within each invariance mode, we test many metrics. We will report raw p‑values, and optionally control the false discovery rate (Benjamini–Hochberg) per family (linearity, slopes, intercepts). Boolean decisions remain margin‑based (NHST and/or TOST) with the guardrails above.

## 6. Decision matrix
For each mode separately, construct a Boolean matrix \(B\) with rows = metrics and columns:
\[
 B = \{\text{Linear},\;\text{SlopesEqual\_K},\;\text{SlopesEqual\_DS},\;\text{InterceptsEqual\_K},\;\text{InterceptsEqual\_DS},\;\text{Overall}\}.
\]
- Linear is from (4).
- SlopesEqual_* from (6)+(7) (or TOST equivalence in the finalized version).
- InterceptsEqual_* from (8)+(9) (or TOST at \(x_0\)).
- Overall is the logical AND of Linear, both slope decisions, and both intercept decisions. NA propagates if prerequisites are not met (e.g., only one \(K\) level present).

## 7. Mapping guardrails back to the natural scale
Let \(T\) be the transform and \(g=T^{-1}\) its inverse. A standardized difference \(\Delta\) at \(y_{\text{std}}\) corresponds to an approximate change on the natural scale via the delta method:
\[
 \Delta_y\;\approx\; g'(\mu^*)\; s(y^*)\; \Delta,\tag{10}
\]
where \(\mu^*\) is a reference point (e.g., median of \(y^*\)). For logit, \(g'(u)=\frac{e^u}{(1+e^u)^2}\); for log1p, \(g'(u)=e^u\). This permits reporting \(\delta_s,\delta_b\) in interpretable units when needed.

## 8. Calibration and defaults
We will calibrate \(\alpha,\varepsilon_{R^2},\delta_s,\delta_b\) by simulation, targeting detection of practically meaningful effects on the natural scales, with robustness checks across invariance modes and dataset sizes. Defaults used in the baseline implementation are \(\alpha=0.01\), \(\varepsilon_{R^2}=0.02\), \(\delta_s=\delta_b=0.10\) on the standardized scale.

## 9. Implementation notes
- Each metric is analyzed independently; outputs are aggregated into the Boolean matrix.
- In the code, we will expose switches for: covariance type (HC3/CR/BOOT), linearity method (quad/GAM/CV), and equivalence testing (on/off, \(x_0\)).
- All scripts run headlessly and save per‑mode CSV/JSON artifacts to enable CI regression checks.

