"""Benchmarking and simulation utilities for PMD.

This module provides a minimal, extensible foundation to simulate count
compositions, apply controlled effects (e.g., non-shared clusters), and
compute comparison metrics including PMD and common distances.

Design goals
- Keep abstractions simple (dataclasses + small pure functions).
- Avoid pseudocounts by default; return NaN when undefined. Optionally allow
  a global +1 (log1p-style) pseudocount when mathematically justified.
- Make RNG explicit and reproducible.

Note: This is an initial implementation (Phase 2 MVP starter). It currently
focuses on B=2 columns (two groups). Future versions will add multi-column
support and parallel runners per the plan in docs/benchmarking.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance as ssd

from .percent_max_diff import (
    get_pmd as _get_pmd_raw,
    get_detailed_null_vects as _get_null_vects,
)


# -----------------------------
# Configuration dataclasses
# -----------------------------


@dataclass
class SimulationConfig:
    """Configuration for a two-column (B=2) simulation grid.

    Parameters
    ----------
    K : int
        Number of features (rows/clusters).
    N1, N2 : int
        Library sizes (total counts) for the two columns.
    s_values : Iterable[int]
        Non-shared cluster counts to sweep (0..K). s=0 means full overlap.
    iters : int
        Repeats per (K, N1, N2, s) setting.
    num_boot : int
        Bootstrap iterations for PMD null debiasing.
    random_state : Optional[int]
        Seed for reproducibility.
    pseudocount_mode : str
        One of {"none", "one"}. "none" avoids any pseudocounts; "one" applies a
        global +1 to all counts for probability-based metrics that may require
        strictly positive support. Undefined metrics with mode="none" return NaN.
    compute_metrics : List[str]
        Metrics to compute. Supported: [
            "pmd_raw", "pmd", "chi2", "chi2_p", "cramers_v",
            "inv_simpson", "entropy",
            "jsd", "tv", "hellinger", "braycurtis", "canberra", "cosine"
        ]
    composition : str
        One of {"uniform", "dirichlet"}. If "dirichlet", use `dirichlet_alpha`.
    dirichlet_alpha : Union[float, Sequence[float], None]
        Concentration(s) for Dirichlet base composition when `composition=="dirichlet"`.
    effect : str
        One of {"overlap_shift", "fold_change"}.
    effect_indices : Optional[Sequence[int]]
        Feature indices affected for fold-change effect (0-based). If None, picks one.
    effect_fold : float
        Fold-change applied to affected features in the second column when
        `effect=="fold_change"`.
    sampling_model : str
        One of {"multinomial", "poisson_multinomial", "poisson_independent"}.
        - multinomial: fixed depths N1, N2.
        - poisson_multinomial: draw depths D1~Pois(N1), D2~Pois(N2), then multinomial.
        - poisson_independent: draw counts independently Pois(Nj * p_j).
    null_model : str
        One of {"permutation", "parametric"} for debiasing.
    store_null_draws : bool
        If True, collect and return null PMD draws per run.
    n_workers : int
        Number of worker processes for parallel simulation. 1 runs sequentially.
    executor : str
        Deprecated. One of {"sequential", "multiprocessing", "ray"} for backward
        compatibility. Use `n_workers` instead.
    show_progress : bool
        If True and `tqdm` is installed, display a progress bar during grids.
    expand_union : bool
        If True (and effect == 'overlap_shift'), simulate a non-wrapping label
        shift so the union of clusters is K + s (as in the R script), by
        padding the composition for column 2 into new indices. When False,
        use cyclic shift with union size K.
    """

    K: int
    N1: int
    N2: int
    s_values: Iterable[int] = field(default_factory=lambda: (0,))
    iters: int = 1
    num_boot: int = 300
    random_state: Optional[int] = None
    pseudocount_mode: str = "none"  # "none" | "one"
    compute_metrics: List[str] = field(
        default_factory=lambda: [
            "pmd_raw",
            "pmd",
            "chi2",
            "chi2_p",
            "cramers_v",
            "cramers_v_bc",
            "inv_simpson",
            "entropy",
            "jsd",
            "tv",
            "hellinger",
            "braycurtis",
            "canberra",
            "cosine",
        ]
    )
    composition: str = "uniform"
    dirichlet_alpha: Union[float, Sequence[float], None] = None
    effect: str = "overlap_shift"
    effect_indices: Optional[Sequence[int]] = None
    effect_fold: float = 2.0
    sampling_model: str = "multinomial"
    null_model: str = "permutation"
    store_null_draws: bool = False
    n_workers: int = 1
    executor: str = "sequential"
    show_progress: bool = False
    expand_union: bool = False
    warn_eps: float = 1e-9
    entropy_method: str = "hypergeo"  # "hypergeo" | "bootstrap"


# -----------------------------
# Composition generators & effects
# -----------------------------


def uniform_probs(K: int) -> np.ndarray:
    """Uniform composition over K features.

    Returns
    -------
    p : ndarray of shape (K,)
        Uniform probability vector.
    """

    if K <= 0:
        raise ValueError("K must be positive")
    return np.full(K, 1.0 / K, dtype=float)


def dirichlet_probs(K: int, alpha: Union[float, Sequence[float]], rng: np.random.Generator) -> np.ndarray:
    """Dirichlet composition over K features.

    alpha can be a scalar or a length-K vector.
    """
    if np.isscalar(alpha):
        a = np.full(K, float(alpha), dtype=float)
    else:
        a = np.asarray(alpha, dtype=float)
        if a.shape[0] != K:
            raise ValueError("dirichlet_alpha length must equal K")
    return rng.dirichlet(a)


def apply_overlap_shift(p: np.ndarray, s: int, *, K: Optional[int] = None) -> np.ndarray:
    """Create a second composition by cyclically shifting indices by s.

    Mimics the R script's construction of `K - s` overlapping clusters.

    Parameters
    ----------
    p : ndarray
        Base composition (length K).
    s : int
        Number of non-shared clusters (shift). s=0 yields identical p.
    K : Optional[int]
        Number of features; inferred from len(p) if None.
    """

    if K is None:
        K = int(p.shape[0])
    s = int(s) % K
    if s == 0:
        return p.copy()
    return np.roll(p, s)


def apply_fold_change(
    p: np.ndarray,
    indices: Sequence[int],
    fold: float,
) -> np.ndarray:
    """Apply a multiplicative fold-change to selected indices and renormalize.

    Parameters
    ----------
    p : ndarray
        Base composition vector.
    indices : sequence of int
        Indices to scale.
    fold : float
        Multiplicative factor (>0). Values >1 enrich, <1 deplete.
    """
    if fold <= 0:
        raise ValueError("fold must be > 0")
    out = p.astype(float).copy()
    out = np.maximum(out, 0.0)
    idx = np.asarray(indices, dtype=int)
    out[idx] *= float(fold)
    total = out.sum()
    if total <= 0:
        raise ValueError("fold-change produced zero-sum composition")
    return out / total


# -----------------------------
# Sampling models
# -----------------------------


def sample_counts_multinomial(N: int, p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw counts from a Multinomial with total N and composition p."""

    if N < 0:
        raise ValueError("N must be non-negative")
    if N == 0:
        return np.zeros_like(p, dtype=int)
    return rng.multinomial(int(N), p).astype(np.int64)


def _sample_counts_for_column(
    N_or_mu: int,
    p: np.ndarray,
    rng: np.random.Generator,
    *,
    model: str,
) -> np.ndarray:
    """Sample a single column according to the chosen sampling model."""
    if model == "multinomial":
        return sample_counts_multinomial(int(N_or_mu), p, rng)
    elif model == "poisson_multinomial":
        depth = int(rng.poisson(max(0.0, float(N_or_mu))))
        return sample_counts_multinomial(depth, p, rng)
    elif model == "poisson_independent":
        mu_vec = float(N_or_mu) * p
        return rng.poisson(mu_vec).astype(np.int64)
    else:
        raise ValueError(f"unknown sampling model: {model}")


def simulate_two_column_counts(
    K: int,
    N1: int,
    N2: int,
    s: int,
    rng: np.random.Generator,
    *,
    base_probs: Optional[np.ndarray] = None,
    composition: str = "uniform",
    dirichlet_alpha: Union[float, Sequence[float], None] = None,
    effect: str = "overlap_shift",
    effect_indices: Optional[Sequence[int]] = None,
    effect_fold: float = 2.0,
    sampling_model: str = "multinomial",
    expand_union: bool = False,
) -> np.ndarray:
    """Simulate a K×2 count matrix with `s` non-shared clusters (overlap shift).

    Uses uniform base composition unless `base_probs` is provided.
    """

    # Base composition
    if base_probs is not None:
        p1 = np.asarray(base_probs, dtype=float)
    else:
        if composition == "uniform":
            p1 = uniform_probs(K)
        elif composition == "dirichlet":
            if dirichlet_alpha is None:
                dirichlet_alpha = 1.0
            p1 = dirichlet_probs(K, dirichlet_alpha, rng)
        else:
            raise ValueError("unknown composition model")
    if p1.shape[0] != K:
        raise ValueError("base_probs length must equal K")
    # Effect on column 2
    if effect == "overlap_shift":
        if expand_union and s > 0:
            newK = int(K + s)
            p1_pad = np.zeros(newK, dtype=float)
            p2_pad = np.zeros(newK, dtype=float)
            p1_pad[0:K] = p1
            # Non-wrapping shift: place p1 starting at index s for column 2
            p2_pad[s : s + K] = p1
            p1_use, p2_use = p1_pad, p2_pad
        else:
            p2 = apply_overlap_shift(p1, s, K=K)
            p1_use, p2_use = p1, p2
    elif effect == "fold_change":
        idx = (
            [int(i) for i in effect_indices]
            if effect_indices is not None and len(effect_indices) > 0
            else [int(s % K)]  # pick one index based on s for reproducibility
        )
        p2 = apply_fold_change(p1, idx, effect_fold)
        p1_use, p2_use = p1, p2
    else:
        raise ValueError("unknown effect model")

    x1 = _sample_counts_for_column(N1, p1_use, rng, model=sampling_model)
    x2 = _sample_counts_for_column(N2, p2_use, rng, model=sampling_model)
    return np.stack([x1, x2], axis=1)


# -----------------------------
# Utilities for probabilities and entropy
# -----------------------------


def _normalize_counts_to_probs(
    counts: np.ndarray, mode: str = "none"
) -> Tuple[np.ndarray, bool]:
    """Convert a 1D count vector to probabilities.

    Parameters
    ----------
    counts : ndarray
        Non-negative integer counts.
    mode : {"none", "one"}
        "none" does no smoothing. "one" adds a global +1 pseudocount.

    Returns
    -------
    probs : ndarray
        Probability vector (sums to 1 when sum(counts) > 0; otherwise zeros).
    ok : bool
        False when normalization is undefined (sum==0 and mode=="none").
    """

    x = np.asarray(counts, dtype=float)
    if np.any(x < 0):
        return np.zeros_like(x), False
    if mode == "one":
        x = x + 1.0
    total = x.sum()
    if total <= 0:
        return np.zeros_like(x), False
    return (x / total), True


def _downsample_column_hypergeo(col_counts: np.ndarray, d: int, rng: np.random.Generator) -> np.ndarray:
    """Downsample a single column to depth d without replacement (multivariate hypergeometric)."""
    counts = col_counts.astype(int).copy()
    out = np.zeros_like(counts)
    total = int(counts.sum())
    d_rem = int(d)
    if total <= 0 or d_rem <= 0:
        return out
    K = counts.shape[0]
    for i in range(K - 1):
        ngood = int(counts[i])
        nbad = int(total - ngood)
        if d_rem <= 0 or (ngood + nbad) <= 0:
            xi = 0
        else:
            xi = int(rng.hypergeometric(ngood, nbad, d_rem))
        out[i] = xi
        d_rem -= xi
        total -= ngood
    out[-1] = max(0, d_rem)
    return out


def shannon_entropy_across_columns(
    X: np.ndarray,
    *,
    downsample: bool = True,
    downsample_method: str = "hypergeo",
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Mean per-row Shannon entropy of the 2-vector across columns.

    If `downsample=True`, columns are downsampled to equal depth before entropy.
    Methods:
      - "hypergeo": without replacement (multivariate hypergeometric), mirrors R.
      - "bootstrap": with replacement quick approximation.
    Special case: if K == 1, duplicate the single row to avoid undefined entropy.
    """

    X = np.asarray(X, dtype=np.int64)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must be shape (K, 2)")
    K = X.shape[0]
    if K == 1:
        X = np.vstack([X, X])
        K = 2
    if downsample:
        if rng is None:
            rng = np.random.default_rng()
        d = int(min(X[:, 0].sum(), X[:, 1].sum()))
        if d == 0:
            return float("nan")
        if downsample_method == "hypergeo":
            col0 = _downsample_column_hypergeo(X[:, 0], d, rng)
            col1 = _downsample_column_hypergeo(X[:, 1], d, rng)
        elif downsample_method == "bootstrap":
            def draw_boot(col_counts: np.ndarray) -> np.ndarray:
                p, ok = _normalize_counts_to_probs(col_counts, mode="none")
                if not ok:
                    return np.zeros_like(col_counts)
                idx = rng.choice(np.arange(K), size=d, replace=True, p=p)
                out = np.zeros(K, dtype=int)
                np.add.at(out, idx, 1)
                return out

            col0 = draw_boot(X[:, 0])
            col1 = draw_boot(X[:, 1])
        else:
            raise ValueError("downsample_method must be 'hypergeo' or 'bootstrap'")
        Xds = np.stack([col0, col1], axis=1)
    else:
        Xds = X

    # For each row, compute entropy over its two counts across columns
    entropies = []
    for i in range(K):
        v = Xds[i, :].astype(float)
        s = v.sum()
        if s <= 0:
            entropies.append(np.nan)
            continue
        p = v / s
        entropies.append(stats.entropy(p, base=2))
    return float(np.nanmean(entropies))


# -----------------------------
# Metrics
# -----------------------------


def cramers_v_stat(X: np.ndarray, *, bias_correction: bool = False) -> Tuple[float, float, float]:
    """Chi-square, p-value, and (optionally bias-corrected) Cramer's V.

    Handles degenerate rows/columns by removing any with zero totals before
    calling chi2_contingency. Returns NaNs if the table is < 2x2 after filtering.

    Bias correction follows Bergsma (2013) and matches rcompanion::cramerV(bias.correct=TRUE).
    """
    X = np.asarray(X, dtype=float)
    # Drop all-zero rows/cols to avoid zero expected counts
    row_mask = X.sum(axis=1) > 0
    col_mask = X.sum(axis=0) > 0
    Xf = X[row_mask][:, col_mask]
    if Xf.size == 0 or Xf.shape[0] < 2 or Xf.shape[1] < 2:
        return float("nan"), float("nan"), float("nan")
    try:
        chi2, pval, dof, expected = stats.chi2_contingency(Xf, correction=False)
    except Exception:
        return float("nan"), float("nan"), float("nan")
    N = Xf.sum()
    if N <= 0:
        return float("nan"), float("nan"), float("nan")
    r, k = Xf.shape
    phi2 = chi2 / N
    if not bias_correction:
        v = np.sqrt(phi2 / max(1e-12, min(k - 1, r - 1)))
    else:
        phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(1.0, N - 1))
        rcorr = r - ((r - 1) ** 2) / max(1.0, N - 1)
        kcorr = k - ((k - 1) ** 2) / max(1.0, N - 1)
        denom = max(1e-12, min(kcorr - 1.0, rcorr - 1.0))
        v = np.sqrt(phi2corr / denom)
    return float(chi2), float(pval), float(v)


def inverse_simpson_per_column(X: np.ndarray) -> float:
    """Mean inverse Simpson index across columns."""

    out = []
    for j in range(X.shape[1]):
        col = X[:, j].astype(float)
        total = col.sum()
        if total <= 0:
            out.append(np.nan)
            continue
        p = col / total
        out.append(1.0 / np.sum(p * p))
    return float(np.nanmean(out))


def _pairwise_prob_vectors(X: np.ndarray, mode: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    p1, ok1 = _normalize_counts_to_probs(X[:, 0], mode=mode)
    p2, ok2 = _normalize_counts_to_probs(X[:, 1], mode=mode)
    if not (ok1 and ok2):
        return None, None
    return p1, p2


def js_distance(X: np.ndarray, mode: str = "none") -> float:
    p1, p2 = _pairwise_prob_vectors(X, mode)
    if p1 is None:
        return float("nan")
    # SciPy returns the distance (sqrt divergence)
    return float(ssd.jensenshannon(p1, p2, base=2))


def total_variation(X: np.ndarray, mode: str = "none") -> float:
    p1, p2 = _pairwise_prob_vectors(X, mode)
    if p1 is None:
        return float("nan")
    return 0.5 * float(ssd.cityblock(p1, p2))


def hellinger(X: np.ndarray, mode: str = "none") -> float:
    p1, p2 = _pairwise_prob_vectors(X, mode)
    if p1 is None:
        return float("nan")
    return float(ssd.euclidean(np.sqrt(p1), np.sqrt(p2)) / np.sqrt(2.0))


def bray_curtis(X: np.ndarray, mode: str = "none") -> float:
    # Defined on non-negative vectors; can be used on counts or probabilities
    if mode in ("none", "one"):
        # Use probabilities for consistency with other metrics
        p1, p2 = _pairwise_prob_vectors(X, mode)
        if p1 is None:
            return float("nan")
        return float(ssd.braycurtis(p1, p2))
    raise ValueError("Unsupported mode for bray_curtis")


def canberra(X: np.ndarray, mode: str = "none") -> float:
    # Canberra is defined for non-negative vectors; probabilities recommended
    p1, p2 = _pairwise_prob_vectors(X, mode)
    if p1 is None:
        return float("nan")
    return float(ssd.canberra(p1, p2))


def cosine_distance(X: np.ndarray, mode: str = "none") -> float:
    # Cosine distance on probabilities to focus on composition
    p1, p2 = _pairwise_prob_vectors(X, mode)
    if p1 is None:
        return float("nan")
    return float(ssd.cosine(p1, p2))


def _compute_null_pmd_parametric(X: np.ndarray, *, num_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Generate null PMD by simulating from independence with fixed column totals."""
    R = X.sum(axis=1).astype(float)
    N = float(R.sum())
    if N <= 0:
        return np.full(num_boot, np.nan, dtype=float)
    p_row = R / N
    C = X.sum(axis=0).astype(int)
    out = np.zeros(num_boot, dtype=float)
    for b in range(num_boot):
        col1 = rng.multinomial(int(C[0]), p_row)
        col2 = rng.multinomial(int(C[1]), p_row)
        out[b] = float(_get_pmd_raw(np.stack([col1, col2], axis=1)))
    return out


def compute_pmd_metrics(
    X: np.ndarray,
    *,
    num_boot: int,
    null_model: str = "permutation",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float, Optional[np.ndarray]]:
    """Compute raw PMD, debiased PMD using empirical λ, and λ.

    Returns
    -------
    raw : float
    debiased : float
    lam : float
    """

    if rng is None:
        rng = np.random.default_rng()
    raw = float(_get_pmd_raw(X))
    null_draws: Optional[np.ndarray]
    if null_model == "permutation":
        seed_val = int(rng.integers(0, 2**31 - 1)) if rng is not None else None
        if seed_val is not None:
            np.random.seed(seed_val)
        null_draws, _ = _get_null_vects(X, num_boot=num_boot)
    elif null_model == "parametric":
        null_draws = _compute_null_pmd_parametric(X, num_boot=num_boot, rng=rng)
    else:
        raise ValueError("unknown null_model")
    lam = float(np.nanmean(null_draws))
    debiased = (raw - lam) / (1.0 - lam) if (lam is not None and lam != 1.0) else np.nan
    return raw, debiased, lam, null_draws


# -----------------------------
# Single-run and grid runners
# -----------------------------


def run_single_simulation(
    K: int,
    N1: int,
    N2: int,
    s: int,
    *,
    num_boot: int = 300,
    pseudocount_mode: str = "none",
    rng: Optional[np.random.Generator] = None,
    metrics: Optional[List[str]] = None,
    composition: str = "uniform",
    dirichlet_alpha: Union[float, Sequence[float], None] = None,
    effect: str = "overlap_shift",
    effect_indices: Optional[Sequence[int]] = None,
    effect_fold: float = 2.0,
    sampling_model: str = "multinomial",
    null_model: str = "permutation",
    store_null_draws: bool = False,
    expand_union: bool = False,
    warn_eps: float = 1e-9,
    entropy_method: str = "hypergeo",
) -> Dict[str, Union[float, np.ndarray]]:
    """Simulate one table and compute selected metrics.

    Returns a flat dict suitable for DataFrame construction.
    """

    if rng is None:
        rng = np.random.default_rng()

    X = simulate_two_column_counts(
        K,
        N1,
        N2,
        s,
        rng,
        composition=composition,
        dirichlet_alpha=dirichlet_alpha,
        effect=effect,
        effect_indices=effect_indices,
        effect_fold=effect_fold,
        sampling_model=sampling_model,
        expand_union=expand_union,
    )
    # expected table and small expected count warnings
    if X.sum() > 0:
        R = X.sum(axis=1)[:, None].astype(float)
        C = X.sum(axis=0)[None, :].astype(float)
        E = (R * C) / float(X.sum())
        tiny_E_count = int((E < warn_eps).sum())
        tiny_E_frac = float(tiny_E_count) / float(E.size)
    else:
        tiny_E_count = 0
        tiny_E_frac = float("nan")

    out: Dict[str, float] = {
        "K": float(K),
        "N1": float(N1),
        "N2": float(N2),
        "s": float(s),
        "min_E": float((X.sum(axis=1)[:, None] * X.sum(axis=0)[None, :] / X.sum()).min()) if X.sum() > 0 else np.nan,
        "total_clusters": float(X.shape[0]),
        "observed_clusters": float(int((X.sum(axis=1) > 0).sum())),
        "tiny_E_count": float(tiny_E_count),
        "tiny_E_frac": float(tiny_E_frac),
        "warn_eps": float(warn_eps),
    }

    metrics = metrics or [
        "pmd_raw",
        "pmd",
        "chi2",
        "chi2_p",
        "cramers_v",
        "inv_simpson",
        "entropy",
        "jsd",
        "tv",
        "hellinger",
        "braycurtis",
        "canberra",
        "cosine",
    ]

    # PMD family
    if ("pmd_raw" in metrics) or ("pmd" in metrics):
        raw, debiased, lam, null_draws = compute_pmd_metrics(
            X,
            num_boot=num_boot,
            null_model=null_model,
            rng=rng,
        )
        if "pmd_raw" in metrics:
            out["pmd_raw"] = raw
            out["pmd_lambda"] = lam
        if "pmd" in metrics:
            out["pmd"] = debiased
        if store_null_draws:
            out["pmd_null_draws"] = null_draws

    # Classical stats
    if ("chi2" in metrics) or ("cramers_v" in metrics) or ("chi2_p" in metrics) or ("cramers_v_bc" in metrics):
        chi2, pval, v = cramers_v_stat(X, bias_correction=False)
        if "chi2" in metrics:
            out["chi2"] = chi2
        if "chi2_p" in metrics:
            out["chi2_p"] = pval
        if "cramers_v" in metrics:
            out["cramers_v"] = v
        if "cramers_v_bc" in metrics:
            _, _, vbc = cramers_v_stat(X, bias_correction=True)
            out["cramers_v_bc"] = vbc

    # Diversity/entropy
    if "inv_simpson" in metrics:
        out["inv_simpson"] = inverse_simpson_per_column(X)
    if "entropy" in metrics:
        out["entropy"] = shannon_entropy_across_columns(
            X, downsample=True, downsample_method=entropy_method, rng=rng
        )

    # Distances (probability-based unless noted)
    if "jsd" in metrics:
        out["jsd"] = js_distance(X, mode=pseudocount_mode)
    if "tv" in metrics:
        out["tv"] = total_variation(X, mode=pseudocount_mode)
    if "hellinger" in metrics:
        out["hellinger"] = hellinger(X, mode=pseudocount_mode)
    if "braycurtis" in metrics:
        out["braycurtis"] = bray_curtis(X, mode=pseudocount_mode)
    if "canberra" in metrics:
        out["canberra"] = canberra(X, mode=pseudocount_mode)
    if "cosine" in metrics:
        out["cosine"] = cosine_distance(X, mode=pseudocount_mode)

    return out


def _worker_single(args: Tuple[SimulationConfig, int, int, int]) -> Dict[str, Union[float, np.ndarray]]:
    cfg, s, rep, seed = args
    rep_rng = np.random.default_rng(int(seed))
    row = run_single_simulation(
        cfg.K,
        cfg.N1,
        cfg.N2,
        int(s),
        num_boot=cfg.num_boot,
        pseudocount_mode=cfg.pseudocount_mode,
        rng=rep_rng,
        metrics=cfg.compute_metrics,
        composition=cfg.composition,
        dirichlet_alpha=cfg.dirichlet_alpha,
        effect=cfg.effect,
        effect_indices=cfg.effect_indices,
        effect_fold=cfg.effect_fold,
        sampling_model=cfg.sampling_model,
        null_model=cfg.null_model,
        store_null_draws=cfg.store_null_draws,
        expand_union=cfg.expand_union,
        warn_eps=cfg.warn_eps,
        entropy_method=cfg.entropy_method,
    )
    row["iter"] = float(rep)
    row["seed"] = float(seed)
    return row


def run_simulation_grid(cfg: SimulationConfig) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Run a grid over s_values × iters and return results.

    Returns a DataFrame of results. If `cfg.store_null_draws` is True, also
    returns a second DataFrame with null draws linked by (K,N1,N2,s,iter).
    """

    import time as _time

    rng = np.random.default_rng(cfg.random_state)
    rows: List[Dict[str, Union[float, np.ndarray]]] = []
    null_rows: List[Dict[str, float]] = []

    work_items: List[Tuple[SimulationConfig, int, int, int]] = []
    for s in cfg.s_values:
        for rep in range(cfg.iters):
            seed = int(rng.integers(0, 2**63 - 1, dtype=np.uint64))
            work_items.append((cfg, int(s), int(rep), seed))

    t0 = _time.perf_counter()
    total_items = len(work_items)
    pbar = None
    if cfg.show_progress:
        try:
            from tqdm.auto import tqdm as _tqdm
            pbar = _tqdm(total=total_items, desc="PMD sims")
        except Exception:
            pbar = None

    # Determine worker policy (new n_workers with backward compatibility)
    n_workers = getattr(cfg, "n_workers", 1) or 1
    executor = getattr(cfg, "executor", "sequential") or "sequential"
    use_ray = False
    if executor != "sequential":
        warnings.warn(
            "SimulationConfig.executor is deprecated; use n_workers instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if executor == "multiprocessing" and n_workers <= 1:
            # legacy behavior: default to all CPUs
            cpu_total = os.cpu_count() or 1
            n_workers = max(1, cpu_total)
        elif executor == "ray":
            warnings.warn(
                "Ray execution is deprecated and will fall back to sequential processing.",
                DeprecationWarning,
                stacklevel=2,
            )
            use_ray = False
            n_workers = 1
        elif executor not in ("sequential", "multiprocessing", "ray"):
            raise ValueError(f"unsupported executor '{executor}'")

    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ordered_rows: List[Optional[Dict[str, Union[float, np.ndarray]]]] = [None] * len(work_items)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            future_to_idx = {ex.submit(_worker_single, wi): idx for idx, wi in enumerate(work_items)}
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                row = fut.result()
                ordered_rows[idx] = row
                if pbar is not None:
                    pbar.update(1)
        rows.extend(row for row in ordered_rows if row is not None)
    else:
        for wi in work_items:
            row = _worker_single(wi)
            rows.append(row)
            if pbar is not None:
                pbar.update(1)

    elapsed_total = _time.perf_counter() - t0
    if pbar is not None:
        pbar.close()

    results_df = pd.DataFrame.from_records(rows)
    # Extract null draws if present
    if cfg.store_null_draws and ("pmd_null_draws" in results_df.columns):
        for idx, nd in results_df["pmd_null_draws"].items():
            if isinstance(nd, np.ndarray):
                r = results_df.loc[idx]
                for b, val in enumerate(nd):
                    null_rows.append(
                        {
                            "K": float(cfg.K),
                            "N1": float(cfg.N1),
                            "N2": float(cfg.N2),
                            "s": float(r["s"]),
                            "iter": float(r["iter"]),
                            "boot": float(b),
                            "pmd_null": float(val),
                        }
                    )
        results_df = results_df.drop(columns=["pmd_null_draws"])

    results_df["elapsed_total_sec"] = float(elapsed_total)
    if cfg.store_null_draws:
        null_df = pd.DataFrame.from_records(null_rows)
        return results_df, null_df
    else:
        return results_df


__all__ = [
    "SimulationConfig",
    "uniform_probs",
    "apply_overlap_shift",
    "simulate_two_column_counts",
    "run_single_simulation",
    "run_simulation_grid",
    "pairwise_js_distance",
]


def pairwise_js_distance(P: np.ndarray, base: float = 2.0) -> np.ndarray:
    """Compute pairwise Jensen–Shannon distance for probability columns.

    Parameters
    ----------
    P : ndarray (K, B)
        Columns are distributions (non-negative, sum to 1). No smoothing is applied.
    base : float
        Log base used for divergence; 2.0 bounds distances to [0,1].

    Returns
    -------
    D : ndarray (B, B)
        Symmetric matrix of JSD distances.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2:
        raise ValueError("P must be 2D with shape (K, B)")
    # Ensure columns sum to 1 (best-effort); if a column sums to 0, distance is NaN
    col_sums = P.sum(axis=0)
    valid = col_sums > 0
    P = P[:, valid] / col_sums[valid]
    B = P.shape[1]
    if B == 0:
        return np.full((0, 0), np.nan)
    # Broadcast to (K, B, B)
    P_i = P[:, :, None]
    P_j = P[:, None, :]
    M = 0.5 * (P_i + P_j)
    # Compute KL(P_i || M) safely: only where P_i > 0 and M > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        term_i = np.where(P_i > 0, P_i * (np.log(P_i) - np.log(M)), 0.0)
        term_j = np.where(P_j > 0, P_j * (np.log(P_j) - np.log(M)), 0.0)
    KL_i = term_i.sum(axis=0)
    KL_j = term_j.sum(axis=0)
    JSD = 0.5 * (KL_i + KL_j)
    if base and base != np.e:
        JSD = JSD / np.log(base)
    D = np.sqrt(np.maximum(JSD, 0.0))
    # Expand back to original columns if some were invalid
    out = np.full((col_sums.shape[0], col_sums.shape[0]), np.nan, dtype=float)
    idx = np.where(valid)[0]
    for a, ia in enumerate(idx):
        for b, ib in enumerate(idx):
            out[ia, ib] = D[a, b]
    return out
