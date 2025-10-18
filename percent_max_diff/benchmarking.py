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

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

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


def simulate_two_column_counts(
    K: int,
    N1: int,
    N2: int,
    s: int,
    rng: np.random.Generator,
    *,
    base_probs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Simulate a K×2 count matrix with `s` non-shared clusters (overlap shift).

    Uses uniform base composition unless `base_probs` is provided.
    """

    p1 = uniform_probs(K) if base_probs is None else np.asarray(base_probs, dtype=float)
    if p1.shape[0] != K:
        raise ValueError("base_probs length must equal K")
    p2 = apply_overlap_shift(p1, s, K=K)
    x1 = sample_counts_multinomial(N1, p1, rng)
    x2 = sample_counts_multinomial(N2, p2, rng)
    return np.stack([x1, x2], axis=1)  # shape (K, 2)


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


def shannon_entropy_across_columns(
    X: np.ndarray, *, downsample: bool = True, rng: Optional[np.random.Generator] = None
) -> float:
    """Mean per-row Shannon entropy of the 2-vector across columns.

    If `downsample=True`, columns are downsampled without replacement to the
    minimum depth before computing per-row entropy, to avoid depth bias.
    """

    X = np.asarray(X, dtype=np.int64)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must be shape (K, 2)")
    K = X.shape[0]
    if downsample:
        if rng is None:
            rng = np.random.default_rng()
        d = int(min(X[:, 0].sum(), X[:, 1].sum()))
        if d == 0:
            return float("nan")
        # Draw d categorical samples per column according to column probabilities
        def draw_col(col_counts: np.ndarray) -> np.ndarray:
            p, ok = _normalize_counts_to_probs(col_counts, mode="none")
            if not ok:
                return np.zeros_like(col_counts)
            idx = rng.choice(np.arange(K), size=d, replace=True, p=p)
            out = np.zeros(K, dtype=int)
            np.add.at(out, idx, 1)
            return out

        Xds = np.stack([draw_col(X[:, 0]), draw_col(X[:, 1])], axis=1)
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
        # entropy base 2 to align with JSD docs
        entropies.append(stats.entropy(p, base=2))
    return float(np.nanmean(entropies))


# -----------------------------
# Metrics
# -----------------------------


def chi2_and_cramers_v(X: np.ndarray) -> Tuple[float, float, float]:
    """Compute chi-square test of independence and Cramer's V for (K, 2) table."""

    chi2, pval, dof, expected = stats.chi2_contingency(X, correction=False)
    N = X.sum()
    # For 2 columns, min(K-1, B-1) = 1
    v = np.sqrt(chi2 / (N * 1.0)) if N > 0 else np.nan
    return chi2, pval, v


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


def compute_pmd_metrics(X: np.ndarray, *, num_boot: int) -> Tuple[float, float, float]:
    """Compute raw PMD, debiased PMD using empirical λ, and λ.

    Returns
    -------
    raw : float
    debiased : float
    lam : float
    """

    raw = float(_get_pmd_raw(X))
    null_draws, _ = _get_null_vects(X, num_boot=num_boot)
    lam = float(np.mean(null_draws))
    debiased = (raw - lam) / (1.0 - lam) if lam != 1.0 else np.nan
    return raw, debiased, lam


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
) -> Dict[str, float]:
    """Simulate one table and compute selected metrics.

    Returns a flat dict suitable for DataFrame construction.
    """

    if rng is None:
        rng = np.random.default_rng()

    X = simulate_two_column_counts(K, N1, N2, s, rng)
    out: Dict[str, float] = {
        "K": float(K),
        "N1": float(N1),
        "N2": float(N2),
        "s": float(s),
        "min_E": float((X.sum(axis=1)[:, None] * X.sum(axis=0)[None, :] / X.sum()).min()) if X.sum() > 0 else np.nan,
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
        raw, debiased, lam = compute_pmd_metrics(X, num_boot=num_boot)
        if "pmd_raw" in metrics:
            out["pmd_raw"] = raw
            out["pmd_lambda"] = lam
        if "pmd" in metrics:
            out["pmd"] = debiased

    # Classical stats
    if ("chi2" in metrics) or ("cramers_v" in metrics) or ("chi2_p" in metrics):
        chi2, pval, v = chi2_and_cramers_v(X)
        if "chi2" in metrics:
            out["chi2"] = chi2
        if "chi2_p" in metrics:
            out["chi2_p"] = pval
        if "cramers_v" in metrics:
            out["cramers_v"] = v

    # Diversity/entropy
    if "inv_simpson" in metrics:
        out["inv_simpson"] = inverse_simpson_per_column(X)
    if "entropy" in metrics:
        out["entropy"] = shannon_entropy_across_columns(X, downsample=True, rng=rng)

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


def run_simulation_grid(cfg: SimulationConfig) -> pd.DataFrame:
    """Run a small grid over s_values × iters and return a tidy DataFrame."""

    rng = np.random.default_rng(cfg.random_state)
    rows: List[Dict[str, float]] = []
    for s in cfg.s_values:
        for rep in range(cfg.iters):
            seed = rng.integers(0, 2**63 - 1, dtype=np.uint64)
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
            )
            row["iter"] = float(rep)
            rows.append(row)
    return pd.DataFrame.from_records(rows)


__all__ = [
    "SimulationConfig",
    "uniform_probs",
    "apply_overlap_shift",
    "simulate_two_column_counts",
    "run_single_simulation",
    "run_simulation_grid",
]

