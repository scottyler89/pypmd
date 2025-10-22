import numpy as np
import pandas as pd

from percent_max_diff.benchmarking import SimulationConfig, run_simulation_grid


def test_metric_bounds_small_run():
    cfg = SimulationConfig(K=6, N1=500, N2=500, s_values=range(0, 4), iters=2, num_boot=50, random_state=123)
    df = run_simulation_grid(cfg)
    # Bounds checks (where present)
    for col, lo, hi in [
        ("pmd_raw", 0.0, 1.0),
        ("pmd", 0.0, 1.0),
        ("cramers_v", 0.0, 1.0),
        ("chi2_p", 0.0, 1.0),
        ("jsd", 0.0, 1.0),
        ("tv", 0.0, 1.0),
        ("hellinger", 0.0, 1.0),
        ("braycurtis", 0.0, 1.0),
        ("cosine", 0.0, 1.0),
    ]:
        if col in df.columns:
            vals = df[col].dropna()
            assert ((vals >= lo) & (vals <= hi)).all(), f"{col} out of bounds"


def test_monotone_trend_on_average():
    cfg = SimulationConfig(K=8, N1=1000, N2=1000, s_values=range(0, 6), iters=3, num_boot=30, random_state=7)
    df = run_simulation_grid(cfg)
    means = df.groupby("s")["pmd"].mean(numeric_only=True)
    # Allow minor dips
    dips = sum(means.diff().dropna() < -1e-6)
    assert dips <= 1, f"Too many dips in monotonicity: {dips}"


def test_null_centering_approx_zero():
    cfg = SimulationConfig(K=8, N1=1000, N2=1000, s_values=[0], iters=4, num_boot=50, random_state=123)
    df = run_simulation_grid(cfg)
    assert "pmd" in df.columns
    mu = df["pmd"].mean()
    assert abs(mu) < 0.2, f"PMD* under null not ~0, got {mu}"


def test_deterministic_with_seed():
    cfg = SimulationConfig(K=6, N1=500, N2=500, s_values=range(3), iters=2, num_boot=20, random_state=999)
    df1 = run_simulation_grid(cfg)
    df2 = run_simulation_grid(cfg)
    # Sort by s,iter to compare; drop variable fields
    keep = [c for c in df1.columns if c not in {"elapsed_sec", "seed", "elapsed_total_sec"}]
    d1 = df1[keep].sort_values(by=["s", "iter"]).reset_index(drop=True)
    d2 = df2[keep].sort_values(by=["s", "iter"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(d1, d2, check_exact=False, rtol=1e-12, atol=1e-12)
