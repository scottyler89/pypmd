import json
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmarks.config import default_dataset_pairs


def make_df(generator):
    rows = []
    x_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    labels = [f"{a}_vs_{b}" for a, b in default_dataset_pairs()[:2]]
    for ds in labels:
        for K in [4, 8]:
            for x in x_levels:
                for rep in range(3):
                    rows.append({
                        "dataset_sizes": ds,
                        "NumberOfClusters": K,
                        "percent_different_clusters_numeric": x,
                        "mode": "symmetric",
                        "pmd": generator(x=x, K=K, ds=ds),
                    })
    return pd.DataFrame(rows)


def run_validator(df, tmp_path, *, include_interactions=False):
    inv_csv = tmp_path / "invariance_results.csv"
    df.to_csv(inv_csv, index=False)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        "benchmarks/validate_invariance_onehot.py",
        "--file",
        str(inv_csv),
        "--mode",
        "symmetric",
        "--out-dir",
        str(out_dir),
    ]
    if include_interactions:
        cmd.extend(["--include-interactions", "true"])
    subprocess.run(cmd, check=True)

    mode_dir = out_dir / "symmetric"
    bm_path = mode_dir / "invariance_onehot_boolean_matrix.csv"
    stats_path = mode_dir / "invariance_onehot_stats.json"

    bm = pd.read_csv(bm_path)
    with open(stats_path) as fh:
        stats = json.load(fh)
    return bm, stats, mode_dir


def test_validate_invariance_onehot_linear(tmp_path):
    df = make_df(lambda x, **_: x)
    bm, stats, mode_dir = run_validator(df, tmp_path)

    row = bm.loc[bm["metric"] == "pmd"].iloc[0]
    assert bool(row["Linear_onehot_vs_numeric"])
    assert bool(row["Slope_nonzero"])
    assert bool(row["InterceptsEqual_K_zero"])
    assert bool(row["InterceptsEqual_DS_zero"])
    assert bool(row["Overall"])

    metric_stats = stats["metrics"]["pmd"]
    assert metric_stats["linear_bic"] is True
    assert metric_stats.get("max_abs_CK_coef", 0.0) <= 0.05
    assert metric_stats.get("max_abs_CDS_coef", 0.0) <= 0.05


def test_validate_invariance_onehot_nonlinear(tmp_path):
    df = make_df(lambda x, **_: 0.0 if x < 0.5 else 1.0)
    bm, stats, mode_dir = run_validator(df, tmp_path)

    row = bm.loc[bm["metric"] == "pmd"].iloc[0]
    assert not row["Linear_onehot_vs_numeric"]
    assert bool(row["Slope_nonzero"])
    assert not row["Overall"]

    metric_stats = stats["metrics"]["pmd"]
    assert metric_stats.get("linear_dev") is False
    assert metric_stats.get("dev_improvement_ratio", 0.0) > 0.5


def test_validate_invariance_onehot_intercept_shift(tmp_path):
    df = make_df(lambda x, ds, **_: x + (0.1 if ds == "2000_vs_2000" else 0.0))
    bm, stats, mode_dir = run_validator(df, tmp_path)

    row = bm.loc[bm["metric"] == "pmd"].iloc[0]
    assert not row["InterceptsEqual_DS_zero"]
    assert not row["Overall"]

    metric_stats = stats["metrics"]["pmd"]
    assert metric_stats["p_int_DS"] < 0.01


def test_validate_invariance_onehot_slope_shift(tmp_path):
    df = make_df(lambda x, K, **_: (1.0 if K == 4 else 2.0) * x)
    bm, stats, mode_dir = run_validator(df, tmp_path, include_interactions=True)

    row = bm.loc[bm["metric"] == "pmd"].iloc[0]
    assert not row["SlopesEqual_K_zero"]
    assert not row["Overall"]

    metric_stats = stats["metrics"]["pmd"]
    assert metric_stats["p_slope_K"] < 0.01
    mm_dir = mode_dir / "model_matrices"
    assert (mm_dir / "pmd_numeric_design.csv").exists()
    assert (mm_dir / "pmd_categorical_design.csv").exists()
