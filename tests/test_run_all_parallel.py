import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pandas.testing as pdt


ROOT = Path(__file__).resolve().parents[1]


def _run(tag: str) -> Path:
    cmd = [
        sys.executable,
        os.path.join("benchmarks", "run_all_benchmarks.py"),
        "--K", "4",
        "--N1", "50",
        "--N2", "50",
        "--smax", "1",
        "--iters", "1",
        "--num-boot", "2",
        "--seed", "123",
        "--expand-union", "false",
        "--store-null", "false",
        "--r-compat", "false",
        "--pdf", "false",
        "--super-grid", "false",
        "--lambda-violin-by", "",
        "--lambda-ridgeline-by", "",
        "--validate-invariance", "false",
        "--N1-list", "50",
        "--N2-list", "50",
        "--num-clust-range", "4",
        "--percent-difference", "0,1",
        "--b1-sizes", "50",
        "--b2-sizes", "50",
        "--max-workers", "2",
        "--tag", tag,
    ]
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MKL_THREADING_LAYER": "GNU",
        "PMD_SKIP_PLOTS": "1",
    })
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(ROOT)
    subprocess.run(cmd, check=True, env=env)

    out_root = Path("benchmarks") / "out"
    runs = sorted(out_root.glob(f"*_{tag}"))
    if not runs:
        raise AssertionError("No output directory created for tag")
    return runs[-1]


def _assert_scenario_outputs(base_dir: Path) -> pd.DataFrame:
    expected = {
        "characterize": base_dir / "characterize" / "results.csv",
        "overlays": base_dir / "overlays" / "results.csv",
        "invariance_symmetric": base_dir / "invariance" / "symmetric" / "invariance_results.csv",
        "invariance_expand_b2_only": base_dir / "invariance" / "expand_b2_only" / "invariance_results.csv",
        "invariance_fixed_union_private": base_dir / "invariance" / "fixed_union_private" / "invariance_results.csv",
    }
    for label, path in expected.items():
        assert path.exists(), f"Missing expected output for {label}: {path}"

    unified = base_dir / "unified_results.csv"
    assert unified.exists()
    df = pd.read_csv(unified)
    assert not df.empty
    scenarios = set(df["scenario"].unique())
    assert {"characterize", "overlays", "invariance"}.issubset(scenarios)
    for required in ["dataset_pair_set", "n_workers"]:
        assert required in df.columns, f"unified results missing {required}"
        assert df[required].notna().any()

    manifest = base_dir / "manifest.json"
    assert manifest.exists()

    unified_nulls = base_dir / "unified_nulls.csv"
    assert unified_nulls.exists()
    df_null = pd.read_csv(unified_nulls)
    for required in ["dataset_pair_set", "n_workers"]:
        assert required in df_null.columns, f"unified nulls missing {required}"
    return df


def _prep_for_compare(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    if "K" not in df.columns:
        if "NumberOfClusters" in df.columns:
            df["K"] = df["NumberOfClusters"]
        elif "total_number_of_clusters" in df.columns:
            df["K"] = df["total_number_of_clusters"]
        else:
            df["K"] = 0
    if "x_numeric" in df.columns:
        df["_cmp_x"] = df["x_numeric"]
    elif "percent_different_clusters_numeric" in df.columns:
        df["_cmp_x"] = df["percent_different_clusters_numeric"]
    elif "s" in df.columns:
        df["_cmp_x"] = df["s"]
    else:
        df["_cmp_x"] = 0
    df["_cmp_mode"] = df["mode"] if "mode" in df.columns else "NA"
    df["_cmp_iter"] = df["iter"] if "iter" in df.columns else 0
    df["_cmp_K"] = df["K"]
    # Normalize numeric keys to avoid floating epsilon mismatches
    for col in ["_cmp_K", "_cmp_x", "_cmp_iter"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(8)
    key_cols = ["scenario", "dataset_sizes", "_cmp_mode", "_cmp_K", "_cmp_x", "_cmp_iter"]
    return df, key_cols


def test_run_all_parallel_small(tmp_path):
    tag1 = f"pytest_{int(time.time())}"
    base_dir1 = _run(tag1)
    try:
        df1 = _assert_scenario_outputs(base_dir1)

        # Re-run with the same configuration to verify reproducibility.
        time.sleep(1)  # ensure distinct timestamped directory
        tag2 = f"{tag1}_repeat"
        base_dir2 = _run(tag2)
        df2 = _assert_scenario_outputs(base_dir2)

        common_cols = sorted(set(df1.columns) & set(df2.columns))
        noisy = {"elapsed_total_sec", "braycurtis", "canberra", "chi_neg_log_p", "chi_sq", "chi2"}
        df1_norm, key_cols = _prep_for_compare(df1)
        df2_norm, _ = _prep_for_compare(df2)
        merged = df1_norm.merge(df2_norm, on=key_cols, how="outer", indicator=True, suffixes=("_run1", "_run2"))
        if not (merged["_merge"] == "both").all():
            missing = merged[merged["_merge"] != "both"][key_cols + ["_merge"]]
            raise AssertionError(f"Row alignment mismatch:\n{missing.head()}")
        merged = merged.drop(columns="_merge")
        base_cols = sorted(set(df1.columns) & set(df2.columns))
        key_skip = {
            "scenario", "dataset_sizes", "mode", "K",
            "NumberOfClusters", "total_number_of_clusters",
            "s", "percent_different_clusters_numeric", "x_numeric", "iter",
            "b1_size", "b2_size", "N1", "N2"
        }
        # Compare per-key groups as multisets to ignore internal row ordering within duplicate keys
        for col in base_cols:
            if col in noisy or col in key_skip:
                continue
            col_run1 = f"{col}_run1" if f"{col}_run1" in merged.columns else col
            col_run2 = f"{col}_run2" if f"{col}_run2" in merged.columns else col
            if col_run1 == col_run2:
                continue
            assert col_run1 in merged.columns and col_run2 in merged.columns
            g1 = merged.groupby(key_cols)[col_run1].apply(lambda s: tuple(pd.Series(s).astype(object).tolist()))
            g2 = merged.groupby(key_cols)[col_run2].apply(lambda s: tuple(pd.Series(s).astype(object).tolist()))
            # Sort tuples to compare as multisets; handle NaNs by mapping to a sentinel
            def norm_tup(t):
                arr = [('_nan_' if (isinstance(x, float) and pd.isna(x)) else x) for x in t]
                try:
                    return tuple(sorted(arr))
                except Exception:
                    return tuple(arr)
            g1n = g1.map(norm_tup); g1n.name = col
            g2n = g2.map(norm_tup); g2n.name = col
            pdt.assert_series_equal(
                g1n, g2n,
                check_dtype=False,
                check_exact=True,
                check_names=False,
                obj=f"{col} (grouped)")
    finally:
        if base_dir1.exists():
            shutil.rmtree(base_dir1, ignore_errors=True)
        if 'base_dir2' in locals() and base_dir2.exists():
            shutil.rmtree(base_dir2, ignore_errors=True)
