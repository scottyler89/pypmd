import os
import sys
from pathlib import Path
import pandas as pd

from benchmarks.config import default_dataset_pairs


def _write_invariance(tmp: Path):
    # minimal invariance results: two K values, several percents, include NaN for Cramer's V at 0
    ds_pairs = default_dataset_pairs()
    first_pair = ds_pairs[0]
    ds_label = f"{first_pair[0]}_vs_{first_pair[1]}"
    rows = []
    for K in [4, 8]:
        for p in [0.0, 0.5, 1.0]:
            rows.append({
                "dataset_sizes": ds_label,
                "iter": 0,
                "NumberOfClusters": K,
                "percent_different_clusters_numeric": p,
                "mode": "symmetric",
                "raw_pmd": p,
                "pmd": p,
                "chi_sq": 1.0 + 10*p,
                "chi_neg_log_p": 2.0 + 10*p,
                "cramers_v_bc": (None if p==0.0 else 0.1 + 0.8*p),
                "inverse_simp": 1.0 + p,
                "shannon_entropy": p,
                "b1_size": first_pair[0],
                "b2_size": first_pair[1],
                "total_number_of_clusters": K,
            })
    df = pd.DataFrame(rows)
    (tmp / "invariance_results.csv").write_text(df.to_csv(index=False))


def test_plot_invariance_super_grid(tmp_path):
    _write_invariance(tmp_path)
    sys.path.insert(0, str(Path(".").resolve()))
    import benchmarks.plot_invariance as m
    # Patch argv to pass file + output
    import importlib
    m.parse_args = lambda: type("Args", (), {"file": str(tmp_path / "invariance_results.csv"), "out": str(tmp_path / "invariance_super_grid.png"), "full_grid":"true", "include_raw_percent":"false", "separate_panels":"false"})
    importlib.reload(m)
    m.main()
    assert (tmp_path / "invariance_super_grid.png").exists()


def test_plot_characterize_null_density(tmp_path):
    # Minimal characterize results + null
    res = pd.DataFrame({
        "K":[10]*3, "N1":[2000]*3, "N2":[2000]*3, "s":[0,1,2],
        "pmd_raw":[0.0,0.5,1.0], "pmd":[0.0,0.4,0.9], "chi2":[1.0,5.0,10.0],
        "chi2_p":[0.5, 1e-10, 0.0], "cramers_v_bc":[0.0,0.5,1.0],
        "inv_simpson":[1.0,2.0,1.5], "entropy":[0.0,0.6,0.9],
        "iter":[0,0,0], "seed":[0,0,0], "elapsed_total_sec":[0.0,0.0,0.0]
    })
    null = pd.DataFrame({"pmd_null":[0.02,0.03,0.02,0.01], "s":[0,0,0,0]})
    (tmp_path/"results.csv").write_text(res.to_csv(index=False))
    (tmp_path/"null.csv").write_text(null.to_csv(index=False))
    sys.path.insert(0, str(Path(".").resolve()))
    import benchmarks.plot_characterize_pmd as m
    import importlib
    m.parse_args = lambda: type("Args", (), {"results": str(tmp_path/"results.csv"), "dir":"", "latest_tag":"", "null": str(tmp_path/"null.csv"), "out": str(tmp_path/"characterize_pmd.png"), "pdf":"", "merged_grid":"false", "super_grid":"true", "null_density_out": str(tmp_path/"null_pmd_density.png"), "separate_panels":"false"})
    importlib.reload(m)
    m.main()
    assert (tmp_path/"characterize_pmd.png").exists()
    assert (tmp_path/"null_pmd_density.png").exists()
