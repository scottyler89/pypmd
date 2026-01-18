import os
import sys
from pathlib import Path
import subprocess
import pandas as pd


def test_plot_lambda_k_linestyles(tmp_path):
    # Create a tiny results.csv with multiple dataset_sizes and two K values
    rows = []
    for ds in ["100_vs_100", "250_vs_250"]:
        for K in [4, 8]:
            for i, lam in enumerate([0.02, 0.03, 0.04]):
                rows.append({
                    "dataset_sizes": ds,
                    "K": K,
                    "pmd_lambda": lam,
                    "total_clusters": K,
                    "min_E": 10,
                })
    df = pd.DataFrame(rows)
    df.to_csv(tmp_path / "results.csv", index=False)

    # Run the plotter; it should complete and write files without needing a display
    cmd = [sys.executable, os.path.join("benchmarks", "plot_lambda.py"), "--dir", str(tmp_path)]
    subprocess.run(cmd, check=True)

    # Check outputs exist
    assert (tmp_path / "lambda_density.png").exists()
    assert (tmp_path / "lambda_by_predictors.png").exists()
    # Check legend summary reflects both dataset colors and multiple K values
    import json
    with open(tmp_path / "lambda_legend.json") as fh:
        info = json.load(fh)
    assert info.get("hue_col") in {"dataset_sizes", "dataset_label"}
    assert set(info.get("colors", [])) == {"100_vs_100", "250_vs_250"}
    k_list = info.get("k_values", [])
    assert sorted(set(k_list)) == [4, 8]
