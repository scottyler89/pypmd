import sys
from pathlib import Path
import pandas as pd


def _write_results(tmp_path: Path):
    df = pd.DataFrame(
        {
            "K": [10]*4,
            "N1": [2000]*4,
            "N2": [2000]*4,
            "s": [0, 2, 4, 6],
            "pmd_raw": [0.0, 0.3, 0.6, 0.9],
            "pmd": [0.0, 0.25, 0.5, 0.85],
            "chi2": [1.0, 10.0, 50.0, 100.0],
            "chi2_p": [0.5, 1e-5, 1e-20, 0.0],
            "pmd_lambda": [0.05, 0.06, 0.07, 0.08],
            "min_E": [10.0, 10.0, 10.0, 10.0],
            "total_clusters": [10, 10, 10, 10],
            "observed_clusters": [10, 10, 10, 10],
        }
    )
    (tmp_path / "results.csv").write_text(df.to_csv(index=False))


def test_plot_pmd_focus(tmp_path):
    _write_results(tmp_path)
    sys.path.insert(0, str(Path(".").resolve()))
    import benchmarks.plot_pmd_focus as m
    m.main.__wrapped__ if hasattr(m.main, "__wrapped__") else m.main()


def test_plot_lambda(tmp_path):
    _write_results(tmp_path)
    sys.path.insert(0, str(Path(".").resolve()))
    import benchmarks.plot_lambda as m
    m.main.__wrapped__ if hasattr(m.main, "__wrapped__") else m.main()

