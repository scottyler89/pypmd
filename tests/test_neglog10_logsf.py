import os
import sys
import numpy as np
from scipy import stats

from percent_max_diff.benchmarking import cramers_v_stat


def test_logsf_neglog10_stable():
    # Construct a strongly separated 2x2 table with enormous chi2
    X = np.array([[50000, 0], [0, 50000]], dtype=int)
    chi2, pval, _ = cramers_v_stat(X, bias_correction=False)
    # p-value may underflow to 0 via chi2_contingency; logsf should remain finite
    dof = (X.shape[0] - 1) * (X.shape[1] - 1)
    neglog10 = -stats.chi2.logsf(chi2, dof) / np.log(10.0)
    assert np.isfinite(neglog10)
    assert neglog10 > 50  # extremely significant


def test_plot_characterize_handles_zero_p(tmp_path):
    # Prepare a tiny results.csv where chi2_p underflows to zero but chi2 is present
    import pandas as pd
    from importlib import reload
    import benchmarks.plot_characterize_pmd as plotmod

    df = pd.DataFrame(
        {
            "K": [10, 10, 10],
            "N1": [2000, 2000, 2000],
            "N2": [2000, 2000, 2000],
            "s": [0, 5, 10],
            "pmd_raw": [0.0, 0.5, 1.0],
            "pmd": [0.0, 0.4, 0.9],
            "chi2": [1e6, 5e5, 1e6],
            "chi2_p": [0.0, 0.0, 0.0],  # simulate underflow
            "cramers_v_bc": [0.0, 0.5, 1.0],
            "inv_simpson": [1.0, 2.0, 1.5],
            "entropy": [0.0, 0.6, 0.9],
            "observed_clusters": [10, 10, 10],
            "iter": [0, 0, 0],
            "seed": [0, 0, 0],
            "elapsed_total_sec": [0.0, 0.0, 0.0],
        }
    )
    res_path = tmp_path / "results.csv"
    df.to_csv(res_path, index=False)

    out_path = tmp_path / "characterize_pmd.png"
    # Emulate CLI invocation to use argparse in the module
    argv = ["plot_characterize_pmd.py", str(res_path), "--out", str(out_path)]
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        reload(plotmod)
        plotmod.main()
    finally:
        sys.argv = old_argv
    assert out_path.exists()

