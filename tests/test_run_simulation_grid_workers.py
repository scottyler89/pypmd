import pandas as pd
import pandas.testing as pdt

from percent_max_diff.benchmarking import SimulationConfig, run_simulation_grid

def make_cfg(n_workers: int) -> SimulationConfig:
    return SimulationConfig(
        K=6,
        N1=200,
        N2=200,
        s_values=[0, 2],
        iters=2,
        num_boot=10,
        random_state=123,
        sampling_model="multinomial",
        null_model="permutation",
        store_null_draws=False,
        show_progress=False,
        n_workers=n_workers,
    )

def test_run_simulation_grid_parallel_matches_sequential():
    cfg_seq = make_cfg(1)
    cfg_par = make_cfg(2)
    df_seq = run_simulation_grid(cfg_seq)
    df_par = run_simulation_grid(cfg_par)
    # Ensure identical ordering and values
    df_seq_sorted = df_seq.sort_values(["s", "iter"]).reset_index(drop=True)
    df_par_sorted = df_par.sort_values(["s", "iter"]).reset_index(drop=True)
    for col in ["elapsed_total_sec"]:
        if col in df_seq_sorted.columns:
            df_seq_sorted = df_seq_sorted.drop(columns=col)
        if col in df_par_sorted.columns:
            df_par_sorted = df_par_sorted.drop(columns=col)
    pdt.assert_frame_equal(df_seq_sorted, df_par_sorted, check_dtype=False)
