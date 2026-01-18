import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmarks import config
from benchmarks import run_all_benchmarks as rab
from benchmarks import run_multi_config as rmc
from benchmarks import run_invariance_property as rip
from benchmarks import run_characterize_pmd as rch


def _parse_defaults(module):
    argv = sys.argv
    sys.argv = [module.__file__]
    try:
        ns = module.parse_args()
    finally:
        sys.argv = argv
    return ns


def test_run_all_defaults_match_config():
    ns = _parse_defaults(rab)
    assert ns.dataset_pair_set == config.DEFAULT_DATASET_PAIR_SET
    n1, n2 = config.default_dataset_pairs_str()
    assert (ns.N1_list or n1) == n1
    assert (ns.N2_list or n2) == n2
    inv = config.default_invariance()
    assert ns.num_clust_range == inv["k_list_str"]
    assert ns.percent_difference == inv["pdiffs_str"]
    assert (ns.b1_sizes or n1) == n1
    assert (ns.b2_sizes or n2) == n2
    assert ns.max_workers is None


def test_run_multi_defaults_match_config():
    ns = _parse_defaults(rmc)
    assert ns.dataset_pair_set == config.DEFAULT_DATASET_PAIR_SET
    n1, n2 = config.default_dataset_pairs_str()
    assert (ns.N1_list or n1) == n1
    assert (ns.N2_list or n2) == n2
    assert ns.max_workers is None


def test_run_invariance_defaults_match_config():
    ns = _parse_defaults(rip)
    assert ns.dataset_pair_set == config.DEFAULT_DATASET_PAIR_SET
    n1, n2 = config.default_dataset_pairs_str()
    inv = config.default_invariance()
    assert (ns.b1_sizes or n1) == n1
    assert (ns.b2_sizes or n2) == n2
    assert ns.num_clust_range == inv["k_list_str"]
    assert ns.percent_difference == inv["pdiffs_str"]
    assert ns.max_workers is None


def test_run_characterize_defaults_match_config():
    ns = _parse_defaults(rch)
    assert ns.smax == config.CHAR_SMAX
    assert ns.iters == config.CHAR_ITERS
    assert ns.num_boot == config.CHAR_NUM_BOOT
    assert ns.max_workers is None


def test_dataset_pair_sets_partition():
    signal = config.default_dataset_pairs("signal")
    low = config.default_dataset_pairs("low_signal")
    all_pairs = config.default_dataset_pairs("all")
    assert signal and low
    assert set(signal).isdisjoint(set(low))
    assert signal + low == all_pairs
