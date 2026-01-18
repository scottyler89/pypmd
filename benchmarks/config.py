"""Centralized benchmark defaults (SSoT) and small helpers.

Import these defaults from all runners/plotters to avoid hard-coding.
Also houses the parallelization policy so runners stay consistent."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os


# Dataset size pairs used across scenarios (b1_size, b2_size)
DATASET_PAIRS_SIGNAL: List[Tuple[int, int]] = [
    (10000, 10000),
    (10000, 1000),
    (1000, 1000),
    (250, 2000),
    (250, 250),
]

DATASET_PAIRS_LOW_SIGNAL: List[Tuple[int, int]] = [
    (100, 100),
    (25, 25),
    (10, 10),
    (5, 5),
    (1, 1),
]

DATASET_PAIR_SETS: Dict[str, List[Tuple[int, int]]] = {
    "signal": DATASET_PAIRS_SIGNAL,
    "low_signal": DATASET_PAIRS_LOW_SIGNAL,
}
DATASET_PAIR_SETS["all"] = DATASET_PAIR_SETS["signal"] + DATASET_PAIR_SETS["low_signal"]

DEFAULT_DATASET_PAIR_SET: str = "signal"

# Parallel defaults (0/1 means sequential, >1 enables multiprocessing)
DEFAULT_MAX_WORKERS: int = 0


# Characterize defaults
CHAR_SMAX: int = 10
CHAR_ITERS: int = 5
CHAR_NUM_BOOT: int = 200


# Invariance defaults
INV_K_LIST: List[int] = [4, 8, 12, 16]
INV_PDIFFS: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
INV_ITERS: int = 5
INV_NUM_BOOT: int = 200


def dataset_pair_sets() -> List[str]:
    return list(DATASET_PAIR_SETS.keys())


def default_dataset_pairs(region: str = DEFAULT_DATASET_PAIR_SET) -> List[Tuple[int, int]]:
    if region not in DATASET_PAIR_SETS:
        raise ValueError(f"Unknown dataset pair set '{region}'. Options: {sorted(DATASET_PAIR_SETS)}")
    return list(DATASET_PAIR_SETS[region])


def default_dataset_pairs_str(region: str = DEFAULT_DATASET_PAIR_SET) -> Tuple[str, str]:
    pairs = default_dataset_pairs(region)
    n1 = ",".join(str(a) for a, _ in pairs)
    n2 = ",".join(str(b) for _, b in pairs)
    return n1, n2


def default_invariance() -> Dict[str, object]:
    return {
        "k_list": list(INV_K_LIST),
        "k_list_str": ",".join(str(k) for k in INV_K_LIST),
        "pdiffs": list(INV_PDIFFS),
        "pdiffs_str": ",".join(str(x) for x in INV_PDIFFS),
        "iters": INV_ITERS,
        "num_boot": INV_NUM_BOOT,
    }


def resolve_max_workers(requested: Optional[int]) -> int:
    """Normalize requested workers against policy and hardware.

    - None → DEFAULT_MAX_WORKERS
    - 0 or 1 → sequential (returns 1)
    - <0 → all available CPUs
    - >available CPUs → clamp to hardware count
    """
    req = DEFAULT_MAX_WORKERS if requested is None else requested
    if req in (0, 1):
        return 1
    cpu_total = os.cpu_count() or 1
    if req < 0:
        return cpu_total
    if req > cpu_total:
        return cpu_total
    return max(1, req)


def parallel_env() -> Dict[str, str]:
    """Return environment overrides keeping BLAS/OpenMP single-threaded per worker."""
    return {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MKL_THREADING_LAYER": "GNU",
        "NUMEXPR_NUM_THREADS": "1",
    }


# Unified results schema helpers
RENAME_MAP_PER_SCENARIO: Dict[str, Dict[str, str]] = {
    # invariance CSV normalizations
    "invariance": {
        "percent_different_clusters_numeric": "x_numeric",
        "NumberOfClusters": "K",
        "cramers_v_bc": "cramers_v",
    },
    # characterize/overlays often already aligned; prefer BC if present
    "characterize": {
        "cramers_v_bc": "cramers_v",
    },
    "overlays": {
        "cramers_v_bc": "cramers_v",
    },
}


@dataclass
class Palette:
    datasets: List[str] = None


def default_palette_for_datasets(labels: List[str]) -> Dict[str, str]:
    """Return a stable color palette for dataset labels."""
    # Simple qualitative palette drawn deterministically from label order
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    colors: Dict[str, str] = {}
    for i, lab in enumerate(labels):
        colors[lab] = base[i % len(base)]
    return colors


def _defaults_payload() -> Dict[str, object]:
    """Return a JSON-serialisable snapshot of benchmark defaults."""
    inv = default_invariance()
    payload = {
        "dataset_pair_sets": {
            name: default_dataset_pairs(name)
            for name in DATASET_PAIR_SETS
        },
        "default_dataset_pair_set": DEFAULT_DATASET_PAIR_SET,
        "parallel": {
            "default_max_workers": DEFAULT_MAX_WORKERS,
            "policy": "0 or 1 -> sequential; >1 -> multiprocessing; -1 -> all cores",
        },
        "characterize": {
            "smax": CHAR_SMAX,
            "iters": CHAR_ITERS,
            "num_boot": CHAR_NUM_BOOT,
        },
        "invariance": {
            "k_list": inv["k_list"],
            "percent_differences": inv["pdiffs"],
            "iters": inv["iters"],
            "num_boot": inv["num_boot"],
        },
    }
    return payload


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect benchmark default configuration.")
    parser.add_argument(
        "--print",
        dest="emit",
        action="store_true",
        help="Print the current benchmark defaults (JSON).",
    )
    args = parser.parse_args(argv)
    if args.emit:
        print(json.dumps(_defaults_payload(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
