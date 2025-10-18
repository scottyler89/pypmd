"""Top-level package for percent_max_diff (PMD).

Public API:

- `pmd`: main class computing PMD and related statistics.
- `PMD`: alias to `pmd` for conventional class-style naming.
- `get_pmd`: low-level function returning unadjusted PMD for a 2D count matrix.
- `get_pmd_pairs`: compute pairwise PMD between columns.

Avoid star imports; we surface only the supported symbols via `__all__`.
"""

from .percent_max_diff import pmd, get_pmd, get_pmd_pairs

# Conventional alias without breaking backwards compatibility
PMD = pmd

__all__ = ["pmd", "PMD", "get_pmd", "get_pmd_pairs"]

# Keep a simple version marker in the package for convenience
__version__ = "0.99.5"
