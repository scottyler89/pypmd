"""pypmd: convenience alias for the percent_max_diff API.

This package re-exports the public API from `percent_max_diff` so users can
import either of the following:

- `from percent_max_diff import pmd, PMD, get_pmd, get_pmd_pairs`
- `from pypmd import pmd, PMD, get_pmd, get_pmd_pairs`
"""

from percent_max_diff import pmd, PMD, get_pmd, get_pmd_pairs, __version__

__all__ = ["pmd", "PMD", "get_pmd", "get_pmd_pairs", "__version__"]

