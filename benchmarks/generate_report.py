#!/usr/bin/env python3
"""Generate a simple markdown report for a base output directory.

Scans characterize/, overlays/, invariance/ for CSVs and images and creates a
report.md with links and brief summaries.
"""

from __future__ import annotations

import argparse
import glob
import os
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base output directory")
    ap.add_argument("--out", default="", help="output markdown path (default base/report.md)")
    return ap.parse_args()


def section(title: str) -> str:
    return f"\n## {title}\n\n"


def main() -> None:
    args = parse_args()
    base = args.base
    out = args.out or os.path.join(base, "report.md")
    parts = [f"# PMD Benchmark Report\n\nBase: {base}\n"]

    # Characterize
    ch = os.path.join(base, "characterize")
    if os.path.isdir(ch):
        parts.append(section("Characterize"))
        for name in ["results.csv","null.csv","results_r.csv","null_r.csv","characterize_pmd.png","characterize_comparators.png","characterize_pmd.pdf","characterize_super_grid.png","characterize_merged.png","lambda_model.csv"]:
            fp = os.path.join(ch, name)
            if os.path.exists(fp):
                parts.append(f"- {name}: {os.path.relpath(fp, base)}\n")

    # Overlays
    ov = os.path.join(base, "overlays")
    if os.path.isdir(ov):
        parts.append(section("Overlays (Multi-config)"))
        for name in ["results.csv","null.csv","results_r.csv","null_r.csv","characterize_pmd.png","characterize_comparators.png","characterize_super_grid.png","characterize_merged.png"]:
            fp = os.path.join(ov, name)
            if os.path.exists(fp):
                parts.append(f"- {name}: {os.path.relpath(fp, base)}\n")

    # Invariance
    inv = os.path.join(base, "invariance")
    if os.path.isdir(inv):
        parts.append(section("Invariance"))
        for name in ["invariance_results.csv","invariance_results_r.csv","invariance.png","invariance_super_grid.png"]:
            fp = os.path.join(inv, name)
            if os.path.exists(fp):
                parts.append(f"- {name}: {os.path.relpath(fp, base)}\n")

    with open(out, "w") as fh:
        fh.write("".join(parts))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

