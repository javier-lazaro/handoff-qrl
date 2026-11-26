#!/usr/bin/env python3
"""Generate/check the deterministic cross-environment paper analysis tables."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from handoff_qrl.csvcheck import csv_equivalent
from hybrid_training_qrl.paper_analysis_tables import TABLE_FIELDS, build_paper_analysis_tables
from hybrid_training_qrl.paper_tables import write_csv

ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = ROOT / "results"
OUTPUT_DIR = ROOT / "analysis/tables"


def _write_tables(output_dir: Path) -> dict[str, Path]:
    tables = build_paper_analysis_tables(PAPER_ROOT)
    output: dict[str, Path] = {}
    for filename, rows in tables.items():
        path = output_dir / filename
        write_csv(path, rows, fieldnames=TABLE_FIELDS[filename])
        output[filename] = path
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate temporarily and fail on any substantive table difference.",
    )
    args = parser.parse_args()

    if args.check:
        with tempfile.TemporaryDirectory(prefix="paper-analysis-") as tmp:
            generated = _write_tables(Path(tmp))
            mismatches: list[str] = []
            for filename, generated_path in generated.items():
                tracked = OUTPUT_DIR / filename
                equivalent, detail = csv_equivalent(tracked, generated_path)
                if not equivalent:
                    mismatches.append(filename)
                    print(f"{filename}: {detail}")
            if mismatches:
                raise SystemExit(
                    "paper analysis tables are stale or missing: " + ", ".join(sorted(mismatches))
                )
        print("Paper cross-environment analysis tables are reproducible: PASS")
        return 0

    written = _write_tables(OUTPUT_DIR)
    print("Paper cross-environment analysis tables generated from canonical results/*")
    tables = build_paper_analysis_tables(PAPER_ROOT)
    print(f"  condition summary: {len(tables['condition_summary.csv'])} rows")
    print(f"  descriptive FC/teacher references: {len(tables['fc_reference_contrasts.csv'])} rows")
    print(f"  descriptive continuation references: {len(tables['continuation_reference_contrasts.csv'])} rows")
    print(f"  allocation normalization: {len(tables['allocation_normalized.csv'])} rows")
    print("  environments: LunarLander retrospective + MinAtar prospective confirmatory")
    print("  resource accounting: grad/no-grad + fixed KD / online decomposition")
    print("  timing/energy: MinAtar serial resource cohort seeds 0-4 only")
    print("  KD diagnostics: common H/C action-agreement + policy-KL surface")
    print(f"  output directory: {OUTPUT_DIR.relative_to(ROOT)} ({len(written)} tables)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
