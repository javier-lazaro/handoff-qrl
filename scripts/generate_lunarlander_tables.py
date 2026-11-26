#!/usr/bin/env python3
"""Regenerate LunarLander paper-analysis tables from canonical evidence."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from handoff_qrl.csvcheck import csv_equivalent
from hybrid_training_qrl.paper_stats import load_statistics_contract
from hybrid_training_qrl.paper_tables import (
    LUNAR_PREDICTION_FIELDS,
    P3_FIELDS,
    PRIMARY_TABLE_FIELDS,
    RQ3_FIELDS,
    build_lunar_tables,
    write_csv,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/lunarlander/data"
OUTPUT = ROOT / "analysis/tables/lunarlander"
CONTRACT = ROOT / "configs/minatar/protocol.yaml"

FIELDS = {
    "rq1_allocation.csv": PRIMARY_TABLE_FIELDS,
    "rq2_attribution.csv": PRIMARY_TABLE_FIELDS,
    "rq3_decomposition.csv": RQ3_FIELDS,
    "p3_classical_continuation.csv": P3_FIELDS,
    "replication_predictions.csv": LUNAR_PREDICTION_FIELDS,
}


def _build_tables() -> dict[str, list[dict[str, object]]]:
    contract = load_statistics_contract(CONTRACT)
    return build_lunar_tables(
        data_dir=DATA,
        contract=contract,
        seeds=tuple(range(20)),
    )


def _check_tables(tables: dict[str, list[dict[str, object]]]) -> None:
    with tempfile.TemporaryDirectory(prefix="paper-lunarlander-") as tmp:
        tmp_dir = Path(tmp)
        mismatches: list[str] = []
        for filename, rows in tables.items():
            generated = tmp_dir / filename
            write_csv(generated, rows, fieldnames=FIELDS[filename])
            tracked = OUTPUT / filename
            equivalent, detail = csv_equivalent(tracked, generated)
            if not equivalent:
                mismatches.append(filename)
                print(f"{filename}: {detail}")
        if mismatches:
            raise SystemExit(
                "LunarLander analysis tables are stale or missing: "
                + ", ".join(sorted(mismatches))
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate temporarily and fail on any substantive table difference.",
    )
    args = parser.parse_args()

    tables = _build_tables()
    if args.check:
        _check_tables(tables)
        print("LunarLander analysis tables are reproducible: PASS")
        return 0

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, rows in tables.items():
        write_csv(OUTPUT / filename, rows, fieldnames=FIELDS[filename])

    print("LunarLander analysis tables materialised under analysis/tables/lunarlander")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
