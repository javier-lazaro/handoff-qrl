"""Strict CSV comparison with machine-precision tolerance for floating fields."""

from __future__ import annotations

import csv
import math
from pathlib import Path

FLOAT_ABS_TOL = 1e-12
FLOAT_REL_TOL = 1e-12


def _numeric_equal(left: str, right: str) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except ValueError:
        return False
    return math.isclose(
        left_value,
        right_value,
        rel_tol=FLOAT_REL_TOL,
        abs_tol=FLOAT_ABS_TOL,
    )


def csv_equivalent(tracked: Path, generated: Path) -> tuple[bool, str | None]:
    """Compare CSVs strictly, allowing only sub-picounit floating round-off.

    Headers, row counts, column counts, and non-numeric cells must match exactly.
    Numeric cells may differ only within 1e-12 absolute/relative tolerance. This
    absorbs platform-dependent last-bit floating arithmetic while remaining far
    below any precision reported in the manuscript.
    """

    if not tracked.is_file():
        return False, "tracked file is missing"

    with tracked.open(newline="", encoding="utf-8") as left_handle, generated.open(
        newline="", encoding="utf-8"
    ) as right_handle:
        left_rows = list(csv.reader(left_handle))
        right_rows = list(csv.reader(right_handle))

    if not left_rows or not right_rows:
        if left_rows == right_rows:
            return True, None
        return False, "one CSV is empty"

    if left_rows[0] != right_rows[0]:
        return False, "header differs"
    if len(left_rows) != len(right_rows):
        return False, f"row count differs: {len(left_rows) - 1} != {len(right_rows) - 1}"

    for row_index, (left_row, right_row) in enumerate(zip(left_rows[1:], right_rows[1:]), start=2):
        if len(left_row) != len(right_row):
            return False, f"column count differs on row {row_index}"
        for column_index, (left, right) in enumerate(zip(left_row, right_row), start=1):
            if left == right or _numeric_equal(left, right):
                continue
            column = left_rows[0][column_index - 1]
            return False, f"row {row_index}, column {column!r}: {left!r} != {right!r}"

    return True, None
