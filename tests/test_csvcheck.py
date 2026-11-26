from pathlib import Path

from handoff_qrl.csvcheck import csv_equivalent


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_csv_equivalent_accepts_last_bit_float_roundoff(tmp_path: Path) -> None:
    tracked = _write(tmp_path / "tracked.csv", "label,value\nA,25.960570349824152\n")
    generated = _write(tmp_path / "generated.csv", "label,value\nA,25.960570349824156\n")
    assert csv_equivalent(tracked, generated) == (True, None)


def test_csv_equivalent_rejects_substantive_numeric_change(tmp_path: Path) -> None:
    tracked = _write(tmp_path / "tracked.csv", "label,value\nA,25.960570349824152\n")
    generated = _write(tmp_path / "generated.csv", "label,value\nA,25.9605703499\n")
    equivalent, detail = csv_equivalent(tracked, generated)
    assert not equivalent
    assert "value" in str(detail)


def test_csv_equivalent_requires_exact_non_numeric_cells(tmp_path: Path) -> None:
    tracked = _write(tmp_path / "tracked.csv", "label,value\nA,1.0\n")
    generated = _write(tmp_path / "generated.csv", "label,value\nB,1.0\n")
    equivalent, detail = csv_equivalent(tracked, generated)
    assert not equivalent
    assert "label" in str(detail)
