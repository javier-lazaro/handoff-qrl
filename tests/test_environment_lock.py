from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _requirements(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_cpu_lock_is_public_and_curated() -> None:
    cpu = _requirements(ROOT / "environment/requirements-cpu.txt")
    assert cpu == [
        "numpy==2.5.0",
        "pandas==3.0.3",
        "PyYAML==6.0.3",
        "gymnasium==1.3.0",
        "Box2D==2.3.10",
        "pygame==2.6.1",
        "MinAtar==1.0.15",
        "pennylane==0.45.1",
        "pennylane_lightning==0.45.0",
        "cloudpickle==3.1.2",
        "tqdm==4.68.3",
        "psutil==7.2.2",
        "codecarbon==3.2.8",
        "tensorboard==2.21.0",
    ]
    lowered = "\n".join(cpu).lower()
    assert "git+" not in lowered
    assert "git@" not in lowered
    assert "-e " not in lowered
    assert not any(line.startswith(("/", "file:")) for line in cpu)


def test_no_developer_environment_snapshot_is_distributed() -> None:
    assert not (ROOT / "environment/original-pip-freeze.txt").exists()
    assert not (ROOT / "analysis/training_environment.json").exists()


def test_analysis_lock_matches_frozen_figure_environment() -> None:
    assert _requirements(ROOT / "environment/requirements-analysis.txt") == [
        "numpy==2.5.0",
        "pandas==3.0.2",
        "matplotlib==3.10.8",
        "PyYAML==6.0.3",
        "contourpy==1.4.0",
        "cycler==0.12.1",
        "fonttools==4.68.0",
        "kiwisolver==1.5.1",
        "packaging==26.3",
        "pillow==13.0.0",
        "pyparsing==3.3.3",
        "python-dateutil==3.0.0",
        "six==1.17.0",
        "tzdata==2026.6",
    ]
