from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from handoff_qrl.experiments import CONDITIONS, default_output_dir, normalise_environment, run_experiment

ROOT = Path(__file__).resolve().parents[1]


def test_public_environment_names() -> None:
    assert normalise_environment("minatar") == "minatar"
    assert normalise_environment("MinAtar-Breakout") == "minatar"
    assert normalise_environment("lunarlander") == "lunarlander"
    assert normalise_environment("LunarLander-v3") == "lunarlander"


def test_public_condition_matrix() -> None:
    assert CONDITIONS == (
        "full_classical",
        "H(0)",
        "C(0)",
        "H(0.25)",
        "C(0.25)",
        "H(0.5)",
        "C(0.5)",
        "H(0.75)",
        "C(0.75)",
        "H(1)",
    )


def test_default_output_is_environment_condition_seed_scoped() -> None:
    assert str(default_output_dir("minatar", "H(0.5)", 7)).replace("\\", "/") == (
        "runs/reproduction/minatar/h_0p5/seed7"
    )


def test_paired_classical_control_requires_hybrid_source() -> None:
    with pytest.raises(ValueError, match="requires --paired-source"):
        run_experiment(
            environment="minatar",
            condition="C(0.5)",
            seed=0,
            output_dir="unused",
        )


def test_lunar_protocol_records_final_dataset_permutation_and_pairing() -> None:
    protocol = yaml.safe_load((ROOT / "configs/lunarlander/protocol.yaml").read_text(encoding="utf-8"))
    dataset = protocol["kd"]["dataset"]
    assert dataset["permutation_seed_base"] == 51000
    assert dataset["full_pool_permuted_before_kd"] is True
    assert dataset["paired_hybrid_classical_dataset_bytes_required"] is True
    assert protocol["pairing"]["H(1)"]["reuses_same_seed_full_classical_final_teacher"] is True
