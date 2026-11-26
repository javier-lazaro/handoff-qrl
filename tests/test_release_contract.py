from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_manuscript_regression_manifest_is_complete() -> None:
    expected = json.loads(
        (ROOT / "tests/paper_expected_sha256.json").read_text(encoding="utf-8")
    )
    required = {
        "numbers.tex",
        "tables/tab_conditions.tex",
        "tables/tab_families.tex",
        "tables/tab_references.tex",
        *{f"figures/Fig{i}.pdf" for i in range(1, 12)},
    }
    assert set(expected) == required
    assert all(len(value) == 64 for value in expected.values())


def test_public_sources_and_protocols_are_present() -> None:
    assert (ROOT / "results/lunarlander/manifest.json").is_file()
    assert (ROOT / "results/minatar/manifest.json").is_file()
    assert (ROOT / "configs/minatar/protocol.yaml").is_file()
    assert (ROOT / "configs/lunarlander/protocol.yaml").is_file()
    assert (ROOT / "environment/requirements-cpu.txt").is_file()
    assert (ROOT / "environment/requirements-analysis.txt").is_file()


def test_development_lineage_artifacts_are_absent() -> None:
    for relative in (
        "environment/original-pip-freeze.txt",
        "analysis/training_environment.json",
        "RELEASE_SANITIZATION.json",
        "configs/lunarlander/development_freeze.yaml",
        "results/lunarlander/provenance",
        "results/minatar/provenance",
    ):
        assert not (ROOT / relative).exists(), relative


def test_manifests_inventory_only_distributed_public_files() -> None:
    for env in ("lunarlander", "minatar"):
        root = ROOT / "results" / env
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        paths = [str(item["path"]) for item in manifest["files"]]
        assert all(not path.startswith("provenance/") for path in paths)
        assert all((root / path).is_file() for path in paths)


def test_results_share_one_public_schema() -> None:
    expected_data = {
        "evaluation_endpoints.csv",
        "evaluation_episodes.csv.gz",
        "full_classical_matrix.csv",
        "handoff_matrix.csv",
        "kd_diagnostics.csv",
        "quantum_forward_accounting.csv",
        "training_episode_returns.csv.gz",
        "training_update_metrics.csv.gz",
    }
    expected_analysis = {
        "rq1_allocation.csv",
        "rq2_attribution.csv",
        "rq3_decomposition.csv",
        "p3_classical_continuation.csv",
        "replication_predictions.csv",
    }
    for env in ("lunarlander", "minatar"):
        assert {p.name for p in (ROOT / "results" / env / "data").iterdir()} == expected_data
        assert {p.name for p in (ROOT / "analysis/tables" / env).iterdir()} == expected_analysis
