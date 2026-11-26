#!/usr/bin/env python3
"""Validate the symmetric public LunarLander and MinAtar evidence surfaces."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

RESULTS = Path("results")
ANALYSIS = Path("analysis/tables")
ENVIRONMENTS = ("lunarlander", "minatar")
SEEDS = tuple(range(20))
HANDOFF_CONDITIONS = (
    "C(0)", "H(0)", "C(0.25)", "H(0.25)", "C(0.5)", "H(0.5)",
    "C(0.75)", "H(0.75)", "H(1)",
)
ALL_CONDITIONS = ("full_classical", *HANDOFF_CONDITIONS)
ONLINE_CONDITIONS = (
    "full_classical", "C(0)", "H(0)", "C(0.25)", "H(0.25)",
    "C(0.5)", "H(0.5)", "C(0.75)", "H(0.75)",
)
KD_CONDITIONS = (
    "C(0.25)", "H(0.25)", "C(0.5)", "H(0.5)",
    "C(0.75)", "H(0.75)", "H(1)",
)
CANONICAL_DATA = (
    "evaluation_endpoints.csv",
    "evaluation_episodes.csv.gz",
    "full_classical_matrix.csv",
    "handoff_matrix.csv",
    "kd_diagnostics.csv",
    "quantum_forward_accounting.csv",
    "training_episode_returns.csv.gz",
    "training_update_metrics.csv.gz",
)
DERIVED_TABLES = (
    "rq1_allocation.csv",
    "rq2_attribution.csv",
    "rq3_decomposition.csv",
    "p3_classical_continuation.csv",
    "replication_predictions.csv",
)
EXPECTED_HANDOFF_CELLS = {
    (seed, condition) for seed in SEEDS for condition in HANDOFF_CONDITIONS
}
EXPECTED_KD_CELLS = {
    (seed, condition) for seed in SEEDS for condition in KD_CONDITIONS
}
EXPECTED_TRAINING_CELLS = {
    (seed, condition) for seed in SEEDS for condition in ONLINE_CONDITIONS
}
EXPECTED_ENDPOINT_CELLS = {
    (seed, condition, "final") for seed in SEEDS for condition in ALL_CONDITIONS
} | {
    (seed, condition, role)
    for seed in SEEDS
    for condition in KD_CONDITIONS
    for role in ("teacher", "post_kd")
}


class ValidationError(RuntimeError):
    pass


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValidationError(message)


def _load_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Missing JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"Expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    _require(path.is_file(), f"Missing CSV: {path}")
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_manifest(root: Path, environment: str) -> dict[str, Any]:
    manifest = _load_json(root / "manifest.json")
    _require(manifest.get("dataset_kind") == "paper_evidence", f"Unexpected dataset kind: {environment}")
    _require(manifest.get("environment") == environment, f"Manifest environment mismatch: {environment}")
    _require(manifest.get("status") == "COMPLETE", f"Incomplete evidence surface: {environment}")
    _require(manifest.get("active_seeds") == list(SEEDS), f"Seed-set drift: {environment}")
    _require(set(manifest.get("conditions", [])) == set(ALL_CONDITIONS), f"Condition-set drift: {environment}")
    contract = manifest.get("training_curve_contract", {})
    _require(contract.get("full_classical_is_shared_teacher_trajectory") is True, f"Teacher-trajectory contract drift: {environment}")
    _require(contract.get("staged_conditions_contain_post_handoff_training_only") is True, f"Staged-curve contract drift: {environment}")

    files = manifest.get("files")
    _require(isinstance(files, list), f"Manifest has no file inventory: {environment}")
    inventory = {str(item.get("path")): item for item in files if isinstance(item, dict)}
    expected_inventory = {"README.md", *{f"data/{name}" for name in CANONICAL_DATA}}
    _require(set(inventory) == expected_inventory, f"Public inventory drift: {environment}: {sorted(inventory)}")
    for relative, item in inventory.items():
        path = root / relative
        _require(path.is_file(), f"Manifest file missing: {path}")
        _require(_sha256(path) == str(item.get("sha256")), f"Manifest SHA mismatch: {path}")

    row_counts = manifest.get("row_counts", {})
    _require(set(row_counts) == set(CANONICAL_DATA), f"Row-count inventory drift: {environment}")
    return manifest


def _validate_derived_tables(environment: str, handoff: list[dict[str, str]]) -> None:
    final_by_condition: dict[str, dict[int, float]] = {}
    for row in handoff:
        final_by_condition.setdefault(row["condition"], {})[int(row["seed"])] = float(row["final_return_mean"])

    root = ANALYSIS / environment
    for filename in DERIVED_TABLES:
        _require((root / filename).is_file(), f"Missing derived table: {root / filename}")

    for filename in ("rq1_allocation.csv", "rq2_attribution.csv"):
        table = _read_csv(root / filename)
        _require(len(table) == 4, f"{environment} {filename}: expected four contrasts")
        for row in table:
            left, right = row["left"], row["right"]
            _require(
                set(final_by_condition[left]) == set(SEEDS) == set(final_by_condition[right]),
                f"Coverage drift: {environment} {left}-{right}",
            )
            paired = [final_by_condition[left][seed] - final_by_condition[right][seed] for seed in SEEDS]
            _require(
                math.isclose(
                    sum(paired) / len(paired),
                    float(row["paired_mean_difference"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"Paired mean drift: {environment} {left}-{right}",
            )
            _require(int(row["n"]) == len(SEEDS), f"n drift: {environment} {left}-{right}")
            _require(row.get("iqm_difference", "") != "", f"Missing IQM: {environment} {left}-{right}")
            _require(
                row.get("probability_of_improvement", "") != "",
                f"Missing probability of improvement: {environment} {left}-{right}",
            )

    _require(len(_read_csv(root / "rq3_decomposition.csv")) == 3, f"{environment} RQ3 row count drifted")
    _require(len(_read_csv(root / "p3_classical_continuation.csv")) == 3, f"{environment} P3 row count drifted")
    predictions = _read_csv(root / "replication_predictions.csv")
    _require(
        {row["prediction"] for row in predictions} == {"P1", "P2", "P3"},
        f"{environment} prediction table drifted",
    )


def _stream_training(path: Path, value_field: str) -> tuple[dict[tuple[int, str], int], int]:
    counts: dict[tuple[int, str], int] = {}
    rows = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["seed"]), row["condition"])
            _require(key in EXPECTED_TRAINING_CELLS, f"Unexpected training cell: {key}")
            if value_field == "episode_return":
                _require(math.isfinite(float(row[value_field])), f"Non-finite training return: {key}")
            else:
                _require(int(float(row[value_field])) >= 0, f"Invalid global step: {key}")
            counts[key] = counts.get(key, 0) + 1
            rows += 1
    return counts, rows


def validate_environment(environment: str) -> None:
    root = RESULTS / environment
    _require(root.is_dir(), f"Missing result root: {root}")
    manifest = _validate_manifest(root, environment)
    data = root / "data"

    full_classical = _read_csv(data / "full_classical_matrix.csv")
    _require(len(full_classical) == len(SEEDS), f"{environment}: full-classical row count drifted")
    _require(
        {int(row["seed"]) for row in full_classical} == set(SEEDS),
        f"{environment}: full-classical seed coverage drifted",
    )

    handoff = _read_csv(data / "handoff_matrix.csv")
    observed_handoff = {(int(row["seed"]), row["condition"]) for row in handoff}
    _require(observed_handoff == EXPECTED_HANDOFF_CELLS, f"{environment}: handoff coverage drifted")
    _require(all(row.get("status") in ("", "PASS") for row in handoff), f"{environment}: non-PASS handoff row")

    accounting = _read_csv(data / "quantum_forward_accounting.csv")
    _require(
        {(int(row["seed"]), row["condition"]) for row in accounting} == EXPECTED_HANDOFF_CELLS,
        f"{environment}: accounting coverage drifted",
    )
    for row in accounting:
        total = int(row["combined_total"])
        grad = int(row["combined_grad"])
        nograd = int(row["combined_nograd"])
        _require(
            total == grad + nograd and row["identity_status"] == "PASS",
            f"{environment}: QNode identity failed: {row['condition']} seed{row['seed']}",
        )
        if row["family"] == "compact_classical":
            _require(total == 0, f"{environment}: classical QNode calls: {row['condition']} seed{row['seed']}")
        else:
            _require(total > 0, f"{environment}: hybrid condition has no QNode calls: {row['condition']} seed{row['seed']}")

    kd = _read_csv(data / "kd_diagnostics.csv")
    _require(
        {(int(row["seed"]), row["condition"]) for row in kd} == EXPECTED_KD_CELLS,
        f"{environment}: KD diagnostic coverage drifted",
    )

    endpoints = _read_csv(data / "evaluation_endpoints.csv")
    observed_endpoints = {
        (int(row["seed"]), row["condition"], row["endpoint_role"])
        for row in endpoints
    }
    _require(observed_endpoints == EXPECTED_ENDPOINT_CELLS, f"{environment}: evaluation endpoint coverage drifted")

    episode_counts: dict[tuple[int, str, str], int] = {}
    evaluation_rows = 0
    with gzip.open(data / "evaluation_episodes.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["seed"]), row["condition"], row["endpoint_role"])
            _require(key in EXPECTED_ENDPOINT_CELLS, f"{environment}: unexpected evaluation cell: {key}")
            _require(math.isfinite(float(row["episode_return"])), f"{environment}: non-finite evaluation return: {key}")
            episode_counts[key] = episode_counts.get(key, 0) + 1
            evaluation_rows += 1
    _require(set(episode_counts) == EXPECTED_ENDPOINT_CELLS, f"{environment}: evaluation episode coverage drifted")
    _require(set(episode_counts.values()) == {100}, f"{environment}: expected 100 evaluation episodes per endpoint")

    episode_curve_counts, training_episode_rows = _stream_training(
        data / "training_episode_returns.csv.gz", "episode_return"
    )
    update_curve_counts, training_update_rows = _stream_training(
        data / "training_update_metrics.csv.gz", "global_step"
    )
    _require(set(episode_curve_counts) == EXPECTED_TRAINING_CELLS, f"{environment}: training episode coverage drifted")
    _require(set(update_curve_counts) == EXPECTED_TRAINING_CELLS, f"{environment}: training update coverage drifted")

    observed_counts = {
        "evaluation_endpoints.csv": len(endpoints),
        "evaluation_episodes.csv.gz": evaluation_rows,
        "full_classical_matrix.csv": len(full_classical),
        "handoff_matrix.csv": len(handoff),
        "kd_diagnostics.csv": len(kd),
        "quantum_forward_accounting.csv": len(accounting),
        "training_episode_returns.csv.gz": training_episode_rows,
        "training_update_metrics.csv.gz": training_update_rows,
    }
    _require(
        observed_counts == {key: int(value) for key, value in manifest["row_counts"].items()},
        f"{environment}: manifest row counts drifted",
    )
    _validate_derived_tables(environment, handoff)

    print(f"{environment}: public evidence OK")
    print(
        f"  8 canonical data files; {len(handoff)} handoff rows; "
        f"{len(endpoints)} endpoints; {evaluation_rows} evaluation episodes"
    )
    print(f"  training rows: {training_episode_rows} episodes / {training_update_rows} updates")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=("all", *ENVIRONMENTS), default="all")
    args = parser.parse_args()
    selected: Iterable[str] = ENVIRONMENTS if args.environment == "all" else (args.environment,)
    for environment in selected:
        validate_environment(environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
