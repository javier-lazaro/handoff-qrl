"""Paper-facing paper table builders over canonical evidence surfaces.

This module is deliberately model-free: it consumes canonical CSV evidence and
materialises the preregistered paper statistical tables.  It never trains or
evaluates an agent.  Both environments converge on these same builders: MinAtar is prospective and
LunarLander is retrospective under the same frozen statistical contract.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from hybrid_training_qrl.paper_stats import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    ContrastSpec,
    PairedContrastResult,
    PaperStatisticsContract,
    PaperStatisticsError,
    evaluate_holm_family,
    exact_sign_flip_pvalue,
    percentile_bootstrap_ci,
)

INTERIOR_HANDOFFS = (0.25, 0.5, 0.75)
PRIMARY_TABLE_FIELDS = (
    "family",
    "contrast",
    "left",
    "right",
    "n",
    "paired_mean_difference",
    "paired_std_difference",
    "paired_median_difference",
    "ci95_low",
    "ci95_high",
    "wins_left",
    "ties",
    "sign_flip_p",
    "holm_p",
    "holm_reject_0_05",
    "iqm_difference",
    "iqm_ci95_low",
    "iqm_ci95_high",
    "probability_of_improvement",
)
RQ3_FIELDS = (
    "handoff_t",
    "n",
    "teacher_return_mean",
    "teacher_return_ci95_low",
    "teacher_return_ci95_high",
    "hybrid_post_kd_mean",
    "hybrid_post_kd_ci95_low",
    "hybrid_post_kd_ci95_high",
    "classical_post_kd_mean",
    "classical_post_kd_ci95_low",
    "classical_post_kd_ci95_high",
    "hybrid_online_gain_mean",
    "hybrid_online_gain_ci95_low",
    "hybrid_online_gain_ci95_high",
    "classical_online_gain_mean",
    "classical_online_gain_ci95_low",
    "classical_online_gain_ci95_high",
    "post_kd_gap_h_minus_c_mean",
    "post_kd_gap_h_minus_c_ci95_low",
    "post_kd_gap_h_minus_c_ci95_high",
    "final_gap_h_minus_c_mean",
    "final_gap_h_minus_c_ci95_low",
    "final_gap_h_minus_c_ci95_high",
    "online_gain_interaction_I_mean",
    "online_gain_interaction_I_ci95_low",
    "online_gain_interaction_I_ci95_high",
)
P3_FIELDS = (
    "prediction",
    "handoff_t",
    "contrast",
    "n",
    "paired_mean_difference",
    "ci95_low",
    "ci95_high",
    "sign_flip_p_unadjusted",
    "wins_classical",
    "status",
)
PREDICTION_FIELDS = (
    "prediction",
    "definition",
    "minatar_status",
    "return_condition_met",
    "resource_condition_met",
    "holm_condition_met",
    "note",
)
LUNAR_PREDICTION_FIELDS = (
    "prediction",
    "definition",
    "lunar_status",
    "return_condition_met",
    "resource_condition_met",
    "holm_condition_met",
    "note",
)


class PaperTableError(ValueError):
    """Raised when a canonical paper surface cannot support the requested table."""


def _canonical_condition(row: Mapping[str, str]) -> str:
    """Return the canonical public condition label."""

    return str(row.get("condition") or "").strip()



def _quantum_forward_total(row: Mapping[str, str]) -> int:
    """Read the sample-level quantum-forward total from either environment schema."""

    value = row.get("combined_total")
    if value in (None, ""):
        value = row.get("qforward_total")
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise PaperTableError(f"Invalid quantum-forward total: {value!r}") from exc


def read_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise PaperTableError(f"Missing CSV: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _condition_rows(
    handoff_rows: Sequence[Mapping[str, str]],
    condition: str,
    seeds: Sequence[int],
) -> list[Mapping[str, str]]:
    selected = sorted(
        (row for row in handoff_rows if _canonical_condition(row) == condition),
        key=lambda row: int(row["seed"]),
    )
    observed = [int(row["seed"]) for row in selected]
    if observed != list(seeds):
        raise PaperTableError(
            f"{condition}: expected exactly seeds {list(seeds)}, observed {observed}"
        )
    if any(row.get("status") not in (None, "", "PASS") for row in selected):
        raise PaperTableError(f"{condition}: canonical handoff surface contains non-PASS rows")
    return selected


def _final_returns_by_condition(
    handoff_rows: Sequence[Mapping[str, str]],
    conditions: Iterable[str],
    seeds: Sequence[int],
) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for condition in conditions:
        rows = _condition_rows(handoff_rows, condition, seeds)
        try:
            values[condition] = np.asarray([float(row["final_return_mean"]) for row in rows])
        except (KeyError, TypeError, ValueError) as exc:
            raise PaperTableError(f"{condition}: invalid final_return_mean") from exc
    return values


def _primary_row(family: str, result: PairedContrastResult) -> dict[str, Any]:
    return {
        "family": family,
        "contrast": result.contrast,
        "left": result.left,
        "right": result.right,
        "n": result.n,
        "paired_mean_difference": result.paired_mean_difference,
        "paired_std_difference": result.paired_std_difference,
        "paired_median_difference": result.paired_median_difference,
        "ci95_low": result.ci95_low,
        "ci95_high": result.ci95_high,
        "wins_left": result.wins_left,
        "ties": result.ties,
        "sign_flip_p": result.sign_flip_p,
        "holm_p": result.holm_p,
        "holm_reject_0_05": result.holm_reject_0_05,
        "iqm_difference": result.iqm_difference,
        "iqm_ci95_low": result.iqm_ci95_low,
        "iqm_ci95_high": result.iqm_ci95_high,
        "probability_of_improvement": result.probability_of_improvement,
    }


def build_primary_family(
    handoff_rows: Sequence[Mapping[str, str]],
    *,
    family: str,
    contrasts: Sequence[ContrastSpec],
    seeds: Sequence[int],
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    conditions = {spec.left for spec in contrasts} | {spec.right for spec in contrasts}
    values = _final_returns_by_condition(handoff_rows, conditions, seeds)
    try:
        results = evaluate_holm_family(
            values,
            contrasts,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
    except PaperStatisticsError as exc:
        raise PaperTableError(str(exc)) from exc
    return [_primary_row(family, result) for result in results]


def _mean_ci(
    values: np.ndarray,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[float, float, float]:
    low, high = percentile_bootstrap_ci(
        values,
        statistic="mean",
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    return float(np.mean(values)), low, high


def build_rq3_decomposition(
    handoff_rows: Sequence[Mapping[str, str]],
    *,
    seeds: Sequence[int],
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for handoff_index, handoff_t in enumerate(INTERIOR_HANDOFFS):
        label = f"{handoff_t:g}"
        hybrid = _condition_rows(handoff_rows, f"H({label})", seeds)
        classical = _condition_rows(handoff_rows, f"C({label})", seeds)

        try:
            teacher_h = np.asarray([float(row["teacher_return_mean"]) for row in hybrid])
            teacher_c = np.asarray([float(row["teacher_return_mean"]) for row in classical])
            h_post = np.asarray([float(row["post_kd_return_mean"]) for row in hybrid])
            c_post = np.asarray([float(row["post_kd_return_mean"]) for row in classical])
            h_final = np.asarray([float(row["final_return_mean"]) for row in hybrid])
            c_final = np.asarray([float(row["final_return_mean"]) for row in classical])
        except (KeyError, TypeError, ValueError) as exc:
            raise PaperTableError(f"RQ3 handoff t={handoff_t}: missing/invalid endpoint") from exc

        if not np.array_equal(teacher_h, teacher_c):
            raise PaperTableError(
                f"RQ3 handoff t={handoff_t}: H/C teacher endpoint is not exactly matched by seed"
            )

        h_online = h_final - h_post
        c_online = c_final - c_post
        metrics = (
            ("teacher_return", teacher_h),
            ("hybrid_post_kd", h_post),
            ("classical_post_kd", c_post),
            ("hybrid_online_gain", h_online),
            ("classical_online_gain", c_online),
            ("post_kd_gap_h_minus_c", h_post - c_post),
            ("final_gap_h_minus_c", h_final - c_final),
            ("online_gain_interaction_I", h_online - c_online),
        )
        row: dict[str, Any] = {"handoff_t": handoff_t, "n": len(seeds)}
        for metric_index, (name, values) in enumerate(metrics):
            mean, low, high = _mean_ci(
                values,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + 2000 + handoff_index * 100 + metric_index,
            )
            row[f"{name}_mean"] = mean
            row[f"{name}_ci95_low"] = low
            row[f"{name}_ci95_high"] = high
        rows.append(row)
    return rows


def _full_classical_returns(
    rows: Sequence[Mapping[str, str]],
    seeds: Sequence[int],
) -> np.ndarray:
    selected = sorted(rows, key=lambda row: int(row["seed"]))
    observed = [int(row["seed"]) for row in selected]
    if observed != list(seeds):
        raise PaperTableError(
            f"Full classical: expected exactly seeds {list(seeds)}, observed {observed}"
        )
    if any(row.get("status") not in (None, "", "PASS") for row in selected):
        raise PaperTableError("Full classical surface contains non-PASS rows")
    try:
        return np.asarray([float(row["final_return_mean"]) for row in selected])
    except (KeyError, TypeError, ValueError) as exc:
        raise PaperTableError("Full classical: invalid final_return_mean") from exc


def build_p3_classical_continuation(
    handoff_rows: Sequence[Mapping[str, str]],
    full_classical_rows: Sequence[Mapping[str, str]],
    *,
    seeds: Sequence[int],
    role: str,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    if role not in {"retrospective", "prospective"}:
        raise PaperTableError(f"Unsupported P3 role: {role!r}")
    full_classical = _full_classical_returns(full_classical_rows, seeds)
    rows: list[dict[str, Any]] = []
    for index, handoff_t in enumerate(INTERIOR_HANDOFFS):
        label = f"{handoff_t:g}"
        hybrid_rows = _condition_rows(handoff_rows, f"H({label})", seeds)
        hybrid = np.asarray([float(row["final_return_mean"]) for row in hybrid_rows])
        differences = full_classical - hybrid
        low, high = percentile_bootstrap_ci(
            differences,
            statistic="mean",
            samples=bootstrap_samples,
            seed=bootstrap_seed + 3500 + index,
        )
        rows.append(
            {
                "prediction": "P3",
                "handoff_t": handoff_t,
                "contrast": "full classical continuation - hybrid switch",
                "n": len(seeds),
                "paired_mean_difference": float(np.mean(differences)),
                "ci95_low": low,
                "ci95_high": high,
                "sign_flip_p_unadjusted": exact_sign_flip_pvalue(differences),
                "wins_classical": int(np.count_nonzero(differences > 0.0)),
                "status": (
                    f"supported_{role}" if low > 0.0 else f"not_supported_{role}"
                ),
            }
        )
    return rows


def _qforward_cost_by_condition(
    accounting_rows: Sequence[Mapping[str, str]],
    *,
    seeds: Sequence[int],
) -> dict[str, int]:
    costs: dict[str, int] = {}
    hybrid_conditions = ("H(0)", "H(0.25)", "H(0.5)", "H(0.75)", "H(1)")
    for condition in hybrid_conditions:
        selected = sorted(
            (row for row in accounting_rows if _canonical_condition(row) == condition),
            key=lambda row: int(row["seed"]),
        )
        observed = [int(row["seed"]) for row in selected]
        if observed != list(seeds):
            raise PaperTableError(
                f"{condition}: quantum accounting does not cover the canonical seed set"
            )
        totals = {_quantum_forward_total(row) for row in selected}
        if len(totals) != 1:
            raise PaperTableError(
                f"{condition}: quantum-forward cost must be schedule-fixed, "
                f"observed {sorted(totals)}"
            )
        costs[condition] = totals.pop()
    return costs


def build_minatar_replication_predictions(
    rq1_rows: Sequence[Mapping[str, Any]],
    p3_rows: Sequence[Mapping[str, Any]],
    accounting_rows: Sequence[Mapping[str, str]],
    *,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    by_left = {str(row["left"]): row for row in rq1_rows}
    qforward = _qforward_cost_by_condition(accounting_rows, seeds=seeds)

    p1 = by_left["H(1)"]
    p1_return = float(p1["paired_mean_difference"]) > 0.0
    p1_resource = qforward["H(1)"] < qforward["H(0)"]
    p1_holm = bool(p1["holm_reject_0_05"])
    p1_confirmed = p1_return and p1_resource and p1_holm

    interior: list[dict[str, Any]] = []
    for label in ("H(0.25)", "H(0.5)", "H(0.75)"):
        row = by_left[label]
        item = {
            "label": label,
            "positive": float(row["paired_mean_difference"]) > 0.0,
            "cheaper": qforward[label] < qforward["H(0)"],
            "holm": bool(row["holm_reject_0_05"]),
        }
        interior.append(item)
    p2_directional = any(item["positive"] and item["cheaper"] for item in interior)
    p2_confirmed = any(
        item["positive"] and item["cheaper"] and item["holm"] for item in interior
    )

    p3_return = len(p3_rows) == 3 and all(
        float(row["paired_mean_difference"]) > 0.0 for row in p3_rows
    )
    p3_supported = len(p3_rows) == 3 and all(
        str(row["status"]) == "supported_prospective" for row in p3_rows
    )

    return [
        {
            "prediction": "P1",
            "definition": "H(1) > H(0) in final return with lower quantum-forward cost",
            "minatar_status": (
                "confirmed_prospective" if p1_confirmed else "not_supported_prospective"
            ),
            "return_condition_met": p1_return,
            "resource_condition_met": p1_resource,
            "holm_condition_met": p1_holm,
            "note": "Prospective replication of the retrospective Lunar result.",
        },
        {
            "prediction": "P2",
            "definition": "At least one interior H(t) Pareto-dominates H(0)",
            "minatar_status": (
                "confirmed_prospective" if p2_confirmed else "not_supported_prospective"
            ),
            "return_condition_met": p2_directional,
            "resource_condition_met": any(item["cheaper"] for item in interior),
            "holm_condition_met": p2_confirmed,
            "note": json.dumps(interior, sort_keys=True),
        },
        {
            "prediction": "P3",
            "definition": (
                "Full-capacity classical continuation beats switching to H(t) "
                "at each interior handoff"
            ),
            "minatar_status": (
                "supported_prospective" if p3_supported else "not_supported_prospective"
            ),
            "return_condition_met": p3_return,
            "resource_condition_met": None,
            "holm_condition_met": None,
            "note": "Secondary prediction; no third Holm family is created.",
        },
    ]



def build_lunar_replication_predictions(
    rq1_rows: Sequence[Mapping[str, Any]],
    p3_rows: Sequence[Mapping[str, Any]],
    accounting_rows: Sequence[Mapping[str, str]],
    *,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """Materialise the retrospective Lunar P1/P2/P3 status without new tests."""

    by_left = {str(row["left"]): row for row in rq1_rows}
    qforward = _qforward_cost_by_condition(accounting_rows, seeds=seeds)

    p1 = by_left["H(1)"]
    p1_return = float(p1["paired_mean_difference"]) > 0.0
    p1_resource = qforward["H(1)"] < qforward["H(0)"]
    p1_holm = bool(p1["holm_reject_0_05"])
    p1_confirmed = p1_return and p1_resource and p1_holm

    interior: list[dict[str, Any]] = []
    for label in ("H(0.25)", "H(0.5)", "H(0.75)"):
        row = by_left[label]
        interior.append(
            {
                "label": label,
                "positive": float(row["paired_mean_difference"]) > 0.0,
                "cheaper": qforward[label] < qforward["H(0)"],
                "holm": bool(row["holm_reject_0_05"]),
            }
        )
    p2_directional = any(item["positive"] and item["cheaper"] for item in interior)
    p2_confirmed = any(
        item["positive"] and item["cheaper"] and item["holm"] for item in interior
    )

    p3_return = len(p3_rows) == 3 and all(
        float(row["paired_mean_difference"]) > 0.0 for row in p3_rows
    )
    p3_supported = len(p3_rows) == 3 and all(
        str(row["status"]) == "supported_retrospective" for row in p3_rows
    )

    return [
        {
            "prediction": "P1",
            "definition": "H(1) > H(0) in final return with lower quantum-forward cost",
            "lunar_status": (
                "confirmed_retrospective" if p1_confirmed else "not_confirmed_retrospective"
            ),
            "return_condition_met": p1_return,
            "resource_condition_met": p1_resource,
            "holm_condition_met": p1_holm,
            "note": "Retrospective Lunar result; MinAtar is the prospective replication.",
        },
        {
            "prediction": "P2",
            "definition": "At least one interior H(t) Pareto-dominates H(0)",
            "lunar_status": (
                "confirmed_retrospective"
                if p2_confirmed
                else "directionally_consistent_not_confirmatory"
                if p2_directional
                else "not_supported_retrospective"
            ),
            "return_condition_met": p2_directional,
            "resource_condition_met": p2_directional,
            "holm_condition_met": p2_confirmed,
            "note": json.dumps(interior, sort_keys=True),
        },
        {
            "prediction": "P3",
            "definition": (
                "Full-capacity classical continuation beats switching to H(t) "
                "at each interior handoff"
            ),
            "lunar_status": (
                "supported_retrospective" if p3_supported else "not_fully_supported_retrospective"
            ),
            "return_condition_met": p3_return,
            "resource_condition_met": None,
            "holm_condition_met": None,
            "note": "Secondary prediction; no third Holm family is created.",
        },
    ]


def build_lunar_tables(
    *,
    data_dir: str | Path,
    contract: PaperStatisticsContract,
    seeds: Sequence[int],
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, list[dict[str, Any]]]:
    """Regenerate LunarLander analysis from the canonical public evidence."""

    root = Path(data_dir)
    handoff = read_csv(root / "handoff_matrix.csv")
    full_classical = read_csv(root / "full_classical_matrix.csv")
    accounting = read_csv(root / "quantum_forward_accounting.csv")

    rq1 = build_primary_family(
        handoff,
        family="allocation",
        contrasts=contract.holm_families["allocation"],
        seeds=seeds,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    rq2 = build_primary_family(
        handoff,
        family="attribution",
        contrasts=contract.holm_families["attribution"],
        seeds=seeds,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    rq3 = build_rq3_decomposition(
        handoff,
        seeds=seeds,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    p3 = build_p3_classical_continuation(
        handoff,
        full_classical,
        seeds=seeds,
        role="retrospective",
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    predictions = build_lunar_replication_predictions(
        rq1,
        p3,
        accounting,
        seeds=seeds,
    )
    return {
        "rq1_allocation.csv": rq1,
        "rq2_attribution.csv": rq2,
        "rq3_decomposition.csv": rq3,
        "p3_classical_continuation.csv": p3,
        "replication_predictions.csv": predictions,
    }


def build_minatar_tables(
    *,
    data_dir: str | Path,
    contract: PaperStatisticsContract,
    seeds: Sequence[int],
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, list[dict[str, Any]]]:
    root = Path(data_dir)
    handoff = read_csv(root / "handoff_matrix.csv")
    full_classical = read_csv(root / "full_classical_matrix.csv")
    accounting = read_csv(root / "quantum_forward_accounting.csv")

    rq1 = build_primary_family(
        handoff,
        family="allocation",
        contrasts=contract.holm_families["allocation"],
        seeds=seeds,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    rq2 = build_primary_family(
        handoff,
        family="attribution",
        contrasts=contract.holm_families["attribution"],
        seeds=seeds,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    rq3 = build_rq3_decomposition(
        handoff,
        seeds=seeds,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    p3 = build_p3_classical_continuation(
        handoff,
        full_classical,
        seeds=seeds,
        role="prospective",
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    predictions = build_minatar_replication_predictions(
        rq1,
        p3,
        accounting,
        seeds=seeds,
    )
    return {
        "rq1_allocation.csv": rq1,
        "rq2_attribution.csv": rq2,
        "rq3_decomposition.csv": rq3,
        "p3_classical_continuation.csv": p3,
        "replication_predictions.csv": predictions,
    }


def update_manifest_inventory(manifest_path: str | Path, relative_paths: Sequence[str]) -> None:
    path = Path(manifest_path)
    root = path.parent
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperTableError(f"Cannot read manifest: {path}") from exc
    files = manifest.get("files")
    if not isinstance(files, list):
        raise PaperTableError(f"Manifest has no files inventory: {path}")

    by_path: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in files:
        if not isinstance(item, dict) or not item.get("path"):
            raise PaperTableError(f"Malformed manifest file entry: {item!r}")
        relative = str(item["path"])
        by_path[relative] = dict(item)
        order.append(relative)

    for relative in relative_paths:
        file_path = root / relative
        if not file_path.is_file():
            raise PaperTableError(f"Cannot inventory missing generated table: {file_path}")
        by_path[relative] = {
            "bytes": file_path.stat().st_size,
            "path": relative,
            "sha256": sha256(file_path),
        }
        if relative not in order:
            # Keep data tables before provenance entries for readability.
            insertion = next(
                (index for index, current in enumerate(order) if current.startswith("provenance/")),
                len(order),
            )
            order.insert(insertion, relative)

    manifest["files"] = [by_path[relative] for relative in order]
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
