"""Cross-environment paper analysis tables derived from canonical paper evidence.

The per-environment ``results/*`` trees remain the scientific source of
truth.  This module only produces deterministic paper-facing summaries from
those frozen surfaces; it never reads raw ``runs/`` artifacts and never trains
or evaluates an agent.

Raw reward magnitudes are intentionally kept environment-scoped.  The only
quantities intended for side-by-side cross-environment reading are explicitly
normalised within each environment (e.g. return change relative to H(0), or
quantum-forward savings relative to H(0)).  Timing/energy summaries are limited
to the MinAtar seeds 0--4 serial resource cohort.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hybrid_training_qrl.paper_stats import (
    DEFAULT_BOOTSTRAP_SEED,
    interquartile_mean,
    paired_contrast,
)
from hybrid_training_qrl.paper_tables import PRIMARY_TABLE_FIELDS, read_csv

ENVIRONMENTS = ("lunarlander", "minatar")
HYBRID_CONDITIONS = ("H(0)", "H(0.25)", "H(0.5)", "H(0.75)", "H(1)")
INTERIOR_HANDOFFS = (0.25, 0.5, 0.75)
EXPECTED_SEEDS = tuple(range(20))
SERIAL_RESOURCE_SEEDS = tuple(range(5))
FIXED_KD_QFORWARD_TOTAL = 3_391_488
KD_DATASET_ENV_INTERACTIONS = 65_536 + 8_192
TOTAL_ENV_INTERACTIONS = {"lunarlander": 1_048_576, "minatar": 5_001_216}

CONDITION_ORDER = (
    "full_classical",
    "H(0)",
    "H(0.25)",
    "H(0.5)",
    "H(0.75)",
    "H(1)",
    "C(0)",
    "C(0.25)",
    "C(0.5)",
    "C(0.75)",
)


CONDITION_SUMMARY_FIELDS = (
    "environment",
    "evidence_role",
    "family",
    "condition",
    "handoff_t",
    "n",
    "mean_final_return",
    "sd_final_return",
    "median_final_return",
    "iqm_final_return",
    "min_final_return",
    "max_final_return",
    "mean_qforward_total",
    "mean_qforward_grad",
    "mean_qforward_nograd",
    "total_env_interactions",
)
FC_REFERENCE_CONTRAST_FIELDS = (
    "environment",
    "family",
    "confirmatory",
    *PRIMARY_TABLE_FIELDS[1:],
    "retention_pct_of_teacher_mean",
    "pearson_r_same_seed",
)
CONTINUATION_REFERENCE_CONTRAST_FIELDS = (
    "environment",
    "family",
    "confirmatory",
    "handoff_t",
    "contrast",
    "right_head",
    "n",
    "paired_mean_difference",
    "ci95_low",
    "ci95_high",
    "sign_flip_p_unadjusted",
    "wins_classical",
    "status",
    "iqm_difference",
    "iqm_ci95_low",
    "iqm_ci95_high",
    "probability_of_improvement",
)

ALLOCATION_NORMALIZED_FIELDS = (
    "environment",
    "evidence_role",
    "condition",
    "handoff_t",
    "n",
    "mean_final_return",
    "paired_delta_vs_h0",
    "relative_delta_vs_h0_pct",
    "mean_qforward_total",
    "qforward_saved_vs_h0_pct",
    "holm_p_vs_h0",
    "holm_reject_0_05_vs_h0",
)
ENVIRONMENT_SUMMARY_FIELDS = (
    "environment",
    "evidence_role",
    "n",
    "full_classical_mean_return",
    "h0_mean_return",
    "h1_mean_return",
    "h1_retention_pct_of_full_classical",
    "h1_paired_delta_vs_h0",
    "h1_holm_p_vs_h0",
    "h1_holm_reject_0_05_vs_h0",
    "h1_qforward_saved_vs_h0_pct",
    "rq2_any_holm_reject_0_05",
)
RESOURCE_BREAKDOWN_FIELDS = (
    "environment",
    "condition",
    "handoff_t",
    "n",
    "mean_qforward_grad",
    "mean_qforward_nograd",
    "mean_qforward_total",
    "grad_share_pct",
    "nograd_share_pct",
    "mean_kd_qforward_total",
    "mean_online_qforward_total",
    "kd_share_pct",
    "kd_to_online_qforward_ratio",
)
SERIAL_TIMING_FIELDS = (
    "environment",
    "timing_scope",
    "family",
    "condition",
    "handoff_t",
    "n_serial_seeds",
    "kd_qnode_time_mean_seconds",
    "kd_qnode_time_std_seconds",
    "online_qnode_time_mean_seconds",
    "online_qnode_time_std_seconds",
    "qnode_time_mean_seconds",
    "qnode_time_std_seconds",
    "kd_wall_clock_mean_seconds",
    "kd_wall_clock_std_seconds",
    "online_wall_clock_mean_seconds",
    "online_wall_clock_std_seconds",
    "online_ppo_update_time_mean_seconds",
    "online_ppo_update_time_std_seconds",
    "combined_energy_mean_kwh",
    "combined_energy_std_kwh",
    "combined_emissions_mean_kg",
    "combined_emissions_std_kg",
)
KD_HC_DIAGNOSTIC_FIELDS = (
    "environment",
    "evidence_role",
    "handoff_t",
    "n",
    "hybrid_action_agreement_mean",
    "classical_action_agreement_mean",
    "h_minus_c_action_agreement",
    "hybrid_policy_kl_mean",
    "classical_policy_kl_mean",
    "h_minus_c_policy_kl",
    "post_kd_gap_h_minus_c_mean",
    "diagnostic_checkpoint",
)

TABLE_FIELDS = {
    "condition_summary.csv": CONDITION_SUMMARY_FIELDS,
    "fc_reference_contrasts.csv": FC_REFERENCE_CONTRAST_FIELDS,
    "continuation_reference_contrasts.csv": CONTINUATION_REFERENCE_CONTRAST_FIELDS,
    "allocation_normalized.csv": ALLOCATION_NORMALIZED_FIELDS,
    "environment_summary.csv": ENVIRONMENT_SUMMARY_FIELDS,
    "resource_breakdown.csv": RESOURCE_BREAKDOWN_FIELDS,
    "minatar_serial_timing.csv": SERIAL_TIMING_FIELDS,
    "kd_hc_diagnostics.csv": KD_HC_DIAGNOSTIC_FIELDS,
}


class PaperAnalysisTableError(ValueError):
    """Raised when canonical evidence cannot support a derived paper table."""


def _float(row: Mapping[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise PaperAnalysisTableError(f"Invalid {key}: {row.get(key)!r}") from exc
    if not np.isfinite(value):
        raise PaperAnalysisTableError(f"Non-finite {key}: {value!r}")
    return value


def _int(row: Mapping[str, str], key: str) -> int:
    try:
        return int(float(row[key]))
    except (KeyError, TypeError, ValueError) as exc:
        raise PaperAnalysisTableError(f"Invalid {key}: {row.get(key)!r}") from exc


def _mean(values: Sequence[float], *, label: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise PaperAnalysisTableError(f"{label}: expected non-empty finite values")
    return float(np.mean(array))


def _std(values: Sequence[float], *, label: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2 or not np.isfinite(array).all():
        raise PaperAnalysisTableError(f"{label}: expected at least two finite values")
    return float(np.std(array, ddof=1))


def _condition_label(environment: str, row: Mapping[str, str]) -> str:
    del environment
    return str(row.get("condition", "")).strip()



def _evidence_role(environment: str) -> str:
    if environment == "lunarlander":
        return "retrospective"
    if environment == "minatar":
        return "prospective_confirmatory"
    raise PaperAnalysisTableError(f"Unknown environment: {environment}")


def _canonical_hybrid_rows(
    result_root: Path,
    environment: str,
    *,
    seeds: Sequence[int],
) -> dict[str, list[dict[str, str]]]:
    rows = read_csv(result_root / "data/handoff_matrix.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("family") != "hybrid":
            continue
        grouped[_condition_label(environment, row)].append(row)
    if set(grouped) != set(HYBRID_CONDITIONS):
        raise PaperAnalysisTableError(
            f"{environment}: hybrid condition coverage drifted: {sorted(grouped)}"
        )
    expected_seeds = list(seeds)
    for condition in HYBRID_CONDITIONS:
        grouped[condition].sort(key=lambda row: _int(row, "seed"))
        observed = [_int(row, "seed") for row in grouped[condition]]
        if observed != expected_seeds:
            raise PaperAnalysisTableError(
                f"{environment} {condition}: expected seeds {expected_seeds}, observed {observed}"
            )
    return grouped


def _analysis_table(result_root: Path, filename: str) -> list[dict[str, str]]:
    path = result_root.parents[1] / "analysis/tables" / result_root.name / filename
    return read_csv(path)


def _rq_rows(result_root: Path, filename: str) -> list[dict[str, str]]:
    rows = _analysis_table(result_root, filename)
    if len(rows) != 4:
        raise PaperAnalysisTableError(f"{result_root.name} {filename}: expected four rows")
    return rows



def _full_classical_mean(result_root: Path, environment: str, seeds: Sequence[int]) -> float:
    rows = read_csv(result_root / "data/full_classical_matrix.csv")
    selected = sorted(rows, key=lambda row: _int(row, "seed"))
    observed = [_int(row, "seed") for row in selected]
    if observed != list(seeds):
        raise PaperAnalysisTableError(
            f"{environment}: full-classical seed coverage drifted: {observed}"
        )
    return _mean([_float(row, "final_return_mean") for row in selected], label="full classical")



def build_allocation_normalized(
    paper_root: str | Path,
    *,
    seeds: Sequence[int] = EXPECTED_SEEDS,
) -> list[dict[str, Any]]:
    """Build within-environment allocation/resource normalisation for H(t)."""

    paper_root = Path(paper_root)
    output: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        root = paper_root / environment
        grouped = _canonical_hybrid_rows(root, environment, seeds=seeds)
        means = {
            condition: _mean(
                [_float(row, "final_return_mean") for row in grouped[condition]],
                label=f"{environment} {condition} return",
            )
            for condition in HYBRID_CONDITIONS
        }
        qmeans = {
            condition: _mean(
                [_float(row, "qforward_total") for row in grouped[condition]],
                label=f"{environment} {condition} qforward",
            )
            for condition in HYBRID_CONDITIONS
        }
        h0_return = means["H(0)"]
        h0_qforward = qmeans["H(0)"]
        if h0_return == 0.0 or h0_qforward <= 0.0:
            raise PaperAnalysisTableError(f"{environment}: invalid H(0) normalisation denominator")
        rq1 = {row["left"]: row for row in _rq_rows(root, "rq1_allocation.csv")}
        for condition in HYBRID_CONDITIONS:
            if condition == "H(0)":
                delta = 0.0
                holm_p: float | str = ""
                reject: bool | str = ""
            else:
                if condition not in rq1 or rq1[condition].get("right") != "H(0)":
                    raise PaperAnalysisTableError(
                        f"{environment}: RQ1 is missing canonical contrast {condition}-H(0)"
                    )
                delta = _float(rq1[condition], "paired_mean_difference")
                holm_p = _float(rq1[condition], "holm_p")
                reject = rq1[condition].get("holm_reject_0_05") == "True"
            row0 = grouped[condition][0]
            output.append(
                {
                    "environment": environment,
                    "evidence_role": _evidence_role(environment),
                    "condition": condition,
                    "handoff_t": _float(row0, "handoff_t"),
                    "n": len(seeds),
                    "mean_final_return": means[condition],
                    "paired_delta_vs_h0": delta,
                    "relative_delta_vs_h0_pct": 100.0 * delta / abs(h0_return),
                    "mean_qforward_total": qmeans[condition],
                    "qforward_saved_vs_h0_pct": 100.0
                    * (1.0 - qmeans[condition] / h0_qforward),
                    "holm_p_vs_h0": holm_p,
                    "holm_reject_0_05_vs_h0": reject,
                }
            )
    return output


def build_environment_summary(
    paper_root: str | Path,
    *,
    seeds: Sequence[int] = EXPECTED_SEEDS,
) -> list[dict[str, Any]]:
    paper_root = Path(paper_root)
    allocation = build_allocation_normalized(paper_root, seeds=seeds)
    by_environment = {
        environment: {row["condition"]: row for row in allocation if row["environment"] == environment}
        for environment in ENVIRONMENTS
    }
    output: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        root = paper_root / environment
        full_classical = _full_classical_mean(root, environment, seeds)
        h0 = by_environment[environment]["H(0)"]
        h1 = by_environment[environment]["H(1)"]
        rq2 = _rq_rows(root, "rq2_attribution.csv")
        output.append(
            {
                "environment": environment,
                "evidence_role": _evidence_role(environment),
                "n": len(seeds),
                "full_classical_mean_return": full_classical,
                "h0_mean_return": h0["mean_final_return"],
                "h1_mean_return": h1["mean_final_return"],
                "h1_retention_pct_of_full_classical": 100.0
                * float(h1["mean_final_return"])
                / full_classical,
                "h1_paired_delta_vs_h0": h1["paired_delta_vs_h0"],
                "h1_holm_p_vs_h0": h1["holm_p_vs_h0"],
                "h1_holm_reject_0_05_vs_h0": h1["holm_reject_0_05_vs_h0"],
                "h1_qforward_saved_vs_h0_pct": h1["qforward_saved_vs_h0_pct"],
                "rq2_any_holm_reject_0_05": any(
                    row.get("holm_reject_0_05") == "True" for row in rq2
                ),
            }
        )
    return output


def _resource_columns(environment: str) -> dict[str, str]:
    del environment
    return {
        "condition": "condition",
        "grad": "combined_grad",
        "nograd": "combined_nograd",
        "total": "combined_total",
        "kd_grad": "kd_grad",
        "kd_nograd": "kd_nograd",
        "online_grad": "online_grad",
        "online_nograd": "online_nograd",
    }



def build_resource_breakdown(
    paper_root: str | Path,
    *,
    seeds: Sequence[int] = EXPECTED_SEEDS,
) -> list[dict[str, Any]]:
    """Summarise sample-level quantum-forward composition for hybrid conditions."""

    paper_root = Path(paper_root)
    output: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        root = paper_root / environment
        cols = _resource_columns(environment)
        rows = read_csv(root / "data/quantum_forward_accounting.csv")
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            if environment == "minatar" and row.get("family") != "hybrid":
                continue
            condition = row.get(cols["condition"], "")
            if condition in HYBRID_CONDITIONS:
                grouped[condition].append(row)
        if set(grouped) != set(HYBRID_CONDITIONS):
            raise PaperAnalysisTableError(
                f"{environment}: QNode condition coverage drifted: {sorted(grouped)}"
            )
        for condition in HYBRID_CONDITIONS:
            selected = sorted(grouped[condition], key=lambda row: _int(row, "seed"))
            if [_int(row, "seed") for row in selected] != list(seeds):
                raise PaperAnalysisTableError(
                    f"{environment} {condition}: QNode seed coverage drifted"
                )
            grad = _mean([_float(row, cols["grad"]) for row in selected], label="grad")
            nograd = _mean([_float(row, cols["nograd"]) for row in selected], label="nograd")
            total = _mean([_float(row, cols["total"]) for row in selected], label="total")
            kd_total = _mean(
                [
                    _float(row, cols["kd_grad"]) + _float(row, cols["kd_nograd"])
                    for row in selected
                ],
                label="KD qforward",
            )
            online_total = _mean(
                [
                    _float(row, cols["online_grad"]) + _float(row, cols["online_nograd"])
                    for row in selected
                ],
                label="online qforward",
            )
            if not np.isclose(grad + nograd, total, rtol=0.0, atol=1e-9):
                raise PaperAnalysisTableError(f"{environment} {condition}: grad/nograd identity failed")
            if not np.isclose(kd_total + online_total, total, rtol=0.0, atol=1e-9):
                raise PaperAnalysisTableError(f"{environment} {condition}: KD/online identity failed")
            if condition != "H(0)" and not np.isclose(
                kd_total, FIXED_KD_QFORWARD_TOTAL, rtol=0.0, atol=0.0
            ):
                raise PaperAnalysisTableError(
                    f"{environment} {condition}: frozen KD cost drifted ({kd_total})"
                )
            output.append(
                {
                    "environment": environment,
                    "condition": condition,
                    "handoff_t": float(condition[2:-1]),
                    "n": len(seeds),
                    "mean_qforward_grad": grad,
                    "mean_qforward_nograd": nograd,
                    "mean_qforward_total": total,
                    "grad_share_pct": 100.0 * grad / total,
                    "nograd_share_pct": 100.0 * nograd / total,
                    "mean_kd_qforward_total": kd_total,
                    "mean_online_qforward_total": online_total,
                    "kd_share_pct": 100.0 * kd_total / total,
                    "kd_to_online_qforward_ratio": ""
                    if online_total == 0.0
                    else kd_total / online_total,
                }
            )
    return output


def _serial_stat(rows: Sequence[Mapping[str, str]], key: str) -> tuple[float, float]:
    values = [_float(row, key) for row in rows]
    return _mean(values, label=key), _std(values, label=key)


def build_minatar_serial_timing(
    paper_root: str | Path,
    *,
    serial_seeds: Sequence[int] = SERIAL_RESOURCE_SEEDS,
) -> list[dict[str, Any]]:
    """Summarise only the preregistered MinAtar serial timing/resource cohort."""

    root = Path(paper_root) / "minatar"
    rows = read_csv(root / "data/handoff_matrix.csv")
    selected = [row for row in rows if _int(row, "seed") in serial_seeds]
    expected_conditions = {
        "C(0)",
        "C(0.25)",
        "C(0.5)",
        "C(0.75)",
        *HYBRID_CONDITIONS,
    }
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        if row.get("timing_scope") != "paper_facing_serial":
            raise PaperAnalysisTableError(
                f"MinAtar seed{row.get('seed')} {row.get('condition')}: serial timing scope drifted"
            )
        grouped[(row["family"], row["condition"])].append(row)
    if {condition for _, condition in grouped} != expected_conditions:
        raise PaperAnalysisTableError("MinAtar serial timing condition coverage drifted")

    output: list[dict[str, Any]] = []
    for (family, condition), condition_rows in sorted(
        grouped.items(), key=lambda item: (float(item[0][1][2:-1]), item[0][0])
    ):
        condition_rows.sort(key=lambda row: _int(row, "seed"))
        observed = [_int(row, "seed") for row in condition_rows]
        if observed != list(serial_seeds):
            raise PaperAnalysisTableError(
                f"MinAtar {condition}: expected serial seeds {list(serial_seeds)}, observed {observed}"
            )
        values: dict[str, float] = {}
        for source, prefix in (
            ("kd_qnode_time_seconds", "kd_qnode_time"),
            ("online_qnode_time_seconds", "online_qnode_time"),
            ("qnode_time_seconds", "qnode_time"),
            ("kd_wall_clock", "kd_wall_clock"),
            ("online_wall_clock", "online_wall_clock"),
            ("online_ppo_update_time_seconds", "online_ppo_update_time"),
            ("combined_energy_kwh", "combined_energy"),
            ("combined_emissions_kg", "combined_emissions"),
        ):
            mean, std = _serial_stat(condition_rows, source)
            values[f"{prefix}_mean"] = mean
            values[f"{prefix}_std"] = std
        output.append(
            {
                "environment": "minatar",
                "timing_scope": "paper_facing_serial_seeds_0_4",
                "family": family,
                "condition": condition,
                "handoff_t": _float(condition_rows[0], "handoff_t"),
                "n_serial_seeds": len(serial_seeds),
                "kd_qnode_time_mean_seconds": values["kd_qnode_time_mean"],
                "kd_qnode_time_std_seconds": values["kd_qnode_time_std"],
                "online_qnode_time_mean_seconds": values["online_qnode_time_mean"],
                "online_qnode_time_std_seconds": values["online_qnode_time_std"],
                "qnode_time_mean_seconds": values["qnode_time_mean"],
                "qnode_time_std_seconds": values["qnode_time_std"],
                "kd_wall_clock_mean_seconds": values["kd_wall_clock_mean"],
                "kd_wall_clock_std_seconds": values["kd_wall_clock_std"],
                "online_wall_clock_mean_seconds": values["online_wall_clock_mean"],
                "online_wall_clock_std_seconds": values["online_wall_clock_std"],
                "online_ppo_update_time_mean_seconds": values["online_ppo_update_time_mean"],
                "online_ppo_update_time_std_seconds": values["online_ppo_update_time_std"],
                "combined_energy_mean_kwh": values["combined_energy_mean"],
                "combined_energy_std_kwh": values["combined_energy_std"],
                "combined_emissions_mean_kg": values["combined_emissions_mean"],
                "combined_emissions_std_kg": values["combined_emissions_std"],
            }
        )
    return output


def _lunar_kd_summary(result_root: Path, handoff_t: float) -> tuple[float, float, float, float, str]:
    label = f"{handoff_t:g}"
    rows = read_csv(result_root / "data/kd_diagnostics.csv")
    hybrid = [row for row in rows if row.get("family") == "hybrid" and row.get("condition") == f"H({label})"]
    classical = [row for row in rows if row.get("family") == "compact_classical" and row.get("condition") == f"C({label})"]
    if len(hybrid) != 20 or len(classical) != 20:
        raise PaperAnalysisTableError(f"Lunar t={handoff_t}: expected 20 H and 20 C KD diagnostics")
    return (
        _mean([_float(row, "validation_action_agreement") for row in hybrid], label="Lunar H agreement"),
        _mean([_float(row, "validation_action_agreement") for row in classical], label="Lunar C agreement"),
        _mean([_float(row, "validation_policy_kl") for row in hybrid], label="Lunar H KL"),
        _mean([_float(row, "validation_policy_kl") for row in classical], label="Lunar C KL"),
        "epoch_20",
    )



def _minatar_kd_summary(result_root: Path, handoff_t: float) -> tuple[float, float, float, float, str]:
    condition = f"{handoff_t:g}"
    hlabel = f"H({condition})"
    clabel = f"C({condition})"
    rows = read_csv(result_root / "data/kd_diagnostics.csv")
    hybrid = [row for row in rows if row.get("family") == "hybrid" and row.get("condition") == hlabel]
    classical = [
        row
        for row in rows
        if row.get("family") == "compact_classical" and row.get("condition") == clabel
    ]
    if len(hybrid) != 20 or len(classical) != 20:
        raise PaperAnalysisTableError(
            f"MinAtar t={handoff_t}: expected 20 H and 20 C KD diagnostics"
        )
    return (
        _mean([_float(row, "validation_action_agreement") for row in hybrid], label="MinAtar H agreement"),
        _mean([_float(row, "validation_action_agreement") for row in classical], label="MinAtar C agreement"),
        _mean([_float(row, "validation_policy_kl") for row in hybrid], label="MinAtar H KL"),
        _mean([_float(row, "validation_policy_kl") for row in classical], label="MinAtar C KL"),
        "validation_selected_best_epoch",
    )


def build_kd_hc_diagnostics(paper_root: str | Path) -> list[dict[str, Any]]:
    """Build common H/C KD diagnostics using metrics present in both environments."""

    paper_root = Path(paper_root)
    output: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        root = paper_root / environment
        rq3 = {float(row["handoff_t"]): row for row in _analysis_table(root, "rq3_decomposition.csv")}
        for handoff_t in INTERIOR_HANDOFFS:
            if environment == "lunarlander":
                h_agree, c_agree, h_kl, c_kl, checkpoint = _lunar_kd_summary(root, handoff_t)
            else:
                h_agree, c_agree, h_kl, c_kl, checkpoint = _minatar_kd_summary(root, handoff_t)
            if handoff_t not in rq3:
                raise PaperAnalysisTableError(f"{environment}: missing RQ3 t={handoff_t}")
            output.append(
                {
                    "environment": environment,
                    "evidence_role": _evidence_role(environment),
                    "handoff_t": handoff_t,
                    "n": 20,
                    "hybrid_action_agreement_mean": h_agree,
                    "classical_action_agreement_mean": c_agree,
                    "h_minus_c_action_agreement": h_agree - c_agree,
                    "hybrid_policy_kl_mean": h_kl,
                    "classical_policy_kl_mean": c_kl,
                    "h_minus_c_policy_kl": h_kl - c_kl,
                    "post_kd_gap_h_minus_c_mean": _float(
                        rq3[handoff_t], "post_kd_gap_h_minus_c_mean"
                    ),
                    "diagnostic_checkpoint": checkpoint,
                }
            )
    return output


def _sorted_seed_rows(rows: Sequence[Mapping[str, str]], *, environment: str, label: str, seeds: Sequence[int]) -> list[Mapping[str, str]]:
    selected = sorted(rows, key=lambda row: _int(row, "seed"))
    observed = [_int(row, "seed") for row in selected]
    if observed != list(seeds):
        raise PaperAnalysisTableError(
            f"{environment} {label}: expected seeds {list(seeds)}, observed {observed}"
        )
    return selected


def _condition_rows_all(
    result_root: Path,
    environment: str,
    *,
    seeds: Sequence[int],
) -> dict[str, list[Mapping[str, str]]]:
    """Resolve every final public condition to seed-level endpoint rows."""

    handoff = read_csv(result_root / "data/handoff_matrix.csv")
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in handoff:
        condition = str(row.get("condition", "")).strip()
        if condition in CONDITION_ORDER:
            grouped[condition].append(row)

    grouped["full_classical"] = read_csv(result_root / "data/full_classical_matrix.csv")
    expected = set(CONDITION_ORDER)
    if set(grouped) != expected:
        raise PaperAnalysisTableError(
            f"{environment}: final-condition coverage drifted: {sorted(grouped)}"
        )
    return {
        condition: _sorted_seed_rows(grouped[condition], environment=environment, label=condition, seeds=seeds)
        for condition in CONDITION_ORDER
    }



def _condition_family(condition: str) -> str:
    if condition == "full_classical":
        return "full_classical"
    if condition.startswith("H("):
        return "hybrid"
    if condition.startswith("C("):
        return "compact_classical"
    raise PaperAnalysisTableError(f"Unknown final condition: {condition}")


def _condition_handoff_t(condition: str) -> float | str:
    if condition == "full_classical":
        return ""
    return float(condition[2:-1])


def _condition_qforward(row: Mapping[str, str], component: str) -> float:
    value = row.get(component, "")
    if value in (None, ""):
        return 0.0
    return _float(row, component)


def build_condition_summary(
    paper_root: str | Path,
    *,
    seeds: Sequence[int] = EXPECTED_SEEDS,
) -> list[dict[str, Any]]:
    """Summarise all final conditions without marginal confidence intervals."""

    paper_root = Path(paper_root)
    output: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        root = paper_root / environment
        grouped = _condition_rows_all(root, environment, seeds=seeds)
        for condition in CONDITION_ORDER:
            rows = grouped[condition]
            values = np.asarray([_float(row, "final_return_mean") for row in rows], dtype=np.float64)
            q_total = [_condition_qforward(row, "qforward_total") for row in rows]
            q_grad = [_condition_qforward(row, "qforward_grad") for row in rows]
            q_nograd = [_condition_qforward(row, "qforward_nograd") for row in rows]
            staged = condition not in {"full_classical", "H(0)", "C(0)"}
            total_env = TOTAL_ENV_INTERACTIONS[environment] + (
                KD_DATASET_ENV_INTERACTIONS if staged else 0
            )
            output.append(
                {
                    "environment": environment,
                    "evidence_role": _evidence_role(environment),
                    "family": _condition_family(condition),
                    "condition": condition,
                    "handoff_t": _condition_handoff_t(condition),
                    "n": len(values),
                    "mean_final_return": float(np.mean(values)),
                    "sd_final_return": float(np.std(values, ddof=1)),
                    "median_final_return": float(np.median(values)),
                    "iqm_final_return": interquartile_mean(values),
                    "min_final_return": float(np.min(values)),
                    "max_final_return": float(np.max(values)),
                    "mean_qforward_total": _mean(q_total, label=f"{environment} {condition} qforward total"),
                    "mean_qforward_grad": _mean(q_grad, label=f"{environment} {condition} qforward grad"),
                    "mean_qforward_nograd": _mean(q_nograd, label=f"{environment} {condition} qforward nograd"),
                    "total_env_interactions": total_env,
                }
            )
    return output


def _primary_descriptive_row(result: Any) -> dict[str, Any]:
    return {
        "family": "descriptive_reference",
        "confirmatory": False,
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
        "holm_p": "",
        "holm_reject_0_05": "",
        "iqm_difference": result.iqm_difference,
        "iqm_ci95_low": result.iqm_ci95_low,
        "iqm_ci95_high": result.iqm_ci95_high,
        "probability_of_improvement": result.probability_of_improvement,
        "retention_pct_of_teacher_mean": "",
        "pearson_r_same_seed": "",
    }


def build_fc_reference_contrasts(
    paper_root: str | Path,
    *,
    seeds: Sequence[int] = EXPECTED_SEEDS,
) -> list[dict[str, Any]]:
    """Descriptive full-classical/teacher references using the frozen paired engine."""

    paper_root = Path(paper_root)
    output: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        root = paper_root / environment
        grouped = _condition_rows_all(root, environment, seeds=seeds)
        full_classical = np.asarray(
            [_float(row, "final_return_mean") for row in grouped["full_classical"]], dtype=np.float64
        )
        for condition in ("H(0)", "C(0)"):
            left = np.asarray(
                [_float(row, "final_return_mean") for row in grouped[condition]], dtype=np.float64
            )
            output.append(
                {
                    "environment": environment,
                    **_primary_descriptive_row(
                        paired_contrast(
                            left,
                            full_classical,
                            left_name=condition,
                            right_name="full_classical",
                        )
                    ),
                }
            )

        h1_rows = grouped["H(1)"]
        h1 = np.asarray([_float(row, "final_return_mean") for row in h1_rows], dtype=np.float64)
        teacher = np.asarray([_float(row, "teacher_return_mean") for row in h1_rows], dtype=np.float64)
        result = _primary_descriptive_row(
            paired_contrast(h1, teacher, left_name="H(1)", right_name="teacher")
        )
        result["retention_pct_of_teacher_mean"] = 100.0 * float(np.mean(h1)) / float(np.mean(teacher))
        result["pearson_r_same_seed"] = float(np.corrcoef(h1, teacher)[0, 1])
        output.append({"environment": environment, **result})
    return output


def build_continuation_reference_contrasts(
    paper_root: str | Path,
    *,
    seeds: Sequence[int] = EXPECTED_SEEDS,
) -> list[dict[str, Any]]:
    """Build descriptive full-classical continuation references for H(t) and C(t).

    These rows mirror P3's sign convention and frozen per-handoff bootstrap seed,
    but are explicitly descriptive and receive no multiplicity adjustment.
    """

    paper_root = Path(paper_root)
    output: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        root = paper_root / environment
        grouped = _condition_rows_all(root, environment, seeds=seeds)
        full_classical = np.asarray(
            [_float(row, "final_return_mean") for row in grouped["full_classical"]],
            dtype=np.float64,
        )
        for handoff_index, handoff_t in enumerate(INTERIOR_HANDOFFS):
            label = f"{handoff_t:g}"
            bootstrap_seed = DEFAULT_BOOTSTRAP_SEED + 3500 + handoff_index
            for right_head in ("H", "C"):
                condition = f"{right_head}({label})"
                right = np.asarray(
                    [_float(row, "final_return_mean") for row in grouped[condition]],
                    dtype=np.float64,
                )
                result = paired_contrast(
                    full_classical,
                    right,
                    left_name="full classical continuation",
                    right_name=(
                        "hybrid switch" if right_head == "H" else "compact classical switch"
                    ),
                    bootstrap_seed=bootstrap_seed,
                )
                output.append(
                    {
                        "environment": environment,
                        "family": "descriptive_reference",
                        "confirmatory": False,
                        "handoff_t": handoff_t,
                        "contrast": result.contrast,
                        "right_head": right_head,
                        "n": result.n,
                        "paired_mean_difference": result.paired_mean_difference,
                        "ci95_low": result.ci95_low,
                        "ci95_high": result.ci95_high,
                        "sign_flip_p_unadjusted": result.sign_flip_p,
                        "wins_classical": result.wins_left,
                        "status": "descriptive_reference",
                        "iqm_difference": result.iqm_difference,
                        "iqm_ci95_low": result.iqm_ci95_low,
                        "iqm_ci95_high": result.iqm_ci95_high,
                        "probability_of_improvement": result.probability_of_improvement,
                    }
                )
    return output


def build_paper_analysis_tables(
    paper_root: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    """Build every deterministic cross-environment paper table."""

    return {
        "condition_summary.csv": build_condition_summary(paper_root),
        "fc_reference_contrasts.csv": build_fc_reference_contrasts(paper_root),
        "continuation_reference_contrasts.csv": build_continuation_reference_contrasts(paper_root),
        "allocation_normalized.csv": build_allocation_normalized(paper_root),
        "environment_summary.csv": build_environment_summary(paper_root),
        "resource_breakdown.csv": build_resource_breakdown(paper_root),
        "minatar_serial_timing.csv": build_minatar_serial_timing(paper_root),
        "kd_hc_diagnostics.csv": build_kd_hc_diagnostics(paper_root),
    }
