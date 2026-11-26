"""Writers for schedule-aware experiment summary artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from hybrid_training_qrl.schedules import ScheduleSpec

SCHEDULE_SUMMARY = "schedule_summary.json"
PHASE_SUMMARY = "phase_summary.csv"
QUANTUM_SPEND_SUMMARY = "quantum_spend_summary.json"
TRANSITION_SUMMARY = "transition_summary.json"
KD_BRIDGE_SUMMARY = "kd_bridge_summary.json"
ENERGY_SUMMARY = "energy_summary.json"
DOWNSTREAM_SUMMARY = "downstream_summary.json"
AUDIT_REPORT = "audit_report.json"

PHASE_SUMMARY_FIELDS = [
    "phase_id",
    "phase_name",
    "phase_type",
    "start_step",
    "end_step",
    "phase_budget_steps",
    "phase_configured_env_steps",
    "phase_effective_env_steps",
    "phase_step_overshoot",
    "phase_start_global_step",
    "phase_final_global_step",
    "phase_rollout_batch_size",
    "phase_num_iterations",
    "phase_rollout_aligned",
    "initialization_mode",
    "continuation_source_run_dir",
    "continuation_source_checkpoint",
    "continuation_source_global_step",
    "continuation_seed",
    "continuation_final_global_step",
    "continuation_target_global_step",
    "continuation_target_reached",
    "continuation_optimizer_state_restored",
    "continuation_rng_state_restored",
    "qnode_active",
    "wall_clock",
    "qnode_calls",
    "qnode_time",
    "qnode_calls_actor",
    "qnode_calls_critic",
    "qnode_time_actor",
    "qnode_time_critic",
    "training_evaluation_count",
    "training_evaluation_artifact_present",
    "training_evaluation_contract_ok",
    "kd_dataset_collection_env_steps",
    "codecarbon_energy_kwh",
    "codecarbon_emissions_kg",
    "trainability_status",
    "trainability_sampled_updates",
    "trainability_approx_kl_mean",
    "trainability_approx_kl_final",
    "trainability_clipfrac_mean",
    "trainability_clipfrac_final",
    "trainability_explained_variance_mean",
    "trainability_explained_variance_final",
    "trainability_loss_policy_final",
    "trainability_loss_value_final",
    "trainability_loss_entropy_final",
    "trainability_actor_q_grad_mean",
    "trainability_actor_q_grad_final",
    "trainability_actor_q_grad_zero_fraction",
    "trainability_critic_q_grad_mean",
    "trainability_critic_q_grad_final",
    "trainability_critic_q_grad_zero_fraction",
    "trainability_actor_classical_grad_mean",
    "trainability_critic_classical_grad_mean",
    "trainability_actor_q_output_std_mean",
    "trainability_actor_q_output_std_final",
    "trainability_critic_q_output_std_mean",
    "trainability_critic_q_output_std_final",
    "trainability_actor_logits_std_mean",
    "trainability_actor_logits_std_final",
    "trainability_actor_action_max_prob_mean",
    "trainability_actor_action_max_prob_final",
    "trainability_critic_values_std_mean",
    "trainability_critic_values_std_final",
    "trainability_summary_path",
    "optimization_diagnostics_status",
    "optimization_diagnostics_sampled_optimizer_steps",
    "optimization_gradient_clipping_mode",
    "optimization_learning_rate_mode",
    "optimization_learning_rate_actor_mean",
    "optimization_learning_rate_critic_mean",
    "optimization_learning_rate_shared_mean",
    "optimization_clip_active_fraction",
    "optimization_clip_active_actor_fraction",
    "optimization_clip_active_critic_fraction",
    "optimization_clip_active_shared_fraction",
    "optimization_clip_coefficient_actor_mean",
    "optimization_clip_coefficient_actor_min",
    "optimization_clip_coefficient_critic_mean",
    "optimization_clip_coefficient_critic_min",
    "optimization_clip_coefficient_shared_mean",
    "optimization_clip_coefficient_shared_min",
    "optimization_grad_global_pre_clip_mean",
    "optimization_grad_global_pre_clip_max",
    "optimization_clip_coefficient_global_mean",
    "optimization_clip_coefficient_global_min",
    "optimization_grad_actor_pre_clip_mean",
    "optimization_grad_actor_pre_clip_final",
    "optimization_grad_critic_pre_clip_mean",
    "optimization_grad_critic_pre_clip_final",
    "optimization_grad_actor_post_clip_mean",
    "optimization_grad_actor_post_clip_final",
    "optimization_grad_critic_post_clip_mean",
    "optimization_grad_critic_post_clip_final",
    "optimization_grad_actor_to_critic_ratio_mean",
    "optimization_grad_actor_to_critic_ratio_final",
    "optimization_grad_actor_share_mean",
    "optimization_grad_critic_share_mean",
    "optimization_update_actor_norm_mean",
    "optimization_update_critic_norm_mean",
    "optimization_update_actor_to_weight_ratio_mean",
    "optimization_update_actor_to_weight_ratio_final",
    "optimization_update_critic_to_weight_ratio_mean",
    "optimization_update_critic_to_weight_ratio_final",
    "optimization_update_actor_to_critic_ratio_mean",
    "optimization_update_actor_to_critic_ratio_final",
    "optimization_update_actor_to_weight_per_preclip_grad_mean",
    "optimization_update_critic_to_weight_per_preclip_grad_mean",
    "optimization_diagnostics_summary_path",
    "phase_run_dir",
    "config",
]


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_phase_summary(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_rows = [{field: row.get(field) for field in PHASE_SUMMARY_FIELDS} for row in rows]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PHASE_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(normalized_rows)


def merge_phase_metrics(
    base_rows: list[dict[str, Any]],
    phase_metrics: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Merge runner-derived metrics into phase rows.

    Runner-derived training and KD metrics are keyed by ``phase_id``. Policy
    evaluation is deliberately excluded and is written by the post-hoc evaluator.
    """

    phase_metrics = phase_metrics or {}
    rows: list[dict[str, Any]] = []
    for row in base_rows:
        phase_id = int(row["phase_id"])
        merged = dict(row)
        merged.update(
            {
                "phase_configured_env_steps": row.get("phase_budget_steps"),
                "phase_effective_env_steps": row.get("phase_budget_steps"),
                "phase_step_overshoot": 0,
                "phase_start_global_step": None,
                "phase_final_global_step": None,
                "phase_rollout_batch_size": None,
                "phase_num_iterations": None,
                "phase_rollout_aligned": None,
                "initialization_mode": None,
                "continuation_source_run_dir": None,
                "continuation_source_checkpoint": None,
                "continuation_source_global_step": None,
                "continuation_seed": None,
                "continuation_final_global_step": None,
                "continuation_target_global_step": None,
                "continuation_target_reached": None,
                "continuation_optimizer_state_restored": None,
                "continuation_rng_state_restored": None,
                "wall_clock": 0.0,
                "qnode_calls": 0,
                "qnode_time": 0.0,
                "qnode_calls_actor": 0,
                "qnode_calls_critic": 0,
                "qnode_time_actor": 0.0,
                "qnode_time_critic": 0.0,
                "training_evaluation_count": 0,
                "training_evaluation_artifact_present": False,
                "training_evaluation_contract_ok": True,
                "kd_dataset_collection_env_steps": 0,
                "codecarbon_energy_kwh": None,
                "codecarbon_emissions_kg": None,
            }
        )
        merged.update(phase_metrics.get(phase_id, {}))
        rows.append(merged)
    return rows


def build_quantum_spend_summary(
    schedule: ScheduleSpec,
    phase_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate training/KD resource expenditure only.

    Policy evaluation is intentionally absent from this artifact. Post-hoc
    performance lives in posthoc_performance_*.json and never changes the
    training resource totals.
    """

    qnode_calls_total = sum(int(row.get("qnode_calls") or 0) for row in phase_rows)
    qnode_time_total = sum(float(row.get("qnode_time") or 0.0) for row in phase_rows)
    qnode_calls_actor_total = sum(
        int(row.get("qnode_calls_actor") or 0) for row in phase_rows
    )
    qnode_calls_critic_total = sum(
        int(row.get("qnode_calls_critic") or 0) for row in phase_rows
    )
    qnode_time_actor_total = sum(
        float(row.get("qnode_time_actor") or 0.0) for row in phase_rows
    )
    qnode_time_critic_total = sum(
        float(row.get("qnode_time_critic") or 0.0) for row in phase_rows
    )
    training_wall_clock_total = sum(
        float(row.get("wall_clock") or 0.0) for row in phase_rows
    )

    total_steps = schedule.total_env_step_budget
    active_steps = schedule.qnode_active_steps_budget
    effective_total_steps = sum(
        int(row.get("phase_effective_env_steps") or row.get("phase_budget_steps") or 0)
        for row in phase_rows
    )
    effective_active_steps = sum(
        int(row.get("phase_effective_env_steps") or row.get("phase_budget_steps") or 0)
        for row in phase_rows
        if str(row.get("qnode_active", "")).lower() in {"true", "1"}
        or row.get("qnode_active") is True
    )
    kd_dataset_collection_env_steps = sum(
        int(row.get("kd_dataset_collection_env_steps") or 0) for row in phase_rows
    )

    def safe_div(numerator: float, denominator: float) -> float | None:
        return None if denominator == 0 else numerator / denominator

    return {
        "contract_version": 2,
        "resource_scope": "training_and_kd_only",
        "periodic_evaluation_included": False,
        "posthoc_evaluation_included": False,
        "schedule_name": schedule.schedule_name,
        "schedule_family": schedule.schedule_family,
        "online_qnode_active_env_step_ratio": schedule.qnode_active_steps_ratio,
        "qnode_active_steps_ratio": schedule.qnode_active_steps_ratio,
        "qnode_active_steps_budget": active_steps,
        "total_env_step_budget": total_steps,
        "total_env_steps_effective": effective_total_steps,
        "total_env_step_overshoot": effective_total_steps - total_steps,
        "kd_dataset_collection_env_steps": kd_dataset_collection_env_steps,
        "total_environment_interactions_accounted": (
            effective_total_steps + kd_dataset_collection_env_steps
        ),
        "qnode_active_env_steps_effective": effective_active_steps,
        "qnode_active_steps_ratio_effective": safe_div(
            float(effective_active_steps), float(effective_total_steps)
        ),
        "qnode_calls_total": qnode_calls_total,
        "qnode_time_total": qnode_time_total,
        "qnode_calls_actor_total": qnode_calls_actor_total,
        "qnode_calls_critic_total": qnode_calls_critic_total,
        "qnode_time_actor_total": qnode_time_actor_total,
        "qnode_time_critic_total": qnode_time_critic_total,
        "training_wall_clock_total": training_wall_clock_total,
        "wall_clock_total": training_wall_clock_total,
        "qnode_time_share_total": safe_div(qnode_time_total, training_wall_clock_total),
        "qnode_calls_per_total_env_step": safe_div(
            float(qnode_calls_total), float(total_steps)
        ),
        "qnode_calls_per_effective_env_step": safe_div(
            float(qnode_calls_total), float(effective_total_steps)
        ),
        "qnode_time_per_total_env_step": safe_div(qnode_time_total, float(total_steps)),
        "qnode_time_per_effective_env_step": safe_div(
            qnode_time_total, float(effective_total_steps)
        ),
        "qnode_calls_per_qnode_active_step": safe_div(
            float(qnode_calls_total), float(active_steps)
        ),
        "qnode_calls_per_effective_qnode_active_step": safe_div(
            float(qnode_calls_total), float(effective_active_steps)
        ),
        "qnode_time_per_qnode_active_step": safe_div(qnode_time_total, float(active_steps)),
        "qnode_time_per_effective_qnode_active_step": safe_div(
            qnode_time_total, float(effective_active_steps)
        ),
        "qnode_calls_by_phase": {
            str(row["phase_id"]): int(row.get("qnode_calls") or 0) for row in phase_rows
        },
        "qnode_time_by_phase": {
            str(row["phase_id"]): float(row.get("qnode_time") or 0.0) for row in phase_rows
        },
        "qnode_calls_actor_by_phase": {
            str(row["phase_id"]): int(row.get("qnode_calls_actor") or 0)
            for row in phase_rows
        },
        "qnode_calls_critic_by_phase": {
            str(row["phase_id"]): int(row.get("qnode_calls_critic") or 0)
            for row in phase_rows
        },
        "qnode_time_actor_by_phase": {
            str(row["phase_id"]): float(row.get("qnode_time_actor") or 0.0)
            for row in phase_rows
        },
        "qnode_time_critic_by_phase": {
            str(row["phase_id"]): float(row.get("qnode_time_critic") or 0.0)
            for row in phase_rows
        },
    }


def write_schedule_artifacts(
    run_dir: str | Path,
    schedule: ScheduleSpec,
    *,
    phase_metrics: dict[int, dict[str, Any]] | None = None,
    quantum_spend_summary: dict[str, Any] | None = None,
    transition_summary: dict[str, Any] | None = None,
    kd_bridge_summary: dict[str, Any] | None = None,
    energy_summary: dict[str, Any] | None = None,
    downstream_summary: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write training/KD artifacts without embedding policy evaluation results."""

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    phase_rows = merge_phase_metrics(schedule.default_phase_rows(), phase_metrics)
    if quantum_spend_summary is None:
        quantum_spend_summary = build_quantum_spend_summary(schedule, phase_rows)

    paths = {
        "schedule_summary": run_dir / SCHEDULE_SUMMARY,
        "phase_summary": run_dir / PHASE_SUMMARY,
        "quantum_spend_summary": run_dir / QUANTUM_SPEND_SUMMARY,
    }
    schedule_summary = schedule.to_schedule_summary()
    effective_total_steps = sum(
        int(row.get("phase_effective_env_steps") or row.get("phase_budget_steps") or 0)
        for row in phase_rows
    )
    effective_active_steps = sum(
        int(row.get("phase_effective_env_steps") or row.get("phase_budget_steps") or 0)
        for row in phase_rows
        if str(row.get("qnode_active", "")).lower() in {"true", "1"}
        or row.get("qnode_active") is True
    )
    execution_scope = str(schedule.metadata.get("execution_scope", "paper_facing"))
    if execution_scope == "development_parallelism_gate":
        codecarbon_scope = "disabled_engineering_only"
    elif execution_scope == "parallel_performance_campaign":
        codecarbon_scope = "disabled_parallel_performance_campaign"
    else:
        codecarbon_scope = "training_and_kd_only"
    schedule_summary.update(
        {
            "evaluation_protocol": "posthoc_only",
            "training_evaluation_enabled": False,
            "codecarbon_scope": codecarbon_scope,
            "total_env_steps_effective": effective_total_steps,
            "total_env_step_overshoot": effective_total_steps - schedule.total_env_step_budget,
            "qnode_active_env_steps_effective": effective_active_steps,
            "qnode_active_steps_ratio_effective": (
                effective_active_steps / effective_total_steps
                if effective_total_steps > 0
                else 0.0
            ),
        }
    )
    write_json(paths["schedule_summary"], schedule_summary)
    write_phase_summary(paths["phase_summary"], phase_rows)
    write_json(paths["quantum_spend_summary"], quantum_spend_summary)

    if transition_summary is not None:
        paths["transition_summary"] = run_dir / TRANSITION_SUMMARY
        write_json(paths["transition_summary"], transition_summary)

    if kd_bridge_summary is not None:
        paths["kd_bridge_summary"] = run_dir / KD_BRIDGE_SUMMARY
        write_json(paths["kd_bridge_summary"], kd_bridge_summary)

    if energy_summary is not None:
        paths["energy_summary"] = run_dir / ENERGY_SUMMARY
        write_json(paths["energy_summary"], energy_summary)

    if downstream_summary is not None:
        paths["downstream_summary"] = run_dir / DOWNSTREAM_SUMMARY
        write_json(paths["downstream_summary"], downstream_summary)

    return paths
