"""Transition and KD bridge summaries for delayed QNode schedules."""

from __future__ import annotations

from typing import Any

from hybrid_training_qrl.schedules import PhaseSpec, ScheduleSpec


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _float_or_zero(value: Any) -> float:
    parsed = _float_or_none(value)
    return 0.0 if parsed is None else parsed


def _int_or_zero(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return 0


def _safe_div(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else numerator / denominator


def build_kd_bridge_summary(
    *,
    schedule: ScheduleSpec,
    kd_phase: PhaseSpec,
    kd_metrics: dict[str, Any],
    total_qnode_time: float,
) -> dict[str, Any]:
    """Build the paper-facing summary for a real policy/value KD bridge."""

    kd_qnode_time = _float_or_zero(kd_metrics.get("kd_qnode_time"))
    return {
        "schedule_name": schedule.schedule_name,
        "transition_type": schedule.transition_type,
        "bridge_type": kd_metrics.get("bridge_type", schedule.bridge_type),
        "bridge_mode": kd_metrics.get("bridge_mode", "offline_policy_value_kd_micro"),
        "kd_phase_id": kd_phase.phase_id,
        "kd_samples": _int_or_zero(kd_metrics.get("kd_samples")),
        "kd_epochs": _int_or_zero(kd_metrics.get("kd_epochs")),
        "kd_wall_clock": _float_or_zero(kd_metrics.get("kd_wall_clock")),
        "kd_bridge_wall_clock_total": _float_or_zero(
            kd_metrics.get("kd_bridge_wall_clock_total")
        ),
        "kd_dataset_export_wall_clock": _float_or_none(
            kd_metrics.get("kd_dataset_export_wall_clock")
        ),
        "kd_dataset_collection_env_steps": _int_or_zero(
            kd_metrics.get("kd_dataset_collection_env_steps")
        ),
        "kd_dataset_episodes_touched": _int_or_zero(
            kd_metrics.get("kd_dataset_episodes_touched")
        ),
        "kd_dataset_done_count": _int_or_zero(kd_metrics.get("kd_dataset_done_count")),
        "kd_dataset_reward_mean": _float_or_none(kd_metrics.get("kd_dataset_reward_mean")),
        "kd_qnode_calls": _int_or_zero(kd_metrics.get("kd_qnode_calls")),
        "kd_qnode_time": kd_qnode_time,
        "qnode_calls_actor": _int_or_zero(kd_metrics.get("qnode_calls_actor")),
        "qnode_calls_critic": _int_or_zero(kd_metrics.get("qnode_calls_critic")),
        "qnode_time_actor": _float_or_zero(kd_metrics.get("qnode_time_actor")),
        "qnode_time_critic": _float_or_zero(kd_metrics.get("qnode_time_critic")),
        "kd_policy_weight": _float_or_none(kd_metrics.get("kd_policy_weight")),
        "kd_value_weight": _float_or_none(kd_metrics.get("kd_value_weight")),
        "kd_total_loss_initial": _float_or_none(kd_metrics.get("kd_total_loss_initial")),
        "kd_total_loss_final": _float_or_none(kd_metrics.get("kd_total_loss_final")),
        "kd_policy_loss_initial": _float_or_none(kd_metrics.get("kd_policy_loss_initial")),
        "kd_policy_loss_final": _float_or_none(kd_metrics.get("kd_policy_loss_final")),
        "kd_value_loss_initial": _float_or_none(kd_metrics.get("kd_value_loss_initial")),
        "kd_value_loss_final": _float_or_none(kd_metrics.get("kd_value_loss_final")),
        "kd_value_mse": _float_or_none(kd_metrics.get("kd_value_mse")),
        "kd_loss_initial": _float_or_none(kd_metrics.get("kd_loss_initial")),
        "kd_loss_final": _float_or_none(kd_metrics.get("kd_loss_final")),
        "kd_policy_kl": _float_or_none(kd_metrics.get("kd_policy_kl")),
        "kd_action_agreement": _float_or_none(kd_metrics.get("kd_action_agreement")),
        "student_actor_head_path": kd_metrics.get("student_actor_head_path"),
        "student_critic_head_path": kd_metrics.get("student_critic_head_path"),
        "bridge_cost_share": _safe_div(kd_qnode_time, float(total_qnode_time)),
        "codecarbon_enabled": bool(kd_metrics.get("codecarbon_enabled", False)),
        "codecarbon_tracker_status": kd_metrics.get("codecarbon_tracker_status"),
        "codecarbon_error": kd_metrics.get("codecarbon_error"),
        "codecarbon_output_file": kd_metrics.get("codecarbon_output_file"),
        "codecarbon_energy_kwh": _float_or_none(
            kd_metrics.get("codecarbon_energy_kwh")
        ),
        "codecarbon_emissions_kg": _float_or_none(
            kd_metrics.get("codecarbon_emissions_kg")
        ),
        "codecarbon_duration_seconds": _float_or_none(
            kd_metrics.get("codecarbon_duration_seconds")
        ),
    }


def build_transition_summary(
    *,
    schedule: ScheduleSpec,
    transition_phase: PhaseSpec,
    before_phase_metrics: dict[str, Any] | None = None,
    after_phase_metrics: dict[str, Any] | None = None,
    phase_metrics: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write the training-time transition placeholder.

    Policy returns are deliberately absent during training. The post-hoc evaluator
    replaces this file after evaluating the teacher, immediate post-KD policy and
    final policy with the common stochastic protocol.
    """

    del before_phase_metrics, after_phase_metrics, phase_metrics
    transition_step = sum(
        phase.steps for phase in schedule.phases if phase.phase_id < transition_phase.phase_id
    )
    return {
        "contract_version": 2,
        "status": "pending_posthoc",
        "schedule_name": schedule.schedule_name,
        "schedule_family": schedule.schedule_family,
        "transition_type": schedule.transition_type,
        "bridge_type": schedule.bridge_type,
        "transition_phase_id": transition_phase.phase_id,
        "transition_step": transition_step,
        "evaluation_mode": "stochastic_primary",
        "teacher_return_posthoc": None,
        "post_kd_return_posthoc": None,
        "final_return_posthoc": None,
        "immediate_transition_drop": None,
        "final_delta_vs_teacher": None,
        "recovered_by_budget_end": None,
        "recovery_step": None,
        "recovery_wall_clock": None,
        "recovery_qnode_calls": None,
        "recovery_qnode_time": None,
        "note": (
            "Training does not evaluate policies. Run scripts/evaluate_checkpoints.py "
            "with phase-scope=canonical-endpoints to complete this artifact."
        ),
    }
