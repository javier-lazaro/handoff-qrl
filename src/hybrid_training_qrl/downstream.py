"""Downstream adaptation summaries for QNode activation schedules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hybrid_training_qrl.schedules import PhaseSpec, ScheduleSpec


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


def _float_or_zero(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def build_downstream_summary(
    *,
    schedule: ScheduleSpec,
    representation_phase: PhaseSpec,
    kd_phase: PhaseSpec,
    downstream_phase: PhaseSpec,
    kd_metrics: dict[str, Any],
    downstream_metrics: dict[str, Any],
    representation_run_dir: str | Path,
    kd_phase_dir: str | Path,
    downstream_run_dir: str | Path,
    composition_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact summary for downstream hybrid adaptation.

    Qdownstream reports representation/development and downstream adaptation as
    separate concepts.  The schedule-level quantum summary still aggregates all
    QNode spend, while this artifact records the downstream-specific cost and
    frozen-feature semantics.
    """

    composition_summary = composition_summary or {}
    downstream_qnode_calls = _int_or_zero(downstream_metrics.get("qnode_calls"))
    downstream_qnode_time = _float_or_zero(downstream_metrics.get("qnode_time"))

    return {
        "schedule_name": schedule.schedule_name,
        "schedule_family": schedule.schedule_family,
        "downstream_mode": "policy_value_kd_plus_frozen_ft",
        "downstream_cost_reported_separately": True,
        "representation_phase_id": representation_phase.phase_id,
        "kd_phase_id": kd_phase.phase_id,
        "downstream_ft_phase_id": downstream_phase.phase_id,
        "representation_steps": representation_phase.steps,
        "downstream_env_steps": downstream_phase.steps,
        "freeze_feature_extractor": True,
        "feature_interface_trainable_during_phase": False,
        "feature_extractor_runtime_frozen": composition_summary.get(
            "feature_extractor_runtime_frozen"
        ),
        "feature_extractor_runtime_frozen_after_training": composition_summary.get(
            "feature_extractor_runtime_frozen_after_training"
        ),
        "student_actor_head_path": kd_metrics.get("student_actor_head_path"),
        "student_critic_head_path": kd_metrics.get("student_critic_head_path"),
        "teacher_checkpoint": composition_summary.get("teacher_checkpoint"),
        "downstream_checkpoint_path": composition_summary.get("checkpoint_path"),
        "representation_run_dir": str(representation_run_dir),
        "kd_phase_dir": str(kd_phase_dir),
        "downstream_run_dir": str(downstream_run_dir),
        "downstream_qnode_calls": downstream_qnode_calls,
        "downstream_qnode_time": downstream_qnode_time,
        "downstream_qnode_calls_actor": _int_or_zero(
            downstream_metrics.get("qnode_calls_actor")
        ),
        "downstream_qnode_calls_critic": _int_or_zero(
            downstream_metrics.get("qnode_calls_critic")
        ),
        "downstream_qnode_time_actor": _float_or_zero(
            downstream_metrics.get("qnode_time_actor")
        ),
        "downstream_qnode_time_critic": _float_or_zero(
            downstream_metrics.get("qnode_time_critic")
        ),
        "evaluation_protocol": "posthoc_only",
        "posthoc_status": "pending",
        "final_return_posthoc": None,
        "kd_qnode_calls": _int_or_zero(kd_metrics.get("kd_qnode_calls")),
        "kd_qnode_time": _float_or_zero(kd_metrics.get("kd_qnode_time")),
        "kd_policy_loss_final": _maybe_float(kd_metrics.get("kd_policy_loss_final")),
        "kd_value_loss_final": _maybe_float(kd_metrics.get("kd_value_loss_final")),
        "kd_value_mse": _maybe_float(kd_metrics.get("kd_value_mse")),
    }
