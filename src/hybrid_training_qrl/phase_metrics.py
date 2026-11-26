"""Collect phase-level metrics from SimplyQRL run artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from hybrid_training_qrl.execution import PHASE_EXECUTION_SUMMARY
from hybrid_training_qrl.simplyqrl_adapter import CONTINUATION_SUMMARY

TRAINABILITY_SUMMARY = "trainability_summary.json"
OPTIMIZATION_DIAGNOSTICS_SUMMARY = "optimization_diagnostics_summary.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


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
    parsed = _float_or_none(value)
    return 0.0 if parsed is None else parsed


def collect_phase_metrics(phase_run_dir: str | Path) -> dict[str, Any]:
    """Collect best-effort metrics from a SimplyQRL phase run directory.

    Missing diagnostic artifacts are tolerated. An ``evaluations.csv`` artifact
    is not tolerated by the run audit because paper-facing training is post-hoc only.
    """

    phase_run_dir = Path(phase_run_dir)
    summary = _read_json(phase_run_dir / "summary.json")
    execution = _read_json(phase_run_dir / PHASE_EXECUTION_SUMMARY)
    evaluations_path = phase_run_dir / "evaluations.csv"
    eval_rows = _read_csv_rows(evaluations_path)
    trainability = _read_json(phase_run_dir / TRAINABILITY_SUMMARY)
    optimization = _read_json(phase_run_dir / OPTIMIZATION_DIAGNOSTICS_SUMMARY)
    continuation = _read_json(phase_run_dir / CONTINUATION_SUMMARY)

    qnode_calls_actor = _int_or_zero(summary.get("qnode_calls_actor"))
    qnode_calls_critic = _int_or_zero(summary.get("qnode_calls_critic"))
    qnode_time_actor = _float_or_zero(summary.get("qnode_time_actor"))
    qnode_time_critic = _float_or_zero(summary.get("qnode_time_critic"))

    # Contract-aware total: all schedules report actor/critic diagnostics when
    # available, while paper-facing summaries use the aggregate total.  This
    # keeps Q100 and delayed schedules comparable even if actor/critic counts
    # differ slightly.
    if qnode_calls_actor or qnode_calls_critic:
        qnode_calls = qnode_calls_actor + qnode_calls_critic
    else:
        qnode_calls = _int_or_zero(summary.get("qnode_calls_total"))

    if qnode_time_actor or qnode_time_critic:
        qnode_time = qnode_time_actor + qnode_time_critic
    else:
        qnode_time = _float_or_zero(summary.get("qnode_time_total"))

    metrics = {
        "phase_run_dir": str(phase_run_dir),
        "phase_configured_env_steps": _int_or_zero(execution.get("configured_timesteps")),
        "phase_effective_env_steps": _int_or_zero(execution.get("effective_env_steps")),
        "phase_step_overshoot": _int_or_zero(execution.get("step_overshoot")),
        "phase_start_global_step": _int_or_zero(execution.get("start_global_step")),
        "phase_final_global_step": _int_or_zero(execution.get("final_global_step")),
        "phase_rollout_batch_size": _int_or_zero(execution.get("rollout_batch_size")),
        "phase_num_iterations": _int_or_zero(execution.get("actual_iterations")),
        "phase_rollout_aligned": execution.get("rollout_aligned"),
        "initialization_mode": continuation.get("initialization_mode"),
        "continuation_source_run_dir": continuation.get("source_run_dir"),
        "continuation_source_checkpoint": continuation.get("source_checkpoint"),
        "continuation_source_global_step": _int_or_zero(
            continuation.get("source_global_step")
        ),
        "continuation_seed": continuation.get("continuation_seed"),
        "continuation_final_global_step": _int_or_zero(
            continuation.get("continuation_final_global_step")
        ),
        "continuation_target_global_step": _int_or_zero(
            continuation.get("target_cumulative_global_step")
        ),
        "continuation_target_reached": continuation.get("target_global_step_reached"),
        "continuation_optimizer_state_restored": continuation.get(
            "optimizer_state_restored"
        ),
        "continuation_rng_state_restored": continuation.get("rng_state_restored"),
        "wall_clock": _float_or_zero(summary.get("wall_clock_elapsed")),
        "qnode_calls": qnode_calls,
        "qnode_time": qnode_time,
        "qnode_calls_actor": qnode_calls_actor,
        "qnode_calls_critic": qnode_calls_critic,
        "qnode_time_actor": qnode_time_actor,
        "qnode_time_critic": qnode_time_critic,
        "training_evaluation_count": len(eval_rows),
        "training_evaluation_artifact_present": evaluations_path.exists(),
        "training_evaluation_contract_ok": not eval_rows and not evaluations_path.exists(),
        "kd_dataset_collection_env_steps": _int_or_zero(
            summary.get("kd_dataset_collection_env_steps")
        ),
        "codecarbon_energy_kwh": _float_or_none(summary.get("codecarbon_energy_kwh")),
        "codecarbon_emissions_kg": _float_or_none(summary.get("codecarbon_emissions_kg")),
        "codecarbon_duration_seconds": _float_or_none(
            summary.get("codecarbon_duration_seconds")
        ),
        "codecarbon_enabled": bool(summary.get("codecarbon_enabled", False)),
        "codecarbon_output_file": summary.get("codecarbon_output_file"),
        "trainability_status": trainability.get("status"),
        "trainability_sampled_updates": _int_or_zero(trainability.get("sampled_updates")),
        "trainability_approx_kl_mean": _float_or_none(trainability.get("approx_kl_mean")),
        "trainability_approx_kl_final": _float_or_none(trainability.get("approx_kl_final")),
        "trainability_clipfrac_mean": _float_or_none(trainability.get("clipfrac_mean")),
        "trainability_clipfrac_final": _float_or_none(trainability.get("clipfrac_final")),
        "trainability_explained_variance_mean": _float_or_none(
            trainability.get("explained_variance_mean")
        ),
        "trainability_explained_variance_final": _float_or_none(
            trainability.get("explained_variance_final")
        ),
        "trainability_loss_policy_final": _float_or_none(
            trainability.get("loss_policy_final")
        ),
        "trainability_loss_value_final": _float_or_none(
            trainability.get("loss_value_final")
        ),
        "trainability_loss_entropy_final": _float_or_none(
            trainability.get("loss_entropy_final")
        ),
        "trainability_actor_q_grad_mean": _float_or_none(
            trainability.get("grad_actor_q_mean_mean")
        ),
        "trainability_actor_q_grad_final": _float_or_none(
            trainability.get("grad_actor_q_mean_final")
        ),
        "trainability_actor_q_grad_zero_fraction": _float_or_none(
            trainability.get("grad_actor_q_zero_fraction_mean")
        ),
        "trainability_critic_q_grad_mean": _float_or_none(
            trainability.get("grad_critic_q_mean_mean")
        ),
        "trainability_critic_q_grad_final": _float_or_none(
            trainability.get("grad_critic_q_mean_final")
        ),
        "trainability_critic_q_grad_zero_fraction": _float_or_none(
            trainability.get("grad_critic_q_zero_fraction_mean")
        ),
        "trainability_actor_classical_grad_mean": _float_or_none(
            trainability.get("grad_actor_classical_mean_mean")
        ),
        "trainability_critic_classical_grad_mean": _float_or_none(
            trainability.get("grad_critic_classical_mean_mean")
        ),
        "trainability_actor_q_output_std_mean": _float_or_none(
            trainability.get("actor_q_output_std_mean")
        ),
        "trainability_actor_q_output_std_final": _float_or_none(
            trainability.get("actor_q_output_std_final")
        ),
        "trainability_critic_q_output_std_mean": _float_or_none(
            trainability.get("critic_q_output_std_mean")
        ),
        "trainability_critic_q_output_std_final": _float_or_none(
            trainability.get("critic_q_output_std_final")
        ),
        "trainability_actor_logits_std_mean": _float_or_none(
            trainability.get("actor_logits_std_mean")
        ),
        "trainability_actor_logits_std_final": _float_or_none(
            trainability.get("actor_logits_std_final")
        ),
        "trainability_actor_action_max_prob_mean": _float_or_none(
            trainability.get("actor_action_max_prob_mean_mean")
        ),
        "trainability_actor_action_max_prob_final": _float_or_none(
            trainability.get("actor_action_max_prob_mean_final")
        ),
        "trainability_critic_values_std_mean": _float_or_none(
            trainability.get("critic_values_std_mean")
        ),
        "trainability_critic_values_std_final": _float_or_none(
            trainability.get("critic_values_std_final")
        ),
        "trainability_summary_path": str(phase_run_dir / TRAINABILITY_SUMMARY)
        if trainability
        else None,
        "optimization_diagnostics_status": optimization.get("status"),
        "optimization_diagnostics_sampled_optimizer_steps": _int_or_zero(
            optimization.get("sampled_optimizer_steps")
        ),
        "optimization_gradient_clipping_mode": optimization.get(
            "gradient_clipping_mode"
        ),
        "optimization_learning_rate_mode": optimization.get("learning_rate_mode"),
        "optimization_learning_rate_actor_mean": _float_or_none(
            optimization.get("learning_rate_actor_mean")
        ),
        "optimization_learning_rate_critic_mean": _float_or_none(
            optimization.get("learning_rate_critic_mean")
        ),
        "optimization_learning_rate_shared_mean": _float_or_none(
            optimization.get("learning_rate_shared_mean")
        ),
        "optimization_clip_active_fraction": _float_or_none(
            optimization.get("clip_active_fraction")
        ),
        "optimization_clip_active_actor_fraction": _float_or_none(
            optimization.get("clip_active_actor_fraction")
        ),
        "optimization_clip_active_critic_fraction": _float_or_none(
            optimization.get("clip_active_critic_fraction")
        ),
        "optimization_clip_active_shared_fraction": _float_or_none(
            optimization.get("clip_active_shared_fraction")
        ),
        "optimization_clip_coefficient_actor_mean": _float_or_none(
            optimization.get("clip_coefficient_actor_mean")
        ),
        "optimization_clip_coefficient_actor_min": _float_or_none(
            optimization.get("clip_coefficient_actor_min")
        ),
        "optimization_clip_coefficient_critic_mean": _float_or_none(
            optimization.get("clip_coefficient_critic_mean")
        ),
        "optimization_clip_coefficient_critic_min": _float_or_none(
            optimization.get("clip_coefficient_critic_min")
        ),
        "optimization_clip_coefficient_shared_mean": _float_or_none(
            optimization.get("clip_coefficient_shared_mean")
        ),
        "optimization_clip_coefficient_shared_min": _float_or_none(
            optimization.get("clip_coefficient_shared_min")
        ),
        "optimization_grad_global_pre_clip_mean": _float_or_none(
            optimization.get("grad_global_pre_clip_mean")
        ),
        "optimization_grad_global_pre_clip_max": _float_or_none(
            optimization.get("grad_global_pre_clip_max")
        ),
        "optimization_clip_coefficient_global_mean": _float_or_none(
            optimization.get("clip_coefficient_global_mean")
        ),
        "optimization_clip_coefficient_global_min": _float_or_none(
            optimization.get("clip_coefficient_global_min")
        ),
        "optimization_grad_actor_pre_clip_mean": _float_or_none(
            optimization.get("grad_actor_pre_clip_mean")
        ),
        "optimization_grad_actor_pre_clip_final": _float_or_none(
            optimization.get("grad_actor_pre_clip_final")
        ),
        "optimization_grad_critic_pre_clip_mean": _float_or_none(
            optimization.get("grad_critic_pre_clip_mean")
        ),
        "optimization_grad_critic_pre_clip_final": _float_or_none(
            optimization.get("grad_critic_pre_clip_final")
        ),
        "optimization_grad_actor_post_clip_mean": _float_or_none(
            optimization.get("grad_actor_post_clip_mean")
        ),
        "optimization_grad_actor_post_clip_final": _float_or_none(
            optimization.get("grad_actor_post_clip_final")
        ),
        "optimization_grad_critic_post_clip_mean": _float_or_none(
            optimization.get("grad_critic_post_clip_mean")
        ),
        "optimization_grad_critic_post_clip_final": _float_or_none(
            optimization.get("grad_critic_post_clip_final")
        ),
        "optimization_grad_actor_to_critic_ratio_mean": _float_or_none(
            optimization.get("grad_actor_to_critic_ratio_pre_clip_mean")
        ),
        "optimization_grad_actor_to_critic_ratio_final": _float_or_none(
            optimization.get("grad_actor_to_critic_ratio_pre_clip_final")
        ),
        "optimization_grad_actor_share_mean": _float_or_none(
            optimization.get("grad_actor_share_pre_clip_mean")
        ),
        "optimization_grad_critic_share_mean": _float_or_none(
            optimization.get("grad_critic_share_pre_clip_mean")
        ),
        "optimization_update_actor_norm_mean": _float_or_none(
            optimization.get("update_actor_norm_mean")
        ),
        "optimization_update_critic_norm_mean": _float_or_none(
            optimization.get("update_critic_norm_mean")
        ),
        "optimization_update_actor_to_weight_ratio_mean": _float_or_none(
            optimization.get("update_actor_to_weight_ratio_mean")
        ),
        "optimization_update_actor_to_weight_ratio_final": _float_or_none(
            optimization.get("update_actor_to_weight_ratio_final")
        ),
        "optimization_update_critic_to_weight_ratio_mean": _float_or_none(
            optimization.get("update_critic_to_weight_ratio_mean")
        ),
        "optimization_update_critic_to_weight_ratio_final": _float_or_none(
            optimization.get("update_critic_to_weight_ratio_final")
        ),
        "optimization_update_actor_to_critic_ratio_mean": _float_or_none(
            optimization.get("update_actor_to_critic_ratio_mean")
        ),
        "optimization_update_actor_to_critic_ratio_final": _float_or_none(
            optimization.get("update_actor_to_critic_ratio_final")
        ),
        "optimization_update_actor_to_weight_per_preclip_grad_mean": _float_or_none(
            optimization.get("update_actor_to_weight_per_preclip_grad_mean")
        ),
        "optimization_update_critic_to_weight_per_preclip_grad_mean": _float_or_none(
            optimization.get("update_critic_to_weight_per_preclip_grad_mean")
        ),
        "optimization_diagnostics_summary_path": str(
            phase_run_dir / OPTIMIZATION_DIAGNOSTICS_SUMMARY
        )
        if optimization
        else None,
    }

    return metrics


def build_energy_summary(
    *,
    total_env_step_budget: int,
    qnode_active_steps_budget: int,
    phase_metrics: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    """Build an aggregate energy summary when CodeCarbon data is available."""

    energy = 0.0
    emissions = 0.0
    duration = 0.0
    enabled = False
    output_files: list[str] = []

    for metrics in phase_metrics.values():
        if metrics.get("codecarbon_enabled"):
            enabled = True
        energy += _float_or_zero(metrics.get("codecarbon_energy_kwh"))
        emissions += _float_or_zero(metrics.get("codecarbon_emissions_kg"))
        duration += _float_or_zero(metrics.get("codecarbon_duration_seconds"))
        output_file = metrics.get("codecarbon_output_file")
        if output_file:
            output_files.append(str(output_file))

    if not enabled and energy == 0.0 and emissions == 0.0 and duration == 0.0:
        return None

    def safe_div(numerator: float, denominator: float) -> float | None:
        return None if denominator == 0 else numerator / denominator

    return {
        "contract_version": 2,
        "resource_scope": "training_and_kd_only",
        "periodic_evaluation_included": False,
        "posthoc_evaluation_included": False,
        "codecarbon_enabled": enabled,
        "energy_kwh": energy,
        "emissions_kg": emissions,
        "duration_seconds": duration,
        "energy_per_env_step": safe_div(energy, float(total_env_step_budget)),
        "energy_per_qnode_active_step": safe_div(energy, float(qnode_active_steps_budget)),
        "emissions_per_env_step": safe_div(emissions, float(total_env_step_budget)),
        "emissions_per_qnode_active_step": safe_div(
            emissions,
            float(qnode_active_steps_budget),
        ),
        "codecarbon_output_files": output_files,
    }
