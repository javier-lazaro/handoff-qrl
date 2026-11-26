"""Audit schedule-aware bundled experiment run directories."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hybrid_training_qrl.summaries import (
    DOWNSTREAM_SUMMARY,
    ENERGY_SUMMARY,
    KD_BRIDGE_SUMMARY,
    PHASE_SUMMARY,
    QUANTUM_SPEND_SUMMARY,
    SCHEDULE_SUMMARY,
    TRANSITION_SUMMARY,
    AUDIT_REPORT,
    write_json,
)

REQUIRED_ARTIFACTS = (SCHEDULE_SUMMARY, PHASE_SUMMARY, QUANTUM_SPEND_SUMMARY)
REQUIRED_SCHEDULE_FIELDS = {
    "schedule_name",
    "schedule_family",
    "total_env_step_budget",
    "qnode_active_steps_budget",
    "qnode_active_steps_ratio",
    "phase_count",
    "evaluation_protocol",
    "training_evaluation_enabled",
    "codecarbon_scope",
}
REQUIRED_QUANTUM_FIELDS = {
    "schedule_name",
    "qnode_calls_total",
    "qnode_time_total",
    "qnode_time_per_total_env_step",
    "qnode_time_per_qnode_active_step",
    "qnode_calls_by_phase",
    "qnode_time_by_phase",
    "qnode_calls_actor_total",
    "qnode_calls_critic_total",
    "qnode_time_actor_total",
    "qnode_time_critic_total",
    "qnode_calls_actor_by_phase",
    "qnode_calls_critic_by_phase",
    "qnode_time_actor_by_phase",
    "qnode_time_critic_by_phase",
    "resource_scope",
    "periodic_evaluation_included",
    "posthoc_evaluation_included",
    "training_wall_clock_total",
    "kd_dataset_collection_env_steps",
    "total_environment_interactions_accounted",
}
REQUIRED_PHASE_FIELDS = {
    "phase_id",
    "phase_name",
    "phase_type",
    "start_step",
    "end_step",
    "phase_budget_steps",
    "qnode_active",
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
}

REQUIRED_DOWNSTREAM_FIELDS = {
    "schedule_name",
    "downstream_mode",
    "downstream_cost_reported_separately",
    "representation_phase_id",
    "kd_phase_id",
    "downstream_ft_phase_id",
    "freeze_feature_extractor",
    "feature_interface_trainable_during_phase",
    "student_actor_head_path",
    "student_critic_head_path",
    "downstream_qnode_calls",
    "downstream_qnode_time",
    "downstream_qnode_calls_actor",
    "downstream_qnode_calls_critic",
}


@dataclass
class AuditResult:
    run_dir: Path
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "PASS" if self.ok else "FAIL",
            "ok": self.ok,
            "run_dir": str(self.run_dir),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


def write_audit_report(
    result: AuditResult,
    path: str | Path | None = None,
) -> Path:
    """Persist one audit result and return the report path."""

    report_path = Path(path) if path is not None else result.run_dir / AUDIT_REPORT
    write_json(report_path, result.to_dict())
    return report_path


def _read_json(path: Path, result: AuditResult) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001 - CLI audit should report any parse problem.
        result.add_error(f"Could not read {path.name}: {exc}")
        return {}
    if not isinstance(data, dict):
        result.add_error(f"{path.name} must contain a JSON object.")
        return {}
    return data


def _audit_json_fields(
    payload: dict[str, Any],
    *,
    required_fields: set[str],
    artifact_name: str,
    result: AuditResult,
) -> None:
    missing = sorted(field for field in required_fields if field not in payload)
    if missing:
        result.add_error(f"{artifact_name} is missing fields: {', '.join(missing)}")


def _audit_phase_summary(path: Path, result: AuditResult) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as exc:  # noqa: BLE001
        result.add_error(f"Could not read {path.name}: {exc}")
        return []

    if not rows:
        result.add_error(f"{path.name} must contain at least one phase row.")
        return []

    fieldnames = set(reader.fieldnames or [])
    missing = sorted(REQUIRED_PHASE_FIELDS - fieldnames)
    if missing:
        result.add_error(f"{path.name} is missing columns: {', '.join(missing)}")
    return rows




def _int_value(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def _float_value(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _close_enough(left: float, right: float, *, atol: float = 1e-9) -> bool:
    return abs(left - right) <= atol


def _audit_actor_critic_totals(
    *,
    quantum: dict[str, Any],
    phase_rows: list[dict[str, str]],
    result: AuditResult,
) -> None:
    calls_total = _int_value(quantum.get("qnode_calls_total"))
    calls_actor_total = _int_value(quantum.get("qnode_calls_actor_total"))
    calls_critic_total = _int_value(quantum.get("qnode_calls_critic_total"))
    if None not in {calls_total, calls_actor_total, calls_critic_total}:
        if calls_total != calls_actor_total + calls_critic_total:
            result.add_error(
                "quantum_spend_summary.json has inconsistent QNode call totals: "
                f"qnode_calls_total={calls_total}, actor+critic="
                f"{calls_actor_total + calls_critic_total}."
            )

    time_total = _float_value(quantum.get("qnode_time_total"))
    time_actor_total = _float_value(quantum.get("qnode_time_actor_total"))
    time_critic_total = _float_value(quantum.get("qnode_time_critic_total"))
    if None not in {time_total, time_actor_total, time_critic_total}:
        if not _close_enough(time_total, time_actor_total + time_critic_total):
            result.add_error(
                "quantum_spend_summary.json has inconsistent QNode time totals: "
                f"qnode_time_total={time_total}, actor+critic="
                f"{time_actor_total + time_critic_total}."
            )

    for row in phase_rows:
        phase_id = row.get("phase_id", "<unknown>")
        calls = _int_value(row.get("qnode_calls"))
        calls_actor = _int_value(row.get("qnode_calls_actor"))
        calls_critic = _int_value(row.get("qnode_calls_critic"))
        if None not in {calls, calls_actor, calls_critic}:
            if calls != calls_actor + calls_critic:
                result.add_error(
                    f"phase_summary.csv phase {phase_id} has inconsistent QNode calls: "
                    f"qnode_calls={calls}, actor+critic={calls_actor + calls_critic}."
                )

        qtime = _float_value(row.get("qnode_time"))
        qtime_actor = _float_value(row.get("qnode_time_actor"))
        qtime_critic = _float_value(row.get("qnode_time_critic"))
        if None not in {qtime, qtime_actor, qtime_critic}:
            if not _close_enough(qtime, qtime_actor + qtime_critic):
                result.add_error(
                    f"phase_summary.csv phase {phase_id} has inconsistent QNode time: "
                    f"qnode_time={qtime}, actor+critic={qtime_actor + qtime_critic}."
                )



def _audit_posthoc_only_training_contract(
    *,
    schedule: dict[str, Any],
    quantum: dict[str, Any],
    phase_rows: list[dict[str, str]],
    result: AuditResult,
) -> None:
    if schedule.get("evaluation_protocol") != "posthoc_only":
        result.add_error("schedule_summary.json must set evaluation_protocol='posthoc_only'.")
    if schedule.get("training_evaluation_enabled") is not False:
        result.add_error("schedule_summary.json must set training_evaluation_enabled=false.")
    allowed_codecarbon_scopes = {
        "training_and_kd_only",
        "disabled_engineering_only",
        "disabled_parallel_performance_campaign",
    }
    if schedule.get("codecarbon_scope") not in allowed_codecarbon_scopes:
        result.add_error(
            "schedule_summary.json has an invalid codecarbon_scope; expected one of "
            f"{sorted(allowed_codecarbon_scopes)}."
        )

    if quantum.get("resource_scope") != "training_and_kd_only":
        result.add_error("quantum_spend_summary.json has an invalid resource_scope.")
    if quantum.get("periodic_evaluation_included") is not False:
        result.add_error("quantum_spend_summary.json must exclude periodic evaluation.")
    if quantum.get("posthoc_evaluation_included") is not False:
        result.add_error("quantum_spend_summary.json must exclude post-hoc evaluation.")

    for row in phase_rows:
        phase_id = row.get("phase_id", "<unknown>")
        count = _int_value(row.get("training_evaluation_count"))
        artifact = str(row.get("training_evaluation_artifact_present", "")).lower()
        contract_ok = str(row.get("training_evaluation_contract_ok", "")).lower()
        if count not in {0, None}:
            result.add_error(
                f"phase_summary.csv phase {phase_id} reports {count} in-training evaluations."
            )
        if artifact not in {"false", "0"}:
            result.add_error(
                f"phase_summary.csv phase {phase_id} contains an in-training evaluation artifact."
            )
        if contract_ok not in {"true", "1"}:
            result.add_error(
                f"phase_summary.csv phase {phase_id} violates the post-hoc-only contract."
            )


def _audit_downstream_summary(
    *,
    run_dir: Path,
    result: AuditResult,
) -> None:
    path = run_dir / DOWNSTREAM_SUMMARY
    if not path.exists():
        result.add_error(f"Missing required downstream artifact: {DOWNSTREAM_SUMMARY}")
        return

    payload = _read_json(path, result)
    _audit_json_fields(
        payload,
        required_fields=REQUIRED_DOWNSTREAM_FIELDS,
        artifact_name=DOWNSTREAM_SUMMARY,
        result=result,
    )

    if payload.get("downstream_mode") != "policy_value_kd_plus_frozen_ft":
        result.add_error(
            "downstream_summary.json has unexpected downstream_mode: "
            f"{payload.get('downstream_mode')!r}."
        )
    if payload.get("freeze_feature_extractor") is not True:
        result.add_error("downstream_summary.json must set freeze_feature_extractor=true.")
    if payload.get("feature_interface_trainable_during_phase") is not False:
        result.add_error(
            "downstream_summary.json must set "
            "feature_interface_trainable_during_phase=false."
        )
    if payload.get("downstream_cost_reported_separately") is not True:
        result.add_error(
            "downstream_summary.json must set downstream_cost_reported_separately=true."
        )

    calls = _int_value(payload.get("downstream_qnode_calls"))
    calls_actor = _int_value(payload.get("downstream_qnode_calls_actor"))
    calls_critic = _int_value(payload.get("downstream_qnode_calls_critic"))
    if None not in {calls, calls_actor, calls_critic}:
        if calls != calls_actor + calls_critic:
            result.add_error(
                "downstream_summary.json has inconsistent downstream QNode calls: "
                f"downstream_qnode_calls={calls}, actor+critic={calls_actor + calls_critic}."
            )

    for field_name in ("student_actor_head_path", "student_critic_head_path"):
        value = payload.get(field_name)
        if isinstance(value, str) and value:
            if not Path(value).exists():
                result.add_warning(f"{DOWNSTREAM_SUMMARY}.{field_name} does not exist on this machine: {value}")


def _audit_checkpoint_continuation(
    *,
    schedule: dict[str, Any],
    phase_rows: list[dict[str, str]],
    result: AuditResult,
) -> None:
    continuation_rows = [
        row for row in phase_rows if row.get("initialization_mode") == "checkpoint_resume"
    ]
    if not continuation_rows:
        return
    if len(continuation_rows) != 1:
        result.add_error(
            "Checkpoint continuation schedules currently require exactly one resumed PPO phase."
        )
        return

    row = continuation_rows[0]
    required = (
        "continuation_source_checkpoint",
        "continuation_source_global_step",
        "continuation_seed",
        "continuation_final_global_step",
        "continuation_target_global_step",
        "continuation_target_reached",
        "continuation_optimizer_state_restored",
        "continuation_rng_state_restored",
    )
    for field_name in required:
        if row.get(field_name) in {None, ""}:
            result.add_error(
                f"phase_summary.csv continuation row is missing {field_name}."
            )

    source_step = _int_value(row.get("continuation_source_global_step"))
    final_step = _int_value(row.get("continuation_final_global_step"))
    target_step = _int_value(row.get("continuation_target_global_step"))
    effective_steps = _int_value(row.get("phase_effective_env_steps"))
    if None not in {source_step, final_step, effective_steps}:
        if final_step != source_step + effective_steps:
            result.add_error(
                "Continuation global-step accounting is inconsistent: "
                f"final={final_step}, source+effective={source_step + effective_steps}."
            )
    if None not in {final_step, target_step} and final_step != target_step:
        result.add_error(
            "Continuation did not reach its declared cumulative target: "
            f"final={final_step}, target={target_step}."
        )

    if str(row.get("continuation_target_reached")).lower() != "true":
        result.add_error("Continuation phase must report continuation_target_reached=true.")
    if str(row.get("continuation_optimizer_state_restored")).lower() != "true":
        result.add_error("Continuation phase must restore optimizer state.")
    if str(row.get("continuation_rng_state_restored")).lower() != "false":
        result.add_error(
            "Continuation provenance must explicitly report RNG state as not restored."
        )

    cumulative_target = _int_value(schedule.get("cumulative_env_step_target"))
    if cumulative_target is not None and target_step is not None:
        if cumulative_target != target_step:
            result.add_error(
                "schedule_summary.json cumulative_env_step_target does not match the "
                f"continuation target ({cumulative_target} vs {target_step})."
            )

    checkpoint = row.get("continuation_source_checkpoint")
    if checkpoint and not Path(str(checkpoint)).exists():
        result.add_warning(
            "Continuation source checkpoint no longer exists on this machine: "
            f"{checkpoint}"
        )


D1_DIAGNOSTIC_MANIFEST = "safe_finetune_d1_manifest.json"
D2_DIAGNOSTIC_MANIFEST = "safe_finetune_d2_manifest.json"
D3_DIAGNOSTIC_MANIFEST = "safe_finetune_d3_manifest.json"
D4A_DIAGNOSTIC_MANIFEST = "safe_finetune_d4a_manifest.json"


def _audit_safe_online_reentry_diagnostic(
    *,
    run_dir: Path,
    schedule: dict[str, Any],
    result: AuditResult,
) -> None:
    """Audit incremental safe-reentry provenance without full Q50 summaries."""

    gate = str(schedule.get("diagnostic_gate", ""))
    if gate == "D1":
        artifact = D1_DIAGNOSTIC_MANIFEST
        expected_scope = "one_rollout_one_update_safe_online_reentry"
        expected_resource_scope = "incremental_gate_d1_training_only"
        expected_steps = 16384
    elif gate == "D2":
        artifact = D2_DIAGNOSTIC_MANIFEST
        expected_scope = "four_rollout_short_horizon_safe_online_reentry"
        expected_resource_scope = "incremental_gate_d2_training_only"
        expected_steps = 65536
    elif gate == "D3":
        artifact = D3_DIAGNOSTIC_MANIFEST
        expected_scope = "medium_horizon_continuation_rollout4_to_rollout16"
        expected_resource_scope = "incremental_gate_d3_rollouts5_to16_only"
        expected_steps = 196608
    elif gate == "D4A":
        artifact = D4A_DIAGNOSTIC_MANIFEST
        expected_scope = "medium_horizon_update_epochs2_rollout4_to_rollout16"
        expected_resource_scope = "incremental_gate_d4a_rollouts5_to16_only"
        expected_steps = 196608
    else:
        result.add_error(
            "Safe online re-entry diagnostics must declare diagnostic_gate as D1, D2, D3 or D4A."
        )
        return

    path = run_dir / artifact
    if not path.exists():
        result.add_error(f"Missing required safe-reentry diagnostic artifact: {artifact}")
        return

    payload = _read_json(path, result)
    required = {
        "gate",
        "scope",
        "candidate",
        "hybrid_environment_steps",
        "source_kd_manifest",
        "source_kd_manifest_sha256",
        "source_teacher_checkpoint",
        "source_teacher_checkpoint_sha256",
        "source_actor_head",
        "source_actor_head_sha256",
        "source_critic_head",
        "source_critic_head_sha256",
        "final_checkpoint",
        "resource_scope",
        "posthoc_required",
    }
    if gate in {"D2", "D3", "D4A"}:
        required.update({"milestone_rollouts", "milestone_checkpoints", "phase_run_dir"})
    if gate in {"D3", "D4A"}:
        required.update(
            {
                "source_d2_manifest",
                "source_d2_manifest_sha256",
                "source_d2_rollout4_checkpoint",
                "source_d2_rollout4_checkpoint_sha256",
                "source_cumulative_online_rollout",
                "additional_rollouts",
                "cumulative_online_rollout_target",
                "optimizer_state_restored",
                "rng_state_restored",
                "continuation_seed",
            }
        )
    _audit_json_fields(
        payload,
        required_fields=required,
        artifact_name=artifact,
        result=result,
    )
    if payload.get("gate") != gate:
        result.add_error(f"{artifact} must set gate={gate!r}.")
    if payload.get("scope") != expected_scope:
        result.add_error(f"{artifact} has an invalid scope.")
    if payload.get("resource_scope") != expected_resource_scope:
        result.add_error(f"{artifact} has an invalid resource_scope.")
    if payload.get("posthoc_required") is not True:
        result.add_error(f"{artifact} must require post-hoc evaluation.")

    declared_steps = _int_value(payload.get("hybrid_environment_steps"))
    schedule_steps = _int_value(schedule.get("total_env_steps_effective"))
    if declared_steps != expected_steps:
        result.add_error(
            f"{artifact} must declare hybrid_environment_steps={expected_steps}."
        )
    if None not in {declared_steps, schedule_steps} and declared_steps != schedule_steps:
        result.add_error(
            f"{artifact} hybrid_environment_steps does not match schedule_summary.json "
            f"({declared_steps} vs {schedule_steps})."
        )

    checkpoint = payload.get("final_checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        result.add_error(f"{artifact} has no final_checkpoint.")
    elif not Path(checkpoint).exists():
        result.add_error(f"{artifact} final_checkpoint does not exist: {checkpoint}")

    if gate in {"D2", "D3", "D4A"}:
        expected_milestones = [1, 2, 4] if gate == "D2" else [8, 16]
        milestones = payload.get("milestone_rollouts")
        if milestones != expected_milestones:
            result.add_error(
                f"{artifact} must declare milestone_rollouts={expected_milestones}."
            )
        milestone_checkpoints = payload.get("milestone_checkpoints")
        if not isinstance(milestone_checkpoints, dict):
            result.add_error(f"{artifact}.milestone_checkpoints must be a mapping.")
        else:
            for rollout in expected_milestones:
                milestone = milestone_checkpoints.get(str(rollout))
                if not isinstance(milestone, str) or not milestone:
                    result.add_error(
                        f"{artifact} is missing the rollout-{rollout} checkpoint path."
                    )
                elif not Path(milestone).exists():
                    result.add_error(
                        f"{artifact} rollout-{rollout} checkpoint does not exist: {milestone}"
                    )

    if gate == "D3" and payload.get("diagnostic_variant") == "staged_actor_lr_increase":
        if payload.get("optimizer_learning_rates_overridden_after_resume") is not True:
            result.add_error(
                f"{artifact} must record the staged optimizer LR override."
            )
        expected_lrs = {
            "source_actor_learning_rate": 2e-4,
            "actor_learning_rate": 5e-4,
            "critic_learning_rate": 2e-4,
            "shared_learning_rate": 2e-4,
        }
        for field, expected in expected_lrs.items():
            value = payload.get(field)
            try:
                actual = float(value)
            except (TypeError, ValueError):
                result.add_error(f"{artifact} must declare numeric {field}.")
                continue
            if abs(actual - expected) > 1e-12:
                result.add_error(
                    f"{artifact} {field} must equal {expected}, got {actual}."
                )

    if gate in {"D3", "D4A"}:
        if payload.get("source_cumulative_online_rollout") != 4:
            result.add_error(f"{artifact} must resume from cumulative online rollout 4.")
        if payload.get("additional_rollouts") != 12:
            result.add_error(f"{artifact} must add exactly 12 rollouts.")
        if payload.get("cumulative_online_rollout_target") != 16:
            result.add_error(f"{artifact} must target cumulative online rollout 16.")
        if payload.get("optimizer_state_restored") is not True:
            result.add_error(f"{artifact} must restore optimizer state.")
        if payload.get("rng_state_restored") is not False:
            result.add_error(f"{artifact} must explicitly report rng_state_restored=false.")
        for field in ("source_d2_manifest", "source_d2_rollout4_checkpoint"):
            value = payload.get(field)
            if isinstance(value, str) and value and not Path(value).exists():
                result.add_error(f"{artifact} {field} does not exist: {value}")

    if gate == "D4A":
        if payload.get("diagnostic_variant") != "update_epochs_reduction":
            result.add_error(f"{artifact} must identify the update-epochs intervention.")
        if payload.get("update_epochs") != 2 or payload.get("control_update_epochs") != 4:
            result.add_error(f"{artifact} must declare update_epochs 2 versus control 4.")
        if payload.get("optimizer_state_restored") is not True:
            result.add_error(f"{artifact} must restore optimizer state.")
        if payload.get("optimizer_learning_rates_overridden_after_resume") is not False:
            result.add_error(f"{artifact} must not override restored learning rates.")
        if payload.get("optimizer_step_count") != 6144:
            result.add_error(f"{artifact} must declare 6,144 optimizer steps.")
        if payload.get("control_optimizer_step_count") != 12288:
            result.add_error(f"{artifact} must declare 12,288 D3A control optimizer steps.")
        if payload.get("anchor_policy_probe_required") is not True:
            result.add_error(f"{artifact} must require the post-hoc anchor-policy probe.")


def _audit_policy_value_kd_summary(
    *,
    run_dir: Path,
    result: AuditResult,
) -> None:
    path = run_dir / KD_BRIDGE_SUMMARY
    if not path.exists():
        return
    payload = _read_json(path, result)
    if payload.get("bridge_type") != "policy_value_kd":
        result.add_error(
            "kd_bridge_summary.json must use bridge_type='policy_value_kd' for supported KD schedules."
        )
    for field_name in (
        "student_actor_head_path",
        "student_critic_head_path",
        "kd_value_loss_final",
        "kd_policy_loss_final",
        "qnode_calls_actor",
        "qnode_calls_critic",
    ):
        if field_name not in payload or payload.get(field_name) in {None, ""}:
            result.add_error(
                f"kd_bridge_summary.json is missing policy-value field: {field_name}"
            )

def audit_run_dir(
    run_dir: str | Path,
    *,
    require_energy: bool = False,
    strict_optional: bool = False,
) -> AuditResult:
    """Audit one schedule-aware run directory."""

    run_dir = Path(run_dir)
    result = AuditResult(run_dir=run_dir)

    if not run_dir.exists():
        result.add_error(f"Run directory does not exist: {run_dir}")
        return result

    for artifact in REQUIRED_ARTIFACTS:
        if not (run_dir / artifact).exists():
            result.add_error(f"Missing required artifact: {artifact}")

    if not result.ok:
        return result

    schedule = _read_json(run_dir / SCHEDULE_SUMMARY, result)
    quantum = _read_json(run_dir / QUANTUM_SPEND_SUMMARY, result)
    phase_rows = _audit_phase_summary(run_dir / PHASE_SUMMARY, result)

    _audit_json_fields(
        schedule,
        required_fields=REQUIRED_SCHEDULE_FIELDS,
        artifact_name=SCHEDULE_SUMMARY,
        result=result,
    )
    _audit_json_fields(
        quantum,
        required_fields=REQUIRED_QUANTUM_FIELDS,
        artifact_name=QUANTUM_SPEND_SUMMARY,
        result=result,
    )

    phase_count = schedule.get("phase_count")
    if isinstance(phase_count, int) and phase_rows and phase_count != len(phase_rows):
        result.add_error(
            f"schedule_summary.json phase_count={phase_count}, "
            f"but phase_summary.csv has {len(phase_rows)} rows."
        )

    _audit_actor_critic_totals(
        quantum=quantum,
        phase_rows=phase_rows,
        result=result,
    )
    _audit_posthoc_only_training_contract(
        schedule=schedule,
        quantum=quantum,
        phase_rows=phase_rows,
        result=result,
    )
    _audit_checkpoint_continuation(
        schedule=schedule,
        phase_rows=phase_rows,
        result=result,
    )

    schedule_name = str(schedule.get("schedule_name", ""))
    schedule_family = str(schedule.get("schedule_family", ""))
    transition_type = schedule.get("transition_type")
    bridge_type = schedule.get("bridge_type")

    # Gate D1 retains the source Q50 scientific identity, but it is an
    # incremental one-update diagnostic rather than a complete delayed schedule.
    # Its source KD/transition provenance lives in a dedicated manifest.
    incremental_diagnostic = schedule_family == "safe_online_reentry_diagnostic"
    if incremental_diagnostic:
        _audit_safe_online_reentry_diagnostic(
            run_dir=run_dir,
            schedule=schedule,
            result=result,
        )

    transition_expected = not incremental_diagnostic and (
        schedule_name.startswith("delayed_hybrid_")
        or schedule_name.startswith("delayed_classical_")
        or schedule_name == "downstream_hybrid_adaptation"
        or transition_type not in {None, "", "none"}
    )
    kd_expected = not incremental_diagnostic and (
        "kd" in schedule_name or bridge_type not in {None, "", "none"}
    )

    transition_required = (
        schedule_name.startswith("delayed_hybrid_")
        or schedule_name.startswith("delayed_classical_")
        or schedule_name == "downstream_hybrid_adaptation"
    ) and not incremental_diagnostic
    kd_required = (
        "kd" in schedule_name or bridge_type not in {None, "", "none"}
    ) and not incremental_diagnostic

    if transition_expected and not (run_dir / TRANSITION_SUMMARY).exists():
        message = f"Expected transition artifact for {schedule_name}: {TRANSITION_SUMMARY}"
        if transition_required or strict_optional:
            result.add_error(message)
        else:
            result.add_warning(message)

    if kd_expected and not (run_dir / KD_BRIDGE_SUMMARY).exists():
        message = f"Expected KD bridge artifact for {schedule_name}: {KD_BRIDGE_SUMMARY}"
        if kd_required or strict_optional:
            result.add_error(message)
        else:
            result.add_warning(message)

    if kd_required and (run_dir / KD_BRIDGE_SUMMARY).exists():
        _audit_policy_value_kd_summary(run_dir=run_dir, result=result)

    if schedule_name.startswith("delayed_classical_"):
        qnode_calls = _int_value(quantum.get("qnode_calls_total"))
        qnode_time = _float_value(quantum.get("qnode_time_total"))
        if qnode_calls not in {None, 0}:
            result.add_error(
                f"{schedule_name} must record zero QNode calls, got {qnode_calls}."
            )
        if qnode_time is not None and abs(qnode_time) > 1e-12:
            result.add_error(
                f"{schedule_name} must record zero QNode time, got {qnode_time}."
            )

    if schedule_name == "downstream_hybrid_adaptation":
        _audit_downstream_summary(run_dir=run_dir, result=result)


    effective_total = _int_value(schedule.get("total_env_steps_effective"))
    quantum_effective_total = _int_value(quantum.get("total_env_steps_effective"))
    if effective_total is not None and quantum_effective_total is not None:
        if effective_total != quantum_effective_total:
            result.add_error(
                "Effective environment-step totals differ between schedule_summary.json "
                f"({effective_total}) and quantum_spend_summary.json "
                f"({quantum_effective_total})."
            )

    if effective_total is not None and phase_rows:
        phase_effective_total = sum(
            _int_value(row.get("phase_effective_env_steps")) or 0 for row in phase_rows
        )
        if effective_total != phase_effective_total:
            result.add_error(
                "schedule_summary.json total_env_steps_effective does not match "
                f"phase_summary.csv ({effective_total} vs {phase_effective_total})."
            )

    energy_path = run_dir / ENERGY_SUMMARY
    energy_required = require_energy or schedule.get("codecarbon_scope") == "training_and_kd_only"
    if energy_required and not energy_path.exists():
        result.add_error(f"Missing required energy artifact: {ENERGY_SUMMARY}")
    elif energy_path.exists():
        energy = _read_json(energy_path, result)
        if energy.get("resource_scope") != "training_and_kd_only":
            result.add_error("energy_summary.json has an invalid resource_scope.")
        if energy.get("periodic_evaluation_included") is not False:
            result.add_error("energy_summary.json must exclude periodic evaluation.")
        if energy.get("posthoc_evaluation_included") is not False:
            result.add_error("energy_summary.json must exclude post-hoc evaluation.")

    return result
