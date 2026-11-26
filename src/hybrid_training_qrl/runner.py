"""Schedule-aware experiment runner for the bundled experiment implementation.

The repository owns paper-side orchestration, while SimplyQRL executes concrete
PPO phases. Maintained schedules use real policy/value KD bridges; historical
accounting-only placeholders live exclusively under ``archive/``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hybrid_training_qrl.audit import audit_run_dir, write_audit_report
from hybrid_training_qrl.composed_finetune import run_composed_policy_value_finetune_phase
from hybrid_training_qrl.downstream import build_downstream_summary
from hybrid_training_qrl.kd_bridge import run_offline_policy_value_kd_bridge
from hybrid_training_qrl.phase_metrics import build_energy_summary, collect_phase_metrics
from hybrid_training_qrl.schedules import PhaseSpec, ScheduleSpec, load_schedule_spec
from hybrid_training_qrl.simplyqrl_adapter import (
    resolve_phase_resume_checkpoint,
    run_ppo_phase,
)
from hybrid_training_qrl.summaries import build_quantum_spend_summary, write_schedule_artifacts
from hybrid_training_qrl.transition import build_kd_bridge_summary, build_transition_summary

SUPPORTED_SINGLE_PHASE_SCHEDULES = {
    "classical_warmup",
    "compact_classical_c0",
    "immediate_hybrid_q100",
}
SUPPORTED_DOWNSTREAM_SCHEDULES = {"downstream_hybrid_adaptation"}
SUPPORTED_DELAYED_HYBRID_POLICY_VALUE_KD_SCHEDULES = {
    "delayed_hybrid_q25_kd_bridge",
    "delayed_hybrid_q50_kd_bridge",
    "delayed_hybrid_q75_kd_bridge",
}
SUPPORTED_DELAYED_CLASSICAL_POLICY_VALUE_KD_SCHEDULES = {
    "delayed_classical_q25_kd_bridge",
    "delayed_classical_q50_kd_bridge",
    "delayed_classical_q75_kd_bridge",
}
SUPPORTED_DELAYED_POLICY_VALUE_KD_SCHEDULES = (
    SUPPORTED_DELAYED_HYBRID_POLICY_VALUE_KD_SCHEDULES
    | SUPPORTED_DELAYED_CLASSICAL_POLICY_VALUE_KD_SCHEDULES
)
SUPPORTED_SCHEDULES = (
    SUPPORTED_SINGLE_PHASE_SCHEDULES
    | SUPPORTED_DELAYED_POLICY_VALUE_KD_SCHEDULES
    | SUPPORTED_DOWNSTREAM_SCHEDULES
)
PPO_PHASE_TYPES = {"classical_ppo", "hybrid_ppo"}


@dataclass(frozen=True)
class ScheduleRunResult:
    run_dir: Path
    phase_run_dirs: dict[int, Path]
    dry_run: bool = False


def _slug(value: str | None) -> str:
    value = value or "unknown"
    safe = []
    for char in value.lower():
        if char.isalnum():
            safe.append(char)
        elif char in {"-", "_", "."}:
            safe.append("_")
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "unknown"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve_phase_config(schedule_config_path: Path, phase: PhaseSpec) -> Path:
    if not phase.config:
        raise ValueError(
            f"Phase {phase.phase_id} ({phase.phase_name}) does not define a SimplyQRL config."
        )
    config_path = Path(phase.config)
    if not config_path.is_absolute():
        # Prefer paths relative to the repository root/current working directory.
        # Fall back to schedule-relative paths for hand-written local configs.
        cwd_path = Path.cwd() / config_path
        if cwd_path.exists():
            return cwd_path.resolve()
        config_path = schedule_config_path.parent / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Phase config not found: {phase.config}")
    return config_path.resolve()


def _validate_supported_schedule(schedule: ScheduleSpec) -> None:
    if schedule.schedule_name not in SUPPORTED_SCHEDULES:
        supported = ", ".join(sorted(SUPPORTED_SCHEDULES))
        raise NotImplementedError(
            f"Schedule {schedule.schedule_name!r} is not supported by the maintained runner. "
            f"Supported schedules: {supported}."
        )

    if schedule.schedule_name in SUPPORTED_SINGLE_PHASE_SCHEDULES:
        if len(schedule.phases) != 1:
            raise NotImplementedError(
                f"Schedule {schedule.schedule_name!r} has {len(schedule.phases)} phases. "
                "This schedule is expected to be single-phase."
            )
        phase = schedule.phases[0]
        if phase.phase_type not in PPO_PHASE_TYPES:
            raise NotImplementedError(
                f"Unsupported phase_type {phase.phase_type!r}. "
                "Single-phase schedules support only classical_ppo and hybrid_ppo."
            )
        initialization = str(phase.metadata.get("initialization", "fresh"))
        if initialization not in {"fresh", "fresh_hybrid", "checkpoint_resume", ""}:
            raise ValueError(
                f"Unsupported single-phase PPO initialization: {initialization!r}."
            )
        if initialization == "checkpoint_resume":
            if not phase.metadata.get("resume_from_run_dir") and not phase.metadata.get(
                "resume_from_checkpoint"
            ):
                raise ValueError(
                    "checkpoint_resume requires resume_from_run_dir or "
                    "resume_from_checkpoint."
                )
            if phase.metadata.get("preserve_optimizer_state") is not True:
                raise ValueError(
                    "checkpoint_resume phases must explicitly set "
                    "preserve_optimizer_state=true."
                )
        return

    if schedule.schedule_name in SUPPORTED_DELAYED_POLICY_VALUE_KD_SCHEDULES:
        _validate_delayed_policy_value_kd_schedule(schedule)
        return

    if schedule.schedule_name in SUPPORTED_DOWNSTREAM_SCHEDULES:
        _validate_downstream_hybrid_adaptation_schedule(schedule)
        return

    raise AssertionError(f"Unhandled supported schedule: {schedule.schedule_name}")


def _validate_declared_budgets(schedule: ScheduleSpec) -> None:
    phase_steps_total = sum(phase.steps for phase in schedule.phases)
    if phase_steps_total != schedule.total_env_step_budget:
        raise ValueError(
            f"{schedule.schedule_name} declares total_env_step_budget="
            f"{schedule.total_env_step_budget}, but phase steps sum to {phase_steps_total}."
        )

    qnode_active_step_budget = sum(
        phase.steps for phase in schedule.phases if phase.qnode_active and phase.steps > 0
    )
    if qnode_active_step_budget != schedule.qnode_active_steps_budget:
        raise ValueError(
            f"{schedule.schedule_name} declares qnode_active_steps_budget="
            f"{schedule.qnode_active_steps_budget}, but qnode-active PPO phase steps "
            f"sum to {qnode_active_step_budget}."
        )


def _head_family_from_phase_metadata(phase: PhaseSpec, key: str) -> str:
    config = phase.metadata.get(key)
    if not isinstance(config, dict):
        return ""
    return str(config.get("family") or config.get("type") or "").strip().lower()


def _validate_delayed_policy_value_kd_schedule(schedule: ScheduleSpec) -> None:
    is_hybrid = schedule.schedule_name in SUPPORTED_DELAYED_HYBRID_POLICY_VALUE_KD_SCHEDULES
    is_classical = schedule.schedule_name in SUPPORTED_DELAYED_CLASSICAL_POLICY_VALUE_KD_SCHEDULES
    if not (is_hybrid or is_classical):
        raise AssertionError(f"Unhandled delayed schedule: {schedule.schedule_name}")

    target_kind = "hybrid" if is_hybrid else "classical"
    target_phase_type = "hybrid_ppo" if is_hybrid else "classical_ppo"

    if len(schedule.phases) != 3:
        raise NotImplementedError(
            f"{schedule.schedule_name} expects exactly 3 phases: "
            f"classical_ppo, kd_bridge, {target_phase_type}."
        )

    _validate_declared_budgets(schedule)

    phase_types = [phase.phase_type for phase in schedule.phases]
    expected = ["classical_ppo", "kd_bridge", target_phase_type]
    if phase_types != expected:
        raise NotImplementedError(
            f"{schedule.schedule_name} expects phase types {expected}, got {phase_types}."
        )

    classical_phase, kd_phase, target_phase = schedule.phases

    if classical_phase.qnode_active:
        raise ValueError(
            f"{schedule.schedule_name} phase 1 classical_warmup must have qnode_active=false."
        )

    expected_kd_qnode = is_hybrid
    expected_online_qnode = is_hybrid
    if kd_phase.qnode_active is not expected_kd_qnode:
        raise ValueError(
            f"{schedule.schedule_name} phase 2 kd_bridge qnode_active must be "
            f"{str(expected_kd_qnode).lower()} for a delayed {target_kind} target."
        )
    if target_phase.qnode_active is not expected_online_qnode:
        raise ValueError(
            f"{schedule.schedule_name} phase 3 qnode_active must be "
            f"{str(expected_online_qnode).lower()} for a delayed {target_kind} target."
        )

    if classical_phase.steps <= 0:
        raise ValueError(f"{schedule.schedule_name} phase 1 classical_warmup must have steps > 0.")
    if kd_phase.steps != 0:
        raise ValueError(f"{schedule.schedule_name} phase 2 kd_bridge must have steps = 0.")
    if target_phase.steps <= 0:
        raise ValueError(f"{schedule.schedule_name} phase 3 finetune must have steps > 0.")

    for phase in (classical_phase, target_phase):
        if not phase.config:
            raise ValueError(
                f"{schedule.schedule_name} PPO phase {phase.phase_id} "
                f"({phase.phase_name}) requires a config path."
            )
    if kd_phase.config:
        raise ValueError(
            f"{schedule.schedule_name} kd_bridge does not use a SimplyQRL PPO config path."
        )

    bridge_mode = str(kd_phase.metadata.get("bridge_mode", ""))
    if bridge_mode != "offline_policy_value_kd_micro":
        raise ValueError(
            f"{schedule.schedule_name} requires bridge_mode='offline_policy_value_kd_micro', "
            f"got {bridge_mode!r}."
        )

    required = {
        "kd_samples",
        "kd_epochs",
        "kd_batch_size",
        "student_actor_head",
        "student_critic_head",
    }
    missing = sorted(key for key in required if key not in kd_phase.metadata)
    if missing:
        raise ValueError(
            "offline_policy_value_kd_micro phase is missing required metadata: "
            + ", ".join(missing)
        )

    for key in ("student_actor_head", "student_critic_head"):
        family = _head_family_from_phase_metadata(kd_phase, key)
        if family != target_kind:
            raise ValueError(
                f"{schedule.schedule_name} {key} must declare family/type={target_kind!r}; "
                f"got {family!r}."
            )

    initialization = str(target_phase.metadata.get("initialization", ""))
    if initialization != "kd_composed_policy_value":
        raise ValueError(
            f"{schedule.schedule_name} requires phase 3 "
            "initialization=kd_composed_policy_value."
        )

    if bool(target_phase.metadata.get("freeze_feature_extractor", False)):
        raise ValueError(
            f"{schedule.schedule_name} is a delayed end-to-end schedule; "
            "phase 3 must keep freeze_feature_extractor=false. "
            "Frozen feature interfaces are reserved for downstream schedules."
        )

    if not bool(target_phase.metadata.get("rebuild_optimizer", True)):
        raise ValueError(f"{schedule.schedule_name} phase 3 requires rebuild_optimizer=true.")

    if is_classical and schedule.qnode_active_steps_budget != 0:
        raise ValueError(
            f"{schedule.schedule_name} is a no-QNode attribution control and must declare "
            "qnode_active_steps_budget=0."
        )



def _validate_downstream_hybrid_adaptation_schedule(schedule: ScheduleSpec) -> None:
    if len(schedule.phases) != 3:
        raise NotImplementedError(
            f"{schedule.schedule_name} expects exactly 3 phases: "
            "classical_ppo, kd_bridge, hybrid_ppo."
        )

    _validate_declared_budgets(schedule)

    phase_types = [phase.phase_type for phase in schedule.phases]
    expected = ["classical_ppo", "kd_bridge", "hybrid_ppo"]
    if phase_types != expected:
        raise NotImplementedError(
            f"{schedule.schedule_name} expects phase types {expected}, got {phase_types}."
        )

    representation_phase, kd_phase, downstream_phase = schedule.phases

    if representation_phase.qnode_active:
        raise ValueError(
            f"{schedule.schedule_name} phase 1 representation/classical phase must have "
            "qnode_active=false."
        )
    if not kd_phase.qnode_active:
        raise ValueError(f"{schedule.schedule_name} phase 2 kd_bridge must have qnode_active=true.")
    if not downstream_phase.qnode_active:
        raise ValueError(
            f"{schedule.schedule_name} phase 3 downstream_finetune must have qnode_active=true."
        )

    if representation_phase.steps <= 0:
        raise ValueError(f"{schedule.schedule_name} phase 1 must have steps > 0.")
    if kd_phase.steps != 0:
        raise ValueError(f"{schedule.schedule_name} phase 2 kd_bridge must have steps = 0.")
    if downstream_phase.steps <= 0:
        raise ValueError(f"{schedule.schedule_name} phase 3 downstream_finetune must have steps > 0.")

    for phase in (representation_phase, downstream_phase):
        if not phase.config:
            raise ValueError(
                f"{schedule.schedule_name} PPO phase {phase.phase_id} "
                f"({phase.phase_name}) requires a config path."
            )
    if kd_phase.config:
        raise ValueError(f"{schedule.schedule_name} kd_bridge does not use a SimplyQRL PPO config path.")

    bridge_mode = str(kd_phase.metadata.get("bridge_mode", ""))
    if bridge_mode != "offline_policy_value_kd_micro":
        raise ValueError(
            f"{schedule.schedule_name} requires bridge_mode='offline_policy_value_kd_micro', "
            f"got {bridge_mode!r}."
        )

    required = {
        "kd_samples",
        "kd_epochs",
        "kd_batch_size",
        "student_actor_head",
        "student_critic_head",
    }
    missing = sorted(key for key in required if key not in kd_phase.metadata)
    if missing:
        raise ValueError(
            "offline_policy_value_kd_micro phase is missing required metadata: "
            + ", ".join(missing)
        )

    initialization = str(downstream_phase.metadata.get("initialization", ""))
    if initialization != "kd_composed_policy_value":
        raise ValueError(
            f"{schedule.schedule_name} requires phase 3 "
            "initialization=kd_composed_policy_value."
        )

    if not bool(downstream_phase.metadata.get("freeze_feature_extractor", False)):
        raise ValueError(
            f"{schedule.schedule_name} is a downstream schedule; phase 3 must set "
            "freeze_feature_extractor=true."
        )

    if bool(downstream_phase.metadata.get("feature_interface_trainable_during_phase", True)):
        raise ValueError(
            f"{schedule.schedule_name} phase 3 must set "
            "feature_interface_trainable_during_phase=false."
        )

    if not bool(downstream_phase.metadata.get("rebuild_optimizer", True)):
        raise ValueError(f"{schedule.schedule_name} phase 3 requires rebuild_optimizer=true.")

def _phase_run_name(schedule: ScheduleSpec, phase: PhaseSpec) -> str:
    return f"{_slug(schedule.schedule_name)}_phase{phase.phase_id}_{_slug(phase.phase_name)}"


def _build_parent_run_dir(
    *,
    output_root: Path,
    schedule: ScheduleSpec,
    run_name: str | None,
) -> Path:
    if run_name:
        return output_root / run_name
    return output_root / f"{_timestamp()}_{_slug(schedule.schedule_name)}_{_slug(schedule.env_id)}"


def _phase_parent_dir(run_dir: Path, phase: PhaseSpec) -> Path:
    return run_dir / "phases" / f"phase_{phase.phase_id}_{_slug(phase.phase_name)}"


def _phase_rows_with_metrics(
    schedule: ScheduleSpec,
    phase_metrics: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    phase_rows = []
    for base_row in schedule.default_phase_rows():
        metrics = phase_metrics.get(int(base_row["phase_id"]), {})
        merged = dict(base_row)
        merged.update(metrics)
        phase_rows.append(merged)
    return phase_rows


def _run_ppo_schedule_phase(
    *,
    schedule_config_path: Path,
    schedule: ScheduleSpec,
    phase: PhaseSpec,
    run_dir: Path,
) -> Path:
    phase_config = _resolve_phase_config(schedule_config_path, phase)
    phase_parent = _phase_parent_dir(run_dir, phase)
    phase_parent.mkdir(parents=True, exist_ok=True)
    return run_ppo_phase(
        config_path=phase_config,
        phase=phase,
        phase_output_dir=phase_parent,
        schedule_name=schedule.schedule_name,
        run_name=_phase_run_name(schedule, phase),
    )


def _finalize_schedule_run(
    *,
    run_dir: Path,
    schedule: ScheduleSpec,
    phase_metrics: dict[int, dict[str, Any]],
    transition_summary: dict[str, Any] | None = None,
    kd_bridge_summary: dict[str, Any] | None = None,
    downstream_summary: dict[str, Any] | None = None,
    audit: bool = True,
) -> None:
    phase_rows = _phase_rows_with_metrics(schedule, phase_metrics)
    quantum_spend = build_quantum_spend_summary(schedule, phase_rows)
    energy_summary = build_energy_summary(
        total_env_step_budget=schedule.total_env_step_budget,
        qnode_active_steps_budget=schedule.qnode_active_steps_budget,
        phase_metrics=phase_metrics,
    )

    write_schedule_artifacts(
        run_dir,
        schedule,
        phase_metrics=phase_metrics,
        quantum_spend_summary=quantum_spend,
        transition_summary=transition_summary,
        kd_bridge_summary=kd_bridge_summary,
        energy_summary=energy_summary,
        downstream_summary=downstream_summary,
    )

    if audit:
        audit_result = audit_run_dir(run_dir)
        write_audit_report(audit_result)
        if not audit_result.ok:
            errors = "\n".join(f"ERROR: {message}" for message in audit_result.errors)
            raise RuntimeError(f"Schedule output audit failed for {run_dir}\n{errors}")


def _print_dry_run_phase(
    *,
    schedule_config_path: Path,
    phase: PhaseSpec,
) -> None:
    print(
        "phase: "
        f"{phase.phase_id} {phase.phase_name} "
        f"({phase.phase_type}, steps={phase.steps}, qnode_active={phase.qnode_active})"
    )
    if phase.phase_type in PPO_PHASE_TYPES:
        print(f"config: {_resolve_phase_config(schedule_config_path, phase)}")
        initialization = phase.metadata.get("initialization")
        if initialization:
            print(f"initialization: {initialization}")
        if initialization == "checkpoint_resume":
            source = resolve_phase_resume_checkpoint(phase)
            if source is None:
                raise AssertionError("checkpoint_resume did not resolve a source checkpoint")
            source_run_dir, source_checkpoint = source
            print(f"resume source run: {source_run_dir}")
            print(f"resume checkpoint: {source_checkpoint}")
            print(
                "additional/cumulative steps: "
                f"{phase.steps} / "
                f"{phase.metadata.get('target_cumulative_global_step', '<unspecified>')}"
            )
            print(f"continuation seed: {phase.metadata.get('continuation_seed', '<config>')}")
            print("optimizer state: restored; RNG/environment state: not stored by source checkpoint")
        if "freeze_feature_extractor" in phase.metadata:
            print(f"freeze_feature_extractor: {phase.metadata['freeze_feature_extractor']}")
    else:
        bridge_mode = phase.metadata.get("bridge_mode", "<missing>")
        print(f"config: <none> ({phase.phase_type}, bridge_mode={bridge_mode})")


def dry_run_schedule(
    *,
    config_path: str | Path,
    output_root: str | Path,
    run_name: str | None = None,
) -> ScheduleRunResult:
    config_path = Path(config_path).resolve()
    output_root = Path(output_root)
    schedule = load_schedule_spec(config_path)
    _validate_supported_schedule(schedule)
    run_dir = _build_parent_run_dir(output_root=output_root, schedule=schedule, run_name=run_name)

    print("DRY RUN OK")
    print(f"schedule: {schedule.schedule_name}")
    print(f"env_id: {schedule.env_id}")
    print(f"run_dir: {run_dir}")
    for phase in schedule.phases:
        _print_dry_run_phase(schedule_config_path=config_path, phase=phase)

    return ScheduleRunResult(run_dir=run_dir, phase_run_dirs={}, dry_run=True)


def run_schedule(
    *,
    config_path: str | Path,
    output_root: str | Path,
    run_name: str | None = None,
    audit: bool = True,
) -> ScheduleRunResult:
    """Run a supported schedule and write schedule-aware artifacts."""

    config_path = Path(config_path).resolve()
    output_root = Path(output_root)
    schedule = load_schedule_spec(config_path)
    _validate_supported_schedule(schedule)

    run_dir = _build_parent_run_dir(output_root=output_root, schedule=schedule, run_name=run_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "schedule_config.yaml")

    if schedule.schedule_name in SUPPORTED_SINGLE_PHASE_SCHEDULES:
        result = _run_single_phase_schedule(
            config_path=config_path,
            schedule=schedule,
            run_dir=run_dir,
            audit=audit,
        )
    elif schedule.schedule_name in SUPPORTED_DELAYED_POLICY_VALUE_KD_SCHEDULES:
        result = _run_delayed_policy_value_kd_schedule(
            config_path=config_path,
            schedule=schedule,
            run_dir=run_dir,
            audit=audit,
        )
    elif schedule.schedule_name in SUPPORTED_DOWNSTREAM_SCHEDULES:
        result = _run_downstream_hybrid_adaptation_schedule(
            config_path=config_path,
            schedule=schedule,
            run_dir=run_dir,
            audit=audit,
        )
    else:
        raise AssertionError(f"Unhandled supported schedule: {schedule.schedule_name}")

    print("Audit this directory:")
    print(f"  python scripts/audit_quantum_budget_outputs.py --run-dir {result.run_dir}")
    return result


def _run_single_phase_schedule(
    *,
    config_path: Path,
    schedule: ScheduleSpec,
    run_dir: Path,
    audit: bool,
) -> ScheduleRunResult:
    phase = schedule.phases[0]
    concrete_phase_run_dir = _run_ppo_schedule_phase(
        schedule_config_path=config_path,
        schedule=schedule,
        phase=phase,
        run_dir=run_dir,
    )

    phase_metrics = {phase.phase_id: collect_phase_metrics(concrete_phase_run_dir)}
    _finalize_schedule_run(
        run_dir=run_dir,
        schedule=schedule,
        phase_metrics=phase_metrics,
        audit=audit,
    )

    return ScheduleRunResult(
        run_dir=run_dir,
        phase_run_dirs={phase.phase_id: concrete_phase_run_dir},
        dry_run=False,
    )


def _run_delayed_policy_value_kd_schedule(
    *,
    config_path: Path,
    schedule: ScheduleSpec,
    run_dir: Path,
    audit: bool,
) -> ScheduleRunResult:
    classical_phase, kd_phase, target_phase = schedule.phases
    is_classical_control = (
        schedule.schedule_name in SUPPORTED_DELAYED_CLASSICAL_POLICY_VALUE_KD_SCHEDULES
    )
    phase_run_dirs: dict[int, Path] = {}
    phase_metrics: dict[int, dict[str, Any]] = {}

    classical_run_dir = _run_ppo_schedule_phase(
        schedule_config_path=config_path,
        schedule=schedule,
        phase=classical_phase,
        run_dir=run_dir,
    )
    phase_run_dirs[classical_phase.phase_id] = classical_run_dir
    phase_metrics[classical_phase.phase_id] = collect_phase_metrics(classical_run_dir)

    kd_phase_dir = _phase_parent_dir(run_dir, kd_phase)
    kd_phase_dir.mkdir(parents=True, exist_ok=True)
    bridge_mode = str(kd_phase.metadata.get("bridge_mode", ""))
    if bridge_mode != "offline_policy_value_kd_micro":
        # Validation should catch this before phase execution. Keep the explicit
        # guard here so direct/internal calls cannot revive archived placeholders.
        raise ValueError(
            "Maintained delayed schedules require a real "
            "bridge_mode='offline_policy_value_kd_micro'; "
            f"got {bridge_mode!r}."
        )

    teacher_checkpoint = classical_run_dir / "checkpoint.zip"
    if not teacher_checkpoint.exists():
        raise FileNotFoundError(
            "Cannot run offline policy-value KD bridge: phase 1 checkpoint was not found at "
            f"{teacher_checkpoint}."
        )
    teacher_config_path = _resolve_phase_config(config_path, classical_phase)
    kd_result = run_offline_policy_value_kd_bridge(
        schedule=schedule,
        phase=kd_phase,
        phase_dir=kd_phase_dir,
        teacher_checkpoint=teacher_checkpoint,
        teacher_config_path=teacher_config_path,
    )
    kd_metrics = kd_result.metrics

    phase_run_dirs[kd_phase.phase_id] = kd_phase_dir
    phase_metrics[kd_phase.phase_id] = kd_metrics

    target_initialization = str(target_phase.metadata.get("initialization", "fresh_hybrid"))
    if target_initialization == "kd_composed_policy_value":
        teacher_checkpoint = classical_run_dir / "checkpoint.zip"
        actor_head_path = Path(kd_metrics.get("student_actor_head_path", ""))
        critic_head_path = Path(kd_metrics.get("student_critic_head_path", ""))
        if not teacher_checkpoint.exists():
            raise FileNotFoundError(f"Missing teacher checkpoint for composed phase: {teacher_checkpoint}")
        if not actor_head_path.exists():
            raise FileNotFoundError(f"Missing KD actor head artifact for composed phase: {actor_head_path}")
        if not critic_head_path.exists():
            raise FileNotFoundError(f"Missing KD critic head artifact for composed phase: {critic_head_path}")

        phase_config = _resolve_phase_config(config_path, target_phase)
        phase_parent = _phase_parent_dir(run_dir, target_phase)
        phase_parent.mkdir(parents=True, exist_ok=True)
        target_run_dir = run_composed_policy_value_finetune_phase(
            config_path=phase_config,
            phase=target_phase,
            phase_output_dir=phase_parent,
            schedule_name=schedule.schedule_name,
            run_name=_phase_run_name(schedule, target_phase),
            teacher_checkpoint=teacher_checkpoint,
            teacher_config_path=_resolve_phase_config(config_path, classical_phase),
            student_actor_head_path=actor_head_path,
            student_critic_head_path=critic_head_path,
            freeze_feature_extractor=bool(target_phase.metadata.get("freeze_feature_extractor", False)),
            rebuild_optimizer=bool(target_phase.metadata.get("rebuild_optimizer", True)),
            use_best_heads=bool(target_phase.metadata.get("use_best_heads", True)),
        )
    elif target_initialization == "fresh_hybrid":
        target_run_dir = _run_ppo_schedule_phase(
            schedule_config_path=config_path,
            schedule=schedule,
            phase=target_phase,
            run_dir=run_dir,
        )
    else:
        raise ValueError(f"Unsupported delayed phase 3 initialization: {target_initialization!r}")
    phase_run_dirs[target_phase.phase_id] = target_run_dir
    phase_metrics[target_phase.phase_id] = collect_phase_metrics(target_run_dir)

    if is_classical_control:
        actual_calls = sum(int(metrics.get("qnode_calls") or 0) for metrics in phase_metrics.values())
        actual_time = sum(float(metrics.get("qnode_time") or 0.0) for metrics in phase_metrics.values())
        if actual_calls != 0 or abs(actual_time) > 1e-12:
            raise RuntimeError(
                f"{schedule.schedule_name} is a no-QNode attribution control but recorded "
                f"qnode_calls={actual_calls}, qnode_time={actual_time}."
            )

    total_qnode_time = sum(float(metrics.get("qnode_time") or 0.0) for metrics in phase_metrics.values())
    kd_bridge_summary = build_kd_bridge_summary(
        schedule=schedule,
        kd_phase=kd_phase,
        kd_metrics=kd_metrics,
        total_qnode_time=total_qnode_time,
    )
    transition_summary = build_transition_summary(
        schedule=schedule,
        transition_phase=kd_phase,
        before_phase_metrics=phase_metrics.get(classical_phase.phase_id),
        after_phase_metrics=phase_metrics.get(target_phase.phase_id),
        phase_metrics=phase_metrics,
    )

    _finalize_schedule_run(
        run_dir=run_dir,
        schedule=schedule,
        phase_metrics=phase_metrics,
        transition_summary=transition_summary,
        kd_bridge_summary=kd_bridge_summary,
        audit=audit,
    )

    return ScheduleRunResult(run_dir=run_dir, phase_run_dirs=phase_run_dirs, dry_run=False)


def _load_json_object(path: Path) -> dict[str, Any]:
    import json

    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_downstream_hybrid_adaptation_schedule(
    *,
    config_path: Path,
    schedule: ScheduleSpec,
    run_dir: Path,
    audit: bool,
) -> ScheduleRunResult:
    representation_phase, kd_phase, downstream_phase = schedule.phases
    phase_run_dirs: dict[int, Path] = {}
    phase_metrics: dict[int, dict[str, Any]] = {}

    representation_run_dir = _run_ppo_schedule_phase(
        schedule_config_path=config_path,
        schedule=schedule,
        phase=representation_phase,
        run_dir=run_dir,
    )
    phase_run_dirs[representation_phase.phase_id] = representation_run_dir
    phase_metrics[representation_phase.phase_id] = collect_phase_metrics(representation_run_dir)

    kd_phase_dir = _phase_parent_dir(run_dir, kd_phase)
    kd_phase_dir.mkdir(parents=True, exist_ok=True)
    teacher_checkpoint = representation_run_dir / "checkpoint.zip"
    if not teacher_checkpoint.exists():
        raise FileNotFoundError(
            "Cannot run downstream policy-value KD bridge: phase 1 checkpoint was not found at "
            f"{teacher_checkpoint}."
        )
    representation_config_path = _resolve_phase_config(config_path, representation_phase)
    kd_result = run_offline_policy_value_kd_bridge(
        schedule=schedule,
        phase=kd_phase,
        phase_dir=kd_phase_dir,
        teacher_checkpoint=teacher_checkpoint,
        teacher_config_path=representation_config_path,
    )
    kd_metrics = kd_result.metrics
    phase_run_dirs[kd_phase.phase_id] = kd_phase_dir
    phase_metrics[kd_phase.phase_id] = kd_metrics

    actor_head_path = Path(kd_metrics.get("student_actor_head_path", ""))
    critic_head_path = Path(kd_metrics.get("student_critic_head_path", ""))
    if not actor_head_path.exists():
        raise FileNotFoundError(f"Missing KD actor head artifact for downstream phase: {actor_head_path}")
    if not critic_head_path.exists():
        raise FileNotFoundError(f"Missing KD critic head artifact for downstream phase: {critic_head_path}")

    phase_config = _resolve_phase_config(config_path, downstream_phase)
    phase_parent = _phase_parent_dir(run_dir, downstream_phase)
    phase_parent.mkdir(parents=True, exist_ok=True)
    downstream_run_dir = run_composed_policy_value_finetune_phase(
        config_path=phase_config,
        phase=downstream_phase,
        phase_output_dir=phase_parent,
        schedule_name=schedule.schedule_name,
        run_name=_phase_run_name(schedule, downstream_phase),
        teacher_checkpoint=teacher_checkpoint,
        teacher_config_path=representation_config_path,
        student_actor_head_path=actor_head_path,
        student_critic_head_path=critic_head_path,
        freeze_feature_extractor=True,
        rebuild_optimizer=bool(downstream_phase.metadata.get("rebuild_optimizer", True)),
        use_best_heads=bool(downstream_phase.metadata.get("use_best_heads", True)),
    )
    phase_run_dirs[downstream_phase.phase_id] = downstream_run_dir
    phase_metrics[downstream_phase.phase_id] = collect_phase_metrics(downstream_run_dir)

    total_qnode_time = sum(float(metrics.get("qnode_time") or 0.0) for metrics in phase_metrics.values())
    kd_bridge_summary = build_kd_bridge_summary(
        schedule=schedule,
        kd_phase=kd_phase,
        kd_metrics=kd_metrics,
        total_qnode_time=total_qnode_time,
    )
    transition_summary = build_transition_summary(
        schedule=schedule,
        transition_phase=kd_phase,
        before_phase_metrics=phase_metrics.get(representation_phase.phase_id),
        after_phase_metrics=phase_metrics.get(downstream_phase.phase_id),
        phase_metrics=phase_metrics,
    )
    composition_summary = _load_json_object(downstream_run_dir / "composition_summary.json")
    downstream_summary = build_downstream_summary(
        schedule=schedule,
        representation_phase=representation_phase,
        kd_phase=kd_phase,
        downstream_phase=downstream_phase,
        kd_metrics=kd_metrics,
        downstream_metrics=phase_metrics[downstream_phase.phase_id],
        representation_run_dir=representation_run_dir,
        kd_phase_dir=kd_phase_dir,
        downstream_run_dir=downstream_run_dir,
        composition_summary=composition_summary,
    )

    _finalize_schedule_run(
        run_dir=run_dir,
        schedule=schedule,
        phase_metrics=phase_metrics,
        transition_summary=transition_summary,
        kd_bridge_summary=kd_bridge_summary,
        downstream_summary=downstream_summary,
        audit=audit,
    )

    return ScheduleRunResult(run_dir=run_dir, phase_run_dirs=phase_run_dirs, dry_run=False)
