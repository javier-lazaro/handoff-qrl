"""Schema helpers for maintained QNode-activation schedules.

The schema is dependency-light and owned by this experiment repository. It
represents schedule identity, phase metadata, QNode-active budgets and
transition labels without importing SimplyQRL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ALLOWED_SCHEDULE_NAMES = {
    "classical_warmup",
    "compact_classical_c0",
    "immediate_hybrid_q100",
    "delayed_hybrid_q75_kd_bridge",
    "delayed_hybrid_q50_kd_bridge",
    "delayed_hybrid_q25_kd_bridge",
    "delayed_classical_q75_kd_bridge",
    "delayed_classical_q50_kd_bridge",
    "delayed_classical_q25_kd_bridge",
    "downstream_hybrid_adaptation",
}

ALLOWED_PHASE_TYPES = {
    "classical_ppo",
    "hybrid_ppo",
    "kd_bridge",
}


class ScheduleValidationError(ValueError):
    """Raised when a schedule specification is malformed."""


def _require_mapping(data: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ScheduleValidationError(f"{context} must be a mapping/dict.")
    return data


def _require_str(data: dict[str, Any], key: str, *, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ScheduleValidationError(f"{context}.{key} must be a non-empty string.")
    return value.strip()


def _require_non_negative_int(data: dict[str, Any], key: str, *, context: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ScheduleValidationError(f"{context}.{key} must be a non-negative integer.")
    return value


def _require_ratio(data: dict[str, Any], key: str, *, context: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ScheduleValidationError(f"{context}.{key} must be a number in [0, 1].")
    ratio = float(value)
    if ratio < 0.0 or ratio > 1.0:
        raise ScheduleValidationError(f"{context}.{key} must be in [0, 1], got {ratio}.")
    return ratio


@dataclass(frozen=True)
class PhaseSpec:
    """One schedule phase.

    ``steps`` is the environment-step budget assigned to the phase.  Transition
    or KD-only phases may legitimately use ``steps=0`` while still incurring
    wall-clock/QNode cost later when the runner records metrics.
    """

    phase_id: int
    phase_name: str
    phase_type: str
    qnode_active: bool
    steps: int
    config: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, index: int) -> "PhaseSpec":
        data = _require_mapping(raw, context=f"phases[{index}]")
        phase_id = data.get("phase_id", index + 1)
        if isinstance(phase_id, bool) or not isinstance(phase_id, int) or phase_id <= 0:
            raise ScheduleValidationError(f"phases[{index}].phase_id must be a positive integer.")

        phase_name = _require_str(data, "phase_name", context=f"phases[{index}]")
        phase_type = _require_str(data, "phase_type", context=f"phases[{index}]")
        if phase_type not in ALLOWED_PHASE_TYPES:
            allowed = ", ".join(sorted(ALLOWED_PHASE_TYPES))
            raise ScheduleValidationError(
                f"Unsupported phase_type {phase_type!r} in phases[{index}]. Allowed: {allowed}."
            )

        qnode_active = data.get("qnode_active")
        if not isinstance(qnode_active, bool):
            raise ScheduleValidationError(f"phases[{index}].qnode_active must be true/false.")

        steps = _require_non_negative_int(data, "steps", context=f"phases[{index}]")
        config = data.get("config")
        if config is not None and not isinstance(config, str):
            raise ScheduleValidationError(f"phases[{index}].config must be a string when set.")

        reserved = {"phase_id", "phase_name", "phase_type", "qnode_active", "steps", "config"}
        metadata = {key: value for key, value in data.items() if key not in reserved}
        return cls(
            phase_id=phase_id,
            phase_name=phase_name,
            phase_type=phase_type,
            qnode_active=qnode_active,
            steps=steps,
            config=config,
            metadata=metadata,
        )

    def to_summary_row(self, *, start_step: int) -> dict[str, Any]:
        end_step = start_step + self.steps
        return {
            "phase_id": self.phase_id,
            "phase_name": self.phase_name,
            "phase_type": self.phase_type,
            "start_step": start_step,
            "end_step": end_step,
            "phase_budget_steps": self.steps,
            "qnode_active": self.qnode_active,
            "config": self.config,
        }


@dataclass(frozen=True)
class ScheduleSpec:
    """Paper-facing schedule identity and phase plan."""

    schedule_name: str
    schedule_family: str
    total_env_step_budget: int
    qnode_active_steps_budget: int
    qnode_active_steps_ratio: float
    phases: tuple[PhaseSpec, ...]
    env_id: str | None = None
    transition_type: str | None = None
    bridge_type: str | None = None
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ScheduleSpec":
        data = _require_mapping(raw, context="schedule")
        schedule_name = _require_str(data, "schedule_name", context="schedule")
        if schedule_name not in ALLOWED_SCHEDULE_NAMES:
            allowed = ", ".join(sorted(ALLOWED_SCHEDULE_NAMES))
            raise ScheduleValidationError(
                f"Unsupported schedule_name {schedule_name!r}. Allowed: {allowed}."
            )

        schedule_family = _require_str(data, "schedule_family", context="schedule")
        total_env_step_budget = _require_non_negative_int(
            data, "total_env_step_budget", context="schedule"
        )
        qnode_active_steps_budget = _require_non_negative_int(
            data, "qnode_active_steps_budget", context="schedule"
        )
        if qnode_active_steps_budget > total_env_step_budget:
            raise ScheduleValidationError(
                "schedule.qnode_active_steps_budget cannot exceed schedule.total_env_step_budget."
            )

        qnode_active_steps_ratio = _require_ratio(
            data, "qnode_active_steps_ratio", context="schedule"
        )

        if total_env_step_budget > 0:
            expected_ratio = qnode_active_steps_budget / total_env_step_budget
            if abs(expected_ratio - qnode_active_steps_ratio) > 1e-6:
                raise ScheduleValidationError(
                    "schedule.qnode_active_steps_ratio does not match "
                    "qnode_active_steps_budget / total_env_step_budget "
                    f"({qnode_active_steps_ratio} vs {expected_ratio})."
                )

        raw_phases = data.get("phases")
        if not isinstance(raw_phases, list) or not raw_phases:
            raise ScheduleValidationError("schedule.phases must be a non-empty list.")
        phases = tuple(PhaseSpec.from_mapping(item, index=i) for i, item in enumerate(raw_phases))

        phase_ids = [phase.phase_id for phase in phases]
        if len(set(phase_ids)) != len(phase_ids):
            raise ScheduleValidationError("phase_id values must be unique within a schedule.")

        env_id = data.get("env_id")
        if env_id is not None and not isinstance(env_id, str):
            raise ScheduleValidationError("schedule.env_id must be a string when set.")

        transition_type = data.get("transition_type")
        if transition_type is not None and not isinstance(transition_type, str):
            raise ScheduleValidationError("schedule.transition_type must be a string when set.")

        bridge_type = data.get("bridge_type")
        if bridge_type is not None and not isinstance(bridge_type, str):
            raise ScheduleValidationError("schedule.bridge_type must be a string when set.")

        notes = data.get("notes")
        if notes is not None and not isinstance(notes, str):
            raise ScheduleValidationError("schedule.notes must be a string when set.")

        reserved = {
            "schedule_name",
            "schedule_family",
            "total_env_step_budget",
            "qnode_active_steps_budget",
            "qnode_active_steps_ratio",
            "phases",
            "env_id",
            "transition_type",
            "bridge_type",
            "notes",
        }
        metadata = {key: value for key, value in data.items() if key not in reserved}

        return cls(
            schedule_name=schedule_name,
            schedule_family=schedule_family,
            total_env_step_budget=total_env_step_budget,
            qnode_active_steps_budget=qnode_active_steps_budget,
            qnode_active_steps_ratio=qnode_active_steps_ratio,
            phases=phases,
            env_id=env_id,
            transition_type=transition_type,
            bridge_type=bridge_type,
            notes=notes,
            metadata=metadata,
        )

    def to_schedule_summary(self) -> dict[str, Any]:
        return {
            "schedule_name": self.schedule_name,
            "schedule_family": self.schedule_family,
            "env_id": self.env_id,
            "total_env_step_budget": self.total_env_step_budget,
            "qnode_active_steps_budget": self.qnode_active_steps_budget,
            "qnode_active_steps_ratio": self.qnode_active_steps_ratio,
            "phase_count": len(self.phases),
            "transition_type": self.transition_type,
            "bridge_type": self.bridge_type,
            "notes": self.notes,
            **self.metadata,
        }

    def default_phase_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        start_step = 0
        for phase in self.phases:
            row = phase.to_summary_row(start_step=start_step)
            rows.append(row)
            start_step = row["end_step"]
        return rows


def load_schedule_spec(path: str | Path) -> ScheduleSpec:
    """Load and validate a schedule YAML file."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ScheduleSpec.from_mapping(raw)
