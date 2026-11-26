"""Phase execution accounting independent of SimplyQRL internals.

SimplyQRL rounds PPO training requests to complete rollout batches.  This protocol needs
both the declared budget and the number of environment steps that actually ran,
especially for multi-phase schedules where independent rounding can otherwise
change the intended Q25/Q50/Q75 ratios.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from hybrid_training_qrl.summaries import write_json

PHASE_EXECUTION_SUMMARY = "phase_execution_summary.json"


def build_phase_execution_summary(
    *,
    model: Any,
    configured_timesteps: int,
    start_global_step: int,
) -> dict[str, Any]:
    """Return configured/effective PPO step accounting for one phase."""

    configured_timesteps = int(configured_timesteps)
    start_global_step = int(start_global_step)
    final_global_step = int(getattr(model, "global_step", start_global_step))
    effective_env_steps = final_global_step - start_global_step
    num_envs = int(getattr(model, "num_envs"))
    num_steps = int(getattr(model, "num_steps"))
    rollout_batch_size = num_envs * num_steps
    expected_iterations = (
        math.ceil(configured_timesteps / rollout_batch_size)
        if configured_timesteps > 0
        else 0
    )
    actual_iterations = int(getattr(model, "num_iterations", expected_iterations))

    return {
        "configured_timesteps": configured_timesteps,
        "start_global_step": start_global_step,
        "final_global_step": final_global_step,
        "effective_env_steps": effective_env_steps,
        "step_overshoot": effective_env_steps - configured_timesteps,
        "num_envs": num_envs,
        "num_steps": num_steps,
        "rollout_batch_size": rollout_batch_size,
        "expected_iterations": expected_iterations,
        "actual_iterations": actual_iterations,
        "rollout_aligned": configured_timesteps % rollout_batch_size == 0,
    }


def write_phase_execution_summary(
    phase_run_dir: str | Path,
    *,
    model: Any,
    configured_timesteps: int,
    start_global_step: int,
) -> dict[str, Any]:
    """Persist and return phase execution accounting."""

    payload = build_phase_execution_summary(
        model=model,
        configured_timesteps=configured_timesteps,
        start_global_step=start_global_step,
    )
    write_json(Path(phase_run_dir) / PHASE_EXECUTION_SUMMARY, payload)
    return payload
