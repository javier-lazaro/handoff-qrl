"""Small adapter around SimplyQRL's YAML-driven PPO runner."""

from __future__ import annotations

import copy
import json
import math
import zipfile
from pathlib import Path
from typing import Any

from hybrid_training_qrl.execution import write_phase_execution_summary
from hybrid_training_qrl.schedules import PhaseSpec
from hybrid_training_qrl.summaries import write_json

CONTINUATION_SUMMARY = "continuation_summary.json"


def _ensure_mapping(payload: Any, *, key: str) -> dict[str, Any]:
    value = payload.get(key)
    if value is None:
        value = {}
        payload[key] = value
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping at {key!r}, got {type(value).__name__}.")
    return value


def prepare_ppo_config_for_phase(
    *,
    config_path: str | Path,
    phase: PhaseSpec,
    phase_output_dir: str | Path,
    schedule_name: str,
    run_name: str,
    total_timesteps: int | None = None,
) -> dict[str, Any]:
    """Load a SimplyQRL YAML config and override runtime/reporting fields."""

    from simplyqrl.runners.train_ppo import load_yaml

    config = copy.deepcopy(load_yaml(config_path))
    total_timesteps = int(phase.steps if total_timesteps is None else total_timesteps)

    run_config = _ensure_mapping(config, key="run")
    run_config["run_name"] = run_name
    run_config["total_timesteps"] = total_timesteps
    run_config.setdefault("progress_bar", False)

    reporting = _ensure_mapping(config, key="reporting")
    reporting["enabled"] = True
    reporting["output_dir"] = str(phase_output_dir)
    reporting["experiment_name"] = None
    reporting["run_name"] = run_name
    reporting["regime"] = schedule_name
    reporting.setdefault("hardware_profile", False)

    evaluation = _ensure_mapping(reporting, key="evaluation")
    if bool(evaluation.get("enabled", False)):
        raise ValueError(
            "Paper-facing schedule phases must disable in-training evaluation. "
            "Set reporting.evaluation.enabled=false and evaluate only with "
            "scripts/evaluate_checkpoints.py after training."
        )
    evaluation.clear()
    evaluation["enabled"] = False

    resources = _ensure_mapping(reporting, key="resources")
    if resources.get("enabled") is not True:
        raise ValueError("Schedule phases must enable resource reporting.")

    execution_contract = str(phase.metadata.get("execution_contract", "")).strip()
    config_metadata = config.get("metadata") or {}
    if not isinstance(config_metadata, dict):
        raise TypeError("Expected YAML mapping at 'metadata'.")
    execution_scope = str(config_metadata.get("execution_scope", "")).strip()
    paper_facing = config_metadata.get("paper_facing")

    timing_scope = str(config_metadata.get("timing_scope", "")).strip()
    paper_facing_performance = config_metadata.get("paper_facing_performance")
    parallel_execution = config.get("parallel_execution") or {}
    if not isinstance(parallel_execution, dict):
        raise TypeError("Expected YAML mapping at 'parallel_execution'.")

    engineering_parallel_gate = (
        execution_contract == "parallelism_gate_non_paper"
        and execution_scope == "development_parallelism_gate"
        and paper_facing is False
    )
    parallel_performance_campaign = (
        execution_contract == "parallel_performance_campaign"
        and execution_scope == "parallel_performance_campaign"
        and paper_facing is True
        and paper_facing_performance is True
        and timing_scope == "operational_only"
    )
    # MinAtar uses generated per-seed control YAMLs.  Keep the exemption
    # deliberately fail-closed: the phase must opt into the exact execution
    # contract and the source PPO config must carry every frozen parallel marker.
    minatar_parallel_performance_campaign = (
        execution_contract == "minatar_parallel_performance_campaign"
        and config.get("artifact_kind") == "frozen_parallel_expansion_training"
        and config.get("seed_role") == "concurrent_performance_expansion_after_freeze"
        and config.get("paper_facing") is True
        and parallel_execution.get("timing_scope") == "operational_only"
        and parallel_execution.get("codecarbon_scope") == "disabled"
        and parallel_execution.get("parent_managed_seed_worker") is True
    )
    if engineering_parallel_gate:
        if resources.get("use_codecarbon") is not False:
            raise ValueError(
                "Parallelism-gate workloads must explicitly disable CodeCarbon. "
                "Their timings are operational-only, not paper-facing measurements."
            )
    elif parallel_performance_campaign:
        if resources.get("use_codecarbon") is not False:
            raise ValueError(
                "Parallel performance-campaign workloads must explicitly disable "
                "CodeCarbon because concurrent trackers cannot provide independent "
                "paper-facing energy measurements."
            )
    elif minatar_parallel_performance_campaign:
        if resources.get("use_codecarbon") is not False:
            raise ValueError(
                "MinAtar parallel performance-campaign workloads must explicitly "
                "disable CodeCarbon because concurrent trackers cannot provide "
                "independent paper-facing energy measurements."
            )
    else:
        protected_contracts = {
            "parallelism_gate_non_paper",
            "parallel_performance_campaign",
            "minatar_parallel_performance_campaign",
        }
        protected_scopes = {
            "development_parallelism_gate",
            "parallel_performance_campaign",
        }
        if execution_contract in protected_contracts or execution_scope in protected_scopes:
            raise ValueError(
                "The parallel CodeCarbon exemption is fail-closed: execution_contract "
                "and all campaign-specific paper-facing/timing markers must match an "
                "approved parallel execution contract exactly."
            )
        if resources.get("use_codecarbon") is not True:
            raise ValueError(
                "Paper-facing schedule phases must enable CodeCarbon during training. "
                "Post-hoc evaluation remains untracked."
            )
    resources.setdefault("sample_every_updates", 10)
    resources.setdefault("include_system", True)
    resources.setdefault("include_process", True)
    resources.setdefault("include_torch_cuda", True)

    quantum = _ensure_mapping(reporting, key="quantum")
    quantum.setdefault("enabled", bool(phase.qnode_active))
    quantum.setdefault("include_zeros", True)

    return config


def _resolve_checkpoint_from_run_dir(raw_run_dir: str | Path) -> tuple[Path, Path]:
    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(
            "Checkpoint continuation source run does not exist: " f"{run_dir}"
        )

    preferred = sorted(run_dir.glob("phases/*/checkpoint.zip"))
    candidates = preferred or sorted(run_dir.rglob("checkpoint.zip"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint.zip was found under continuation source run {run_dir}."
        )
    if len(candidates) != 1:
        rendered = "\n".join(f"  - {path}" for path in candidates)
        raise RuntimeError(
            "Checkpoint continuation requires exactly one source checkpoint. "
            f"Found {len(candidates)} under {run_dir}:\n{rendered}"
        )
    return run_dir, candidates[0].resolve()


def resolve_phase_resume_checkpoint(phase: PhaseSpec) -> tuple[Path, Path] | None:
    """Resolve a phase-level checkpoint continuation source, if configured."""

    initialization = str(phase.metadata.get("initialization", "fresh"))
    if initialization != "checkpoint_resume":
        return None

    raw_checkpoint = phase.metadata.get("resume_from_checkpoint")
    raw_run_dir = phase.metadata.get("resume_from_run_dir")
    if raw_checkpoint and raw_run_dir:
        raise ValueError(
            "checkpoint_resume must define only one of resume_from_checkpoint or "
            "resume_from_run_dir."
        )
    if raw_checkpoint:
        checkpoint = Path(str(raw_checkpoint)).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = Path.cwd() / checkpoint
        checkpoint = checkpoint.resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Continuation checkpoint not found: {checkpoint}")
        return checkpoint.parent, checkpoint
    if raw_run_dir:
        return _resolve_checkpoint_from_run_dir(str(raw_run_dir))
    raise ValueError(
        "checkpoint_resume requires resume_from_run_dir or resume_from_checkpoint."
    )


def _checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(checkpoint_path, "r") as archive:
        payload = json.loads(archive.read("meta.json"))
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint meta.json is not a mapping: {checkpoint_path}")
    return payload


def _float_equal(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    except (TypeError, ValueError):
        return left == right


def _validate_loaded_checkpoint_against_config(model: Any, config: dict[str, Any]) -> None:
    ppo_config = dict(config.get("ppo") or {})
    expected_agent_type = ppo_config.get("agent_type")
    if expected_agent_type is not None and model.agent_type != expected_agent_type:
        raise ValueError(
            "Continuation checkpoint agent_type mismatch: "
            f"checkpoint={model.agent_type!r}, config={expected_agent_type!r}."
        )

    expected_agent_config = ppo_config.get("agent_config")
    if expected_agent_config is not None and model.agent_config != expected_agent_config:
        raise ValueError(
            "Continuation checkpoint agent_config does not match the continuation config. "
            "This usually means the wrong DR-depth checkpoint was selected."
        )

    comparable = {
        "learning_rate": "learning_rate",
        "learning_rate_mode": "learning_rate_mode",
        "actor_learning_rate": "actor_learning_rate",
        "critic_learning_rate": "critic_learning_rate",
        "shared_learning_rate": "shared_learning_rate",
        "num_envs": "num_envs",
        "num_steps": "num_steps",
        "anneal_lr": "anneal_lr",
        "gamma": "gamma",
        "gae_lambda": "gae_lambda",
        "num_minibatches": "num_minibatches",
        "update_epochs": "update_epochs",
        "norm_adv": "norm_adv",
        "clip_coef": "clip_coef",
        "clip_vloss": "clip_vloss",
        "ent_coef": "ent_coef",
        "vf_coef": "vf_coef",
        "max_grad_norm": "max_grad_norm",
        "gradient_clipping_mode": "gradient_clipping_mode",
        "actor_max_grad_norm": "actor_max_grad_norm",
        "critic_max_grad_norm": "critic_max_grad_norm",
        "shared_max_grad_norm": "shared_max_grad_norm",
        "target_kl": "target_kl",
    }
    mismatches: list[str] = []
    for config_key, model_attr in comparable.items():
        if config_key not in ppo_config:
            continue
        expected = ppo_config[config_key]
        actual = getattr(model, model_attr)
        if not _float_equal(actual, expected):
            mismatches.append(f"{config_key}: checkpoint={actual!r}, config={expected!r}")
    if mismatches:
        raise ValueError(
            "Continuation PPO hyperparameters do not match the source checkpoint:\n  - "
            + "\n  - ".join(mismatches)
        )


def _build_model_from_checkpoint_resume(
    *,
    config: dict[str, Any],
    phase: PhaseSpec,
    run_name: str,
) -> tuple[Any, dict[str, Any]]:
    from simplyqrl.ppo import PPO
    from simplyqrl.runners.train_ppo import resolve_env

    resolved = resolve_phase_resume_checkpoint(phase)
    if resolved is None:
        raise AssertionError("checkpoint resume requested without a resolved checkpoint")
    source_run_dir, checkpoint_path = resolved
    source_meta = _checkpoint_metadata(checkpoint_path)

    expected_source_global_step = phase.metadata.get("expected_source_global_step")
    source_global_step = int(source_meta.get("global_step", -1))
    if expected_source_global_step is not None and source_global_step != int(
        expected_source_global_step
    ):
        raise ValueError(
            "Continuation source global_step mismatch: "
            f"checkpoint={source_global_step}, expected={expected_source_global_step}."
        )

    env_id, env_factory = resolve_env(config.get("env"))
    del env_id  # The checkpoint retains its environment label; only custom factories are needed.

    device_policy = config.get("device_policy")
    if device_policy is None:
        device_policy = (config.get("ppo") or {}).get("device_policy")
    device = "cpu"
    if isinstance(device_policy, dict):
        device = str(device_policy.get("torch_device", "cpu"))

    ppo_config = dict(config.get("ppo") or {})
    hparam_overrides: dict[str, Any] = {}
    # A fresh continuation process cannot restore the old RNG/environment state
    # from SimplyQRL 1.5 checkpoints. Allow an explicit continuation seed so the
    # second segment uses a deterministic *new* stream instead of replaying seed 0.
    if "seed" in ppo_config:
        hparam_overrides["seed"] = int(ppo_config["seed"])

    model = PPO.load(
        str(checkpoint_path),
        env=env_factory,
        device=device,
        reporting_config=dict(config.get("reporting") or {}),
        run_name=run_name,
        hparams_overrides=hparam_overrides or None,
    )
    _validate_loaded_checkpoint_against_config(model, config)

    optimizer_lrs = [float(group["lr"]) for group in model.optimizer.param_groups]
    continuation = {
        "initialization_mode": "checkpoint_resume",
        "source_run_dir": str(source_run_dir),
        "source_checkpoint": str(checkpoint_path),
        "source_global_step": source_global_step,
        "continuation_seed": int(model.seed),
        "optimizer_state_restored": True,
        "optimizer_learning_rates_at_resume": optimizer_lrs,
        "rng_state_restored": False,
        "rng_note": (
            "SimplyQRL 1.5 checkpoints persist policy, optimizer and global_step, but not "
            "Python/NumPy/Torch/environment RNG state. The continuation config therefore "
            "uses an explicit deterministic continuation seed."
        ),
    }
    return model, continuation


def _attach_trainability_monitor(model: Any, config: dict[str, Any]) -> None:
    diagnostics = config.get("diagnostics") or {}
    trainability = diagnostics.get("trainability") if isinstance(diagnostics, dict) else None
    if isinstance(trainability, dict) and bool(trainability.get("enabled", False)):
        from hybrid_training_qrl.trainability import TrainabilityMonitor

        model.monitor_manager.add(
            TrainabilityMonitor(
                model=model,
                sample_every_updates=int(trainability.get("sample_every_updates", 1)),
                forward_sample_every_calls=int(
                    trainability.get("forward_sample_every_calls", 32)
                ),
                backward_sample_every_calls=int(
                    trainability.get("backward_sample_every_calls", 32)
                ),
            )
        )


def run_ppo_phase(
    *,
    config_path: str | Path,
    phase: PhaseSpec,
    phase_output_dir: str | Path,
    schedule_name: str,
    run_name: str,
    save_checkpoint: bool = True,
) -> Path:
    """Run one SimplyQRL PPO phase and return the concrete run directory.

    Fresh phases are created through SimplyQRL's YAML builder. A phase with
    ``initialization: checkpoint_resume`` instead restores policy weights,
    optimizer state and ``global_step`` from an earlier SimplyQRL checkpoint,
    then trains for the *additional* ``phase.steps`` budget.
    """

    from simplyqrl.runners.train_ppo import build_ppo_from_config

    config = prepare_ppo_config_for_phase(
        config_path=config_path,
        phase=phase,
        phase_output_dir=phase_output_dir,
        schedule_name=schedule_name,
        run_name=run_name,
    )

    initialization = str(phase.metadata.get("initialization", "fresh"))
    continuation: dict[str, Any] | None = None
    if initialization == "checkpoint_resume":
        model, continuation = _build_model_from_checkpoint_resume(
            config=config,
            phase=phase,
            run_name=run_name,
        )
        total_timesteps = int(phase.steps)
        progress_bar = bool((config.get("run") or {}).get("progress_bar", False))
    elif initialization in {"fresh", "fresh_hybrid", ""}:
        model, total_timesteps, progress_bar = build_ppo_from_config(config)
    else:
        raise ValueError(f"Unsupported PPO phase initialization: {initialization!r}")

    _attach_trainability_monitor(model, config)

    try:
        start_global_step = int(model.global_step)
        if continuation is not None:
            continuation["continuation_start_global_step"] = start_global_step
            target_global_step = phase.metadata.get("target_cumulative_global_step")
            if target_global_step is not None:
                continuation["target_cumulative_global_step"] = int(target_global_step)

        model.train(total_timesteps=total_timesteps, progress_bar=progress_bar)
        if model.artifact_logger is None:
            raise RuntimeError("SimplyQRL reporting is disabled; cannot collect phase artifacts.")

        phase_run_dir = Path(model.artifact_logger.run_dir)
        execution_summary = write_phase_execution_summary(
            phase_run_dir,
            model=model,
            configured_timesteps=total_timesteps,
            start_global_step=start_global_step,
        )
        if continuation is not None:
            continuation.update(
                {
                    "additional_configured_env_steps": int(total_timesteps),
                    "additional_effective_env_steps": int(
                        execution_summary["effective_env_steps"]
                    ),
                    "continuation_final_global_step": int(
                        execution_summary["final_global_step"]
                    ),
                    "target_global_step_reached": (
                        continuation.get("target_cumulative_global_step") is None
                        or int(execution_summary["final_global_step"])
                        == int(continuation["target_cumulative_global_step"])
                    ),
                }
            )
            execution_summary.update(continuation)
            write_json(phase_run_dir / "phase_execution_summary.json", execution_summary)
            write_json(phase_run_dir / CONTINUATION_SUMMARY, continuation)

        if save_checkpoint:
            checkpoint_base = phase_run_dir / "checkpoint"
            checkpoint_metadata = {
                "schedule_name": schedule_name,
                "phase_id": phase.phase_id,
                "phase_name": phase.phase_name,
                "phase_type": phase.phase_type,
                "phase_steps": phase.steps,
                "phase_effective_env_steps": execution_summary["effective_env_steps"],
                "phase_step_overshoot": execution_summary["step_overshoot"],
            }
            if continuation is not None:
                checkpoint_metadata.update(
                    {
                        "initialization_mode": "checkpoint_resume",
                        "source_checkpoint": continuation["source_checkpoint"],
                        "source_global_step": continuation["source_global_step"],
                        "continuation_seed": continuation["continuation_seed"],
                        "rng_state_restored": False,
                    }
                )
            model.save(str(checkpoint_base), metadata=checkpoint_metadata)
        return phase_run_dir
    finally:
        close = getattr(model, "close", None)
        if callable(close):
            close()
