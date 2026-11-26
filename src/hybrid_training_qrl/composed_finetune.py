"""Composed policy-value fine-tuning utilities for delayed schedules.

This module consumes SimplyQRL's staged composition API unlocked in SimplyQRL
5A.  It remains paper-repo code because it binds schedule phases, checkpoint
locations, reporting directories and composition summaries.
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable

from hybrid_training_qrl.execution import write_phase_execution_summary
from hybrid_training_qrl.schedules import PhaseSpec
from hybrid_training_qrl.simplyqrl_adapter import prepare_ppo_config_for_phase
from hybrid_training_qrl.summaries import write_json




class _UpdateCheckpointMonitor:
    """Save optimizer-complete checkpoints after selected PPO updates.

    The object implements the SimplyQRL monitor interface without importing
    SimplyQRL at module import time.  It performs no evaluation and therefore
    does not contaminate the post-hoc-only resource contract.
    """

    def __init__(
        self,
        *,
        model: Any,
        output_dir: Path,
        updates: Iterable[int],
        metadata: dict[str, Any],
    ) -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.updates = {int(value) for value in updates}
        self.metadata = dict(metadata)
        self.saved: dict[int, Path] = {}

    def on_training_start(self, state: dict[str, Any]) -> None:
        return None

    def on_rollout_start(self, state: dict[str, Any]) -> None:
        return None

    def on_episode_end(self, state: dict[str, Any]) -> None:
        return None

    def on_rollout_end(self, state: dict[str, Any]) -> None:
        return None

    def on_update_start(self, state: dict[str, Any]) -> None:
        return None

    def on_backward_end(self, state: dict[str, Any]) -> None:
        return None

    def on_update_end(self, state: dict[str, Any]) -> None:
        update = int(state.get("update", 0))
        if update not in self.updates or update in self.saved:
            return
        milestone_dir = self.output_dir / f"rollout_{update}"
        milestone_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_base = milestone_dir / "checkpoint"
        self.model.save(
            str(checkpoint_base),
            metadata={
                **self.metadata,
                "diagnostic_milestone_update": update,
                "diagnostic_milestone_global_step": int(state.get("global_step", 0)),
                "optimizer_state_included": True,
                "posthoc_evaluation_included": False,
            },
        )
        self.saved[update] = milestone_dir / "checkpoint.zip"

    def on_evaluation_end(self, state: dict[str, Any]) -> None:
        return None

    def on_training_end(self, state: dict[str, Any]) -> None:
        return None

    def on_training_error(self, state: dict[str, Any]) -> None:
        return None




class _LabeledUpdateCheckpointMonitor:
    """Save checkpoints at local updates using cumulative rollout labels."""

    def __init__(
        self,
        *,
        model: Any,
        output_dir: Path,
        update_labels: dict[int, str],
        metadata: dict[str, Any],
    ) -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.update_labels = {int(key): str(value) for key, value in update_labels.items()}
        self.metadata = dict(metadata)
        self.saved: dict[str, Path] = {}

    def on_training_start(self, state: dict[str, Any]) -> None:
        return None

    def on_rollout_start(self, state: dict[str, Any]) -> None:
        return None

    def on_episode_end(self, state: dict[str, Any]) -> None:
        return None

    def on_rollout_end(self, state: dict[str, Any]) -> None:
        return None

    def on_update_start(self, state: dict[str, Any]) -> None:
        return None

    def on_backward_end(self, state: dict[str, Any]) -> None:
        return None

    def on_update_end(self, state: dict[str, Any]) -> None:
        update = int(state.get("update", 0))
        label = self.update_labels.get(update)
        if label is None or label in self.saved:
            return
        milestone_dir = self.output_dir / label
        milestone_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_base = milestone_dir / "checkpoint"
        self.model.save(
            str(checkpoint_base),
            metadata={
                **self.metadata,
                "continuation_local_update": update,
                "continuation_milestone_label": label,
                "continuation_milestone_global_step": int(state.get("global_step", 0)),
                "optimizer_state_included": True,
                "posthoc_evaluation_included": False,
            },
        )
        self.saved[label] = milestone_dir / "checkpoint.zip"

    def on_evaluation_end(self, state: dict[str, Any]) -> None:
        return None

    def on_training_end(self, state: dict[str, Any]) -> None:
        return None

    def on_training_error(self, state: dict[str, Any]) -> None:
        return None


def _ensure_mapping(payload: dict[str, Any], *, key: str) -> dict[str, Any]:
    value = payload.get(key)
    if value is None:
        value = {}
        payload[key] = value
    if not isinstance(value, dict):
        raise TypeError(f"Expected config mapping at {key!r}, got {type(value).__name__}.")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _phase_runtime_config(
    *,
    config_path: str | Path,
    phase: PhaseSpec,
    phase_output_dir: str | Path,
    schedule_name: str,
    run_name: str,
) -> tuple[dict[str, Any], int, bool]:
    config = prepare_ppo_config_for_phase(
        config_path=config_path,
        phase=phase,
        phase_output_dir=phase_output_dir,
        schedule_name=schedule_name,
        run_name=run_name,
    )
    run_config = _ensure_mapping(config, key="run")
    total_timesteps = int(run_config.get("total_timesteps", phase.steps))
    progress_bar = bool(run_config.get("progress_bar", False))
    return config, total_timesteps, progress_bar




def _resolve_env_override_from_config(config_path: str | Path | None):
    """Return the teacher env factory for custom-env checkpoints.

    PPO checkpoints store the environment label as ``env_id``.  Custom envs
    created via YAML factories (for example MinAtar frame stacking) must pass
    the factory again when loading the checkpoint.
    """

    if config_path is None:
        return None

    from simplyqrl.runners.train_ppo import load_yaml, resolve_env

    config = load_yaml(config_path)
    _env_id, env_factory = resolve_env(config.get("env"))
    return env_factory


def _torch_device_from_config(config: dict[str, Any]) -> str:
    device_policy = config.get("device_policy")
    if isinstance(device_policy, dict) and device_policy.get("torch_device") is not None:
        return str(device_policy["torch_device"])
    ppo = config.get("ppo")
    if isinstance(ppo, dict):
        nested = ppo.get("device_policy")
        if isinstance(nested, dict) and nested.get("torch_device") is not None:
            return str(nested["torch_device"])
    return "cpu"



PPO_TRAINING_HPARAM_KEYS = (
    "seed",
    "learning_rate",
    "learning_rate_mode",
    "actor_learning_rate",
    "critic_learning_rate",
    "shared_learning_rate",
    "num_envs",
    "num_steps",
    "anneal_lr",
    "gamma",
    "gae_lambda",
    "num_minibatches",
    "update_epochs",
    "norm_adv",
    "clip_coef",
    "clip_vloss",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "gradient_clipping_mode",
    "actor_max_grad_norm",
    "critic_max_grad_norm",
    "shared_max_grad_norm",
    "target_kl",
)


def _phase_training_hparams_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract PPO training hparams that must survive checkpoint reload.

    Composed phases load a classical teacher checkpoint and then install KD
    hybrid heads before continuing PPO.  The checkpoint supplies the base
    policy weights, but the active training phase must use the hparams from
    its own YAML config, not stale/default values from the teacher checkpoint.
    """

    ppo = config.get("ppo")
    if not isinstance(ppo, dict):
        return {}
    return {key: ppo[key] for key in PPO_TRAINING_HPARAM_KEYS if key in ppo}


def _model_training_hparams(model: Any) -> dict[str, Any]:
    return {
        key: getattr(model, key)
        for key in PPO_TRAINING_HPARAM_KEYS
        if hasattr(model, key)
    }


def _hparam_mismatches(
    *,
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    mismatches: dict[str, dict[str, Any]] = {}
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if actual_value != expected_value:
            mismatches[key] = {"expected": expected_value, "actual": actual_value}
    return mismatches


def _optimizer_group_learning_rates(model: Any) -> dict[str, float]:
    optimizer = getattr(model, "optimizer", None)
    if optimizer is None:
        return {}
    rates: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group_{index}"))
        rates[name] = float(group["lr"])
    return rates


def _expected_optimizer_group_learning_rates(config: dict[str, Any]) -> dict[str, float]:
    ppo = dict(config.get("ppo") or {})
    mode = str(ppo.get("learning_rate_mode", "global"))
    base = float(ppo.get("learning_rate", 0.0))
    if mode == "global":
        return {"global": base}
    return {
        "actor": float(ppo.get("actor_learning_rate", base)),
        "critic": float(ppo.get("critic_learning_rate", base)),
        "shared": float(ppo.get("shared_learning_rate", base)),
    }


def _set_optimizer_group_learning_rates(
    model: Any, *, expected: dict[str, float]
) -> dict[str, float]:
    """Apply phase-config learning rates to restored optimizer parameter groups.

    Adam moments and step counters remain untouched.  This is used only for
    staged continuation experiments that intentionally change LR after a
    checkpoint boundary while preserving optimizer state.
    """

    optimizer = getattr(model, "optimizer", None)
    if optimizer is None:
        raise RuntimeError("Cannot override learning rates without an optimizer")
    applied: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group_{index}"))
        if name not in expected:
            raise RuntimeError(
                f"Restored optimizer group {name!r} has no phase-config learning rate"
            )
        value = float(expected[name])
        group["lr"] = value
        if "initial_lr" in group:
            group["initial_lr"] = value
        applied[name] = value
    return applied


def _optimizer_group_lr_mismatches(
    *, expected: dict[str, float], actual: dict[str, float]
) -> dict[str, dict[str, Any]]:
    mismatches: dict[str, dict[str, Any]] = {}
    for name, actual_lr in actual.items():
        expected_lr = expected.get(name)
        if expected_lr is None or abs(actual_lr - expected_lr) > 1e-12:
            mismatches[name] = {"expected": expected_lr, "actual": actual_lr}
    for required in ("actor", "critic"):
        if required in expected and required not in actual:
            mismatches[required] = {"expected": expected[required], "actual": None}
    return mismatches


def _feature_extractor_param_status(model: Any) -> dict[str, Any]:
    extractor = getattr(model.agent, "feature_extractor", None)
    if extractor is None:
        return {
            "feature_extractor_has_parameters": False,
            "feature_extractor_trainable_parameters": 0,
        }
    params = list(extractor.parameters())
    return {
        "feature_extractor_has_parameters": bool(params),
        "feature_extractor_trainable_parameters": int(
            sum(1 for parameter in params if parameter.requires_grad)
        ),
    }

def _summary_from_artifact_logger(model: Any) -> Path:
    if model.artifact_logger is None:
        raise RuntimeError("SimplyQRL reporting is disabled; cannot collect composed phase artifacts.")
    return Path(model.artifact_logger.run_dir)


def run_composed_policy_value_finetune_phase(
    *,
    config_path: str | Path,
    phase: PhaseSpec,
    phase_output_dir: str | Path,
    schedule_name: str,
    run_name: str,
    teacher_checkpoint: str | Path,
    student_actor_head_path: str | Path,
    student_critic_head_path: str | Path,
    freeze_feature_extractor: bool,
    rebuild_optimizer: bool = True,
    use_best_heads: bool = True,
    save_checkpoint: bool = True,
    teacher_config_path: str | Path | None = None,
    checkpoint_milestone_updates: Iterable[int] | None = None,
    model_configurator: Callable[[Any, Path], Any] | None = None,
) -> Path:
    """Run one PPO phase from a policy-value KD-composed teacher checkpoint."""

    from simplyqrl.ppo import PPO
    from simplyqrl.staged import (
        compose_policy_from_teacher,
        is_feature_extractor_frozen,
        rebuild_ppo_optimizer_after_composition,
    )
    from simplyqrl.staged.head_artifacts import load_head_config

    config, total_timesteps, progress_bar = _phase_runtime_config(
        config_path=config_path,
        phase=phase,
        phase_output_dir=phase_output_dir,
        schedule_name=schedule_name,
        run_name=run_name,
    )
    reporting_config = dict(config.get("reporting") or {"enabled": False})
    torch_device = _torch_device_from_config(config)
    phase_training_hparams_expected = _phase_training_hparams_from_config(config)

    teacher_env_override = _resolve_env_override_from_config(teacher_config_path)
    actor_head_config = load_head_config(student_actor_head_path)
    critic_head_config = load_head_config(student_critic_head_path)
    actor_head_family = str(actor_head_config.get("family", "unknown"))
    critic_head_family = str(critic_head_config.get("family", "unknown"))
    if actor_head_family != critic_head_family:
        raise ValueError(
            "Policy-value composition requires actor and critic heads from the same family; "
            f"got actor={actor_head_family!r}, critic={critic_head_family!r}."
        )
    composed_agent_type = f"composed_policy_value_{actor_head_family}"

    # The teacher optimizer references the pre-composition classical heads and
    # may also have a different parameter-group layout. Load policy state only;
    # the active phase optimizer is rebuilt after installing the KD heads.
    model = PPO.load(
        str(teacher_checkpoint),
        env=teacher_env_override,
        device=torch_device,
        allow_pickle=True,
        reporting_config=reporting_config,
        run_name=run_name,
        hparams_overrides=phase_training_hparams_expected,
        load_optimizer_state=False,
    )
    try:
        start_global_step = int(model.global_step)
        compose_policy_from_teacher(
            model,
            actor_head_artifact=student_actor_head_path,
            critic_head_artifact=student_critic_head_path,
            freeze_feature_extractor=bool(freeze_feature_extractor),
            use_best=bool(use_best_heads),
            qnode_scope="actor",
            critic_qnode_scope="critic",
        )

        # The loaded PPO was built before the hybrid heads were installed. Apply
        # SimplyQRL's QNode wrapper placement policy again after composition.
        apply_policy = getattr(model, "_apply_qnode_wrapper_policy", None)
        if callable(apply_policy):
            apply_policy()

        optimizer_rebuilt = False
        if rebuild_optimizer:
            rebuild_ppo_optimizer_after_composition(model)
            optimizer_rebuilt = True

        optimizer_group_learning_rates = _optimizer_group_learning_rates(model)
        expected_optimizer_group_learning_rates = (
            _expected_optimizer_group_learning_rates(config)
        )
        optimizer_group_lr_mismatches = _optimizer_group_lr_mismatches(
            expected=expected_optimizer_group_learning_rates,
            actual=optimizer_group_learning_rates,
        )
        if optimizer_group_lr_mismatches:
            raise RuntimeError(
                "Composed PPO optimizer does not match the phase config: "
                f"{optimizer_group_lr_mismatches}"
            )

        phase_run_dir = _summary_from_artifact_logger(model)
        composition_summary_path = phase_run_dir / "composition_summary.json"
        model_extension = (
            model_configurator(model, phase_run_dir)
            if model_configurator is not None
            else None
        )
        feature_status = _feature_extractor_param_status(model)
        phase_training_hparams_actual = _model_training_hparams(model)
        phase_training_hparam_mismatches = _hparam_mismatches(
            expected=phase_training_hparams_expected,
            actual=phase_training_hparams_actual,
        )
        composition_summary = {
            "initialization": "kd_composed_policy_value",
            "composition_scope": "policy_value",
            "schedule_name": schedule_name,
            "phase_id": phase.phase_id,
            "phase_name": phase.phase_name,
            "teacher_checkpoint": str(teacher_checkpoint),
            "teacher_config_path": str(teacher_config_path) if teacher_config_path is not None else None,
            "teacher_env_factory_overridden": bool(teacher_env_override is not None),
            "student_actor_head_path": str(student_actor_head_path),
            "student_critic_head_path": str(student_critic_head_path),
            "actor_head_source": "kd_bridge",
            "critic_head_source": "kd_bridge",
            "base_checkpoint_agent_type": "v15_classical",
            "effective_agent_type": composed_agent_type,
            "student_actor_head_family": actor_head_family,
            "student_critic_head_family": critic_head_family,
            "phase_training_hparams_expected": phase_training_hparams_expected,
            "phase_training_hparams_actual": phase_training_hparams_actual,
            "phase_training_hparam_mismatches": phase_training_hparam_mismatches,
            "hparams_match_phase_config": not bool(phase_training_hparam_mismatches),
            "freeze_feature_extractor": bool(freeze_feature_extractor),
            "feature_interface_trainable_during_phase": not bool(freeze_feature_extractor),
            "feature_extractor_runtime_frozen": bool(is_feature_extractor_frozen(model)),
            **feature_status,
            "optimizer_rebuilt": bool(optimizer_rebuilt),
            "optimizer_parameter_group_count": len(model.optimizer.param_groups),
            "optimizer_group_learning_rates": optimizer_group_learning_rates,
            "expected_optimizer_group_learning_rates": (
                expected_optimizer_group_learning_rates
            ),
            "optimizer_group_lr_mismatches": optimizer_group_lr_mismatches,
            "use_best_heads": bool(use_best_heads),
        }
        write_json(composition_summary_path, composition_summary)

        milestone_monitor = None
        milestone_updates = tuple(sorted({int(value) for value in (checkpoint_milestone_updates or ())}))
        if milestone_updates:
            milestone_monitor = _UpdateCheckpointMonitor(
                model=model,
                output_dir=phase_run_dir / "milestones",
                updates=milestone_updates,
                metadata={
                    "schedule_name": schedule_name,
                    "phase_id": phase.phase_id,
                    "phase_name": phase.phase_name,
                    "phase_type": phase.phase_type,
                    "initialization": "kd_composed_policy_value",
                    "composition_scope": "policy_value",
                },
            )
            model.monitor_manager.add(milestone_monitor)

        model.train(total_timesteps=total_timesteps, progress_bar=progress_bar)
        execution_summary = write_phase_execution_summary(
            phase_run_dir,
            model=model,
            configured_timesteps=total_timesteps,
            start_global_step=start_global_step,
        )

        if save_checkpoint:
            checkpoint_base = phase_run_dir / "checkpoint"
            model.save(
                str(checkpoint_base),
                metadata={
                    "schedule_name": schedule_name,
                    "phase_id": phase.phase_id,
                    "phase_name": phase.phase_name,
                    "phase_type": phase.phase_type,
                    "phase_steps": phase.steps,
                    "phase_effective_env_steps": execution_summary["effective_env_steps"],
                    "phase_step_overshoot": execution_summary["step_overshoot"],
                    "initialization": "kd_composed_policy_value",
                    "composition_scope": "policy_value",
                    "feature_extractor_runtime_frozen": bool(is_feature_extractor_frozen(model)),
                },
            )

        # Refresh the artifact after training/checkpointing so it captures the
        # final state of the composition metadata as well.
        final_summary = dict(composition_summary)
        final_summary.update(
            {
                "phase_run_dir": str(phase_run_dir),
                "checkpoint_path": str(phase_run_dir / "checkpoint.zip")
                if save_checkpoint
                else None,
                "feature_extractor_runtime_frozen_after_training": bool(
                    is_feature_extractor_frozen(model)
                ),
                **_feature_extractor_param_status(model),
                "qnode_stats_after_training": getattr(model.agent, "get_qnode_stats", lambda: {})(),
                "checkpoint_milestone_updates": list(milestone_updates),
                "checkpoint_milestones": (
                    {str(update): str(path) for update, path in milestone_monitor.saved.items()}
                    if milestone_monitor is not None
                    else {}
                ),
                "model_extension": (
                    model_extension.metadata()
                    if model_extension is not None
                    and callable(getattr(model_extension, "metadata", None))
                    else None
                ),
            }
        )
        write_json(composition_summary_path, final_summary)
        return phase_run_dir
    finally:
        close = getattr(model, "close", None)
        if callable(close):
            close()



def run_composed_policy_value_checkpoint_resume_phase(
    *,
    config_path: str | Path,
    phase: PhaseSpec,
    phase_output_dir: str | Path,
    schedule_name: str,
    run_name: str,
    teacher_checkpoint: str | Path,
    teacher_config_path: str | Path | None,
    student_actor_head_path: str | Path,
    student_critic_head_path: str | Path,
    resume_checkpoint: str | Path,
    expected_source_global_step: int,
    target_cumulative_global_step: int,
    continuation_seed: int,
    checkpoint_milestone_update_labels: dict[int, str],
    apply_phase_learning_rates_after_resume: bool = False,
    freeze_feature_extractor: bool = False,
    use_best_heads: bool = True,
    save_checkpoint: bool = True,
) -> Path:
    """Resume a composed hybrid PPO checkpoint with optimizer state intact.

    SimplyQRL 1.5 checkpoint metadata retains the classical teacher identity
    after head composition.  A direct ``PPO.load`` would therefore rebuild the
    wrong architecture.  This function reconstructs teacher + KD heads first,
    then restores policy and optimizer state from the composed checkpoint.
    """

    import torch
    from simplyqrl.ppo import PPO
    from simplyqrl.staged import (
        compose_policy_from_teacher,
        is_feature_extractor_frozen,
        rebuild_ppo_optimizer_after_composition,
    )

    config, total_timesteps, progress_bar = _phase_runtime_config(
        config_path=config_path,
        phase=phase,
        phase_output_dir=phase_output_dir,
        schedule_name=schedule_name,
        run_name=run_name,
    )
    reporting_config = dict(config.get("reporting") or {"enabled": False})
    torch_device = _torch_device_from_config(config)
    phase_training_hparams_expected = _phase_training_hparams_from_config(config)
    if int(phase_training_hparams_expected.get("seed", continuation_seed)) != int(
        continuation_seed
    ):
        raise RuntimeError("Continuation seed does not match the active PPO config")

    teacher_env_override = _resolve_env_override_from_config(teacher_config_path)
    model = PPO.load(
        str(teacher_checkpoint),
        env=teacher_env_override,
        device=torch_device,
        allow_pickle=True,
        reporting_config=reporting_config,
        run_name=run_name,
        hparams_overrides=phase_training_hparams_expected,
        load_optimizer_state=False,
    )
    try:
        compose_policy_from_teacher(
            model,
            actor_head_artifact=student_actor_head_path,
            critic_head_artifact=student_critic_head_path,
            freeze_feature_extractor=bool(freeze_feature_extractor),
            use_best=bool(use_best_heads),
            qnode_scope="actor",
            critic_qnode_scope="critic",
        )
        apply_policy = getattr(model, "_apply_qnode_wrapper_policy", None)
        if callable(apply_policy):
            apply_policy()
        rebuild_ppo_optimizer_after_composition(model)

        resume_checkpoint = Path(resume_checkpoint).expanduser().resolve()
        with zipfile.ZipFile(resume_checkpoint, "r") as archive:
            source_meta = json.loads(archive.read("meta.json"))
            policy_state = torch.load(
                BytesIO(archive.read("policy.pth")), map_location=torch_device
            )
            optimizer_state = torch.load(
                BytesIO(archive.read("optim.pth")), map_location=torch_device
            )
        source_global_step = int(source_meta.get("global_step", -1))
        if source_global_step != int(expected_source_global_step):
            raise RuntimeError(
                "Composed continuation source global_step mismatch: "
                f"checkpoint={source_global_step}, expected={expected_source_global_step}"
            )
        model.agent.load_state_dict(policy_state)
        model.optimizer.load_state_dict(optimizer_state)
        model.global_step = source_global_step

        optimizer_group_learning_rates_restored = _optimizer_group_learning_rates(model)
        expected_optimizer_group_learning_rates = _expected_optimizer_group_learning_rates(config)
        optimizer_group_learning_rates_overridden: dict[str, float] = {}
        if apply_phase_learning_rates_after_resume:
            optimizer_group_learning_rates_overridden = _set_optimizer_group_learning_rates(
                model, expected=expected_optimizer_group_learning_rates
            )
        optimizer_group_learning_rates = _optimizer_group_learning_rates(model)
        optimizer_group_lr_mismatches = _optimizer_group_lr_mismatches(
            expected=expected_optimizer_group_learning_rates,
            actual=optimizer_group_learning_rates,
        )
        if optimizer_group_lr_mismatches:
            raise RuntimeError(
                "Restored composed PPO optimizer does not match the phase config: "
                f"{optimizer_group_lr_mismatches}"
            )

        phase_run_dir = _summary_from_artifact_logger(model)
        composition_summary_path = phase_run_dir / "composition_summary.json"
        phase_training_hparams_actual = _model_training_hparams(model)
        phase_training_hparam_mismatches = _hparam_mismatches(
            expected=phase_training_hparams_expected,
            actual=phase_training_hparams_actual,
        )
        composition_summary = {
            "initialization": "checkpoint_resume_composed_policy_value",
            "composition_scope": "policy_value",
            "schedule_name": schedule_name,
            "phase_id": phase.phase_id,
            "phase_name": phase.phase_name,
            "teacher_checkpoint": str(teacher_checkpoint),
            "teacher_config_path": (
                str(teacher_config_path) if teacher_config_path is not None else None
            ),
            "student_actor_head_path": str(student_actor_head_path),
            "student_critic_head_path": str(student_critic_head_path),
            "resume_checkpoint": str(resume_checkpoint),
            "resume_checkpoint_global_step": source_global_step,
            "optimizer_state_restored": True,
            "rng_state_restored": False,
            "continuation_seed": int(continuation_seed),
            "target_cumulative_global_step": int(target_cumulative_global_step),
            "base_checkpoint_agent_type": "v15_classical",
            "effective_agent_type": "composed_policy_value_hybrid",
            "phase_training_hparams_expected": phase_training_hparams_expected,
            "phase_training_hparams_actual": phase_training_hparams_actual,
            "phase_training_hparam_mismatches": phase_training_hparam_mismatches,
            "hparams_match_phase_config": not bool(phase_training_hparam_mismatches),
            "freeze_feature_extractor": bool(freeze_feature_extractor),
            "feature_interface_trainable_during_phase": not bool(freeze_feature_extractor),
            "feature_extractor_runtime_frozen": bool(is_feature_extractor_frozen(model)),
            **_feature_extractor_param_status(model),
            "optimizer_rebuilt_before_restore": True,
            "optimizer_parameter_group_count": len(model.optimizer.param_groups),
            "apply_phase_learning_rates_after_resume": bool(
                apply_phase_learning_rates_after_resume
            ),
            "optimizer_group_learning_rates_restored": (
                optimizer_group_learning_rates_restored
            ),
            "optimizer_group_learning_rates_overridden": (
                optimizer_group_learning_rates_overridden
            ),
            "optimizer_group_learning_rates": optimizer_group_learning_rates,
            "expected_optimizer_group_learning_rates": expected_optimizer_group_learning_rates,
            "optimizer_group_lr_mismatches": optimizer_group_lr_mismatches,
            "use_best_heads": bool(use_best_heads),
        }
        write_json(composition_summary_path, composition_summary)

        milestone_monitor = _LabeledUpdateCheckpointMonitor(
            model=model,
            output_dir=phase_run_dir / "milestones",
            update_labels=checkpoint_milestone_update_labels,
            metadata={
                "schedule_name": schedule_name,
                "phase_id": phase.phase_id,
                "phase_name": phase.phase_name,
                "phase_type": phase.phase_type,
                "initialization": "checkpoint_resume_composed_policy_value",
                "composition_scope": "policy_value",
                "source_checkpoint": str(resume_checkpoint),
                "source_global_step": source_global_step,
            },
        )
        model.monitor_manager.add(milestone_monitor)

        start_global_step = int(model.global_step)
        model.train(total_timesteps=total_timesteps, progress_bar=progress_bar)
        execution_summary = write_phase_execution_summary(
            phase_run_dir,
            model=model,
            configured_timesteps=total_timesteps,
            start_global_step=start_global_step,
        )
        final_global_step = int(execution_summary["final_global_step"])
        continuation_summary = {
            "initialization_mode": "checkpoint_resume",
            "source_run_dir": str(resume_checkpoint.parent),
            "source_checkpoint": str(resume_checkpoint),
            "source_global_step": source_global_step,
            "continuation_seed": int(continuation_seed),
            "optimizer_state_restored": True,
            "apply_phase_learning_rates_after_resume": bool(
                apply_phase_learning_rates_after_resume
            ),
            "optimizer_learning_rates_restored_before_override": (
                optimizer_group_learning_rates_restored
            ),
            "optimizer_learning_rates_overridden": (
                optimizer_group_learning_rates_overridden
            ),
            "optimizer_learning_rates_at_resume": [
                float(group["lr"]) for group in model.optimizer.param_groups
            ],
            "rng_state_restored": False,
            "rng_note": (
                "SimplyQRL 1.5 checkpoints do not persist Python/NumPy/Torch/environment "
                "RNG state; Gate D3 uses an explicit deterministic continuation seed."
            ),
            "continuation_start_global_step": start_global_step,
            "target_cumulative_global_step": int(target_cumulative_global_step),
            "additional_configured_env_steps": int(total_timesteps),
            "additional_effective_env_steps": int(execution_summary["effective_env_steps"]),
            "continuation_final_global_step": final_global_step,
            "target_global_step_reached": final_global_step
            == int(target_cumulative_global_step),
            "checkpoint_milestones": {
                label: str(path) for label, path in milestone_monitor.saved.items()
            },
        }
        write_json(phase_run_dir / "continuation_summary.json", continuation_summary)
        execution_summary.update(continuation_summary)
        write_json(phase_run_dir / "phase_execution_summary.json", execution_summary)

        if final_global_step != int(target_cumulative_global_step):
            raise RuntimeError(
                "Gate D3 continuation did not reach its cumulative target: "
                f"final={final_global_step}, target={target_cumulative_global_step}"
            )
        expected_labels = set(checkpoint_milestone_update_labels.values())
        if set(milestone_monitor.saved) != expected_labels:
            raise RuntimeError(
                "Gate D3 continuation did not save all requested milestones: "
                f"saved={sorted(milestone_monitor.saved)}, expected={sorted(expected_labels)}"
            )

        if save_checkpoint:
            checkpoint_base = phase_run_dir / "checkpoint"
            model.save(
                str(checkpoint_base),
                metadata={
                    "schedule_name": schedule_name,
                    "phase_id": phase.phase_id,
                    "phase_name": phase.phase_name,
                    "phase_type": phase.phase_type,
                    "phase_steps": phase.steps,
                    "phase_effective_env_steps": execution_summary["effective_env_steps"],
                    "phase_step_overshoot": execution_summary["step_overshoot"],
                    "initialization": "checkpoint_resume_composed_policy_value",
                    "composition_scope": "policy_value",
                    "source_checkpoint": str(resume_checkpoint),
                    "source_global_step": source_global_step,
                    "continuation_seed": int(continuation_seed),
                    "rng_state_restored": False,
                },
            )

        final_summary = dict(composition_summary)
        final_summary.update(
            {
                "phase_run_dir": str(phase_run_dir),
                "checkpoint_path": str(phase_run_dir / "checkpoint.zip")
                if save_checkpoint
                else None,
                "feature_extractor_runtime_frozen_after_training": bool(
                    is_feature_extractor_frozen(model)
                ),
                **_feature_extractor_param_status(model),
                "qnode_stats_after_training": getattr(
                    model.agent, "get_qnode_stats", lambda: {}
                )(),
                "checkpoint_milestone_update_labels": {
                    str(update): label
                    for update, label in checkpoint_milestone_update_labels.items()
                },
                "checkpoint_milestones": {
                    label: str(path) for label, path in milestone_monitor.saved.items()
                },
            }
        )
        write_json(composition_summary_path, final_summary)
        return phase_run_dir
    finally:
        close = getattr(model, "close", None)
        if callable(close):
            close()
