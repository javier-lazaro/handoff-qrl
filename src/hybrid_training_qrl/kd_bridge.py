"""Offline KD bridge utilities for delayed QNode schedules.

The functions in this module intentionally stay in the experiment repository.
They use SimplyQRL's public staged API (``PPO.load``, ``get_teacher_outputs``,
``build_head_from_config`` and ``save_head_artifact``) but own the paper-specific
KD dataset/export/training artifacts and metrics.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from hybrid_training_qrl.schedules import PhaseSpec, ScheduleSpec
from hybrid_training_qrl.summaries import write_json


@dataclass(frozen=True)
class KDBridgeResult:
    """Result of an offline policy-value KD bridge phase."""

    phase_dir: Path
    metrics: dict[str, Any]
    student_actor_head_path: Path
    student_critic_head_path: Path
    dataset_path: Path


def _int_meta(phase: PhaseSpec, key: str, default: int) -> int:
    value = phase.metadata.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer, got boolean {value!r}.")
    return int(value)


def _float_meta(phase: PhaseSpec, key: str, default: float) -> float:
    value = phase.metadata.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a number, got boolean {value!r}.")
    return float(value)


def _bool_meta(phase: PhaseSpec, key: str, default: bool) -> bool:
    value = phase.metadata.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _as_numpy_batch(outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value) for key, value in outputs.items()}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_div(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else numerator / denominator


def _start_codecarbon(
    *,
    enabled: bool,
    phase_dir: Path,
    project_name: str,
    measure_power_secs: int,
    log_level: str,
    fail_on_error: bool,
    output_file: str = "kd_codecarbon.csv",
):
    if not enabled:
        return None, {
            "codecarbon_enabled": False,
            "codecarbon_tracker_status": "disabled",
            "codecarbon_error": None,
            "codecarbon_output_file": None,
        }
    try:
        from codecarbon import EmissionsTracker

        tracker_kwargs = {
            "project_name": project_name,
            "output_dir": str(phase_dir),
            "output_file": str(output_file),
            "log_level": log_level,
            "measure_power_secs": int(measure_power_secs),
            "save_to_file": True,
        }
        try:
            tracker = EmissionsTracker(**tracker_kwargs)
        except TypeError:
            tracker_kwargs.pop("save_to_file", None)
            tracker_kwargs.pop("measure_power_secs", None)
            tracker = EmissionsTracker(**tracker_kwargs)
        tracker.start()
        return tracker, {
            "codecarbon_enabled": True,
            "codecarbon_tracker_status": "running",
            "codecarbon_error": None,
            "codecarbon_output_file": str(phase_dir / output_file),
        }
    except Exception as exc:
        if fail_on_error:
            raise
        return None, {
            "codecarbon_enabled": True,
            "codecarbon_tracker_status": "error",
            "codecarbon_error": repr(exc),
            "codecarbon_output_file": str(phase_dir / output_file),
        }


def _stop_codecarbon(tracker: Any, state: dict[str, Any], *, fail_on_error: bool) -> dict[str, Any]:
    if tracker is None:
        return dict(state)
    try:
        emissions = tracker.stop()
        result = dict(state)
        result["codecarbon_tracker_status"] = "stopped"
        result["codecarbon_emissions_kg"] = None if emissions is None else float(emissions)
        output_path = Path(str(result["codecarbon_output_file"]))
        if output_path.exists():
            import csv

            with output_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                result["codecarbon_energy_kwh"] = _safe_float(last.get("energy_consumed"))
                result["codecarbon_duration_seconds"] = _safe_float(last.get("duration"))
        return result
    except Exception as exc:
        if fail_on_error:
            raise
        result = dict(state)
        result["codecarbon_tracker_status"] = "error"
        result["codecarbon_error"] = repr(exc)
        return result


def _resolve_env_override_from_config(config_path: str | Path | None):
    """Return an env factory for custom teacher env configs when needed.

    SimplyQRL checkpoints store ``env_id`` as a label.  For custom factories
    such as MinAtarFrameStack4 this label is intentionally not a Gymnasium id,
    so loading the checkpoint for KD must pass the original env factory again.
    Standard Gymnasium ids return ``None`` and keep the default PPO.load path.
    """

    if config_path is None:
        return None

    from simplyqrl.runners.train_ppo import load_yaml, resolve_env

    config = load_yaml(config_path)
    _env_id, env_factory = resolve_env(config.get("env"))
    return env_factory


def _load_teacher(
    checkpoint_path: str | Path,
    *,
    device: str,
    teacher_config_path: str | Path | None = None,
):
    from simplyqrl.ppo import PPO

    env_override = _resolve_env_override_from_config(teacher_config_path)

    return PPO.load(
        str(checkpoint_path),
        env=env_override,
        device=device,
        allow_pickle=True,
        reporting_config={"enabled": False},
        run_name="KD-teacher-loaded",
    )


def export_teacher_dataset(
    *,
    teacher_checkpoint: str | Path,
    phase_dir: str | Path,
    kd_samples: int,
    deterministic: bool = True,
    device: str = "cpu",
    seed: int | None = None,
    teacher_config_path: str | Path | None = None,
    output_name: str = "part_00000.npz",
    policy_seed: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Export a small feature/logit/value/action KD dataset from a PPO teacher."""

    if kd_samples <= 0:
        raise ValueError("kd_samples must be positive for offline_policy_value_kd_micro.")

    phase_dir = Path(phase_dir)
    dataset_dir = phase_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / str(output_name)

    export_start_time = time.time()
    teacher = _load_teacher(
        teacher_checkpoint,
        device=device,
        teacher_config_path=teacher_config_path,
    )
    try:
        if seed is None:
            seed = int(getattr(teacher, "seed", 0))
        if policy_seed is None:
            policy_seed = int(seed)
        random.seed(int(policy_seed))
        np.random.seed(int(policy_seed))
        torch.manual_seed(int(policy_seed))
        obs_np, _ = teacher.envs.reset(seed=seed)

        features: list[np.ndarray] = []
        logits: list[np.ndarray] = []
        values: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        greedy_actions: list[np.ndarray] = []
        rewards: list[float] = []
        dones: list[bool] = []
        env_indices: list[int] = []
        episode_ids: list[int] = []
        step_ids: list[int] = []

        num_envs = int(getattr(teacher, "num_envs", np.asarray(obs_np).shape[0]))
        episode_ids_by_env = np.arange(num_envs, dtype=np.int64)
        step_ids_by_env = np.zeros(num_envs, dtype=np.int64)
        next_episode_id = int(num_envs)
        while len(actions) < kd_samples:
            outputs = _as_numpy_batch(
                teacher.get_teacher_outputs(obs_np, deterministic=deterministic)
            )
            action_np = np.asarray(outputs["actions"])
            greedy_action_np = np.argmax(np.asarray(outputs["logits"]), axis=-1)
            next_obs_np, reward_np, terminated_np, truncated_np, _infos = teacher.envs.step(
                action_np
            )
            done_np = np.logical_or(terminated_np, truncated_np)

            batch_size = int(action_np.shape[0]) if action_np.ndim > 0 else 1
            for idx in range(batch_size):
                if len(actions) >= kd_samples:
                    break
                features.append(np.asarray(outputs["features"])[idx].copy())
                logits.append(np.asarray(outputs["logits"])[idx].copy())
                values.append(np.asarray(outputs["values"])[idx].copy())
                actions.append(np.asarray(action_np[idx]).copy())
                greedy_actions.append(np.asarray(greedy_action_np[idx]).copy())
                rewards.append(float(np.asarray(reward_np)[idx]))
                done = bool(np.asarray(done_np)[idx])
                dones.append(done)
                env_indices.append(int(idx))
                episode_ids.append(int(episode_ids_by_env[idx]))
                step_ids.append(int(step_ids_by_env[idx]))

            for idx in range(batch_size):
                if bool(np.asarray(done_np)[idx]):
                    episode_ids_by_env[idx] = next_episode_id
                    next_episode_id += 1
                    step_ids_by_env[idx] = 0
                else:
                    step_ids_by_env[idx] += 1

            obs_np = next_obs_np

        features_np = np.stack(features).astype(np.float32)
        logits_np = np.stack(logits).astype(np.float32)
        values_np = np.stack(values).astype(np.float32).reshape(len(values), -1)
        actions_np = np.asarray(actions, dtype=np.int64).reshape(-1)
        greedy_actions_np = np.asarray(greedy_actions, dtype=np.int64).reshape(-1)
        rewards_np = np.asarray(rewards, dtype=np.float32)
        dones_np = np.asarray(dones, dtype=np.bool_)
        env_indices_np = np.asarray(env_indices, dtype=np.int64)
        episode_ids_np = np.asarray(episode_ids, dtype=np.int64)
        step_ids_np = np.asarray(step_ids, dtype=np.int64)

        np.savez_compressed(
            dataset_path,
            features=features_np,
            teacher_logits=logits_np,
            teacher_values=values_np,
            teacher_actions=actions_np,
            teacher_actions_sampled=actions_np,
            teacher_actions_greedy=greedy_actions_np,
            rewards=rewards_np,
            dones=dones_np,
            env_indices=env_indices_np,
            episode_ids=episode_ids_np,
            step_ids=step_ids_np,
        )

        metadata = {
            "dataset_path": str(dataset_path),
            "teacher_checkpoint": str(teacher_checkpoint),
            "teacher_config_path": str(teacher_config_path) if teacher_config_path is not None else None,
            "teacher_env_factory_overridden": bool(
                _resolve_env_override_from_config(teacher_config_path) is not None
            ),
            "kd_samples": int(kd_samples),
            "feature_dim": int(features_np.shape[1]),
            "action_dim": int(logits_np.shape[1]),
            "value_dim": int(values_np.shape[1]),
            "deterministic": bool(deterministic),
            "seed": int(seed),
            "environment_seed": int(seed),
            "policy_seed": int(policy_seed),
            "episodes_touched": int(np.unique(episode_ids_np).size),
            "vector_env_count": int(num_envs),
            "vectorized_episode_tracking": True,
            "reward_mean": float(np.mean(rewards_np)) if rewards_np.size else None,
            "done_count": int(np.sum(dones_np)),
            "environment_interactions": int(features_np.shape[0]),
            "dataset_export_wall_clock": float(time.time() - export_start_time),
        }
        summary_name = (
            "kd_dataset_summary.json"
            if str(output_name) == "part_00000.npz"
            else f"{Path(output_name).stem}_summary.json"
        )
        write_json(phase_dir / summary_name, metadata)
        return dataset_path, metadata
    finally:
        close = getattr(teacher, "close", None)
        if callable(close):
            close()


def _load_dataset(dataset_path: str | Path) -> dict[str, torch.Tensor]:
    with np.load(dataset_path) as data:
        return {
            "features": torch.as_tensor(data["features"], dtype=torch.float32),
            "teacher_logits": torch.as_tensor(data["teacher_logits"], dtype=torch.float32),
            "teacher_values": torch.as_tensor(data["teacher_values"], dtype=torch.float32).reshape(-1, 1),
            "teacher_actions": torch.as_tensor(
                data["teacher_actions_greedy"]
                if "teacher_actions_greedy" in data.files
                else data["teacher_actions"],
                dtype=torch.long,
            ),
        }


def _complete_head_config(
    *,
    raw_config: dict[str, Any],
    feature_dim: int,
    output_dim: int,
    role: str,
    label: str,
) -> dict[str, Any]:
    cfg = copy.deepcopy(raw_config)
    cfg.setdefault("type", cfg.get("family", "hybrid"))
    cfg.setdefault("family", cfg.get("type", "hybrid"))
    cfg["role"] = str(cfg.get("role", role)).lower()
    if cfg["role"] != role:
        raise ValueError(f"{label}.role must be {role!r}, got {cfg['role']!r}.")
    cfg["input_dim"] = int(cfg.get("input_dim", feature_dim))
    cfg["output_dim"] = int(cfg.get("output_dim", output_dim))
    if int(cfg["input_dim"]) != int(feature_dim):
        raise ValueError(
            f"{label}.input_dim={cfg['input_dim']} does not match dataset feature_dim={feature_dim}."
        )
    if int(cfg["output_dim"]) != int(output_dim):
        raise ValueError(
            f"{label}.output_dim={cfg['output_dim']} does not match expected output_dim={output_dim}."
        )
    return cfg


def _policy_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)


def _policy_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction="batchmean",
    )


def _evaluate_policy_value_heads(
    *,
    actor_head: torch.nn.Module,
    critic_head: torch.nn.Module,
    features: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_values: torch.Tensor,
    teacher_actions: torch.Tensor,
    temperature: float,
    policy_weight: float,
    value_weight: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    policy_losses: list[float] = []
    value_losses: list[float] = []
    total_losses: list[float] = []
    kls: list[float] = []
    correct = 0
    count = 0

    actor_training = actor_head.training
    critic_training = critic_head.training
    actor_head.eval()
    critic_head.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            x = features[start:end].to(device)
            teacher_policy = teacher_logits[start:end].to(device)
            teacher_value = teacher_values[start:end].to(device)
            teacher_action = teacher_actions[start:end].to(device)

            student_policy = actor_head(x)
            student_value = critic_head(x).reshape_as(teacher_value)

            policy = _policy_loss(
                student_policy,
                teacher_policy,
                temperature=temperature,
            )
            value = F.mse_loss(student_value, teacher_value)
            total = policy_weight * policy + value_weight * value
            kl = _policy_kl(student_policy, teacher_policy)

            policy_losses.append(float(policy.detach().cpu()))
            value_losses.append(float(value.detach().cpu()))
            total_losses.append(float(total.detach().cpu()))
            kls.append(float(kl.detach().cpu()))
            predicted = torch.argmax(student_policy, dim=-1)
            correct += int((predicted == teacher_action).sum().detach().cpu())
            count += int(teacher_action.numel())

    if actor_training:
        actor_head.train()
    if critic_training:
        critic_head.train()

    agreement = float(correct / count) if count else 0.0
    return {
        "total_loss": float(np.mean(total_losses)),
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "policy_kl": float(np.mean(kls)),
        "value_mse": float(np.mean(value_losses)),
        "action_agreement": agreement,
    }


def _qnode_totals(snapshot: dict[str, Any]) -> tuple[int, float, int, int, float, float]:
    actor_calls = int(snapshot.get("qnode_calls_actor", 0) or 0)
    critic_calls = int(snapshot.get("qnode_calls_critic", 0) or 0)
    actor_time = float(snapshot.get("qnode_time_actor", 0.0) or 0.0)
    critic_time = float(snapshot.get("qnode_time_critic", 0.0) or 0.0)

    if actor_calls or critic_calls:
        total_calls = actor_calls + critic_calls
    else:
        total_calls = int(snapshot.get("qnode_calls_total", 0) or 0)

    if actor_time or critic_time:
        total_time = actor_time + critic_time
    else:
        total_time = float(snapshot.get("qnode_time_total", 0.0) or 0.0)

    return total_calls, total_time, actor_calls, critic_calls, actor_time, critic_time


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _state_dict_sha256(*modules: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for module in modules:
        for key, value in sorted(module.state_dict().items()):
            digest.update(key.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _snapshot_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float | int]:
    keys = (
        "qnode_calls_total",
        "qnode_time_total",
        "qnode_calls_actor",
        "qnode_calls_critic",
        "qnode_time_actor",
        "qnode_time_critic",
    )
    result: dict[str, float | int] = {}
    for key in keys:
        delta = (after.get(key, 0) or 0) - (before.get(key, 0) or 0)
        result[key] = int(delta) if "calls" in key else float(delta)
    return result


def _metric_payload(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def train_policy_value_kd_heads(
    *,
    dataset_path: str | Path,
    phase_dir: str | Path,
    student_actor_head_config: dict[str, Any],
    student_critic_head_config: dict[str, Any],
    kd_epochs: int,
    kd_batch_size: int,
    kd_learning_rate: float,
    kd_temperature: float,
    kd_policy_weight: float,
    kd_value_weight: float,
    device: str = "cpu",
    validation_dataset_path: str | Path | None = None,
    student_init_seed: int = 0,
    dataloader_seed: int = 0,
) -> tuple[Path, Path, dict[str, Any]]:
    """Train actor/critic heads with deterministic init and validation selection.

    ``validation_dataset_path`` is mandatory for the KD-calibration path.
    It remains optional only for backwards compatibility with archived micro
    schedules; in that legacy mode the training dataset is reused for selection.
    """

    from simplyqrl.instrumentation import QNodeStats
    from simplyqrl.staged import build_head_from_config, normalize_head_config, save_head_artifact

    if kd_epochs <= 0:
        raise ValueError("kd_epochs must be positive for offline policy-value KD.")
    if kd_batch_size <= 0:
        raise ValueError("kd_batch_size must be positive for offline policy-value KD.")

    phase_dir = Path(phase_dir)
    phase_dir.mkdir(parents=True, exist_ok=True)
    train_data = _load_dataset(dataset_path)
    validation_path = Path(validation_dataset_path) if validation_dataset_path else Path(dataset_path)
    validation_data = _load_dataset(validation_path)

    features = train_data["features"]
    teacher_logits = train_data["teacher_logits"]
    teacher_values = train_data["teacher_values"]
    teacher_actions = train_data["teacher_actions"]
    val_features = validation_data["features"]
    val_teacher_logits = validation_data["teacher_logits"]
    val_teacher_values = validation_data["teacher_values"]
    val_teacher_actions = validation_data["teacher_actions"]

    feature_dim = int(features.shape[1])
    action_dim = int(teacher_logits.shape[1])
    value_dim = int(teacher_values.shape[1])
    if int(val_features.shape[1]) != feature_dim:
        raise ValueError("Validation feature dimension does not match the training dataset.")
    if int(val_teacher_logits.shape[1]) != action_dim:
        raise ValueError("Validation action dimension does not match the training dataset.")
    if int(val_teacher_values.shape[1]) != value_dim:
        raise ValueError("Validation value dimension does not match the training dataset.")

    actor_config = normalize_head_config(
        _complete_head_config(
            raw_config=student_actor_head_config,
            feature_dim=feature_dim,
            output_dim=action_dim,
            role="actor",
            label="student_actor_head",
        )
    )
    critic_config = normalize_head_config(
        _complete_head_config(
            raw_config=student_critic_head_config,
            feature_dim=feature_dim,
            output_dim=value_dim,
            role="critic",
            label="student_critic_head",
        )
    )

    _seed_all(int(student_init_seed))
    qnode_stats = QNodeStats()
    actor_head = build_head_from_config(actor_config, qnode_stats=qnode_stats, qnode_scope="actor")
    critic_head = build_head_from_config(critic_config, qnode_stats=qnode_stats, qnode_scope="critic")
    initial_state_hash = _state_dict_sha256(actor_head, critic_head)

    torch_device = torch.device(device)
    actor_head.to(torch_device)
    critic_head.to(torch_device)
    optimizer = torch.optim.Adam(
        list(actor_head.parameters()) + list(critic_head.parameters()),
        lr=float(kd_learning_rate),
    )
    permutation_generator = torch.Generator(device="cpu")
    permutation_generator.manual_seed(int(dataloader_seed))

    train_eval_kwargs = dict(
        actor_head=actor_head,
        critic_head=critic_head,
        features=features,
        teacher_logits=teacher_logits,
        teacher_values=teacher_values,
        teacher_actions=teacher_actions,
        temperature=float(kd_temperature),
        policy_weight=float(kd_policy_weight),
        value_weight=float(kd_value_weight),
        batch_size=int(kd_batch_size),
        device=torch_device,
    )
    validation_eval_kwargs = dict(
        actor_head=actor_head,
        critic_head=critic_head,
        features=val_features,
        teacher_logits=val_teacher_logits,
        teacher_values=val_teacher_values,
        teacher_actions=val_teacher_actions,
        temperature=float(kd_temperature),
        policy_weight=float(kd_policy_weight),
        value_weight=float(kd_value_weight),
        batch_size=int(kd_batch_size),
        device=torch_device,
    )

    metrics_path = phase_dir / "kd_metrics.jsonl"
    start_time = time.time()
    qnode_start = qnode_stats.snapshot()
    initial_train = _evaluate_policy_value_heads(**train_eval_kwargs)
    initial_validation = _evaluate_policy_value_heads(**validation_eval_kwargs)

    best_epoch = 0
    best_loss = float(initial_validation["total_loss"])
    best_actor_state_dict = copy.deepcopy(actor_head.state_dict())
    best_critic_state_dict = copy.deepcopy(critic_head.state_dict())
    optimization_qnode = {
        "qnode_calls_total": 0,
        "qnode_time_total": 0.0,
        "qnode_calls_actor": 0,
        "qnode_calls_critic": 0,
        "qnode_time_actor": 0.0,
        "qnode_time_critic": 0.0,
    }

    with metrics_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "kd_initial",
                    "epoch": 0,
                    **_metric_payload("train", initial_train),
                    **_metric_payload("validation", initial_validation),
                },
                sort_keys=True,
            )
            + "\n"
        )

        num_samples = int(features.shape[0])
        for epoch in range(1, int(kd_epochs) + 1):
            permutation = torch.randperm(num_samples, generator=permutation_generator)
            batch_total_losses: list[float] = []
            batch_policy_losses: list[float] = []
            batch_value_losses: list[float] = []
            actor_head.train()
            critic_head.train()
            before_optimization = qnode_stats.snapshot()
            for start_index in range(0, num_samples, int(kd_batch_size)):
                idx = permutation[start_index : start_index + int(kd_batch_size)]
                x = features[idx].to(torch_device)
                teacher_policy = teacher_logits[idx].to(torch_device)
                teacher_value = teacher_values[idx].to(torch_device)

                student_policy = actor_head(x)
                student_value = critic_head(x).reshape_as(teacher_value)
                policy = _policy_loss(student_policy, teacher_policy, temperature=float(kd_temperature))
                value = F.mse_loss(student_value, teacher_value)
                loss = float(kd_policy_weight) * policy + float(kd_value_weight) * value

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                batch_total_losses.append(float(loss.detach().cpu()))
                batch_policy_losses.append(float(policy.detach().cpu()))
                batch_value_losses.append(float(value.detach().cpu()))

            after_optimization = qnode_stats.snapshot()
            delta = _snapshot_delta(before_optimization, after_optimization)
            for key, value in delta.items():
                optimization_qnode[key] = optimization_qnode[key] + value

            validation = _evaluate_policy_value_heads(**validation_eval_kwargs)
            if float(validation["total_loss"]) < best_loss:
                best_loss = float(validation["total_loss"])
                best_epoch = int(epoch)
                best_actor_state_dict = copy.deepcopy(actor_head.state_dict())
                best_critic_state_dict = copy.deepcopy(critic_head.state_dict())

            handle.write(
                json.dumps(
                    {
                        "event": "kd_epoch",
                        "epoch": int(epoch),
                        "train_total_loss_mean": float(np.mean(batch_total_losses)),
                        "train_policy_loss_mean": float(np.mean(batch_policy_losses)),
                        "train_value_loss_mean": float(np.mean(batch_value_losses)),
                        **_metric_payload("validation", validation),
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    final_actor_state_dict = copy.deepcopy(actor_head.state_dict())
    final_critic_state_dict = copy.deepcopy(critic_head.state_dict())
    final_train = _evaluate_policy_value_heads(**train_eval_kwargs)
    final_validation = _evaluate_policy_value_heads(**validation_eval_kwargs)

    actor_head.load_state_dict(best_actor_state_dict)
    critic_head.load_state_dict(best_critic_state_dict)
    best_train = _evaluate_policy_value_heads(**train_eval_kwargs)
    best_validation = _evaluate_policy_value_heads(**validation_eval_kwargs)
    actor_head.load_state_dict(final_actor_state_dict)
    critic_head.load_state_dict(final_critic_state_dict)

    kd_wall_clock = time.time() - start_time
    qnode_snapshot = qnode_stats.snapshot()
    total_calls, total_time, actor_calls, critic_calls, actor_time, critic_time = _qnode_totals(qnode_snapshot)
    evaluation_qnode = _snapshot_delta(qnode_start, qnode_snapshot)
    for key, value in optimization_qnode.items():
        evaluation_qnode[key] = evaluation_qnode[key] - value

    common_metadata = {
        "bridge_mode": "offline_policy_value_kd",
        "bridge_type": "policy_value_kd",
        "selection_metric": "validation_total_loss",
        "best_epoch": int(best_epoch),
        "kd_epochs": int(kd_epochs),
        "kd_samples": int(features.shape[0]),
        "kd_validation_samples": int(val_features.shape[0]),
        "kd_batch_size": int(kd_batch_size),
        "kd_temperature": float(kd_temperature),
        "kd_learning_rate": float(kd_learning_rate),
        "kd_policy_weight": float(kd_policy_weight),
        "kd_value_weight": float(kd_value_weight),
        "student_init_seed": int(student_init_seed),
        "dataloader_seed": int(dataloader_seed),
        "student_initialization_sha256": initial_state_hash,
        **_metric_payload("train_initial", initial_train),
        **_metric_payload("validation_initial", initial_validation),
        **_metric_payload("train_final", final_train),
        **_metric_payload("validation_final", final_validation),
        **_metric_payload("train_best", best_train),
        **_metric_payload("validation_best", best_validation),
    }

    student_actor_head_path = save_head_artifact(
        actor_head,
        actor_config,
        phase_dir / "student_actor_head.pt",
        best_state_dict=best_actor_state_dict,
        metadata={**common_metadata, "role": "actor"},
    )
    student_critic_head_path = save_head_artifact(
        critic_head,
        critic_config,
        phase_dir / "student_critic_head.pt",
        best_state_dict=best_critic_state_dict,
        metadata={**common_metadata, "role": "critic"},
    )

    summary: dict[str, Any] = {
        **common_metadata,
        "dataset_path": str(Path(dataset_path)),
        "validation_dataset_path": str(validation_path),
        "kd_wall_clock": float(kd_wall_clock),
        "kd_policy_kl": float(best_validation["policy_kl"]),
        "kd_value_mse": float(best_validation["value_mse"]),
        "kd_action_agreement": float(best_validation["action_agreement"]),
        "kd_total_loss_initial": float(initial_validation["total_loss"]),
        "kd_total_loss_final": float(final_validation["total_loss"]),
        "kd_total_loss_best": float(best_validation["total_loss"]),
        "kd_policy_loss_initial": float(initial_validation["policy_loss"]),
        "kd_policy_loss_final": float(final_validation["policy_loss"]),
        "kd_policy_loss_best": float(best_validation["policy_loss"]),
        "kd_value_loss_initial": float(initial_validation["value_loss"]),
        "kd_value_loss_final": float(final_validation["value_loss"]),
        "kd_value_loss_best": float(best_validation["value_loss"]),
        "kd_loss_initial": float(initial_validation["total_loss"]),
        "kd_loss_final": float(final_validation["total_loss"]),
        "student_actor_head_path": str(student_actor_head_path),
        "student_critic_head_path": str(student_critic_head_path),
        "kd_metrics_path": str(metrics_path),
        **qnode_snapshot,
    }
    summary["kd_qnode_calls"] = int(total_calls)
    summary["kd_qnode_time"] = float(total_time)
    summary["qnode_calls"] = int(total_calls)
    summary["qnode_time"] = float(total_time)
    summary["qnode_calls_actor"] = int(actor_calls)
    summary["qnode_time_actor"] = float(actor_time)
    summary["qnode_calls_critic"] = int(critic_calls)
    summary["qnode_time_critic"] = float(critic_time)
    for key, value in optimization_qnode.items():
        summary[f"kd_optimization_{key}"] = value
    for key, value in evaluation_qnode.items():
        summary[f"kd_monitoring_{key}"] = value

    write_json(phase_dir / "kd_summary.json", summary)
    write_json(
        phase_dir / "kd_config.json",
        {
            "student_actor_head": actor_config,
            "student_critic_head": critic_config,
            "kd_policy_weight": float(kd_policy_weight),
            "kd_value_weight": float(kd_value_weight),
            "student_init_seed": int(student_init_seed),
            "dataloader_seed": int(dataloader_seed),
            "selection_metric": "validation_total_loss",
        },
    )
    return student_actor_head_path, student_critic_head_path, summary


def run_offline_policy_value_kd_bridge(
    *,
    schedule: ScheduleSpec,
    phase: PhaseSpec,
    phase_dir: str | Path,
    teacher_checkpoint: str | Path,
    teacher_config_path: str | Path | None = None,
) -> KDBridgeResult:
    """Run the technical offline policy-value KD bridge for Q25 smokes."""

    phase_dir = Path(phase_dir)
    phase_dir.mkdir(parents=True, exist_ok=True)

    kd_samples = _int_meta(phase, "kd_samples", 128)
    kd_epochs = _int_meta(phase, "kd_epochs", 3)
    kd_batch_size = _int_meta(phase, "kd_batch_size", 32)
    kd_learning_rate = _float_meta(phase, "kd_learning_rate", 1e-3)
    kd_temperature = _float_meta(phase, "kd_temperature", 1.0)
    kd_policy_weight = _float_meta(phase, "kd_policy_weight", 1.0)
    kd_value_weight = _float_meta(phase, "kd_value_weight", 0.5)
    deterministic = _bool_meta(phase, "deterministic", True)
    use_codecarbon = _bool_meta(phase, "use_codecarbon", False)
    codecarbon_measure_power_secs = _int_meta(phase, "codecarbon_measure_power_secs", 1)
    codecarbon_log_level = str(phase.metadata.get("codecarbon_log_level", "error"))
    fail_on_codecarbon_error = _bool_meta(phase, "fail_on_codecarbon_error", False)
    device = str(phase.metadata.get("device", "cpu"))
    seed = phase.metadata.get("seed")
    seed = None if seed is None else int(seed)
    student_init_seed = _int_meta(phase, "student_init_seed", 0)
    dataloader_seed = _int_meta(phase, "dataloader_seed", 0)
    student_actor_head = phase.metadata.get("student_actor_head")
    student_critic_head = phase.metadata.get("student_critic_head")
    if not isinstance(student_actor_head, dict):
        raise ValueError(
            "offline_policy_value_kd_micro requires a student_actor_head mapping in phase metadata."
        )
    if not isinstance(student_critic_head, dict):
        raise ValueError(
            "offline_policy_value_kd_micro requires a student_critic_head mapping in phase metadata."
        )

    tracker, codecarbon_state = _start_codecarbon(
        enabled=use_codecarbon,
        phase_dir=phase_dir,
        project_name=f"{schedule.schedule_name}-kd-bridge",
        measure_power_secs=codecarbon_measure_power_secs,
        log_level=codecarbon_log_level,
        fail_on_error=fail_on_codecarbon_error,
    )
    bridge_start_time = time.time()
    try:
        dataset_path, dataset_summary = export_teacher_dataset(
            teacher_checkpoint=teacher_checkpoint,
            phase_dir=phase_dir,
            kd_samples=kd_samples,
            deterministic=deterministic,
            device=device,
            seed=seed,
            teacher_config_path=teacher_config_path,
        )
        (
            student_actor_head_path,
            student_critic_head_path,
            kd_summary,
        ) = train_policy_value_kd_heads(
            dataset_path=dataset_path,
            phase_dir=phase_dir,
            student_actor_head_config=student_actor_head,
            student_critic_head_config=student_critic_head,
            kd_epochs=kd_epochs,
            kd_batch_size=kd_batch_size,
            kd_learning_rate=kd_learning_rate,
            kd_temperature=kd_temperature,
            kd_policy_weight=kd_policy_weight,
            kd_value_weight=kd_value_weight,
            device=device,
            student_init_seed=student_init_seed,
            dataloader_seed=dataloader_seed,
        )
    finally:
        codecarbon_summary = _stop_codecarbon(
            tracker,
            codecarbon_state,
            fail_on_error=fail_on_codecarbon_error,
        )
    bridge_wall_clock = time.time() - bridge_start_time
    kd_summary.update(codecarbon_summary)
    kd_summary["kd_bridge_wall_clock_total"] = float(bridge_wall_clock)
    kd_summary["kd_dataset_export_wall_clock"] = _safe_float(
        dataset_summary.get("dataset_export_wall_clock")
    )
    kd_summary["kd_dataset_collection_env_steps"] = int(
        dataset_summary.get("environment_interactions", kd_samples)
    )
    write_json(phase_dir / "kd_summary.json", kd_summary)

    metrics = {
        "phase_run_dir": str(phase_dir),
        "wall_clock": float(kd_summary.get("kd_bridge_wall_clock_total", 0.0)),
        "qnode_calls": int(kd_summary.get("kd_qnode_calls", 0)),
        "qnode_time": float(kd_summary.get("kd_qnode_time", 0.0)),
        "qnode_calls_actor": int(kd_summary.get("qnode_calls_actor", 0)),
        "qnode_calls_critic": int(kd_summary.get("qnode_calls_critic", 0)),
        "qnode_time_actor": float(kd_summary.get("qnode_time_actor", 0.0)),
        "qnode_time_critic": float(kd_summary.get("qnode_time_critic", 0.0)),
        "codecarbon_energy_kwh": _safe_float(kd_summary.get("codecarbon_energy_kwh")),
        "codecarbon_emissions_kg": _safe_float(kd_summary.get("codecarbon_emissions_kg")),
        "codecarbon_duration_seconds": _safe_float(
            kd_summary.get("codecarbon_duration_seconds")
        ),
        "codecarbon_enabled": bool(kd_summary.get("codecarbon_enabled", False)),
        "codecarbon_tracker_status": kd_summary.get("codecarbon_tracker_status"),
        "codecarbon_error": kd_summary.get("codecarbon_error"),
        "codecarbon_output_file": kd_summary.get("codecarbon_output_file"),
        "bridge_mode": "offline_policy_value_kd_micro",
        "bridge_type": "policy_value_kd",
        "kd_samples": int(kd_samples),
        "kd_epochs": int(kd_epochs),
        "kd_wall_clock": float(kd_summary.get("kd_wall_clock", 0.0)),
        "kd_bridge_wall_clock_total": float(
            kd_summary.get("kd_bridge_wall_clock_total", 0.0)
        ),
        "kd_dataset_export_wall_clock": _safe_float(
            kd_summary.get("kd_dataset_export_wall_clock")
        ),
        "kd_dataset_collection_env_steps": int(
            kd_summary.get("kd_dataset_collection_env_steps", kd_samples)
        ),
        "kd_qnode_calls": int(kd_summary.get("kd_qnode_calls", 0)),
        "kd_qnode_time": float(kd_summary.get("kd_qnode_time", 0.0)),
        "kd_policy_weight": float(kd_policy_weight),
        "kd_value_weight": float(kd_value_weight),
        "kd_total_loss_initial": _safe_float(kd_summary.get("kd_total_loss_initial")),
        "kd_total_loss_final": _safe_float(kd_summary.get("kd_total_loss_final")),
        "kd_policy_loss_initial": _safe_float(kd_summary.get("kd_policy_loss_initial")),
        "kd_policy_loss_final": _safe_float(kd_summary.get("kd_policy_loss_final")),
        "kd_value_loss_initial": _safe_float(kd_summary.get("kd_value_loss_initial")),
        "kd_value_loss_final": _safe_float(kd_summary.get("kd_value_loss_final")),
        "kd_value_mse": _safe_float(kd_summary.get("kd_value_mse")),
        "kd_loss_initial": _safe_float(kd_summary.get("kd_loss_initial")),
        "kd_loss_final": _safe_float(kd_summary.get("kd_loss_final")),
        "kd_policy_kl": _safe_float(kd_summary.get("kd_policy_kl")),
        "kd_action_agreement": _safe_float(kd_summary.get("kd_action_agreement")),
        "student_actor_head_path": str(student_actor_head_path),
        "student_critic_head_path": str(student_critic_head_path),
        "dataset_path": str(dataset_path),
        "dataset_feature_dim": int(dataset_summary["feature_dim"]),
        "dataset_action_dim": int(dataset_summary["action_dim"]),
        "dataset_value_dim": int(dataset_summary["value_dim"]),
        "teacher_config_path": dataset_summary.get("teacher_config_path"),
        "kd_dataset_episodes_touched": int(dataset_summary.get("episodes_touched", 0)),
        "kd_dataset_done_count": int(dataset_summary.get("done_count", 0)),
        "kd_dataset_reward_mean": _safe_float(dataset_summary.get("reward_mean")),
        "teacher_env_factory_overridden": bool(
            dataset_summary.get("teacher_env_factory_overridden", False)
        ),
    }
    write_json(phase_dir / "kd_bridge_phase_summary.json", metrics)
    return KDBridgeResult(
        phase_dir=phase_dir,
        metrics=metrics,
        student_actor_head_path=student_actor_head_path,
        student_critic_head_path=student_critic_head_path,
        dataset_path=dataset_path,
    )
