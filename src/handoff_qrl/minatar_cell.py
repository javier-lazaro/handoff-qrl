"""Release-facing single-cell runner for the frozen MinAtar paper protocol.

This module deliberately implements the paper protocol directly rather than
publishing the historical campaign machinery.  A *cell* is one
(condition, training seed) pair.  Supported conditions are the full classical
reference, H(t) for t in {0, .25, .5, .75, 1}, and C(t) for t in
{0, .25, .5, .75}.  C(1) is undefined by design.

The runner is a protocol reproduction utility, not a bit-for-bit replay tool.
The paper's canonical numerical results live under ``results``.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

from hybrid_training_qrl.envs.minatar import make_minatar_breakout
from hybrid_training_qrl.kd_bridge import export_teacher_dataset, train_policy_value_kd_heads
from simplyqrl.ppo import PPO
from simplyqrl.runners.train_ppo import build_ppo_from_config
from simplyqrl.staged import (
    compose_policy_from_teacher,
    is_feature_extractor_frozen,
    rebuild_ppo_optimizer_after_composition,
)

TOTAL_UPDATES = 4_884
ENV_STEPS_PER_UPDATE = 8 * 128
TOTAL_ENV_STEPS = TOTAL_UPDATES * ENV_STEPS_PER_UPDATE
HANDOFF_UPDATES = {0.0: 0, 0.25: 1_221, 0.5: 2_442, 0.75: 3_663, 1.0: 4_884}
EVAL_EPISODES = 100
EVAL_ENV_SEED_START = 15_000
EVAL_POLICY_SEED_START = 25_000
EVAL_MAX_STEPS = 1_000


@dataclass(frozen=True)
class CellSpec:
    condition: str
    family: str
    handoff_t: float | None
    seed: int


@dataclass(frozen=True)
class Budget:
    total_updates: int = TOTAL_UPDATES
    num_envs: int = 8
    num_steps: int = 128
    update_epochs: int = 4
    num_minibatches: int = 4
    kd_train_samples: int = 65_536
    kd_validation_samples: int = 8_192
    kd_epochs: int = 20
    kd_batch_size: int = 128
    eval_episodes: int = EVAL_EPISODES
    max_eval_steps: int = EVAL_MAX_STEPS

    @property
    def env_steps_per_update(self) -> int:
        return self.num_envs * self.num_steps

    @classmethod
    def smoke(cls) -> "Budget":
        # Mechanics-only workload.  No performance claim may be made from it.
        return cls(
            total_updates=2,
            num_envs=2,
            num_steps=16,
            update_epochs=1,
            num_minibatches=2,
            kd_train_samples=32,
            kd_validation_samples=16,
            kd_epochs=1,
            kd_batch_size=16,
            eval_episodes=2,
            max_eval_steps=100,
        )


def parse_condition(raw: str, seed: int) -> CellSpec:
    condition = raw.strip()
    if condition in {"full_classical", "classical"}:
        return CellSpec("full_classical", "full_classical", None, int(seed))
    if len(condition) < 4 or condition[0] not in {"H", "C"} or not condition.endswith(")"):
        raise ValueError(f"Unsupported condition: {raw!r}")
    try:
        handoff_t = float(condition[2:-1])
    except ValueError as exc:
        raise ValueError(f"Invalid handoff condition: {raw!r}") from exc
    if handoff_t not in HANDOFF_UPDATES:
        raise ValueError(f"Handoff must be one of {sorted(HANDOFF_UPDATES)}, got {handoff_t}.")
    family = "hybrid" if condition.startswith("H") else "compact_classical"
    canonical = f"{condition[0]}({handoff_t:g})"
    if canonical == "C(1)":
        raise ValueError("C(1) is intentionally undefined: it would just be a compact copy of the teacher.")
    return CellSpec(canonical, family, handoff_t, int(seed))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(xs: list[float]) -> float:
    return float(statistics.fmean(xs))


def _std(xs: list[float]) -> float:
    return float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0


def _qstats(model: PPO) -> dict[str, Any]:
    getter = getattr(model.agent, "get_qnode_stats", None)
    if not callable(getter):
        return {"qnode_calls_total": 0, "qnode_time_total": 0.0}
    return dict(getter())


def _env_config() -> dict[str, Any]:
    return {
        "factory": "hybrid_training_qrl.envs.minatar:make_minatar_breakout",
        "label": "MinAtarBreakoutSingleFrame",
        "kwargs": {"frame_stack": 1},
    }


def _extractor_config(features_dim: int = 32) -> dict[str, Any]:
    return {
        "type": "cnn",
        "features_dim": int(features_dim),
        "tensor_format": "channels_last",
        "normalize_images": False,
    }


def _full_classical_head() -> dict[str, Any]:
    return {"type": "classical", "hidden_sizes": [64], "activation": "tanh"}


def _compact_classical_head(*, role: str | None = None, input_dim: int = 32) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "type": "classical",
        "family": "classical",
        "hidden_sizes": [8, 64],
        "activation": "tanh",
    }
    if role:
        cfg["role"] = role
        cfg["input_dim"] = int(input_dim)
        cfg["output_dim"] = 6 if role == "actor" else 1
    return cfg


def _hybrid_head(*, role: str | None = None, smoke: bool = False) -> dict[str, Any]:
    # The full protocol uses one complete D32 pass through an 8-qubit circuit.
    # Smoke mode preserves the same interface but swaps to a single D8 feature
    # interface only to make CI inexpensive; it is explicitly non-paper-facing.
    if smoke:
        quantum_core = {
            "factory": "hybrid_training_qrl.quantum.circuits:build_angle_rx_ryrz_cnot_reuploading",
            "q_output_dim": 8,
            "kwargs": {
                "n_qubits": 8,
                "n_layers_q": 1,
                "entangler": "cnot_circular",
                "qml_device": "default.qubit",
                "diff_method": "backprop",
            },
        }
        hidden_sizes = [8]
    else:
        quantum_core = {
            "factory": "hybrid_training_qrl.quantum.circuits:build_chunked_angle_rx_ryrz_cnot",
            "q_output_dim": 8,
            "kwargs": {
                "n_qubits": 8,
                "entangler": "cnot_circular",
                "qml_device": "default.qubit",
                "diff_method": "backprop",
            },
        }
        hidden_sizes = [64]
    cfg: dict[str, Any] = {
        "type": "hybrid",
        "family": "hybrid",
        "input_adapter": {"type": "identity"},
        "quantum_core": quantum_core,
        "postprocessing": {"reuse_repetitions": 1, "use_output_scaling": False},
        "hidden_sizes": hidden_sizes,
        "activation": "tanh",
    }
    if role:
        cfg["role"] = role
        cfg["input_dim"] = 8 if smoke else 32
        cfg["output_dim"] = 6 if role == "actor" else 1
    return cfg


def _ppo_runtime_config(
    *,
    seed: int,
    family: str,
    total_timesteps: int,
    budget: Budget,
    run_name: str,
    smoke: bool,
) -> dict[str, Any]:
    if family == "full_classical":
        head_type = "classical"
        head = _full_classical_head()
        agent_type = "v15_classical"
        extractor_dim = 8 if smoke else 32
    elif family == "compact_classical":
        head_type = "classical"
        head = _compact_classical_head()
        agent_type = "v15_classical"
        extractor_dim = 8 if smoke else 32
    elif family == "hybrid":
        head_type = "hybrid"
        head = _hybrid_head(smoke=smoke)
        agent_type = "v15_hybrid"
        extractor_dim = 8 if smoke else 32
    else:
        raise ValueError(family)

    return {
        "run": {"run_name": run_name, "total_timesteps": int(total_timesteps), "progress_bar": False},
        "env": _env_config(),
        "device_policy": {"torch_device": "cpu", "allow_cuda_fallback": False},
        "reporting": {"enabled": False},
        "ppo": {
            "seed": int(seed),
            "learning_rate": 2.5e-4,
            "num_envs": budget.num_envs,
            "num_steps": budget.num_steps,
            "anneal_lr": False,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "num_minibatches": budget.num_minibatches,
            "update_epochs": budget.update_epochs,
            "norm_adv": True,
            "clip_coef": 0.2,
            "clip_vloss": True,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "agent_type": agent_type,
            "agent_config": {
                "extractor": _extractor_config(extractor_dim),
                "head_type": head_type,
                "head": head,
            },
        },
    }


def _save_model(model: PPO, path: Path, *, metadata: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = path.with_suffix("")
    model.save(str(base), metadata=metadata)
    checkpoint = base.with_suffix(".zip")
    if not checkpoint.is_file():
        raise RuntimeError(f"Checkpoint was not created: {checkpoint}")
    return checkpoint


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _teacher_config_path(output_dir: Path, cfg: dict[str, Any]) -> Path:
    path = output_dir / "teacher_config.yaml"
    _write_yaml(path, cfg)
    return path


def _train_fresh(
    *, family: str, seed: int, steps: int, output_dir: Path, budget: Budget, smoke: bool
) -> tuple[PPO, Path, Path]:
    cfg = _ppo_runtime_config(
        seed=seed,
        family=family,
        total_timesteps=steps,
        budget=budget,
        run_name=f"release_{family}_seed{seed}",
        smoke=smoke,
    )
    config_path = _teacher_config_path(output_dir, cfg)
    model, total_timesteps, _ = build_ppo_from_config(cfg)
    model.train(total_timesteps=total_timesteps, progress_bar=False)
    checkpoint = _save_model(
        model,
        output_dir / "checkpoint.zip",
        metadata={"release_runner": True, "family": family, "training_seed": seed},
    )
    return model, checkpoint, config_path


def _seed_contract(seed: int) -> dict[str, int]:
    return {
        "train_environment_seed": 11_000 + seed,
        "train_policy_seed": 21_000 + seed,
        "validation_environment_seed": 31_000 + seed,
        "validation_policy_seed": 41_000 + seed,
        "student_init_seed": 61_000 + seed,
        "dataloader_seed": 71_000 + seed,
    }


def _load_teacher(checkpoint: Path, config_path: Path, *, run_name: str) -> PPO:
    from simplyqrl.runners.train_ppo import load_yaml, resolve_env

    cfg = load_yaml(config_path)
    _label, factory = resolve_env(cfg["env"])
    return PPO.load(
        str(checkpoint),
        env=factory,
        device="cpu",
        allow_pickle=True,
        reporting_config={"enabled": False},
        run_name=run_name,
        load_optimizer_state=False,
    )


def _build_kd(
    *,
    teacher_checkpoint: Path,
    teacher_config_path: Path,
    family: str,
    seed: int,
    output_dir: Path,
    budget: Budget,
    smoke: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    seeds = _seed_contract(seed)
    data_dir = output_dir / "dataset"
    train_path, train_summary = export_teacher_dataset(
        teacher_checkpoint=teacher_checkpoint,
        phase_dir=data_dir,
        kd_samples=budget.kd_train_samples,
        deterministic=False,
        device="cpu",
        seed=seeds["train_environment_seed"],
        policy_seed=seeds["train_policy_seed"],
        teacher_config_path=teacher_config_path,
        output_name="train.npz",
    )
    val_path, val_summary = export_teacher_dataset(
        teacher_checkpoint=teacher_checkpoint,
        phase_dir=data_dir,
        kd_samples=budget.kd_validation_samples,
        deterministic=False,
        device="cpu",
        seed=seeds["validation_environment_seed"],
        policy_seed=seeds["validation_policy_seed"],
        teacher_config_path=teacher_config_path,
        output_name="validation.npz",
    )
    actor_cfg = _hybrid_head(role="actor", smoke=smoke) if family == "hybrid" else _compact_classical_head(role="actor", input_dim=8 if smoke else 32)
    critic_cfg = _hybrid_head(role="critic", smoke=smoke) if family == "hybrid" else _compact_classical_head(role="critic", input_dim=8 if smoke else 32)
    actor, critic, kd_summary = train_policy_value_kd_heads(
        dataset_path=train_path,
        validation_dataset_path=val_path,
        phase_dir=output_dir / "kd",
        student_actor_head_config=actor_cfg,
        student_critic_head_config=critic_cfg,
        kd_epochs=budget.kd_epochs,
        kd_batch_size=budget.kd_batch_size,
        kd_learning_rate=1e-3,
        kd_temperature=1.0,
        kd_policy_weight=1.0,
        kd_value_weight=0.5,
        device="cpu",
        student_init_seed=seeds["student_init_seed"],
        dataloader_seed=seeds["dataloader_seed"],
    )
    summary = {
        "seed_contract": seeds,
        "train_dataset": str(train_path),
        "train_dataset_sha256": _sha256(train_path),
        "validation_dataset": str(val_path),
        "validation_dataset_sha256": _sha256(val_path),
        "train_dataset_summary": train_summary,
        "validation_dataset_summary": val_summary,
        "kd_summary": kd_summary,
    }
    _write_json(output_dir / "kd_release_summary.json", summary)
    return actor, critic, summary


def _compose(
    *,
    teacher_checkpoint: Path,
    teacher_config_path: Path,
    actor_head: Path,
    critic_head: Path,
    freeze_feature_extractor: bool,
    rebuild_optimizer: bool,
    run_name: str,
) -> PPO:
    model = _load_teacher(teacher_checkpoint, teacher_config_path, run_name=run_name)
    compose_policy_from_teacher(
        model,
        actor_head_artifact=actor_head,
        critic_head_artifact=critic_head,
        freeze_feature_extractor=freeze_feature_extractor,
        use_best=True,
        qnode_scope="actor",
        critic_qnode_scope="critic",
    )
    apply_policy = getattr(model, "_apply_qnode_wrapper_policy", None)
    if callable(apply_policy):
        apply_policy()
    if rebuild_optimizer:
        rebuild_ppo_optimizer_after_composition(model)
    return model


def _evaluate(
    model: PPO,
    *,
    output_dir: Path,
    episodes: int,
    max_steps: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    returns: list[float] = []
    lengths: list[int] = []
    before = _qstats(model)
    was_training = model.agent.training
    model.agent.eval()
    start = time.perf_counter()
    try:
        for episode_idx in range(int(episodes)):
            env_seed = EVAL_ENV_SEED_START + episode_idx
            policy_seed = EVAL_POLICY_SEED_START + episode_idx
            torch.manual_seed(policy_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(policy_seed)
            env = make_minatar_breakout(frame_stack=1)
            try:
                obs, _ = env.reset(seed=env_seed)
                ret = 0.0
                length = 0
                done = False
                terminated = truncated = False
                while not done and length < int(max_steps):
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
                    with torch.no_grad():
                        logits = model.agent(obs_t)
                        action = int(torch.distributions.Categorical(logits=logits).sample().item())
                    obs, reward, terminated, truncated, _ = env.step(action)
                    ret += float(reward)
                    length += 1
                    done = bool(terminated or truncated)
            finally:
                env.close()
            returns.append(ret)
            lengths.append(length)
            rows.append({
                "episode_index": episode_idx,
                "environment_seed": env_seed,
                "policy_seed": policy_seed,
                "episode_return": ret,
                "episode_length": length,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })
    finally:
        if was_training:
            model.agent.train()
    after = _qstats(model)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "episodes": int(episodes),
        "environment_seed_start": EVAL_ENV_SEED_START,
        "policy_seed_start": EVAL_POLICY_SEED_START,
        "max_eval_steps": int(max_steps),
        "return_mean": _mean(returns),
        "return_std": _std(returns),
        "return_median": float(statistics.median(returns)),
        "return_min": float(min(returns)),
        "return_max": float(max(returns)),
        "episode_length_mean": _mean([float(x) for x in lengths]),
        "evaluation_seconds": float(time.perf_counter() - start),
        "evaluation_qnode_calls": int(after.get("qnode_calls_total", 0)) - int(before.get("qnode_calls_total", 0)),
        "excluded_from_training_resource_totals": True,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def _effective_handoff_updates(spec: CellSpec, budget: Budget, smoke: bool) -> int:
    if spec.handoff_t is None:
        return budget.total_updates
    if not smoke:
        return HANDOFF_UPDATES[spec.handoff_t]
    # Preserve the qualitative staged structure in a two-update smoke.
    if spec.handoff_t == 0.0:
        return 0
    if spec.handoff_t == 1.0:
        return budget.total_updates
    return 1


def run_cell(
    *,
    condition: str,
    seed: int,
    output_dir: str | Path,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run one MinAtar protocol cell and return its release summary."""
    spec = parse_condition(condition, seed)
    budget = Budget.smoke() if smoke else Budget()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    total_steps = budget.total_updates * budget.env_steps_per_update
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "status": "RUNNING",
        "environment": "MinAtar Breakout",
        "condition": spec.condition,
        "family": spec.family,
        "handoff_t": spec.handoff_t,
        "training_seed": spec.seed,
        "smoke": bool(smoke),
        "paper_facing": not smoke,
        "protocol_reproduction_not_bitwise_replay": True,
        "total_updates": budget.total_updates,
        "total_env_steps": total_steps,
    }

    model: PPO | None = None
    try:
        if spec.family == "full_classical":
            model, checkpoint, _cfg = _train_fresh(
                family="full_classical", seed=spec.seed, steps=total_steps,
                output_dir=output / "train", budget=budget, smoke=smoke,
            )
            summary["training_qnode_stats"] = _qstats(model)
            summary["final_checkpoint"] = str(checkpoint)

        elif spec.handoff_t == 0.0:
            model, checkpoint, _cfg = _train_fresh(
                family=spec.family, seed=spec.seed, steps=total_steps,
                output_dir=output / "train", budget=budget, smoke=smoke,
            )
            summary["training_qnode_stats"] = _qstats(model)
            summary["final_checkpoint"] = str(checkpoint)

        else:
            handoff_updates = _effective_handoff_updates(spec, budget, smoke)
            prefix_steps = handoff_updates * budget.env_steps_per_update
            remaining_steps = total_steps - prefix_steps
            teacher, teacher_checkpoint, teacher_config = _train_fresh(
                family="full_classical", seed=spec.seed, steps=prefix_steps,
                output_dir=output / "teacher", budget=budget, smoke=smoke,
            )
            teacher.close()
            actor, critic, kd_summary = _build_kd(
                teacher_checkpoint=teacher_checkpoint,
                teacher_config_path=teacher_config,
                family=spec.family,
                seed=spec.seed,
                output_dir=output / "bridge",
                budget=budget,
                smoke=smoke,
            )
            freeze = math.isclose(float(spec.handoff_t), 1.0)
            model = _compose(
                teacher_checkpoint=teacher_checkpoint,
                teacher_config_path=teacher_config,
                actor_head=actor,
                critic_head=critic,
                freeze_feature_extractor=freeze,
                rebuild_optimizer=remaining_steps > 0,
                run_name=f"release_{spec.condition}_seed{spec.seed}",
            )
            if remaining_steps > 0:
                model.train(total_timesteps=remaining_steps, progress_bar=False)
            if freeze and not is_feature_extractor_frozen(model):
                raise AssertionError("H(1) must keep the teacher feature extractor frozen.")
            checkpoint = _save_model(
                model,
                output / "final" / "checkpoint.zip",
                metadata={
                    "release_runner": True,
                    "condition": spec.condition,
                    "training_seed": spec.seed,
                    "handoff_t": spec.handoff_t,
                },
            )
            summary["teacher_checkpoint"] = str(teacher_checkpoint)
            summary["teacher_checkpoint_sha256"] = _sha256(teacher_checkpoint)
            summary["kd"] = kd_summary
            summary["remaining_online_env_steps"] = int(remaining_steps)
            summary["training_qnode_stats"] = _qstats(model)
            summary["final_checkpoint"] = str(checkpoint)

        if model is None:
            raise AssertionError("Cell runner produced no model.")
        summary["evaluation"] = _evaluate(
            model,
            output_dir=output / "evaluation",
            episodes=budget.eval_episodes,
            max_steps=budget.max_eval_steps,
        )
        summary["status"] = "PASS"
        summary["wall_clock_seconds"] = float(time.perf_counter() - started)
        _write_json(output / "cell_summary.json", summary)
        return summary
    finally:
        if model is not None:
            close = getattr(model, "close", None)
            if callable(close):
                close()
