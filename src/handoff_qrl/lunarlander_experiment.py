"""One-condition LunarLander reproduction from the frozen paper protocol."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml

from hybrid_training_qrl.kd_bridge import export_teacher_dataset, train_policy_value_kd_heads
from simplyqrl.ppo import PPO
from simplyqrl.runners.train_ppo import build_ppo_from_config, load_yaml, resolve_env
from simplyqrl.staged import (
    compose_policy_from_teacher,
    is_feature_extractor_frozen,
    rebuild_ppo_optimizer_after_composition,
)

TOTAL_ROLLOUTS = 64
NUM_ENVS = 16
NUM_STEPS = 1024
ROLLOUT_STEPS = NUM_ENVS * NUM_STEPS
TOTAL_ENV_STEPS = TOTAL_ROLLOUTS * ROLLOUT_STEPS
HANDOFF_ROLLOUTS = {0.0: 0, 0.25: 16, 0.5: 32, 0.75: 48, 1.0: 64}
EVAL_EPISODES = 100
EVAL_ENV_SEED_START = 15_000
EVAL_POLICY_SEED_START = 25_000
EVAL_MAX_STEPS = 1_000


@dataclass(frozen=True)
class ExperimentSpec:
    condition: str
    family: str
    handoff_t: float | None
    seed: int


@dataclass(frozen=True)
class Budget:
    total_rollouts: int = TOTAL_ROLLOUTS
    num_envs: int = NUM_ENVS
    num_steps: int = NUM_STEPS
    update_epochs: int = 4
    num_minibatches: int = 256
    kd_train_samples: int = 65_536
    kd_validation_samples: int = 8_192
    kd_epochs: int = 20
    kd_batch_size: int = 128
    eval_episodes: int = EVAL_EPISODES
    max_eval_steps: int = EVAL_MAX_STEPS

    @property
    def rollout_steps(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def total_env_steps(self) -> int:
        return self.total_rollouts * self.rollout_steps

    @classmethod
    def smoke(cls) -> "Budget":
        return cls(
            total_rollouts=2,
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


def parse_condition(raw: str, seed: int) -> ExperimentSpec:
    condition = raw.strip()
    if condition in {"full_classical", "classical"}:
        return ExperimentSpec("full_classical", "full_classical", None, int(seed))
    if len(condition) < 4 or condition[0] not in {"H", "C"} or not condition.endswith(")"):
        raise ValueError(f"Unsupported condition: {raw!r}")
    try:
        handoff_t = float(condition[2:-1])
    except ValueError as exc:
        raise ValueError(f"Invalid handoff condition: {raw!r}") from exc
    if handoff_t not in HANDOFF_ROLLOUTS:
        raise ValueError(f"Handoff must be one of {sorted(HANDOFF_ROLLOUTS)}, got {handoff_t}.")
    family = "hybrid" if condition.startswith("H") else "compact_classical"
    canonical = f"{condition[0]}({handoff_t:g})"
    if canonical == "C(1)":
        raise ValueError("C(1) is not part of the paper's attribution family.")
    return ExperimentSpec(canonical, family, handoff_t, int(seed))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _std(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _qstats(model: PPO) -> dict[str, Any]:
    getter = getattr(model.agent, "get_qnode_stats", None)
    if not callable(getter):
        return {"qnode_calls_total": 0, "qnode_time_total": 0.0}
    return dict(getter())


def _seed_contract(seed: int) -> dict[str, int]:
    return {
        "train_environment_seed": 11_000 + seed,
        "train_policy_seed": 21_000 + seed,
        "validation_environment_seed": 31_000 + seed,
        "validation_policy_seed": 41_000 + seed,
        "permutation_seed": 51_000 + seed,
        "student_init_seed": 61_000 + seed,
        "dataloader_seed": 71_000 + seed,
    }


def _identity_extractor() -> dict[str, Any]:
    return {"type": "identity"}


def _full_classical_agent() -> dict[str, Any]:
    return {
        "extractor": _identity_extractor(),
        "head_type": "classical",
        "net_arch": {"pi": [128, 128], "vf": [128, 128]},
        "activation": "tanh",
    }


def _compact_classical_agent() -> dict[str, Any]:
    return {
        "extractor": _identity_extractor(),
        "head_type": "classical",
        "net_arch": {"pi": [8], "vf": [8]},
        "activation": "tanh",
    }


def _hybrid_head(*, role: str | None = None, smoke: bool = False) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "type": "hybrid",
        "family": "hybrid",
        "quantum_core": {
            "factory": "hybrid_training_qrl.quantum.circuits:build_angle_rx_ryrz_cnot_reuploading",
            "q_output_dim": 8,
            "kwargs": {
                "n_qubits": 8,
                "n_layers_q": 1 if smoke else 5,
                "entangler": "cnot_circular",
                "qml_device": "default.qubit",
                "diff_method": "backprop",
            },
        },
        "postprocessing": {"reuse_repetitions": 1, "use_output_scaling": False},
        "hidden_sizes": [8],
        "activation": "tanh",
    }
    if role is not None:
        cfg.update({"role": role, "input_dim": 8, "output_dim": 4 if role == "actor" else 1})
    return cfg


def _compact_head(*, role: str) -> dict[str, Any]:
    return {
        "type": "classical",
        "family": "classical",
        "role": role,
        "input_dim": 8,
        "output_dim": 4 if role == "actor" else 1,
        "hidden_sizes": [8],
        "activation": "tanh",
    }


def _common_ppo(seed: int, budget: Budget) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "learning_rate": 0.001,
        "num_envs": budget.num_envs,
        "num_steps": budget.num_steps,
        "anneal_lr": False,
        "gamma": 0.999,
        "gae_lambda": 0.98,
        "num_minibatches": budget.num_minibatches,
        "update_epochs": budget.update_epochs,
        "norm_adv": True,
        "clip_coef": 0.2,
        "clip_vloss": False,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }


def _runtime_config(
    *, seed: int, family: str, total_timesteps: int, budget: Budget, run_name: str, smoke: bool
) -> dict[str, Any]:
    ppo = _common_ppo(seed, budget)
    if family == "full_classical":
        ppo.update(
            {
                "learning_rate": 0.0003,
                "agent_type": "v15_classical",
                "agent_config": _full_classical_agent(),
            }
        )
    else:
        ppo.update(
            {
                "gradient_clipping_mode": "separate_actor_critic",
                "actor_max_grad_norm": 0.5,
                "critic_max_grad_norm": 0.5,
                "shared_max_grad_norm": 0.5,
                "learning_rate_mode": "separate_actor_critic",
                "actor_learning_rate": 0.002,
                "critic_learning_rate": 0.001,
                "shared_learning_rate": 0.001,
            }
        )
        if family == "compact_classical":
            ppo.update(
                {"agent_type": "v15_classical", "agent_config": _compact_classical_agent()}
            )
        elif family == "hybrid":
            ppo.update(
                {
                    "agent_type": "v15_hybrid",
                    "agent_config": {
                        "extractor": _identity_extractor(),
                        "head_type": "hybrid",
                        "head": _hybrid_head(smoke=smoke),
                    },
                }
            )
        else:
            raise ValueError(family)
    return {
        "run": {
            "run_name": run_name,
            "total_timesteps": int(total_timesteps),
            "progress_bar": False,
        },
        "env": {"id": "LunarLander-v3"},
        "device_policy": {"torch_device": "cpu", "allow_cuda_fallback": False},
        "reporting": {"enabled": False},
        "ppo": ppo,
    }


def _save_model(model: PPO, path: Path, *, metadata: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = path.with_suffix("")
    model.save(str(base), metadata=metadata)
    checkpoint = base.with_suffix(".zip")
    if not checkpoint.is_file():
        raise RuntimeError(f"Checkpoint was not created: {checkpoint}")
    return checkpoint


def _train_fresh(
    *, family: str, seed: int, steps: int, output_dir: Path, budget: Budget, smoke: bool
) -> tuple[PPO, Path, Path]:
    cfg = _runtime_config(
        seed=seed,
        family=family,
        total_timesteps=steps,
        budget=budget,
        run_name=f"reproduction_lunarlander_{family}_seed{seed}",
        smoke=smoke,
    )
    config_path = output_dir / "config.yaml"
    _write_yaml(config_path, cfg)
    model, total_timesteps, _ = build_ppo_from_config(cfg)
    model.train(total_timesteps=total_timesteps, progress_bar=False)
    checkpoint = _save_model(
        model,
        output_dir / "checkpoint.zip",
        metadata={"reproduction_runner": True, "family": family, "training_seed": seed},
    )
    return model, checkpoint, config_path


def _permute_full_pool(source: Path, target: Path, *, seed: int) -> Path:
    with np.load(source) as data:
        count = int(np.asarray(data["features"]).shape[0])
        permutation = np.random.default_rng(int(seed)).permutation(count)
        payload = {key: np.asarray(data[key])[permutation] for key in data.files}
    payload["source_indices"] = np.asarray(permutation, dtype=np.int64)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **payload)
    return target


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
    train_pool, train_pool_summary = export_teacher_dataset(
        teacher_checkpoint=teacher_checkpoint,
        phase_dir=output_dir,
        kd_samples=budget.kd_train_samples,
        deterministic=False,
        device="cpu",
        seed=seeds["train_environment_seed"],
        policy_seed=seeds["train_policy_seed"],
        teacher_config_path=teacher_config_path,
        output_name="train_pool.npz",
    )
    validation, validation_summary = export_teacher_dataset(
        teacher_checkpoint=teacher_checkpoint,
        phase_dir=output_dir,
        kd_samples=budget.kd_validation_samples,
        deterministic=False,
        device="cpu",
        seed=seeds["validation_environment_seed"],
        policy_seed=seeds["validation_policy_seed"],
        teacher_config_path=teacher_config_path,
        output_name="validation.npz",
    )
    # The frozen LunarLander protocol uses the full stochastic pool through the frozen 51000+seed
    # permutation before KD. Even at 64k/64k this changes sample order and is
    # therefore part of the reproducibility contract.
    train = _permute_full_pool(
        train_pool,
        output_dir / "dataset/train.npz",
        seed=seeds["permutation_seed"],
    )
    actor_cfg = _hybrid_head(role="actor", smoke=smoke) if family == "hybrid" else _compact_head(role="actor")
    critic_cfg = _hybrid_head(role="critic", smoke=smoke) if family == "hybrid" else _compact_head(role="critic")
    actor, critic, kd_summary = train_policy_value_kd_heads(
        dataset_path=train,
        validation_dataset_path=validation,
        phase_dir=output_dir / "kd",
        student_actor_head_config=actor_cfg,
        student_critic_head_config=critic_cfg,
        kd_epochs=budget.kd_epochs,
        kd_batch_size=budget.kd_batch_size,
        kd_learning_rate=0.001,
        kd_temperature=1.0,
        kd_policy_weight=1.0,
        kd_value_weight=0.5,
        device="cpu",
        student_init_seed=seeds["student_init_seed"],
        dataloader_seed=seeds["dataloader_seed"],
    )
    summary = {
        "seed_contract": seeds,
        "collection_policy": "stochastic_teacher",
        "train_pool": str(train_pool),
        "train_pool_sha256": _sha256(train_pool),
        "train_dataset": str(train),
        "train_dataset_sha256": _sha256(train),
        "validation_dataset": str(validation),
        "validation_dataset_sha256": _sha256(validation),
        "train_pool_summary": train_pool_summary,
        "validation_dataset_summary": validation_summary,
        "permutation_seed": seeds["permutation_seed"],
        "kd_summary": kd_summary,
    }
    _write_json(output_dir / "kd_summary.json", summary)
    return actor, critic, summary


def _phase_hparams(seed: int, budget: Budget) -> dict[str, Any]:
    ppo = _common_ppo(seed, budget)
    ppo.update(
        {
            "gradient_clipping_mode": "separate_actor_critic",
            "actor_max_grad_norm": 0.5,
            "critic_max_grad_norm": 0.5,
            "shared_max_grad_norm": 0.5,
            "learning_rate_mode": "separate_actor_critic",
            "actor_learning_rate": 0.002,
            "critic_learning_rate": 0.001,
            "shared_learning_rate": 0.001,
        }
    )
    return ppo


def _compose(
    *,
    teacher_checkpoint: Path,
    teacher_config_path: Path,
    actor_head: Path,
    critic_head: Path,
    seed: int,
    budget: Budget,
    freeze_feature_extractor: bool,
    rebuild_optimizer: bool,
    run_name: str,
) -> PPO:
    teacher_cfg = load_yaml(teacher_config_path)
    _label, env_factory = resolve_env(teacher_cfg["env"])
    model = PPO.load(
        str(teacher_checkpoint),
        env=env_factory,
        device="cpu",
        allow_pickle=True,
        reporting_config={"enabled": False},
        run_name=run_name,
        hparams_overrides=_phase_hparams(seed, budget),
        load_optimizer_state=False,
    )
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


def _evaluate(model: PPO, *, output_dir: Path, episodes: int, max_steps: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    returns: list[float] = []
    lengths: list[int] = []
    before = _qstats(model)
    was_training = model.agent.training
    model.agent.eval()
    started = time.perf_counter()
    try:
        for episode_idx in range(int(episodes)):
            env_seed = EVAL_ENV_SEED_START + episode_idx
            policy_seed = EVAL_POLICY_SEED_START + episode_idx
            torch.manual_seed(policy_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(policy_seed)
            env = gym.make("LunarLander-v3")
            try:
                obs, _ = env.reset(seed=env_seed)
                ret = 0.0
                length = 0
                terminated = truncated = False
                while not (terminated or truncated) and length < int(max_steps):
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
                    with torch.no_grad():
                        logits = model.agent(obs_t)
                        action = int(torch.distributions.Categorical(logits=logits).sample().item())
                    obs, reward, terminated, truncated, _ = env.step(action)
                    ret += float(reward)
                    length += 1
            finally:
                env.close()
            returns.append(ret)
            lengths.append(length)
            rows.append(
                {
                    "episode_index": episode_idx,
                    "environment_seed": env_seed,
                    "policy_seed": policy_seed,
                    "episode_return": ret,
                    "episode_length": length,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
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
        "evaluation_seconds": float(time.perf_counter() - started),
        "evaluation_qnode_calls": int(after.get("qnode_calls_total", 0))
        - int(before.get("qnode_calls_total", 0)),
        "excluded_from_training_resource_totals": True,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def _effective_handoff_rollouts(spec: ExperimentSpec, budget: Budget, smoke: bool) -> int:
    if spec.handoff_t is None:
        return budget.total_rollouts
    if not smoke:
        return HANDOFF_ROLLOUTS[spec.handoff_t]
    if spec.handoff_t == 0.0:
        return 0
    if spec.handoff_t == 1.0:
        return budget.total_rollouts
    return 1


def run_experiment(
    *, condition: str, seed: int, output_dir: str | Path, smoke: bool = False
) -> dict[str, Any]:
    """Run one LunarLander condition; full matrices add exact cross-condition pairing."""
    spec = parse_condition(condition, seed)
    budget = Budget.smoke() if smoke else Budget()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "status": "RUNNING",
        "environment": "LunarLander-v3",
        "condition": spec.condition,
        "family": spec.family,
        "handoff_t": spec.handoff_t,
        "training_seed": spec.seed,
        "smoke": bool(smoke),
        "paper_protocol": not smoke,
        "total_rollouts": budget.total_rollouts,
        "total_env_steps": budget.total_env_steps,
    }
    model: PPO | None = None
    try:
        if spec.family == "full_classical":
            model, checkpoint, _cfg = _train_fresh(
                family="full_classical",
                seed=spec.seed,
                steps=budget.total_env_steps,
                output_dir=output / "train",
                budget=budget,
                smoke=smoke,
            )
            summary["training_qnode_stats"] = _qstats(model)
            summary["final_checkpoint"] = str(checkpoint)
        elif spec.handoff_t == 0.0:
            model, checkpoint, _cfg = _train_fresh(
                family=spec.family,
                seed=spec.seed,
                steps=budget.total_env_steps,
                output_dir=output / "train",
                budget=budget,
                smoke=smoke,
            )
            summary["training_qnode_stats"] = _qstats(model)
            summary["final_checkpoint"] = str(checkpoint)
        else:
            handoff_rollouts = _effective_handoff_rollouts(spec, budget, smoke)
            prefix_steps = handoff_rollouts * budget.rollout_steps
            remaining_steps = budget.total_env_steps - prefix_steps
            teacher, teacher_checkpoint, teacher_config = _train_fresh(
                family="full_classical",
                seed=spec.seed,
                steps=prefix_steps,
                output_dir=output / "teacher",
                budget=budget,
                smoke=smoke,
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
                seed=spec.seed,
                budget=budget,
                freeze_feature_extractor=freeze,
                rebuild_optimizer=remaining_steps > 0,
                run_name=f"reproduction_lunarlander_{spec.condition}_seed{spec.seed}",
            )
            if remaining_steps > 0:
                model.train(total_timesteps=remaining_steps, progress_bar=False)
            if freeze and not is_feature_extractor_frozen(model):
                raise AssertionError("H(1) must keep the teacher feature extractor frozen.")
            checkpoint = _save_model(
                model,
                output / "final/checkpoint.zip",
                metadata={
                    "reproduction_runner": True,
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
            raise AssertionError("Experiment runner produced no model.")
        summary["evaluation"] = _evaluate(
            model,
            output_dir=output / "evaluation",
            episodes=budget.eval_episodes,
            max_steps=budget.max_eval_steps,
        )
        summary["status"] = "PASS"
        summary["wall_clock_seconds"] = float(time.perf_counter() - started)
        _write_json(output / "experiment_summary.json", summary)
        return summary
    finally:
        if model is not None:
            close = getattr(model, "close", None)
            if callable(close):
                close()
