"""Protocol-matched classical controls that reuse the exact H(t) teacher and KD data."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any


def _load_summary(root: Path) -> dict[str, Any]:
    for name in ("experiment_summary.json", "cell_summary.json"):
        path = root / name
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
    raise FileNotFoundError(f"No experiment summary found under {root}")


def _copy_exact(source: Path, target: Path, sha256) -> Path:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if sha256(source) != sha256(target):
        raise RuntimeError(f"Byte-identical paired artifact copy failed: {source} -> {target}")
    return target


def _validate_source(source: Path, *, expected_condition: str, seed: int, smoke: bool) -> dict[str, Any]:
    summary = _load_summary(source.resolve())
    if summary.get("status") != "PASS":
        raise RuntimeError(f"Paired hybrid source is not PASS: {source}")
    if summary.get("condition") != expected_condition:
        raise RuntimeError(
            f"Expected paired source {expected_condition}, got {summary.get('condition')!r}."
        )
    if int(summary.get("training_seed", -1)) != int(seed):
        raise RuntimeError("Paired source training seed does not match the requested control.")
    if bool(summary.get("smoke")) != bool(smoke):
        raise RuntimeError("Paired source smoke/full-protocol mode does not match the requested control.")
    return summary


def _paired_paths(summary: dict[str, Any]) -> tuple[Path, Path, Path]:
    kd = summary.get("kd") or {}
    teacher = Path(str(summary.get("teacher_checkpoint", "")))
    train = Path(str(kd.get("train_dataset", "")))
    validation = Path(str(kd.get("validation_dataset", "")))
    for path in (teacher, train, validation):
        if not path.is_file():
            raise FileNotFoundError(f"Missing paired source artifact: {path}")
    return teacher, train, validation


def _minatar_teacher_config(source: Path, summary: dict[str, Any]) -> Path:
    standalone = source / "teacher/teacher_config.yaml"
    if standalone.is_file():
        return standalone
    shared_root = summary.get("shared_teacher_trajectory")
    if shared_root:
        shared = Path(str(shared_root)) / "train/teacher_config.yaml"
        if shared.is_file():
            return shared
    explicit = summary.get("teacher_config")
    if explicit and Path(str(explicit)).is_file():
        return Path(str(explicit))
    raise FileNotFoundError(f"Could not resolve teacher config for paired source {source}")


def run_paired_control(
    *,
    environment: str,
    handoff_t: float,
    seed: int,
    hybrid_source_dir: str | Path,
    output_dir: str | Path,
    smoke: bool = False,
) -> dict[str, Any]:
    if handoff_t not in {0.25, 0.5, 0.75}:
        raise ValueError("Paired controls are defined only for t in {0.25, 0.5, 0.75}.")
    env = environment.strip().lower()
    if env == "minatar":
        return _run_minatar(handoff_t, seed, Path(hybrid_source_dir), Path(output_dir), smoke)
    if env == "lunarlander":
        return _run_lunar(handoff_t, seed, Path(hybrid_source_dir), Path(output_dir), smoke)
    raise ValueError(environment)


def _run_minatar(t: float, seed: int, source: Path, output: Path, smoke: bool) -> dict[str, Any]:
    from hybrid_training_qrl.kd_bridge import train_policy_value_kd_heads
    from handoff_qrl import minatar_cell as m

    condition = f"C({t:g})"
    source = source.resolve()
    source_summary = _validate_source(source, expected_condition=f"H({t:g})", seed=seed, smoke=smoke)
    teacher_checkpoint, source_train, source_validation = _paired_paths(source_summary)
    teacher_config = _minatar_teacher_config(source, source_summary)

    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    budget = m.Budget.smoke() if smoke else m.Budget()
    seeds = m._seed_contract(seed)
    train = _copy_exact(source_train, output / "bridge/dataset/train.npz", m._sha256)
    validation = _copy_exact(source_validation, output / "bridge/dataset/validation.npz", m._sha256)
    actor, critic, kd_summary = train_policy_value_kd_heads(
        dataset_path=train,
        validation_dataset_path=validation,
        phase_dir=output / "bridge/kd",
        student_actor_head_config=m._compact_classical_head(role="actor", input_dim=8 if smoke else 32),
        student_critic_head_config=m._compact_classical_head(role="critic", input_dim=8 if smoke else 32),
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
    remaining_steps = int(source_summary["remaining_online_env_steps"])
    started = time.perf_counter()
    model = m._compose(
        teacher_checkpoint=teacher_checkpoint,
        teacher_config_path=teacher_config,
        actor_head=actor,
        critic_head=critic,
        freeze_feature_extractor=False,
        rebuild_optimizer=True,
        run_name=f"reproduction_{condition}_seed{seed}",
    )
    try:
        model.train(total_timesteps=remaining_steps, progress_bar=False)
        checkpoint = m._save_model(
            model,
            output / "final/checkpoint.zip",
            metadata={"reproduction_runner": True, "condition": condition, "training_seed": seed},
        )
        paired_kd = {
            "seed_contract": seeds,
            "train_dataset": str(train),
            "train_dataset_sha256": m._sha256(train),
            "validation_dataset": str(validation),
            "validation_dataset_sha256": m._sha256(validation),
            "paired_hybrid_source": str(source),
            "paired_source_train_sha256": m._sha256(source_train),
            "paired_source_validation_sha256": m._sha256(source_validation),
            "kd_summary": kd_summary,
        }
        summary: dict[str, Any] = {
            "status": "PASS",
            "environment": "MinAtar Breakout",
            "condition": condition,
            "family": "compact_classical",
            "handoff_t": t,
            "training_seed": int(seed),
            "smoke": bool(smoke),
            "paper_protocol": not smoke,
            "protocol_pairing": "exact_teacher_checkpoint_and_kd_dataset_bytes",
            "paired_hybrid_source": str(source),
            "teacher_checkpoint": str(teacher_checkpoint),
            "teacher_checkpoint_sha256": m._sha256(teacher_checkpoint),
            "teacher_config": str(teacher_config),
            "kd": paired_kd,
            "remaining_online_env_steps": remaining_steps,
            "training_qnode_stats": m._qstats(model),
            "final_checkpoint": str(checkpoint),
        }
        summary["evaluation"] = m._evaluate(
            model,
            output_dir=output / "evaluation",
            episodes=budget.eval_episodes,
            max_steps=budget.max_eval_steps,
        )
        summary["wall_clock_seconds"] = float(time.perf_counter() - started)
        m._write_json(output / "experiment_summary.json", summary)
        return summary
    finally:
        close = getattr(model, "close", None)
        if callable(close):
            close()


def _run_lunar(t: float, seed: int, source: Path, output: Path, smoke: bool) -> dict[str, Any]:
    from hybrid_training_qrl.kd_bridge import train_policy_value_kd_heads
    from handoff_qrl import lunarlander_experiment as m

    condition = f"C({t:g})"
    source = source.resolve()
    source_summary = _validate_source(source, expected_condition=f"H({t:g})", seed=seed, smoke=smoke)
    teacher_checkpoint, source_train, source_validation = _paired_paths(source_summary)
    teacher_config = source / "teacher/config.yaml"
    if not teacher_config.is_file():
        raise FileNotFoundError(f"Missing paired teacher config: {teacher_config}")

    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    budget = m.Budget.smoke() if smoke else m.Budget()
    seeds = m._seed_contract(seed)
    train = _copy_exact(source_train, output / "bridge/dataset/train.npz", m._sha256)
    validation = _copy_exact(source_validation, output / "bridge/dataset/validation.npz", m._sha256)
    actor, critic, kd_summary = train_policy_value_kd_heads(
        dataset_path=train,
        validation_dataset_path=validation,
        phase_dir=output / "bridge/kd",
        student_actor_head_config=m._compact_head(role="actor"),
        student_critic_head_config=m._compact_head(role="critic"),
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
    remaining_steps = int(source_summary["remaining_online_env_steps"])
    started = time.perf_counter()
    model = m._compose(
        teacher_checkpoint=teacher_checkpoint,
        teacher_config_path=teacher_config,
        actor_head=actor,
        critic_head=critic,
        seed=seed,
        budget=budget,
        freeze_feature_extractor=False,
        rebuild_optimizer=True,
        run_name=f"reproduction_lunarlander_{condition}_seed{seed}",
    )
    try:
        model.train(total_timesteps=remaining_steps, progress_bar=False)
        checkpoint = m._save_model(
            model,
            output / "final/checkpoint.zip",
            metadata={"reproduction_runner": True, "condition": condition, "training_seed": seed},
        )
        paired_kd = {
            "seed_contract": seeds,
            "train_dataset": str(train),
            "train_dataset_sha256": m._sha256(train),
            "validation_dataset": str(validation),
            "validation_dataset_sha256": m._sha256(validation),
            "paired_hybrid_source": str(source),
            "paired_source_train_sha256": m._sha256(source_train),
            "paired_source_validation_sha256": m._sha256(source_validation),
            "kd_summary": kd_summary,
        }
        summary: dict[str, Any] = {
            "status": "PASS",
            "environment": "LunarLander-v3",
            "condition": condition,
            "family": "compact_classical",
            "handoff_t": t,
            "training_seed": int(seed),
            "smoke": bool(smoke),
            "paper_protocol": not smoke,
            "protocol_pairing": "exact_teacher_checkpoint_and_kd_dataset_bytes",
            "paired_hybrid_source": str(source),
            "teacher_checkpoint": str(teacher_checkpoint),
            "teacher_checkpoint_sha256": m._sha256(teacher_checkpoint),
            "teacher_config": str(teacher_config),
            "kd": paired_kd,
            "remaining_online_env_steps": remaining_steps,
            "training_qnode_stats": m._qstats(model),
            "final_checkpoint": str(checkpoint),
        }
        summary["evaluation"] = m._evaluate(
            model,
            output_dir=output / "evaluation",
            episodes=budget.eval_episodes,
            max_steps=budget.max_eval_steps,
        )
        summary["wall_clock_seconds"] = float(time.perf_counter() - started)
        m._write_json(output / "experiment_summary.json", summary)
        return summary
    finally:
        close = getattr(model, "close", None)
        if callable(close):
            close()
