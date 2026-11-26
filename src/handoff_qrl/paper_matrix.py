"""Protocol-faithful full-matrix reproduction helpers.

The matrix launcher preserves cross-condition dependencies that a standalone
condition cannot express: MinAtar handoffs share one full-classical teacher
trajectory per seed, H/C attribution pairs reuse the exact teacher checkpoint
and KD dataset bytes, and LunarLander H(1) reuses the same-seed full-classical
teacher as the original final campaign.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from handoff_qrl.experiments import condition_slug
from handoff_qrl.paired_control import run_paired_control

MATRIX_ORDER = (
    "full_classical",
    "H(0)",
    "C(0)",
    "H(0.25)",
    "C(0.25)",
    "H(0.5)",
    "C(0.5)",
    "H(0.75)",
    "C(0.75)",
    "H(1)",
)


def _out(root: Path, environment: str, condition: str, seed: int) -> Path:
    return root / environment / condition_slug(condition) / f"seed{seed}"


def _load_pass(path: Path) -> dict[str, Any] | None:
    summary = path / "experiment_summary.json"
    if not summary.is_file():
        return None
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if payload.get("status") == "PASS" else None


def run_seed_matrix(
    *,
    environment: str,
    seed: int,
    output_root: str | Path = "runs/reproduction",
    smoke: bool = False,
    resume: bool = False,
) -> dict[str, dict[str, Any]]:
    env = environment.strip().lower()
    root = Path(output_root)
    if env == "minatar":
        return _run_minatar_seed(seed, root, smoke, resume)
    if env == "lunarlander":
        return _run_lunar_seed(seed, root, smoke, resume)
    raise ValueError(environment)


class _MilestoneMonitor:
    def __init__(self, *, model, output_dir: Path, labels_by_update: dict[int, tuple[str, ...]]) -> None:
        self.model = model
        self.output_dir = output_dir
        self.labels_by_update = labels_by_update
        self.saved: dict[str, Path] = {}

    def on_training_start(self, state): return None
    def on_rollout_start(self, state): return None
    def on_episode_end(self, state): return None
    def on_rollout_end(self, state): return None
    def on_update_start(self, state): return None
    def on_backward_end(self, state): return None
    def on_evaluation_end(self, state): return None
    def on_training_end(self, state): return None
    def on_training_error(self, state): return None

    def on_update_end(self, state):
        update = int(state.get("update", 0))
        for label in self.labels_by_update.get(update, ()):
            target = self.output_dir / label / "checkpoint"
            target.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(
                str(target),
                metadata={
                    "reproduction_teacher_milestone": label,
                    "update": update,
                    "global_step": int(state.get("global_step", 0)),
                    "optimizer_state_included": True,
                },
            )
            self.saved[label] = target.with_suffix(".zip")


def _minatar_teacher_labels(smoke: bool) -> dict[int, tuple[str, ...]]:
    if smoke:
        return {1: ("t025", "t050", "t075"), 2: ("t100",)}
    return {
        1221: ("t025",),
        2442: ("t050",),
        3663: ("t075",),
        4884: ("t100",),
    }


def _minatar_full_classical(seed: int, output: Path, smoke: bool) -> dict[str, Any]:
    from handoff_qrl import minatar_cell as m
    from simplyqrl.runners.train_ppo import build_ppo_from_config

    budget = m.Budget.smoke() if smoke else m.Budget()
    total_steps = budget.total_updates * budget.env_steps_per_update
    output.mkdir(parents=True, exist_ok=True)
    cfg = m._ppo_runtime_config(
        seed=seed,
        family="full_classical",
        total_timesteps=total_steps,
        budget=budget,
        run_name=f"reproduction_minatar_full_classical_seed{seed}",
        smoke=smoke,
    )
    config_path = m._teacher_config_path(output / "train", cfg)
    model, total_timesteps, _ = build_ppo_from_config(cfg)
    monitor = _MilestoneMonitor(
        model=model,
        output_dir=output / "train/milestones",
        labels_by_update=_minatar_teacher_labels(smoke),
    )
    model.monitor_manager.add(monitor)
    started = time.perf_counter()
    try:
        model.train(total_timesteps=total_timesteps, progress_bar=False)
        final_checkpoint = m._save_model(
            model,
            output / "train/checkpoint.zip",
            metadata={"reproduction_runner": True, "family": "full_classical", "training_seed": seed},
        )
        expected_labels = {label for labels in _minatar_teacher_labels(smoke).values() for label in labels}
        if set(monitor.saved) != expected_labels:
            raise RuntimeError(
                f"Teacher milestone capture incomplete: expected {sorted(expected_labels)}, "
                f"got {sorted(monitor.saved)}"
            )
        summary: dict[str, Any] = {
            "status": "PASS",
            "environment": "MinAtar Breakout",
            "condition": "full_classical",
            "family": "full_classical",
            "handoff_t": None,
            "training_seed": int(seed),
            "smoke": bool(smoke),
            "paper_protocol": not smoke,
            "shared_teacher_trajectory": True,
            "teacher_config": str(config_path),
            "teacher_milestones": {key: str(value) for key, value in monitor.saved.items()},
            "training_qnode_stats": m._qstats(model),
            "final_checkpoint": str(final_checkpoint),
            "total_updates": budget.total_updates,
            "total_env_steps": total_steps,
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


def _minatar_handoff_from_teacher(
    *,
    t: float,
    seed: int,
    teacher_summary: dict[str, Any],
    teacher_root: Path,
    output: Path,
    smoke: bool,
) -> dict[str, Any]:
    from handoff_qrl import minatar_cell as m

    budget = m.Budget.smoke() if smoke else m.Budget()
    label = {0.25: "t025", 0.5: "t050", 0.75: "t075", 1.0: "t100"}[t]
    milestone_path = Path(str(teacher_summary["teacher_milestones"][label]))
    teacher_config = teacher_root / "train/teacher_config.yaml"
    if not milestone_path.is_file() or not teacher_config.is_file():
        raise FileNotFoundError("Shared MinAtar teacher trajectory is incomplete.")

    output.mkdir(parents=True, exist_ok=True)
    actor, critic, kd_summary = m._build_kd(
        teacher_checkpoint=milestone_path,
        teacher_config_path=teacher_config,
        family="hybrid",
        seed=seed,
        output_dir=output / "bridge",
        budget=budget,
        smoke=smoke,
    )
    handoff_updates = m._effective_handoff_updates(m.parse_condition(f"H({t:g})", seed), budget, smoke)
    remaining_steps = (budget.total_updates - handoff_updates) * budget.env_steps_per_update
    freeze = math.isclose(t, 1.0)
    model = m._compose(
        teacher_checkpoint=milestone_path,
        teacher_config_path=teacher_config,
        actor_head=actor,
        critic_head=critic,
        freeze_feature_extractor=freeze,
        rebuild_optimizer=remaining_steps > 0,
        run_name=f"reproduction_H({t:g})_seed{seed}",
    )
    started = time.perf_counter()
    try:
        if remaining_steps > 0:
            model.train(total_timesteps=remaining_steps, progress_bar=False)
        if freeze and not m.is_feature_extractor_frozen(model):
            raise AssertionError("H(1) must keep the teacher feature extractor frozen.")
        checkpoint = m._save_model(
            model,
            output / "final/checkpoint.zip",
            metadata={"reproduction_runner": True, "condition": f"H({t:g})", "training_seed": seed},
        )
        summary: dict[str, Any] = {
            "status": "PASS",
            "environment": "MinAtar Breakout",
            "condition": f"H({t:g})",
            "family": "hybrid",
            "handoff_t": t,
            "training_seed": int(seed),
            "smoke": bool(smoke),
            "paper_protocol": not smoke,
            "shared_teacher_trajectory": str(teacher_root),
            "teacher_checkpoint": str(milestone_path),
            "teacher_checkpoint_sha256": m._sha256(milestone_path),
            "kd": kd_summary,
            "remaining_online_env_steps": int(remaining_steps),
            "training_qnode_stats": m._qstats(model),
            "final_checkpoint": str(checkpoint),
            "total_updates": budget.total_updates,
            "total_env_steps": budget.total_updates * budget.env_steps_per_update,
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


def _run_minatar_seed(seed: int, root: Path, smoke: bool, resume: bool) -> dict[str, dict[str, Any]]:
    from handoff_qrl import minatar_cell as m
    from handoff_qrl.experiments import run_experiment

    results: dict[str, dict[str, Any]] = {}
    teacher_root = _out(root, "minatar", "full_classical", seed)
    teacher_summary = _load_pass(teacher_root) if resume else None
    if teacher_summary is None or "teacher_milestones" not in teacher_summary:
        if teacher_root.exists() and any(teacher_root.iterdir()):
            raise FileExistsError(
                f"Full-classical output exists without reusable milestones: {teacher_root}. "
                "Remove it or choose a new output root."
            )
        teacher_summary = _minatar_full_classical(seed, teacher_root, smoke)
    results["full_classical"] = teacher_summary

    for condition in ("H(0)", "C(0)"):
        output = _out(root, "minatar", condition, seed)
        existing = _load_pass(output) if resume else None
        results[condition] = existing or run_experiment(
            environment="minatar", condition=condition, seed=seed, output_dir=output, smoke=smoke
        )

    for t in (0.25, 0.5, 0.75):
        h_condition = f"H({t:g})"
        h_output = _out(root, "minatar", h_condition, seed)
        h_summary = _load_pass(h_output) if resume else None
        if h_summary is None:
            h_summary = _minatar_handoff_from_teacher(
                t=t, seed=seed, teacher_summary=teacher_summary,
                teacher_root=teacher_root, output=h_output, smoke=smoke,
            )
        results[h_condition] = h_summary

        c_condition = f"C({t:g})"
        c_output = _out(root, "minatar", c_condition, seed)
        c_summary = _load_pass(c_output) if resume else None
        results[c_condition] = c_summary or run_paired_control(
            environment="minatar", handoff_t=t, seed=seed,
            hybrid_source_dir=h_output, output_dir=c_output, smoke=smoke,
        )

    h1_output = _out(root, "minatar", "H(1)", seed)
    h1_summary = _load_pass(h1_output) if resume else None
    results["H(1)"] = h1_summary or _minatar_handoff_from_teacher(
        t=1.0, seed=seed, teacher_summary=teacher_summary,
        teacher_root=teacher_root, output=h1_output, smoke=smoke,
    )
    return results


def _lunar_h1_from_teacher(
    *, seed: int, teacher_root: Path, teacher_summary: dict[str, Any], output: Path, smoke: bool
) -> dict[str, Any]:
    from handoff_qrl import lunarlander_experiment as m

    budget = m.Budget.smoke() if smoke else m.Budget()
    teacher_checkpoint = Path(str(teacher_summary["final_checkpoint"]))
    teacher_config = teacher_root / "train/config.yaml"
    if not teacher_checkpoint.is_file() or not teacher_config.is_file():
        raise FileNotFoundError("LunarLander full-classical teacher source is incomplete.")
    output.mkdir(parents=True, exist_ok=True)
    actor, critic, kd_summary = m._build_kd(
        teacher_checkpoint=teacher_checkpoint,
        teacher_config_path=teacher_config,
        family="hybrid",
        seed=seed,
        output_dir=output / "bridge",
        budget=budget,
        smoke=smoke,
    )
    model = m._compose(
        teacher_checkpoint=teacher_checkpoint,
        teacher_config_path=teacher_config,
        actor_head=actor,
        critic_head=critic,
        seed=seed,
        budget=budget,
        freeze_feature_extractor=True,
        rebuild_optimizer=False,
        run_name=f"reproduction_lunarlander_H(1)_seed{seed}",
    )
    started = time.perf_counter()
    try:
        if not m.is_feature_extractor_frozen(model):
            raise AssertionError("H(1) must keep the teacher feature extractor frozen.")
        checkpoint = m._save_model(
            model, output / "final/checkpoint.zip",
            metadata={"reproduction_runner": True, "condition": "H(1)", "training_seed": seed},
        )
        summary: dict[str, Any] = {
            "status": "PASS",
            "environment": "LunarLander-v3",
            "condition": "H(1)",
            "family": "hybrid",
            "handoff_t": 1.0,
            "training_seed": int(seed),
            "smoke": bool(smoke),
            "paper_protocol": not smoke,
            "shared_full_classical_teacher": str(teacher_root),
            "teacher_checkpoint": str(teacher_checkpoint),
            "teacher_checkpoint_sha256": m._sha256(teacher_checkpoint),
            "kd": kd_summary,
            "remaining_online_env_steps": 0,
            "training_qnode_stats": m._qstats(model),
            "final_checkpoint": str(checkpoint),
            "total_rollouts": budget.total_rollouts,
            "total_env_steps": budget.total_env_steps,
        }
        summary["evaluation"] = m._evaluate(
            model, output_dir=output / "evaluation",
            episodes=budget.eval_episodes, max_steps=budget.max_eval_steps,
        )
        summary["wall_clock_seconds"] = float(time.perf_counter() - started)
        m._write_json(output / "experiment_summary.json", summary)
        return summary
    finally:
        close = getattr(model, "close", None)
        if callable(close):
            close()


def _run_lunar_seed(seed: int, root: Path, smoke: bool, resume: bool) -> dict[str, dict[str, Any]]:
    from handoff_qrl.experiments import run_experiment

    results: dict[str, dict[str, Any]] = {}
    classical_output = _out(root, "lunarlander", "full_classical", seed)
    classical = _load_pass(classical_output) if resume else None
    classical = classical or run_experiment(
        environment="lunarlander", condition="full_classical", seed=seed,
        output_dir=classical_output, smoke=smoke,
    )
    results["full_classical"] = classical

    for condition in ("H(0)", "C(0)"):
        output = _out(root, "lunarlander", condition, seed)
        existing = _load_pass(output) if resume else None
        results[condition] = existing or run_experiment(
            environment="lunarlander", condition=condition, seed=seed, output_dir=output, smoke=smoke
        )

    for t in (0.25, 0.5, 0.75):
        h_condition = f"H({t:g})"
        h_output = _out(root, "lunarlander", h_condition, seed)
        h = _load_pass(h_output) if resume else None
        h = h or run_experiment(
            environment="lunarlander", condition=h_condition, seed=seed, output_dir=h_output, smoke=smoke
        )
        results[h_condition] = h
        c_condition = f"C({t:g})"
        c_output = _out(root, "lunarlander", c_condition, seed)
        c = _load_pass(c_output) if resume else None
        results[c_condition] = c or run_paired_control(
            environment="lunarlander", handoff_t=t, seed=seed,
            hybrid_source_dir=h_output, output_dir=c_output, smoke=smoke,
        )

    h1_output = _out(root, "lunarlander", "H(1)", seed)
    h1 = _load_pass(h1_output) if resume else None
    results["H(1)"] = h1 or _lunar_h1_from_teacher(
        seed=seed, teacher_root=classical_output, teacher_summary=classical,
        output=h1_output, smoke=smoke,
    )
    return results
