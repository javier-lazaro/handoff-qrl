"""Public experiment dispatch for the paper reproducibility repository."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

ENVIRONMENT_ALIASES = {
    "lunarlander": "lunarlander",
    "lunarlander-v3": "lunarlander",
    "minatar": "minatar",
    "minatar-breakout": "minatar",
}

CONDITIONS = (
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


def normalise_environment(raw: str) -> str:
    key = raw.strip().lower()
    try:
        return ENVIRONMENT_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported environment {raw!r}. Choose 'lunarlander' or 'minatar'."
        ) from exc


def condition_slug(condition: str) -> str:
    if condition == "full_classical":
        return condition
    return condition.lower().replace("(", "_").replace(")", "").replace(".", "p")


def default_output_dir(environment: str, condition: str, seed: int) -> Path:
    env = normalise_environment(environment)
    return Path("runs/reproduction") / env / condition_slug(condition) / f"seed{int(seed)}"


def _paired_handoff(condition: str) -> float | None:
    condition = condition.strip()
    if not (condition.startswith("C(") and condition.endswith(")")):
        return None
    value = float(condition[2:-1])
    return value if value in {0.25, 0.5, 0.75} else None


def _run_minatar_fresh_zero(*, condition: str, seed: int, output: Path, smoke: bool) -> dict[str, Any]:
    """Run H(0) or C(0) with a runtime-valid SimplyQRL head config."""
    from handoff_qrl import minatar_cell as m
    from simplyqrl.runners.train_ppo import build_ppo_from_config

    spec = m.parse_condition(condition, seed)
    if spec.handoff_t != 0.0:
        raise ValueError(condition)
    budget = m.Budget.smoke() if smoke else m.Budget()
    total_steps = budget.total_updates * budget.env_steps_per_update
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cfg = m._ppo_runtime_config(
        seed=seed,
        family=spec.family,
        total_timesteps=total_steps,
        budget=budget,
        run_name=f"reproduction_{spec.condition}_seed{seed}",
        smoke=smoke,
    )
    # `family` is KD metadata, not an ActorCritic runtime-head field.
    cfg["ppo"]["agent_config"]["head"].pop("family", None)
    config_path = output / "train/ppo_config.yaml"
    m._write_yaml(config_path, cfg)

    model = None
    started = time.perf_counter()
    try:
        model, configured_timesteps, _ = build_ppo_from_config(cfg)
        model.train(total_timesteps=configured_timesteps, progress_bar=False)
        checkpoint = m._save_model(
            model,
            output / "train/checkpoint.zip",
            metadata={
                "reproduction_runner": True,
                "condition": spec.condition,
                "training_seed": seed,
            },
        )
        summary: dict[str, Any] = {
            "status": "PASS",
            "environment": "MinAtar Breakout",
            "condition": spec.condition,
            "family": spec.family,
            "handoff_t": 0.0,
            "training_seed": int(seed),
            "smoke": bool(smoke),
            "paper_protocol": not smoke,
            "total_updates": budget.total_updates,
            "total_env_steps": total_steps,
            "training_qnode_stats": m._qstats(model),
            "final_checkpoint": str(checkpoint),
            "config": str(config_path),
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
        if model is not None:
            close = getattr(model, "close", None)
            if callable(close):
                close()


def _run_lunar_h0(*, seed: int, output: Path, smoke: bool) -> dict[str, Any]:
    """Run LunarLander H(0) with a runtime-valid hybrid head config."""
    from handoff_qrl import lunarlander_experiment as m
    from simplyqrl.runners.train_ppo import build_ppo_from_config

    budget = m.Budget.smoke() if smoke else m.Budget()
    total_steps = budget.total_env_steps
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cfg = m._runtime_config(
        seed=seed,
        family="hybrid",
        total_timesteps=total_steps,
        budget=budget,
        run_name=f"reproduction_lunarlander_H(0)_seed{seed}",
        smoke=smoke,
    )
    cfg["ppo"]["agent_config"]["head"].pop("family", None)
    config_path = output / "train/config.yaml"
    m._write_yaml(config_path, cfg)

    model = None
    started = time.perf_counter()
    try:
        model, configured_timesteps, _ = build_ppo_from_config(cfg)
        model.train(total_timesteps=configured_timesteps, progress_bar=False)
        checkpoint = m._save_model(
            model,
            output / "train/checkpoint.zip",
            metadata={"reproduction_runner": True, "condition": "H(0)", "training_seed": seed},
        )
        summary: dict[str, Any] = {
            "status": "PASS",
            "environment": "LunarLander-v3",
            "condition": "H(0)",
            "family": "hybrid",
            "handoff_t": 0.0,
            "training_seed": int(seed),
            "smoke": bool(smoke),
            "paper_protocol": not smoke,
            "total_rollouts": budget.total_rollouts,
            "total_env_steps": total_steps,
            "training_qnode_stats": m._qstats(model),
            "final_checkpoint": str(checkpoint),
            "config": str(config_path),
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
        if model is not None:
            close = getattr(model, "close", None)
            if callable(close):
                close()


def run_experiment(
    *,
    environment: str,
    condition: str,
    seed: int,
    output_dir: str | Path | None = None,
    smoke: bool = False,
    paired_source_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run one condition from the frozen paper protocol.

    C(.25/.5/.75) requires the matching H(t) output so the teacher checkpoint
    and KD dataset bytes are reused exactly, as in the paper protocol.
    """
    env = normalise_environment(environment)
    output = Path(output_dir) if output_dir is not None else default_output_dir(env, condition, seed)
    paired_t = _paired_handoff(condition)
    if paired_t is not None:
        if paired_source_dir is None:
            raise ValueError(
                f"{condition} requires --paired-source pointing to the matching H({paired_t:g}) "
                "run. Use scripts/run_experiments.py --conditions all to reproduce the full "
                "paired matrix automatically."
            )
        from handoff_qrl.paired_control import run_paired_control

        return run_paired_control(
            environment=env,
            handoff_t=paired_t,
            seed=int(seed),
            hybrid_source_dir=paired_source_dir,
            output_dir=output,
            smoke=bool(smoke),
        )

    canonical = condition.strip()
    if env == "minatar" and canonical in {"H(0)", "C(0)"}:
        return _run_minatar_fresh_zero(
            condition=canonical,
            seed=int(seed),
            output=output,
            smoke=bool(smoke),
        )

    if env == "lunarlander" and canonical == "H(0)":
        return _run_lunar_h0(seed=int(seed), output=output, smoke=bool(smoke))

    if env == "minatar":
        from handoff_qrl.minatar_cell import run_cell

        summary = run_cell(
            condition=condition,
            seed=int(seed),
            output_dir=output,
            smoke=bool(smoke),
        )
        legacy_summary = output.resolve() / "cell_summary.json"
        experiment_summary = output.resolve() / "experiment_summary.json"
        experiment_summary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if legacy_summary.is_file():
            legacy_summary.unlink()
        return summary

    from handoff_qrl.lunarlander_experiment import run_experiment as run_lunarlander

    return run_lunarlander(
        condition=condition,
        seed=int(seed),
        output_dir=output,
        smoke=bool(smoke),
    )
