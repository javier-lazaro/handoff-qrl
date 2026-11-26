"""YAML-driven PPO runner for SimplyQRL.

Minimal YAML contract:

    python -m simplyqrl.runners.train_ppo --config path/to/config.yaml

The runner intentionally stays thin:
- read YAML
- resolve environment
- build PPO
- call train(total_timesteps)
- let PPO/reporting generate all artifacts

Paper campaign orchestration remains downstream; this runner is the maintained
single-run YAML entrypoint for the library.
"""

from __future__ import annotations

import argparse
import importlib
from functools import partial
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import yaml

from simplyqrl.envs import CartPolePixelsEnv, MinAtarBreakoutEnv, MinAtarGymEnv
from simplyqrl.ppo import PPO


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    if not isinstance(config, dict):
        raise TypeError(f"Config file must contain a YAML mapping: {path}")

    return config


def import_from_string(target: str) -> Any:
    """Import an object from 'module:attribute' or 'module.attribute'."""

    if ":" in target:
        module_name, attr_name = target.split(":", 1)
    else:
        module_name, attr_name = target.rsplit(".", 1)

    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def resolve_builtin_env_factory(
    name: str,
    kwargs: dict[str, Any],
) -> Callable[[], gym.Env]:
    """Resolve built-in SimplyQRL environment factories."""

    normalized = name.lower().replace("-", "_")

    if normalized in {"cartpole_pixels", "cartpole_pixels_env"}:
        return partial(CartPolePixelsEnv, **kwargs)

    if normalized in {"minatar_breakout", "minatar_breakout_env"}:
        return partial(MinAtarBreakoutEnv, **kwargs)

    if normalized in {"minatar", "minatar_env"}:
        game = kwargs.pop("game", "breakout")
        return partial(MinAtarGymEnv, game=game, **kwargs)

    raise KeyError(name)


def resolve_env(env_config: Any) -> tuple[str, Callable[[], gym.Env] | None]:
    """Resolve environment config into (env_id, env_factory_or_none).

    Supported forms:

    env: CartPole-v1

    env:
      id: CartPole-v1

    env:
      factory: cartpole_pixels
      kwargs:
        image_size: 64
      label: CartPolePixels

    env:
      factory: simplyqrl.envs:CartPolePixelsEnv
      kwargs:
        image_size: 64
      label: CartPolePixels
    """

    if env_config is None:
        return "CartPole-v1", None

    if isinstance(env_config, str):
        return env_config, None

    if not isinstance(env_config, dict):
        raise TypeError(
            "env must be either a string env id or a mapping with 'id'/'factory'."
        )

    if "id" in env_config:
        return str(env_config["id"]), None

    if "factory" not in env_config:
        raise ValueError("env mapping must define either 'id' or 'factory'.")

    factory_name = str(env_config["factory"])
    kwargs = dict(env_config.get("kwargs") or {})
    label = str(env_config.get("label") or factory_name)

    try:
        factory = resolve_builtin_env_factory(factory_name, kwargs)
        return label, factory
    except KeyError:
        pass

    target = import_from_string(factory_name)

    if isinstance(target, type) and issubclass(target, gym.Env):
        return label, partial(target, **kwargs)

    if callable(target):
        return label, partial(target, **kwargs)

    raise TypeError(f"Resolved env factory is not callable: {factory_name}")


def build_ppo_from_config(config: dict[str, Any]) -> tuple[PPO, int, bool]:
    """Build PPO from a YAML config."""

    run_config = dict(config.get("run") or {})
    ppo_config = dict(config.get("ppo") or {})

    # Optional convenience schema:
    #
    # agent:
    #   type: v15_classical
    #   config: {...}
    agent_config_block = config.get("agent")
    if isinstance(agent_config_block, dict):
        ppo_config.setdefault(
            "agent_type",
            agent_config_block.get("type")
            or agent_config_block.get("agent_type")
            or "v15_classical",
        )
        if "config" in agent_config_block:
            ppo_config.setdefault("agent_config", agent_config_block["config"])
        elif "agent_config" in agent_config_block:
            ppo_config.setdefault("agent_config", agent_config_block["agent_config"])

    total_timesteps = int(
        run_config.get(
            "total_timesteps",
            config.get("total_timesteps", ppo_config.pop("total_timesteps", 100_000)),
        )
    )

    progress_bar = bool(
        run_config.get(
            "progress_bar",
            config.get("progress_bar", ppo_config.pop("progress_bar", False)),
        )
    )

    if "run_name" in run_config:
        ppo_config.setdefault("run_name", run_config["run_name"])

    env_config = config.get("env", ppo_config.pop("env", None))
    env_id, env_factory = resolve_env(env_config)

    ppo_config.setdefault("env_id", env_id)
    if env_factory is not None:
        ppo_config["env"] = env_factory

    reporting_config = (
        config.get("reporting")
        if "reporting" in config
        else config.get("reporting_config")
    )
    if reporting_config is not None:
        ppo_config["reporting_config"] = dict(reporting_config)

    # Device policy convenience schema.
    #
    # In the legacy YAML format the PPO constructor arguments usually live
    # under ``ppo:``, while top-level sections such as ``run:``, ``env:`` and
    # ``reporting:`` are lifted into PPO kwargs by this runner. Accepting a
    # top-level ``device_policy:`` keeps placement policy visible as a runtime
    # concern instead of hiding it inside architecture config. If callers place
    # ``device_policy`` directly under ``ppo:``, it is already present in
    # ``ppo_config`` and is left untouched.
    root_device_policy = config.get("device_policy")
    if root_device_policy is not None:
        if not isinstance(root_device_policy, dict):
            raise TypeError("device_policy must be a YAML mapping.")
        ppo_config.setdefault("device_policy", dict(root_device_policy))

    model = PPO(**ppo_config)
    return model, total_timesteps, progress_bar


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run SimplyQRL PPO from a YAML config.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    model, total_timesteps, progress_bar = build_ppo_from_config(config)

    model.train(
        total_timesteps=total_timesteps,
        progress_bar=progress_bar,
    )

    if model.artifact_logger is not None:
        print(f"Run directory: {model.artifact_logger.run_dir}")


if __name__ == "__main__":
    main()
