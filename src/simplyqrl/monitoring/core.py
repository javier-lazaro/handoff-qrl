"""Core training monitor for SimplyQRL algorithms."""

from __future__ import annotations

from typing import Any

from simplyqrl.monitoring.base import BaseMonitor
from simplyqrl.reporting.logger import ArtifactLogger


class CoreTrainingMonitor(BaseMonitor):
    """Default monitor for essential training metrics.

    This monitor is intentionally broad: it handles the core metrics that every
    training run should expose, avoiding one tiny callback per metric.
    """

    def __init__(self, logger: ArtifactLogger) -> None:
        self.logger = logger
        self.num_updates = 0
        self.num_episodes = 0
        self.best_episode_return: float | None = None
        self.last_episode_return: float | None = None
        self.last_episode_length: int | None = None

    def on_training_start(self, state: dict[str, Any]) -> None:
        self.logger.update_summary(
            {
                "total_timesteps": state.get("total_timesteps"),
                "seed": state.get("seed"),
                "env_id": state.get("env_id"),
                "agent_type": state.get("agent_type"),
                "trainable_params": state.get("trainable_params"),
                "device": state.get("device"),
            }
        )
        self.logger.log_event(
            "training_start",
            {
                "global_step": state.get("global_step", 0),
                "total_timesteps": state.get("total_timesteps"),
                "device": state.get("device"),
            },
        )

    def on_episode_end(self, state: dict[str, Any]) -> None:
        episode_return = float(state["episode_return"])
        episode_length = int(state["episode_length"])
        self.num_episodes += 1
        self.last_episode_return = episode_return
        self.last_episode_length = episode_length

        if self.best_episode_return is None or episode_return > self.best_episode_return:
            self.best_episode_return = episode_return

        self.logger.log_event(
            "episode",
            {
                "global_step": int(state["global_step"]),
                "episode_return": episode_return,
                "episode_length": episode_length,
                "num_episodes": self.num_episodes,
            },
        )

    def on_rollout_end(self, state: dict[str, Any]) -> None:
        self.logger.log_event(
            "rollout",
            {
                "global_step": int(state["global_step"]),
                "update": int(state["update"]),
                "rollout_time": float(state["rollout_time"]),
            },
        )

    def on_update_end(self, state: dict[str, Any]) -> None:
        self.num_updates += 1

        payload = {
            "global_step": int(state["global_step"]),
            "update": int(state["update"]),
            "update_time": float(state["update_time"]),
            "steps_per_second": float(state["steps_per_second"]),
        }

        for key in (
            "loss_policy",
            "loss_value",
            "loss_entropy",
            "loss_total",
            "approx_kl",
            "clipfrac",
            "explained_variance",
            "learning_rate",
            "grad_norm",
        ):
            if key in state and state[key] is not None:
                payload[key] = float(state[key])

        if "optimizer_steps" in state and state["optimizer_steps"] is not None:
            payload["optimizer_steps"] = int(state["optimizer_steps"])

        self.logger.log_event("update", payload)

    def on_training_end(self, state: dict[str, Any]) -> None:
        self.logger.log_event(
            "training_end",
            {
                "global_step": int(state.get("global_step", 0)),
                "wall_clock_elapsed": float(state.get("wall_clock_elapsed", 0.0)),
                "num_updates": self.num_updates,
                "num_episodes": self.num_episodes,
            },
        )
        self.logger.update_summary(
            {
                "final_global_step": int(state.get("global_step", 0)),
                "wall_clock_elapsed": float(state.get("wall_clock_elapsed", 0.0)),
                "num_updates": self.num_updates,
                "num_episodes": self.num_episodes,
                "last_episode_return": self.last_episode_return,
                "last_episode_length": self.last_episode_length,
                "best_episode_return": self.best_episode_return,
            }
        )

    def on_training_error(self, state: dict[str, Any]) -> None:
        self.logger.log_event(
            "training_error",
            {
                "global_step": int(state.get("global_step", 0)),
                "error": state.get("error"),
            },
        )
        self.logger.update_summary(
            {
                "status": "failed",
                "error": state.get("error"),
            }
        )
