"""Evaluation monitor for SimplyQRL algorithms."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch

from simplyqrl.monitoring.base import BaseMonitor
from simplyqrl.reporting.logger import ArtifactLogger


class EvaluationMonitor(BaseMonitor):
    """Periodic evaluation monitor.

    This monitor owns the evaluation artifact (`evaluations.csv`) and writes an
    evaluation event to `metrics.jsonl`. The algorithm decides when to call it.
    """

    def __init__(
        self,
        logger: ArtifactLogger,
        *,
        eval_freq: int,
        eval_episodes: int = 5,
        max_eval_steps: int = 1_000,
        deterministic: bool = True,
    ) -> None:
        if eval_freq <= 0:
            raise ValueError("eval_freq must be > 0 when EvaluationMonitor is enabled.")
        if eval_episodes <= 0:
            raise ValueError("eval_episodes must be > 0.")

        self.logger = logger
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)
        self.max_eval_steps = int(max_eval_steps)
        self.deterministic = bool(deterministic)

        self.next_eval_step = self.eval_freq
        self.num_evaluations = 0
        self.best_eval_return_mean: float | None = None

        self.evaluations_path = self.logger.run_dir / "evaluations.csv"
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.evaluations_path.exists():
            return

        with self.evaluations_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "evaluation_index",
                    "global_step",
                    "eval_return_mean",
                    "eval_return_std",
                    "eval_length_mean",
                    "eval_length_std",
                    "eval_time",
                    "eval_episodes",
                ],
            )
            writer.writeheader()

    def should_evaluate(self, global_step: int) -> bool:
        return int(global_step) >= self.next_eval_step

    def mark_evaluated(self, global_step: int) -> None:
        while self.next_eval_step <= int(global_step):
            self.next_eval_step += self.eval_freq

    def _obs_to_tensor(self, obs: Any, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    def evaluate(
        self,
        *,
        agent: torch.nn.Module,
        env_factory: Callable[[], gym.Env],
        device: torch.device,
        global_step: int,
        seed: int,
    ) -> dict[str, Any]:
        was_training = agent.training
        agent.eval()

        returns: list[float] = []
        lengths: list[int] = []

        start_time = time.time()

        try:
            for episode_idx in range(self.eval_episodes):
                env = env_factory()
                obs, _ = env.reset(seed=int(seed) + 10_000 + episode_idx)

                done = False
                episode_return = 0.0
                episode_length = 0

                while not done and episode_length < self.max_eval_steps:
                    obs_t = self._obs_to_tensor(obs, device)

                    with torch.no_grad():
                        logits = agent(obs_t)

                        if self.deterministic:
                            action = int(torch.argmax(logits, dim=-1).item())
                        else:
                            probs = torch.distributions.Categorical(logits=logits)
                            action = int(probs.sample().item())

                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_return += float(reward)
                    episode_length += 1
                    done = bool(terminated or truncated)

                env.close()

                returns.append(episode_return)
                lengths.append(episode_length)

        finally:
            if was_training:
                agent.train()

        eval_time = time.time() - start_time
        self.mark_evaluated(global_step)

        return {
            "global_step": int(global_step),
            "eval_return_mean": float(np.mean(returns)),
            "eval_return_std": float(np.std(returns)),
            "eval_length_mean": float(np.mean(lengths)),
            "eval_length_std": float(np.std(lengths)),
            "eval_time": float(eval_time),
            "eval_episodes": self.eval_episodes,
        }

    def on_evaluation_end(self, state: dict[str, Any]) -> None:
        self.num_evaluations += 1

        eval_return_mean = float(state["eval_return_mean"])

        if (
            self.best_eval_return_mean is None
            or eval_return_mean > self.best_eval_return_mean
        ):
            self.best_eval_return_mean = eval_return_mean

        row = {
            "evaluation_index": self.num_evaluations,
            "global_step": int(state["global_step"]),
            "eval_return_mean": eval_return_mean,
            "eval_return_std": float(state["eval_return_std"]),
            "eval_length_mean": float(state["eval_length_mean"]),
            "eval_length_std": float(state["eval_length_std"]),
            "eval_time": float(state["eval_time"]),
            "eval_episodes": int(state["eval_episodes"]),
        }

        with self.evaluations_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

        self.logger.log_event("evaluation", row)
        self.logger.update_summary(
            {
                "num_evaluations": self.num_evaluations,
                "last_eval_return_mean": row["eval_return_mean"],
                "last_eval_return_std": row["eval_return_std"],
                "last_eval_length_mean": row["eval_length_mean"],
                "best_eval_return_mean": self.best_eval_return_mean,
            }
        )
