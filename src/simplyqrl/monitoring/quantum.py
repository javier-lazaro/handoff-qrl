"""Quantum monitor for SimplyQRL algorithms."""

from __future__ import annotations

from typing import Any

from simplyqrl.monitoring.base import BaseMonitor
from simplyqrl.reporting.logger import ArtifactLogger


def _zero_snapshot() -> dict[str, Any]:
    return {
        "qnode_calls_total": 0,
        "qnode_calls_grad_total": 0,
        "qnode_calls_nograd_total": 0,
        "qnode_layer_forwards_total": 0,
        "qnode_time_total": 0.0,
        "qnode_calls_actor": 0,
        "qnode_layer_forwards_actor": 0,
        "qnode_time_actor": 0.0,
        "qnode_calls_critic": 0,
        "qnode_layer_forwards_critic": 0,
        "qnode_time_critic": 0.0,
    }


class QuantumMonitor(BaseMonitor):
    """Monitor cumulative quantum execution statistics.

    Hybrid heads update a shared QNodeStats object. This monitor reads the
    accumulated stats at update/training end and writes quantum-cost events.
    """

    def __init__(
        self,
        logger: ArtifactLogger,
        *,
        agent: Any,
        include_zeros: bool = True,
    ) -> None:
        self.logger = logger
        self.agent = agent
        self.include_zeros = bool(include_zeros)
        self.last_snapshot = _zero_snapshot()

    def _snapshot(self) -> dict[str, Any]:
        if hasattr(self.agent, "get_qnode_stats"):
            snapshot = self.agent.get_qnode_stats()
            return {**_zero_snapshot(), **snapshot}
        return _zero_snapshot()

    def _delta(self, current: dict[str, Any]) -> dict[str, Any]:
        delta: dict[str, Any] = {}
        for key, value in current.items():
            previous = self.last_snapshot.get(key, 0)
            if isinstance(value, float) or isinstance(previous, float):
                delta[f"{key}_delta"] = float(value) - float(previous)
            else:
                delta[f"{key}_delta"] = int(value) - int(previous)
        return delta

    def on_training_start(self, state: dict[str, Any]) -> None:
        snapshot = self._snapshot()
        self.last_snapshot = dict(snapshot)

        if self.include_zeros:
            self.logger.update_summary(snapshot)

    def on_update_end(self, state: dict[str, Any]) -> None:
        snapshot = self._snapshot()
        delta = self._delta(snapshot)

        delta_calls = int(delta.get("qnode_calls_total_delta", 0))
        delta_time = float(delta.get("qnode_time_total_delta", 0.0))

        if delta_calls == 0 and not self.include_zeros:
            self.last_snapshot = dict(snapshot)
            return

        rollout_time = float(state.get("rollout_time", 0.0) or 0.0)
        update_time = float(state.get("update_time", 0.0) or 0.0)
        iteration_observed_time = rollout_time + update_time

        global_step = int(state.get("global_step", 0) or 0)

        qnode_time_share_iteration = (
            delta_time / iteration_observed_time
            if iteration_observed_time > 0.0
            else 0.0
        )
        qnode_calls_per_step = (
            snapshot["qnode_calls_total"] / global_step if global_step > 0 else 0.0
        )
        qnode_time_per_step = (
            snapshot["qnode_time_total"] / global_step if global_step > 0 else 0.0
        )

        payload = {
            "global_step": global_step,
            "update": int(state.get("update", 0) or 0),
            "rollout_time": rollout_time,
            "update_time": update_time,
            "iteration_observed_time": iteration_observed_time,
            **snapshot,
            **delta,
            "qnode_time_share_iteration": float(qnode_time_share_iteration),
            "qnode_calls_per_step": float(qnode_calls_per_step),
            "qnode_time_per_step": float(qnode_time_per_step),
        }

        self.logger.log_event("quantum", payload)

        self.logger.update_summary(
            {
                **snapshot,
                "qnode_time_share_iteration_last": float(qnode_time_share_iteration),
                "qnode_calls_per_step": float(qnode_calls_per_step),
                "qnode_time_per_step": float(qnode_time_per_step),
            }
        )

        self.last_snapshot = dict(snapshot)

    def on_training_end(self, state: dict[str, Any]) -> None:
        snapshot = self._snapshot()
        global_step = int(state.get("global_step", 0) or 0)

        qnode_calls_per_step = (
            snapshot["qnode_calls_total"] / global_step if global_step > 0 else 0.0
        )
        qnode_time_per_step = (
            snapshot["qnode_time_total"] / global_step if global_step > 0 else 0.0
        )

        self.logger.update_summary(
            {
                **snapshot,
                "qnode_calls_per_step": float(qnode_calls_per_step),
                "qnode_time_per_step": float(qnode_time_per_step),
            }
        )
