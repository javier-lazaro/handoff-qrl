"""Base monitoring primitives for SimplyQRL algorithms."""

from __future__ import annotations

from typing import Any


class BaseMonitor:
    """Base class for algorithm monitors.

    Monitors observe training events and may write through the artifact logger.
    They should not own the training loop.
    """

    def on_training_start(self, state: dict[str, Any]) -> None:
        pass

    def on_rollout_start(self, state: dict[str, Any]) -> None:
        pass

    def on_episode_end(self, state: dict[str, Any]) -> None:
        pass

    def on_rollout_end(self, state: dict[str, Any]) -> None:
        pass

    def on_update_start(self, state: dict[str, Any]) -> None:
        pass

    def on_backward_end(self, state: dict[str, Any]) -> None:
        pass

    def on_optimizer_step_end(self, state: dict[str, Any]) -> None:
        pass

    def on_update_end(self, state: dict[str, Any]) -> None:
        pass

    def on_evaluation_end(self, state: dict[str, Any]) -> None:
        pass

    def on_training_end(self, state: dict[str, Any]) -> None:
        pass

    def on_training_error(self, state: dict[str, Any]) -> None:
        pass


class MonitorManager:
    """Dispatch training events to a list of monitors."""

    def __init__(self, monitors: list[BaseMonitor] | None = None) -> None:
        self.monitors = monitors or []

    def add(self, monitor: BaseMonitor) -> None:
        self.monitors.append(monitor)

    def _dispatch(self, hook: str, state: dict[str, Any]) -> None:
        for monitor in self.monitors:
            callback = getattr(monitor, hook, None)
            if callback is None:
                # External monitors written against an earlier SimplyQRL
                # monitoring contract may not implement hooks added later.
                # Treat monitor hooks as optional so new observability events
                # remain backwards-compatible and cannot break training.
                continue
            if not callable(callback):
                raise TypeError(
                    f"Monitor {type(monitor).__name__}.{hook} exists but is not callable."
                )
            callback(state)

    def on_training_start(self, state: dict[str, Any]) -> None:
        self._dispatch("on_training_start", state)

    def on_rollout_start(self, state: dict[str, Any]) -> None:
        self._dispatch("on_rollout_start", state)

    def on_episode_end(self, state: dict[str, Any]) -> None:
        self._dispatch("on_episode_end", state)

    def on_rollout_end(self, state: dict[str, Any]) -> None:
        self._dispatch("on_rollout_end", state)

    def on_update_start(self, state: dict[str, Any]) -> None:
        self._dispatch("on_update_start", state)

    def on_backward_end(self, state: dict[str, Any]) -> None:
        self._dispatch("on_backward_end", state)

    def on_optimizer_step_end(self, state: dict[str, Any]) -> None:
        self._dispatch("on_optimizer_step_end", state)

    def on_update_end(self, state: dict[str, Any]) -> None:
        self._dispatch("on_update_end", state)

    def on_evaluation_end(self, state: dict[str, Any]) -> None:
        self._dispatch("on_evaluation_end", state)

    def on_training_end(self, state: dict[str, Any]) -> None:
        self._dispatch("on_training_end", state)

    def on_training_error(self, state: dict[str, Any]) -> None:
        self._dispatch("on_training_error", state)
