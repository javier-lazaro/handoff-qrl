"""SimplyQRL 1.5 public package interface."""

from __future__ import annotations

from typing import Any

__version__ = "1.5.0.dev0"
__all__ = ["PPO", "__version__"]


def __getattr__(name: str) -> Any:
    """Load the PPO entrypoint lazily to keep package metadata lightweight."""

    if name == "PPO":
        from simplyqrl.ppo import PPO

        return PPO
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
