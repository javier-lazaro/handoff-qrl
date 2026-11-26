"""MinAtar environment factories owned by the paper repository.

SimplyQRL intentionally accepts external environment factories, so paper-specific
wrapping choices such as frame stacking live here rather than in the reusable
library.  Observations are HWC float32 arrays compatible with SimplyQRL's
CNNExtractor configured with ``tensor_format: channels_last``.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FrameStackedMinAtarEnv(gym.Env):
    """Frame-stacked MinAtar wrapper.

    The base SimplyQRL MinAtar wrapper exposes observations as ``(H, W, C)``.
    This wrapper stacks the last ``frame_stack`` observations along the channel
    dimension, yielding ``(H, W, C * frame_stack)``.
    """

    metadata: dict[str, Any] = {}

    def __init__(self, game: str = "breakout", frame_stack: int = 4) -> None:
        super().__init__()
        if int(frame_stack) <= 0:
            raise ValueError("frame_stack must be a positive integer.")

        from simplyqrl.envs import MinAtarGymEnv

        self.game = str(game)
        self.frame_stack = int(frame_stack)
        self.env = MinAtarGymEnv(game=self.game)
        self.action_space = self.env.action_space

        base_space = self.env.observation_space
        if len(base_space.shape) != 3:
            raise ValueError(
                "FrameStackedMinAtarEnv expects HWC base observations, "
                f"got shape={base_space.shape}."
            )

        height, width, channels = (int(dim) for dim in base_space.shape)
        self.base_observation_shape = (height, width, channels)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(height, width, channels * self.frame_stack),
            dtype=np.float32,
        )
        self._frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)

    def _stack(self) -> np.ndarray:
        if len(self._frames) != self.frame_stack:
            raise RuntimeError(
                f"Expected {self.frame_stack} frames, got {len(self._frames)}."
            )
        return np.concatenate(list(self._frames), axis=-1).astype(np.float32, copy=False)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = np.asarray(obs, dtype=np.float32)
        self._frames.clear()
        for _ in range(self.frame_stack):
            self._frames.append(obs.copy())
        return self._stack(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(np.asarray(obs, dtype=np.float32))
        return self._stack(), float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()


def make_minatar_breakout(frame_stack: int = 4) -> FrameStackedMinAtarEnv:
    """Build the paper-facing MinAtar Breakout environment."""

    return FrameStackedMinAtarEnv(game="breakout", frame_stack=frame_stack)
