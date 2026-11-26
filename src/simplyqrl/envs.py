"""Environment utilities and lightweight wrappers for SimplyQRL 1.5."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv


def _make_factory(base: str | Callable[[], gym.Env] | gym.Env, num_envs: int) -> Callable[[], gym.Env]:
    if isinstance(base, str):
        return lambda: gym.make(base)

    if callable(base):
        return base

    if isinstance(base, gym.Env):
        if base.spec is not None and base.spec.id is not None:
            return lambda: gym.make(base.spec.id)
        if num_envs == 1:
            return lambda: base
        raise ValueError(
            "A concrete gym.Env without spec.id can only be used with num_envs=1. "
            "Prefer passing an environment factory instead."
        )

    raise TypeError(f"Unsupported environment source: {type(base)!r}")


def make_vec_env(base: str | Callable[[], gym.Env] | gym.Env, *, num_envs: int, seed: int):
    """Create a SyncVectorEnv from an env id, env factory or env instance."""

    factory = _make_factory(base, num_envs)

    def _thunk(idx: int):
        def _inner():
            env = factory()
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.reset(seed=seed + idx)
            return env

        return _inner

    return SyncVectorEnv([_thunk(i) for i in range(num_envs)])


class MinAtarGymEnv(gym.Env):
    """Gymnasium wrapper for MinAtar environments.

    Observations are exposed as float32 arrays with shape (H, W, C), compatible
    with CNNExtractor using channels_last format.
    """

    metadata: dict[str, Any] = {}

    def __init__(self, game: str = "breakout") -> None:
        super().__init__()
        from minatar import Environment

        self.game = game
        self.env = Environment(game)

        state = self.env.state()
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=state.shape,
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.env.num_actions())

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.env.seed(seed)
        # MinAtar's Environment.reset() delegates to the game but leaves the
        # sticky-action history untouched. Episode resets must not inherit the
        # previous episode's last action.
        self.env.last_action = 0
        self.env.reset()
        return self.env.state().astype(np.float32), {}

    def step(self, action):
        reward, terminated = self.env.act(int(action))
        obs = self.env.state().astype(np.float32)
        return obs, float(reward), bool(terminated), False, {}

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()


class MinAtarBreakoutEnv(MinAtarGymEnv):
    def __init__(self) -> None:
        super().__init__(game="breakout")


class CartPolePixelsEnv(gym.Env):
    """Pixel-observation CartPole wrapper.

    The underlying dynamics are CartPole-v1, but observations are RGB frames
    resized to (image_size, image_size, 3).
    """

    metadata: dict[str, Any] = {}

    def __init__(self, image_size: int = 64) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.image_size, self.image_size, 3),
            dtype=np.uint8,
        )

    def _get_pixels(self):
        import cv2

        frame = self.env.render()
        frame = cv2.resize(
            frame,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        return frame.astype(np.uint8)

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        return self._get_pixels(), {}

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(int(action))
        return self._get_pixels(), float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        self.env.close()
