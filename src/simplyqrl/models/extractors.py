"""Feature extractors for vector and visual observations.

These modules are intentionally independent from the legacy PPO/DQN code.
They provide the base abstraction needed by SimplyQRL 1.5:

    observation -> feature extractor -> head

The goal is to support both state-based environments and visual environments
without forcing visual observations to be flattened directly into the policy head.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Literal

import torch
from torch import nn


TensorFormat = Literal["auto", "channels_first", "channels_last"]


def _as_shape(shape: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(shape, int):
        return (shape,)
    return tuple(int(dim) for dim in shape)


def _prod(shape: Sequence[int]) -> int:
    return int(prod(shape))


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation: {name}")


class BaseFeatureExtractor(nn.Module):
    """Base class for all feature extractors."""

    @property
    def output_dim(self) -> int:
        raise NotImplementedError


class IdentityExtractor(BaseFeatureExtractor):
    """Flatten observations into a feature vector.

    This is the default extractor for classic state-based environments such as
    CartPole-v1 and LunarLander-v3.
    """

    def __init__(self, observation_shape: int | Sequence[int], flatten: bool = True) -> None:
        super().__init__()
        self.observation_shape = _as_shape(observation_shape)
        self.flatten = flatten
        self._output_dim = _prod(self.observation_shape)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.float()

        # Add batch dimension when receiving a single observation.
        if x.ndim == len(self.observation_shape):
            x = x.unsqueeze(0)

        if self.flatten:
            x = torch.flatten(x, start_dim=1)

        return x


class MLPExtractor(BaseFeatureExtractor):
    """Small MLP feature extractor for vector observations."""

    def __init__(
        self,
        observation_shape: int | Sequence[int],
        features_dim: int = 64,
        hidden_sizes: Sequence[int] = (64,),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.observation_shape = _as_shape(observation_shape)
        self.input_dim = _prod(self.observation_shape)
        self._output_dim = int(features_dim)

        layers: list[nn.Module] = []
        previous_dim = self.input_dim

        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(previous_dim, int(hidden_dim)))
            layers.append(_activation(activation))
            previous_dim = int(hidden_dim)

        layers.append(nn.Linear(previous_dim, self._output_dim))
        layers.append(_activation(activation))

        self.network = nn.Sequential(*layers)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.float()

        if x.ndim == len(self.observation_shape):
            x = x.unsqueeze(0)

        x = torch.flatten(x, start_dim=1)
        return self.network(x)


class CNNExtractor(BaseFeatureExtractor):
    """Compact CNN extractor for visual observations.

    Designed for small visual RL observations such as MinAtar and CartPole Pixels.
    It accepts either HWC or CHW observations and projects them into a compact
    latent vector that can be consumed by a classical or hybrid head.
    """

    def __init__(
        self,
        observation_shape: Sequence[int],
        features_dim: int = 128,
        tensor_format: TensorFormat = "auto",
        normalize_images: bool = True,
    ) -> None:
        super().__init__()

        self.observation_shape = _as_shape(observation_shape)
        if len(self.observation_shape) != 3:
            raise ValueError(
                "CNNExtractor expects a 3D observation shape, e.g. (H, W, C) or (C, H, W). "
                f"Got {self.observation_shape}."
            )

        self.tensor_format = tensor_format
        self.normalize_images = normalize_images
        self.channels_last = self._infer_channels_last(self.observation_shape, tensor_format)

        if self.channels_last:
            height, width, channels = self.observation_shape
        else:
            channels, height, width = self.observation_shape

        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)
        self._output_dim = int(features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(self.channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, self.channels, self.height, self.width)
            n_flatten = self.cnn(dummy).shape[1]

        self.projection = nn.Sequential(
            nn.Linear(n_flatten, self._output_dim),
            nn.ReLU(),
        )

    @staticmethod
    def _infer_channels_last(shape: tuple[int, ...], tensor_format: TensorFormat) -> bool:
        if tensor_format == "channels_last":
            return True
        if tensor_format == "channels_first":
            return False

        # Heuristic:
        # - RGB images and MinAtar usually come as HWC, e.g. (400, 600, 3) or (10, 10, C).
        # - CHW usually starts with a small channel count, e.g. (3, 84, 84).
        first, _, last = shape
        if first in {1, 3, 4} and last not in {1, 3, 4}:
            return False
        return True

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.float()

        # Add batch dimension when receiving a single image.
        if x.ndim == 3:
            x = x.unsqueeze(0)

        if self.channels_last:
            x = x.permute(0, 3, 1, 2).contiguous()

        # Keep binary MinAtar observations unchanged, but scale uint8/RGB images.
        if self.normalize_images and x.numel() > 0 and torch.max(x).item() > 1.5:
            x = x / 255.0

        x = self.cnn(x)
        return self.projection(x)
