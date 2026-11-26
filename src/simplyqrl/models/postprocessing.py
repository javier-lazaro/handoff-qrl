"""Post-processing modules for classical and hybrid heads.

In SimplyQRL 1.5, a head may contain several internal stages:

    feature vector -> VQC/linear block -> observables -> post-processing -> output projection

This file keeps post-processing explicit so that techniques such as Output Reuse
and Output Scaling remain visible and can be accounted for separately.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class BasePostProcessor(nn.Module):
    """Base class for post-processing modules."""

    @property
    def output_dim(self) -> int:
        raise NotImplementedError


class IdentityPostProcessor(BasePostProcessor):
    """No-op post-processing."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)

    @property
    def output_dim(self) -> int:
        return self.input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class OutputReuse(BasePostProcessor):
    """Replicate the feature/readout vector along the last dimension.

    This implements the Output Reuse idea used in hybrid QRL pipelines, where
    quantum measurements are repeated before the classical inference layer.
    """

    def __init__(self, input_dim: int, repetitions: int) -> None:
        super().__init__()
        if repetitions < 1:
            raise ValueError("OutputReuse repetitions must be >= 1.")

        self.input_dim = int(input_dim)
        self.repetitions = int(repetitions)

    @property
    def output_dim(self) -> int:
        return self.input_dim * self.repetitions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.repetitions == 1:
            return x
        return torch.cat([x] * self.repetitions, dim=-1)


class OutputScaling(BasePostProcessor):
    """Learnable or fixed output scaling.

    Scaling can be scalar or per-feature. Keeping it as a module makes its
    parameters visible for parameter accounting.
    """

    def __init__(
        self,
        input_dim: int,
        init_scale: float = 1.0,
        trainable: bool = True,
        per_feature: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.per_feature = bool(per_feature)

        shape = (self.input_dim,) if self.per_feature else (1,)
        scale = torch.full(shape, float(init_scale), dtype=torch.float32)

        if trainable:
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer("scale", scale)

    @property
    def output_dim(self) -> int:
        return self.input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class PostProcessingStack(BasePostProcessor):
    """Sequential stack of post-processing modules."""

    def __init__(self, processors: Sequence[BasePostProcessor]) -> None:
        super().__init__()
        self.processors = nn.ModuleList(processors)

        if len(self.processors) == 0:
            raise ValueError("PostProcessingStack requires at least one processor.")

    @property
    def output_dim(self) -> int:
        return self.processors[-1].output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for processor in self.processors:
            x = processor(x)
        return x


def build_postprocessing_stack(
    input_dim: int,
    reuse_repetitions: int = 1,
    use_output_scaling: bool = False,
    output_scaling_trainable: bool = True,
    output_scaling_per_feature: bool = False,
) -> BasePostProcessor:
    """Build a standard post-processing stack from simple config values."""

    processors: list[BasePostProcessor] = []

    if reuse_repetitions > 1:
        reuse = OutputReuse(input_dim=input_dim, repetitions=reuse_repetitions)
        processors.append(reuse)
        current_dim = reuse.output_dim
    else:
        current_dim = int(input_dim)

    if use_output_scaling:
        processors.append(
            OutputScaling(
                input_dim=current_dim,
                trainable=output_scaling_trainable,
                per_feature=output_scaling_per_feature,
            )
        )

    if not processors:
        return IdentityPostProcessor(input_dim=input_dim)

    return PostProcessingStack(processors)
