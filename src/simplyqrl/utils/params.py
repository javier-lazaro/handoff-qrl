"""Utilities for parameter accounting.

For the Budgeted QRL work, parameter counts must distinguish between encoder,
VQC, post-processing and classical inference layers. Counting only total
parameters is not enough because hybrid heads often combine quantum parameters
with classical post-processing capacity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, asdict

from torch import nn


def count_parameters(module: nn.Module | None, trainable_only: bool = False) -> int:
    """Count parameters in a torch module."""

    if module is None:
        return 0

    parameters = module.parameters()
    if trainable_only:
        return int(sum(p.numel() for p in parameters if p.requires_grad))

    return int(sum(p.numel() for p in parameters))


def count_trainable_params(module: nn.Module | None) -> int:
    """Count trainable parameters in a torch module."""

    return count_parameters(module, trainable_only=True)


@dataclass(frozen=True)
class ParameterBreakdown:
    """Structured parameter report for a SimplyQRL agent."""

    trainable_params_total: int
    trainable_params_encoder: int = 0
    trainable_params_head_total: int = 0
    trainable_params_vqc: int = 0
    trainable_params_postprocessing: int = 0
    trainable_params_inference_nn: int = 0
    trainable_params_other: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def build_parameter_breakdown(
    *,
    encoder: nn.Module | None = None,
    head: nn.Module | None = None,
    vqc: nn.Module | None = None,
    postprocessing: nn.Module | None = None,
    inference_nn: nn.Module | None = None,
    other_modules: Mapping[str, nn.Module | None] | None = None,
) -> ParameterBreakdown:
    """Build a structured parameter breakdown.

    Parameters are counted by explicitly provided modules. This avoids relying
    on fragile name matching and makes the accounting compatible with future
    classical, quantum and hybrid heads.
    """

    encoder_params = count_trainable_params(encoder)
    head_params = count_trainable_params(head)
    vqc_params = count_trainable_params(vqc)
    postprocessing_params = count_trainable_params(postprocessing)
    inference_params = count_trainable_params(inference_nn)

    other_params = 0
    if other_modules:
        other_params = sum(count_trainable_params(module) for module in other_modules.values())

    total = (
        encoder_params
        + head_params
        + vqc_params
        + postprocessing_params
        + inference_params
        + other_params
    )

    return ParameterBreakdown(
        trainable_params_total=total,
        trainable_params_encoder=encoder_params,
        trainable_params_head_total=head_params,
        trainable_params_vqc=vqc_params,
        trainable_params_postprocessing=postprocessing_params,
        trainable_params_inference_nn=inference_params,
        trainable_params_other=other_params,
    )
