"""Reusable optimizer and gradient-clipping helpers for SimplyQRL PPO.

The PPO algorithm owns the loss and update loop.  This module owns the
branch-aware mechanics needed by heterogeneous actor-critic agents:

- validating disjoint ``shared``/``actor``/``critic`` parameter groups;
- building one Adam with either a global learning rate or per-branch rates;
- annealing those rates without losing their relative scale;
- applying global or per-branch gradient clipping across mixed CPU/CUDA
  parameter sets.

Keeping these operations outside :mod:`simplyqrl.ppo` makes the optimization
contract explicit and reusable without introducing paper-specific scheduling
logic into the library.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn, optim

OPTIMIZATION_BRANCHES = ("shared", "actor", "critic")
SUPPORTED_GRADIENT_CLIPPING_MODES = frozenset({"global", "separate_actor_critic"})
SUPPORTED_LEARNING_RATE_MODES = frozenset({"global", "separate_actor_critic"})

ParameterGroups = dict[str, list[nn.Parameter]]


def clip_grad_norm_mixed_device(
    parameters: Iterable[nn.Parameter],
    max_norm: float,
) -> torch.Tensor:
    """Clip gradients for parameters that may span CPU and CUDA devices.

    The returned norm is the pre-clipping global L2 norm on CPU, matching the
    contract of :func:`torch.nn.utils.clip_grad_norm_` closely enough for PPO
    reporting and diagnostics.
    """

    params = [parameter for parameter in parameters if parameter.grad is not None]
    if not params:
        return torch.tensor(0.0)

    norms = [parameter.grad.detach().norm(2).to("cpu") for parameter in params]
    total_norm = torch.norm(torch.stack(norms), 2)

    clip_coef = float(max_norm) / (float(total_norm) + 1e-6)
    if clip_coef < 1.0:
        for parameter in params:
            parameter.grad.detach().mul_(clip_coef)

    return total_norm


def resolve_optimization_parameter_groups(
    agent: nn.Module,
    *,
    require_branch_groups: bool,
) -> ParameterGroups:
    """Return disjoint trainable ``shared``/``actor``/``critic`` groups.

    Agents that do not expose ``get_optimization_parameter_groups`` remain
    supported only for the legacy-equivalent global optimization path.  The
    modular SimplyQRL actor-critic agent exposes the method and therefore
    supports both global and branch-aware optimization.
    """

    getter = getattr(agent, "get_optimization_parameter_groups", None)
    if not callable(getter):
        if require_branch_groups:
            raise TypeError(
                "Separate actor/critic clipping or learning rates require an agent "
                "that exposes get_optimization_parameter_groups()."
            )
        return {
            "shared": [
                parameter for parameter in agent.parameters() if parameter.requires_grad
            ],
            "actor": [],
            "critic": [],
        }

    raw = getter()
    if not isinstance(raw, Mapping):
        raise TypeError("get_optimization_parameter_groups() must return a mapping.")

    groups: ParameterGroups = {
        name: [
            parameter
            for parameter in raw.get(name, [])
            if isinstance(parameter, nn.Parameter) and parameter.requires_grad
        ]
        for name in OPTIMIZATION_BRANCHES
    }

    seen: set[int] = set()
    for group_name, parameters in groups.items():
        for parameter in parameters:
            if id(parameter) in seen:
                raise RuntimeError(
                    "Optimization parameter groups must be disjoint; "
                    f"duplicate parameter found in {group_name!r}."
                )
            seen.add(id(parameter))

    expected = {
        id(parameter) for parameter in agent.parameters() if parameter.requires_grad
    }
    if seen != expected:
        raise RuntimeError(
            "Optimization parameter groups must cover all trainable PPO parameters "
            f"exactly (missing={len(expected - seen)}, extra={len(seen - expected)})."
        )

    return groups


def build_adam_optimizer(
    agent: nn.Module,
    parameter_groups: ParameterGroups,
    *,
    learning_rate_mode: str,
    learning_rate: float,
    branch_learning_rates: Mapping[str, float],
    eps: float = 1e-5,
) -> optim.Adam:
    """Build one Adam using global or branch-specific learning-rate groups."""

    if learning_rate_mode == "global":
        parameters = [
            parameter for parameter in agent.parameters() if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("No trainable parameters were available for PPO Adam.")
        return optim.Adam(
            [{"params": parameters, "lr": learning_rate, "name": "global"}],
            lr=learning_rate,
            eps=eps,
        )

    optimizer_groups: list[dict[str, Any]] = []
    for name in OPTIMIZATION_BRANCHES:
        parameters = parameter_groups[name]
        if not parameters:
            continue
        optimizer_groups.append(
            {
                "params": parameters,
                "lr": float(branch_learning_rates[name]),
                "name": name,
            }
        )

    if not optimizer_groups:
        raise RuntimeError("No trainable parameters were available for PPO Adam.")
    return optim.Adam(optimizer_groups, eps=eps)


def base_learning_rate_for_group(
    group: Mapping[str, Any],
    *,
    learning_rate_mode: str,
    learning_rate: float,
    branch_learning_rates: Mapping[str, float],
) -> float:
    """Return the configured base LR associated with an optimizer group."""

    name = str(group.get("name", "global"))
    if learning_rate_mode == "global" or name == "global":
        return float(learning_rate)
    if name not in branch_learning_rates:
        raise KeyError(f"Unknown optimizer parameter-group name: {name!r}.")
    return float(branch_learning_rates[name])


def set_annealed_learning_rates(
    optimizer: optim.Optimizer,
    fraction: float,
    *,
    learning_rate_mode: str,
    learning_rate: float,
    branch_learning_rates: Mapping[str, float],
) -> None:
    """Apply one annealing fraction while preserving branch LR ratios."""

    for group in optimizer.param_groups:
        group["lr"] = float(fraction) * base_learning_rate_for_group(
            group,
            learning_rate_mode=learning_rate_mode,
            learning_rate=learning_rate,
            branch_learning_rates=branch_learning_rates,
        )


def current_branch_learning_rates(
    optimizer: optim.Optimizer,
    *,
    learning_rate_mode: str,
) -> dict[str, float | None]:
    """Return effective LRs normalized to shared/actor/critic reporting keys."""

    if learning_rate_mode == "global":
        current = float(optimizer.param_groups[0]["lr"])
        return {name: current for name in OPTIMIZATION_BRANCHES}

    rates: dict[str, float | None] = {name: None for name in OPTIMIZATION_BRANCHES}
    for group in optimizer.param_groups:
        name = str(group.get("name", ""))
        if name in rates:
            rates[name] = float(group["lr"])
    return rates


def apply_gradient_clipping(
    agent: nn.Module,
    parameter_groups: ParameterGroups,
    *,
    gradient_clipping_mode: str,
    max_grad_norm: float,
    branch_max_grad_norms: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    """Apply global or branch-specific clipping without changing PPO loss semantics."""

    if gradient_clipping_mode == "global":
        total_norm = clip_grad_norm_mixed_device(agent.parameters(), max_grad_norm)
        return total_norm, None

    returned_group_norms = {
        name: clip_grad_norm_mixed_device(
            parameter_groups[name],
            float(branch_max_grad_norms[name]),
        )
        for name in OPTIMIZATION_BRANCHES
    }
    total_norm = torch.sqrt(
        sum(
            norm.detach().to("cpu").double().pow(2)
            for norm in returned_group_norms.values()
        )
    )
    return total_norm, returned_group_norms
