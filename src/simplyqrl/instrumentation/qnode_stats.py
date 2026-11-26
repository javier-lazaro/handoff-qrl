"""Quantum execution statistics for SimplyQRL.

The goal is to measure the cost of the quantum layer without changing the
training loop. We wrap the quantum layer forward pass and accumulate effective
sample-level quantum evaluations and wall-clock time.

Notes:
- `qnode_calls_*` are effective sample-level calls, estimated from batch size.
- `qnode_calls_grad_total` / `qnode_calls_nograd_total` split those calls by
  whether PyTorch autograd was enabled for the forward. This records autograd
  mode, not the number of backward executions.
- `qnode_layer_forwards_*` count PyTorch module forward calls.
- Time is measured around the quantum layer forward call and includes the
  PennyLane/TorchLayer overhead observed by the training loop. It is forward
  time only; any later backward pass is outside this timer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class ScopeStats:
    qnode_calls: int = 0
    qnode_calls_grad: int = 0
    qnode_layer_forwards: int = 0
    qnode_time: float = 0.0


@dataclass
class QNodeStats:
    """Accumulates quantum layer execution statistics."""

    scopes: dict[str, ScopeStats] = field(default_factory=dict)

    def _scope(self, scope: str) -> ScopeStats:
        if scope not in self.scopes:
            self.scopes[scope] = ScopeStats()
        return self.scopes[scope]

    def record(
        self,
        *,
        scope: str,
        elapsed: float,
        effective_calls: int,
        layer_forwards: int = 1,
        grad_enabled: bool = False,
    ) -> None:
        stats = self._scope(scope)
        stats.qnode_calls += int(effective_calls)
        if grad_enabled:
            stats.qnode_calls_grad += int(effective_calls)
        stats.qnode_layer_forwards += int(layer_forwards)
        stats.qnode_time += float(elapsed)

    def reset(self) -> None:
        self.scopes.clear()

    def snapshot(self) -> dict[str, Any]:
        actor = self.scopes.get("actor", ScopeStats())
        critic = self.scopes.get("critic", ScopeStats())

        total_calls = sum(scope.qnode_calls for scope in self.scopes.values())
        total_grad_calls = sum(scope.qnode_calls_grad for scope in self.scopes.values())
        total_forwards = sum(scope.qnode_layer_forwards for scope in self.scopes.values())
        total_time = sum(scope.qnode_time for scope in self.scopes.values())

        return {
            "qnode_calls_total": int(total_calls),
            "qnode_calls_grad_total": int(total_grad_calls),
            "qnode_calls_nograd_total": int(total_calls - total_grad_calls),
            "qnode_layer_forwards_total": int(total_forwards),
            "qnode_time_total": float(total_time),
            "qnode_calls_actor": int(actor.qnode_calls),
            "qnode_layer_forwards_actor": int(actor.qnode_layer_forwards),
            "qnode_time_actor": float(actor.qnode_time),
            "qnode_calls_critic": int(critic.qnode_calls),
            "qnode_layer_forwards_critic": int(critic.qnode_layer_forwards),
            "qnode_time_critic": float(critic.qnode_time),
        }


def estimate_effective_qnode_calls(inputs: Any) -> int:
    """Estimate effective sample-level QNode calls from input batch shape."""

    if isinstance(inputs, torch.Tensor):
        if inputs.ndim <= 1:
            return 1
        return int(inputs.shape[0])

    return 1


class TimedQuantumLayer(nn.Module):
    """Wrap a quantum layer and update shared QNodeStats on every forward."""

    def __init__(
        self,
        layer: nn.Module,
        *,
        stats: QNodeStats,
        scope: str,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.stats = stats
        self.scope = str(scope)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        effective_calls = estimate_effective_qnode_calls(inputs)
        grad_enabled = torch.is_grad_enabled()
        start = time.perf_counter()
        output = self.layer(inputs)
        elapsed = time.perf_counter() - start

        self.stats.record(
            scope=self.scope,
            elapsed=elapsed,
            effective_calls=effective_calls,
            layer_forwards=1,
            grad_enabled=grad_enabled,
        )

        return output
