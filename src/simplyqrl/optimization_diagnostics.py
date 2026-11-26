"""Observational optimizer diagnostics for actor-critic PPO.

The probe in this module measures gradients and actual Adam parameter updates
without changing the loss, optimizer, clipping policy or update order. It is
used to diagnose whether a joint actor-critic optimizer behaves differently for
classical and hybrid heads, including global versus branch-wise clipping.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
from torch import nn


GROUP_NAMES = ("shared", "actor", "critic")
SUPPORTED_GRADIENT_CLIPPING_MODES = {"global", "separate_actor_critic"}


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _unique_trainable(parameters: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    result: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        result.append(parameter)
    return result


def _l2_norm_tensors(tensors: Iterable[torch.Tensor]) -> float:
    squared = 0.0
    for tensor in tensors:
        detached = tensor.detach()
        if detached.numel() == 0:
            continue
        value = torch.sum(detached.double() * detached.double()).to("cpu")
        squared += float(value)
    return math.sqrt(max(squared, 0.0))


def _gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    return _l2_norm_tensors(
        parameter.grad for parameter in parameters if parameter.grad is not None
    )


def _parameter_norm(parameters: Iterable[nn.Parameter]) -> float:
    return _l2_norm_tensors(parameter.data for parameter in parameters)


def _safe_ratio(numerator: float, denominator: float, eps: float = 1e-16) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if abs(denominator) <= eps:
        return None
    return numerator / denominator


def _normalise_gradient_clipping_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in SUPPORTED_GRADIENT_CLIPPING_MODES:
        supported = ", ".join(sorted(SUPPORTED_GRADIENT_CLIPPING_MODES))
        raise ValueError(
            f"Unsupported gradient_clipping_mode={value!r}. Supported: {supported}."
        )
    return mode


def _group_learning_rate(
    optimizer: torch.optim.Optimizer,
    parameters: Iterable[nn.Parameter],
) -> float | None:
    parameter_ids = {id(parameter) for parameter in parameters}
    if not parameter_ids:
        return None

    learning_rates: set[float] = set()
    covered: set[int] = set()
    for optimizer_group in optimizer.param_groups:
        lr = float(optimizer_group["lr"])
        for parameter in optimizer_group.get("params", []):
            if id(parameter) in parameter_ids:
                covered.add(id(parameter))
                learning_rates.add(lr)

    if covered != parameter_ids:
        raise RuntimeError(
            "Optimization diagnostics could not resolve the optimizer learning rate "
            "for every parameter in a diagnostic group."
        )
    if len(learning_rates) != 1:
        raise RuntimeError(
            "A diagnostic parameter group spans optimizer parameter groups with "
            "different learning rates. Split the diagnostic group first."
        )
    return next(iter(learning_rates))


class OptimizationDiagnosticsProbe:
    """Capture sparse pre/post-clipping gradients and actual optimizer updates."""

    GROUP_NAMES = GROUP_NAMES

    def __init__(
        self,
        *,
        agent: nn.Module,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float,
        sample_every_optimizer_steps: int = 64,
        gradient_clipping_mode: str = "global",
        group_max_grad_norms: dict[str, float] | None = None,
    ) -> None:
        if sample_every_optimizer_steps <= 0:
            raise ValueError("sample_every_optimizer_steps must be > 0.")

        getter = getattr(agent, "get_optimization_parameter_groups", None)
        if not callable(getter):
            raise TypeError(
                "Agent does not expose get_optimization_parameter_groups(); "
                "optimizer diagnostics are only supported for modular actor-critic agents."
            )

        raw_groups = getter()
        self.groups = {
            name: _unique_trainable(raw_groups.get(name, []))
            for name in self.GROUP_NAMES
        }
        self.optimizer = optimizer
        self.max_grad_norm = float(max_grad_norm)
        self.gradient_clipping_mode = _normalise_gradient_clipping_mode(
            gradient_clipping_mode
        )
        supplied_group_norms = dict(group_max_grad_norms or {})
        self.group_max_grad_norms = {
            name: float(supplied_group_norms.get(name, self.max_grad_norm))
            for name in self.GROUP_NAMES
        }
        for name, limit in self.group_max_grad_norms.items():
            if limit <= 0:
                raise ValueError(f"{name}_max_grad_norm must be > 0, got {limit}.")

        self.sample_every_optimizer_steps = int(sample_every_optimizer_steps)
        self.optimizer_step = 0
        self._active: dict[str, Any] | None = None

        all_ids: set[int] = set()
        for name in self.GROUP_NAMES:
            for parameter in self.groups[name]:
                if id(parameter) in all_ids:
                    raise RuntimeError(
                        "Optimization diagnostics parameter groups must be disjoint."
                    )
                all_ids.add(id(parameter))

        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group.get("params", [])
            if isinstance(parameter, nn.Parameter) and parameter.requires_grad
        }
        missing = all_ids - optimizer_ids
        extra = optimizer_ids - all_ids
        if missing or extra:
            raise RuntimeError(
                "Optimization diagnostics parameter groups must cover the PPO optimizer "
                f"exactly (missing={len(missing)}, unclassified={len(extra)})."
            )

    def begin_step(
        self,
        *,
        update: int,
        epoch: int,
        minibatch: int,
        global_step: int,
        losses: dict[str, Any],
    ) -> bool:
        """Start a sampled optimizer step after ``loss.backward()`` and before clipping."""

        self.optimizer_step += 1
        if self.optimizer_step % self.sample_every_optimizer_steps != 0:
            self._active = None
            return False

        group_grad_pre = {
            name: _gradient_norm(parameters) for name, parameters in self.groups.items()
        }
        global_grad_pre = math.sqrt(
            sum(value * value for value in group_grad_pre.values())
        )
        global_clip_coefficient = min(
            1.0,
            self.max_grad_norm / (global_grad_pre + 1e-6),
        )

        snapshots: dict[int, torch.Tensor] = {}
        group_parameter_norm_pre: dict[str, float] = {}
        group_learning_rates: dict[str, float | None] = {}
        for name, parameters in self.groups.items():
            group_parameter_norm_pre[name] = _parameter_norm(parameters)
            group_learning_rates[name] = _group_learning_rate(self.optimizer, parameters)
            for parameter in parameters:
                snapshots[id(parameter)] = parameter.detach().clone()

        payload: dict[str, Any] = {
            "optimizer_step": self.optimizer_step,
            "global_step": int(global_step),
            "update": int(update),
            "epoch": int(epoch),
            "minibatch": int(minibatch),
            # Backward-compatible alias for single-LR runs.
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "learning_rate_mode": (
                "separate_actor_critic"
                if len(self.optimizer.param_groups) > 1
                else "global"
            ),
            "max_grad_norm": self.max_grad_norm,
            "gradient_clipping_mode": self.gradient_clipping_mode,
            "grad_global_pre_clip": global_grad_pre,
            "clip_coefficient_global": (
                global_clip_coefficient
                if self.gradient_clipping_mode == "global"
                else None
            ),
            "clip_active": False,
            "parameter_shared_count": sum(p.numel() for p in self.groups["shared"]),
            "parameter_actor_count": sum(p.numel() for p in self.groups["actor"]),
            "parameter_critic_count": sum(p.numel() for p in self.groups["critic"]),
            "snapshots": snapshots,
        }
        for key, value in losses.items():
            parsed = _finite_float(value)
            if parsed is not None:
                payload[key] = parsed

        any_clip_active = False
        for name in self.GROUP_NAMES:
            payload[f"learning_rate_{name}"] = group_learning_rates[name]
            payload[f"max_grad_norm_{name}"] = self.group_max_grad_norms[name]
            payload[f"grad_{name}_pre_clip"] = group_grad_pre[name]
            payload[f"parameter_{name}_norm_pre"] = group_parameter_norm_pre[name]
            if group_grad_pre[name] <= 0.0:
                coefficient = 1.0
            elif self.gradient_clipping_mode == "global":
                coefficient = global_clip_coefficient
            else:
                coefficient = min(
                    1.0,
                    self.group_max_grad_norms[name] / (group_grad_pre[name] + 1e-6),
                )
            active = coefficient < 1.0
            payload[f"clip_coefficient_{name}"] = coefficient
            payload[f"clip_active_{name}"] = active
            any_clip_active = any_clip_active or active
        payload["clip_active"] = any_clip_active

        payload["grad_actor_to_critic_ratio_pre_clip"] = _safe_ratio(
            group_grad_pre["actor"], group_grad_pre["critic"]
        )
        global_sq = global_grad_pre * global_grad_pre
        payload["grad_actor_share_pre_clip"] = (
            None if global_sq <= 0 else group_grad_pre["actor"] ** 2 / global_sq
        )
        payload["grad_critic_share_pre_clip"] = (
            None if global_sq <= 0 else group_grad_pre["critic"] ** 2 / global_sq
        )
        payload["grad_shared_share_pre_clip"] = (
            None if global_sq <= 0 else group_grad_pre["shared"] ** 2 / global_sq
        )
        self._active = payload
        return True

    def capture_post_clip(
        self,
        returned_global_norm: Any = None,
        *,
        returned_group_norms: dict[str, Any] | None = None,
    ) -> None:
        """Capture branch norms after clipping and cross-check the clip helper."""

        if self._active is None:
            return
        returned = _finite_float(returned_global_norm)
        if returned is not None:
            self._active["clip_returned_global_norm"] = returned
            self._active["grad_global_pre_clip_consistency_abs_error"] = abs(
                returned - float(self._active["grad_global_pre_clip"])
            )

        for name, value in dict(returned_group_norms or {}).items():
            if name not in self.GROUP_NAMES:
                continue
            parsed = _finite_float(value)
            if parsed is None:
                continue
            self._active[f"clip_returned_{name}_norm"] = parsed
            self._active[f"grad_{name}_pre_clip_consistency_abs_error"] = abs(
                parsed - float(self._active[f"grad_{name}_pre_clip"])
            )

        group_grad_post = {
            name: _gradient_norm(parameters) for name, parameters in self.groups.items()
        }
        global_grad_post = math.sqrt(
            sum(value * value for value in group_grad_post.values())
        )
        self._active["grad_global_post_clip"] = global_grad_post
        for name in self.GROUP_NAMES:
            self._active[f"grad_{name}_post_clip"] = group_grad_post[name]
        self._active["grad_actor_to_critic_ratio_post_clip"] = _safe_ratio(
            group_grad_post["actor"], group_grad_post["critic"]
        )

    def finish_step(self) -> dict[str, Any] | None:
        """Capture actual parameter deltas after ``optimizer.step()``."""

        if self._active is None:
            return None

        snapshots: dict[int, torch.Tensor] = self._active.pop("snapshots")
        update_norms: dict[str, float] = {}
        parameter_norms_after: dict[str, float] = {}

        for name, parameters in self.groups.items():
            deltas: list[torch.Tensor] = []
            for parameter in parameters:
                before = snapshots.get(id(parameter))
                if before is None:
                    continue
                deltas.append(parameter.detach() - before.to(parameter.device))
            update_norms[name] = _l2_norm_tensors(deltas)
            parameter_norms_after[name] = _parameter_norm(parameters)

        for name in self.GROUP_NAMES:
            update_norm = update_norms[name]
            parameter_norm_pre = float(self._active[f"parameter_{name}_norm_pre"])
            self._active[f"parameter_{name}_norm_post"] = parameter_norms_after[name]
            self._active[f"update_{name}_norm"] = update_norm
            self._active[f"update_{name}_to_weight_ratio"] = _safe_ratio(
                update_norm, parameter_norm_pre
            )
            branch_lr = self._active.get(f"learning_rate_{name}")
            self._active[f"update_{name}_per_lr"] = (
                None
                if branch_lr is None
                else _safe_ratio(update_norm, float(branch_lr))
            )

        self._active["update_actor_to_critic_ratio"] = _safe_ratio(
            update_norms["actor"], update_norms["critic"]
        )
        self._active["update_actor_to_weight_per_preclip_grad"] = _safe_ratio(
            self._active.get("update_actor_to_weight_ratio") or 0.0,
            self._active.get("grad_actor_pre_clip") or 0.0,
        )
        self._active["update_critic_to_weight_per_preclip_grad"] = _safe_ratio(
            self._active.get("update_critic_to_weight_ratio") or 0.0,
            self._active.get("grad_critic_pre_clip") or 0.0,
        )

        result = dict(self._active)
        self._active = None
        return result
