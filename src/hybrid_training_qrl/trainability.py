"""Paper-side trainability diagnostics for Q100 calibration.

The monitor plugs into SimplyQRL's public monitor hooks without changing the
library.  It samples gradient groups and forward activations during PPO updates
and writes compact, audit-friendly artifacts next to the normal run outputs.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from simplyqrl.monitoring.base import BaseMonitor

TRAINABILITY_UPDATES = "trainability_updates.csv"
TRAINABILITY_SUMMARY = "trainability_summary.json"


class _ScalarAccumulator:
    def __init__(self) -> None:
        self.values: list[float] = []

    def add(self, value: Any) -> None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(parsed):
            self.values.append(parsed)

    def mean(self) -> float | None:
        return None if not self.values else float(sum(self.values) / len(self.values))

    def minimum(self) -> float | None:
        return None if not self.values else float(min(self.values))

    def maximum(self) -> float | None:
        return None if not self.values else float(max(self.values))

    def zero_fraction(self, threshold: float = 1e-10) -> float | None:
        if not self.values:
            return None
        return float(sum(abs(value) <= threshold for value in self.values) / len(self.values))


class _TensorAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.minimum: float | None = None
        self.maximum: float | None = None

    def add(self, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return
        values = tensor.detach().float().reshape(-1).cpu()
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            return
        self.count += int(finite.numel())
        self.sum += float(finite.sum())
        self.sum_sq += float((finite * finite).sum())
        current_min = float(finite.min())
        current_max = float(finite.max())
        self.minimum = current_min if self.minimum is None else min(self.minimum, current_min)
        self.maximum = current_max if self.maximum is None else max(self.maximum, current_max)

    def mean(self) -> float | None:
        return None if self.count == 0 else self.sum / self.count

    def std(self) -> float | None:
        if self.count == 0:
            return None
        mean = self.sum / self.count
        variance = max(self.sum_sq / self.count - mean * mean, 0.0)
        return math.sqrt(variance)

    def max_abs(self) -> float | None:
        if self.minimum is None or self.maximum is None:
            return None
        return max(abs(self.minimum), abs(self.maximum))


def _first_tensor(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(output, dict):
        for item in output.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _unique_parameters(modules: Iterable[torch.nn.Module | None]) -> list[torch.nn.Parameter]:
    parameters: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters(recurse=True):
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            parameters.append(parameter)
    return parameters


def _gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float | None:
    squared = 0.0
    found = False
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        finite = gradient.detach().float()
        if not torch.isfinite(finite).all():
            continue
        squared += float(torch.sum(finite * finite))
        found = True
    return math.sqrt(squared) if found else None


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _last(values: list[float]) -> float | None:
    return None if not values else float(values[-1])


class TrainabilityMonitor(BaseMonitor):
    """Low-overhead diagnostics for a SimplyQRL PPO model.

    Sampling is deliberately sparse because hybrid PPO invokes actor and critic
    QNodes many times per update.  The monitor observes only optimizer-update
    forwards, not environment rollouts or evaluation episodes.
    """

    def __init__(
        self,
        *,
        model: Any,
        sample_every_updates: int = 1,
        forward_sample_every_calls: int = 32,
        backward_sample_every_calls: int = 32,
    ) -> None:
        if model.artifact_logger is None:
            raise RuntimeError("TrainabilityMonitor requires SimplyQRL reporting.")

        self.model = model
        self.agent = model.agent
        self.logger = model.artifact_logger
        self.run_dir = Path(self.logger.run_dir)
        self.sample_every_updates = max(int(sample_every_updates), 1)
        self.forward_sample_every_calls = max(int(forward_sample_every_calls), 1)
        self.backward_sample_every_calls = max(int(backward_sample_every_calls), 1)

        actor = getattr(self.agent, "actor_head", None)
        critic = getattr(self.agent, "critic_head", None)
        extractor = getattr(self.agent, "feature_extractor", None)

        self.parameter_groups = {
            "actor_q": _unique_parameters([getattr(actor, "q_layer", None)]),
            "actor_classical": _unique_parameters(
                [getattr(actor, "postprocessing", None), getattr(actor, "inference_nn", None)]
            ),
            "critic_q": _unique_parameters([getattr(critic, "q_layer", None)]),
            "critic_classical": _unique_parameters(
                [getattr(critic, "postprocessing", None), getattr(critic, "inference_nn", None)]
            ),
            "extractor": _unique_parameters([extractor]),
        }

        self.n_actions = int(getattr(self.agent, "n_actions", 0) or 0)
        self.rows: list[dict[str, Any]] = []
        self._hooks: list[Any] = []
        self._active = False
        self._forward_calls: dict[str, int] = {}
        self._backward_calls = 0
        self._gradients: dict[str, _ScalarAccumulator] = {}
        self._tensors: dict[str, _TensorAccumulator] = {}
        self._action_prob_sum: torch.Tensor | None = None
        self._action_prob_count = 0
        self._action_max_prob = _ScalarAccumulator()

        self._register_hook(getattr(actor, "q_layer", None), "actor_q_output")
        self._register_hook(getattr(critic, "q_layer", None), "critic_q_output")
        self._register_hook(getattr(actor, "inference_nn", None), "actor_logits")
        self._register_hook(getattr(critic, "inference_nn", None), "critic_values")

    def _register_hook(self, module: torch.nn.Module | None, name: str) -> None:
        if module is None:
            return

        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not self._active:
                return
            calls = self._forward_calls.get(name, 0) + 1
            self._forward_calls[name] = calls
            if calls % self.forward_sample_every_calls != 0:
                return
            tensor = _first_tensor(output)
            if tensor is None:
                return
            self._tensors[name].add(tensor)
            if name == "actor_logits" and tensor.ndim >= 2:
                probabilities = torch.softmax(tensor.detach().float(), dim=-1).cpu()
                vector = probabilities.reshape(-1, probabilities.shape[-1]).sum(dim=0)
                if self._action_prob_sum is None:
                    self._action_prob_sum = torch.zeros_like(vector)
                self._action_prob_sum += vector
                self._action_prob_count += int(probabilities.reshape(-1, probabilities.shape[-1]).shape[0])
                for value in probabilities.max(dim=-1).values.reshape(-1).tolist():
                    self._action_max_prob.add(value)

        self._hooks.append(module.register_forward_hook(hook))

    def _reset_update(self) -> None:
        self._forward_calls = {}
        self._backward_calls = 0
        self._gradients = {name: _ScalarAccumulator() for name in self.parameter_groups}
        self._tensors = {
            name: _TensorAccumulator()
            for name in ("actor_q_output", "critic_q_output", "actor_logits", "critic_values")
        }
        self._action_prob_sum = None
        self._action_prob_count = 0
        self._action_max_prob = _ScalarAccumulator()

    def on_rollout_start(self, state: dict[str, Any]) -> None:
        pass

    def on_episode_end(self, state: dict[str, Any]) -> None:
        pass

    def on_rollout_end(self, state: dict[str, Any]) -> None:
        pass

    def on_evaluation_end(self, state: dict[str, Any]) -> None:
        pass

    def on_training_start(self, state: dict[str, Any]) -> None:
        self.logger.update_summary(
            {
                "trainability_diagnostics_enabled": True,
                "trainability_sample_every_updates": self.sample_every_updates,
                "trainability_forward_sample_every_calls": self.forward_sample_every_calls,
                "trainability_backward_sample_every_calls": self.backward_sample_every_calls,
            }
        )

    def on_update_start(self, state: dict[str, Any]) -> None:
        update = int(state.get("update", 0))
        self._active = update % self.sample_every_updates == 0
        if self._active:
            self._reset_update()

    def on_backward_end(self, state: dict[str, Any]) -> None:
        if not self._active:
            return
        self._backward_calls += 1
        if self._backward_calls % self.backward_sample_every_calls != 0:
            return
        for name, parameters in self.parameter_groups.items():
            self._gradients[name].add(_gradient_norm(parameters))

    def on_update_end(self, state: dict[str, Any]) -> None:
        if not self._active:
            return

        row: dict[str, Any] = {
            "global_step": int(state.get("global_step", 0)),
            "update": int(state.get("update", 0)),
            "loss_policy": state.get("loss_policy"),
            "loss_value": state.get("loss_value"),
            "loss_entropy": state.get("loss_entropy"),
            "loss_total": state.get("loss_total"),
            "approx_kl": state.get("approx_kl"),
            "clipfrac": state.get("clipfrac"),
            "explained_variance": state.get("explained_variance"),
            "learning_rate": state.get("learning_rate"),
            "learning_rate_mode": state.get("learning_rate_mode"),
            "learning_rate_actor": state.get("learning_rate_actor"),
            "learning_rate_critic": state.get("learning_rate_critic"),
            "learning_rate_shared": state.get("learning_rate_shared"),
            "grad_norm_total": state.get("grad_norm"),
            "optimizer_steps": state.get("optimizer_steps"),
            "actor_action_max_prob_mean": self._action_max_prob.mean(),
        }

        for name, accumulator in self._gradients.items():
            row[f"grad_{name}_mean"] = accumulator.mean()
            row[f"grad_{name}_min"] = accumulator.minimum()
            row[f"grad_{name}_max"] = accumulator.maximum()
            row[f"grad_{name}_zero_fraction"] = accumulator.zero_fraction()

        for name, accumulator in self._tensors.items():
            row[f"{name}_mean"] = accumulator.mean()
            row[f"{name}_std"] = accumulator.std()
            row[f"{name}_max_abs"] = accumulator.max_abs()

        if self._action_prob_sum is not None and self._action_prob_count > 0:
            action_probabilities = self._action_prob_sum / self._action_prob_count
            row["actor_action_prob_mean"] = json.dumps(
                [float(value) for value in action_probabilities.tolist()]
            )
        else:
            row["actor_action_prob_mean"] = None

        self.rows.append(row)
        self._append_csv(row)
        self.logger.log_event("trainability_update", row)
        self._active = False

    def _fieldnames(self) -> list[str]:
        fields = [
            "global_step",
            "update",
            "loss_policy",
            "loss_value",
            "loss_entropy",
            "loss_total",
            "approx_kl",
            "clipfrac",
            "explained_variance",
            "learning_rate",
            "learning_rate_mode",
            "learning_rate_actor",
            "learning_rate_critic",
            "learning_rate_shared",
            "grad_norm_total",
            "optimizer_steps",
        ]
        for name in self.parameter_groups:
            fields.extend(
                [
                    f"grad_{name}_mean",
                    f"grad_{name}_min",
                    f"grad_{name}_max",
                    f"grad_{name}_zero_fraction",
                ]
            )
        for name in ("actor_q_output", "critic_q_output", "actor_logits", "critic_values"):
            fields.extend([f"{name}_mean", f"{name}_std", f"{name}_max_abs"])
        fields.extend(["actor_action_max_prob_mean", "actor_action_prob_mean"])
        return fields

    def _append_csv(self, row: dict[str, Any]) -> None:
        path = self.run_dir / TRAINABILITY_UPDATES
        write_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames(), extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow({field: row.get(field) for field in self._fieldnames()})

    def _series(self, field: str) -> list[float]:
        values: list[float] = []
        for row in self.rows:
            value = row.get(field)
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(parsed):
                values.append(parsed)
        return values

    def _build_summary(self, *, status: str, error: str | None = None) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "status": status,
            "error": error,
            "sampled_updates": len(self.rows),
            "sample_every_updates": self.sample_every_updates,
            "forward_sample_every_calls": self.forward_sample_every_calls,
            "backward_sample_every_calls": self.backward_sample_every_calls,
            "updates_path": str(self.run_dir / TRAINABILITY_UPDATES),
        }

        fields = [
            "loss_policy",
            "loss_value",
            "loss_entropy",
            "loss_total",
            "approx_kl",
            "clipfrac",
            "explained_variance",
            "grad_norm_total",
            "grad_actor_q_mean",
            "grad_actor_classical_mean",
            "grad_critic_q_mean",
            "grad_critic_classical_mean",
            "grad_extractor_mean",
            "grad_actor_q_zero_fraction",
            "grad_critic_q_zero_fraction",
            "actor_q_output_std",
            "critic_q_output_std",
            "actor_logits_std",
            "critic_values_std",
            "actor_action_max_prob_mean",
        ]
        for field in fields:
            values = self._series(field)
            summary[f"{field}_mean"] = _mean(values)
            summary[f"{field}_final"] = _last(values)
            summary[f"{field}_min"] = min(values) if values else None
            summary[f"{field}_max"] = max(values) if values else None

        if self.rows:
            summary["actor_action_prob_mean_final"] = self.rows[-1].get(
                "actor_action_prob_mean"
            )
        return summary

    def _write_summary(self, *, status: str, error: str | None = None) -> None:
        summary = self._build_summary(status=status, error=error)
        path = self.run_dir / TRAINABILITY_SUMMARY
        with path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        self.logger.update_summary(
            {
                "trainability_diagnostics_status": status,
                "trainability_sampled_updates": len(self.rows),
                "trainability_summary_path": str(path),
            }
        )

    def _remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def on_training_end(self, state: dict[str, Any]) -> None:
        self._write_summary(status="PASS")
        self._remove_hooks()

    def on_training_error(self, state: dict[str, Any]) -> None:
        self._write_summary(status="FAIL", error=str(state.get("error")))
        self._remove_hooks()
