"""Persistent reporting for PPO actor-critic optimization diagnostics."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from simplyqrl.monitoring.base import BaseMonitor
from simplyqrl.reporting.logger import ArtifactLogger

OPTIMIZATION_DIAGNOSTICS_STEPS = "optimization_diagnostics_steps.csv"
OPTIMIZATION_DIAGNOSTICS_SUMMARY = "optimization_diagnostics_summary.json"


class OptimizationDiagnosticsMonitor(BaseMonitor):
    """Write sparse optimizer-step diagnostics and a compact final summary."""

    def __init__(
        self,
        logger: ArtifactLogger,
        *,
        sample_every_optimizer_steps: int,
    ) -> None:
        self.logger = logger
        self.run_dir = Path(logger.run_dir)
        self.sample_every_optimizer_steps = int(sample_every_optimizer_steps)
        self.rows: list[dict[str, Any]] = []
        self.steps_path = self.run_dir / OPTIMIZATION_DIAGNOSTICS_STEPS
        self.summary_path = self.run_dir / OPTIMIZATION_DIAGNOSTICS_SUMMARY

    def on_training_start(self, state: dict[str, Any]) -> None:
        self.logger.update_summary(
            {
                "optimization_diagnostics_enabled": True,
                "optimization_diagnostics_sample_every_optimizer_steps": (
                    self.sample_every_optimizer_steps
                ),
            }
        )

    def on_optimizer_step_end(self, state: dict[str, Any]) -> None:
        row = {
            key: value
            for key, value in state.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        self.rows.append(row)
        self._append_csv(row)
        self.logger.log_event("optimization_diagnostics_step", row)

    def on_training_end(self, state: dict[str, Any]) -> None:
        self._write_summary(status="PASS")

    def on_training_error(self, state: dict[str, Any]) -> None:
        self._write_summary(status="FAIL", error=str(state.get("error")))

    def _fieldnames(self) -> list[str]:
        return [
            "optimizer_step",
            "global_step",
            "update",
            "epoch",
            "minibatch",
            "learning_rate",
            "learning_rate_mode",
            "learning_rate_shared",
            "learning_rate_actor",
            "learning_rate_critic",
            "gradient_clipping_mode",
            "max_grad_norm",
            "max_grad_norm_shared",
            "max_grad_norm_actor",
            "max_grad_norm_critic",
            "loss_policy_minibatch",
            "loss_value_minibatch",
            "loss_entropy_minibatch",
            "loss_total_minibatch",
            "grad_global_pre_clip",
            "clip_returned_global_norm",
            "grad_global_pre_clip_consistency_abs_error",
            "clip_returned_shared_norm",
            "clip_returned_actor_norm",
            "clip_returned_critic_norm",
            "grad_shared_pre_clip_consistency_abs_error",
            "grad_actor_pre_clip_consistency_abs_error",
            "grad_critic_pre_clip_consistency_abs_error",
            "grad_global_post_clip",
            "clip_coefficient_global",
            "clip_coefficient_shared",
            "clip_coefficient_actor",
            "clip_coefficient_critic",
            "clip_active",
            "clip_active_shared",
            "clip_active_actor",
            "clip_active_critic",
            "parameter_shared_count",
            "parameter_actor_count",
            "parameter_critic_count",
            "grad_shared_pre_clip",
            "grad_actor_pre_clip",
            "grad_critic_pre_clip",
            "grad_shared_post_clip",
            "grad_actor_post_clip",
            "grad_critic_post_clip",
            "grad_actor_to_critic_ratio_pre_clip",
            "grad_actor_to_critic_ratio_post_clip",
            "grad_shared_share_pre_clip",
            "grad_actor_share_pre_clip",
            "grad_critic_share_pre_clip",
            "parameter_shared_norm_pre",
            "parameter_actor_norm_pre",
            "parameter_critic_norm_pre",
            "parameter_shared_norm_post",
            "parameter_actor_norm_post",
            "parameter_critic_norm_post",
            "update_shared_norm",
            "update_actor_norm",
            "update_critic_norm",
            "update_shared_to_weight_ratio",
            "update_actor_to_weight_ratio",
            "update_critic_to_weight_ratio",
            "update_shared_per_lr",
            "update_actor_per_lr",
            "update_critic_per_lr",
            "update_actor_to_critic_ratio",
            "update_actor_to_weight_per_preclip_grad",
            "update_critic_to_weight_per_preclip_grad",
        ]

    def _append_csv(self, row: dict[str, Any]) -> None:
        write_header = not self.steps_path.exists()
        with self.steps_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=self._fieldnames(),
                extrasaction="ignore",
            )
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

    @staticmethod
    def _mean(values: list[float]) -> float | None:
        return None if not values else float(sum(values) / len(values))

    @staticmethod
    def _last(values: list[float]) -> float | None:
        return None if not values else float(values[-1])

    @staticmethod
    def _minimum(values: list[float]) -> float | None:
        return None if not values else float(min(values))

    @staticmethod
    def _maximum(values: list[float]) -> float | None:
        return None if not values else float(max(values))

    def _build_summary(self, *, status: str, error: str | None = None) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "status": status,
            "error": error,
            "sampled_optimizer_steps": len(self.rows),
            "sample_every_optimizer_steps": self.sample_every_optimizer_steps,
            "steps_path": str(self.steps_path),
        }

        for field in (
            "clip_active",
            "clip_active_shared",
            "clip_active_actor",
            "clip_active_critic",
        ):
            values = self._series(field)
            summary[f"{field}_fraction"] = self._mean(values)

        modes = [
            str(row.get("gradient_clipping_mode"))
            for row in self.rows
            if row.get("gradient_clipping_mode") not in (None, "")
        ]
        summary["gradient_clipping_mode"] = modes[-1] if modes else None

        lr_modes = [
            str(row.get("learning_rate_mode"))
            for row in self.rows
            if row.get("learning_rate_mode") not in (None, "")
        ]
        summary["learning_rate_mode"] = lr_modes[-1] if lr_modes else None

        fields = [
            "learning_rate_shared",
            "learning_rate_actor",
            "learning_rate_critic",
            "grad_global_pre_clip",
            "clip_returned_global_norm",
            "grad_global_pre_clip_consistency_abs_error",
            "clip_returned_shared_norm",
            "clip_returned_actor_norm",
            "clip_returned_critic_norm",
            "grad_shared_pre_clip_consistency_abs_error",
            "grad_actor_pre_clip_consistency_abs_error",
            "grad_critic_pre_clip_consistency_abs_error",
            "grad_global_post_clip",
            "clip_coefficient_global",
            "clip_coefficient_shared",
            "clip_coefficient_actor",
            "clip_coefficient_critic",
            "grad_actor_pre_clip",
            "grad_critic_pre_clip",
            "grad_shared_pre_clip",
            "grad_actor_post_clip",
            "grad_critic_post_clip",
            "grad_shared_post_clip",
            "grad_actor_to_critic_ratio_pre_clip",
            "grad_actor_share_pre_clip",
            "grad_critic_share_pre_clip",
            "update_actor_norm",
            "update_critic_norm",
            "update_shared_norm",
            "update_actor_to_weight_ratio",
            "update_critic_to_weight_ratio",
            "update_shared_to_weight_ratio",
            "update_actor_to_critic_ratio",
            "update_actor_to_weight_per_preclip_grad",
            "update_critic_to_weight_per_preclip_grad",
        ]
        for field in fields:
            values = self._series(field)
            summary[f"{field}_mean"] = self._mean(values)
            summary[f"{field}_final"] = self._last(values)
            summary[f"{field}_min"] = self._minimum(values)
            summary[f"{field}_max"] = self._maximum(values)

        return summary

    def _write_summary(self, *, status: str, error: str | None = None) -> None:
        payload = self._build_summary(status=status, error=error)
        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        self.logger.update_summary(
            {
                "optimization_diagnostics_status": status,
                "optimization_diagnostics_sampled_optimizer_steps": len(self.rows),
                "optimization_diagnostics_summary_path": str(self.summary_path),
            }
        )
