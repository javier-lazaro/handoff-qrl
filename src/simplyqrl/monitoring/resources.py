"""Resource monitor for SimplyQRL algorithms.

This monitor captures lightweight process/system resource metrics. CodeCarbon is
intentionally not handled here yet; it will later be added as an optional probe.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import psutil
import torch

from simplyqrl.monitoring.base import BaseMonitor
from simplyqrl.reporting.logger import ArtifactLogger


def _bytes_to_mb(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024**2)


class ResourceMonitor(BaseMonitor):
    """Monitor lightweight CPU/RAM/CUDA memory resource metrics."""

    def __init__(
        self,
        logger: ArtifactLogger,
        *,
        sample_every_updates: int = 1,
        include_system: bool = True,
        include_process: bool = True,
        include_torch_cuda: bool = True,
        torch_cuda_device_index: int | None = None,
        use_codecarbon: bool = False,
        codecarbon_output_file: str = "codecarbon.csv",
        codecarbon_project_name: str | None = None,
        codecarbon_measure_power_secs: int = 1,
        codecarbon_log_level: str = "error",
        fail_on_codecarbon_error: bool = False,
    ) -> None:
        if sample_every_updates <= 0:
            raise ValueError("sample_every_updates must be > 0.")

        self.logger = logger
        self.sample_every_updates = int(sample_every_updates)
        self.include_system = bool(include_system)
        self.include_process = bool(include_process)
        self.include_torch_cuda = bool(include_torch_cuda)
        self.torch_cuda_device_index = torch_cuda_device_index

        self.use_codecarbon = bool(use_codecarbon)
        self.codecarbon_output_file = str(codecarbon_output_file)
        self.codecarbon_project_name = codecarbon_project_name or logger.context.run_name
        self.codecarbon_measure_power_secs = int(codecarbon_measure_power_secs)
        self.codecarbon_log_level = str(codecarbon_log_level)
        self.fail_on_codecarbon_error = bool(fail_on_codecarbon_error)

        self.codecarbon_tracker = None
        self.codecarbon_status = "disabled"
        self.codecarbon_error: str | None = None
        self.codecarbon_emissions_kg: float | None = None
        self.codecarbon_output_path = self.logger.run_dir / self.codecarbon_output_file

        self.process = psutil.Process(os.getpid())

        self.num_resource_samples = 0
        self.process_rss_mb_peak = 0.0
        self.process_vms_mb_peak = 0.0
        self.cuda_memory_allocated_mb_peak = 0.0
        self.cuda_memory_reserved_mb_peak = 0.0

        # Prime psutil CPU counters.
        try:
            psutil.cpu_percent(interval=None)
            self.process.cpu_percent(interval=None)
        except Exception:
            pass

    def _build_codecarbon_tracker(self):
        from codecarbon import EmissionsTracker

        kwargs = {
            "project_name": self.codecarbon_project_name,
            "output_dir": str(self.logger.run_dir),
            "output_file": self.codecarbon_output_file,
            "save_to_file": True,
            "log_level": self.codecarbon_log_level,
            "measure_power_secs": self.codecarbon_measure_power_secs,
        }

        try:
            return EmissionsTracker(**kwargs)
        except TypeError:
            # Compatibility fallback for older/newer CodeCarbon signatures.
            fallback_kwargs = {
                "project_name": self.codecarbon_project_name,
                "output_dir": str(self.logger.run_dir),
                "output_file": self.codecarbon_output_file,
                "log_level": self.codecarbon_log_level,
            }
            return EmissionsTracker(**fallback_kwargs)

    def _start_codecarbon(self) -> None:
        if not self.use_codecarbon:
            return

        try:
            self.codecarbon_tracker = self._build_codecarbon_tracker()
            self.codecarbon_tracker.start()
            self.codecarbon_status = "running"
        except Exception as exc:
            self.codecarbon_status = "error"
            self.codecarbon_error = repr(exc)
            if self.fail_on_codecarbon_error:
                raise

    def _read_codecarbon_last_row(self) -> dict[str, Any]:
        path = Path(self.codecarbon_output_path)
        if not path.exists():
            return {}

        try:
            with path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            return rows[-1] if rows else {}
        except Exception as exc:
            return {"codecarbon_read_error": repr(exc)}

    def _stop_codecarbon(self) -> dict[str, Any]:
        if not self.use_codecarbon:
            return {
                "codecarbon_enabled": False,
                "codecarbon_tracker_status": "disabled",
            }

        if self.codecarbon_tracker is None:
            return {
                "codecarbon_enabled": True,
                "codecarbon_tracker_status": self.codecarbon_status,
                "codecarbon_error": self.codecarbon_error,
                "codecarbon_output_file": str(self.codecarbon_output_path),
            }

        emissions_kg = None
        try:
            emissions_kg = self.codecarbon_tracker.stop()
            self.codecarbon_status = "stopped"
            self.codecarbon_emissions_kg = (
                float(emissions_kg) if emissions_kg is not None else None
            )
        except Exception as exc:
            self.codecarbon_status = "error"
            self.codecarbon_error = repr(exc)
            if self.fail_on_codecarbon_error:
                raise

        row = self._read_codecarbon_last_row()

        def _float_or_none(value):
            try:
                return float(value)
            except Exception:
                return None

        energy_kwh = (
            _float_or_none(row.get("energy_consumed"))
            or _float_or_none(row.get("energy_kwh"))
        )
        duration_seconds = _float_or_none(row.get("duration"))
        emissions_from_csv = _float_or_none(row.get("emissions"))

        return {
            "codecarbon_enabled": True,
            "codecarbon_tracker_status": self.codecarbon_status,
            "codecarbon_error": self.codecarbon_error,
            "codecarbon_output_file": str(self.codecarbon_output_path),
            "codecarbon_emissions_kg": (
                self.codecarbon_emissions_kg
                if self.codecarbon_emissions_kg is not None
                else emissions_from_csv
            ),
            "codecarbon_energy_kwh": energy_kwh,
            "codecarbon_duration_seconds": duration_seconds,
        }

    def _sample_process(self) -> dict[str, Any]:
        if not self.include_process:
            return {}

        memory = self.process.memory_info()

        process_rss_mb = _bytes_to_mb(memory.rss) or 0.0
        process_vms_mb = _bytes_to_mb(memory.vms) or 0.0

        self.process_rss_mb_peak = max(self.process_rss_mb_peak, process_rss_mb)
        self.process_vms_mb_peak = max(self.process_vms_mb_peak, process_vms_mb)

        try:
            process_cpu_percent = float(self.process.cpu_percent(interval=None))
        except Exception:
            process_cpu_percent = None

        return {
            "process_rss_mb": process_rss_mb,
            "process_vms_mb": process_vms_mb,
            "process_cpu_percent": process_cpu_percent,
            "process_rss_mb_peak": self.process_rss_mb_peak,
            "process_vms_mb_peak": self.process_vms_mb_peak,
        }

    def _sample_system(self) -> dict[str, Any]:
        if not self.include_system:
            return {}

        vm = psutil.virtual_memory()

        try:
            system_cpu_percent = float(psutil.cpu_percent(interval=None))
        except Exception:
            system_cpu_percent = None

        return {
            "system_cpu_percent": system_cpu_percent,
            "system_ram_total_mb": _bytes_to_mb(vm.total),
            "system_ram_used_mb": _bytes_to_mb(vm.used),
            "system_ram_available_mb": _bytes_to_mb(vm.available),
            "system_ram_percent": float(vm.percent),
        }

    def _sample_torch_cuda(self) -> dict[str, Any]:
        if not self.include_torch_cuda:
            return {}

        if not torch.cuda.is_available():
            return {
                "torch_cuda_available": False,
            }

        device_index = (
            int(self.torch_cuda_device_index)
            if self.torch_cuda_device_index is not None
            else torch.cuda.current_device()
        )

        allocated_mb = _bytes_to_mb(torch.cuda.memory_allocated(device_index)) or 0.0
        reserved_mb = _bytes_to_mb(torch.cuda.memory_reserved(device_index)) or 0.0
        max_allocated_mb = _bytes_to_mb(torch.cuda.max_memory_allocated(device_index)) or 0.0
        max_reserved_mb = _bytes_to_mb(torch.cuda.max_memory_reserved(device_index)) or 0.0

        self.cuda_memory_allocated_mb_peak = max(
            self.cuda_memory_allocated_mb_peak,
            allocated_mb,
            max_allocated_mb,
        )
        self.cuda_memory_reserved_mb_peak = max(
            self.cuda_memory_reserved_mb_peak,
            reserved_mb,
            max_reserved_mb,
        )

        free_mb = None
        total_mb = None
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            free_mb = _bytes_to_mb(free_bytes)
            total_mb = _bytes_to_mb(total_bytes)
        except Exception:
            pass

        return {
            "torch_cuda_available": True,
            "cuda_device_index": int(device_index),
            "cuda_device_name": torch.cuda.get_device_name(device_index),
            "cuda_memory_allocated_mb": allocated_mb,
            "cuda_memory_reserved_mb": reserved_mb,
            "cuda_memory_max_allocated_mb": max_allocated_mb,
            "cuda_memory_max_reserved_mb": max_reserved_mb,
            "cuda_memory_allocated_mb_peak": self.cuda_memory_allocated_mb_peak,
            "cuda_memory_reserved_mb_peak": self.cuda_memory_reserved_mb_peak,
            "cuda_memory_free_mb": free_mb,
            "cuda_memory_total_mb": total_mb,
        }

    def sample(self) -> dict[str, Any]:
        sample = {}
        sample.update(self._sample_process())
        sample.update(self._sample_system())
        sample.update(self._sample_torch_cuda())
        return sample

    def on_training_start(self, state: dict[str, Any]) -> None:
        self._start_codecarbon()

        sample = self.sample()
        payload = {
            "global_step": int(state.get("global_step", 0) or 0),
            "phase": "training_start",
            **sample,
            "codecarbon_enabled": self.use_codecarbon,
            "codecarbon_tracker_status": self.codecarbon_status,
            "codecarbon_error": self.codecarbon_error,
            "codecarbon_output_file": (
                str(self.codecarbon_output_path) if self.use_codecarbon else None
            ),
        }

        self.logger.log_event("resources", payload)
        self.logger.update_summary(
            {
                "resources_enabled": True,
                **payload,
            }
        )

    def on_update_end(self, state: dict[str, Any]) -> None:
        update = int(state.get("update", 0) or 0)

        if update % self.sample_every_updates != 0:
            return

        self.num_resource_samples += 1
        sample = self.sample()

        payload = {
            "global_step": int(state.get("global_step", 0) or 0),
            "update": update,
            "phase": "update_end",
            "num_resource_samples": self.num_resource_samples,
            **sample,
        }

        self.logger.log_event("resources", payload)
        self.logger.update_summary(
            {
                "num_resource_samples": self.num_resource_samples,
                **sample,
            }
        )

    def on_training_end(self, state: dict[str, Any]) -> None:
        codecarbon_summary = self._stop_codecarbon()
        sample = self.sample()

        payload = {
            "global_step": int(state.get("global_step", 0) or 0),
            "phase": "training_end",
            **sample,
            **codecarbon_summary,
        }

        self.logger.log_event("resources", payload)
        self.logger.update_summary(
            {
                "num_resource_samples": self.num_resource_samples,
                **sample,
                **codecarbon_summary,
            }
        )

    def on_training_error(self, state: dict[str, Any]) -> None:
        codecarbon_summary = self._stop_codecarbon()
        self.logger.log_event(
            "resources",
            {
                "global_step": int(state.get("global_step", 0) or 0),
                "phase": "training_error",
                **codecarbon_summary,
            },
        )
        self.logger.update_summary(codecarbon_summary)
