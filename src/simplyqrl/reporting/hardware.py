"""Hardware profile utilities for SimplyQRL experiments.

A hardware profile is generated once per campaign and referenced by each run via
`hardware_profile_id`. This avoids duplicating machine specs in every run while
keeping experiments traceable.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from simplyqrl.reporting.context import get_git_commit, write_json


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except Exception:
        return None


def _run_command(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _collect_os_info() -> dict[str, Any]:
    """Collect OS information in a cross-platform-friendly format."""

    release_info: dict[str, str] = {}

    if hasattr(platform, "freedesktop_os_release"):
        try:
            release_info = dict(platform.freedesktop_os_release())
        except Exception:
            release_info = {}

    return {
        "system": platform.system(),
        "platform": platform.platform(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "os_name": release_info.get("PRETTY_NAME") or platform.platform(),
        "os_id": release_info.get("ID"),
        "os_version_id": release_info.get("VERSION_ID"),
        "os_codename": (
            release_info.get("VERSION_CODENAME")
            or release_info.get("UBUNTU_CODENAME")
        ),
    }


def _collect_cpu_info() -> dict[str, Any]:
    cpu_info: dict[str, Any] = {
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }

    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()
        cpu_info["brand_raw"] = info.get("brand_raw")
        cpu_info["arch"] = info.get("arch")
    except Exception as exc:
        cpu_info["cpuinfo_error"] = repr(exc)

    return cpu_info


def _collect_memory_info() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "total_bytes": int(vm.total),
        "total_gb": round(vm.total / (1024**3), 3),
    }


def _collect_torch_gpu_info() -> list[dict[str, Any]]:
    try:
        import torch

        if not torch.cuda.is_available():
            return []

        gpus: list[dict[str, Any]] = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gpus.append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "total_memory_bytes": int(props.total_memory),
                    "total_memory_gb": round(props.total_memory / (1024**3), 3),
                    "capability": list(props.major_minor) if hasattr(props, "major_minor") else [
                        props.major,
                        props.minor,
                    ],
                    "multi_processor_count": int(props.multi_processor_count),
                }
            )
        return gpus

    except Exception as exc:
        return [{"error": repr(exc)}]


def _collect_nvidia_smi_info() -> dict[str, Any]:
    driver_version = _run_command(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    cuda_version = _run_command(
        ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"]
    )
    gpu_rows = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,power.limit",
            "--format=csv,noheader,nounits",
        ]
    )

    parsed_gpus: list[dict[str, Any]] = []
    if gpu_rows:
        for row in gpu_rows.splitlines():
            parts = [part.strip() for part in row.split(",")]
            if len(parts) >= 4:
                parsed_gpus.append(
                    {
                        "index": parts[0],
                        "name": parts[1],
                        "memory_total_mib": parts[2],
                        "power_limit_w": parts[3],
                    }
                )

    return {
        "driver_version": driver_version.splitlines()[0] if driver_version else None,
        "cuda_version_reported_by_driver": cuda_version.splitlines()[0] if cuda_version else None,
        "gpus": parsed_gpus,
    }


def _stable_hardware_profile_id(profile: dict[str, Any]) -> str:
    """Compute a stable profile id excluding volatile timestamps."""

    stable = dict(profile)
    stable.pop("generated_at", None)

    payload = json.dumps(stable, sort_keys=True, default=repr, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def collect_hardware_profile(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Collect a hardware/software profile for the current machine."""

    profile: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": _utc_now_iso(),
        "hostname": socket.gethostname(),
        "os": _collect_os_info(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "cpu": _collect_cpu_info(),
        "memory": _collect_memory_info(),
        "gpu": {
            "torch": _collect_torch_gpu_info(),
            "nvidia_smi": _collect_nvidia_smi_info(),
        },
        "software": {
            "simplyqrl": _safe_version("simplyqrl"),
            "torch": _safe_version("torch"),
            "gymnasium": _safe_version("gymnasium"),
            "pennylane": _safe_version("pennylane"),
            "pennylane-lightning": _safe_version("pennylane-lightning"),
            "codecarbon": _safe_version("codecarbon"),
            "numpy": _safe_version("numpy"),
            "python": platform.python_version(),
        },
        "git": {
            "commit": get_git_commit(repo_root),
            "repo_root": str(repo_root) if repo_root is not None else None,
        },
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
        },
    }

    profile["hardware_profile_id"] = _stable_hardware_profile_id(profile)
    return profile


def hardware_profile_path(output_dir: str | Path, experiment_name: str | None = None) -> Path:
    base = Path(output_dir)
    if experiment_name:
        base = base / experiment_name
    return base / "hardware_profile.json"


def ensure_hardware_profile(
    *,
    output_dir: str | Path,
    experiment_name: str | None = None,
    repo_root: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[str, Path, dict[str, Any]]:
    """Create or load the campaign-level hardware profile.

    Returns:
        (hardware_profile_id, path, profile)
    """

    path = hardware_profile_path(output_dir, experiment_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        with path.open("r", encoding="utf-8") as f:
            profile = json.load(f)

        profile_id = profile.get("hardware_profile_id")
        if not profile_id:
            profile_id = _stable_hardware_profile_id(profile)
            profile["hardware_profile_id"] = profile_id
            write_json(path, profile)

        return str(profile_id), path, profile

    profile = collect_hardware_profile(repo_root=repo_root)
    write_json(path, profile)

    return str(profile["hardware_profile_id"]), path, profile
