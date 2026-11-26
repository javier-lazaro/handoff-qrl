"""Run context utilities for SimplyQRL reporting.

This module defines the identity and filesystem layout of a run. It does not
collect metrics; it only answers: where does this run live, what is its id, and
what config produced it?
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """Convert common Python objects to JSON/YAML-safe values."""

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    # Callables, transformers, custom env factories, etc.
    return repr(value)


def stable_config_hash(config: dict[str, Any]) -> str:
    safe_config = _json_safe(config)
    payload = json.dumps(safe_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def get_git_commit(repo_root: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


@dataclass(frozen=True)
class RunContext:
    run_id: str
    run_name: str
    output_dir: Path
    run_dir: Path
    config_hash: str
    created_at: str
    experiment_name: str | None = None
    regime: str | None = None
    hardware_profile_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        data["run_dir"] = str(self.run_dir)
        return data


def build_run_context(
    *,
    config: dict[str, Any],
    output_dir: str | Path = "runs",
    run_name: str | None = None,
    experiment_name: str | None = None,
    regime: str | None = None,
    hardware_profile_id: str | None = None,
) -> RunContext:
    output_path = Path(output_dir)
    created_at = utc_now_iso()
    config_hash = stable_config_hash(config)

    safe_run_name = run_name or str(config.get("run_name") or "simplyqrl_run")
    safe_run_name = safe_run_name.replace("/", "_").replace(" ", "_")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{safe_run_name}_{timestamp}_{config_hash}"

    if experiment_name:
        run_dir = output_path / experiment_name / run_id
    else:
        run_dir = output_path / run_id

    return RunContext(
        run_id=run_id,
        run_name=safe_run_name,
        output_dir=output_path,
        run_dir=run_dir,
        config_hash=config_hash,
        created_at=created_at,
        experiment_name=experiment_name,
        regime=regime,
        hardware_profile_id=hardware_profile_id,
    )


def build_basic_runtime_metadata(repo_root: str | Path | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "git_commit": get_git_commit(repo_root),
    }

    try:
        import torch

        metadata["torch_version"] = torch.__version__
        metadata["torch_cuda_available"] = bool(torch.cuda.is_available())
        metadata["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            metadata["torch_cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        metadata["torch_error"] = repr(exc)

    try:
        import gymnasium as gym

        metadata["gymnasium_version"] = gym.__version__
    except Exception as exc:
        metadata["gymnasium_error"] = repr(exc)

    try:
        import pennylane as qml

        metadata["pennylane_version"] = qml.__version__
    except Exception as exc:
        metadata["pennylane_error"] = repr(exc)

    return metadata


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(_json_safe(data), f, sort_keys=False, allow_unicode=True)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, indent=2, sort_keys=True)
        f.write("\n")
