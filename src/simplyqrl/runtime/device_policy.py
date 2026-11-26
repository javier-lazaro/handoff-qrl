"""Device policy resolution for SimplyQRL.

SimplyQRL exposes one runtime placement decision to users:

* ``torch_device``: where the PyTorch/PPO side of the pipeline lives
  (rollout buffers, CNN/MLP extractors, classical heads, optimisers).

Quantum simulator selection is intentionally not part of ``device_policy``. It
belongs to the quantum-core config (``quantum_core.kwargs.qml_device`` and
``quantum_core.kwargs.diff_method``).

Hybrid agents may still need an internal CPU/GPU boundary between classical
PyTorch modules and a PennyLane/TorchLayer QNode. That boundary is selected
automatically from the quantum backend and reported as metadata, rather than
being exposed as a user-facing config option.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class DevicePolicy:
    """Resolved runtime placement policy for the main PyTorch pipeline."""

    requested_torch_device: str
    torch_device: torch.device
    allow_cuda_fallback: bool
    cuda_available: bool
    cuda_device_count: int
    cuda_device_index: int | None
    cuda_device_name: str | None
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_torch_device": self.requested_torch_device,
            "torch_device": str(self.torch_device),
            "torch_device_type": self.torch_device.type,
            "allow_cuda_fallback": self.allow_cuda_fallback,
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "cuda_device_index": self.cuda_device_index,
            "cuda_device_name": self.cuda_device_name,
            "fallback_reason": self.fallback_reason,
        }


def _cuda_metadata(device: torch.device | None = None) -> tuple[int | None, str | None]:
    if not torch.cuda.is_available():
        return None, None

    index = device.index if device is not None and device.index is not None else 0
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device index {index} is out of range for "
            f"cuda_device_count={torch.cuda.device_count()}."
        )
    return int(index), torch.cuda.get_device_name(index)


def _resolve_device(
    requested: str,
    *,
    role: str,
    allow_cuda_fallback: bool,
    auto_uses_cuda: bool,
) -> tuple[torch.device, str | None]:
    requested = str(requested or "cpu").strip().lower()
    fallback_reason = None

    if requested == "auto":
        if auto_uses_cuda and torch.cuda.is_available():
            return torch.device("cuda:0"), None
        return torch.device("cpu"), (
            f"{role}=auto resolved to cpu because cuda is unavailable."
            if auto_uses_cuda and not torch.cuda.is_available()
            else None
        )

    try:
        device = torch.device(requested)
    except Exception as exc:
        raise ValueError(f"Invalid {role}: {requested!r}.") from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            if allow_cuda_fallback:
                return torch.device("cpu"), f"{role}={requested!r} fell back to cpu: cuda unavailable."
            raise RuntimeError(
                f"{role}={requested!r} was requested, but torch.cuda.is_available() is False. "
                "Use 'cpu', 'auto', or set allow_cuda_fallback=true."
            )

        index = device.index if device.index is not None else 0
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(
                f"{role}={requested!r} refers to CUDA index {index}, but "
                f"cuda_device_count={torch.cuda.device_count()}."
            )
        return torch.device(f"cuda:{index}"), None

    if device.type != "cpu":
        raise ValueError(
            f"Unsupported {role}: {requested!r}. SimplyQRL currently supports 'cpu', "
            "'cuda[:index]' and 'auto'."
        )

    return device, fallback_reason


def resolve_device_policy(config: Mapping[str, Any] | None = None) -> DevicePolicy:
    """Resolve a user/runtime device policy mapping.

    Supported config keys:

    ``torch_device``
        Main PyTorch placement. Default: ``"cpu"``. ``"auto"`` uses CUDA when
        available.
    ``allow_cuda_fallback``
        If true, explicit CUDA requests fall back to CPU when CUDA is missing.
        Default: false.

    Quantum backend selection belongs in ``quantum_core.kwargs.qml_device`` and
    ``quantum_core.kwargs.diff_method``. The Torch/PennyLane boundary placement
    for hybrid heads is inferred internally and reported as metadata.
    """

    cfg = dict(config or {})
    supported = {"torch_device", "allow_cuda_fallback"}
    unknown = sorted(set(cfg) - supported)
    if unknown:
        raise KeyError(
            f"Unknown device_policy keys: {unknown}. Supported keys: {sorted(supported)}. "
            "Quantum backend selection belongs in quantum_core.kwargs.qml_device and "
            "quantum_core.kwargs.diff_method."
        )

    requested_torch = str(cfg.get("torch_device", "cpu"))
    allow_cuda_fallback = bool(cfg.get("allow_cuda_fallback", False))

    torch_device, fallback_reason = _resolve_device(
        requested_torch,
        role="device_policy.torch_device",
        allow_cuda_fallback=allow_cuda_fallback,
        auto_uses_cuda=True,
    )

    cuda_index, cuda_name = _cuda_metadata(torch_device if torch_device.type == "cuda" else None)

    return DevicePolicy(
        requested_torch_device=requested_torch,
        torch_device=torch_device,
        allow_cuda_fallback=allow_cuda_fallback,
        cuda_available=bool(torch.cuda.is_available()),
        cuda_device_count=int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        cuda_device_index=cuda_index if torch_device.type == "cuda" else None,
        cuda_device_name=cuda_name if torch_device.type == "cuda" else None,
        fallback_reason=fallback_reason,
    )
