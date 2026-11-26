"""Quantum-core registry and factory contract.

A quantum core is any torch ``nn.Module`` satisfying:

    input:  Tensor[batch, input_dim]
    output: Tensor[batch, q_output_dim]

It can be provided either as a built-in SimplyQRL preset by name or as an
external importable factory path.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
from torch import nn

from simplyqrl.quantum.presets import BUILTIN_QUANTUM_CORE_PRESETS


QuantumCoreFactory = Callable[..., nn.Module]
QuantumCoreSpec = Mapping[str, Any]


@dataclass(frozen=True)
class QuantumCoreBuild:
    layer: nn.Module
    q_output_dim: int
    config: dict[str, Any]


def import_from_string(path: str) -> Any:
    """Import ``module.submodule:attribute`` or ``module.submodule.attribute``."""

    if not isinstance(path, str) or not path.strip():
        raise ValueError("Import path must be a non-empty string.")

    path = path.strip()
    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    else:
        module_name, _, attr_name = path.rpartition(".")

    if not module_name or not attr_name:
        raise ValueError(
            f"Invalid import path {path!r}. Use 'package.module:object' or "
            "'package.module.object'."
        )

    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportError(f"Object {attr_name!r} not found in module {module_name!r}.") from exc


def normalize_quantum_core_config(config: QuantumCoreSpec) -> dict[str, Any]:
    """Validate and normalize a quantum-core config block."""

    if not isinstance(config, Mapping):
        raise TypeError("quantum_core must be a mapping/dict.")

    cfg = dict(config)
    has_name = bool(cfg.get("name"))
    has_factory = bool(cfg.get("factory"))

    if has_name == has_factory:
        raise ValueError("quantum_core must define exactly one of: 'name' or 'factory'.")

    if "q_output_dim" not in cfg:
        raise KeyError("quantum_core requires q_output_dim.")

    q_output_dim = int(cfg["q_output_dim"])
    if q_output_dim <= 0:
        raise ValueError("quantum_core.q_output_dim must be positive.")

    kwargs = dict(cfg.get("kwargs", {}))
    if not isinstance(kwargs, dict):
        raise TypeError("quantum_core.kwargs must be a dict.")

    normalized: dict[str, Any] = {
        "q_output_dim": q_output_dim,
        "kwargs": kwargs,
    }

    if has_name:
        name = str(cfg["name"])
        if name not in BUILTIN_QUANTUM_CORE_PRESETS:
            available = ", ".join(sorted(BUILTIN_QUANTUM_CORE_PRESETS))
            raise KeyError(f"Unknown built-in quantum_core preset {name!r}. Available: {available}")
        normalized["name"] = name
    else:
        factory = str(cfg["factory"])
        normalized["factory"] = factory

    return normalized


def _resolve_factory(config: Mapping[str, Any]) -> QuantumCoreFactory:
    if "name" in config:
        return BUILTIN_QUANTUM_CORE_PRESETS[str(config["name"])]

    factory = import_from_string(str(config["factory"]))
    if not callable(factory):
        raise TypeError(f"quantum_core.factory {config['factory']!r} is not callable.")
    return factory


def build_quantum_core(config: QuantumCoreSpec, *, input_dim: int) -> QuantumCoreBuild:
    """Build a quantum core from a canonical config.

    The selected factory is called with:

        factory(input_dim=input_dim, **quantum_core.kwargs)

    and must return a ``torch.nn.Module``.
    """

    normalized = normalize_quantum_core_config(config)
    factory = _resolve_factory(normalized)
    kwargs = dict(normalized["kwargs"])

    layer = factory(input_dim=int(input_dim), **kwargs)
    if not isinstance(layer, nn.Module):
        raise TypeError(
            "quantum_core factory must return torch.nn.Module, "
            f"got {type(layer).__name__}."
        )

    return QuantumCoreBuild(
        layer=layer,
        q_output_dim=int(normalized["q_output_dim"]),
        config=normalized,
    )


def validate_quantum_core_forward(
    layer: nn.Module,
    *,
    input_dim: int,
    q_output_dim: int,
    batch_size: int = 2,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Run a lightweight shape check for a quantum core.

    This is intentionally separate from ``build_quantum_core`` because a QNode
    forward can be relatively expensive and some callers may not want validation
    during construction.
    """

    x = torch.zeros((int(batch_size), int(input_dim)), dtype=dtype)
    with torch.no_grad():
        y = layer(x)

    if not isinstance(y, torch.Tensor):
        raise TypeError(f"quantum_core forward must return torch.Tensor, got {type(y).__name__}.")

    expected = (int(batch_size), int(q_output_dim))
    if tuple(y.shape) != expected:
        raise ValueError(
            f"quantum_core forward shape mismatch: expected {expected}, got {tuple(y.shape)}."
        )

    return {
        "input_shape": list(x.shape),
        "output_shape": list(y.shape),
        "q_output_dim": int(q_output_dim),
    }
