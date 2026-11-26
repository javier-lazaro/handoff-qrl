"""Save/load utilities for staged SimplyQRL head artifacts.

A head artifact is the canonical way to move a KD-trained actor/critic head
between workflows:

    offline KD trainer -> student_head.pt -> composed policy / plug-back smoke

The module stores the canonical `head_config` contract needed to reconstruct
classical and hybrid heads, including external quantum-core factories.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from simplyqrl.instrumentation import QNodeStats
from simplyqrl.models.heads import (
    ClassicalHead,
    HybridHead,
    normalize_input_adapter_config,
)

HEAD_ARTIFACT_FORMAT = "simplyqrl_head_artifact"
HEAD_ARTIFACT_VERSION = 1

_CLASSICAL_ALIASES = {"classical", "classical_head", "mlp"}
_HYBRID_ALIASES = {"hybrid", "hybrid_head", "vqc"}
_SUPPORTED_ROLES = {"actor", "critic", "unknown"}


def _as_plain_dict(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config)


def _as_int_tuple(values: Any) -> tuple[int, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        stripped = values.strip()
        if not stripped:
            return ()
        return tuple(int(part.strip()) for part in stripped.split(",") if part.strip())
    if isinstance(values, Sequence):
        return tuple(int(value) for value in values)
    raise TypeError(f"hidden_sizes must be a sequence, string or None, got {type(values)!r}")


def _infer_family(config: Mapping[str, Any]) -> str:
    raw = (
        config.get("family")
        or config.get("type")
        or config.get("head_type")
        or config.get("student_type")
    )
    if raw is None:
        raise KeyError("Head config requires one of: family, type, head_type, student_type")

    family = str(raw).lower()
    if family in _CLASSICAL_ALIASES:
        return "classical"
    if family in _HYBRID_ALIASES:
        return "hybrid"
    raise ValueError(f"Unsupported head family/type: {raw!r}")


def normalize_head_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical, serializable head config.

    Accepted inputs must follow the canonical SimplyQRL 1.5 head contract.
    """

    cfg = _as_plain_dict(config)
    family = _infer_family(cfg)

    if "input_dim" not in cfg:
        raise KeyError("Head config requires input_dim")

    output_dim = cfg.get("output_dim", cfg.get("n_actions"))
    if output_dim is None:
        raise KeyError("Head config requires output_dim or n_actions")

    role = str(cfg.get("role", "actor")).lower()
    if role not in _SUPPORTED_ROLES:
        raise ValueError(f"Unsupported head role: {role!r}")

    normalized: dict[str, Any] = {
        "family": family,
        "type": family,
        "role": role,
        "input_dim": int(cfg["input_dim"]),
        "output_dim": int(output_dim),
        "hidden_sizes": list(_as_int_tuple(cfg.get("hidden_sizes", ()))),
        "activation": cfg.get("activation", "tanh"),
        "final_std": float(cfg.get("final_std", 0.01 if role == "actor" else 1.0)),
    }

    if family == "hybrid":
        postprocessing = dict(cfg.get("postprocessing", {}))

        if "quantum_core" not in cfg:
            raise KeyError("Hybrid HeadArtifact config requires a canonical quantum_core block.")

        hybrid_fields: dict[str, Any] = {
            "quantum_core": dict(cfg["quantum_core"]),
            "postprocessing": postprocessing,
        }
        if "input_adapter" in cfg:
            hybrid_fields["input_adapter"] = normalize_input_adapter_config(
                cfg["input_adapter"],
                input_dim=int(normalized["input_dim"]),
            )
        normalized.update(hybrid_fields)

    return normalized


def build_head_from_config(
    config: Mapping[str, Any],
    *,
    qnode_stats: QNodeStats | None = None,
    qnode_scope: str | None = None,
) -> nn.Module:
    """Build a SimplyQRL head from canonical config."""

    cfg = normalize_head_config(config)
    family = cfg["family"]

    if family == "classical":
        return ClassicalHead(
            input_dim=int(cfg["input_dim"]),
            output_dim=int(cfg["output_dim"]),
            hidden_sizes=tuple(cfg.get("hidden_sizes", ())),
            activation=cfg.get("activation", "tanh"),
            final_std=float(cfg.get("final_std", 1.0)),
        )

    if family == "hybrid":
        return HybridHead(
            input_dim=int(cfg["input_dim"]),
            output_dim=int(cfg["output_dim"]),
            quantum_core=dict(cfg["quantum_core"]),
            input_adapter=cfg.get("input_adapter"),
            hidden_sizes=tuple(cfg.get("hidden_sizes", ())),
            activation=cfg.get("activation", "tanh"),
            postprocessing=dict(cfg.get("postprocessing", {})),
            final_std=float(cfg.get("final_std", 1.0)),
            qnode_stats=qnode_stats,
            qnode_scope=qnode_scope,
        )

    raise ValueError(f"Unsupported head family: {family!r}")



def _canonical_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return a wrapper-independent state dict for head artifacts.

    TimedQuantumLayer is instrumentation, not model structure. Artifacts should
    therefore store `q_layer.*` keys even if the in-memory head used an
    instrumented `q_layer.layer.*` wrapper during training.
    """

    canonical: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        out_key = key
        if key.startswith("q_layer.layer."):
            out_key = "q_layer." + key[len("q_layer.layer.") :]
        canonical[out_key] = value
    return canonical


def _adapt_state_dict_for_head(
    head: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Adapt wrapper-independent q_layer keys to the target head instance."""

    target_keys = set(head.state_dict().keys())
    adapted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key in target_keys:
            adapted[key] = value
            continue

        # Canonical artifact -> instrumented in-memory head.
        if key.startswith("q_layer.") and not key.startswith("q_layer.layer."):
            wrapped_key = "q_layer.layer." + key[len("q_layer.") :]
            if wrapped_key in target_keys:
                adapted[wrapped_key] = value
                continue

        # Legacy instrumented artifact -> uninstrumented in-memory head.
        if key.startswith("q_layer.layer."):
            unwrapped_key = "q_layer." + key[len("q_layer.layer.") :]
            if unwrapped_key in target_keys:
                adapted[unwrapped_key] = value
                continue

        adapted[key] = value

    return adapted

def _load_payload(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Head artifact must be a dict payload, got {type(payload)!r}")
    return payload


def _extract_head_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw_config = payload.get("head_config")
    if raw_config is None:
        raise KeyError("Head artifact requires head_config")
    if not isinstance(raw_config, Mapping):
        raise TypeError(f"head_config must be a mapping, got {type(raw_config)!r}")
    return normalize_head_config(raw_config)


def save_head_artifact(
    head: nn.Module,
    config: Mapping[str, Any],
    path: str | Path,
    *,
    best_state_dict: Mapping[str, torch.Tensor] | None = None,
    metadata: Mapping[str, Any] | None = None,
    extra_payload: Mapping[str, Any] | None = None,
) -> Path:
    """Save a head and its canonical config as a reusable artifact."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    head_config = normalize_head_config(config)
    payload: dict[str, Any] = {
        "format": HEAD_ARTIFACT_FORMAT,
        "version": HEAD_ARTIFACT_VERSION,
        "head_config": head_config,
        "model_state_dict": _canonical_state_dict(head.state_dict()),
        "metadata": dict(metadata or {}),
    }

    if best_state_dict is not None:
        payload["best_model_state_dict"] = _canonical_state_dict(best_state_dict)
    if extra_payload is not None:
        payload.update(dict(extra_payload))

    torch.save(payload, out_path)
    return out_path


def load_head_artifact(
    path: str | Path,
    *,
    use_best: bool = True,
    map_location: str | torch.device = "cpu",
    qnode_stats: QNodeStats | None = None,
    qnode_scope: str | None = None,
    strict: bool = True,
) -> nn.Module:
    """Load a head artifact and return the reconstructed head module.

    The artifact must contain canonical `head_config` and `model_state_dict`.
    """

    payload = _load_payload(path, map_location=map_location)
    head_config = _extract_head_config(payload)
    head = build_head_from_config(
        head_config,
        qnode_stats=qnode_stats,
        qnode_scope=qnode_scope,
    )

    state_key = (
        "best_model_state_dict"
        if use_best and "best_model_state_dict" in payload
        else "model_state_dict"
    )
    if state_key not in payload:
        raise KeyError(f"Head artifact does not contain {state_key!r}")

    state_dict = _adapt_state_dict_for_head(head, payload[state_key])
    head.load_state_dict(state_dict, strict=strict)
    return head


def load_head_config(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load only the canonical head config from an artifact."""

    payload = _load_payload(path, map_location=map_location)
    return _extract_head_config(payload)
