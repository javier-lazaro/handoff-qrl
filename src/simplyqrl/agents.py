"""Public agent factory for the SimplyQRL 1.5 modular PPO pipeline.

All supported agents use the same architecture contract:

    observation -> feature extractor -> actor/critic heads

Canonical agent types are ``v15_classical`` and ``v15_hybrid``. Shorter names
from earlier development configs remain explicit compatibility aliases; they do
not select separate legacy implementations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from simplyqrl.models.actor_critic import build_actor_critic_agent

CANONICAL_CLASSICAL_AGENT_TYPE = "v15_classical"
CANONICAL_HYBRID_AGENT_TYPE = "v15_hybrid"

AGENT_TYPE_ALIASES = {
    "v15_classical": CANONICAL_CLASSICAL_AGENT_TYPE,
    "modular_classical": CANONICAL_CLASSICAL_AGENT_TYPE,
    "classical": CANONICAL_CLASSICAL_AGENT_TYPE,
    "mlp": CANONICAL_CLASSICAL_AGENT_TYPE,
    "v15_hybrid": CANONICAL_HYBRID_AGENT_TYPE,
    "modular_hybrid": CANONICAL_HYBRID_AGENT_TYPE,
    "hybrid": CANONICAL_HYBRID_AGENT_TYPE,
}


def canonical_agent_type(agent_type: str) -> str:
    """Resolve a public agent type or compatibility alias to its canonical name."""

    normalized = str(agent_type).strip().lower()
    try:
        return AGENT_TYPE_ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(AGENT_TYPE_ALIASES))
        raise ValueError(
            f"Unsupported agent type: {agent_type!r}. Supported types: {supported}"
        ) from exc


def _normalise_classical_config(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(config or {})
    cfg.setdefault("head_type", "classical")
    cfg.setdefault("extractor", {"type": "identity"})
    cfg.setdefault("net_arch", {"pi": [64], "vf": [64]})
    cfg.setdefault("activation", "tanh")
    return cfg


def _default_quantum_core(n_qubits: int) -> dict[str, Any]:
    return {
        "name": "skolik_rx_cz_reuploading",
        "q_output_dim": int(n_qubits),
        "kwargs": {
            "n_qubits": int(n_qubits),
            "n_layers_q": 1,
            "ent": False,
        },
    }


def _normalise_hybrid_config(
    obs_shape: tuple[int, ...],
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    cfg = dict(config or {})
    cfg.setdefault("head_type", "hybrid")
    cfg.setdefault("extractor", {"type": "identity"})
    cfg.setdefault("net_arch", {"pi": [], "vf": []})
    cfg.setdefault("activation", "tanh")

    if "head" not in cfg:
        n_qubits = int(np.prod(obs_shape))
        cfg["head"] = {
            "type": "hybrid",
            "quantum_core": _default_quantum_core(n_qubits),
            "postprocessing": {
                "reuse_repetitions": 1,
                "use_output_scaling": False,
            },
            "hidden_sizes": [],
            "activation": cfg.get("activation", "tanh"),
        }

    return cfg


def build_agent(
    type: str,
    obs_shape: tuple[int, ...],
    n_actions: int,
    config: dict[str, Any] | None = None,
    is_qnet: bool = False,
):
    """Build a SimplyQRL PPO actor-critic agent.

    ``type`` and ``is_qnet`` retain the original factory signature for external
    callers. Q-network agents are intentionally outside the SimplyQRL 1.5 scope.
    """

    if is_qnet:
        raise NotImplementedError(
            "SimplyQRL 1.5 supports PPO actor-critic agents only; "
            "DQN/Q-network agents are outside the maintained 1.5 pipeline."
        )

    canonical_type = canonical_agent_type(type)
    obs_shape = tuple(int(dim) for dim in obs_shape)

    if canonical_type == CANONICAL_CLASSICAL_AGENT_TYPE:
        normalized_config = _normalise_classical_config(config)
    else:
        normalized_config = _normalise_hybrid_config(obs_shape, config)

    return build_actor_critic_agent(
        obs_shape=obs_shape,
        n_actions=int(n_actions),
        config=normalized_config,
    )


__all__ = [
    "AGENT_TYPE_ALIASES",
    "CANONICAL_CLASSICAL_AGENT_TYPE",
    "CANONICAL_HYBRID_AGENT_TYPE",
    "build_agent",
    "canonical_agent_type",
]
