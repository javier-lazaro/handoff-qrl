"""Policy/value heads for SimplyQRL 1.5.

A head receives a latent feature vector and produces either:
- action logits for the actor, or
- a scalar value for the critic.

Hybrid heads make the input/quantum/readout structure explicit:

    features -> input_adapter -> VQC -> observables -> post-processing -> inference NN -> output
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from simplyqrl.instrumentation import QNodeStats, TimedQuantumLayer
from simplyqrl.models.postprocessing import BasePostProcessor, build_postprocessing_stack
from simplyqrl.quantum import build_quantum_core


def layer_init(layer: nn.Linear, std: float = 2**0.5, bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal layer initialization used by CleanRL PPO."""

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def make_activation(activation: str | type[nn.Module] | nn.Module) -> nn.Module:
    """Build activation module from string, class or instance."""

    if isinstance(activation, nn.Module):
        return activation

    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation()

    if isinstance(activation, str):
        name = activation.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "tanh":
            return nn.Tanh()
        if name == "gelu":
            return nn.GELU()
        if name == "elu":
            return nn.ELU()

    raise ValueError(f"Unsupported activation: {activation!r}")



def normalize_input_adapter_config(
    config: Mapping[str, Any] | None,
    *,
    input_dim: int,
) -> dict[str, Any]:
    """Validate and normalize the optional head-local input adapter.

    The SimplyQRL staged-policy contract distinguishes the global feature
    extractor from a head-local adapter.  The paper protocol needs the canonical
    visual use case: a trainable ``Linear(D32, D8)`` immediately before the
    VQC.  The adapter remains classical and is therefore outside QNode timing.
    """

    input_dim = int(input_dim)
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}.")

    cfg = dict(config or {})
    adapter_type = str(cfg.get("type", "identity")).strip().lower()

    if adapter_type == "identity":
        unknown = set(cfg) - {"type", "output_dim"}
        if unknown:
            raise ValueError(
                "Unsupported identity input_adapter key(s): "
                + ", ".join(sorted(unknown))
            )
        if "output_dim" in cfg and int(cfg["output_dim"]) != input_dim:
            raise ValueError(
                "identity input_adapter.output_dim must match input_dim: "
                f"output_dim={int(cfg['output_dim'])}, input_dim={input_dim}"
            )
        return {"type": "identity", "output_dim": input_dim}

    if adapter_type == "linear":
        unknown = set(cfg) - {"type", "output_dim", "activation", "bias"}
        if unknown:
            raise ValueError(
                "Unsupported linear input_adapter key(s): "
                + ", ".join(sorted(unknown))
            )
        if "output_dim" not in cfg:
            raise KeyError("linear input_adapter requires output_dim")
        output_dim = int(cfg["output_dim"])
        if output_dim <= 0:
            raise ValueError("input_adapter.output_dim must be positive")
        activation = str(cfg.get("activation", "identity")).strip().lower()
        if activation not in {"identity", "relu", "tanh", "gelu", "elu"}:
            raise ValueError(f"Unsupported input_adapter activation: {activation!r}")
        return {
            "type": "linear",
            "output_dim": output_dim,
            "activation": activation,
            "bias": bool(cfg.get("bias", True)),
        }

    raise ValueError(
        f"Unsupported input_adapter type: {adapter_type!r}. "
        "Supported types: identity, linear."
    )


def build_input_adapter(
    config: Mapping[str, Any] | None,
    *,
    input_dim: int,
) -> tuple[nn.Module, int, dict[str, Any]]:
    """Build the canonical head-local input adapter.

    Returns ``(module, output_dim, normalized_config)`` so the quantum core can
    be constructed against the adapted dimensionality rather than the global
    feature width.
    """

    normalized = normalize_input_adapter_config(config, input_dim=input_dim)
    adapter_type = normalized["type"]

    if adapter_type == "identity":
        return nn.Identity(), int(input_dim), normalized

    linear = nn.Linear(
        int(input_dim),
        int(normalized["output_dim"]),
        bias=bool(normalized["bias"]),
    )
    torch.nn.init.orthogonal_(linear.weight, 2**0.5)
    if linear.bias is not None:
        torch.nn.init.constant_(linear.bias, 0.0)
    activation_name = str(normalized["activation"])
    if activation_name == "identity":
        module: nn.Module = linear
    else:
        module = nn.Sequential(linear, make_activation(activation_name))
    return module, int(normalized["output_dim"]), normalized


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: Sequence[int] = (),
    activation: str | type[nn.Module] | nn.Module = "tanh",
    final_std: float = 1.0,
) -> nn.Sequential:
    """Build a simple MLP with CleanRL-style initialization."""

    layers: list[nn.Module] = []
    previous_dim = int(input_dim)

    for hidden_dim in hidden_sizes:
        layers.append(layer_init(nn.Linear(previous_dim, int(hidden_dim))))
        layers.append(make_activation(activation))
        previous_dim = int(hidden_dim)

    layers.append(layer_init(nn.Linear(previous_dim, int(output_dim)), std=final_std))
    return nn.Sequential(*layers)


class ClassicalHead(nn.Module):
    """Purely classical MLP head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Sequence[int] = (),
        activation: str | type[nn.Module] | nn.Module = "tanh",
        final_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)

        self.network = build_mlp(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_sizes=self.hidden_sizes,
            activation=activation,
            final_std=final_std,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features.float())


class HybridHead(nn.Module):
    """Hybrid input-adapter + quantum-core + readout head.

    Canonical contract:

        features -> input_adapter -> quantum_core -> quantum_features
                 -> postprocessing -> inference NN -> output

    ``quantum_core`` is a serializable config pointing either to a built-in
    preset or to an external importable factory. HybridHead does not branch on
    specific circuit names; those live in the quantum-core registry.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        quantum_core: dict[str, Any],
        input_adapter: dict[str, Any] | None = None,
        hidden_sizes: Sequence[int] = (),
        activation: str | type[nn.Module] | nn.Module = "tanh",
        postprocessing: dict[str, Any] | None = None,
        final_std: float = 1.0,
        qnode_stats: QNodeStats | None = None,
        qnode_scope: str | None = None,
    ) -> None:
        super().__init__()

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        (
            self.input_adapter,
            self.quantum_input_dim,
            self.input_adapter_config,
        ) = build_input_adapter(input_adapter, input_dim=self.input_dim)

        core_build = build_quantum_core(quantum_core, input_dim=self.quantum_input_dim)
        self.quantum_core_config = core_build.config
        self.q_output_dim = int(core_build.q_output_dim)
        self.q_layer = core_build.layer
        self.qnode_wrapper_torch_device = torch.device("cpu")
        self.qnode_wrapper_placement_reason = "default_cpu_qnode_wrapper"

        if qnode_stats is not None:
            self.q_layer = TimedQuantumLayer(
                self.q_layer,
                stats=qnode_stats,
                scope=qnode_scope or "quantum",
            )

        post_cfg = postprocessing or {}
        self.postprocessing: BasePostProcessor = build_postprocessing_stack(
            input_dim=self.q_output_dim,
            reuse_repetitions=int(post_cfg.get("reuse_repetitions", 1)),
            use_output_scaling=bool(post_cfg.get("use_output_scaling", False)),
            output_scaling_trainable=bool(post_cfg.get("output_scaling_trainable", True)),
            output_scaling_per_feature=bool(post_cfg.get("output_scaling_per_feature", False)),
        )

        self.inference_nn = build_mlp(
            input_dim=self.postprocessing.output_dim,
            output_dim=self.output_dim,
            hidden_sizes=tuple(int(size) for size in hidden_sizes),
            activation=activation,
            final_std=final_std,
        )

    def set_qnode_wrapper_torch_device(
        self,
        device: str | torch.device,
        *,
        reason: str = "auto",
    ) -> None:
        """Place the QNode Torch wrapper module on SimplyQRL's internal device.

        The wrapper device is intentionally not a user-facing device-policy
        option. Users choose the main ``torch_device`` and the PennyLane
        backend (``qml_device``/``diff_method``); SimplyQRL chooses the
        Torch device used by the QNode wrapper to avoid invalid CPU/GPU mixes
        at the TorchLayer/PennyLane boundary.
        """

        self.qnode_wrapper_torch_device = torch.device(device)
        self.qnode_wrapper_placement_reason = str(reason)
        self.q_layer.to(self.qnode_wrapper_torch_device)

    def quantum_backend_name(self) -> str:
        """Return the configured PennyLane backend name when known."""

        kwargs = self.quantum_core_config.get("kwargs", {})
        backend = kwargs.get("qml_device")
        if backend is None:
            if "factory" in self.quantum_core_config:
                return "factory_default"
            return "default.qubit"
        return str(backend).strip().lower()

    def quantum_diff_method(self) -> str:
        """Return the configured PennyLane differentiation method when known."""

        kwargs = self.quantum_core_config.get("kwargs", {})
        diff_method = kwargs.get("diff_method")
        if diff_method is None:
            if "factory" in self.quantum_core_config:
                return "factory_default"
            return "backprop"
        return str(diff_method).strip().lower()

    def _classical_torch_device(self, fallback: torch.device) -> torch.device:
        for module in (self.input_adapter, self.inference_nn, self.postprocessing):
            for parameter in module.parameters(recurse=True):
                return parameter.device
            for buffer in module.buffers(recurse=True):
                return buffer.device
        return fallback

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        classical_device = self._classical_torch_device(features.device)

        adapter_inputs = features.float().to(classical_device)
        adapted = self.input_adapter(adapter_inputs)
        q_inputs = adapted.to(self.qnode_wrapper_torch_device)
        q_out = self.q_layer(q_inputs)

        if q_out.device != classical_device:
            q_out = q_out.to(classical_device)

        processed = self.postprocessing(q_out)
        return self.inference_nn(processed)
