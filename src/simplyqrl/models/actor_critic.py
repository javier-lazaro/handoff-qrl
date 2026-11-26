"""Actor-critic agent built from SimplyQRL 1.5 components."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.distributions.categorical import Categorical

from simplyqrl.instrumentation import QNodeStats
from simplyqrl.models.extractors import CNNExtractor, IdentityExtractor, MLPExtractor
from simplyqrl.models.heads import ClassicalHead, HybridHead
from simplyqrl.utils.params import ParameterBreakdown, count_trainable_params



def _reject_unknown_keys(
    config: dict[str, Any],
    *,
    allowed: set[str],
    context: str,
    suggestions: dict[str, str] | None = None,
) -> None:
    """Fail fast on config keys that would otherwise be silently ignored.

    Staged visual workflows depend on stable extractor/head contracts.
    A typo such as ``output_dim`` instead of ``features_dim`` can change the
    actual architecture without raising an error, so the v1.5 builders validate
    their local config surface explicitly.
    """

    suggestions = suggestions or {}
    unknown = sorted(set(config) - allowed)
    if not unknown:
        return

    details = []
    for key in unknown:
        if key in suggestions:
            details.append(f"{key!r} (did you mean {suggestions[key]!r}?)")
        else:
            details.append(repr(key))

    raise ValueError(
        f"Unsupported {context} config key(s): {', '.join(details)}. "
        f"Allowed keys: {sorted(allowed)}."
    )


def _validate_extractor_config(config: dict[str, Any], extractor_type: str) -> None:
    common = {"type"}
    suggestions = {
        "output_dim": "features_dim",
        "channels_last": "tensor_format",
    }

    if extractor_type == "identity":
        _reject_unknown_keys(
            config,
            allowed=common,
            context="identity extractor",
            suggestions=suggestions,
        )
        return

    if extractor_type == "mlp":
        _reject_unknown_keys(
            config,
            allowed=common | {"features_dim", "hidden_sizes", "activation"},
            context="mlp extractor",
            suggestions=suggestions,
        )
        return

    if extractor_type == "cnn":
        _reject_unknown_keys(
            config,
            allowed=common | {"features_dim", "tensor_format", "normalize_images"},
            context="cnn extractor",
            suggestions=suggestions,
        )
        return


def _validate_head_config(config: dict[str, Any], head_type: str) -> None:
    common = {"type", "activation", "hidden_sizes", "final_std"}

    if head_type == "classical":
        _reject_unknown_keys(
            config,
            allowed=common,
            context="classical head",
        )
        return

    if head_type == "hybrid":
        _reject_unknown_keys(
            config,
            allowed=common | {"input_adapter", "quantum_core", "postprocessing"},
            context="hybrid head",
            suggestions={
                "circ_type": "quantum_core",
                "n_qubits": "quantum_core.kwargs.n_qubits",
                "n_layers_q": "quantum_core.kwargs.n_layers_q",
                "ent": "quantum_core.kwargs.ent",
            },
        )
        if "quantum_core" not in config:
            raise KeyError("Hybrid head config requires a canonical quantum_core block.")
        return

def _module_parameter_devices(module: nn.Module) -> list[str]:
    devices: set[str] = set()
    for parameter in module.parameters(recurse=True):
        devices.add(str(parameter.device))
    for buffer in module.buffers(recurse=True):
        devices.add(str(buffer.device))
    return sorted(devices)


_GPU_NATIVE_QML_BACKENDS = {
    "lightning.gpu",
}


def _select_qnode_wrapper_torch_device(
    *,
    torch_device: torch.device,
    qml_device: str,
) -> tuple[torch.device, str]:
    """Select the Torch device used by the QNode wrapper for hybrid heads.

    Users choose the main PyTorch runtime device (``device_policy.torch_device``)
    and the PennyLane backend (``quantum_core.kwargs.qml_device``). SimplyQRL
    chooses the Torch device used by the QNode wrapper. CPU-backed PennyLane
    backends are forced to a CPU wrapper even when the classical PyTorch
    runtime runs on CUDA; known GPU-native QML backends can keep the wrapper on
    the main CUDA device.
    """

    qml_device_normalised = str(qml_device or "factory_default").strip().lower()

    if torch_device.type != "cuda":
        return torch.device("cpu"), "main_torch_device_cpu__qnode_wrapper_cpu"

    if qml_device_normalised in _GPU_NATIVE_QML_BACKENDS:
        return torch_device, f"qml_backend_gpu_native__qnode_wrapper_matches_torch:{qml_device_normalised}"

    return torch.device("cpu"), f"qml_backend_cpu_backed__qnode_wrapper_forced_cpu:{qml_device_normalised}"


def _normalise_net_arch(net_arch: Any) -> tuple[list[int], list[int]]:
    if net_arch is None:
        return [64], [64]

    if isinstance(net_arch, dict):
        return list(net_arch.get("pi", [])), list(net_arch.get("vf", []))

    if isinstance(net_arch, list):
        return list(net_arch), list(net_arch)

    raise ValueError("net_arch must be a dict, list or None.")


def build_feature_extractor(
    observation_shape: tuple[int, ...],
    config: dict[str, Any] | None,
) -> nn.Module:
    cfg = dict(config or {})
    extractor_type = cfg.get("type", "identity").lower()
    _validate_extractor_config(cfg, extractor_type)

    if extractor_type == "identity":
        return IdentityExtractor(observation_shape)

    if extractor_type == "mlp":
        return MLPExtractor(
            observation_shape=observation_shape,
            features_dim=int(cfg.get("features_dim", 64)),
            hidden_sizes=tuple(cfg.get("hidden_sizes", (64,))),
            activation=cfg.get("activation", "tanh"),
        )

    if extractor_type == "cnn":
        return CNNExtractor(
            observation_shape=observation_shape,
            features_dim=int(cfg.get("features_dim", 128)),
            tensor_format=cfg.get("tensor_format", "auto"),
            normalize_images=bool(cfg.get("normalize_images", True)),
        )

    raise ValueError(f"Unsupported extractor type: {extractor_type!r}")


def build_head(
    head_type: str,
    input_dim: int,
    output_dim: int,
    config: dict[str, Any],
    *,
    final_std: float,
    qnode_stats: QNodeStats | None = None,
    qnode_scope: str | None = None,
) -> nn.Module:
    head_type = head_type.lower()
    config = dict(config or {})
    _validate_head_config(config, head_type)
    activation = config.get("activation", "tanh")
    hidden_sizes = tuple(config.get("hidden_sizes", ()))

    if head_type == "classical":
        return ClassicalHead(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            final_std=final_std,
        )

    if head_type == "hybrid":
        return HybridHead(
            input_dim=input_dim,
            output_dim=output_dim,
            quantum_core=dict(config["quantum_core"]),
            input_adapter=config.get("input_adapter"),
            hidden_sizes=hidden_sizes,
            activation=activation,
            postprocessing=config.get("postprocessing", {}),
            final_std=final_std,
            qnode_stats=qnode_stats,
            qnode_scope=qnode_scope,
        )

    raise ValueError(f"Unsupported head type: {head_type!r}")


class ActorCriticAgent(nn.Module):
    """Modular actor-critic agent.

    Pipeline:

        observation -> feature_extractor -> actor_head  -> logits
                                      └-> critic_head -> value
    """

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        n_actions: int,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.observation_shape = tuple(int(x) for x in observation_shape)
        self.n_actions = int(n_actions)
        self.config = config or {}
        self.qnode_stats = QNodeStats()

        self.feature_extractor = build_feature_extractor(
            self.observation_shape,
            self.config.get("extractor", {"type": "identity"}),
        )

        default_head_type = self.config.get("head_type", "classical").lower()
        default_activation = self.config.get("activation", "tanh")
        pi_arch, vf_arch = _normalise_net_arch(self.config.get("net_arch", None))

        common_head_cfg = dict(self.config.get("head", {}))
        common_head_cfg.setdefault("activation", default_activation)

        actor_head_cfg = {
            **common_head_cfg,
            **self.config.get("actor_head", {}),
        }
        actor_head_cfg.setdefault("hidden_sizes", pi_arch)

        critic_head_cfg = {
            **common_head_cfg,
            **self.config.get("critic_head", {}),
        }
        critic_head_cfg.setdefault("hidden_sizes", vf_arch)

        self.actor_head = build_head(
            actor_head_cfg.get("type", default_head_type),
            input_dim=self.feature_extractor.output_dim,
            output_dim=self.n_actions,
            config=actor_head_cfg,
            final_std=float(actor_head_cfg.get("final_std", 0.01)),
            qnode_stats=self.qnode_stats,
            qnode_scope="actor",
        )

        self.critic_head = build_head(
            critic_head_cfg.get("type", default_head_type),
            input_dim=self.feature_extractor.output_dim,
            output_dim=1,
            config=critic_head_cfg,
            final_std=float(critic_head_cfg.get("final_std", 1.0)),
            qnode_stats=self.qnode_stats,
            qnode_scope="critic",
        )

    def extract_features(self, observations: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(observations)

    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(observations)
        return self.critic_head(features)

    def get_action_and_value(
        self,
        observations: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(observations)
        logits = self.actor_head(features)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        value = self.critic_head(features)
        return action, probs.log_prob(action), probs.entropy(), value

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(observations)
        return self.actor_head(features)

    def get_logits(self, observations: torch.Tensor) -> torch.Tensor:
        """Return policy logits for a batch of observations.

        This is intentionally explicit because KD export needs a stable teacher API
        for feature/logit dataset export, instead of relying on ad-hoc access to
        internal modules from downstream scripts.
        """
        features = self.extract_features(observations)
        return self.actor_head(features)

    def get_value_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """Return critic values from an already computed feature batch."""
        return self.critic_head(features)

    def get_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """Return actor logits from an already computed feature batch."""
        return self.actor_head(features)

    def get_teacher_outputs(
        self,
        observations: torch.Tensor,
        *,
        deterministic: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Return the standard outputs needed for KD dataset export.

        The returned tensors stay on the same device as the model and keep the
        computation graph if the caller is not under ``torch.no_grad()``. This
        makes the method usable both for offline export and for smoke tests.
        """
        features = self.extract_features(observations)
        logits = self.get_logits_from_features(features)
        values = self.get_value_from_features(features)
        distribution = Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()

        return {
            "features": features,
            "logits": logits,
            "values": values,
            "actions": actions,
        }

    def set_feature_extractor_trainable(self, trainable: bool) -> None:
        """Enable/disable gradients for the feature extractor."""
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = bool(trainable)

    def freeze_feature_extractor(self) -> None:
        """Freeze the feature extractor used as staged/KD interface."""
        self.set_feature_extractor_trainable(False)

    def replace_actor_head(self, actor_head: nn.Module) -> None:
        """Replace the policy head while preserving the feature extractor.

        This staged-composition hook is used by downstream workflows that train
        or distil a replacement policy head outside PPO and then plug it back
        into an existing actor-critic agent. The critic may be preserved or
        replaced separately with :meth:`replace_critic_head`.
        """
        self.actor_head = actor_head

    def replace_critic_head(self, critic_head: nn.Module) -> None:
        """Replace the value head while preserving the feature extractor.

        This enables policy-value staged composition where both actor and critic
        heads are swapped, for example after policy-value KD.
        """
        self.critic_head = critic_head

    def configure_qnode_wrappers(self, torch_device: str | torch.device) -> None:
        """Apply SimplyQRL's internal QNode-wrapper placement policy.

        This keeps the public runtime contract simple: users choose the main
        Torch runtime device, while each quantum core declares its PennyLane
        backend via ``quantum_core.kwargs.qml_device``. SimplyQRL then chooses
        where to place the QNode Torch wrapper.
        """

        main_device = torch.device(torch_device)
        for head in (self.actor_head, self.critic_head):
            setter = getattr(head, "set_qnode_wrapper_torch_device", None)
            if not callable(setter):
                continue

            backend_getter = getattr(head, "quantum_backend_name", None)
            qml_device = backend_getter() if callable(backend_getter) else "unknown"
            boundary_device, reason = _select_qnode_wrapper_torch_device(
                torch_device=main_device,
                qml_device=qml_device,
            )
            setter(boundary_device, reason=reason)

    def get_device_metadata(self) -> dict[str, Any]:
        """Return lightweight runtime placement metadata for reporting."""

        metadata: dict[str, Any] = {
            "agent_devices": _module_parameter_devices(self),
            "feature_extractor_devices": _module_parameter_devices(self.feature_extractor),
            "actor_head_devices": _module_parameter_devices(self.actor_head),
            "critic_head_devices": _module_parameter_devices(self.critic_head),
        }

        actor_adapter = getattr(self.actor_head, "input_adapter", None)
        critic_adapter = getattr(self.critic_head, "input_adapter", None)
        if isinstance(actor_adapter, nn.Module):
            metadata["actor_input_adapter_devices"] = _module_parameter_devices(actor_adapter)
        if isinstance(critic_adapter, nn.Module):
            metadata["critic_input_adapter_devices"] = _module_parameter_devices(critic_adapter)

        actor_q_layer = getattr(self.actor_head, "q_layer", None)
        critic_q_layer = getattr(self.critic_head, "q_layer", None)
        if actor_q_layer is not None:
            metadata["actor_qnode_wrapper_parameter_devices"] = _module_parameter_devices(actor_q_layer)
            metadata["actor_qnode_wrapper_torch_device"] = str(
                getattr(self.actor_head, "qnode_wrapper_torch_device", "unknown")
            )
            metadata["actor_qnode_wrapper_placement_reason"] = str(
                getattr(self.actor_head, "qnode_wrapper_placement_reason", "unknown")
            )
            backend_getter = getattr(self.actor_head, "quantum_backend_name", None)
            diff_getter = getattr(self.actor_head, "quantum_diff_method", None)
            metadata["actor_pennylane_backend"] = backend_getter() if callable(backend_getter) else "unknown"
            metadata["actor_pennylane_diff_method"] = diff_getter() if callable(diff_getter) else "unknown"
        if critic_q_layer is not None:
            metadata["critic_qnode_wrapper_parameter_devices"] = _module_parameter_devices(critic_q_layer)
            metadata["critic_qnode_wrapper_torch_device"] = str(
                getattr(self.critic_head, "qnode_wrapper_torch_device", "unknown")
            )
            metadata["critic_qnode_wrapper_placement_reason"] = str(
                getattr(self.critic_head, "qnode_wrapper_placement_reason", "unknown")
            )
            backend_getter = getattr(self.critic_head, "quantum_backend_name", None)
            diff_getter = getattr(self.critic_head, "quantum_diff_method", None)
            metadata["critic_pennylane_backend"] = backend_getter() if callable(backend_getter) else "unknown"
            metadata["critic_pennylane_diff_method"] = diff_getter() if callable(diff_getter) else "unknown"

        return metadata

    def get_qnode_stats(self) -> dict[str, Any]:
        """Return cumulative QNode execution statistics."""
        return self.qnode_stats.snapshot()

    def get_optimization_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Return disjoint trainable groups for branch-aware optimization.

        ``shared`` contains the feature extractor, while ``actor`` and ``critic``
        contain their respective heads. The contract is consumed by PPO optimizer
        construction, gradient clipping and optional diagnostics.
        """

        groups = {
            "shared": [
                parameter
                for parameter in self.feature_extractor.parameters(recurse=True)
                if parameter.requires_grad
            ],
            "actor": [
                parameter
                for parameter in self.actor_head.parameters(recurse=True)
                if parameter.requires_grad
            ],
            "critic": [
                parameter
                for parameter in self.critic_head.parameters(recurse=True)
                if parameter.requires_grad
            ],
        }

        seen: dict[int, str] = {}
        for group_name, parameters in groups.items():
            for parameter in parameters:
                previous = seen.get(id(parameter))
                if previous is not None:
                    raise RuntimeError(
                        "Optimization diagnostics require disjoint parameter groups; "
                        f"parameter appears in both {previous!r} and {group_name!r}."
                    )
                seen[id(parameter)] = group_name
        return groups

    def get_parameter_breakdown(self) -> ParameterBreakdown:
        actor_vqc = getattr(self.actor_head, "q_layer", None)
        critic_vqc = getattr(self.critic_head, "q_layer", None)

        actor_post = getattr(self.actor_head, "postprocessing", None)
        critic_post = getattr(self.critic_head, "postprocessing", None)

        actor_inf = getattr(self.actor_head, "inference_nn", None)
        critic_inf = getattr(self.critic_head, "inference_nn", None)

        trainable_params_encoder = count_trainable_params(self.feature_extractor)
        trainable_params_head_total = (
            count_trainable_params(self.actor_head) + count_trainable_params(self.critic_head)
        )
        trainable_params_vqc = count_trainable_params(actor_vqc) + count_trainable_params(critic_vqc)
        trainable_params_postprocessing = (
            count_trainable_params(actor_post) + count_trainable_params(critic_post)
        )
        trainable_params_inference_nn = (
            count_trainable_params(actor_inf) + count_trainable_params(critic_inf)
        )

        return ParameterBreakdown(
            trainable_params_total=count_trainable_params(self),
            trainable_params_encoder=trainable_params_encoder,
            trainable_params_head_total=trainable_params_head_total,
            trainable_params_vqc=trainable_params_vqc,
            trainable_params_postprocessing=trainable_params_postprocessing,
            trainable_params_inference_nn=trainable_params_inference_nn,
            trainable_params_other=0,
        )

    def get_agent_info(self) -> None:
        print("\nAGENT INFORMATION:")
        print("* Agent type: Actor-Critic 1.5")
        print(f"* Observation shape: {self.observation_shape}")
        print(f"* Feature dim: {self.feature_extractor.output_dim}")
        print("* Parameter breakdown:")
        for key, value in self.get_parameter_breakdown().to_dict().items():
            print(f"  - {key}: {value}")
        print("\n", self)


# Compatibility alias for checkpoints/scripts authored during the 1.5 migration.
# New code should use ``ActorCriticAgent``.
ActorCriticAgent15 = ActorCriticAgent


def build_actor_critic_agent(
    obs_shape: tuple[int, ...],
    n_actions: int,
    config: dict[str, Any] | None = None,
) -> ActorCriticAgent:
    obs_shape = tuple(int(x) for x in obs_shape)
    return ActorCriticAgent(
        observation_shape=obs_shape,
        n_actions=int(n_actions),
        config=config or {},
    )
