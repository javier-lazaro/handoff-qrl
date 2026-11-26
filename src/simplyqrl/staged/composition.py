"""Composed-policy utilities for staged SimplyQRL workflows.

The core staged use case is plug-back after offline KD:

    teacher PPO checkpoint + KD head artifact(s) -> composed PPO policy

The composition contract supports both actor-only and policy-value workflows:

* actor-only composition replaces the policy head while preserving the value head;
* policy-value composition replaces both policy and value heads.

The feature extractor can be frozen explicitly for downstream adaptation, or left
trainable for end-to-end staged fine-tuning. Optimizer rebuilding is intentionally
kept explicit via :func:`rebuild_ppo_optimizer_after_composition`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from simplyqrl.instrumentation import QNodeStats
from simplyqrl.staged.head_artifacts import load_head_artifact, load_head_config


def _get_policy(container: Any) -> nn.Module:
    """Return the policy/agent object from a PPO-like container or policy itself."""

    policy = getattr(container, "agent", container)
    if not isinstance(policy, nn.Module):
        raise TypeError(
            "Expected a torch.nn.Module policy or an object with an `.agent` module, "
            f"got {type(container)!r}."
        )
    return policy


def _get_device(container: Any, policy: nn.Module) -> torch.device:
    device = getattr(container, "device", None)
    if device is not None:
        return torch.device(device)

    try:
        return next(policy.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _policy_feature_dim(policy: nn.Module) -> int:
    feature_extractor = getattr(policy, "feature_extractor", None)
    if feature_extractor is None:
        raise AttributeError("Policy does not expose a feature_extractor")

    output_dim = getattr(feature_extractor, "output_dim", None)
    if output_dim is None:
        raise AttributeError("Policy feature_extractor does not expose output_dim")
    return int(output_dim)


def _policy_action_dim(policy: nn.Module, container: Any | None = None) -> int | None:
    n_actions = getattr(policy, "n_actions", None)
    if n_actions is not None:
        return int(n_actions)

    envs = getattr(container, "envs", None) if container is not None else None
    action_space = getattr(envs, "single_action_space", None)
    n = getattr(action_space, "n", None)
    if n is not None:
        return int(n)

    actor_head = getattr(policy, "actor_head", None)
    output_dim = getattr(actor_head, "output_dim", None)
    if output_dim is not None:
        return int(output_dim)

    return None


def _resolve_qnode_stats(policy: nn.Module, qnode_stats: QNodeStats | None) -> QNodeStats | None:
    if qnode_stats is not None:
        return qnode_stats
    return getattr(policy, "qnode_stats", None)


def _load_or_use_head(
    head_artifact: str | Path | nn.Module,
    *,
    use_best: bool,
    device: torch.device,
    qnode_stats: QNodeStats | None,
    qnode_scope: str,
) -> tuple[nn.Module, dict[str, Any] | None]:
    """Return ``(head, head_config)`` from an artifact path or module."""

    if isinstance(head_artifact, (str, Path)):
        artifact_path = Path(head_artifact)
        head_config = load_head_config(artifact_path)
        head = load_head_artifact(
            artifact_path,
            use_best=use_best,
            qnode_stats=qnode_stats,
            qnode_scope=qnode_scope,
        )
    elif isinstance(head_artifact, nn.Module):
        head = head_artifact
        head_config = None
    else:
        raise TypeError(
            "Head artifact must be a path or torch.nn.Module, "
            f"got {type(head_artifact)!r}."
        )

    head.to(device)
    return head, head_config


def _validate_head_role(head_config: dict[str, Any] | None, *, expected_role: str, label: str) -> None:
    if head_config is None:
        return
    role = str(head_config.get("role", "unknown")).lower()
    if role not in {expected_role, "unknown"}:
        raise ValueError(
            f"{label} HeadArtifact role mismatch: expected {expected_role!r} or 'unknown', got {role!r}."
        )


def _validate_input_dim(
    *,
    label: str,
    head: nn.Module,
    head_config: dict[str, Any] | None,
    expected_input_dim: int,
) -> None:
    head_input_dim = getattr(head, "input_dim", None)
    if head_input_dim is not None and int(head_input_dim) != expected_input_dim:
        raise ValueError(
            f"{label} head input_dim does not match teacher feature extractor: "
            f"head={int(head_input_dim)}, teacher_features={expected_input_dim}."
        )

    if head_config is not None:
        cfg_input_dim = int(head_config["input_dim"])
        if cfg_input_dim != expected_input_dim:
            raise ValueError(
                f"{label} HeadArtifact input_dim does not match teacher feature extractor: "
                f"artifact={cfg_input_dim}, teacher_features={expected_input_dim}."
            )


def _validate_output_dim(
    *,
    label: str,
    head: nn.Module,
    head_config: dict[str, Any] | None,
    expected_output_dim: int | None,
) -> None:
    if expected_output_dim is None:
        return

    head_output_dim = getattr(head, "output_dim", None)
    if head_output_dim is not None and int(head_output_dim) != int(expected_output_dim):
        raise ValueError(
            f"{label} head output_dim does not match expected output dim: "
            f"head={int(head_output_dim)}, expected={int(expected_output_dim)}."
        )

    if head_config is not None:
        cfg_output_dim = int(head_config["output_dim"])
        if cfg_output_dim != int(expected_output_dim):
            raise ValueError(
                f"{label} HeadArtifact output_dim does not match expected output dim: "
                f"artifact={cfg_output_dim}, expected={int(expected_output_dim)}."
            )


def compose_policy_from_teacher(
    teacher: Any,
    actor_head_artifact: str | Path | nn.Module,
    *,
    critic_head_artifact: str | Path | nn.Module | None = None,
    freeze_feature_extractor: bool = False,
    use_best: bool = True,
    validate_shapes: bool = True,
    qnode_stats: QNodeStats | None = None,
    qnode_scope: str = "actor",
    critic_qnode_scope: str = "critic",
) -> Any:
    """Compose a teacher policy with staged actor and optional critic heads.

    Parameters
    ----------
    teacher:
        Either a PPO-like object exposing `.agent` or a policy module directly.
        The object is modified in-place and returned.
    actor_head_artifact:
        Path to a canonical/legacy actor HeadArtifact, or an already constructed
        `nn.Module` actor head.
    critic_head_artifact:
        Optional path/module for a value head. If provided, the teacher critic is
        replaced as well, enabling policy-value staged composition.
    freeze_feature_extractor:
        If true, call the policy's `freeze_feature_extractor()` method after
        replacing heads. Use this for downstream/frozen-representation workflows.
        Leave it false for end-to-end staged fine-tuning.
    use_best:
        When loading from artifact, prefer `best_model_state_dict` if present.
    validate_shapes:
        Validate that head input/output dimensions are compatible with the
        teacher feature extractor, action space and scalar value output.
    qnode_stats:
        Optional QNodeStats object to inject into hybrid heads. If omitted and
        the teacher policy exposes `.qnode_stats`, that object is reused.
    qnode_scope:
        Scope label used for actor hybrid-head QNodeStats instrumentation.
    critic_qnode_scope:
        Scope label used for critic hybrid-head QNodeStats instrumentation.

    Notes
    -----
    This function does not rebuild the PPO optimizer. Online fine-tuning should
    explicitly call :func:`rebuild_ppo_optimizer_after_composition` after
    composition so the optimizer points to the newly installed modules and
    respects any frozen parameters.
    """

    policy = _get_policy(teacher)
    device = _get_device(teacher, policy)
    shared_qnode_stats = _resolve_qnode_stats(policy, qnode_stats)

    actor_head, actor_head_config = _load_or_use_head(
        actor_head_artifact,
        use_best=use_best,
        device=device,
        qnode_stats=shared_qnode_stats,
        qnode_scope=qnode_scope,
    )
    _validate_head_role(actor_head_config, expected_role="actor", label="Actor")

    critic_head: nn.Module | None = None
    critic_head_config: dict[str, Any] | None = None
    if critic_head_artifact is not None:
        critic_head, critic_head_config = _load_or_use_head(
            critic_head_artifact,
            use_best=use_best,
            device=device,
            qnode_stats=shared_qnode_stats,
            qnode_scope=critic_qnode_scope,
        )
        _validate_head_role(critic_head_config, expected_role="critic", label="Critic")

    if validate_shapes:
        expected_input_dim = _policy_feature_dim(policy)
        expected_action_dim = _policy_action_dim(policy, teacher)

        _validate_input_dim(
            label="Actor",
            head=actor_head,
            head_config=actor_head_config,
            expected_input_dim=expected_input_dim,
        )
        _validate_output_dim(
            label="Actor",
            head=actor_head,
            head_config=actor_head_config,
            expected_output_dim=expected_action_dim,
        )

        if critic_head is not None:
            _validate_input_dim(
                label="Critic",
                head=critic_head,
                head_config=critic_head_config,
                expected_input_dim=expected_input_dim,
            )
            _validate_output_dim(
                label="Critic",
                head=critic_head,
                head_config=critic_head_config,
                expected_output_dim=1,
            )

    if not hasattr(policy, "replace_actor_head"):
        raise AttributeError("Policy does not expose replace_actor_head(head)")
    policy.replace_actor_head(actor_head)

    if critic_head is not None:
        if not hasattr(policy, "replace_critic_head"):
            raise AttributeError("Policy does not expose replace_critic_head(head)")
        policy.replace_critic_head(critic_head)

    if freeze_feature_extractor:
        if not hasattr(policy, "freeze_feature_extractor"):
            raise AttributeError("Policy does not expose freeze_feature_extractor()")
        policy.freeze_feature_extractor()

    # Make the composed policy explicit in metadata-like attributes without
    # requiring a dedicated wrapper class.
    setattr(policy, "_simplyqrl_composed_actor_head", True)
    setattr(policy, "_simplyqrl_composed_critic_head", critic_head is not None)
    setattr(policy, "_simplyqrl_composition_scope", "policy_value" if critic_head is not None else "policy")
    setattr(policy, "_simplyqrl_composition_freeze_feature_extractor", bool(freeze_feature_extractor))
    if actor_head_config is not None:
        setattr(policy, "_simplyqrl_composed_actor_head_config", dict(actor_head_config))
    if critic_head_config is not None:
        setattr(policy, "_simplyqrl_composed_critic_head_config", dict(critic_head_config))

    return teacher



def rebuild_ppo_optimizer_after_composition(
    ppo: Any,
    *,
    learning_rate: float | None = None,
    eps: float = 1e-5,
    optimizer_cls: type[torch.optim.Optimizer] | None = None,
    optimizer_kwargs: dict[str, Any] | None = None,
) -> torch.optim.Optimizer:
    """Rebuild a PPO optimizer after replacing staged policy components.

    ``compose_policy_from_teacher`` replaces modules inside the policy in-place.
    A pre-existing PPO optimizer still points to the old parameter objects, so
    online fine-tuning must rebuild the optimizer before calling ``train``.

    By default, PPO's native ``rebuild_optimizer`` method is used so global or
    branch-specific learning rates and parameter ownership are preserved. Custom
    optimizer arguments retain the explicit global-optimizer compatibility path.

    Only trainable parameters are included. This keeps downstream/frozen-feature
    compositions explicit and avoids optimizer state for intentionally frozen
    modules.
    """

    if not hasattr(ppo, "agent"):
        raise AttributeError("Expected a PPO-like object exposing `.agent`.")

    native_rebuilder = getattr(ppo, "rebuild_optimizer", None)
    if (
        callable(native_rebuilder)
        and optimizer_cls is None
        and optimizer_kwargs is None
        and learning_rate is None
    ):
        return native_rebuilder()

    learning_rate_mode = str(getattr(ppo, "learning_rate_mode", "global"))
    if learning_rate_mode != "global":
        raise ValueError(
            "Explicit optimizer overrides are supported only with "
            "learning_rate_mode='global'. Call PPO.rebuild_optimizer() without "
            "overrides to preserve branch-specific learning rates."
        )

    if optimizer_cls is None:
        optimizer_cls = torch.optim.Adam

    if learning_rate is None:
        if not hasattr(ppo, "learning_rate"):
            raise AttributeError(
                "learning_rate was not provided and the PPO-like object does not "
                "expose `.learning_rate`."
            )
        learning_rate = float(ppo.learning_rate)

    group_resolver = getattr(ppo, "_resolve_optimization_parameter_groups", None)
    if callable(group_resolver):
        ppo.optimization_parameter_groups = group_resolver()

    kwargs = dict(optimizer_kwargs or {})
    kwargs.setdefault("lr", float(learning_rate))

    # PPO uses Adam(eps=1e-5). Preserve that default for Adam while allowing
    # callers to pass arbitrary optimizer kwargs for other optimizers.
    if optimizer_cls is torch.optim.Adam:
        kwargs.setdefault("eps", float(eps))

    trainable_parameters = [parameter for parameter in ppo.agent.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("Cannot rebuild PPO optimizer: no trainable agent parameters remain.")

    optimizer = optimizer_cls(trainable_parameters, **kwargs)
    ppo.optimizer = optimizer

    probe_factory = getattr(ppo, "_create_optimization_diagnostics_probe", None)
    if getattr(ppo, "optimization_diagnostics_probe", None) is not None and callable(
        probe_factory
    ):
        ppo.optimization_diagnostics_probe = probe_factory()

    return optimizer


def is_feature_extractor_frozen(policy_or_teacher: Any) -> bool:
    """Return whether all feature extractor parameters are frozen.

    Stateless extractors such as IdentityExtractor have no parameters and are
    considered frozen for staged-composition purposes.
    """

    policy = _get_policy(policy_or_teacher)
    feature_extractor = getattr(policy, "feature_extractor", None)
    if feature_extractor is None:
        raise AttributeError("Policy does not expose a feature_extractor")

    params = list(feature_extractor.parameters())
    if not params:
        return True
    return all(not parameter.requires_grad for parameter in params)
