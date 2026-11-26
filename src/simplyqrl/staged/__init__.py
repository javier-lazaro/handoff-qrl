"""Small public staged-policy utility surface for SimplyQRL.

The staged namespace intentionally exposes only generic architecture-level
utilities: independent head artifacts and policy composition. Paper-specific
teacher dataset exporters, KD trainers, campaigns, analysis and plotting should
live in downstream experiment repositories.
"""

from simplyqrl.staged.composition import (
    compose_policy_from_teacher,
    is_feature_extractor_frozen,
    rebuild_ppo_optimizer_after_composition,
)
from simplyqrl.staged.head_artifacts import (
    HEAD_ARTIFACT_FORMAT,
    HEAD_ARTIFACT_VERSION,
    build_head_from_config,
    load_head_artifact,
    load_head_config,
    normalize_head_config,
    save_head_artifact,
)

__all__ = [
    "compose_policy_from_teacher",
    "is_feature_extractor_frozen",
    "rebuild_ppo_optimizer_after_composition",
    "HEAD_ARTIFACT_FORMAT",
    "HEAD_ARTIFACT_VERSION",
    "build_head_from_config",
    "load_head_artifact",
    "load_head_config",
    "normalize_head_config",
    "save_head_artifact",
]
