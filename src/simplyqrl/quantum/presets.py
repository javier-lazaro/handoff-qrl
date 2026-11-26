"""Built-in quantum-core presets for SimplyQRL.

These presets adapt the existing SimplyQRL qlayer builders to the new
quantum-core contract used by HybridHead:

    torch.Tensor[batch, input_dim] -> torch.Tensor[batch, q_output_dim]

The registry treats these presets exactly like external user factories.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from simplyqrl.embeddings import build_embedding_fn
from simplyqrl.qlayers import (
    build_basic_qlayer,
    build_dr_qlayer,
    build_hsiao_qlayer,
    build_skolik_qlayer,
)


def build_skolik_rx_cz_reuploading(
    *,
    input_dim: int,
    n_qubits: int,
    n_layers_q: int = 1,
    ent: bool = False,
    emb_indices: list[int] | None = None,
    transform_fn: Any | None = None,
    qml_device: str = "default.qubit",
    diff_method: str = "backprop",
    **_: Any,
) -> nn.Module:
    """Skolik-style RX reuploading preset.

    The ``input_dim`` argument is accepted to satisfy the quantum-core factory
    contract, even if this preset mostly relies on ``n_qubits`` and optional
    ``emb_indices`` internally.
    """

    _ = input_dim
    return build_skolik_qlayer(
        n_qubits=n_qubits,
        n_layers=n_layers_q,
        emb_indices=emb_indices,
        transform_fn=transform_fn,
        ent=ent,
        qml_device=qml_device,
        diff_method=diff_method,
    )


def build_dr_rx_reuploading(
    *,
    input_dim: int,
    n_qubits: int,
    n_layers_q: int = 1,
    ent: bool = False,
    emb_indices: list[int] | None = None,
    transform_fn: Any | None = None,
    qml_device: str = "default.qubit",
    diff_method: str = "backprop",
    **_: Any,
) -> nn.Module:
    """Data-reuploading preset based on the existing SimplyQRL DR layer."""

    _ = input_dim
    return build_dr_qlayer(
        n_qubits=n_qubits,
        n_layers=n_layers_q,
        emb_indices=emb_indices,
        transform_fn=transform_fn,
        ent=ent,
        qml_device=qml_device,
        diff_method=diff_method,
    )


def build_hsiao_rx_unentangled(
    *,
    input_dim: int,
    n_qubits: int,
    n_layers_q: int = 1,
    emb_type: str = "multi",
    emb_indices: list[int] | None = None,
    transform_fn: Any | None = None,
    ent: bool = False,
    qml_device: str = "default.qubit",
    diff_method: str = "backprop",
    **_: Any,
) -> nn.Module:
    """Hsiao-style preset based on the existing SimplyQRL Hsiao layer."""

    _ = input_dim
    return build_hsiao_qlayer(
        n_qubits=n_qubits,
        n_layers=n_layers_q,
        emb_type=emb_type,
        emb_indices=emb_indices,
        transform_fn=transform_fn,
        ent=ent,
        qml_device=qml_device,
        diff_method=diff_method,
    )


def build_basic_angle_entangler(
    *,
    input_dim: int,
    n_qubits: int,
    n_layers_q: int = 1,
    embedding: str | Any = "angle",
    embedding_kwargs: dict[str, Any] | None = None,
    qml_device: str = "default.qubit",
    diff_method: str = "backprop",
    **_: Any,
) -> nn.Module:
    """Basic configurable embedding + BasicEntanglerLayers preset."""

    _ = input_dim
    embedding_fn = build_embedding_fn(embedding, embedding_kwargs or {})
    return build_basic_qlayer(
        embedding_fn=embedding_fn,
        n_qubits=n_qubits,
        n_layers=n_layers_q,
        qml_device=qml_device,
        diff_method=diff_method,
    )


BUILTIN_QUANTUM_CORE_PRESETS = {
    "skolik_rx_cz_reuploading": build_skolik_rx_cz_reuploading,
    "dr_rx_reuploading": build_dr_rx_reuploading,
    "hsiao_rx_unentangled": build_hsiao_rx_unentangled,
    "basic_angle_entangler": build_basic_angle_entangler,
}
