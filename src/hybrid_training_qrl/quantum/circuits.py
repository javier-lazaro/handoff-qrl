"""Paper-side VQC factories for the SimplyQRL quantum-core contract.

These factories intentionally live outside SimplyQRL.  They validate that the
library can consume externally-defined quantum circuits through the contract:

    torch.Tensor[batch, input_dim] -> torch.Tensor[batch, q_output_dim]

Each factory returns a ``qml.qnn.TorchLayer`` and can be referenced from YAML via
``quantum_core.factory``.
"""

from __future__ import annotations

from typing import Literal

import pennylane as qml
from torch import nn

Entangler = Literal[
    "none",
    "cnot",
    "cnot_circular",
    "cz",
    "cz_circular",
    "full_cnot",
    "full_cz",
]


def _validate_positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _apply_entanglement(entangler: str, n_qubits: int) -> None:
    entangler = str(entangler).lower()
    if entangler in {"none", "false", "off", ""} or n_qubits <= 1:
        return

    if entangler in {"cnot", "cnot_linear"}:
        for wire in range(n_qubits - 1):
            qml.CNOT(wires=[wire, wire + 1])
        return

    if entangler == "cnot_circular":
        for wire in range(n_qubits):
            qml.CNOT(wires=[wire, (wire + 1) % n_qubits])
        return

    if entangler in {"cz", "cz_linear"}:
        for wire in range(n_qubits - 1):
            qml.CZ(wires=[wire, wire + 1])
        return

    if entangler == "cz_circular":
        for wire in range(n_qubits):
            qml.CZ(wires=[wire, (wire + 1) % n_qubits])
        return

    if entangler == "full_cnot":
        for control in range(n_qubits):
            for target in range(n_qubits):
                if control != target:
                    qml.CNOT(wires=[control, target])
        return

    if entangler == "full_cz":
        for left in range(n_qubits):
            for right in range(left + 1, n_qubits):
                qml.CZ(wires=[left, right])
        return

    raise ValueError(f"Unsupported entangler: {entangler!r}.")


def build_angle_rx_ryrz_cnot_reuploading(
    *,
    input_dim: int,
    n_qubits: int,
    n_layers_q: int = 4,
    entangler: Entangler = "cnot_circular",
    qml_device: str = "lightning.qubit",
    diff_method: str = "adjoint",
    **_: object,
) -> nn.Module:
    """Angle RX embedding + RY/RZ ansatz + entanglement, repeated as DR layers.

    This corresponds to the paper-side "Circuit A" mock-up:

    for each data-reuploading layer:
        RX feature embedding -> RY/RZ trainable rotations -> entanglement

    The factory intentionally requires ``input_dim == n_qubits`` so that a YAML
    config cannot silently pad/truncate the state representation.  Use an MLP or
    CNN extractor with ``features_dim == n_qubits`` when the raw observation
    dimension differs from the circuit width.
    """

    input_dim = _validate_positive_int("input_dim", input_dim)
    n_qubits = _validate_positive_int("n_qubits", n_qubits)
    n_layers_q = _validate_positive_int("n_layers_q", n_layers_q)

    if input_dim != n_qubits:
        raise ValueError(
            "build_angle_rx_ryrz_cnot_reuploading requires input_dim == n_qubits. "
            f"Got input_dim={input_dim}, n_qubits={n_qubits}."
        )

    dev = qml.device(qml_device, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        for layer in range(n_layers_q):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")

            for wire in range(n_qubits):
                qml.RY(weights[layer, 0, wire], wires=wire)
                qml.RZ(weights[layer, 1, wire], wires=wire)

            _apply_entanglement(entangler, n_qubits)

        return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

    weight_shapes = {"weights": (n_layers_q, 2, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)



def build_chunked_angle_rx_ryrz_cnot(
    *,
    input_dim: int,
    n_qubits: int,
    entangler: Entangler = "cnot_circular",
    qml_device: str = "lightning.qubit",
    diff_method: str = "adjoint",
    **_: object,
) -> nn.Module:
    """Consume a wider feature vector through sequential angle-encoding chunks.

    The input is split into contiguous ``n_qubits``-wide chunks.  Each chunk is
    uploaded exactly once with RX angle encoding, followed by an independent
    trainable RY/RZ block and the configured entangler.  This lets an 8-qubit
    circuit consume all 32 MinAtar D32 features without a learned D32->D8
    bottleneck:

        x[0:8]   -> RX -> RY/RZ -> entangle
        x[8:16]  -> RX -> RY/RZ -> entangle
        x[16:24] -> RX -> RY/RZ -> entangle
        x[24:32] -> RX -> RY/RZ -> entangle

    ``input_dim`` must be an exact positive multiple of ``n_qubits`` so the
    factory cannot silently pad or truncate features.
    """

    input_dim = _validate_positive_int("input_dim", input_dim)
    n_qubits = _validate_positive_int("n_qubits", n_qubits)
    if input_dim % n_qubits != 0:
        raise ValueError(
            "build_chunked_angle_rx_ryrz_cnot requires input_dim to be an exact "
            f"multiple of n_qubits. Got input_dim={input_dim}, n_qubits={n_qubits}."
        )

    n_chunks = input_dim // n_qubits
    dev = qml.device(qml_device, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        for chunk in range(n_chunks):
            start = chunk * n_qubits
            stop = start + n_qubits
            qml.AngleEmbedding(
                inputs[..., start:stop],
                wires=range(n_qubits),
                rotation="X",
            )

            for wire in range(n_qubits):
                qml.RY(weights[chunk, 0, wire], wires=wire)
                qml.RZ(weights[chunk, 1, wire], wires=wire)

            _apply_entanglement(entangler, n_qubits)

        return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

    weight_shapes = {"weights": (n_chunks, 2, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


def build_chunked_angle_rx_ryrz_cnot_reuploading(
    *,
    input_dim: int,
    n_qubits: int,
    n_reuploads: int = 5,
    entangler: Entangler = "cnot_circular",
    qml_device: str = "lightning.qubit",
    diff_method: str = "adjoint",
    **_: object,
) -> nn.Module:
    """Consume a wider feature vector with repeated full chunked data reuploading.

    A *full pass* uploads every ``n_qubits``-wide chunk once, with an
    independent RY/RZ variational block and entangler after each chunk.  Data
    reuploading repeats that entire full-feature pass ``n_reuploads`` times.

    For the paper MinAtar D32 / 8Q / DR5 case this is therefore:

        for reupload in 0..4:
            x[0:8]   -> RX -> RY/RZ -> entangle
            x[8:16]  -> RX -> RY/RZ -> entangle
            x[16:24] -> RX -> RY/RZ -> entangle
            x[24:32] -> RX -> RY/RZ -> entangle

    giving 5 full D32 passes, 20 encoding+variational blocks and 320 trainable
    VQC parameters per head.  ``input_dim`` must be an exact positive multiple
    of ``n_qubits`` so no feature is silently padded or truncated.
    """

    input_dim = _validate_positive_int("input_dim", input_dim)
    n_qubits = _validate_positive_int("n_qubits", n_qubits)
    n_reuploads = _validate_positive_int("n_reuploads", n_reuploads)
    if input_dim % n_qubits != 0:
        raise ValueError(
            "build_chunked_angle_rx_ryrz_cnot_reuploading requires input_dim "
            "to be an exact multiple of n_qubits. "
            f"Got input_dim={input_dim}, n_qubits={n_qubits}."
        )

    n_chunks = input_dim // n_qubits
    dev = qml.device(qml_device, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        for reupload in range(n_reuploads):
            for chunk in range(n_chunks):
                start = chunk * n_qubits
                stop = start + n_qubits
                qml.AngleEmbedding(
                    inputs[..., start:stop],
                    wires=range(n_qubits),
                    rotation="X",
                )

                for wire in range(n_qubits):
                    qml.RY(weights[reupload, chunk, 0, wire], wires=wire)
                    qml.RZ(weights[reupload, chunk, 1, wire], wires=wire)

                _apply_entanglement(entangler, n_qubits)

        return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

    weight_shapes = {"weights": (n_reuploads, n_chunks, 2, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


def build_amplitude_fully_entangled(
    *,
    input_dim: int,
    n_qubits: int,
    n_layers_q: int = 4,
    entangler: Entangler = "full_cnot",
    qml_device: str = "lightning.qubit",
    diff_method: str = "adjoint",
    normalize: bool = True,
    pad_with: float = 0.0,
    **_: object,
) -> nn.Module:
    """Amplitude embedding + repeated fully entangled trainable layers.

    This is the paper-side "Circuit B" mock-up.  It is intentionally qubit
    efficient: e.g. an 8-dimensional feature vector can be encoded in 3 qubits.
    Output Reuse remains outside the VQC and is configured through SimplyQRL's
    ``postprocessing.reuse_repetitions``.
    """

    input_dim = _validate_positive_int("input_dim", input_dim)
    n_qubits = _validate_positive_int("n_qubits", n_qubits)
    n_layers_q = _validate_positive_int("n_layers_q", n_layers_q)

    max_features = 2**n_qubits
    if input_dim > max_features:
        raise ValueError(
            "Amplitude embedding cannot encode more features than 2**n_qubits. "
            f"Got input_dim={input_dim}, n_qubits={n_qubits}, max_features={max_features}."
        )

    dev = qml.device(qml_device, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        qml.AmplitudeEmbedding(
            inputs,
            wires=range(n_qubits),
            pad_with=float(pad_with),
            normalize=bool(normalize),
        )

        for layer in range(n_layers_q):
            for wire in range(n_qubits):
                qml.RY(weights[layer, 0, wire], wires=wire)
                qml.RZ(weights[layer, 1, wire], wires=wire)

            _apply_entanglement(entangler, n_qubits)

        return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

    weight_shapes = {"weights": (n_layers_q, 2, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)
