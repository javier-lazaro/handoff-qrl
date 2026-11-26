"""Shared statistical primitives for the frozen paper protocol.

The final paper contract lives in ``configs/minatar/protocol.yaml``.  This
module centralises the statistical operations used by both LunarLander and
MinAtar so that the two evidence surfaces cannot silently drift apart.

The scientific contract fixes the training seed as the statistical unit, the
paired mean difference as the primary estimand, percentile paired bootstrap
confidence intervals, exact paired sign-flip tests when feasible, Holm
correction within the preregistered RQ1/RQ2 families, and IQM / probability of
improvement as secondary estimands.

The resampling count and RNG seed are implementation-level reproducibility
constants rather than additional scientific degrees of freedom.  Their defaults
preserve the values already frozen in the LunarLander paper amendment and are
shared by both environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import yaml

DEFAULT_BOOTSTRAP_SAMPLES = 50_000
DEFAULT_BOOTSTRAP_SEED = 20_260_914
DEFAULT_ALPHA = 0.05
MAX_EXACT_SIGN_FLIP_PAIRS = 20
DEFAULT_MONTE_CARLO_SIGN_FLIP_SAMPLES = 1_000_000


class PaperStatisticsError(ValueError):
    """Raised when a paper statistical input or contract is malformed."""


@dataclass(frozen=True)
class ContrastSpec:
    """One ordered preregistered contrast, e.g. ``H(0.25)-H(0)``."""

    expression: str
    left: str
    right: str


@dataclass(frozen=True)
class PaperStatisticsContract:
    """Statistical subset of the frozen paper protocol contract."""

    unit: str
    primary_estimand: str
    confidence_interval: str
    confirmatory_p_value: str
    confirmatory_p_value_fallback: str | None
    multiplicity_method: str
    holm_families: Mapping[str, tuple[ContrastSpec, ...]]
    secondary_estimands: tuple[str, ...]


@dataclass(frozen=True)
class PairedContrastResult:
    """Analysis result for one paired seed-level contrast."""

    contrast: str
    left: str
    right: str
    n: int
    paired_mean_difference: float
    paired_std_difference: float
    paired_median_difference: float
    ci95_low: float
    ci95_high: float
    wins_left: int
    ties: int
    sign_flip_p: float
    iqm_difference: float
    iqm_ci95_low: float
    iqm_ci95_high: float
    probability_of_improvement: float
    holm_p: float | None = None
    holm_reject_0_05: bool | None = None


def _as_finite_1d(values: Iterable[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise PaperStatisticsError(f"{name} must be a non-empty one-dimensional sample")
    if not np.all(np.isfinite(array)):
        raise PaperStatisticsError(f"{name} contains NaN or infinite values")
    return array


def _as_paired(
    left: Iterable[float] | np.ndarray,
    right: Iterable[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left_array = _as_finite_1d(left, name="left")
    right_array = _as_finite_1d(right, name="right")
    if left_array.shape != right_array.shape:
        raise PaperStatisticsError(
            f"Paired samples must have identical shape: {left_array.shape} != {right_array.shape}"
        )
    return left_array, right_array


def parse_contrast(expression: str) -> ContrastSpec:
    """Parse a frozen paper contrast expression into left/right condition labels."""

    compact = str(expression).replace(" ", "")
    parts = compact.split("-")
    if len(parts) != 2 or not all(parts):
        raise PaperStatisticsError(f"Unsupported contrast expression: {expression!r}")
    left, right = parts
    return ContrastSpec(expression=compact, left=left, right=right)


def load_statistics_contract(path: str | Path) -> PaperStatisticsContract:
    """Load and fail-closed validate the statistical section of the paper YAML contract."""

    contract_path = Path(path)
    try:
        payload = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PaperStatisticsError(f"Paper contract not found: {contract_path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("statistics"), dict):
        raise PaperStatisticsError(f"Paper contract has no statistics mapping: {contract_path}")

    statistics = payload["statistics"]
    confirmatory = statistics.get("confirmatory_p_value")
    multiplicity = statistics.get("multiplicity")
    if not isinstance(confirmatory, dict):
        raise PaperStatisticsError("statistics.confirmatory_p_value must be a mapping")
    if not isinstance(multiplicity, dict):
        raise PaperStatisticsError("statistics.multiplicity must be a mapping")

    if statistics.get("unit") != "training_seed":
        raise PaperStatisticsError("Paper statistical unit must remain training_seed")
    if statistics.get("primary_estimand") != "paired_mean_difference":
        raise PaperStatisticsError("Paper primary estimand must remain paired_mean_difference")
    if statistics.get("confidence_interval") != "paired_bootstrap":
        raise PaperStatisticsError("Paper confidence interval must remain paired_bootstrap")
    if confirmatory.get("primary") != "exact_sign_flip":
        raise PaperStatisticsError("Paper confirmatory test must remain exact_sign_flip")
    if multiplicity.get("method") != "holm":
        raise PaperStatisticsError("Paper multiplicity correction must remain Holm")

    raw_families = multiplicity.get("families")
    if not isinstance(raw_families, dict) or not raw_families:
        raise PaperStatisticsError("Paper Holm families are missing")
    families: dict[str, tuple[ContrastSpec, ...]] = {}
    for family_name, expressions in raw_families.items():
        if not isinstance(expressions, list) or not expressions:
            raise PaperStatisticsError(f"Holm family {family_name!r} must be a non-empty list")
        families[str(family_name)] = tuple(parse_contrast(str(item)) for item in expressions)

    secondary = statistics.get("secondary_estimands")
    if not isinstance(secondary, list):
        raise PaperStatisticsError("statistics.secondary_estimands must be a list")
    secondary_tuple = tuple(str(item) for item in secondary)
    if secondary_tuple != ("iqm", "probability_of_improvement"):
        raise PaperStatisticsError(
            "Paper secondary estimands must remain ['iqm', 'probability_of_improvement']"
        )

    return PaperStatisticsContract(
        unit=str(statistics["unit"]),
        primary_estimand=str(statistics["primary_estimand"]),
        confidence_interval=str(statistics["confidence_interval"]),
        confirmatory_p_value=str(confirmatory["primary"]),
        confirmatory_p_value_fallback=(
            str(confirmatory["fallback_if_exact_infeasible"])
            if confirmatory.get("fallback_if_exact_infeasible") is not None
            else None
        ),
        multiplicity_method=str(multiplicity["method"]),
        holm_families=families,
        secondary_estimands=secondary_tuple,
    )


def paired_mean_difference(
    left: Iterable[float] | np.ndarray,
    right: Iterable[float] | np.ndarray,
) -> float:
    """Return mean(left - right) while preserving the seed-level pairing."""

    left_array, right_array = _as_paired(left, right)
    return float(np.mean(left_array - right_array))


def _iqm_weights(n: int) -> np.ndarray:
    """Return exact empirical-quantile weights for the central 50% interval."""

    if n <= 0:
        raise PaperStatisticsError("IQM requires at least one observation")
    edges_left = np.arange(n, dtype=float) / n
    edges_right = np.arange(1, n + 1, dtype=float) / n
    overlap = np.maximum(0.0, np.minimum(edges_right, 0.75) - np.maximum(edges_left, 0.25))
    weights = overlap / 0.5
    if not np.isclose(weights.sum(), 1.0):
        raise PaperStatisticsError("Internal IQM weighting error")
    return weights


def interquartile_mean(values: Iterable[float] | np.ndarray) -> float:
    """Compute the IQM as the mean over the central 50% empirical quantile mass.

    Fractional boundary weights are used when 25% of the sample size is not an
    integer, avoiding a hidden rounding rule.  For paper's n=20 this reduces to
    the ordinary mean of the middle ten ordered seed-level values.
    """

    array = np.sort(_as_finite_1d(values, name="values"))
    return float(array @ _iqm_weights(array.size))


def _bootstrap_statistics(samples: np.ndarray, *, statistic: str) -> np.ndarray:
    if statistic == "mean":
        return samples.mean(axis=1)
    if statistic == "iqm":
        ordered = np.sort(samples, axis=1)
        return ordered @ _iqm_weights(samples.shape[1])
    raise PaperStatisticsError(f"Unsupported bootstrap statistic: {statistic!r}")


def percentile_bootstrap_ci(
    values: Iterable[float] | np.ndarray,
    *,
    statistic: str = "mean",
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a one-dimensional seed-level sample.

    When ``values`` are paired differences this is the paired bootstrap used by
    paper.  ``statistic`` may be ``"mean"`` (primary) or ``"iqm"`` (secondary).
    """

    array = _as_finite_1d(values, name="values")
    if samples <= 0:
        raise PaperStatisticsError("Bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise PaperStatisticsError("Bootstrap confidence must lie strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    chunk_size = min(samples, 10_000)
    bootstrapped: list[np.ndarray] = []
    remaining = samples
    while remaining:
        current = min(chunk_size, remaining)
        indices = rng.integers(0, array.size, size=(current, array.size))
        bootstrapped.append(_bootstrap_statistics(array[indices], statistic=statistic))
        remaining -= current

    estimates = np.concatenate(bootstrapped)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [tail, 1.0 - tail])
    return float(low), float(high)


def exact_sign_flip_pvalue(values: Iterable[float] | np.ndarray) -> float:
    """Two-sided exact paired sign-flip p-value.

    paper has n=20 training seeds, so enumerating all 2**20 sign assignments is
    feasible and avoids Monte Carlo noise in the confirmatory p-values.
    """

    array = _as_finite_1d(values, name="paired differences")
    n = array.size
    if n > MAX_EXACT_SIGN_FLIP_PAIRS:
        raise PaperStatisticsError(
            f"Exact sign-flip supports at most {MAX_EXACT_SIGN_FLIP_PAIRS} pairs, got {n}"
        )

    observed = abs(float(array.mean()))
    total = 1 << n
    extreme = 0
    bit_positions = np.arange(n, dtype=np.uint64)
    chunk_size = 1 << min(n, 16)
    tolerance = 1e-12

    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        codes = np.arange(start, stop, dtype=np.uint64)[:, None]
        bits = ((codes >> bit_positions) & 1).astype(np.int8)
        signs = bits * 2 - 1
        means = (signs * array[None, :]).mean(axis=1)
        extreme += int(np.count_nonzero(np.abs(means) + tolerance >= observed))

    return float(extreme / total)


def monte_carlo_sign_flip_pvalue(
    values: Iterable[float] | np.ndarray,
    *,
    samples: int = DEFAULT_MONTE_CARLO_SIGN_FLIP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> float:
    """Two-sided Monte Carlo paired sign-flip p-value with +1 correction."""

    array = _as_finite_1d(values, name="paired differences")
    if samples <= 0:
        raise PaperStatisticsError("Monte Carlo sign-flip samples must be positive")

    observed = abs(float(array.mean()))
    rng = np.random.default_rng(seed)
    remaining = samples
    extreme = 0
    tolerance = 1e-12
    while remaining:
        current = min(remaining, 10_000)
        signs = rng.integers(0, 2, size=(current, array.size), dtype=np.int8) * 2 - 1
        means = (signs * array[None, :]).mean(axis=1)
        extreme += int(np.count_nonzero(np.abs(means) + tolerance >= observed))
        remaining -= current
    return float((extreme + 1) / (samples + 1))


def sign_flip_pvalue(
    values: Iterable[float] | np.ndarray,
    *,
    exact_max_pairs: int = MAX_EXACT_SIGN_FLIP_PAIRS,
    monte_carlo_samples: int = DEFAULT_MONTE_CARLO_SIGN_FLIP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> float:
    """Use the frozen exact test when feasible, otherwise the declared MC fallback."""

    array = _as_finite_1d(values, name="paired differences")
    if array.size <= exact_max_pairs:
        return exact_sign_flip_pvalue(array)
    return monte_carlo_sign_flip_pvalue(array, samples=monte_carlo_samples, seed=seed)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down family-wise error correction, returned in input order."""

    if not p_values:
        raise PaperStatisticsError("Holm adjustment requires at least one p-value")
    values = [float(value) for value in p_values]
    if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise PaperStatisticsError("Holm p-values must be finite and lie in [0, 1]")

    m = len(values)
    order = sorted(range(m), key=lambda index: values[index])
    adjusted = [1.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_probability_of_improvement(
    left: Iterable[float] | np.ndarray,
    right: Iterable[float] | np.ndarray,
) -> float:
    """Probability that the left member of a matched seed pair improves on the right.

    This is the paired paper definition: wins receive mass 1, losses 0 and exact
    ties 0.5.  It therefore respects the frozen training-seed pairing rather
    than constructing all cross-seed pairs.
    """

    left_array, right_array = _as_paired(left, right)
    differences = left_array - right_array
    wins = np.count_nonzero(differences > 0.0)
    ties = np.count_nonzero(differences == 0.0)
    return float((wins + 0.5 * ties) / differences.size)


def paired_contrast(
    left: Iterable[float] | np.ndarray,
    right: Iterable[float] | np.ndarray,
    *,
    left_name: str,
    right_name: str,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> PairedContrastResult:
    """Compute primary and preregistered secondary statistics for one paired contrast."""

    left_array, right_array = _as_paired(left, right)
    differences = left_array - right_array
    mean_low, mean_high = percentile_bootstrap_ci(
        differences,
        statistic="mean",
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    iqm_low, iqm_high = percentile_bootstrap_ci(
        differences,
        statistic="iqm",
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    std = float(np.std(differences, ddof=1)) if differences.size > 1 else 0.0
    return PairedContrastResult(
        contrast=f"{left_name} - {right_name}",
        left=left_name,
        right=right_name,
        n=int(differences.size),
        paired_mean_difference=float(np.mean(differences)),
        paired_std_difference=std,
        paired_median_difference=float(np.median(differences)),
        ci95_low=mean_low,
        ci95_high=mean_high,
        wins_left=int(np.count_nonzero(differences > 0.0)),
        ties=int(np.count_nonzero(differences == 0.0)),
        sign_flip_p=sign_flip_pvalue(differences, seed=bootstrap_seed),
        iqm_difference=interquartile_mean(differences),
        iqm_ci95_low=iqm_low,
        iqm_ci95_high=iqm_high,
        probability_of_improvement=paired_probability_of_improvement(left_array, right_array),
    )


def evaluate_holm_family(
    values_by_condition: Mapping[str, Iterable[float] | np.ndarray],
    contrasts: Sequence[ContrastSpec],
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[PairedContrastResult, ...]:
    """Evaluate one frozen contrast family and apply Holm within that family only.

    The same base bootstrap seed is used for both environments.  A deterministic
    ``+ index`` offset gives each ordered contrast its own stream while keeping
    regeneration stable and independent of call order outside the family.
    """

    if not contrasts:
        raise PaperStatisticsError("A Holm family must contain at least one contrast")
    if not 0.0 < alpha < 1.0:
        raise PaperStatisticsError("alpha must lie strictly between 0 and 1")

    raw: list[PairedContrastResult] = []
    for index, spec in enumerate(contrasts):
        if spec.left not in values_by_condition or spec.right not in values_by_condition:
            raise PaperStatisticsError(
                f"Missing condition for {spec.expression}: need {spec.left!r} and {spec.right!r}"
            )
        raw.append(
            paired_contrast(
                values_by_condition[spec.left],
                values_by_condition[spec.right],
                left_name=spec.left,
                right_name=spec.right,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + index,
            )
        )

    adjusted = holm_adjust([item.sign_flip_p for item in raw])
    return tuple(
        PairedContrastResult(
            **{
                **item.__dict__,
                "holm_p": adjusted[index],
                "holm_reject_0_05": bool(adjusted[index] < alpha),
            }
        )
        for index, item in enumerate(raw)
    )
