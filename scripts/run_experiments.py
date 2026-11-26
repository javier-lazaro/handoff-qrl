#!/usr/bin/env python3
"""Rerun a paper experiment matrix sequentially."""
from __future__ import annotations

import argparse
from pathlib import Path

from handoff_qrl.experiments import CONDITIONS, condition_slug, run_experiment
from handoff_qrl.paper_matrix import MATRIX_ORDER, run_seed_matrix


def _parse_seeds(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            start, end = int(start_raw), int(end_raw)
            if end < start:
                raise argparse.ArgumentTypeError(f"Invalid seed range: {token}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    if not values:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return sorted(set(values))


def _parse_conditions(raw: str) -> tuple[list[str], bool]:
    if raw.strip().lower() == "all":
        return list(MATRIX_ORDER), True
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in requested if item not in CONDITIONS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown conditions: {', '.join(unknown)}")
    if not requested:
        raise argparse.ArgumentTypeError("At least one condition is required")
    order = {name: index for index, name in enumerate(MATRIX_ORDER)}
    return sorted(set(requested), key=order.__getitem__), False


def _output(root: Path, environment: str, condition: str, seed: int) -> Path:
    return root / environment / condition_slug(condition) / f"seed{seed}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun the paper matrix. With --conditions all, cross-condition dependencies are "
            "preserved exactly: shared MinAtar teacher milestones, byte-identical H/C KD data, "
            "and the original LunarLander H(1) teacher reuse."
        )
    )
    parser.add_argument("--environment", required=True, choices=["lunarlander", "minatar"])
    parser.add_argument("--seeds", type=_parse_seeds, default=_parse_seeds("0-19"))
    parser.add_argument("--conditions", default="all")
    parser.add_argument("--output-root", type=Path, default=Path("runs/reproduction"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    conditions, full_matrix = _parse_conditions(args.conditions)

    print(f"environment: {args.environment}")
    print(f"seeds: {args.seeds}")
    print(f"conditions: {conditions}")
    print(f"full protocol matrix: {full_matrix}")

    if args.dry_run:
        for seed in args.seeds:
            for condition in conditions:
                print(f"seed {seed}: {condition} -> {_output(args.output_root, args.environment, condition, seed)}")
        if full_matrix:
            print("dependencies: exact cross-condition sharing enabled")
        return 0

    if full_matrix:
        for index, seed in enumerate(args.seeds, start=1):
            print(f"[seed {index}/{len(args.seeds)}] {seed}", flush=True)
            run_seed_matrix(
                environment=args.environment,
                seed=seed,
                output_root=args.output_root,
                smoke=args.smoke,
                resume=args.resume,
            )
        print("PASS")
        return 0

    # A requested subset stays simple. Paired C(t) conditions require their
    # matching H(t) output to exist under the same output root.
    for seed in args.seeds:
        for condition in conditions:
            output = _output(args.output_root, args.environment, condition, seed)
            if args.resume and (output / "experiment_summary.json").is_file():
                print(f"SKIP {condition} seed {seed}: summary exists")
                continue
            paired_source = None
            if condition.startswith("C(") and condition not in {"C(0)"}:
                t = float(condition[2:-1])
                paired_source = _output(args.output_root, args.environment, f"H({t:g})", seed)
                if not (paired_source / "experiment_summary.json").is_file():
                    raise RuntimeError(
                        f"{condition} seed {seed} requires its matching hybrid run first: "
                        f"{paired_source}. Include H({t:g}) in --conditions or use --conditions all."
                    )
            print(f"RUN {condition} seed {seed}", flush=True)
            run_experiment(
                environment=args.environment,
                condition=condition,
                seed=seed,
                output_dir=output,
                smoke=args.smoke,
                paired_source_dir=paired_source,
            )
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
