#!/usr/bin/env python3
"""Rerun one LunarLander or MinAtar paper condition."""
from __future__ import annotations

import argparse
from pathlib import Path

from handoff_qrl.experiments import default_output_dir, run_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", required=True, choices=["lunarlander", "minatar"])
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--paired-source",
        type=Path,
        help="Matching H(t) output required for C(0.25), C(0.5) and C(0.75).",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    output = args.output or default_output_dir(args.environment, args.condition, args.seed)
    summary = run_experiment(
        environment=args.environment,
        condition=args.condition,
        seed=args.seed,
        output_dir=output,
        smoke=args.smoke,
        paired_source_dir=args.paired_source,
    )
    print(f"PASS: {summary['environment']} / {summary['condition']} / seed {summary['training_seed']}")
    print(f"output: {Path(output).resolve()}")
    print(f"return mean: {summary['evaluation']['return_mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
