#!/usr/bin/env python3
"""Validate the saved result surface and regenerate all manuscript artefacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_CREATION_DATE = re.compile(rb"/CreationDate \(D:\d{14}Z\)")


def _env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src if not env.get("PYTHONPATH") else src + os.pathsep + env["PYTHONPATH"]
    return env


def _run(*args: str) -> None:
    command = [sys.executable, *args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=_env(), check=True)


def _sha256(path: Path, *, normalise_pdf: bool = False) -> str:
    payload = path.read_bytes()
    if normalise_pdf:
        payload = _CREATION_DATE.sub(b"/CreationDate (D:00000000000000Z)", payload)
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "build/paper")
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out

    # Prove that the saved evidence and the derived statistical surface agree.
    _run("scripts/generate_lunarlander_tables.py", "--check")
    _run("scripts/generate_minatar_tables.py", "--check")
    _run("scripts/generate_analysis_tables.py", "--check")
    _run("scripts/validate_frozen_results.py", "--environment", "all")

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    _run("figures/make_paper_tables.py", "--root", str(ROOT), "--out", str(out))
    _run("figures/plot_paper_figures.py", "--root", str(ROOT), "--out", str(out / "figures"))
    _run("figures/make_fig1.py", "--out", str(out / "figures"))

    expected = json.loads(
        (ROOT / "tests/paper_expected_sha256.json").read_text(encoding="utf-8")
    )
    failures: list[str] = []
    for relative, wanted in expected.items():
        path = out / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        got = _sha256(path, normalise_pdf=path.suffix.lower() == ".pdf")
        if got != wanted:
            failures.append(f"hash mismatch: {relative}: expected {wanted}, got {got}")
    if failures:
        raise SystemExit(
            "Paper reproduction FAILED\n" + "\n".join(f"- {item}" for item in failures)
        )
    print(
        f"Paper reproduction PASS: {len(expected)} manuscript artefacts "
        "match the saved paper results."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
