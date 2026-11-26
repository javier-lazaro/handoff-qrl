#!/usr/bin/env python3
"""Fail if private workspace metadata or legacy release layout leaks into the public tree."""
from __future__ import annotations

import gzip
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "build", "dist", "runs", "__pycache__"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".csv", ".cff", ".tex", ".gitignore"}
PATTERN_SOURCES = (
    ("absolute user path", r"/(?:home|Users|mnt|scratch|tmp)/"),
    ("private VCS transport", r"(?:git\+ssh://|git@github\.com|^-e\s+git\+)"),
    ("development run/config path", r"(?:runs|configs)/s\d+/"),
    ("development sprint marker", r"(?:^|[^A-Za-z0-9])s\d{2}(?:[^A-Za-z0-9]|$)"),
    ("legacy research-track marker", r"rt3"),
    ("raw Git revision", r"\b[0-9a-f]{40}\b"),
    ("scheduler/infrastructure marker", r"\bslurm\b"),
    ("private key", r"BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY"),
    (
        "legacy public-data filename",
        r"\b(?:main_matrix|classical_delayed_matrix|c0_matrix|c0_episodes|"
        r"c0_training_episode_returns|c0_training_update_metrics|kd_metrics|"
        r"mechanistic_decomposition|audit)\.(?:csv(?:\.gz)?|json)\b",
    ),
    (
        "legacy public-data column",
        r"\b(?:paper_label|qforward_grad_kd|qforward_nograd_kd|"
        r"qforward_grad_ppo|qforward_nograd_ppo)\b",
    ),
)
PATTERN_LABELS = tuple(label for label, _ in PATTERN_SOURCES)
COMBINED_PATTERN = re.compile(
    "|".join(f"(?P<p{index}>{source})" for index, (_, source) in enumerate(PATTERN_SOURCES)),
    re.I,
)
FORBIDDEN_PATHS = {
    "environment/original-pip-freeze.txt",
    "analysis/training_environment.json",
    "analysis/rt3",
    "results/paper",
    "RELEASE_SANITIZATION.json",
    "configs/lunarlander/development_freeze.yaml",
    "results/lunarlander/provenance",
    "results/minatar/provenance",
}


def _iter_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        yield path


def _scan_lines(path: Path):
    if path.name.endswith(".csv.gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
            yield from handle
    elif path.name == ".gitignore" or path.suffix.lower() in TEXT_SUFFIXES:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            yield from handle


def main() -> int:
    errors: list[str] = []
    for forbidden in sorted(FORBIDDEN_PATHS):
        if (ROOT / forbidden).exists():
            errors.append(f"forbidden development/legacy artifact: {forbidden}")

    for path in _iter_files():
        rel = path.relative_to(ROOT)
        if any("rt3" in part.lower() for part in rel.parts):
            errors.append(f"legacy research-track path: {rel}")
            continue
        for lineno, line in enumerate(_scan_lines(path), start=1):
            match = COMBINED_PATTERN.search(line)
            if match is None:
                continue
            assert match.lastgroup is not None
            label = PATTERN_LABELS[int(match.lastgroup[1:])]
            errors.append(f"{label}: {rel}:{lineno}")
            break

    if errors:
        raise SystemExit("Release hygiene FAILED\n" + "\n".join(f"- {x}" for x in sorted(set(errors))))
    print("Release hygiene PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
