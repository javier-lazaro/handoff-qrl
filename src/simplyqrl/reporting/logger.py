"""Artifact logger for SimplyQRL runs.

The logger writes artifacts. It does not decide which metrics should exist.
That responsibility belongs to monitors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from simplyqrl.reporting.context import (
    RunContext,
    build_basic_runtime_metadata,
    utc_now_iso,
    write_json,
    write_yaml,
)


class ArtifactLogger:
    """Write canonical run artifacts to disk."""

    def __init__(
        self,
        context: RunContext,
        config: dict[str, Any],
        *,
        repo_root: str | Path | None = None,
    ) -> None:
        self.context = context
        self.config = config
        self.repo_root = repo_root
        self.run_dir = context.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.metadata_path = self.run_dir / "run_metadata.json"
        self.config_path = self.run_dir / "config.yaml"

        self._metrics_file = self.metrics_path.open("a", encoding="utf-8")
        self.summary: dict[str, Any] = {
            "status": "running",
            "created_at": context.created_at,
            "run_id": context.run_id,
            "run_name": context.run_name,
            "config_hash": context.config_hash,
        }

        self.write_config()
        self.write_metadata(status="running")
        self.write_summary()

    def write_config(self) -> None:
        write_yaml(self.config_path, self.config)

    def write_metadata(self, *, status: str) -> None:
        metadata = {
            **self.context.to_dict(),
            "status": status,
            "updated_at": utc_now_iso(),
            "runtime": build_basic_runtime_metadata(self.repo_root),
        }
        write_json(self.metadata_path, metadata)

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "event_type": event_type,
            "logged_at": utc_now_iso(),
            **payload,
        }
        self._metrics_file.write(json.dumps(event, sort_keys=True, default=repr) + "\n")
        self._metrics_file.flush()

    def update_summary(self, payload: dict[str, Any]) -> None:
        self.summary.update(payload)
        self.write_summary()

    def write_summary(self) -> None:
        write_json(self.summary_path, self.summary)

    def close(self, *, status: str = "finished", final_summary: dict[str, Any] | None = None) -> None:
        if final_summary:
            self.summary.update(final_summary)

        self.summary["status"] = status
        self.summary["finished_at"] = utc_now_iso()
        self.write_summary()
        self.write_metadata(status=status)

        if not self._metrics_file.closed:
            self._metrics_file.close()
