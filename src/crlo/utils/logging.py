"""Append-flush JSON / JSONL logger with atomic-rename single-event writes.

Two APIs:

- :meth:`RunLogger.log_event` — atomic single-shot JSON dump (used for
  gating decisions, eval summaries, final scores).
- :meth:`RunLogger.log_step` — append a single record to a JSONL stream,
  flushed per write so a killed run loses at most the in-flight line.

All paths are resolved against the run's ``base_dir``.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, TextIO


class RunLogger:
    """Per-run logger backed by JSON / JSONL files under ``base_dir``."""

    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._streams: dict[str, TextIO] = {}

    def log_event(self, name: str, payload: Mapping[str, Any]) -> Path:
        """Atomic JSON dump to ``base_dir/{name}.json``.

        Writes to a ``.tmp`` sibling then ``rename``s to make the visible
        file flip atomically.
        """
        path = self.base_dir / f"{name}.json"
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(dict(payload), indent=2, default=_default))
        tmp.replace(path)
        return path

    def log_step(self, stream: str, payload: Mapping[str, Any]) -> None:
        """Append one record to ``base_dir/{stream}.jsonl`` and flush."""
        fh = self._streams.get(stream)
        if fh is None:
            fh = (self.base_dir / f"{stream}.jsonl").open("a")
            self._streams[stream] = fh
        fh.write(json.dumps(dict(payload), default=_default))
        fh.write("\n")
        fh.flush()

    def close(self) -> None:
        """Close every open JSONL stream."""
        for fh in self._streams.values():
            fh.close()
        self._streams.clear()

    def __enter__(self) -> RunLogger:
        """Context-manager entry returns ``self``."""
        return self

    def __exit__(self, *_: Any) -> None:
        """Close all open streams on exit."""
        self.close()


def _default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON-serialisable")


def log_run_config(config: Mapping[str, Any], *, command: str, argv: list[str] | None) -> None:
    """Record resolved arguments and the observed environment without changing it."""
    output = config.get("output_dir")
    if not output:
        return
    versions = {}
    for package in ("jax", "numpy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    RunLogger(output).log_event(
        "resolved_config",
        {
            "command": command,
            "config": dict(config),
            "argv": list(sys.argv[1:] if argv is None else argv),
            "runtime": {
                "cpu_affinity": sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None,
                "environment": {
                    key: os.environ.get(key)
                    for key in (
                        "JAX_PLATFORMS",
                        "XLA_PYTHON_CLIENT_PREALLOCATE",
                        "JAX_COMPILATION_CACHE_DIR",
                        "MUJOCO_GL",
                        "XLA_FLAGS",
                        "OMP_NUM_THREADS",
                        "PYTHONUNBUFFERED",
                    )
                },
            },
            "python": sys.version,
            **versions,
        },
    )
