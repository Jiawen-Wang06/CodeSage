"""End-to-end request tracing for system evaluation.

Every question produces one Trace: the user input, each pipeline step (history window, query rewrite,
retrieval and its sub-steps, answer generation, citation check) and the final output. Traces are written
as one JSON object per line so an annotation tool can read them without depending on this package.

Privacy: API keys are never passed into a trace. Every string is passed through `redact_secrets` plus
e-mail masking when the trace is serialized, and any extra values given in `scrub` (such as the active
API key) are replaced if they appear anywhere. Error messages are not recorded, only exception types,
because provider errors can echo parts of a key.
"""

import json
import logging
import os
import re
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from codesage.redaction import REDACTED, redact_secrets

SCHEMA_VERSION = 1

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}\b")
_WRITE_LOCK = threading.Lock()
_log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Step:
    """One unit of work inside a trace. Steps can contain child steps."""

    def __init__(self, name: str, kind: str):
        self.name = name
        self.kind = kind
        self.started_at = _now_iso()
        self.duration_ms: float | None = None
        self.status = "ok"
        self.input: dict[str, Any] = {}
        self.output: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self.error: dict[str, str] | None = None
        self.children: list["Step"] = []
        self._t0 = time.perf_counter()

    def finish(self) -> None:
        if self.duration_ms is None:
            self.duration_ms = round((time.perf_counter() - self._t0) * 1000, 2)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "input": self.input,
            "output": self.output,
            "metadata": self.metadata,
        }
        if self.error:
            data["error"] = self.error
        if self.children:
            data["children"] = [child.to_dict() for child in self.children]
        return data


class Trace:
    """Collects everything that happens between a user question and the final answer."""

    def __init__(
        self,
        *,
        session_id: str,
        turn_index: int,
        repo: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        scrub: Iterable[str] = (),
    ):
        self.trace_id = uuid.uuid4().hex
        self.session_id = session_id
        self.turn_index = turn_index
        self.repo = repo or {}
        self.config = config or {}
        self.started_at = _now_iso()
        self.ended_at: str | None = None
        self.duration_ms: float | None = None
        self.status = "ok"
        self.error: dict[str, str] | None = None
        self.input: dict[str, Any] = {}
        self.output: dict[str, Any] = {}
        self.steps: list[Step] = []
        # Held in memory only so they can be removed from the output; never serialized.
        self._scrub = tuple(value for value in scrub if value and len(value) >= 8)
        self._t0 = time.perf_counter()

    def __repr__(self) -> str:
        return f"Trace(trace_id={self.trace_id!r}, status={self.status!r}, steps={len(self.steps)})"

    @contextmanager
    def step(self, name: str, kind: str, parent: Step | None = None) -> Iterator[Step]:
        current = Step(name, kind)
        (parent.children if parent else self.steps).append(current)
        try:
            yield current
        except Exception as exc:
            current.status = "error"
            current.error = {"type": type(exc).__name__}
            raise
        finally:
            current.finish()

    def fail(self, exc: BaseException) -> None:
        self.status = "error"
        self.error = {"type": type(exc).__name__}

    def finish(self) -> None:
        if self.ended_at is None:
            self.ended_at = _now_iso()
            self.duration_ms = round((time.perf_counter() - self._t0) * 1000, 2)

    def to_dict(self) -> dict[str, Any]:
        self.finish()
        data = {
            "schema_version": SCHEMA_VERSION,
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "repo": self.repo,
            "config": self.config,
            "input": self.input,
            "steps": [step.to_dict() for step in self.steps],
            "output": self.output,
        }
        return sanitize(data, self._scrub)


def sanitize(value: Any, scrub: tuple[str, ...] = ()) -> Any:
    """Recursively mask secrets and e-mail addresses in every string of a JSON-like value."""
    if isinstance(value, str):
        for secret in scrub:
            value = value.replace(secret, REDACTED)
        return _EMAIL.sub(REDACTED, redact_secrets(value))
    if isinstance(value, dict):
        return {str(key): sanitize(item, scrub) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item, scrub) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize(str(value), scrub)


def instance_name() -> str:
    """A short id for this server instance, so scaled-out instances never append to the same file."""
    raw = os.getenv("WEBSITE_INSTANCE_ID") or socket.gethostname() or "local"
    return re.sub(r"[^A-Za-z0-9]", "", raw)[:12] or "local"


def write_trace(trace: Trace, directory: Path, retention_days: int | None = None) -> Path | None:
    """Append the trace to a daily, per-instance JSONL file. Tracing must never break a user request."""
    try:
        line = json.dumps(trace.to_dict(), ensure_ascii=False)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"traces-{datetime.now(timezone.utc):%Y-%m-%d}-{instance_name()}.jsonl"
        with _WRITE_LOCK, path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if retention_days:
            _prune_once_per_day(directory, retention_days)
        return path
    except Exception as exc:  # noqa: BLE001
        _log.warning("Could not write trace %s (%s)", trace.trace_id, type(exc).__name__)
        return None


_last_prune: dict[str, str] = {}


def _prune_once_per_day(directory: Path, retention_days: int) -> None:
    today = datetime.now(timezone.utc).date()
    key = str(directory.resolve())
    if _last_prune.get(key) == today.isoformat():
        return
    _last_prune[key] = today.isoformat()
    cutoff = today - timedelta(days=retention_days)
    for path in directory.glob("traces-*.jsonl"):
        match = re.match(r"traces-(\d{4}-\d{2}-\d{2})", path.name)
        if match and datetime.strptime(match[1], "%Y-%m-%d").date() < cutoff:
            try:
                path.unlink()
            except OSError:
                pass


def read_traces(directory: Path) -> Iterator[dict[str, Any]]:
    """Yield every trace stored in `directory`, oldest file first."""
    for path in sorted(directory.glob("traces-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
