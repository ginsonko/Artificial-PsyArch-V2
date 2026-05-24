# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import read_json


class StorageError(RuntimeError):
    pass


def safe_slug(value: str, *, fallback: str = "item") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or fallback


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True)
class StorageLayout:
    repo_root: Path
    outputs_root: Path
    runs_root: Path


AUTONOMOUS_SESSION_STATUS_RELATIVE_PATH = Path("live") / "autonomous_session_status.json"


def build_storage_layout(repo_root: Path, outputs_root_value: str) -> StorageLayout:
    root = repo_root.resolve()
    outputs_root = Path(outputs_root_value)
    if not outputs_root.is_absolute():
        outputs_root = (root / outputs_root).resolve()
    runs_root = (outputs_root / "runs").resolve()
    outputs_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    return StorageLayout(repo_root=root, outputs_root=outputs_root, runs_root=runs_root)


def make_run_id(prefix: str = "run") -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return f"{safe_slug(prefix, fallback='run')}_{stamp}_{now_ms() % 1000:03d}"


def make_run_dir(layout: StorageLayout, run_id: str) -> Path:
    rid = safe_slug(run_id, fallback="run")
    run_dir = (layout.runs_root / rid).resolve()
    try:
        run_dir.relative_to(layout.runs_root.resolve())
    except ValueError as exc:
        raise StorageError(f"Invalid run id: {run_id}") from exc
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def chunk_bounds(tick_index: int, chunk_size: int) -> tuple[int, int]:
    start = (int(tick_index) // int(chunk_size)) * int(chunk_size)
    end = start + int(chunk_size) - 1
    return start, end


def chunk_file(run_dir: Path, *, kind: str, tick_index: int, chunk_size: int) -> Path:
    start, end = chunk_bounds(tick_index, chunk_size)
    return run_dir / "chunks" / f"ticks_{start:06d}_{end:06d}.{kind}.jsonl"


def read_autonomous_session_status(run_dir: Path) -> dict[str, Any]:
    payload = read_json(run_dir / AUTONOMOUS_SESSION_STATUS_RELATIVE_PATH, default={})
    return payload if isinstance(payload, dict) else {}


def overlay_manifest_with_session_status(manifest: dict[str, Any], session_status: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(manifest or {})
    if not session_status:
        return merged
    session_state = str(session_status.get("status", "") or "").strip()
    if session_state:
        merged["status"] = session_state
    merged["autonomous_session_status_summary"] = {
        "session_id": str(session_status.get("session_id", "") or ""),
        "status": session_state,
        "active": bool(session_status.get("active", False)),
        "paused": bool(session_status.get("paused", False)),
        "stopping": bool(session_status.get("stopping", False)),
        "tick_done": int(session_status.get("tick_done", 0) or 0),
        "max_ticks": int(session_status.get("max_ticks", 0) or 0),
        "recoverable": bool(session_status.get("recoverable", False)),
        "updated_at_ms": int(session_status.get("updated_at_ms", 0) or 0),
        "finished_at_ms": int(session_status.get("finished_at_ms", 0) or 0),
        "last_stop_reason": str(session_status.get("last_stop_reason", "") or ""),
        "goal": dict(session_status.get("session_goal", {}) or {}),
        "health": dict(session_status.get("session_health", {}) or {}),
        "context": dict(session_status.get("session_context", {}) or {}),
        "lifecycle": dict(session_status.get("lifecycle", {}) or {}),
    }
    return merged


def _manifest_run_timestamp_ms(manifest: dict[str, Any], *, directory_mtime_ms: int) -> int:
    updated_at_ms = int(manifest.get("updated_at_ms", 0) or 0)
    if updated_at_ms > 0:
        return updated_at_ms
    finished_at_ms = int(manifest.get("finished_at_ms", 0) or 0)
    if finished_at_ms > 0:
        return finished_at_ms
    created_at_ms = int(manifest.get("created_at_ms", 0) or 0)
    if created_at_ms > 0:
        return created_at_ms
    return int(directory_mtime_ms or 0)


def list_runs(layout: StorageLayout, limit: int = 32) -> list[dict[str, Any]]:
    if not layout.runs_root.exists():
        return []
    dirs = [p for p in layout.runs_root.iterdir() if p.is_dir()]
    rows: list[dict[str, Any]] = []
    for directory in dirs:
        stat_result = directory.stat()
        directory_mtime_ms = int(stat_result.st_mtime * 1000)
        manifest = read_json(directory / "manifest.json", default={}) or {}
        session_status = read_autonomous_session_status(directory)
        display_manifest = overlay_manifest_with_session_status(manifest, session_status)
        created_at_ms = int(manifest.get("created_at_ms", 0) or 0)
        finished_at_ms = int(manifest.get("finished_at_ms", 0) or 0)
        updated_at_ms = int(manifest.get("updated_at_ms", 0) or 0)
        run_timestamp_ms = _manifest_run_timestamp_ms(manifest, directory_mtime_ms=directory_mtime_ms)
        rows.append(
            {
                "run_id": directory.name,
                "status": str(display_manifest.get("status", "unknown") or "unknown"),
                "label": str(display_manifest.get("label", "") or ""),
                "tick_done": int(manifest.get("tick_done", 0) or 0),
                "tick_planned": int(manifest.get("tick_planned", 0) or 0),
                "updated_at_ms": updated_at_ms,
                "finished_at_ms": finished_at_ms,
                "created_at_ms": created_at_ms,
                "run_timestamp_ms": run_timestamp_ms,
                "directory_mtime_ms": directory_mtime_ms,
                "session_status": dict(display_manifest.get("autonomous_session_status_summary", {}) or {}),
                "path": str(directory),
            }
        )
    rows.sort(
        key=lambda item: (
            int(item.get("run_timestamp_ms", 0) or 0),
            int(item.get("updated_at_ms", 0) or 0),
            int(item.get("finished_at_ms", 0) or 0),
            int(item.get("created_at_ms", 0) or 0),
            int(item.get("directory_mtime_ms", 0) or 0),
            str(item.get("run_id", "") or ""),
        ),
        reverse=True,
    )
    return rows[: max(1, int(limit))]
