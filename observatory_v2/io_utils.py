# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Iterator


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    temp_path = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                os.replace(temp_path, path)
                last_error = None
                break
            except PermissionError as exc:
                last_error = exc
                if attempt >= 5:
                    raise
                time.sleep(0.01 * (attempt + 1))
        if last_error is not None:
            raise last_error
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                yield payload
