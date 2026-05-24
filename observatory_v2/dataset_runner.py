# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path
from typing import Any

from .app import ObservatoryV2App
from .config import load_config, repo_root


CONTROL_MODES = {
    "autonomous_session_status",
    "pause_autonomous_session",
    "resume_autonomous_session",
    "stop_autonomous_session",
}

ASYNC_SESSION_MODES = {
    "autonomous_session",
    "recover_autonomous_session",
}


def _resolve_path(raw_path: str, *, dataset_path: Path) -> Path:
    clean = str(raw_path or "").strip()
    if not clean:
        raise ValueError("path is empty")
    candidate = Path(clean)
    if not candidate.is_absolute():
        candidate = (dataset_path.parent / candidate).resolve()
    return candidate


def _read_bytes_from_path(raw_path: str, *, dataset_path: Path) -> bytes:
    candidate = _resolve_path(raw_path, dataset_path=dataset_path)
    if not candidate.exists():
        raise FileNotFoundError(f"dataset file not found: {candidate}")
    return candidate.read_bytes()


def _decode_inline_bytes(value: str, *, field_name: str) -> bytes:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError(f"{field_name} is empty")
    try:
        return base64.b64decode(clean, validate=True)
    except Exception as exc:  # pragma: no cover - exact text varies
        raise ValueError(f"{field_name} is not valid base64") from exc


def _load_binary_blob(
    row: dict[str, Any],
    *,
    inline_key: str,
    path_key: str,
    dataset_path: Path,
) -> bytes | None:
    inline_value = row.get(inline_key)
    if inline_value is not None:
        return _decode_inline_bytes(str(inline_value), field_name=inline_key)
    path_value = row.get(path_key)
    if path_value is not None:
        return _read_bytes_from_path(str(path_value), dataset_path=dataset_path)
    return None


def _normalize_feedback(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("external_feedback must be an object")
    return dict(raw)


def _build_multimodal_item(raw: Any, *, dataset_path: Path) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"text": str(raw or ""), "source_type": "external_text"}
    item: dict[str, Any] = {
        "text": str(raw.get("text", "") or ""),
        "source_type": str(raw.get("source_type", "multimodal_input") or "multimodal_input"),
    }
    image_bytes = _load_binary_blob(raw, inline_key="image_b64", path_key="image_path", dataset_path=dataset_path)
    if image_bytes is not None:
        item["image_bytes"] = image_bytes
    audio_bytes = _load_binary_blob(raw, inline_key="audio_b64", path_key="audio_path", dataset_path=dataset_path)
    if audio_bytes is not None:
        item["audio_bytes"] = audio_bytes
    if bool(raw.get("capture_screen", False)):
        item["capture_screen"] = True
    feedback = _normalize_feedback(raw.get("external_feedback"))
    if feedback:
        item["external_feedback"] = feedback
    if isinstance(raw.get("stream_source_meta"), dict):
        item["stream_source_meta"] = dict(raw.get("stream_source_meta", {}) or {})
    if isinstance(raw.get("stream_frame_meta"), dict):
        item["stream_frame_meta"] = dict(raw.get("stream_frame_meta", {}) or {})
    if isinstance(raw.get("stream_chunk_meta"), dict):
        item["stream_chunk_meta"] = dict(raw.get("stream_chunk_meta", {}) or {})
    return item


def _normalize_texts(payload: list[Any]) -> list[str]:
    return [str(item or "") for item in payload]


def _normalize_multimodal_items(payload: list[Any], *, dataset_path: Path) -> list[dict[str, Any]]:
    return [_build_multimodal_item(item, dataset_path=dataset_path) for item in payload]


def _resolve_optional_int(raw: Any) -> int | None:
    if raw is None or raw == "":
        return None
    return int(raw)


def _resolve_optional_float(raw: Any) -> float | None:
    if raw is None or raw == "":
        return None
    return float(raw)


def _resolve_optional_bool(raw: Any) -> bool | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        return raw
    clean = str(raw).strip().lower()
    if clean in {"1", "true", "yes", "on"}:
        return True
    if clean in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"cannot parse bool value: {raw}")


def _resolve_headers(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("external_teacher_http_headers must be an object")
    return dict(raw)


def _resolve_reward_schedule(raw: Any) -> list[dict[str, Any]] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("reward_schedule must be a list")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"reward_schedule[{index}] must be an object")
        rows.append(
            {
                "tick_index": int(item.get("tick_index", 0) or 0),
                "reward": float(item.get("reward", 0.0) or 0.0),
                "punishment": float(item.get("punishment", 0.0) or 0.0),
            }
        )
    return rows


def _hook_value(payload: dict[str, Any], *, section: str, key: str) -> Any:
    block = payload.get(section, {})
    if isinstance(block, dict) and key in block:
        return block.get(key)
    return payload.get(key)


def _normalize_wait_hook(raw: Any) -> dict[str, Any] | None:
    if raw in (None, False, ""):
        return None
    if raw is True:
        return {"timeout_sec": 120.0, "stop_on_timeout": False, "poll_interval_ms": 50}
    if isinstance(raw, (int, float)):
        return {"timeout_sec": float(raw), "stop_on_timeout": False, "poll_interval_ms": 50}
    if not isinstance(raw, dict):
        raise ValueError("wait_for_session must be bool/number/object")
    return {
        "timeout_sec": float(raw.get("timeout_sec", 120.0) or 120.0),
        "stop_on_timeout": bool(raw.get("stop_on_timeout", False)),
        "poll_interval_ms": int(raw.get("poll_interval_ms", 50) or 50),
    }


def _normalize_session_control_hook(raw: Any) -> dict[str, Any] | None:
    if raw in (None, False, ""):
        return None
    if raw is True:
        return {"delay_ms": 0, "timeout_sec": 20.0, "poll_interval_ms": 50}
    if isinstance(raw, (int, float)):
        return {"delay_ms": int(raw), "timeout_sec": 20.0, "poll_interval_ms": 50}
    if not isinstance(raw, dict):
        raise ValueError("session control hook must be bool/number/object")
    return {
        "delay_ms": int(raw.get("delay_ms", 0) or 0),
        "timeout_sec": float(raw.get("timeout_sec", 20.0) or 20.0),
        "poll_interval_ms": int(raw.get("poll_interval_ms", 50) or 50),
    }


def _normalize_status_snapshot_hook(raw: Any, *, dataset_path: Path) -> dict[str, Any] | None:
    if raw in (None, False, ""):
        return None
    if raw is True:
        return {"path": None}
    if isinstance(raw, str):
        return {"path": str(_resolve_path(raw, dataset_path=dataset_path))}
    if not isinstance(raw, dict):
        raise ValueError("status_snapshot must be bool/string/object")
    path_value = raw.get("path")
    return {
        "path": (str(_resolve_path(str(path_value), dataset_path=dataset_path)) if path_value else None),
    }


def _normalize_forget_hook(raw: Any) -> dict[str, Any] | None:
    if raw in (None, False, ""):
        return None
    if isinstance(raw, (int, float, str)):
        return {"keep_latest": int(raw)}
    if not isinstance(raw, dict):
        raise ValueError("forget must be number/string/object")
    keep_latest = raw.get("keep_latest", raw.get("forget_keep_latest", 128))
    protect_memory_kinds = raw.get("protect_memory_kinds", [])
    if protect_memory_kinds is None:
        protect_memory_kinds = []
    if not isinstance(protect_memory_kinds, list):
        raise ValueError("forget.protect_memory_kinds must be a list")
    max_memory_count = raw.get("max_memory_count")
    clean: dict[str, Any] = {
        "keep_latest": int(keep_latest or 0),
        "strategy": str(raw.get("strategy", "latest_only") or "latest_only"),
        "min_reality_weight": float(raw.get("min_reality_weight", 0.0) or 0.0),
        "min_total_item_energy": float(raw.get("min_total_item_energy", 0.0) or 0.0),
        "protect_memory_kinds": [str(item or "") for item in protect_memory_kinds if str(item or "")],
        "dry_run": bool(raw.get("dry_run", False)),
    }
    if max_memory_count not in (None, "", False):
        clean["max_memory_count"] = int(max_memory_count)
    return clean


def _resolve_hooks(payload: dict[str, Any], *, dataset_path: Path) -> dict[str, Any]:
    checkpoint_out_alias = _hook_value(payload, section="after", key="export_runtime_path")
    checkpoint_out = _hook_value(payload, section="after", key="save_checkpoint_path")
    if checkpoint_out_alias and checkpoint_out and str(checkpoint_out_alias) != str(checkpoint_out):
        raise ValueError("save_checkpoint_path and export_runtime_path disagree")
    checkpoint_target = checkpoint_out if checkpoint_out is not None else checkpoint_out_alias
    hooks = {
        "before": {
            "import_runtime_path": None,
            "import_memory_bundle_dir": None,
        },
        "after": {
            "save_checkpoint_path": None,
            "export_memory_bundle_dir": None,
            "inspect_memory_bundle": bool(_hook_value(payload, section="after", key="inspect_memory_bundle")),
            "forget_keep_latest": _resolve_optional_int(_hook_value(payload, section="after", key="forget_keep_latest")),
            "forget": _normalize_forget_hook(_hook_value(payload, section="after", key="forget")),
            "wait_for_session": _normalize_wait_hook(_hook_value(payload, section="after", key="wait_for_session")),
            "status_snapshot": _normalize_status_snapshot_hook(_hook_value(payload, section="after", key="status_snapshot"), dataset_path=dataset_path),
            "pause_session": _normalize_session_control_hook(_hook_value(payload, section="after", key="pause_session")),
            "resume_session": _normalize_session_control_hook(_hook_value(payload, section="after", key="resume_session")),
            "stop_session": _normalize_session_control_hook(_hook_value(payload, section="after", key="stop_session")),
        },
    }
    import_runtime_path = _hook_value(payload, section="before", key="import_runtime_path")
    if import_runtime_path:
        hooks["before"]["import_runtime_path"] = str(_resolve_path(str(import_runtime_path), dataset_path=dataset_path))
    import_memory_bundle_dir = _hook_value(payload, section="before", key="import_memory_bundle_dir")
    if import_memory_bundle_dir:
        hooks["before"]["import_memory_bundle_dir"] = str(_resolve_path(str(import_memory_bundle_dir), dataset_path=dataset_path))
    if checkpoint_target:
        hooks["after"]["save_checkpoint_path"] = str(_resolve_path(str(checkpoint_target), dataset_path=dataset_path))
    export_memory_bundle_dir = _hook_value(payload, section="after", key="export_memory_bundle_dir")
    if export_memory_bundle_dir:
        hooks["after"]["export_memory_bundle_dir"] = str(_resolve_path(str(export_memory_bundle_dir), dataset_path=dataset_path))
    return hooks


def _parse_autonomous_common(payload: dict[str, Any]) -> dict[str, Any]:
    teacher_mode = str(payload.get("teacher_mode", "") or "").strip() or None
    llm_gate_mode = str(payload.get("llm_gate_mode", "") or "").strip() or None
    external_teacher_mode = str(payload.get("external_teacher_mode", "") or "").strip() or None
    return {
        "text_hint": str(payload.get("text_hint", "") or ""),
        "stop_on_capture_failures": _resolve_optional_int(payload.get("stop_on_capture_failures")),
        "stop_on_action_errors": _resolve_optional_int(payload.get("stop_on_action_errors")),
        "stop_on_idle_ticks": _resolve_optional_int(payload.get("stop_on_idle_ticks")),
        "idle_backoff_ms": _resolve_optional_int(payload.get("idle_backoff_ms")),
        "auto_feedback_enabled": _resolve_optional_bool(payload.get("auto_feedback_enabled")),
        "teacher_mode": teacher_mode,
        "llm_gate_mode": llm_gate_mode,
        "external_teacher_enabled": _resolve_optional_bool(payload.get("external_teacher_enabled")),
        "external_teacher_mode": external_teacher_mode,
        "external_teacher_stub_response_path": (
            str(payload.get("external_teacher_stub_response_path", "") or "").strip() or None
        ),
        "external_teacher_fail_open": _resolve_optional_bool(payload.get("external_teacher_fail_open")),
        "external_teacher_max_retries": _resolve_optional_int(payload.get("external_teacher_max_retries")),
        "external_teacher_retry_backoff_ms": _resolve_optional_int(payload.get("external_teacher_retry_backoff_ms")),
        "external_teacher_http_endpoint": (
            str(payload.get("external_teacher_http_endpoint", "") or "").strip() or None
        ),
        "external_teacher_http_headers": _resolve_headers(payload.get("external_teacher_http_headers")),
    }


def _parse_run_spec(payload: dict[str, Any], *, dataset_path: Path, default_label: str) -> dict[str, Any]:
    mode = str(payload.get("mode", "") or "").strip().lower()
    tick_interval_ms = int(payload.get("tick_interval_ms", 0) or 0)
    reset_runtime = bool(payload.get("reset_runtime", False))
    delay_ms = int(payload.get("delay_ms", 0) or 0)
    label = str(payload.get("label", "") or "").strip() or default_label
    hooks = _resolve_hooks(payload, dataset_path=dataset_path)

    if hooks["before"]["import_runtime_path"] and reset_runtime:
        raise ValueError("import_runtime_path cannot be combined with reset_runtime=true")
    if hooks["before"]["import_memory_bundle_dir"] and reset_runtime:
        raise ValueError("import_memory_bundle_dir cannot be combined with reset_runtime=true")

    if isinstance(payload.get("texts"), list):
        return {
            "mode": "text",
            "texts": _normalize_texts(list(payload.get("texts", []) or [])),
            "label": label,
            "tick_interval_ms": tick_interval_ms,
            "reset_runtime": reset_runtime,
            "delay_ms": delay_ms,
            "hooks": hooks,
        }

    if isinstance(payload.get("items"), list):
        return {
            "mode": "multimodal",
            "items": _normalize_multimodal_items(list(payload.get("items", []) or []), dataset_path=dataset_path),
            "label": label,
            "tick_interval_ms": tick_interval_ms,
            "reset_runtime": reset_runtime,
            "delay_ms": delay_ms,
            "hooks": hooks,
        }

    if not mode:
        raise ValueError("dataset format unsupported")

    common = {
        "mode": mode,
        "label": label,
        "tick_interval_ms": tick_interval_ms,
        "reset_runtime": reset_runtime,
        "delay_ms": delay_ms,
        "hooks": hooks,
    }

    if mode == "continue_from_checkpoint":
        checkpoint_path = payload.get("checkpoint_path")
        if checkpoint_path is None:
            raise ValueError("continue_from_checkpoint requires checkpoint_path")
        common.update(
            {
                "checkpoint_path": str(_resolve_path(str(checkpoint_path), dataset_path=dataset_path)),
                "texts": _normalize_texts(list(payload.get("texts", []) or [])),
            }
        )
        return common

    if mode == "audio_stream":
        audio_bytes = _load_binary_blob(payload, inline_key="audio_b64", path_key="audio_path", dataset_path=dataset_path)
        if audio_bytes is None:
            raise ValueError("audio_stream dataset requires audio_b64 or audio_path")
        common.update(
            {
                "audio_bytes": audio_bytes,
                "text_prefix": str(payload.get("text_prefix", "") or ""),
                "tick_window_ms": _resolve_optional_int(payload.get("tick_window_ms")),
            }
        )
        return common

    if mode == "image_stream":
        frame_paths = payload.get("frame_paths", []) or []
        frames_b64 = payload.get("frames_b64", []) or []
        frame_bytes_list: list[bytes] = []
        if frame_paths:
            if not isinstance(frame_paths, list):
                raise ValueError("frame_paths must be a list")
            frame_bytes_list = [_read_bytes_from_path(str(path), dataset_path=dataset_path) for path in frame_paths]
        elif frames_b64:
            if not isinstance(frames_b64, list):
                raise ValueError("frames_b64 must be a list")
            frame_bytes_list = [_decode_inline_bytes(str(value), field_name="frames_b64") for value in frames_b64]
        strip_image_bytes = _load_binary_blob(payload, inline_key="strip_image_b64", path_key="strip_image_path", dataset_path=dataset_path)
        if not frame_bytes_list and strip_image_bytes is None:
            raise ValueError("image_stream dataset requires frame_paths / frames_b64 / strip_image_b64 / strip_image_path")
        common.update(
            {
                "frame_bytes_list": frame_bytes_list or None,
                "strip_image_bytes": strip_image_bytes,
                "frame_count": _resolve_optional_int(payload.get("frame_count")),
                "text_prefix": str(payload.get("text_prefix", "") or ""),
            }
        )
        return common

    if mode == "video_stream":
        video_bytes = _load_binary_blob(payload, inline_key="video_b64", path_key="video_path", dataset_path=dataset_path)
        if video_bytes is None:
            raise ValueError("video_stream dataset requires video_b64 or video_path")
        video_path = str(payload.get("video_path", "") or "")
        common.update(
            {
                "video_bytes": video_bytes,
                "video_name": str(payload.get("video_name", "") or (Path(video_path).name if video_path else "")),
                "text_prefix": str(payload.get("text_prefix", "") or ""),
                "tick_fps": _resolve_optional_float(payload.get("tick_fps")),
                "frame_stride": _resolve_optional_int(payload.get("frame_stride")),
                "max_frames": _resolve_optional_int(payload.get("max_frames")),
            }
        )
        return common

    if mode == "webcam_stream":
        common.update(
            {
                "text_prefix": str(payload.get("text_prefix", "") or ""),
                "max_frames": _resolve_optional_int(payload.get("max_frames")),
                "device_index": int(payload.get("device_index", 0) or 0),
                "frame_width": _resolve_optional_int(payload.get("frame_width")),
                "frame_height": _resolve_optional_int(payload.get("frame_height")),
            }
        )
        return common

    if mode == "microphone_stream":
        common.update(
            {
                "text_prefix": str(payload.get("text_prefix", "") or ""),
                "max_windows": _resolve_optional_int(payload.get("max_windows")),
                "tick_window_ms": _resolve_optional_int(payload.get("tick_window_ms")),
                "sample_rate": int(payload.get("sample_rate", 16000) or 16000),
                "channels": int(payload.get("channels", 1) or 1),
                "device_index": payload.get("device_index"),
            }
        )
        return common

    if mode == "autonomous_run":
        common.update(_parse_autonomous_common(payload))
        common.update(
            {
                "ticks": max(1, int(payload.get("ticks", 4) or 4)),
                "reward_schedule": _resolve_reward_schedule(payload.get("reward_schedule")),
            }
        )
        return common

    if mode == "autonomous_session":
        common.update(_parse_autonomous_common(payload))
        common.update(
            {
                "max_ticks": _resolve_optional_int(payload.get("max_ticks")),
                "wait_for_completion": bool(payload.get("wait_for_completion", False)),
            }
        )
        return common

    if mode == "recover_autonomous_session":
        common.update(
            {
                "run_id": str(payload.get("run_id", "") or "").strip() or None,
                "wait_for_completion": bool(payload.get("wait_for_completion", False)),
            }
        )
        return common

    if mode in CONTROL_MODES:
        common.update(
            {
                "timeout_sec": float(payload.get("timeout_sec", 20.0) or 20.0),
                "poll_interval_ms": int(payload.get("poll_interval_ms", 50) or 50),
            }
        )
        return common

    raise ValueError(f"dataset mode unsupported: {mode}")


def load_run_specs(path: Path, *, default_label: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if payload and any(isinstance(item, dict) for item in payload):
            return [
                {
                    "mode": "multimodal",
                    "items": _normalize_multimodal_items(payload, dataset_path=path),
                    "label": default_label,
                    "tick_interval_ms": 0,
                    "reset_runtime": False,
                    "delay_ms": 0,
                    "hooks": _resolve_hooks({}, dataset_path=path),
                }
            ]
        return [
            {
                "mode": "text",
                "texts": _normalize_texts(payload),
                "label": default_label,
                "tick_interval_ms": 0,
                "reset_runtime": False,
                "delay_ms": 0,
                "hooks": _resolve_hooks({}, dataset_path=path),
            }
        ]
    if not isinstance(payload, dict):
        raise ValueError("dataset format unsupported")
    if isinstance(payload.get("runs"), list):
        rows = []
        for index, run_payload in enumerate(payload.get("runs", []) or []):
            if not isinstance(run_payload, dict):
                raise ValueError(f"runs[{index}] must be an object")
            run_default_label = str(run_payload.get("label", "") or payload.get("label", "") or f"{default_label} #{index + 1}")
            rows.append(_parse_run_spec(run_payload, dataset_path=path, default_label=run_default_label))
        if not rows:
            raise ValueError("runs must not be empty")
        return rows
    return [_parse_run_spec(payload, dataset_path=path, default_label=str(payload.get("label", "") or default_label))]


def _apply_before_hooks(app: ObservatoryV2App, spec: dict[str, Any]) -> dict[str, Any]:
    hooks = dict(spec.get("hooks", {}) or {})
    before = dict(hooks.get("before", {}) or {})
    result: dict[str, Any] = {}
    import_runtime_path = str(before.get("import_runtime_path", "") or "").strip()
    if import_runtime_path:
        result["import_runtime"] = app.load_checkpoint(Path(import_runtime_path))
    import_memory_bundle_dir = str(before.get("import_memory_bundle_dir", "") or "").strip()
    if import_memory_bundle_dir:
        result["import_memory_bundle"] = app.import_memory_deployment_bundle(Path(import_memory_bundle_dir))
    return result


def _sleep_before_spec(spec: dict[str, Any]) -> None:
    delay_ms = int(spec.get("delay_ms", 0) or 0)
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)


def _wait_for_session_status(
    app: ObservatoryV2App,
    *,
    predicate,
    timeout_sec: float,
    poll_interval_ms: int,
) -> dict[str, Any]:
    deadline = time.time() + max(0.1, float(timeout_sec))
    last_status = app.get_autonomous_session_status()
    while time.time() < deadline:
        last_status = app.get_autonomous_session_status()
        if predicate(last_status):
            return {"ok": True, "status": last_status}
        time.sleep(max(0.01, int(poll_interval_ms) / 1000.0))
    last_status = app.get_autonomous_session_status()
    return {"ok": False, "timeout": True, "status": last_status}


def _wait_for_completion(
    app: ObservatoryV2App,
    *,
    timeout_sec: float,
    stop_on_timeout: bool = False,
) -> dict[str, Any]:
    waited = bool(app.wait_for_idle(timeout_sec=float(timeout_sec)))
    status = app.get_autonomous_session_status()
    if waited:
        return {"ok": True, "status": status}
    if stop_on_timeout and bool(status.get("active", False)):
        stop_result = app.stop_autonomous_session()
        app.wait_for_idle(timeout_sec=10.0)
        return {
            "ok": False,
            "timeout": True,
            "stop_result": stop_result,
            "status": app.get_autonomous_session_status(),
        }
    return {"ok": False, "timeout": True, "status": status}


def _run_single_spec(app: ObservatoryV2App, spec: dict[str, Any]) -> dict[str, Any]:
    mode = str(spec.get("mode", "") or "").strip().lower()
    common = {
        "label": str(spec.get("label", "") or ""),
        "tick_interval_ms": int(spec.get("tick_interval_ms", 0) or 0),
        "reset_runtime": bool(spec.get("reset_runtime", False)),
    }
    if mode == "text":
        return app.start_text_run(texts=list(spec.get("texts", []) or []), **common)
    if mode == "multimodal":
        return app.start_multimodal_run(items=list(spec.get("items", []) or []), **common)
    if mode == "continue_from_checkpoint":
        return app.continue_from_checkpoint(
            checkpoint_path=Path(str(spec.get("checkpoint_path", "") or "")),
            texts=list(spec.get("texts", []) or []),
            label=common["label"],
            tick_interval_ms=common["tick_interval_ms"],
        )
    if mode == "audio_stream":
        return app.start_audio_stream_run(
            audio_bytes=bytes(spec.get("audio_bytes") or b""),
            text_prefix=str(spec.get("text_prefix", "") or ""),
            tick_window_ms=spec.get("tick_window_ms"),
            **common,
        )
    if mode == "image_stream":
        return app.start_image_stream_run(
            frame_bytes_list=spec.get("frame_bytes_list"),
            strip_image_bytes=spec.get("strip_image_bytes"),
            frame_count=spec.get("frame_count"),
            text_prefix=str(spec.get("text_prefix", "") or ""),
            **common,
        )
    if mode == "video_stream":
        return app.start_video_stream_run(
            video_bytes=bytes(spec.get("video_bytes") or b""),
            video_name=str(spec.get("video_name", "") or ""),
            text_prefix=str(spec.get("text_prefix", "") or ""),
            tick_fps=spec.get("tick_fps"),
            frame_stride=spec.get("frame_stride"),
            max_frames=spec.get("max_frames"),
            **common,
        )
    if mode == "webcam_stream":
        return app.start_webcam_stream_run(
            text_prefix=str(spec.get("text_prefix", "") or ""),
            max_frames=spec.get("max_frames"),
            device_index=int(spec.get("device_index", 0) or 0),
            frame_width=spec.get("frame_width"),
            frame_height=spec.get("frame_height"),
            **common,
        )
    if mode == "microphone_stream":
        device_index = spec.get("device_index")
        return app.start_microphone_stream_run(
            text_prefix=str(spec.get("text_prefix", "") or ""),
            max_windows=spec.get("max_windows"),
            tick_window_ms=spec.get("tick_window_ms"),
            sample_rate=int(spec.get("sample_rate", 16000) or 16000),
            channels=int(spec.get("channels", 1) or 1),
            device_index=(None if device_index is None else int(device_index)),
            **common,
        )
    if mode == "autonomous_run":
        return app.start_autonomous_run(
            ticks=int(spec.get("ticks", 4) or 4),
            text_hint=str(spec.get("text_hint", "") or ""),
            tick_interval_ms=common["tick_interval_ms"],
            reset_runtime=common["reset_runtime"],
            label=common["label"],
            reward_schedule=spec.get("reward_schedule"),
            stop_on_capture_failures=spec.get("stop_on_capture_failures"),
            stop_on_action_errors=spec.get("stop_on_action_errors"),
            stop_on_idle_ticks=spec.get("stop_on_idle_ticks"),
            idle_backoff_ms=spec.get("idle_backoff_ms"),
            auto_feedback_enabled=spec.get("auto_feedback_enabled"),
            teacher_mode=spec.get("teacher_mode"),
            llm_gate_mode=spec.get("llm_gate_mode"),
            external_teacher_enabled=spec.get("external_teacher_enabled"),
            external_teacher_mode=spec.get("external_teacher_mode"),
            external_teacher_stub_response_path=spec.get("external_teacher_stub_response_path"),
            external_teacher_fail_open=spec.get("external_teacher_fail_open"),
            external_teacher_max_retries=spec.get("external_teacher_max_retries"),
            external_teacher_retry_backoff_ms=spec.get("external_teacher_retry_backoff_ms"),
            external_teacher_http_endpoint=spec.get("external_teacher_http_endpoint"),
            external_teacher_http_headers=spec.get("external_teacher_http_headers"),
        )
    if mode == "autonomous_session":
        return app.start_autonomous_session(
            text_hint=str(spec.get("text_hint", "") or ""),
            tick_interval_ms=common["tick_interval_ms"],
            reset_runtime=common["reset_runtime"],
            label=common["label"],
            max_ticks=spec.get("max_ticks"),
            stop_on_capture_failures=spec.get("stop_on_capture_failures"),
            stop_on_action_errors=spec.get("stop_on_action_errors"),
            stop_on_idle_ticks=spec.get("stop_on_idle_ticks"),
            idle_backoff_ms=spec.get("idle_backoff_ms"),
            auto_feedback_enabled=spec.get("auto_feedback_enabled"),
            teacher_mode=spec.get("teacher_mode"),
            llm_gate_mode=spec.get("llm_gate_mode"),
            external_teacher_enabled=spec.get("external_teacher_enabled"),
            external_teacher_mode=spec.get("external_teacher_mode"),
            external_teacher_stub_response_path=spec.get("external_teacher_stub_response_path"),
            external_teacher_fail_open=spec.get("external_teacher_fail_open"),
            external_teacher_max_retries=spec.get("external_teacher_max_retries"),
            external_teacher_retry_backoff_ms=spec.get("external_teacher_retry_backoff_ms"),
            external_teacher_http_endpoint=spec.get("external_teacher_http_endpoint"),
            external_teacher_http_headers=spec.get("external_teacher_http_headers"),
        )
    if mode == "recover_autonomous_session":
        tick_interval_ms = int(spec.get("tick_interval_ms", 0) or 0)
        return app.recover_autonomous_session(
            run_id=spec.get("run_id"),
            tick_interval_ms=(tick_interval_ms if tick_interval_ms > 0 else None),
        )
    if mode == "autonomous_session_status":
        return {"ok": True, "status": app.get_autonomous_session_status()}
    if mode == "pause_autonomous_session":
        request = app.pause_autonomous_session()
        waited = _wait_for_session_status(
            app,
            predicate=lambda status: bool(status.get("paused", False)) or str(status.get("status", "")) == "paused",
            timeout_sec=float(spec.get("timeout_sec", 20.0) or 20.0),
            poll_interval_ms=int(spec.get("poll_interval_ms", 50) or 50),
        )
        return {"request": request, "wait": waited}
    if mode == "resume_autonomous_session":
        request = app.resume_autonomous_session()
        waited = _wait_for_session_status(
            app,
            predicate=lambda status: (not bool(status.get("paused", False))) and str(status.get("status", "")) == "running",
            timeout_sec=float(spec.get("timeout_sec", 20.0) or 20.0),
            poll_interval_ms=int(spec.get("poll_interval_ms", 50) or 50),
        )
        return {"request": request, "wait": waited}
    if mode == "stop_autonomous_session":
        request = app.stop_autonomous_session()
        waited = _wait_for_completion(
            app,
            timeout_sec=float(spec.get("timeout_sec", 20.0) or 20.0),
            stop_on_timeout=False,
        )
        return {"request": request, "wait": waited}
    raise ValueError(f"unsupported dataset mode: {mode}")


def _spec_auto_wait(spec: dict[str, Any]) -> bool:
    mode = str(spec.get("mode", "") or "").strip().lower()
    if mode in CONTROL_MODES:
        return False
    if mode in ASYNC_SESSION_MODES:
        return bool(spec.get("wait_for_completion", False))
    return True


def _status_snapshot_payload(app: ObservatoryV2App) -> dict[str, Any]:
    status = app.get_autonomous_session_status()
    run_id = str(status.get("run_id", "") or "")
    manifest = app.get_manifest(run_id) if run_id else {}
    if run_id and bool(status.get("active", False)) is False:
        for _ in range(20):
            manifest = app.get_manifest(run_id)
            manifest_status = str((manifest or {}).get("status", "") or "")
            session_status = str((status or {}).get("status", "") or "")
            if manifest_status == session_status:
                break
            time.sleep(0.02)
            status = app.get_autonomous_session_status()
    return {
        "status": status,
        "manifest": manifest,
    }


def _write_json_snapshot(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _apply_after_hooks(app: ObservatoryV2App, spec: dict[str, Any]) -> dict[str, Any]:
    hooks = dict(spec.get("hooks", {}) or {})
    after = dict(hooks.get("after", {}) or {})
    result: dict[str, Any] = {}

    pause_cfg = after.get("pause_session")
    if pause_cfg:
        if int(pause_cfg.get("delay_ms", 0) or 0) > 0:
            time.sleep(int(pause_cfg.get("delay_ms", 0) or 0) / 1000.0)
        result["pause_session"] = {
            "request": app.pause_autonomous_session(),
            "wait": _wait_for_session_status(
                app,
                predicate=lambda status: bool(status.get("paused", False)) or str(status.get("status", "")) == "paused",
                timeout_sec=float(pause_cfg.get("timeout_sec", 20.0) or 20.0),
                poll_interval_ms=int(pause_cfg.get("poll_interval_ms", 50) or 50),
            ),
        }

    resume_cfg = after.get("resume_session")
    if resume_cfg:
        if int(resume_cfg.get("delay_ms", 0) or 0) > 0:
            time.sleep(int(resume_cfg.get("delay_ms", 0) or 0) / 1000.0)
        result["resume_session"] = {
            "request": app.resume_autonomous_session(),
            "wait": _wait_for_session_status(
                app,
                predicate=lambda status: (not bool(status.get("paused", False))) and str(status.get("status", "")) == "running",
                timeout_sec=float(resume_cfg.get("timeout_sec", 20.0) or 20.0),
                poll_interval_ms=int(resume_cfg.get("poll_interval_ms", 50) or 50),
            ),
        }

    stop_cfg = after.get("stop_session")
    if stop_cfg:
        if int(stop_cfg.get("delay_ms", 0) or 0) > 0:
            time.sleep(int(stop_cfg.get("delay_ms", 0) or 0) / 1000.0)
        request = app.stop_autonomous_session()
        result["stop_session"] = {
            "request": request,
            "wait": _wait_for_completion(
                app,
                timeout_sec=float(stop_cfg.get("timeout_sec", 20.0) or 20.0),
                stop_on_timeout=False,
            ),
        }

    wait_cfg = after.get("wait_for_session")
    if wait_cfg:
        result["wait_for_session"] = _wait_for_completion(
            app,
            timeout_sec=float(wait_cfg.get("timeout_sec", 120.0) or 120.0),
            stop_on_timeout=bool(wait_cfg.get("stop_on_timeout", False)),
        )

    status_snapshot_cfg = after.get("status_snapshot")
    if status_snapshot_cfg:
        snapshot_payload = _status_snapshot_payload(app)
        result["status_snapshot"] = snapshot_payload
        snapshot_path = status_snapshot_cfg.get("path")
        if snapshot_path:
            _write_json_snapshot(Path(str(snapshot_path)), snapshot_payload)
            result["status_snapshot_file"] = str(snapshot_path)

    save_checkpoint_path = str(after.get("save_checkpoint_path", "") or "").strip()
    if save_checkpoint_path:
        checkpoint_path = Path(save_checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        result["save_checkpoint"] = app.save_checkpoint(checkpoint_path)

    export_memory_bundle_dir = str(after.get("export_memory_bundle_dir", "") or "").strip()
    if export_memory_bundle_dir:
        bundle_dir = Path(export_memory_bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        export_result = app.export_memory_deployment_bundle(bundle_dir)
        result["export_memory_bundle"] = export_result
        if bool(after.get("inspect_memory_bundle", False)):
            result["inspect_memory_bundle"] = app.inspect_memory_deployment_bundle(bundle_dir)

    forget_cfg = after.get("forget")
    if forget_cfg:
        result["forget"] = app.forget_cold_memories(
            keep_latest=int(forget_cfg.get("keep_latest", 128) or 128),
            strategy=str(forget_cfg.get("strategy", "latest_only") or "latest_only"),
            min_reality_weight=float(forget_cfg.get("min_reality_weight", 0.0) or 0.0),
            min_total_item_energy=float(forget_cfg.get("min_total_item_energy", 0.0) or 0.0),
            protect_memory_kinds=list(forget_cfg.get("protect_memory_kinds", []) or []),
            max_memory_count=forget_cfg.get("max_memory_count"),
            dry_run=bool(forget_cfg.get("dry_run", False)),
        )
    else:
        forget_keep_latest = after.get("forget_keep_latest")
        if forget_keep_latest is not None:
            result["forget"] = app.forget_cold_memories(keep_latest=int(forget_keep_latest))

    return result


def _resolve_manifest_for_row(app: ObservatoryV2App, result: dict[str, Any]) -> dict[str, Any]:
    run_id = str(result.get("run_id", "") or "")
    if not run_id and isinstance(result.get("status"), dict):
        run_id = str((result.get("status", {}) or {}).get("run_id", "") or "")
    if not run_id and isinstance(result.get("request"), dict):
        run_id = str((result.get("request", {}) or {}).get("run_id", "") or "")
    return app.get_manifest(run_id) if run_id else {}


def run_dataset_file(
    dataset_path: Path,
    *,
    default_label: str = "Phase10 批量实验",
    timeout_sec: float = 600.0,
    app: ObservatoryV2App | None = None,
    repo_root_value: Path | None = None,
    outputs_root_override: str | None = None,
) -> dict[str, Any]:
    resolved_dataset_path = Path(dataset_path).resolve()
    raw_payload = json.loads(resolved_dataset_path.read_text(encoding="utf-8"))
    dataset_config_overrides = None
    if isinstance(raw_payload, dict) and raw_payload.get("config_overrides") is not None:
        if not isinstance(raw_payload.get("config_overrides"), dict):
            raise ValueError("config_overrides must be an object")
        dataset_config_overrides = dict(raw_payload.get("config_overrides", {}) or {})
    if app is not None and dataset_config_overrides:
        raise ValueError("config_overrides cannot be used when app is provided explicitly")

    run_specs = load_run_specs(resolved_dataset_path, default_label=str(default_label or "Phase10 批量实验"))
    owned_app = app is None
    runtime_app = app or ObservatoryV2App(
        config=load_config(overrides=dataset_config_overrides or None),
        repo_root_value=(repo_root_value or repo_root()),
        outputs_root_override=outputs_root_override,
    )

    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(run_specs):
        before_artifacts = _apply_before_hooks(runtime_app, spec)
        _sleep_before_spec(spec)
        result = _run_single_spec(runtime_app, spec)
        if _spec_auto_wait(spec):
            if not runtime_app.wait_for_idle(timeout_sec=float(timeout_sec)):
                raise TimeoutError(f"run did not finish in time: {result.get('run_id', '')}")
        after_artifacts = _apply_after_hooks(runtime_app, spec)
        manifest = _resolve_manifest_for_row(runtime_app, result)
        rows.append(
            {
                "batch_index": index,
                "mode": spec.get("mode", ""),
                "label": str(manifest.get("label", "") or spec.get("label", "") or ""),
                "result": result,
                "manifest": manifest,
                "session_status": runtime_app.get_autonomous_session_status(),
                "artifacts": {
                    "before": before_artifacts,
                    "after": after_artifacts,
                },
            }
        )

    payload: dict[str, Any] = {
        "dataset": str(resolved_dataset_path),
        "run_count": len(rows),
        "runs": rows,
    }
    if dataset_config_overrides:
        payload["config_overrides"] = dataset_config_overrides
    if owned_app:
        payload["runtime_summary"] = runtime_app.export_runtime_summary()
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AP 二期 dataset runner 工程化入口")
    parser.add_argument("--dataset", required=True, help="JSON dataset path")
    parser.add_argument("--label", default="Phase10 批量实验", help="dataset 未提供 label 时使用")
    parser.add_argument("--outputs-root", default="", help="可选，覆盖输出目录，适合隔离测试或临时批量实验")
    parser.add_argument("--timeout-sec", type=float, default=600.0, help="每个 run 等待完成的最长秒数")
    args = parser.parse_args(argv)
    result = run_dataset_file(
        Path(args.dataset),
        default_label=str(args.label or "Phase10 批量实验"),
        timeout_sec=float(args.timeout_sec),
        repo_root_value=repo_root(),
        outputs_root_override=(str(args.outputs_root or "").strip() or None),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


__all__ = [
    "load_run_specs",
    "run_dataset_file",
    "main",
]
