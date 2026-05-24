# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import base64
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .app import AppError, ObservatoryV2App


def _read_static_index() -> bytes:
    path = Path(__file__).resolve().parent / "web_static" / "index.html"
    return path.read_bytes()


class ObservatoryHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], app: ObservatoryV2App):
        super().__init__(server_address, ObservatoryRequestHandler)
        self.app = app


class ObservatoryRequestHandler(BaseHTTPRequestHandler):
    server: ObservatoryHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path or "/"
        if path == "/":
            self._send_html(_read_static_index())
            return
        if path == "/api/health":
            self._send_json({"ok": True, "service": "observatory_v2", "status": "ready", "server_meta": self.server.app.service_meta()})
            return
        if path == "/api/config":
            self._send_json(self.server.app.config_public())
            return
        if path == "/api/live":
            self._send_json(self.server.app.get_live_snapshot())
            return
        if path == "/api/runs":
            self._send_json({"runs": self.server.app.list_run_infos(limit=32)})
            return
        if path == "/api/runs/overview-batch":
            query = urllib.parse.parse_qs(parsed.query or "", keep_blank_values=False)
            raw_limit = (query.get("limit", ["8"]) or ["8"])[0]
            try:
                limit = int(raw_limit or 8)
            except ValueError:
                limit = 8
            run_ids = query.get("run_id", []) or None
            self._send_json(self.server.app.build_run_overview_batch(limit=limit, run_ids=run_ids))
            return
        if path == "/api/executor/status":
            self._send_json(self.server.app.executor_status())
            return
        if path == "/api/autonomous-session/status":
            self._send_json(self.server.app.get_autonomous_session_status())
            return
        if path == "/api/executor/screen-preview":
            self._send_json(self.server.app.capture_screen_preview())
            return
        if path == "/api/runtime/export":
            self._send_json(self.server.app.export_runtime())
            return
        if path == "/api/runtime/summary":
            self._send_json(self.server.app.export_runtime_summary())
            return
        if path == "/api/rules":
            self._send_json(self.server.app.get_rules_payload())
            return
        if path == "/api/tuner":
            self._send_json(self.server.app.get_tuner_payload())
            return
        if path == "/api/runs/latest":
            run_id = self.server.app.latest_run_id()
            if not run_id:
                self._send_json({"run_id": "", "manifest": {}}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json({"run_id": run_id, "manifest": self.server.app.get_manifest(run_id)})
            return
        parts = [p for p in path.split("/") if p]
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "runs" and parts[3] == "manifest":
            run_id = parts[2]
            manifest = self.server.app.get_manifest(run_id)
            if not manifest:
                self._send_json({"error": "run not found", "run_id": run_id}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(manifest)
            return
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "runs" and parts[3] == "overview":
            run_id = parts[2]
            payload = self.server.app.build_run_overview(run_id)
            if not payload:
                self._send_json({"error": "run not found", "run_id": run_id}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(payload)
            return
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "runs" and parts[3] == "rollup":
            run_id = parts[2]
            payload = self.server.app.get_run_rollup(run_id)
            if not payload:
                self._send_json({"error": "run not found", "run_id": run_id}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(payload)
            return
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "runs" and parts[3] == "events":
            run_id = parts[2]
            query = urllib.parse.parse_qs(parsed.query or "", keep_blank_values=False)
            raw_limit = (query.get("limit", ["120"]) or ["120"])[0]
            try:
                limit = int(raw_limit or 120)
            except ValueError:
                limit = 120
            self._send_json({"run_id": run_id, "events": self.server.app.get_run_events(run_id, limit=limit)})
            return
        if len(parts) >= 6 and parts[0] == "api" and parts[1] == "runs" and parts[3] == "assets":
            run_id = parts[2]
            rel_path = "/".join(parts[4:])
            if not rel_path:
                self._send_json({"error": "asset path required", "run_id": run_id}, status=HTTPStatus.BAD_REQUEST)
                return
            text = self.server.app.get_run_text_asset(run_id, rel_path)
            if not text:
                self._send_json({"error": "asset not found", "run_id": run_id, "rel_path": rel_path}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json({"run_id": run_id, "rel_path": rel_path, "text": text})
            return
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "runs" and parts[3] == "ticks":
            run_id = parts[2]
            self._send_json({"run_id": run_id, "ticks": self.server.app.list_tick_summaries(run_id)})
            return
        if len(parts) == 5 and parts[0] == "api" and parts[1] == "runs" and parts[3] == "ticks":
            run_id = parts[2]
            try:
                tick_index = int(parts[4])
            except ValueError:
                self._send_json({"error": "invalid tick index"}, status=HTTPStatus.BAD_REQUEST)
                return
            summary = self.server.app.get_tick_summary(run_id, tick_index)
            if not summary:
                self._send_json({"error": "tick not found", "run_id": run_id, "tick_index": tick_index}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(summary)
            return
        if len(parts) == 6 and parts[0] == "api" and parts[1] == "runs" and parts[3] == "ticks" and parts[5] == "sidecar":
            run_id = parts[2]
            try:
                tick_index = int(parts[4])
            except ValueError:
                self._send_json({"error": "invalid tick index"}, status=HTTPStatus.BAD_REQUEST)
                return
            sidecar = self.server.app.get_tick_sidecar(run_id, tick_index)
            if not sidecar:
                self._send_json({"error": "tick sidecar not found", "run_id": run_id, "tick_index": tick_index}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(sidecar)
            return
        self._send_json({"error": "not found", "path": path}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path or "/"
        if path == "/api/runs/demo/start":
            body = self._read_json_body()
            try:
                result = self.server.app.start_demo_run(
                    tick_count=body.get("tick_count"),
                    tick_interval_ms=body.get("tick_interval_ms"),
                    label=body.get("label"),
                )
            except AppError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.CONFLICT)
                return
            self._send_json({"ok": True, **result}, status=HTTPStatus.ACCEPTED)
            return
        if path == "/api/runs/text/start":
            body = self._read_json_body()
            texts = body.get("texts", [])
            if not isinstance(texts, list):
                self._send_json({"ok": False, "error": "texts must be a list"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                result = self.server.app.start_text_run(
                    texts=[str(item or "") for item in texts],
                    label=body.get("label"),
                    tick_interval_ms=int(body.get("tick_interval_ms", 0) or 0),
                    reset_runtime=bool(body.get("reset_runtime", False)),
                )
            except AppError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.CONFLICT)
                return
            self._send_json({"ok": True, **result}, status=HTTPStatus.ACCEPTED)
            return
        if path == "/api/runs/multimodal/start":
            body = self._read_json_body()
            items = body.get("items", [])
            if not isinstance(items, list):
                self._send_json({"ok": False, "error": "items must be a list"}, status=HTTPStatus.BAD_REQUEST)
                return
            clean_items = []
            for raw in items:
                if not isinstance(raw, dict):
                    continue
                item = {"text": str(raw.get("text", "") or ""), "source_type": str(raw.get("source_type", "multimodal_input") or "multimodal_input")}
                image_b64 = str(raw.get("image_b64", "") or "")
                audio_b64 = str(raw.get("audio_b64", "") or "")
                if isinstance(raw.get("external_feedback"), dict):
                    item["external_feedback"] = dict(raw.get("external_feedback", {}) or {})
                if image_b64:
                    try:
                        item["image_bytes"] = base64.b64decode(image_b64)
                    except Exception:
                        pass
                if audio_b64:
                    try:
                        item["audio_bytes"] = base64.b64decode(audio_b64)
                    except Exception:
                        pass
                clean_items.append(item)
            try:
                result = self.server.app.start_multimodal_run(
                    items=clean_items,
                    label=body.get("label"),
                    tick_interval_ms=int(body.get("tick_interval_ms", 0) or 0),
                    reset_runtime=bool(body.get("reset_runtime", False)),
                )
            except AppError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.CONFLICT)
                return
            self._send_json({"ok": True, **result}, status=HTTPStatus.ACCEPTED)
            return
        if path == "/api/runs/screen/start":
            body = self._read_json_body()
            ticks = max(1, int(body.get("ticks", 1) or 1))
            items = [{"text": str(body.get("text", "") or ""), "source_type": "screen_capture_run", "capture_screen": True} for _ in range(ticks)]
            try:
                result = self.server.app.start_multimodal_run(
                    items=items,
                    label=str(body.get("label", "") or "Screen Capture Run"),
                    tick_interval_ms=int(body.get("tick_interval_ms", 0) or 0),
                    reset_runtime=bool(body.get("reset_runtime", False)),
                    run_kind="phase17_screen_capture_run",
                )
            except AppError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.CONFLICT)
                return
            self._send_json({"ok": True, **result}, status=HTTPStatus.ACCEPTED)
            return
        if path == "/api/runs/audio-stream/start":
            body = self._read_json_body()
            audio_b64 = str(body.get("audio_b64", "") or "")
            if not audio_b64:
                self._send_json({"ok": False, "error": "audio_b64 is required"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                audio_bytes = base64.b64decode(audio_b64)
            except Exception:
                self._send_json({"ok": False, "error": "audio_b64 decode failed"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                result = self.server.app.start_audio_stream_run(
                    audio_bytes=audio_bytes,
                    text_prefix=str(body.get("text_prefix", "") or ""),
                    tick_window_ms=int(body.get("tick_window_ms", 0) or 0) or None,
                    label=str(body.get("label", "") or "Audio Stream Run"),
                    tick_interval_ms=int(body.get("tick_interval_ms", 0) or 0),
                    reset_runtime=bool(body.get("reset_runtime", False)),
                )
            except AppError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.CONFLICT)
                return
            self._send_json({"ok": True, **result}, status=HTTPStatus.ACCEPTED)
            return
        if path == "/api/runs/image-stream/start":
            body = self._read_json_body()
            frame_list = body.get("frames_b64", [])
            strip_b64 = str(body.get("strip_image_b64", "") or "")
            frame_bytes_list: list[bytes] = []
            if isinstance(frame_list, list):
                for item in frame_list:
                    raw = str(item or "")
                    if not raw:
                        continue
                    try:
                        frame_bytes_list.append(base64.b64decode(raw))
                    except Exception:
                        continue
            strip_image_bytes = None
            if strip_b64:
                try:
                    strip_image_bytes = base64.b64decode(strip_b64)
                except Exception:
                    self._send_json({"ok": False, "error": "strip_image_b64 decode failed"}, status=HTTPStatus.BAD_REQUEST)
                    return
            if not frame_bytes_list and strip_image_bytes is None:
                self._send_json({"ok": False, "error": "frames_b64 or strip_image_b64 is required"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                result = self.server.app.start_image_stream_run(
                    frame_bytes_list=frame_bytes_list or None,
                    strip_image_bytes=strip_image_bytes,
                    frame_count=int(body.get("frame_count", 1) or 1),
                    text_prefix=str(body.get("text_prefix", "") or ""),
                    label=str(body.get("label", "") or "Image Stream Run"),
                    tick_interval_ms=int(body.get("tick_interval_ms", 0) or 0),
                    reset_runtime=bool(body.get("reset_runtime", False)),
                )
            except AppError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.CONFLICT)
                return
            self._send_json({"ok": True, **result}, status=HTTPStatus.ACCEPTED)
            return
        if path == "/api/runs/video-stream/start":
            body = self._read_json_body()
            video_b64 = str(body.get("video_b64", "") or "")
            if not video_b64:
                self._send_json({"ok": False, "error": "video_b64 is required"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                video_bytes = base64.b64decode(video_b64)
            except Exception:
                self._send_json({"ok": False, "error": "video_b64 decode failed"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                result = self.server.app.start_video_stream_run(
                    video_bytes=video_bytes,
                    video_name=str(body.get("video_name", "") or ""),
                    text_prefix=str(body.get("text_prefix", "") or ""),
                    tick_fps=(float(body.get("tick_fps", 0.0) or 0.0) or None),
                    frame_stride=(int(body.get("frame_stride", 0) or 0) or None),
                    max_frames=(int(body.get("max_frames", 0) or 0) or None),
                    label=str(body.get("label", "") or "Video Stream Run"),
                    tick_interval_ms=int(body.get("tick_interval_ms", 0) or 0),
                    reset_runtime=bool(body.get("reset_runtime", False)),
                )
            except AppError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.CONFLICT)
                return
            self._send_json({"ok": True, **result}, status=HTTPStatus.ACCEPTED)
            return
        if path == "/api/runs/webcam-stream/start":
            body = self._read_json_body()
            try:
                result = self.server.app.start_webcam_stream_run(
                    text_prefix=str(body.get("text_prefix", "") or ""),
                    max_frames=(int(body.get("max_frames", 0) or 0) or None),
                    device_index=int(body.get("device_index", 0) or 0),
                    frame_width=(int(body.get("frame_width", 0) or 0) or None),
                    frame_height=(int(body.get("frame_height", 0) or 0) or None),
                    label=str(body.get("label", "") or "Webcam Stream Run"),
                    tick_interval_ms=int(body.get("tick_interval_ms", 0) or 0),
                    reset_runtime=bool(body.get("reset_runtime", False)),
                )
            except AppError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.CONFLICT)
                return
            self._send_json({"ok": True, **result}, status=HTTPStatus.ACCEPTED)
            return
        if path == "/api/runs/microphone-stream/start":
            body = self._read_json_body()
            try:
                raw_device_index = int(body.get("device_index", -1) or -1)
                result = self.server.app.start_microphone_stream_run(
                    text_prefix=str(body.get("text_prefix", "") or ""),
                    max_windows=(int(body.get("max_windows", 0) or 0) or None),
                    tick_window_ms=(int(body.get("tick_window_ms", 0) or 0) or None),
                    sample_rate=int(body.get("sample_rate", 16000) or 16000),
                    channels=int(body.get("channels", 1) or 1),
                    device_index=(None if raw_device_index < 0 else raw_device_index),
                    label=str(body.get("label", "") or "Microphone Stream Run"),
                    tick_interval_ms=int(body.get("tick_interval_ms", 0) or 0),
                    reset_runtime=bool(body.get("reset_runtime", False)),
                )
            except AppError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.CONFLICT)
                return
            self._send_json({"ok": True, **result}, status=HTTPStatus.ACCEPTED)
            return
        if path == "/api/runs/autonomous/start":
            body = self._read_json_body()
            try:
                result = self.server.app.start_autonomous_run(
                    ticks=max(1, int(body.get("ticks", 1) or 1)),
                    text_hint=str(body.get("text_hint", "") or ""),
                    tick_interval_ms=int(body.get("tick_interval_ms", 0) or 0),
                    reset_runtime=bool(body.get("reset_runtime", False)),
                    label=str(body.get("label", "") or "Autonomous Loop Run"),
                    reward_schedule=list(body.get("reward_schedule", []) or []),
                    stop_on_capture_failures=(int(body.get("stop_on_capture_failures", 0) or 0) or None),
                    stop_on_action_errors=(int(body.get("stop_on_action_errors", 0) or 0) or None),
                    stop_on_idle_ticks=(int(body.get("stop_on_idle_ticks", 0) or 0) or None),
                    idle_backoff_ms=(int(body.get("idle_backoff_ms", 0) or 0) or None),
                    auto_feedback_enabled=(body.get("auto_feedback_enabled") if isinstance(body.get("auto_feedback_enabled"), bool) else None),
                    teacher_mode=(str(body.get("teacher_mode", "") or "").strip() or None),
                    llm_gate_mode=(str(body.get("llm_gate_mode", "") or "").strip() or None),
                    external_teacher_enabled=(body.get("external_teacher_enabled") if isinstance(body.get("external_teacher_enabled"), bool) else None),
                    external_teacher_mode=(str(body.get("external_teacher_mode", "") or "").strip() or None),
                    external_teacher_stub_response_path=(str(body.get("external_teacher_stub_response_path", "") or "").strip() or None),
                    external_teacher_fail_open=(body.get("external_teacher_fail_open") if isinstance(body.get("external_teacher_fail_open"), bool) else None),
                    external_teacher_max_retries=(
                        int(body.get("external_teacher_max_retries"))
                        if "external_teacher_max_retries" in body
                        and body.get("external_teacher_max_retries") is not None
                        and int(body.get("external_teacher_max_retries", 0) or 0) > 0
                        else None
                    ),
                    external_teacher_retry_backoff_ms=(
                        int(body.get("external_teacher_retry_backoff_ms"))
                        if "external_teacher_retry_backoff_ms" in body
                        and body.get("external_teacher_retry_backoff_ms") is not None
                        else None
                    ),
                    external_teacher_http_endpoint=(str(body.get("external_teacher_http_endpoint", "") or "").strip() or None),
                    external_teacher_http_headers=(dict(body.get("external_teacher_http_headers", {}) or {}) if isinstance(body.get("external_teacher_http_headers"), dict) else None),
                )
            except AppError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.CONFLICT)
                return
            self._send_json({"ok": True, **result}, status=HTTPStatus.ACCEPTED)
            return
        if path == "/api/autonomous-session/start":
            body = self._read_json_body()
            try:
                result = self.server.app.start_autonomous_session(
                    text_hint=str(body.get("text_hint", "") or ""),
                    tick_interval_ms=int(body.get("tick_interval_ms", 0) or 0),
                    reset_runtime=bool(body.get("reset_runtime", False)),
                    label=str(body.get("label", "") or "Autonomous Session"),
                    max_ticks=(int(body.get("max_ticks", 0) or 0) or None),
                    stop_on_capture_failures=(int(body.get("stop_on_capture_failures", 0) or 0) or None),
                    stop_on_action_errors=(int(body.get("stop_on_action_errors", 0) or 0) or None),
                    stop_on_idle_ticks=(int(body.get("stop_on_idle_ticks", 0) or 0) or None),
                    idle_backoff_ms=(int(body.get("idle_backoff_ms", 0) or 0) or None),
                    auto_feedback_enabled=(body.get("auto_feedback_enabled") if isinstance(body.get("auto_feedback_enabled"), bool) else None),
                    teacher_mode=(str(body.get("teacher_mode", "") or "").strip() or None),
                    llm_gate_mode=(str(body.get("llm_gate_mode", "") or "").strip() or None),
                    external_teacher_enabled=(body.get("external_teacher_enabled") if isinstance(body.get("external_teacher_enabled"), bool) else None),
                    external_teacher_mode=(str(body.get("external_teacher_mode", "") or "").strip() or None),
                    external_teacher_stub_response_path=(str(body.get("external_teacher_stub_response_path", "") or "").strip() or None),
                    external_teacher_fail_open=(body.get("external_teacher_fail_open") if isinstance(body.get("external_teacher_fail_open"), bool) else None),
                    external_teacher_max_retries=(
                        int(body.get("external_teacher_max_retries"))
                        if "external_teacher_max_retries" in body
                        and body.get("external_teacher_max_retries") is not None
                        and int(body.get("external_teacher_max_retries", 0) or 0) > 0
                        else None
                    ),
                    external_teacher_retry_backoff_ms=(
                        int(body.get("external_teacher_retry_backoff_ms"))
                        if "external_teacher_retry_backoff_ms" in body
                        and body.get("external_teacher_retry_backoff_ms") is not None
                        else None
                    ),
                    external_teacher_http_endpoint=(str(body.get("external_teacher_http_endpoint", "") or "").strip() or None),
                    external_teacher_http_headers=(dict(body.get("external_teacher_http_headers", {}) or {}) if isinstance(body.get("external_teacher_http_headers"), dict) else None),
                )
            except AppError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.CONFLICT)
                return
            self._send_json({"ok": True, **result}, status=HTTPStatus.ACCEPTED)
            return
        if path == "/api/autonomous-session/pause":
            self._send_json(self.server.app.pause_autonomous_session(), status=HTTPStatus.OK)
            return
        if path == "/api/autonomous-session/resume":
            self._send_json(self.server.app.resume_autonomous_session(), status=HTTPStatus.OK)
            return
        if path == "/api/autonomous-session/stop":
            self._send_json(self.server.app.stop_autonomous_session(), status=HTTPStatus.OK)
            return
        if path == "/api/autonomous-session/recover":
            body = self._read_json_body()
            try:
                result = self.server.app.recover_autonomous_session(run_id=(str(body.get("run_id", "") or "").strip() or None))
            except AppError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.CONFLICT)
                return
            self._send_json(result, status=HTTPStatus.OK)
            return
        if path == "/api/runtime/import":
            body = self._read_json_body()
            try:
                result = self.server.app.import_runtime(body)
            except AppError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result, status=HTTPStatus.OK)
            return
        if path == "/api/runtime/memory-bundle/export":
            body = self._read_json_body()
            directory = str(body.get("directory", "") or "").strip()
            if not directory:
                self._send_json({"ok": False, "error": "directory is required"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                result = self.server.app.export_memory_deployment_bundle(Path(directory))
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result, status=HTTPStatus.OK)
            return
        if path == "/api/runtime/memory-bundle/import":
            body = self._read_json_body()
            directory = str(body.get("directory", "") or "").strip()
            if not directory:
                self._send_json({"ok": False, "error": "directory is required"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                result = self.server.app.import_memory_deployment_bundle(Path(directory))
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result, status=HTTPStatus.OK)
            return
        if path == "/api/runtime/memory-bundle/inspect":
            body = self._read_json_body()
            directory = str(body.get("directory", "") or "").strip()
            if not directory:
                self._send_json({"ok": False, "error": "directory is required"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                result = self.server.app.inspect_memory_deployment_bundle(Path(directory))
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result, status=HTTPStatus.OK)
            return
        if path == "/api/runtime/forget":
            body = self._read_json_body()
            try:
                protect_memory_kinds = body.get("protect_memory_kinds", [])
                if not isinstance(protect_memory_kinds, list):
                    protect_memory_kinds = []
                raw_limit = body.get("max_memory_count", None)
                max_memory_count = None
                if raw_limit not in (None, "", False):
                    max_memory_count = int(raw_limit)
                result = self.server.app.forget_cold_memories(
                    keep_latest=int(body.get("keep_latest", 128) or 128),
                    min_reality_weight=float(body.get("min_reality_weight", 0.0) or 0.0),
                    min_total_item_energy=float(body.get("min_total_item_energy", 0.0) or 0.0),
                    protect_memory_kinds=[str(item or "") for item in protect_memory_kinds],
                    max_memory_count=max_memory_count,
                    strategy=str(body.get("strategy", "latest_only") or "latest_only"),
                    dry_run=bool(body.get("dry_run", False)),
                )
            except AppError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"ok": True, **result}, status=HTTPStatus.OK)
            return
        if path == "/api/rules":
            body = self._read_json_body()
            try:
                result = self.server.app.save_rules_payload(body)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(
                {
                    "ok": True,
                    "rules": result.get("payload", {}),
                    "warnings": result.get("warnings", []),
                    "stats": result.get("stats", {}),
                },
                status=HTTPStatus.OK,
            )
            return
        if path == "/api/rules/simulate":
            body = self._read_json_body()
            try:
                result = self.server.app.simulate_rules(body)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"ok": True, "result": result}, status=HTTPStatus.OK)
            return
        if path == "/api/rules/validate":
            body = self._read_json_body()
            try:
                result = self.server.app.validate_rules_payload(body)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(
                {
                    "ok": True,
                    "rules": result.get("payload", {}),
                    "warnings": result.get("warnings", []),
                    "stats": result.get("stats", {}),
                },
                status=HTTPStatus.OK,
            )
            return
        if path == "/api/tuner":
            body = self._read_json_body()
            try:
                result = self.server.app.save_tuner_payload(body)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(
                {
                    "ok": True,
                    "tuner": result.get("payload", {}),
                    "warnings": result.get("warnings", []),
                    "stats": result.get("stats", {}),
                },
                status=HTTPStatus.OK,
            )
            return
        if path == "/api/tuner/validate":
            body = self._read_json_body()
            try:
                result = self.server.app.validate_tuner_payload(body)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(
                {
                    "ok": True,
                    "tuner": result.get("payload", {}),
                    "warnings": result.get("warnings", []),
                    "stats": result.get("stats", {}),
                },
                status=HTTPStatus.OK,
            )
            return
        if path == "/api/executor/manual-action":
            body = self._read_json_body()
            action_name = str(body.get("action_name", "") or "").strip()
            if not action_name:
                self._send_json({"ok": False, "error": "action_name is required"}, status=HTTPStatus.BAD_REQUEST)
                return
            result = self.server.app.execute_manual_action(action_name=action_name, params=dict(body.get("params", {}) or {}))
            self._send_json({"ok": True, "result": result}, status=HTTPStatus.OK)
            return
        self._send_json({"error": "not found", "path": path}, status=HTTPStatus.NOT_FOUND)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        return payload

    def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, payload: bytes, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)


def create_server(app: ObservatoryV2App, *, host: str, port: int) -> ObservatoryHTTPServer:
    return ObservatoryHTTPServer((host, int(port)), app)


def run_server(app: ObservatoryV2App, *, host: str, port: int, open_browser: bool = False) -> None:
    server = create_server(app, host=host, port=port)
    bound_host, bound_port = server.server_address[:2]
    url = f"http://{bound_host}:{bound_port}"
    print("======================================")
    print("AP 二期最小观测台已启动")
    print("访问地址:", url)
    print("最新输出目录:", app.layout.outputs_root)
    print("停止方式: 关闭本窗口或按 Ctrl+C")
    print("======================================")
    if open_browser:
        threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
