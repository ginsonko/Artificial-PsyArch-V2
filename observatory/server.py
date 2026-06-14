from __future__ import annotations

import argparse
from collections import deque
import io
import json
import math
import re
import struct
import tempfile
import wave
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from core.runtime.engine import APV21Runtime
from observatory.reconstruct import reconstruct_tick_observatory
from observatory.render_model import build_inner_world_render_model, render_observatory_shell_html


MAX_JSON_BODY_BYTES = 1_000_000
ASSET_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")
PLAYBACK_BUFFER_MAX_FRAMES = 16
PLAYBACK_FRAME_INTERVAL_MS = 100
PLAYBACK_INSPECT_INTERVAL_OPTIONS_MS = [100, 250, 500]


class APV21ObservatoryApp:
    """
    Small APV2.1-native observatory application boundary.

    The runtime remains the source of truth. The server only runs ticks, expands
    the latest white-box reconstruction after the tick, and exposes compact
    render/asset contracts for the browser.
    """

    def __init__(
        self,
        *,
        runtime: APV21Runtime | None = None,
        root_dir: Path | str | None = None,
        warm_demo_memory: bool = True,
    ) -> None:
        self.runtime = runtime or APV21Runtime()
        self.root_dir = Path(root_dir).resolve() if root_dir else Path(__file__).resolve().parents[1]
        self.warm_demo_memory = bool(warm_demo_memory)
        self._demo_memory_warmed = False
        self.latest_trace: dict | None = None
        self.latest_reconstruction: dict | None = None
        self.latest_render_model: dict | None = None
        self.attention_timeline: deque[dict] = deque(maxlen=10)
        self.playback_buffer: deque[dict] = deque(maxlen=PLAYBACK_BUFFER_MAX_FRAMES)
        self._playback_frame_seq = 0
        self._demo_frame_index = 0
        self._demo_audio_index = 0

    def health(self) -> dict:
        return {
            "schema_id": "apv21_observatory_health/v1",
            "service": "apv21_native_observatory",
            "latest_tick_index": None if self.latest_trace is None else self.latest_trace.get("tick_index"),
            "has_latest_render_model": self.latest_render_model is not None,
            "playback_buffer_frames": len(self.playback_buffer),
            "asset_layer_enabled": bool(self.runtime.config.multimodal_assets.enabled),
            "inner_playback_source": "state_pool_numeric_channels",
            "asset_store": self.runtime.asset_store.summary(),
        }

    def ensure_latest(self, *, inline_assets: bool = True) -> dict:
        if self.latest_render_model is None:
            self.run_tick(text="inner world preview", use_demo_media=True, inline_assets=inline_assets)
        return dict(self.latest_render_model or {})

    def run_tick(
        self,
        *,
        text: str = "",
        image_bytes: bytes | None = None,
        audio_bytes: bytes | None = None,
        use_demo_media: bool = True,
        inline_assets: bool = True,
    ) -> dict:
        clean_text = str(text or "inner world preview")
        if use_demo_media:
            image_bytes = image_bytes if image_bytes is not None else self._demo_image_bytes()
            audio_bytes = audio_bytes if audio_bytes is not None else generated_demo_wav(phrase_index=self._demo_audio_index)
            self._demo_audio_index += 1
            self._warm_demo_history_once(image_bytes=image_bytes, audio_bytes=audio_bytes)

        trace = self.runtime.process_multimodal_tick(clean_text, image_bytes=image_bytes, audio_bytes=audio_bytes)
        reconstruction = reconstruct_tick_observatory(
            trace,
            snapshot_lookup=self.runtime.memory.snapshot_by_id,
            successor_lookup=self.runtime.memory.successor_links,
        )
        render_model = build_inner_world_render_model(reconstruction, inline_assets=inline_assets)
        self._append_attention_timeline(render_model)
        render_model["attention_timeline"] = self._attention_timeline_view()
        self.latest_trace = trace
        self.latest_reconstruction = reconstruction
        self.latest_render_model = render_model
        self._append_playback_frame(render_model)
        return {
            "schema_id": "apv21_observatory_tick_response/v1",
            "trace": trace,
            "reconstruction": reconstruction,
            "render_model": render_model,
        }

    def reconstruction(self) -> dict:
        self.ensure_latest(inline_assets=True)
        return dict(self.latest_reconstruction or {})

    def playback_buffer_view(self) -> dict:
        """
        Return recent render models as an observatory-only playback window.

        The buffer is deliberately outside AP cognition. It never feeds back
        into the runtime, memory, planner, or state pool; it only lets a human
        inspect the 5-10 tick scale where gaze and audio flow become visible.
        """

        if not self.playback_buffer:
            self.ensure_latest(inline_assets=True)
        frames = [dict(frame) for frame in self.playback_buffer]
        latest_tick = frames[-1].get("tick_index") if frames else None
        return {
            "schema_id": "apv21_observatory_playback_buffer/v1",
            "source": "observatory_render_model_buffer",
            "cognition_feedback": "none",
            "frame_interval_ms": PLAYBACK_FRAME_INTERVAL_MS,
            "inspect_interval_options_ms": list(PLAYBACK_INSPECT_INTERVAL_OPTIONS_MS),
            "max_frames": PLAYBACK_BUFFER_MAX_FRAMES,
            "frame_count": len(frames),
            "latest_tick_index": latest_tick,
            "frames": frames,
        }

    def _append_playback_frame(self, render_model: dict) -> None:
        """
        Store a compact multi-tick observatory frame.

        Gaze delta is copied out explicitly because future AP action-learning
        audits should learn from parameterized actions, not just the bare
        `action::move_gaze_to` label. This still remains a read-only trace.
        """

        self._playback_frame_seq += 1
        model = dict(render_model or {})
        self.playback_buffer.append(
            {
                "schema_id": "apv21_observatory_playback_frame/v1",
                "frame_seq": self._playback_frame_seq,
                "tick_index": model.get("tick_index"),
                "gaze_delta": _playback_gaze_delta(model),
                "render_model": model,
            }
        )

    def _append_attention_timeline(self, render_model: dict) -> None:
        """
        Keep a short observatory-only view of AP's recent attention flow.

        This is deliberately not written back into the runtime or memory. It is
        a white-box playback aid for the humanlike 5-10 tick observation window:
        where the gaze was, how sharp the fovea was, and which B/C layers were
        active across recent ticks.
        """

        model = dict(render_model or {})
        vision = dict(model.get("vision_panel", {}) or {})
        focus = dict(vision.get("focus_overlay", {}) or {})
        detail = dict(model.get("detail_panel", {}) or {})
        layers = dict(detail.get("memory_layers", {}) or {})
        action = dict(model.get("action_panel", {}) or {})
        if not action:
            action = dict((dict(model.get("detail_panel", {}) or {}).get("action", {}) or {}))
        gaze_action = _timeline_gaze_action(action)
        clarity = dict(focus.get("clarity", {}) or {})
        resolution = dict(focus.get("resolution_summary", {}) or {})
        self.attention_timeline.append(
            {
                "schema_id": "apv21_attention_timeline_row/v1",
                "tick_index": model.get("tick_index"),
                "gaze_center_norm": list(focus.get("center_norm", []) or []),
                "clarity": {
                    "near_focus": float(clarity.get("near_focus", 0.0) or 0.0),
                    "far_periphery": float(clarity.get("far_periphery", 0.0) or 0.0),
                },
                "resolution": {
                    "policy": str(resolution.get("policy", "") or ""),
                    "max_color_cells": int(resolution.get("max_color_cells", 0) or 0),
                    "min_color_cells": int(resolution.get("min_color_cells", 0) or 0),
                    "focus_detail_patch_count": int(resolution.get("focus_detail_patch_count", 0) or 0),
                    "max_focus_patch_cells": int(resolution.get("max_focus_patch_cells", 0) or 0),
                },
                "fast_bn_labels": _timeline_labels(layers.get("fast_bn", [])),
                "slow_bn_labels": _timeline_labels(layers.get("slow_bn_prime", [])),
                "fast_cn_labels": _timeline_labels(layers.get("fast_cn", []), predicted=True),
                "slow_cn_labels": _timeline_labels(layers.get("slow_cn_prime", []), predicted=True),
                "gaze_action": gaze_action,
            }
        )

    def _attention_timeline_view(self) -> list[dict]:
        return [dict(row) for row in self.attention_timeline]

    def asset_record(self, asset_id: str, *, include_payload: bool = True) -> dict | None:
        if not bool(self.runtime.config.multimodal_assets.enabled):
            return None
        clean = str(asset_id or "")
        if not ASSET_ID_RE.fullmatch(clean):
            return None
        return self.runtime.asset_store.get(clean, include_payload=include_payload)

    def _warm_demo_history_once(self, *, image_bytes: bytes | None, audio_bytes: bytes | None) -> None:
        if not self.warm_demo_memory or self._demo_memory_warmed:
            return
        self._demo_memory_warmed = True
        # Warm only the stable baseline. The visible demo sequence should still
        # contain genuine later surprises: side objects appear, move, and vanish
        # after AP has a small expectation of the baseline scene.
        baseline_image = generated_gaze_flow_demo_image(frame_index=0) or image_bytes
        baseline_audio = generated_demo_wav(phrase_index=0) or audio_bytes
        self.runtime.process_multimodal_tick("inner world preview memory", image_bytes=baseline_image, audio_bytes=baseline_audio)
        self.runtime.process_idle_maintenance(include_heavy=True, budget=4, max_ms=50.0)

    def _demo_image_bytes(self) -> bytes | None:
        generated = generated_gaze_flow_demo_image(frame_index=self._demo_frame_index)
        self._demo_frame_index += 1
        if generated:
            return generated
        evidence_root = self.root_dir / "legacy_apv2" / "evidence"
        if not evidence_root.exists():
            return None
        for path in evidence_root.rglob("*.png"):
            try:
                return path.read_bytes()
            except OSError:
                continue
        return None


def generated_gaze_flow_demo_image(*, frame_index: int = 0) -> bytes | None:
    """
    Observatory demo image sequence tuned for attention-flow validation.

    The frame is still ordinary external input. The sequence changes side
    objects over time so AP can be pulled by the same generic signals humans
    use: new appearance, motion, high contrast, disappearance mismatch, and
    fatigue on already-clear targets. No gaze action is scripted here.
    """

    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None
    image = Image.new("RGB", (160, 100), (238, 240, 242))
    draw = ImageDraw.Draw(image)
    step = int(frame_index) % 12

    # A stable central object gives the system an expectation anchor. Its
    # brightness is reduced later so side changes can win by competition rather
    # than by a hard-coded scan order.
    center_fill = (245, 190, 70) if step not in {10, 11} else (190, 150, 76)
    draw.ellipse((65, 18, 95, 48), fill=center_fill, outline=(85, 65, 28), width=3)

    left_box = (14, 26, 48, 74)
    right_box = (112, 20, 148, 80)
    if step in {1, 2, 3, 4, 6, 9, 10, 11}:
        left_fill = (60, 110, 220)
        if step == 2:
            left_fill = (85, 160, 255)
        if step == 9:
            left_box = (20, 18, 56, 66)
        if step == 10:
            left_box = (10, 30, 44, 78)
        draw.rectangle(left_box, fill=left_fill, outline=(20, 35, 70), width=3)
    elif step == 8:
        # A faint outline-like trace makes absence visible without drawing the
        # former object as an active object. This supports mismatch inspection.
        draw.rectangle(left_box, outline=(198, 205, 214), width=1)

    if step in {4, 5, 6, 7, 8, 9, 10, 11}:
        right_fill = (230, 80, 58)
        if step == 5:
            right_box = (108, 18, 144, 78)
        if step == 6:
            right_box = (100, 14, 136, 72)
        if step == 7:
            right_fill = (255, 55, 42)
        if step == 10:
            right_box = (116, 26, 152, 84)
        draw.rectangle(right_box, fill=right_fill, outline=(70, 35, 28), width=3)

    if step in {2, 7}:
        # A tiny contrast glint gives the sensor a real brightness/edge event.
        box = left_box if step == 2 else right_box
        draw.rectangle((box[0] + 5, box[1] + 5, box[0] + 14, box[1] + 14), fill=(255, 252, 220))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

def create_observatory_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8767,
    app: APV21ObservatoryApp | None = None,
) -> ThreadingHTTPServer:
    observatory_app = app or APV21ObservatoryApp()

    class Handler(APV21ObservatoryHandler):
        pass

    Handler.app = observatory_app
    return ThreadingHTTPServer((str(host), int(port)), Handler)


class APV21ObservatoryHandler(BaseHTTPRequestHandler):
    app: APV21ObservatoryApp
    server_version = "APV21Observatory/0.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path in {"/", "/index.html"}:
            self._send_html(render_observatory_shell_html())
            return
        if path == "/api/health":
            self._send_json(self.app.health())
            return
        if path == "/api/inner-world/render-model":
            inline_assets = _bool_query(query, "inline_assets", default=True)
            self._send_json(self.app.ensure_latest(inline_assets=inline_assets))
            return
        if path == "/api/inner-world/playback-buffer":
            self._send_json(self.app.playback_buffer_view())
            return
        if path == "/api/observatory/reconstruction":
            self._send_json(self.app.reconstruction())
            return
        if path.startswith("/api/assets/"):
            self._send_asset(unquote(path.removeprefix("/api/assets/")))
            return
        self._send_error(HTTPStatus.NOT_FOUND, "not_found")

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        if parsed.path == "/api/feedback":
            self._handle_feedback_post()
            return
        if parsed.path != "/api/tick":
            self._send_error(HTTPStatus.NOT_FOUND, "not_found")
            return
        body = self._read_json_body()
        if body is None:
            return
        response = self.app.run_tick(
            text=str(body.get("text", "") or "inner world preview"),
            use_demo_media=bool(body.get("use_demo_media", True)),
            inline_assets=bool(body.get("inline_assets", True)),
        )
        self._send_json(response)

    def do_PUT(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        if parsed.path != "/api/feedback":
            self._send_error(HTTPStatus.NOT_FOUND, "not_found")
            return
        self._handle_feedback_post()

    def _handle_feedback_post(self) -> None:
        body = self._read_json_body()
        if body is None:
            return
        queued = self.app.runtime.queue_external_feedback(
            reward=float(body.get("reward", 0.0) or 0.0),
            punishment=float(body.get("punishment", 0.0) or 0.0),
            correctness=float(body.get("correctness", 0.0) or 0.0),
            confidence=float(body.get("confidence", 1.0) or 1.0),
            source=str(body.get("source", "") or "observatory_api"),
            notes=[str(note or "") for note in list(body.get("notes", []) or []) if str(note or "")],
        )
        self._send_json(queued)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json_body(self) -> dict | None:
        raw_length = self.headers.get("content-length", "0")
        try:
            length = int(raw_length)
        except ValueError:
            self._send_error(HTTPStatus.LENGTH_REQUIRED, "invalid_content_length")
            return None
        if length < 0 or length > MAX_JSON_BODY_BYTES:
            self._send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "json_body_too_large")
            return None
        try:
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8") if raw else "{}")
        except Exception:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_json")
            return None
        if not isinstance(payload, dict):
            self._send_error(HTTPStatus.BAD_REQUEST, "json_object_required")
            return None
        return payload

    def _send_asset(self, asset_id: str) -> None:
        if not ASSET_ID_RE.fullmatch(str(asset_id or "")):
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_asset_id")
            return
        record = self.app.asset_record(asset_id, include_payload=True)
        if not record:
            self._send_error(HTTPStatus.NOT_FOUND, "asset_not_found")
            return
        payload = bytes(record.get("payload_bytes", b"") or b"")
        if not payload:
            self._send_error(HTTPStatus.NOT_FOUND, "asset_payload_not_found")
            return
        payload_ref = dict(record.get("payload_ref", {}) or {})
        content_type = _content_type_for_encoding(str(payload_ref.get("encoding", "") or "bytes"))
        range_header = str(self.headers.get("range", "") or "")
        byte_range = _parse_range_header(range_header, len(payload))
        if byte_range is not None:
            start, end = byte_range
            chunk = payload[start : end + 1]
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("content-type", content_type)
            self.send_header("accept-ranges", "bytes")
            self.send_header("content-range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("content-length", str(len(chunk)))
            self.send_header("cache-control", "private, max-age=300")
            self.end_headers()
            self.wfile.write(chunk)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", content_type)
        self.send_header("accept-ranges", "bytes")
        self.send_header("content-length", str(len(payload)))
        self.send_header("cache-control", "private, max-age=300")
        self.end_headers()
        self.wfile.write(payload)

    def _send_html(self, html_text: str) -> None:
        data = html_text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status: HTTPStatus, code: str) -> None:
        self._send_json({"schema_id": "apv21_observatory_error/v1", "error": code}, status=status)


def generated_demo_audio_info(*, phrase_index: int = 0) -> dict:
    """
    Return demo audio with provenance for tests and human comparison.

    The phrase/provenance are observatory diagnostics only. AP receives the WAV
    bytes through the normal numeric audio sensor, and the inner audio player
    still reconstructs from state-pool numeric payloads rather than replaying
    this source.
    """

    phrase = _demo_audio_phrase(phrase_index)
    sapi = _sapi_wav_bytes(phrase)
    if sapi:
        return {"phrase": phrase, "generator": "windows_sapi", "wav_bytes": sapi}
    return {
        "phrase": phrase,
        "generator": "speech_like_fallback",
        "wav_bytes": _speech_like_fallback_wav(phrase_index=phrase_index),
    }


def generated_demo_wav(*, phrase_index: int = 0) -> bytes:
    return bytes(generated_demo_audio_info(phrase_index=phrase_index).get("wav_bytes", b"") or b"")


def _demo_audio_phrase(phrase_index: int) -> str:
    phrases = [
        "苹果在左边。",
        "香蕉在中间。",
        "红色方块在右边。",
        "请看左边。",
        "现在右边动了。",
    ]
    return phrases[int(phrase_index) % len(phrases)]


def _sapi_wav_bytes(text: str) -> bytes | None:
    """
    Generate a short Mandarin phrase through Windows SAPI when available.

    This is external audio input, not an inner-world playback shortcut. AP still
    receives only WAV bytes through the numeric audio sensor; the observatory
    later plays state-pool synthesis from numeric payloads.
    """

    try:
        import win32com.client  # type: ignore
    except Exception:
        return None
    tmp_path = ""
    try:
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        voices = speaker.GetVoices()
        for idx in range(voices.Count):
            voice = voices.Item(idx)
            desc = str(voice.GetDescription() or "")
            if "Huihui" in desc or "Chinese" in desc or "中文" in desc:
                speaker.Voice = voice
                break
        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as handle:
            tmp_path = handle.name
        stream.Open(tmp_path, 3, False)
        speaker.AudioOutputStream = stream
        speaker.Rate = 0
        speaker.Volume = 100
        speaker.Speak(str(text or ""), 0)
        stream.Close()
        return Path(tmp_path).read_bytes()
    except Exception:
        return None
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass


def _speech_like_fallback_wav(*, phrase_index: int = 0) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        frames = []
        syllables = [320, 420, 520, 390, 610]
        base = syllables[int(phrase_index) % len(syllables)]
        for idx in range(9600):
            syllable = idx // 1600
            local = idx % 1600
            voiced = 0.18 < local / 1600.0 < 0.82
            freq = base + (syllable % 3) * 65
            envelope = (0.20 + 0.80 * math.sin(math.pi * local / 1600.0)) if voiced else 0.02
            value = int(
                10500
                * envelope
                * (
                    math.sin(2 * math.pi * freq * idx / 16000)
                    + 0.22 * math.sin(2 * math.pi * (freq * 2.05) * idx / 16000)
                )
                / 1.22
            )
            frames.append(struct.pack("<h", value))
        wav_file.writeframes(b"".join(frames))
    return buf.getvalue()


def _bool_query(query: dict[str, list[str]], key: str, *, default: bool) -> bool:
    values = query.get(key)
    if not values:
        return bool(default)
    value = str(values[-1] or "").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _timeline_labels(rows: object, *, predicted: bool = False) -> list[str]:
    labels: list[str] = []
    for row in list(rows or [])[:3]:
        if not isinstance(row, dict):
            continue
        source = list(row.get("predicted_labels" if predicted else "top_labels", []) or row.get("top_labels", []) or [])
        for label in source:
            clean = str(label or "")
            if clean and clean not in labels:
                labels.append(clean)
            if len(labels) >= 4:
                return labels
        fallback = str(row.get("memory_id", "") or row.get("successor_memory_id", "") or "")
        if fallback and fallback not in labels:
            labels.append(fallback)
        if len(labels) >= 4:
            return labels
    return labels


def _timeline_gaze_action(action: dict) -> dict:
    """
    Extract an observatory-only summary of AP's visual-gaze decision.

    This does not feed back into cognition. It lets the 5-10 tick view explain
    whether the gaze moved because planner selected a visual action and which
    target/reason carried the drive.
    """

    selected = [dict(row) for row in list((action or {}).get("selected_actions", []) or []) if isinstance(row, dict)]
    visual_trace = dict((action or {}).get("visual_gaze", {}) or {})
    events = [dict(row) for row in list(visual_trace.get("events", []) or []) if isinstance(row, dict)]
    visual_ids = {
        "action::move_gaze_to",
        "action::nudge_gaze",
        "action::scan_visual_field",
        "action::hold_gaze",
        "action::zoom_visual_focus",
        "action::widen_visual_focus",
    }
    selected_visual = next((row for row in selected if str(row.get("action_id", "") or "") in visual_ids), {})
    event = _primary_gaze_event(events)
    return {
        "schema_id": "apv21_timeline_gaze_action/v1",
        "selected": bool(selected_visual or event),
        "action_id": str(event.get("action_id", selected_visual.get("action_id", "")) or ""),
        "target": str(event.get("target", (selected_visual.get("params", {}) or {}).get("target", "")) or ""),
        "reason": str(event.get("reason", (selected_visual.get("params", {}) or {}).get("reason", "")) or ""),
        "drive": round(float(event.get("drive", selected_visual.get("drive", 0.0)) or 0.0), 4),
        "movement_distance": round(float(event.get("movement_distance", 0.0) or 0.0), 4),
        "target_fatigue": _note_value(selected_visual, "target_fatigue"),
        "parameter_memory_bias": "parameter_memory_bias" in list(selected_visual.get("notes", []) or []),
        "parameter_drive_bias": _note_value(selected_visual, "parameter_drive_bias"),
        "learned_parameter_hint": dict((selected_visual.get("params", {}) or {}).get("learned_parameter_hint", {}) or {}),
        "old_center_norm": [
            round(float(event.get("old_center_x", 0.0) or 0.0), 4),
            round(float(event.get("old_center_y", 0.0) or 0.0), 4),
        ]
        if event
        else [],
        "new_center_norm": [
            round(float(event.get("center_x", 0.0) or 0.0), 4),
            round(float(event.get("center_y", 0.0) or 0.0), 4),
        ]
        if event
        else [],
    }


def _note_value(row: dict, key: str) -> float:
    prefix = f"{key}="
    for note in list((row or {}).get("notes", []) or []):
        text = str(note or "")
        if not text.startswith(prefix):
            continue
        try:
            return round(float(text.removeprefix(prefix)), 4)
        except ValueError:
            return 0.0
    return 0.0


def _playback_gaze_delta(render_model: dict) -> dict:
    """
    Derive a parameterized gaze-action trace for playback and later audits.

    AP should eventually learn gaze movement as action + parameters + outcome:
    where the focus started, where it ended, and how far/directionally it moved.
    The render model already carries the action event; this helper only formats
    those parameters for the observatory playback buffer.
    """

    action = dict((render_model or {}).get("action_panel", {}) or {})
    if not action:
        action = dict(((render_model or {}).get("detail_panel", {}) or {}).get("action", {}) or {})
    gaze_action = _timeline_gaze_action(action)
    old_center = list(gaze_action.get("old_center_norm", []) or [])
    new_center = list(gaze_action.get("new_center_norm", []) or [])
    delta_x = 0.0
    delta_y = 0.0
    if len(old_center) >= 2 and len(new_center) >= 2:
        delta_x = float(new_center[0] or 0.0) - float(old_center[0] or 0.0)
        delta_y = float(new_center[1] or 0.0) - float(old_center[1] or 0.0)
    return {
        "schema_id": "apv21_playback_gaze_delta/v1",
        "selected": bool(gaze_action.get("selected", False)),
        "action_id": str(gaze_action.get("action_id", "") or ""),
        "target": str(gaze_action.get("target", "") or ""),
        "reason": str(gaze_action.get("reason", "") or ""),
        "old_center_norm": old_center,
        "new_center_norm": new_center,
        "delta_norm": [round(delta_x, 4), round(delta_y, 4)] if old_center and new_center else [],
        "movement_distance": round(float(gaze_action.get("movement_distance", 0.0) or 0.0), 4),
        "target_fatigue": round(float(gaze_action.get("target_fatigue", 0.0) or 0.0), 4),
        "parameter_memory_bias": bool(gaze_action.get("parameter_memory_bias", False)),
        "parameter_drive_bias": round(float(gaze_action.get("parameter_drive_bias", 0.0) or 0.0), 4),
        "learned_parameter_hint": dict(gaze_action.get("learned_parameter_hint", {}) or {}),
    }


def _primary_gaze_event(events: list[dict]) -> dict:
    if not events:
        return {}
    for row in events:
        if float(row.get("movement_distance", 0.0) or 0.0) > 0.0001:
            return row
    for action_id in ("action::move_gaze_to", "action::nudge_gaze", "action::hold_gaze"):
        for row in events:
            if str(row.get("action_id", "") or "") == action_id:
                return row
    return events[0]


def _parse_range_header(range_header: str, payload_length: int) -> tuple[int, int] | None:
    text = str(range_header or "").strip().lower()
    total = max(0, int(payload_length))
    if not text.startswith("bytes=") or total <= 0:
        return None
    spec = text.removeprefix("bytes=").split(",", 1)[0].strip()
    if "-" not in spec:
        return None
    start_text, end_text = spec.split("-", 1)
    try:
        if start_text == "":
            suffix = max(0, int(end_text))
            if suffix <= 0:
                return None
            start = max(0, total - suffix)
            end = total - 1
        else:
            start = max(0, int(start_text))
            end = total - 1 if end_text == "" else min(total - 1, int(end_text))
    except ValueError:
        return None
    if start > end or start >= total:
        return None
    return start, end


def _content_type_for_encoding(encoding: str) -> str:
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "wav": "audio/wav",
        "json": "application/json; charset=utf-8",
    }.get(str(encoding or "").lower(), "application/octet-stream")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the APV2.1-native localhost observatory.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--no-warm-demo-memory", action="store_true")
    args = parser.parse_args(argv)

    app = APV21ObservatoryApp(warm_demo_memory=not args.no_warm_demo_memory)
    server = create_observatory_server(host=args.host, port=args.port, app=app)
    actual_host, actual_port = server.server_address[:2]
    print(f"APV2.1 native observatory running at http://{actual_host}:{actual_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping APV2.1 native observatory.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
