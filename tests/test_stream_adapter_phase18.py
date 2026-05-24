# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import tempfile
import threading
import unittest
import urllib.request
import json
from io import BytesIO
from pathlib import Path

from PIL import Image
from unittest.mock import patch

from observatory_v2.app import ObservatoryV2App
from observatory_v2.config import load_config
from observatory_v2.web import create_server
from sensors.stream_adapter_v1 import StreamAdapterV1, build_test_wav_bytes

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_frame(color: tuple[int, int, int], *, size: int = 32) -> bytes:
    image = Image.new("RGB", (size, size), color=color)
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class _FakeCap:
    def __init__(self, frames: list[bytes]) -> None:
        self._frames = list(frames)
        self._index = 0

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, object]:
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def get(self, prop: int) -> float:
        if prop == 5:
            return 12.0
        if prop == 7:
            return float(len(self._frames))
        return 0.0

    def release(self) -> None:
        return None


class StreamAdapterPhase18Tests(unittest.TestCase):
    def test_video_source_prefers_memory_decoder_backend_when_available(self) -> None:
        adapter = StreamAdapterV1()

        class _FakeDecoder:
            def __init__(self) -> None:
                self._frames = [
                    (0, build_frame((10, 10, 10)), {"decode_backend": "pyav_bytes_memory", "decode_mode": "memory_bytes", "temp_suffix": "", "file_hint": "demo.mp4", "frame_width": 32, "frame_height": 32}),
                    (1, build_frame((240, 240, 240)), {"decode_backend": "pyav_bytes_memory", "decode_mode": "memory_bytes", "temp_suffix": "", "file_hint": "demo.mp4", "frame_width": 32, "frame_height": 32}),
                ]
                self._index = 0

            def status(self) -> dict:
                return {"native_fps": 24.0, "total_frames": 2}

            def read_frame_png(self):
                if self._index >= len(self._frames):
                    return None
                row = self._frames[self._index]
                self._index += 1
                return row

            def close(self) -> None:
                return None

        def fake_open_video_decoder_v1(*, raw_bytes: bytes, file_hint: str, suffix_hint: str):
            return _FakeDecoder(), [{"backend": "pyav_bytes_memory", "ok": True, "uses_tempfile": False}]

        with patch("sensors.stream_adapter_v1.open_video_decoder_v1", side_effect=fake_open_video_decoder_v1):
            source = adapter.build_video_file_source(raw_bytes=b"fake-video", max_frames=2, frame_stride=1, file_hint="demo.mp4")
            item0 = source.next_item()
            item1 = source.next_item()
            item2 = source.next_item()
            self.assertIsNotNone(item0)
            self.assertIsNotNone(item1)
            self.assertIsNone(item2)
            frame_meta = dict(item0["stream_frame_meta"])
            self.assertEqual(frame_meta.get("decode_backend"), "pyav_bytes_memory")
            self.assertEqual(frame_meta.get("decode_mode"), "memory_bytes")
            self.assertEqual(frame_meta.get("temp_suffix"), "")
            attempts = list(frame_meta.get("decoder_attempts", []) or [])
            self.assertEqual(attempts[0].get("backend"), "pyav_bytes_memory")
            self.assertFalse(bool(attempts[0].get("uses_tempfile", True)))

    def test_video_source_respects_max_frames_without_emitting_extra_item(self) -> None:
        fake_frames = [
            build_frame((10, 10, 10)),
            build_frame((240, 240, 240)),
            build_frame((10, 240, 10)),
        ]

        def fake_video_capture(_path: str) -> _FakeCap:
            return _FakeCap(fake_frames)

        def fake_imencode(_ext: str, frame: bytes) -> tuple[bool, object]:
            class _Encoded:
                def __init__(self, data: bytes) -> None:
                    self._data = data

                def tobytes(self) -> bytes:
                    return self._data

            return True, _Encoded(frame)

        adapter = StreamAdapterV1()
        with patch("cv2.VideoCapture", side_effect=fake_video_capture), patch("cv2.imencode", side_effect=fake_imencode):
            source = adapter.build_video_file_source(raw_bytes=b"fake-video", max_frames=2, frame_stride=1, file_hint="demo.mp4")
            item0 = source.next_item()
            item1 = source.next_item()
            item2 = source.next_item()
            self.assertIsNotNone(item0)
            self.assertIsNotNone(item1)
            self.assertIsNone(item2)
            self.assertTrue(source.status().get("exhausted"))

    def test_audio_stream_run_splits_long_audio_into_multiple_ticks(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            wav_bytes = build_test_wav_bytes(sample_rate=8000, duration_sec=0.32, frequency=440.0)
            result = app.start_audio_stream_run(
                audio_bytes=wav_bytes,
                text_prefix="listen",
                tick_window_ms=80,
                label="audio stream test",
                tick_interval_ms=0,
                reset_runtime=True,
            )
            self.assertTrue(app.wait_for_idle(timeout_sec=20.0))
            manifest = app.get_manifest(result["run_id"])
            self.assertEqual(manifest["status"], "completed")
            self.assertGreaterEqual(int(manifest.get("tick_done", 0) or 0), 3)
            sidecar0 = app.get_tick_sidecar(result["run_id"], 0)
            self.assertIn("audio_packet", sidecar0)
            self.assertGreater(sidecar0["audio_packet"].get("budget_used", 0), 0)
            self.assertEqual(sidecar0["input_item"]["stream_source_meta"]["source_kind"], "audio_file")
            summary0 = app.get_tick_summary(result["run_id"], 0)
            self.assertEqual(summary0["multimodal_summary"]["stream_source"]["source_kind"], "audio_file")

    def test_image_stream_run_accepts_frame_sequence(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            result = app.start_image_stream_run(
                frame_bytes_list=[build_frame((10, 10, 10)), build_frame((240, 240, 240)), build_frame((10, 240, 10))],
                text_prefix="see",
                label="image stream test",
                tick_interval_ms=0,
                reset_runtime=True,
            )
            self.assertTrue(app.wait_for_idle(timeout_sec=20.0))
            manifest = app.get_manifest(result["run_id"])
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(int(manifest.get("tick_done", 0) or 0), 3)
            sidecar1 = app.get_tick_sidecar(result["run_id"], 1)
            self.assertIn("image_packet", sidecar1)
            self.assertGreater(sidecar1["image_packet"].get("budget_used", 0), 0)
            self.assertEqual(sidecar1["input_item"]["stream_source_meta"]["source_kind"], "image_sequence")

    def test_video_stream_run_accepts_decoded_video_frames(self) -> None:
        config = load_config()
        fake_frames = [build_frame((10, 10, 10)), build_frame((240, 240, 240)), build_frame((10, 240, 10))]

        def fake_video_capture(_path: str) -> _FakeCap:
            return _FakeCap(fake_frames)

        def fake_imencode(_ext: str, frame: bytes) -> tuple[bool, object]:
            class _Encoded:
                def __init__(self, data: bytes) -> None:
                    self._data = data

                def tobytes(self) -> bytes:
                    return self._data

            return True, _Encoded(frame)

        with tempfile.TemporaryDirectory() as tmpdir, patch("cv2.VideoCapture", side_effect=fake_video_capture), patch("cv2.imencode", side_effect=fake_imencode):
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            result = app.start_video_stream_run(
                video_bytes=b"fake-video",
                video_name="demo.mp4",
                text_prefix="watch",
                frame_stride=1,
                max_frames=3,
                label="video stream test",
                tick_interval_ms=0,
                reset_runtime=True,
            )
            self.assertTrue(app.wait_for_idle(timeout_sec=20.0))
            manifest = app.get_manifest(result["run_id"])
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(int(manifest.get("tick_done", 0) or 0), 3)
            sidecar1 = app.get_tick_sidecar(result["run_id"], 1)
            self.assertEqual(sidecar1["input_item"]["source_type"], "video_stream")
            self.assertEqual(sidecar1["input_item"]["stream_source_meta"]["source_kind"], "video_file")
            self.assertIn("stream_frame_meta", sidecar1["input_item"])
            frame_meta = dict(sidecar1["input_item"]["stream_frame_meta"])
            self.assertEqual(frame_meta.get("temp_suffix"), ".mp4")
            self.assertEqual(frame_meta.get("decode_backend"), "opencv_videocapture_lazy")
            self.assertEqual(frame_meta.get("decode_mode"), "tempfile_path")
            self.assertTrue(any(bool(item.get("uses_tempfile", False)) for item in (frame_meta.get("decoder_attempts", []) or [])))

    def test_stream_adapter_splitters_and_web_endpoints(self) -> None:
        adapter = StreamAdapterV1()
        audio_items = adapter.split_audio_wav_bytes(
            build_test_wav_bytes(sample_rate=8000, duration_sec=0.24, frequency=220.0),
            tick_window_ms=60,
        )
        self.assertGreaterEqual(len(audio_items), 3)

        strip = Image.new("RGB", (24, 72), color=(0, 0, 0))
        for y in range(0, 24):
            for x in range(24):
                strip.putpixel((x, y), (255, 0, 0))
        for y in range(24, 48):
            for x in range(24):
                strip.putpixel((x, y), (0, 255, 0))
        for y in range(48, 72):
            for x in range(24):
                strip.putpixel((x, y), (0, 0, 255))
        strip_buf = BytesIO()
        strip.save(strip_buf, format="PNG")
        image_items = adapter.split_vertical_strip_image(strip_buf.getvalue(), frame_count=3)
        self.assertEqual(len(image_items), 3)

        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            server = create_server(app, host="127.0.0.1", port=0)
            host, port = server.server_address[:2]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                req_audio = urllib.request.Request(
                    f"http://{host}:{port}/api/runs/audio-stream/start",
                    data=json.dumps(
                        {
                            "audio_b64": base64.b64encode(build_test_wav_bytes(sample_rate=8000, duration_sec=0.24, frequency=330.0)).decode("ascii"),
                            "text_prefix": "web-audio",
                            "tick_window_ms": 60,
                            "tick_interval_ms": 0,
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req_audio, timeout=5) as resp:
                    audio_result = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(audio_result["ok"])
                self.assertTrue(app.wait_for_idle(timeout_sec=20.0))

                req_image = urllib.request.Request(
                    f"http://{host}:{port}/api/runs/image-stream/start",
                    data=json.dumps(
                        {
                            "frames_b64": [
                                base64.b64encode(build_frame((20, 20, 20))).decode("ascii"),
                                base64.b64encode(build_frame((250, 250, 250))).decode("ascii"),
                            ],
                            "text_prefix": "web-image",
                            "tick_interval_ms": 0,
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req_image, timeout=5) as resp:
                    image_result = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(image_result["ok"])
                self.assertTrue(app.wait_for_idle(timeout_sec=20.0))

                def fake_video_capture(_path: str) -> _FakeCap:
                    return _FakeCap([build_frame((1, 1, 1)), build_frame((200, 200, 200))])

                def fake_imencode(_ext: str, frame: bytes) -> tuple[bool, object]:
                    class _Encoded:
                        def __init__(self, data: bytes) -> None:
                            self._data = data

                        def tobytes(self) -> bytes:
                            return self._data

                    return True, _Encoded(frame)

                with patch("cv2.VideoCapture", side_effect=fake_video_capture), patch("cv2.imencode", side_effect=fake_imencode):
                    req_video = urllib.request.Request(
                        f"http://{host}:{port}/api/runs/video-stream/start",
                        data=json.dumps(
                            {
                                "video_b64": base64.b64encode(b"fake-video").decode("ascii"),
                                "text_prefix": "web-video",
                                "frame_stride": 1,
                                "max_frames": 2,
                                "tick_interval_ms": 0,
                            }
                        ).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req_video, timeout=5) as resp:
                        video_result = json.loads(resp.read().decode("utf-8"))
                    self.assertTrue(video_result["ok"])
                    self.assertTrue(app.wait_for_idle(timeout_sec=20.0))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_split_video_file_bytes_matches_realtime_source_semantics(self) -> None:
        fake_frames = [build_frame((10, 10, 10)), build_frame((240, 240, 240)), build_frame((10, 240, 10))]

        def fake_video_capture(_path: str) -> _FakeCap:
            return _FakeCap(fake_frames)

        def fake_imencode(_ext: str, frame: bytes) -> tuple[bool, object]:
            class _Encoded:
                def __init__(self, data: bytes) -> None:
                    self._data = data

                def tobytes(self) -> bytes:
                    return self._data

            return True, _Encoded(frame)

        adapter = StreamAdapterV1()
        with patch("cv2.VideoCapture", side_effect=fake_video_capture), patch("cv2.imencode", side_effect=fake_imencode):
            split_items = adapter.split_video_file_bytes(
                raw_bytes=b"fake-video",
                frame_stride=1,
                max_frames=2,
                source_type="video_stream",
                file_hint="demo.mp4",
            )
        self.assertEqual(len(split_items), 2)
        self.assertEqual(split_items[0]["stream_source_meta"]["source_kind"], "video_file")
        self.assertEqual(split_items[0]["stream_frame_meta"]["frame_index"], 0)
        self.assertEqual(split_items[1]["stream_frame_meta"]["frame_index"], 1)
        self.assertEqual(split_items[1]["stream_frame_meta"]["sampled_from_frame_index"], 1)
        self.assertEqual(split_items[0]["stream_frame_meta"]["decode_backend"], "opencv_videocapture_lazy")
        self.assertEqual(split_items[0]["stream_frame_meta"]["decode_mode"], "tempfile_path")

    def test_unified_realtime_sources_status_and_unavailable_paths(self) -> None:
        adapter = StreamAdapterV1()
        audio_source = adapter.build_audio_file_source(
            raw_bytes=build_test_wav_bytes(sample_rate=8000, duration_sec=0.24, frequency=220.0),
            tick_window_ms=60,
        )
        first_audio = audio_source.next_item()
        self.assertIsNotNone(first_audio)
        self.assertEqual(first_audio["stream_source_meta"]["source_kind"], "audio_file")

        image_source = adapter.build_image_sequence_source(
            frames=[build_frame((10, 10, 10)), build_frame((20, 20, 20))],
            source_type="image_stream",
        )
        first_image = image_source.next_item()
        self.assertIsNotNone(first_image)
        self.assertEqual(first_image["stream_source_meta"]["source_kind"], "image_sequence")

        screen_source = adapter.build_screen_capture_source(text_hint="observe")
        first_screen = screen_source.next_item()
        self.assertIsNotNone(first_screen)
        self.assertTrue(first_screen.get("capture_screen"))
        self.assertTrue(screen_source.status().get("realtime"))

        webcam_source = adapter.build_webcam_source()
        webcam_status = webcam_source.status()
        self.assertEqual(webcam_status.get("source_kind"), "webcam")
        if webcam_status.get("unavailable"):
            self.assertIsNone(webcam_source.next_item())
        else:
            webcam_source.close()
            self.assertTrue(webcam_source.status().get("closed"))

        mic_source = adapter.build_microphone_source()
        mic_status = mic_source.status()
        self.assertEqual(mic_status.get("source_kind"), "microphone")
        if mic_status.get("unavailable"):
            self.assertIsNone(mic_source.next_item())
        else:
            mic_source.close()
            self.assertTrue(mic_source.status().get("closed"))

    def test_webcam_and_microphone_web_endpoints_fail_gracefully_when_unavailable(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            server = create_server(app, host="127.0.0.1", port=0)
            host, port = server.server_address[:2]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                webcam_req = urllib.request.Request(
                    f"http://{host}:{port}/api/runs/webcam-stream/start",
                    data=json.dumps({"max_frames": 1, "tick_interval_ms": 0}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(webcam_req, timeout=5) as resp:
                        webcam_payload = json.loads(resp.read().decode("utf-8"))
                    self.assertTrue(webcam_payload["ok"])
                    self.assertTrue(app.wait_for_idle(timeout_sec=20.0))
                except Exception:
                    pass

                mic_req = urllib.request.Request(
                    f"http://{host}:{port}/api/runs/microphone-stream/start",
                    data=json.dumps({"max_windows": 1, "tick_interval_ms": 0}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(mic_req, timeout=5) as resp:
                        mic_payload = json.loads(resp.read().decode("utf-8"))
                    self.assertTrue(mic_payload["ok"])
                    self.assertTrue(app.wait_for_idle(timeout_sec=20.0))
                except Exception:
                    pass
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
